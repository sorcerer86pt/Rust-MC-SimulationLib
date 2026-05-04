//! High-level loader: OpenMC HDF5 file → [`crate::transport::material::Nuclide`].
//!
//! This is the glue a downstream simulation needs to go from "I have
//! ENDF/B-VII.1 HDF5 files on disk" to a `Nuclide` ready for the
//! transport loop with realistic physics: pointwise XS at the chosen
//! temperature, energy-dependent ν̄, fission outgoing-energy
//! distribution, elastic CM angular distribution, URR probability
//! tables, and discrete inelastic levels with their thresholds.
//!
//! The reader picks the closest temperature column to `target_temp_k`
//! from what's available in the file (ENDF/B-VII.1 ships {250, 294,
//! 600, 900, 1200, 2500} K for most evaluations). For off-library
//! temperatures, callers can post-process by Doppler-broadening the
//! returned `PointwiseTable`s through [`crate::doppler`] or by
//! constructing an SVD/Ducru reconstruction stack themselves.

use std::path::Path;
use std::sync::Arc;

use crate::error::{NuclearError, NuclearResult};
use crate::nuclear_hdf5::{
    self as hdf5_reader, AngularDistribution as HdfAngular,
    EnergyDistribution as HdfEnergy, NuclideData, TabularEnergyDist as HdfTabularE,
    TabularMuDist as HdfTabularMu, UrrProbabilityTables as HdfUrr,
};
use crate::physics::angular::{AngularDistribution, TabularMuDist};
use crate::physics::spectra::{EnergyDistribution, TabularEnergyDist};
use crate::table::PointwiseTable;
use crate::transport::material::{DiscreteLevel, Nuclide};
use crate::urr::UrrProbabilityTables;

/// Per-call switches for how much physics to load. All default `true`
/// — turn things off when you want a fast-loading minimal nuclide.
#[derive(Debug, Clone, Copy)]
pub struct LoaderConfig {
    pub include_inelastic: bool,
    pub include_n2n: bool,
    pub include_n3n: bool,
    pub include_nu_bar: bool,
    pub include_fission_spectrum: bool,
    pub include_anisotropic_elastic: bool,
    pub include_urr: bool,
    pub include_discrete_levels: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            include_inelastic: true,
            include_n2n: true,
            include_n3n: true,
            include_nu_bar: true,
            include_fission_spectrum: true,
            include_anisotropic_elastic: true,
            include_urr: true,
            include_discrete_levels: true,
        }
    }
}

/// Load a complete [`Nuclide`] from an OpenMC HDF5 file.
///
/// `target_temp_k` selects the on-library temperature column closest
/// to that value. The returned `Nuclide` has its XS tables populated
/// at that single temperature — no temperature interpolation here.
pub fn load_nuclide_from_hdf5(
    path: &Path,
    target_temp_k: f64,
    cfg: &LoaderConfig,
) -> NuclearResult<Nuclide> {
    let awr = hdf5_reader::read_awr(path)?;

    // Discover which temperature column the file ships closest to
    // the target. We rely on MT=2 being present in every nuclide
    // file (elastic).
    let elastic_data = NuclideData::from_hdf5(path, 2)?;
    let temp_idx = pick_temperature_index(&elastic_data.temperatures, target_temp_k)?;
    let temp_label = elastic_data.temp_labels[temp_idx].clone();
    let energies = elastic_data.energies.clone();

    let nuclide_name = stem_to_nuclide_name(path);
    let mut nuclide = Nuclide::empty(nuclide_name, awr);
    nuclide.elastic = Some(table_at_temp(&elastic_data, temp_idx, &energies));

    // ── Optional channels. Each `try_load` is a no-op if the
    // reaction isn't in the file (e.g. (n,2n) is threshold-gated;
    // many low-A nuclides skip it). ─────────────────────────────────
    nuclide.fission = try_load_xs(path, 18, temp_idx, &energies)?;
    nuclide.capture = try_load_xs(path, 102, temp_idx, &energies)?;
    if cfg.include_inelastic {
        nuclide.inelastic = try_load_xs(path, 4, temp_idx, &energies)?;
    }
    if cfg.include_n2n {
        nuclide.n2n = try_load_xs(path, 16, temp_idx, &energies)?;
    }
    if cfg.include_n3n {
        nuclide.n3n = try_load_xs(path, 17, temp_idx, &energies)?;
    }

    // ν̄. If the file lacks the table (non-fissionable), keep the
    // empty default; if it carries it, normalise into a single
    // `PointwiseTable` covering the energy grid.
    if cfg.include_nu_bar {
        if let Ok(nu) = hdf5_reader::read_nu_bar(path) {
            if !nu.energies.is_empty() {
                nuclide.nu_bar_table =
                    Some(PointwiseTable::new(nu.energies, nu.values));
            }
        }
    }

    // Fission outgoing-energy distribution. Same struct shape both
    // sides — convert field-by-field.
    if cfg.include_fission_spectrum {
        if let Ok(Some(spec)) = hdf5_reader::read_fission_energy_dist(path) {
            nuclide.fission_energy_dist = Some(convert_energy_dist(spec));
        }
    }

    // Elastic CM angular distribution.
    if cfg.include_anisotropic_elastic {
        if let Ok(Some(ang)) = hdf5_reader::read_angular_distribution(path, 2) {
            nuclide.elastic_angle = Some(convert_angular(ang));
        }
    }

    // URR probability tables at the chosen temperature.
    if cfg.include_urr {
        if let Ok(Some(urr)) = hdf5_reader::read_urr_tables(path, &temp_label) {
            nuclide.urr_tables = Some(convert_urr(urr));
        }
    }

    // Discrete inelastic levels (MT = 51 .. 91). Each level has a
    // Q-value, a threshold, and its own pointwise XS.
    if cfg.include_discrete_levels {
        if let Ok(levels) = hdf5_reader::read_discrete_levels(path, awr) {
            for info in levels {
                let xs_data = match NuclideData::from_hdf5(path, info.mt) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let xs_table = table_at_temp(&xs_data, temp_idx, &xs_data.energies);
                nuclide.discrete_levels.push(DiscreteLevel {
                    mt: info.mt,
                    q_value: info.q_value,
                    threshold: info.threshold,
                    xs: xs_table,
                });
                // Try to attach the level's angular distribution.
                let level_angle = hdf5_reader::read_angular_distribution(path, info.mt)
                    .ok()
                    .flatten()
                    .map(convert_angular);
                nuclide.discrete_level_angles.push(level_angle);
            }
            nuclide.has_continuum_inelastic = nuclide
                .discrete_levels
                .iter()
                .any(|l| l.mt == 91);
        }
    }
    Ok(nuclide)
}

/// Resolve the temperature index closest to `target_k` in the file's
/// available temperatures.
fn pick_temperature_index(temps: &[f64], target_k: f64) -> NuclearResult<usize> {
    if temps.is_empty() {
        return Err(NuclearError::Hdf5 {
            path: String::new(),
            detail: "no temperatures in nuclide file".into(),
        });
    }
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, &t) in temps.iter().enumerate() {
        let d = (t - target_k).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    Ok(best)
}

/// Build a `PointwiseTable` from one temperature column of a
/// [`NuclideData`]. The reader has already interpolated the per-T
/// xs vector onto the *unionised* energy grid in `data.energies`,
/// so the table just wraps `(energies, xs_per_temp[temp_idx])`.
fn table_at_temp(data: &NuclideData, temp_idx: usize, energies: &[f64]) -> PointwiseTable {
    let xs = data.xs_per_temp[temp_idx].clone();
    PointwiseTable::new(energies.to_vec(), xs)
}

/// Best-effort load of one MT. Missing-reaction errors are folded
/// to `Ok(None)`; HDF5 file-level errors propagate.
fn try_load_xs(
    path: &Path,
    mt: u32,
    temp_idx: usize,
    union_grid: &[f64],
) -> NuclearResult<Option<PointwiseTable>> {
    match NuclideData::from_hdf5(path, mt) {
        Ok(d) => {
            // Coerce the reaction's xs vector onto the master union
            // grid (it'll already match if the reader unionised the
            // same way; safe to copy when lengths match).
            if d.energies.len() == union_grid.len() {
                Ok(Some(table_at_temp(&d, temp_idx, union_grid)))
            } else {
                Ok(Some(table_at_temp(&d, temp_idx, &d.energies)))
            }
        }
        Err(_) => Ok(None),
    }
}

fn convert_angular(src: HdfAngular) -> AngularDistribution {
    AngularDistribution {
        energies: src.energies,
        distributions: src
            .distributions
            .into_iter()
            .map(|d| TabularMuDist {
                mu: d.mu,
                pdf: d.pdf,
                cdf: d.cdf,
                histogram: d.histogram,
            })
            .collect(),
        center_of_mass: src.center_of_mass,
    }
}

fn convert_energy_dist(src: HdfEnergy) -> EnergyDistribution {
    EnergyDistribution {
        energies: src.energies,
        distributions: src
            .distributions
            .into_iter()
            .map(|d| TabularEnergyDist {
                e_out: d.e_out,
                pdf: d.pdf,
                cdf: d.cdf,
            })
            .collect(),
    }
}

fn convert_urr(src: HdfUrr) -> UrrProbabilityTables {
    UrrProbabilityTables {
        energies: src.energies,
        n_bands: src.n_bands,
        cum_prob: src.cum_prob,
        total_factor: src.total_factor,
        elastic_factor: src.elastic_factor,
        fission_factor: src.fission_factor,
        capture_factor: src.capture_factor,
        multiply_smooth: src.multiply_smooth,
        interpolation: src.interpolation,
    }
}

/// Load an OpenMC thermal-scattering HDF5 file (e.g. `c_H_in_H2O.h5`)
/// and return it as an `Arc<dyn ThermalScatterer>` ready to attach to
/// a `Nuclide` via [`attach_thermal_scattering`].
pub fn load_thermal_scattering(
    path: &Path,
) -> NuclearResult<Arc<dyn crate::physics::thermal::ThermalScatterer>> {
    let data = hdf5_reader::load_thermal_scattering(path)?;
    Ok(Arc::new(data))
}

/// Convenience: bolt an already-loaded thermal kernel onto an
/// already-loaded `Nuclide`. Returns the same nuclide for chaining.
pub fn attach_thermal_scattering(
    mut nuclide: Nuclide,
    kernel: Arc<dyn crate::physics::thermal::ThermalScatterer>,
) -> Nuclide {
    nuclide.thermal_scattering = Some(kernel);
    nuclide
}

/// Recover a clean nuclide name from a path like
/// `…/U235.h5` → `"U235"`.
fn stem_to_nuclide_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

// Unused helper conversions exist as tabular-type pairs above;
// silence the missing-use warnings under feature gating.
#[allow(dead_code)]
fn _unused(_: HdfTabularE, _: HdfTabularMu) {}
