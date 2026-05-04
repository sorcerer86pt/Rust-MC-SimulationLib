//! Materials: composition + macroscopic cross sections + per-nuclide
//! XS evaluators for the transport loop.

use std::sync::Arc;

use crate::physics::angular::AngularDistribution;
use crate::physics::spectra::EnergyDistribution;
use crate::table::PointwiseTable;
use crate::urr::UrrProbabilityTables;

/// Per-reaction microscopic cross sections for one nuclide at one
/// energy (barns).
#[derive(Debug, Clone, Copy, Default)]
pub struct MicroXs {
    pub total: f64,
    pub elastic: f64,
    pub inelastic: f64,
    pub n2n: f64,
    pub n3n: f64,
    pub fission: f64,
    pub capture: f64,
    pub nu_bar: f64,
    pub awr: f64,
}

/// Optional discrete-inelastic level (`MT = 51..91`).
pub struct DiscreteLevel {
    pub mt: u32,
    pub q_value: f64,
    pub threshold: f64,
    pub xs: PointwiseTable,
}

/// Self-contained nuclide: pointwise cross sections + sampling
/// distributions + optional URR. Built once from your nuclear-data
/// source (raw HDF5, ENDF, your own format) and used by the
/// transport loop.
pub struct Nuclide {
    pub name: String,
    /// A / m_n.
    pub awr: f64,
    /// Constant nu-bar fallback when the table is absent.
    pub nu_bar_const: f64,
    /// Energy-dependent prompt + delayed nu-bar; optional.
    pub nu_bar_table: Option<PointwiseTable>,
    pub elastic: Option<PointwiseTable>,
    pub inelastic: Option<PointwiseTable>,
    pub n2n: Option<PointwiseTable>,
    pub n3n: Option<PointwiseTable>,
    pub fission: Option<PointwiseTable>,
    pub capture: Option<PointwiseTable>,
    /// Discrete inelastic levels (MT=51..91), in incident-energy
    /// order. Each carries its own threshold + Q + cross section.
    pub discrete_levels: Vec<DiscreteLevel>,
    /// Per-level CM-frame angular distribution; aligned with
    /// `discrete_levels`. `None` entries → isotropic.
    pub discrete_level_angles: Vec<Option<AngularDistribution>>,
    /// Whether `discrete_levels` includes a continuum MT=91 entry.
    pub has_continuum_inelastic: bool,
    /// Elastic-scatter angular distribution (MT=2).
    pub elastic_angle: Option<AngularDistribution>,
    /// Fission outgoing-energy distribution (continuous tabulated).
    pub fission_energy_dist: Option<EnergyDistribution>,
    /// (n, 2n) outgoing-energy distribution.
    pub n2n_edist: Option<EnergyDistribution>,
    /// (n, 3n) outgoing-energy distribution.
    pub n3n_edist: Option<EnergyDistribution>,
    /// MT=91 continuum outgoing-energy distribution.
    pub inelastic_continuum_edist: Option<EnergyDistribution>,
    /// Unresolved Resonance Range probability tables.
    pub urr_tables: Option<UrrProbabilityTables>,
}

impl Nuclide {
    /// Empty nuclide with the given name + AWR. Use the field
    /// setters to populate the data you have.
    pub fn empty(name: impl Into<String>, awr: f64) -> Self {
        Self {
            name: name.into(),
            awr,
            nu_bar_const: 0.0,
            nu_bar_table: None,
            elastic: None,
            inelastic: None,
            n2n: None,
            n3n: None,
            fission: None,
            capture: None,
            discrete_levels: Vec::new(),
            discrete_level_angles: Vec::new(),
            has_continuum_inelastic: false,
            elastic_angle: None,
            fission_energy_dist: None,
            n2n_edist: None,
            n3n_edist: None,
            inelastic_continuum_edist: None,
            urr_tables: None,
        }
    }

    /// Energy-dependent ν̄.
    pub fn nu_bar_at(&self, energy: f64) -> f64 {
        self.nu_bar_table
            .as_ref()
            .map_or(self.nu_bar_const, |t| t.lookup(energy))
    }

    /// Microscopic cross sections for all channels at `energy`. The
    /// inelastic XS is taken from `inelastic` when present, otherwise
    /// from the sum of accessible discrete-level cross sections.
    pub fn micro_xs(&self, energy: f64) -> MicroXs {
        let elastic = self.elastic.as_ref().map_or(0.0, |t| t.lookup(energy));
        let inelastic = match &self.inelastic {
            Some(t) => t.lookup(energy),
            None if !self.discrete_levels.is_empty() => self
                .discrete_levels
                .iter()
                .filter(|l| energy >= l.threshold)
                .map(|l| l.xs.lookup(energy).max(0.0))
                .sum(),
            None => 0.0,
        };
        let n2n = self.n2n.as_ref().map_or(0.0, |t| t.lookup(energy));
        let n3n = self.n3n.as_ref().map_or(0.0, |t| t.lookup(energy));
        let fission = self.fission.as_ref().map_or(0.0, |t| t.lookup(energy));
        let capture = self.capture.as_ref().map_or(0.0, |t| t.lookup(energy));
        let total = elastic + inelastic + n2n + n3n + fission + capture;
        MicroXs {
            total,
            elastic,
            inelastic,
            n2n,
            n3n,
            fission,
            capture,
            nu_bar: self.nu_bar_at(energy),
            awr: self.awr,
        }
    }
}

/// Composition of a material: nuclides + atom densities.
pub struct Material {
    pub name: String,
    /// Temperature in Kelvin. Used by the free-gas thermal-scatter
    /// branch in [`crate::physics::scatter::elastic_scatter_aniso`].
    pub temperature_k: f64,
    /// `(nuclide, atom_density_per_b_per_cm)` pairs.
    pub nuclides: Vec<(Arc<Nuclide>, f64)>,
}

impl Material {
    pub fn new(name: impl Into<String>, temperature_k: f64) -> Self {
        Self {
            name: name.into(),
            temperature_k,
            nuclides: Vec::new(),
        }
    }

    pub fn add(&mut self, nuc: Arc<Nuclide>, atom_density: f64) {
        self.nuclides.push((nuc, atom_density));
    }

    /// Macroscopic total cross section Σ_t at `energy`, in 1/cm.
    /// `Σ = Σ_n N_n · σ_t,n(E)` summed across nuclides.
    pub fn macro_total(&self, energy: f64) -> f64 {
        self.nuclides
            .iter()
            .map(|(n, density)| density * n.micro_xs(energy).total)
            .sum()
    }

    /// Number density of nuclides in the material (atoms/b·cm).
    pub fn total_atom_density(&self) -> f64 {
        self.nuclides.iter().map(|(_, d)| *d).sum()
    }
}
