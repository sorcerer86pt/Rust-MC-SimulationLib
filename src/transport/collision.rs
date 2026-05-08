//! Collision dispatch: pick a nuclide → pick a reaction → call the
//! kinematics routine → update the particle.

use crate::physics::scatter::{
    elastic_scatter, elastic_scatter_aniso, inelastic_scatter, rotate_direction,
};
use crate::physics::spectra::evaporation;
use crate::rng::Pcg64;
use crate::transport::material::{Material, MicroXs, Nuclide};
use crate::transport::particle::{FissionSite, Particle};

/// Outcome of a collision the transport loop has to handle.
#[derive(Debug)]
pub enum CollisionOutcome {
    /// Particle scattered; energy + direction already updated.
    Scatter,
    /// Inelastic on a discrete level (or continuum MT=91).
    /// `q_value_ev` is the level Q (negative for excitation).
    InelasticScatter { q_value_ev: f64 },
    /// Particle absorbed (capture or absorbed minor channel).
    Absorption,
    /// Fission — particle is gone; `sites` seed the next generation.
    Fission { sites: Vec<FissionSite> },
    /// (n, 2n) / (n, 3n): primary continues, secondaries banked
    /// for the *current* generation (not the fission bank).
    Multiplicity { secondaries: Vec<SecondaryNeutron> },
}

/// A non-fission secondary neutron emitted in (n, 2n) / (n, 3n).
#[derive(Debug, Clone)]
pub struct SecondaryNeutron {
    pub pos: crate::geometry::Vec3,
    pub dir: crate::geometry::Vec3,
    pub energy: f64,
}

/// Pick which nuclide reacts with the incoming neutron, given
/// per-nuclide cross sections that have already been computed at
/// `particle.energy`.
///
/// Returns the chosen index into `material.nuclides` and the
/// micro-XS struct to feed into [`process_collision`].
pub fn sample_nuclide_at_energy(
    material: &Material,
    energy: f64,
    rng: &mut Pcg64,
) -> Option<(usize, MicroXs)> {
    let mut cum_total = 0.0_f64;
    let temperature_k = material.temperature_k;
    let nuclide_xs: Vec<MicroXs> = material
        .nuclides
        .iter()
        .map(|(n, _)| n.micro_xs_at_temp(energy, temperature_k))
        .collect();
    let total: f64 = material
        .nuclides
        .iter()
        .zip(nuclide_xs.iter())
        .map(|((_, density), xs)| density * xs.total)
        .sum();
    if total <= 0.0 {
        return None;
    }
    let xi = rng.uniform() * total;
    for (i, ((_, density), xs)) in material.nuclides.iter().zip(nuclide_xs.iter()).enumerate() {
        cum_total += density * xs.total;
        if xi < cum_total {
            return Some((i, *xs));
        }
    }
    let last = material.nuclides.len() - 1;
    Some((last, nuclide_xs[last]))
}

/// Pick a reaction channel and apply its kinematics. Returns the
/// [`CollisionOutcome`] the transport loop dispatches on.
pub fn process_collision(
    particle: &mut Particle,
    nuclide: &Nuclide,
    xs: &MicroXs,
    temperature_k: f64,
    rng: &mut Pcg64,
) -> CollisionOutcome {
    particle.n_collisions += 1;
    let xi = rng.uniform() * xs.total;
    let mut cum = 0.0_f64;

    cum += xs.elastic;
    if xi < cum {
        // Bound-atom thermal kernel takes over below energy_max.
        if let Some(thermal) = &nuclide.thermal_scattering {
            if particle.energy <= thermal.energy_max() {
                let (e_out, mu_lab) = thermal.sample(particle.energy, temperature_k, rng);
                particle.dir = rotate_direction(particle.dir, mu_lab, rng);
                particle.energy = e_out.max(1e-5);
                return CollisionOutcome::Scatter;
            }
        }
        let (e, d) = elastic_scatter_aniso(
            particle.energy,
            particle.dir,
            xs.awr,
            nuclide.elastic_angle.as_ref(),
            temperature_k,
            rng,
        );
        particle.energy = e;
        particle.dir = d;
        return CollisionOutcome::Scatter;
    }

    cum += xs.inelastic;
    if xi < cum {
        // Sample a discrete level proportional to its threshold-gated
        // cross section. Continuum MT=91 (when present) is one of
        // the entries; we sample its outgoing energy from the
        // continuum-tabulated distribution.
        let (q_value, level_idx) = sample_inelastic_level(particle.energy, xs.awr, nuclide, rng);
        let angle = level_idx.and_then(|i| {
            nuclide
                .discrete_level_angles
                .get(i)
                .and_then(|o| o.as_ref())
        });
        let (e, d) = inelastic_scatter(particle.energy, particle.dir, xs.awr, q_value, angle, rng);
        particle.energy = e;
        particle.dir = d;
        return CollisionOutcome::InelasticScatter {
            q_value_ev: q_value,
        };
    }

    cum += xs.n2n;
    if xi < cum {
        let secondaries = sample_multiplicity(particle, xs.awr, nuclide, 2, rng);
        return CollisionOutcome::Multiplicity { secondaries };
    }
    cum += xs.n3n;
    if xi < cum {
        let secondaries = sample_multiplicity(particle, xs.awr, nuclide, 3, rng);
        return CollisionOutcome::Multiplicity { secondaries };
    }

    cum += xs.fission;
    if xi < cum {
        let sites = sample_fission_sites(particle, xs, nuclide, rng);
        particle.kill();
        return CollisionOutcome::Fission { sites };
    }

    // Remainder: capture / absorption.
    particle.kill();
    CollisionOutcome::Absorption
}

/// Choose which discrete level was excited and return its Q-value.
fn sample_inelastic_level(
    energy: f64,
    _awr: f64,
    nuclide: &Nuclide,
    rng: &mut Pcg64,
) -> (f64, Option<usize>) {
    if nuclide.discrete_levels.is_empty() {
        // No level data → use a constant 45 keV-equivalent excitation.
        return (-45_000.0, None);
    }
    let mut xs_sum = 0.0_f64;
    let mut accessible: Vec<(usize, f64)> = Vec::new();
    for (i, level) in nuclide.discrete_levels.iter().enumerate() {
        if energy > level.threshold {
            let s = level.xs.lookup(energy).max(0.0);
            if s > 0.0 {
                xs_sum += s;
                accessible.push((i, s));
            }
        }
    }
    if accessible.is_empty() || xs_sum <= 0.0 {
        return (-45_000.0, None);
    }
    let xi = rng.uniform() * xs_sum;
    let mut cum = 0.0_f64;
    for &(idx, xs) in &accessible {
        cum += xs;
        if xi < cum {
            let level = &nuclide.discrete_levels[idx];
            return (level.q_value, Some(idx));
        }
    }
    // Safe: `accessible` is non-empty (the `is_empty()` early
    // return above guarantees that), so `last()` is `Some(_)`.
    let Some(&(last_idx, _)) = accessible.last() else {
        return (-45_000.0, None);
    };
    (nuclide.discrete_levels[last_idx].q_value, Some(last_idx))
}

/// Sample fission sites for a fission collision: how many secondaries
/// to bank, and where + with what energy.
fn sample_fission_sites(
    particle: &Particle,
    xs: &MicroXs,
    nuclide: &Nuclide,
    rng: &mut Pcg64,
) -> Vec<FissionSite> {
    let n_floor = xs.nu_bar.floor();
    let frac = xs.nu_bar - n_floor;
    let n_neutrons = if rng.uniform() < frac {
        n_floor as usize + 1
    } else {
        n_floor as usize
    };
    (0..n_neutrons)
        .map(|_| FissionSite {
            pos: particle.pos,
            energy: sample_fission_energy(particle.energy, nuclide, rng),
            weight: 1.0,
        })
        .collect()
}

fn sample_fission_energy(incident: f64, nuclide: &Nuclide, rng: &mut Pcg64) -> f64 {
    if let Some(d) = &nuclide.fission_energy_dist {
        return d.sample(incident, rng).max(1e-5);
    }
    // Fallback: U-235 thermal Watt parameters.
    crate::physics::spectra::watt_u235_thermal(rng)
}

fn sample_multiplicity(
    particle: &Particle,
    _awr: f64,
    nuclide: &Nuclide,
    multiplicity: usize,
    rng: &mut Pcg64,
) -> Vec<SecondaryNeutron> {
    // Outgoing energies from the channel's tabulated dist when
    // available, evaporation otherwise.
    let edist = match multiplicity {
        2 => nuclide.n2n_edist.as_ref(),
        3 => nuclide.n3n_edist.as_ref(),
        _ => None,
    };
    let mut out = Vec::with_capacity(multiplicity);
    for _ in 0..multiplicity {
        let e = match edist {
            Some(d) => d.sample(particle.energy, rng).max(1e-5),
            None => evaporation(particle.energy / 10.0, particle.energy, rng),
        };
        let (u, v, w) = rng.isotropic_direction();
        out.push(SecondaryNeutron {
            pos: particle.pos,
            dir: crate::geometry::Vec3::new(u, v, w),
            energy: e,
        });
    }
    out
}

/// Convenience wrapper for "do a collision in this material at this
/// particle's current energy". Picks the nuclide via
/// [`sample_nuclide_at_energy`], then [`process_collision`].
pub fn collide_in_material(
    particle: &mut Particle,
    material: &Material,
    rng: &mut Pcg64,
) -> Option<CollisionOutcome> {
    let (nuc_idx, xs) = sample_nuclide_at_energy(material, particle.energy, rng)?;
    let nuc = &material.nuclides[nuc_idx].0;
    Some(process_collision(
        particle,
        nuc,
        &xs,
        material.temperature_k,
        rng,
    ))
}

/// Bring `elastic_scatter` into scope so down-stream callers that
/// want a "no anisotropy, no thermal" path can call it directly via
/// `transport::collision::elastic_scatter_simple`.
pub fn elastic_scatter_simple(particle: &mut Particle, awr: f64, rng: &mut Pcg64) {
    let (e, d) = elastic_scatter(particle.energy, particle.dir, awr, rng);
    particle.energy = e;
    particle.dir = d;
}
