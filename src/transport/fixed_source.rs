//! Fixed-source neutron transport. Drop-in alternative to the
//! k-eigenvalue driver in [`crate::transport::simulate`] for problems
//! where the source is *not* a self-sustaining fission chain:
//!
//!   * shielding analysis (γ-streaming through structural barriers)
//!   * source-driven sub-criticals (ADS, accelerator targets)
//!   * activation / dose calculations
//!   * neutron irradiation experiments
//!
//! Each batch independently samples `n_particles_per_batch` source
//! neutrons from a user-supplied [`NeutronSource`] and tracks them
//! to extinction. There is no fission bank, no power iteration, and
//! no entropy convergence — every batch is a clean Monte Carlo
//! estimator. Tally accumulation is delegated to whichever
//! [`crate::tally::FluxTally`] (or other scorer) the caller passes
//! in; this module's job is just the transport.

use crate::geometry::surface::BoundaryCondition;
use crate::geometry::{Cell, Surface, Vec3, ray};
use crate::rng::Pcg64;
use crate::tally::FluxTally;
use crate::transport::collision::{CollisionOutcome, collide_in_material};
use crate::transport::material::Material;
use crate::transport::particle::{Particle, ParticleStatus};

/// Configuration for a fixed-source run.
pub struct FixedSourceConfig {
    pub n_batches: u32,
    pub n_particles_per_batch: u32,
    pub seed: u64,
}

impl Default for FixedSourceConfig {
    fn default() -> Self {
        Self {
            n_batches: 50,
            n_particles_per_batch: 10_000,
            seed: 1,
        }
    }
}

/// Source-particle sampler. Implementors return one fully-defined
/// `(pos, dir, energy, weight)` particle per call. Common
/// distributions are provided as concrete types in this module.
pub trait NeutronSource: Send + Sync {
    fn sample(&self, rng: &mut Pcg64) -> SourceParticle;
}

#[derive(Debug, Clone, Copy)]
pub struct SourceParticle {
    pub pos: Vec3,
    pub dir: Vec3,
    pub energy: f64,
    pub weight: f64,
}

/// Monoenergetic isotropic point source.
pub struct PointSource {
    pub pos: Vec3,
    pub energy_ev: f64,
}

impl NeutronSource for PointSource {
    fn sample(&self, rng: &mut Pcg64) -> SourceParticle {
        let (u, v, w) = rng.isotropic_direction();
        SourceParticle {
            pos: self.pos,
            dir: Vec3::new(u, v, w),
            energy: self.energy_ev,
            weight: 1.0,
        }
    }
}

/// Monoenergetic isotropic source uniformly sampled inside a box.
pub struct BoxSource {
    pub min: Vec3,
    pub max: Vec3,
    pub energy_ev: f64,
}

impl NeutronSource for BoxSource {
    fn sample(&self, rng: &mut Pcg64) -> SourceParticle {
        let x = self.min.x + rng.uniform() * (self.max.x - self.min.x);
        let y = self.min.y + rng.uniform() * (self.max.y - self.min.y);
        let z = self.min.z + rng.uniform() * (self.max.z - self.min.z);
        let (u, v, w) = rng.isotropic_direction();
        SourceParticle {
            pos: Vec3::new(x, y, z),
            dir: Vec3::new(u, v, w),
            energy: self.energy_ev,
            weight: 1.0,
        }
    }
}

/// Watt fission spectrum (U-235 thermal Cranberg parameters) at a
/// fixed point. Useful for activation / dose downstream of a fission
/// converter.
pub struct WattFissionSource {
    pub pos: Vec3,
}

impl NeutronSource for WattFissionSource {
    fn sample(&self, rng: &mut Pcg64) -> SourceParticle {
        let (u, v, w) = rng.isotropic_direction();
        SourceParticle {
            pos: self.pos,
            dir: Vec3::new(u, v, w),
            energy: crate::physics::spectra::watt_u235_thermal(rng),
            weight: 1.0,
        }
    }
}

/// Per-batch fixed-source result.
#[derive(Debug, Clone)]
pub struct FixedSourceBatch {
    pub batch: u32,
    pub n_collisions: u64,
    pub n_leaked: u64,
    pub n_absorbed: u64,
    pub n_fissioned: u64,
}

/// Aggregate fixed-source result.
#[derive(Debug, Clone)]
pub struct FixedSourceResult {
    pub batches: Vec<FixedSourceBatch>,
    pub total_histories: u64,
}

/// Run a fixed-source simulation.
///
/// `cell_material(cell_idx) → mat_idx` selects the material for each
/// geometry cell. The optional `tally` is `end_batch()`-rolled at the
/// end of every batch so per-bin σ is over independent batches.
#[allow(clippy::too_many_arguments)]
pub fn run_fixed_source<S: NeutronSource>(
    cfg: &FixedSourceConfig,
    cells: &[Cell],
    surfaces: &[Surface],
    cell_material: impl Fn(usize) -> usize,
    materials: &[Material],
    source: &S,
    tally: Option<&mut FluxTally>,
) -> FixedSourceResult {
    let mut rng = Pcg64::new(cfg.seed, 1);
    let mut batches = Vec::with_capacity(cfg.n_batches as usize);
    let mut tally_box = tally;

    for batch in 0..cfg.n_batches {
        let mut n_collisions = 0_u64;
        let mut n_leaked = 0_u64;
        let mut n_absorbed = 0_u64;
        let mut n_fissioned = 0_u64;

        for _ in 0..cfg.n_particles_per_batch {
            let s = source.sample(&mut rng);
            let cell_idx = ray::find_cell(s.pos, surfaces, cells).unwrap_or(0);
            let mut p = Particle::new(s.pos, s.dir, s.energy, cell_idx);
            p.weight = s.weight;
            transport_one(
                &mut p,
                cells,
                surfaces,
                &cell_material,
                materials,
                &mut rng,
                &mut n_collisions,
                &mut n_leaked,
                &mut n_absorbed,
                &mut n_fissioned,
                tally_box.as_deref_mut(),
            );
        }

        if let Some(t) = tally_box.as_deref_mut() {
            t.end_batch();
        }

        batches.push(FixedSourceBatch {
            batch,
            n_collisions,
            n_leaked,
            n_absorbed,
            n_fissioned,
        });
    }

    FixedSourceResult {
        total_histories: u64::from(cfg.n_batches) * u64::from(cfg.n_particles_per_batch),
        batches,
    }
}

#[allow(clippy::too_many_arguments)]
fn transport_one(
    p: &mut Particle,
    cells: &[Cell],
    surfaces: &[Surface],
    cell_material: &impl Fn(usize) -> usize,
    materials: &[Material],
    rng: &mut Pcg64,
    n_collisions: &mut u64,
    n_leaked: &mut u64,
    n_absorbed: &mut u64,
    n_fissioned: &mut u64,
    mut tally: Option<&mut FluxTally>,
) {
    while p.is_alive() {
        let mat_idx = cell_material(p.cell_idx);
        let material = &materials[mat_idx];
        let sigma_t = material.macro_total(p.energy).max(1e-30);
        let dist_collision = -rng.uniform().ln() / sigma_t;
        let trace = ray::trace_step(p.pos, p.dir, p.cell_idx, surfaces, cells);
        let dist_surface = trace.as_ref().map_or(f64::INFINITY, |h| h.distance);

        if dist_collision < dist_surface {
            // Score the track-length segment up to the collision site.
            if let Some(t) = tally.as_deref_mut() {
                t.score_track(p.cell_idx, p.energy, dist_collision, p.weight);
            }
            p.advance(dist_collision);
            let outcome = collide_in_material(p, material, rng);
            *n_collisions += 1;
            match outcome {
                Some(CollisionOutcome::Absorption) => {
                    *n_absorbed += 1;
                }
                Some(CollisionOutcome::Fission { .. }) => {
                    *n_fissioned += 1;
                    // Fixed-source mode: ignore the fission bank.
                    p.kill();
                }
                Some(CollisionOutcome::Multiplicity { .. }) => {
                    // Multiplicity secondaries are dropped in the
                    // simplest fixed-source model. Add them to a
                    // stack here if you need (n,2n) lensing.
                    p.kill();
                }
                _ => {}
            }
        } else if let Some(hit) = trace {
            if let Some(t) = tally.as_deref_mut() {
                t.score_track(p.cell_idx, p.energy, hit.distance, p.weight);
            }
            p.advance(hit.distance + 1e-10);
            match surfaces[hit.surface_idx].boundary_condition() {
                BoundaryCondition::Vacuum => {
                    *n_leaked += 1;
                    p.kill();
                }
                BoundaryCondition::Reflective => {
                    let n = surfaces[hit.surface_idx].normal_at(p.pos);
                    let dot = p.dir.dot(n);
                    p.dir = Vec3::new(
                        p.dir.x - 2.0 * dot * n.x,
                        p.dir.y - 2.0 * dot * n.y,
                        p.dir.z - 2.0 * dot * n.z,
                    );
                    p.advance(2e-10);
                    p.status = ParticleStatus::Alive;
                    if let Some(c) = ray::find_cell(p.pos, surfaces, cells) {
                        p.cell_idx = c;
                    }
                }
                BoundaryCondition::Transmission => {
                    if let Some(c) = ray::find_cell(p.pos, surfaces, cells) {
                        p.cell_idx = c;
                    } else {
                        *n_leaked += 1;
                        p.kill();
                    }
                }
            }
        } else {
            *n_leaked += 1;
            p.kill();
        }
    }
}
