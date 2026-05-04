//! k-eigenvalue power-iteration driver. One geometry + one material
//! per cell + a fission source bank — runs `n_batches` of
//! `n_particles` each, accumulates k-eff statistics, returns the
//! per-batch history.
//!
//! Geometry coupling: the caller provides a `cell_material` lookup
//! that maps a [`crate::geometry::Cell`] index to a
//! [`crate::transport::material::Material`] index. Boundary
//! conditions are read from each [`crate::geometry::Surface`]'s
//! [`crate::geometry::surface::BoundaryCondition`].

use crate::geometry::surface::BoundaryCondition;
use crate::geometry::{ray, Cell, Surface, Vec3};
use crate::rng::Pcg64;
use crate::transport::collision::{collide_in_material, CollisionOutcome};
use crate::transport::material::Material;
use crate::transport::particle::{FissionBank, FissionSite, Particle, ParticleStatus};

/// Configuration for one k-eigenvalue run.
pub struct EigenvalueConfig {
    pub n_batches: u32,
    pub n_inactive: u32,
    pub n_particles_per_batch: u32,
    pub seed: u64,
}

impl Default for EigenvalueConfig {
    fn default() -> Self {
        Self {
            n_batches: 100,
            n_inactive: 20,
            n_particles_per_batch: 10_000,
            seed: 1,
        }
    }
}

/// Per-batch result.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch: u32,
    pub k_collision: f64,
    pub source_entropy: f64,
    pub n_collisions: u64,
    pub n_fissions: u64,
    pub n_leaked: u64,
}

/// Final k-eff aggregate over the active batches.
#[derive(Debug, Clone)]
pub struct EigenvalueResult {
    pub k_mean: f64,
    pub k_sigma: f64,
    pub batch_history: Vec<BatchResult>,
}

/// Run a k-eigenvalue power iteration.
///
/// `cell_material(cell_idx) → mat_idx` selects the material for each
/// geometry cell. The initial source is built by sampling
/// `n_particles_per_batch` uniform points inside `source_box`, each
/// with energy `source_energy_eV` and isotropic direction.
#[allow(clippy::too_many_arguments)]
pub fn run_eigenvalue(
    cfg: &EigenvalueConfig,
    cells: &[Cell],
    surfaces: &[Surface],
    cell_material: impl Fn(usize) -> usize,
    materials: &[Material],
    source_box: ([f64; 3], [f64; 3]),
    source_energy_ev: f64,
) -> EigenvalueResult {
    let mut rng = Pcg64::new(cfg.seed, 1);
    let mut source = build_initial_source(
        cfg.n_particles_per_batch as usize,
        cells,
        surfaces,
        source_box,
        source_energy_ev,
        &mut rng,
    );
    let mut history = Vec::with_capacity(cfg.n_batches as usize);
    let mut k_active_sum = 0.0_f64;
    let mut k_active_sum_sq = 0.0_f64;
    let mut n_active = 0_u32;

    for batch in 0..cfg.n_batches {
        let mut bank = FissionBank::new();
        let mut n_collisions = 0_u64;
        let mut n_fissions = 0_u64;
        let mut n_leaked = 0_u64;
        let mut weight_in = 0.0_f64;
        let mut nu_sum = 0.0_f64;

        for site in &source {
            weight_in += site.weight;
            let (u, v, w) = rng.isotropic_direction();
            let initial_dir = Vec3::new(u, v, w);
            let cell_idx = ray::find_cell(site.pos, surfaces, cells).unwrap_or(0);
            let mut p = Particle::new(site.pos, initial_dir, site.energy, cell_idx);
            transport_one(
                &mut p,
                cells,
                surfaces,
                &cell_material,
                materials,
                &mut rng,
                &mut n_collisions,
                &mut n_fissions,
                &mut nu_sum,
                &mut bank,
                &mut n_leaked,
            );
        }

        // Collision-estimator k-eff: ⟨ν · σ_f / σ_t⟩ accumulated as
        // nu_sum / N_in for this batch.
        let k_collision = if weight_in > 0.0 {
            nu_sum / weight_in
        } else {
            0.0
        };

        // Build next source.
        let next_source: Vec<FissionSite> = if bank.sites.is_empty() {
            source.clone()
        } else {
            let n_target = cfg.n_particles_per_batch as usize;
            let n_pool = bank.sites.len();
            (0..n_target)
                .map(|_| {
                    let mut idx = (rng.uniform() * n_pool as f64) as usize;
                    if idx >= n_pool {
                        idx = n_pool - 1;
                    }
                    bank.sites[idx].clone()
                })
                .collect()
        };

        let entropy = shannon_entropy(&next_source, source_box, 6);
        let res = BatchResult {
            batch,
            k_collision,
            source_entropy: entropy,
            n_collisions,
            n_fissions,
            n_leaked,
        };
        history.push(res);

        if batch >= cfg.n_inactive {
            k_active_sum += k_collision;
            k_active_sum_sq += k_collision * k_collision;
            n_active += 1;
        }
        source = next_source;
    }

    let k_mean = if n_active > 0 {
        k_active_sum / n_active as f64
    } else {
        0.0
    };
    let k_sigma = if n_active > 1 {
        let n = n_active as f64;
        let var = (k_active_sum_sq - k_active_sum * k_active_sum / n) / (n - 1.0);
        var.max(0.0).sqrt() / n.sqrt()
    } else {
        0.0
    };
    EigenvalueResult {
        k_mean,
        k_sigma,
        batch_history: history,
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
    n_fissions: &mut u64,
    nu_sum: &mut f64,
    bank: &mut FissionBank,
    n_leaked: &mut u64,
) {
    while p.is_alive() {
        let mat_idx = cell_material(p.cell_idx);
        let material = &materials[mat_idx];
        let sigma_t = material.macro_total(p.energy).max(1e-30);
        let dist_collision = -rng.uniform().ln() / sigma_t;
        let trace = ray::trace_step(p.pos, p.dir, p.cell_idx, surfaces, cells);
        let dist_surface = trace.as_ref().map_or(f64::INFINITY, |h| h.distance);

        if dist_collision < dist_surface {
            p.advance(dist_collision);
            let outcome = collide_in_material(p, material, rng);
            *n_collisions += 1;
            match outcome {
                Some(CollisionOutcome::Fission { sites }) => {
                    *n_fissions += 1;
                    *nu_sum += sites.len() as f64;
                    for s in sites {
                        bank.push(s);
                    }
                }
                Some(CollisionOutcome::Multiplicity { secondaries }) => {
                    for s in secondaries {
                        let mut sec_p =
                            Particle::new(s.pos, s.dir, s.energy, p.cell_idx);
                        transport_secondary(
                            &mut sec_p,
                            cells,
                            surfaces,
                            cell_material,
                            materials,
                            rng,
                            n_collisions,
                            n_fissions,
                            nu_sum,
                            bank,
                            n_leaked,
                        );
                    }
                }
                _ => {}
            }
        } else if let Some(hit) = trace {
            // Cross the surface (with a small nudge to clear it).
            p.advance(hit.distance + 1e-10);
            let bc = surfaces[hit.surface_idx].boundary_condition();
            match bc {
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
                    // Step a little so we're inside the new cell.
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
                        // Outside geometry — vacuum semantics.
                        *n_leaked += 1;
                        p.kill();
                    }
                }
            }
        } else {
            // No surface hit anywhere — escape.
            *n_leaked += 1;
            p.kill();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transport_secondary(
    p: &mut Particle,
    cells: &[Cell],
    surfaces: &[Surface],
    cell_material: &impl Fn(usize) -> usize,
    materials: &[Material],
    rng: &mut Pcg64,
    n_collisions: &mut u64,
    n_fissions: &mut u64,
    nu_sum: &mut f64,
    bank: &mut FissionBank,
    n_leaked: &mut u64,
) {
    while p.is_alive() {
        let mat_idx = cell_material(p.cell_idx);
        let material = &materials[mat_idx];
        let sigma_t = material.macro_total(p.energy).max(1e-30);
        let dist_collision = -rng.uniform().ln() / sigma_t;
        let trace = ray::trace_step(p.pos, p.dir, p.cell_idx, surfaces, cells);
        let dist_surface = trace.as_ref().map_or(f64::INFINITY, |h| h.distance);
        if dist_collision < dist_surface {
            p.advance(dist_collision);
            let outcome = collide_in_material(p, material, rng);
            *n_collisions += 1;
            match outcome {
                Some(CollisionOutcome::Fission { sites }) => {
                    *n_fissions += 1;
                    *nu_sum += sites.len() as f64;
                    for s in sites {
                        bank.push(s);
                    }
                }
                Some(CollisionOutcome::Multiplicity { .. }) => {
                    // Don't recurse arbitrarily.
                    p.kill();
                }
                _ => {}
            }
        } else if let Some(hit) = trace {
            p.advance(hit.distance + 1e-10);
            let bc = surfaces[hit.surface_idx].boundary_condition();
            match bc {
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

fn build_initial_source(
    n: usize,
    cells: &[Cell],
    surfaces: &[Surface],
    source_box: ([f64; 3], [f64; 3]),
    energy: f64,
    rng: &mut Pcg64,
) -> Vec<FissionSite> {
    let ([xmin, ymin, zmin], [xmax, ymax, zmax]) = source_box;
    let mut sites = Vec::with_capacity(n);
    let mut tries = 0;
    while sites.len() < n {
        let x = xmin + rng.uniform() * (xmax - xmin);
        let y = ymin + rng.uniform() * (ymax - ymin);
        let z = zmin + rng.uniform() * (zmax - zmin);
        let pos = Vec3::new(x, y, z);
        if ray::find_cell(pos, surfaces, cells).is_some() {
            sites.push(FissionSite {
                pos,
                energy,
                weight: 1.0,
            });
        }
        tries += 1;
        if tries > 100 * n {
            // Saturate: caller's source_box doesn't intersect any
            // cell. Return whatever we have rather than spin.
            break;
        }
    }
    sites
}

/// Shannon entropy of the source spatial distribution on an
/// `n_bins`³ Cartesian mesh over `source_box`. Returns the entropy
/// in bits; max value is `3·log₂(n_bins)` for a uniform source.
pub fn shannon_entropy(
    sites: &[FissionSite],
    source_box: ([f64; 3], [f64; 3]),
    n_bins: usize,
) -> f64 {
    let ([xmin, ymin, zmin], [xmax, ymax, zmax]) = source_box;
    let dx = (xmax - xmin).max(1e-30);
    let dy = (ymax - ymin).max(1e-30);
    let dz = (zmax - zmin).max(1e-30);
    let total_bins = n_bins * n_bins * n_bins;
    let mut counts = vec![0_u32; total_bins];
    let n_total = sites.len();
    if n_total == 0 {
        return 0.0;
    }
    for s in sites {
        let i = (((s.pos.x - xmin) / dx * n_bins as f64).floor() as usize).min(n_bins - 1);
        let j = (((s.pos.y - ymin) / dy * n_bins as f64).floor() as usize).min(n_bins - 1);
        let k = (((s.pos.z - zmin) / dz * n_bins as f64).floor() as usize).min(n_bins - 1);
        counts[i * n_bins * n_bins + j * n_bins + k] += 1;
    }
    let mut h = 0.0_f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n_total as f64;
            h -= p * p.log2();
        }
    }
    h
}
