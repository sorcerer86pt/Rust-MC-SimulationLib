//! Photon fixed-source driver. Same skeleton as
//! `transport::fixed_source::run_fixed_source` for neutrons, with the
//! collision dispatch swapped for the photon kernels in
//! [`crate::photon::interactions`]. Supports a stack of pending
//! photons per primary so pair-production annihilation γ's are
//! tracked to extinction.

use crate::geometry::bvh::Bvh;
use crate::geometry::surface::BoundaryCondition;
use crate::geometry::{Cell, Surface, Vec3, ray};
use crate::photon::interactions::{
    PAIR_THRESHOLD_EV, PhotonReaction, sample_coherent_forward, sample_compton_free,
    sample_pair, sample_photoelectric,
};
use crate::photon::material::PhotonMaterial;
use crate::photon::source::{PhotonSource, SourcePhoton};
use crate::physics::scatter::rotate_direction;
use crate::rng::Pcg64;
use crate::tally::FluxTally;

#[derive(Debug, Clone, Copy)]
pub struct PhotonFixedSourceConfig {
    pub n_batches: u32,
    pub n_particles_per_batch: u32,
    pub seed: u64,
    /// Lowest energy a photon is tracked to (eV). Below this, photons
    /// deposit their remaining energy locally and die.
    pub energy_cutoff_ev: f64,
}

impl Default for PhotonFixedSourceConfig {
    fn default() -> Self {
        Self {
            n_batches: 50,
            n_particles_per_batch: 5_000,
            seed: 1,
            energy_cutoff_ev: 1.0e3, // 1 keV
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhotonBatch {
    pub batch: u32,
    pub n_collisions: u64,
    pub n_leaked: u64,
    pub n_absorbed: u64,
    pub total_energy_deposited: f64,
}

#[derive(Debug, Clone)]
pub struct PhotonFixedSourceResult {
    pub batches: Vec<PhotonBatch>,
    pub total_histories: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActivePhoton {
    pos: Vec3,
    dir: Vec3,
    energy: f64,
    weight: f64,
    cell_idx: usize,
}

/// Run a photon fixed-source simulation.
#[allow(clippy::too_many_arguments)]
pub fn run_photon_fixed_source<S: PhotonSource>(
    cfg: &PhotonFixedSourceConfig,
    cells: &[Cell],
    surfaces: &[Surface],
    cell_material: impl Fn(usize) -> usize,
    materials: &[PhotonMaterial],
    source: &S,
    tally: Option<&mut FluxTally>,
) -> PhotonFixedSourceResult {
    let mut rng = Pcg64::new(cfg.seed, 1);
    let bvh = Bvh::build(cells);
    let mut batches = Vec::with_capacity(cfg.n_batches as usize);
    let mut tally_box = tally;

    for batch in 0..cfg.n_batches {
        let mut n_collisions = 0_u64;
        let mut n_leaked = 0_u64;
        let mut n_absorbed = 0_u64;
        let mut total_energy_deposited = 0.0_f64;

        for _ in 0..cfg.n_particles_per_batch {
            let s: SourcePhoton = source.sample(&mut rng);
            let cell_idx =
                ray::find_cell_bvh(s.pos, surfaces, cells, &bvh).unwrap_or(0);
            let mut stack: Vec<ActivePhoton> = vec![ActivePhoton {
                pos: s.pos,
                dir: s.dir,
                energy: s.energy,
                weight: s.weight,
                cell_idx,
            }];

            while let Some(mut p) = stack.pop() {
                transport_one(
                    &mut p,
                    cells,
                    surfaces,
                    &bvh,
                    &cell_material,
                    materials,
                    cfg.energy_cutoff_ev,
                    &mut rng,
                    &mut stack,
                    &mut n_collisions,
                    &mut n_leaked,
                    &mut n_absorbed,
                    &mut total_energy_deposited,
                    tally_box.as_deref_mut(),
                );
            }
        }

        if let Some(t) = tally_box.as_deref_mut() {
            t.end_batch();
        }

        batches.push(PhotonBatch {
            batch,
            n_collisions,
            n_leaked,
            n_absorbed,
            total_energy_deposited,
        });
    }

    PhotonFixedSourceResult {
        total_histories: u64::from(cfg.n_batches) * u64::from(cfg.n_particles_per_batch),
        batches,
    }
}

#[allow(clippy::too_many_arguments)]
fn transport_one(
    p: &mut ActivePhoton,
    cells: &[Cell],
    surfaces: &[Surface],
    bvh: &Bvh,
    cell_material: &impl Fn(usize) -> usize,
    materials: &[PhotonMaterial],
    energy_cutoff: f64,
    rng: &mut Pcg64,
    stack: &mut Vec<ActivePhoton>,
    n_collisions: &mut u64,
    n_leaked: &mut u64,
    n_absorbed: &mut u64,
    total_energy_deposited: &mut f64,
    mut tally: Option<&mut FluxTally>,
) {
    loop {
        if p.energy < energy_cutoff {
            *total_energy_deposited += p.weight * p.energy;
            *n_absorbed += 1;
            return;
        }
        let mat_idx = cell_material(p.cell_idx);
        let material = &materials[mat_idx];
        let sigma_t = material.macro_total(p.energy).max(1e-30);
        let dist_collision = -rng.uniform().ln() / sigma_t;
        let trace = ray::trace_step_opt(p.pos, p.dir, p.cell_idx, surfaces, cells, Some(bvh));
        let dist_surface = trace.as_ref().map_or(f64::INFINITY, |h| h.distance);

        if dist_collision < dist_surface {
            if let Some(t) = tally.as_deref_mut() {
                t.score_track(p.cell_idx, p.energy, dist_collision, p.weight);
            }
            p.pos = p.pos + p.dir * dist_collision;
            *n_collisions += 1;
            let reaction = match material.sample_reaction(p.energy, rng) {
                Some(r) => r,
                None => return,
            };
            match reaction {
                PhotonReaction::Coherent => {
                    let o = sample_coherent_forward(p.energy);
                    p.energy = o.energy_out;
                    // μ = 1 → no direction change.
                }
                PhotonReaction::Incoherent => {
                    let o = sample_compton_free(p.energy, rng);
                    *total_energy_deposited += p.weight * o.local_deposition;
                    p.dir = rotate_direction(p.dir, o.mu, rng);
                    p.energy = o.energy_out;
                }
                PhotonReaction::Photoelectric => {
                    let o = sample_photoelectric(p.energy);
                    *total_energy_deposited += p.weight * o.local_deposition;
                    *n_absorbed += 1;
                    return;
                }
                PhotonReaction::PairProduction => {
                    if p.energy < PAIR_THRESHOLD_EV {
                        // Should not happen; treat as photoelectric
                        // for safety.
                        *total_energy_deposited += p.weight * p.energy;
                        *n_absorbed += 1;
                        return;
                    }
                    let o = sample_pair(p.energy, rng).expect("above threshold");
                    *total_energy_deposited += p.weight * o.local_deposition;
                    // Bank both 511 keV annihilation γ's. Pick one
                    // isotropic axis; the second is back-to-back.
                    for (i, (e_anni, _)) in o.annihilation_photons.iter().enumerate() {
                        let (u, v, w) = rng.isotropic_direction();
                        let mut dir = Vec3::new(u, v, w);
                        if i == 1 {
                            dir = Vec3::new(-dir.x, -dir.y, -dir.z);
                        }
                        stack.push(ActivePhoton {
                            pos: p.pos,
                            dir,
                            energy: *e_anni,
                            weight: p.weight,
                            cell_idx: p.cell_idx,
                        });
                    }
                    *n_absorbed += 1;
                    return;
                }
            }
        } else if let Some(hit) = trace {
            if let Some(t) = tally.as_deref_mut() {
                t.score_track(p.cell_idx, p.energy, hit.distance, p.weight);
            }
            p.pos = p.pos + p.dir * (hit.distance + 1e-10);
            match surfaces[hit.surface_idx].boundary_condition() {
                BoundaryCondition::Vacuum => {
                    *n_leaked += 1;
                    return;
                }
                BoundaryCondition::Reflective => {
                    let n = surfaces[hit.surface_idx].normal_at(p.pos);
                    let dot = p.dir.dot(n);
                    p.dir = Vec3::new(
                        p.dir.x - 2.0 * dot * n.x,
                        p.dir.y - 2.0 * dot * n.y,
                        p.dir.z - 2.0 * dot * n.z,
                    );
                    p.pos = p.pos + p.dir * 2.0e-10;
                    if let Some(c) = ray::find_cell_bvh(p.pos, surfaces, cells, bvh) {
                        p.cell_idx = c;
                    }
                }
                BoundaryCondition::Transmission => {
                    if let Some(c) = ray::find_cell_bvh(p.pos, surfaces, cells, bvh) {
                        p.cell_idx = c;
                    } else {
                        *n_leaked += 1;
                        return;
                    }
                }
            }
        } else {
            *n_leaked += 1;
            return;
        }
    }
}
