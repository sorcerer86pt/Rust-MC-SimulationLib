#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end k-eigenvalue power iteration on a bare critical sphere
//! ("Godiva-like") using synthetic, energy-independent cross
//! sections. The point is to exercise the *full* transport pipeline
//! shipped in this crate end-to-end:
//!
//!   * Geometry: one [`Surface::Sphere`] with a [`BoundaryCondition::Vacuum`]
//!     boundary, a single [`Cell`] inside it, no leakage path back.
//!   * Nuclide: just `elastic`, `fission`, `capture` constant-XS
//!     pointwise tables + a constant ν̄ = 2.43.
//!   * Material: 100 % U-235 at metallic density (~4.788e-2 a/b·cm).
//!   * Source: 64 isotropic source points in a small box at the
//!     centre, 1 MeV.
//!   * Driver: [`run_eigenvalue`] with collision-estimator k-eff and
//!     Shannon-entropy convergence on a 6³ mesh.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --example 09_keigenvalue_godiva
//! ```

use std::sync::Arc;

use rust_mc_sim::geometry::cell::{CellFill, inside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::{Aabb, Cell, CellId, Surface, Vec3};
use rust_mc_sim::table::PointwiseTable;
use rust_mc_sim::transport::material::{Material, Nuclide};
use rust_mc_sim::transport::simulate::{EigenvalueConfig, run_eigenvalue};

fn main() {
    // 1) Synthetic U-235 nuclide. Energy-independent XS, νbar = 2.43.
    let energies = vec![1.0e-5, 1.0e7];
    let mut nuc = Nuclide::empty("U235_synth", 233.0);
    nuc.elastic = Some(PointwiseTable::new(energies.clone(), vec![4.0, 4.0]));
    nuc.fission = Some(PointwiseTable::new(energies.clone(), vec![1.4, 1.4]));
    nuc.capture = Some(PointwiseTable::new(energies, vec![0.6, 0.6]));
    nuc.nu_bar_const = 2.43;
    let nuc = Arc::new(nuc);

    // 2) Material: pure U-235 at metallic density.
    let mut mat = Material::new("Uranium", 293.6);
    mat.add(nuc, 4.788e-2);

    // 3) Geometry: sphere of radius 8.74 cm with vacuum boundary.
    let radius_cm = 8.74_f64;
    let surfaces = vec![Surface::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: radius_cm,
        bc: BoundaryCondition::Vacuum,
    }];
    let cells = vec![
        Cell::new(CellId(0), inside(0), CellFill::Material(0)).with_aabb(Aabb {
            min: Vec3::new(-radius_cm, -radius_cm, -radius_cm),
            max: Vec3::new(radius_cm, radius_cm, radius_cm),
        }),
    ];
    let materials = [mat];

    // 4) Run a small k-eigenvalue power iteration. With synthetic
    //    XS at this radius and density, k will land somewhere near
    //    1, but the headline result is *that the loop runs*: the
    //    Shannon entropy and k-collision values stabilise.
    let cfg = EigenvalueConfig {
        n_batches: 60,
        n_inactive: 20,
        n_particles_per_batch: 2_000,
        seed: 42,
    };

    println!(
        "Bare U-235 sphere, R = {radius_cm} cm, {} batches × {} particles, vacuum BC",
        cfg.n_batches, cfg.n_particles_per_batch
    );

    let half = radius_cm * 0.5;
    let result = run_eigenvalue(
        &cfg,
        &cells,
        &surfaces,
        |_| 0,
        &materials,
        ([-half, -half, -half], [half, half, half]),
        1.0e6,
    );

    println!("\n  batch │  k_collision │ entropy(bits) │  collisions │ leaked");
    println!("  ──────┼──────────────┼───────────────┼─────────────┼───────");
    for b in &result.batch_history {
        println!(
            "  {:>5} │   {:.5}    │     {:.3}     │ {:>11} │ {:>5}",
            b.batch, b.k_collision, b.source_entropy, b.n_collisions, b.n_leaked
        );
    }
    println!(
        "\n  k_eff (active batches) = {:.5} ± {:.5}",
        result.k_mean, result.k_sigma
    );
}
