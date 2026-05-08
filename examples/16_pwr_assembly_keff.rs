#![allow(clippy::unwrap_used, clippy::expect_used)]
//! 17 × 17 PWR fuel assembly k_∞ benchmark — full ENDF/B-VII.1
//! physics on a Westinghouse / CP1 assembly footprint with 24
//! water-filled guide-tube positions and 1 central instrumentation
//! thimble. Same nuclide set and S(α,β) kernel as
//! `examples/10_pwr_pin_cell.rs`; cell count is ≈ 530 (vs 3 for
//! the pin cell), which is the regime where the BVH wiring buys
//! most of its speedup.
//!
//! With the `preview` feature on, a window opens showing the
//! geometry before the simulation starts — close it (Esc / X) and
//! the eigenvalue run begins. With only the `nuclear` feature the
//! preview is dropped and the run starts immediately.
//!
//! Run with (preview + simulation):
//!
//! ```bash
//! cargo run --release --features "nuclear preview" \
//!     --example 16_pwr_assembly_keff -- \
//!     /path/to/endfb-vii.1-hdf5/neutron
//! ```

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rust_mc_sim::geometry::Surface;
use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, Region, between, inside, outside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::nuclear::loader::{
    LoaderConfig, attach_thermal_scattering, load_nuclide_from_hdf5, load_thermal_scattering,
};
use rust_mc_sim::transport::material::{Material, Nuclide};
use rust_mc_sim::transport::simulate::{EigenvalueConfig, run_eigenvalue};

const PITCH: f64 = 1.260;
const N_PIN_SIDE: i32 = 17;
const HALF_ASSEMBLY: f64 = (N_PIN_SIDE as f64) * PITCH * 0.5; // 10.71 cm
const PELLET_R: f64 = 0.4096;
const CLAD_OR: f64 = 0.4750;

const T_FUEL: f64 = 900.0;
const T_CLAD: f64 = 600.0;
const T_MOD: f64 = 600.0;

const N_U235_FUEL: f64 = 8.65e-4;
const N_U238_FUEL: f64 = 2.225e-2;
const N_O16_FUEL: f64 = 4.624e-2;
const N_ZR_TOTAL: f64 = 4.32e-2;
const F_ZR90: f64 = 0.5145;
const F_ZR91: f64 = 0.1122;
const F_ZR92: f64 = 0.1715;
const F_ZR94: f64 = 0.1738;
const N_H1_MOD: f64 = 4.836e-2;
const N_O16_MOD: f64 = 2.418e-2;

// Westinghouse 17 × 17 guide-tube + instrumentation pattern (1-indexed
// row/col converted to 0-indexed). 24 guide tubes + central
// instrumentation thimble = 25 non-fuel positions.
const NON_FUEL_POSITIONS: &[(i32, i32)] = &[
    (2, 5),
    (2, 8),
    (2, 11),
    (3, 3),
    (3, 13),
    (5, 2),
    (5, 5),
    (5, 8),
    (5, 11),
    (5, 14),
    (8, 2),
    (8, 5),
    (8, 8),
    (8, 11),
    (8, 14),
    (11, 2),
    (11, 5),
    (11, 8),
    (11, 11),
    (11, 14),
    (13, 3),
    (13, 13),
    (14, 5),
    (14, 8),
    (14, 11),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let data_dir = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!(
                "usage: 16_pwr_assembly_keff <data_dir>\n\n\
                 <data_dir> is the directory containing the per-nuclide\n\
                 OpenMC HDF5 files (U235.h5, …, c_H_in_H2O.h5)."
            );
            std::process::exit(2);
        }
    };

    println!(
        "17 × 17 Westinghouse PWR fuel assembly — ENDF/B-VII.1 HDF5\n\
         data dir: {}\n",
        data_dir.display()
    );

    // ── Nuclides ───────────────────────────────────────────────────
    let lcfg = LoaderConfig::default();
    let t0 = Instant::now();
    let u235 = Arc::new(load("U235", &data_dir, T_FUEL, &lcfg)?);
    let u238 = Arc::new(load("U238", &data_dir, T_FUEL, &lcfg)?);
    let o16_fuel = Arc::new(load("O16", &data_dir, T_FUEL, &lcfg)?);
    let h1_free = load("H1", &data_dir, T_MOD, &lcfg)?;
    let kernel = load_thermal_scattering(&data_dir.join("c_H_in_H2O.h5"))?;
    let h1 = Arc::new(attach_thermal_scattering(h1_free, kernel));
    let o16_mod = Arc::new(load("O16", &data_dir, T_MOD, &lcfg)?);
    let zr90 = Arc::new(load("Zr90", &data_dir, T_CLAD, &lcfg)?);
    let zr91 = Arc::new(load("Zr91", &data_dir, T_CLAD, &lcfg)?);
    let zr92 = Arc::new(load("Zr92", &data_dir, T_CLAD, &lcfg)?);
    let zr94 = Arc::new(load("Zr94", &data_dir, T_CLAD, &lcfg)?);
    let load_s = t0.elapsed().as_secs_f64();
    println!("loaded 9 nuclides + S(α,β) in {load_s:.1} s");

    // ── Materials ──────────────────────────────────────────────────
    let mut fuel = Material::new("UO2 fuel (3.7 % ²³⁵U)", T_FUEL);
    fuel.add(u235, N_U235_FUEL);
    fuel.add(u238, N_U238_FUEL);
    fuel.add(o16_fuel, N_O16_FUEL);

    let mut clad = Material::new("Zircaloy-4 clad", T_CLAD);
    clad.add(zr90, N_ZR_TOTAL * F_ZR90);
    clad.add(zr91, N_ZR_TOTAL * F_ZR91);
    clad.add(zr92, N_ZR_TOTAL * F_ZR92);
    clad.add(zr94, N_ZR_TOTAL * F_ZR94);

    let mut moderator = Material::new("light water + S(α,β)", T_MOD);
    moderator.add(h1, N_H1_MOD);
    moderator.add(o16_mod, N_O16_MOD);
    let materials = [fuel, clad, moderator];

    // ── Geometry: 4 pitch planes + per-pin cylinders ───────────────
    let mut surfaces: Vec<Surface> = Vec::new();
    let s_xn = push_surface(
        &mut surfaces,
        Surface::PlaneX {
            x0: -HALF_ASSEMBLY,
            bc: BoundaryCondition::Reflective,
        },
    );
    let s_xp = push_surface(
        &mut surfaces,
        Surface::PlaneX {
            x0: HALF_ASSEMBLY,
            bc: BoundaryCondition::Reflective,
        },
    );
    let s_yn = push_surface(
        &mut surfaces,
        Surface::PlaneY {
            y0: -HALF_ASSEMBLY,
            bc: BoundaryCondition::Reflective,
        },
    );
    let s_yp = push_surface(
        &mut surfaces,
        Surface::PlaneY {
            y0: HALF_ASSEMBLY,
            bc: BoundaryCondition::Reflective,
        },
    );

    let mut cells: Vec<Cell> = Vec::new();
    let mut cell_materials: Vec<usize> = Vec::new();
    let mut clad_surfaces: Vec<usize> = Vec::new();
    let mut n_fuel_pins = 0;

    for j in 0..N_PIN_SIDE {
        for i in 0..N_PIN_SIDE {
            if NON_FUEL_POSITIONS.contains(&(i, j)) {
                continue;
            }
            n_fuel_pins += 1;
            let cx = (i as f64 - (N_PIN_SIDE as f64 - 1.0) * 0.5) * PITCH;
            let cy = (j as f64 - (N_PIN_SIDE as f64 - 1.0) * 0.5) * PITCH;
            let s_pellet = push_surface(
                &mut surfaces,
                Surface::CylinderZ {
                    center_x: cx,
                    center_y: cy,
                    radius: PELLET_R,
                    bc: BoundaryCondition::Transmission,
                },
            );
            let s_clad = push_surface(
                &mut surfaces,
                Surface::CylinderZ {
                    center_x: cx,
                    center_y: cy,
                    radius: CLAD_OR,
                    bc: BoundaryCondition::Transmission,
                },
            );
            clad_surfaces.push(s_clad);
            cells.push(
                Cell::new(
                    CellId(cells.len() as u32),
                    inside(s_pellet),
                    CellFill::Material(0),
                )
                .with_aabb_from_region(&surfaces),
            );
            cell_materials.push(0);
            cells.push(
                Cell::new(
                    CellId(cells.len() as u32),
                    between(s_pellet, s_clad),
                    CellFill::Material(1),
                )
                .with_aabb_from_region(&surfaces),
            );
            cell_materials.push(1);
        }
    }

    // Moderator = inside the assembly box ∩ outside every clad.
    let inside_box =
        Region::Intersection(Box::new(between(s_xn, s_xp)), Box::new(between(s_yn, s_yp)));
    let mut mod_region = inside_box;
    for &c in &clad_surfaces {
        mod_region = Region::Intersection(Box::new(mod_region), Box::new(outside(c)));
    }
    cells.push(
        Cell::new(
            CellId(cells.len() as u32),
            mod_region,
            CellFill::Material(2),
        )
        .with_aabb_from_region(&surfaces),
    );
    cell_materials.push(2);

    println!(
        "geometry: {n_fuel_pins} fuel pins, {} guide-tube + 1 instrumentation positions (water-filled),\n          {} surfaces, {} cells",
        NON_FUEL_POSITIONS.len(),
        surfaces.len(),
        cells.len()
    );

    // Optional preview pass — when the `preview` feature is enabled
    // this opens a window so you can verify the geometry before the
    // (slow) simulation starts. Close the window to continue.
    #[cfg(feature = "preview")]
    {
        use rust_mc_sim::preview::{Viewport, preview_geometry};
        println!("\nopening preview window… close (Esc / X) to start the simulation.");
        preview_geometry(
            Viewport::square_centered(HALF_ASSEMBLY * 1.15, 0.0, 800),
            "rust-mc-sim — 17×17 PWR assembly preview",
            &cells,
            &surfaces,
            &materials,
            |idx| cell_materials[idx],
            None,
        );
    }

    let cfg = EigenvalueConfig {
        n_batches: 30,
        n_inactive: 10,
        n_particles_per_batch: 2_000,
        seed: 7,
    };
    println!(
        "driver: {} batches × {} particles, {} inactive, reflective walls (infinite-lattice k_∞)\n",
        cfg.n_batches, cfg.n_particles_per_batch, cfg.n_inactive
    );

    let sim_t0 = Instant::now();
    let result = run_eigenvalue(
        &cfg,
        &cells,
        &surfaces,
        |cell_idx| cell_materials[cell_idx],
        &materials,
        (
            [-HALF_ASSEMBLY * 0.6, -HALF_ASSEMBLY * 0.6, -0.5],
            [HALF_ASSEMBLY * 0.6, HALF_ASSEMBLY * 0.6, 0.5],
        ),
        2.0e6,
    );
    let sim_s = sim_t0.elapsed().as_secs_f64();

    let n_active_hist: u64 = result
        .batch_history
        .iter()
        .filter(|b| b.batch >= cfg.n_inactive)
        .map(|_| u64::from(cfg.n_particles_per_batch))
        .sum();
    let ns_per_history = (sim_s * 1.0e9) / (n_active_hist.max(1) as f64);
    let total_collisions: u64 = result.batch_history.iter().map(|b| b.n_collisions).sum();
    let collisions_per_history = total_collisions as f64
        / (u64::from(cfg.n_batches) * u64::from(cfg.n_particles_per_batch)) as f64;

    println!(
        "\n  k_∞ (active batches) = {:.5} ± {:.5}",
        result.k_mean, result.k_sigma
    );
    println!(
        "  sim time   = {:.2} s   ({:.0} ns/history active)",
        sim_s, ns_per_history
    );
    println!(
        "  collisions = {} total ({:.1} per source neutron)",
        total_collisions, collisions_per_history
    );
    Ok(())
}

fn load(
    name: &str,
    dir: &std::path::Path,
    target_t: f64,
    cfg: &LoaderConfig,
) -> Result<Nuclide, Box<dyn std::error::Error>> {
    let path = dir.join(format!("{name}.h5"));
    Ok(load_nuclide_from_hdf5(&path, target_t, cfg)?)
}

fn push_surface(surfaces: &mut Vec<Surface>, surface: Surface) -> usize {
    let idx = surfaces.len();
    surfaces.push(surface);
    idx
}
