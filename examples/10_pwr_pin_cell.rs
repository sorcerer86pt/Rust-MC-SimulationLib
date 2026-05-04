//! Light-water reactor pin cell — medium-class three-loop PWR with
//! real ENDF/B-VII.1 nuclear data.
//!
//! Geometry and materials match the Westinghouse 17×17 fuel design
//! used by Almaraz NPP (Spain, ≈977 MWe net) and the French CP1/CP2
//! 900-MWe class. Both plants share the same nominal pin cell:
//!
//!   * pin pitch                p = 1.260 cm
//!   * UO₂ pellet outer radius  r_p = 0.4096 cm
//!   * Zircaloy-4 outer radius  r_c = 0.4750 cm
//!   * fresh-fuel enrichment    3.7 wt % ²³⁵U
//!   * fuel temperature         900 K (≈ centerline, hot full power)
//!   * coolant / clad temp      600 K (≈ T_avg 583 K, snapped to library)
//!   * boron concentration      0 ppm (HZP-like, fresh core)
//!
//! Boundary conditions are reflective on all four pitch walls — this
//! is the infinite-lattice k_∞ problem against which OpenMC, Serpent,
//! and MCNP regression-validate their PWR pin-cell models.
//!
//! Data source: ENDF/B-VII.1 OpenMC HDF5 (`https://openmc.org/data/`).
//! Nine nuclides loaded with energy-dependent ν̄, fission outgoing-
//! energy distributions, elastic CM angular distributions, URR
//! probability tables, and discrete inelastic levels. S(α,β) thermal
//! scattering for H in H₂O is *not* included in this example; for
//! that, layer in the `nuclear::thermal` module.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features nuclear --example 10_pwr_pin_cell -- \
//!     /path/to/endfb-vii.1-hdf5/neutron
//! ```
//!
//! The first positional argument is the directory containing the
//! per-nuclide HDF5 files (`U235.h5`, `U238.h5`, `O16.h5`, `H1.h5`,
//! `Zr90.h5`, …).

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, Region, between, inside, outside};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::Surface;
use rust_mc_sim::nuclear::loader::{LoaderConfig, load_nuclide_from_hdf5};
use rust_mc_sim::transport::material::{Material, Nuclide};
use rust_mc_sim::transport::simulate::{EigenvalueConfig, run_eigenvalue};

// ── Geometry constants (cm) ─────────────────────────────────────────
const PITCH: f64 = 1.260;
const HALF_PITCH: f64 = PITCH * 0.5;
const PELLET_R: f64 = 0.4096;
const CLAD_OR: f64 = 0.4750;

// ── Material temperatures (K) — snap to ENDF/B-VII.1 library cols ───
const T_FUEL: f64 = 900.0;
const T_CLAD: f64 = 600.0;
const T_MOD: f64 = 600.0;

// ── Atom densities (atoms / b·cm) ───────────────────────────────────
// UO₂ at 10.41 g/cm³ (95 % theoretical density) and 3.7 wt % ²³⁵U.
const N_U235_FUEL: f64 = 8.65e-4;
const N_U238_FUEL: f64 = 2.225e-2;
const N_O16_FUEL: f64 = 4.624e-2;

// Zircaloy-4 at 6.55 g/cm³ — Zr makes up 98.3 wt %; treat as a
// natural Zr mix (89.4 % Zr-90, 10.6 % Zr-91/92/94 split).
const N_ZR_TOTAL: f64 = 4.32e-2;
const F_ZR90: f64 = 0.5145; // natural abundance
const F_ZR91: f64 = 0.1122;
const F_ZR92: f64 = 0.1715;
const F_ZR94: f64 = 0.1738;

// Light water at 0.7245 g/cm³ (155 bar, 583 K) — fresh-fuel HZP-like.
const N_H1_MOD: f64 = 4.836e-2;
const N_O16_MOD: f64 = 2.418e-2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let data_dir = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!(
                "usage: 10_pwr_pin_cell <data_dir>\n\
                 \n\
                 <data_dir> is the directory containing the per-nuclide\n\
                 OpenMC HDF5 files (U235.h5, U238.h5, O16.h5, H1.h5,\n\
                 Zr90.h5, Zr91.h5, Zr92.h5, Zr94.h5). Download from\n\
                 https://openmc.org/data/ if you don't already have it."
            );
            std::process::exit(2);
        }
    };

    println!(
        "Almaraz / CP1-class PWR pin cell — ENDF/B-VII.1 HDF5\n\
         data dir: {}\n",
        data_dir.display()
    );

    // ── Nuclide load ────────────────────────────────────────────────
    let cfg = LoaderConfig::default();
    let load_t0 = Instant::now();
    let u235 = Arc::new(load("U235", &data_dir, T_FUEL, &cfg)?);
    let u238 = Arc::new(load("U238", &data_dir, T_FUEL, &cfg)?);
    let o16_fuel = Arc::new(load("O16", &data_dir, T_FUEL, &cfg)?);
    let h1 = Arc::new(load("H1", &data_dir, T_MOD, &cfg)?);
    let o16_mod = Arc::new(load("O16", &data_dir, T_MOD, &cfg)?);
    let zr90 = Arc::new(load("Zr90", &data_dir, T_CLAD, &cfg)?);
    let zr91 = Arc::new(load("Zr91", &data_dir, T_CLAD, &cfg)?);
    let zr92 = Arc::new(load("Zr92", &data_dir, T_CLAD, &cfg)?);
    let zr94 = Arc::new(load("Zr94", &data_dir, T_CLAD, &cfg)?);
    let load_ms = load_t0.elapsed().as_secs_f64() * 1000.0;
    println!("loaded 9 nuclides in {load_ms:.0} ms\n");

    // ── Materials ───────────────────────────────────────────────────
    let mut fuel = Material::new("UO2_3.7%", T_FUEL);
    fuel.add(u235, N_U235_FUEL);
    fuel.add(u238, N_U238_FUEL);
    fuel.add(o16_fuel, N_O16_FUEL);

    let mut clad = Material::new("Zircaloy-4", T_CLAD);
    clad.add(zr90, N_ZR_TOTAL * F_ZR90);
    clad.add(zr91, N_ZR_TOTAL * F_ZR91);
    clad.add(zr92, N_ZR_TOTAL * F_ZR92);
    clad.add(zr94, N_ZR_TOTAL * F_ZR94);

    let mut moderator = Material::new("H2O_583K", T_MOD);
    moderator.add(h1, N_H1_MOD);
    moderator.add(o16_mod, N_O16_MOD);
    let materials = [fuel, clad, moderator];

    // ── Geometry ────────────────────────────────────────────────────
    let surfaces = vec![
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: PELLET_R,
            bc: BoundaryCondition::Transmission,
        },
        Surface::CylinderZ {
            center_x: 0.0,
            center_y: 0.0,
            radius: CLAD_OR,
            bc: BoundaryCondition::Transmission,
        },
        Surface::PlaneX {
            x0: -HALF_PITCH,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneX {
            x0: HALF_PITCH,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: -HALF_PITCH,
            bc: BoundaryCondition::Reflective,
        },
        Surface::PlaneY {
            y0: HALF_PITCH,
            bc: BoundaryCondition::Reflective,
        },
    ];

    let inside_box = Region::Intersection(Box::new(between(2, 3)), Box::new(between(4, 5)));
    let cell_fuel = Cell::new(CellId(0), inside(0), CellFill::Material(0));
    let cell_clad = Cell::new(CellId(1), between(0, 1), CellFill::Material(1));
    let cell_mod = Cell::new(
        CellId(2),
        Region::Intersection(Box::new(outside(1)), Box::new(inside_box)),
        CellFill::Material(2),
    );
    let cells = vec![cell_fuel, cell_clad, cell_mod];

    // ── k-eigenvalue power iteration ────────────────────────────────
    let cfg = EigenvalueConfig {
        n_batches: 80,
        n_inactive: 20,
        n_particles_per_batch: 5_000,
        seed: 7,
    };
    println!(
        "geometry: pitch {PITCH:.3} cm, pellet R {PELLET_R:.4} cm, clad R {CLAD_OR:.4} cm\n\
         driver: {} batches × {} particles, {} inactive, reflective walls\n",
        cfg.n_batches, cfg.n_particles_per_batch, cfg.n_inactive
    );

    let sim_t0 = Instant::now();
    let result = run_eigenvalue(
        &cfg,
        &cells,
        &surfaces,
        |cell_idx| match cell_idx {
            0 => 0,
            1 => 1,
            _ => 2,
        },
        &materials,
        (
            [-HALF_PITCH * 0.6, -HALF_PITCH * 0.6, -0.5],
            [HALF_PITCH * 0.6, HALF_PITCH * 0.6, 0.5],
        ),
        2.0e6,
    );
    let sim_s = sim_t0.elapsed().as_secs_f64();

    println!("  batch │  k_collision │ entropy │  collisions │ leaked │ fissions");
    println!("  ──────┼──────────────┼─────────┼─────────────┼────────┼─────────");
    for b in &result.batch_history {
        println!(
            "  {:>5} │   {:.5}    │  {:.3}  │ {:>11} │ {:>6} │ {:>8}",
            b.batch, b.k_collision, b.source_entropy, b.n_collisions, b.n_leaked, b.n_fissions
        );
    }
    let n_active_hist: u64 = result
        .batch_history
        .iter()
        .filter(|b| b.batch >= cfg.n_inactive)
        .map(|_| u64::from(cfg.n_particles_per_batch))
        .sum();
    let ns_per_history = (sim_s * 1.0e9) / (n_active_hist.max(1) as f64);
    println!(
        "\n  k_∞ (active batches) = {:.5} ± {:.5}\n  sim time = {:.2} s  ({:.0} ns/history, active)",
        result.k_mean, result.k_sigma, sim_s, ns_per_history
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
    let nuc = load_nuclide_from_hdf5(&path, target_t, cfg)?;
    Ok(nuc)
}
