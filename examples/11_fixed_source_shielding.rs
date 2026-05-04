//! Fixed-source shielding demo — Watt-fission point source streaming
//! through a slab of light water, with leakage tallied at the far
//! face. Demonstrates the [`run_fixed_source`] driver and the
//! [`FluxTally`] track-length scorer on real ENDF/B-VII.1 data.
//!
//! Geometry (1-D-like, infinite in y, z):
//!
//! ```text
//!   x = 0     ─────────────  ────────────  ─────────────  x = X_max
//!     │       │ source      │  H₂O slab  │      vacuum         │
//!     │       │ region      │            │                     │
//!     │  Vacuum BC          ↑           Vacuum BC
//! ```
//!
//! Tally bins record cell-resolved track-length flux in two
//! coarse energy windows: fast (E ≥ 0.625 eV) and thermal
//! (E < 0.625 eV). Reading the ratio of (thermal flux at far face)
//! / (source intensity) gives a back-of-envelope shielding
//! transmission factor.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features nuclear --example 11_fixed_source_shielding -- \
//!     /path/to/endfb-vii.1-hdf5/neutron
//! ```

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, between};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::{Surface, Vec3};
use rust_mc_sim::nuclear::loader::{
    LoaderConfig, attach_thermal_scattering, load_nuclide_from_hdf5,
    load_thermal_scattering,
};
use rust_mc_sim::tally::{FluxBin, FluxTally};
use rust_mc_sim::transport::fixed_source::{
    FixedSourceConfig, WattFissionSource, run_fixed_source,
};
use rust_mc_sim::transport::material::{Material, Nuclide};

const SLAB_X_MIN: f64 = 0.0;
const SLAB_X_MAX: f64 = 30.0;
const Y_HALF: f64 = 50.0;
const Z_HALF: f64 = 50.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let data_dir = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: 11_fixed_source_shielding <data_dir>");
            std::process::exit(2);
        }
    };

    println!(
        "Watt-fission source through 30 cm light-water slab\n\
         data dir: {}\n",
        data_dir.display()
    );

    // ── Materials ──────────────────────────────────────────────────
    let cfg = LoaderConfig::default();
    let h1_free = load("H1", &data_dir, 600.0, &cfg)?;
    let kernel = load_thermal_scattering(&data_dir.join("c_H_in_H2O.h5"))?;
    let h1 = Arc::new(attach_thermal_scattering(h1_free, kernel));
    let o16 = Arc::new(load("O16", &data_dir, 600.0, &cfg)?);
    let mut water = Material::new("H2O_600K", 600.0);
    water.add(h1, 4.836e-2);
    water.add(o16, 2.418e-2);
    let materials = [water];

    // ── Geometry: a slab between x = 0 and x = SLAB_X_MAX, with
    // ── y, z bounded by a vacuum box. ──────────────────────────────
    let surfaces = vec![
        Surface::PlaneX {
            x0: SLAB_X_MIN,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneX {
            x0: SLAB_X_MAX,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneY {
            y0: -Y_HALF,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneY {
            y0: Y_HALF,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneZ {
            z0: -Z_HALF,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneZ {
            z0: Z_HALF,
            bc: BoundaryCondition::Vacuum,
        },
    ];

    use rust_mc_sim::geometry::cell::Region;
    let inside_box = Region::Intersection(
        Box::new(Region::Intersection(
            Box::new(between(0, 1)),
            Box::new(between(2, 3)),
        )),
        Box::new(between(4, 5)),
    );
    let cell = Cell::new(CellId(0), inside_box, CellFill::Material(0));
    let cells = vec![cell];

    // ── Tally: cell-0 flux split into thermal and fast windows. ───
    let bins = vec![
        FluxBin {
            cell: Some(0),
            e_lo: 0.0,
            e_hi: 0.625,
        },
        FluxBin {
            cell: Some(0),
            e_lo: 0.625,
            e_hi: f64::INFINITY,
        },
    ];
    let mut tally = FluxTally::new(bins, 5);

    // ── Source: Watt fission spectrum at x = 0.5 cm (just inside
    // the front face of the slab), isotropic. ─────────────────────
    let source = WattFissionSource {
        pos: Vec3::new(0.5, 0.0, 0.0),
    };

    let cfg = FixedSourceConfig {
        n_batches: 30,
        n_particles_per_batch: 5_000,
        seed: 17,
    };
    println!(
        "driver: {} batches × {} particles, slab depth {} cm",
        cfg.n_batches,
        cfg.n_particles_per_batch,
        SLAB_X_MAX - SLAB_X_MIN
    );

    let result = run_fixed_source(
        &cfg,
        &cells,
        &surfaces,
        |_| 0,
        &materials,
        &source,
        Some(&mut tally),
    );

    let total_collisions: u64 = result.batches.iter().map(|b| b.n_collisions).sum();
    let total_leaked: u64 = result.batches.iter().map(|b| b.n_leaked).sum();
    let total_absorbed: u64 = result.batches.iter().map(|b| b.n_absorbed).sum();

    println!(
        "\n  histories run     {}\n  total collisions  {}\n  total leaked      {}\n  total absorbed    {}",
        result.total_histories, total_collisions, total_leaked, total_absorbed
    );

    println!("\nTrack-length flux (per batch, per source neutron):");
    println!("  bin              │ mean        │ σ");
    println!("  ─────────────────┼─────────────┼────────────");
    for (i, bin) in tally.bins().iter().enumerate() {
        let mean = tally.mean(i) / cfg.n_particles_per_batch as f64;
        let sigma = tally.sigma(i) / cfg.n_particles_per_batch as f64;
        let label = if bin.e_hi.is_infinite() {
            format!("E ≥ {:.3} eV", bin.e_lo)
        } else {
            format!("{:.3} eV ≤ E < {:.3} eV", bin.e_lo, bin.e_hi)
        };
        println!("  {label:<17}│ {mean:>11.4e} │ {sigma:>10.4e}");
    }
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
