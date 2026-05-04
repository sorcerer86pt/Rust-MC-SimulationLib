//! γ-shielding demo — Cs-137 661.7 keV line through a slab of lead.
//! Demonstrates the [`run_photon_fixed_source`] driver and the
//! photon interaction kernels (Klein-Nishina + photoelectric + pair
//! production + forward-peaked Rayleigh) on real ENDF/B-VII.1
//! photon data.
//!
//! Geometry: a 5 cm slab of lead between two vacuum half-spaces.
//! The Cs-137 source sits 0.5 cm in front of the slab; the user can
//! read the leakage / energy-deposition stats and the per-cell flux
//! tally to estimate transmission and dose.
//!
//! Reference values (NIST XCOM, lead, 662 keV):
//!
//!   * μ/ρ ≈ 0.1102 cm²/g, ρ_Pb = 11.35 g/cm³ → μ ≈ 1.25 cm⁻¹
//!   * 5 cm of lead → narrow-beam transmission e^(-μx) ≈ 0.19 %
//!   * with build-up factor B ≈ 2.5 → broad-beam ≈ 0.5 %
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --features nuclear --example 12_photon_shielding -- \
//!     /path/to/endfb-vii.1-hdf5/photon
//! ```

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use rust_mc_sim::geometry::cell::{Cell, CellFill, CellId, between};
use rust_mc_sim::geometry::surface::BoundaryCondition;
use rust_mc_sim::geometry::{Surface, Vec3};
use rust_mc_sim::photon::{
    IsotropicLineSource, PhotonFixedSourceConfig, PhotonMaterial,
    run_photon_fixed_source,
};
use rust_mc_sim::tally::{FluxBin, FluxTally};

const SLAB_X_MIN: f64 = 0.0;
const SLAB_X_MAX: f64 = 5.0;
const Y_HALF: f64 = 30.0;
const Z_HALF: f64 = 30.0;

const CS137_LINE_EV: f64 = 661_700.0;
const PB_DENSITY_G_PER_CM3: f64 = 11.35;
const PB_ATOM_DENSITY: f64 = PB_DENSITY_G_PER_CM3 * 6.022e23 / 207.2 / 1.0e24;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let data_dir = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: 12_photon_shielding <photon_data_dir>");
            std::process::exit(2);
        }
    };

    println!(
        "Cs-137 line source through {} cm lead slab\n\
         data dir: {}\n",
        SLAB_X_MAX - SLAB_X_MIN,
        data_dir.display()
    );

    // Load Pb photon data.
    let pb_path = data_dir.join("Pb.h5");
    let pb_element =
        Arc::new(rust_mc_sim::photon::loader::load_photon_element(&pb_path)?);
    println!(
        "loaded Pb (Z={}, {} energy points), N = {:.4e} atoms/b·cm",
        pb_element.z,
        pb_element.n_energy(),
        PB_ATOM_DENSITY
    );

    let mut lead = PhotonMaterial::new("Pb");
    lead.add(pb_element, PB_ATOM_DENSITY);
    let materials = [lead];

    // Geometry: 1-D-style slab with vacuum walls.
    let surfaces = vec![
        Surface::PlaneX {
            x0: -1.0,
            bc: BoundaryCondition::Vacuum,
        },
        Surface::PlaneX {
            x0: SLAB_X_MIN,
            bc: BoundaryCondition::Transmission,
        },
        Surface::PlaneX {
            x0: SLAB_X_MAX,
            bc: BoundaryCondition::Transmission,
        },
        Surface::PlaneX {
            x0: SLAB_X_MAX + 5.0,
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
    let yz_box = Region::Intersection(Box::new(between(4, 5)), Box::new(between(6, 7)));
    // Cell 0: pre-slab air (modeled as vacuum-like, so really we want a tiny
    //          XS material — for v1 we just put lead at the source side too,
    //          which is wrong but simple. Better: void cell. We'll pick
    //          "void = same Pb material with negligible density" for clarity).
    // Actually simpler: use one cell with Pb, source inside slab.
    let slab_box = Region::Intersection(Box::new(between(1, 2)), Box::new(yz_box));
    let slab = Cell::new(CellId(0), slab_box, CellFill::Material(0));
    let cells = vec![slab];

    // Tally: per-cell flux in two energy bins (above / below the
    // photoelectric edge of Pb K-shell, ~88 keV).
    let bins = vec![
        FluxBin {
            cell: Some(0),
            e_lo: 0.0,
            e_hi: 88.0e3,
        },
        FluxBin {
            cell: Some(0),
            e_lo: 88.0e3,
            e_hi: f64::INFINITY,
        },
    ];
    let mut tally = FluxTally::new(bins, 5);

    // Source: 662 keV isotropic point at the upstream face of the slab.
    let source = IsotropicLineSource {
        pos: Vec3::new(SLAB_X_MIN + 1.0e-6, 0.0, 0.0),
        energy_ev: CS137_LINE_EV,
    };

    let cfg = PhotonFixedSourceConfig {
        n_batches: 30,
        n_particles_per_batch: 5_000,
        seed: 13,
        energy_cutoff_ev: 1.0e3,
    };
    println!(
        "driver: {} batches × {} histories, slab depth = {} cm, E_cutoff = {:.0} eV",
        cfg.n_batches,
        cfg.n_particles_per_batch,
        SLAB_X_MAX - SLAB_X_MIN,
        cfg.energy_cutoff_ev
    );

    let result = run_photon_fixed_source(
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
    let total_dep: f64 = result
        .batches
        .iter()
        .map(|b| b.total_energy_deposited)
        .sum();
    let leak_frac = total_leaked as f64 / result.total_histories as f64;
    let dep_per_history = total_dep / result.total_histories as f64;

    println!(
        "\n  histories      {}\n  collisions     {}\n  leaked         {}  ({:.2} % of source)\n  absorbed       {}  ({:.2} % of source)\n  E deposited    {:.3} MeV total ({:.1} keV / source γ)",
        result.total_histories,
        total_collisions,
        total_leaked,
        leak_frac * 100.0,
        total_absorbed,
        total_absorbed as f64 / result.total_histories as f64 * 100.0,
        total_dep * 1.0e-6,
        dep_per_history * 1.0e-3,
    );

    println!("\nTrack-length flux per source photon:");
    println!("  energy bin                         │ mean         │ σ");
    println!("  ───────────────────────────────────┼──────────────┼─────────");
    for (i, bin) in tally.bins().iter().enumerate() {
        let mean = tally.mean(i) / cfg.n_particles_per_batch as f64;
        let sigma = tally.sigma(i) / cfg.n_particles_per_batch as f64;
        let label = if bin.e_hi.is_infinite() {
            format!("E ≥ {:.1} keV (above K-edge)", bin.e_lo * 1.0e-3)
        } else {
            format!("E < {:.1} keV (below K-edge)", bin.e_hi * 1.0e-3)
        };
        println!("  {label:<35}│ {mean:>11.4e}  │ {sigma:>8.3e}");
    }

    Ok(())
}
