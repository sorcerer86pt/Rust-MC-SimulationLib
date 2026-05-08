#![allow(clippy::unwrap_used, clippy::expect_used)]
//! URR probability-table sampling.
//!
//! Run with: `cargo run --release --example 08_urr_sampling`

use rust_mc_sim::Pcg64;
use rust_mc_sim::urr::UrrProbabilityTables;

fn main() {
    // Two-energy, four-band table. Cumulative probs: 0.25, 0.50,
    // 0.75, 1.0 → uniform across the four bands. Per-band factors
    // chosen so the bands are easily distinguishable.
    let t = UrrProbabilityTables {
        energies: vec![1.0e3, 1.0e4],
        n_bands: 4,
        cum_prob: vec![vec![0.25, 0.50, 0.75, 1.0], vec![0.25, 0.50, 0.75, 1.0]],
        total_factor: vec![vec![0.5, 1.0, 1.5, 2.0], vec![0.4, 0.9, 1.4, 1.9]],
        elastic_factor: vec![vec![0.5, 1.0, 1.5, 2.0], vec![0.4, 0.9, 1.4, 1.9]],
        fission_factor: vec![vec![0.5, 1.0, 1.5, 2.0], vec![0.4, 0.9, 1.4, 1.9]],
        capture_factor: vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.5, 0.5, 0.5, 0.5]],
        multiply_smooth: true,
        interpolation: 2,
    };

    println!("URR probe at three energies:");
    let mut rng = Pcg64::new(42, 1);
    for &e in &[1.0e3, 5.5e3, 1.0e4] {
        // Average over many ξ draws — should reproduce the
        // band-averaged factor at that energy.
        let mut tot_total = 0.0;
        let mut tot_capture = 0.0;
        let n = 100_000;
        for _ in 0..n {
            let xi = rng.uniform();
            let f = t.sample(e, xi);
            tot_total += f.total;
            tot_capture += f.capture;
        }
        println!(
            "  E = {e:>7.0} eV  ⟨total⟩ = {:.3}  ⟨capture⟩ = {:.3}",
            tot_total / n as f64,
            tot_capture / n as f64
        );
    }
    println!();
    println!("(at E=1000 eV the table averages to 1.25 / 1.0)");
    println!("(at E=10000 eV  the table averages to 1.15 / 0.5)");
    println!("(at E=5500 eV   linear interpolation between the two)");
}
