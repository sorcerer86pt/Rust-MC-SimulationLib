#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Ducru-weighted reconstruction at off-grid target columns.
//!
//! Demonstrates both the raw weights (L2-optimal) and the
//! partition-of-unity normalisation (peak-preserving). For a smooth
//! exponential test function both schemes are within 0.5% of truth at
//! the off-grid mid-points; for a function with sharp resonance peaks
//! the normalisation is what keeps the peaks at the right height.
//!
//! Run with: `cargo run --release --example 03_ducru_blending`

use rust_mc_sim::{ducru_unity_weights, ducru_weights, nearest_k_columns};

fn main() {
    let library = vec![300.0, 600.0, 900.0, 1200.0, 1500.0, 2500.0];

    // f(t) = e^{-t/1000} as a smooth proxy for a temperature-dependent
    // average cross section. Sample at the library; reconstruct at the
    // mid-points and report the error.
    let f = |t: f64| (-t / 1000.0).exp();
    let f_train: Vec<f64> = library.iter().copied().map(f).collect();

    println!("library T (K): {library:?}");
    println!();
    println!("target_T  truth      raw 5-pt   3-pt unity  unity_err  raw_err");
    println!("--------  --------   --------   ----------  ---------  -------");
    for &target in &[400.0, 750.0, 1050.0, 1350.0, 1800.0] {
        let truth = f(target);

        let raw_w = ducru_weights(&library, target);
        let raw_est: f64 = raw_w.iter().zip(f_train.iter()).map(|(w, fv)| w * fv).sum();

        let chosen = nearest_k_columns(&library, target, 3);
        let sub: Vec<f64> = chosen.iter().map(|&i| library[i]).collect();
        let unity_w = ducru_unity_weights(&sub, target);
        let unity_est: f64 = chosen
            .iter()
            .zip(unity_w.iter())
            .map(|(&i, &w)| w * f_train[i])
            .sum();

        println!(
            "{target:>7.0}   {truth:>8.5}   {raw_est:>8.5}    {unity_est:>9.5}    \
             {:>8.2e}   {:>7.2e}",
            ((unity_est - truth) / truth).abs(),
            ((raw_est - truth) / truth).abs()
        );
    }

    println!();
    println!("on-library targets (collapse to one-hot, error = 0):");
    for &target in &library {
        let w = ducru_weights(&library, target);
        let est: f64 = w.iter().zip(f_train.iter()).map(|(w, fv)| w * fv).sum();
        let truth = f(target);
        let err = (est - truth).abs();
        assert!(
            err < 1e-10,
            "at training column {target} expected exact, got err={err}"
        );
        println!("  T={target}: exact match (err = {err:.2e})");
    }
}
