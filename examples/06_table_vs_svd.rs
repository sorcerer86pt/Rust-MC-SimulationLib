//! Side-by-side comparison: PointwiseTable (the production-baseline
//! lookup) against SvdKernel (the SVD-compressed reconstruction)
//! against the Ducru-blended off-grid SVD reconstruction.
//!
//! Demonstrates how to call the same `f(x)` at the same point with
//! all three representations and what each one costs vs the truth.
//!
//! Run with: `cargo run --release --example 06_table_vs_svd`

use std::sync::Arc;

use rust_mc_sim::{PointwiseTable, SvdKernel, ducru_unity_weights, nearest_k_columns};

fn main() {
    // Smooth two-axis function f(x, T) = sin(x) · exp(-T/1000) sampled
    // on a log-spaced row axis (10,000 points) and 6 training columns.
    let n_rows = 10_000;
    let n_cols = 6;
    let row_axis: Vec<f64> = (0..n_rows)
        .map(|i| 1e-3 * 1.005f64.powi(i as i32))
        .collect();
    let columns: [f64; 6] = [300.0, 600.0, 900.0, 1200.0, 1500.0, 2500.0];

    let mut data = vec![0.0_f64; n_rows * n_cols];
    for i in 0..n_rows {
        let x = row_axis[i];
        for (j, &t) in columns.iter().enumerate() {
            data[i * n_cols + j] = x.sin().abs() * (-t / 1000.0_f64).exp() + 0.01;
        }
    }
    let row_axis: Arc<[f64]> = Arc::from(row_axis.into_boxed_slice());

    // ── Pointwise tables — one per training column ──
    // Production baseline. Memory cost = n_rows × n_cols × 8 bytes.
    let mut tables: Vec<PointwiseTable> = (0..n_cols)
        .map(|j| {
            let xs: Vec<f64> = (0..n_rows).map(|i| data[i * n_cols + j]).collect();
            PointwiseTable::from_shared(Arc::clone(&row_axis), xs)
        })
        .collect();
    for t in tables.iter_mut() {
        t.build_hash(8192);
    }

    // ── SVD kernel — one for the whole 2-tensor ──
    // Stores rank × (n_rows + n_cols) instead of n_rows × n_cols.
    let kernel = SvdKernel::from_data(
        &data,
        Arc::clone(&row_axis),
        n_rows,
        n_cols,
        4, // rank-4 truncation
    );

    // ── Memory comparison ──
    let table_bytes: usize = tables.iter().map(|t| t.memory_bytes()).sum();
    let svd_bytes = kernel.memory_bytes();
    println!("=== rust-mc-sim — table vs SVD ===");
    println!();
    println!("  data shape: {n_rows} rows × {n_cols} cols (training columns: {columns:?})");
    println!();
    println!("  memory:");
    println!(
        "    {} pointwise tables: {:.1} KB ({} × {} × 8B)",
        n_cols,
        table_bytes as f64 / 1024.0,
        n_rows,
        n_cols
    );
    println!(
        "    SVD kernel rank-4:   {:.1} KB (basis {} × 4 + Vᵀ 4 × {})",
        svd_bytes as f64 / 1024.0,
        n_rows,
        n_cols
    );
    println!(
        "    SVD/table memory:    {:.2}×",
        svd_bytes as f64 / table_bytes as f64
    );

    // ── Accuracy at on-grid columns ──
    println!();
    println!("  on-grid lookup at T = {} K, x = 5.3:", columns[2]);
    let x_probe: f64 = 5.3;
    let truth_2 = x_probe.sin().abs() * (-columns[2] / 1000.0_f64).exp() + 0.01;
    let table_2 = tables[2].lookup(x_probe);
    let svd_coeffs = kernel.coeffs_at_col(2);
    let svd_2 = kernel.reconstruct_at(kernel.row_index(x_probe), &svd_coeffs);
    println!("    truth      : {truth_2:.6e}");
    println!(
        "    table      : {table_2:.6e}   (rel err {:.2e})",
        ((table_2 - truth_2) / truth_2).abs()
    );
    println!(
        "    SVD rank-4 : {svd_2:.6e}   (rel err {:.2e})",
        ((svd_2 - truth_2) / truth_2).abs()
    );

    // ── Off-grid reconstruction at T = 750 K ──
    println!();
    println!("  off-grid reconstruction at T = 750 K, x = 5.3:");
    let target_t = 750.0;
    let truth_off = x_probe.sin().abs() * (-target_t / 1000.0_f64).exp() + 0.01;
    println!("    truth                       : {truth_off:.6e}");

    // Pointwise table approach: stochastic two-endpoint pseudo-interp
    // (or linear interp between adjacent endpoints — shown here).
    // OpenMC ships the stochastic variant for resonance-region channels
    // because pure linear interpolation of a resonance peak is biased.
    let lo_idx = 1; // T = 600 K
    let hi_idx = 2; // T = 900 K
    let frac = (target_t - columns[lo_idx]) / (columns[hi_idx] - columns[lo_idx]);
    let lin_interp =
        (1.0 - frac) * tables[lo_idx].lookup(x_probe) + frac * tables[hi_idx].lookup(x_probe);
    println!(
        "    linear interp (2 tables)    : {lin_interp:.6e}   (rel err {:.2e})",
        ((lin_interp - truth_off) / truth_off).abs()
    );

    // Full Ducru raw weights — N tables but L2-optimal.
    let n_full = 5; // 5 of 6 columns
    let raw_w = rust_mc_sim::ducru::ducru_weights(&columns[..n_full], target_t);
    let raw_est: f64 = (0..n_full)
        .map(|j| raw_w[j] * tables[j].lookup(x_probe))
        .sum();
    println!(
        "    Ducru raw {n_full}-pt           : {raw_est:.6e}   (rel err {:.2e})",
        ((raw_est - truth_off) / truth_off).abs()
    );

    // SVD with 3-pt unity Ducru — production-grade reconstruction.
    let chosen = nearest_k_columns(&columns, target_t, 3);
    let sub: Vec<f64> = chosen.iter().map(|&i| columns[i]).collect();
    let unity_w = ducru_unity_weights(&sub, target_t);
    let mut weights_full = vec![0.0_f64; n_cols];
    for (k, &col) in chosen.iter().enumerate() {
        weights_full[col] = unity_w[k];
    }
    let svd_off = kernel.reconstruct_with_weights(&weights_full);
    let svd_est = kernel.reconstruct_at(kernel.row_index(x_probe), &svd_off);
    println!(
        "    SVD rank-4 + 3-pt unity Ducru: {svd_est:.6e}   (rel err {:.2e})",
        ((svd_est - truth_off) / truth_off).abs()
    );

    // ── Summary ──
    println!();
    println!("  Summary:");
    println!("    * table is exact at training columns (no compression error)");
    println!("    * SVD rank-4 is near-exact at training columns and stays");
    println!(
        "      sub-percent off-grid at the cost of {:.1}× memory of one table",
        svd_bytes as f64 / tables[0].memory_bytes() as f64
    );
    println!("    * for reactor-grade accuracy (≤ 1% global L2) on smooth data,");
    println!("      both representations are production-viable; pick by what you");
    println!("      have more of: column counts (favours SVD) or rows (favours tables).");
}
