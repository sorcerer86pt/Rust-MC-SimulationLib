//! Truncated SVD on a smooth 2D function.
//!
//! Run with: `cargo run --release --example 01_svd_basics`

use std::sync::Arc;

use tensor_compress::{SvdKernel, ducru_unity_weights, nearest_k_columns};

fn main() {
    // Smooth function f(x, t) = sin(x) · exp(-t / 1000) sampled on a
    // log-spaced row axis (e.g. energies) and a small set of training
    // columns (e.g. temperatures).
    let n_rows = 1000;
    let n_cols = 6;
    let row_axis: Vec<f64> = (0..n_rows).map(|i| 1e-3 * 1.02f64.powi(i as i32)).collect();
    let columns: [f64; 6] = [300.0, 600.0, 900.0, 1200.0, 1500.0, 2500.0];

    let mut data = vec![0.0_f64; n_rows * n_cols];
    for i in 0..n_rows {
        let x = row_axis[i];
        for (j, &t) in columns.iter().enumerate() {
            data[i * n_cols + j] = x.sin() * (-t / 1000.0_f64).exp();
        }
    }

    let row_axis: Arc<[f64]> = Arc::from(row_axis.into_boxed_slice());
    let kernel = SvdKernel::from_data(&data, row_axis, n_rows, n_cols, 4);

    println!(
        "kernel built: rank={}, basis = {} bytes, n_rows={}, n_cols={}",
        kernel.rank(),
        kernel.memory_bytes(),
        kernel.n_rows(),
        kernel.n_cols()
    );

    // Reconstruct at training column 2 (T = 900 K) at three random rows.
    let coeffs = kernel.coeffs_at_col(2);
    for row in [50, 500, 950] {
        let from_kernel = kernel.reconstruct_at(row, &coeffs);
        let truth = data[row * n_cols + 2];
        println!(
            "row={row} on-grid recon={from_kernel:.6e}  truth={truth:.6e}  rel_err={:.2e}",
            ((from_kernel - truth) / truth).abs()
        );
    }

    // Off-grid reconstruction at T = 750 K via partition-of-unity
    // 3-point Ducru weights (the production-grade variant). Picks the
    // three library columns nearest to 750 K, computes their unity-
    // normalised weights, and feeds them straight into the kernel.
    let chosen = nearest_k_columns(&columns, 750.0, 3);
    let sub: Vec<f64> = chosen.iter().map(|&i| columns[i]).collect();
    let weights_3pt = ducru_unity_weights(&sub, 750.0);
    // Build a length-`n_cols` weight vector that's zero outside the
    // 3-point subset (so reconstruct_with_weights against the full
    // V^T does the right thing).
    let mut weights_full = vec![0.0_f64; n_cols];
    for (k, &col) in chosen.iter().enumerate() {
        weights_full[col] = weights_3pt[k];
    }
    let off = kernel.reconstruct_with_weights(&weights_full);

    let truth_at_750 = (1e-3_f64 * 1.02f64.powi(500)).sin() * (-750.0_f64 / 1000.0).exp();
    let est = kernel.reconstruct_at(500, &off);
    println!(
        "off-grid T=750 K (3-pt unity Ducru): recon={est:.6e}  truth={truth_at_750:.6e}  rel_err={:.2e}",
        ((est - truth_at_750) / truth_at_750).abs()
    );
}
