//! Stress example for at-scale workloads.
//!
//! Builds N independent SvdKernels from synthetic data, both
//! sequentially and (with `--features parallel`) in parallel, and
//! reports wall-clock time + total memory. Tune N via the env var
//! `RUST_MC_SIM_STRESS_N` (default 10000).
//!
//! Run with the parallel feature for the speedup:
//!
//! ```bash
//! cargo run --release --features parallel --example 05_stress_200k
//! RUST_MC_SIM_STRESS_N=50000 cargo run --release --features parallel --example 05_stress_200k
//! ```
//!
//! At 200k-nuclide library scale you'd typically:
//!   1. Read HDF5 nuclide files into raw `Vec<f64>` matrices in
//!      parallel (use `rayon` or async I/O — the file open is the
//!      bottleneck, not the SVD).
//!   2. Hand the matrices to `from_data_many_par` for fan-out SVD.
//!   3. Persist the resulting `(basis, vt)` factor pairs to disk if
//!      you want to amortise the load across runs (see
//!      `SvdKernel::from_factors`).

use std::sync::Arc;
use std::time::Instant;

use rust_mc_sim::{
    SvdKernel,
    batch::{KernelInput, from_data_many, total_kernel_bytes},
};

#[cfg(feature = "parallel")]
use rust_mc_sim::batch::from_data_many_par;

fn main() {
    let n: usize = std::env::var("RUST_MC_SIM_STRESS_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    // 80 rows × 6 columns is small enough that overhead-per-kernel
    // dominates — the perfect stress test for the batch APIs. At
    // 200k kernels with rank 5 this matches a typical depletion-
    // library shape.
    let n_rows = 80;
    let n_cols = 6;
    let rank = 5;

    let row_axis: Arc<[f64]> = Arc::from(
        (0..n_rows)
            .map(|i| 1e-3 * 1.05f64.powi(i as i32))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );

    println!("=== rust-mc-sim stress: {n} kernels ===");
    println!("  shape per kernel: {n_rows} rows × {n_cols} cols, rank = {rank}");
    println!();

    // Generate N synthetic matrices. Deterministic per-index; cheap.
    let t_gen = Instant::now();
    let mats: Vec<Vec<f64>> = (0..n)
        .map(|s| {
            let mut m = vec![0.0_f64; n_rows * n_cols];
            for i in 0..n_rows {
                for j in 0..n_cols {
                    m[i * n_cols + j] = ((i + 1) as f64).ln() * ((j + 1) as f64)
                        + 0.001 * (s as f64) * ((i + j) as f64).cos();
                }
            }
            m
        })
        .collect();
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    println!("  generated {n} matrices in {gen_ms:.0} ms");

    let inputs: Vec<KernelInput<'_>> = mats
        .iter()
        .map(|m| KernelInput {
            matrix_row_major: m,
            row_axis: Arc::clone(&row_axis),
            n_rows,
            n_cols,
            rank,
        })
        .collect();

    // Warm up faer's thread-local SVD state before measuring (the
    // first SVD call per thread allocates lazily; the warm-up
    // amortises that out of both timings).
    let warmup_inputs: Vec<KernelInput<'_>> = inputs.iter().take(64).cloned().collect();
    let _ = from_data_many(&warmup_inputs);

    // Sequential reference run.
    let t_seq = Instant::now();
    let seq: Vec<SvdKernel> = from_data_many(&inputs);
    let seq_ms = t_seq.elapsed().as_secs_f64() * 1000.0;
    let seq_bytes = total_kernel_bytes(&seq);
    println!();
    println!("  sequential:");
    println!(
        "    wall:        {seq_ms:>8.0} ms ({:.1} µs/kernel)",
        seq_ms * 1e3 / n as f64
    );
    println!(
        "    memory:      {:>8.1} MB ({:.1} KB/kernel)",
        seq_bytes as f64 / (1024.0 * 1024.0),
        seq_bytes as f64 / 1024.0 / n as f64
    );

    // Drop seq to free memory before the parallel run on tight boxes.
    drop(seq);

    #[cfg(feature = "parallel")]
    {
        let t_par = Instant::now();
        let par: Vec<SvdKernel> = from_data_many_par(&inputs);
        let par_ms = t_par.elapsed().as_secs_f64() * 1000.0;
        let par_bytes = total_kernel_bytes(&par);
        let speedup = seq_ms / par_ms;
        println!();
        println!("  parallel (rayon, all cores):");
        println!(
            "    wall:        {par_ms:>8.0} ms ({:.1} µs/kernel)",
            par_ms * 1e3 / n as f64
        );
        println!(
            "    memory:      {:>8.1} MB ({:.1} KB/kernel)",
            par_bytes as f64 / (1024.0 * 1024.0),
            par_bytes as f64 / 1024.0 / n as f64
        );
        println!("    speedup:     {speedup:>8.2}× over sequential");

        // Project to 200k.
        let proj_seq = seq_ms * 200_000.0 / n as f64 / 1000.0;
        let proj_par = par_ms * 200_000.0 / n as f64 / 1000.0;
        let proj_mem = par_bytes as f64 * 200_000.0 / n as f64 / 1024.0 / 1024.0 / 1024.0;
        println!();
        println!("  projection to 200k kernels (linear):");
        println!("    sequential:  ~{proj_seq:>5.1} s wall");
        println!("    parallel:    ~{proj_par:>5.1} s wall");
        println!("    memory:      ~{proj_mem:>5.1} GB");
    }
    #[cfg(not(feature = "parallel"))]
    {
        println!();
        println!("  (built without `--features parallel`; rebuild with");
        println!("   `cargo run --release --features parallel --example 05_stress_200k`");
        println!("   to see the rayon-driven speedup over sequential.)");
    }
}
