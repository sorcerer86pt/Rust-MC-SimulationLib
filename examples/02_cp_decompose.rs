//! CP/PARAFAC decomposition of a synthetic 3-tensor with mixed structure.
//!
//! Reports per-rank reconstruction error and memory cost so callers
//! can pick a working rank for their problem.
//!
//! Run with: `cargo run --release --example 02_cp_decompose`

use rust_mc_sim::{cp_greedy_rank1, max_abs_error, relative_l2_error};

fn main() {
    // A 3-tensor with three distinguishable components: a smooth
    // multiplicative term, a cross-coupled cosine, and a sparse
    // bump. CP should pick all three up by rank ≤ 4.
    let n_a = 60;
    let n_b = 6;
    let n_c = 12;

    let mut tensor = vec![0.0_f64; n_a * n_b * n_c];
    for i in 0..n_a {
        for t in 0..n_b {
            for l in 0..n_c {
                let v1 = (i as f64 + 1.0).ln() * (t as f64 + 1.0).sqrt() * (l as f64 + 1.0);
                let v2 = ((i + l) as f64 / 7.0).cos() * ((t + 1) as f64);
                let v3 = if (i + 30).abs_diff(50) < 2 && t == 3 && l == 7 {
                    5.0
                } else {
                    0.0
                };
                tensor[i * n_b * n_c + t * n_c + l] = v1 + v2 + v3;
            }
        }
    }

    let cp = cp_greedy_rank1(&tensor, n_a, n_b, n_c, 8, 500, 1e-10);
    println!(
        "shape: ({}, {}, {})  fitted rank: {}",
        n_a, n_b, n_c, cp.rank
    );
    println!(
        "cp memory: {} bytes (factor matrices + per-component σ)",
        cp.memory_bytes()
    );
    println!();
    println!("rank  rel_L2     max_abs    cp_KB");
    println!("----  ---------  ---------  -----");
    for k in 1..=cp.rank {
        let recon = cp.reconstruct(k);
        let l2 = relative_l2_error(&tensor, &recon);
        let abs_err = max_abs_error(&tensor, &recon);
        let bytes = k * (n_a + n_b + n_c + 1) * std::mem::size_of::<f64>();
        println!(
            "{:>4}  {:>9.2e}  {:>9.2e}  {:>5.2}",
            k,
            l2,
            abs_err,
            bytes as f64 / 1024.0
        );
    }
}
