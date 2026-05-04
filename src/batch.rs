//! Sequential and (with `feature = "parallel"`) rayon batch APIs
//! for many independent decompositions.

use std::sync::Arc;

use crate::cdf::LogDecimatedCdf;
use crate::cp::{CpDecomposition, cp_greedy_rank1};
use crate::svd::{Svd, SvdKernel};

/// Input descriptor for a single SVD decomposition.
pub struct SvdInput<'a> {
    pub matrix_row_major: &'a [f64],
    pub n_rows: usize,
    pub n_cols: usize,
}

/// Sequential batch: decompose every input, return one [`Svd`] per.
pub fn decompose_many(inputs: &[SvdInput<'_>]) -> Vec<Svd> {
    inputs
        .iter()
        .map(|i| Svd::decompose(i.matrix_row_major, i.n_rows, i.n_cols))
        .collect()
}

/// Parallel batch: decompose every input across rayon's thread pool.
/// Order is preserved (output index `k` corresponds to input index `k`).
#[cfg(feature = "parallel")]
pub fn decompose_many_par(inputs: &[SvdInput<'_>]) -> Vec<Svd> {
    use rayon::prelude::*;
    inputs
        .par_iter()
        .map(|i| Svd::decompose(i.matrix_row_major, i.n_rows, i.n_cols))
        .collect()
}

/// Input descriptor for building one [`SvdKernel`].
#[derive(Clone)]
pub struct KernelInput<'a> {
    pub matrix_row_major: &'a [f64],
    pub row_axis: Arc<[f64]>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub rank: usize,
}

/// Sequential batch: build one [`SvdKernel`] per input.
pub fn from_data_many(inputs: &[KernelInput<'_>]) -> Vec<SvdKernel> {
    inputs
        .iter()
        .map(|i| {
            SvdKernel::from_data(
                i.matrix_row_major,
                Arc::clone(&i.row_axis),
                i.n_rows,
                i.n_cols,
                i.rank,
            )
        })
        .collect()
}

/// Parallel batch: build one [`SvdKernel`] per input across rayon.
#[cfg(feature = "parallel")]
pub fn from_data_many_par(inputs: &[KernelInput<'_>]) -> Vec<SvdKernel> {
    use rayon::prelude::*;
    inputs
        .par_iter()
        .map(|i| {
            SvdKernel::from_data(
                i.matrix_row_major,
                Arc::clone(&i.row_axis),
                i.n_rows,
                i.n_cols,
                i.rank,
            )
        })
        .collect()
}

/// Input descriptor for one CP/PARAFAC decomposition.
pub struct CpInput<'a> {
    pub tensor: &'a [f64],
    pub n_a: usize,
    pub n_b: usize,
    pub n_c: usize,
    pub max_rank: usize,
    pub max_iter: usize,
    pub tol: f64,
}

impl CpInput<'_> {
    /// Default convergence parameters (`max_iter = 200`, `tol = 1e-9`).
    pub fn with_defaults<'a>(
        tensor: &'a [f64],
        n_a: usize,
        n_b: usize,
        n_c: usize,
        max_rank: usize,
    ) -> CpInput<'a> {
        CpInput {
            tensor,
            n_a,
            n_b,
            n_c,
            max_rank,
            max_iter: 200,
            tol: 1e-9,
        }
    }
}

/// Sequential batch CP/PARAFAC.
pub fn cp_many(inputs: &[CpInput<'_>]) -> Vec<CpDecomposition> {
    inputs
        .iter()
        .map(|i| cp_greedy_rank1(i.tensor, i.n_a, i.n_b, i.n_c, i.max_rank, i.max_iter, i.tol))
        .collect()
}

/// Parallel batch CP/PARAFAC.
#[cfg(feature = "parallel")]
pub fn cp_many_par(inputs: &[CpInput<'_>]) -> Vec<CpDecomposition> {
    use rayon::prelude::*;
    inputs
        .par_iter()
        .map(|i| cp_greedy_rank1(i.tensor, i.n_a, i.n_b, i.n_c, i.max_rank, i.max_iter, i.tol))
        .collect()
}

/// Input descriptor for one [`LogDecimatedCdf`].
pub struct CdfInput<'a> {
    pub intensities: &'a [Vec<f64>],
    pub axis: &'a [f64],
    pub n_decimated: usize,
}

/// Sequential batch CDF construction.
pub fn cdf_many(inputs: &[CdfInput<'_>]) -> Vec<LogDecimatedCdf> {
    inputs
        .iter()
        .map(|i| LogDecimatedCdf::from_intensities(i.intensities, i.axis, i.n_decimated))
        .collect()
}

/// Parallel batch CDF construction.
#[cfg(feature = "parallel")]
pub fn cdf_many_par(inputs: &[CdfInput<'_>]) -> Vec<LogDecimatedCdf> {
    use rayon::prelude::*;
    inputs
        .par_iter()
        .map(|i| LogDecimatedCdf::from_intensities(i.intensities, i.axis, i.n_decimated))
        .collect()
}

/// Aggregate memory used by a slice of [`SvdKernel`]s.
/// Useful for a quick check at load time: "did I just allocate the
/// expected ~N * rank * n_rows * 8 bytes, or did something explode?"
pub fn total_kernel_bytes(kernels: &[SvdKernel]) -> usize {
    kernels.iter().map(|k| k.memory_bytes()).sum()
}

/// Aggregate memory used by a slice of [`CpDecomposition`]s.
pub fn total_cp_bytes(decomps: &[CpDecomposition]) -> usize {
    decomps.iter().map(|d| d.memory_bytes()).sum()
}

/// Aggregate memory used by a slice of [`LogDecimatedCdf`]s.
pub fn total_cdf_bytes(cdfs: &[LogDecimatedCdf]) -> usize {
    cdfs.iter().map(|c| c.memory_bytes()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_matrix(seed: usize, n_rows: usize, n_cols: usize) -> Vec<f64> {
        // Deterministic synthetic data depending on `seed` so each
        // batch entry is distinguishable.
        let mut out = vec![0.0; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                let x = (i + 1) as f64;
                let t = (j + 1) as f64;
                out[i * n_cols + j] =
                    (x.ln() * t).sin() + 0.1 * (seed as f64 + 1.0) * (i + j) as f64;
            }
        }
        out
    }

    #[test]
    fn sequential_batch_preserves_per_input_decomposition() {
        let n_rows = 50;
        let n_cols = 4;
        let mats: Vec<Vec<f64>> = (0..8).map(|s| synth_matrix(s, n_rows, n_cols)).collect();
        let inputs: Vec<SvdInput<'_>> = mats
            .iter()
            .map(|m| SvdInput {
                matrix_row_major: m,
                n_rows,
                n_cols,
            })
            .collect();
        let batch = decompose_many(&inputs);
        assert_eq!(batch.len(), 8);
        for (k, svd) in batch.iter().enumerate() {
            // Compare round-trip recon against an independently-built
            // decomposition for the same input.
            let solo = Svd::decompose(&mats[k], n_rows, n_cols);
            let r1 = svd.reconstruct(svd.rank);
            let r2 = solo.reconstruct(solo.rank);
            for (a, b) in r1.iter().zip(r2.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_batch_matches_sequential() {
        let n_rows = 80;
        let n_cols = 5;
        let mats: Vec<Vec<f64>> = (0..16).map(|s| synth_matrix(s, n_rows, n_cols)).collect();
        let inputs: Vec<SvdInput<'_>> = mats
            .iter()
            .map(|m| SvdInput {
                matrix_row_major: m,
                n_rows,
                n_cols,
            })
            .collect();
        let seq = decompose_many(&inputs);
        let par = decompose_many_par(&inputs);
        assert_eq!(seq.len(), par.len());
        for (a, b) in seq.iter().zip(par.iter()) {
            // Reconstructions match; raw factors may have free sign
            // flips (faer-internal) so we don't compare them directly.
            let ra = a.reconstruct(a.rank);
            let rb = b.reconstruct(b.rank);
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-10);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_kernel_batch_matches_sequential() {
        let n_rows = 60;
        let n_cols = 4;
        let row_axis: Arc<[f64]> = Arc::from(
            (0..n_rows)
                .map(|i| 1e-3 * 1.05f64.powi(i as i32))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let mats: Vec<Vec<f64>> = (0..8).map(|s| synth_matrix(s, n_rows, n_cols)).collect();
        let inputs: Vec<KernelInput<'_>> = mats
            .iter()
            .map(|m| KernelInput {
                matrix_row_major: m,
                row_axis: Arc::clone(&row_axis),
                n_rows,
                n_cols,
                rank: 3,
            })
            .collect();
        let seq = from_data_many(&inputs);
        let par = from_data_many_par(&inputs);
        assert_eq!(seq.len(), par.len());
        // Compare reconstructions at column 1.
        for (a, b) in seq.iter().zip(par.iter()) {
            let ca = a.coeffs_at_col(1);
            let cb = b.coeffs_at_col(1);
            for row in 0..n_rows {
                let ra = a.reconstruct_at(row, &ca);
                let rb = b.reconstruct_at(row, &cb);
                assert!((ra - rb).abs() < 1e-10);
            }
        }
    }
}
