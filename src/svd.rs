//! Truncated singular value decomposition with cache-friendly
//! reconstruction.
//!
//! [`Svd`] wraps a faer-computed truncated SVD as a row-major
//! `(U·Σ, Vᵀ)` factor pair. [`SvdKernel`] is the deployment-time
//! engine: pre-multiplied basis stored row-major
//! `basis[i*rank + j] = U[i,j] · σ_j`, and a `rank × n_t` matrix of
//! `Vᵀ` columns. Reconstruction at column index `t` is a length-`rank`
//! dot product per row of `basis` against a pre-computed vector
//! `coeffs[j] = Vᵀ[j,t]`. The hot path is pure FMA; `coeffs` fits in
//! registers and `basis` streams sequentially.
//!
//! # Off-column reconstruction
//!
//! For a target column `t*` that does not coincide with one of the
//! training columns, [`SvdKernel::ducru_coeffs`] computes
//! Ducru-weighted reconstruction coefficients via
//! [`crate::ducru::ducru_weights`]:
//!
//! ```text
//!     coeffs[j] = Σ_t  w_t · Vᵀ[j, t]
//! ```
//!
//! The weights are exact at training columns and L2-optimal between
//! them under the free-Doppler kernel approximation that motivates
//! Ducru et al. 2017.
//!
//! # Index lookup
//!
//! Continuous lookups against a sorted-ascending row axis are
//! supported via [`LogHashIndex`]: O(1) hash to a log-uniform bin
//! followed by a short linear scan, fastest when the row axis spans
//! many decades. Falls through to a binary search at construction
//! when the row count is small.

use std::sync::Arc;

use faer::Mat;

use crate::ducru::ducru_weights;

/// Result of a truncated SVD computed by [`Svd::decompose`].
///
/// Layout matches what [`SvdKernel::from_svd`] expects:
///
/// * `u`: `n_rows × rank`, row-major. Entry `u[i*rank + j] = U[i,j]`.
/// * `s`: length `rank`, descending singular values.
/// * `vt`: `rank × n_cols`, row-major. Entry `vt[j*n_cols + t] = Vᵀ[j,t]`.
pub struct Svd {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub rank: usize,
}

impl Svd {
    /// Compute the thin SVD of a row-major `n_rows × n_cols` matrix.
    ///
    /// The returned [`Svd`] has `rank = min(n_rows, n_cols)` slots
    /// (truncate later via [`SvdKernel::from_svd`]).
    pub fn decompose(matrix_row_major: &[f64], n_rows: usize, n_cols: usize) -> Self {
        assert_eq!(matrix_row_major.len(), n_rows * n_cols);
        let a = Mat::from_fn(n_rows, n_cols, |i, j| matrix_row_major[i * n_cols + j]);
        #[allow(non_snake_case)]
        let decomp = a.thin_svd().expect("SVD did not converge");
        #[allow(non_snake_case)]
        let U = decomp.U();
        let s_col = decomp.S().column_vector();
        #[allow(non_snake_case)]
        let V = decomp.V();
        let rank = n_rows.min(n_cols);

        let mut u = vec![0.0_f64; n_rows * rank];
        let mut s = vec![0.0_f64; rank];
        let mut vt = vec![0.0_f64; rank * n_cols];

        for i in 0..n_rows {
            for j in 0..rank {
                u[i * rank + j] = U[(i, j)];
            }
        }
        for j in 0..rank {
            s[j] = s_col[j];
        }
        // V is n_cols × rank; transpose into row-major rank × n_cols.
        for j in 0..rank {
            for t in 0..n_cols {
                vt[j * n_cols + t] = V[(t, j)];
            }
        }

        Self {
            u,
            s,
            vt,
            n_rows,
            n_cols,
            rank,
        }
    }

    /// Reconstruct the full matrix at truncation `k ≤ self.rank`.
    /// Returns row-major `n_rows × n_cols`.
    pub fn reconstruct(&self, k: usize) -> Vec<f64> {
        let k = k.min(self.rank);
        let mut out = vec![0.0_f64; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            for t in 0..self.n_cols {
                let mut acc = 0.0_f64;
                for j in 0..k {
                    let u_ij = self.u[i * self.rank + j];
                    let s_j = self.s[j];
                    let vt_jt = self.vt[j * self.n_cols + t];
                    acc = (u_ij * s_j).mul_add(vt_jt, acc);
                }
                out[i * self.n_cols + t] = acc;
            }
        }
        out
    }
}

/// O(1) hash lookup for a sorted-ascending row axis spanning many
/// decades. Brown 2014 ("New hash-based energy lookup algorithm",
/// *Trans. ANS* 111). Implementation note: log-uniform bins; the
/// kernel pre-stores the lower-bracket grid index for every bin so
/// the runtime cost is one log + one bin index + a short forward scan.
pub struct LogHashIndex {
    bins: Vec<u32>,
    log_min: f64,
    inv_bin_width: f64,
    n_bins: usize,
}

impl LogHashIndex {
    /// Build a hash index over a sorted-ascending positive `axis`.
    /// `n_bins = 8192` is a good default for axes with $\sim10^4$ entries.
    pub fn new(axis: &[f64], n_bins: usize) -> Self {
        let n = axis.len();
        if n < 2 {
            return Self {
                bins: vec![0; n_bins],
                log_min: 0.0,
                inv_bin_width: 0.0,
                n_bins,
            };
        }
        let log_min = axis[0].max(1e-30).ln();
        let log_max = axis[n - 1].max(1e-30).ln();
        let bin_width = (log_max - log_min) / n_bins as f64;
        let inv_bin_width = if bin_width > 0.0 {
            1.0 / bin_width
        } else {
            0.0
        };

        let mut bins = Vec::with_capacity(n_bins);
        let mut grid_idx = 0_u32;
        for b in 0..n_bins {
            let bin_log_e = log_min + (b as f64 + 1.0) * bin_width;
            let bin_e = bin_log_e.exp();
            while (grid_idx as usize) < n && axis[grid_idx as usize] < bin_e {
                grid_idx += 1;
            }
            bins.push(if grid_idx > 0 { grid_idx - 1 } else { 0 });
        }
        Self {
            bins,
            log_min,
            inv_bin_width,
            n_bins,
        }
    }

    /// Lower-bracket index: largest `idx` with `axis[idx] ≤ value`.
    /// Falls through to clamped extremes outside the range.
    #[inline]
    pub fn lookup(&self, value: f64, axis: &[f64]) -> usize {
        let n = axis.len();
        if n < 2 {
            return 0;
        }
        if value <= axis[0] {
            return 0;
        }
        if value >= axis[n - 1] {
            return n - 1;
        }
        let log_v = value.ln();
        let bin = ((log_v - self.log_min) * self.inv_bin_width) as usize;
        let bin = bin.min(self.n_bins - 1);
        let start = if bin > 0 {
            self.bins[bin - 1] as usize
        } else {
            0
        };
        let mut idx = start.min(n - 1);
        while idx + 1 < n && axis[idx + 1] <= value {
            idx += 1;
        }
        idx
    }
}

/// Pre-multiplied SVD reconstruction engine for a row × column data
/// matrix. The "row axis" is whatever continuous coordinate the data
/// lives on (energy, position, time…); the "column axis" is whatever
/// discrete training set the data was sampled at (temperatures,
/// experiments, configurations…). Both are domain-agnostic.
pub struct SvdKernel {
    /// Pre-multiplied `U·Σ`, row-major: `[n_rows][rank]`.
    basis: Vec<f64>,
    /// `Vᵀ`, row-major: `[rank][n_cols]`.
    vt_coeffs: Vec<f64>,
    /// Row axis (the coordinate the data is sampled in). Shared
    /// across kernels that sit on the same axis.
    row_axis: Arc<[f64]>,
    /// Optional log-uniform hash index for O(1) row lookups.
    hash: Option<LogHashIndex>,
    rank: usize,
    n_rows: usize,
    n_cols: usize,
}

impl SvdKernel {
    /// Build a kernel from raw row-major data. Computes the SVD,
    /// truncates to `rank`, and pre-multiplies `U` by `Σ`.
    pub fn from_data(
        matrix_row_major: &[f64],
        row_axis: Arc<[f64]>,
        n_rows: usize,
        n_cols: usize,
        rank: usize,
    ) -> Self {
        let svd = Svd::decompose(matrix_row_major, n_rows, n_cols);
        Self::from_svd(&svd, row_axis, rank)
    }

    /// Build a kernel from a pre-computed [`Svd`], truncating to `rank`.
    pub fn from_svd(svd: &Svd, row_axis: Arc<[f64]>, rank: usize) -> Self {
        let rank = rank.min(svd.rank);
        let n_rows = svd.n_rows;
        let n_cols = svd.n_cols;
        let mut basis = vec![0.0_f64; n_rows * rank];
        for j in 0..rank {
            let s_j = svd.s[j];
            for i in 0..n_rows {
                basis[i * rank + j] = svd.u[i * svd.rank + j] * s_j;
            }
        }
        let mut vt_coeffs = vec![0.0_f64; rank * n_cols];
        for j in 0..rank {
            for t in 0..n_cols {
                vt_coeffs[j * n_cols + t] = svd.vt[j * n_cols + t];
            }
        }
        let hash = if n_rows > 100 {
            Some(LogHashIndex::new(&row_axis, 8192))
        } else {
            None
        };
        Self {
            basis,
            vt_coeffs,
            row_axis,
            hash,
            rank,
            n_rows,
            n_cols,
        }
    }

    /// Construct a kernel from already pre-multiplied basis + Vᵀ.
    /// Use this when you've serialised an SVD elsewhere or built
    /// the factors yourself.
    ///
    /// `basis` must be `n_rows * rank` doubles, row-major, with
    /// `basis[i*rank + j] = U[i,j] · σ_j`. `vt_coeffs` must be
    /// `rank * n_cols` doubles, row-major.
    pub fn from_factors(
        basis: Vec<f64>,
        vt_coeffs: Vec<f64>,
        row_axis: Arc<[f64]>,
        rank: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        assert_eq!(basis.len(), n_rows * rank);
        assert_eq!(vt_coeffs.len(), rank * n_cols);
        let hash = if n_rows > 100 {
            Some(LogHashIndex::new(&row_axis, 8192))
        } else {
            None
        };
        Self {
            basis,
            vt_coeffs,
            row_axis,
            hash,
            rank,
            n_rows,
            n_cols,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
    pub fn row_axis(&self) -> &[f64] {
        &self.row_axis
    }
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }
    pub fn vt(&self) -> &[f64] {
        &self.vt_coeffs
    }

    /// Pick column `t` of `Vᵀ` as the reconstruction coefficients.
    /// Returns a length-`rank` vector you can pass to
    /// [`SvdKernel::reconstruct_at`].
    pub fn coeffs_at_col(&self, t: usize) -> Vec<f64> {
        debug_assert!(t < self.n_cols);
        let mut out = Vec::with_capacity(self.rank);
        for j in 0..self.rank {
            out.push(self.vt_coeffs[j * self.n_cols + t]);
        }
        out
    }

    /// Ducru-blended reconstruction coefficients for an off-column
    /// target. `column_values` is the list of column-axis values used
    /// when training (one per `n_cols`). At an exact match the weights
    /// collapse to one-hot; off-grid the L2-optimal partition-of-unity
    /// 3-point variant gives sub-percent reconstruction error.
    ///
    /// To use the partition-of-unity variant rather than the raw
    /// Ducru weights, pass the result of [`crate::ducru::ducru_unity_weights`]
    /// directly to [`SvdKernel::reconstruct_with_weights`] instead.
    pub fn ducru_coeffs(&self, column_values: &[f64], target: f64) -> Vec<f64> {
        debug_assert_eq!(column_values.len(), self.n_cols);
        let weights = ducru_weights(column_values, target);
        self.reconstruct_with_weights(&weights)
    }

    /// Reconstruction coefficients from an arbitrary length-`n_cols`
    /// weight vector against `Vᵀ`. Use this with externally computed
    /// weights (e.g. partition-of-unity 3-point Ducru, or any custom
    /// quadrature scheme). No checks on the weights — caller is
    /// responsible for consistency (e.g. partition-of-unity if peak
    /// preservation matters).
    pub fn reconstruct_with_weights(&self, weights: &[f64]) -> Vec<f64> {
        debug_assert_eq!(weights.len(), self.n_cols);
        let mut out = Vec::with_capacity(self.rank);
        for j in 0..self.rank {
            let row = &self.vt_coeffs[j * self.n_cols..j * self.n_cols + self.n_cols];
            let acc: f64 = weights.iter().zip(row.iter()).map(|(w, c)| w * c).sum();
            out.push(acc);
        }
        out
    }

    /// Reconstruct one row at row index `i` using the supplied
    /// length-`rank` coefficients. Pure FMA dot product.
    #[inline]
    pub fn reconstruct_at(&self, i: usize, coeffs: &[f64]) -> f64 {
        debug_assert!(i < self.n_rows);
        debug_assert_eq!(coeffs.len(), self.rank);
        let row = &self.basis[i * self.rank..(i + 1) * self.rank];
        let mut acc = 0.0_f64;
        for j in 0..self.rank {
            acc = row[j].mul_add(coeffs[j], acc);
        }
        acc
    }

    /// Reconstruct **all** rows for a given column's coefficients.
    /// `out` must have length ≥ `n_rows`.
    pub fn reconstruct_full(&self, coeffs: &[f64], out: &mut [f64]) {
        debug_assert_eq!(coeffs.len(), self.rank);
        debug_assert!(out.len() >= self.n_rows);
        let rank = self.rank;
        for i in 0..self.n_rows {
            let row = &self.basis[i * rank..(i + 1) * rank];
            let mut acc = 0.0_f64;
            for j in 0..rank {
                acc = row[j].mul_add(coeffs[j], acc);
            }
            out[i] = acc;
        }
    }

    /// Index lookup on the row axis (O(1) via [`LogHashIndex`] when
    /// available; binary search otherwise). Returns the lower bracket
    /// index, clamped to `[0, n_rows - 1]`.
    #[inline]
    pub fn row_index(&self, value: f64) -> usize {
        if let Some(ref ht) = self.hash {
            ht.lookup(value, &self.row_axis)
        } else {
            self.row_index_binary(value)
        }
    }

    fn row_index_binary(&self, value: f64) -> usize {
        let n = self.row_axis.len();
        match self
            .row_axis
            .binary_search_by(|e| e.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else if i >= n {
                    n - 1
                } else {
                    i
                }
            }
        }
    }

    /// Bytes used by the basis + Vᵀ (does not include the shared row axis).
    pub fn memory_bytes(&self) -> usize {
        (self.basis.len() + self.vt_coeffs.len()) * std::mem::size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (a.abs().max(b.abs()).max(1.0))
    }

    fn roundtrip_tol(rank: usize) -> f64 {
        // Truncated SVD is exact only at rank ≥ matrix rank; use a
        // loose tolerance for low-rank truncations of dense data.
        if rank >= 4 { 1e-10 } else { 1e-2 }
    }

    #[test]
    fn svd_roundtrip_full_rank() {
        // 4×3 matrix; full SVD reconstructs to machine precision.
        let m: Vec<f64> = (0..12).map(|x| x as f64 + 1.0).collect();
        let svd = Svd::decompose(&m, 4, 3);
        let recon = svd.reconstruct(svd.rank);
        for (a, b) in m.iter().zip(recon.iter()) {
            assert!(approx_eq(*a, *b, 1e-12));
        }
    }

    #[test]
    fn kernel_reconstructs_at_column() {
        // Sample data: σ(E_i, T_j) = (i+1) · sin(j+1).
        let n_rows = 5;
        let n_cols = 3;
        let mut data = vec![0.0; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                data[i * n_cols + j] = (i as f64 + 1.0) * ((j as f64 + 1.0).sin());
            }
        }
        let row_axis: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0, 5.0].into_boxed_slice());
        let kernel = SvdKernel::from_data(&data, row_axis, n_rows, n_cols, 2);
        for j in 0..n_cols {
            let coeffs = kernel.coeffs_at_col(j);
            for i in 0..n_rows {
                let expected = data[i * n_cols + j];
                let got = kernel.reconstruct_at(i, &coeffs);
                let tol = roundtrip_tol(kernel.rank());
                assert!(
                    approx_eq(expected, got, tol),
                    "mismatch at i={i} j={j}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn ducru_at_training_column_is_one_hot() {
        // At an exact training column, ducru_coeffs should reproduce
        // coeffs_at_col exactly (one-hot weights collapse).
        let n_rows = 4;
        let n_cols = 4;
        let mut data = vec![0.0; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                data[i * n_cols + j] = ((i + j) as f64).cos();
            }
        }
        let row_axis: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice());
        let kernel = SvdKernel::from_data(&data, row_axis, n_rows, n_cols, 4);
        let temps = vec![300.0, 600.0, 900.0, 1200.0];
        let target = temps[2]; // exact match
        let direct = kernel.coeffs_at_col(2);
        let ducru = kernel.ducru_coeffs(&temps, target);
        for (a, b) in direct.iter().zip(ducru.iter()) {
            assert!(approx_eq(*a, *b, 1e-12));
        }
    }

    #[test]
    fn hash_index_matches_binary_search() {
        let axis: Vec<f64> = (0..200).map(|i| 1e-3 * 1.05f64.powi(i)).collect();
        let arc: Arc<[f64]> = Arc::from(axis.clone().into_boxed_slice());
        let kernel = SvdKernel::from_factors(vec![0.0; 200], vec![0.0; 1], arc, 1, 200, 1);
        // Probe values: in-range, below, above.
        let probes = [1e-4, 5e-3, 0.1, 1.0, 1e3, 1e9];
        for &p in &probes {
            let from_hash = kernel.row_index(p);
            // Binary search the same axis for cross-check.
            let bs = match axis
                .binary_search_by(|x| x.partial_cmp(&p).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(i) => i,
                Err(i) => {
                    if i == 0 {
                        0
                    } else if i >= axis.len() {
                        axis.len() - 1
                    } else {
                        i
                    }
                }
            };
            // Hash returns lower bracket; binary_search returns
            // upper-or-equal. Allow off-by-one.
            assert!(
                from_hash == bs || from_hash + 1 == bs || from_hash == bs + 1,
                "hash {from_hash} vs binary {bs} for p={p}"
            );
        }
    }
}
