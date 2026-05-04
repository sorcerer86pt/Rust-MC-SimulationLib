//! Log-decimated cumulative distribution `F_k(x)` for inverse-transform
//! sampling of categorical outcomes whose probabilities depend on a
//! continuous coordinate.
//!
//! Given a 2-tensor `σ_k(x)` of "intensities" for category `k ∈ 0..K`
//! at sample point `x`, we typically want to:
//!
//! 1. compute the total intensity `Σ(x) = Σ_k σ_k(x)` (e.g. for
//!    deciding *whether* an event happens — this is the main XS sum);
//! 2. sample which category `k` was selected, with probability
//!    `σ_k(x) / Σ(x)`.
//!
//! Storing the per-category intensities at full resolution and
//! recomputing the CDF at every sample is expensive. The categorical
//! distribution `F_k(x) = Σ_{k'≤k} σ_{k'}(x) / Σ(x)` is bounded in
//! `[0, 1]` and (for typical scientific tabulations) varies smoothly
//! in `x` because peaks in the individual `σ_k` cancel in the ratio.
//!
//! [`LogDecimatedCdf`] tabulates `F_k(x)` on a log-spaced grid (default
//! ~200 points) and samples by bracketed binary scan with linear
//! interpolation in `log(x)`. For a 41-category × 6-column input
//! this drops storage from hundreds of MB to ~400 KB at sub-pcm
//! reconstruction error.
//!
//! For an off-column target (e.g. an off-grid temperature) callers
//! should pre-blend the per-column intensities via the Ducru weights
//! in [`crate::ducru`] before constructing the CDF, then store only
//! a single column (`n_cols = 1`).
//!
//! # Quick example
//!
//! ```
//! use rust_mc_sim::cdf::LogDecimatedCdf;
//! // Three categories whose probabilities depend on x ∈ [1, 100].
//! // Simple smooth shapes for illustration.
//! let xs: Vec<f64> = (0..50).map(|i| 1.0 * 1.1f64.powi(i)).collect();
//! let n_cat = 3;
//! let mut intensities = vec![vec![0.0_f64; xs.len()]; n_cat];
//! for (j, &x) in xs.iter().enumerate() {
//!     intensities[0][j] = 1.0 / x;             // declines with x
//!     intensities[1][j] = (x.ln() / 5.0).max(0.0); // grows with x
//!     intensities[2][j] = 1.0;                  // flat
//! }
//! let cdf = LogDecimatedCdf::from_intensities(&intensities, &xs, 200);
//! // Sample at x = 10 with ξ = 0.42:
//! let k = cdf.sample(10.0, 0.42);
//! assert!(k < 3);
//! ```

/// Pre-tabulated cumulative distribution `F_k(x)` over `K` categories.
///
/// Layout: `cdf_flat[ed * n_cols * n_categories + col * n_categories + k]`.
/// `n_cols == 1` for the typical case where columns have already been
/// pre-blended (e.g. via Ducru weights at a target temperature).
pub struct LogDecimatedCdf {
    /// Number of categories (`K`).
    pub n_categories: usize,
    /// Number of column slices stored. Use `1` for the pre-blended
    /// single-target case; `>1` if you want to interpolate at lookup
    /// time (caller manages column selection).
    pub n_cols: usize,
    /// Number of decimated points on the continuous axis.
    pub n_points: usize,
    /// `log10(x_min)` of the decimated grid.
    pub log_x_min: f64,
    /// `log10(x_max)` of the decimated grid.
    pub log_x_max: f64,
    /// Flat data, row-major in (point, col, category).
    pub cdf_flat: Vec<f64>,
}

impl LogDecimatedCdf {
    /// Construct a CDF from per-category intensities sampled on a
    /// shared, sorted-ascending positive `axis`.
    ///
    /// `intensities[k][j]` is the intensity of category `k` at
    /// `axis[j]`. `n_decimated` is the number of log-spaced points to
    /// keep; defaults of 200 give sub-pcm reconstruction accuracy on
    /// most physically smooth data.
    ///
    /// Stores a single column (`n_cols = 1`) — pre-blend across
    /// configurations before calling this if you want off-grid
    /// support.
    pub fn from_intensities(intensities: &[Vec<f64>], axis: &[f64], n_decimated: usize) -> Self {
        assert!(!intensities.is_empty(), "need at least one category");
        let n_categories = intensities.len();
        for row in intensities.iter() {
            assert_eq!(
                row.len(),
                axis.len(),
                "every category must be sampled at every axis point"
            );
        }
        let n_dec = n_decimated.max(2);
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        for &x in axis {
            if x > 0.0 {
                if x < x_min {
                    x_min = x;
                }
                if x > x_max {
                    x_max = x;
                }
            }
        }
        assert!(
            x_min.is_finite() && x_max.is_finite() && x_min < x_max,
            "axis must contain at least two distinct positive values"
        );
        let log_x_min = x_min.log10();
        let log_x_max = x_max.log10();

        let bsearch = |x: f64| -> (usize, f64) {
            if x <= axis[0] {
                return (0, 0.0);
            }
            if x >= axis[axis.len() - 1] {
                return (axis.len() - 1, 0.0);
            }
            let mut lo = 0usize;
            let mut hi = axis.len() - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if axis[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let span = axis[hi] - axis[lo];
            let alpha = if span > 0.0 {
                (x - axis[lo]) / span
            } else {
                0.0
            };
            (lo, alpha)
        };

        let mut cdf_flat = vec![0.0_f64; n_dec * n_categories];
        for ed in 0..n_dec {
            let frac = ed as f64 / (n_dec - 1) as f64;
            let log_x = log_x_min + frac * (log_x_max - log_x_min);
            let x = 10f64.powf(log_x);
            let (idx, alpha) = bsearch(x);
            let nxt = (idx + 1).min(axis.len() - 1);
            let mut total = 0.0_f64;
            for k in 0..n_categories {
                let lo = intensities[k][idx].max(0.0);
                let hi = intensities[k][nxt].max(0.0);
                let v = lo + alpha * (hi - lo);
                total += v;
            }
            let row_off = ed * n_categories;
            if total <= 1e-30 {
                cdf_flat[row_off + n_categories - 1] = 1.0;
                continue;
            }
            let inv = 1.0 / total;
            let mut running = 0.0_f64;
            for k in 0..n_categories {
                let lo = intensities[k][idx].max(0.0);
                let hi = intensities[k][nxt].max(0.0);
                let v = lo + alpha * (hi - lo);
                running += v * inv;
                cdf_flat[row_off + k] = running;
            }
            cdf_flat[row_off + n_categories - 1] = 1.0;
        }

        Self {
            n_categories,
            n_cols: 1,
            n_points: n_dec,
            log_x_min,
            log_x_max,
            cdf_flat,
        }
    }

    /// Bytes used by the flat CDF data.
    pub fn memory_bytes(&self) -> usize {
        self.cdf_flat.len() * std::mem::size_of::<f64>()
    }

    /// Look up `F_k(x)`, linearly interpolated in `log10(x)` between
    /// the two bracketing grid points. Returns `1.0` for the last
    /// category (CDF invariant) and saturates outside the tabulated
    /// range.
    #[inline]
    pub fn lookup(&self, x: f64, category: usize) -> f64 {
        if category + 1 >= self.n_categories {
            return 1.0;
        }
        if x <= 0.0 {
            return 0.0;
        }
        let log_x = x.log10();
        if log_x <= self.log_x_min {
            return self.cdf_flat[category];
        }
        if log_x >= self.log_x_max {
            let last = self.n_points - 1;
            return self.cdf_flat[last * self.n_categories + category];
        }
        let frac = (log_x - self.log_x_min) / (self.log_x_max - self.log_x_min);
        let f_idx = frac * (self.n_points - 1) as f64;
        let idx = f_idx.floor() as usize;
        let alpha = f_idx - idx as f64;
        let lo = self.cdf_flat[idx * self.n_categories + category];
        let hi = self.cdf_flat[(idx + 1) * self.n_categories + category];
        lo + alpha * (hi - lo)
    }

    /// Inverse-transform sample: return the smallest category index
    /// `k` such that `F_k(x) ≥ ξ`. `ξ` should be a uniform draw in
    /// `[0, 1]`.
    #[inline]
    pub fn sample(&self, x: f64, xi: f64) -> usize {
        for k in 0..self.n_categories - 1 {
            if xi <= self.lookup(x, k) {
                return k;
            }
        }
        self.n_categories - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_invariants() {
        // Three categories, smooth intensities.
        let axis: Vec<f64> = (0..50).map(|i| 1.0 * 1.1f64.powi(i)).collect();
        let mut intensities = vec![vec![0.0_f64; axis.len()]; 3];
        for (j, &x) in axis.iter().enumerate() {
            intensities[0][j] = 1.0 / x;
            intensities[1][j] = (x.ln() / 5.0).max(0.0);
            intensities[2][j] = 1.0;
        }
        let cdf = LogDecimatedCdf::from_intensities(&intensities, &axis, 200);
        // F_{K-1}(x) == 1 for all x.
        for x in [1.0, 5.0, 10.0, 50.0] {
            assert!((cdf.lookup(x, 2) - 1.0).abs() < 1e-12);
        }
        // F_k monotone non-decreasing in k.
        for x in [1.0, 5.0, 10.0, 50.0] {
            let f0 = cdf.lookup(x, 0);
            let f1 = cdf.lookup(x, 1);
            let f2 = cdf.lookup(x, 2);
            assert!(f0 <= f1 + 1e-12);
            assert!(f1 <= f2 + 1e-12);
        }
    }

    #[test]
    fn sampling_distribution_matches_intensities() {
        // At a fixed x with intensities [1, 2, 1], expect samples to
        // hit categories at frequencies 0.25 / 0.5 / 0.25 over many
        // pseudo-uniform draws.
        let axis: Vec<f64> = (0..30).map(|i| 1.0 * 1.2f64.powi(i)).collect();
        let intensities: Vec<Vec<f64>> = vec![
            vec![1.0; axis.len()],
            vec![2.0; axis.len()],
            vec![1.0; axis.len()],
        ];
        let cdf = LogDecimatedCdf::from_intensities(&intensities, &axis, 200);
        let n = 10_000;
        let mut counts = [0_u32; 3];
        for i in 0..n {
            let xi = (i as f64 + 0.5) / n as f64;
            let k = cdf.sample(5.0, xi);
            counts[k] += 1;
        }
        let frac0 = counts[0] as f64 / n as f64;
        let frac1 = counts[1] as f64 / n as f64;
        let frac2 = counts[2] as f64 / n as f64;
        assert!((frac0 - 0.25).abs() < 0.01, "got {frac0} for cat 0");
        assert!((frac1 - 0.50).abs() < 0.01, "got {frac1} for cat 1");
        assert!((frac2 - 0.25).abs() < 0.01, "got {frac2} for cat 2");
    }
}
