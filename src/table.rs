//! Pointwise table ([`PointwiseTable`]) with log-log interpolation
//! and OpenMC-style stochastic two-endpoint temperature
//! pseudo-interpolation ([`StochTempTable`]).

use std::sync::Arc;

use crate::svd::LogHashIndex;

/// Pointwise table for `f(x)` on a single column (e.g. one
/// temperature, one configuration). Log-log interpolation between
/// bracketing rows; falls back to linear interpolation when either
/// bracket value is non-positive.
pub struct PointwiseTable {
    /// Row axis (sorted ascending; positive). Shared across tables on
    /// the same grid via `Arc::clone`.
    row_axis: Arc<[f64]>,
    /// Tabulated `f(x)` values, same length as `row_axis`.
    values: Vec<f64>,
    /// Optional log-uniform hash index for O(1) row lookup.
    hash: Option<LogHashIndex>,
}

impl PointwiseTable {
    /// Build from a shared row axis and owned values.
    /// **No** hash index is built; call [`PointwiseTable::build_hash`]
    /// to opt in for hot tables.
    pub fn from_shared(row_axis: Arc<[f64]>, values: Vec<f64>) -> Self {
        debug_assert_eq!(row_axis.len(), values.len());
        Self {
            row_axis,
            values,
            hash: None,
        }
    }

    /// Build from owned `(row_axis, values)`. Convenience constructor;
    /// internally allocates an `Arc<[f64]>` for the row axis.
    pub fn new(row_axis: Vec<f64>, values: Vec<f64>) -> Self {
        debug_assert_eq!(row_axis.len(), values.len());
        Self::from_shared(Arc::from(row_axis.into_boxed_slice()), values)
    }

    /// Build (or rebuild) a [`LogHashIndex`] for O(1) lookup.
    /// `n_bins = 8192` is a good default for axes with $\sim10^4$
    /// entries.
    pub fn build_hash(&mut self, n_bins: usize) {
        self.hash = Some(LogHashIndex::new(&self.row_axis, n_bins));
    }

    /// Drop the hash index (frees its memory). Subsequent lookups
    /// fall back to binary search.
    pub fn drop_hash(&mut self) {
        self.hash = None;
    }

    /// Whether the hash index is currently built.
    pub fn has_hash(&self) -> bool {
        self.hash.is_some()
    }

    /// Bytes used by the values array (does **not** include the
    /// shared row axis or the hash index).
    pub fn memory_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>()
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.row_axis.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.row_axis.is_empty()
    }

    /// Read-only access to the row axis.
    pub fn row_axis(&self) -> &[f64] {
        &self.row_axis
    }

    /// Read-only access to the tabulated values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Lower-bracket index: largest `idx` where `row_axis[idx] ≤ x`,
    /// clamped to `[0, n-1]`. Use this once when several tables share
    /// the same row axis and you want one search to feed many lookups.
    #[inline]
    pub fn bracket_idx(&self, x: f64) -> usize {
        let n = self.row_axis.len();
        if n == 0 {
            return 0;
        }
        if let Some(ref h) = self.hash {
            return h.lookup(x, &self.row_axis);
        }
        match self
            .row_axis
            .binary_search_by(|e| e.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) if i >= n => n - 1,
            Err(i) => i - 1,
        }
    }

    /// Lookup `f(x)` with binary search + log-log interpolation.
    /// Saturates at the table edges.
    #[inline]
    pub fn lookup(&self, x: f64) -> f64 {
        let n = self.row_axis.len();
        if n == 0 {
            return 0.0;
        }
        let idx = self.bracket_idx(x);
        self.lookup_at_idx(x, idx)
    }

    /// Lookup with a pre-computed lower-bracket `idx` (skips the
    /// search). Caller is responsible for `idx` matching `x` against
    /// the same row axis. Produces exactly the same value as
    /// [`PointwiseTable::lookup`] when `idx` is the lower bracket.
    #[inline]
    pub fn lookup_at_idx(&self, x: f64, idx: usize) -> f64 {
        let n = self.row_axis.len();
        if n == 0 {
            return 0.0;
        }
        if x <= self.row_axis[0] {
            return self.values[0];
        }
        if idx + 1 >= n {
            return self.values[n - 1];
        }
        let x_lo = self.row_axis[idx];
        let x_hi = self.row_axis[idx + 1];
        let v_lo = self.values[idx];
        let v_hi = self.values[idx + 1];
        if v_lo <= 0.0 || v_hi <= 0.0 {
            // Fallback to linear interpolation when either bracket
            // is non-positive (log-log undefined).
            let frac = (x - x_lo) / (x_hi - x_lo);
            return v_lo + frac * (v_hi - v_lo);
        }
        // Log-log interpolation. exp2/log2 is 3-5× faster than powf
        // on most modern CPUs.
        let f = (x / x_lo).ln() / (x_hi / x_lo).ln();
        let ratio = v_hi / v_lo;
        v_lo * f64::exp2(f * ratio.log2())
    }

    /// Batch lookup for benchmarking. Single-threaded.
    pub fn batch_lookup(&self, xs: &[f64], out: &mut [f64]) {
        for (x, o) in xs.iter().zip(out.iter_mut()) {
            *o = self.lookup(*x);
        }
    }
}

// ── Stochastic two-endpoint table ─────────────────────────────────────

thread_local! {
    /// Per-thread splitmix64 state for cheap `ξ` draws used by
    /// [`StochTempTable::lookup_at_idx`] / [`StochTempTable::draw_pick`].
    /// Kept separate from any "physics" RNG so you don't perturb a
    /// reproducible Monte Carlo stream just because you chose to look
    /// up cross sections via stochastic temperature interpolation.
    static STOCH_STATE: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0x9E37_79B9_7F4A_7C15) };
}

#[inline]
fn draw_xi() -> f64 {
    STOCH_STATE.with(|c| {
        let mut z = c.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        c.set(z);
        (z >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    })
}

/// A pointwise table that may be backed by two endpoints with a
/// stochastic per-lookup pick. The pick is exact at endpoints and
/// linearly interpolant in expectation between them, matching the
/// OpenMC stochastic temperature pseudo-interpolation convention.
///
/// Use [`StochTempTable::single`] when the target sits exactly on a
/// library endpoint (no picking, performance matches
/// [`PointwiseTable`]). Use [`StochTempTable::stochastic`] when the
/// target sits between two endpoints — every lookup draws `ξ ∈ [0,1)`
/// and returns the lower endpoint with probability
/// `(T_hi − T) / (T_hi − T_lo)`.
///
/// For consistency across multiple channels in a single physics
/// event, draw the pick once at the start of the event with
/// [`StochTempTable::draw_pick`] and feed it into each channel's
/// [`StochTempTable::lookup_at_idx_with_pick`]. All partial channels
/// for the event will then come from the same library endpoint —
/// required for thermodynamic consistency in OpenMC-style stochastic
/// interpolation.
pub struct StochTempTable {
    lo: PointwiseTable,
    hi: Option<PointwiseTable>,
    /// Probability of selecting the lower endpoint per lookup.
    p_lo: f64,
}

impl StochTempTable {
    /// Wrap a single endpoint (target sits exactly on a library
    /// column — no stochastic picking).
    pub fn single(table: PointwiseTable) -> Self {
        Self {
            lo: table,
            hi: None,
            p_lo: 1.0,
        }
    }

    /// Build a two-endpoint stochastic table. `target` should lie in
    /// `[col_lo, col_hi]`; the lower-endpoint probability is clamped
    /// to `[0, 1]`.
    pub fn stochastic(
        lo: PointwiseTable,
        hi: PointwiseTable,
        target: f64,
        col_lo: f64,
        col_hi: f64,
    ) -> Self {
        let p_lo = if (col_hi - col_lo).abs() < 1e-6 {
            1.0
        } else {
            ((col_hi - target) / (col_hi - col_lo)).clamp(0.0, 1.0)
        };
        Self {
            lo,
            hi: Some(hi),
            p_lo,
        }
    }

    /// True if both endpoints are populated (stochastic mode).
    pub fn is_stochastic(&self) -> bool {
        self.hi.is_some()
    }

    /// Probability of selecting the lower endpoint per lookup.
    pub fn p_lo(&self) -> f64 {
        self.p_lo
    }

    /// Bytes used by the values arrays of both endpoints.
    pub fn memory_bytes(&self) -> usize {
        self.lo.memory_bytes() + self.hi.as_ref().map_or(0, PointwiseTable::memory_bytes)
    }

    /// Lower-bracket grid index (the two endpoints share a row axis).
    #[inline]
    pub fn bracket_idx(&self, x: f64) -> usize {
        self.lo.bracket_idx(x)
    }

    /// Stochastic lookup with internal `ξ` draw and internal bracket
    /// search.
    #[inline]
    pub fn lookup(&self, x: f64) -> f64 {
        let idx = self.bracket_idx(x);
        self.lookup_at_idx(x, idx)
    }

    /// Stochastic lookup at a pre-computed `idx`. Draws its own `ξ`
    /// per call — use [`StochTempTable::lookup_at_idx_with_pick`]
    /// when several channels in the same physics event must share an
    /// endpoint pick (the typical case).
    #[inline]
    pub fn lookup_at_idx(&self, x: f64, idx: usize) -> f64 {
        match &self.hi {
            Some(hi) if draw_xi() > self.p_lo => hi.lookup_at_idx(x, idx),
            _ => self.lo.lookup_at_idx(x, idx),
        }
    }

    /// Stochastic lookup with an externally-chosen endpoint pick.
    /// Pass `use_hi = true` to pick the upper endpoint, `false` for
    /// the lower. Get a consistent pick from
    /// [`StochTempTable::draw_pick`] once per physics event and use
    /// the returned bool for every channel.
    #[inline]
    pub fn lookup_at_idx_with_pick(&self, x: f64, idx: usize, use_hi: bool) -> f64 {
        match &self.hi {
            Some(hi) if use_hi => hi.lookup_at_idx(x, idx),
            _ => self.lo.lookup_at_idx(x, idx),
        }
    }

    /// Draw one consistent pick for this physics event. Returns
    /// `true` to select the upper endpoint, `false` for the lower.
    /// For single-endpoint tables always returns `false`.
    #[inline]
    pub fn draw_pick(&self) -> bool {
        match &self.hi {
            Some(_) => draw_xi() > self.p_lo,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_exact_at_grid_points() {
        let xs = vec![1.0, 10.0, 100.0, 1000.0];
        let vs = vec![2.0, 4.0, 8.0, 16.0];
        let t = PointwiseTable::new(xs.clone(), vs.clone());
        for (x, v) in xs.iter().zip(vs.iter()) {
            let got = t.lookup(*x);
            assert!(
                (got - v).abs() < 1e-12,
                "exact-at-grid: x={x} v={v}, got {got}"
            );
        }
    }

    #[test]
    fn lookup_log_log_between_grid_points() {
        // f(x) = 2 x has log-log linear shape (slope 1).
        let xs = vec![1.0, 10.0, 100.0];
        let vs = vec![2.0, 20.0, 200.0];
        let t = PointwiseTable::new(xs, vs);
        for &x in &[2.0, 5.0, 30.0] {
            let want = 2.0 * x;
            let got = t.lookup(x);
            assert!((got - want).abs() < 1e-9, "x={x}: want {want}, got {got}");
        }
    }

    #[test]
    fn lookup_saturates_outside_range() {
        let t = PointwiseTable::new(vec![1.0, 10.0], vec![5.0, 7.0]);
        assert!((t.lookup(0.5) - 5.0).abs() < 1e-12);
        assert!((t.lookup(100.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn hash_lookup_matches_binary_search() {
        let xs: Vec<f64> = (0..200).map(|i| 1e-3 * 1.05f64.powi(i)).collect();
        let vs: Vec<f64> = xs.iter().map(|x| x * 1.7 + 0.5).collect();
        let mut t = PointwiseTable::new(xs.clone(), vs.clone());
        let probes: Vec<f64> = (0..50).map(|i| xs[0] * 1.07f64.powi(i + 3)).collect();
        let mut without_hash = vec![0.0; probes.len()];
        t.batch_lookup(&probes, &mut without_hash);
        t.build_hash(8192);
        let mut with_hash = vec![0.0; probes.len()];
        t.batch_lookup(&probes, &mut with_hash);
        for (a, b) in without_hash.iter().zip(with_hash.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "hash and binary search disagree: {a} vs {b}"
            );
        }
    }

    #[test]
    fn stoch_single_collapses_to_pointwise() {
        let t = PointwiseTable::new(vec![1.0, 10.0], vec![2.0, 20.0]);
        let s = StochTempTable::single(t);
        assert!(!s.is_stochastic());
        assert!((s.lookup(5.0) - 2.0 * 5.0).abs() < 1e-9);
    }

    #[test]
    fn stoch_pick_endpoints_at_target() {
        // Target = col_lo: p_lo = 1, every lookup picks lo.
        // Target = col_hi: p_lo = 0, every lookup picks hi.
        let lo = PointwiseTable::new(vec![1.0, 10.0], vec![2.0, 20.0]);
        let hi = PointwiseTable::new(vec![1.0, 10.0], vec![10.0, 100.0]);
        let s_at_lo = StochTempTable::stochastic(
            PointwiseTable::new(vec![1.0, 10.0], vec![2.0, 20.0]),
            PointwiseTable::new(vec![1.0, 10.0], vec![10.0, 100.0]),
            300.0,
            300.0,
            900.0,
        );
        for _ in 0..1000 {
            let v = s_at_lo.lookup(5.0);
            // p_lo = 1 → always lo
            assert!((v - 2.0 * 5.0).abs() < 1e-9);
        }
        let s_at_hi = StochTempTable::stochastic(lo, hi, 900.0, 300.0, 900.0);
        for _ in 0..1000 {
            let v = s_at_hi.lookup(5.0);
            assert!((v - 10.0 * 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn stoch_consistent_pick_across_channels() {
        // Two channels share a stochastic table — using draw_pick once
        // per "event" and feeding into both channels' lookups must
        // give consistent endpoint picks across the channels.
        let lo_a = PointwiseTable::new(vec![1.0, 10.0], vec![2.0, 20.0]);
        let hi_a = PointwiseTable::new(vec![1.0, 10.0], vec![10.0, 100.0]);
        let lo_b = PointwiseTable::new(vec![1.0, 10.0], vec![3.0, 30.0]);
        let hi_b = PointwiseTable::new(vec![1.0, 10.0], vec![15.0, 150.0]);
        let chan_a = StochTempTable::stochastic(lo_a, hi_a, 600.0, 300.0, 900.0);
        let chan_b = StochTempTable::stochastic(lo_b, hi_b, 600.0, 300.0, 900.0);
        // 50% probability of each endpoint per event.
        let mut hi_count: i32 = 0;
        for _ in 0..10_000 {
            let pick = chan_a.draw_pick();
            let va = chan_a.lookup_at_idx_with_pick(5.0, chan_a.bracket_idx(5.0), pick);
            let vb = chan_b.lookup_at_idx_with_pick(5.0, chan_b.bracket_idx(5.0), pick);
            // Channel a yields 10 (lo) or 50 (hi) at x=5.
            // Channel b yields 15 (lo) or 75 (hi) at x=5.
            // Both must be on the same side.
            let a_is_hi = (va - 50.0).abs() < (va - 10.0).abs();
            let b_is_hi = (vb - 75.0).abs() < (vb - 15.0).abs();
            assert_eq!(
                a_is_hi, b_is_hi,
                "channels disagreed on endpoint pick: a={va} b={vb}"
            );
            if a_is_hi {
                hi_count += 1;
            }
        }
        // 50/50 ± a few sigma.
        assert!(
            (hi_count - 5000).abs() < 200,
            "expected ~5000 hi picks, got {hi_count}"
        );
    }
}
