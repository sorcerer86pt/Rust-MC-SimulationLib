use crate::geometry::Vec3;

/// One scoring bin: optional cell filter + optional [E_lo, E_hi).
#[derive(Debug, Clone)]
pub struct FluxBin {
    pub cell: Option<usize>,
    pub e_lo: f64,
    pub e_hi: f64,
}

impl FluxBin {
    pub fn all_energies(cell: Option<usize>) -> Self {
        Self {
            cell,
            e_lo: 0.0,
            e_hi: f64::INFINITY,
        }
    }
    pub fn matches(&self, cell_idx: usize, energy: f64) -> bool {
        if let Some(c) = self.cell {
            if c != cell_idx {
                return false;
            }
        }
        energy >= self.e_lo && energy < self.e_hi
    }
}

/// Track-length flux estimator: φ_bin += w · ℓ for each (cell,
/// energy)-matched track segment. Per-batch and accumulating
/// statistics over active batches.
#[derive(Debug, Clone)]
pub struct FluxTally {
    bins: Vec<FluxBin>,
    /// Accumulator for the *current* batch only; reset by `end_batch`.
    batch_sum: Vec<f64>,
    /// Sum of per-batch values across active batches.
    sum: Vec<f64>,
    /// Sum of squares of per-batch values across active batches.
    sum_sq: Vec<f64>,
    n_active: u32,
    n_inactive: u32,
    n_recorded: u32,
}

impl FluxTally {
    pub fn new(bins: Vec<FluxBin>, n_inactive: u32) -> Self {
        let n = bins.len();
        Self {
            bins,
            batch_sum: vec![0.0; n],
            sum: vec![0.0; n],
            sum_sq: vec![0.0; n],
            n_active: 0,
            n_inactive,
            n_recorded: 0,
        }
    }

    /// Score a single track segment of length `length` with
    /// statistical weight `weight` at energy `energy` in cell
    /// `cell_idx`. Increments every matching bin.
    #[inline]
    pub fn score_track(&mut self, cell_idx: usize, energy: f64, length: f64, weight: f64) {
        let contrib = weight * length;
        for (i, bin) in self.bins.iter().enumerate() {
            if bin.matches(cell_idx, energy) {
                self.batch_sum[i] += contrib;
            }
        }
    }

    /// Close out the current batch. After `n_inactive` warm-up
    /// batches, the per-batch sum contributes to the mean+σ.
    pub fn end_batch(&mut self) {
        if self.n_recorded >= self.n_inactive {
            for i in 0..self.bins.len() {
                let v = self.batch_sum[i];
                self.sum[i] += v;
                self.sum_sq[i] += v * v;
            }
            self.n_active += 1;
        }
        self.n_recorded += 1;
        for v in &mut self.batch_sum {
            *v = 0.0;
        }
    }

    pub fn bins(&self) -> &[FluxBin] {
        &self.bins
    }

    pub fn n_active(&self) -> u32 {
        self.n_active
    }

    /// Mean per-batch flux for bin `i` (track-length sum per batch).
    /// Divide by source weight × volume to get a normalised flux.
    pub fn mean(&self, i: usize) -> f64 {
        if self.n_active == 0 {
            0.0
        } else {
            self.sum[i] / self.n_active as f64
        }
    }

    pub fn sigma(&self, i: usize) -> f64 {
        if self.n_active < 2 {
            return 0.0;
        }
        let n = self.n_active as f64;
        let var = (self.sum_sq[i] - self.sum[i] * self.sum[i] / n) / (n - 1.0);
        var.max(0.0).sqrt() / n.sqrt()
    }
}

/// Convenience: track length from `from` to `to` is `(to-from).norm()`.
#[inline]
pub fn segment_length(from: Vec3, to: Vec3) -> f64 {
    let d = to - from;
    (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_filter_isolates_scoring() {
        let bins = vec![
            FluxBin::all_energies(Some(0)),
            FluxBin::all_energies(Some(1)),
        ];
        let mut t = FluxTally::new(bins, 0);
        t.score_track(0, 1.0, 2.0, 1.0);
        t.score_track(1, 1.0, 5.0, 1.0);
        t.end_batch();
        assert!((t.mean(0) - 2.0).abs() < 1e-12);
        assert!((t.mean(1) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn energy_window_excludes_outside_band() {
        let bins = vec![FluxBin {
            cell: None,
            e_lo: 1.0,
            e_hi: 10.0,
        }];
        let mut t = FluxTally::new(bins, 0);
        t.score_track(0, 0.5, 1.0, 1.0); // below
        t.score_track(0, 5.0, 1.0, 1.0); // in
        t.score_track(0, 100.0, 1.0, 1.0); // above
        t.end_batch();
        assert!((t.mean(0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inactive_batches_dropped_from_stats() {
        let bins = vec![FluxBin::all_energies(None)];
        let mut t = FluxTally::new(bins, 1);
        t.score_track(0, 1.0, 7.0, 1.0);
        t.end_batch(); // inactive — not counted
        t.score_track(0, 1.0, 3.0, 1.0);
        t.end_batch();
        assert_eq!(t.n_active(), 1);
        assert!((t.mean(0) - 3.0).abs() < 1e-12);
    }
}
