/// Running k-eff estimator over active batches: mean + 1σ of the
/// per-batch collision estimator. Inactive batches are accumulated
/// for the history but excluded from the statistic.
#[derive(Debug, Clone, Default)]
pub struct KeffTally {
    history: Vec<f64>,
    n_inactive: u32,
    sum: f64,
    sum_sq: f64,
    n_active: u32,
}

impl KeffTally {
    pub fn new(n_inactive: u32) -> Self {
        Self {
            history: Vec::new(),
            n_inactive,
            sum: 0.0,
            sum_sq: 0.0,
            n_active: 0,
        }
    }

    /// Record one batch's collision-estimator k. Returns the batch
    /// index of the value just recorded.
    pub fn record(&mut self, k_collision: f64) -> u32 {
        let idx = self.history.len() as u32;
        self.history.push(k_collision);
        if idx >= self.n_inactive {
            self.sum += k_collision;
            self.sum_sq += k_collision * k_collision;
            self.n_active += 1;
        }
        idx
    }

    pub fn mean(&self) -> f64 {
        if self.n_active == 0 {
            0.0
        } else {
            self.sum / self.n_active as f64
        }
    }

    /// Standard error of the mean, σ/√N over active batches.
    pub fn sigma(&self) -> f64 {
        if self.n_active < 2 {
            return 0.0;
        }
        let n = self.n_active as f64;
        let var = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
        var.max(0.0).sqrt() / n.sqrt()
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }

    pub fn n_active(&self) -> u32 {
        self.n_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_batches_excluded() {
        let mut t = KeffTally::new(2);
        t.record(0.5);
        t.record(0.5);
        t.record(1.0);
        t.record(1.0);
        assert_eq!(t.n_active(), 2);
        assert!((t.mean() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sigma_of_constant_sequence_is_zero() {
        let mut t = KeffTally::new(0);
        for _ in 0..10 {
            t.record(1.234);
        }
        assert!(t.sigma() < 1e-12);
    }
}
