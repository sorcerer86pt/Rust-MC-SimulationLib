//! URR probability-table sampling. OpenMC convention:
//! correlated-`ξ` band lookup at both bracketing energies,
//! per-channel XS interpolated between them.

/// Probability-table block for one nuclide / one temperature.
pub struct UrrProbabilityTables {
    /// Energy grid (eV, sorted ascending).
    pub energies: Vec<f64>,
    /// Number of probability bands (typically 20).
    pub n_bands: usize,
    /// Cumulative band probabilities `[n_energy][n_bands]`. Each row
    /// is the per-band CDF: monotone non-decreasing, last entry
    /// `≈ 1.0`.
    pub cum_prob: Vec<Vec<f64>>,
    pub total_factor: Vec<Vec<f64>>,
    pub elastic_factor: Vec<Vec<f64>>,
    pub fission_factor: Vec<Vec<f64>>,
    pub capture_factor: Vec<Vec<f64>>,
    /// `true` → factor multiplies the smooth XS;
    /// `false` → factor *is* the absolute XS.
    pub multiply_smooth: bool,
    /// ENDF interpolation between adjacent URR energies. 2 = lin-lin
    /// (default), 5 = log-log.
    pub interpolation: u8,
}

/// Sampled URR factors for one collision.
#[derive(Debug, Clone, Copy)]
pub struct UrrFactors {
    pub total: f64,
    pub elastic: f64,
    pub fission: f64,
    pub capture: f64,
}

impl UrrProbabilityTables {
    /// `true` if `energy` falls inside the URR grid.
    #[inline]
    pub fn in_range(&self, energy: f64) -> bool {
        if self.energies.is_empty() {
            return false;
        }
        energy >= self.energies[0] && energy <= *self.energies.last().unwrap_or(&0.0)
    }

    /// Sample URR factors at `energy` using the supplied `ξ ∈ [0, 1)`.
    ///
    /// `ξ` is consumed *once* and the same value is used to pick a
    /// band at both bracketing energies (correlated draw). Returns
    /// XS factors if `multiply_smooth = true`, absolute XS values
    /// otherwise.
    pub fn sample(&self, energy: f64, xi: f64) -> UrrFactors {
        let n_e = self.energies.len();
        if n_e == 0 {
            return UrrFactors {
                total: 1.0,
                elastic: 1.0,
                fission: 1.0,
                capture: 1.0,
            };
        }

        let i_lo = match self
            .energies
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => i,
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };

        let pick = |idx: usize| -> (f64, f64, f64, f64) {
            let cum = &self.cum_prob[idx];
            let mut band = cum.len() - 1;
            for (j, &cp) in cum.iter().enumerate() {
                if xi < cp {
                    band = j;
                    break;
                }
            }
            (
                self.total_factor[idx][band],
                self.elastic_factor[idx][band],
                self.fission_factor[idx][band],
                self.capture_factor[idx][band],
            )
        };

        if n_e == 1 || i_lo + 1 >= n_e || energy <= self.energies[0] {
            let (total, elastic, fission, capture) = pick(i_lo.min(n_e - 1));
            return UrrFactors {
                total,
                elastic,
                fission,
                capture,
            };
        }

        let e_lo = self.energies[i_lo];
        let e_hi = self.energies[i_lo + 1];
        let f = match self.interpolation {
            5 => (energy / e_lo).ln() / (e_hi / e_lo).ln(),
            _ => (energy - e_lo) / (e_hi - e_lo),
        };
        let (t_lo, el_lo, fi_lo, c_lo) = pick(i_lo);
        let (t_hi, el_hi, fi_hi, c_hi) = pick(i_lo + 1);
        UrrFactors {
            total: (1.0 - f) * t_lo + f * t_hi,
            elastic: (1.0 - f) * el_lo + f * el_hi,
            fission: (1.0 - f) * fi_lo + f * fi_hi,
            capture: (1.0 - f) * c_lo + f * c_hi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_table() -> UrrProbabilityTables {
        // 2 energies × 4 bands: every band has identical factors so
        // sampling is deterministic regardless of ξ.
        UrrProbabilityTables {
            energies: vec![1.0e3, 1.0e4],
            n_bands: 4,
            cum_prob: vec![vec![0.25, 0.50, 0.75, 1.0], vec![0.25, 0.50, 0.75, 1.0]],
            total_factor: vec![vec![1.1, 1.2, 1.3, 1.4], vec![2.1, 2.2, 2.3, 2.4]],
            elastic_factor: vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0]],
            fission_factor: vec![vec![0.5, 0.5, 0.5, 0.5], vec![1.5, 1.5, 1.5, 1.5]],
            capture_factor: vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.6, 0.7, 0.8, 0.9]],
            multiply_smooth: true,
            interpolation: 2,
        }
    }

    #[test]
    fn in_range_works() {
        let t = small_table();
        assert!(!t.in_range(100.0));
        assert!(t.in_range(5.0e3));
        assert!(!t.in_range(1.0e5));
    }

    #[test]
    fn sample_at_grid_edges_uses_single_energy() {
        let t = small_table();
        // ξ = 0.4 picks band 1 (cum_prob > 0.4 first at idx 1) at the
        // lower energy. total_factor[0][1] = 1.2.
        let f_lo = t.sample(1.0e3, 0.4);
        assert!((f_lo.total - 1.2).abs() < 1e-12);
        // At upper edge, band 1 → total_factor[1][1] = 2.2.
        let f_hi = t.sample(1.0e4, 0.4);
        assert!((f_hi.total - 2.2).abs() < 1e-12);
    }

    #[test]
    fn sample_interpolates_between_energies() {
        let t = small_table();
        // Mid-energy 5500 eV, lin-lin interp f = 0.5.
        // total at lo = 1.2, total at hi = 2.2 → midpoint 1.7.
        let f = t.sample(5.5e3, 0.4);
        assert!((f.total - 1.7).abs() < 1e-12, "got {}", f.total);
    }
}
