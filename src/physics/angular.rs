//! Tabulated `μ` distributions: stochastic-bin selection between
//! incident-energy slots, PDF-aware inverse-CDF sampling within
//! the chosen bin (histogram or linear-linear).

use crate::rng::Pcg64;

/// Tabulated angular distribution: an energy grid plus one `μ`
/// distribution per energy.
pub struct AngularDistribution {
    /// Energy grid (eV, sorted ascending) at which distributions are
    /// tabulated.
    pub energies: Vec<f64>,
    /// One `(μ, pdf, cdf, histogram)` distribution per energy.
    pub distributions: Vec<TabularMuDist>,
    /// `true` if the distribution is in the center-of-mass frame.
    /// `false` for lab frame.
    pub center_of_mass: bool,
}

/// Tabulated `μ` distribution at a single energy.
///
/// `mu`, `pdf`, `cdf` are parallel arrays. `cdf[0]` is `0.0`,
/// `cdf[n-1]` is `1.0`. `histogram = true` indicates that the PDF is
/// constant within each bin (ENDF interpolation 1 → linear CDF);
/// `false` indicates linear-linear interpolation between
/// `(mu, pdf)` breakpoints (ENDF interpolation 2 → quadratic CDF).
pub struct TabularMuDist {
    pub mu: Vec<f64>,
    pub pdf: Vec<f64>,
    pub cdf: Vec<f64>,
    pub histogram: bool,
}

impl AngularDistribution {
    /// Sample the scattering cosine `μ` at `energy`.
    ///
    /// Stochastic bin selection between the two energy slots
    /// bracketing `energy`, then one inverse-CDF draw within the
    /// chosen bin. Two random draws total. Returns isotropic
    /// `2ξ - 1` if the distribution is empty.
    pub fn sample_mu(&self, energy: f64, rng: &mut Pcg64) -> f64 {
        if self.energies.is_empty() {
            return 2.0 * rng.uniform() - 1.0;
        }
        let n = self.energies.len();
        if energy <= self.energies[0] {
            return self.distributions[0].sample(rng);
        }
        if energy >= self.energies[n - 1] {
            return self.distributions[n - 1].sample(rng);
        }
        let idx = match self
            .energies
            .binary_search_by(|e| e.partial_cmp(&energy).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) => return self.distributions[i].sample(rng),
            Err(i) => {
                if i > 0 {
                    i - 1
                } else {
                    0
                }
            }
        };
        if idx + 1 >= n {
            return self.distributions[idx].sample(rng);
        }
        let e_lo = self.energies[idx];
        let e_hi = self.energies[idx + 1];
        let r = (energy - e_lo) / (e_hi - e_lo);
        let pick_hi = rng.uniform() < r;
        let dist = if pick_hi {
            &self.distributions[idx + 1]
        } else {
            &self.distributions[idx]
        };
        dist.sample(rng).clamp(-1.0, 1.0)
    }
}

impl TabularMuDist {
    /// Sample `μ` using inverse CDF, drawing a fresh random number.
    pub fn sample(&self, rng: &mut Pcg64) -> f64 {
        self.sample_with_xi(rng.uniform())
    }

    /// Sample `μ` using inverse CDF with a pre-drawn `ξ ∈ [0, 1)`.
    /// Deterministic — useful for testing and for shared-pick
    /// patterns where multiple channels must consume the same `ξ`.
    pub fn sample_with_xi(&self, xi: f64) -> f64 {
        let n = self.cdf.len();
        if n < 2 {
            return 2.0 * xi - 1.0;
        }
        let idx = match self
            .cdf
            .binary_search_by(|c| c.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less))
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
        let idx = idx.min(n - 2);
        let cdf_lo = self.cdf[idx];
        let cdf_hi = self.cdf[idx + 1];
        let mu_lo = self.mu[idx];
        let mu_hi = self.mu[idx + 1];
        let dmu = mu_hi - mu_lo;

        if (cdf_hi - cdf_lo).abs() < 1e-15 || dmu.abs() < 1e-15 {
            return mu_lo.clamp(-1.0, 1.0);
        }

        if self.histogram {
            let frac = (xi - cdf_lo) / (cdf_hi - cdf_lo);
            return (mu_lo + frac * dmu).clamp(-1.0, 1.0);
        }

        // Linear-linear: solve quadratic CDF(μ) = ξ for x = μ - μ_lo.
        // PDF(μ) = pdf_lo + (pdf_hi - pdf_lo)/dmu · (μ - μ_lo)
        // CDF(μ) = cdf_lo + pdf_lo·x + 0.5·(pdf_hi - pdf_lo)/dmu · x²
        let pdf_lo = if idx < self.pdf.len() {
            self.pdf[idx]
        } else {
            0.0
        };
        let pdf_hi = if idx + 1 < self.pdf.len() {
            self.pdf[idx + 1]
        } else {
            pdf_lo
        };
        let a = (pdf_hi - pdf_lo) / (2.0 * dmu);
        let b = pdf_lo;
        let c = cdf_lo - xi;

        let x = if a.abs() < 1e-14 {
            if b.abs() < 1e-30 {
                (xi - cdf_lo) / (cdf_hi - cdf_lo) * dmu
            } else {
                -c / b
            }
        } else {
            let disc = (b * b - 4.0 * a * c).max(0.0);
            let sqrt_disc = disc.sqrt();
            (-b + sqrt_disc) / (2.0 * a)
        };

        let x = x.clamp(0.0, dmu);
        (mu_lo + x).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_iso_bin() -> TabularMuDist {
        // Histogram CDF, isotropic in [-1, 1]: PDF = 0.5 everywhere.
        TabularMuDist {
            mu: vec![-1.0, 1.0],
            pdf: vec![0.5, 0.5],
            cdf: vec![0.0, 1.0],
            histogram: true,
        }
    }

    #[test]
    fn isotropic_histogram_uniform_in_minus1_to_1() {
        let d = build_iso_bin();
        let mut sum = 0.0_f64;
        let n = 50_000;
        let mut rng = Pcg64::new(42, 1);
        for _ in 0..n {
            let mu = d.sample(&mut rng);
            assert!((-1.0..=1.0).contains(&mu));
            sum += mu;
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.02, "mean μ should be ~0, got {mean}");
    }

    #[test]
    fn forward_peaked_linear_linear() {
        // Linear-linear with PDF rising from 0 at μ=-1 to 1 at μ=+1.
        // CDF = 0.25 (μ + 1)² ; analytic mean = 1/3.
        let mu = vec![-1.0, 1.0];
        let pdf = vec![0.0, 1.0];
        let cdf = vec![0.0, 1.0];
        let d = TabularMuDist {
            mu,
            pdf,
            cdf,
            histogram: false,
        };
        let mut rng = Pcg64::new(7, 1);
        let mut sum = 0.0_f64;
        let n = 100_000;
        for _ in 0..n {
            sum += d.sample(&mut rng);
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 1.0 / 3.0).abs() < 0.01,
            "expected mean ≈ 1/3, got {mean}"
        );
    }
}
