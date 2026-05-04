//! CP / PARAFAC of a 3-tensor via greedy rank-1 power-iteration
//! deflation.

#![allow(clippy::needless_range_loop)]

/// Stored CP decomposition of a 3-tensor. Factor matrices are stored
/// flat, column-major in the rank index: `a[r*n_a + i]`, `b[r*n_b + t]`,
/// `c[r*n_c + k]`. Per-component magnitude `sigma[r]` is the
/// contribution of the `r`-th outer product.
pub struct CpDecomposition {
    pub rank: usize,
    pub n_a: usize,
    pub n_b: usize,
    pub n_c: usize,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub sigma: Vec<f64>,
}

impl CpDecomposition {
    /// Reconstruct the full tensor at truncation rank `k ≤ self.rank`.
    /// Returns flat `out[i * n_b * n_c + t * n_c + l]` row-major.
    pub fn reconstruct(&self, k: usize) -> Vec<f64> {
        let k = k.min(self.rank);
        let mut out = vec![0.0_f64; self.n_a * self.n_b * self.n_c];
        for r in 0..k {
            let s = self.sigma[r];
            let a_off = r * self.n_a;
            let b_off = r * self.n_b;
            let c_off = r * self.n_c;
            for i in 0..self.n_a {
                let ai_s = self.a[a_off + i] * s;
                for t in 0..self.n_b {
                    let bt = self.b[b_off + t];
                    let scale = ai_s * bt;
                    let row = i * self.n_b * self.n_c + t * self.n_c;
                    for l in 0..self.n_c {
                        out[row + l] += scale * self.c[c_off + l];
                    }
                }
            }
        }
        out
    }

    /// Bytes used by the factor matrices and component magnitudes.
    pub fn memory_bytes(&self) -> usize {
        (self.a.len() + self.b.len() + self.c.len() + self.sigma.len()) * std::mem::size_of::<f64>()
    }
}

/// Decompose a flat 3-tensor (row-major
/// `tensor[i * n_b * n_c + t * n_c + l]`) into a rank-`max_rank` CP
/// approximation via greedy rank-1 deflation.
///
/// `max_iter` caps the alternating-power-iteration count per
/// component; `tol` is the relative convergence threshold on the
/// component magnitude between successive iterations. Component
/// fitting can stop early if the residual collapses to numerical
/// zero.
pub fn cp_greedy_rank1(
    tensor: &[f64],
    n_a: usize,
    n_b: usize,
    n_c: usize,
    max_rank: usize,
    max_iter: usize,
    tol: f64,
) -> CpDecomposition {
    assert_eq!(tensor.len(), n_a * n_b * n_c);

    let mut residual: Vec<f64> = tensor.to_vec();
    let mut a = Vec::with_capacity(max_rank * n_a);
    let mut b = Vec::with_capacity(max_rank * n_b);
    let mut c = Vec::with_capacity(max_rank * n_c);
    let mut sigma = Vec::with_capacity(max_rank);

    let mut rng_state = 0x853c49e6748fea9b_u64;
    let next_uniform = |state: &mut u64| {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (*state >> 33) as f64;
        x / (1_u64 << 31) as f64 - 1.0
    };

    for _r in 0..max_rank {
        let mut bv: Vec<f64> = (0..n_b).map(|_| next_uniform(&mut rng_state)).collect();
        let mut cv: Vec<f64> = (0..n_c).map(|_| next_uniform(&mut rng_state)).collect();
        normalize(&mut bv);
        normalize(&mut cv);
        let mut av: Vec<f64> = vec![0.0; n_a];

        let mut prev_norm = 0.0_f64;
        for _it in 0..max_iter {
            for i in 0..n_a {
                let mut s = 0.0_f64;
                for t in 0..n_b {
                    let bt = bv[t];
                    let row = i * n_b * n_c + t * n_c;
                    for l in 0..n_c {
                        s += residual[row + l] * bt * cv[l];
                    }
                }
                av[i] = s;
            }
            normalize(&mut av);

            for t in 0..n_b {
                let mut s = 0.0_f64;
                for i in 0..n_a {
                    let ai = av[i];
                    let row = i * n_b * n_c + t * n_c;
                    for l in 0..n_c {
                        s += residual[row + l] * ai * cv[l];
                    }
                }
                bv[t] = s;
            }
            normalize(&mut bv);

            for l in 0..n_c {
                let mut s = 0.0_f64;
                for i in 0..n_a {
                    let ai = av[i];
                    for t in 0..n_b {
                        let row = i * n_b * n_c + t * n_c;
                        s += residual[row + l] * ai * bv[t];
                    }
                }
                cv[l] = s;
            }
            let cv_norm = cv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if cv_norm < 1e-30 {
                break;
            }
            for cv_l in cv.iter_mut() {
                *cv_l /= cv_norm;
            }

            let converged = (cv_norm - prev_norm).abs() < tol * cv_norm.max(1e-30);
            prev_norm = cv_norm;
            if converged {
                break;
            }
        }

        let mut s = 0.0_f64;
        for i in 0..n_a {
            let ai = av[i];
            for t in 0..n_b {
                let bt = bv[t];
                let row = i * n_b * n_c + t * n_c;
                for l in 0..n_c {
                    s += residual[row + l] * ai * bt * cv[l];
                }
            }
        }
        if s.abs() < 1e-20 {
            break;
        }

        sigma.push(s);
        a.extend_from_slice(&av);
        b.extend_from_slice(&bv);
        c.extend_from_slice(&cv);

        for i in 0..n_a {
            let ai_s = av[i] * s;
            for t in 0..n_b {
                let bt = bv[t];
                let scale = ai_s * bt;
                let row = i * n_b * n_c + t * n_c;
                for l in 0..n_c {
                    residual[row + l] -= scale * cv[l];
                }
            }
        }
    }

    let rank = sigma.len();
    CpDecomposition {
        rank,
        n_a,
        n_b,
        n_c,
        a,
        b,
        c,
        sigma,
    }
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-30 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Relative Frobenius error of `reconstruction` vs `original`.
/// Returns `0.0` if `original` is identically zero.
pub fn relative_l2_error(original: &[f64], reconstruction: &[f64]) -> f64 {
    assert_eq!(original.len(), reconstruction.len());
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..original.len() {
        let d = original[i] - reconstruction[i];
        num += d * d;
        den += original[i] * original[i];
    }
    if den < 1e-30 {
        return 0.0;
    }
    (num / den).sqrt()
}

/// Maximum absolute element-wise error.
pub fn max_abs_error(original: &[f64], reconstruction: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..original.len() {
        let d = (original[i] - reconstruction[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank1_tensor_recovers_at_rank1() {
        let n_a = 4;
        let n_b = 3;
        let n_c = 5;
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, 1.5, 2.5];
        let c = [1.0, -1.0, 2.0, -0.5, 3.0];
        let mut tensor = vec![0.0; n_a * n_b * n_c];
        for i in 0..n_a {
            for t in 0..n_b {
                for l in 0..n_c {
                    tensor[i * n_b * n_c + t * n_c + l] = a[i] * b[t] * c[l];
                }
            }
        }
        let cp = cp_greedy_rank1(&tensor, n_a, n_b, n_c, 1, 200, 1e-12);
        let recon = cp.reconstruct(1);
        let err = relative_l2_error(&tensor, &recon);
        assert!(
            err < 1e-8,
            "rank-1 recovery should be near-exact, got rel L2 = {err}"
        );
    }

    #[test]
    fn rank2_sum_recovers_well_at_rank4() {
        let n_a = 6;
        let n_b = 4;
        let n_c = 5;
        let mut tensor = vec![0.0; n_a * n_b * n_c];
        for i in 0..n_a {
            for t in 0..n_b {
                for l in 0..n_c {
                    let v1 = (i as f64 + 1.0) * (t as f64 + 1.0) * (l as f64 + 1.0);
                    let v2 = ((i + l) as f64).cos() * ((t + 1) as f64);
                    tensor[i * n_b * n_c + t * n_c + l] = v1 + v2;
                }
            }
        }
        let cp = cp_greedy_rank1(&tensor, n_a, n_b, n_c, 4, 500, 1e-10);
        let recon = cp.reconstruct(4);
        let err = relative_l2_error(&tensor, &recon);
        assert!(err < 0.05, "rank-4 should reach within 5%, got {err}");
    }
}
