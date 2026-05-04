//! Six-group point-kinetics:
//!
//! ```text
//! dn/dt    = (ρ − β) / Λ · n + Σᵢ λᵢ Cᵢ + S
//! dCᵢ/dt   = βᵢ / Λ · n − λᵢ Cᵢ          (i = 1..6)
//! ```
//!
//! Solved on a piecewise-constant ρ and S interval as a homogeneous
//! 8×8 linear system y' = M·y via the augmented trick
//! `M = [[A, b], [0, 0]]`. Time advancement is the matrix exponential
//! `exp(M·dt)` computed by Padé(13)-with-scaling-and-squaring. Stable
//! for the full physical range of Λ and λᵢ; the prompt jump on a
//! reactivity step is handled correctly without any λᵢ-dependent
//! stiffness manoeuvre.

use faer::Mat;
use faer::prelude::Solve;

/// Six-group kinetic parameters. Defaults are the standard Keepin
/// thermal U-235 set used in textbook benchmarks.
#[derive(Debug, Clone, Copy)]
pub struct KineticsParams {
    /// Delayed-neutron group fractions βᵢ (dimensionless).
    pub beta_i: [f64; 6],
    /// Decay constants λᵢ (1/s).
    pub lambda_i: [f64; 6],
    /// Mean prompt-neutron generation time Λ (s).
    pub gen_time: f64,
}

impl KineticsParams {
    /// Total delayed-neutron fraction β = Σᵢ βᵢ.
    pub fn beta_total(&self) -> f64 {
        self.beta_i.iter().sum()
    }

    /// Keepin thermal U-235 set (β_total = 0.0065).
    pub fn keepin_u235_thermal() -> Self {
        Self {
            beta_i: [
                0.000_215, 0.001_424, 0.001_274, 0.002_568, 0.000_748, 0.000_273,
            ],
            lambda_i: [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],
            gen_time: 1.0e-4,
        }
    }
}

/// Reactor state: prompt-neutron density n and the six precursor
/// concentrations Cᵢ. Units are arbitrary (whatever the caller chose
/// for `n` propagates linearly to Cᵢ).
#[derive(Debug, Clone, Copy)]
pub struct KineticsState {
    pub n: f64,
    pub c: [f64; 6],
    pub time: f64,
}

/// Equilibrium state at given `n`: dCᵢ/dt = 0 ⇒ Cᵢ = βᵢ n / (λᵢ Λ).
pub fn equilibrium_state(n: f64, p: &KineticsParams) -> KineticsState {
    let mut c = [0.0; 6];
    for i in 0..6 {
        c[i] = p.beta_i[i] * n / (p.lambda_i[i] * p.gen_time);
    }
    KineticsState { n, c, time: 0.0 }
}

/// Stateful point-kinetics integrator. Hold the parameters; advance
/// the state under a piecewise-constant (ρ, S) program.
pub struct PointKinetics {
    pub params: KineticsParams,
    pub state: KineticsState,
}

impl PointKinetics {
    pub fn new(params: KineticsParams, state: KineticsState) -> Self {
        Self { params, state }
    }

    /// Advance by `dt` under constant reactivity `rho` (in absolute
    /// units, *not* dollars) and external source rate `s`. Returns
    /// the new state.
    pub fn step(&mut self, rho: f64, s: f64, dt: f64) -> KineticsState {
        let p = &self.params;
        let beta = p.beta_total();
        // Build augmented 8×8 M:
        //   M[0][0]   = (ρ − β)/Λ
        //   M[0][i+1] = λᵢ
        //   M[i+1][0] = βᵢ/Λ
        //   M[i+1][i+1] = −λᵢ
        //   M[0][7]   = S        (last column = inhomogeneous term)
        let mut m = [[0.0_f64; 8]; 8];
        m[0][0] = (rho - beta) / p.gen_time;
        for i in 0..6 {
            m[0][i + 1] = p.lambda_i[i];
            m[i + 1][0] = p.beta_i[i] / p.gen_time;
            m[i + 1][i + 1] = -p.lambda_i[i];
        }
        m[0][7] = s;

        let exp_m_dt = expm_pade_8(&m, dt);

        // y_old = [n, c1..c6, 1]
        let mut y = [0.0_f64; 8];
        y[0] = self.state.n;
        for i in 0..6 {
            y[i + 1] = self.state.c[i];
        }
        y[7] = 1.0;

        let mut y_new = [0.0_f64; 8];
        for r in 0..8 {
            let mut acc = 0.0;
            for c in 0..8 {
                acc += exp_m_dt[r][c] * y[c];
            }
            y_new[r] = acc;
        }

        let mut c_new = [0.0_f64; 6];
        for i in 0..6 {
            c_new[i] = y_new[i + 1];
        }
        let new_state = KineticsState {
            n: y_new[0],
            c: c_new,
            time: self.state.time + dt,
        };
        self.state = new_state;
        new_state
    }
}

/// Specialisation of [`expm_pade`] to 8×8 stack-allocated matrices.
fn expm_pade_8(m: &[[f64; 8]; 8], t: f64) -> [[f64; 8]; 8] {
    let mut a = Mat::<f64>::zeros(8, 8);
    for r in 0..8 {
        for c in 0..8 {
            a[(r, c)] = m[r][c] * t;
        }
    }
    let e = expm_pade(&a);
    let mut out = [[0.0_f64; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            out[r][c] = e[(r, c)];
        }
    }
    out
}

/// Matrix exponential `exp(A)` for a real square matrix via
/// Padé(13) with scaling-and-squaring. Coefficients are
/// from Higham, *SIAM J. Matrix Anal. Appl.* 26 (2005) 1179.
pub fn expm_pade(a: &Mat<f64>) -> Mat<f64> {
    assert_eq!(a.nrows(), a.ncols(), "expm requires a square matrix");
    let n = a.nrows();

    // Compute the 1-norm (max column sum).
    let norm = matrix_one_norm(a);

    // Padé(13) order parameter: scale so ‖A/2^s‖₁ ≤ 5.371920351148152.
    let theta_13 = 5.371_920_351_148_152_f64;
    let s = if norm <= theta_13 {
        0
    } else {
        ((norm / theta_13).log2().ceil() as i32).max(0) as u32
    };
    let scale = 2.0_f64.powi(s as i32);

    // A ← A / 2^s.
    let mut a_scaled = a.cloned();
    if s > 0 {
        for r in 0..n {
            for c in 0..n {
                a_scaled[(r, c)] /= scale;
            }
        }
    }

    // Padé(13) coefficients.
    let b: [f64; 14] = [
        64_764_752_532_480_000.0,
        32_382_376_266_240_000.0,
        7_771_770_303_897_600.0,
        1_187_353_796_428_800.0,
        129_060_195_264_000.0,
        10_559_470_521_600.0,
        670_442_572_800.0,
        33_522_128_640.0,
        1_323_241_920.0,
        40_840_800.0,
        960_960.0,
        16_380.0,
        182.0,
        1.0,
    ];

    let i_n = identity(n);
    let a2 = mat_mul(&a_scaled, &a_scaled);
    let a4 = mat_mul(&a2, &a2);
    let a6 = mat_mul(&a4, &a2);

    // U = A · (A6·(b13·A6 + b11·A4 + b9·A2) + b7·A6 + b5·A4 + b3·A2 + b1·I)
    let mut inner_u = scaled_add(&a6, b[13], &a4, b[11]);
    inner_u = scaled_add(&inner_u, 1.0, &a2, b[9]);
    let inner_u = mat_mul(&a6, &inner_u);
    let mut u_inner = scaled_add(&inner_u, 1.0, &a6, b[7]);
    u_inner = scaled_add(&u_inner, 1.0, &a4, b[5]);
    u_inner = scaled_add(&u_inner, 1.0, &a2, b[3]);
    u_inner = scaled_add(&u_inner, 1.0, &i_n, b[1]);
    let u = mat_mul(&a_scaled, &u_inner);

    // V = A6·(b12·A6 + b10·A4 + b8·A2) + b6·A6 + b4·A4 + b2·A2 + b0·I
    let mut inner_v = scaled_add(&a6, b[12], &a4, b[10]);
    inner_v = scaled_add(&inner_v, 1.0, &a2, b[8]);
    let inner_v = mat_mul(&a6, &inner_v);
    let mut v = scaled_add(&inner_v, 1.0, &a6, b[6]);
    v = scaled_add(&v, 1.0, &a4, b[4]);
    v = scaled_add(&v, 1.0, &a2, b[2]);
    v = scaled_add(&v, 1.0, &i_n, b[0]);

    // R = (V − U)⁻¹ · (V + U)
    let p = scaled_add(&v, 1.0, &u, 1.0);
    let q = scaled_add(&v, 1.0, &u, -1.0);
    let lu = q.partial_piv_lu();
    let mut r = lu.solve(&p);

    // Square s times.
    for _ in 0..s {
        r = mat_mul(&r, &r);
    }
    r
}

fn matrix_one_norm(a: &Mat<f64>) -> f64 {
    let n = a.ncols();
    let m = a.nrows();
    let mut max_col = 0.0_f64;
    for c in 0..n {
        let mut s = 0.0_f64;
        for r in 0..m {
            s += a[(r, c)].abs();
        }
        if s > max_col {
            max_col = s;
        }
    }
    max_col
}

fn identity(n: usize) -> Mat<f64> {
    let mut i = Mat::<f64>::zeros(n, n);
    for k in 0..n {
        i[(k, k)] = 1.0;
    }
    i
}

fn mat_mul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    a * b
}

fn scaled_add(a: &Mat<f64>, alpha: f64, b: &Mat<f64>, beta: f64) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::<f64>::zeros(m, n);
    for r in 0..m {
        for c in 0..n {
            out[(r, c)] = alpha * a[(r, c)] + beta * b[(r, c)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm_diagonal_decay() {
        // exp(diag(-1, -2)) = diag(e^-1, e^-2)
        let mut a = Mat::<f64>::zeros(2, 2);
        a[(0, 0)] = -1.0;
        a[(1, 1)] = -2.0;
        let e = expm_pade(&a);
        assert!((e[(0, 0)] - (-1.0_f64).exp()).abs() < 1e-12);
        assert!((e[(1, 1)] - (-2.0_f64).exp()).abs() < 1e-12);
        assert!(e[(0, 1)].abs() < 1e-14);
        assert!(e[(1, 0)].abs() < 1e-14);
    }

    #[test]
    fn expm_rotation_pi_over_2() {
        // exp(t · [[0,1],[-1,0]]) is rotation by t. At t = π/2 the
        // result should be [[0,1],[-1,0]].
        let mut a = Mat::<f64>::zeros(2, 2);
        a[(0, 0)] = 0.0;
        a[(0, 1)] = std::f64::consts::FRAC_PI_2;
        a[(1, 0)] = -std::f64::consts::FRAC_PI_2;
        a[(1, 1)] = 0.0;
        let e = expm_pade(&a);
        assert!(e[(0, 0)].abs() < 1e-12);
        assert!((e[(0, 1)] - 1.0).abs() < 1e-12);
        assert!((e[(1, 0)] + 1.0).abs() < 1e-12);
        assert!(e[(1, 1)].abs() < 1e-12);
    }

    #[test]
    fn expm_zero_is_identity() {
        let a = Mat::<f64>::zeros(5, 5);
        let e = expm_pade(&a);
        for r in 0..5 {
            for c in 0..5 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((e[(r, c)] - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn equilibrium_state_drift_is_zero() {
        let p = KineticsParams::keepin_u235_thermal();
        let s0 = equilibrium_state(1.0, &p);
        let mut k = PointKinetics::new(p, s0);
        // Zero reactivity, no source — equilibrium should hold.
        let s1 = k.step(0.0, 0.0, 1.0);
        assert!((s1.n - 1.0).abs() < 1e-9, "n drifted to {}", s1.n);
        for i in 0..6 {
            let want = p.beta_i[i] / (p.lambda_i[i] * p.gen_time);
            assert!(
                (s1.c[i] - want).abs() / want < 1e-7,
                "C{} drifted: got {}, want {}",
                i,
                s1.c[i],
                want
            );
        }
    }

    #[test]
    fn delayed_critical_step_inhour_response() {
        // Step ρ = β/2 (positive, sub-prompt) for 1 s and check that
        // n grows but stays bounded — period dominated by delayed
        // groups, not prompt.
        let p = KineticsParams::keepin_u235_thermal();
        let beta = p.beta_total();
        let s0 = equilibrium_state(1.0, &p);
        let mut k = PointKinetics::new(p, s0);
        let s1 = k.step(0.5 * beta, 0.0, 1.0);
        // Prompt-jump approximation gives n_+ = β/(β-ρ) = 2 followed
        // by a slow rise on the inhour period. The full transient
        // adds fast-group catch-up (λ_6 = 3/s has 4 half-lives in
        // 1 s), so the expected band is wider than PJA alone.
        assert!(s1.n > 1.5, "no prompt-jump rise, got {}", s1.n);
        assert!(s1.n < 5.0, "n shot up super-prompt, got {}", s1.n);
    }

    #[test]
    fn shutdown_decays_via_longest_lived_precursor() {
        // Strong negative ρ kills prompt; long-term decay rate should
        // approach −λ₁ (longest-lived group, ~80 s half-life).
        let p = KineticsParams::keepin_u235_thermal();
        let s0 = equilibrium_state(1.0, &p);
        let mut k = PointKinetics::new(p, s0);
        let s1 = k.step(-10.0 * p.beta_total(), 0.0, 60.0);
        let s2 = k.step(-10.0 * p.beta_total(), 0.0, 60.0);
        let observed_decay = (s2.n / s1.n).ln() / 60.0;
        // Should be approximately −λ₁ = −0.0124. Looser bound here:
        // long-step transient still has groups 2–3 alive.
        assert!(
            observed_decay < -0.005 && observed_decay > -0.05,
            "decay rate {} outside expected band",
            observed_decay
        );
    }
}
