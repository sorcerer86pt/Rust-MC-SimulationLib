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

use crate::expm::expm_pade;

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

#[cfg(test)]
mod tests {
    use super::*;

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
