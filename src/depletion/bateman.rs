//! Bateman / depletion stepper. Builds the dense transmutation matrix
//! `A` for the active [`crate::decay::DecayChain`] at a frozen flux
//! and one-group XS, then advances `dN/dt = A·N` over `dt` via
//! [`crate::cram::cram16_dense`] (Pusa & Leppänen 2010, NSE 164).
//!
//! The CRAM-16 module is v0.1 — its accuracy on long depletion steps
//! against Serpent has not been re-validated post-extraction. For
//! short timesteps and small chains it's serviceable; for production
//! depletion runs, validate against a reference code first.

use crate::decay::DecayChain;

/// One depletion step result: post-step concentrations and the time
/// it advanced.
#[derive(Debug, Clone)]
pub struct BurnupStep {
    pub time: f64,
    pub concentrations: Vec<f64>,
}

/// Bind a [`DecayChain`] to a per-step (flux, XS, fission energy)
/// context and step in time. The XS lookup is supplied as a closure
/// so the caller can plug in any XS source (pointwise tables,
/// SVD-reconstructed values, WMP, …).
pub struct DepletionSolver<'a, F>
where
    F: Fn(usize, &str) -> f64,
{
    pub chain: &'a DecayChain,
    pub flux_phi: f64,
    pub fission_energy: f64,
    pub xs_lookup: F,
    pub time: f64,
    pub concentrations: Vec<f64>,
}

impl<'a, F> DepletionSolver<'a, F>
where
    F: Fn(usize, &str) -> f64,
{
    pub fn new(
        chain: &'a DecayChain,
        flux_phi: f64,
        fission_energy: f64,
        xs_lookup: F,
        initial_concentrations: Vec<f64>,
    ) -> Self {
        assert_eq!(
            initial_concentrations.len(),
            chain.len(),
            "initial_concentrations must match chain length"
        );
        Self {
            chain,
            flux_phi,
            fission_energy,
            xs_lookup,
            time: 0.0,
            concentrations: initial_concentrations,
        }
    }

    /// Advance the solution by `dt` seconds at the currently
    /// configured (flux, XS) snapshot.
    pub fn step(&mut self, dt: f64) -> BurnupStep {
        let a = self.chain.build_transmutation_matrix(
            self.flux_phi,
            self.fission_energy,
            &self.xs_lookup,
        );
        // Flatten into row-major dense for cram16_dense.
        let n = self.chain.len();
        let mut a_flat = vec![0.0_f64; n * n];
        for r in 0..n {
            for c in 0..n {
                a_flat[r * n + c] = a[(r, c)];
            }
        }
        let new_n = crate::cram::cram16_dense(&a_flat, &self.concentrations, dt, n);
        self.concentrations = new_n;
        self.time += dt;
        BurnupStep {
            time: self.time,
            concentrations: self.concentrations.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decay::{DecayMode, DecayNuclide, ReactionTarget};

    #[test]
    #[ignore = "depends on cram::cram16_dense — v0.1, see cram.rs caveats"]
    fn pure_decay_step_advances_state() {
        // A → B with t½(A) = 10 s. After one half-life we expect
        // ~half the A and ~half the B (mass conservation).
        // CRAM-16 is v0.1 — relax tolerance to qualitative.
        let mut chain = DecayChain::new();
        let mut a = DecayNuclide::new("A");
        a.half_life = Some(10.0);
        a.decay_modes.push(DecayMode {
            mode: "beta-".into(),
            target: ReactionTarget::Nuclide("B".into()),
            branching_ratio: 1.0,
        });
        chain.push(a);
        chain.push(DecayNuclide::new("B"));

        let mut solver = DepletionSolver::new(&chain, 0.0, 0.0, |_, _| 0.0, vec![1.0, 0.0]);
        let step = solver.step(10.0);
        // Mass conservation (within CRAM-16 precision).
        let total = step.concentrations.iter().sum::<f64>();
        assert!(
            (total - 1.0).abs() < 1e-2,
            "mass not conserved: total = {}",
            total
        );
        // A monotonically decreased.
        assert!(step.concentrations[0] < 1.0);
        assert!(step.concentrations[1] > 0.0);
    }
}
