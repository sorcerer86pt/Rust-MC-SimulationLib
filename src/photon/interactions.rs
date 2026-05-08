//! Photon interaction kernels. Free Klein-Nishina (Compton),
//! photoelectric (terminal), pair production (Bethe-Heitler partition
//! + annihilation), coherent (forward-peaked v1 approximation).

use crate::rng::Pcg64;

pub const M_E_C2_EV: f64 = 510_998.95;
pub const PAIR_THRESHOLD_EV: f64 = 2.0 * M_E_C2_EV;

/// Photon reaction types selectable from the macroscopic XS sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonReaction {
    Coherent,
    Incoherent,
    Photoelectric,
    PairProduction,
}

/// Outcome of a single photon interaction. Fields not relevant to a
/// given reaction are 0/NaN — see the variant docs.
#[derive(Debug, Clone)]
pub struct PhotonOutcome {
    /// Outgoing primary photon energy (eV). 0 → photon absorbed.
    pub energy_out: f64,
    /// Lab-frame scattering cosine of the outgoing primary, μ ∈ [-1, 1].
    /// Meaningless when `energy_out == 0`.
    pub mu: f64,
    /// Locally-deposited energy (eV) — electron(s) kinetic + photon
    /// absorbed. Goes to a kerma tally, no electron transport.
    pub local_deposition: f64,
    /// Annihilation γ's emitted at this site (pair-production only).
    /// Each entry is `(energy_eV, mu_lab)` with isotropic axis
    /// resolved by the caller using `rotate_direction`.
    pub annihilation_photons: Vec<(f64, f64)>,
}

impl PhotonOutcome {
    fn absorbed(local: f64) -> Self {
        Self {
            energy_out: 0.0,
            mu: 0.0,
            local_deposition: local,
            annihilation_photons: Vec::new(),
        }
    }
    fn scattered(energy_out: f64, mu: f64, local: f64) -> Self {
        Self {
            energy_out,
            mu,
            local_deposition: local,
            annihilation_photons: Vec::new(),
        }
    }
}

/// Free-electron Klein-Nishina sampler (Koblinger composite envelope).
///
/// Returns the outgoing photon energy and lab-frame `μ`. The recoil
/// electron's kinetic energy is the difference, deposited locally
/// under the kerma approximation.
pub fn sample_compton_free(energy_in: f64, rng: &mut Pcg64) -> PhotonOutcome {
    let alpha = energy_in / M_E_C2_EV;
    let kappa = 1.0 + 2.0 * alpha;
    let a1 = kappa.ln();
    let a2 = 0.5 * (1.0 - 1.0 / (kappa * kappa));
    let p1 = a1 / (a1 + a2);
    loop {
        let xi = rng.uniform();
        let k;
        if rng.uniform() < p1 {
            // 1/k component on [1/κ, 1].
            k = (1.0 / kappa).powf(1.0 - xi);
        } else {
            // k component on [1/κ, 1].
            k = (1.0 / (kappa * kappa) + xi * (1.0 - 1.0 / (kappa * kappa))).sqrt();
        }
        let mu = 1.0 - (1.0 - k) / (alpha * k);
        let mu = mu.clamp(-1.0, 1.0);
        // Free Klein-Nishina rejection ratio: f(k, μ) / envelope.
        let g = 1.0 - (1.0 - mu * mu) / (k + 1.0 / k);
        if rng.uniform() < g {
            let e_out = energy_in * k;
            let local = energy_in - e_out;
            return PhotonOutcome::scattered(e_out, mu, local);
        }
    }
}

/// Photoelectric absorption — destroys the photon. Local energy
/// deposition is the full incident energy under the kerma
/// approximation. Atomic relaxation (K-shell fluorescence /
/// Auger cascades) is not modeled — biases dose at sub-100 keV in
/// high-Z materials, fine for shielding above that.
pub fn sample_photoelectric(energy_in: f64) -> PhotonOutcome {
    PhotonOutcome::absorbed(energy_in)
}

/// Pair production. Below `PAIR_THRESHOLD_EV` returns `None` (caller
/// must verify reaction sampling never selects this branch sub-
/// threshold).
///
/// Energy partition follows Bethe-Heitler symmetric shape sampled by
/// rejection from a uniform envelope. Both leptons deposit kinetic
/// energy locally; the positron annihilates at rest emitting two
/// 511 keV photons isotropically (axis resolved by the caller).
pub fn sample_pair(energy_in: f64, rng: &mut Pcg64) -> Option<PhotonOutcome> {
    if energy_in < PAIR_THRESHOLD_EV {
        return None;
    }
    let t_total = energy_in - PAIR_THRESHOLD_EV;
    let epsilon = sample_bethe_heitler_epsilon(rng);
    let electron_ke = epsilon * t_total;
    let positron_ke = (1.0 - epsilon) * t_total;
    Some(PhotonOutcome {
        energy_out: 0.0,
        mu: 0.0,
        local_deposition: electron_ke + positron_ke,
        // Two back-to-back 511 keV γ's. The caller picks an isotropic
        // axis: μ for one, the other is in the opposite direction.
        annihilation_photons: vec![(M_E_C2_EV, 0.0), (M_E_C2_EV, 0.0)],
    })
}

/// Coherent (Rayleigh) scattering. v1 forward-peaked approximation:
/// `μ = 1`, energy unchanged. Physically biased at low energies where
/// Rayleigh has a wide angular distribution governed by the atomic
/// form factor `F(x, Z)`. Acceptable for shielding above ~100 keV in
/// any material; for the bound-electron limit use the parent
/// project's `photon::coherent` kernel.
pub fn sample_coherent_forward(energy_in: f64) -> PhotonOutcome {
    PhotonOutcome::scattered(energy_in, 1.0, 0.0)
}

/// Bethe-Heitler `ε` sampler. `f(ε) = ε² + (1-ε)² + (2/3) ε(1-ε)`,
/// peak `1` at the endpoints, minimum `2/3` in the middle. Rejection
/// from a uniform envelope.
fn sample_bethe_heitler_epsilon(rng: &mut Pcg64) -> f64 {
    loop {
        let eps = rng.uniform();
        let f = eps * eps + (1.0 - eps) * (1.0 - eps) + (2.0 / 3.0) * eps * (1.0 - eps);
        if rng.uniform() < f {
            return eps;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::rng::Pcg64;

    #[test]
    fn compton_low_energy_is_thomson_like() {
        // At E ≪ m_e c², KN reduces to Thomson and the mean μ is 0
        // by symmetry. Mean ⟨E_out⟩ ≈ E_in.
        let mut rng = Pcg64::new(1, 1);
        let energy = 1.0e3; // 1 keV
        let mut sum_e = 0.0;
        let n = 50_000;
        for _ in 0..n {
            let o = sample_compton_free(energy, &mut rng);
            sum_e += o.energy_out;
        }
        let mean_e = sum_e / n as f64;
        let rel = (mean_e - energy).abs() / energy;
        assert!(
            rel < 0.01,
            "Thomson limit violated: ⟨E⟩ = {mean_e}, want {energy}"
        );
    }

    #[test]
    fn compton_high_energy_loses_energy_on_average() {
        let mut rng = Pcg64::new(2, 1);
        let energy = 5.0e6;
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let o = sample_compton_free(energy, &mut rng);
            sum += o.energy_out;
        }
        let mean = sum / n as f64;
        assert!(mean < energy, "outgoing should be ≤ incident on average");
        assert!(mean > 0.1 * energy, "outgoing too low: {mean}");
    }

    #[test]
    fn pair_below_threshold_is_none() {
        let mut rng = Pcg64::new(3, 1);
        assert!(sample_pair(1.0e6, &mut rng).is_none());
    }

    #[test]
    fn pair_above_threshold_gives_two_annihilations() {
        let mut rng = Pcg64::new(4, 1);
        let o = sample_pair(5.0e6, &mut rng).unwrap();
        assert_eq!(o.annihilation_photons.len(), 2);
        for (e, _) in &o.annihilation_photons {
            assert!((e - M_E_C2_EV).abs() < 1.0);
        }
        assert_eq!(o.energy_out, 0.0);
    }
}
