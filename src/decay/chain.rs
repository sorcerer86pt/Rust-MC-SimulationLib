use std::collections::HashMap;

/// Where a decay or transmutation reaction sends its product. Some
/// channels in OpenMC chain files have no `target` attribute — those
/// are treated as `Lost` (mass leaves the tracked set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReactionTarget {
    Nuclide(String),
    Lost,
}

/// One radioactive-decay branch. Branching ratios across all decay
/// modes of a parent nuclide should sum to 1 ± round-off.
#[derive(Debug, Clone)]
pub struct DecayMode {
    /// Decay channel label: "alpha", "beta-", "ec/beta+", "it",
    /// "sf", "n", "p", "2n", "2p", … straight from chain.xml.
    pub mode: String,
    pub target: ReactionTarget,
    pub branching_ratio: f64,
}

/// One transmutation reaction (n,γ), (n,2n), fission, …
#[derive(Debug, Clone)]
pub struct ReactionChannel {
    /// "(n,gamma)", "(n,2n)", "fission", …
    pub mt: String,
    pub target: ReactionTarget,
    pub q_value: f64,
    pub branching_ratio: f64,
}

/// Chain entry for one nuclide: half-life, decay modes, and the
/// transmutation channels that connect it to other nuclides.
#[derive(Debug, Clone)]
pub struct DecayNuclide {
    pub name: String,
    /// Half-life in seconds; `None` for stable nuclides.
    pub half_life: Option<f64>,
    /// Total recoverable decay-heat energy per decay (eV). 0.0 when
    /// not provided by the chain file.
    pub decay_energy: f64,
    pub decay_modes: Vec<DecayMode>,
    pub reactions: Vec<ReactionChannel>,
    /// Energy-dependent fission-product yields keyed by parent
    /// energy (eV), if any. Empty when this nuclide is not a
    /// fissionable one in the chain.
    pub fission_yields: Option<crate::fission_yields::FissionYields>,
}

impl DecayNuclide {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            half_life: None,
            decay_energy: 0.0,
            decay_modes: Vec::new(),
            reactions: Vec::new(),
            fission_yields: None,
        }
    }

    /// Decay constant λ = ln 2 / t½. Returns 0 for stable nuclides.
    pub fn decay_constant(&self) -> f64 {
        match self.half_life {
            Some(t) if t > 0.0 => std::f64::consts::LN_2 / t,
            _ => 0.0,
        }
    }
}

/// A full depletion chain: many nuclides linked by decay branches and
/// reaction channels. Wrapped with name → index lookup so the
/// transmutation matrix can be assembled in O(N + edges).
#[derive(Debug, Clone, Default)]
pub struct DecayChain {
    pub nuclides: Vec<DecayNuclide>,
    name_to_idx: HashMap<String, usize>,
}

impl DecayChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, n: DecayNuclide) -> usize {
        let idx = self.nuclides.len();
        self.name_to_idx.insert(n.name.clone(), idx);
        self.nuclides.push(n);
        idx
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.nuclides.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nuclides.is_empty()
    }

    /// Build the dense transmutation matrix `A` (1/s) such that
    /// `dN/dt = A·N`, where `N` is the column vector of atom
    /// densities indexed by chain position.
    ///
    /// `flux_phi`: scalar neutron flux (1/cm²·s).
    /// `xs_lookup(parent_idx, mt) → microscopic XS in barns`.
    /// `fission_energy`: incident-neutron energy (eV) used to look
    /// up fission yields. Use 0.0253 (thermal) for LWR analysis.
    ///
    /// Reaction-channel sigma·flux contributions add to the off-diag
    /// term `A[child, parent] += σ_b · 1e-24 · φ · branching` and
    /// subtract from the diagonal `A[parent, parent] -= σ_b · 1e-24
    /// · φ`. Decay terms add `λ·branch` and subtract `λ` similarly.
    /// Fission products are spread across the chain using the parent
    /// nuclide's `fission_yields` evaluated at `fission_energy`.
    pub fn build_transmutation_matrix(
        &self,
        flux_phi: f64,
        fission_energy: f64,
        xs_lookup: impl Fn(usize, &str) -> f64,
    ) -> faer::Mat<f64> {
        let n = self.nuclides.len();
        let mut a = faer::Mat::<f64>::zeros(n, n);

        for (parent_idx, parent) in self.nuclides.iter().enumerate() {
            // Decay.
            let lambda = parent.decay_constant();
            if lambda > 0.0 {
                a[(parent_idx, parent_idx)] -= lambda;
                for m in &parent.decay_modes {
                    if let ReactionTarget::Nuclide(child) = &m.target {
                        if let Some(child_idx) = self.index_of(child) {
                            a[(child_idx, parent_idx)] += lambda * m.branching_ratio;
                        }
                    }
                }
            }
            // Transmutation reactions.
            for r in &parent.reactions {
                let sigma = xs_lookup(parent_idx, &r.mt);
                if sigma <= 0.0 {
                    continue;
                }
                let rate = sigma * 1.0e-24 * flux_phi;
                a[(parent_idx, parent_idx)] -= rate;
                if r.mt == "fission" {
                    if let Some(yields) = &parent.fission_yields {
                        for (product, y) in yields.products_at_energy(fission_energy) {
                            if let Some(p_idx) = self.index_of(&product) {
                                a[(p_idx, parent_idx)] += rate * y;
                            }
                        }
                    }
                } else if let ReactionTarget::Nuclide(child) = &r.target {
                    if let Some(child_idx) = self.index_of(child) {
                        a[(child_idx, parent_idx)] += rate * r.branching_ratio;
                    }
                }
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_constant_for_stable_is_zero() {
        let n = DecayNuclide::new("Fe56");
        assert_eq!(n.decay_constant(), 0.0);
    }

    #[test]
    fn decay_constant_matches_half_life() {
        let mut n = DecayNuclide::new("Co60");
        n.half_life = Some(1.663e8); // 5.27 years in s
        let lambda = n.decay_constant();
        assert!((lambda - std::f64::consts::LN_2 / 1.663e8).abs() < 1e-20);
    }

    #[test]
    fn matrix_balances_for_pure_decay_chain() {
        // A → B → (stable). Mass conservation: column sums are zero
        // when no flux losses (no `Lost` targets in this case).
        let mut chain = DecayChain::new();
        let mut a = DecayNuclide::new("A");
        a.half_life = Some(1.0);
        a.decay_modes.push(DecayMode {
            mode: "beta-".into(),
            target: ReactionTarget::Nuclide("B".into()),
            branching_ratio: 1.0,
        });
        chain.push(a);

        let mut b = DecayNuclide::new("B");
        b.half_life = Some(2.0);
        b.decay_modes.push(DecayMode {
            mode: "beta-".into(),
            target: ReactionTarget::Nuclide("C".into()),
            branching_ratio: 1.0,
        });
        chain.push(b);
        chain.push(DecayNuclide::new("C"));

        let m = chain.build_transmutation_matrix(0.0, 0.0, |_, _| 0.0);
        // Column 0 (A): −λ_A on diag, +λ_A on row B → sum 0
        let col0 = m[(0, 0)] + m[(1, 0)] + m[(2, 0)];
        assert!(col0.abs() < 1e-15);
        // Column 1 (B): −λ_B on diag, +λ_B on row C → sum 0
        let col1 = m[(0, 1)] + m[(1, 1)] + m[(2, 1)];
        assert!(col1.abs() < 1e-15);
    }
}
