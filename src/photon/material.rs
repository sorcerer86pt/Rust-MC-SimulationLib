//! Photon material: a homogeneous mixture of elements at given atom
//! densities. Macroscopic XS aggregation across the five reaction
//! channels, with a flight-distance / channel sampler matching the
//! neutron `Material` interface so the transport loops have similar
//! shape.

use std::sync::Arc;

use crate::photon::data::{PhotonElement, interpolate_log_log};
use crate::photon::interactions::PhotonReaction;
use crate::rng::Pcg64;

/// Composition of a photon target.
pub struct PhotonMaterial {
    pub name: String,
    /// `(element, atom_density_per_b_per_cm)` pairs.
    pub elements: Vec<(Arc<PhotonElement>, f64)>,
}

impl PhotonMaterial {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            elements: Vec::new(),
        }
    }

    pub fn add(&mut self, element: Arc<PhotonElement>, atom_density: f64) {
        self.elements.push((element, atom_density));
    }

    /// Macroscopic total XS Σ_t at `energy` (1/cm). Sum over channels
    /// and elements.
    pub fn macro_total(&self, energy: f64) -> f64 {
        let mut sigma = 0.0_f64;
        for (e, density) in &self.elements {
            sigma += density
                * (interpolate_log_log(&e.energy, &e.coherent_xs, energy)
                    + interpolate_log_log(&e.energy, &e.incoherent_xs, energy)
                    + interpolate_log_log(&e.energy, &e.photoelectric_xs, energy)
                    + interpolate_log_log(&e.energy, &e.pair_production_nuclear_xs, energy)
                    + interpolate_log_log(&e.energy, &e.pair_production_electron_xs, energy));
        }
        sigma
    }

    /// Macroscopic XS for one channel at `energy` (1/cm). Used for
    /// post-collision sampling — pick reaction first, then element
    /// within the chosen reaction.
    pub fn macro_channel(&self, channel: PhotonReaction, energy: f64) -> f64 {
        let mut sigma = 0.0_f64;
        for (e, density) in &self.elements {
            let xs = match channel {
                PhotonReaction::Coherent => interpolate_log_log(&e.energy, &e.coherent_xs, energy),
                PhotonReaction::Incoherent => {
                    interpolate_log_log(&e.energy, &e.incoherent_xs, energy)
                }
                PhotonReaction::Photoelectric => {
                    interpolate_log_log(&e.energy, &e.photoelectric_xs, energy)
                }
                PhotonReaction::PairProduction => {
                    interpolate_log_log(&e.energy, &e.pair_production_nuclear_xs, energy)
                        + interpolate_log_log(&e.energy, &e.pair_production_electron_xs, energy)
                }
            };
            sigma += density * xs;
        }
        sigma
    }

    /// Sample a reaction channel proportional to its macroscopic XS.
    /// Returns `None` only when the total XS is non-positive (caller
    /// should not have reached a collision).
    pub fn sample_reaction(&self, energy: f64, rng: &mut Pcg64) -> Option<PhotonReaction> {
        let sigma_coh = self.macro_channel(PhotonReaction::Coherent, energy);
        let sigma_inc = self.macro_channel(PhotonReaction::Incoherent, energy);
        let sigma_pe = self.macro_channel(PhotonReaction::Photoelectric, energy);
        let sigma_pp = self.macro_channel(PhotonReaction::PairProduction, energy);
        let total = sigma_coh + sigma_inc + sigma_pe + sigma_pp;
        if total <= 0.0 {
            return None;
        }
        let xi = rng.uniform() * total;
        if xi < sigma_coh {
            Some(PhotonReaction::Coherent)
        } else if xi < sigma_coh + sigma_inc {
            Some(PhotonReaction::Incoherent)
        } else if xi < sigma_coh + sigma_inc + sigma_pe {
            Some(PhotonReaction::Photoelectric)
        } else {
            Some(PhotonReaction::PairProduction)
        }
    }
}
