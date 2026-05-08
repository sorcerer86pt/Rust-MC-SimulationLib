//! Photon (γ) transport. Slim v1 covering shielding-quality kinematics:
//!
//!   * **Compton (incoherent)** — free Klein-Nishina sampling with
//!     Koblinger composite envelope. No bound-electron `S(x, Z)/Z`
//!     rejection and no Doppler broadening from Compton profiles —
//!     adequate for ≥ ~10 keV photons through low-to-medium Z.
//!   * **Photoelectric** — terminal absorption (kerma approximation;
//!     atomic relaxation / fluorescence is not modeled).
//!   * **Pair production** — Bethe-Heitler ε partition; both leptons
//!     deposit kinetic energy locally; one isotropic axis emits the
//!     two 511 keV annihilation γ's back-to-back.
//!   * **Coherent (Rayleigh)** — forward-peaked approximation (μ = 1,
//!     no energy loss). High-energy limit; biased at low E. Listed as
//!     a v1 simplification — full atomic-form-factor sampling sits in
//!     the parent project's `photon::coherent`.
//!
//! What you can simulate end-to-end:
//!   * γ shielding through arbitrary CSG geometries
//!   * dose / fluence at a target via [`crate::tally::FluxTally`]
//!   * fixed γ source from monoenergetic line (Cs-137, Co-60, …)
//!     plus a Watt-style fission-spectrum γ proxy
//!
//! Coupled n-γ (capture / fission γ production from neutron transport)
//! is not yet wired — the HDF5 reader carries `PhotonProduct` data,
//! but the neutron collision dispatch does not currently bank γ's. A
//! follow-up commit can add that hook.

pub mod data;
pub mod interactions;
pub mod material;
pub mod source;
pub mod transport;

pub use data::PhotonElement;
pub use interactions::{
    M_E_C2_EV, PAIR_THRESHOLD_EV, PhotonOutcome, PhotonReaction, sample_compton_free, sample_pair,
    sample_photoelectric,
};
pub use material::PhotonMaterial;
pub use source::{IsotropicLineSource, MonoBoxSource, PhotonSource, SourcePhoton};
pub use transport::{PhotonFixedSourceConfig, PhotonFixedSourceResult, run_photon_fixed_source};

#[cfg(feature = "nuclear")]
pub mod loader;
