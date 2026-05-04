//! `rust-mc-sim` — low-rank approximations for tabulated
//! multi-way data, with a (feature-gated) OpenMC-compatible
//! nuclear-data layer.
//!
//! Extracted from
//! [open_rust_mc](https://github.com/sorcerer86pt/open_rust_mc) so
//! the validated algorithms can be reused outside the original
//! Monte Carlo neutron transport context.
//!
//! # Layers
//!
//! Pure math (always available):
//!
//! * [`svd`] — truncated SVD via faer + cache-friendly
//!   reconstruction kernel with optional log-uniform hash index.
//! * [`cp`] — CP / PARAFAC decomposition of a 3-tensor (greedy
//!   rank-1 deflation).
//! * [`ducru`] — Ducru-2017 free-Doppler weights for off-grid
//!   reconstruction, raw and partition-of-unity normalised.
//! * [`cdf`] — log-decimated CDF with inverse-transform sampling
//!   for categorical outcomes whose probabilities depend on a
//!   continuous coordinate.
//! * [`rng`] — PCG-64 RNG used by [`cdf`] sampling and the
//!   nuclear-data layer; exposed so callers can plug their own
//!   reproducible streams in.
//!
//! Nuclear-data layer (`feature = "nuclear"`, pulls in `hdf5-pure`):
//!
//! * [`nuclear::wmp`] — Windowed Multipole evaluator with
//!   Humlicek W4 Faddeeva.
//! * [`nuclear::thermal`] — S(α,β) thermal scattering kernels and
//!   sampling.
//! * [`nuclear::hdf5`] — pure-Rust OpenMC HDF5 reader (cross
//!   sections, level metadata, angular distributions, energy
//!   distributions, URR, S(α,β)).
//!
//! # Re-exports
//!
//! For convenience, the pure-math types are re-exported at the
//! crate root so the typical caller can write
//! `use rust_mc_sim::{SvdKernel, ducru_unity_weights};`.

pub mod batch;
pub mod cdf;
pub mod cp;
pub mod ducru;
pub mod error;
pub mod rng;
pub mod svd;

#[cfg(feature = "nuclear")]
pub mod nuclear {
    //! Nuclear-data-specific layer (feature `"nuclear"`).
    //!
    //! Ports of the WMP, S(α,β), and OpenMC HDF5 reader extracted
    //! from `open_rust_mc`. The internal data types match OpenMC's
    //! HDF5 conventions; the public APIs are consistent with the
    //! original engine so anyone reading the parent project's
    //! source can map calls 1-to-1.

    pub mod wmp {
        pub use crate::nuclear_wmp::*;
    }
    pub mod thermal {
        pub use crate::nuclear_thermal::*;
    }
    pub mod hdf5 {
        pub use crate::nuclear_hdf5::*;
    }
}

// Internal modules backing `nuclear::*`. Kept private so the public
// paths are stable across implementation reshuffles.
#[cfg(feature = "nuclear")]
mod nuclear_hdf5;
#[cfg(feature = "nuclear")]
mod nuclear_thermal;
#[cfg(feature = "nuclear")]
mod nuclear_wmp;

// Convenience re-exports for the pure-math API.
pub use cdf::LogDecimatedCdf;
pub use cp::{CpDecomposition, cp_greedy_rank1, max_abs_error, relative_l2_error};
pub use ducru::{ducru_unity_weights, ducru_weights, nearest_k_columns};
pub use rng::Pcg64;
pub use svd::{LogHashIndex, Svd, SvdKernel};
