//! Reusable building blocks for Monte Carlo simulation in Rust.
//! Extracted from
//! [open_rust_mc](https://github.com/sorcerer86pt/open_rust_mc).
//! Math layer is always available; the `nuclear` feature adds
//! OpenMC HDF5 / WMP / S(α,β); `parallel` adds rayon batch APIs.

pub mod batch;
pub mod cdf;
pub mod cp;
pub mod cram;
pub mod decay;
pub mod depletion;
pub mod doppler;
pub mod ducru;
pub mod error;
pub mod expm;
pub mod fission_yields;
pub mod geometry;
pub mod kinetics;
pub mod photon;
pub mod physics;
#[cfg(feature = "preview")]
pub mod preview;
pub mod rng;
pub mod svd;
pub mod table;
pub mod tally;
pub mod transport;
pub mod urr;

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
    pub mod loader {
        pub use crate::nuclear_loader::*;
    }
}

// Internal modules backing `nuclear::*`. Kept private so the public
// paths are stable across implementation reshuffles.
#[cfg(feature = "nuclear")]
mod nuclear_hdf5;
#[cfg(feature = "nuclear")]
mod nuclear_loader;
#[cfg(feature = "nuclear")]
mod nuclear_thermal;
#[cfg(feature = "nuclear")]
mod nuclear_wmp;

// Convenience re-exports for the pure-math API.
pub use cdf::LogDecimatedCdf;
pub use cp::{CpDecomposition, cp_greedy_rank1, max_abs_error, relative_l2_error};
pub use ducru::{ducru_constrained_weights, ducru_unity_weights, ducru_weights, nearest_k_columns};
pub use rng::Pcg64;
pub use svd::{LogHashIndex, Svd, SvdKernel};
pub use table::{PointwiseTable, StochTempTable};
