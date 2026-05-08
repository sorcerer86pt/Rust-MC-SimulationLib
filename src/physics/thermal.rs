//! Thermal-scattering trait. Concrete S(α,β) data lives in the
//! feature-gated `nuclear_thermal` module, but the transport loop
//! only needs this small interface — so it stays feature-independent
//! and the main `Nuclide` / `Material` types can carry an optional
//! `Arc<dyn ThermalScatterer>` without dragging in the OpenMC HDF5
//! dependency.

use crate::rng::Pcg64;

/// Bound-atom thermal-scattering kernel. Above [`Self::energy_max`],
/// callers should fall back to the free-atom elastic model; below it,
/// the bound-atom kernel replaces the elastic channel.
pub trait ThermalScatterer: Send + Sync {
    /// Highest incident energy (eV) at which the bound kernel applies.
    fn energy_max(&self) -> f64;

    /// Total bound thermal-scattering XS (barns) at `energy` and
    /// `temperature_k`. Implementations are free to pick a single
    /// library temperature (stochastic or nearest-neighbour); this
    /// API is the steady-state value the macro-XS sums against.
    fn total_xs(&self, energy: f64, temperature_k: f64) -> f64;

    /// Sample `(E_out, μ_lab)` from the bound-atom kernel.
    fn sample(&self, energy: f64, temperature_k: f64, rng: &mut Pcg64) -> (f64, f64);
}
