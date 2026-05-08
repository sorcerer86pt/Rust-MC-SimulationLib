//! Photon-interaction data, element-indexed. Slim v1 — five-channel
//! cross sections on a shared energy grid, no form factors / Compton
//! profiles / subshells / bremsstrahlung. The full data layout from
//! the parent project's `photon::data` (with sampling auxiliaries) is
//! a future expansion if anyone wants bound-electron Compton, exact
//! Rayleigh, or atomic relaxation.

/// All photon-interaction cross sections for one element (Z), on a
/// shared energy grid.
#[derive(Debug, Clone)]
pub struct PhotonElement {
    pub z: u32,
    pub symbol: String,
    /// Photon energy grid in eV, ascending.
    pub energy: Vec<f64>,
    pub coherent_xs: Vec<f64>,
    pub incoherent_xs: Vec<f64>,
    pub photoelectric_xs: Vec<f64>,
    pub pair_production_nuclear_xs: Vec<f64>,
    pub pair_production_electron_xs: Vec<f64>,
}

impl PhotonElement {
    pub fn n_energy(&self) -> usize {
        self.energy.len()
    }

    /// Total XS (barns/atom) at grid point `i`.
    pub fn total_xs_at(&self, i: usize) -> f64 {
        self.coherent_xs[i]
            + self.incoherent_xs[i]
            + self.photoelectric_xs[i]
            + self.pair_production_nuclear_xs[i]
            + self.pair_production_electron_xs[i]
    }
}

/// Log-log interpolation on a strictly ascending grid. Returns 0 below
/// the grid, and saturates the last value above it.
#[inline]
pub fn interpolate_log_log(xs_grid: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs_grid.len();
    if n == 0 || x <= 0.0 {
        return 0.0;
    }
    if x <= xs_grid[0] {
        return ys[0];
    }
    if x >= xs_grid[n - 1] {
        return ys[n - 1];
    }
    let i =
        match xs_grid.binary_search_by(|v| v.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less)) {
            Ok(i) => return ys[i],
            Err(i) => i - 1,
        };
    let x_lo = xs_grid[i];
    let x_hi = xs_grid[i + 1];
    let y_lo = ys[i].max(1.0e-30);
    let y_hi = ys[i + 1].max(1.0e-30);
    let t = (x.ln() - x_lo.ln()) / (x_hi.ln() - x_lo.ln());
    (y_lo.ln() + t * (y_hi.ln() - y_lo.ln())).exp()
}
