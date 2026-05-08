//! Per-incident-energy independent (or cumulative) fission-product
//! yields. Stored as a sparse list of (energy, products, yields)
//! tables; queries interpolate linearly in incident energy and
//! return a flat (product, yield) list.

use std::collections::BTreeMap;

/// Fission-product yields at one incident-neutron energy.
#[derive(Debug, Clone, Default)]
pub struct YieldTable {
    /// Product names parallel to `yields`.
    pub products: Vec<String>,
    /// Yields parallel to `products` (atoms / fission).
    pub yields: Vec<f64>,
}

impl YieldTable {
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.products
            .iter()
            .map(String::as_str)
            .zip(self.yields.iter().copied())
    }
}

/// Energy-keyed yield tables. Lookup interpolates linearly between
/// the two bracketing tables and saturates outside the grid.
#[derive(Debug, Clone, Default)]
pub struct FissionYields {
    /// Sorted (energy_eV → table). Use `BTreeMap` so iteration is
    /// in energy order.
    pub tables: BTreeMap<u64, YieldTable>,
}

impl FissionYields {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, energy_ev: f64, table: YieldTable) {
        // Encode energy as bits for total-ordering map key (no
        // negative energies in practice, no NaN).
        self.tables.insert(energy_ev.to_bits(), table);
    }

    pub fn energies(&self) -> impl Iterator<Item = f64> + '_ {
        self.tables.keys().copied().map(f64::from_bits)
    }

    /// All (product, yield) at `energy_ev`. Linear interpolation
    /// between bracketing tables, saturation outside. Products that
    /// only appear in one bracket are still returned with the
    /// linear contribution from that bracket.
    pub fn products_at_energy(&self, energy_ev: f64) -> Vec<(String, f64)> {
        if self.tables.is_empty() {
            return Vec::new();
        }
        let energies: Vec<f64> = self.energies().collect();
        // Find bracket.
        let last = energies.len() - 1;
        let (lo_e, hi_e, w) = if energy_ev <= energies[0] {
            (energies[0], energies[0], 0.0)
        } else if energy_ev >= energies[last] {
            (energies[last], energies[last], 0.0)
        } else {
            let upper = energies
                .iter()
                .position(|&e| e >= energy_ev)
                .unwrap_or(last);
            let lower = upper - 1;
            let e_lo = energies[lower];
            let e_hi = energies[upper];
            let w = (energy_ev - e_lo) / (e_hi - e_lo);
            (e_lo, e_hi, w)
        };
        let lo = self.tables.get(&lo_e.to_bits()).expect("table missing");
        let hi = self.tables.get(&hi_e.to_bits()).expect("table missing");

        let mut acc: BTreeMap<String, f64> = BTreeMap::new();
        for (p, y) in lo.iter() {
            *acc.entry(p.to_string()).or_insert(0.0) += (1.0 - w) * y;
        }
        for (p, y) in hi.iter() {
            *acc.entry(p.to_string()).or_insert(0.0) += w * y;
        }
        acc.into_iter().collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn t(products: &[&str], yields: &[f64]) -> YieldTable {
        YieldTable {
            products: products.iter().map(|s| s.to_string()).collect(),
            yields: yields.to_vec(),
        }
    }

    #[test]
    fn single_table_returns_unchanged() {
        let mut fy = FissionYields::new();
        fy.insert(0.0253, t(&["Cs137", "Sr90"], &[0.06, 0.04]));
        let got = fy.products_at_energy(0.0253);
        let mut as_map: std::collections::HashMap<_, _> = got.into_iter().collect();
        assert!((as_map.remove("Cs137").unwrap() - 0.06).abs() < 1e-15);
        assert!((as_map.remove("Sr90").unwrap() - 0.04).abs() < 1e-15);
    }

    #[test]
    fn linear_interpolation_between_two_energies() {
        let mut fy = FissionYields::new();
        fy.insert(0.0253, t(&["Cs137"], &[0.06]));
        fy.insert(5.0e5, t(&["Cs137"], &[0.07]));
        // Halfway → average.
        let got = fy.products_at_energy(2.5e5);
        let y = got.iter().find(|(p, _)| p == "Cs137").unwrap().1;
        assert!((y - 0.065).abs() < 1e-3);
    }

    #[test]
    fn saturates_outside_grid() {
        let mut fy = FissionYields::new();
        fy.insert(0.0253, t(&["Cs137"], &[0.06]));
        fy.insert(5.0e5, t(&["Cs137"], &[0.07]));
        let lo = fy.products_at_energy(1e-5);
        let hi = fy.products_at_energy(1e8);
        assert!((lo[0].1 - 0.06).abs() < 1e-15);
        assert!((hi[0].1 - 0.07).abs() < 1e-15);
    }
}
