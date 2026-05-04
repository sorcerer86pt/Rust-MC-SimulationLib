//! Photon source distributions for fixed-source runs. Mirrors the
//! `transport::fixed_source::NeutronSource` shape so the two
//! transport drivers stay symmetric.

use crate::geometry::Vec3;
use crate::rng::Pcg64;

/// One sampled source photon.
#[derive(Debug, Clone, Copy)]
pub struct SourcePhoton {
    pub pos: Vec3,
    pub dir: Vec3,
    pub energy: f64,
    pub weight: f64,
}

pub trait PhotonSource: Send + Sync {
    fn sample(&self, rng: &mut Pcg64) -> SourcePhoton;
}

/// Monoenergetic isotropic point source at `pos` emitting line `energy_ev`.
/// Cs-137 → 661.7 keV, Co-60 → 1.173 / 1.332 MeV (use one line at a
/// time and weight downstream), Na-22 → 1.275 MeV (+ 511 keV
/// annihilation pair handled implicitly).
pub struct IsotropicLineSource {
    pub pos: Vec3,
    pub energy_ev: f64,
}

impl PhotonSource for IsotropicLineSource {
    fn sample(&self, rng: &mut Pcg64) -> SourcePhoton {
        let (u, v, w) = rng.isotropic_direction();
        SourcePhoton {
            pos: self.pos,
            dir: Vec3::new(u, v, w),
            energy: self.energy_ev,
            weight: 1.0,
        }
    }
}

/// Monoenergetic isotropic source uniformly sampled inside a box.
pub struct MonoBoxSource {
    pub min: Vec3,
    pub max: Vec3,
    pub energy_ev: f64,
}

impl PhotonSource for MonoBoxSource {
    fn sample(&self, rng: &mut Pcg64) -> SourcePhoton {
        let x = self.min.x + rng.uniform() * (self.max.x - self.min.x);
        let y = self.min.y + rng.uniform() * (self.max.y - self.min.y);
        let z = self.min.z + rng.uniform() * (self.max.z - self.min.z);
        let (u, v, w) = rng.isotropic_direction();
        SourcePhoton {
            pos: Vec3::new(x, y, z),
            dir: Vec3::new(u, v, w),
            energy: self.energy_ev,
            weight: 1.0,
        }
    }
}
