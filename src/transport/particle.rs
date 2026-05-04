//! Particle state + fission bank.

use crate::geometry::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleStatus {
    Alive,
    Dead,
}

/// A single neutron being transported.
#[derive(Debug, Clone)]
pub struct Particle {
    pub pos: Vec3,
    pub dir: Vec3,
    /// Energy (eV).
    pub energy: f64,
    pub weight: f64,
    pub cell_idx: usize,
    pub status: ParticleStatus,
    pub n_collisions: u32,
}

impl Particle {
    pub fn new(pos: Vec3, dir: Vec3, energy: f64, cell_idx: usize) -> Self {
        Self {
            pos,
            dir,
            energy,
            weight: 1.0,
            cell_idx,
            status: ParticleStatus::Alive,
            n_collisions: 0,
        }
    }

    #[inline]
    pub fn advance(&mut self, distance: f64) {
        self.pos = self.pos + self.dir * distance;
    }

    #[inline]
    pub fn kill(&mut self) {
        self.status = ParticleStatus::Dead;
    }

    #[inline]
    pub fn is_alive(&self) -> bool {
        self.status == ParticleStatus::Alive
    }
}

/// One fission site for the next generation's source bank.
#[derive(Debug, Clone)]
pub struct FissionSite {
    pub pos: Vec3,
    pub energy: f64,
    pub weight: f64,
}

#[derive(Debug, Default)]
pub struct FissionBank {
    pub sites: Vec<FissionSite>,
}

impl FissionBank {
    pub fn new() -> Self {
        Self { sites: Vec::new() }
    }

    pub fn push(&mut self, site: FissionSite) {
        self.sites.push(site);
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    pub fn clear(&mut self) {
        self.sites.clear();
    }
}
