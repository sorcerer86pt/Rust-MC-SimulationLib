//! PCG-64 pseudo-random generator.
//!
//! Used by the [`crate::nuclear::thermal`] sampling routines and
//! exposed for callers that want a cheap, reproducible, parallel-safe
//! RNG with deterministic skip-ahead semantics.
//!
//! Imported from `open_rust_mc/src/transport/rng.rs` unchanged
//! (PCG-XSH-RR 64/32 stream variant of O'Neill 2014).

/// PCG-XSH-RR 64/32 generator.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u64,
    inc: u64,
}

impl Pcg64 {
    /// Build a new RNG with the given seed and stream identifier.
    /// Different `stream` values yield uncorrelated sequences from the
    /// same `seed` — useful for per-particle / per-thread parallelism.
    pub fn new(seed: u64, stream: u64) -> Self {
        let inc = (stream << 1) | 1;
        let mut rng = Self { state: 0, inc };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Deterministic per-particle seed: same `(batch, particle_id)`
    /// always produces the same stream.
    pub fn for_particle(batch: u64, particle_id: u64) -> Self {
        let seed = batch
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(particle_id);
        Self::new(seed, particle_id)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform `f64 ∈ [0, 1)` with full mantissa precision.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        (a * 67_108_864 + b) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `-ln(ξ) / rate`, the standard exponential sample.
    #[inline]
    pub fn exponential(&mut self, rate: f64) -> f64 {
        -self.uniform().ln() / rate
    }

    /// Uniform direction on the unit sphere as `(x, y, z)`.
    #[inline]
    pub fn isotropic_direction(&mut self) -> (f64, f64, f64) {
        let mu = 2.0 * self.uniform() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * self.uniform();
        let sin_theta = (1.0 - mu * mu).sqrt();
        (sin_theta * phi.cos(), sin_theta * phi.sin(), mu)
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn stream(&self) -> u64 {
        self.inc >> 1
    }

    /// Reconstruct from saved `(state, stream)`.
    pub fn from_state(state: u64, stream: u64) -> Self {
        Self {
            state,
            inc: (stream << 1) | 1,
        }
    }

    /// Discrete inverse-CDF sampling: weights are non-negative and sum to `total`.
    #[inline]
    pub fn discrete(&mut self, weights: &[f64], total: f64) -> usize {
        let xi = self.uniform() * total;
        let mut cumulative = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if xi < cumulative {
                return i;
            }
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_in_range() {
        let mut rng = Pcg64::new(42, 1);
        for _ in 0..10_000 {
            let x = rng.uniform();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn deterministic_per_particle() {
        let mut a = Pcg64::for_particle(1, 100);
        let mut b = Pcg64::for_particle(1, 100);
        for _ in 0..100 {
            assert_eq!(a.uniform().to_bits(), b.uniform().to_bits());
        }
    }

    #[test]
    fn isotropic_direction_unit_norm() {
        let mut rng = Pcg64::new(42, 1);
        for _ in 0..1000 {
            let (u, v, w) = rng.isotropic_direction();
            let len = (u * u + v * v + w * w).sqrt();
            assert!((len - 1.0).abs() < 1e-10);
        }
    }
}
