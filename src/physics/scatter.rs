//! Elastic + inelastic two-body kinematics. Energies in eV,
//! directions are unit [`Vec3`]. `elastic_scatter_aniso` switches
//! to the free-gas branch below `400·kT`.

use crate::physics::angular::AngularDistribution;
use crate::rng::Pcg64;

/// Boltzmann constant in eV/K (OpenMC value).
pub const K_BOLTZMANN: f64 = 8.617_333e-5;

/// Three-component direction / velocity vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l > 0.0 {
            Self::new(self.x / l, self.y / l, self.z / l)
        } else {
            self
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Cold-target elastic scatter, isotropic in CM. Standard textbook
/// formula. Returns `(new_energy_eV, new_direction)`.
pub fn elastic_scatter(energy: f64, dir: Vec3, awr: f64, rng: &mut Pcg64) -> (f64, Vec3) {
    let mu_cm = 2.0 * rng.uniform() - 1.0;
    let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
    let new_energy = energy * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm);
    let mu_lab = if awr > 1.0 + 1e-10 {
        (1.0 + awr * mu_cm) / (1.0 + 2.0 * awr * mu_cm + awr * awr).sqrt()
    } else {
        // Hydrogen special case
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };
    let new_dir = rotate_direction(dir, mu_lab, rng);
    (new_energy.max(1e-11), new_dir)
}

/// Production elastic scatter with optional anisotropic μ from a
/// tabulated angular distribution and free-gas thermal correction
/// for `E < 400·kT`.
///
/// `temperature` in K; pass `0.0` to disable the free-gas branch.
pub fn elastic_scatter_aniso(
    energy: f64,
    dir: Vec3,
    awr: f64,
    angle_dist: Option<&AngularDistribution>,
    temperature: f64,
    rng: &mut Pcg64,
) -> (f64, Vec3) {
    let kt = K_BOLTZMANN * temperature;
    if temperature > 0.0 && energy < 400.0 * kt {
        return free_gas_scatter(energy, dir, awr, kt, angle_dist, rng);
    }
    let mu_cm = match angle_dist {
        Some(dist) if dist.center_of_mass => dist.sample_mu(energy, rng),
        _ => 2.0 * rng.uniform() - 1.0,
    };
    let alpha = ((awr - 1.0) / (awr + 1.0)).powi(2);
    let new_energy = energy * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm);
    let mu_lab = if awr > 1.0 + 1e-10 {
        (1.0 + awr * mu_cm) / (1.0 + 2.0 * awr * mu_cm + awr * awr).sqrt()
    } else {
        ((1.0 + mu_cm) * 0.5).max(0.0).sqrt()
    };
    let new_dir = rotate_direction(dir, mu_lab, rng);
    (new_energy.max(1e-11), new_dir)
}

/// Free-gas thermal scattering: target nucleus has thermal motion.
/// Implements the standard four-step procedure: sample target
/// velocity from Maxwell-Boltzmann, compute relative velocity,
/// scatter in the relative-motion CM frame, transform back to lab.
pub fn free_gas_scatter(
    energy: f64,
    dir: Vec3,
    awr: f64,
    kt: f64,
    angle_dist: Option<&AngularDistribution>,
    rng: &mut Pcg64,
) -> (f64, Vec3) {
    let v_n = (2.0 * energy).sqrt();
    let sigma = (kt / awr).sqrt();
    // Maxwell-Boltzmann target velocity via three independent
    // Box-Muller normals.
    let normal = |rng: &mut Pcg64| -> f64 {
        sigma
            * (-2.0 * rng.uniform().ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.uniform()).cos()
    };
    let v_target = Vec3::new(normal(rng), normal(rng), normal(rng));
    let v_neutron = dir * v_n;
    let v_rel = v_neutron - v_target;
    let v_rel_mag = v_rel.length();
    if v_rel_mag < 1e-20 {
        return (energy, dir);
    }
    let mu_reduced = awr / (awr + 1.0);
    let e_rel = 0.5 * mu_reduced * v_rel_mag * v_rel_mag;
    let mu_cm = match angle_dist {
        Some(dist) if dist.center_of_mass => dist.sample_mu(e_rel, rng),
        _ => 2.0 * rng.uniform() - 1.0,
    };
    let v_cm = (v_neutron + v_target * awr) * (1.0 / (1.0 + awr));
    let v_n_cm_dir = v_rel.normalized();
    let new_v_n_cm_dir = rotate_direction(v_n_cm_dir, mu_cm, rng);
    let v_n_cm_after = new_v_n_cm_dir * (v_rel_mag * awr / (1.0 + awr));
    let v_n_lab = v_n_cm_after + v_cm;
    let v_n_lab_mag = v_n_lab.length();
    if v_n_lab_mag < 1e-20 {
        return (1e-11, dir);
    }
    let new_energy = 0.5 * v_n_lab_mag * v_n_lab_mag;
    let new_dir = v_n_lab * (1.0 / v_n_lab_mag);
    (new_energy.max(1e-11), new_dir)
}

/// Two-body inelastic kinematics with discrete-level Q (eV).
///
/// `q_value` is signed: negative for endothermic level excitation
/// (the typical case for inelastic-to-bound-state). Below threshold
/// `(|Q|·(A+1)/A)` falls back to elastic scattering.
pub fn inelastic_scatter(
    energy: f64,
    dir: Vec3,
    awr: f64,
    q_value: f64,
    angle_dist: Option<&AngularDistribution>,
    rng: &mut Pcg64,
) -> (f64, Vec3) {
    let threshold = if q_value < 0.0 {
        (-q_value) * (awr + 1.0) / awr
    } else {
        0.0
    };
    if energy < threshold {
        return elastic_scatter(energy, dir, awr, rng);
    }
    let e_cm = energy * awr / (awr + 1.0);
    let e_cm_out = e_cm + q_value;
    if e_cm_out <= 0.0 {
        return elastic_scatter(energy, dir, awr, rng);
    }
    let mu_cm = match angle_dist {
        Some(dist) => dist.sample_mu(energy, rng),
        None => 2.0 * rng.uniform() - 1.0,
    };
    let a_plus_1 = awr + 1.0;
    let e_neutron_cm = e_cm_out * awr / a_plus_1;
    let v_n = (2.0 * e_neutron_cm).sqrt();
    let v_cm_sys = (2.0 * energy / (a_plus_1 * a_plus_1)).sqrt();
    let e_lab_out = 0.5 * (v_n * v_n + v_cm_sys * v_cm_sys + 2.0 * v_n * v_cm_sys * mu_cm);
    let e_lab_out = e_lab_out.max(1e-5);
    let mu_lab = if v_n + v_cm_sys > 1e-20 {
        (v_cm_sys + v_n * mu_cm)
            / (v_n * v_n + v_cm_sys * v_cm_sys + 2.0 * v_n * v_cm_sys * mu_cm).sqrt()
    } else {
        2.0 * rng.uniform() - 1.0
    };
    let mu_lab = mu_lab.clamp(-1.0, 1.0);
    let new_dir = rotate_direction(dir, mu_lab, rng);
    (e_lab_out, new_dir)
}

/// Rotate `dir` by a polar angle (given as its cosine `mu`) and a
/// uniform-random azimuth. Uses the standard rotation formula with a
/// special case for directions nearly along the z-axis.
pub fn rotate_direction(dir: Vec3, mu: f64, rng: &mut Pcg64) -> Vec3 {
    let phi = 2.0 * std::f64::consts::PI * rng.uniform();
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    let (u, v, w) = (dir.x, dir.y, dir.z);
    if w.abs() > 0.999_999 {
        let sign = w.signum();
        return Vec3::new(sin_theta * cos_phi, sign * sin_theta * sin_phi, sign * mu);
    }
    let inv_sqrt = 1.0 / (1.0 - w * w).sqrt();
    Vec3::new(
        mu * u + sin_theta * (u * w * cos_phi - v * sin_phi) * inv_sqrt,
        mu * v + sin_theta * (v * w * cos_phi + u * sin_phi) * inv_sqrt,
        mu * w - sin_theta * cos_phi * (1.0 - w * w) * inv_sqrt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_energy_bounded() {
        let mut rng = Pcg64::new(42, 1);
        for _ in 0..10_000 {
            let (e_new, _) = elastic_scatter(1.0e6, Vec3::new(0.0, 0.0, 1.0), 1.0, &mut rng);
            assert!(e_new >= 0.0);
            assert!(e_new <= 1.0e6 * 1.0001);
        }
    }

    #[test]
    fn elastic_heavy_nucleus_small_loss() {
        let mut rng = Pcg64::new(42, 1);
        let e0 = 1.0e6;
        let awr = 238.0;
        let mut total = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let (e_new, _) = elastic_scatter(e0, Vec3::new(0.0, 0.0, 1.0), awr, &mut rng);
            total += e_new / e0;
        }
        let avg = total / n as f64;
        assert!(avg > 0.99, "avg ratio = {avg}");
    }

    #[test]
    fn elastic_direction_stays_unit() {
        let mut rng = Pcg64::new(42, 1);
        for _ in 0..1000 {
            let d = Vec3::new(0.5, 0.5, 1.0 / 2.0_f64.sqrt()).normalized();
            let (_, new_dir) = elastic_scatter(1.0e6, d, 12.0, &mut rng);
            assert!((new_dir.length() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn inelastic_below_threshold_falls_back_to_elastic() {
        // Q = -2 MeV → threshold = 2 * (12+1)/12 ≈ 2.17 MeV. At 1 MeV
        // we should fall back to elastic.
        let mut rng = Pcg64::new(7, 1);
        let (e_new, _) = inelastic_scatter(
            1.0e6,
            Vec3::new(0.0, 0.0, 1.0),
            12.0,
            -2.0e6,
            None,
            &mut rng,
        );
        // Elastic on C-12 keeps most of the energy.
        assert!(e_new > 0.5e6);
    }

    #[test]
    fn rotate_direction_unit_norm() {
        let mut rng = Pcg64::new(42, 1);
        for _ in 0..1000 {
            let d = Vec3::new(1.0, 0.0, 0.0);
            let new_d = rotate_direction(d, rng.uniform() * 2.0 - 1.0, &mut rng);
            assert!((new_d.length() - 1.0).abs() < 1e-9);
        }
    }
}
