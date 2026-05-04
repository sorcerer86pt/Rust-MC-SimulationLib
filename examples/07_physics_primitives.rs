//! Physics primitives: scatter + angular + spectra in one place.
//!
//! Run with: `cargo run --release --example 07_physics_primitives`

use rust_mc_sim::Pcg64;
use rust_mc_sim::physics::angular::{AngularDistribution, TabularMuDist};
use rust_mc_sim::physics::scatter::{
    Vec3, elastic_scatter, elastic_scatter_aniso, inelastic_scatter,
};
use rust_mc_sim::physics::spectra::{evaporation, maxwellian, watt_u235_thermal};

fn main() {
    let mut rng = Pcg64::new(42, 1);

    // ── Elastic scatter, isotropic CM, on hydrogen ──
    let dir = Vec3::new(0.0, 0.0, 1.0);
    let (e_out, _) = elastic_scatter(2.0e6, dir, 1.0, &mut rng);
    println!("elastic on H-1, E_in = 2 MeV → E_out = {e_out:.3e} eV");

    // ── Anisotropic elastic with a forward-peaked μ distribution ──
    let aniso = AngularDistribution {
        energies: vec![1.0e3, 1.0e7],
        distributions: vec![
            TabularMuDist {
                mu: vec![-1.0, 0.0, 1.0],
                pdf: vec![0.25, 0.5, 0.25],
                cdf: vec![0.0, 0.375, 1.0],
                histogram: false,
            },
            TabularMuDist {
                mu: vec![-1.0, 0.5, 1.0],
                pdf: vec![0.1, 0.3, 0.6],
                cdf: vec![0.0, 0.6, 1.0],
                histogram: false,
            },
        ],
        center_of_mass: true,
    };
    let mut sum_mu = 0.0;
    let n = 50_000;
    for _ in 0..n {
        let (_e, _d) = elastic_scatter_aniso(5.0e6, dir, 12.0, Some(&aniso), 0.0, &mut rng);
        // For a fairer mean check, sample μ_cm directly:
        let mu = aniso.sample_mu(5.0e6, &mut rng);
        sum_mu += mu;
    }
    println!(
        "forward-peaked μ at 5 MeV → mean μ_CM = {:.3} (forward-peaked → > 0)",
        sum_mu / n as f64
    );

    // ── Inelastic two-body, Q = -2 MeV ──
    let (e_in_out, _) = inelastic_scatter(5.0e6, dir, 12.0, -2.0e6, None, &mut rng);
    println!("inelastic on C-12, Q=-2MeV, E_in=5MeV → E_out = {e_in_out:.3e} eV");

    // ── Outgoing-energy spectra ──
    let watt_xs: Vec<f64> = (0..50_000).map(|_| watt_u235_thermal(&mut rng)).collect();
    let watt_mean = watt_xs.iter().sum::<f64>() / watt_xs.len() as f64;
    println!("Watt(U-235 thermal) sample mean: {watt_mean:.3e} eV");

    let max_xs: Vec<f64> = (0..50_000).map(|_| maxwellian(1.5e6, &mut rng)).collect();
    let max_mean = max_xs.iter().sum::<f64>() / max_xs.len() as f64;
    println!("Maxwellian(T=1.5 MeV) sample mean: {max_mean:.3e} eV (3T/2 ≈ 2.25e6)");

    let evap_xs: Vec<f64> = (0..50_000)
        .map(|_| evaporation(0.5e6, 5.0e6, &mut rng))
        .collect();
    let evap_mean = evap_xs.iter().sum::<f64>() / evap_xs.len() as f64;
    println!("evaporation(T=0.5 MeV, capped 5 MeV) sample mean: {evap_mean:.3e} eV");
}
