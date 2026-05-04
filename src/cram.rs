//! Matrix-exponential propagator for Bateman equations.
//!
//! The function name `cram16_dense` is preserved for API stability
//! with the original CRAM-16 sketch in this crate, but the current
//! implementation is **dense Padé(13) with scaling-and-squaring**
//! via [`crate::expm::expm_pade`]. The CRAM-16 partial-fraction
//! poles and residues are kept as public constants for callers who
//! want to plug in a validated CRAM kernel themselves; the published
//! coefficient sets in this crate (Pusa & Leppänen 2010, *NSE* 164;
//! Pusa 2013, arXiv:1206.2880) shipped with two competing
//! conventions (additive partial fraction vs OpenMC's incomplete
//! partial fraction) and reconciling them against a Serpent or
//! ORIGEN reference is out of scope here.
//!
//! Padé(13) is robust on the full spectrum and adequate for the
//! depletion problem sizes this crate was designed for (up to a
//! few thousand nuclides). For the 5 000+ nuclide chains where
//! CRAM's sparsity advantage matters, swap this function for a
//! validated CRAM-48 implementation.

use faer::Mat;

/// CRAM-16 pole/residue table from Pusa, *Ann. Nucl. Energy* 38
/// (2011) 1657, Table III. Real and imaginary parts of the eight
/// (θ_k, α_k) pairs in the lower half-plane (the upper half-plane
/// pairs are complex conjugates).
pub const CRAM16_THETA_RE: [f64; 8] = [
    -1.084_391_707_834_4e+01,
    -5.264_971_343_442_4e+00,
    5.948_152_268_951_177e+00,
    3.509_103_608_414_918e+00,
    6.416_177_699_099_435e+00,
    1.419_375_897_185_666e+00,
    4.993_174_737_717_997e+00,
    -1.413_036_697_886_109e+00,
];
pub const CRAM16_THETA_IM: [f64; 8] = [
    1.927_744_616_792_731_8e+01,
    1.622_022_147_316_792_8e+01,
    3.587_457_362_018_322_4e+00,
    8.436_198_985_884_374e+00,
    1.194_122_393_370_990_4e+01,
    1.092_536_348_449_672e+01,
    5.996_881_713_603_942e+00,
    1.369_633_186_206_625_3e+01,
];
pub const CRAM16_ALPHA_RE: [f64; 8] = [
    -5.090_152_186_522_492_2e-07,
    2.115_174_218_246_607e-04,
    1.133_977_517_848_393e+02,
    1.505_958_527_002_581_5e+01,
    -6.450_087_802_553_964e+01,
    -1.479_300_711_355_799_8e+00,
    -6.251_839_587_481_6e+01,
    4.102_313_683_541_12e-02,
];
pub const CRAM16_ALPHA_IM: [f64; 8] = [
    -2.422_001_765_285_228e-05,
    4.389_296_964_738_067_5e-03,
    1.019_472_170_421_585_5e+02,
    -5.751_405_277_642_215e+00,
    -2.245_944_076_265_209e+02,
    1.768_658_832_175_792_2e+00,
    -2.530_585_697_955_287_3e+01,
    -1.574_346_617_345_546_2e-01,
];

/// `α₀` constant from the CRAM-16 partial-fraction expansion.
pub const CRAM16_ALPHA0: f64 = 2.124_853_710_495_224e-16;

/// Compute `exp(A · t) · n0` for a dense real square matrix `A` via
/// CRAM-16. Returns the new column vector `n(t) = exp(A t) · n0`.
///
/// `A` is `n × n` row-major; `n0` is length `n`; the output is
/// length `n`.
///
/// **Caveats** (v0.1):
/// * Dense direct solver via faer — fine up to a few thousand
///   nuclides; production depletion libraries (5 000+ nuclides)
///   need sparse + iterative.
/// * No special handling for zero rows/columns (decay-only species)
///   — the polynomial system handles them correctly but at full
///   `n²` cost.
/// * Validate against your reference (e.g. ORIGEN-S, Serpent's
///   built-in CRAM-48) before relying on inventories that drive
///   reactivity.
pub fn cram16_dense(a_row_major: &[f64], n0: &[f64], t: f64, n: usize) -> Vec<f64> {
    assert_eq!(a_row_major.len(), n * n);
    assert_eq!(n0.len(), n);
    if t == 0.0 {
        return n0.to_vec();
    }
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = a_row_major[i * n + j] * t;
        }
    }
    let exp_a = crate::expm::expm_pade(&a);
    let mut out = vec![0.0_f64; n];
    for r in 0..n {
        let mut acc = 0.0_f64;
        for c in 0..n {
            acc += exp_a[(r, c)] * n0[c];
        }
        out[r] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * a.abs().max(b.abs()).max(1e-30)
    }

    #[test]
    fn diagonal_decay() {
        // Single isotope decay: A = -λ, n(t) = n0 · exp(-λ t).
        let lambda = 1.0;
        let n0 = vec![1.0_f64];
        let a = vec![-lambda];
        for t in [0.0, 0.5, 1.0, 5.0] {
            let n_t = cram16_dense(&a, &n0, t, 1);
            let want = (-lambda * t).exp();
            assert!(
                approx_eq(n_t[0], want, 1e-10),
                "exp(-{lambda}·{t}): want {want}, got {}",
                n_t[0]
            );
        }
    }

    #[test]
    fn two_isotope_chain() {
        // A → B with rate λ. dA/dt = -λ A, dB/dt = +λ A.
        // Closed form: A(t) = A0 e^{-λt}, B(t) = A0 (1 - e^{-λt}).
        let lambda = 0.5_f64;
        let a = vec![-lambda, 0.0, lambda, 0.0];
        let n0 = vec![1.0, 0.0];
        for t in [0.5_f64, 1.0_f64, 2.0_f64, 5.0_f64] {
            let n_t = cram16_dense(&a, &n0, t, 2);
            let want_a = (-lambda * t).exp();
            let want_b = 1.0 - want_a;
            assert!(approx_eq(n_t[0], want_a, 1e-9));
            assert!(approx_eq(n_t[1], want_b, 1e-9));
        }
    }

    #[test]
    fn zero_time_is_identity() {
        let a = vec![-2.0, 1.0, 0.5, -3.0];
        let n0 = vec![7.0, 11.0];
        let n_t = cram16_dense(&a, &n0, 0.0, 2);
        for (a, b) in n_t.iter().zip(n0.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
