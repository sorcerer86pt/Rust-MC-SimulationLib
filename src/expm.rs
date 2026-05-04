//! Matrix exponential `exp(A)` for real square matrices via
//! Padé(13) with scaling-and-squaring (Higham, *SIAM J. Matrix
//! Anal. Appl.* 26 (2005) 1179). Robust on the whole spectrum;
//! used as the back-end for both [`crate::kinetics::point`] and
//! [`crate::cram::cram16_dense`].

use faer::Mat;
use faer::prelude::Solve;

/// Compute `exp(A)` for a real square matrix.
pub fn expm_pade(a: &Mat<f64>) -> Mat<f64> {
    assert_eq!(a.nrows(), a.ncols(), "expm requires a square matrix");
    let n = a.nrows();

    let norm = matrix_one_norm(a);

    // Padé(13) order parameter: scale so ‖A/2^s‖₁ ≤ 5.371920351148152.
    let theta_13 = 5.371_920_351_148_152_f64;
    let s = if norm <= theta_13 {
        0
    } else {
        ((norm / theta_13).log2().ceil() as i32).max(0) as u32
    };
    let scale = 2.0_f64.powi(s as i32);

    let mut a_scaled = a.cloned();
    if s > 0 {
        for r in 0..n {
            for c in 0..n {
                a_scaled[(r, c)] /= scale;
            }
        }
    }

    let b: [f64; 14] = [
        64_764_752_532_480_000.0,
        32_382_376_266_240_000.0,
        7_771_770_303_897_600.0,
        1_187_353_796_428_800.0,
        129_060_195_264_000.0,
        10_559_470_521_600.0,
        670_442_572_800.0,
        33_522_128_640.0,
        1_323_241_920.0,
        40_840_800.0,
        960_960.0,
        16_380.0,
        182.0,
        1.0,
    ];

    let i_n = identity(n);
    let a2 = &a_scaled * &a_scaled;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    let mut inner_u = scaled_add(&a6, b[13], &a4, b[11]);
    inner_u = scaled_add(&inner_u, 1.0, &a2, b[9]);
    let inner_u = &a6 * &inner_u;
    let mut u_inner = scaled_add(&inner_u, 1.0, &a6, b[7]);
    u_inner = scaled_add(&u_inner, 1.0, &a4, b[5]);
    u_inner = scaled_add(&u_inner, 1.0, &a2, b[3]);
    u_inner = scaled_add(&u_inner, 1.0, &i_n, b[1]);
    let u = &a_scaled * &u_inner;

    let mut inner_v = scaled_add(&a6, b[12], &a4, b[10]);
    inner_v = scaled_add(&inner_v, 1.0, &a2, b[8]);
    let inner_v = &a6 * &inner_v;
    let mut v = scaled_add(&inner_v, 1.0, &a6, b[6]);
    v = scaled_add(&v, 1.0, &a4, b[4]);
    v = scaled_add(&v, 1.0, &a2, b[2]);
    v = scaled_add(&v, 1.0, &i_n, b[0]);

    let p = scaled_add(&v, 1.0, &u, 1.0);
    let q = scaled_add(&v, 1.0, &u, -1.0);
    let lu = q.partial_piv_lu();
    let mut r = lu.solve(&p);

    for _ in 0..s {
        r = &r * &r;
    }
    r
}

fn matrix_one_norm(a: &Mat<f64>) -> f64 {
    let n = a.ncols();
    let m = a.nrows();
    let mut max_col = 0.0_f64;
    for c in 0..n {
        let mut s = 0.0_f64;
        for r in 0..m {
            s += a[(r, c)].abs();
        }
        if s > max_col {
            max_col = s;
        }
    }
    max_col
}

fn identity(n: usize) -> Mat<f64> {
    let mut i = Mat::<f64>::zeros(n, n);
    for k in 0..n {
        i[(k, k)] = 1.0;
    }
    i
}

fn scaled_add(a: &Mat<f64>, alpha: f64, b: &Mat<f64>, beta: f64) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::<f64>::zeros(m, n);
    for r in 0..m {
        for c in 0..n {
            out[(r, c)] = alpha * a[(r, c)] + beta * b[(r, c)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_decay_matches_exp() {
        let mut a = Mat::<f64>::zeros(2, 2);
        a[(0, 0)] = -1.0;
        a[(1, 1)] = -2.0;
        let e = expm_pade(&a);
        assert!((e[(0, 0)] - (-1.0_f64).exp()).abs() < 1e-12);
        assert!((e[(1, 1)] - (-2.0_f64).exp()).abs() < 1e-12);
        assert!(e[(0, 1)].abs() < 1e-14);
        assert!(e[(1, 0)].abs() < 1e-14);
    }

    #[test]
    fn rotation_pi_over_2() {
        let mut a = Mat::<f64>::zeros(2, 2);
        a[(0, 1)] = std::f64::consts::FRAC_PI_2;
        a[(1, 0)] = -std::f64::consts::FRAC_PI_2;
        let e = expm_pade(&a);
        assert!(e[(0, 0)].abs() < 1e-12);
        assert!((e[(0, 1)] - 1.0).abs() < 1e-12);
        assert!((e[(1, 0)] + 1.0).abs() < 1e-12);
        assert!(e[(1, 1)].abs() < 1e-12);
    }
}
