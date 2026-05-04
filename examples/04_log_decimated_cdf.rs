//! Build a log-decimated CDF and verify the sampled distribution
//! matches the underlying intensities at a fixed point.
//!
//! Run with: `cargo run --release --example 04_log_decimated_cdf`

use tensor_compress::{Pcg64, cdf::LogDecimatedCdf};

fn main() {
    // Three categories with intensities that depend on `x`.
    let xs: Vec<f64> = (0..100).map(|i| 1e-3 * 1.1f64.powi(i)).collect();
    let n_cat = 3;
    let mut intensities = vec![vec![0.0_f64; xs.len()]; n_cat];
    for (j, &x) in xs.iter().enumerate() {
        // Smooth, monotone-different shapes so the CDF is non-trivial.
        intensities[0][j] = 1.0 / (1.0 + x); // decays
        intensities[1][j] = (x.ln() + 5.0).max(0.0); // grows
        intensities[2][j] = 0.5 + (x * 0.3).cos().abs(); // oscillates
    }

    let cdf = LogDecimatedCdf::from_intensities(&intensities, &xs, 200);
    println!(
        "CDF: {} categories × {} log-spaced points = {} bytes",
        cdf.n_categories,
        cdf.n_points,
        cdf.memory_bytes()
    );

    // Verify the empirical sampling distribution at x = 1.0 matches
    // the analytic intensities normalised by their sum. With 100k
    // draws each category should land within ~0.5% of its expected
    // share.
    let target_x = 1.0;
    let mut totals = [0.0_f64; 3];
    let nearest_j = xs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (target_x - **a)
                .abs()
                .partial_cmp(&(target_x - **b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(j, _)| j)
        .unwrap();
    for k in 0..n_cat {
        totals[k] = intensities[k][nearest_j];
    }
    let total: f64 = totals.iter().sum();
    let expected: Vec<f64> = totals.iter().map(|t| t / total).collect();

    let mut counts = [0_u64; 3];
    let n_draws = 100_000;
    let mut rng = Pcg64::new(7, 1);
    for _ in 0..n_draws {
        let xi = rng.uniform();
        let k = cdf.sample(target_x, xi);
        counts[k] += 1;
    }
    let observed: Vec<f64> = counts.iter().map(|c| *c as f64 / n_draws as f64).collect();

    println!();
    println!("at x = {target_x}, expected vs observed sampling fractions:");
    println!("  cat   expected    observed   |Δ|");
    println!("  ---   --------    --------   ----");
    for k in 0..n_cat {
        let dev = (observed[k] - expected[k]).abs();
        println!(
            "  {k:>3}    {:>7.4}     {:>7.4}    {:>5.4}",
            expected[k], observed[k], dev
        );
    }
}
