//! Ducru free-Doppler reconstruction weights (Ducru et al., JCP 335,
//! 2017, Eq. 31). Raw and partition-of-unity variants. Use the
//! 3-point unity form on the nearest three columns for production.

/// Raw Ducru (2017) Eq. 31 weights.
///
/// `column_values` are the training column values (e.g.
/// temperatures); `target` is the column value to reconstruct at.
/// Returns one weight per training column. **Not** normalised.
///
/// If `target` matches a training column to within 0.01 (in the same
/// units as the column values), returns a one-hot vector at that
/// column.
pub fn ducru_weights(column_values: &[f64], target: f64) -> Vec<f64> {
    let n = column_values.len();
    let t = target;

    for (idx, &t_j) in column_values.iter().enumerate() {
        if (t - t_j).abs() < 0.01 {
            let mut w = vec![0.0; n];
            w[idx] = 1.0;
            return w;
        }
    }

    let mut weights = Vec::with_capacity(n);
    for j in 0..n {
        let t_j = column_values[j];
        let leading = (t_j * t).sqrt() / (t_j + t);

        let mut product = 1.0_f64;
        for (i, &t_i) in column_values.iter().enumerate() {
            if i == j {
                continue;
            }
            let num1 = t - t_i;
            let den1 = t + t_i;
            let num2 = t_j + t_i;
            let den2 = t_j - t_i;
            if den2.abs() < 1e-10 {
                continue;
            }
            product *= (num1 / den1) * (num2 / den2);
        }

        weights.push(leading * product);
    }
    weights
}

/// Partition-of-unity normalised Ducru weights: `w ← w / Σ w`.
///
/// Falls back to a uniform `1/N` split if the raw weights sum to zero
/// (degenerate, near-collision configuration).
///
/// In a 3-point setup over the columns nearest to the target, this is
/// the production-default scheme for problems where peak-height
/// preservation matters more than the small global L2 increase
/// introduced by re-normalisation.
pub fn ducru_unity_weights(column_values: &[f64], target: f64) -> Vec<f64> {
    let raw = ducru_weights(column_values, target);
    let s: f64 = raw.iter().sum();
    if s.abs() < 1e-12 {
        return vec![1.0 / column_values.len() as f64; column_values.len()];
    }
    raw.iter().map(|w| w / s).collect()
}

/// Pick the `k` training columns whose values lie closest to
/// `target`, returning their indices in ascending order.
///
/// Typical use: `nearest_k_columns(temps, target, 3)` then pass the
/// 3 chosen columns through [`ducru_unity_weights`].
pub fn nearest_k_columns(column_values: &[f64], target: f64, k: usize) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..column_values.len()).collect();
    idxs.sort_by(|&a, &b| {
        let da = (column_values[a] - target).abs();
        let db = (column_values[b] - target).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    idxs.truncate(k);
    idxs.sort();
    idxs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn raw_weights_are_one_hot_at_exact_match() {
        let temps = vec![300.0, 600.0, 900.0, 1200.0];
        for (i, &t) in temps.iter().enumerate() {
            let w = ducru_weights(&temps, t);
            for (k, &v) in w.iter().enumerate() {
                let want = if k == i { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(v, want, 1e-12),
                    "weight {k} at exact match T={t}: got {v}, want {want}"
                );
            }
        }
    }

    #[test]
    fn unity_weights_sum_to_one() {
        let temps = vec![294.0, 600.0, 900.0, 1200.0, 2500.0];
        for &target in &[450.0, 750.0, 1100.0, 1800.0] {
            let w = ducru_unity_weights(&temps, target);
            let s: f64 = w.iter().sum();
            assert!(
                approx_eq(s, 1.0, 1e-12),
                "Σw at target {target} = {s}, expected 1.0"
            );
        }
    }

    #[test]
    fn unity_weights_at_exact_match_one_hot() {
        let temps = vec![294.0, 600.0, 900.0];
        let w = ducru_unity_weights(&temps, 600.0);
        assert!(approx_eq(w[0], 0.0, 1e-12));
        assert!(approx_eq(w[1], 1.0, 1e-12));
        assert!(approx_eq(w[2], 0.0, 1e-12));
    }

    #[test]
    fn nearest_k_returns_correct_subset() {
        let temps = vec![294.0, 600.0, 900.0, 1200.0, 2500.0];
        let chosen = nearest_k_columns(&temps, 800.0, 3);
        // 800 K → distances {506, 200, 100, 400, 1700}; nearest 3 are
        // indices 1, 2, 3 (600, 900, 1200), in ascending order.
        assert_eq!(chosen, vec![1, 2, 3]);
    }

    #[test]
    fn unity_3point_reproduces_smooth_function_to_high_accuracy() {
        // f(t) = exp(-t/1000) sampled at training cols; check that
        // the unity-normalised 3-point reconstruction matches truth
        // at off-grid targets to better than 1%.
        let temps = vec![300.0, 600.0, 900.0, 1200.0, 1500.0];
        let f = |t: f64| (-t / 1000.0).exp();
        let f_train: Vec<f64> = temps.iter().copied().map(f).collect();
        let mut max_rel = 0.0_f64;
        for &target in &[450.0, 750.0, 1050.0, 1350.0] {
            let chosen = nearest_k_columns(&temps, target, 3);
            let sub: Vec<f64> = chosen.iter().map(|&i| temps[i]).collect();
            let w = ducru_unity_weights(&sub, target);
            let est: f64 = chosen
                .iter()
                .zip(w.iter())
                .map(|(&i, &wj)| wj * f_train[i])
                .sum();
            let truth = f(target);
            let rel = ((est - truth) / truth).abs();
            if rel > max_rel {
                max_rel = rel;
            }
        }
        assert!(
            max_rel < 0.01,
            "max rel error {max_rel} above 1% — Ducru unity should be tighter on smooth functions"
        );
    }
}
