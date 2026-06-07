use ndarray::Array2;

/// Return a new ndarray with the softmax function applied column-wise.
/// Softmax: σ(x_i) = e^(x_i) / sum(e^(x_j)) for each column j
pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let max_per_col = x.map_axis(ndarray::Axis(0), |col| {
        col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });
    let exp_shifted = x - &max_per_col.insert_axis(ndarray::Axis(0));
    let exp_values = exp_shifted.mapv(|v| v.exp());
    let sum_exp = exp_values
        .sum_axis(ndarray::Axis(0))
        .insert_axis(ndarray::Axis(0));
    &exp_values / &sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_softmax() {
        let input = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 1.0]).unwrap();
        let output = softmax(&input);
        assert_eq!(output.shape(), &[4, 1]);
        output.columns().into_iter().for_each(|col| {
            let col_sum: f64 = col.sum();
            assert!((col_sum - 1.0).abs() < 1e-6);
        });
    }
}
