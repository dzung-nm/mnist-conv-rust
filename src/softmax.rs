use ndarray::Array2;

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let max_per_row = x.map_axis(ndarray::Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });
    let exp_shifted = x - &max_per_row.insert_axis(ndarray::Axis(1));
    let exp_values = exp_shifted.mapv(|v| v.exp());
    let sum_exp = exp_values
        .sum_axis(ndarray::Axis(1))
        .insert_axis(ndarray::Axis(1));
    &exp_values / &sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_softmax() {
        let input = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 1.0]).unwrap();
        let output = softmax(&input);
        assert_eq!(output.shape(), &[2, 2]);
        output.rows().into_iter().for_each(|row| {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Softmax output does not sum to 1");
        });
    }
}
