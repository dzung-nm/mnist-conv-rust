use ndarray::Array2;

/// Returns a new ndarray with the sigmoid function applied element-wise.
/// Sigmoid: σ(x) = 1 / (1 + e^(-x))
pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Returns a new ndarray with the sigmoid prime function applied element-wise.
/// Sigmoid prime: σ'(x) = σ(x) * (1 - σ(x))
pub fn sigmoid_prime(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| {
        let s = 1.0 / (1.0 + (-v).exp());
        s * (1.0 - s)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let input = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 2.0]).unwrap();
        let expected = Array2::from_shape_vec((2, 2), vec![0.5, 0.7310585786300049, 0.2689414213699951, 0.8807970779778823]).unwrap();
        let result = sigmoid(&input);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sigmoid_prime() {
        let input = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 2.0]).unwrap();
        let expected = Array2::from_shape_vec((2, 2), vec![0.25, 0.19661193324148185, 0.19661193324148185, 0.10499358540350662]).unwrap();
        let result = sigmoid_prime(&input);
        assert_eq!(result, expected);
    }
}
