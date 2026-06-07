use ndarray::Array2;

/// Return the maximum value in the input array of f64. Panics if the input array is empty.
/// Warning: if there is a f64::NAN value in the array, the result will be f64::NAN.
pub fn arr_max(a: &Vec<f64>) -> f64 {
    if a.is_empty() {
        panic!("arr_max: input array is empty");
    }
    a.iter().max_by(|x, y| x.total_cmp(y)).unwrap().clone()
}

/// Same as arr_max but for a slice of f64 instead of a Vec. Panics if the input array is empty.
pub fn slice_max(a: &[f64]) -> f64 {
    if a.is_empty() {
        panic!("slice_max: input array is empty");
    }
    a.iter().max_by(|x, y| x.total_cmp(y)).unwrap().clone()
}

/// 'output' is one-hot label, return the index of the maximum value as the predicted label
pub fn get_predicted_label(output: &Array2<f64>) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(index, _)| index)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arr_max() {
        let a = vec![4.0, 2.0, -3.0, 40.0, 0.0];
        assert_eq!(arr_max(&a), 40.0);

        let b = vec![-3.0, -2.0, 1.0, f64::MAX, 0.0, f64::MIN];
        assert_eq!(arr_max(&b), f64::MAX);

        let c = vec![-3.0, -2.0, 1.0, f64::NAN, 0.0];
        assert!(arr_max(&c).is_nan());
    }

    #[test]
    #[should_panic = "arr_max: input array is empty"]
    fn test_arr_max_is_empty() {
        let a = vec![];
        arr_max(&a);
    }

    #[test]
    fn test_slice_max() {
        let a = vec![4.0, 2.0, -3.0, 40.0, 0.0];
        assert_eq!(slice_max(&a[0..]), 40.0);

        let b = vec![-3.0, -2.0, 1.0, f64::MAX, 0.0, f64::MIN];
        assert_eq!(slice_max(&b[0..]), f64::MAX);

        let c = vec![-3.0, -2.0, 1.0, f64::NAN, 0.0];
        assert!(slice_max(&c[0..]).is_nan());
    }

    #[test]
    #[should_panic = "slice_max: input array is empty"]
    fn test_slice_max_is_empty() {
        let a = vec![];
        slice_max(&a[0..]);
    }

    #[test]
    fn test_get_predicted_label() {
        let a = Array2::from_shape_vec(
            (10, 1),
            vec![0.1, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        )
        .unwrap();
        assert_eq!(get_predicted_label(&a), 1);
    }
}
