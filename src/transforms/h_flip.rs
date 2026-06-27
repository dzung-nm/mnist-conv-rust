use ndarray::Array3;

/// Horizontally flip an image
///
/// # Arguments
/// * `image` - Array3 of shape (c, h, w) representing the image to be flipped.
///     The first dimension is the number of channels (e.g., 1 for grayscale, 3 for RGB),
///     the second dimension is the height, and the third dimension is the width.
///
/// # Returns
/// A new Array3 of the same shape as the input, containing the horizontally flipped image.
pub fn h_flip(image: &Array3<f64>) -> Array3<f64> {
    let (c, h, w) = image.dim();
    let mut flipped_image = Array3::<f64>::zeros((c, h, w));

    for channel in 0..c {
        for row in 0..h {
            for col in 0..w {
                flipped_image[[channel, row, col]] = image[[channel, row, w - 1 - col]];
            }
        }
    }

    flipped_image
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_h_flip() {
        let image = array![
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ],
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0]
            ]
        ];
        let result = array![
            [
                [3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0]
            ],
            [
                [9.0, 8.0, 7.0],
                [12.0, 11.0, 10.0]
            ]
        ];

        let h_flip = h_flip(&image);
        assert_eq!(h_flip, result);
    }
}
