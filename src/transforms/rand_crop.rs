use ndarray::{s, Array3};
use rand::RngExt;

pub fn rand_crop(img: &Array3<f64>, pad: usize) -> Array3<f64> {
    let mut rng = rand::rng();
    let (c, h, w) = img.dim();

    // Create a larger, zero-initialized canvas
    let mut padded = Array3::<f64>::zeros((c, h + 2 * pad, w + 2 * pad));

    // Insert the original image into the center of the canvas
    padded.slice_mut(s![.., pad..(pad + h), pad..(pad + w)]).assign(img);

    // Pick a random top-left starting index for the 32x32 window
    let start_h = rng.random_range(0..=(2 * pad));
    let start_w = rng.random_range(0..=(2 * pad));

    // Slice out the new 32x32 view and clone it into an owned Array3
    padded.slice(s![.., start_h..(start_h + h), start_w..(start_w + w)]).to_owned()
}
