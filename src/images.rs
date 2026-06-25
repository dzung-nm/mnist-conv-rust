use image::{GrayImage, Luma, RgbImage, Rgb};
use ndarray::Array2;

/// Save a MNIST image (from memory) to a file
///
/// # Arguments
/// * `image_data` - Array2 of shape (784, 1) containing normalized pixel values [0.0, 1.0]
/// * `filename` - Output filename (e.g., "digit_5.png")
///
/// # Example
/// ```ignore
/// use ndarray::Array2;
/// let image_data = Array2::zeros((784, 1));
/// save_image_mnist(&image_data, "digit_5.png");
/// ```
pub fn save_image_mnist(image: &Array2<f64>, filename: &str) -> std::io::Result<()> {
    // MNIST images are 28x28 pixels
    const WIDTH: u32 = 28;
    const HEIGHT: u32 = 28;

    // Create a new grayscale image
    let mut img = GrayImage::new(WIDTH, HEIGHT);

    // Fill the image with pixel data
    for (idx, &pixel_value) in image.iter().enumerate() {
        let x = (idx % WIDTH as usize) as u32;
        let y = (idx / WIDTH as usize) as u32;

        // Convert from [0.0, 1.0] to [0, 255]
        let pixel_u8 = (pixel_value * 255.0).clamp(0.0, 255.0) as u8;
        img.put_pixel(x, y, Luma([pixel_u8]));
    }

    // Save the image to file
    img.save(filename)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(())
}

/// Save a CIFAR-10 image (from memory) to a file
///
/// # Arguments
/// * `image_data` - Array2 of shape (3072, 1) containing normalized RGB pixel values [0.0, 1.0]
///   Layout: first 1024 bytes are red channel, next 1024 are green, last 1024 are blue
/// * `filename` - Output filename
///
/// # Example
/// ```ignore
/// use ndarray::Array2;
/// let image_data = Array2::zeros((3072, 1));
/// save_image_cifar10(&image_data, "airplane.png");
/// ```
pub fn save_image_cifar10(image: &Array2<f64>, filename: &str) -> std::io::Result<()> {
    // CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    const CHANNEL_SIZE: usize = 1024; // 32 * 32

    // Create a new RGB image
    let mut img = RgbImage::new(WIDTH, HEIGHT);

    // Extract RGB channels from the flattened array
    // CIFAR-10 format: [R R R ... G G G ... B B B ...]
    let pixels: Vec<f64> = image.iter().copied().collect();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;

            // Get RGB values from respective channels
            let r = (pixels[idx] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (pixels[idx + CHANNEL_SIZE] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (pixels[idx + CHANNEL_SIZE * 2] * 255.0).clamp(0.0, 255.0) as u8;

            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // Save the image to file
    img.save(filename)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(())
}
