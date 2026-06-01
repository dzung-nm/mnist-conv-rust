use std::fs;
use std::io;
use std::path::Path;

pub fn unzip(zip_path_str: &str) -> Result<(), Box<dyn std::error::Error>> {
    let zip_path = Path::new(zip_path_str);
    let output_dir = zip_path.parent().unwrap_or(Path::new("."));

    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    println!("Scanning archive: {}", zip_path_str);

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name();

        // Skip directory entries if any accidentally exist
        if name.ends_with('/') {
            continue;
        }

        // Skip macOS metadata junk files and hidden system files
        if name.starts_with("__MACOSX/") || name.starts_with(".") || name.contains("/.") {
            continue;
        }

        // Clean and build the output path
        let enclosed_name = file.enclosed_name().ok_or("Invalid file path in zip")?;
        let out_path = output_dir.join(enclosed_name);

        // skip extraction if the target file already exists
        if out_path.exists() {
            println!("Skipping: {} (Already exists)", out_path.display());
            continue;
        }

        let mut out_file = fs::File::create(&out_path)?;
        io::copy(&mut file, &mut out_file)?;
        println!("Extracted: {}", out_path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unzip() {
        unzip("data/mnist.zip").expect("Failed to unzip mnist.zip");
    }
}
