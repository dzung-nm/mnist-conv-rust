use std::f64::consts::PI;
use rand::RngExt;
use rand::distr::Open01;

// Box-Muller transform to generate a random number from a standard normal distribution
pub fn box_muller_random() -> f64 {
    let u: f64 = rand::rng().sample(Open01);
    let v: f64 = rand::rng().sample(Open01);
    (-2.0 * u.ln()).sqrt() * (v * 2.0 * PI).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_muller_random() {
        let mut x = Vec::new();
        for _ in 0..10 {
            x.push(box_muller_random());
        }
        println!("10 random samples from standard normal distribution {:?}", x);
    }
}
