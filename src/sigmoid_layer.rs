use ndarray::Array2;

use crate::base_layer::*;
use crate::box_muller::box_muller_random;
use crate::sigmoid::*;

pub struct SigmoidLayer {
    base: BaseLayer,
}

impl SigmoidLayer {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let weights = Array2::from_shape_fn((n_out, n_in), |_| {
            box_muller_random() * (1.0 / (n_in as f64).sqrt()) // Xavier initialization
        });
        let biases = Array2::from_shape_fn((n_out, 1), |_| box_muller_random());

        SigmoidLayer {
            base: BaseLayer {
                input_size: n_in,
                output_size: n_out,
                weights,
                biases,
            },
        }
    }
}

impl Layer for SigmoidLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        "SigmoidLayer, weight init method = Xavier".to_string()
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::Sigmoid
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        sigmoid(z)
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        sigmoid_prime(z)
    }

    fn forward(&self, input: &Array2<f64>, _is_training: bool) -> ForwardData {
        let base = self.get_base();
        let z = base.weights.dot(input) + &base.biases;
        let activation = self.activate(&z);
        ForwardData {
            z,
            activation,
            cache: None,
        }
    }

    fn backward(
        &self,
        input: &Array2<f64>,
        output_error: &Array2<f64>,
        forward_data: &ForwardData,
    ) -> BackwardData {
        let delta = output_error * self.activate_prime(&forward_data.z);
        let nabla_w = delta.dot(&input.t());

        // Propagated error for the previous layer: W_l^T · δ_l
        let input_gradient = self.get_base().weights.t().dot(&delta);

        BackwardData {
            input_gradient,
            nabla_b: delta,
            nabla_w,
        }
    }
}
