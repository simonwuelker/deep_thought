use ndarray::prelude::*;

#[allow(non_upper_case_globals)]
const e: f64 = 2.71; // there is probably some native implementation

// TODO: replace this import
use ndarray_rand::rand_distr::num_traits::Pow;

/// Possible activation functions to apply on a Layer's Z value
pub enum Activation {
    /// values < 0 become 0 
    ReLU,
    /// no changes, f(x) = x
    Linear,
    /// squash every input between 0 and 1
    Sigmoid,
}

impl Activation {
    /// compute the result of this activation function for a given input (forward propagate)
    pub fn compute(&self, inp: &Array1<f64>) -> Array1<f64> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { x } else { 0. }),
            Activation::Linear => inp.clone(),
            Activation::Sigmoid => inp.map(|x| 1. / (1. + e.pow(-1. * x) as f64 )),
        }
    }

    /// compute the derivative of the activation function for a given input
    pub fn derivative(&self, inp: &Array1<f64>) -> Array1<f64> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { 1. } else { 0. }),
            Activation::Linear => Array1::ones(inp.raw_dim()),
            Activation::Sigmoid => inp.map(|&x| (e.pow(-1. * x)) / ((1. + e.pow(-1. * x)).pow(2)))
        }
    }
}

impl Default for Activation {
    fn default() -> Activation {
        Activation::Linear
    }
}
