use ndarray::prelude::*;

// TODO: replace this import
use ndarray_rand::rand_distr::num_traits::Pow;

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
    pub fn compute(&self, inp: &Array1<f32>) -> Array1<f32> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { x } else { 0. }),
            Activation::Linear => inp.clone(),
            Activation::Sigmoid => inp.map(|x| 1. / (1. + 2.71.pow(-1. * x) as f32 )),
        }
    }

    /// compute the derivative of the activation function for a given input
    pub fn derivative(&self, inp: &Array1<f32>) -> Array1<f32> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { 1. } else { 0. }),
            Activation::Linear => Array1::ones(inp.raw_dim()),
            Activation::Sigmoid => {
                unimplemented!()
            },
        }
    }
}

impl Default for Activation {
    fn default() -> Activation {
        Activation::Linear
    }
}
