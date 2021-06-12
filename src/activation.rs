use ndarray::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Possible activation functions to apply on a Layer's Z value
/// Each Activation function must be continuous and differentiable
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Activation {
    /// values < 0 become 0
    ReLU,
    /// no changes, f(x) = x
    Linear,
    /// squash every input between 0 and 1
    Sigmoid,
    /// Values < 0 get scaled down by a lot. Similar to ReLU except gradients don't become 0. LeakyReLu(0) = ReLU
    LeakyReLU(f64),
    /// Sum of all output values is 1. Useful for getting a probability distribution over the action space
    Softmax,
}

impl Activation {
    /// compute the result of this activation function for a given input (forward propagate)
    pub fn compute(&self, inp: &Array2<f64>) -> Array2<f64> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { x } else { 0. }),
            Activation::Linear => inp.clone(),
            Activation::Sigmoid => inp.map(|x| 1. / (1. + (-x).exp())),
            Activation::LeakyReLU(slope) => inp.map(|&x| if x > 0. { x } else { slope * x }),
            Activation::Softmax => {
                let tmp = inp.map(|x| x.exp());
                let sum = tmp.sum();
                tmp / sum
            }
        }
    }

    /// compute the derivative of the activation function for a given input
    pub fn derivative(&self, inp: &Array2<f64>) -> Array2<f64> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { 1. } else { 0. }),
            Activation::Linear => Array2::ones(inp.dim()),
            Activation::Sigmoid => {
                self.compute(inp) * (Array::<f64, _>::ones(inp.raw_dim()) - self.compute(inp))
            }
            Activation::LeakyReLU(slope) => inp.map(|&x| if x > 0. { 1. } else { *slope }),
            Activation::Softmax => {
                let sum = inp.map(|x| x.exp()).sum();
                unimplemented!();
            }
        }
    }
}

impl Default for Activation {
    fn default() -> Activation {
        Activation::Linear
    }
}
