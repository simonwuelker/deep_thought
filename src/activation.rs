use ndarray::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// build a array of array2s from an array of diagonals. Really just a batched version of Array2::from_diag
fn array3_from_diags(diags: &Array2<f64>) -> Array3<f64> {
    let mut result = Array::zeros((diags.nrows(), diags.ncols(), diags.ncols()));

    for (batch_ix, mut elem) in result.axis_iter_mut(Axis(0)).enumerate() {
        elem.diag_mut().assign(&diags.slice(s![batch_ix, ..]));
    }
    result
}

/// Possible activation functions to apply on a Layer's Z value
/// Each Activation function must be continuous and differentiable
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Activation {
    /// values < 0 become 0
    ReLU,
    /// no changes, f(x) = x
    Linear,
    /// squash every input into a range between 0 and 1
    Sigmoid,
    /// Values < 0 get scaled down by a lot. Similar to ReLU except gradients don't become 0. LeakyReLu(0) = ReLU
    LeakyReLU(f64),
    /// Sum of all output values is 1. Useful for getting a probability distribution over the action space
    Softmax,
    /// Sqash every input into a range between -1 and 1
    Tanh,
}

impl Activation {
    /// compute the result of this activation function for a given input (forward propagate)
    pub fn compute(&self, inp: &Array2<f64>) -> Array2<f64> {
        match &self {
            Activation::ReLU => inp.map(|&x| if x > 0. { x } else { 0. }),
            Activation::Linear => inp.clone(),
            Activation::Sigmoid => inp.map(|x| 1. / (1. + (-x).exp())),
            Activation::LeakyReLU(slope) => inp.map(|&x| if x > 0. { x } else { slope * x }),
            Activation::Tanh => inp.map(|&x| ((2. * x).exp() - 1.) / ((2. * x).exp() + 1.)),
            Activation::Softmax => {
                // shift the values by -max(inputs) to prevent overflow (does not affect derivative)
                let max = inp.iter().max_by(|a, b| 
                    if a > b {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }).unwrap();
                let tmp = inp.map(|x| (x - max).exp());
                let sum = tmp.sum();
                tmp / sum
            }
        }
    }

    /// compute the derivative of the activation function for a given input
    /// within a batch, the value v_ji means "how much does a change in the input node i_j affect the output node o_i
    pub fn derivative(&self, inp: &Array2<f64>) -> Array3<f64> {
        match &self {
            Activation::ReLU => array3_from_diags(&inp.map(|&x| if x > 0. { 1. } else { 0. })),
            Activation::Linear => array3_from_diags(&Array2::ones(inp.dim())),
            Activation::Sigmoid => array3_from_diags(&(self.compute(inp) * (Array::<f64, _>::ones(inp.dim()) - self.compute(inp)))),
            Activation::LeakyReLU(slope) => array3_from_diags(&inp.map(|&x| if x > 0. { 1. } else { *slope })),
            Activation::Tanh => array3_from_diags(&(-1. * self.compute(inp) * self.compute(inp) + 1.)),
            Activation::Softmax => {
                let out = self.compute(inp);
                let mut result: Array3<f64> = Array3::zeros((inp.ncols(), inp.nrows(), inp.nrows()));
                // do the computation for every batch seperately
                for (index, mut matrix) in result.axis_iter_mut(Axis(0)).enumerate() {
                    let s = out.slice(s![.., index]).clone().insert_axis(Axis(1));
                    let jacob = Array2::from_diag(&out.slice(s![.., index])) - s.dot(&s.t());
                    matrix.assign(&jacob);
                }
                result
            }
        }
    }
}

impl Default for Activation {
    fn default() -> Activation {
        Activation::Linear
    }
}
