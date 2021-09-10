use crate::autograd::Dual;
use ndarray::prelude::*;
use num_traits::{Float, Zero, One};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// build a array of array2s from an array of diagonals. Really just a batched version of Array2::from_diag
fn array3_from_diags<F: Float, const N: usize>(
    diags: &Array2<Dual<F, N>>,
) -> Array3<Dual<F, N>> {
    let mut result = Array3::<Dual<F, N>>::zeros((diags.nrows(), diags.ncols(), diags.ncols()));

    for (batch_ix, mut elem) in result.axis_iter_mut(Axis(0)).enumerate() {
        elem.diag_mut().assign(&diags.slice(s![batch_ix, ..]));
    }
    result
}

/// Possible activation functions to apply on a Layer's Z value
/// Each Activation function must be continuous and differentiable
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Activation<F, const N: usize> {
    /// values < 0 become 0
    ReLU,
    /// no changes, f(x) = x
    Linear,
    /// squash every input into a range between 0 and 1
    Sigmoid,
    /// Values < 0 get scaled down by a lot. Similar to ReLU except gradients don't become 0. LeakyReLu(0) = ReLU
    LeakyReLU(Dual<F, N>),
    /// Sum of all output values is 1. Useful for getting a probability distribution over the action space
    Softmax,
    /// Sqash every input into a range between -1 and 1
    Tanh,
}

impl<F: Float, const N: usize> Activation<F, N> {
    /// compute the result of this activation function for a given input (forward propagate)
    pub fn compute(
        &self,
        inp: &Array2<Dual<F, N>>,
    ) -> Array2<Dual<F, N>> {
        match &self {
            Activation::ReLU => inp.map(|&x| {
                if x > Dual::zero() {
                    x
                } else {
                    Dual::zero()
                }
            }),
            Activation::Linear => inp.clone(),
            Activation::Sigmoid => inp.map(|x| Dual::<F, N>::one() / (Dual::<F, N>::one() + (-x).exp())),
            Activation::LeakyReLU(slope) => inp.map(|&x| if x > Dual::zero() { x } else { slope * x }),
            Activation::Tanh => inp.map(|&x| x.tanh()),
            Activation::Softmax => {
                // shift the values by -max(inputs) to prevent overflow (does not affect derivative)
                let max = inp
                    .iter()
                    .max_by(|a, b| {
                        a.partial_cmp(b).unwrap()
                    })
                    .unwrap();
                let tmp = inp.map(|x| (x - max).exp());
                let sum = tmp.sum();
                tmp.map(|x| x / sum)
            }
        }
    }
}

impl<F, const N: usize> Default for Activation<F, N> {
    fn default() -> Activation<F, N> {
        Activation::Linear
    }
}
