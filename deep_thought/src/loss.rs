use ndarray::prelude::*;
use crate::autograd::Dual;
use num_traits::Float;

/// A continuous, derivable function to describe how close one value is to another
pub enum Loss {
    /// Mean Squared Error Loss
    MSE,
}

impl Loss {
    /// compute the loss for a given output/target pair
    pub fn compute<F: Float, const N: usize>(&self, output: &Array2<Dual<F, N>>, target: &Array2<F>) -> Array2<Dual<F, N>> {
        match &self {
            Loss::MSE => (output - target) * (output - target),
        }
    }
}
