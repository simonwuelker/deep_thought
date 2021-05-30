use ndarray::prelude::*;

pub enum Loss {
    MSE,
}

impl Loss {
    /// compute the loss for a given output/target pair
    pub fn compute(&self, output: Array1<f32>, target: Array1<f32>) -> f32 {
        match &self {
            Loss::MSE => ((&output - &target) * (&output - &target)).mean().unwrap(),
        }
    }

    /// compute the derivative of the loss for a given output/target pair
    /// (how sensitive the result of the loss.compute fn is to changes in the output)
    pub fn derivative(&self, output: Array1<f32>, _target: Array1<f32>) -> Array1<f32> {
        match &self {
            Loss::MSE => 2. * &output * (1. / output.len() as f32 )
        }
    }
}
