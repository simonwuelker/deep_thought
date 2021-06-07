use ndarray::prelude::*;

pub enum Loss {
    MSE,
}

impl Loss {
    /// compute the loss for a given output/target pair
    pub fn compute(&self, output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        match &self {
            Loss::MSE => ((output - target) * (output - target)),
        }
    }

    /// compute the derivative of the loss for a given output/target pair
    /// (how sensitive the result of the loss.compute fn is to changes in the output)
    pub fn derivative(&self, output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        match &self {
            Loss::MSE => output - target, // factor 2 is irrelevant because its constant
        }
    }
}
