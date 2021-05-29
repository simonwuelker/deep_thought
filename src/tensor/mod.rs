pub mod gradient;
use gradient::*;

/// A tensor is a matrix that keeps track of its gradient (the mathematical 
/// Operations being performed on it). It also keeps mutable references to
/// every object that influenced it's final value, meaning it can be used 
/// for backpropagation
pub struct Tensor<'a> {
    data: Vec<Vec<f32>>,
    gradient: Option<Operation<'a>>,
}

impl Tensor<'_> {
    pub fn backprop(&self) {
        if let Some(gradient) = &self.gradient {
            gradient.backprop(0.0);
        }
    }

    /// delete all the history from that tensor
    pub fn zero_grad(&mut self) {
        self.gradient = None;
    }
}
