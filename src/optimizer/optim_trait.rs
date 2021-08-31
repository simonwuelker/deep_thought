use crate::autograd::Dual;
use crate::neural_network::NeuralNetwork;
use num_traits::Float;
use std::fmt;

/// Implement this for your custom optimizers
pub trait Optimizer<F: Float + fmt::Debug, const N: usize> {
    /// Create a new Optimizer instance based on the networks shape. Some Optimizers require knowledge about hidden layer sizes
    /// which is why you need to pass a reference to the network here
    fn new(net: &NeuralNetwork<F, N>) -> Self;

    /// Optimizes the provided network's parameters based on their corresponding delta values
    /// (which are already computed at this point)
    fn step(&mut self, net: &mut NeuralNetwork<F, N>, loss: Dual<F, N>);
}
