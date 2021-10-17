use crate::autograd::Dual;
use crate::neural_network::NeuralNetwork;
use num_traits::Float;

/// Implement this for your custom optimizers
pub trait Optimizer<F, const N: usize> {
    /// Create a new instance of the Optimizer
    fn new() -> Self;

    /// Optimizes the provided network's parameters based on their corresponding delta values
    /// (which are already computed at this point)
    fn step(&mut self, net: &mut NeuralNetwork<F, N>, loss: Dual<F, N>);
}
