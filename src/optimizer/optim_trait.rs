use crate::neural_network::NeuralNetwork;

/// Implement this for your custom optimizers
pub trait Optimizer {
    /// Create a new Optimizer instance based on the networks shape. Some Optimizers require knowledge about hidden layer sizes
    /// which is why you need to pass a reference to the network here
    fn new(net: &NeuralNetwork) -> Self;

    /// Optimizes the provided network's parameters based on their corresponding delta values
    /// (which are already computed at this point)
    fn step(&mut self, net: &mut NeuralNetwork);
}
