use crate::{
    activation::Activation, autograd::Dual, error::Error, loss::Loss, optimizer::Optimizer,
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use num_traits::{Float, Zero};
use rand_distr::{Distribution, Normal};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A Neural Network consisting of a an input/output and any number of additional hidden [`Layer`]s
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuralNetwork<F: Float + fmt::Debug, const N: usize> {
    pub layers: Vec<Layer<F, N>>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(non_snake_case)] // non snake case kinda makes sense with matrices
/// A single neuron layer with an associated [`Activation`] function
pub struct Layer<F: Float + fmt::Debug, const N: usize> {
    /// Weight matrix
    pub W: Array2<Dual<F, N>>,
    /// Bias vector
    pub B: Array2<Dual<F, N>>,
    /// Activation function to allow for nonlinear transformations
    activation: Activation,
}

impl<F, const N: usize> Layer<F, N>
where
    F: Float + fmt::Debug,
    f64: Into<F>,
{
    /// Construct a new layer with provided dimensions. Weights are initialized using [Glorot/Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a.html)
    /// Biases are always initialized to zeros.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let std: Dual<F, N> = (2. / (input_dim + output_dim) as f64).sqrt().into();
        Self {
            W: Array2::<Dual<F, N>>::random(
                (output_dim, input_dim),
                Normal::new(Dual::<F, N>::zero(), std).unwrap(),
            ),
            B: Array2::<Dual<F, N>>::zeros((output_dim, 1)),
            activation: Activation::default(),
        }
    }

    /// define a activation function for that layer (default is f(x) = x )
    pub fn activation(mut self, a: Activation) -> Self {
        self.activation = a;
        self
    }
    /// forward-pass a batch of input vectors through the layer
    pub fn forward(&mut self, inp: &Array2<Dual<F, N>>) -> Array2<Dual<F, N>> {
        let z = self.W.dot(inp) + self.B;
        self.activation.compute(&z)
    }
}

impl<F, const N: usize> NeuralNetwork<F, N>
where
    F: Float + fmt::Debug,
    f64: Into<F>,
{
    /// Initialize a empty Neural Network
    pub fn new() -> NeuralNetwork<F, N> {
        NeuralNetwork { layers: vec![] }
    }

    /// Get the number of tunable parameters inside the network
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.W.len() + layer.B.len())
            .sum()
    }

    /// add a hidden layer to the network
    pub fn add_layer(mut self, layer: Layer<F, N>) -> NeuralNetwork<F, N> {
        self.layers.push(layer);
        self
    }

    /// forward-pass a batch of input vectors through the network
    pub fn forward(&mut self, inp: &Array2<Dual<F, N>>) -> Array2<Dual<F, N>> {
        let mut input = inp.to_owned();
        for index in 0..self.layers.len() {
            input = self.layers[index].forward(&input);
        }
        input.to_owned()
    }
}
