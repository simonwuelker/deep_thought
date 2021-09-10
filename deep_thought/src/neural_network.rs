use crate::{
    activation::Activation,
    autograd::{Dual, DualDistribution},
    error::Error,
    loss::Loss,
    optimizer::Optimizer,
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use num_traits::{Num, Float};
use rand_distr::{StandardNormal, Normal, Distribution};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A Neural Network consisting of a an input/output and any number of additional hidden [`Layer`]s
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuralNetwork<F: Num + Copy, const N: usize> {
    pub layers: Vec<Layer<F, N>>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(non_snake_case)] // non snake case kinda makes sense with matrices
/// A single neuron layer with an associated [`Activation`] function
pub struct Layer<F: Num + Copy, const N: usize> {
    /// Weight matrix
    pub W: Array2<Dual<F, N>>,
    /// Bias vector
    pub B: Array2<Dual<F, N>>,
    /// Activation function to allow for nonlinear transformations
    activation: Activation<F, N>,
}

impl<F: Float, const N: usize> Layer<F, N> 
where StandardNormal: Distribution<F> {
    /// Construct a new layer with provided dimensions. Weights are initialized using [Glorot/Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a.html)
    /// Biases are always initialized to zeros.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let std = (2. / (input_dim + output_dim) as f64).sqrt();
        let dist = DualDistribution::new(Normal::new(F::zero(), F::from(std).unwrap()).unwrap());
        Self {
            W: Array2::<Dual<F, N>>::random((output_dim, input_dim), dist),
            B: Array2::<Dual<F, N>>::zeros((output_dim, 1)),
            activation: Activation::default(),
        }
    }
}

impl<F: 'static + Float, const N: usize> Layer<F, N> {
    /// define a activation function for that layer (default is f(x) = x )
    pub fn activation(mut self, a: Activation<F, N>) -> Self {
        self.activation = a;
        self
    }
    /// forward-pass a batch of input vectors through the layer
    pub fn forward(&mut self, inp: &Array2<Dual<F, N>>) -> Array2<Dual<F, N>> {
        let z = self.W.dot(inp) + &self.B;
        self.activation.compute(&z)
    }
}

impl<F: 'static + Float, const N: usize> NeuralNetwork<F, N> {
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
