use crate::{activation::Activation, error::Error, loss::Loss, optimizer::Optimizer};
use anyhow::Result;
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Normal};
// use rand_distr::{Normal, Distribution};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A Neural Network consisting of a an input/output and any number of additional hidden [`Layer`]s
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BuiltNeuralNetwork {
    pub layers: Vec<BuiltLayer>,
}

pub struct Layer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub activation: Activation,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(non_snake_case)] // non snake case kinda makes sense with matrices
/// A single neuron layer with an associated [`Activation`] function
pub struct BuiltLayer {
    /// Weight matrix
    pub W: Array2<f64>,
    /// Bias vector
    pub B: Array2<f64>,
    /// Activation function which turns self.Z into self.A
    activation: Activation,
}

impl Layer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim: input_dim,
            output_dim: output_dim,
            activation: Activation::default(),
        }
    }

    /// define a activation function for that layer (default is f(x) = x )
    pub fn activation(mut self, a: Activation) -> Self {
        self.activation = a;
        self
    }
}

impl BuiltLayer {
    /// forward-pass a batch of input vectors through the layer
    pub fn forward(&mut self, inp: &Array2<f64>) -> Array2<f64>{
        let z = self.W.dot(inp) + &self.B;
        self.activation.compute(&z)
    }
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork { layers: vec![] }
    }

    /// Get the number of tunable parameters inside the network
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        
        for layer in &self.layers {
            total += layer.input_dim * layer.output_dim + layer.output_dim;
        }
        total
    }

    /// add a hidden layer to the network
    pub fn add_layer(mut self, layer: Layer) -> NeuralNetwork {
        self.layers.push(layer);
        self
    }

    /// construct a new layer with provided dimensions. Weights are initialized using [Glorot/Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a.html)
    /// Biases are always intialized to zeros
    pub fn build(&self) -> BuiltNeuralNetwork {
        let mut layers = vec![];
        for layer in &self.layers {
            let std = (2. / (layer.input_dim + layer.output_dim) as f64).sqrt();
            layers.push(BuiltLayer {
                W: Array::random((layer.output_dim, layer.input_dim), Normal::new(0., std).unwrap()),
                // W: Array::from_shape_simple_fn((output_dim, input_dim), || Dual::variable(normal.sample
                B: Array::zeros((layer.output_dim, 1)),
                activation: Activation::default(),
            });
        }
        BuiltNeuralNetwork {
            layers: layers,
        }
    }
}

impl BuiltNeuralNetwork {
    /// forward-pass a batch of input vectors through the network
    pub fn forward(&mut self, inp: &Array2<f64>) -> Array2<f64> {
        let mut input = inp.to_owned();
        for index in 0..self.layers.len() {
            input = self.layers[index].forward(&input);
        }
        input.to_owned()
    }
}
