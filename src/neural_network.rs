use anyhow::Result;
use ndarray::prelude::*;
use crate::error::Error;
use ndarray_rand::{
    RandomExt,
    rand_distr::Uniform,
};

pub struct NeuralNetworkBuilder {
    layers: Vec<Layer>,
    learning_rate: f32,
}

pub struct Layer {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_dim: usize,
    output_dim: usize,
    d_weights: Array2<f32>,
    d_bias: Array2<f32>,
}

impl Layer {
    /// construct a new layer with provided dimensions and random weights/biases
    pub fn new(input_dim: usize, output_dim: usize) -> Layer {
        Layer {
            weights: Array::random((output_dim, input_dim), Uniform::new(-1.0, 1.)),
            bias: Array::random((output_dim, input_dim), Uniform::new(-1.0, 1.)),
            input_dim: input_dim,
            output_dim: output_dim,
            d_weights: Array::zeros((output_dim, input_dim)),
            d_bias: Array::zeros((output_dim, input_dim)),
        }
    }

    /// construct a layer from provided weight/bias parameters
    pub fn from_parameters(parameters: (Array2<f32>, Array2<f32>)) -> Result<Layer> {
        if parameters.0.raw_dim() != parameters.1.raw_dim() {
            return Err(Error::MismatchedDimensions{
                expected: parameters.0.raw_dim().into_dyn(), 
                found: parameters.1.raw_dim().into_dyn(),
            }.into());
        }
        let input_dim = parameters.0.ncols();
        let output_dim = parameters.0.nrows();
        Ok(Layer {
            weights: parameters.0,
            bias: parameters.1,
            d_weights: Array2::zeros((output_dim, input_dim)),
            d_bias: Array2::zeros((output_dim, input_dim)),
            input_dim: input_dim,
            output_dim: output_dim,
        })
    }

    /// get the weights/biases of the neurons
    pub fn get_parameters(&self) -> (Array2<f32>, Array2<f32>) {
        (self.weights.clone(), self.bias.clone())
    }

    /// manually set weights/biases for the neurons
    pub fn set_parameters(&mut self, parameters: (Array2<f32>, Array2<f32>)) -> Result<()> {
        // make sure the dimensions match before replacing the old ones
        if self.weights.raw_dim() != parameters.0.raw_dim() {
            return Err(Error::MismatchedDimensions{
                expected: self.weights.raw_dim().into_dyn(), 
                found: parameters.0.raw_dim().into_dyn(),
            }.into());
        }
        else if self.bias.raw_dim() != parameters.1.raw_dim() {
            return Err(Error::MismatchedDimensions{
                expected: self.bias.raw_dim().into_dyn(),
                found: parameters.1.raw_dim().into_dyn(),
            }.into())
        }

        self.weights = parameters.0;
        self.bias = parameters.1;

        Ok(())
    }

    /// forward-pass a input vector through the layer
    pub fn forward(&self, inp: Array1<f32>) -> Array1<f32> {
        ((&inp * &self.weights) + &self.bias).sum_axis(Axis(1))
    }
}

impl NeuralNetworkBuilder {
    pub fn new() -> NeuralNetworkBuilder {
        NeuralNetworkBuilder {
            layers: vec![],
            learning_rate: 0.1,
        }
    }

    /// add a hidden layer to the network
    pub fn add_layer(mut self, layer: Layer) -> NeuralNetworkBuilder {
        self.layers.push(layer);
        self
    }

    // /// add an activation function to the network
    // pub fn add_activation(mut self, activation: &Activation) -> NeuralNetworkBuilder {
    //     self.operations.push(Operation::Activation(Box::new(activation)));
    //     self
    // }

    /// forward-pass a 1D vector through the network
    pub fn forward(&self, inp_: Array1<f32>) -> Array1<f32> {
        let mut inp = inp_.clone();
        for layer in &self.layers {
            inp = layer.forward(inp);
        }
        inp
    }

    pub fn backprop(&mut self) {
    }
}
