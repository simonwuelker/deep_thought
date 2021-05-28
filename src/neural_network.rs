use serde::{Serialize, Deserialize};
use anyhow::Result;
use ndarray::prelude::*;
use crate::error::Error;
use ndarray_rand::{
    RandomExt,
    rand_distr::Uniform,
};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetworkBuilder {
    layers: Vec<Layer>,
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl Layer {
    /// construct a new layer with provided dimensions and random weights/biases
    pub fn new(input_dim: usize, output_dim: usize) -> Layer {
        Layer {
            weights: Array::random((output_dim, input_dim), Uniform::new(-1.0, 1.)),
            bias: Array::random((output_dim, input_dim), Uniform::new(-1.0, 1.)),
            input_dim: input_dim,
            output_dim: output_dim,
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
        Ok(Layer {
            input_dim: parameters.0.ncols(),
            output_dim: parameters.0.nrows(),
            weights: parameters.0,
            bias: parameters.1,
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
        }
    }

    pub fn add_layer(mut self, layer: Layer) -> NeuralNetworkBuilder {
        self.layers.push(layer);
        self
    }

    /// forward-pass a 1D vector through the network
    pub fn forward(&self, inp_: Array1<f32>) -> Array1<f32> {
        let mut inp = inp_.clone();
        for layer in &self.layers {
            inp = layer.forward(inp);
        }
        inp
    }

     // /// add a activation function
     // pub fn add_activation(&mut self, dyn fn(f32) -> f32) {
     // }

    /// Serialize the network to a provided path
    pub fn serialize(&self) -> Result<()> {
        unimplemented!()
    }

    /// Load the network from a provided path 
    pub fn deserialize(path: String) -> Result<NeuralNetworkBuilder> {
        unimplemented!()
    }
}
