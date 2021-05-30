use anyhow::Result;
use ndarray::prelude::*;
use crate::{
    error::Error,
    loss::Loss,
    activation::Activation,
};
use ndarray_rand::{
    RandomExt,
    rand_distr::Uniform,
};

pub struct NeuralNetworkBuilder {
    layers: Vec<Layer>,
    learning_rate: f32,
}

#[allow(non_snake_case)] // snake case kinda makes sense with matrices
pub struct Layer {
    /// weight matrix
    W: Array2<f32>,
    /// bias matrix
    B: Array2<f32>,
    /// number of input dimensions
    input_dim: usize,
    /// number of output dimensions
    output_dim: usize,
    /// some activation function
    activation: Activation,
    /// inp * weight  + bias
    Z: Array1<f32>,
    /// activation(Z), the actual activation of the neurons
    A: Array1<f32>,
    d_weights: Array2<f32>,
    d_bias: Array2<f32>,
}

impl Layer {
    /// construct a new layer with provided dimensions and random weights/biases
    pub fn new(input_dim: usize, output_dim: usize) -> Layer {
        Layer {
            W: Array::random((output_dim, input_dim), Uniform::new(-1.0, 1.)),
            B: Array::random((output_dim, input_dim), Uniform::new(-1.0, 1.)),
            input_dim: input_dim,
            output_dim: output_dim,
            activation: Activation::default(),
            Z: Array::zeros(output_dim),
            A: Array::zeros(output_dim),
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
            W: parameters.0,
            B: parameters.1,
            Z: Array::zeros(output_dim),
            A: Array::zeros(output_dim),
            activation: Activation::default(),
            d_weights: Array::zeros((output_dim, input_dim)),
            d_bias: Array2::zeros((output_dim, input_dim)),
            input_dim: input_dim,
            output_dim: output_dim,
        })
    }

    /// get the weights/biases of the neurons
    pub fn get_parameters(&self) -> (Array2<f32>, Array2<f32>) {
        (self.W.clone(), self.B.clone())
    }

    /// manually set weights/biases for the neurons
    pub fn set_parameters(&mut self, parameters: (Array2<f32>, Array2<f32>)) -> Result<()> {
        // make sure the dimensions match before replacing the old ones
        if self.W.raw_dim() != parameters.0.raw_dim() {
            return Err(Error::MismatchedDimensions{
                expected: self.W.raw_dim().into_dyn(), 
                found: parameters.0.raw_dim().into_dyn(),
            }.into());
        }
        else if self.B.raw_dim() != parameters.1.raw_dim() {
            return Err(Error::MismatchedDimensions{
                expected: self.B.raw_dim().into_dyn(),
                found: parameters.1.raw_dim().into_dyn(),
            }.into())
        }

        self.W = parameters.0;
        self.B = parameters.1;

        Ok(())
    }

    /// define a activation function for that layer (default is f(x) = x )
    pub fn activation(mut self, a: Activation) -> Layer {
        self.activation = a;
        self
    }

    /// forward-pass a input vector through the layer
    pub fn forward(&mut self, inp: &Array1<f32>) {
        self.Z = ((inp * &self.W) + &self.B).sum_axis(Axis(1));
        self.A = self.activation.compute(&self.Z);
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

    /// forward-pass a 1D vector through the network
    pub fn forward(&mut self, inp: Array1<f32>) -> Array1<f32> {
        for index in 0..self.layers.len() {
            if index == 0 {
                self.layers[index].forward(&inp);
            } else {
                let prev_activation = self.layers[index - 1].A.clone();
                self.layers[index].forward(&prev_activation);
            }
        }
        self.layers.iter().last().unwrap().A.clone()
    }

    pub fn backprop(&mut self, target: Array1<f32>, loss: Loss) {
        // .enumerate is used in a confusing way here but its actually easier
        // let mut dz: Array2<f32>;
        // for (index, layer) in &mut self.layers.iter().reverse().enumerate() {
        //     if index == 0 {
        //         dz = match loss {
        //             loss::MSE => target - layer.activation,
        //         }
        //     } else {
        //         dz = 
        //     }
        // }
    }
}

