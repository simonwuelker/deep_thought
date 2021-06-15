use crate::{activation::Activation, error::Error, loss::Loss, optimizer::Optimizer};
use anyhow::Result;
use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Normal, RandomExt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// A Neural Network consisting of a an input/output and any number of additional hidden [`Layer`]s
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(non_snake_case)] // non snake case kinda makes sense with matrices
/// A single neuron layer with an associated [`Activation`] function
pub struct Layer {
    /// Weight matrix
    pub W: Array2<f64>,
    /// Bias vector
    pub B: Array2<f64>,
    /// Accumulated weight gradients
    pub dW: Array2<f64>,
    /// Accumulated bias gradients
    pub dB: Array2<f64>,
    /// Activation function which turns self.Z into self.A
    activation: Activation,
    /// inp * weight  + bias
    Z: Array2<f64>,
    /// Activation(Z), the actual activation of the neurons
    A: Array2<f64>,
}

impl Layer {
    /// construct a new layer with provided dimensions. Weights are initialized using [Glorot/Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a.html)
    /// Biases are always intialized to zeros
    pub fn new(input_dim: usize, output_dim: usize) -> Layer {
        let std = (2. / (input_dim + output_dim) as f64).sqrt();
        Layer {
            W: Array::random((output_dim, input_dim), Normal::new(0., std).unwrap()),
            B: Array::zeros((output_dim, 1)),
            activation: Activation::default(),
            Z: Array::zeros((0, output_dim)),
            A: Array::zeros((0, output_dim)),
            dW: Array::zeros((output_dim, input_dim)),
            dB: Array::zeros((output_dim, 1)),
        }
    }

    /// construct a layer from provided weight/bias parameters
    pub fn from_parameters(parameters: (Array2<f64>, Array2<f64>)) -> Result<Layer> {
        let input_dim = parameters.0.ncols();
        let output_dim = parameters.0.nrows();
        Ok(Layer {
            W: parameters.0,
            B: parameters.1,
            Z: Array::zeros((0, output_dim)),
            A: Array::zeros((0, output_dim)),
            activation: Activation::default(),
            dW: Array::zeros((output_dim, input_dim)),
            dB: Array::zeros((output_dim, 1)),
        })
    }

    /// get the weights/biases of the neurons
    pub fn get_parameters(&self) -> (Array2<f64>, Array2<f64>) {
        (self.W.clone(), self.B.clone())
    }

    /// manually set weights/biases for the neurons
    pub fn set_parameters(&mut self, parameters: (Array2<f64>, Array2<f64>)) -> Result<()> {
        // make sure the dimensions match before replacing the old ones
        if self.W.raw_dim() != parameters.0.raw_dim() {
            return Err(Error::MismatchedDimensions {
                expected: self.W.raw_dim().into_dyn(),
                found: parameters.0.raw_dim().into_dyn(),
            }
            .into());
        } else if self.B.raw_dim() != parameters.1.raw_dim() {
            return Err(Error::MismatchedDimensions {
                expected: self.B.raw_dim().into_dyn(),
                found: parameters.1.raw_dim().into_dyn(),
            }
            .into());
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

    /// forward-pass a batch of input vectors through the layer
    pub fn forward(&mut self, inp: &Array2<f64>) {
        self.Z = self.W.dot(inp) + &self.B;
        self.A = self.activation.compute(&self.Z);
    }
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork { layers: vec![] }
    }

    /// add a hidden layer to the network
    pub fn add_layer(mut self, layer: Layer) -> NeuralNetwork {
        self.layers.push(layer);
        self
    }

    /// forward-pass a batch of input vectors through the network
    pub fn forward(&mut self, inp: &Array2<f64>) -> Array2<f64> {
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

    /// Backpropagate the output through the network and adjust weights/biases to further match the
    /// desired target
    /// Input has shape (in_size, num_batches)
    pub fn backprop<O>(
        &mut self,
        input: Array2<f64>,
        target: Array2<f64>,
        loss: &Loss,
        optimizer: &mut O,
    ) where
        O: Optimizer,
    {
        let num_layers = self.layers.len();
        // Initial dz for the last layer
        // dz is the error in the layers Z value - sometimes also denoted as delta
        let mut dz = (&self.layers[num_layers - 1]
            .activation
            .derivative(&self.layers[num_layers - 1].Z)
            * loss.derivative(&self.layers[num_layers - 1].A, &target)).sum_axis(Axis(1));

        for n in (0..num_layers).rev() {
            let nth_layer = &self.layers[n];

            // determine the vector that is fed into the nth layer
            let nth_layer_input = if n == 0 {
                input.clone()
            } else {
                self.layers[n - 1].A.clone()
            };

            // find the derivative of the cost function with respect to the nth layers Z value
            if n != num_layers - 1 {
                // this might be wrong
                dz = (&self.layers[n + 1].W.t().dot(&dz)
                    * nth_layer.activation.derivative(&nth_layer.Z)).sum_axis(Axis(1));
            }

            let nth_layer_mut = &mut self.layers[n];
            nth_layer_mut.dW = dz.dot(&nth_layer_input.t());
            nth_layer_mut.dB = dz.clone();
        }

        // Actually optimize the network
        optimizer.step(self);
    }
}
