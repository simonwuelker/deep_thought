use anyhow::Result;
use ndarray::prelude::*;
use crate::{
    error::Error,
    loss::Loss,
    activation::Activation,
};
use ndarray_rand::{
    RandomExt,
    rand_distr::Normal,
};

pub struct NeuralNetworkBuilder {
    layers: Vec<Layer>,
    lr: f64,
    momentum: f64,
}

#[allow(non_snake_case)] // non snake case kinda makes sense with matrices
pub struct Layer {
    /// Weight matrix
    W: Array2<f64>,
    /// Bias vector
    B: Array2<f64>,
    /// Activation function which turns self.Z into self.A
    activation: Activation,
    /// inp * weight  + bias
    Z: Array2<f64>,
    /// Activation(Z), the actual activation of the neurons
    A: Array2<f64>,
    /// Accumulated weight gradients
    dW: Array2<f64>,
    /// Accumulated bias gradients
    dB: Array2<f64>,
}

impl Layer {
    /// construct a new layer with provided dimensions and random weights/biases
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
        let output_dim = parameters.0.nrows();
        Ok(Layer {
            W: parameters.0,
            B: parameters.1,
            Z: Array::zeros((0, output_dim)),
            A: Array::zeros((0, output_dim)),
            activation: Activation::default(),
            dW: Array::zeros((0,0)),
            dB: Array::zeros((0, 0)),
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

    /// forward-pass a batch of input vectors through the layer
    pub fn forward(&mut self, inp: &Array2<f64>) {
        self.Z = self.W.dot(inp) + &self.B;
        self.A = self.activation.compute(&self.Z);
    }
}

impl NeuralNetworkBuilder {
    pub fn new() -> NeuralNetworkBuilder {
        NeuralNetworkBuilder {
            layers: vec![],
            lr: 0.01,
            momentum: 0.,
        }
    }

    /// add a hidden layer to the network
    pub fn add_layer(mut self, layer: Layer) -> NeuralNetworkBuilder {
        self.layers.push(layer);
        self
    }

    /// manually set the learning rate, default is 0.01
    pub fn learning_rate(mut self, lr: f64) -> NeuralNetworkBuilder {
        if lr < 0. { panic!("learning rate must be >= 0, got {}", lr); }
        self.lr = lr;
        self
    }

    /// set the momentum for gradient descent
    pub fn momentum(mut self, momentum: f64) -> NeuralNetworkBuilder {
        if momentum < 0. { panic!("momentum must be >= 0, got {}", momentum); }
        self.momentum = momentum;
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
    pub fn backprop(&mut self, input: Array2<f64>, target: Array2<f64>, loss: &Loss) {
        let num_layers = self.layers.len();
        // Initial dz for the last layer
        // dz is the error in the layers Z value - sometimes also denoted as delta
        let mut dz = &self.layers[num_layers - 1].activation.derivative(&self.layers[num_layers - 1].Z) * 
            loss.derivative(&self.layers[num_layers - 1].A, &target);

        for n in (0..num_layers).rev() {
            // println!("Optimizing Layer {}", n);
            let nth_layer = &self.layers[n];

            // determine the vector that is fed into the nth layer
            let nth_layer_input = if n == 0 {
               &input 
            } else {
                &self.layers[n - 1].A
            };

            // find the derivative of the cost function with respect to the nth layers Z value
            if n != num_layers - 1 {
                dz = &self.layers[n + 1].W.to_owned().reversed_axes().dot(&dz) *
                    nth_layer.activation.derivative(&nth_layer.Z);
            }
            
            let dw = &dz.dot(&nth_layer_input.to_owned().reversed_axes());
            let db = (&dz.sum_axis(Axis(1))).to_shape((dz.nrows(), 1)).unwrap().to_owned(); // need to add an extra dim

            let nth_layer_mut = &mut self.layers[n];
            nth_layer_mut.dW = self.momentum * &nth_layer_mut.dW + (1. - self.momentum) * dw;
            nth_layer_mut.dB = self.momentum * &nth_layer_mut.dB + (1. - self.momentum) * db;
        }
        for layer in &mut self.layers {
            layer.W = &layer.W - &layer.dW * self.lr;
            layer.B = &layer.B - &layer.dB * self.lr;
        }
    }
}

