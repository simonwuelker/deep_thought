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
    lr: f64,
}

#[allow(non_snake_case)] // snake case kinda makes sense with matrices
pub struct Layer {
    /// weight matrix
    W: Array2<f64>,
    /// bias vector
    B: Array1<f64>,
    /// number of input dimensions
    input_dim: usize,
    /// number of output dimensions
    output_dim: usize,
    /// some activation function
    activation: Activation,
    /// inp * weight  + bias
    Z: Array1<f64>,
    /// activation(Z), the actual activation of the neurons
    A: Array1<f64>,
}

impl Layer {
    /// construct a new layer with provided dimensions and random weights/biases
    pub fn new(input_dim: usize, output_dim: usize) -> Layer {
        Layer {
            W: Array::random((output_dim, input_dim), Uniform::new(-1., 1.)),
            B: Array::random(output_dim, Uniform::new(-1., 1.)),
            input_dim: input_dim,
            output_dim: output_dim,
            activation: Activation::default(),
            Z: Array::zeros(output_dim),
            A: Array::zeros(output_dim),
        }
    }

    /// construct a layer from provided weight/bias parameters
    pub fn from_parameters(parameters: (Array2<f64>, Array1<f64>)) -> Result<Layer> {
        let input_dim = parameters.0.ncols();
        let output_dim = parameters.0.nrows();
        Ok(Layer {
            W: parameters.0,
            B: parameters.1,
            Z: Array::zeros(output_dim),
            A: Array::zeros(output_dim),
            activation: Activation::default(),
            input_dim: input_dim,
            output_dim: output_dim,
        })
    }

    /// get the weights/biases of the neurons
    pub fn get_parameters(&self) -> (Array2<f64>, Array1<f64>) {
        (self.W.clone(), self.B.clone())
    }

    /// manually set weights/biases for the neurons
    pub fn set_parameters(&mut self, parameters: (Array2<f64>, Array1<f64>)) -> Result<()> {
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
    pub fn forward(&mut self, inp: &Array1<f64>) {
        self.Z = self.W.dot(inp) + &self.B;
        self.A = self.activation.compute(&self.Z);
    }
}

impl NeuralNetworkBuilder {
    pub fn new() -> NeuralNetworkBuilder {
        NeuralNetworkBuilder {
            layers: vec![],
            lr: 0.01,
        }
    }

    /// add a hidden layer to the network
    pub fn add_layer(mut self, layer: Layer) -> NeuralNetworkBuilder {
        self.layers.push(layer);
        self
    }

    /// manually set the learning rate, default is 0.01
    pub fn learning_rate(mut self, lr: f64) -> NeuralNetworkBuilder {
        self.lr = lr;
        self
    }

    /// forward-pass a 1D vector through the network
    pub fn forward(&mut self, inp: &Array1<f64>) -> Array1<f64> {
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
    pub fn backprop(&mut self, input: Array1<f64>, target: Array1<f64>, loss: Loss) {
        let num_layers = self.layers.len();

        // backpropagation for last layer is a bit special because of the cost function
        let last_layer = &self.layers[num_layers - 1];
        let mut dz = last_layer.activation.derivative(&last_layer.Z) * loss.derivative(&last_layer.A, &target);
        let mut dw = &dz * &self.layers[num_layers - 2].A;
        let mut db = &dz;

        let last_layer_mut = &mut self.layers[num_layers - 1];
        last_layer_mut.W = &last_layer_mut.W - dw * self.lr;
        last_layer_mut.B = &last_layer_mut.B - db * self.lr;

        // all the layers in the middle are the same
        for n_ in 1..num_layers - 1 {
            // i forgot why .reverse doesnt work so this will have to do
            let n = num_layers - n_ - 1;

            let nth_layer = &self.layers[n];
            // dz = nth_layer.activation.derivative(&nth_layer.Z) * &self.layers[n + 1].W.sum_axis(Axis(0)) * dz;
            dz = nth_layer.activation.derivative(&nth_layer.Z) * &self.layers[n + 1].W.sum_axis(Axis(0)) * &dz;
            println!("dz: {}", &dz);
            println!("A: {}", &self.layers[n - 1].A);
            panic!("no worki");
            dw = &dz * &self.layers[n - 1].A;
            db = &dz;

            let nth_layer_mut = &mut self.layers[n];
            nth_layer_mut.W = &nth_layer_mut.W - dw * self.lr;
            nth_layer_mut.B = &nth_layer_mut.B - db * self.lr;

        }
        // first layer is a bit special again bc its input isn't the previous layer's activation, it's the input!
        let first_layer = &self.layers[0];
        dz = dz * (first_layer.activation.derivative(&first_layer.Z) * &self.layers[1].W).sum_axis(Axis(0));
        let dw = &dz * &input;
        let db = &dz;

        let first_layer_mut = &mut self.layers[0];
        first_layer_mut.W = &first_layer_mut.W - dw * self.lr;
        first_layer_mut.B = &first_layer_mut.B - db * self.lr
    }
}

