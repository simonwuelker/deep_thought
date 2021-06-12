use crate::prelude::*;
use ndarray::prelude::*;

/// Implements stochastic gradient descent (optionally with momentum)
pub struct SGD {
    lr: f64,
    momentum: f64,
    v_weight: Vec<Array2<f64>>,
    v_bias: Vec<Array2<f64>>,
}

impl Optimizer for SGD {
    fn new(net: &NeuralNetwork) -> Self {
        let mut v_weight = vec![];
        let mut v_bias = vec![];

        for layer in &net.layers {
            v_weight.push(Array2::zeros(layer.W.dim()));
            v_bias.push(Array2::zeros(layer.B.dim()));
        }

        SGD {
            lr: 0.01,
            momentum: 0.,
            v_weight: v_weight,
            v_bias: v_bias,
        }
    }

    fn step(&mut self, net: &mut NeuralNetwork) {
        for (index, layer) in &mut net.layers.iter_mut().enumerate() {
            // update velocity vector
            self.v_weight[index] = self.momentum * &self.v_weight[index] + self.lr * &layer.dW;
            self.v_bias[index] = self.momentum * &self.v_bias[index] + self.lr * &layer.dB;

            // update network parameters
            layer.W = &layer.W + &self.v_weight[index];
            layer.B = &layer.B + &self.v_bias[index];
        }
    }
}

impl SGD {
    /// Set the learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        if lr < 0. {
            panic!("momentum must be >= 0, got {}", lr);
        }
        self.lr = lr;
        self
    }

    /// Set the momentum
    pub fn momentum(mut self, momentum: f64) -> Self {
        if momentum < 0. {
            panic!("momentum must be >= 0, got {}", momentum);
        }
        self.momentum = momentum;
        self
    }
}
