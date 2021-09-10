use crate::autograd::Dual;
use crate::optimizer::Optimizer;
use crate::prelude::*;
use ndarray::prelude::*;
use num_traits::Float;
use std::fmt;

/// Implements stochastic gradient descent (optionally with momentum)
pub struct SGD<F: Float + fmt::Debug, const N: usize> {
    lr: f64,
    momentum: f64,
    v_weight: Array1<Dual<F, N>>,
    v_bias: Array1<Dual<F, N>>,
}

impl<F, const N: usize> Optimizer<F, N> for SGD<F, N>
where
    F: Float + fmt::Debug,
    f64: Into<F>,
{
    fn new(net: &NeuralNetwork<F, N>) -> Self {
        SGD {
            lr: 0.01,
            momentum: 0.,
            // wrong, too many biases/weights
            v_weight: Array1::<Dual<F, N>>::zeros(N),
            v_bias: Array1::<Dual<F, N>>::zeros(N),
        }
    }

    fn step(&mut self, net: &mut NeuralNetwork<F, N>, loss: Dual<F, N>) {
        // for (index, layer) in &mut net.layers.iter_mut().enumerate() {
        //     // update velocity vector
        //     self.v_weight[index] = self.momentum * &self.v_weight[index] + self.lr * &layer.dW;
        //     self.v_bias[index] = self.momentum * &self.v_bias[index] + self.lr * &layer.dB;

        //     // update network parameters
        //     layer.W = &layer.W + &self.v_weight[index];
        //     layer.B = &layer.B + &self.v_bias[index];
        // }
    }
}

impl<F: Float + fmt::Debug, const N: usize> SGD<F, N> {
    /// Set the learning rate
    ///
    /// **Panics** if the learning is below 0
    pub fn learning_rate(mut self, lr: f64) -> Self {
        if lr < 0. {
            panic!("learning rate must be >= 0, got {}", lr);
        }
        self.lr = lr;
        self
    }

    /// Set the momentum
    ///
    /// **Panics** if the momentum is below 0
    pub fn momentum(mut self, momentum: f64) -> Self {
        if momentum < 0. {
            panic!("momentum must be >= 0, got {}", momentum);
        }
        self.momentum = momentum;
        self
    }
}
