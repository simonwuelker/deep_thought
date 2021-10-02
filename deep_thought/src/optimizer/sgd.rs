use crate::autograd::Dual;
use crate::optimizer::Optimizer;
use crate::prelude::*;
use ndarray::prelude::*;
use num_traits::Float;

/// Implements stochastic gradient descent (optionally with momentum)
pub struct SGD<F, const N: usize> {
    /// learning rate
    lr: F,
    /// how much the previous change affects the current change
    momentum: F,
    /// velocity of each parameter
    v: [F; N],
}

impl<F, const N: usize> Optimizer<F, N> for SGD<F, N>
where
    F: Float,
{
    fn new() -> Self {
        SGD {
            lr: F::from(0.01).unwrap(),
            momentum: F::zero(),
            v: [F::zero(); N],
        }
    }

    fn step(&mut self, net: &mut NeuralNetwork<F, N>, loss: Dual<F, N>) {
        // Update parameter velocities, not sure if the formula is correct
        let zipped = self.v.zip(loss.e);
        self.v = zipped.map(|(v, d)| self.momentum * v + self.lr * d);

        // Update the network's parameters

        //     // update network parameters
        //     layer.W = &layer.W + &self.v_weight[index];
        //     layer.B = &layer.B + &self.v_bias[index];
        // }
    }
}

impl<F, const N: usize> SGD<F, N> {
    /// Set the learning rate
    pub fn learning_rate(mut self, lr: F) -> Self {
        self.lr = lr;
        self
    }

    /// Set the momentum
    pub fn momentum(mut self, momentum: F) -> Self {
        self.momentum = momentum;
        self
    }
}
