//! This crate implements basic feedforward-neural Networks in rust.
//!
//! A basix XOR training example might look like this:
//! ```rust
//! fn main() -> Result<()>{
//!     // Build the input and label arrays
//!     let inputs = array![
//!         [0., 0.],
//!         [0., 1.],
//!         [1., 0.],
//!         [1., 1.],
//!     ];
//!     let labels = array![[0.], [1.], [1.], [0.]];
//!     
//!     let dataset = Dataset::new(inputs, labels, 1., BatchSize::One)?;
//!     let loss_fn = Loss::MSE;
//!
//!     // Build the neural net
//!     let mut net = NeuralNetworkBuilder::new()
//!         .learning_rate(0.3)
//!         .momentum(0.1)
//!         .add_layer(Layer::new(2, 3).activation(Activation::Sigmoid))
//!         .add_layer(Layer::new(3, 3).activation(Activation::Sigmoid))
//!         .add_layer(Layer::new(3, 1).activation(Activation::Sigmoid));
//!     
//!     // train the network
//!     for epoch in 0..11000 {
//!         for (samples, labels) in dataset.iter_train().into_iter() {
//!             let _out = net.forward(&samples);
//!             if epoch % 100 == 0 {
//!                 println!("training epoch {}", epoch);
//!                 println!("  Loss: {}\n", &loss_fn.compute(&_out, &labels).mean().unwrap());
//!             }
//!             net.backprop(samples, labels, &loss_fn);
//!         }
//!     }
//!     
//!     // evaluate the net
//!     let mut total_loss: f64 = 0.;
//!     // should ofc be iter_test but this dataset is kinda minimalistic
//!     let test_iter = dataset.iter_train();
//!     let num_test_samples = test_iter.num_batches * test_iter.batch_size;
//!     for (sample, label) in test_iter {
//!         let out = net.forward(&sample);
//!         total_loss += loss_fn.compute(&out, &label).sum();
//!         println!("{} == {}", out.map(|&x| x.round()), label);
//!     }
//!     
//!     println!("Mean loss over {} test samples: {:.2}", num_test_samples, total_loss / num_test_samples as f64);
//!     Ok(())
//! }
//! ```
//! Feature Flags
// #![feature(test)]

pub mod activation;
pub mod dataset;
pub mod error;
pub mod loss;
pub mod neural_network;
pub mod prelude;

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use anyhow::Result;
    use ndarray::prelude::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn simple_net_test() {
        let mut net = NeuralNetworkBuilder::new()
            .learning_rate(0.05)
            .add_layer(Layer::new(1, 1))
            .add_layer(Layer::new(1, 1));

        let inp = array![[0.6]];
        let target = array![[0.3]];

        let mut last_loss = 50.0;
        for _index in 0..100 {
            let out = net.forward(&inp);
            last_loss = Loss::MSE.compute(&out, &target).mean().unwrap();
            println!("In: {} Out: {} Loss: {}", &inp, &out, &last_loss);
            net.backprop(inp.clone(), target.clone(), &Loss::MSE);
        }
        assert!(last_loss < 0.001);
        assert!(false);
    }

    #[test]
    /// assert that the normalization/denormalization of datasets work
    fn dataset_normalization() -> Result<()> {
        let sample_orig = Array::random((3, 3), Uniform::new(-1., 1.));
        let label_orig = Array::random((3, 2), Uniform::new(-1., 1.));
        let dataset = Dataset::new(sample_orig.clone(), label_orig.clone(), 1.0, BatchSize::One)?;
        for (index, (sample_norm, label_norm)) in dataset.iter_train().enumerate() {
            let target_sample = sample_orig.slice(s![index..index + 1, ..]);
            let target_label = label_orig.slice(s![index..index + 1, ..]);

            let denormalized_sample = &dataset.denormalize_records(sample_norm);
            let denormalized_label = &dataset.denormalize_labels(label_norm);
            assert!(target_sample.abs_diff_eq(denormalized_sample, 0.01));
            assert!(target_label.abs_diff_eq(denormalized_label, 0.01));
        }
        Ok(())
    }
}
