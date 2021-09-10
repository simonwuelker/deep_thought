#![feature(array_zip, negative_impls, auto_traits)]

// //! This crate implements basic feedforward-neural Networks in rust.
// //!
// //! A basix XOR training example might look like this:
// //! ```rust
// //! fn main() -> Result<()> {
// //!    // Build the input and label arrays
// //!    let inputs = array![
// //!        [0., 0.],
// //!        [0., 1.],
// //!        [1., 0.],
// //!        [1., 1.],
// //!    ];
// //!    let labels = array![[0.], [1.], [1.], [0.]];
// //!
// //!    let dataset = Dataset::new(inputs, labels, 1., BatchSize::One)?;
// //!    let loss_fn = Loss::MSE;
// //!
// //!    // Build the neural net
// //!    let mut net = NeuralNetwork::new()
// //!        .learning_rate(0.3)
// //!        .momentum(0.1)
// //!        .add_layer(Layer::new(2, 3).activation(Activation::Sigmoid))
// //!        .add_layer(Layer::new(3, 3).activation(Activation::Sigmoid))
// //!        .add_layer(Layer::new(3, 1).activation(Activation::Sigmoid));
// //!
// //!    // train the network
// //!    for epoch in 0..11000 {
// //!        for (samples, labels) in dataset.iter_train() {
// //!            let _out = net.forward(&samples);
// //!            if epoch % 100 == 0 {
// //!                println!("training epoch {}", epoch);
// //!                println!("  Loss: {}\n", &loss_fn.compute(&_out, &labels).mean().unwrap());
// //!            }
// //!            net.backprop(samples, labels, &loss_fn);
// //!        }
// //!    }
// //!
// //!    // evaluate the net
// //!    let mut total_loss: f64 = 0.;
// //!    // should ofc be iter_test but this dataset is kinda minimalistic
// //!    let test_iter = dataset.iter_train();
// //!    let num_test_samples = test_iter.num_batches * test_iter.batch_size;
// //!    for (sample, label) in test_iter {
// //!        let out = net.forward(&sample);
// //!        total_loss += loss_fn.compute(&out, &label).sum();
// //!        println!("{} == {}", out.map(|&x| x.round()), label);
// //!    }
// //!
// //!    println!("Mean loss over {} test samples: {:.2}", num_test_samples, total_loss / num_test_samples as f64);
// //!    Ok(())
// //! }
// //! ```

/// Activation functions
pub mod activation;
/// Automatic tensor differentiation
pub mod autograd;
/// Dataset object which is used to split and normalize data
pub mod dataset;
/// Common errors
pub mod error;
/// Loss functions
pub mod loss;
/// Neural networks, Layers and math
pub mod neural_network;
/// Contains various different Types of optimizers
pub mod optimizer;
/// Common imports
pub mod prelude;
