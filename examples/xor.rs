use anyhow::Result;
use ndarray::prelude::*;
use rust_nn::{
    loss::Loss,
    activation::Activation,
    dataset::{Dataset, BatchSize},
    neural_network::{Layer, NeuralNetworkBuilder},
};


fn main() -> Result<()>{
    // Build the input and label arrays
    let inputs = array![
        [1., 1.],
        [1., 0.],
        [0., 1.],
        [0., 0.],
    ];
    let labels = array![[0.], [1.], [1.], [0.]];
    
    let dataset = Dataset::new(inputs, labels, 1., BatchSize::All)?;
    let loss_fn = Loss::MSE;
    
    // Build the neural net
    let mut net = NeuralNetworkBuilder::new()
        .learning_rate(0.005)
        .momentum(0.3)
        .add_layer(Layer::new(2, 3).activation(Activation::ReLU))
        .add_layer(Layer::new(3, 1).activation(Activation::ReLU));
    
    // train the network
    for epoch in 0..1000 {
        println!("training epoch {}", epoch);
        for (samples, labels) in dataset.iter_train().into_iter() {
            let _out = net.forward(&samples);
            println!("{}", &loss_fn.compute(&_out, &labels));
            net.backprop(samples, labels, &loss_fn);
        }
    }
    
    // evaluate the net 
    let mut total_loss: f64 = 0.;
    // should ofc be iter_test but this dataset is kinda minimalistic
    let test_iter = dataset.iter_train();
    let num_test_samples = test_iter.num_batches * test_iter.batch_size;
    for (sample, label) in test_iter {
        let out = net.forward(&sample);
        total_loss += loss_fn.compute(&out, &label).sum();
    }
    
    println!("Mean loss over {} test samples: {:.2}", num_test_samples, total_loss / num_test_samples as f64);
    Ok(())
}
