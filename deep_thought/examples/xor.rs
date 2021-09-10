use anyhow::Result;
use deep_thought::optimizer::Optimizer;
use deep_thought::prelude::*;
use ndarray::prelude::*;
use num_traits::Float;

// Network size must be known at compile-time
const _NUM_PARAMETERS: usize = 19;

fn main() -> Result<()> {
    // Build the input and label arrays
    let inputs = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let labels = array![[0.], [1.], [1.], [0.]];

    let dataset = Dataset::raw(inputs, labels, 1., BatchSize::One)?;
    let loss_fn = Loss::MSE;

    // Build the neural net
    let mut net = NeuralNetwork::new()
        .add_layer(Layer::new(2, 3).activation(Activation::Sigmoid))
        .add_layer(Layer::new(3, 3).activation(Activation::Sigmoid))
        .add_layer(Layer::new(3, 1).activation(Activation::Sigmoid));

    // let mut optim = optimizer::SGD::new(&net).learning_rate(0.3).momentum(0.);

    // train the network
    for epoch in 0..11000 {
        let mut epoch_loss = 0.;

        for (samples, labels) in dataset.iter_train() {
            let out = net.forward(&samples);
            epoch_loss += &loss_fn.compute(&out, &labels).mean().unwrap();
            // optim.step(&mut net, &out);
        }

        if epoch % 100 == 0 {
            println!("training epoch {}", epoch);
            println!("Mean Loss: {}\n", epoch_loss / dataset.length() as f64);
            println!("data len {}", dataset.length());
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
        println!("{} == {}", out.map(|&x| x.round()), label);
    }

    println!(
        "Mean loss over {} test samples: {:.2}",
        num_test_samples,
        total_loss / num_test_samples as f64
    );
    Ok(())
}
