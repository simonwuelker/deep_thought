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
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
    ];
    let labels = array![[0.], [1.], [1.], [0.]];
    
    let dataset = Dataset::new(inputs, labels, 1., BatchSize::One)?;
    let loss_fn = Loss::MSE;

    // convoluted layerbuilding for test purposes
    let l1_bias = array![[0.33503600319382953], [0.2281878581838248], [-0.07632319718961056]];
    let l2_bias = array![[-0.061635521334861165], [0.22311927096806938], [0.3858446076254327]];
    let l3_bias = array![[-0.320750721292129]];

    let l1_weight = array![
        [0.4719902228641588, 0.07456852960846994],
        [0.24821372390126983, 0.15926089678311706],
        [0.4350420676050757, -0.3635419453638451],
    ];
    let l2_weight = array![
        [0.08115954636864608, 0.22281905787076073, -0.13564626922849476],
        [-0.13098361424948313, -0.44693379386553556, 0.38802147329361136],
        [0.14989085959488024, 0.40187661883580583, -0.11136940661186756],
    ];
    let l3_weight = array![
        [0.3645358671970196, -0.24000831624897967, 0.14783022232855392],
    ];
    let l1 = Layer::from_parameters((l1_weight, l1_bias))?.activation(Activation::Sigmoid);
    let l2 = Layer::from_parameters((l2_weight, l2_bias))?.activation(Activation::Sigmoid);
    let l3 = Layer::from_parameters((l3_weight, l3_bias))?.activation(Activation::Sigmoid);
    
    // Build the neural net
    let mut net = NeuralNetworkBuilder::new()
        .learning_rate(0.3)
        .momentum(0.1)
        .add_layer(l1)
        .add_layer(l2)
        .add_layer(l3);
        // .add_layer(Layer::new(2, 3))
        // .add_layer(Layer::new(3, 3))
        // .add_layer(Layer::new(3, 1));
    
    // train the network
    for epoch in 0..11000 {
        for (samples, labels) in dataset.iter_train().into_iter() {
            let _out = net.forward(&samples);
            // println!("out: {}", _out);
            // panic!("no more!");
            if epoch % 100 == 0 {
                println!("training epoch {}", epoch);
                println!("  Loss: {}\n", &loss_fn.compute(&_out, &labels).mean().unwrap());
            }
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
        println!("{} == {}", out.map(|&x| x.round()), label);
    }
    
    println!("Mean loss over {} test samples: {:.2}", num_test_samples, total_loss / num_test_samples as f64);
    Ok(())
}
