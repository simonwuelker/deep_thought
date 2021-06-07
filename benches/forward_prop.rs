use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_nn::neural_network::{NeuralNetworkBuilder, Layer};
use ndarray::Array2;
use ndarray_rand::{
    RandomExt,
    rand_distr::Uniform,
};

fn criterion_benchmark(c: &mut Criterion) {
    // Build the neural net
    let mut net = NeuralNetworkBuilder::new()
        .add_layer(Layer::new(50, 100))
        .add_layer(Layer::new(100, 10));

    // construct some arbitrary input of 10 batches
    let inp = Array2::random((50, 10), Uniform::new(-1., 1.));

    c.bench_function("Forward pass", |b| b.iter(|| net.forward(black_box(&inp))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
