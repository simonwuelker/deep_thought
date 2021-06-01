//! Neural Networks in Rust. Not much more for now
// #![feature(test)]

pub mod neural_network;
pub mod activation;
pub mod loss;
pub mod error;
pub mod dataset;


#[cfg(test)]
mod tests {
    #[test]
    fn simple_net_test() {
        let mut net = NeuralNetworkBuilder::new()
            .learning_rate(0.05)
            .add_layer(Layer::new(1, 1))
            .add_layer(Layer::new(1, 1));

        let inp = array![0.6];
        let target = array![0.3];

        let mut last_loss = 50.0;
        for index in 0..100 {
            let out = net.forward(&inp);
            last_loss = Loss::MSE.compute(&out, &target);
            println!("In: {} Out: {} Loss: {}", &inp, &out, &last_loss);
            net.backprop(inp.clone(), target.clone(), Loss::MSE);
        }
        assert!(last_loss < 0.001);
        assert!(false);
    }

    // #[test]
    // fn dataset_normalization() {
    //     let original = Array::random((3, 3), Uniform::new(-1., 1.)),
    //     let dataset = Dataset::new(original,
    // }

    #[test]
    fn can_we_use_dot() {
        let inp = array![1., 2., 3.];
        let weights = array![
            [0.5, 0.3, 1.2],
            [1.6, 2.0, 0.0],
            [1.7, 0.3, 0.6],
            [0.1, 0.2, 0.4],
        ];

        let the_previous_way = (&inp * &weights).sum_axis(Axis(1));
        let the_new_way = weights.dot(&inp);
        assert_eq!(the_previous_way, the_new_way);
    }

    // use crate::neural_network::{Layer, NeuralNetworkBuilder};
    // use ndarray::prelude::*;
    // #[test]
    // fn layer_forward_pass() -> Result<()>{
    //     let weights = Array::from_elem((2, 4), 0.5);
    //     let bias = Array::from_elem((2, 4), -1.0);
    //     let l = Layer::from_parameters((weights, bias))?;

    //     let a = array![0., 1., 2., 3.];
    //     l.forward(a);
    //     assert_eq!(array![-1.0, -1.0], l.A);
    //     Ok(())
    // }

    // use test::Bencher;
    // // just a test benchmark
    // #[bench]
    // fn bench_add_two(b: &mut Bencher) {
    //     b.iter(|| 2+2)
    // }   
}
