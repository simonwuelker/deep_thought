//! Neural Networks in Rust. Not much more for now
// #![feature(test)]

pub mod neural_network;
pub mod activation;
pub mod loss;
pub mod error;
pub mod dataset;
pub mod prelude;


#[cfg(test)]
mod tests {
    use anyhow::Result;
    use crate::prelude::*;
    use ndarray::prelude::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;


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
            let target_sample = sample_orig.slice(s![index..index+1, ..]);
            let target_label = label_orig.slice(s![index..index+1, ..]);

            let denormalized_sample = &dataset.denormalize_records(sample_norm);
            let denormalized_label = &dataset.denormalize_labels(label_norm);
            assert!(target_sample.abs_diff_eq(denormalized_sample, 0.01));
            assert!(target_label.abs_diff_eq(denormalized_label, 0.01));
        }
        Ok(())
    }
}
