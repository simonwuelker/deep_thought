//! Neural Networks in Rust. Not much more for now
// #![feature(test)]

mod neural_network;
mod activation;
mod error;
mod tensor;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct HeartFailureRecord {
        age: f32,
        anaemia: u16,
        creatinine_phosphokinase: u16,
        diabetes: u16,
        ejection_fraction: u16,
        high_blood_pressure: u16,
        platelets: f32,
        serum_creatinine: f32,
        serum_sodium: u16,
        sex: u16,
        smoking: u16,
        time: u16,
        death_event: u16,
    }

    #[test]
    fn heart_failure_classification() -> Result<()>{
        // Dataset from https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
        let mut rdr = csv::Reader::from_path("datasets/heart_failure_clinical_records_dataset.csv")?;
        for result in rdr.deserialize() {
            let record: HeartFailureRecord = result?;
            println!("{:?}", record);
        }
        Ok(())
    }
    use crate::*;
    #[test]
    fn activation_functions() {
        // ReLU
        let mut x = -1.0;
        let mut y = 1.0;
        activation::relu(&mut x);
        activation::relu(&mut y);
        assert_eq!(0.0, x);
        assert_eq!(1.0, y);

        // Sigmoid
        x = 0.0;
        activation::sigmoid(&mut x);
        assert_eq!(0.5, x);
    }

    use crate::neural_network::{Layer, NeuralNetworkBuilder};
    use ndarray::prelude::*;
    #[test]
    fn layer_forward_pass() -> Result<()>{
        let weights = Array::from_elem((2, 4), 0.5);
        let bias = Array::from_elem((2, 4), -1.0);
        let l = Layer::from_parameters((weights, bias))?;

        let a = array![0., 1., 2., 3.];
        let res = l.forward(a);
        assert_eq!(array![-1.0, -1.0], res);
        Ok(())
    }

    #[test]
    fn network_forward_pass() {
        let network = NeuralNetworkBuilder::new()
            .add_layer(Layer::new(10, 15))
            .add_layer(Layer::new(15, 20))
            .add_layer(Layer::new(20, 10));
        let res = network.forward(Array1::ones(10));
        println!("{:?}", res);
        assert!(false);
    }

    // use test::Bencher;
    // // just a test benchmark
    // #[bench]
    // fn bench_add_two(b: &mut Bencher) {
    //     b.iter(|| 2+2)
    // }   
}
