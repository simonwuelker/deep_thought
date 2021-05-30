//! Neural Networks in Rust. Not much more for now
// #![feature(test)]

mod neural_network;
mod activation;
mod loss;
mod error;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct HeartFailureRecord {
        age: f32,
        anaemia: f32,
        creatinine_phosphokinase: f32,
        diabetes: f32,
        ejection_fraction: f32,
        high_blood_pressure: f32,
        platelets: f32,
        serum_creatinine: f32,
        serum_sodium: f32,
        sex: f32,
        smoking: f32,
        time: f32,
        death_event: f32,
    }

    impl HeartFailureRecord {
        fn build_input(&self) -> Array1<f32> {
            Array::from_vec(vec![
                self.age, self.anaemia, self.creatinine_phosphokinase, self.diabetes, self.ejection_fraction, self.high_blood_pressure,
                self.platelets, self.serum_creatinine, self.serum_sodium, self.sex, self.smoking, self.time,
            ])
        }

        fn build_target(&self) -> Array1<f32> {
            Array::from_vec(vec![
                self.death_event,
            ])
        }
    }

    use ndarray::prelude::*;
    use crate::{
        activation::Activation,
        neural_network::{Layer, NeuralNetworkBuilder},
        loss::Loss,
    };

    #[test]
    fn heart_failure_classification() -> Result<()>{
        // Dataset from https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
        let mut rdr = csv::Reader::from_path("datasets/heart_failure_clinical_records_dataset.csv")?;
        // can probably be done via sth like rdr.deserialize().into_vec()
        let mut records: Vec<HeartFailureRecord> = vec![];
        for result in rdr.deserialize() {
            let record: HeartFailureRecord = result?;
            records.push(record);
        }

        let train_records = &records[..200];
        let test_records = &records[200..];

        // Build the neural net
        let mut net = NeuralNetworkBuilder::new()
            .add_layer(Layer::new(12, 20))
            .add_layer(Layer::new(20, 10))
            .add_layer(Layer::new(10, 5))
            .add_layer(Layer::new(5, 1).activation(Activation::Sigmoid));

        // evaluate the net 
        let mut total_loss: f32 = 0.;
        for record in test_records {
            let out = net.forward(record.build_input());
            let target = record.build_target();
            total_loss += Loss::MSE.compute(out, target);
        }

        println!("Mean loss over 100 test samples: {}", total_loss / 100.);
        assert!(false);

        Ok(())
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

    #[test]
    fn network_forward_pass() {
        let mut network = NeuralNetworkBuilder::new()
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
