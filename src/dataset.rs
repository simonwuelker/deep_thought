use anyhow::Result;
use ndarray::prelude::*;
use crate::error::Error;
use ndarray::iter::LanesIter;

pub struct Dataset {
    /// ratio between number of training and number of testing samples
    train_test_split: f64,
    /// normalized record data data contained by the dataset
    records: Array2<f64>,
    /// normalized labels to the records
    labels: Array2<f64>,
    /// mean of record columns, used to de-normalize the records
    record_means: Array1<f64>,
    /// mean of label columns, used to de-normalize the labels
    label_means: Array1<f64>,
}

impl Dataset {
    /// create a new dataset, normalizing the inputs and labels
    pub fn new(records: Array2<f64>, labels: Array2<f64>, train_test_split: f64) -> Result<Dataset> {
        let record_means = records.mean_axis(Axis(0)).ok_or(Error::NoData)?;
        let label_means = labels.mean_axis(Axis(0)).ok_or(Error::NoData)?;
        
        Ok(Dataset {
            train_test_split: train_test_split,
            records: records / &record_means,
            labels: labels / &label_means,
            record_means: record_means,
            label_means: label_means,
        })
    }

    /// denormalize a record vector into its original form
    pub fn denormalize_record(&self, normalized: Array1<f64>) -> Array1<f64> {
        normalized * &self.record_means
    }

    /// denormalize a label vector into its original form
    pub fn denormalize_label(&self, normalized: Array1<f64>) -> Array1<f64> {
        normalized * &self.label_means
    }

    /// return an iterator over training examples/labels in (sample, label) tupels
    pub fn iter_train(&self) -> TrainIterator {
        let num_train = (self.records.nrows() as f64 * self.train_test_split) as usize;
        TrainIterator {
            index: 0,
            num_samples: num_train,
            samples: self.records.slice(s![0..num_train, ..]).to_owned(),
            labels: self.labels.slice(s![..num_train, ..]).to_owned(),
        }
    }
}

// BIG TODO: use lifetimes and array views here instead of cloning everything, this is slow!
pub struct TrainIterator {
    index: usize,
    num_samples: usize,
    samples: Array2<f64>,
    labels: Array2<f64>,
}

impl Iterator for TrainIterator {
    type Item = (Array1<f64>, Array1<f64>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.num_samples {
            None
        }
        else {
            self.index += 1;
            Some((self.samples.index_axis(Axis(0), self.index-1).to_owned(), self.labels.index_axis(Axis(0), self.index-1).to_owned()))
        }
    }
}
// struct TrainIterator<'a> {
//     index: usize,
//     num_samples: usize,
//     samples: ArrayView<'a, f64, Dim<[usize; 2]>>,
//     labels: ArrayView<'a, f64, Dim<[usize; 2]>>
// }
// 
// impl<'a, 'b : 'a> Iterator for TrainIterator<'a> {
//     type Item = (ArrayView<'b, f64, Dim<[usize; 1]>>, ArrayView<'b, f64, Dim<[usize;1 ]>>);
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index > self.num_samples {
//             None
//         }
//         else {
//             self.index += 1;
//             Some((self.samples.index_axis(Axis(0), self.index-1), self.labels.index_axis(Axis(0), self.index-1)))
//         }
//     }
// }
