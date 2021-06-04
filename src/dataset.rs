use anyhow::Result;
use ndarray::prelude::*;
use crate::error::Error;

/// Number of training examples to run before optimizing the net once.
/// If the number of examples does not fit evenly,
/// mod(num_example, batchsize) examples are disregarded.
pub enum BatchSize {
    /// Batch gradient descent
    All,
    /// Stochastic gradient descent, equivalent to `BatchSize::Number(1)`
    One,
    /// Mini batch gradient descent
    Number(usize),
}

pub struct Dataset {
    /// Ratio between number of training and number of testing samples
    train_test_split: f64,
    /// Normalized record data data contained by the dataset
    records: Array2<f64>,
    /// Normalized labels to the records
    labels: Array2<f64>,
    /// Mean of record columns, used to de-normalize the records
    record_means: Array1<f64>,
    /// Mean of label columns, used to de-normalize the labels
    label_means: Array1<f64>,
    /// Size of one batch
    batch_size: BatchSize,
}

impl Dataset {
    /// Create a new dataset, normalizing the inputs and labels
    pub fn new(records: Array2<f64>, labels: Array2<f64>, train_test_split: f64, batch_size: BatchSize) -> Result<Dataset> {
        let record_means = records.mean_axis(Axis(0)).ok_or(Error::NoData)?;
        let label_means = labels.mean_axis(Axis(0)).ok_or(Error::NoData)?;
        
        Ok(Dataset {
            train_test_split: train_test_split,
            records: records / &record_means,
            labels: labels / &label_means,
            record_means: record_means,
            label_means: label_means,
            batch_size: batch_size,
        })
    }

    /// Denormalize a record vector into its original form
    pub fn denormalize_record(&self, normalized: Array1<f64>) -> Array1<f64> {
        normalized * &self.record_means
    }

    /// Denormalize a label vector into its original form
    pub fn denormalize_label(&self, normalized: Array1<f64>) -> Array1<f64> {
        normalized * &self.label_means
    }

    /// Return an iterator over training examples/labels in (sample, label) tupels
    pub fn iter_train(&self) -> SampleIterator {
        let num_train = (self.records.nrows() as f64 * self.train_test_split) as usize;

        let batch_size = match self.batch_size {
            BatchSize::One => 1,
            BatchSize::All => num_train,
            BatchSize::Number(num) => num,
        };
        
        SampleIterator {
            index: 0,
            num_batches: num_train.div_euclid(batch_size),
            batch_size: batch_size,
            samples: self.records.slice(s![..num_train, ..]).to_owned(),
            labels: self.labels.slice(s![..num_train, ..]).to_owned(),
        }
    }

    /// Return an iterator over testing examples/labels in (sample, label) tupels
    pub fn iter_test(&self) -> SampleIterator {
        let num_train = (self.records.nrows() as f64 * self.train_test_split) as usize;
        let num_test = self.records.nrows() - num_train;

        let batch_size = match self.batch_size {
            BatchSize::One => 1,
            BatchSize::All => num_test,
            BatchSize::Number(num) => num,
        };

        SampleIterator {
            index: 0,
            num_batches: num_test.div_euclid(batch_size),
            batch_size: batch_size,
            samples: self.records.slice(s![num_train.., ..]).to_owned(),
            labels: self.labels.slice(s![num_train.., ..]).to_owned(),
        }
    }
}

// BIG TODO: use lifetimes and array views here instead of cloning everything, this is slow!
/// An iterator over training/testing data. Yields (samples, labels) pairs where both
/// samples and labels have the shape (num_fields x batch_size)
pub struct SampleIterator {
    index: usize,
    num_batches: usize,
    batch_size: usize,
    samples: Array2<f64>,
    labels: Array2<f64>,
}

impl Iterator for SampleIterator {
    type Item = (Array2<f64>, Array2<f64>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.num_batches {
            None
        }
        else {
            let batched_samples = self.samples.slice(s![self.index * self.batch_size..(self.index + 1) * self.batch_size, ..]).to_owned();
            let batched_labels = self.labels.slice(s![self.index * self.batch_size..(self.index + 1) * self.batch_size, ..]).to_owned();
            self.index += 1;
            Some((batched_samples.reversed_axes(), batched_labels.reversed_axes()))
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
