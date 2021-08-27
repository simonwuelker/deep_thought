use crate::error::Error;
use anyhow::Result;
use ndarray::prelude::*;

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
    /// Create a new dataset from the given data. Data is split into training and testing data based on the train_test_split argument.
    /// All Samples and labels are normalized by column, meaning that the mean across a column is always approximately 1
    pub fn new(
        records: Array2<f64>,
        labels: Array2<f64>,
        train_test_split: f64,
        batch_size: BatchSize,
    ) -> Result<Dataset> {
        let record_means = records.mean_axis(Axis(0)).ok_or(Error::NoData)?;
        let label_means = labels.mean_axis(Axis(0)).ok_or(Error::NoData)?;

        // normalization temporarily turned off because debug
        Ok(Dataset {
            train_test_split: train_test_split,
            records: records / &record_means,
            labels: labels / &label_means,
            record_means: record_means,
            label_means: label_means,
            batch_size: batch_size,
        })
    }

    /// Create a new dataset from a given data. Data is split into training and testing data based on the `train_test_split`
    /// argument. Data is not normalized.
    pub fn raw(
        records: Array2<f64>,
        labels: Array2<f64>,
        train_test_split: f64,
        batch_size: BatchSize,
    ) -> Result<Dataset> {
        Ok(Dataset {
            train_test_split: train_test_split,
            record_means: Array1::ones(records.ncols()),
            label_means: Array1::ones(labels.ncols()),
            records: records,
            labels: labels,
            batch_size: batch_size,
        })
    }

    /// Get the number of entries within the dataset
    pub fn length(&self) -> usize {
        self.records.len_of(Axis(0))
    }

    /// Denormalize a batch of record vectors into its original form
    pub fn denormalize_records(&self, normalized: Array2<f64>) -> Array2<f64> {
        normalized * &self.record_means
    }

    /// Denormalize a batch of label vectors into its original form
    pub fn denormalize_labels(&self, normalized: Array2<f64>) -> Array2<f64> {
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
    pub num_batches: usize,
    pub batch_size: usize,
    samples: Array2<f64>,
    labels: Array2<f64>,
}

impl Iterator for SampleIterator {
    type Item = (Array2<f64>, Array2<f64>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.num_batches {
            None
        } else {
            let batched_samples = self
                .samples
                .slice(s![
                    self.index * self.batch_size..(self.index + 1) * self.batch_size,
                    ..
                ])
                .to_owned();
            let batched_labels = self
                .labels
                .slice(s![
                    self.index * self.batch_size..(self.index + 1) * self.batch_size,
                    ..
                ])
                .to_owned();
            self.index += 1;
            Some((
                batched_samples.reversed_axes(),
                batched_labels.reversed_axes(),
            ))
        }
    }
}
// struct SampleIterator<'a> {
//     index: usize,
//     pub num_batches: usize,
//     pub batch_size: usize,
//     samples: ArrayView<'a, f64, Dim<[usize; 2]>>,
//     labels: ArrayView<'a, f64, Dim<[usize; 2]>>
// }
//
// impl<'a> Iterator for SampleIterator<'a> {
//     type Item = (ArrayView<'a, f64, Dim<[usize; 2]>>, ArrayView<'a, f64, Dim<[usize; 2]>>);
//     fn next(&mut self) -> Option<(ArrayView<'a, f64, Dim<[usize; 2]>>, ArrayView<'a, f64, Dim<[usize; 2]>>)> {
//         if self.index > self.num_batches {
//             None
//         }
//         else {
//             let batched_samples: ArrayView<'a, f64, Dim<[usize; 2]>>  = self.samples.slice(s![self.index * self.batch_size..(self.index + 1) * self.batch_size, ..]);
//             let batched_labels = self.labels.slice(s![self.index * self.batch_size..(self.index + 1) * self.batch_size, ..]);
//             self.index += 1;
//             Some((batched_samples, batched_labels))
//         }
//     }
// }
