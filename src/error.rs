use thiserror::Error;
use ndarray::IxDyn;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Mismatched Dimensions: expected {expected:?}, found {found:?}")]
    MismatchedDimensions{expected: IxDyn, found: IxDyn},
}
