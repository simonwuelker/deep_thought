use ndarray::IxDyn;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Mismatched Dimensions: expected {expected:?}, found {found:?}")]
    MismatchedDimensions { expected: IxDyn, found: IxDyn },
    #[error("Expected some data but there is none")]
    NoData,
}
