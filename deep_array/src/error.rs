/// Errors that may arise when working with arrays.
use thiserror::Error as ErrorTrait;

#[derive(ErrorTrait, Debug)]
pub enum Error {
    /// An index exceeds the size of the axis
    #[error("Index {ix} is out of bounds for axis {axis_ix} with size {axis_size}")]
    IndexOutOfBounds {
        ix: usize,
        axis_ix: usize,
        axis_size: usize,
    },

    /// An offset exceeds the array size
    #[error("Element offset {offset} exceeds number of elements in the array ({bound})")]
    OffsetOutOfBounds { offset: usize, bound: usize },

    /// Trying to reshape into an incompatible shape
    #[error("Cannot reshape array of size {size} into shape {new_shape:?}")]
    IncompatibleShape {
        size: usize,
        new_shape: Vec<usize>,
    }
}
