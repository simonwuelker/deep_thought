//! Errors that may arise when working with arrays.
use thiserror::Error as ErrorTrait;

/// Errors that may arise when working with arrays.
#[derive(ErrorTrait, Debug)]
pub enum Error {
    /// An index exceeds the size of the axis
    #[error("Index {ix} is out of bounds for axis {axis_ix} with size {axis_size}")]
    IndexOutOfBounds {
        /// The requested
        ix: usize,
        /// The index of the requested axis
        axis_ix: usize,
        /// The size of the axis
        axis_size: usize,
    },

    /// An offset exceeds the array size
    #[error("Element offset {offset} exceeds number of elements in the array ({bound})")]
    OffsetOutOfBounds {
        /// The requested offset
        offset: usize,
        /// The number of elements in the array
        bound: usize,
    },

    /// Trying to reshape into an incompatible shape
    #[error("Cannot reshape array of size {size} into shape {new_shape:?}")]
    ReshapeIncompatibleShape {
        /// Original Shape
        size: usize,
        /// New Shape
        new_shape: Vec<usize>,
    },
}
