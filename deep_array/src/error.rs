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
}
