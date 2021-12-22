//! Contains the [BaseArray] type as well as the [BorrowedArray] Subtype.

use crate::array_trait::{Array, Initialize};

/// A n-dimensional array
pub struct BaseArray<T, const N: usize> {
    /// Pointer to the first element
    ptr: *mut T,
    /// Number of bytes between successive elements. Different Axes are allowed to have different
    /// strides
    stride: [usize; N],
    /// Axis sizes
    shape: [usize; N],
}

// All attributes are pub(crate) avoid having to define private constructors
/// A view of an [Array], containing references to the original data.
pub struct BorrowedArray<T, const N: usize> {
    /// Reference to the first element
    pub(crate) ptr: *mut T,
    /// Number of bytes between successive elements. Different Axes are allowed to have different
    /// strides
    pub(crate) stride: [usize; N],
    /// Axis sizes
    pub(crate) shape: [usize; N],
}

/// Type alias for one-dimensional arrays
pub type Array1<T> = BaseArray<T, 1>;
/// Type alias for two-dimensional arrays
pub type Array2<T> = BaseArray<T, 2>;
/// Type alias for three-dimensional arrays
pub type Array3<T> = BaseArray<T, 3>;

impl<T, const N: usize> Array<T, N> for BaseArray<T, N> {
    fn ptr(&self) -> *mut T {
        self.ptr
    }

    fn shape(&self) -> [usize; N] {
        self.shape.clone()
    }

    fn stride(&self) -> [usize; N] {
        self.stride.clone()
    }
}

impl<T, const N: usize> Array<T, N> for BorrowedArray<T, N> {
    fn ptr(&self) -> *mut T {
        self.ptr
    }

    fn shape(&self) -> [usize; N] {
        self.shape.clone()
    }

    fn stride(&self) -> [usize; N] {
        self.stride.clone()
    }
}

impl<T: Clone, const N: usize> Initialize<T, N> for BaseArray<T, N> {
    unsafe fn from_raw_parts(ptr: *mut T, stride: [usize; N], shape: [usize; N]) -> Self {
        Self {
            ptr: ptr,
            stride: stride,
            shape: shape,
        }
    }
}

impl<T: PartialEq, const N: usize> PartialEq for BaseArray<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            false
        } else {
            for offset in 0..self.size() {
                // Safe because offset will never exceed self.size() and other.size() == self.size()
                unsafe {
                    if *self._get_unchecked(offset) != *other._get_unchecked(offset) {
                        return false;
                    }
                }
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    /// A wrapper function to allow running code before/after the test.
    /// In this case, we are initializing a logger.
    /// See also https://github.com/rust-lang/rfcs/issues/1664
    #[cfg_attr(feature = "log", ctor::ctor)]
    #[allow(dead_code)]
    fn init_logger() {
        env_logger::init();
    }

    #[test]
    fn index() -> Result<(), error::Error> {
        let mut a: Array1<usize> = Array1::fill(1, &[3]);
        *a.get_mut(&[0])? = 3;
        assert_eq!(*a.get(&[0])?, 3);
        Ok(())
    }

    #[test]
    fn calc_internal_index() -> Result<(), error::Error> {
        let a: Array3<u8> = Array3::fill(1, &[2, 2, 2]);

        assert_eq!(a._get_internal_ix(&[0, 0, 0])?, 0);
        assert_eq!(a._get_internal_ix(&[0, 0, 1])?, 1);
        assert_eq!(a._get_internal_ix(&[0, 1, 0])?, 2);
        assert_eq!(a._get_internal_ix(&[0, 1, 1])?, 3);
        assert_eq!(a._get_internal_ix(&[1, 0, 0])?, 4);
        assert_eq!(a._get_internal_ix(&[1, 0, 1])?, 5);
        assert_eq!(a._get_internal_ix(&[1, 1, 0])?, 6);
        assert_eq!(a._get_internal_ix(&[1, 1, 1])?, 7);
        Ok(())
    }

    #[test]
    fn fill() -> Result<(), error::Error> {
        let a: Array1<u8> = Array1::fill(1, &[3]);
        assert_eq!(*a.get(&[0])?, 1);
        assert_eq!(*a.get(&[1])?, 1);
        assert_eq!(*a.get(&[2])?, 1);
        Ok(())
    }

    // #[test]
    // fn reshape() -> Result<(), error::Error> {
    //     let a: Array<usize, 3> = Array::fill(1, [2, 2, 2]);
    //     let b = a.reshape([1, 8])?;
    //     // assert!(a.reshape([1, 8])? == Array::fill(1, [1, 8]));

    //     Ok(())
    // }

    #[test]
    fn partial_eq() {
        let a: Array1<usize> = Array1::fill(0, &[2]); // equal to a and same object
        let b: Array1<usize> = Array1::fill(0, &[2]); // equal to a and  different object
        let c: Array1<usize> = Array1::fill(1, &[2]); // not equal to a and different object

        // TODO: replace with assert_eq/assert_ne as soon as Debug is implemented
        assert!(a == a);
        assert!(a == b);
        assert!(a != c);
    }

    #[test]
    fn borrow() -> Result<(), error::Error> {
        let mut a: Array2<usize> = Array2::fill(0, &[4, 4]);
        *a.get_mut(&[1, 1])? = 1;

        let b = a.borrow(&[1, 1], &[2, 2])?;

        assert_eq!(b.get(&[0, 0])?, a.get(&[1, 1])?);
        Ok(())
    }
}
