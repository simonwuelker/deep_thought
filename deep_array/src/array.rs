//! Contains the base [Array] type as well as the [BorrowedArray] Subtype.

use crate::error::Error;

/// A n-dimensional array
pub struct Array<T, const N: usize> {
    /// Pointer to the first element
    pub(crate) ptr: *mut T,
    /// Number of bytes between successive elements. Different Axes are allowed to have different
    /// strides
    pub(crate) stride: [usize; N],
    /// Axis sizes
    pub(crate) dim: [usize; N],
}

/// A view of an [Array], containing references to the original data.
pub struct BorrowedArray<T, const N: usize> {
    /// Reference to the first element
    pub(crate) ptr: *mut T,
    /// Number of bytes between successive elements. Different Axes are allowed to have different
    /// strides
    pub(crate) stride: [usize; N],
    /// Axis sizes
    pub(crate) dim: [usize; N],
}

impl<T, const N: usize> Array<T, N> {
    /// Return a immutable reference to an object within the array.
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    pub(crate) fn _get(&self, offset: usize) -> Result<&T, Error> {
        if offset < self.size() {
            unsafe { Ok(self._get_unchecked(offset)) }
        } else {
            Err(Error::OffsetOutOfBounds {
                offset: offset,
                bound: self.size(),
            })
        }
    }

    /// Return an immutable reference to an object within the array without doing boundary checks
    ///
    /// # Safety
    ///
    /// This is unsafe because there are no boundary checks for the offset, meaning that its possible to
    /// read past the end, causing undefined behaviour.
    pub(crate) unsafe fn _get_unchecked(&self, offset: usize) -> &T {
        &*self.ptr.add(offset)
    }

    /// Return a mutable reference to an object stored within the array
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    pub(crate) fn _get_mut(&mut self, offset: usize) -> Result<&mut T, Error> {
        if offset < self.size() {
            unsafe { Ok(self._get_mut_unchecked(offset)) }
        } else {
            Err(Error::OffsetOutOfBounds {
                offset: offset,
                bound: self.size(),
            })
        }
    }

    /// Return a mutable reference to an object stored within the array
    ///
    /// # Safety
    ///
    /// This is unsafe because there are no boundary checks for the offset, meaning that its possible to
    /// read past the end, causing undefined behaviour.
    pub(crate) unsafe fn _get_mut_unchecked(&mut self, offset: usize) -> &mut T {
        &mut *self.ptr.add(offset)
    }

    /// Get the internal byte offset from dimension indices and array stride 
    /// or an Error if the index is out of bounds
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    pub(crate) fn _get_internal_ix(&self, ix: &[usize; N]) -> Result<usize, Error> {
        let mut internal_ix = 0;
        for (axis_ix, (ix, axis_size)) in ix.zip(self.dim).iter().enumerate() {
            if ix >= axis_size {
                return Err(Error::IndexOutOfBounds {
                    ix: *ix,
                    axis_ix: axis_ix,
                    axis_size: *axis_size,
                });
            } else {
                internal_ix += ix * self.stride[axis_ix];
            }
        }
        Ok(internal_ix)
    }

    /// Return a reference to an element at an Index or an Error if the index is out of bounds
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    ///
    /// # Examples
    /// ```
    /// use deep_array::Array;
    ///
    /// # fn main() -> Result<(), deep_array::error::Error> {
    /// let a: Array<usize, 3> = Array::fill(0, [2, 2, 2]);
    /// assert_eq!(*a.get([0, 0, 0])?, 0);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn get(&self, ix: [usize; N]) -> Result<&T, Error> {
        let internal_ix = self._get_internal_ix(&ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_unchecked(internal_ix)) };
    }

    /// Return a mutable reference to an element at an Index or an Error if the index is out of bounds
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    ///
    /// # Examples
    /// ```
    /// use deep_array::Array;
    ///
    /// # fn main() -> Result<(), deep_array::error::Error> {
    /// let mut a: Array<usize, 3> = Array::fill(0, [2, 2, 2]);
    /// *a.get_mut([0, 0, 0]).unwrap() = 1;
    /// assert_eq!(*a.get([0, 0, 0])?, 1);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn get_mut(&mut self, ix: [usize; N]) -> Result<&mut T, Error> {
        let internal_ix = self._get_internal_ix(&ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_mut_unchecked(internal_ix)) };
    }

    /// Get the number of elements in the array
    ///
    /// # Examples
    /// ```
    /// use deep_array::Array;
    ///
    /// let a: Array<usize, 3> = Array::fill(0, [2, 2, 2]);
    /// assert_eq!(a.size(), 8);
    /// ```
    pub fn size(&self) -> usize {
        self.dim.iter().product()
    }

    /// Create a new instance of [Array] where every element is a clone of item.
    ///
    /// # Examples
    /// ```
    /// use deep_array::Array;
    ///
    /// let a: Array<usize, 3> = Array::fill(0, [2, 2, 2]);
    /// ```
    pub fn fill(item: T, shape: [usize; N]) -> Self
    where
        T: Clone,
    {
        // safe because we wont be reading from the uninitialized memory
        let mut a: Self;
        unsafe {
            a = Array::uninitialized(shape);
        }
        for offset in 0..a.size() {
            // safe because the offset never exceeds the array size
            unsafe { *a._get_mut_unchecked(offset) = item.clone() }
        }
        a
    }

    /// Create a view of the array, containing references to the original data. Data must be
    /// borrowed in rectangular shape.
    /// 
    /// # Panics
    /// This function will panic if any of the indices in i1 are larger than the indices in i2. 
    /// Thats because i1 represents the lower bound, and i2 represents the upper bound.
    /// (Not that it **is** allowed for the lower bound to equal the upper bound)
    pub fn borrow(&self, i1: &[usize; N], i2: &[usize; N]) -> Result<BorrowedArray<T, N>, Error> {
        assert!(i1.iter().zip(i2.iter()).all(|(a, b)| a <= b));
        
        let borrow_dims: Vec<usize> = i1.iter().zip(i2.iter()).map(|(a, b)| b - a).collect();
        Ok(BorrowedArray {
            ptr: self.ptr.add(self._get_internal_ix(i1)?),
            stride: self.stride.clone(),
            dim: borrow_dims,
        })
    }

    // /// Try to broadcast the array into another shape. Return an Error if the shapes are incompatible.
    // /// This operation does not change the actual values in memory, only the array dimensions change.
    // pub fn reshape<const M: usize>(self, dim: [usize; M]) -> Result<Array<T, M>, Error> {
    //     if self.size() != dim.iter().product() {
    //         return Err(Error::ReshapeIncompatibleShape {
    //             size: self.size(),
    //             new_shape: dim.to_vec(),
    //         });
    //     }
    //     let self_nodrop = std::mem::ManuallyDrop::new(self);
    //     let res = Ok(Array {
    //         ptr: *ptr,
    //         stride: self.stride,
    //         dim: dim,
    //     });

    //     // Prevent calling [std::ops::Drop] on self as it would deallocate the data now owned by the result,
    //     // causing a double free
    //     // unsafe {
    //     //     std::ptr::drop_in_place(&self.dim as *mut usize);
    //     //     std::mem::forget(self);
    //     // }

    //     return res
    // }
}

impl<T: PartialEq, const N: usize> PartialEq for Array<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.dim != other.dim {
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
    fn init_logger() {
        env_logger::init();
    }

    #[test]
    fn index() -> Result<(), error::Error> {
        let mut a: Array<usize, 1> = Array::fill(1, [3]);
        *a.get_mut([0])? = 3;
        assert_eq!(*a.get([0])?, 3);
        Ok(())
    }

    #[test]
    fn calc_internal_index() -> Result<(), error::Error> {
        let a: Array<u8, 3> = Array::fill(1, [2, 2, 2]);

        assert_eq!(a._get_internal_ix([0, 0, 0])?, 0);
        assert_eq!(a._get_internal_ix([0, 0, 1])?, 1);
        assert_eq!(a._get_internal_ix([0, 1, 0])?, 2);
        assert_eq!(a._get_internal_ix([0, 1, 1])?, 3);
        assert_eq!(a._get_internal_ix([1, 0, 0])?, 4);
        assert_eq!(a._get_internal_ix([1, 0, 1])?, 5);
        assert_eq!(a._get_internal_ix([1, 1, 0])?, 6);
        assert_eq!(a._get_internal_ix([1, 1, 1])?, 7);
        Ok(())
    }

    #[test]
    fn fill() -> Result<(), error::Error> {
        let a: Array<u8, 1> = Array::fill(1, [3]);
        assert_eq!(*a.get([0])?, 1);
        assert_eq!(*a.get([1])?, 1);
        assert_eq!(*a.get([2])?, 1);
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
        let a: Array<usize, 1> = Array::fill(0, [2]); // equal to a and same object
        let b: Array<usize, 1> = Array::fill(0, [2]); // equal to a and  different object
        let c: Array<usize, 1> = Array::fill(1, [2]); // not equal to a and different object

        // TODO: replace with assert_eq/assert_ne as soon as Debug is implemented
        assert!(a == a);
        assert!(a == b);
        assert!(a != c);
    }
}
