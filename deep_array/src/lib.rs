#![feature(array_zip)]
#[deny(missing_docs, undocumented_unsafe_blocks)]

/// deep_array provides the Array<T, N> type, which is a n-dimensional array.

mod error;
mod allocation;

use crate::error::Error;

/// A n-dimensional array
pub struct Array<T, const N: usize> {
    /// Pointer to the first element
    ptr: *mut T,
    /// Number of bytes between two elements
    stride: usize,
    /// Number of elements within the vector
    dim: [usize; N],
}

impl<T, const N: usize> Array<T, N> {
    /// Return a immutable reference to an object within the array
    /// 
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// Err(Error::IndexOutOfBounds)
    fn _get(&self, offset: usize) -> Result<&T, Error> {
        if offset < self.size() {
            unsafe { Ok(&*self.ptr.add(offset)) }
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
    unsafe fn _get_unchecked(&self, offset: usize) -> &T {
        &*self.ptr.add(offset)
    }

    /// Return a mutable reference to an object stored within the array
    /// 
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// Err(Error::IndexOutOfBounds)
    fn _get_mut(&mut self, offset: usize) -> Result<&mut T, Error> {
        if offset < self.size() {
            unsafe { Ok(&mut *self.ptr.add(offset)) }
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
    unsafe fn _get_mut_unchecked(&mut self, offset: usize) -> &mut T {
        &mut *self.ptr.add(offset)
    }

    /// Get the internal element offset from dimension indices or an Error if the index is out of bounds
    /// 
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// Err(Error::IndexOutOfBounds)
    fn _get_internal_ix(&self, ix: [usize; N]) -> Result<usize, Error> {
        let mut internal_ix = 0;
        for (axis_ix, (ix, axis_size)) in ix.zip(self.dim).iter().enumerate() {
            if ix >= axis_size {
                return Err(Error::IndexOutOfBounds {
                    ix: *ix,
                    axis_ix: axis_ix,
                    axis_size: *axis_size,
                });
            } else {
                let tmp_ix: usize = self.dim.iter().skip(axis_ix + 1).product::<usize>() * ix;
                internal_ix += tmp_ix;
            }
        }
        Ok(internal_ix)
    }

    /// Return a reference to an element at an Index or an Error if the index is out of bounds
    /// 
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// Err(Error::IndexOutOfBounds)
    pub fn get(&self, ix: [usize; N]) -> Result<&T, Error> {
        let internal_ix = self._get_internal_ix(ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_unchecked(internal_ix)) };
    }

    /// Return a mutable reference to an element at an Index or an Error if the index is out of bounds
    /// 
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// Err(Error::IndexOutOfBounds)
    pub fn get_mut(&mut self, ix: [usize; N]) -> Result<&mut T, Error> {
        let internal_ix = self._get_internal_ix(ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_mut_unchecked(internal_ix)) };
    }

    /// Get the number of elements in the array
    pub fn size(&self) -> usize {
        self.dim.iter().product()
    }

    // /// Try to broadcast the array into another shape. Return an Error if the shapes are incompatible.
    // /// This operation does not change the actual values in memory, only the array dimensions change.
    // pub fn reshape(&mut self, dim: &[usize]) -> Result<(), Error> {
    //     if self.size() != dim.iter().product() {
    //         Err(Error::IncompatibleShape {
    //             size: self.size(),
    //             new_shape: dim.to_vec(),
    //         })
    //     }
    //     self.dim = dim;
    //     Ok(())
    // }

}


#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn index() {
        // Safe because we won't be reading from uninitialized memory. TODO: make this safe
        let mut a: Array<usize, 1>;
        unsafe {
            a = Array::uninitialized([3]);
        }
        *a.get_mut([0]).unwrap() = 3;
        assert_eq!(*a.get([0]).unwrap(), 3);
    }

    #[test]
    fn calc_internal_index() {
        // Safe because we won't be reading from uninitialized memory. TODO: make this safe
        let a: Array<usize, 3>;
        unsafe {
            a = Array::uninitialized([2, 2, 2]);
        }
        assert_eq!(a._get_internal_ix([0, 0, 0]).unwrap(), 0);
        assert_eq!(a._get_internal_ix([0, 0, 1]).unwrap(), 1);
        assert_eq!(a._get_internal_ix([0, 1, 0]).unwrap(), 2);
        assert_eq!(a._get_internal_ix([0, 1, 1]).unwrap(), 3);
        assert_eq!(a._get_internal_ix([1, 0, 0]).unwrap(), 4);
        assert_eq!(a._get_internal_ix([1, 0, 1]).unwrap(), 5);
        assert_eq!(a._get_internal_ix([1, 1, 0]).unwrap(), 6);
        assert_eq!(a._get_internal_ix([1, 1, 1]).unwrap(), 7);
    }

    // #[test]
    // fn reshape() {
    //     let a: Array<usize, 3> = Array::new([2, 2, 2]);
    //     assert_eq!(a.reshape(&[1, 8]), Ok(()));
    //     assert_eq!(a.reshape(&[1, 8]), Err(_));
    // }
}
