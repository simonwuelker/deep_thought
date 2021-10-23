#![feature(array_zip)]
mod error;

use crate::error::Error;
use std::alloc::{alloc, dealloc, Layout};

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
    /// Create an uninitialized Array
    pub fn new(dim: [usize; N]) -> Self {
        let stride = std::mem::size_of::<T>();
        let ptr = unsafe {
            let layout = Layout::from_size_align_unchecked(dim.iter().product(), stride);
            alloc(layout) as *mut T
        };
        Self {
            ptr: ptr,
            stride: stride,
            dim: dim,
        }
    }

    /// Return a immutable reference to an object within the array
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
    unsafe fn _get_unchecked(&self, offset: usize) -> &T {
        &*self.ptr.add(offset)
    }

    /// Return a mutable reference to an object stored within the array
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
    unsafe fn _get_mut_unchecked(&mut self, offset: usize) -> &mut T {
        &mut *self.ptr.add(offset)
    }

    /// Get the internal element offset from dimension indices or an Error if the index is out of bounds
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
    pub fn get(&self, ix: [usize; N]) -> Result<&T, Error> {
        let internal_ix = self._get_internal_ix(ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_unchecked(internal_ix)) };
    }

    /// Return a mutable reference to an element at an Index or an Error if the index is out of bounds
    pub fn get_mut(&mut self, ix: [usize; N]) -> Result<&mut T, Error> {
        let internal_ix = self._get_internal_ix(ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_mut_unchecked(internal_ix)) };
    }

    /// Get the number of elements in the array
    pub fn size(&self) -> usize {
        self.dim.iter().product()
    }
}

impl<T, const N: usize> Drop for Array<T, N> {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.ptr as *mut u8,
                Layout::from_size_align_unchecked(self.dim.iter().product(), self.stride),
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn alloc_dealloc() {
        let mut a: Array<usize, 1> = Array::new([3]);
        {
            let ref_1 = a._get_mut(0).unwrap();
            *ref_1 = 3;
        }
        let ref_2 = a._get(0).unwrap();
        assert!(*ref_2 == 3);
    }

    #[test]
    fn index() {
        let mut a: Array<usize, 1> = Array::new([3]);
        *a.get_mut([0]).unwrap() = 3;
        assert!(*a.get([0]).unwrap() == 3);
    }

    #[test]
    fn calc_internal_index() {
        let a: Array<usize, 3> = Array::new([2, 2, 2]);
        assert_eq!(a._get_internal_ix([0, 0, 0]).unwrap(), 0);
        assert_eq!(a._get_internal_ix([0, 0, 1]).unwrap(), 1);
        assert_eq!(a._get_internal_ix([0, 1, 0]).unwrap(), 2);
        assert_eq!(a._get_internal_ix([0, 1, 1]).unwrap(), 3);
        assert_eq!(a._get_internal_ix([1, 0, 0]).unwrap(), 4);
        assert_eq!(a._get_internal_ix([1, 0, 1]).unwrap(), 5);
        assert_eq!(a._get_internal_ix([1, 1, 0]).unwrap(), 6);
        assert_eq!(a._get_internal_ix([1, 1, 1]).unwrap(), 7);
    }
}
