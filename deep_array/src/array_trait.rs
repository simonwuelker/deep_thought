//! Contains the core [ArrayTrait] trait, which is the foundation of all matrix operations.
//! It is implemented by [crate::array::Array] and [crate::array::BorrowedArray].
//! Users can define their own array types by implementing [ArrayTrait].

use crate::allocation::{stride_packed, stride_strided};
use crate::array::BorrowedArray;
use crate::error::Error;
use std::alloc::{alloc, Layout};

/// Trait defining core Array behaviour
pub trait Array<T, const N: usize> {
    /// Get the shape of the array
    fn shape(&self) -> [usize; N];

    /// Get the stride of the array. The Stride refers to the number of bytes between successive
    /// elements in memory.
    fn stride(&self) -> [usize; N];

    /// Get a pointer to the first element in the array
    fn ptr(&self) -> *mut T;

    /// Return a immutable reference to an object within the array.
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    fn _get(&self, offset: usize) -> Result<&T, Error> {
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
    unsafe fn _get_unchecked(&self, offset: usize) -> &T {
        &*self.ptr().add(offset)
    }

    /// Return a mutable reference to an object stored within the array
    ///
    /// # Safety
    ///
    /// This is unsafe because there are no boundary checks for the offset, meaning that its possible to
    /// read past the end, causing undefined behaviour.
    unsafe fn _get_mut_unchecked(&mut self, offset: usize) -> &mut T {
        &mut *self.ptr().add(offset)
    }

    /// Get the internal byte offset from dimension indices and array stride
    /// or an Error if the index is out of bounds
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    fn _get_internal_ix(&self, ix: &[usize; N]) -> Result<usize, Error> {
        let mut internal_ix = 0;
        for (axis_ix, (ix, axis_size)) in ix.zip(self.shape()).iter().enumerate() {
            if ix >= axis_size {
                return Err(Error::IndexOutOfBounds {
                    ix: *ix,
                    axis_ix: axis_ix,
                    axis_size: *axis_size,
                });
            } else {
                internal_ix += ix * self.stride()[axis_ix];
            }
        }
        Ok(internal_ix)
    }

    /// Return a mutable reference to an object stored within the array
    ///
    /// # Errors
    ///
    /// If the any of the provided indices is outside of the array bounds, this function will return
    /// an error
    fn _get_mut(&mut self, offset: usize) -> Result<&mut T, Error> {
        if offset < self.size() {
            unsafe { Ok(self._get_mut_unchecked(offset)) }
        } else {
            Err(Error::OffsetOutOfBounds {
                offset: offset,
                bound: self.size(),
            })
        }
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
    /// use deep_array::*;
    ///
    /// # fn main() -> Result<(), deep_array::error::Error> {
    /// let a: Array3<usize> = Array3::fill(0, &[2, 2, 2]);
    /// assert_eq!(*a.get(&[0, 0, 0])?, 0);
    /// #   Ok(())
    /// # }
    /// ```
    fn get(&self, ix: &[usize; N]) -> Result<&T, Error> {
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
    /// use deep_array::*;
    ///
    /// # fn main() -> Result<(), deep_array::error::Error> {
    /// let mut a: Array3<usize> = Array3::fill(0, &[2, 2, 2]);
    /// *a.get_mut(&[0, 0, 0]).unwrap() = 1;
    /// assert_eq!(*a.get(&[0, 0, 0])?, 1);
    /// #   Ok(())
    /// # }
    /// ```
    fn get_mut(&mut self, ix: &[usize; N]) -> Result<&mut T, Error> {
        let internal_ix = self._get_internal_ix(&ix)?;
        // Safe because internal_ix is boundary checked
        return unsafe { Ok(self._get_mut_unchecked(internal_ix)) };
    }

    /// Get the number of elements in the array
    ///
    /// # Examples
    /// ```
    /// use deep_array::*;
    ///
    /// let a: Array3<usize> = Array3::fill(0, &[2, 2, 2]);
    /// assert_eq!(a.size(), 8);
    /// ```
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    // TODO: This trait does NOT belong here
    // /// Create a new instance of [Array] where every element is a clone of item.
    // ///
    // /// # Examples
    // /// ```
    // /// use deep_array::Array;
    // ///
    // /// let a: Array3<usize> = Array3::fill(0, [2, 2, 2]);
    // /// ```
    // fn fill(item: T, shape: &[usize; N]) -> Self
    // where
    //     T: Clone,
    // {
    //     // safe because we wont be reading from the uninitialized memory
    //     let mut a: Self;
    //     unsafe {
    //         a = Self::uninitialized(shape);
    //     }
    //     for offset in 0..a.size() {
    //         // safe because the offset never exceeds the array size
    //         unsafe { *a._get_mut_unchecked(offset) = item.clone() }
    //     }
    //     a
    // }

    /// Create a view of the array, containing references to the original data. Data must be
    /// borrowed in rectangular shape.
    ///
    /// # Panics
    /// This function will panic if any of the indices in i1 are larger than the indices in i2.
    /// Thats because i1 represents the lower bound, and i2 represents the upper bound.
    /// Note that it **is** allowed for the lower bound to equal the upper bound)
    fn borrow(&self, i1: &[usize; N], i2: &[usize; N]) -> Result<BorrowedArray<T, N>, Error> {
        assert!(i1.iter().zip(i2.iter()).all(|(a, b)| a <= b));

        let borrow_dims: Vec<usize> = i1.iter().zip(i2.iter()).map(|(a, b)| b - a).collect();
        Ok(BorrowedArray {
            ptr: unsafe { self.ptr().add(self._get_internal_ix(i1)?) },
            stride: self.stride().to_owned(),
            shape: borrow_dims.try_into().unwrap(),
        })
    }

    // /// Try to broadcast the array into another shape. Return an Error if the shapes are incompatible.
    // /// This operation does not change the actual values in memory, only the array dimensions change.
    // fn reshape<const M: usize>(self, dim: [usize; M]) -> Result<Array<T, M>, Error> {
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

/// A Trait defined by array that can be initialized.
/// (Borrowed Arrays cannot be created from scratch and therefore do not implement [Initialize])
pub trait Initialize<T: Clone, const N: usize>: Array<T, N> + Sized {
    /// Build an instance of `Self` from a pointer, the stride and the shape.
    ///
    /// # Safety
    /// Implementors are neither required nor supposed to check if the address `ptr` is pointing to
    /// is a valid array. As such, [from_raw_parts] is unsafe.
    unsafe fn from_raw_parts(ptr: *mut T, stride: [usize; N], shape: [usize; N]) -> Self;

    /// Create an uninitialized Array.
    ///
    /// # Safety
    ///
    /// This is unsafe because it leaves the arrays contents uninitialized, meaning that reading from them will
    /// cause undefined behaviour.
    unsafe fn uninitialized(shape: &[usize; N]) -> Self {
        let layout = Layout::array::<T>(shape.iter().product()).unwrap();
        let ptr = alloc(layout.clone()) as *mut T;

        Self::from_raw_parts(ptr, stride_packed(&shape, std::mem::size_of::<T>()), shape.to_owned())
    }

    /// Create a new instance of [Array] where every element is a clone of item.
    ///
    /// # Examples
    /// ```
    /// use deep_array::*;
    ///
    /// let a: Array3<usize> = Array3::fill(0, &[2, 2, 2]);
    /// ```
    fn fill(item: T, shape: &[usize; N]) -> Self {
        // safe because we wont be reading from the uninitialized memory
        let mut a: Self;
        unsafe {
            a = Self::uninitialized(shape);
        }

        for offset in 0..a.size() {
            // safe because the offset never exceeds the array size
            unsafe { *a._get_mut_unchecked(offset) = item.clone() }
        }
        a
    }
}
