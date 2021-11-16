//! Stuff about allocating/freeing heap memory

use crate::Array;
use std::alloc::{alloc, dealloc, Layout};
use std::ops::Rem;

#[cfg(target_pointer_width = "64")]
const WORD_SIZE: usize = 8;
#[cfg(target_pointer_width = "32")]
const WORD_SIZE: usize = 4;

/// pack array elements as tight as possible into on continuous block. This might lead
/// to inefficiencies with array alignment (See [stride_strided])
pub fn stride_packed<const N: usize>(dim: &[usize; N], elem_size: usize) -> [usize; N] {
    let mut stride = [elem_size; N];
    for index in (0..N - 1).rev() {
        stride[index] = dim[index + 1] * stride[index + 1];
    }
    stride
}

/// align array elements to word boundaries so the cpu can load them using less 
/// instructions
pub fn stride_strided<const N: usize>(dim: &[usize; N], elem_size: usize) -> [usize; N] {
    // TODO: clean this up, this is a really stupid way of doing it
    let add = if elem_size.rem(WORD_SIZE) == 0 {
        0
    } else {
        1
    };
    let padded_size = ((elem_size / WORD_SIZE) + add) * WORD_SIZE;
    let mut stride = [padded_size; N];
    for index in (0..N - 1).rev() {
        stride[index] = dim[index + 1] * stride[index + 1];
    }
    stride
}

impl<T, const N: usize> Array<T, N> {
    /// Create an uninitialized Array.
    ///
    /// # Safety
    ///
    /// This is unsafe because it leaves the arrays contents uninitialized, meaning that reading from them will
    /// cause undefined behaviour.
    pub unsafe fn uninitialized(dim: [usize; N]) -> Self {
        let layout = Layout::array::<T>(dim.iter().product()).unwrap();
        let ptr = alloc(layout.clone()) as *mut T;

        Self {
            ptr: ptr,
            stride: stride_packed(&dim, std::mem::size_of::<T>()),
            dim: dim,
        }
    }
}

impl<T, const N: usize> Drop for Array<T, N> {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.ptr as *mut u8,
                Layout::array::<T>(self.size()).unwrap(),
            )
        };
    }
}

impl<T: Copy, const N: usize> Clone for Array<T, N> {
    fn clone(&self) -> Self {
        let cloned: Self;

        // Safe because we won't be reading from uninitialized memory.
        unsafe {
            cloned = Array::uninitialized(self.dim);
        }

        // Safe because
        // * T is Copy
        // * self.ptr is valid for self.size() reads
        // * cloned.ptr is valid for self.size() writes
        // * both self and cloned are properly aligned
        // * self and cloned do not overlap
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, cloned.ptr, self.size());
        }

        cloned
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use crate::allocation::stride_packed;

    #[test]
    fn alloc_dealloc() {
        // Safe because we won't be reading from uninitialized memory.
        let mut a: Array<usize, 1>;
        unsafe {
            a = Array::uninitialized([2]);
        }
        // Write
        let ref_1 = a._get_mut(0).unwrap();
        *ref_1 = 3;

        // Read
        let ref_2 = a._get(0).unwrap();

        assert_eq!(*ref_2, 3);
    }

    #[test]
    fn clone() {
        let mut a: Array<usize, 1> = Array::fill(0, [2]);
        let mut b = a.clone();

        // assert that the arrays can be mutated independently of each other
        *a.get_mut([1]).unwrap() = 3;
        *b.get_mut([1]).unwrap() = 4;
        assert_eq!(*a.get([1]).unwrap(), 3);
        assert_eq!(*b.get([1]).unwrap(), 4);
    }

    #[test]
    fn test_stride_packed() {
        let stride = stride_packed(&[2, 3, 4], 2);
        assert_eq!(stride, [24, 8, 2]);
    }

    #[test]
    fn test_stride_strided() {
        // TODO: add actual test
        let _stride = stride_strided(&[2, 3, 4], 2);
    }
}

