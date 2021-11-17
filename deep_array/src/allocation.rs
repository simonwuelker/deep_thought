//! Stuff about allocating/freeing heap memory

use crate::array_trait::Initialize;
use crate::{Array, BaseArray};
use std::alloc::{alloc, dealloc, Layout};

#[cfg(target_pointer_width = "64")]
const WORD_SIZE: usize = 8;
#[cfg(target_pointer_width = "32")]
const WORD_SIZE: usize = 4;

/// pack array elements as tight as possible into on continuous block. This might lead
/// to inefficiencies with array alignment (See [stride_strided])
pub fn stride_packed<const N: usize>(shape: &[usize; N], elem_size: usize) -> [usize; N] {
    let mut stride = [elem_size; N];
    for index in (0..N - 1).rev() {
        stride[index] = shape[index + 1] * stride[index + 1];
    }
    stride
}

/// align array elements to word boundaries so the cpu can load them using less
/// instructions
pub fn stride_strided<const N: usize>(shape: &[usize; N], elem_size: usize) -> [usize; N] {
    // the padded size is the next largest multiple of the word size
    let padded_size = (1 + ((elem_size - 1) / WORD_SIZE)) * WORD_SIZE;

    let mut stride = [padded_size; N];
    for index in (0..N - 1).rev() {
        stride[index] = shape[index + 1] * stride[index + 1];
    }
    stride
}

// FIXME: Other array types are not being dropped properly, this is dangerous!
impl<T, const N: usize> Drop for BaseArray<T, N> {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.ptr() as *mut u8,
                Layout::array::<T>(self.size()).unwrap(),
            )
        };
    }
}

impl<T: Copy, const N: usize> Clone for BaseArray<T, N> {
    fn clone(&self) -> Self {
        let cloned: Self;

        // Safe because we won't be reading from uninitialized memory.
        unsafe {
            cloned = BaseArray::uninitialized(&self.shape());
        }

        // Safe because
        // * T is Copy
        // * self.ptr is valid for self.size() reads
        // * cloned.ptr is valid for self.size() writes
        // * both self and cloned are properly aligned
        // * self and cloned do not overlap
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr(), cloned.ptr(), self.size());
        }

        cloned
    }
}

#[cfg(test)]
mod tests {
    use crate::allocation::{stride_packed, stride_strided};
    use crate::*;

    #[test]
    fn alloc_dealloc() {
        let mut a: Array1<usize>;
        // Safe because we won't be reading from uninitialized memory.
        unsafe {
            a = Array1::uninitialized(&[2]);
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
        let mut a: Array1<usize> = Array1::fill(0, &[2]);
        let mut b = a.clone();

        // assert that the arrays can be mutated independently of each other
        *a.get_mut(&[1]).unwrap() = 3;
        *b.get_mut(&[1]).unwrap() = 4;
        assert_eq!(*a.get(&[1]).unwrap(), 3);
        assert_eq!(*b.get(&[1]).unwrap(), 4);
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
