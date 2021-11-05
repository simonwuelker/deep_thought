/// Stuff about allocating/freeing heap memory
use crate::Array;
use std::alloc::{alloc, dealloc, Layout};

impl<T, const N: usize> Array<T, N> {
    /// Create an uninitialized Array.
    ///
    /// # Safety
    ///
    /// This is unsafe because it leaves the arrays contents uninitialized, meaning that reading from them will
    /// cause undefined behaviour.
    pub unsafe fn uninitialized(dim: [usize; N]) -> Self {
        let stride = std::mem::size_of::<T>();
        let layout = Layout::array::<T>(dim.iter().product()).unwrap();
        let ptr = alloc(layout.clone()) as *mut T;

        Self {
            ptr: ptr,
            stride: stride,
            dim: dim,
        }
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
}
