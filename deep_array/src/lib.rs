use std::alloc::{alloc, dealloc, Layout};

/// A one dimensional vector
pub struct Array<T> {
    /// Pointer to the first element
    ptr: *mut T,
    /// Number of bytes between two elements
    stride: usize,
    /// Number of elements within the vector
    dim: usize,
}

impl<T> Array<T> {
    /// Create an uninitialized Array<T>
    pub fn new(len: usize) -> Self {
        let stride = std::mem::size_of::<T>();
        let ptr = unsafe {
            let layout = Layout::from_size_align_unchecked(len, stride);
            alloc(layout) as *mut T
        };
        Self {
            ptr: ptr,
            stride: stride,
            dim: len,
        }
    }

    /// Get a immutable reference to an object within the array
    pub fn get(&self, ix: usize) -> Option<&T> {
        if ix < self.dim {
            unsafe {
                Some(&*self.ptr.add(ix))
            }
        } else {
            None
        }
    }

    /// Get an immutable reference to an object within the array without doing boundary checks
    pub unsafe fn get_unchecked(&self, ix: usize) -> Option<&T> {
        Some(&*self.ptr.add(ix))
    }

    /// Get a mutable reference to an object stored within the array
    pub fn get_mut(&mut self, ix: usize) -> Option<&mut T> {
        if ix < self.dim {
            unsafe {
                Some(&mut *self.ptr.add(ix))
            }
        } else {
            None
        }
    }

    /// Get a mutable reference to an object stored within the array
    pub unsafe fn get_mut_unchecked(&mut self, ix: usize) -> Option<&mut T> {
        Some(&mut *self.ptr.add(ix))
    }

}

impl<T> Drop for Array<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.ptr as *mut u8,
                Layout::from_size_align_unchecked(self.dim, self.stride),
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn alloc_dealloc() {
        let mut a: Array<usize> = Array::new(3);
        {
            let ref_1 = a.get_mut(0).unwrap();
            *ref_1 = 3;
        }
        let ref_2 = a.get(0).unwrap();
        assert!(*ref_2 == 3);
    }
}
