/// Basic arithmetic operations on array (add, sub, mul, div, etc)
use crate::Array;
use std::ops::*;

impl<T: Add<Output = T> + Copy, const N: usize> Add for Array<T, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.dim, other.dim);

        // Safe because we won't be reading from uninitialized memory.
        let mut res: Array<T, N>;
        unsafe {
            res = Array::uninitialized(self.dim);
        }

        for offset in 0..self.size() {
            // safe because offset will never exceed self.size()
            // and both self and other have the same size (as asserted before)
            unsafe {
                *res._get_mut_unchecked(offset) =
                    *self._get_unchecked(offset) + *other._get_unchecked(offset);
            }
        }
        res
    }
}

mod tests {
    use crate::*;

    #[test]
    fn add() {
        let a: Array<usize, 2> = Array::fill(1, [2, 2]);
        let b: Array<usize, 2> = Array::fill(2, [2, 2]);
        let c: Array<usize, 2> = Array::fill(3, [2, 2]);

        assert!(a + b == c);
    }
}
