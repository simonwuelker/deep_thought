//! Rand implementations for dual numbers

use crate::autograd::Dual;
use num_traits::Num;
use rand::distributions::Standard;
use rand::prelude::*;

impl<F, const N: usize> Distribution<Dual<F, N>> for Standard
where
    F: Num + Copy,
    Standard: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Dual<F, N> {
        Dual::constant(self.sample(rng))
    }
}

/// A generic random value distribution for dual numbers.
#[derive(Clone, Copy, Debug)]
pub struct DualDistribution<D> {
    val: D,
}

impl<D> DualDistribution<D> {
    /// Creates a Dual distribution from an independent Distribution
    pub fn new(val: D) -> Self {
        DualDistribution { val }
    }
}

impl<F, D, const N: usize> Distribution<Dual<F, N>> for DualDistribution<D>
where
    F: Num + Copy,
    D: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Dual<F, N> {
        Dual::constant(F::from(self.val.sample(rng)))
    }
}
