use std::fmt::{self, Debug};
use std::ops::*;
use num_traits::{Float, pow::Pow};

// pub struct DualGenerator<I: Iterator<Item=usize>> {
//     iter: I,
// }

// impl Dualgenerator {
//     pub fn new() -> Self {
//         Self {
//             iter: 0..
//         }
//     }
// 
//     /// return a new variable
//     pub fn next(&mut self, val: f32) -> Dual {
//         let e = 
//         Dual {
//             val:
//         }
//     }
// }

#[derive(Debug, Clone, Copy)]
pub struct Dual<F: Float + Debug, const N: usize> {
    pub val: F,
    /// e^2 = 0 but e != 0
    pub e: [F; N],
}

impl<F, const N: usize> Dual<F, N> where
    F: Float + Debug,
    f64: Into<F>, {
    /// Create a constant dual number, meaning it has a derivative
    /// of zero
    pub fn constant(val: F) -> Self {
        Self {
            val: val,
            e: [0_f64.into(); N],
        }
    }

    /// Create a variable dual number, meaning it has a derivative
    /// of one. Every variable must be assigned a unique index
    pub fn variable(val: F, index: usize) -> Self {
        let mut e = [0_f64.into(); N];
        e[index] = 1_f64.into();
        Self { val: val, e: e }
    }

    /// Invert the gradient
    pub fn conjugate(&self) -> Self {
        Self {
            val: self.val,
            e: self.e.map(|a| a.neg()),
        }
    }
}

impl<F: Float + Debug, const N: usize> fmt::Display for Dual<F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut t = f.debug_tuple("Dual number");
        t.field(&self.val);
        for e_val in &self.e {
            t.field(e_val);
        }
        t.finish()
    }
}

impl<F: Float + Debug, const N: usize> Add for Dual<F, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let zipped = &self.e.zip(other.e);
        let e = zipped.map(|(a, b)| a + b);
        Self {
            val: self.val + other.val,
            e: e,
        }
    }
}

impl<F: Float + Debug, const N: usize> Sub for Dual<F, N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let zipped = &self.e.zip(other.e);
        let e = zipped.map(|(a, b)| a + b);
        Self {
            val: self.val + other.val,
            e: e,
        }
    }
}

impl<F: Float + Debug, const N: usize> Mul for Dual<F, N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let e1 = self.e.map(|e| other.val * e);
        let e2 = other.e.map(|e| self.val * e);
        let sum = e1.zip(e2).map(|(a, b)| a + b);

        Self {
            val: self.val * other.val,
            e: sum,
        }
    }
}

impl<F: Float + Debug, const N: usize> Div for Dual<F, N> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let e1 = self.e.map(|e| other.val * e);
        let e2 = other.e.map(|e| self.val * e);
        let res = e1.zip(e2).map(|(a, b)| (a - b) / (other.val * other.val));
        Self {
            val: self.val / other.val,
            e: res,
        }
    }
}

impl<F: Float + Debug, const N: usize> AddAssign for Dual<F, N> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<F: Float + Debug, const N: usize> SubAssign for Dual<F, N> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<F: Float + Debug, const N: usize> MulAssign for Dual<F, N> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<F: Float + Debug, const N: usize> DivAssign for Dual<F, N> {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl<R: Into<F>, F: Float + Debug, const N: usize> Pow<R> for Dual<F, N> 
where F: Float + Debug,
    f64: Into<F> {
    type Output = Self;

    fn pow(self, power: R) -> Self {
        let p: F = power.into();
        let e = self.e.map(|x| x * p * self.val.powf(p - 1_f64.into()));
        Self {
            val: self.val.powf(p),
            e: e,
        }
    }
}

