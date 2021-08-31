use num_traits::*;
use num_traits::float::FloatCore;
use std::fmt;
use std::ops::*;

#[derive(Debug, Clone, Copy)]
pub struct Dual<F, const N: usize> {
    pub val: F,
    /// e^2 = 0 but e != 0
    pub e: [F; N],
}

pub type Dual32<const N: usize> = Dual<f32, N>;
pub type Dual64<const N: usize> = Dual<f64, N>;

impl<F: Copy + Num, const N: usize> From<F> for Dual<F, N> {
    #[inline]
    fn from(x: F) -> Self {
        Self::constant(x)
    }
}

impl<'a, F: Copy + Num, const N: usize> From<&'a F> for Dual<F, N> {
    #[inline]
    fn from(x: &F) -> Self {
        From::from(x.clone())
    }
}

macro_rules! forward_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b, F: Copy + Num, const N: usize> $imp<&'b Dual<F, N>> for &'a Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: &Dual<F, N>) -> Self::Output {
                self.clone().$method(other.clone())
            }
        }
    };
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, F: Copy + Num, const N: usize> $imp<Dual<F, N>> for &'a Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: Dual<F, N>) -> Self::Output {
                self.clone().$method(other)
            }
        }
    };
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, F: Copy + Num, const N: usize> $imp<&'a Dual<F, N>> for Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: &Dual<F, N>) -> Self::Output {
                self.$method(other.clone())
            }
        }
    };
}

macro_rules! forward_all_binop {
    (impl $imp:ident, $method:ident) => {
        forward_ref_ref_binop!(impl $imp, $method);
        forward_ref_val_binop!(impl $imp, $method);
        forward_val_ref_binop!(impl $imp, $method);
    };
}

impl<F: Num + Copy, const N: usize> Dual<F, N>
{
    /// Create a new dual number, providing both its real part and the derivatives
    pub fn new(val: F, e: [F; N]) -> Self {
        Dual {
            val: val,
            e: e,
        }
    }

    /// Create a constant dual number, meaning it has a derivative
    /// of zero
    pub fn constant(val: F) -> Self {
        Self {
            val: val,
            e: [F::zero().clone(); N],
        }
    }

    /// Create a variable dual number, meaning it has a derivative
    /// of one. Every variable must be assigned a unique index
    pub fn variable(val: F, index: usize) -> Self {
        let mut e = [F::zero(); N];
        e[index] = F::one();
        Self { val: val, e: e }
    }
}

impl<F: Num + Neg<Output = F> + Copy, const N: usize> Dual<F, N> {
    /// Invert the gradient
    pub fn conj(&self) -> Self {
        Self {
            val: self.val.clone(),
            e: self.e.clone().map(|a| a.neg()),
        }
    }
}

impl<F: Num + fmt::Debug, const N: usize> fmt::Display for Dual<F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut t = f.debug_tuple("Dual");
        t.field(&self.val);
        for e_val in &self.e {
            t.field(e_val);
        }
        t.finish()
    }
}

impl<F: FloatCore, const N: usize> Dual<F, N> {
    /// Checks if the given Dual number is NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        self.val.is_nan() || self.e.iter().any(|x| x.is_nan())
    }

    /// Checks if the given Dual number is infinite
    #[inline]
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.val.is_infinite() || self.e.iter().any(|x| x.is_infinite()))
    }

    /// Checks if the given Dual number is finite
    #[inline]
    pub fn is_finite(self) -> bool {
        self.val.is_finite() && self.e.iter().all(|x| x.is_finite())
    }

    /// Checks if the given Dual number is normal
    #[inline]
    pub fn is_normal(self) -> bool {
        self.val.is_normal() && self.e.iter().all(|x| x.is_normal())
    }
}

// Basic Numeric Operations with Dual
impl<F: Num + Copy, const N: usize> Add for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let zipped = self.e.clone().zip(other.e);
        let e = zipped.clone().map(|(a, b)| a + b);
        Self {
            val: self.val + other.val,
            e: e,
        }
    }
}
forward_all_binop!(impl Add, add);

impl<F: Num + Copy, const N: usize> Sub for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        let zipped = self.e.clone().zip(other.e);
        let e = zipped.clone().map(|(a, b)| a - b);
        Self {
            val: self.val - other.val,
            e: e,
        }
    }
}
forward_all_binop!(impl Sub, sub);

impl<F: Num + Copy, const N: usize> Mul for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        let e1 = self.e.clone().map(|e| other.val.clone() * e);
        let e2 = other.e.map(|e| self.val.clone() * e);
        let sum = e1.zip(e2).map(|(a, b)| a + b);

        Self {
            val: self.val * other.val,
            e: sum,
        }
    }
}
forward_all_binop!(impl Mul, mul);

impl<F: Num + Copy, const N: usize> Div for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        let e1 = self.e.clone().map(|e| other.val.clone() * e);
        let e2 = other.e.clone().map(|e| self.val.clone() * e);
        Self {
            val: self.val / other.clone().val,
            e: e1.zip(e2).map(|(a, b)| (a - b) / (other.val.clone() * other.val.clone())),
        }
    }
}
forward_all_binop!(impl Div, div);

macro_rules! real_arithmetic {
    (@forward $imp:ident::$method:ident for $($real:ident),*) => (
        impl<'a, F: Copy + Num, const N: usize> $imp<&'a F> for Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: &F) -> Self::Output {
                self.$method(other.clone())
            }
        }
        impl<'a, F: Copy + Num, const N: usize> $imp<F> for &'a Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: F) -> Self::Output {
                self.clone().$method(other)
            }
        }
        impl<'a, 'b, F: Copy + Num, const N: usize> $imp<&'a F> for &'b Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: &F) -> Self::Output {
                self.clone().$method(other.clone())
            }
        }
        $(
            impl<'a, const N: usize> $imp<&'a Dual<$real, N>> for $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn $method(self, other: &Dual<$real, N>) -> Dual<$real, N> {
                    self.$method(other.clone())
                }
            }
            impl<'a, const N: usize> $imp<Dual<$real, N>> for &'a $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn $method(self, other: Dual<$real, N>) -> Dual<$real, N> {
                    self.clone().$method(other)
                }
            }
            impl<'a, 'b, const N: usize> $imp<&'a Dual<$real, N>> for &'b $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn $method(self, other: &Dual<$real, N>) -> Dual<$real, N> {
                    self.clone().$method(other.clone())
                }
            }
        )*
    );
    ($($real:ident),*) => (
        real_arithmetic!(@forward Add::add for $($real),*);
        real_arithmetic!(@forward Sub::sub for $($real),*);
        real_arithmetic!(@forward Mul::mul for $($real),*);
        real_arithmetic!(@forward Div::div for $($real),*);
        real_arithmetic!(@forward Rem::rem for $($real),*);

        $(
            impl<const N: usize> Add<Dual<$real, N>> for $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn add(self, other: Dual<$real, N>) -> Self::Output {
                    Self::Output::new(self + other.val, other.e)
                }
            }

            impl<const N: usize> Sub<Dual<$real, N>> for $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn sub(self, other: Dual<$real, N>) -> Self::Output  {
                    unimplemented!();
                    // Self::Output::new(self - other.val, other.e.map(|x| x.neg()))
                }
            }

            impl<const N: usize> Mul<Dual<$real, N>> for $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn mul(self, other: Dual<$real, N>) -> Self::Output {
                    Self::Output::new(self * other.val, other.e.map(|x| self * x))
                }
            }

            impl<const N: usize> Div<Dual<$real, N>> for $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn div(self, other: Dual<$real, N>) -> Self::Output {
                    Self::Output::new(self / other.val, [$real::zero(); N])
                }
            }

            impl<const N: usize> Rem<Dual<$real, N>> for $real {
                type Output = Dual<$real, N>;

                #[inline]
                fn rem(self, other: Dual<$real, N>) -> Self::Output {
                    Self::Output::constant(self) % other
                }
            }
        )*
    );
}

impl<F: Copy + Num, const N: usize> Add<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn add(self, other: F) -> Self::Output {
        Self::Output::new(self.val + other, self.e)
    }
}

impl<F: Copy + Num, const N: usize> Sub<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn sub(self, other: F) -> Self::Output {
        Self::Output::new(self.val - other, self.e)
    }
}

impl<F: Copy + Num, const N: usize> Mul<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn mul(self, other: F) -> Self::Output {
        Self::Output::new(self.val * other.clone(), self.e.map(|x| x * other.clone()))
    }
}

impl<F: Copy + Num, const N: usize> Div<F> for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn div(self, other: F) -> Self::Output {
        Self::Output::new(self.val / other.clone(), self.e.map(|x| x / other.clone()))
    }
}

impl<F: Copy + Num, const N: usize> Rem<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn rem(self, other: F) -> Self::Output {
        Self::Output::new(self.val % other.clone(), self.e.map(|x| x / other.clone()))
    }
}

real_arithmetic!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32, f64);

mod opassign {
    use core::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
    use crate::autograd::Dual;
    use num_traits::*;

    impl<F: NumAssign + Copy, const N: usize> AddAssign for Dual<F, N> {
        fn add_assign(&mut self, other: Self) {
            let zipped = self.e.clone().zip(other.e);
            self.val += other.val;
            self.e = zipped.clone().map(|(a, b)| a + b);
        }
    }

    impl<F: NumAssign + Copy, const N: usize> SubAssign for Dual<F, N> {
        fn sub_assign(&mut self, other: Self) {
            let zipped = self.e.clone().zip(other.e);
            self.val -= other.val;
            self.e = zipped.clone().map(|(a, b)| a - b);
        }
    }

    impl<F: NumAssign + Copy, const N: usize> MulAssign for Dual<F, N> {
        fn mul_assign(&mut self, other: Self) {
            let e1 = self.e.clone().map(|e| other.val.clone() * e);
            let e2 = other.e.map(|e| self.val.clone() * e);
            self.val *= other.val;
            self.e = e1.zip(e2).map(|(a, b)| a + b);
        }
    }

    impl<F: NumAssign + Copy, const N: usize> DivAssign for Dual<F, N> {
        fn div_assign(&mut self, other: Self) {
            let e1 = self.e.clone().map(|e| other.val.clone() * e);
            let e2 = other.e.clone().map(|e| self.val.clone() * e);
            self.e = e1.zip(e2).map(|(a, b)| (a - b) / (other.val.clone() * other.val.clone()));
            self.val /= other.val;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> RemAssign for Dual<F, N> {
        fn rem_assign(&mut self, other: Self) {
            let e1 = self.e.clone().map(|e| other.val.clone() * e);
            let e2 = other.e.clone().map(|e| self.val.clone() * e);
            let res = e1.zip(e2).map(|(a, b)| (a - b) / (other.val.clone() * other.val.clone()));
            self.val %= other.val;
            self.e = res;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> AddAssign<F> for Dual<F, N> {
        fn add_assign(&mut self, other: F) {
            self.val += other;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> SubAssign<F> for Dual<F, N> {
        fn sub_assign(&mut self, other: F) {
            self.val -= other;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> MulAssign<F> for Dual<F, N> {
        fn mul_assign(&mut self, other: F) {
            self.e = self.e.clone().map(|x| x * other.clone());
            self.val *= other;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> DivAssign<F> for Dual<F, N> {
        fn div_assign(&mut self, other: F) {
            self.e = self.e.clone().map(|x| x / other.clone());
            self.val /= other;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> RemAssign<F> for Dual<F, N> {
        fn rem_assign(&mut self, other: F) {
            self.e = self.e.clone().map(|x| x / other.clone());
            self.val %= other;
        }
    }

    macro_rules! forward_op_assign {
        (impl $imp:ident, $method:ident) => {
            impl<'a, F: Copy + NumAssign, const N: usize> $imp<&'a Dual<F, N>>
                for Dual<F, N>
            {
                #[inline]
                fn $method(&mut self, other: &Self) {
                    self.$method(other.clone())
                }
            }
            impl<'a, F: Copy + NumAssign, const N: usize> $imp<&'a F> for Dual<F, N> {
                #[inline]
                fn $method(&mut self, other: &F) {
                    self.$method(other.clone())
                }
            }
        };
    }

    forward_op_assign!(impl AddAssign, add_assign);
    forward_op_assign!(impl SubAssign, sub_assign);
    forward_op_assign!(impl MulAssign, mul_assign);
    forward_op_assign!(impl DivAssign, div_assign);
    forward_op_assign!(impl RemAssign, rem_assign);
}

impl<F: Num + Neg<Output = F>, const N: usize> Neg for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Dual {
            val: self.val.neg(),
            e: self.e,
        }
    }
}

impl<'a, F: Num + Neg<Output = F> + Copy, const N: usize> Neg for &'a Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Dual {
            val: self.val.clone().neg(),
            e: self.e.clone(),
        }
    }
}

// impl<R: Into<F>, F: Num + fmt::Debug, const N: usize> Pow<R> for Dual<F, N>
// where
//     f64: Into<F>,
// {
//     type Output = Self;
//     fn pow(self, power: R) -> Self::Output {
//         let p: F = power.into();
//         let e = self.e.map(|x| x * p * self.val.powf(p - 1_f64.into()));
//         Self {
//             val: self.val.pow(p),
//             e: e,
//         }
//     }
// }

impl<F: Num + Copy, const N: usize> Zero for Dual<F, N>
{
    fn zero() -> Self {
        Dual {
            val: F::zero(),
            e: [F::zero(); N],
        }
    }

    fn is_zero(&self) -> bool {
        self.val.is_zero() && self.e.iter().all(|x| x.is_zero())
    }
}

impl<F: Num + Copy, const N: usize> One for Dual<F, N>
{
    fn one() -> Self {
        Dual {
            val: F::one(),
            e: [F::zero(); N],
        }
    }

    fn is_one(&self) -> bool {
        self.val.is_one() && self.e.iter().all(|x| x.is_zero())
    }
}

impl<F: Num, const N: usize> PartialEq for Dual<F, N> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<F: Num, const N: usize> Rem for Dual<F, N> {
    type Output = Self;
    fn rem(self, other: Self) -> Self::Output {
        Dual {
            val: self.val % other.val,
            e: self.e,
        }
    }
}
