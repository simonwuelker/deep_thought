use num_traits::float::FloatCore;
use num_traits::*;
use std::cmp::Ordering;
use std::fmt;
use std::ops::*;

#[derive(Debug, Clone, Copy)]
pub struct Dual<F, const N: usize> {
    /// real value
    pub val: F,
    /// e^2 = 0 but e != 0
    pub e: [F; N],
}

pub type Dual32<const N: usize> = Dual<f32, N>;
pub type Dual64<const N: usize> = Dual<f64, N>;

impl<F: Num + PartialOrd + Copy, const N: usize> From<F> for Dual<F, N> {
    #[inline]
    fn from(x: F) -> Self {
        Self::constant(x)
    }
}

impl<'a, F: Num + PartialOrd + Copy, const N: usize> From<&'a F> for Dual<F, N> {
    #[inline]
    fn from(x: &F) -> Self {
        From::from(x.clone())
    }
}

macro_rules! forward_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b, F: Num + PartialOrd + Copy, const N: usize> $imp<&'b Dual<F, N>>
            for &'a Dual<F, N>
        {
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
        impl<'a, F: Num + PartialOrd + Copy, const N: usize> $imp<Dual<F, N>> for &'a Dual<F, N> {
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
        impl<'a, F: Num + PartialOrd + Copy, const N: usize> $imp<&'a Dual<F, N>> for Dual<F, N> {
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

impl<F: Num + Copy, const N: usize> Dual<F, N> {
    /// Create a new dual number, providing both its real part and the derivatives
    pub fn new(val: F, e: [F; N]) -> Self {
        Dual { val: val, e: e }
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

impl<F: Num + PartialOrd + Copy + Neg<Output = F>, const N: usize> Dual<F, N> {
    /// Invert the gradient
    pub fn conj(&self) -> Self {
        Self {
            val: self.val.clone(),
            e: self.e.clone().map(|a| a.neg()),
        }
    }
}

impl<F: Num + PartialOrd + Copy + fmt::Debug, const N: usize> fmt::Display for Dual<F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut t = f.debug_tuple("Dual");
        t.field(&self.val);
        for e_val in &self.e {
            t.field(e_val);
        }
        t.finish()
    }
}

impl<F: Num + PartialOrd + FloatCore, const N: usize> Dual<F, N> {
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

// Basic Num + PartialOrd + Copyeric Operations with Dual
impl<F: Num + PartialOrd + Copy, const N: usize> Add for Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy, const N: usize> Sub for Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy, const N: usize> Mul for Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy, const N: usize> Div for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self::Output {
        let e1 = self.e.clone().map(|e| other.val.clone() * e);
        let e2 = other.e.clone().map(|e| self.val.clone() * e);
        Self {
            val: self.val / other.clone().val,
            e: e1
                .zip(e2)
                .map(|(a, b)| (a - b) / (other.val.clone() * other.val.clone())),
        }
    }
}
forward_all_binop!(impl Div, div);

macro_rules! real_arithmetic {
    (@forward $imp:ident::$method:ident for $($real:ident),*) => (
        impl<'a, F: Num + PartialOrd + Copy, const N: usize> $imp<&'a F> for Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: &F) -> Self::Output {
                self.$method(other.clone())
            }
        }
        impl<'a, F: Num + PartialOrd + Copy, const N: usize> $imp<F> for &'a Dual<F, N> {
            type Output = Dual<F, N>;

            #[inline]
            fn $method(self, other: F) -> Self::Output {
                self.clone().$method(other)
            }
        }
        impl<'a, 'b, F: Num + PartialOrd + Copy, const N: usize> $imp<&'a F> for &'b Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy, const N: usize> Add<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn add(self, other: F) -> Self::Output {
        Self::Output::new(self.val + other, self.e)
    }
}

impl<F: Num + PartialOrd + Copy, const N: usize> Sub<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn sub(self, other: F) -> Self::Output {
        Self::Output::new(self.val - other, self.e)
    }
}

impl<F: Num + PartialOrd + Copy, const N: usize> Mul<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn mul(self, other: F) -> Self::Output {
        Self::Output::new(self.val * other.clone(), self.e.map(|x| x * other.clone()))
    }
}

impl<F: Num + PartialOrd + Copy, const N: usize> Div<F> for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn div(self, other: F) -> Self::Output {
        Self::Output::new(self.val / other.clone(), self.e.map(|x| x / other.clone()))
    }
}

impl<F: Num + PartialOrd + Copy, const N: usize> Rem<F> for Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn rem(self, other: F) -> Self::Output {
        Self::Output::new(self.val % other.clone(), self.e.map(|x| x / other.clone()))
    }
}

real_arithmetic!(f32, f64);

mod opassign {
    use crate::autograd::Dual;
    use core::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
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
            self.e = e1
                .zip(e2)
                .map(|(a, b)| (a - b) / (other.val.clone() * other.val.clone()));
            self.val /= other.val;
        }
    }

    impl<F: NumAssign + Copy, const N: usize> RemAssign for Dual<F, N> {
        fn rem_assign(&mut self, other: Self) {
            let e1 = self.e.clone().map(|e| other.val.clone() * e);
            let e2 = other.e.clone().map(|e| self.val.clone() * e);
            let res = e1
                .zip(e2)
                .map(|(a, b)| (a - b) / (other.val.clone() * other.val.clone()));
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
            impl<'a, F: Copy + NumAssign, const N: usize> $imp<&'a Dual<F, N>> for Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy + Neg<Output = F>, const N: usize> Neg for Dual<F, N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Dual {
            val: self.val.neg(),
            e: self.e,
        }
    }
}

impl<'a, F: Num + PartialOrd + Copy + Neg<Output = F>, const N: usize> Neg for &'a Dual<F, N> {
    type Output = Dual<F, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        Dual {
            val: self.val.clone().neg(),
            e: self.e.clone(),
        }
    }
}

// impl<R: Into<F>, F: Num + PartialOrd + Copy + fmt::Debug, const N: usize> Pow<R> for Dual<F, N>
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

impl<F: Num + PartialOrd + Copy, const N: usize> Zero for Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy, const N: usize> One for Dual<F, N> {
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

impl<F: Num + PartialOrd + Copy, const N: usize> Rem for Dual<F, N> {
    type Output = Self;
    fn rem(self, other: Self) -> Self::Output {
        Dual {
            val: self.val % other.val,
            e: self.e,
        }
    }
}

// Auto trait is needed to prevent conflicting trait implementations
pub auto trait NotADual {}
impl<F, const N: usize> !NotADual for Dual<F, N> {}

impl<F: PartialEq, const N: usize> PartialEq for Dual<F, N> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<F, D, const N: usize> PartialEq<D> for Dual<F, N>
where
    F: PartialEq<D>,
    D: NotADual,
{
    fn eq(&self, other: &D) -> bool {
        self.val == *other
    }
}

impl<F: PartialOrd, const N: usize> PartialOrd for Dual<F, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<F, D, const N: usize> PartialOrd<D> for Dual<F, N>
where
    F: PartialOrd<D>,
    D: NotADual,
{
    fn partial_cmp(&self, other: &D) -> Option<Ordering> {
        self.val.partial_cmp(&other)
    }
}

impl<F: Num + Copy + PartialOrd, const N: usize> Num for Dual<F, N> {
    type FromStrRadixErr = <F as Num>::FromStrRadixErr;
    fn from_str_radix(string: &str, radix: u32) -> Result<Self, <Self as Num>::FromStrRadixErr> {
        Ok(Dual::constant(F::from_str_radix(string, radix)?))
    }
}

macro_rules! float_impl_basic {
    ($ty:ty, $($name:ident),*) => {
        $(fn $name() -> Self {
            Dual::constant(<$ty as Float>::$name())
        })*
    }
}

macro_rules! float_impl_passthrough {
    ($result:ty, $($name:ident),*) => {
        $(fn $name(self) -> $result {
            (self.val).$name()
        })*
    }
}

macro_rules! float_impl_self_passthrough {
    ($($name:ident),*) => {
        $(fn $name(self) -> Self {
            Dual::constant(self.val.$name())
        })*
    }
}

impl<F: Float, const N: usize> Float for Dual<F, N> {
    float_impl_basic!(
        F,
        nan,
        infinity,
        neg_infinity,
        neg_zero,
        min_value,
        max_value,
        min_positive_value
    );
    float_impl_passthrough!(
        bool,
        is_nan,
        is_infinite,
        is_finite,
        is_normal,
        is_sign_positive,
        is_sign_negative
    );
    float_impl_passthrough!((u64, i16, i8), integer_decode);
    float_impl_passthrough!(::std::num::FpCategory, classify);
    float_impl_self_passthrough!(floor, ceil, round, trunc);

    fn fract(self) -> Self {
        Dual::new(self.val.fract(), [F::one(); N])
    }

    fn abs(self) -> Self {
        let e = self.e.map(|x| self.val.signum() * x);
        Dual::new(self.val.abs(), e)
    }

    fn signum(self) -> Self {
        Dual::new(self.val.signum(), [F::zero(); N])
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        let e = self.e.zip(a.e).zip(b.e).map(|((d, e), f)| d * a.val + self.val * e * f);
        Dual::new(
            self.val.mul_add(a.val, b.val),
            e,
        )
    }

    fn recip(self) -> Self {
        Dual::<F, N>::one() / self
    }

    fn powi(self, n: i32) -> Self {
        let exp = F::from(n).unwrap();
        let e = self.e.map(|x| exp * x.powi(n - 1));
        Dual::new(self.val.powi(n), e)
    }

    fn powf(self, n: Self) -> Self {
        let val = self.val.powf(n.val);
        let e = self.e.zip(n.e).map(|(a, b)| n.val * self.val.powf(b - F::one()) * a + val * self.val.ln() * b);
        Dual::new(
            val,
            e,
        )
    }

    fn sqrt(self) -> Self {
        let val = self.val.sqrt();
        let e = self.e.map(|x| x / (F::from(2).unwrap() * val));
        Dual::new(val, e)
    }

    fn exp(self) -> Self {
        let val = self.val.exp();
        let e = self.e.map(|x| x * val);
        Dual::new(val, e)
    }

    fn exp2(self) -> Self {
        let val = self.val.exp2();
        let e = self.e.map(|x| val * x * F::from(2).unwrap().ln());
        Dual::new(val, e)
    }

    fn ln(self) -> Self {
        let e = self.e.map(|x| x / self.val);
        Dual::new(self.val.ln(), e)
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        let e = self.e.map(|x| x / (self.val * F::from(10).unwrap().ln()));
        Dual::new(self.val.log2(), e)
    }

    fn log10(self) -> Self {
        let e = self.e.map(|x| x / (self.val * F::from(10).unwrap().ln()));
        Dual::new(self.val.log10(), e)
    }

    fn max(self, other: Self) -> Self {
        let e = if self.val >= other.val {
            self.e
        } else {
            other.e
        };
        Dual::new(self.val.max(other.val), e)
    }

    fn min(self, other: Self) -> Self {
        let e = if self.val <= other.val {
            self.e
        } else {
            other.e
        };
        Dual::new(self.val.min(other.val), e)
    }

    fn abs_sub(self, other: Self) -> Self {
        let e = if self.val > other.val {
            self.e.zip(other.e).map(|(a, b)| a - b)
        } else {
            [F::zero(); N]
        };

        Dual::new((self.val - other.val).max(F::zero()), e)
    }

    fn cbrt(self) -> Self {
        let real = self.val.cbrt();
        let e = self.e.map(|x| x / (F::from(3).unwrap() * real.powi(2)));
        Dual::new(real, e)
    }

    fn hypot(self, other: Self) -> Self {
        let real = self.val.hypot(other.val);
        let zipped = self.e.zip(other.e);
        let e = zipped.map(|(a, b)| (self.val * b + other.val * a) / real);
        Dual::new(real, e)
    }

    fn sin(self) -> Self {
        let e = self.e.map(|x| self.val.cos() * x);
        Dual::new(self.val.sin(), e)
    }

    fn cos(self) -> Self {
        let e = self.e.map(|x| -self.val.sin() * x);
        Dual::new(self.val.cos(), e)
    }

    fn tan(self) -> Self {
        let cos = self.val.cos();
        let e = self.e.map(|x| x / cos.powi(2));
        Dual::new(self.val.tan(), e)
    }

    fn asin(self) -> Self {
        let e = self.e.map(|x| x / (F::one() - self.val.powi(2)).sqrt());
        Dual::new(self.val.asin(), e)
    }

    fn acos(self) -> Self {
        let e = self.e.map(|x| -x / (F::one() - self.val.powi(2)).sqrt());
        Dual::new(self.val.acos(), e)
    }

    fn atan(self) -> Self {
        let e = self.e.map(|x| x / (self.val.powi(2) + F::one()));
        Dual::new(self.val.atan(), e)
    }

    fn atan2(self, other: Self) -> Self {
        let zipped = self.e.zip(other.e);
        let e = zipped
            .map(|(a, b)| (self.val * b - other.val * a) / (self.val.powi(2) + other.val.powi(2)));
        Dual::new(self.val.atan2(other.val), e)
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Self {
        let e = self.e.map(|x| x * self.val.exp());
        Dual::new(self.val.exp_m1(), e)
    }

    fn ln_1p(self) -> Self {
        let e = self.e.map(|x| x / (self.val + F::one()));
        Dual::new(self.val.ln_1p(), e)
    }

    fn sinh(self) -> Self {
        let e = self.e.map(|x| x * self.val.cosh());
        Dual::new(self.val.sinh(), e)
    }

    fn cosh(self) -> Self {
        let e = self.e.map(|x| x * x.sinh());
        Dual::new(self.val.cosh(), e)
    }

    fn tanh(self) -> Self {
        let cosh = self.val.cosh();
        let e = self.e.map(|x| x / cosh.powi(2));
        Dual::new(self.val.tanh(), e)
    }

    fn asinh(self) -> Self {
        let e = self.e.map(|x| x / (self.val.powi(2) + F::one()).sqrt());
        Dual::new(self.val.asinh(), e)
    }

    fn acosh(self) -> Self {
        let e = self
            .e
            .map(|x| x / ((self.val + F::one()).sqrt() * (self.val - F::one()).sqrt()));
        Dual::new(self.val.acosh(), e)
    }

    fn atanh(self) -> Self {
        let e = self.e.map(|x| x / (F::one() - self.val.powi(2)));
        Dual::new(self.val.atanh(), e)
    }
}
