use std::{iter::Sum, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};



#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct c64 {
    pub(crate) a: f64,
    pub(crate) b: f64,
}

impl c64 {
    /// Creates a new complex number (a + bi)
    pub fn new(a: f64, b: f64) -> Self {
        c64 { a, b }
    }

    /// Returns the conjugate of the complex number
    pub fn conjugate(self) -> Self {
        c64 { a: self.a, b: -self.b }
    }

    /// Returns the magnitude (absolute value) of the complex number
    pub fn mag(self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    pub fn re(self) -> f64 {
        self.a
    }

    pub fn im(self) -> f64 {
        self.b
    }

    pub fn arg(self) -> f64 {
        self.b.atan2(self.a)
    }

    pub fn sqrt(self) -> Self {
        let r = self.mag().sqrt();
        let theta = self.arg() / 2.0;
        c64::new(r * theta.cos(), r * theta.sin())
    }

}

// implement zero i ONE
impl c64 {
    pub const ZERO: Self = c64 { a: 0.0, b: 0.0 };
    pub const ONE: Self = c64 { a: 1.0, b: 0.0 };
    pub const I: Self = c64 { a: 0.0, b: 1.0 };
    pub const NEG_I: Self = c64 { a: 0.0, b: -1.0 };

    pub const EPSILON: Self = c64 { a: f64::EPSILON, b: f64::EPSILON };
}

// Implement addition for complex numbers: (a + bi) + (c + di) = (a+c) + (b+d)i
impl Add for c64 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        c64 {
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

// add for f64 + complex
impl Add<c64> for f64 {
    type Output = c64;
    fn add(self, other: c64) -> c64 {
        c64 {
            a: self + other.a,
            b: other.b,
        }
    }
}

// add for complex + f64
impl Add<f64> for c64 {
    type Output = c64;
    fn add(self, other: f64) -> c64 {
        c64 {
            a: self.a + other,
            b: self.b,
        }
    }
}

// Implement subtraction for complex numbers: (a + bi) - (c + di) = (a-c) + (b-d)i
impl Sub for c64 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        c64 {
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

// sub for f64 - complex
impl Sub<c64> for f64 {
    type Output = c64;
    fn sub(self, other: c64) -> c64 {
        c64 {
            a: self - other.a,
            b: -other.b,
        }
    }
}

// sub for complex - f64
impl Sub<f64> for c64 {
    type Output = c64;
    fn sub(self, other: f64) -> c64 {
        c64 {
            a: self.a - other,
            b: self.b,
        }
    }
}

// Implement multiplication for complex numbers:
// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
impl Mul for c64 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        c64 {
            a: self.a * other.a - self.b * other.b,
            b: self.a * other.b + self.b * other.a,
        }
    }
}

// implment mul for f64 * complex
impl Mul<f64> for c64 {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        c64 {
            a: self.a * other,
            b: self.b * other,
        }
    }
}

// implment mul for complex * f64
impl Mul<c64> for f64 {
    type Output = c64;
    fn mul(self, other: c64) -> c64 {
        c64 {
            a: self * other.a,
            b: self * other.b,
        }
    }
}

// Multiply c64 by f32
impl Mul<f32> for c64 {
    type Output = Self;
    fn mul(self, other: f32) -> Self {
        self * (other as f64)
    }
}

impl Mul<c64> for f32 {
    type Output = c64;
    fn mul(self, other: c64) -> c64 {
        (self as f64) * other
    }
}

// Multiply c64 by i32
impl Mul<i32> for c64 {
    type Output = Self;
    fn mul(self, other: i32) -> Self {
        self * (other as f64)
    }
}

impl Mul<c64> for i32 {
    type Output = c64;
    fn mul(self, other: c64) -> c64 {
        (self as f64) * other
    }
}

// Multiply c64 by i64
impl Mul<i64> for c64 {
    type Output = Self;
    fn mul(self, other: i64) -> Self {
        self * (other as f64)
    }
}

impl Mul<c64> for i64 {
    type Output = c64;
    fn mul(self, other: c64) -> c64 {
        (self as f64) * other
    }
}

// Implement division for complex numbers:
// (a + bi) / (c + di) = ((ac + bd) / (c^2+d^2)) + ((bc - ad) / (c^2+d^2))i
impl Div for c64 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let denom = other.a * other.a + other.b * other.b;
        if denom == 0.0 {
            panic!("Division by zero in complex division");
        }
        c64 {
            a: (self.a * other.a + self.b * other.b) / denom,
            b: (self.b * other.a - self.a * other.b) / denom,
        }
    }
}

// Implement division for f64 / complex
impl Div<c64> for f64 {
    type Output = c64;
    fn div(self, other: c64) -> c64 {
        let denom = other.a * other.a + other.b * other.b;
        if denom == 0.0 {
            panic!("Division by zero in complex division");
        }
        c64 {
            a: self * other.a / denom,
            b: -self * other.b / denom,
        }
    }
}

// implent div for complex / f64
impl Div<f64> for c64 {
    type Output = c64;
    fn div(self, other: f64) -> c64 {
        c64 {
            a: self.a / other,
            b: self.b / other,
        }
    }
}

// assignement 
impl AddAssign for c64 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for c64 {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl MulAssign for c64 {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl DivAssign for c64 {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

// partial equality
impl PartialEq<f64> for c64 {
    fn eq(&self, other: &f64) -> bool {
        self.a == *other && self.b == 0.0
    }
}

impl PartialEq<c64> for f64 {
    fn eq(&self, other: &c64) -> bool {
        *self == other.a && 0.0 == other.b
    }
}

// Implement equality for complex numbers: (a + bi) == (c + di) if a == c and b == d
impl PartialEq for c64 {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b
    }
}

// partial ordering
impl PartialOrd<f64> for c64 {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        // throw an error if the imaginary part is not zero
        if self.b != 0.0 {
            None
        } else {
            self.a.partial_cmp(other)
        }
    }
}

// partial ordering for f64
impl PartialOrd<c64> for f64 {
    fn partial_cmp(&self, other: &c64) -> Option<std::cmp::Ordering> {
        // throw an error if the imaginary part is not zero
        if other.b != 0.0 {
            None
        } else {
            self.partial_cmp(&other.a)
        }
    }
}

// Implement ordering for complex numbers: (a + bi) < (c + di) if a < c or (a == c and b < d)
impl PartialOrd for c64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if other.b != 0.0 || self.b != 0.0 {
            None
        } else {
            self.a.partial_cmp(&other.a)
        }
    }
}

// Implement negation for complex numbers: -(a + bi) = (-a) + (-b)i
impl Neg for c64 {
    type Output = Self;
    fn neg(self) -> Self {
        c64 { a: -self.a, b: -self.b }
    }
}

// Implement conversion from a tuple (f64, f64) to c64
impl From<(f64, f64)> for c64 {
    fn from((a, b): (f64, f64)) -> Self {
        c64::new(a, b)
    }
}

// Implement conversion from c64 to a tuple (f64, f64)
impl From<c64> for (f64, f64) {
    fn from(c: c64) -> Self {
        (c.a, c.b)
    }
}

impl From<f64> for c64 {
    fn from(a: f64) -> Self {
        c64::new(a, 0.0)
    }
}

impl From<i64> for c64 {
    fn from(a: i64) -> Self {
        c64::new(a as f64, 0.0)
    }
}

impl From<f32> for c64 {
    fn from(a: f32) -> Self {
        c64::new(a as f64, 0.0)
    }
}


// implement sum
impl Sum for c64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(c64::ZERO, |a, b| a + b)
    }
}

// Add a unified trait for converting various types to c64.
pub trait IntoC64 {
    fn into_c64(self) -> c64;
}

impl IntoC64 for c64 {
    fn into_c64(self) -> c64 {
        self
    }
}

impl IntoC64 for f64 {
    fn into_c64(self) -> c64 {
        c64::new(self, 0.0)
    }
}

impl IntoC64 for f32 {
    fn into_c64(self) -> c64 {
        c64::new(self as f64, 0.0)
    }
}

impl<T: Into<f64>, U: Into<f64>> IntoC64 for (T, U) {
    fn into_c64(self) -> c64 {
        c64::new(self.0.into(), self.1.into())
    }
}

// implement from c64 for f64
impl From<c64> for f64 {
    fn from(c: c64) -> Self {
        c.a
    }
}

// implement from c64 for i64
impl From<c64> for i64 {
    fn from(c: c64) -> Self {
        c.a as i64
    }
}

// implement from c64 for f32
impl From<c64> for f32 {
    fn from(c: c64) -> Self {
        c.a as f32
    }
}




// Optionally, add other implementations following the same pattern...

// Remove old Complexify implementations and add:
pub trait ToComplex {
    fn complex(self) -> c64;
}

impl<T: IntoC64> ToComplex for T {
    fn complex(self) -> c64 {
        self.into_c64()
    }
}

// Extend the conversion for arrays generically.
pub trait ComplexifyArray<const N: usize> {
    fn complex(self) -> [c64; N];
}

impl<T: IntoC64 + Copy, const N: usize> ComplexifyArray<N> for [T; N] {
    fn complex(self) -> [c64; N] {
        std::array::from_fn(|i| self[i].into_c64())
    }
}

// Implement Display for complex numbers
impl std::fmt::Display for c64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.b == 0.0 {
            write!(f, "{}", self.a)
        } else if self.a == 0.0 {
            write!(f, "{}i", self.b)
        } else {
            write!(f, "{} {} {}i", self.a, if self.b < 0.0 { '-' } else { '+' }, self.b.abs())
        }
    }
}