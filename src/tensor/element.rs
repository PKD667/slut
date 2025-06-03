use crate::complex::c64;

// create a trait for weak FP multiplication
pub trait WeakMul {
    fn weak_mul(self, other: f64) -> Self;
}

impl WeakMul for c64 {
    fn weak_mul(self, other: f64) -> Self {
        c64::new(self.re() * other, self.im() * other)
    }
}

impl WeakMul for f64 {
    fn weak_mul(self, other: f64) -> Self {
        self * other
    }
}

impl WeakMul for i64 {
    fn weak_mul(self, other: f64) -> Self {
        (self as f64 * other) as i64
    }
}

// implment mul for f32 * complex
impl WeakMul for f32 {
    fn weak_mul(self, other: f64) -> Self {
        (self as f64 * other) as f32
    }
}


pub trait TensorElement:
 Copy + Clone + Send + Sync + 'static +
    std::ops::Add<Output = Self> +
    std::ops::Sub<Output = Self> +
    std::ops::Mul<Output = Self> +
    std::ops::Div<Output = Self> +
    std::ops::Neg<Output = Self> +
    std::ops::AddAssign +
    std::ops::SubAssign +
    std::ops::MulAssign +
    std::ops::DivAssign +
    std::cmp::PartialEq +
    std::cmp::PartialOrd +

    // Mul with f64
    WeakMul +

    Into<c64> +
    From<c64> +
 std::fmt::Debug +
 std::fmt::Display 
 {
    /// Returns the additive identity.
    fn zero() -> Self;
    fn one() -> Self;

    /// Generate a random value between min and max.
    /// For types where min and max don’t really apply (e.g. bool)
    /// you can provide a custom behavior.
    fn random(min: Self, max: Self) -> Self;

//    fn c(self) -> c64; // convert to complex for maxi

    fn mag(self) -> f64;

    fn conjugate(self) -> Self;

    const EPSILON: Self;
    
}

use rand::Rng;

impl TensorElement for c64 {
    fn zero() -> Self {
        c64::ZERO
    }
    fn one() -> Self {
        c64::ONE
    }

    fn random(min: Self, max: Self) -> Self {
        let mut rng = rand::rng();
        let a = rng.random::<f64>() * (max.re() - min.re()) + min.re();
        let b = rng.random::<f64>() * (max.im() - min.im()) + min.im();
        c64::new(a, b)
    }

    fn mag(self) -> f64 {
        self.mag()
    }

    fn conjugate(self) -> Self {
        self.conjugate()
    }

    const EPSILON: Self = c64::EPSILON;
}

impl TensorElement for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }

    fn random(min: Self, max: Self) -> Self {
        let mut rng = rand::rng();
        rng.random_range(min..max)
    }

    fn mag(self) -> f64 {
        self.abs()
    }

    fn conjugate(self) -> Self {
        self
    }

    const EPSILON: Self = 1e-10;
}

impl TensorElement for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }

    fn random(min: Self, max: Self) -> Self {
        let mut rng = rand::rng();
        rng.random_range(min..max)
    }

    fn mag(self) -> f64 {
        self.abs() as f64
    }

    fn conjugate(self) -> Self {
        self
    }

    const EPSILON: Self = 1e-20;
}

impl TensorElement for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }

    fn random(min: Self, max: Self) -> Self {
        let mut rng = rand::rng();
        rng.random_range(min..max)
    }

    fn mag(self) -> f64 {
        self.abs() as f64
    }

    fn conjugate(self) -> Self {
        self
    }

    const EPSILON: Self = 1;
}