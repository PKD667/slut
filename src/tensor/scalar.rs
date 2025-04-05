use crate::tensor::*;
use crate::units::*;
use crate::complex::*;
use std::marker::PhantomData;

// implement common types of tensors
pub type Scalar<D> = Tensor<D, 1, 1, 1>;

// epsilon scalar
impl<D> Scalar<D> {
    pub const EPSILON: Scalar<D> = Scalar {
        data: [c64 { a: f64::EPSILON, b: 0.0 }],
        _phantom: PhantomData,
    };
}

use crate::dimension::Dimensionless;

pub trait ToScalar {
    // convert by specifying the Unit
    fn scalar<U: Unit>(&self) -> Scalar<U::Dimension>;
}

impl ToScalar for c64 {

    fn scalar<U: Unit>(&self) -> Scalar<U::Dimension> {
        Scalar {
            data: [*self],
            _phantom: PhantomData,
        }
    }
}

impl ToScalar for f64 {
    fn scalar<U: Unit>(&self) -> Scalar<U::Dimension> {
        Scalar {
            data: [U::to_base(self.complex())],
            _phantom: PhantomData,
        }
    }
}

// implement converting a c64 to a Scalar tensor
impl<D> Scalar<D> {
    pub fn from_c64<U: Unit<Dimension = D>>(value: c64) -> Self {
        Scalar {
            data: [U::to_base(value)],
            _phantom: PhantomData,
        }
    }
}

// implement converting a float to a Scalar tensor
impl<D> Scalar<D> {
    pub fn from_f64<U: Unit<Dimension = D>>(value: f64) -> Self {
        Scalar {
            data: [U::to_base(value.complex())],
            _phantom: PhantomData,
        }
    }
}

impl<D> Scalar<D> {
    // Default raw returns c64.
    pub fn raw(&self) -> c64 {
        self.data[0]
    }

    // Generic raw conversion into any type that implements From<c64>.
    pub fn raw_as<T: From<c64>>(&self) -> T {
        T::from(self.data[0])
    }
}

// implement val() im() and re() for scalar tensors
impl<D> Scalar<D> {
    pub fn im(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0].im().complex()],
            _phantom: PhantomData,
        }
    }

    pub fn re(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0].re().complex()],
            _phantom: PhantomData,
        }
    }

    pub fn mag(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0].mag().complex()],
            _phantom: PhantomData,
        }
    }
}
