use crate::tensor::*;
use crate::units::*;
use crate::tensor::element::*;
use std::marker::PhantomData;

pub type Scalar<E: TensorElement, D: Clone> = Tensor<E, D, 1, 1, 1>;

impl<E: TensorElement, D: Clone> Scalar<E, D> {
    // Require TensorElement to provide an EPSILON constant.
    pub const EPSILON: Scalar<E, D> = Scalar::default([E::EPSILON]);

    // Construct a Scalar from a basic f64 value.
    pub fn from<U: Unit<Dimension = D>>(value: E) -> Self {
        Scalar::new::<U>([value])
    }

    // Convert the raw element into any type implementing From<E>.
    pub fn raw_as<T: From<E>>(&self) -> T {
        T::from(self.raw())
    }

    // Return a scalar containing the magnitude.
    pub fn mag(&self) -> Scalar<f64, D> {
        Scalar::default([self.raw().mag()])
    }
}

impl<E: TensorElement, D: Clone> Scalar<E, D> {
    pub fn epsilon(&self) -> Self {
        Self::EPSILON
    }
}

/// A trait to convert a value into a Scalar tensor.
pub trait ToScalar<E: TensorElement> {
    fn scalar<U: Unit>(&self) -> Scalar<E, U::Dimension> where U::Dimension: Clone;
}

impl<E: TensorElement> ToScalar<E> for E {
    fn scalar<U: Unit>(&self) -> Scalar<E, U::Dimension> where U::Dimension: Clone {
        Scalar::new::<U>([*self])
    }
}

