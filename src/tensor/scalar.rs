use crate::tensor::*;
use crate::units::*;
use crate::tensor::element::*;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// A scalar value with a physical dimension.
/// This is a specialized tensor with shape (1,1,1).
pub type Scalar<E: TensorElement, D: Clone> = Tensor<E, D, 1, 1, 1>;

/// The smallest value that can be represented by a scalar type.
pub fn epsilon<E: TensorElement, D: Clone>() -> Scalar<E, D> {
    Scalar::default([E::EPSILON])
}

impl<E: TensorElement, D: Clone> Scalar<E, D> {
    /// Construct a Scalar from a value in the given unit.
    pub fn from<U: Unit<Dimension = D>>(value: E) -> Self {
        Scalar::new::<U>([value])
    }

    /// Convert the raw element into any type implementing From<E>.
    pub fn raw_as<T: From<E>>(&self) -> T {
        T::from(self.raw())
    }

    /// Return a scalar containing the magnitude.
    pub fn mag(&self) -> Scalar<f64, D> {
        Scalar::default([self.raw().mag()])
    }

    /// Get the epsilon value for this scalar type.
    pub fn epsilon(&self) -> Self {
        epsilon::<E, D>()
    }
}

/// A trait to convert a value into a Scalar tensor.
pub trait ToScalar<E: TensorElement> {
    /// Convert the value to a scalar with the given unit.
    fn scalar<U: Unit>(&self) -> Scalar<E, U::Dimension> where U::Dimension: Clone;
}

impl<E: TensorElement> ToScalar<E> for E {
    fn scalar<U: Unit>(&self) -> Scalar<E, U::Dimension> where U::Dimension: Clone {
        Scalar::new::<U>([*self])
    }
}
