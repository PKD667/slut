use crate::tensor::*;
use crate::units::*;
use crate::tensor::element::*;
use std::marker::PhantomData;

pub type Scalar<E: TensorElement, D> = Tensor<E, D, 1, 1, 1>;

impl<E: TensorElement, D> Scalar<E, D> {
    // Require TensorElement to provide an EPSILON constant.
    pub const EPSILON: Scalar<E, D> = Scalar {
        data: [E::EPSILON],
        _phantom: PhantomData,
    };

    // Construct a Scalar from a basic f64 value.
    pub fn from<U: Unit<Dimension = D>>(value: E) -> Self {
        Scalar {
            data: [U::to_base(value.into()).into()],
            _phantom: PhantomData,
        }
    }

    // Return the raw underlying element.
    pub fn item(&self) -> E {
        self.data[0]
    }

    // Convert the raw element into any type implementing From<E>.
    pub fn item_as<T: From<E>>(&self) -> T {
        T::from(self.data[0])
    }

    // Return a scalar containing the magnitude.
    pub fn mag(&self) -> Scalar<f64, D> {
        Scalar {
            data: [self.data[0].mag()],
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement, D> Scalar<E, D> {
    pub fn epsilon(&self) -> Self {
        Self::EPSILON
    }
}

/// A trait to convert a value into a Scalar tensor.
pub trait ToScalar<E: TensorElement> {
    fn scalar<U: Unit>(&self) -> Scalar<E, U::Dimension>;
}

impl<E: TensorElement> ToScalar<E> for E {
    fn scalar<U: Unit>(&self) -> Scalar<E, U::Dimension> {
        Scalar {
            data: [*self],
            _phantom: PhantomData,
        }
    }
}

