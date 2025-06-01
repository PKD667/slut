use crate::tensor::element::TensorElement;
use crate::tensor::base::Tensor;
use crate::tensor::scalar::Scalar;
use std::marker::PhantomData;
use std::ops::{AddAssign, SubAssign};
use crate::tensor::base::Op;

// Type aliases for common tensor shapes
pub type Vector<E: TensorElement, D: Clone, const N: usize> = Tensor<E, D, 1, N, 1>
where
    [(); 1 * N * 1]:;

pub type Matrix<E: TensorElement, D: Clone, const N: usize, const M: usize> = Tensor<E, D, 1, N, M>
where
    [(); 1 * N * M]:;

pub type Vec2<E: TensorElement, D: Clone> = Vector<E, D, 2>;
pub type Vec3<E: TensorElement, D: Clone> = Vector<E, D, 3>;
pub type Vec4<E: TensorElement, D: Clone> = Vector<E, D, 4>;

pub type Mat2<E: TensorElement, D: Clone> = Matrix<E, D, 2, 2>;
pub type Mat3<E: TensorElement, D: Clone> = Matrix<E, D, 3, 3>;
pub type Mat4<E: TensorElement, D: Clone> = Matrix<E, D, 4, 4>;

// Specialized methods for Vec2
impl<E: TensorElement, D: Clone> Vec2<E, D> {
    pub fn raw_tuple(&self) -> (E, E)
    where
        E: TensorElement,
    {
        let data = self.data();
        (data[0], data[1])
    }

    pub fn raw_tuple_as<T: From<E>>(&self) -> (T, T)
    where
        E: TensorElement,
    {
        let data = self.data();
        (T::from(data[0]), T::from(data[1]))
    }
}

// Specialized methods for Vec3
impl<E: TensorElement, D: Clone> Vec3<E, D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E, D> {
        self.get_at(0, 0, 0)
    }

    pub fn y(&self) -> Scalar<E, D> {
        self.get_at(0, 1, 0)
    }

    pub fn z(&self) -> Scalar<E, D> {
        self.get_at(0, 2, 0)
    }
}

// Trait implementations that don't require data access
impl<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize> PartialEq
    for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn eq(&self, other: &Self) -> bool {
        self.data() == other.data()
    }
}

impl<E: TensorElement, D: Clone> PartialOrd for Tensor<E, D, 1, 1, 1>
where
    [(); 1]:,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_data = self.data();
        let other_data = other.data();
        self_data[0].partial_cmp(&other_data[0])
    }
}

impl<E: TensorElement + AddAssign, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    AddAssign for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    Op<E, D, LAYERS, ROWS, COLS>: Copy,
{
    fn add_assign(&mut self, other: Self) {
        *self = self.add(&other);
    }
}

impl<E: TensorElement + SubAssign, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    SubAssign for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    Op<E, D, LAYERS, ROWS, COLS>: Copy,
{
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(&other);
    }
} 