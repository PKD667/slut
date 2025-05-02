use crate::tensor::scalar::Scalar;
use crate::units::Unit;
use std::marker::PhantomData;
use std::{default, ops::*}; // Keep ops import for AddAssign etc.
use std::default::Default;

use crate::complex::c64;
use crate::tensor::element::*;
use crate::tensor::morph::{Morph, DimCombineTransform, DimAdd, DimDivide};

use crate::dimension::{Dimension, DimTransform, DimSquare, DimSqrt, DimInvert, DimMultiply, MultiplyDimensions, InvertDimension, Dimensionless};

#[derive(Copy, Clone)]
pub struct Tensor<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub data: [E; LAYERS * ROWS * COLS],
    pub _phantom: PhantomData<D>,
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn new<U: Unit<Dimension = D>>(values: [E; LAYERS * ROWS * COLS]) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = values
            .iter()
            .map(|&v| E::from(U::to_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn zero() -> Self {
        let data: [E; LAYERS * ROWS * COLS] = [E::zero(); LAYERS * ROWS * COLS];

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn random<U: Unit<Dimension = D>>(min: E, max: E) -> Self {
        let base_min: E = E::from(U::to_base(min.into()));
        let base_max: E = E::from(U::to_base(max.into()));
        let data: [E; LAYERS * ROWS * COLS] = (0..LAYERS * ROWS * COLS)
            .map(|_| {
                E::from(U::from_base(((base_max - base_min) + base_min).weak_mul(rand::random::<f64>()).into()))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn apply_morph<F, DT>(&self, morph: Morph<E, F, D, DT>) -> Tensor<E, DT::Output, LAYERS, ROWS, COLS>
    where
        F: Fn(E) -> E,
        DT: DimTransform<D>,
        E: TensorElement,
        [(); LAYERS * ROWS * COLS]:,
    {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| morph.apply_to_element(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: morph.output_dimension_type(),
        }
    }

    pub fn reciprocal(&self) -> Tensor<E, <DimInvert as DimTransform<D>>::Output, LAYERS, ROWS, COLS>
    where
        DimInvert: DimTransform<D>,
        E: TensorElement,
        [(); LAYERS * ROWS * COLS]:,
    {
        let reciprocal_morph = Morph::<E, _, D, DimInvert>::new(|v| E::one() / v);
        self.apply_morph(reciprocal_morph)
    }

    pub fn get<S: Unit<Dimension = D>>(&self) -> [E; LAYERS * ROWS * COLS] {
        self.data
            .iter()
            .map(|&v| E::from(S::from_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn raw(&self) -> [E; LAYERS * ROWS * COLS] {
        self.data
    }

    pub fn reduce(
        &self,
        f: impl Fn(E, E) -> E,
    ) -> E {
        let mut result = self.data[0];
        for i in 1..LAYERS * ROWS * COLS {
            result = f(result, self.data[i]);
        }
        result
    }

    pub fn reshape<const L: usize, const R: usize, const C: usize>(
        &self,
    ) -> Tensor<E,D, L, R, C>
    where
        [(); L * R * C]:,
    {
        assert_eq!(LAYERS * ROWS * COLS, L * R * C);
        let data: [E; L * R * C] = self
            .data
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn get_at(&self, layer: usize, row: usize, col: usize) -> Scalar<E,D> {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        Scalar::<E,D> {
            data: [self.data[idx]],
            _phantom: PhantomData,
        }
    }

    pub fn set_at(&mut self, layer: usize, row: usize, col: usize, value: Scalar<E,D>) {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        self.data[idx] = value.data[0];
    }

    /// Combines two tensors element-wise using a provided function `f` and a dimension transformation `DCT`.
    pub fn combine_with_transform<F, D2, DCT>(
        &self,
        other: Tensor<E, D2, LAYERS, ROWS, COLS>, // Second tensor has dimension D2
        f: F,                                     // The function to combine elements
    ) -> Tensor<E, DCT::Output, LAYERS, ROWS, COLS> // Output tensor has dimension DCT::Output
    where
        F: Fn(E, E) -> E,
        DCT: DimCombineTransform<D, D2>, // DCT defines how D and D2 combine
        E: TensorElement,
        [(); LAYERS * ROWS * COLS]:,
    {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&v1, &v2)| f(v1, v2)) // Apply the provided function f
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<DCT::Output>, // Use the output dimension from the transform
        }
    }


}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn dtype(&self) -> &'static str {
        std::any::type_name::<E>()
    }

    pub fn cast<T: TensorElement>(&self) -> Tensor<T, D, LAYERS, ROWS, COLS>
    where
        T: TensorElement,
    {
        let data: [T; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| (T::from(v.into() as c64)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

// vector of N elements
pub type Vector<E: TensorElement, D, const N: usize> = Tensor<E, D, 1, N, 1>;

// matrix of N x M elements
pub type Matrix<E: TensorElement,D, const N: usize, const M: usize> = Tensor<E,D, 1, N, M>;

// ----------- SPECIFIC TYPES OF TENSORS ------------

//type alias for a 2D vector
pub type Vec2<E: TensorElement,D> = Vector<E,D, 2>;
//type alias for a 3D vector
pub type Vec3<E: TensorElement,D> = Vector<E,D, 3>;
//type alias for a 4D vector
pub type Vec4<E: TensorElement,D> = Vector<E,D, 4>;

//type alias for a 2x2 matrix
pub type Mat2<E: TensorElement,D> = Matrix<E,D, 2, 2>;
//type alias for a 3x3 matrix
pub type Mat3<E: TensorElement,D> = Matrix<E,D, 3, 3>;
//type alias for a 4x4 matrix
pub type Mat4<E: TensorElement,D> = Matrix<E,D, 4, 4>;

impl<E: TensorElement,D> Vec2<E,D> {
    // Returns a tuple of elements of type E (generic).
    pub fn raw_tuple(&self) -> (E, E)
    where
        E: TensorElement,
    {
        (self.data[0], self.data[1])
    }

    // Generic conversion for a Vec2 into a tuple of type T.
    pub fn raw_tuple_as<T: From<E>>(&self) -> (T, T)
    where
        E: TensorElement,
    {
        (T::from(self.data[0]), T::from(self.data[1]))
    }
}

impl<E: TensorElement,D: Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    E: TensorElement
{

    // Generic conversion for any Tensor into a Vec<T>.
    pub fn raw_as<T: From<E>>(&self) -> Vec<T> {
        self.raw()
            .iter()
            .map(|&v| T::from(v))
            .collect()
    }
}

// implement x() and y() for Vec2
impl<E: TensorElement,D> Vec2<E,D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }
}

// implement x(), y() and z() for Vec3
impl<E: TensorElement,D> Vec3<E,D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }

    pub fn z(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[2]],
            _phantom: PhantomData,
        }
    }
}

// implement += and -= for all tensors
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> AddAssign for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    E: TensorElement + AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        for i in 0..LAYERS {
            for j in 0..ROWS {
                for k in 0..COLS {
                    let idx = i * (ROWS * COLS) + j * COLS + k;
                    self.data[idx] += other.data[idx];
                }
            }
        }
    }
}


