use crate::units::{Unit, STYPE};
use std::marker::PhantomData;
use std::ops::*;

use crate::dimension::Dimension;
use crate::dimension::MultiplyDimensions;
use crate::dimension::InvertDimension;

#[derive(Copy, Clone)]
pub struct Tensor<D, const ROWS: usize, const COLS: usize>
where
    [(); ROWS * COLS]:,
{
    data: [STYPE; ROWS * COLS],
    _phantom: PhantomData<D>,
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    pub fn new<U: Unit<Dimension = D>>(values: [STYPE; ROWS * COLS]) -> Self {
        Self {
            data: values,
            _phantom: PhantomData,
        }
    }

    pub fn get_at<U: Unit<Dimension = D>>(&self, row: usize, col: usize) -> STYPE {
        assert!(row < ROWS && col < COLS);
        let idx = row * COLS + col;
        U::from_base(self.data[idx])
    }

    pub fn get<S: Unit<Dimension = D>>(&self) -> [STYPE; ROWS * COLS] {
        self.data
            .iter()
            .map(|&v| S::from_base(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

// implement common types of tensors
pub type Scalar<D> = Tensor<D, 1, 1>;

// vector of N elements
pub type Vector<D, const N: usize> = Tensor<D, N, 1>;

// matrix of N x M elements
pub type Matrix<D, const N: usize, const M: usize> = Tensor<D, N, M>;

// ----------- SPECIFIC TYPES OF TENSORS ------------

//type alias for a 2D vector
pub type Vec2<D> = Vector<D, 2>;
//type alias for a 3D vector
pub type Vec3<D> = Vector<D, 3>;
//type alias for a 4D vector
pub type Vec4<D> = Vector<D, 4>;

//type alias for a 2x2 matrix
pub type Mat2<D> = Matrix<D, 2, 2>;
//type alias for a 3x3 matrix
pub type Mat3<D> = Matrix<D, 3, 3>;
//type alias for a 4x4 matrix
pub type Mat4<D> = Matrix<D, 4, 4>;

impl<D, const ROWS: usize, const COLS: usize> Add for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let data: [STYPE; ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Sub for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let data: [STYPE; ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<
        const L1: i32,
        const M1: i32,
        const T1: i32,
        const Θ1: i32,
        const I1: i32,
        const N1: i32,
        const J1: i32,
        const L2: i32,
        const M2: i32,
        const T2: i32,
        const Θ2: i32,
        const I2: i32,
        const N2: i32,
        const J2: i32,
        const ROWS: usize,
        const COMMON: usize,
        const COLS: usize,
    > Mul<Tensor<Dimension<L2, M2, T2, Θ2, I2, N2, J2>, COMMON, COLS>>
    for Tensor<Dimension<L1, M1, T1, Θ1, I1, N1, J1>, ROWS, COMMON>
where
    [(); ROWS * COMMON]:,
    [(); COMMON * COLS]:,
    [(); ROWS * COLS]:,
    // Use the helper trait to combine dimensions.
    Dimension<L1, M1, T1, Θ1, I1, N1, J1>:
        MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
{
    type Output = Tensor<
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<
            Dimension<L2, M2, T2, Θ2, I2, N2, J2>,
        >>::Output,
        ROWS,
        COLS,
    >;

    fn mul(
        self,
        other: Tensor<Dimension<L2, M2, T2, Θ2, I2, N2, J2>, COMMON, COLS>,
    ) -> Self::Output {
        let mut result = vec![0.0; ROWS * COLS];

        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum: STYPE = 0.0;
                for k in 0..COMMON {
                    sum += self.data[i * COMMON + k] * other.data[k * COLS + j];
                }
                result[i * COLS + j] = sum;
            }
        }

        let data: [STYPE; ROWS * COLS] = result
            .into_iter()
            .collect::<Vec<STYPE>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    /// Multiplies every element of the tensor by a scalar.
    ///
    /// The scalar is provided as a 1×1 tensor, and the result's dimension
    /// is the product of the original tensor’s dimension and the scalar’s dimension.
    pub fn scale<DS>(
        self,
        scalar: Tensor<DS, 1, 1>,
    ) -> Tensor<<D as MultiplyDimensions<DS>>::Output, ROWS, COLS>
    where
        D: MultiplyDimensions<DS>,
        [(); ROWS * COLS]:,
    {
        let s = scalar.data[0];
        let data: [STYPE; ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v * s)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<<D as MultiplyDimensions<DS>>::Output>,
        }
    }
}

// invert a scalar
impl<D> Tensor<D, 1, 1>
where
    [(); 1]:,
{
    pub fn inv(self) -> Tensor<<D as InvertDimension>::Output, 1, 1>
    where
        D: InvertDimension,
    {
        let data: [STYPE; 1] = [1.0 / self.data[0]];

        Tensor {
            data,
            _phantom: PhantomData::<D::Output>,
        }
    }
}
 

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    /// Returns the norm of the tensor as a 1×1 tensor.
    /// The norm is defined as sqrt(sum(vᵢ²)).
    pub fn norm(&self) -> Tensor<D, 1, 1> {
        let sum: STYPE = self.data.iter().map(|&v| v * v).sum();

        Tensor::<D, 1, 1> {
            data: [sum.sqrt()],
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    pub fn dist(self, other: Self) -> Tensor<D, 1, 1> {
        // norm of sub
        let sub = self - other;
        sub.norm()
    }
}

impl<D, const ROWS: usize, const COLS: usize> std::fmt::Display for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor<{}, {}>:\n", ROWS, COLS)?;
        write!(f, "(")?;
        for i in 0..ROWS {
            write!(f, "(")?;
            for j in 0..COLS {
                write!(f, "{:.2} ", self.data[i * COLS + j])?;
            }
            write!(f, ")")?;
            if (i + 1) < ROWS {
                write!(f, "\n ")?;
            }
        }
        write!(f, ")")?;
        Ok(())
    }
}
