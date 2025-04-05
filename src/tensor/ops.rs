// Tensor operation implementations

use crate::complex::c64;
use crate::dimension::{Dimension, InvertDimension, MultiplyDimensions};
use crate::tensor::Tensor;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};


// -----------------------------------------
// ============= OPERATIONS ================
// -----------------------------------------

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Add for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = self
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

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Sub for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = self
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
        const LAYERS: usize,
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
    > Mul<Tensor<Dimension<L2, M2, T2, Θ2, I2, N2, J2>, LAYERS, COMMON, COLS>>
    for Tensor<Dimension<L1, M1, T1, Θ1, I1, N1, J1>, LAYERS, ROWS, COMMON>
where
    [(); LAYERS * ROWS * COMMON]:,
    [(); LAYERS * COMMON * COLS]:,
    [(); LAYERS * ROWS * COLS]:,
    // Use the helper trait to combine dimensions.
    Dimension<L1, M1, T1, Θ1, I1, N1, J1>:
        MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
{
    type Output = Tensor<
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<
            Dimension<L2, M2, T2, Θ2, I2, N2, J2>,
        >>::Output,
        LAYERS,
        ROWS,
        COLS,
    >;

    fn mul(
        self,
        other: Tensor<Dimension<L2, M2, T2, Θ2, I2, N2, J2>, LAYERS, COMMON, COLS>,
    ) -> Self::Output {
        let mut result = vec![c64::zero(); LAYERS * ROWS * COLS];

        for l in 0..LAYERS {
            for i in 0..ROWS {
                for j in 0..COLS {
                    let mut sum: c64 = c64::zero();
                    for k in 0..COMMON {
                        let a_idx = l * (ROWS * COMMON) + i * COMMON + k;
                        let b_idx = l * (COMMON * COLS) + k * COLS + j;
                        sum += self.data[a_idx] * other.data[b_idx];
                    }
                    let r_idx = l * (ROWS * COLS) + i * COLS + j;
                    result[r_idx] = sum;
                }
            }
        }

        let data: [c64; LAYERS * ROWS * COLS] = result
            .into_iter()
            .collect::<Vec<c64>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Multiplies every element of the tensor by a scalar and auto-normalizes its dimension.
    /// The result's dimension is the product of the original tensor’s dimension and
    /// the scalar’s dimension, normalized automatically.
    pub fn scale<DS>(
        self,
        scalar: Tensor<DS, 1, 1, 1>,
    ) -> Tensor<<D as MultiplyDimensions<DS>>::Output, LAYERS, ROWS, COLS>
    where
        D: MultiplyDimensions<DS>,
        <D as MultiplyDimensions<DS>>::Output:,
        [(); LAYERS * ROWS * COLS]:,
    {
        let s = scalar.data[0];
        let data: [c64; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v * s)
            .collect::<Vec<c64>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<<D as MultiplyDimensions<DS>>::Output>,
        }
    }
}

// invert a scalar
impl<D> Tensor<D, 1, 1, 1>
where
    [(); 1]:,
{
    pub fn inv(self) -> Tensor<<D as InvertDimension>::Output, 1, 1, 1>
    where
        D: InvertDimension,
    {
        let data: [c64; 1] = [1.0 / self.data[0]];

        Tensor {
            data,
            _phantom: PhantomData::<D::Output>,
        }
    }
}

// implement negation for all tensors
impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Neg for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn neg(self) -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| -v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Returns the norm of the tensor as a 1×1×1 tensor.
    /// The norm is defined as sqrt(sum(vᵢ²)).
    pub fn norm(&self) -> Tensor<D, 1, 1, 1> {
        let sum: c64 = self.data.iter().map(|&v| v * v).sum();

        Tensor::<D, 1, 1, 1> {
            data: [sum.sqrt()],
            _phantom: PhantomData,
        }
    }
}

// implement conjugate for all tensors
impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn conjugate(self) -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v.conjugate())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Returns the transpose of this tensor.
    pub fn transpose(self) -> Tensor<D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        let mut transposed = [c64::zero(); LAYERS * COLS * ROWS];
        for l in 0..LAYERS {
            for i in 0..ROWS {
                for j in 0..COLS {
                    // Element at (i, j) moves to (j, i)
                    let src = l * (ROWS * COLS) + i * COLS + j;
                    let dst = l * (COLS * ROWS) + j * ROWS + i;
                    transposed[dst] = self.data[src];
                }
            }
        }
        Tensor::<D, LAYERS, COLS, ROWS> {
            data: transposed,
            _phantom: PhantomData,
        }
    }

    /// Returns the conjugate transpose of this tensor.
    pub fn conjugate_transpose(self) -> Tensor<D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        self.transpose().conjugate()
    }
}

impl<D, const LAYERS: usize, const ROWS: usize> Tensor<D, LAYERS, ROWS, 1>
where
    [(); LAYERS * ROWS * 1]:,
{
    pub fn dist(self, other: Self) -> Tensor<D, 1, 1, 1> {
        // norm of sub
        let sub = self - other;
        sub.norm()
    }
}

// Implement elementwise equality for all tensors.
impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> PartialEq for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn eq(&self, other: &Self) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(&a, &b)| a == b)
    }
}

// Optionally, if c64: Eq then implement Eq.
impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Eq for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    c64: Eq,
{
}

// Implement ordering (>, >=, <, <=) for 1×1×1 tensors only.
impl<D> PartialOrd for Tensor<D, 1, 1, 1>
where
    [(); 1]:,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data[0].partial_cmp(&other.data[0])
    }
}

// implement dot product as macro that does transpose and multiply
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        let a_t = a.transpose();
        let result = a_t * b;
        result
    }};
}

#[macro_export]
macro_rules! inner_product {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        let a_t = a.conjugate_transpose();
        let result = a_t * b;
        result
    }};
}

#[macro_export]
macro_rules! ip {
    ($x:expr, $y:expr) => {
        inner_product!($x, $y)
    };
}

// linear algebraic stuff
// reshape, flatten, etc.

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Returns the number of elements in the tensor.
    pub fn size(&self) -> usize {
        LAYERS * ROWS * COLS
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (LAYERS, ROWS, COLS)
    }

    /// Returns the number of layers in the tensor.
    pub fn layers(&self) -> usize {
        LAYERS
    }

    /// Returns the number of rows in the tensor.
    pub fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns in the tensor.
    pub fn cols(&self) -> usize {
        COLS
    }

    /// Returns the data as a slice.
    pub fn data(&self) -> &[c64] {
        &self.data
    }

    pub fn reshape<const L: usize, const R: usize, const C: usize>(
        &self,
    ) -> Tensor<D, L, R, C>
    where
        [(); L * R * C]:,
    {
        assert_eq!(LAYERS * ROWS * COLS, L * R * C);
        let data: [c64; L * R * C] = self
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

    pub fn flatten(&self) -> Tensor<D, 1, 1, {LAYERS * ROWS * COLS}>
    where
        [(); LAYERS * ROWS * COLS]:,
        [(); 1 * 1 * (LAYERS * ROWS * COLS)]:,
    {
        self.reshape::<1, 1, {LAYERS * ROWS * COLS}>()
    }

}
