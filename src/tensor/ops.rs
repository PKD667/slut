// Tensor operation implementations

use crate::complex::c64;
use crate::dimension::{Dimension, InvertDimension, MultiplyDimensions};
use crate::tensor::element::TensorElement;
use crate::tensor::Tensor;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub, Div};
use crate::*;

use super::Scalar;

// -----------------------------------------
// ============= OPERATIONS ================
// -----------------------------------------

impl<E: TensorElement + Add<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Add for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| a + b)
    }
}

impl<E: TensorElement + Sub<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Sub for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| a - b)
    }
}

impl<
    E: TensorElement + Mul<Output = E> + Add<Output = E> + Copy,
    const LAYERS: usize,
    const L1: i32,
    const M1: i32,
    const T1: i32,
    const Θ1: i32,
    const I1: i32,
    const N1: i32,
    const J1: i32,
    const ROWS: usize,
    const COMMON: usize,
> Tensor<E, Dimension<L1, M1, T1, Θ1, I1, N1, J1>, LAYERS, ROWS, COMMON>
where
    [(); LAYERS * ROWS * COMMON]:,
{
    /// Performs matrix multiplication between two tensors.
    pub fn matmul<
        const L2: i32,
        const M2: i32,
        const T2: i32,
        const Θ2: i32,
        const I2: i32,
        const N2: i32,
        const J2: i32,
        const COLS: usize,
    >(
        self,
        other: Tensor<E, Dimension<L2, M2, T2, Θ2, I2, N2, J2>, LAYERS, COMMON, COLS>,
    ) -> Tensor<
        E,
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<
            Dimension<L2, M2, T2, Θ2, I2, N2, J2>
        >>::Output,
        LAYERS,
        ROWS,
        COLS,
    >
    where
        Dimension<L1, M1, T1, Θ1, I1, N1, J1>: MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
        [(); LAYERS * COMMON * COLS]:,
        [(); LAYERS * ROWS * COLS]:,
        [(); COLS]:,
    {
        // TODO: Abstract away the direct data access
        let data = self.data();
        let other_data = other.data();
        // code above is BAD, VERY BAD
        // SORRY

        // Create a vector to store the output tensor, initializing all entries to zero.
        let mut result = vec![E::zero(); LAYERS * ROWS * COLS];

        // Iterate over layers, rows, and columns to compute each element.
        for layer in 0..LAYERS {
            for row in 0..ROWS {
                for col in 0..COLS {
                    let mut sum = E::zero();
                    // Compute the dot product for the current (layer, row, col) element.
                    for k in 0..COMMON {
                        let index_a = layer * (ROWS * COMMON) + row * COMMON + k;
                        let index_b = layer * (COMMON * COLS) + k * COLS + col;
                        sum = sum + data[index_a] * other_data[index_b];
                    }
                    let index_result = layer * (ROWS * COLS) + row * COLS + col;
                    result[index_result] = sum;
                }
            }
        }

        // Convert the result vector into an array.
        let data: [E; LAYERS * ROWS * COLS] =
            result.into_iter().collect::<Vec<E>>().try_into().unwrap();

        Tensor::default(data)

    }
}

impl<E: TensorElement + Mul<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Multiplies every element of the tensor by a scalar.
    pub fn scale<DS>(
        self,
        scalar: Tensor<E, DS, 1, 1, 1>,
    ) -> Tensor<E, <D as MultiplyDimensions<DS>>::Output, LAYERS, ROWS, COLS>
    where
        D: MultiplyDimensions<DS>,
        <D as MultiplyDimensions<DS>>::Output:,
            {
        self.apply::<_, E, <D as MultiplyDimensions<DS>>::Output>(|v| v * scalar.raw())
    }
}

impl<E, D, DS, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Mul<Tensor<E, DS, 1, 1, 1>> for Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Mul<Output = E> + Copy,
    D: MultiplyDimensions<DS>,
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Tensor<E, <D as MultiplyDimensions<DS>>::Output, LAYERS, ROWS, COLS>;

    fn mul(self, rhs: Tensor<E, DS, 1, 1, 1>) -> Self::Output {
        self.scale(rhs)
    }
}


impl<E, D, DS, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Div<Tensor<E, DS, 1, 1, 1>> for Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Div<Output = E> + Copy,
    DS: InvertDimension,
    D: MultiplyDimensions<<DS as InvertDimension>::Output>,
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Tensor<
        E,
        <D as MultiplyDimensions<<DS as InvertDimension>::Output>>::Output,
        LAYERS,
        ROWS,
        COLS
    >;

    fn div(self, rhs: Tensor<E, DS, 1, 1, 1>) -> Self::Output {
        self.scale(rhs.inv())
    }
}

impl<E: TensorElement + Div<Output = E> + Copy + PartialEq, D> Tensor<E, D, 1, 1, 1>
where
    [(); 1]:,
{
    pub fn inv(self) -> Tensor<E, <D as InvertDimension>::Output, 1, 1, 1>
    where
        D: InvertDimension,
    {
        self.apply::<_, E, <D as InvertDimension>::Output>(|v| E::one() / v)
    }
}

impl<E: TensorElement + Neg<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Neg for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn neg(self) -> Self {
        self.apply::<_, E, D>(|v| -v)
    }
}

impl<E, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Into<c64> + Copy,
    [(); LAYERS * ROWS * COLS]:,
{
    /// Converts the tensor to one with c64 elements by mapping each element via Into<c64>.
    pub fn to_c64(&self) -> Tensor<c64, D, LAYERS, ROWS, COLS> {
        self.apply::<_, c64, D>(|v| v.into())
    }
}

// implement conjugate for all tensors
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn conjugate(self) -> Self {
        self.apply::<_, E, D>(|v| v.conjugate())
    }

    /// Returns the conjugate transpose of this tensor.
    pub fn conjugate_transpose(self) -> Tensor<E,D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        self.transpose().conjugate()
    }
}

// Fix the call site by manually extracting the element for sqrt:
impl<
    E: TensorElement + Into<f64> + Copy,
    const L: i32,
    const M: i32,
    const T: i32,
    const Θ: i32,
    const I: i32,
    const N: i32,
    const J: i32,
    const ROWS: usize
>
Tensor<E, Dimension<L, M, T, Θ, I, N, J>, 1, ROWS, 1>
where
    [(); 1 * ROWS * 1]:,
    [(); 1 * 1 * ROWS]:,
    [(); ROWS * 1 * 1]:,
{
    pub fn norm(
        self
    ) -> Tensor<f64, Dimension<L, M, T, Θ, I, N, J>, 1, 1, 1>
    where
        [(); { <() as ConstAdd<L, L>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<M, M>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<T, T>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<Θ, Θ>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<I, I>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<N, N>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<J, J>>::OUTPUT } as usize]:,
    {
        let ct: Tensor<E, Dimension<L, M, T, Θ, I, N, J>, 1, 1, ROWS> = self.conjugate_transpose();
        let i: Tensor<E, Dimension<_, _, _, _, _, _, _>, 1, 1, 1> = ct.matmul(self);

        // Manually extract the single element and compute sqrt().
        let val:f64 = i.cast::<c64>().raw().sqrt().mag();
        Scalar::default([val])
    }

    pub fn dist(
        self,
        other: Self,
    ) -> Tensor<f64, Dimension<L, M, T, Θ, I, N, J>, 1, 1, 1>
    where
        [(); 1 * ROWS * 1]:,
        [(); 1 * 1 * ROWS]:,
        [(); { <() as ConstAdd<L, L>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<M, M>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<T, T>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<Θ, Θ>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<I, I>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<N, N>>::OUTPUT } as usize]:,
        [(); { <() as ConstAdd<J, J>>::OUTPUT } as usize]:,
    {
       let sub = self - other;
       sub.norm()
    }

}

// implement dot product as macro that does transpose and multiply
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        let a_t = a.transpose();
        let result = a_t.matmul(b);
        result
    }};
}

#[macro_export]
macro_rules! inner_product {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        let a_t = a.conjugate_transpose();
        let result = a_t.matmul(b);
        result
    }};
}

#[macro_export]
macro_rules! ip {
    ($x:expr, $y:expr) => {
        inner_product!($x, $y)
    };
}

// implement all the boolean operations (and, or, xor, not) 
// they should return a tensor of the same size wth 0s and 1s
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn and(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| if a != E::zero() && b != E::zero() { E::one() } else { E::zero() })
    }
}

impl<E: TensorElement + PartialEq + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn or(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| if a != E::zero() || b != E::zero() { E::one() } else { E::zero() })
    }
}

// implement all the comparison operations (eq, ne, gt, ge)
// overload the operators
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn eq(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| if a == b { E::one() } else { E::zero() })
    }
}
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn ne(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| if a != b { E::one() } else { E::zero() })
    }
}
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn gt(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| if a > b { E::one() } else { E::zero() })
    }
}

impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn ge(self, other: Self) -> Self {
        self.combine::<_, E, D>(&other, |a, b| if a >= b { E::one() } else { E::zero() })
    }
}
