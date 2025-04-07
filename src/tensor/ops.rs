// Tensor operation implementations

use crate::complex::c64;
use crate::dimension::{Dimension, InvertDimension, MultiplyDimensions};
use crate::tensor::element::TensorElement;
use crate::tensor::Tensor;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub, Div};
use crate::*;

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
        let data: [E; LAYERS * ROWS * COLS] = self
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

impl<E: TensorElement + Sub<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Sub for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
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
                        sum = sum + self.data[index_a] * other.data[index_b];
                    }
                    let index_result = layer * (ROWS * COLS) + row * COLS + col;
                    result[index_result] = sum;
                }
            }
        }

        // Convert the result vector into an array.
        let data: [E; LAYERS * ROWS * COLS] =
            result.into_iter().collect::<Vec<E>>().try_into().unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
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
        let s = scalar.data[0];
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v * s)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<
<D as MultiplyDimensions<DS>>::Output
>,
        }
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
        let data: [E; 1] = [E::one() / self.data[0]];
        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement + Neg<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Neg for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn neg(self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
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

impl<E, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Into<c64> + Copy,
    [(); LAYERS * ROWS * COLS]:,
{
    /// Converts the tensor to one with c64 elements by mapping each element via Into<c64>.
    pub fn to_c64(&self) -> Tensor<c64, D, LAYERS, ROWS, COLS> {
        let data: [c64; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v.into())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

// implement conjugate for all tensors
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn conjugate(self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
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

    /// Returns the conjugate transpose of this tensor.
    pub fn conjugate_transpose(self) -> Tensor<E,D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        self.transpose().conjugate()
    }
}

impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Returns the transpose of this tensor.
    pub fn transpose(self) -> Tensor<E,D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        let mut transposed = [E::zero(); LAYERS * COLS * ROWS];
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
        Tensor::<E,D, LAYERS, COLS, ROWS> {
            data: transposed,
            _phantom: PhantomData,
        }
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
    const COLS: usize
>
Tensor<E, Dimension<L, M, T, Θ, I, N, J>, 1, 1, COLS>
where
    [(); 1 * 1 * COLS]:,
    [(); 1 * COLS * 1]:,
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
        let ct = self.conjugate_transpose();
        let i = self.matmul(ct);

        // Manually extract the single element and compute sqrt().
        let val: c64 = i.data[0].into();
        let sqrt_val = f64::from(val.sqrt());

        Tensor {
            data: [sqrt_val],
            _phantom: PhantomData,
        }
    }

    pub fn dist(
        self,
        other: Self,
    ) -> Tensor<f64, Dimension<L, M, T, Θ, I, N, J>, 1, 1, 1>
    where
        [(); 1 * 1 * COLS]:,
        [(); 1 * COLS * 1]:,
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
    
// Implement elementwise equality for all tensors.
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> PartialEq for Tensor<E,D, LAYERS, ROWS, COLS>
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
impl<E:TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Eq for Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    c64: Eq,
{
}

// Implement ordering (>, >=, <, <=) for 1×1×1 tensors only.
impl<E: TensorElement,D> PartialOrd for Tensor<E,D, 1, 1, 1>
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

// linear algebraic stuff
// reshape, flatten, etc.

impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
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
    pub fn data(&self) -> &[E] {
        &self.data
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

    pub fn flatten(&self) -> Tensor<E,D, 1, 1, {LAYERS * ROWS * COLS}>
    where
        [(); LAYERS * ROWS * COLS]:,
        [(); 1 * 1 * (LAYERS * ROWS * COLS)]:,
    {
        self.reshape::<1, 1, {LAYERS * ROWS * COLS}>()
    }

}

// implement all the boolean operations (and, or, xor, not) 
// they should return a tensor of the same size wth 0s and 1s
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn and(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a != E::zero() && b != E::zero() { E::one() } else { E::zero() })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement + PartialEq + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn or(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a != E::zero() || b != E::zero() { E::one() } else { E::zero() })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}



// implement all the comparison operations (eq, ne, gt, ge, lt, le)
// overload the operators
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn eq(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a == b { E::one() } else { E::zero() })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn ne(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a != b { E::one() } else { E::one() })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn gt(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a > b { E::one() } else { E::zero() })
            .collect::<Vec<_>>()    
            .try_into()
            .unwrap();
        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn ge(self, other: Self) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a >= b { E::zero() } else { E::one() })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}
