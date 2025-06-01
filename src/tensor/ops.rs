// Tensor operation implementations

use crate::complex::c64;
use crate::dimension::{Dimension, InvertDimension, MultiplyDimensions, SqrtDimension, Dimensionless, ConstAdd};
use crate::tensor::element::TensorElement;
use crate::tensor::base::Tensor;
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
        self.combine(&other, |a, b| a + b)
    }
}

impl<E: TensorElement + Sub<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Sub for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.combine(&other, |a, b| a - b)
    }
}

// Matrix multiplication using outer product sum - simplified constraints
impl<E: TensorElement + Mul<Output = E> + Add<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COMMON: usize>
Tensor<E, D, LAYERS, ROWS, COMMON>
where
    [(); LAYERS * ROWS * COMMON]:,
    D: Copy,
{
    /// Matrix multiplication using outer product sum approach
    /// A @ B = sum_k(A[:, k] ⊗ B[k, :])
    pub fn matmul<DO, const COLS: usize>(
        self,
        other: Tensor<E, DO, LAYERS, COMMON, COLS>,
    ) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS>
    where
        D: MultiplyDimensions<DO>,
        DO: Copy,
        <D as MultiplyDimensions<DO>>::Output: Copy,
        [(); LAYERS * COMMON * COLS]:,
        [(); LAYERS * ROWS * COLS]:,
        [(); LAYERS * ROWS * 1]:,
        [(); LAYERS * 1 * COLS]:,
        [(); LAYERS * COLS * COMMON]:,  // For transpose in get_rows
        [(); LAYERS * COLS * 1]:,       // For get_cols on transposed tensor
    {
        // Get all columns from A and all rows from B
        let cols_a = self.get_cols();
        let rows_b = other.get_rows();
        
        // Initialize result to zero
        let mut result = None;
        
        // For each k in the inner dimension (COMMON)
        for k in 0..COMMON {
            // Get column k from A: (LAYERS, ROWS, 1)
            let col_a = &cols_a[k];
            
            // Get row k from B: (LAYERS, 1, COLS)  
            let row_b = &rows_b[k];
            
            // Compute outer product: col_A ⊗ row_B -> (LAYERS, ROWS, COLS)
            let outer_prod = Self::outer_product(col_a, row_b);
            
            // Add to result
            result = match result {
                None => Some(outer_prod),
                Some(acc) => Some(acc + outer_prod),
            };
        }
        
        result.expect("Matrix multiplication requires at least one iteration (COMMON > 0)")
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
        self.apply_with_dimension::<_, E, <D as MultiplyDimensions<DS>>::Output>(|v| v * scalar.raw())
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
        self.apply_with_dimension::<_, E, <D as InvertDimension>::Output>(|v| E::one() / v)
    }
}

impl<E: TensorElement + Neg<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Neg for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn neg(self) -> Self {
        self.apply(|v| -v)
    }
}

impl<E, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Into<c64> + Copy,
    [(); LAYERS * ROWS * COLS]:,
{
    /// Converts the tensor to one with c64 elements by mapping each element via Into<c64>.
    pub fn to_c64(&self) -> Tensor<c64, D, LAYERS, ROWS, COLS> {
        self.apply_with_dimension::<_, c64, D>(|v| v.into())
    }
}

// implement conjugate for all tensors
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn conjugate(self) -> Self {
        self.apply(|v| v.conjugate())
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
    ) -> Tensor<f64, <Dimension<L, M, T, Θ, I, N, J> as SqrtDimension>::Output, 1, 1, 1>
    where
        Dimension<L, M, T, Θ, I, N, J>: SqrtDimension,
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
        Tensor::<f64, <Dimension<L, M, T, Θ, I, N, J> as SqrtDimension>::Output, 1, 1, 1>::default([val])
    }

    pub fn dist(
        self,
        other: Self,
    ) -> Tensor<f64, <Dimension<L, M, T, Θ, I, N, J> as SqrtDimension>::Output, 1, 1, 1>
    where
        Dimension<L, M, T, Θ, I, N, J>: SqrtDimension,
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

// Boolean operations should only work on dimensionless quantities and return dimensionless
impl<E: TensorElement, const LAYERS: usize, const ROWS: usize, const COLS: usize> 
Tensor<E, Dimensionless, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Logical AND operation - only valid for dimensionless tensors
    pub fn and(self, other: Self) -> Self {
        self.combine(&other, |a, b| if a != E::zero() && b != E::zero() { E::one() } else { E::zero() })
    }

    /// Logical OR operation - only valid for dimensionless tensors
    pub fn or(self, other: Self) -> Self 
    where
        E: PartialEq + Copy,
    {
        self.combine(&other, |a, b| if a != E::zero() || b != E::zero() { E::one() } else { E::zero() })
    }

    /// Logical XOR operation - only valid for dimensionless tensors
    pub fn xor(self, other: Self) -> Self 
    where
        E: PartialEq + Copy,
    {
        self.combine(&other, |a, b| {
            let a_nonzero = a != E::zero();
            let b_nonzero = b != E::zero();
            if (a_nonzero && !b_nonzero) || (!a_nonzero && b_nonzero) { 
                E::one() 
            } else { 
                E::zero() 
            }
        })
    }

    /// Logical NOT operation - only valid for dimensionless tensors
    pub fn not(self) -> Self {
        self.apply(|v| if v == E::zero() { E::one() } else { E::zero() })
    }
}

// Comparison operations should only work between tensors of the same dimension
// and return dimensionless tensors
impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> 
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Element-wise equality comparison - returns dimensionless tensor
    pub fn eq(self, other: Self) -> Tensor<E, Dimensionless, LAYERS, ROWS, COLS> {
        self.combine_with_dimension::<_, E, Dimensionless>(&other, |a, b| if a == b { E::one() } else { E::zero() })
    }

    /// Element-wise inequality comparison - returns dimensionless tensor
    pub fn ne(self, other: Self) -> Tensor<E, Dimensionless, LAYERS, ROWS, COLS> {
        self.combine_with_dimension::<_, E, Dimensionless>(&other, |a, b| if a != b { E::one() } else { E::zero() })
    }

    /// Element-wise greater than comparison - returns dimensionless tensor
    pub fn gt(self, other: Self) -> Tensor<E, Dimensionless, LAYERS, ROWS, COLS> {
        self.combine_with_dimension::<_, E, Dimensionless>(&other, |a, b| if a > b { E::one() } else { E::zero() })
    }

    /// Element-wise greater than or equal comparison - returns dimensionless tensor
    pub fn ge(self, other: Self) -> Tensor<E, Dimensionless, LAYERS, ROWS, COLS> {
        self.combine_with_dimension::<_, E, Dimensionless>(&other, |a, b| if a >= b { E::one() } else { E::zero() })
    }

    /// Element-wise less than comparison - returns dimensionless tensor
    pub fn lt(self, other: Self) -> Tensor<E, Dimensionless, LAYERS, ROWS, COLS> {
        self.combine_with_dimension::<_, E, Dimensionless>(&other, |a, b| if a < b { E::one() } else { E::zero() })
    }

    /// Element-wise less than or equal comparison - returns dimensionless tensor
    pub fn le(self, other: Self) -> Tensor<E, Dimensionless, LAYERS, ROWS, COLS> {
        self.combine_with_dimension::<_, E, Dimensionless>(&other, |a, b| if a <= b { E::one() } else { E::zero() })
    }
}
