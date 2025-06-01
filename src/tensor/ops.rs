// Tensor operation implementations

use crate::complex::c64;
use crate::dimension::{Dimension, InvertDimension, MultiplyDimensions, SqrtDimension, Dimensionless, ConstAdd};
use crate::tensor::element::TensorElement;
use crate::tensor::base::{Tensor, Op, OpType, next_tensor_id};
use std::ops::{Add, Mul, Neg, Sub, Div};
use std::marker::PhantomData;
use crate::*;

use super::Scalar;

// -----------------------------------------
// ============= OPERATIONS ================
// -----------------------------------------

impl<E: TensorElement + Add<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Add for &Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Tensor<E, D, LAYERS, ROWS, COLS>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl<E: TensorElement + Sub<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Sub for &Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Tensor<E, D, LAYERS, ROWS, COLS>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other)
    }
}

// Matrix multiplication using graph operations
impl<E: TensorElement + Mul<Output = E> + Add<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COMMON: usize>
Tensor<E, D, LAYERS, ROWS, COMMON>
where
    [(); LAYERS * ROWS * COMMON]:,
    D: Copy,
{
    /// Matrix multiplication using graph operations
    pub fn matmul<DO: Clone + Copy, const COLS_USIZE: usize>(
        &self,
        other: &Tensor<E, DO, LAYERS, COMMON, COLS_USIZE>,
    ) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS_USIZE>
    where
        D: MultiplyDimensions<DO>,
        <D as MultiplyDimensions<DO>>::Output: Clone + Copy,
        [(); LAYERS * COMMON * COLS_USIZE]:,
        [(); LAYERS * ROWS * COLS_USIZE]:,
    {
        // Create a graph node for matrix multiplication
        let tensor = Tensor::<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS_USIZE> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::MatMul,
                data: None,
                inputs: vec![self.id(), other.id()],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::MatMul,
            inputs: vec![self.id(), other.id()],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }
}

impl<E: TensorElement + Neg<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Neg for &Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Tensor<E, D, LAYERS, ROWS, COLS>;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

// Returns the conjugate transpose of this tensor using graph operations
impl<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize> 
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    [(); LAYERS * COLS * ROWS]:,
{
    /// Returns the conjugate transpose of this tensor.
    pub fn conjugate_transpose(&self) -> Tensor<E, D, LAYERS, COLS, ROWS> {
        self.clone().transpose().conjugate()
    }
}

// Norm and distance macros for arbitrary dimensions
#[macro_export]
macro_rules! norm {
    ($tensor:expr) => {{
        $tensor.compute_norm()
    }};
}

// Simple distance macro using subtraction and norm
#[macro_export]
macro_rules! dist {
    ($a:expr, $b:expr) => {{
        let diff = (&$a).sub(&$b);
        norm!(diff)
    }};
}

// implement dot product as macro that does transpose and multiply
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {{
        let a_ref = &$a;
        let b_ref = &$b;
        let a_t = a_ref.transpose();
        a_t.matmul(b_ref)
    }};
}

#[macro_export]
macro_rules! inner_product {
    ($a:expr, $b:expr) => {{
        let a_ref = &$a;
        let b_ref = &$b;
        let a_h = a_ref.conjugate_transpose();
        a_h.matmul(b_ref)
    }};
}

#[macro_export]
macro_rules! ip {
    ($x:expr, $y:expr) => {
        inner_product!($x, $y)
    };
}

impl<E: TensorElement + Mul<Output = E> + Copy, D: Clone, DO: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Mul<&Tensor<E, DO, LAYERS, ROWS, COLS>> for &Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: MultiplyDimensions<DO>,
    <D as MultiplyDimensions<DO>>::Output: Clone,
{
    type Output = Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS>;

    fn mul(self, other: &Tensor<E, DO, LAYERS, ROWS, COLS>) -> Self::Output {
        self.hadamard(other)
    }
}

impl<E: TensorElement + Div<Output = E> + Copy, D: Clone, DO: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Div<&Tensor<E, DO, LAYERS, ROWS, COLS>> for &Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: MultiplyDimensions<<DO as InvertDimension>::Output>,
    <D as MultiplyDimensions<<DO as InvertDimension>::Output>>::Output: Clone,
    DO: InvertDimension,
{
    type Output = Tensor<E, <D as MultiplyDimensions<<DO as InvertDimension>::Output>>::Output, LAYERS, ROWS, COLS>;

    fn div(self, other: &Tensor<E, DO, LAYERS, ROWS, COLS>) -> Self::Output {
        // Create a graph node for division
        let tensor = Tensor::<E, <D as MultiplyDimensions<<DO as InvertDimension>::Output>>::Output, LAYERS, ROWS, COLS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Div,
                data: None,
                inputs: vec![self.id(), other.id()],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Div,
            inputs: vec![self.id(), other.id()],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }
} 