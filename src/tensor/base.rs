use crate::tensor::scalar::Scalar;
use crate::units::Unit;
use std::marker::PhantomData;
use std::{ops::*, usize};
use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};

use crate::tensor::element::*;
use crate::dimension::{MultiplyDimensions, SqrtDimension, Dimensionless};
use crate::complex::c64;
use crate::tensor::execution::{register_tensor_op, execute_graph};

// Computational Graph Operations - simplified to avoid function pointer issues
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    // Data operations (leaf nodes)
    Constant,
    
    // Unary operations
    Negate,
    Transpose,
    Reshape,
    Sum,
    Broadcast,
    Cut { start: usize },
    GetAt { layer: usize, row: usize, col: usize },
    SliceLayer { layer: usize },
    ComputeNorm,
    
    // Mathematical functions (for apply operations)
    Sin,
    Cos,
    Exp,
    Log,
    Sqrt,
    Abs,
    Conjugate,
    
    // Binary operations
    Add,
    Sub,
    Mul,
    Div,
    Hadamard,
    MatMul,
    
    // Matrix operations
    GetCol { col_idx: usize },
    GetRow { row_idx: usize },
    OuterProduct,
}

#[derive(Debug, Clone)]
pub struct Op<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize> 
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub op_type: OpType,
    pub data: Option<[E; LAYERS * ROWS * COLS]>,
    pub inputs: Vec<TensorId>,
}

// Unique identifier for tensors in the computational graph
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

// Global counter for tensor IDs (in a real implementation, you'd want better ID management)
static mut TENSOR_COUNTER: usize = 0;

pub fn next_tensor_id() -> TensorId {
    unsafe {
        let id = TENSOR_COUNTER;
        TENSOR_COUNTER += 1;
        TensorId(id)
    }
}

// The computational graph tensor
#[derive(Clone)]
pub struct Tensor<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub id: TensorId,
    pub op: Op<E, D, LAYERS, ROWS, COLS>,
    pub _phantom: PhantomData<D>,
}

impl<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize> Copy for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    Op<E, D, LAYERS, ROWS, COLS>: Copy,
{
}

impl<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Create a new tensor from a unit value
    pub fn new<U: crate::units::Unit<Dimension = D>>(data: [E; LAYERS * ROWS * COLS]) -> Self {
        let tensor_id = next_tensor_id();

        // Cache data immediately
        crate::tensor::execution::store_tensor_data::<E, { LAYERS * ROWS * COLS }>(tensor_id, data.clone());

        let tensor = Self {
            id: tensor_id,
            op: Op {
                op_type: OpType::Constant,
                data: Some(data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        };

        crate::tensor::execution::register_typed_tensor_op(tensor.id, &tensor.op, vec![]);
        tensor
    }

    /// Create a tensor with default data (const version, doesn't register in graph)
    pub const fn default(data: [E; LAYERS * ROWS * COLS]) -> Self {
        Self {
            id: TensorId(0), // This will be problematic for const contexts, but works for now
            op: Op {
                op_type: OpType::Constant,
                data: Some(data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        }
    }

    /// Create a tensor with default data and register in graph (non-const version)
    pub fn from_data(data: [E; LAYERS * ROWS * COLS]) -> Self {
        // Generate a unique tensor ID first so we can use it for both caching and the tensor itself
        let tensor_id = next_tensor_id();

        // Clone the data *before* moving it so we can place a copy into the global cache.
        let data_clone = data.clone();

        // Store the constant data in the global cache right away. This ensures that subsequent
        // graph executions can retrieve the values of constant tensors without having to realise
        // them first.
        crate::tensor::execution::store_tensor_data::<E, { LAYERS * ROWS * COLS }>(
            tensor_id,
            data_clone,
        );

        // Now construct the actual tensor, moving the original `data` into the Op definition.
        let tensor = Self {
            id: tensor_id,
            op: Op {
                op_type: OpType::Constant,
                data: Some(data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        };

        // Register with typed system - constants have no inputs but have their own shape
        crate::tensor::execution::register_typed_tensor_op(
            tensor.id,
            &tensor.op,
            vec![] // No input shapes for constants
        );

        tensor
    }

    /// Create a zero tensor (leaf node)
    pub fn zero() -> Self {
        let data: [E; LAYERS * ROWS * COLS] = [E::zero(); LAYERS * ROWS * COLS];
        let tensor_id = next_tensor_id();

        crate::tensor::execution::store_tensor_data::<E, { LAYERS * ROWS * COLS }>(tensor_id, data.clone());

        let tensor = Self {
            id: tensor_id,
            op: Op {
                op_type: OpType::Constant,
                data: Some(data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        };

        crate::tensor::execution::register_typed_tensor_op(tensor.id, &tensor.op, vec![]);
        tensor
    }

    /// Create a random tensor (leaf node)
    pub fn random<U: Unit<Dimension = D>>(min: E, max: E) -> Self {
        let base_min: E = E::from(U::to_base(min.into()));
        let base_max: E = E::from(U::to_base(max.into()));
        let data: [E; LAYERS * ROWS * COLS] = (0..LAYERS * ROWS * COLS)
            .map(|_| {
                E::from(U::from_base(
                    ((base_max - base_min) + base_min)
                        .weak_mul(rand::random::<f64>())
                        .into(),
                ))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let tensor_id = next_tensor_id();
        // cache
        crate::tensor::execution::store_tensor_data::<E, { LAYERS * ROWS * COLS }>(tensor_id, data.clone());

        let tensor = Self {
            id: tensor_id,
            op: Op {
                op_type: OpType::Constant,
                data: Some(data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        };

        crate::tensor::execution::register_typed_tensor_op(tensor.id, &tensor.op, vec![]);
        tensor
    }


    /// Initialize a tensor with a function that takes indices
    pub fn init(f: impl Fn(usize, usize, usize) -> E) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        [(); LAYERS * ROWS * COLS]:,
    {
        let mut data = [E::zero(); LAYERS * ROWS * COLS];
        for l in 0..LAYERS {
            for r in 0..ROWS {
                for c in 0..COLS {
                    let idx = l * (ROWS * COLS) + r * COLS + c;
                    data[idx] = f(l, r, c);
                }
            }
        }
        let tensor_id = next_tensor_id();
        crate::tensor::execution::store_tensor_data::<E, { LAYERS * ROWS * COLS }>(tensor_id, data.clone());

        let tensor = Tensor {
            id: tensor_id,
            op: Op {
                op_type: OpType::Constant,
                data: Some(data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        };

        crate::tensor::execution::register_typed_tensor_op(tensor.id, &tensor.op, vec![]);
        tensor
    }

    /// Initialize a 2D tensor
    pub fn init_2d(f: impl Fn(usize, usize) -> E) -> Tensor<E, D, 1, ROWS, COLS>
    where
        [(); 1 * ROWS * COLS]:,
    {
        Tensor::<E, D, 1, ROWS, COLS>::init(|_, r, c| f(r, c))
    }

    /// Add two tensors (creates graph node)
    pub fn add(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: Add<Output = E>,
    {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Add,
                data: None,
                inputs: vec![self.id, other.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        register_tensor_op(tensor.id, &tensor.op);
        tensor
    }

    /// Subtract two tensors (creates graph node)
    pub fn sub(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: Sub<Output = E>,
    {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Sub,
                data: None,
                inputs: vec![self.id, other.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        register_tensor_op(tensor.id, &tensor.op);
        tensor
    }

    /// Multiply two tensors element-wise (creates graph node)
    pub fn mul(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: Mul<Output = E>,
    {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Mul,
                data: None,
                inputs: vec![self.id, other.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        register_tensor_op(tensor.id, &tensor.op);
        tensor
    }

    /// Divide two tensors element-wise (creates graph node)
    pub fn div(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: Div<Output = E>,
    {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Div,
                data: None,
                inputs: vec![self.id, other.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        register_tensor_op(tensor.id, &tensor.op);
        tensor
    }

    /// Negate tensor (creates graph node)
    pub fn negate(&self) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: std::ops::Neg<Output = E>,
    {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Negate,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        register_tensor_op(tensor.id, &tensor.op);
        tensor
    }

    /// Execute the computational graph and return the computed data
    pub fn realize(&self) -> [E; LAYERS * ROWS * COLS] {
        crate::tensor::execution::execute_tensor(self.id, &self.op)
    }

    /// Get the underlying data reference (forces computation)
    /// Note: This returns a Vec to avoid lifetime issues with static storage
    pub fn data(&self) -> Vec<E> {
        self.realize().to_vec()
    }

    /// Get the underlying data (forces computation)
    pub fn get<S: Unit<Dimension = D>>(&self) -> [E; LAYERS * ROWS * COLS] {
        let data = self.realize();
        data.iter()
            .map(|&v| E::from(S::from_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Get the shape as a tuple
    pub fn shape(&self) -> (usize, usize, usize) {
        (LAYERS, ROWS, COLS)
    }

    /// Get a single element's raw value 
    pub fn raw(&self) -> E where [(); 1 * 1 * 1]: {
        assert_eq!(LAYERS * ROWS * COLS, 1, "raw() can only be called on scalar tensors");
        self.realize()[0]
    }

    /// Slice a specific layer from a 3D tensor
    pub fn slice_layer(&self, layer: usize) -> Tensor<E, D, 1, ROWS, COLS>
    where
        [(); 1 * ROWS * COLS]:,
    {
        assert!(layer < LAYERS, "Layer index {} out of bounds (max: {})", layer, LAYERS - 1);
        
        let tensor = Tensor::<E, D, 1, ROWS, COLS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::SliceLayer { layer },
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::SliceLayer { layer },
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Get element at specific position - returns the value directly as a tensor
    pub fn get_at(&self, layer: usize, row: usize, col: usize) -> Tensor<E, D, 1, 1, 1>
    where
        [(); 1 * 1 * 1]:,
    {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        
        let tensor = Tensor::<E, D, 1, 1, 1> {
            id: next_tensor_id(),
            op: Op::<E, D, 1, 1, 1> {
                op_type: OpType::GetAt { layer, row, col },
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::GetAt { layer, row, col },
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Set element at specific position (this would need special handling in a graph)
    pub fn set_at(&mut self, layer: usize, row: usize, col: usize, value: Scalar<E, D>) {
        // In a computational graph, mutations are tricky - you'd typically create a new tensor
        // This is a placeholder for now
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        todo!("Implement set_at for computational graph - consider using functional updates")
    }

    /// Transpose operation (creates graph node)
    pub fn transpose(self) -> Tensor<E, D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        let tensor = Tensor::<E, D, LAYERS, COLS, ROWS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Transpose,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Transpose,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Reshape operation (creates graph node)
    pub fn reshape<const L: usize, const R: usize, const C: usize>(&self) -> Tensor<E, D, L, R, C>
    where
        [(); L * R * C]:,
    {
        assert_eq!(
            LAYERS * ROWS * COLS,
            L * R * C,
            "Cannot reshape: sizes must match."
        );

        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Reshape,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Reshape,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Cut operation (creates graph node)
    pub fn cut<const LEN: usize>(&self, start: usize) -> Tensor<E, D, 1, 1, LEN>
    where
        [(); 1 * 1 * LEN]:,
        [(); LAYERS * ROWS * COLS]:,
    {
        assert!(start + LEN <= LAYERS * ROWS * COLS);

        let tensor = Tensor::<E, D, 1, 1, LEN> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Cut { start },
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Cut { start },
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Sum all elements to a scalar (creates graph node)
    pub fn sum(&self) -> Tensor<E, D, 1, 1, 1>
    where
        E: std::ops::Add<Output = E>,
        [(); 1 * 1 * 1]:,
    {
        let tensor = Tensor::<E, D, 1, 1, 1> {
            id: next_tensor_id(),
            op: Op::<E, D, 1, 1, 1> {
                op_type: OpType::Sum,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Sum,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Element-wise multiplication (Hadamard product) - creates graph node
    pub fn hadamard<DO: Clone>(
        &self, 
        other: &Tensor<E, DO, LAYERS, ROWS, COLS>
    ) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS>
    where
        E: std::ops::Mul<Output = E>,
        D: MultiplyDimensions<DO>,
        <D as MultiplyDimensions<DO>>::Output: Clone,
    {
        let tensor = Tensor::<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Hadamard,
                data: None,
                inputs: vec![self.id, other.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Hadamard,
            inputs: vec![self.id, other.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Dot product operation for vectors
    pub fn dot<DO: Clone>(&self, other: &Tensor<E, DO, LAYERS, ROWS, COLS>) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, 1, 1, 1>
    where
        E: std::ops::Mul<Output = E> + std::ops::Add<Output = E>,
        D: MultiplyDimensions<DO>,
        <D as MultiplyDimensions<DO>>::Output: Clone,
        [(); 1 * 1 * 1]:,
    {
        self.hadamard(other).sum()
    }

    /// Broadcast this tensor to a larger shape
    pub fn broadcast_to<const TL: usize, const TR: usize, const TC: usize>(
        &self,
    ) -> Tensor<E, D, TL, TR, TC>
    where
        [(); TL * TR * TC]:,
    {
        // Validate broadcasting at compile time
        if !((LAYERS == TL || LAYERS == 1 || TL == 1) &&
             (ROWS == TR || ROWS == 1 || TR == 1) &&
             (COLS == TC || COLS == 1 || TC == 1)) {
            panic!("Invalid broadcast: {:?} cannot be broadcast to {:?}", 
                   (LAYERS, ROWS, COLS), (TL, TR, TC));
        }
        
        let tensor = Tensor::<E, D, TL, TR, TC> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Broadcast,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register with proper compile-time type information
        crate::tensor::execution::register_typed_tensor_op(
            tensor.id, 
            &tensor.op,
            vec![(LAYERS, ROWS, COLS)] // Input shapes
        );
        
        tensor
    }

    /// Return the tensor ID (for debugging and graph introspection)
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Check if this tensor is a leaf node (constant data)
    pub fn is_leaf(&self) -> bool {
        matches!(self.op.op_type, OpType::Constant)
    }

    /// Get the operation type for debugging
    pub fn operation(&self) -> &Op<E, D, LAYERS, ROWS, COLS> {
        &self.op
    }

    /// Get a single column as a column vector - creates a graph node
    pub fn get_col(&self, col_idx: usize) -> Tensor<E, D, LAYERS, ROWS, 1>
    where
        [(); LAYERS * ROWS * 1]:,
    {
        assert!(col_idx < COLS, "Column index {} out of bounds (max: {})", col_idx, COLS - 1);
        
        // Create a graph node for column extraction
        Tensor::<E, D, LAYERS, ROWS, 1> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::GetCol { col_idx },
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        }
    }

    /// Get a single row as a row vector - creates a graph node  
    pub fn get_row(&self, row_idx: usize) -> Tensor<E, D, LAYERS, 1, COLS>
    where
        [(); LAYERS * 1 * COLS]:,
    {
        assert!(row_idx < ROWS, "Row index {} out of bounds (max: {})", row_idx, ROWS - 1);
        
        // Create a graph node for row extraction
        Tensor::<E, D, LAYERS, 1, COLS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::GetRow { row_idx },
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        }
    }

    /// Outer product using broadcast and hadamard - proper graph operations
    pub fn outer_product<DO: Clone, const R_OUT: usize, const C_OUT: usize> (
        vec1: &Tensor<E, D, LAYERS, R_OUT, 1>, 
        vec2: &Tensor<E, DO, LAYERS, 1, C_OUT>
    ) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, R_OUT, C_OUT> 
    where 
        E: std::ops::Mul<Output = E> + Copy,
        D: MultiplyDimensions<DO>, 
        <D as MultiplyDimensions<DO>>::Output: Clone,
        [(); LAYERS * R_OUT * 1]:, 
        [(); LAYERS * 1 * C_OUT]:, 
        [(); LAYERS * R_OUT * C_OUT]:, 
    {
        // Create graph node for outer product
        Tensor::<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, R_OUT, C_OUT> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::OuterProduct,
                data: None,
                inputs: vec![vec1.id, vec2.id],
            },
            _phantom: PhantomData,
        }
    }

    /// Apply sine function to each element (creates graph node)
    pub fn sin(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Sin,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Sin,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Apply cosine function to each element (creates graph node)
    pub fn cos(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Cos,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Cos,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Apply exponential function to each element (creates graph node)
    pub fn exp(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Exp,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Exp,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Apply natural logarithm function to each element (creates graph node)
    pub fn log(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Log,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Log,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Apply square root function to each element (creates graph node)
    pub fn sqrt(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Sqrt,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Sqrt,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Apply absolute value function to each element (creates graph node)
    pub fn abs(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Abs,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Abs,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Apply conjugate function to each element (creates graph node)
    pub fn conjugate(&self) -> Tensor<E, D, LAYERS, ROWS, COLS> {
        let tensor = Tensor {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Conjugate,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::Conjugate,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        tensor
    }

    /// Scale tensor by a scalar
    pub fn scale(&self, scalar: &Tensor<E, D, 1, 1, 1>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: Mul<Output = E>,
        [(); 1 * 1 * 1]:,
    {
        // Create a graph node for element-wise multiplication (broadcasting scalar)
        let tensor = Tensor::<E, D, LAYERS, ROWS, COLS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Mul,
                data: None,
                inputs: vec![self.id, scalar.id],
            },
            _phantom: PhantomData,
        };
        
        // Register operation in global graph
        crate::tensor::execution::register_tensor_op(tensor.id, &tensor.op);
        
        tensor
    }

    /// Comparison operations that return tensors (for graph operations)
    pub fn gt(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<f64, Dimensionless, LAYERS, ROWS, COLS>
    where
        E: PartialOrd,
        [(); LAYERS * ROWS * COLS]:,
    {
        // Create comparison tensor - for now, implement as immediate evaluation
        let self_data = self.realize();
        let other_data = other.realize();
        let mut result_data = [0.0f64; LAYERS * ROWS * COLS];
        
        for i in 0..(LAYERS * ROWS * COLS) {
            result_data[i] = if self_data[i] > other_data[i] { 1.0 } else { 0.0 };
        }
        
        Tensor::from_data(result_data)
    }

    /// Apply a function to each element (immediate evaluation for now)
    pub fn apply<F>(&self, f: F) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        F: Fn(E) -> E,
        E: Copy,
    {
        let current_data = self.realize();
        let new_data: [E; LAYERS * ROWS * COLS] = current_data.map(f);
        Tensor::from_data(new_data)
    }

    /// Logical AND operation - only for dimensionless scalars
    pub fn and(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: PartialEq + From<f64>,
        [(); LAYERS * ROWS * COLS]:,
    {
        let self_data = self.realize();
        let other_data = other.realize();
        let mut result_data = [E::from(0.0); LAYERS * ROWS * COLS];
        
        for i in 0..(LAYERS * ROWS * COLS) {
            let self_bool = self_data[i] != E::from(0.0);
            let other_bool = other_data[i] != E::from(0.0);
            result_data[i] = if self_bool && other_bool { E::from(1.0) } else { E::from(0.0) };
        }
        
        Tensor::from_data(result_data)
    }

    /// Debug: Visualize the computation graph leading to this tensor
    pub fn visualize_graph(&self) {
        crate::tensor::execution::visualize_tensor_graph(self.id);
    }
    
    /// Debug: Export the computation graph to DOT format for external visualization
    pub fn export_dot(&self) -> String {
        crate::tensor::execution::export_dot_graph(self.id)
    }
    
    /// Debug: Print basic information about this tensor
    pub fn debug_info(&self) {
        println!("=== TENSOR DEBUG INFO ===");
        println!("Tensor ID: {:?}", self.id.0);
        println!("Shape: {}x{}x{}", LAYERS, ROWS, COLS);
        println!("Size: {} elements", LAYERS * ROWS * COLS);
        println!("Operation: {:?}", self.op.op_type);
        println!("Inputs: {:?}", self.op.inputs.iter().map(|id| id.0).collect::<Vec<_>>());
        
        if self.op.data.is_some() {
            println!("Has constant data: YES");
        } else {
            println!("Has constant data: NO");
        }
        
        // Check if cached
        if let Ok(cache) = crate::tensor::execution::GLOBAL_CACHE.lock() {
            if cache.contains_key(&self.id) {
                println!("Cache status: CACHED");
            } else {
                println!("Cache status: NOT_CACHED");
            }
        }
        
        // Check if in typed graph
        if let Ok(typed_graph) = crate::tensor::execution::GLOBAL_TYPED_GRAPH.lock() {
            if typed_graph.contains_key(&self.id) {
                println!("Registration: TYPED");
            } else if let Ok(untyped_graph) = crate::tensor::execution::GLOBAL_GRAPH.lock() {
                if untyped_graph.contains_key(&self.id) {
                    println!("Registration: UNTYPED");
                } else {
                    println!("Registration: NOT_REGISTERED");
                }
            }
        }
        
        println!("=== END DEBUG INFO ===\n");
    }
}

// Implementation block for methods requiring SqrtDimension
impl<E, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Into<c64> + Copy,
    D: Clone + SqrtDimension,
    <D as SqrtDimension>::Output: Clone,
    [(); LAYERS * ROWS * COLS]:,
    [(); 1 * 1 * 1]: // For the scalar output tensor
{
    pub fn compute_norm(&self) -> Tensor<f64, <D as SqrtDimension>::Output, 1, 1, 1> {
        let tensor = Tensor::<f64, <D as SqrtDimension>::Output, 1, 1, 1> {
            id: next_tensor_id(),
            op: Op::<f64, <D as SqrtDimension>::Output, 1, 1, 1> {
                op_type: OpType::ComputeNorm,
                data: None,
                inputs: vec![self.id],
            },
            _phantom: PhantomData,
        };
        
        // Register in global graph using the any_op helper
        let any_op = crate::tensor::execution::AnyOp {
            op_type: OpType::ComputeNorm,
            inputs: vec![self.id],
            data: None,
        };
        
        crate::tensor::execution::register_any_op(tensor.id, any_op);
        
        tensor
    }
}

// Arithmetic operations that create graph nodes
impl<E: TensorElement + Add<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Add for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        (&self).add(&other)
    }
}

impl<E: TensorElement + Sub<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Sub for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        (&self).sub(&other)
    }
}

impl<E: TensorElement + Mul<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Mul for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        (&self).mul(&other)
    }
}

impl<E: TensorElement + Div<Output = E> + Copy, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Div for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        (&self).div(&other)
    }
}

// Specialized constructors for common tensor shapes
impl<E: TensorElement, D: Clone, const COLS_USIZE: usize> Tensor<E, D, 1, 1, COLS_USIZE>
where
    [(); 1 * 1 * COLS_USIZE]:,
{
    /// Construct a 1×1×COLS_USIZE tensor directly from the given data array
    pub fn from_array(data: [E; COLS_USIZE]) -> Self {
        let coerced: [E; 1 * 1 * COLS_USIZE] = unsafe { std::mem::transmute_copy(&data) };
        Self {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Constant,
                data: Some(coerced),
                inputs: vec![],
            },
            _phantom: PhantomData,
        }
    }
}

// Static factory methods
impl<E: TensorElement, D: Clone> Tensor<E, D, 0, 0, 0> {
    pub fn merge_cols<const LAYERS: usize, const ROWS: usize, const COL_DIM_SINGLE: usize, const NUM_COLS: usize>(
        columns: [Tensor<E, D, LAYERS, ROWS, COL_DIM_SINGLE>; NUM_COLS],
    ) -> Tensor<E, D, LAYERS, ROWS, NUM_COLS>
    where
        [(); LAYERS * ROWS * COL_DIM_SINGLE]:,
        [(); LAYERS * ROWS * NUM_COLS]:,
    {
        assert_eq!(COL_DIM_SINGLE, 1, "Input tensors must be single columns");

        let mut merged_data = [E::zero(); LAYERS * ROWS * NUM_COLS];

        for l in 0..LAYERS {
            for r in 0..ROWS {
                for col_idx in 0..NUM_COLS {
                    let val = columns[col_idx].realize()[l * ROWS + r];
                    let dst_idx = l * (ROWS * NUM_COLS) + r * NUM_COLS + col_idx;
                    merged_data[dst_idx] = val;
                }
            }
        }

        Tensor::<E, D, LAYERS, ROWS, NUM_COLS> {
            id: next_tensor_id(),
            op: Op {
                op_type: OpType::Constant,
                data: Some(merged_data),
                inputs: vec![],
            },
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Dimensionless;

    #[test]
    fn test_graph_execution_basic_ops() {
        // Create two constant tensors
        let a = Tensor::<f64, Dimensionless, 1, 2, 2>::from_data([1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::<f64, Dimensionless, 1, 2, 2>::from_data([5.0, 6.0, 7.0, 8.0]);
        
        // Test lazy addition - should create a graph node, not immediate computation
        let c = (&a).add(&b);
        assert!(!c.is_leaf()); // Should be a graph node, not a constant
        
        // When we realize it, it should compute the correct result
        let result = c.realize();
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
        
        // Test lazy subtraction
        let d = (&a).sub(&b);
        let sub_result = d.realize();
        assert_eq!(sub_result, [-4.0, -4.0, -4.0, -4.0]);
        
        // Test lazy multiplication
        let e = (&a).mul(&b);
        let mul_result = e.realize();
        assert_eq!(mul_result, [5.0, 12.0, 21.0, 32.0]);
        
        // Test chaining operations
        let f = (&c).add(&d); // (a + b) + (a - b) = 2a
        let chain_result = f.realize();
        assert_eq!(chain_result, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_graph_execution_unary_ops() {
        let a = Tensor::<f64, Dimensionless, 1, 2, 2>::from_data([1.0, 2.0, 3.0, 4.0]);
        
        // Test lazy negation
        let neg_a = (&a).negate();
        assert!(!neg_a.is_leaf());
        let neg_result = neg_a.realize();
        assert_eq!(neg_result, [-1.0, -2.0, -3.0, -4.0]);
        
        // Test transpose (creates graph node)
        let t = a.transpose();
        assert!(!t.is_leaf());
        let transpose_result = t.realize();
        // For a 2x2 matrix [1,2; 3,4], transpose should be [1,3; 2,4]
        assert_eq!(transpose_result, [1.0, 3.0, 2.0, 4.0]);
    }
}
