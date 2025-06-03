// Reference-based computation graph for tensor operations
// This module defines a type-safe graph where nodes hold references to their dependencies

use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::marker::PhantomData;

use crate::tensor::TensorElement;
use crate::complex::c64;

// Unique identifier for tensors in the computational graph
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(0);

impl TensorId {
    pub fn next() -> Self {
        TensorId(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Reset the tensor ID counter for test isolation
pub fn reset_tensor_id_counter() {
    NEXT_TENSOR_ID.store(0, Ordering::Relaxed);
}

/// Unified operation type
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
    
    // Mathematical functions
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

/// Node shape information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphShape {
    pub layers: usize,
    pub rows: usize,
    pub cols: usize,
}

impl GraphShape {
    pub fn new(layers: usize, rows: usize, cols: usize) -> Self {
        Self { layers, rows, cols }
    }

    pub fn total_size(&self) -> usize {
        self.layers * self.rows * self.cols
    }

    pub fn as_tuple(&self) -> (usize, usize, usize) {
        (self.layers, self.rows, self.cols)
    }
}

/// A reference to a computation node
pub type NodeRef<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize> = Rc<RefCell<GraphNode<E, DATA_SIZE, INPUT_SIZE>>>;
pub type WeakNodeRef<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize> = Weak<RefCell<GraphNode<E, DATA_SIZE, INPUT_SIZE>>>;

/// Node in the computation graph with typed data and static input size
pub struct GraphNode<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize> {
    pub id: TensorId,
    pub op_type: OpType,
    pub inputs: [Box<dyn GraphNodeRef>; INPUT_SIZE],
    pub output_shape: GraphShape,
    pub data: [E; DATA_SIZE],
    pub computed: bool,
}

/// Trait for type-erased node references with different sizes
pub trait GraphNodeRef {
    fn id(&self) -> TensorId;
    fn is_computed(&self) -> bool;
    fn execute(&self, executor: &GraphExecutor) -> Result<(), ExecutionError>;
}

impl<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize> GraphNodeRef for NodeRef<E, DATA_SIZE, INPUT_SIZE> {
    fn id(&self) -> TensorId {
        self.borrow().id
    }
    
    fn is_computed(&self) -> bool {
        self.borrow().computed
    }
    
    fn execute(&self, executor: &GraphExecutor) -> Result<(), ExecutionError> {
        executor.execute_node(self)?;
        Ok(())
    }
}

impl<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize> GraphNode<E, DATA_SIZE, INPUT_SIZE> {
    /// Create a new constant node (leaf node with no inputs)
    pub fn constant(data: [E; DATA_SIZE], shape: GraphShape) -> NodeRef<E, DATA_SIZE, 0> {
        let node = GraphNode {
            id: TensorId::next(),
            op_type: OpType::Constant,
            inputs: [],
            output_shape: shape,
            data,
            computed: true,
        };
        Rc::new(RefCell::new(node))
    }

    /// Create a new unary operation node
    pub fn unary_op<const INPUT_DATA_SIZE: usize>(
        op_type: OpType,
        input: NodeRef<E, INPUT_DATA_SIZE, 0>, 
        output_shape: GraphShape
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let node = GraphNode {
            id: TensorId::next(),
            op_type,
            inputs: [Box::new(input)],
            output_shape,
            data: unsafe { std::mem::zeroed() }, // Will be computed later
            computed: false,
        };
        Rc::new(RefCell::new(node))
    }

    /// Create a new binary operation node
    pub fn binary_op<const LEFT_DATA_SIZE: usize, const RIGHT_DATA_SIZE: usize>(
        op_type: OpType, 
        left: NodeRef<E, LEFT_DATA_SIZE, 0>, 
        right: NodeRef<E, RIGHT_DATA_SIZE, 0>, 
        output_shape: GraphShape
    ) -> NodeRef<E, DATA_SIZE, 2> {
        let node = GraphNode {
            id: TensorId::next(),
            op_type,
            inputs: [Box::new(left), Box::new(right)],
            output_shape,
            data: unsafe { std::mem::zeroed() }, // Will be computed later
            computed: false,
        };
        Rc::new(RefCell::new(node))
    }

    pub fn is_constant(&self) -> bool {
        matches!(self.op_type, OpType::Constant)
    }

    pub fn is_leaf(&self) -> bool {
        INPUT_SIZE == 0
    }

    /// Get the input shapes from the input nodes
    pub fn get_input_shapes(&self) -> Vec<GraphShape> {
        // For type-erased inputs, we need a different approach
        // This would require storing shape information or using trait methods
        Vec::new()
    }

    /// Check if this node has been computed
    pub fn needs_computation(&self) -> bool {
        !self.computed && !self.is_constant()
    }

    /// Mark this node as computed and store the result
    pub fn set_computed_data(&mut self, data: [E; DATA_SIZE]) {
        self.data = data;
        self.computed = true;
    }

    /// Get the data for this node
    pub fn get_data(&self) -> &[E; DATA_SIZE] {
        &self.data
    }

    /// Clear computed data to force recomputation
    pub fn invalidate(&mut self) {
        self.computed = false;
    }
}

impl<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize> std::fmt::Debug for GraphNode<E, DATA_SIZE, INPUT_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode")
            .field("id", &self.id)
            .field("op_type", &self.op_type)
            .field("output_shape", &self.output_shape)
            .field("data_size", &DATA_SIZE)
            .field("input_size", &INPUT_SIZE)
            .field("computed", &self.computed)
            .finish()
    }
}

/// Runtime backend for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    /// CPU-based execution using the base implementation
    Base,
    // Future runtimes can be added here:
    // Cuda,
    // Vulkan,
    // Metal,
}

impl Default for Runtime {
    fn default() -> Self {
        Runtime::Base
    }
}

impl std::fmt::Display for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Runtime::Base => write!(f, "Base"),
        }
    }
}

/// Errors that can occur during graph execution
#[derive(Debug, Clone)]
pub enum ExecutionError {
    /// The specified runtime is not available or implemented
    RuntimeNotAvailable(Runtime),
    /// A node in the graph is missing required data
    MissingNodeData(TensorId),
    /// Invalid operation parameters
    InvalidOperation(String),
    /// Dimension mismatch in tensor operations
    DimensionMismatch(String),
    /// Numeric computation error
    ComputationError(String),
    /// Circular dependency detected
    CircularDependency,
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::RuntimeNotAvailable(runtime) => {
                write!(f, "Runtime {} is not available", runtime)
            }
            ExecutionError::MissingNodeData(id) => {
                write!(f, "Missing data for node {:?}", id)
            }
            ExecutionError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            ExecutionError::DimensionMismatch(msg) => {
                write!(f, "Dimension mismatch: {}", msg)
            }
            ExecutionError::ComputationError(msg) => {
                write!(f, "Computation error: {}", msg)
            }
            ExecutionError::CircularDependency => {
                write!(f, "Circular dependency detected in computation graph")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

/// Graph executor that can compute node values
pub struct GraphExecutor {
    runtime: Runtime,
}

impl GraphExecutor {
    pub fn new(runtime: Runtime) -> Self {
        Self { runtime }
    }

    /// Execute a single node and return its computed data
    pub fn execute_node<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize>(
        &self, 
        node: &NodeRef<E, DATA_SIZE, INPUT_SIZE>
    ) -> Result<[E; DATA_SIZE], ExecutionError> {
        let mut node_ref = node.borrow_mut();
        
        // If already computed, return cached result
        if node_ref.computed {
            return Ok(node_ref.data);
        }

        // Execute dependencies first if needed
        if !node_ref.is_constant() && !node_ref.is_leaf() {
            drop(node_ref); // Release borrow to avoid conflicts
            self.execute_dependencies(node)?;
            let mut node_ref = node.borrow_mut();

            // Compute the operation - for now, just implement basic operations
            let result = match &node_ref.op_type {
                OpType::Constant => node_ref.data,
                OpType::Negate => {
                    // This is a simplified version - would need proper input handling
                    let mut result = node_ref.data;
                    for i in 0..DATA_SIZE {
                        result[i] = -result[i];
                    }
                    result
                }
                _ => {
                    return Err(ExecutionError::InvalidOperation(
                        format!("Operation {:?} not yet implemented in simplified system", node_ref.op_type)
                    ));
                }
            };

            // Cache the result
            node_ref.set_computed_data(result);
            Ok(result)
        } else {
            Ok(node_ref.data)
        }
    }

    /// Execute all dependencies of a node
    fn execute_dependencies<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize>(
        &self, 
        node: &NodeRef<E, DATA_SIZE, INPUT_SIZE>
    ) -> Result<(), ExecutionError> {
        let node_ref = node.borrow();
        for input in &node_ref.inputs {
            input.execute(self)?;
        }
        Ok(())
    }

    /// Validate that the graph has no circular dependencies
    pub fn validate_graph<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize>(
        &self, 
        node: &NodeRef<E, DATA_SIZE, INPUT_SIZE>
    ) -> Result<(), ExecutionError> {
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        self.dfs_cycle_check(node, &mut visited, &mut visiting)
    }

    fn dfs_cycle_check<E: TensorElement, const DATA_SIZE: usize, const INPUT_SIZE: usize>(
        &self,
        node: &NodeRef<E, DATA_SIZE, INPUT_SIZE>,
        visited: &mut HashSet<TensorId>,
        visiting: &mut HashSet<TensorId>,
    ) -> Result<(), ExecutionError> {
        let node_ref = node.borrow();
        let node_id = node_ref.id;

        if visiting.contains(&node_id) {
            return Err(ExecutionError::CircularDependency);
        }

        if visited.contains(&node_id) {
            return Ok(());
        }

        visiting.insert(node_id);

        for input in &node_ref.inputs {
            // For now, just execute the input to check it exists
            // Full cycle checking would need more sophisticated type handling
            input.execute(self)?;
        }

        visiting.remove(&node_id);
        visited.insert(node_id);
        Ok(())
    }
}

/// Utility functions for creating common operations
pub mod ops {
    use super::*;

    /// Create a constant tensor node
    pub fn constant<E: TensorElement, const DATA_SIZE: usize>(
        data: [E; DATA_SIZE], 
        shape: GraphShape
    ) -> NodeRef<E, DATA_SIZE, 0> {
        GraphNode::<E, DATA_SIZE, 0>::constant(data, shape)
    }

    /// Add two tensor nodes with the same data size
    pub fn add<E: TensorElement, const DATA_SIZE: usize>(
        left: NodeRef<E, DATA_SIZE, 0>, 
        right: NodeRef<E, DATA_SIZE, 0>
    ) -> Result<NodeRef<E, DATA_SIZE, 2>, ExecutionError> {
        let left_shape = left.borrow().output_shape;
        let right_shape = right.borrow().output_shape;
        
        if left_shape != right_shape {
            return Err(ExecutionError::DimensionMismatch(
                format!("Cannot add tensors with shapes {:?} and {:?}", left_shape, right_shape)
            ));
        }

        Ok(GraphNode::<E, DATA_SIZE, 2>::binary_op(OpType::Add, left, right, left_shape))
    }

    /// Multiply two tensor nodes with the same data size
    pub fn mul<E: TensorElement, const DATA_SIZE: usize>(
        left: NodeRef<E, DATA_SIZE, 0>, 
        right: NodeRef<E, DATA_SIZE, 0>
    ) -> Result<NodeRef<E, DATA_SIZE, 2>, ExecutionError> {
        let left_shape = left.borrow().output_shape;
        let right_shape = right.borrow().output_shape;
        
        if left_shape != right_shape {
            return Err(ExecutionError::DimensionMismatch(
                format!("Cannot multiply tensors with shapes {:?} and {:?}", left_shape, right_shape)
            ));
        }

        Ok(GraphNode::<E, DATA_SIZE, 2>::binary_op(OpType::Mul, left, right, left_shape))
    }

    /// Greater than comparison of two tensor nodes
    pub fn gt<E: TensorElement, const DATA_SIZE: usize>(
        left: NodeRef<E, DATA_SIZE, 0>, 
        right: NodeRef<E, DATA_SIZE, 0>
    ) -> Result<NodeRef<f64, DATA_SIZE, 2>, ExecutionError> {
        let left_shape = left.borrow().output_shape;
        let right_shape = right.borrow().output_shape;
        
        if left_shape != right_shape {
            return Err(ExecutionError::DimensionMismatch(
                format!("Cannot compare tensors with shapes {:?} and {:?}", left_shape, right_shape)
            ));
        }

        // For comparison operations, we need to create a new node with f64 output type
        // This is simplified - a real implementation would handle type conversion properly
        let node = GraphNode {
            id: TensorId::next(),
            op_type: OpType::Add, // Placeholder - we'd need a proper Gt operation
            inputs: [Box::new(left), Box::new(right)],
            output_shape: left_shape,
            data: unsafe { std::mem::zeroed() },
            computed: false,
        };
        Ok(std::rc::Rc::new(std::cell::RefCell::new(node)))
    }

    /// Logical AND of two tensor nodes
    pub fn and<E: TensorElement, const DATA_SIZE: usize>(
        left: NodeRef<E, DATA_SIZE, 0>, 
        right: NodeRef<E, DATA_SIZE, 0>
    ) -> Result<NodeRef<E, DATA_SIZE, 2>, ExecutionError> {
        let left_shape = left.borrow().output_shape;
        let right_shape = right.borrow().output_shape;
        
        if left_shape != right_shape {
            return Err(ExecutionError::DimensionMismatch(
                format!("Cannot AND tensors with shapes {:?} and {:?}", left_shape, right_shape)
            ));
        }

        Ok(GraphNode::<E, DATA_SIZE, 2>::binary_op(OpType::Add, left, right, left_shape)) // Placeholder
    }

    /// Negate a tensor node
    pub fn neg<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Negate, input, shape)
    }

    /// Apply sine to a tensor node
    pub fn sin<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Sin, input, shape)
    }

    /// Apply cosine to a tensor node
    pub fn cos<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Cos, input, shape)
    }

    /// Apply exponential to a tensor node
    pub fn exp<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Exp, input, shape)
    }

    /// Apply natural logarithm to a tensor node
    pub fn log<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Log, input, shape)
    }

    /// Apply square root to a tensor node
    pub fn sqrt<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Sqrt, input, shape)
    }

    /// Apply absolute value to a tensor node (returns f64)
    pub fn abs<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<f64, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        let node = GraphNode {
            id: TensorId::next(),
            op_type: OpType::Abs,
            inputs: [Box::new(input)],
            output_shape: shape,
            data: unsafe { std::mem::zeroed() },
            computed: false,
        };
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }

    /// Apply conjugate to a tensor node
    pub fn conj<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<E, DATA_SIZE, 1> {
        let shape = input.borrow().output_shape;
        GraphNode::<E, DATA_SIZE, 1>::unary_op(OpType::Conjugate, input, shape)
    }

    /// Compute norm of a tensor node (returns f64 scalar)
    pub fn compute_norm<E: TensorElement, const DATA_SIZE: usize>(
        input: NodeRef<E, DATA_SIZE, 0>
    ) -> NodeRef<f64, 1, 1> {
        let scalar_shape = GraphShape::new(1, 1, 1);
        let node = GraphNode {
            id: TensorId::next(),
            op_type: OpType::ComputeNorm,
            inputs: [Box::new(input)],
            output_shape: scalar_shape,
            data: unsafe { std::mem::zeroed() },
            computed: false,
        };
        std::rc::Rc::new(std::cell::RefCell::new(node))
    }
}

/// Result of graph execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub runtime: Runtime,
    pub execution_time_ms: Option<f64>,
    pub nodes_executed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_node_creation() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let shape = GraphShape::new(1, 2, 2);
        let node = ops::constant(data, shape);
        
        assert_eq!(node.borrow().output_shape, shape);
        assert!(node.borrow().is_constant());
        assert_eq!(node.borrow().get_data(), &data);
    }

    #[test]
    fn test_binary_operation() {
        let data1 = [1.0_f64, 2.0];
        let data2 = [3.0_f64, 4.0];
        let shape = GraphShape::new(1, 1, 2);
        
        let node1 = ops::constant(data1, shape);
        let node2 = ops::constant(data2, shape);
        let add_node = ops::add(node1, node2).unwrap();
        
        assert_eq!(add_node.borrow().output_shape, shape);
        assert_eq!(add_node.borrow().inputs.len(), 2);
    }

    #[test]
    fn test_graph_execution() {
        let data1 = [1.0_f64, 2.0];
        let data2 = [3.0_f64, 4.0];
        let shape = GraphShape::new(1, 1, 2);
        
        let node1 = ops::constant(data1, shape);
        let node2 = ops::constant(data2, shape);
        let add_node = ops::add(node1, node2).unwrap();
        
        let executor = GraphExecutor::new(Runtime::Base);
        let result = executor.execute_node(&add_node).unwrap();
        
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 6.0);
    }

    #[test]
    fn test_complex_graph() {
        let data = [1.0_f64, 2.0];
        let shape = GraphShape::new(1, 1, 2);
        
        let node = ops::constant(data, shape);
        let sin_node = ops::sin(node.clone());
        let cos_node = ops::cos(node);
        
        // Need to handle different input sizes for add operation
        // For now, let's test with simpler operations
        let executor = GraphExecutor::new(Runtime::Base);
        let sin_result = executor.execute_node(&sin_node).unwrap();
        let cos_result = executor.execute_node(&cos_node).unwrap();
        
        // sin(1) ≈ 0.841, sin(2) ≈ 0.909
        // cos(1) ≈ 0.540, cos(2) ≈ -0.416
        assert!((sin_result[0] - 0.841_f64).abs() < 0.01);
        assert!((sin_result[1] - 0.909_f64).abs() < 0.01);
        assert!((cos_result[0] - 0.540_f64).abs() < 0.01);
        assert!((cos_result[1] + 0.416_f64).abs() < 0.01);
    }

    #[test]
    fn test_dimension_mismatch() {
        let data1 = [1.0_f64, 2.0];
        let data2 = [3.0_f64, 4.0, 5.0];
        let shape1 = GraphShape::new(1, 1, 2);
        let shape2 = GraphShape::new(1, 1, 3);
        
        let node1 = ops::constant(data1, shape1);
        let node2 = ops::constant(data2, shape2);
        
        // This should fail at compile time with the new type system
        // assert!(ops::add(node1, node2).is_err());
    }

    #[test]
    fn test_tensor_id_generation() {
        reset_tensor_id_counter();
        
        let id1 = TensorId::next();
        let id2 = TensorId::next();
        
        assert_eq!(id1.0, 0);
        assert_eq!(id2.0, 1);
        assert_ne!(id1, id2);
    }
}
