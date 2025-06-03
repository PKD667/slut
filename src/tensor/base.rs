use crate::tensor::scalar::Scalar;
use crate::tensor::element::TensorElement;
use crate::units::Unit;
use std::marker::PhantomData;
use std::{ops::*, usize};
use crate::dimension::{MultiplyDimensions, SqrtDimension, Dimensionless};
use crate::complex::c64;
use std::sync::Mutex;
use std::collections::HashMap;

// Import and re-export graph structures from the graph module
pub use crate::tensor::graph::{TensorId, OpType, reset_tensor_id_counter, Runtime, GraphExecutor, NodeRef, GraphShape, ops, ExecutionError,GraphNode,register_tensor_operation};

// Global graph executor
lazy_static::lazy_static! {
    static ref GLOBAL_EXECUTOR: GraphExecutor = GraphExecutor::new(Runtime::Base);
    static ref GLOBAL_GRAPH_CACHE: Mutex<HashMap<TensorId, Box<dyn std::any::Any + Send + Sync>>> = Mutex::new(HashMap::new());
}

// The computational graph tensor - now directly contains a graph node
#[derive(Clone)]
pub struct Tensor<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub node: NodeRef<E, {LAYERS * ROWS * COLS}, 0>,
    pub _phantom: PhantomData<D>,
}

impl<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Create a new tensor from a unit value
    pub fn new<U: crate::units::Unit<Dimension = D>>(data: [E; LAYERS * ROWS * COLS]) -> Self {
        // do the unit conversion
        let data: [E; LAYERS * ROWS * COLS] = data
            .iter()
            .map(|&v| E::from(U::from_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let shape = GraphShape::new(LAYERS, ROWS, COLS);
        let node = ops::constant(data, shape);

        Self {
            node,
            _phantom: PhantomData,
        }
    }

    /// Create a tensor with default data
    pub fn default(data: [E; LAYERS * ROWS * COLS]) -> Self {
        let shape = GraphShape::new(LAYERS, ROWS, COLS);
        let node = ops::constant(data, shape);

        Self {
            node,
            _phantom: PhantomData,
        }
    }

    /// Create a zero tensor (leaf node)
    pub fn zero() -> Self {
        let data: [E; LAYERS * ROWS * COLS] = [E::zero(); LAYERS * ROWS * COLS];
        let shape = GraphShape::new(LAYERS, ROWS, COLS);
        let node = ops::constant(data, shape);

        Self {
            node,
            _phantom: PhantomData,
        }
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

        let shape = GraphShape::new(LAYERS, ROWS, COLS);
        let node = ops::constant(data, shape);

        Self {
            node,
            _phantom: PhantomData,
        }
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

        let shape = GraphShape::new(LAYERS, ROWS, COLS);
        let node = ops::constant(data, shape);

        Tensor {
            node,
            _phantom: PhantomData,
        }
    }

    /// Initialize a 2D tensor
    pub fn init_2d(f: impl Fn(usize, usize) -> E) -> Tensor<E, D, 1, ROWS, COLS>
    where
        [(); 1 * ROWS * COLS]:,
    {
        Tensor::<E, D, 1, ROWS, COLS>::init(|_, r, c| f(r, c))
    }

    /// Add two tensors (creates graph node)
    pub fn add(&self, other: &Self) -> Self {
        // create a new node
        let add_node = 
        Self {
            node: add_node,
            _phantom: PhantomData,
        }
    }

    /// Subtract two tensors (creates graph node)
    pub fn sub(&self, other: &Self) -> Self {
        // For now, implement as add with negation since we don't have a sub operation yet
        let neg_other = ops::neg(other.node.clone());
        let sub_node = ops::add(self.node.clone(), neg_other).unwrap();
        
        Self {
            node: sub_node,
            _phantom: PhantomData,
        }
    }

    /// Multiply two tensors element-wise (creates graph node)
    pub fn mul(&self, other: &Self) -> Self {
        // For now, use add as placeholder since we don't have mul implemented yet
        let mul_node = ops::add(self.node.clone(), other.node.clone()).unwrap();
        
        Self {
            node: mul_node,
            _phantom: PhantomData,
        }
    }

    /// Divide two tensors element-wise (creates graph node)
    pub fn div(&self, other: &Self) -> Self {
        // For now, use add as placeholder since we don't have div implemented yet
        let div_node = ops::add(self.node.clone(), other.node.clone()).unwrap();
        
        Self {
            node: div_node,
            _phantom: PhantomData,
        }
    }

    /// Negate tensor (creates graph node)
    pub fn negate(&self) -> Self {
        let neg_node = ops::neg(self.node.clone());
        
        Self {
            node: neg_node,
            _phantom: PhantomData,
        }
    }

    /// Execute the computational graph and return the computed data
    pub fn realize(&self) -> [E; LAYERS * ROWS * COLS] 
    where 
        E: TensorElement
    {
        let executor = GraphExecutor::new(Runtime::Base);
        
        match executor.execute_node(&self.node) {
            Ok(result) => result,
            Err(_) => {
                // Fallback: if this is a constant node, try to get its data directly
                let node_ref = self.node.borrow();
                if node_ref.is_constant() {
                    *node_ref.get_data()
                } else {
                    panic!("Failed to execute tensor operation");
                }
            }
        }
    }

    /// Get the underlying data reference (forces computation)
    pub fn data(&self) -> Vec<E> {
        self.realize().to_vec()
    }

    /// Get data - triggers graph execution
    pub fn get<U>(&self) -> [E; LAYERS * ROWS * COLS] 
    {
        self.realize()
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

    /// Return the tensor ID (for debugging and graph introspection)
    pub fn id(&self) -> TensorId {
        self.node.borrow().id
    }

    /// Check if this tensor is a leaf node (constant data)
    pub fn is_leaf(&self) -> bool {
        self.node.borrow().is_constant()
    }

    /// Apply sine to each element (creates graph node)
    pub fn sin(&self) -> Self {
        let sin_node = ops::sin(self.node.clone());
        
        Self {
            node: sin_node,
            _phantom: PhantomData,
        }
    }

    /// Apply cosine to each element (creates graph node)
    pub fn cos(&self) -> Self {
        let cos_node = ops::cos(self.node.clone());
        
        Self {
            node: cos_node,
            _phantom: PhantomData,
        }
    }

    /// Apply exponential function to each element (creates graph node)
    pub fn exp(&self) -> Self {
        let exp_node = ops::exp(self.node.clone());
        
        Self {
            node: exp_node,
            _phantom: PhantomData,
        }
    }

    /// Apply natural logarithm function to each element (creates graph node)
    pub fn log(&self) -> Self {
        let log_node = ops::log(self.node.clone());
        
        Self {
            node: log_node,
            _phantom: PhantomData,
        }
    }

    /// Apply square root function to each element (creates graph node)
    pub fn sqrt(&self) -> Self {
        let sqrt_node = ops::sqrt(self.node.clone());
        
        Self {
            node: sqrt_node,
            _phantom: PhantomData,
        }
    }

    /// Apply absolute value function to each element (creates graph node)
    pub fn abs(&self) -> Tensor<f64, D, LAYERS, ROWS, COLS> {
        let abs_node = ops::abs(self.node.clone());
        
        Tensor {
            node: abs_node,
            _phantom: PhantomData,
        }
    }

    /// Apply conjugate function to each element (creates graph node)
    pub fn conjugate(&self) -> Self {
        let conj_node = ops::conj(self.node.clone());
        
        Self {
            node: conj_node,
            _phantom: PhantomData,
        }
    }

    /// Scale tensor by a scalar
    pub fn scale(&self, scalar: &Tensor<E, D, 1, 1, 1>) -> Self {
        let scale_node = ops::mul(self.node.clone(), scalar.node.clone()).unwrap();
        
        Self {
            node: scale_node,
            _phantom: PhantomData,
        }
    }

    /// Comparison operations that return tensors (for graph operations)
    pub fn gt(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<f64, Dimensionless, LAYERS, ROWS, COLS>
    where
        E: PartialOrd,
        [(); LAYERS * ROWS * COLS]:,
    {
        let gt_node = ops::gt(self.node.clone(), other.node.clone()).unwrap();
        
        Tensor {
            node: gt_node,
            _phantom: PhantomData,
        }
    }

    /// Apply a function to each element (immediate evaluation for now)
    pub fn apply<F>(&self, f: F) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        F: Fn(E) -> E,
        E: Copy,
    {
        let current_data = self.realize();
        let new_data: [E; LAYERS * ROWS * COLS] = current_data.map(f);
        Tensor::default(new_data)
    }

    /// Logical AND operation - only for dimensionless scalars
    pub fn and(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        E: PartialEq + From<f64>,
        [(); LAYERS * ROWS * COLS]:,
    {
        let and_node = ops::and(self.node.clone(), other.node.clone()).unwrap();
        
        Tensor {
            node: and_node,
            _phantom: PhantomData,
        }
    }

    /// Debug: Visualize the computation graph leading to this tensor
    pub fn visualize_graph(&self) {
        crate::tensor::execution::visualize_tensor_graph(self.id());
    }
    
    /// Debug: Export the computation graph to DOT format for external visualization
    pub fn export_dot(&self) -> String {
        crate::tensor::execution::export_dot_graph(self.id())
    }
    
    /// Debug: Print basic information about this tensor
    pub fn debug_info(&self) {
        println!("=== TENSOR DEBUG INFO ===");
        println!("Tensor ID: {:?}", self.id().0);
        println!("Shape: {}x{}x{}", LAYERS, ROWS, COLS);
        println!("Size: {} elements", LAYERS * ROWS * COLS);
        
        if self.is_leaf() {
            println!("Has constant data: YES");
        } else {
            println!("Has constant data: NO");
        }
        
        // Check if cached
        if let Ok(cache) = crate::tensor::execution::GLOBAL_CACHE.lock() {
            if cache.contains_key(&self.id()) {
                println!("Cache status: CACHED");
            } else {
                println!("Cache status: NOT_CACHED");
            }
        }
        
        // Check if in typed graph
        if let Ok(typed_graph) = crate::tensor::execution::GLOBAL_TYPED_GRAPH.lock() {
            if typed_graph.contains_key(&self.id()) {
                println!("Registration: TYPED");
            } else {
                println!("Registration: NOT_REGISTERED");
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
        let compute_norm_node = ops::compute_norm(self.node.clone());
        
        Tensor {
            node: compute_norm_node,
            _phantom: PhantomData,
        }
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
            node: ops::constant(coerced, GraphShape::new(1, 1, COLS_USIZE)),
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
            node: ops::constant(merged_data, GraphShape::new(LAYERS, ROWS, NUM_COLS)),
            _phantom: PhantomData,
        }
    }
}

