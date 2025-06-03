// Graph execution engine - unified typed system

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Mutex, LazyLock};
use crate::tensor::graph::{TensorId, OpType};
use crate::tensor::element::TensorElement;
use crate::complex::c64;
use std::marker::PhantomData;

/// Simple untyped data storage for computed results
#[derive(Clone)]
pub struct TensorResult {
    pub data: Vec<u8>,
    pub size: usize,
}

/// Store operation metadata with compile-time shape information
#[derive(Clone)]
pub struct TypedAnyOp<
    E: TensorElement,
    const IL: usize, const IR: usize, const IC: usize,
    const OL: usize, const OR: usize, const OC: usize,
    const N_INPUTS: usize, // New const generic for number of inputs
>
where 
    [(); IL * IR * IC]:,
    [(); OL * OR * OC]:,
{
    pub op_type: OpType,
    pub inputs: [TensorId; N_INPUTS], // Use fixed-size array
    pub data: Option<[E; IL * IR * IC]>,
    pub input_shapes: [(usize, usize, usize); N_INPUTS], // Use fixed-size array
    pub output_shape: (usize, usize, usize),
    pub element_size: usize,
}

pub trait TypedOperation {
    fn op_type(&self) -> &OpType;
    fn inputs(&self) -> &[TensorId]; // Return a slice
    fn input_shapes(&self) -> &[(usize, usize, usize)]; // Return a slice
    fn output_shape(&self) -> (usize, usize, usize);
    fn element_size(&self) -> usize;
    fn const_bytes(&self) -> Option<&[u8]> { None }
}

impl<E: TensorElement, const IL: usize, const IR: usize, const IC: usize,
     const OL: usize, const OR: usize, const OC: usize, const N_INPUTS: usize> TypedOperation 
for TypedAnyOp<E, IL, IR, IC, OL, OR, OC, N_INPUTS>
where
    [(); IL * IR * IC]:,
    [(); OL * OR * OC]:,
{
    fn op_type(&self) -> &OpType { &self.op_type }
    fn inputs(&self) -> &[TensorId] { &self.inputs } // Return a slice to the array
    fn input_shapes(&self) -> &[(usize, usize, usize)] { &self.input_shapes } // Return a slice to the array
    fn output_shape(&self) -> (usize, usize, usize) { self.output_shape }
    fn element_size(&self) -> usize { self.element_size }

    fn const_bytes(&self) -> Option<&[u8]> {
        self.data.as_ref().map(|array_data| unsafe {
            core::slice::from_raw_parts(
                array_data.as_ptr() as *const u8,
                core::mem::size_of_val(array_data),
            )
        })
    }
}

/// Global cache for tensor computations
pub static GLOBAL_CACHE: LazyLock<Mutex<HashMap<TensorId, TensorResult>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

/// Global registry for tensor operations with proper type information
pub static GLOBAL_TYPED_GRAPH: LazyLock<Mutex<HashMap<TensorId, Box<dyn TypedOperation + Send + Sync>>>> = LazyLock::new(|| {
    Mutex::new(HashMap::new())
});

/// Register a tensor operation with full compile-time type information
pub fn register_typed_tensor_op<
    E: TensorElement, 
    D: Clone, 
    const LAYERS: usize, const ROWS: usize, const COLS: usize, 
    const N_INPUTS: usize, // Now a const generic parameter
>(
    tensor_id: TensorId,
    op: &Op<E, D, LAYERS, ROWS, COLS>,
    actual_input_shapes: [(usize, usize, usize); N_INPUTS],
    actual_inputs: [TensorId; N_INPUTS],
) where
    [(); LAYERS * ROWS * COLS]:,
    [(); N_INPUTS]: // Ensure N_INPUTS is valid for array ops
{
    // Assert that the number of inputs in op matches N_INPUTS
    // This is a runtime check to ensure consistency, though ideally N_INPUTS flows from op structure
    assert_eq!(op.inputs.len(), N_INPUTS, "Mismatch between N_INPUTS and op.inputs.len()");

    let typed_op = TypedAnyOp::<E, LAYERS, ROWS, COLS, LAYERS, ROWS, COLS, N_INPUTS> {
        op_type: op.op_type.clone(),
        inputs: actual_inputs, // op.inputs must be converted to [TensorId; N_INPUTS] before calling
        data: op.data,
        input_shapes: actual_input_shapes, // This will be passed directly
        output_shape: (LAYERS, ROWS, COLS),
        element_size: std::mem::size_of::<E>(),
    };
    
    GLOBAL_TYPED_GRAPH.lock().unwrap().insert(tensor_id, Box::new(typed_op));
}

/// Store tensor data globally
pub fn store_tensor_data<E: TensorElement, const SIZE: usize>(tensor_id: TensorId, data: [E; SIZE]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            std::mem::size_of::<[E; SIZE]>()
        ).to_vec()
    };
    
    if let Ok(mut cache) = GLOBAL_CACHE.lock() {
        cache.insert(tensor_id, TensorResult { data: bytes, size: SIZE });
    }
}

/// Get tensor data globally
pub fn get_tensor_data<E: TensorElement, const SIZE: usize>(tensor_id: TensorId) -> Option<[E; SIZE]> {
    if let Ok(cache) = GLOBAL_CACHE.lock() {
        if let Some(result) = cache.get(&tensor_id) {
            return Some(unsafe {
                let ptr = result.data.as_ptr() as *const E;
                let mut array = [E::zero(); SIZE];
                std::ptr::copy_nonoverlapping(ptr, array.as_mut_ptr(), SIZE);
                array
            });
        }
    }
    None
}

/// Recursively get input data for a tensor by looking up its operation and executing it
fn get_input_data_recursive<E: TensorElement, const SIZE: usize>(
    input_id: TensorId,
    visited: &mut HashSet<TensorId>
) -> [E; SIZE]
where 
    [(); SIZE]:,
{
    // Check for cycles - if we're already processing this tensor, return zeros to break the cycle
    if visited.contains(&input_id) {
        return [E::zero(); SIZE];
    }
    
    // Check if already computed and cached
    if let Some(cached) = get_tensor_data::<E, SIZE>(input_id) {
        return cached;
    }

    // Mark this tensor as being processed
    visited.insert(input_id);

    // Get operation data from the typed graph first, then release the lock
    let op_data = if let Ok(typed_graph) = GLOBAL_TYPED_GRAPH.lock() {
        if let Some(typed_op_box) = typed_graph.get(&input_id) {
            // Extract all the data we need while we have the lock
            Some((
                typed_op_box.op_type().clone(),
                typed_op_box.inputs().to_vec(),
                typed_op_box.const_bytes().map(|b| b.to_vec()),
                typed_op_box.input_shapes().to_vec(),
                typed_op_box.output_shape()
            ))
        } else {
            None
        }
    } else {
        None
    };
    
    // Now we've released the lock, process the operation
    let result = if let Some((op_type, inputs, const_data, input_shapes, output_shape)) = op_data {
        // If it's a constant, get its data directly
        if let Some(const_bytes) = const_data {
            if const_bytes.len() == SIZE * std::mem::size_of::<E>() {
                let result: [E; SIZE] = unsafe {
                    std::slice::from_raw_parts(
                        const_bytes.as_ptr() as *const E,
                        SIZE
                    ).try_into().unwrap()
                };
                store_tensor_data(input_id, result);
                result
            } else {
                [E::zero(); SIZE]
            }
        } else {
            // For non-constants, we need to execute the operation
            match &op_type {
                // Binary operations
                OpType::Add => {
                    if inputs.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = input1[i] + input2[i];
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Sub => {
                    if inputs.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = input1[i] - input2[i];
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Mul => {
                    if inputs.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = input1[i] * input2[i];
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Div => {
                    if inputs.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = input1[i] / input2[i];
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Hadamard => {
                    if inputs.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = input1[i] * input2[i];
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },

                // Unary operations
                OpType::Negate => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = -input1[i];
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },

                // Mathematical functions
                OpType::Sin => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            let complex_val: c64 = input1[i].into();
                            result[i] = E::from(complex_val.sin());
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Cos => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            let complex_val: c64 = input1[i].into();
                            result[i] = E::from(complex_val.cos());
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Exp => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            let complex_val: c64 = input1[i].into();
                            result[i] = E::from(complex_val.exp());
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Log => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            let complex_val: c64 = input1[i].into();
                            result[i] = E::from(complex_val.ln());
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Sqrt => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            let complex_val: c64 = input1[i].into();
                            result[i] = E::from(complex_val.sqrt());
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Abs => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            let mag = input1[i].mag();
                            result[i] = E::from(c64::new(mag, 0.0));
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Conjugate => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut result = [E::zero(); SIZE];
                        for i in 0..SIZE {
                            result[i] = input1[i].conjugate();
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },

                // Shape operations
                OpType::Transpose => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (layers, rows, cols) = input_shapes[0];
                        let mut result = [E::zero(); SIZE];
                        
                        // Transpose: (l, r, c) -> (l, c, r)
                        for l in 0..layers {
                            for r in 0..rows {
                                for c in 0..cols {
                                    let src_idx = l * (rows * cols) + r * cols + c;
                                    let dst_idx = l * (cols * rows) + c * rows + r;
                                    if src_idx < SIZE && dst_idx < SIZE {
                                        result[dst_idx] = input1[src_idx];
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Reshape => {
                    if inputs.len() == 1 {
                        // For reshape, just copy the data as-is since total size is preserved
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        store_tensor_data(input_id, input1);
                        input1
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Sum => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        println!("Sum input: {:?}", input1);
                        let mut sum = E::zero();
                        for element in input1.iter() {
                            sum = sum + *element;
                        }
                        
                        println!("Sum result: {:?}", sum);

                        // For sum, SIZE should be 1 (scalar result)
                        let mut result = [E::zero(); SIZE];
                        if SIZE >= 1 {
                            result[0] = sum;
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Broadcast => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (src_layers, src_rows, src_cols) = input_shapes[0];
                        let (dst_layers, dst_rows, dst_cols) = output_shape;
                        let mut result = [E::zero(); SIZE];
                        
                        for l in 0..dst_layers {
                            for r in 0..dst_rows {
                                for c in 0..dst_cols {
                                    let src_l = if src_layers == 1 { 0 } else { l % src_layers };
                                    let src_r = if src_rows == 1 { 0 } else { r % src_rows };
                                    let src_c = if src_cols == 1 { 0 } else { c % src_cols };
                                    
                                    let src_idx = src_l * (src_rows * src_cols) + src_r * src_cols + src_c;
                                    let dst_idx = l * (dst_rows * dst_cols) + r * dst_cols + c;
                                    
                                    if src_idx < input1.len() && dst_idx < result.len() {
                                        result[dst_idx] = input1[src_idx];
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },

                // Indexing operations
                OpType::SliceLayer { layer } => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (layers, rows, cols) = input_shapes[0];
                        let mut result = [E::zero(); SIZE];
                        
                        if *layer < layers {
                            let layer_size = rows * cols;
                            let src_start = layer * layer_size;
                            for i in 0..layer_size.min(SIZE) {
                                if src_start + i < input1.len() {
                                    result[i] = input1[src_start + i];
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::GetAt { layer, row, col } => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (layers, rows, cols) = input_shapes[0];
                        let mut result = [E::zero(); SIZE];
                        
                        if *layer < layers && *row < rows && *col < cols {
                            let idx = layer * (rows * cols) + row * cols + col;
                            if idx < input1.len() && SIZE >= 1 {
                                result[0] = input1[idx];
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::GetCol { col_idx } => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (layers, rows, cols) = input_shapes[0];
                        let mut result = [E::zero(); SIZE];
                        
                        if *col_idx < cols {
                            for l in 0..layers {
                                for r in 0..rows {
                                    let src_idx = l * (rows * cols) + r * cols + col_idx;
                                    let dst_idx = l * rows + r;
                                    if src_idx < input1.len() && dst_idx < result.len() {
                                        result[dst_idx] = input1[src_idx];
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::GetRow { row_idx } => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (layers, rows, cols) = input_shapes[0];
                        let mut result = [E::zero(); SIZE];
                        
                        if *row_idx < rows {
                            for l in 0..layers {
                                for c in 0..cols {
                                    let src_idx = l * (rows * cols) + row_idx * cols + c;
                                    let dst_idx = l * cols + c;
                                    if src_idx < input1.len() && dst_idx < result.len() {
                                        result[dst_idx] = input1[src_idx];
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::Cut { start } => {
                    if inputs.len() == 1 && input_shapes.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let (layers, rows, cols) = input_shapes[0];
                        let mut result = [E::zero(); SIZE];
                        
                        // Cut from start index in the column dimension
                        for l in 0..layers {
                            for r in 0..rows {
                                for c in 0..(cols - start).min(SIZE / (layers * rows)) {
                                    let src_idx = l * (rows * cols) + r * cols + (start + c);
                                    let dst_idx = l * (rows * (cols - start)) + r * (cols - start) + c;
                                    if src_idx < input1.len() && dst_idx < result.len() {
                                        result[dst_idx] = input1[src_idx];
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },

                // Matrix operations
                OpType::MatMul => {
                    if inputs.len() == 2 && input_shapes.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let (layers1, rows1, common) = input_shapes[0];
                        let (layers2, common2, cols2) = input_shapes[1];
                        let mut result = [E::zero(); SIZE];
                        
                        if common == common2 && layers1 == layers2 {
                            for l in 0..layers1 {
                                for r in 0..rows1 {
                                    for c in 0..cols2 {
                                        let mut sum = E::zero();
                                        for k in 0..common {
                                            let a_idx = l * (rows1 * common) + r * common + k;
                                            let b_idx = l * (common * cols2) + k * cols2 + c;
                                            if a_idx < input1.len() && b_idx < input2.len() {
                                                sum = sum + input1[a_idx] * input2[b_idx];
                                            }
                                        }
                                        let result_idx = l * (rows1 * cols2) + r * cols2 + c;
                                        if result_idx < result.len() {
                                            result[result_idx] = sum;
                                        }
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::OuterProduct => {
                    if inputs.len() == 2 && input_shapes.len() == 2 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let input2 = get_input_data_recursive::<E, SIZE>(inputs[1], visited);
                        let (layers1, rows1, _) = input_shapes[0]; // Should be (L, R, 1)
                        let (layers2, _, cols2) = input_shapes[1]; // Should be (L, 1, C)
                        let mut result = [E::zero(); SIZE];
                        
                        if layers1 == layers2 {
                            for l in 0..layers1 {
                                for r in 0..rows1 {
                                    for c in 0..cols2 {
                                        let a_idx = l * rows1 + r;
                                        let b_idx = l * cols2 + c;
                                        let result_idx = l * (rows1 * cols2) + r * cols2 + c;
                                        
                                        if a_idx < input1.len() && b_idx < input2.len() && result_idx < result.len() {
                                            result[result_idx] = input1[a_idx] * input2[b_idx];
                                        }
                                    }
                                }
                            }
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },
                OpType::ComputeNorm => {
                    if inputs.len() == 1 {
                        let input1 = get_input_data_recursive::<E, SIZE>(inputs[0], visited);
                        let mut sum_sq = 0.0;
                        for element in input1.iter() {
                            let mag = element.mag();
                            sum_sq += mag * mag;
                        }
                        let norm = sum_sq.sqrt();
                        
                        // ComputeNorm should return a scalar (SIZE=1)
                        let mut result = [E::zero(); SIZE];
                        if SIZE >= 1 {
                            result[0] = E::from(c64::new(norm, 0.0));
                        }
                        store_tensor_data(input_id, result);
                        result
                    } else { panic!("Mismatch for operation {:?} on tensor {:?}", op_type, input_id) }
                },

                // Default case for unimplemented operations
                _ => panic!("Unimplemented operation: {:?} for tensor {:?}", op_type, input_id),
            }
        }
    } else {
        panic!("No operation found for tensor {:?}", input_id)
    };

    // Remove from visited set before returning
    visited.remove(&input_id);
    result
}

/// Entry point for recursive computation - creates the visited set
fn get_input_data_safe<E: TensorElement, const SIZE: usize>(input_id: TensorId) -> [E; SIZE]
where 
    [(); SIZE]:,
{
    let mut visited = HashSet::new();
    get_input_data_recursive::<E, SIZE>(input_id, &mut visited)
}

/// Execute the computational graph for a tensor
pub fn execute_graph<E: crate::tensor::element::TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>(
    tensor_id: TensorId,
    op: &crate::tensor::base::Op<E, D, LAYERS, ROWS, COLS>,
) -> [E; LAYERS * ROWS * COLS] {
    get_input_data_safe::<E, { LAYERS * ROWS * COLS }>(tensor_id)
}

/// Execute tensor operation - main entry point for tensor evaluation
pub fn execute_tensor<E: crate::tensor::element::TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>(
    tensor_id: TensorId,
    op: &crate::tensor::base::Op<E, D, LAYERS, ROWS, COLS>,
) -> [E; LAYERS * ROWS * COLS] {
    execute_graph(tensor_id, op)
}

/// Visualize the entire computation graph
pub fn visualize_full_graph() {
    println!("=== FULL COMPUTATION GRAPH ===");
    
    let typed_graph_guard = GLOBAL_TYPED_GRAPH.lock().unwrap();
    let cache_guard = GLOBAL_CACHE.lock().unwrap();

    let typed_ops_refs: Vec<(TensorId, &Box<dyn TypedOperation + Send + Sync>)> = 
        typed_graph_guard.iter().map(|(id, op_box)| (*id, op_box)).collect();
    
    let cached_tensors_cloned: Vec<(TensorId, usize)> = cache_guard.iter()
        .map(|(id, result)| (*id, result.size))
        .collect();
    
    // Now print everything, guards will be dropped at the end of the function
    println!("Typed operations ({} nodes):", typed_ops_refs.len());
    for (id, op_box) in typed_ops_refs {
        println!("  Tensor {:?}:", id.0);
        println!("    Type: {:?}", op_box.op_type());
        println!("    Inputs: {:?}", op_box.inputs().iter().map(|tid| tid.0).collect::<Vec<_>>());
        println!("    Input shapes: {:?}", op_box.input_shapes());
        println!("    Output shape: {:?}", op_box.output_shape());
        println!("    Element size: {} bytes", op_box.element_size());
        if op_box.const_bytes().is_some() {
            println!("    Has constant data: YES");
        }
        println!();
    }
    
    println!("Cached results ({} tensors):", cached_tensors_cloned.len());
    for (id, size) in cached_tensors_cloned {
        println!("  Tensor {:?}: {} elements", id.0, size);
    }
    
    println!("=== END GRAPH ===\n");
}

/// Visualize the computation subgraph leading to a specific tensor
pub fn visualize_tensor_graph(tensor_id: TensorId) {
    println!("=== TENSOR GRAPH: {:?} ===", tensor_id.0);
    
    let typed_graph_guard = GLOBAL_TYPED_GRAPH.lock().unwrap();
    
    // Build a dependency tree using BFS
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut dependency_tree = HashMap::new();
    
    queue.push_back(tensor_id);
    visited.insert(tensor_id);
    
    while let Some(current_id) = queue.pop_front() {
        if let Some(op_box) = typed_graph_guard.get(&current_id) {
            let inputs = op_box.inputs().to_vec();
            dependency_tree.insert(current_id, (op_box.op_type().clone(), inputs.clone(), Some(op_box.output_shape())));
            
            for &input_id in &inputs {
                if !visited.contains(&input_id) {
                    visited.insert(input_id);
                    queue.push_back(input_id);
                }
            }
        }
    }
    
    // Display the subgraph in topological order
    let sorted = topological_sort(&dependency_tree);
    
    println!("Execution order (topologically sorted):");
    for (depth, node_id) in sorted.iter().enumerate() {
        if let Some((op_type, inputs, shape)) = dependency_tree.get(node_id) {
            let shape_str = shape.map_or("unknown".to_string(), |s| format!("{:?}", s));
            println!("  {}: Tensor {:?} <- {:?} (inputs: {:?}) [shape: {}]", 
                depth, node_id.0, op_type, inputs.iter().map(|id| id.0).collect::<Vec<_>>(), shape_str);
        }
    }
    
    let depth = calculate_depth(tensor_id, &dependency_tree);
    println!("Graph depth: {}", depth);
    println!("Total nodes: {}", dependency_tree.len());
    
    println!("=== END TENSOR GRAPH ===\n");
}

fn calculate_depth(tensor_id: TensorId, dependency_tree: &std::collections::HashMap<TensorId, (OpType, Vec<TensorId>, Option<(usize, usize, usize)>)>) -> usize {
    if let Some((_, inputs, _)) = dependency_tree.get(&tensor_id) {
        if inputs.is_empty() {
            1
        } else {
            1 + inputs.iter().map(|&input_id| calculate_depth(input_id, dependency_tree)).max().unwrap_or(0)
        }
    } else {
        0
    }
}

fn topological_sort(dependency_tree: &std::collections::HashMap<TensorId, (OpType, Vec<TensorId>, Option<(usize, usize, usize)>)>) -> Vec<TensorId> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut temp_visited = HashSet::new();
    
    fn dfs(
        node: TensorId,
        dependency_tree: &std::collections::HashMap<TensorId, (OpType, Vec<TensorId>, Option<(usize, usize, usize)>)>,
        visited: &mut HashSet<TensorId>,
        temp_visited: &mut HashSet<TensorId>,
        result: &mut Vec<TensorId>
    ) {
        if temp_visited.contains(&node) {
            return; // Cycle detected, but we'll ignore for now
        }
        if visited.contains(&node) {
            return;
        }
        
        temp_visited.insert(node);
        
        if let Some((_, inputs, _)) = dependency_tree.get(&node) {
            for &input_id in inputs {
                dfs(input_id, dependency_tree, visited, temp_visited, result);
            }
        }
        
        temp_visited.remove(&node);
        visited.insert(node);
        result.push(node);
    }
    
    for &node_id in dependency_tree.keys() {
        if !visited.contains(&node_id) {
            dfs(node_id, dependency_tree, &mut visited, &mut temp_visited, &mut result);
        }
    }
    
    result
}

/// Export the computation graph to DOT format for external visualization
pub fn export_dot_graph(tensor_id: TensorId) -> String {
    let typed_graph_guard = GLOBAL_TYPED_GRAPH.lock().unwrap();
    
    let mut dot = String::new();
    dot.push_str("digraph ComputationGraph {\n");
    dot.push_str("  rankdir=BT;\n");  // Bottom to top
    dot.push_str("  node [shape=box];\n");
    
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut nodes = std::collections::HashMap::new();
    
    queue.push_back(tensor_id);
    visited.insert(tensor_id);
    
    while let Some(current_id) = queue.pop_front() {
        if let Some(op_box) = typed_graph_guard.get(&current_id) {
            let label = format!("T{:?}: {:?}\\nShape: {:?}", current_id.0, op_box.op_type(), op_box.output_shape());
            nodes.insert(current_id, label);
            
            for &input_id in op_box.inputs() {
                if !visited.contains(&input_id) {
                    visited.insert(input_id);
                    queue.push_back(input_id);
                }
            }
        }
    }
    
    for (id, label) in &nodes {
        dot.push_str(&format!("  T{:?} [label=\"{}\"];\n", id.0, label));
    }
    
    for current_id_ref in nodes.keys() {
        let current_id = *current_id_ref;
        if let Some(op_box) = typed_graph_guard.get(&current_id) {
            for &input_id in op_box.inputs() {
                dot.push_str(&format!("  T{:?} -> T{:?};\n", input_id.0, current_id.0));
            }
        }
    }
    
    dot.push_str("}\n");
    dot
}

/// Analyze graph properties and potential issues
pub fn analyze_graph() {
    println!("=== GRAPH ANALYSIS ===");
    
    let (typed_ops_count, cached_count) = {
        let typed_graph_guard = GLOBAL_TYPED_GRAPH.lock().unwrap();
        let typed_ops_count = typed_graph_guard.len();
        drop(typed_graph_guard);
        
        let cache_guard = GLOBAL_CACHE.lock().unwrap();
        let cached_count = cache_guard.len();
        drop(cache_guard);
        
        (typed_ops_count, cached_count)
    };
    
    let mut op_counts = std::collections::HashMap::new();
    let typed_graph_op_types: Vec<OpType> = GLOBAL_TYPED_GRAPH.lock().unwrap().values().map(|b| b.op_type().clone()).collect();

    for op_type in typed_graph_op_types.iter() {
        *op_counts.entry(op_type.clone()).or_insert(0) += 1;
    }
    
    println!("Total typed operations: {}", typed_ops_count);
    println!("Cached results: {}", cached_count);
    println!("Cache hit rate: {:.2}%", if typed_ops_count > 0 { 
        (cached_count as f64 / typed_ops_count as f64) * 100.0 
    } else { 
        0.0 
    });
    
    println!("Operation distribution:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    
    for (op_type, count) in sorted_ops {
        println!("  {:?}: {}", op_type, count);
    }
    
    // Detect potential issues
    if cached_count < typed_ops_count / 2 {
        println!("⚠️  WARNING: Low cache hit rate - many operations may be recomputed");
    }
    
    if typed_ops_count > 1000 {
        println!("⚠️  WARNING: Large computation graph ({} operations) - consider optimization", typed_ops_count);
    }
    
    println!("=== END ANALYSIS ===\n");
}

/// Clear all global state for test isolation
pub fn clear_global_state() {
    // Clear the cache
    if let Ok(mut cache) = GLOBAL_CACHE.lock() {
        cache.clear();
    }
    
    // Clear the typed graph
    if let Ok(mut typed_graph) = GLOBAL_TYPED_GRAPH.lock() {
        typed_graph.clear();
    }
    
    // Reset tensor ID counter
    crate::tensor::base::reset_tensor_id_counter();
}