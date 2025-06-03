#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]
#![allow(mixed_script_confusables)]
#![feature(lazy_type_alias)]
#![feature(const_type_id)]
#![allow(incomplete_features)]


pub mod tensor;


pub mod dimension;
use dimension::*;

pub mod units;
use units::*;

pub mod si;


pub mod complex;

/// Convenience functions for graph debugging and visualization
pub mod debug {
    /// Visualize the entire computation graph
    pub fn visualize_full_graph() {
        crate::tensor::execution::visualize_full_graph();
    }
    
    /// Analyze the graph for potential issues and statistics
    pub fn analyze_graph() {
        crate::tensor::execution::analyze_graph();
    }
}


#[cfg(test)]
mod tests {
    use crate::si::{Force, Joule, MetersPerSecond, Pascal};

    use super::*;
    use tensor::base::Tensor;
    use tensor::element::TensorElement;
    use complex::c64;
    use dimension::Dimensionless;
    use units::Unitless;

    #[test]
    fn test_constant_tensor_creation() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Test basic constant tensor creation
        let tensor: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([1.0, 2.0]);
        let data = tensor.get::<Unitless>();
        assert_eq!(data, [1.0, 2.0]);
    }

    #[test] 
    fn test_tensor_add() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Test basic addition
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([1.0, 2.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([3.0, 4.0]);
        let result = a.add(&b);
        let data = result.get::<Unitless>();
        assert_eq!(data, [4.0, 6.0]);
    }

    #[test]
    fn test_tensor_sub() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Test basic subtraction
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([5.0, 7.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([2.0, 3.0]);
        let result = a.sub(&b);
        let data = result.get::<Unitless>();
        assert_eq!(data, [3.0, 4.0]);
    }

    #[test]
    fn test_tensor_mul() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Test basic element-wise multiplication
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([2.0, 3.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([4.0, 5.0]);
        let result = a.mul(&b);
        println!("Result tensor: {}", result);
        let data = result.get::<Unitless>();
        assert_eq!(data, [8.0, 15.0]);

        // test with some physical units
        let a: Tensor<f64, Length, 1, 2, 1> = Tensor::default([2.0, 3.0]);
        let b: Tensor<f64, Force, 1, 2, 1> = Tensor::default([4.0, 5.0]);
        let result = &a * &b;
        println!("Result tensor: {}", result);
        let data = result.get::<Joule>();
        assert_eq!(data, [8.0, 15.0]);
    }

    #[test]
    fn test_tensor_div() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Test basic element-wise division
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([8.0, 15.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([2.0, 3.0]);
        let result = a.div(&b);

        println!("Result tensor: {:?}", result);
        let data = result.get::<Unitless>();
        assert_eq!(data, [4.0, 5.0]);
    }

    #[test]
    fn test_dimension_consistency_mul_div() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Length * Force -> Energy (Joule)
        let len: Tensor<f64, Length, 1, 1, 2> = Tensor::default([2.0, 3.0]);
        let force: Tensor<f64, Force, 1, 1, 2> = Tensor::default([4.0, 5.0]);
        let energy = &len * &force;
        let energy_vals = energy.get::<Joule>();
        assert_eq!(energy_vals, [8.0, 15.0]);

        // Length / Time -> Velocity (m/s)
        let time: Tensor<f64, Time, 1, 1, 2> = Tensor::default([1.0, 2.0]);
        let velocity = &len / &time; // dimension-aware Div impl
        let vel_vals = velocity.get::<MetersPerSecond>();
        assert_eq!(vel_vals, [2.0, 1.5]);

        // Force / (Length*Length) -> Pressure (Pascal)
        let width: Tensor<f64, Length, 1, 1, 2> = Tensor::default([1.0, 2.0]);
        let area = &len * &width; // Length^2 -> Area
        
        let force2: Tensor<f64, Force, 1, 1, 2> = Tensor::default([2.0, 5.0]);
        let pressure = &force2 / &area;
        let pres_vals = pressure.get::<Pascal>();
        assert_eq!(pres_vals, [1.0, 0.8333333333333334]); // 2.0/2.0=1.0, 5.0/(3.0*2.0)=5.0/6.0≈0.833
    }

    #[test]
    fn test_graph_visualization() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        // Test the new graph visualization system
        println!("\n=== TESTING GRAPH VISUALIZATION ===");
        
        // Create a more complex computation to visualize
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([1.0, 2.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([3.0, 4.0]);
        let c: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([5.0, 6.0]);

        println!("Input tensor A:");
        a.debug_info();
        
        // Create a computation: (a + b) * c
        let sum = a.add(&b);
        let result = sum.mul(&c);
        
        // Show debug info for each tensor
        println!("Input tensor A:");
        a.debug_info();
        
        println!("Input tensor B:");
        b.debug_info();
        
        println!("Input tensor C:");
        c.debug_info();
        
        println!("Sum tensor (a + b):");
        sum.debug_info();
        
        println!("Result tensor ((a + b) * c):");
        result.debug_info();
        
        // Visualize the computation graph for the result
        println!("Graph visualization for result tensor:");
        result.visualize_graph();
        
        // Show full graph analysis
        crate::tensor::execution::analyze_graph();
        
        // Export DOT format for external tools
        let dot_graph = result.export_dot();
        println!("DOT graph representation:");
        println!("{}", dot_graph);
        
        // Now execute and verify the result
        let data = result.get::<Unitless>();
        assert_eq!(data, [20.0, 36.0]); // (1+3)*5 = 20, (2+4)*6 = 36
        
        println!("=== Graph visualization test passed! ===\n");
    }

    #[test]
    fn test_comprehensive_operations() {
        // Clear global state to ensure test isolation
        crate::tensor::execution::clear_global_state();
        
        println!("\n=== TESTING COMPREHENSIVE TENSOR OPERATIONS ===");
        
        // Test unary operations
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([1.0, -2.0]);
        
        // Test negate
        let neg_a = a.negate();
        let neg_data = neg_a.get::<Unitless>();
        assert_eq!(neg_data, [-1.0, 2.0]);
        println!("✓ Negate operation works");
        
        // Test mathematical functions
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([0.0, 1.0]);
        
        let sin_b = b.sin();
        let sin_data = sin_b.get::<Unitless>();
        assert!((sin_data[0] - 0.0).abs() < 1e-10);
        assert!((sin_data[1] - 1.0_f64.sin()).abs() < 1e-10);
        println!("✓ Sin operation works");
        
        let cos_b = b.cos();
        let cos_data = cos_b.get::<Unitless>();
        assert!((cos_data[0] - 1.0).abs() < 1e-10);
        assert!((cos_data[1] - 1.0_f64.cos()).abs() < 1e-10);
        println!("✓ Cos operation works");
        
        let exp_b = b.exp();
        let exp_data = exp_b.get::<Unitless>();
        assert!((exp_data[0] - 1.0).abs() < 1e-10);
        assert!((exp_data[1] - std::f64::consts::E).abs() < 1e-10);
        println!("✓ Exp operation works");
        
        // Test sqrt
        let c: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([1.0, 4.0]);
        let sqrt_c = c.sqrt();
        let sqrt_data = sqrt_c.get::<Unitless>();
        assert!((sqrt_data[0] - 1.0).abs() < 1e-10);
        assert!((sqrt_data[1] - 2.0).abs() < 1e-10);
        println!("✓ Sqrt operation works");
        
        // Test abs
        let d: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([-3.0, 4.0]);
        let abs_d = d.abs();
        let abs_data = abs_d.get::<Unitless>();
        assert!((abs_data[0] - 3.0).abs() < 1e-10);
        assert!((abs_data[1] - 4.0).abs() < 1e-10);
        println!("✓ Abs operation works");
        
        // Test sum operation
        let e: Tensor<f64, Dimensionless, 1, 3, 1> = Tensor::default([1.0, 2.0, 3.0]);
        e.debug_info();
        println!("Data in tensor e: {:?}", e);
        let sum_e = e.sum();
        sum_e.debug_info();
        let sum_data = sum_e.get::<Unitless>();
        assert_eq!(sum_data, [6.0]);
        println!("✓ Sum operation works");
        
        // Test transpose
        let f: Tensor<f64, Dimensionless, 1, 2, 3> = Tensor::default([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t_f = f.transpose();
        let t_data = t_f.get::<Unitless>();
        // Original: (1,2,3) layout: [1,2,3,4,5,6] -> (1,3,2) layout: [1,4,2,5,3,6]
        assert_eq!(t_data, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        println!("✓ Transpose operation works");
        
        // Test reshape
        let g: Tensor<f64, Dimensionless, 1, 2, 3> = Tensor::default([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r_g = g.reshape::<1, 3, 2>();
        let r_data = r_g.get::<Unitless>();
        assert_eq!(r_data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // Same data, different shape
        println!("✓ Reshape operation works");
        
        // Test slice_layer (for multi-layer tensors)
        let h: Tensor<f64, Dimensionless, 2, 2, 1> = Tensor::default([1.0, 2.0, 3.0, 4.0]);
        let slice_h = h.slice_layer(1);
        let slice_data = slice_h.get::<Unitless>();
        assert_eq!(slice_data, [3.0, 4.0]); // Second layer
        println!("✓ SliceLayer operation works");
        
        // Test get_at
        let i: Tensor<f64, Dimensionless, 1, 2, 2> = Tensor::default([1.0, 2.0, 3.0, 4.0]);
        let at_i = i.get_at(0, 1, 0);
        let at_data = at_i.get::<Unitless>();
        assert_eq!(at_data, [3.0]); // Element at (0,1,0)
        println!("✓ GetAt operation works");
        
        // Test get_col
        let j: Tensor<f64, Dimensionless, 1, 2, 3> = Tensor::default([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col_j = j.get_col(1);
        let col_data = col_j.get::<Unitless>();
        assert_eq!(col_data, [2.0, 5.0]); // Column 1: elements at positions (0,0,1) and (0,1,1)
        println!("✓ GetCol operation works");
        
        // Test get_row
        let k: Tensor<f64, Dimensionless, 1, 2, 3> = Tensor::default([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row_k = k.get_row(0);
        let row_data = row_k.get::<Unitless>();
        assert_eq!(row_data, [1.0, 2.0, 3.0]); // Row 0: first row
        println!("✓ GetRow operation works");
        
        // Test broadcast
        let l: Tensor<f64, Dimensionless, 1, 1, 1> = Tensor::default([5.0]);
        let broadcast_l = l.broadcast_to::<1, 2, 3>();
        let broadcast_data = broadcast_l.get::<Unitless>();
        assert_eq!(broadcast_data, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]); // All elements should be 5.0
        println!("✓ Broadcast operation works");
        
        // Test matrix multiplication
        let m1: Tensor<f64, Dimensionless, 1, 2, 2> = Tensor::default([1.0, 2.0, 3.0, 4.0]);
        let m2: Tensor<f64, Dimensionless, 1, 2, 2> = Tensor::default([5.0, 6.0, 7.0, 8.0]);
        let matmul_result = m1.matmul(&m2);
        let matmul_data = matmul_result.get::<Unitless>();
        // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        assert_eq!(matmul_data, [19.0, 22.0, 43.0, 50.0]);
        println!("✓ MatMul operation works");
        
        // Test outer product
        let v1: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([1.0, 2.0]);
        let v2: Tensor<f64, Dimensionless, 1, 3, 1> = Tensor::default([3.0, 4.0, 5.0]);
        let outer_result = v1.outer_product(&v2);
        let outer_data = outer_result.get::<Unitless>();
        // [1] ⊗ [3,4,5] = [1*3, 1*4, 1*5] =t [3, 4, 5]
        // [2]             [2*3, 2*4, 2*5]   [6, 8, 10]
        assert_eq!(outer_data, [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
        println!("✓ OuterProduct operation works");
        
        // Test compute_norm
        let n: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::default([3.0, 4.0]);
        let norm_n = n.compute_norm();
        let norm_data = norm_n.get::<Unitless>();
        assert!((norm_data[0] - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
        println!("✓ ComputeNorm operation works");
        
        println!("=== ALL COMPREHENSIVE OPERATIONS PASSED! ===\n");
    }
}











