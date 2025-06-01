#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]
#![allow(mixed_script_confusables)]
#![feature(lazy_type_alias)]


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
        // Test basic constant tensor creation
        let tensor: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([1.0, 2.0]);
        let data = tensor.get::<Unitless>();
        assert_eq!(data, [1.0, 2.0]);
    }

    #[test] 
    fn test_tensor_add() {
        // Test basic addition
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([1.0, 2.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([3.0, 4.0]);
        let result = a.add(&b);
        let data = result.get::<Unitless>();
        assert_eq!(data, [4.0, 6.0]);
    }

    #[test]
    fn test_tensor_sub() {
        // Test basic subtraction
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([5.0, 7.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([2.0, 3.0]);
        let result = a.sub(&b);
        let data = result.get::<Unitless>();
        assert_eq!(data, [3.0, 4.0]);
    }

    #[test]
    fn test_tensor_mul() {
        // Test basic element-wise multiplication
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([2.0, 3.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([4.0, 5.0]);
        let result = a.mul(&b);
        println!("Result tensor: {}", result);
        let data = result.get::<Unitless>();
        assert_eq!(data, [8.0, 15.0]);

        // test with some physical units
        let a: Tensor<f64, Length, 1, 2, 1> = Tensor::from_data([2.0, 3.0]);
        let b: Tensor<f64, Force, 1, 2, 1> = Tensor::from_data([4.0, 5.0]);
        let result = &a * &b;
        println!("Result tensor: {}", result);
        let data = result.get::<Joule>();
        assert_eq!(data, [8.0, 15.0]);
    }

    #[test]
    fn test_tensor_div() {
        // Test basic element-wise division
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([8.0, 15.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([2.0, 3.0]);
        let result = a.div(&b);

        println!("Result tensor: {:?}", result);
        let data = result.get::<Unitless>();
        assert_eq!(data, [4.0, 5.0]);
    }

    #[test]
    fn test_dimension_consistency_mul_div() {
        // Length * Force -> Energy (Joule)
        let len: Tensor<f64, Length, 1, 1, 2> = Tensor::from_data([2.0, 3.0]);
        let force: Tensor<f64, Force, 1, 1, 2> = Tensor::from_data([4.0, 5.0]);
        let energy = &len * &force;
        let energy_vals = energy.get::<Joule>();
        assert_eq!(energy_vals, [8.0, 15.0]);

        // Length / Time -> Velocity (m/s)
        let time: Tensor<f64, Time, 1, 1, 2> = Tensor::from_data([1.0, 2.0]);
        let velocity = &len / &time; // dimension-aware Div impl
        let vel_vals = velocity.get::<MetersPerSecond>();
        assert_eq!(vel_vals, [2.0, 1.5]);

        // Force / (Length*Length) -> Pressure (Pascal)
        let width: Tensor<f64, Length, 1, 1, 2> = Tensor::from_data([1.0, 2.0]);
        let area = &len * &width; // Length^2 -> Area
        let pressure = &force / &area;
        let pres_vals = pressure.get::<Pascal>();
        assert_eq!(pres_vals, [2.0, 0.8333333333333334]);
    }

    #[test]
    fn test_graph_visualization() {
        // Test the new graph visualization system
        println!("\n=== TESTING GRAPH VISUALIZATION ===");
        
        // Create a more complex computation to visualize
        let a: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([1.0, 2.0]);
        let b: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([3.0, 4.0]);
        let c: Tensor<f64, Dimensionless, 1, 2, 1> = Tensor::from_data([5.0, 6.0]);

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
}











