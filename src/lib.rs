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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::si::*;
    use crate::tensor::*;
    use crate::complex::*;

    #[test]
    fn test_stuff() {
        // Tensor of lengths
        let mass = (10.0).scalar::<Kilogram>();
        let force = Vec2::<f64,Force>::new::<Newton>([1.0, 2.0]);

        //let error = mass + force; // error (expected)

        let mass = mass + Scalar::<f64,Mass>::from::<Gram>(5.0); // works
        println!("{}", mass);
        /*
        Tensor [1x1x1]: M^1
        -- Layer 0 --
        ( 10.005 )
        */

        let acc = force.scale(mass.inv()); // works
        println!("{}", acc);
        /*
        Tensor [1x2x1]: L^1 * T^-2
        -- Layer 0 --
        ( 0.09995002498750624 )
        ( 0.19990004997501248 )
        */

        let time = Scalar::<f64,Time>::new::<Second>([1.0]);
        let vel1 = Vec2::<f64,Velocity>::new::<MetersPerSecond>([10.0, 20.0]);

        let vel2 = vel1 + acc.scale(time); // works
        println!("{:?}", vel2.get::<MetersPerSecond>());
        /*
        [10.099950024987507, 20.199900049975014]
        */

        // try to transpose a tensor
        let tensor = Tensor::<c64,Dimensionless, 1,1, 6>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].complex());
        let tensor_transposed = tensor.transpose();
        println!("{}", tensor);
        println!("{}", tensor_transposed);
        /*
        Tensor [1x1x6]: Dimensionless
        -- Layer 0 --
        ( 1  2  3  4  5  6 )

        Tensor [1x6x1]: Dimensionless
        -- Layer 0 --
        ( 1 )
        ( 2 )
        ( 3 )
        ( 4 )
        ( 5 )
        ( 6 )
         */

        let length = Vec2::<f64,Length>::new::<Meter>([10.0, 20.0]);

        // now try and dot product length and force
        let dot_product = dot!(length, force);
        println!("{}", dot_product);
        /*
        Tensor [1x1x1]: L^2 * M^1 * T^-2
        -- Layer 0 --
        ( 50 )
        */

        assert_dimension!(dot_product, Energy); // works
        //assert_dimension!(dot_product, Force); // error (expected)

        let m1 = Matrix::<f64,Length, 2, 3>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Matrix::<f64,Length, 3, 2>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let m3 = m1.matmul(m2);
        println!("{}", m3);
        /*
        Tensor [1x2x2]: L^2
        -- Layer 0 --
        ( 22  28 )
        ( 49  64 )
        */


        // test openrators
        if (vel1 == vel2) {
            println!("Equal");
        } else {
            println!("Not equal");
        }

        let mass2 = Scalar::<f64,Mass>::from::<Kilogram>(10.0);

        println!("The mass is {}g", mass.get::<Gram>()[0]);

        if (mass == mass2) {
            println!("Equal");
        } else {
            if mass < mass2 {
                println!("Less than");
            } else {
                println!("Greater than");
            }
        }
        /*
        Not equal
        Greater than
        */



        let inv = Scalar::<f64,dim_inv!(Time)>::from::<unit_inv!(Second)>(1.0);

        let mul = Scalar::<f64,dim_div!(Energy,Temperature)>::new::<unit_div!(Joule, Kelvin)>([1.0]);

        assert_dimension!(mul, Entropy); // works
        assert_dimension!(inv, Frequency); // works

        println!("{}", inv);
        println!("{}", mul);

        /*
        Tensor [1x1x1]: T^-1
        -- Layer 0 --
        ( 1 )

        Tensor [1x1x1]: L^2 * M^1 * T^-2 * Θ^-1
        -- Layer 0 --
        ( 1 )
        */

        // test dot product
        let a = Vec2::<f64,Length>::new::<Meter>([1.0, 2.0]);
        let b = Vec2::<f64,Length>::new::<Meter>([3.0, 4.0]);
        let c = dot!(a, b);
        println!("{}", c);
        /*
            Tensor [1x1x1]: L^2
            -- Layer 0 --
            ( 11 )
        */
        

        // invert
        let d = c.inv();
        println!("{}", d);

    }

    #[test]
    fn test_simple() {

        let a = cvec!((2,4), (3,5));
        let b = cvec!((1,2), (3,4));

        let c = ip!(a,b);
        println!("{}", c);
    }

    #[test]
    fn test_conjugate_transpose() {
        let a = cvec!((2,4), (3,5));  // [2+4i, 3+5i]
        let a_h = a.conjugate_transpose();

        println!("{}", a);
        println!("{}", a_h);

        /*
        Tensor [1x2x1]: Dimensionless
        -- Layer 0 --
        ( 2 + 4i )
        ( 3 + 5i )

        Tensor [1x1x2]: Dimensionless
        -- Layer 0 --
        ( 2 - 4i  3 - 5i )
         */



        assert_eq!(a_h.get_at(0,0,0).raw(), c64::new(2.0, -4.0));
        assert_eq!(a_h.get_at(0,0,1).raw(), c64::new(3.0, -5.0));
    }


    #[test]
    fn test_inner_product() {
        let a = cvec!((2,4), (3,5));
        let b = cvec!((1,2), (3,4));

        println!("a:\n{}", a);
        println!("b:\n{}", b);
        println!("a†:\n{}", a.conjugate_transpose());
        println!("a† × b:\n{}", a.conjugate_transpose().matmul(b));
        /*
        a:
        Tensor [1x2x1]: Dimensionless
        -- Layer 0 --
        ( 2 + 4i )
        ( 3 + 5i )

        b:
        Tensor [1x2x1]: Dimensionless
        -- Layer 0 --
        ( 1 + 2i )
        ( 3 + 4i )

        a†:
        Tensor [1x1x2]: Dimensionless
        -- Layer 0 --
        ( 2 - 4i  3 - 5i )

        a† × b:
        Tensor [1x1x1]: Dimensionless
        -- Layer 0 --
        ( 39 - 3i )
         */


        let c = ip!(a,b);
        println!("c: {}",c);
        /*
        c: Tensor [1x1x1]: Dimensionless
        -- Layer 0 --
        ( 39 - 3i )
        */

        let c_ = dless!((39.0, -3.0).complex());
        println!("c_: {}",c_);
        /*
        c_: Tensor [1x1x1]: Dimensionless
        -- Layer 0 --
        ( 39 - 3i )
        */

        // The result should be -17 - 7i
        assert_approx_eq!(c, c_);

        // Test conjugate symmetry
        assert_approx_eq!(ip!(a,b).conjugate(), ip!(b,a));

        // Test linearity
        let d = cvec!((1,1), (2,2));
        let alpha = dless!((2.0, 1.0).complex());

        // (a, αb + d) = α(a,b) + (a,d)
        assert_approx_eq!(
            ip!(a,(b.scale(alpha) + d)),
            ip!(a,b).scale(alpha) + ip!(a,d)
        );

        // Test positive definiteness
        assert!(ip!(a,a).raw().re() >= 0.0);
    }

    #[test]
    fn pure_test() {
        let a = cvec!((2,4), (3,5));
        let b = cvec!((1,2), (3,4));

        let c = ip!(a,b);
        println!("{}", c);

        let c_ = dless!((39.0, -3.0).complex());
        println!("{}", c_);

        assert_approx_eq!(c, c_);

        let d = Natural::<1,1,4>::nat([1.0, 2.0, 3.0, 4.0].complex());
        println!("{}", d);

        let e = dless!((1.0, 2.0).complex());


    }

    #[test]
    fn test_init() {
        
        let f = Natural::<1,3,4>::init(|i, j, k| {
            c64::new(i as f64 + j as f64 + k as f64, 0.0)
        });
        println!("{}", f);

        let v = Matrix::<f64,Dimensionless,10,10>::init_2d(|i, j| {
            (i + j) as f64
        });
        println!("{}", v);
    }

    #[test]
    fn test_cut() {
        // Test cutting a simple tensor
        let tensor = Tensor::<f64, Dimensionless, 1, 1, 6>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // Cut 3 elements starting from index 2
        let cut_result: Tensor<f64, Dimensionless, 1, 1, 3> = tensor.cut::<3>(2);
        println!("Original tensor: {}", tensor);
        println!("Cut result (3 elements from index 2): {}", cut_result);
        
        // Verify the cut contains the correct elements [3.0, 4.0, 5.0]
        assert_eq!(cut_result.get::<Unitless>(), [3.0, 4.0, 5.0]);
        
        // Test cutting from beginning
        let cut_beginning: Tensor<f64, Dimensionless, 1, 1, 2> = tensor.cut::<2>(0);
        assert_eq!(cut_beginning.get::<Unitless>(), [1.0, 2.0]);
        
        // Test cutting from end
        let cut_end: Tensor<f64, Dimensionless, 1, 1, 2> = tensor.cut::<2>(4);
        assert_eq!(cut_end.get::<Unitless>(), [5.0, 6.0]);
        
        // Test with 2D tensor
        let matrix = Matrix::<f64, Dimensionless, 2, 3>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let cut_2d: Tensor<f64, Dimensionless, 1, 1, 4> = matrix.cut::<4>(1);
        assert_eq!(cut_2d.get::<Unitless>(), [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_masked_slice() {
        // Create a test tensor
        let tensor = Tensor::<f64, Dimensionless, 1, 2, 3>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // Create a boolean-like mask (using 0.0 and 1.0)
        let mask = Tensor::<f64, Dimensionless, 1, 2, 3>::new::<Unitless>([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        
        // Apply mask - should select elements where mask > 0: [1.0, 3.0, 4.0, 6.0]
        let masked_result: Tensor<f64, Dimensionless, 1, 1, 4> = tensor.masked_slice::<4, f64, Dimensionless>(&mask);
        println!("Original tensor: {}", tensor);
        println!("Mask: {}", mask);
        println!("Masked result: {}", masked_result);
        
        assert_eq!(masked_result.get::<Unitless>(), [1.0, 3.0, 4.0, 6.0]);
        
        // Test with different mask pattern
        let mask2 = Tensor::<f64, Dimensionless, 1, 2, 3>::new::<Unitless>([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        let masked_result2: Tensor<f64, Dimensionless, 1, 1, 2> = tensor.masked_slice::<2, f64, Dimensionless>(&mask2);
        assert_eq!(masked_result2.get::<Unitless>(), [2.0, 5.0]);
        
        // Test with complex numbers
        let complex_tensor = Tensor::<c64, Dimensionless, 1, 1, 4>::new::<Unitless>([
            c64::new(1.0, 1.0),
            c64::new(2.0, 2.0), 
            c64::new(3.0, 3.0),
            c64::new(4.0, 4.0)
        ]);
        let complex_mask = Tensor::<f64, Dimensionless, 1, 1, 4>::new::<Unitless>([1.0, 0.0, 1.0, 0.0]);
        let complex_result: Tensor<c64, Dimensionless, 1, 1, 2> = complex_tensor.masked_slice::<2, f64, Dimensionless>(&complex_mask);
        
        let expected = [c64::new(1.0, 1.0), c64::new(3.0, 3.0)];
        assert_eq!(complex_result.get::<Unitless>(), expected);
    }

    #[test]
    fn test_reduce() {
        // Test reducing a 2D matrix along rows (sum)
        let matrix = Tensor::<f64, Dimensionless, 1, 3, 2>::new::<Unitless>([
            1.0, 2.0,  // row 0
            3.0, 4.0,  // row 1 
            5.0, 6.0   // row 2
        ]);
        
        // Reduce by summing along rows - should give [9.0, 12.0] (column sums)
        let sum_result: Tensor<f64, Dimensionless, 1, 1, 2> = matrix.reduce(|acc, x| acc + x);
        println!("Original matrix: {}", matrix);
        println!("Row-wise sum: {}", sum_result);
        
        assert_eq!(sum_result.get::<Unitless>(), [9.0, 12.0]);
        
        // Test with product reduction
        let product_result: Tensor<f64, Dimensionless, 1, 1, 2> = matrix.reduce(|acc, x| if acc == 0.0 { x } else { acc * x });
        println!("Row-wise product: {}", product_result);
        assert_eq!(product_result.get::<Unitless>(), [15.0, 48.0]); // [1*3*5, 2*4*6]
        
        // Test with multi-layer tensor
        let tensor_3d = Tensor::<f64, Dimensionless, 2, 2, 3>::new::<Unitless>([
            // Layer 0
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
            // Layer 1  
            7.0, 8.0, 9.0,   // row 0
            10.0, 11.0, 12.0 // row 1
        ]);
        
        let sum_3d: Tensor<f64, Dimensionless, 2, 1, 3> = tensor_3d.reduce(|acc, x| acc + x);
        println!("3D tensor: {}", tensor_3d);
        println!("3D row-wise sum: {}", sum_3d);
        
        // Layer 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        // Layer 1: [7+10, 8+11, 9+12] = [17, 19, 21] 
        assert_eq!(sum_3d.get::<Unitless>(), [5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);
        
        // Test with max reduction
        let max_result: Tensor<f64, Dimensionless, 1, 1, 2> = matrix.reduce(|acc, x| if x > acc { x } else { acc });
        assert_eq!(max_result.get::<Unitless>(), [5.0, 6.0]); // max of each column
        
        // Test with complex numbers
        let complex_matrix = Tensor::<c64, Dimensionless, 1, 2, 2>::new::<Unitless>([
            c64::new(1.0, 1.0), c64::new(2.0, 2.0),
            c64::new(3.0, 3.0), c64::new(4.0, 4.0)
        ]);
        
        let complex_sum: Tensor<c64, Dimensionless, 1, 1, 2> = complex_matrix.reduce(|acc, x| acc + x);
        let expected_complex = [c64::new(4.0, 4.0), c64::new(6.0, 6.0)];
        assert_eq!(complex_sum.get::<Unitless>(), expected_complex);
    }

    #[test]
    fn test_cut_masked_slice_reduce_integration() {
        // Test combining multiple operations
        let tensor = Tensor::<f64, Dimensionless, 1, 3, 4>::new::<Unitless>([
            1.0, 2.0, 3.0, 4.0,    // row 0
            5.0, 6.0, 7.0, 8.0,    // row 1
            9.0, 10.0, 11.0, 12.0  // row 2
        ]);
        
        println!("Original tensor: {}", tensor);
        
        // First, reduce to get column sums: [15, 18, 21, 24]
        let reduced: Tensor<f64, Dimensionless, 1, 1, 4> = tensor.reduce(|acc, x| acc + x);
        println!("Reduced (column sums): {}", reduced);
        assert_eq!(reduced.get::<Unitless>(), [15.0, 18.0, 21.0, 24.0]);
        
        // Then cut the middle 2 elements: [18, 21]
        let cut_reduced: Tensor<f64, Dimensionless, 1, 1, 2> = reduced.cut::<2>(1);
        println!("Cut reduced (middle 2): {}", cut_reduced);
        assert_eq!(cut_reduced.get::<Unitless>(), [18.0, 21.0]);
        
        // Create a mask to select only values > 19
        let mask = cut_reduced.apply_with_dimension::<_, f64, Dimensionless>(|x| if x > 19.0 { 1.0 } else { 0.0 });
        println!("Mask (>19): {}", mask);
        
        // Apply mask to get only values > 19: should be [21.0]
        let final_result: Tensor<f64, Dimensionless, 1, 1, 1> = cut_reduced.masked_slice::<1, f64, Dimensionless>(&mask);
        println!("Final masked result: {}", final_result);
        assert_eq!(final_result.get::<Unitless>(), [21.0]);
    }

    #[test]
    fn test_dimensional_consistency() {
        use crate::si::*;
        use crate::tensor::*;
        use crate::dimension::*;
        
        // Test that basic arithmetic preserves dimensions
        let mass1 = Scalar::<f64, Mass>::from::<Kilogram>(10.0);
        let mass2 = Scalar::<f64, Mass>::from::<Kilogram>(5.0);
        let total_mass = mass1 + mass2;
        assert_dimension!(total_mass, Mass);
        
        // Test that multiplication combines dimensions correctly
        let length = Scalar::<f64, Length>::from::<Meter>(5.0);
        let area = length.scale(length);
        assert_dimension!(area, Area);
        
        // Test that division inverts dimensions correctly
        let time = Scalar::<f64, Time>::from::<Second>(2.0);
        let velocity = length.scale(time.inv());
        assert_dimension!(velocity, Velocity);
        
        // Test that force = mass * acceleration
        let acceleration = Scalar::<f64, Acceleration>::from::<MetersPerSecondSquared>(9.8);
        let force = mass1.scale(acceleration);
        assert_dimension!(force, Force);
        
        // Test dot product dimensions
        let force_vec = Vec2::<f64, Force>::new::<Newton>([10.0, 20.0]);
        let length_vec = Vec2::<f64, Length>::new::<Meter>([2.0, 3.0]);
        let work = dot!(force_vec, length_vec);
        assert_dimension!(work, Energy);
        
        // Test that norm has correct dimension (square root of input dimension)
        let area_vec = Vec2::<f64, Area>::new::<SquareMeter>([9.0, 16.0]);
        let length_norm = area_vec.norm();
        assert_dimension!(length_norm, Length);
        
        // Test that comparison operations return dimensionless tensors
        let mass_comparison = mass1.gt(mass2);
        assert_dimension!(mass_comparison, Dimensionless);
        
        // Test that boolean operations only work on dimensionless tensors
        let bool1 = Scalar::<f64, Dimensionless>::from::<Unitless>(1.0);
        let bool2 = Scalar::<f64, Dimensionless>::from::<Unitless>(0.0);
        let and_result = bool1.and(bool2);
        assert_dimension!(and_result, Dimensionless);
        
        // Test matrix multiplication dimensions
        let m1 = Matrix::<f64, Length, 2, 3>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Matrix::<f64, Time, 3, 2>::new::<Second>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result_matrix = m1.matmul(m2);
        
        // Length × Time should give dimensions of Length × Time
        // (This is a constructed example - in physics this might not be meaningful)
        type LengthTime = dim_mul!(Length, Time);
        assert_dimension!(result_matrix, LengthTime);
        
        println!("All dimensional consistency tests passed!");
    }

    #[test]
    fn test_broadcast_basic() {
        // Test broadcasting a scalar (1x1x1) to a vector (1x1x3)
        let scalar = Scalar::<f64, Dimensionless>::from::<Unitless>(5.0);
        let broadcasted = scalar.broadcast_to::<1, 1, 3>();
        
        println!("Scalar: {}", scalar);
        println!("Broadcasted to vector: {}", broadcasted);
        
        assert_eq!(broadcasted.get::<Unitless>(), [5.0, 5.0, 5.0]);
        
        // Test broadcasting a vector (1x1x2) to a matrix (1x3x2) 
        let vector = Tensor::<f64, Dimensionless, 1, 1, 2>::new::<Unitless>([1.0, 2.0]);
        let broadcasted_matrix = vector.broadcast_to::<1, 3, 2>();
        
        println!("Vector: {}", vector);
        println!("Broadcasted to matrix: {}", broadcasted_matrix);
        
        // Expected: [1, 2] broadcasted to 3 rows gives [[1,2], [1,2], [1,2]]
        let expected_matrix = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        assert_eq!(broadcasted_matrix.get::<Unitless>(), expected_matrix);
        
        // Test broadcasting a vector (1x2x1) to a matrix (1x2x3) - column vector to matrix
        let col_vec = Tensor::<f64, Dimensionless, 1, 2, 1>::new::<Unitless>([3.0, 4.0]);
        let broadcasted_col_matrix = col_vec.broadcast_to::<1, 2, 3>();
        
        println!("Column vector: {}", col_vec);
        println!("Broadcasted to matrix: {}", broadcasted_col_matrix);
        
        // Expected: [3; 4] broadcasted to 3 columns gives [[3,3,3], [4,4,4]]
        let expected_col_matrix = [3.0, 3.0, 3.0, 4.0, 4.0, 4.0];
        assert_eq!(broadcasted_col_matrix.get::<Unitless>(), expected_col_matrix);
    }

    #[test]
    fn test_broadcast_3d() {
        // Test broadcasting with 3D tensors
        let tensor_2d = Matrix::<f64, Dimensionless, 2, 2>::new::<Unitless>([1.0, 2.0, 3.0, 4.0]);
        let broadcasted_3d = tensor_2d.broadcast_to::<3, 2, 2>();
        
        println!("2D tensor: {}", tensor_2d);
        println!("Broadcasted to 3D: {}", broadcasted_3d);
        
        // Should repeat the 2x2 matrix in each of the 3 layers
        let expected = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(broadcasted_3d.get::<Unitless>(), expected);
        
        // Test with different layer broadcasting
        let single_layer = Tensor::<f64, Dimensionless, 1, 1, 3>::new::<Unitless>([5.0, 6.0, 7.0]);
        let multi_layer = single_layer.broadcast_to::<4, 1, 3>();
        
        println!("Single layer: {}", single_layer);
        println!("Multi layer: {}", multi_layer);
        
        let expected_multi = [5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0];
        assert_eq!(multi_layer.get::<Unitless>(), expected_multi);
    }

    #[test] 
    #[should_panic(expected = "Target rows must be divisible by source rows")]
    fn test_broadcast_invalid_dimensions() {
        // This should panic because 5 is not divisible by 2 - Fix: Vec2 is actually (1,2,1)
        let vector = Vec2::<f64, Dimensionless>::new::<Unitless>([1.0, 2.0]);
        let _invalid = vector.broadcast_to::<1, 5, 2>();
    }

    #[test]
    fn test_outer_product_basic() {
        // Test basic outer product of two vectors
        let col_vec = Tensor::<f64, Dimensionless, 1, 3, 1>::new::<Unitless>([1.0, 2.0, 3.0]);
        let row_vec = Tensor::<f64, Dimensionless, 1, 1, 2>::new::<Unitless>([4.0, 5.0]);
        
        let outer = Tensor::<f64, Dimensionless, 1, 3, 1>::outer_product::<3, 2, Dimensionless>(&col_vec, &row_vec);
        
        println!("Column vector: {}", col_vec);
        println!("Row vector: {}", row_vec);
        println!("Outer product: {}", outer);
        
        // Expected: [1,2,3]ᵀ ⊗ [4,5] = [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
        //                                 = [[4, 5], [8, 10], [12, 15]]
        let expected = [4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
        assert_eq!(outer.get::<Unitless>(), expected);
        
        // Test with different dimensions
        let col_vec_2 = Tensor::<f64, Dimensionless, 1, 2, 1>::new::<Unitless>([2.0, 3.0]);
        let row_vec_3 = Tensor::<f64, Dimensionless, 1, 1, 3>::new::<Unitless>([1.0, 0.0, -1.0]);
        
        let outer_2 = Tensor::<f64, Dimensionless, 1, 2, 1>::outer_product::<2, 3, Dimensionless>(&col_vec_2, &row_vec_3);
        
        println!("Outer product 2: {}", outer_2);
        
        // Expected: [2,3]ᵀ ⊗ [1,0,-1] = [[2*1, 2*0, 2*(-1)], [3*1, 3*0, 3*(-1)]]
        //                                = [[2, 0, -2], [3, 0, -3]]
        let expected_2 = [2.0, 0.0, -2.0, 3.0, 0.0, -3.0];
        assert_eq!(outer_2.get::<Unitless>(), expected_2);
    }

    #[test]
    fn test_outer_product_with_dimensions() {
        use crate::si::*;
        
        // Test outer product with physical dimensions
        let force_vec = Tensor::<f64, Force, 1, 2, 1>::new::<Newton>([10.0, 20.0]);
        let length_vec = Tensor::<f64, Length, 1, 1, 3>::new::<Meter>([1.0, 2.0, 3.0]);
        
        let energy_matrix = Tensor::<f64, Force, 1, 2, 1>::outer_product::<2, 3, Length>(&force_vec, &length_vec);
        
        println!("Force vector: {}", force_vec);
        println!("Length vector: {}", length_vec);
        println!("Energy matrix: {}", energy_matrix);
        
        // Force × Length = Energy
        assert_dimension!(energy_matrix, Energy);
        
        // Check values: [10,20]ᵀ ⊗ [1,2,3] = [[10,20,30], [20,40,60]]
        let expected = [10.0, 20.0, 30.0, 20.0, 40.0, 60.0];
        assert_eq!(energy_matrix.get::<Joule>(), expected);
        
        // Test with mass and acceleration to get force matrix
        let mass_vec = Tensor::<f64, Mass, 1, 3, 1>::new::<Kilogram>([1.0, 2.0, 3.0]);
        let accel_vec = Tensor::<f64, Acceleration, 1, 1, 2>::new::<MetersPerSecondSquared>([9.8, 1.0]);
        
        let force_matrix = Tensor::<f64, Mass, 1, 3, 1>::outer_product::<3, 2, Acceleration>(&mass_vec, &accel_vec);
        
        assert_dimension!(force_matrix, Force);
        
        // Expected: [1,2,3]ᵀ ⊗ [9.8,1.0] = [[9.8,1.0], [19.6,2.0], [29.4,3.0]]
        let expected_force = [9.8, 1.0, 19.6, 2.0, 29.4, 3.0];
        
        // Use approximate equality for floating point comparison
        let actual = force_matrix.get::<Newton>();
        assert_eq!(actual.len(), expected_force.len());
        for (a, e) in actual.iter().zip(expected_force.iter()) {
            assert!((a - e).abs() < 1e-10, "Expected {}, got {}", e, a);
        }
    }

    #[test]
    fn test_outer_product_3d() {
        // Test outer product with 3D tensors (multiple layers)
        let col_vec = Tensor::<f64, Dimensionless, 2, 2, 1>::new::<Unitless>([1.0, 2.0, 3.0, 4.0]);
        let row_vec = Tensor::<f64, Dimensionless, 2, 1, 2>::new::<Unitless>([5.0, 6.0, 7.0, 8.0]);
        
        let outer_3d = Tensor::<f64, Dimensionless, 2, 2, 1>::outer_product::<2, 2, Dimensionless>(&col_vec, &row_vec);
        
        println!("3D Column vector: {}", col_vec);
        println!("3D Row vector: {}", row_vec);
        println!("3D Outer product: {}", outer_3d);
        
        // Layer 0: [1,2]ᵀ ⊗ [5,6] = [[5,6], [10,12]]
        // Layer 1: [3,4]ᵀ ⊗ [7,8] = [[21,24], [28,32]]
        let expected_3d = [5.0, 6.0, 10.0, 12.0, 21.0, 24.0, 28.0, 32.0];
        assert_eq!(outer_3d.get::<Unitless>(), expected_3d);
    }

    #[test]
    fn test_broadcast_outer_product_integration() {
        // Test that our outer product implementation using broadcast works correctly
        // compared to manual calculation
        
        let col_vec = Tensor::<f64, Dimensionless, 1, 3, 1>::new::<Unitless>([1.0, 2.0, 3.0]);
        let row_vec = Tensor::<f64, Dimensionless, 1, 1, 4>::new::<Unitless>([2.0, 3.0, 4.0, 5.0]);
        
        // Use our outer product implementation
        let outer_result = Tensor::<f64, Dimensionless, 1, 3, 1>::outer_product::<3, 4, Dimensionless>(&col_vec, &row_vec);
        
        // Manual calculation using broadcast and hadamard (which is what outer_product does internally)
        let broadcasted_col = col_vec.broadcast_to::<1, 3, 4>();
        let broadcasted_row = row_vec.broadcast_to::<1, 3, 4>();
        let manual_result = broadcasted_col.hadamard(&broadcasted_row);
        
        println!("Outer product result: {}", outer_result);
        println!("Manual broadcast result: {}", manual_result);
        
        // Both should give the same result
        assert_eq!(outer_result.get::<Unitless>(), manual_result.get::<Unitless>());
        
        // Verify the actual values
        // [1,2,3]ᵀ ⊗ [2,3,4,5] = [[2,3,4,5], [4,6,8,10], [6,9,12,15]]
        let expected = [2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 8.0, 10.0, 6.0, 9.0, 12.0, 15.0];
        assert_eq!(outer_result.get::<Unitless>(), expected);
    }

    #[test]
    fn test_simple_broadcast_and_outer_product() {
        // Simple test that avoids complex type inference issues
        
        // Test broadcast with explicit types
        let scalar = Scalar::<f64, Dimensionless>::from::<Unitless>(3.0);
        let broadcasted_scalar = scalar.broadcast_to::<1, 1, 3>();
        assert_eq!(broadcasted_scalar.get::<Unitless>(), [3.0, 3.0, 3.0]);
        
        // Test simple outer product
        let col = Tensor::<f64, Dimensionless, 1, 2, 1>::new::<Unitless>([1.0, 2.0]);
        let row = Tensor::<f64, Dimensionless, 1, 1, 2>::new::<Unitless>([3.0, 4.0]);
        
        let result = Tensor::<f64, Dimensionless, 1, 2, 1>::outer_product::<2, 2, Dimensionless>(&col, &row);
        
        // [1,2]ᵀ ⊗ [3,4] = [[3,4], [6,8]]
        let expected = [3.0, 4.0, 6.0, 8.0];
        assert_eq!(result.get::<Unitless>(), expected);
        
        println!("Simple broadcast and outer product tests passed!");
    }

    #[test]
    fn test_outer_product_identity_properties() {
        // Test mathematical properties of outer products
        
        // Test with identity-like vectors
        let e1 = Tensor::<f64, Dimensionless, 1, 2, 1>::new::<Unitless>([1.0, 0.0]);
        let e2 = Tensor::<f64, Dimensionless, 1, 1, 2>::new::<Unitless>([0.0, 1.0]);
        
        let outer_e1_e2 = Tensor::<f64, Dimensionless, 1, 2, 1>::outer_product::<2, 2, Dimensionless>(&e1, &e2);
        
        // e1 ⊗ e2 = [[1*0, 1*1], [0*0, 0*1]] = [[0,1], [0,0]]
        let expected_e1_e2 = [0.0, 1.0, 0.0, 0.0];
        assert_eq!(outer_e1_e2.get::<Unitless>(), expected_e1_e2);
        
        // Test scaling property: (αa) ⊗ b = α(a ⊗ b)
        let a = Tensor::<f64, Dimensionless, 1, 2, 1>::new::<Unitless>([2.0, 3.0]);
        let b = Tensor::<f64, Dimensionless, 1, 1, 3>::new::<Unitless>([1.0, 4.0, 2.0]);
        let alpha = 2.0;
        
        let scaled_a = a.apply(|x| x * alpha);
        let outer_scaled = Tensor::<f64, Dimensionless, 1, 2, 1>::outer_product::<2, 3, Dimensionless>(&scaled_a, &b);
        
        let outer_normal = Tensor::<f64, Dimensionless, 1, 2, 1>::outer_product::<2, 3, Dimensionless>(&a, &b);
        let scaled_after = outer_normal.apply(|x| x * alpha);
        
        // (αa) ⊗ b should equal α(a ⊗ b)
        assert_eq!(outer_scaled.get::<Unitless>(), scaled_after.get::<Unitless>());
        
        println!("Outer product mathematical properties verified!");
    }

    #[test]
    fn test_outer_product_matmul() {
        // Test the new outer product sum matrix multiplication
        let m1 = Matrix::<f64, Dimensionless, 2, 3>::new::<Unitless>([
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0   // row 1
        ]);
        let m2 = Matrix::<f64, Dimensionless, 3, 2>::new::<Unitless>([
            7.0, 8.0,   // row 0
            9.0, 10.0,  // row 1
            11.0, 12.0  // row 2
        ]);

        let result = m1.matmul(m2);
        println!("m1:\n{}", m1);
        println!("m2:\n{}", m2);
        println!("m1 * m2:\n{}", result);
        
        // Expected result:
        // [1 2 3] * [7  8 ]   = [1*7+2*9+3*11  1*8+2*10+3*12]   = [58  64]
        // [4 5 6]   [9  10]     [4*7+5*9+6*11  4*8+5*10+6*12]     [139 154]
        //           [11 12]
        
        let expected = [58.0, 64.0, 139.0, 154.0];
        let actual = result.get::<Unitless>();
        
        for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_eq!(actual_val, expected_val, "Mismatch at index {}: expected {}, got {}", i, expected_val, actual_val);
        }
        
        // Test with different dimensions to verify dimension multiplication
        let m3 = Matrix::<f64, Length, 2, 3>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m4 = Matrix::<f64, Length, 3, 2>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m5 = m3.matmul(m4);
        
        assert_dimension!(m5, Area); // Length * Length = Area
        
        println!("Length matrix result:\n{}", m5);
    }

}











