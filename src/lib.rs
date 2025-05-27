#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]
#![allow(mixed_script_confusables)]

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
    fn test_linalg() {
        
    }



}











