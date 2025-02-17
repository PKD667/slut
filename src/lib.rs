#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]

pub mod tensor;
use tensor::*;

pub mod dimension;
use dimension::*;

pub mod units;
use units::*;

pub mod si;
use si::*;

pub mod complex;
use complex::*;


#[cfg(test)]
mod tests {


    use super::*;

    #[test]
    fn test_stuff() {
        // Tensor of lengths
        let mass = (10.0).scalar::<Kilogram>();
        let force = Vec2::<Force>::new::<Newton>([1.0, 2.0].complex());

        //let error = mass + force; // error (expected)

        let mass = mass + Scalar::<Mass>::from_f64::<Gram>(5.0); // works
        println!("{}", mass);
        /*
        Tensor [1x1]: M^1
        ( 10.005 )
        */

        let acc = force.scale(mass.inv()); // works
        println!("{}", acc);
        /*
        Tensor [2x1]: L^1 * T^-2
        ( 0.9995002 )
        ( 1.9990004 )
        */

        let time = Scalar::<Time>::from_f64::<Second>(1.0);
        let vel1 = Vec2::<Velocity>::new::<MetersPerSecond>([10.0, 20.0].complex());

        let vel2 = vel1 + acc.scale(time); // works
        println!("{:?}", vel2.get::<MetersPerSecond>());
        /*
        [c64 { a: 10.099950024987507, b: 0.0 }, c64 { a: 20.199900049975014, b: 0.0 }]
        */

        // try to transpose a tensor
        let tensor = Tensor::<Dimensionless, 1, 6>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].complex());
        let tensor_transposed = tensor.transpose();
        println!("{}", tensor);
        println!("{}", tensor_transposed);
        /*
        Tensor [1x6]: Dimensionless
        ( 1  2  3  4  5  6 )

        Tensor [6x1]: Dimensionless
        ( 1 )
        ( 2 )
        ( 3 )
        ( 4 )
        ( 5 )
        ( 6 )
         */

        let length = Vec2::<Length>::new::<Meter>([10.0, 20.0].complex());

        // now try and dot product length and force
        let dot_product = dot!(length, force);
        println!("{}", dot_product);
        /*
        Tensor [1x1]: L^2 * M^1 * T^-2
        ( 50 )
        */

        assert_dimension!(dot_product, Energy); // works
        //assert_dimension!(dot_product, Force); // error (expected)

        let m1 = Matrix::<Length, 2, 3>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].complex());
        let m2 = Matrix::<Length, 3, 2>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].complex());

        let m3 = m1 * m2;
        println!("{}", m3);
        /*
        Tensor [2x2]: L^2
        ( 22  28 )
        ( 49  64 )
        */


        // test openrators
        if (vel1 == vel2) {
            println!("Equal");
        } else {
            println!("Not equal");
        }

        let mass2 = Scalar::<Mass>::from_f64::<Kilogram>(10.0);

        if (mass == mass2) {
            println!("Equal");
        } else {
            if mass < mass2 {
                println!("Less than");
            } else {
                println!("Greater than");
            }
        }


        // test macros

        let inv = Scalar::<dim_inv!(Time)>::from_f64::<unit_inv!(Second)>(1.0);

        let mul = Scalar::<dim_div!(Energy,Temperature)>::new::<unit_div!(Joule, Kelvin)>([1.0].complex());

        assert_dimension!(mul, Entropy); // works
        assert_dimension!(inv, Frequency); // works

        println!("{}", inv);
        println!("{}", mul);

        /*
        Tensor [1x1]: T^-1
        ( 1 )

        Tensor [1x1]: L^2 * M^1 * T^-2 * Θ^-1
        ( 1 )
        */

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


        // Should be [2-4i, 3-5i]
        assert_eq!(a_h.get_at::<Unitless>(0,1), c64::new(2.0, -4.0));
        assert_eq!(a_h.get_at::<Unitless>(0,1), c64::new(3.0, -5.0));
    }


    #[test]
    fn test_inner_product() {
        let a = cvec!((2,4), (3,5));
        let b = cvec!((1,2), (3,4));

        println!("a:\n{}", a);
        println!("b:\n{}", b);
        println!("a†:\n{}", a.conjugate_transpose());
        println!("a† × b:\n{}", a.conjugate_transpose() * b);


        let c = ip!(a,b);
        println!("c: {}",c);

        let c_ = dot!(a,b);
        println!("c_: {}",c_);

        // The result should be -17 - 7i
        assert_approx_eq!(c, (39.0, -3.0).complex().dless());

        // Test conjugate symmetry
        assert_approx_eq!(ip!(a,b).conjugate(), ip!(b,a));

        // Test linearity
        let d = cvec!((1,1), (2,2));
        let alpha = (2.0, 1.0).complex().dless();

        // (a, αb + d) = α(a,b) + (a,d)
        assert_approx_eq!(
            ip!(a,(b.scale(alpha) + d)),
            ip!(a,b).scale(alpha) + ip!(a,d)
        );

        // Test positive definiteness
        assert!(ip!(a,a).raw().re() >= 0.0);
    }

}











