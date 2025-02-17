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

pub mod utils;  

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
}











