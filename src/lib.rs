#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]

pub mod tensor;
use tensor::*;

pub mod dimension;
use dimension::*;

pub mod units;
use units::*;

pub mod utils;  
use utils::*;


#[macro_export]
macro_rules! assert_dimension {
    ($value:expr, $expected:ty) => {{
        // Force type inference by attempting an assignment with annotation.
        let _: Tensor<$expected, _, _> = $value;
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main() {
        // Tensor of lengths
        let mass = Scalar::<Mass>::new::<Kilogram>([10.0]);
        let force = Vec2::<Force>::new::<Newton>([10.0, 20.0]);

        let mut vel1 = Vec2::<Velocity>::new::<MetersPerSecond>([10.0, 20.0]);

        //let acc = div!(force, mass); // works
        let acc = force.scale(mass.inv()); // works

        let time = Scalar::<Time>::from::<Second>(1.0);

        let vel2 = vel1 + acc.scale(time); // works

        println!("{:?}", vel2.get::<MetersPerSecond>());

        // try to transpose a tensor
        let tensor = Tensor::<Dimensionless, 1, 6>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor_transposed = tensor.transpose();
        println!("{}", tensor);
        println!("{}", tensor_transposed);

        let length = Vec2::<Length>::new::<Meter>([1.0, 2.0]);

        // now try and dot product length and force
        let dot_product = dot!(length, force);

        println!("{}", dot_product);

        assert_dimension!(dot_product, Energy);

        // test openrators
        if (vel1 == vel2) {
            println!("Equal");
        } else {
            println!("Not equal");
        }

        let mass2 = Scalar::<Mass>::new::<Kilogram>([15.0]);

        if (mass == mass2) {
            println!("Equal");
        } else {
            if mass < mass2 {
                println!("Less than");
            } else {
                println!("Greater than");
            }
        }

    }
}











