#![feature(generic_const_exprs, trait_alias)]

pub mod tensor;
use tensor::*;

pub mod dimension;
use dimension::*;

pub mod units;
use units::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main() {
        // Tensor of lengths
        let mass = Scalar::<Mass>::new::<Kilogram>([10.0]);
        let force = Vec2::<Force>::new::<Newton>([10.0, 20.0]);

        let mut vel = Vec2::<Velocity>::new::<MetersPerSecond>([10.0, 20.0]);

        //let acc = div!(force, mass); // works
        let acc = force.scale(mass.inv()); // works

        let time = Scalar::<Time>::new::<Second>([1.0]);

        vel = vel + acc.scale(time); // works

        println!("{:?}", vel.get::<MetersPerSecond>());

        // try to transpose a tensor
        let tensor = Tensor::<Dimensionless, 1, 6>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor_transposed = tensor.transpose();
        println!("{}", tensor);
        println!("{}", tensor_transposed);
        
        let length = Vec2::<Length>::new::<Meter>([1.0, 2.0]);

        // now try and dot product length and force
        let dot_product = dot!(length, force);

        println!("{}", dot_product);
    }
}
