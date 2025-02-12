#![feature(generic_const_exprs,trait_alias)]

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
        let vel = Vec2::<Velocity>::new::<MetersPerSecond>([3.0, 4.0]);

        let dt = Scalar::<Time>::new::<Second>([10.0]);

        let dist = vel.scale(dt);

        print!("{:?}", dist.get::<Kilometer>());

        let acc = vel.scale(dt.inv());

        print!("{:?}", acc.get::<MeterPerSecondSquared>());
    }
}

