
use crate::units::*;
use crate::tensor::*;
use crate::dimension::*;
use crate::*;

impl<const N: usize> Vector<Force, N> where [(); N * 1]: {
    pub fn apply(&self, mass: Scalar<Mass>, dt: Scalar<Time>) -> Vector<Velocity, N> {
        self.scale(mass.inv()).scale(dt)
    }
}
