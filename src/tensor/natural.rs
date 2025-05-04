use crate::tensor::*;
use crate::complex::*;
use crate::dimension::Dimensionless;
use std::marker::PhantomData;

pub type Natural<const L: usize, const R: usize, const C: usize> =
    Tensor<c64,Dimensionless, L, R, C>;

impl<const L: usize, const R: usize, const C: usize> Tensor<c64,Dimensionless, L, R, C>
where
    [(); L * R * C]:,
{
    pub fn nat(values: [c64; L * R * C]) -> Self {
        Tensor::default(values)
    }

    pub fn randnat(min: c64, max: c64) -> Self {
        let data: [c64; L * R * C] = (0..L * R * C)
            .map(|_| c64::new(rand::random::<f64>() * (max.re() - min.re()) + min.re(), 0.0))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor::default(data)
    }
}

// implement item() for pure scalar
impl Tensor<c64,Dimensionless, 1, 1, 1> {
    pub fn item(&self) -> c64 {
        self.raw()
    }
}