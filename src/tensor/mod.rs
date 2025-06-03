pub mod base;
pub mod element;
pub mod scalar;
pub mod operations;
pub mod specialized;
pub mod natural;
pub mod fmt;
pub mod graph;
pub mod execution;
pub mod ops;

pub use base::*;
pub use element::*;
pub use operations::*;
pub use scalar::*;
pub use specialized::*;
pub use natural::*;
pub use fmt::*;
pub use graph::*;
pub use ops::*;


#[macro_export]
macro_rules! dless {
    ($value:expr) => {{
        use crate::tensor::Scalar;
        use crate::dimension::Dimensionless;
        use crate::units::Unitless;
        Scalar::<_, Dimensionless>::from::<Unitless>($value)
    }};
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let a_val = $a.raw();
        let b_val = $b.raw();
        let diff = if a_val > b_val { a_val - b_val } else { b_val - a_val };
        assert!(diff < crate::complex::c64::from(1e-10), "Values not approximately equal: {:?} vs {:?}", a_val, b_val);
    }};
}

// Type aliases for convenience
pub type Matrix<E, D, const ROWS: usize, const COLS: usize> = Tensor<E, D, 1, ROWS, COLS>
where
    E: crate::tensor::element::TensorElement,
    D: Clone,
    [(); 1 * ROWS * COLS]:;

pub type Vec2<E, D> = Tensor<E, D, 1, 2, 1>
where
    E: crate::tensor::element::TensorElement,
    D: Clone,
    [(); 1 * 2 * 1]:;
