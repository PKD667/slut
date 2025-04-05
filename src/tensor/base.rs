use crate::complex::*;
use crate::tensor::scalar::Scalar;
use crate::units::Unit;
use std::marker::PhantomData;
use std::ops::*;

use crate::complex::c64;


use crate::dimension::Dimension;

#[derive(Copy, Clone)]
pub struct Tensor<D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub data: [c64; LAYERS * ROWS * COLS],
    pub _phantom: PhantomData<D>,
}

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn new<U: Unit<Dimension = D>>(values: [c64; LAYERS * ROWS * COLS]) -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = values
            .iter()
            .map(|&v| U::to_base(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn zero() -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = [c64::zero(); LAYERS * ROWS * COLS];

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn random<U: Unit<Dimension = D>>(min: c64, max: c64) -> Self {
        let data: [c64; LAYERS * ROWS * COLS] = (0..LAYERS * ROWS * COLS)
            .map(|_| c64::new(rand::random::<f64>() * (max.re() - min.re()) + min.re(), 0.0))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn get<S: Unit<Dimension = D>>(&self) -> [c64; LAYERS * ROWS * COLS] {
        self.data
            .iter()
            .map(|&v| S::from_base(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn get_at(&self, layer: usize, row: usize, col: usize) -> Scalar<D> {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        Scalar::<D> {
            data: [self.data[idx]],
            _phantom: PhantomData,
        }
    }

    pub fn set_at(&mut self, layer: usize, row: usize, col: usize, value: Scalar<D>) {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        self.data[idx] = value.data[0];
    }
}



// vector of N elements
pub type Vector<D, const N: usize> = Tensor<D, 1, N, 1>;

// matrix of N x M elements
pub type Matrix<D, const N: usize, const M: usize> = Tensor<D, 1, N, M>;

// ----------- SPECIFIC TYPES OF TENSORS ------------

//type alias for a 2D vector
pub type Vec2<D> = Vector<D, 2>;
//type alias for a 3D vector
pub type Vec3<D> = Vector<D, 3>;
//type alias for a 4D vector
pub type Vec4<D> = Vector<D, 4>;

//type alias for a 2x2 matrix
pub type Mat2<D> = Matrix<D, 2, 2>;
//type alias for a 3x3 matrix
pub type Mat3<D> = Matrix<D, 3, 3>;
//type alias for a 4x4 matrix
pub type Mat4<D> = Matrix<D, 4, 4>;



impl<D> Vec2<D> {
    // Returns a tuple of c64 (default behavior)
    pub fn raw_tuple(&self) -> (c64, c64) {
        (self.data[0], self.data[1])
    }

    // Generic conversion for a Vec2 into a tuple of type T.
    pub fn raw_tuple_as<T: From<c64>>(&self) -> (T, T) {
        (T::from(self.data[0]), T::from(self.data[1]))
    }
}

impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    // Returns a vector of c64 elements (default behavior)
    pub fn raw_vec(&self) -> Vec<c64> {
        self.data.to_vec()
    }

    // Generic conversion for any Tensor into a Vec<T>.
    pub fn raw_vec_as<T: From<c64>>(&self) -> Vec<T> {
        self.data.iter().map(|&x| T::from(x)).collect()
    }
}

// implement x() and y() for Vec2
impl<D> Vec2<D> {
    pub fn x(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }
}

// implement x(), y() and z() for Vec3
impl<D> Vec3<D> {
    pub fn x(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }

    pub fn z(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[2]],
            _phantom: PhantomData,
        }
    }
}


// implement += and -= for all tensors
impl<D, const LAYERS: usize, const ROWS: usize, const COLS: usize> AddAssign for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn add_assign(&mut self, other: Self) {
        for i in 0..LAYERS {
            for j in 0..ROWS {
                for k in 0..COLS {
                    let idx = i * (ROWS * COLS) + j * COLS + k;
                    self.data[idx] += other.data[idx];
                }
            }
        }
    }
}

/// ----- NICE PRINTING -----

impl<const L: i32, const M: i32, const T: i32, const Θ: i32, const I: i32, const N: i32, const J: i32>
    std::fmt::Display for Dimension<L, M, T, Θ, I, N, J>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Format nonzero exponents only
        let mut parts = Vec::new();
        if L != 0 {
            parts.push(format!("L^{}", L));
        }
        if M != 0 {
            parts.push(format!("M^{}", M));
        }
        if T != 0 {
            parts.push(format!("T^{}", T));
        }
        if Θ != 0 {
            parts.push(format!("Θ^{}", Θ));
        }
        if I != 0 {
            parts.push(format!("I^{}", I));
        }
        if N != 0 {
            parts.push(format!("N^{}", N));
        }
        if J != 0 {
            parts.push(format!("J^{}", J));
        }
        if parts.is_empty() {
            write!(f, "Dimensionless")
        } else {
            write!(f, "{}", parts.join(" * "))
        }
    }
}

impl<D: std::fmt::Display + Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> std::fmt::Display
    for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tensor [{}x{}x{}]: {}", LAYERS, ROWS, COLS, D::default())?;
        // For each layer, print the matrix.
        for l in 0..LAYERS {
            writeln!(f, "-- Layer {} --", l)?;
            for i in 0..ROWS {
                write!(f, "(")?;
                for j in 0..COLS {
                    let idx = l * (ROWS * COLS) + i * COLS + j;
                    write!(f, " {} ", self.data[idx])?;
                }
                writeln!(f, ")")?;
            }
        }
        Ok(())
    }
}

impl<D: std::fmt::Debug + Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> std::fmt::Debug
    for Tensor<D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dimension", &D::default())
            .field("shape", &format!("{}x{}x{}", LAYERS, ROWS, COLS))
            .field("data", &self.data)
            .finish()
    }
}

