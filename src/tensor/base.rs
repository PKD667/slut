use crate::tensor::scalar::Scalar;
use crate::units::Unit;
use std::marker::PhantomData;
use std::ops::*;

use crate::complex::c64;
use crate::tensor::element::*;

use crate::dimension::Dimension;

#[derive(Copy, Clone)]
pub struct Tensor<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub data: [E; LAYERS * ROWS * COLS],
    pub _phantom: PhantomData<D>,
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn new<U: Unit<Dimension = D>>(values: [E; LAYERS * ROWS * COLS]) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = values
            .iter()
            .map(|&v| E::from(U::to_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn zero() -> Self {
        let data: [E; LAYERS * ROWS * COLS] = [E::zero(); LAYERS * ROWS * COLS];

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn random<U: Unit<Dimension = D>>(min: E, max: E) -> Self {
        let base_min: E = E::from(U::to_base(min.into()));
        let base_max: E = E::from(U::to_base(max.into()));
        let data: [E; LAYERS * ROWS * COLS] = (0..LAYERS * ROWS * COLS)
            .map(|_| {
                E::from(U::from_base(((base_max - base_min) + base_min).weak_mul(rand::random::<f64>()).into()))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn get<S: Unit<Dimension = D>>(&self) -> [E; LAYERS * ROWS * COLS] {
        self.data
            .iter()
            .map(|&v| E::from(S::from_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn get_at(&self, layer: usize, row: usize, col: usize) -> Scalar<E,D> {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        Scalar::<E,D> {
            data: [self.data[idx]],
            _phantom: PhantomData,
        }
    }

    pub fn set_at(&mut self, layer: usize, row: usize, col: usize, value: Scalar<E,D>) {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        self.data[idx] = value.data[0];
    }
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn dtype(&self) -> &'static str {
        std::any::type_name::<E>()
    }

    pub fn cast<T: TensorElement>(&self) -> Tensor<T, D, LAYERS, ROWS, COLS>
    where
        T: TensorElement,
    {
        let data: [T; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| (T::from(v.into() as c64)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

// vector of N elements
pub type Vector<E: TensorElement, D, const N: usize> = Tensor<E, D, 1, N, 1>;

// matrix of N x M elements
pub type Matrix<E: TensorElement,D, const N: usize, const M: usize> = Tensor<E,D, 1, N, M>;

// ----------- SPECIFIC TYPES OF TENSORS ------------

//type alias for a 2D vector
pub type Vec2<E: TensorElement,D> = Vector<E,D, 2>;
//type alias for a 3D vector
pub type Vec3<E: TensorElement,D> = Vector<E,D, 3>;
//type alias for a 4D vector
pub type Vec4<E: TensorElement,D> = Vector<E,D, 4>;

//type alias for a 2x2 matrix
pub type Mat2<E: TensorElement,D> = Matrix<E,D, 2, 2>;
//type alias for a 3x3 matrix
pub type Mat3<E: TensorElement,D> = Matrix<E,D, 3, 3>;
//type alias for a 4x4 matrix
pub type Mat4<E: TensorElement,D> = Matrix<E,D, 4, 4>;

impl<E: TensorElement,D> Vec2<E,D> {
    // Returns a tuple of elements of type E (generic).
    pub fn raw_tuple(&self) -> (E, E)
    where
        E: TensorElement,
    {
        (self.data[0], self.data[1])
    }

    // Generic conversion for a Vec2 into a tuple of type T.
    pub fn raw_tuple_as<T: From<E>>(&self) -> (T, T)
    where
        E: TensorElement,
    {
        (T::from(self.data[0]), T::from(self.data[1]))
    }
}

impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    E: TensorElement,
{
    // Returns a vector of elements of type E
    pub fn raw_vec(&self) -> Vec<E> {
        self.data.to_vec()
    }

    // Generic conversion for any Tensor into a Vec<T>.
    pub fn raw_vec_as<T: From<E>>(&self) -> Vec<T> {
        self.data.iter().map(|&x| T::from(x)).collect()
    }
}

// implement x() and y() for Vec2
impl<E: TensorElement,D> Vec2<E,D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }
}

// implement x(), y() and z() for Vec3
impl<E: TensorElement,D> Vec3<E,D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }

    pub fn z(&self) -> Scalar<E,D> {
        Scalar::<E,D> {
            data: [self.data[2]],
            _phantom: PhantomData,
        }
    }
}

// implement += and -= for all tensors
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> AddAssign for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    E: TensorElement + AddAssign,
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

impl<E: TensorElement,D: std::fmt::Display + Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> std::fmt::Display
    for Tensor<E,D, LAYERS, ROWS, COLS>
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

impl<E: TensorElement,D: std::fmt::Debug + Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> std::fmt::Debug
    for Tensor<E,D, LAYERS, ROWS, COLS>
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

