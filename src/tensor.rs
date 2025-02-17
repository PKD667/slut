use crate::complex::Complexify;
use crate::units::Unit;
use std::marker::PhantomData;
use std::ops::*;

use crate::complex::c64;


use crate::dimension::Dimension;
use crate::dimension::InvertDimension;
use crate::dimension::MultiplyDimensions;


#[derive(Copy, Clone)]
pub struct Tensor<D, const ROWS: usize, const COLS: usize>
where
    [(); ROWS * COLS]:,
{
    pub data: [c64; ROWS * COLS],
    pub _phantom: PhantomData<D>,
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    pub fn new<U: Unit<Dimension = D>>(values: [c64; ROWS * COLS]) -> Self {


        let data: [c64; ROWS * COLS] = values
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

    pub fn get_at<U: Unit<Dimension = D>>(&self, row: usize, col: usize) -> c64 {
        assert!(row < ROWS && col < COLS);
        let idx = row * COLS + col;
        U::from_base(self.data[idx])
    }

    pub fn get<S: Unit<Dimension = D>>(&self) -> [c64; ROWS * COLS] {
        self.data
            .iter()
            .map(|&v| S::from_base(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn set_at(&mut self, row: usize, col: usize, value: Scalar<D>) {
        assert!(row < ROWS && col < COLS);
        let idx = row * COLS + col;
        self.data[idx] = value.data[0];
    }



}

// implement common types of tensors
pub type Scalar<D> = Tensor<D, 1, 1>;

// vector of N elements
pub type Vector<D, const N: usize> = Tensor<D, N, 1>;

// matrix of N x M elements
pub type Matrix<D, const N: usize, const M: usize> = Tensor<D, N, M>;

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


// zero an N*M tensor
impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    pub fn zero() -> Self {
        let data: [c64; ROWS * COLS] = [c64::zero(); ROWS * COLS];

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

// epsilon scalar
impl<D> Scalar<D> {
    pub const EPSILON: Scalar<D> = Scalar {
        data: [c64 { a: f64::EPSILON, b: 0.0 }],
        _phantom: PhantomData,
    };
}

use crate::dimension::Dimensionless;

pub trait ToScalar {
    fn dless(&self) -> Scalar<Dimensionless>;
    
    // convert by specifying the Unit
    fn scalar<U: Unit>(&self) -> Scalar<U::Dimension>;
}

impl ToScalar  for c64 {
    fn dless(&self) -> Scalar<crate::dimension::Dimensionless> {
        Scalar {
            data: [*self],
            _phantom: PhantomData,
        }
    }

    fn scalar<U: Unit>(&self) -> Scalar<U::Dimension> {
        Scalar {
            data: [*self],
            _phantom: PhantomData,
        }
    }
}

impl ToScalar for f64 {
    fn dless(&self) -> Scalar<crate::dimension::Dimensionless> {
        Scalar {
            data: [self.complex()],
            _phantom: PhantomData,
        }
    }

    fn scalar<U: Unit>(&self) -> Scalar<U::Dimension> {
        Scalar {
            data: [U::to_base(self.complex())],
            _phantom: PhantomData,
        }
    }
}

// -----------------------------------------
// ============= OPERATIONS ================
// -----------------------------------------

impl<D, const ROWS: usize, const COLS: usize> Add for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let data: [c64; ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Sub for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let data: [c64; ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<
        const L1: i32,
        const M1: i32,
        const T1: i32,
        const Θ1: i32,
        const I1: i32,
        const N1: i32,
        const J1: i32,
        const L2: i32,
        const M2: i32,
        const T2: i32,
        const Θ2: i32,
        const I2: i32,
        const N2: i32,
        const J2: i32,
        const ROWS: usize,
        const COMMON: usize,
        const COLS: usize,
    > Mul<Tensor<Dimension<L2, M2, T2, Θ2, I2, N2, J2>, COMMON, COLS>>
    for Tensor<Dimension<L1, M1, T1, Θ1, I1, N1, J1>, ROWS, COMMON>
where
    [(); ROWS * COMMON]:,
    [(); COMMON * COLS]:,
    [(); ROWS * COLS]:,
    // Use the helper trait to combine dimensions.
    Dimension<L1, M1, T1, Θ1, I1, N1, J1>:
        MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
{
    type Output = Tensor<
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<
            Dimension<L2, M2, T2, Θ2, I2, N2, J2>,
        >>::Output,
        ROWS,
        COLS,
    >;

    fn mul(
        self,
        other: Tensor<Dimension<L2, M2, T2, Θ2, I2, N2, J2>, COMMON, COLS>,
    ) -> Self::Output {
        let mut result = vec![c64::zero(); ROWS * COLS];

        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum: c64 = c64::zero();
                for k in 0..COMMON {
                    sum += self.data[i * COMMON + k] * other.data[k * COLS + j];
                }
                result[i * COLS + j] = sum;
            }
        }

        let data: [c64; ROWS * COLS] = result
            .into_iter()
            .collect::<Vec<c64>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    /// Multiplies every element of the tensor by a scalar and auto-normalizes its dimension.
    /// The result's dimension is the product of the original tensor’s dimension and
    /// the scalar’s dimension, normalized automatically.
    pub fn scale<DS>(
        self,
        scalar: Tensor<DS, 1, 1>,
    ) -> Tensor<<D as MultiplyDimensions<DS>>::Output, ROWS, COLS>
    where
        D: MultiplyDimensions<DS>,
        <D as MultiplyDimensions<DS>>::Output:,
        [(); ROWS * COLS]:,
    {
        let s = scalar.data[0];
        let data: [c64; ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v * s)
            .collect::<Vec<c64>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<<D as MultiplyDimensions<DS>>::Output>,
        }
    }
}

//impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
//where
//    [(); ROWS * COLS]:,
//{
//    pub fn div<DS>(
//        self,
//        scalar: Tensor<DS, 1, 1>,
//    ) -> Tensor<
//        <<D as MultiplyDimensions<<DS as InvertDimension>::Output>>::Output as AutoNormalize>::Normalized,
//        ROWS,
//        COLS
//    >
//    where
//        DS: InvertDimension,
//        D: MultiplyDimensions<<DS as InvertDimension>::Output>,
//        // Add the missing bound here:
//        <D as MultiplyDimensions<<DS as InvertDimension>::Output>>::Output: AutoNormalize,
//        [(); ROWS * COLS]:,
//    {
//        self.scale(scalar.inv())
//    }
//}

// invert a scalar
impl<D> Tensor<D, 1, 1>
where
    [(); 1]:,
{
    pub fn inv(self) -> Tensor<<D as InvertDimension>::Output, 1, 1>
    where
        D: InvertDimension,
    {
        let data: [c64; 1] = [1.0 / self.data[0]];

        Tensor {
            data,
            _phantom: PhantomData::<D::Output>,
        }
    }
}

// implement negation for all tensors
impl<D, const ROWS: usize, const COLS: usize> Neg for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    type Output = Self;

    fn neg(self) -> Self {
        let data: [c64; ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| -v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    /// Returns the norm of the tensor as a 1×1 tensor.
    /// The norm is defined as sqrt(sum(vᵢ²)).
    pub fn norm(&self) -> Tensor<D, 1, 1> {
        let sum: c64 = self.data.iter().map(|&v| v * v).sum();

        Tensor::<D, 1, 1> {
            data: [sum.sqrt()],
            _phantom: PhantomData,
        }
    }
}

// implement conjugate for all tensors
impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    pub fn conjugate(self) -> Self {
        let data: [c64; ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v.conjugate())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    /// Returns the transpose of this tensor.
    pub fn transpose(self) -> Tensor<D, COLS, ROWS>
    where
        [(); COLS * ROWS]:,
    {
        let mut transposed = [c64::zero(); COLS * ROWS];
        for i in 0..ROWS {
            for j in 0..COLS {
                // Element at (i, j) moves to (j, i)
                transposed[j * ROWS + i] = self.data[i * COLS + j];
            }
        }
        Tensor::<D, COLS, ROWS> {
            data: transposed,
            _phantom: PhantomData,
        }
    }

    /// Returns the conjugate transpose of this tensor.
    pub fn conjugate_transpose(self) -> Tensor<D, COLS, ROWS>
    where
        [(); COLS * ROWS]:,
    { 
        self.transpose().conjugate()
    }
}

impl<D, const ROWS: usize> Tensor<D, ROWS, 1>
where
    [(); ROWS * 1]:,
{
    pub fn dist(self, other: Self) -> Tensor<D, 1, 1> {
        // norm of sub
        let sub = self - other;
        sub.norm()
    }
}

// Implement elementwise equality for all tensors.
impl<D, const ROWS: usize, const COLS: usize> PartialEq for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    fn eq(&self, other: &Self) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(&a, &b)| a == b)
    }
}

// Optionally, if c64: Eq then implement Eq.
impl<D, const ROWS: usize, const COLS: usize> Eq for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
    c64: Eq,
{
}

// Implement ordering (>, >=, <, <=) for 1×1 tensors only.
impl<D> PartialOrd for Tensor<D, 1, 1>
where
    [(); 1]:,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data[0].partial_cmp(&other.data[0])
    }
}

// implement dot product as macro that does transpose and multiply
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        let a_t = a.transpose();
        let result = a_t * b;
        result
    }};
}

#[macro_export]
macro_rules! inner_product {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        let a_t = a.conjugate_transpose();
        let result = a_t * b;
        result
    }};
}

#[macro_export]
macro_rules! ip {
    ($x:expr, $y:expr) => {
        inner_product!($x, $y)
    }
}

// -----------------------------------------
// ============= INIT AND PRINT ============
// -----------------------------------------

// implement converting a c64 to a Scalar tensor
impl<D> Scalar<D> {

    pub fn from_c64<U: Unit<Dimension = D>>(value: c64) -> Self {
        Scalar { data: [U::to_base(value)], _phantom: PhantomData }
    }
}

// implement converting a float to a Scalar tensor
impl<D> Scalar<D> {

    pub fn from_f64<U: Unit<Dimension = D>>(value: f64) -> Self {
        Scalar { data: [U::to_base(value.complex())], _phantom: PhantomData }
    }
}



/// A trait for converting from c64 to another type.
pub trait Fromc64: Sized {
    fn from_c64(value: c64) -> Self;
}

// Implement for c64 itself (a no‑op conversion)
impl Fromc64 for c64 {
    fn from_c64(value: c64) -> Self {
        value
    }
}

// For example, convert to i32 by rounding.
impl Fromc64 for i32 {
    fn from_c64(value: c64) -> Self {

        if value.im() != 0.0 {
            panic!("Cannot convert complex number to integer");
        }

        value.re().round() as i32
    }
}

// You can add similar implementations for other target types...
impl Fromc64 for i64 {
    fn from_c64(value: c64) -> Self {
        if value.im() != 0.0 {
            panic!("Cannot convert complex number to integer");
        }

        value.re().round() as i64
    }
}

impl<D> Scalar<D> {
    // Default raw returns c64.
    pub fn raw(&self) -> c64 {
        self.data[0]
    }
    
    // Generic raw conversion into any type that implements Fromc64.
    pub fn raw_as<T: Fromc64>(&self) -> T {
        T::from_c64(self.data[0])
    }
}

impl<D> Vec2<D> {
    // Returns a tuple of c64 (default behavior)
    pub fn raw_tuple(&self) -> (c64, c64) {
        (self.data[0], self.data[1])
    }

    // Generic conversion for a Vec2 into a tuple of type T.
    pub fn raw_tuple_as<T: Fromc64>(&self) -> (T, T) {
        (
            T::from_c64(self.data[0]),
            T::from_c64(self.data[1]),
        )
    }
}

impl<D, const ROWS: usize, const COLS: usize> Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    // Returns a vector of c64 elements (default behavior)
    pub fn raw_vec(&self) -> Vec<c64> {
        self.data.to_vec()
    }

    // Generic conversion for any Tensor into a Vec<T>.
    pub fn raw_vec_as<T: Fromc64>(&self) -> Vec<T> {
        self.data.iter().map(|&x| T::from_c64(x)).collect()
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

// implement val() im() and re() for scalar tensors
impl<D> Scalar<D> {

    pub fn im(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0].im().complex()],
            _phantom: PhantomData,
        }
    }

    pub fn re(&self) -> Scalar<D> {
        Scalar::<D> {
            data: [self.data[0].re().complex()],
            _phantom: PhantomData,
        }
    }

}

// implement += and -= for all tensors
impl<D, const ROWS: usize, const COLS: usize> AddAssign for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    fn add_assign(&mut self, other: Self) {
        for i in 0..ROWS {
            for j in 0..COLS {
                self.data[i * COLS + j] += other.data[i * COLS + j];
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
        if L != 0 { parts.push(format!("L^{}", L)); }
        if M != 0 { parts.push(format!("M^{}", M)); }
        if T != 0 { parts.push(format!("T^{}", T)); }
        if Θ != 0 { parts.push(format!("Θ^{}", Θ)); }
        if I != 0 { parts.push(format!("I^{}", I)); }
        if N != 0 { parts.push(format!("N^{}", N)); }
        if J != 0 { parts.push(format!("J^{}", J)); }
        if parts.is_empty() {
            write!(f, "Dimensionless")
        } else {
            write!(f, "{}", parts.join(" * "))
        }
    }
}

impl<D: std::fmt::Display + Default, const ROWS: usize, const COLS: usize> std::fmt::Display
    for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Create a dummy instance of D via Default and let its Display impl format it.
        writeln!(f, "Tensor [{}x{}]: {}", ROWS, COLS, D::default())?;
        for i in 0..ROWS {
            write!(f, "(")?;
            for j in 0..COLS {
                write!(f, " {} ", self.data[i * COLS + j])?;
            }
            writeln!(f, ")")?;
        }
        Ok(())
    }
}

impl<D: std::fmt::Debug + Default, const ROWS: usize, const COLS: usize> std::fmt::Debug for Tensor<D, ROWS, COLS>
where
    [(); ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dimension", &D::default())
            .field("shape", &format!("{}x{}", ROWS, COLS))
            .field("data", &self.data)
            .finish()
    }
}

// As a macro
#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr) => {
        assert_approx_eq!($left, $right, Scalar::EPSILON);
    };
    ($left:expr, $right:expr, $epsilon:expr) => {{
        let left_val = $left;
        let right_val = $right;
        let abs_diff = (left_val - right_val).norm();
        assert!(
            abs_diff.im() < $epsilon && abs_diff.re() < $epsilon,
            "assertion failed: `(left ≈ right)`\n  left: `{}`\n right: `{}`\n  diff: `{}`\n ",
            left_val, right_val, abs_diff,
        );
    }};
}

// macros for math shit
#[macro_export]
macro_rules! cvec {
    () => {
        // A 0-element vector; you may also choose to panic.
        Tensor::<crate::dimension::Dimensionless, 0, 1>::zero()
    };
    ($($x:expr),+ $(,)?) => {{
        // Count the number of elements provided.
        const N: usize = <[()]>::len(&[$(cvec!(@replace $x)),*]);
        Tensor::<crate::dimension::Dimensionless, N, 1>::new::<crate::units::Unitless>([
            $($x.complex()),*
        ])
    }};
    (@replace $_x:expr) => { () };
}