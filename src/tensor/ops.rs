// Tensor operation implementations

use crate::complex::c64;
// Import DimInvert directly from dimension, remove unused ConstAdd
use crate::dimension::{DimInvert, DimTransform, Dimension, Dimensionless, InvertDimension, MultiplyDimensions, SqrtDimension};
use crate::tensor::element::TensorElement;
use crate::tensor::Tensor;
// Update morph imports
use crate::tensor::morph::{Morph, DimIdentity, DimAdd, DimSame};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub, Div};
use std::*;

// -----------------------------------------
// ============= OPERATIONS ================
// -----------------------------------------

// --- Addition ---
impl<E: TensorElement + Add<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Add for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.combine_with_transform::<_, _, DimAdd>(other, |a, b| a + b)
    }
}

// --- Subtraction ---
impl<E: TensorElement + Sub<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Sub for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.combine_with_transform::<_, _, DimAdd>(other, |a, b| a - b)
    }
}

// --- Matrix Multiplication ---
// NOTE: This implementation is specifically tailored for the norm calculation's
// vector dot product (1, 1, ROWS) x (1, ROWS, 1) -> (1, 1, 1) due to the
// restrictive 1x1x1 output signature. It does NOT perform general matrix multiplication.
impl<
    E: TensorElement + Mul<Output = E> + Add<Output = E> + Copy,
    const LAYERS: usize, // Must be 1 for the intended dot product case
    const L1: i32,
    const M1: i32,
    const T1: i32,
    const Θ1: i32,
    const I1: i32,
    const N1: i32,
    const J1: i32,
    const ROWS: usize, // Represents the '1' dimension in (1, 1, COMMON) for ct
    const COMMON: usize, // Represents the 'COMMON' dimension
> Tensor<E, Dimension<L1, M1, T1, Θ1, I1, N1, J1>, LAYERS, ROWS, COMMON> // Expecting self to be ct: (1, 1, COMMON)
where
    [(); LAYERS * ROWS * COMMON]:, // = [(); 1 * 1 * COMMON]
    // Dimension bounds are primarily checked in the function signature below
{
    pub fn matmul<
        const L2: i32,
        const M2: i32,
        const T2: i32,
        const Θ2: i32,
        const I2: i32,
        const N2: i32,
        const J2: i32,
        const COLS: usize, // Represents the '1' dimension in (1, COMMON, 1) for original vector
    >(
        self, // Represents ct: (1, 1, COMMON)
        other: Tensor<E, Dimension<L2, M2, T2, Θ2, I2, N2, J2>, LAYERS, COMMON, COLS>, // Represents original vector: (1, COMMON, 1)
    ) -> Tensor<
        E,
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<
            Dimension<L2, M2, T2, Θ2, I2, N2, J2>
        >>::Output,
         1, // Output is scalar
         1,
         1,
    >
    where
        Dimension<L1, M1, T1, Θ1, I1, N1, J1>: MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>>::Output:, // Ensure output dimension exists
        Dimension<L2, M2, T2, Θ2, I2, N2, J2>: Copy, // Required by PhantomData usage? Or norm's needs? Keep for now.
        // Add the missing bound for the `other` tensor parameter
        [(); LAYERS * COMMON * COLS]:, // Bound for other
        // Bounds for expected shapes:
        [(); 1 * 1 * COMMON]:, // self (ct) shape
        [(); 1 * COMMON * 1]:, // other (vec) shape
        [(); 1 * 1 * 1]:, // Output shape
        // Ensure LAYERS, ROWS, COLS match the expected dot product pattern
        // LAYERS must be 1
        // self.ROWS must be 1
        // other.COLS must be 1
        // self.COMMON must equal other.ROWS
    {
        // Assert shapes for the dot product case this function now specifically handles
        assert_eq!(LAYERS, 1, "matmul (dot product) requires LAYERS = 1");
        assert_eq!(ROWS, 1, "matmul (dot product) requires self.ROWS = 1"); // self is ct (1, 1, COMMON)
        assert_eq!(COLS, 1, "matmul (dot product) requires other.COLS = 1"); // other is vec (1, COMMON, 1)
        // self.COMMON is COMMON, other.ROWS is COMMON - they match by definition

        // Perform the dot product: sum(self[0, 0, k] * other[0, k, 0]) for k in 0..COMMON
        let mut sum = E::zero();
        for k in 0..COMMON {
            // self is ct (1, 1, COMMON), index is 0 * (1*COMMON) + 0 * COMMON + k = k
            // other is vec (1, COMMON, 1), index is 0 * (COMMON*1) + k * 1 + 0 = k
            sum = sum + self.data[k] * other.data[k];
        }

        Tensor::<E, <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<
            Dimension<L2, M2, T2, Θ2, I2, N2, J2>
        >>::Output, 1, 1, 1> {
            data: [sum], // Store the scalar result
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement + Mul<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn scale<DS>(
        self,
        scalar: Tensor<E, DS, 1, 1, 1>,
    ) -> Tensor<E, <D as MultiplyDimensions<DS>>::Output, LAYERS, ROWS, COLS>
    where
        D: MultiplyDimensions<DS>,
        DS: Copy,
        <D as MultiplyDimensions<DS>>::Output:,
        [(); 1]:,
    {
        let s = scalar.data[0];
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v * s)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<<D as MultiplyDimensions<DS>>::Output>,
        }
    }
}

// --- Multiplication by Scalar ---
impl<E, D, DS, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Mul<Tensor<E, DS, 1, 1, 1>> for Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Mul<Output = E> + Copy,
    D: MultiplyDimensions<DS>,
    DS: Copy,
    <D as MultiplyDimensions<DS>>::Output:,
    [(); LAYERS * ROWS * COLS]:,
    [(); 1]:,
{
    type Output = Tensor<E, <D as MultiplyDimensions<DS>>::Output, LAYERS, ROWS, COLS>;

    fn mul(self, rhs: Tensor<E, DS, 1, 1, 1>) -> Self::Output {
        self.scale(rhs)
    }
}

// --- Division by Scalar ---
impl<E, D, DS, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Div<Tensor<E, DS, 1, 1, 1>> for Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Div<Output = E> + Copy + PartialEq,
    DS: InvertDimension + Copy, // DS is the dimension of rhs
    DimInvert: DimTransform<DS>, // Required by rhs.inv()
    <DimInvert as DimTransform<DS>>::Output: Copy, // The output dimension of the inverse transform needs Copy for scale
    D: MultiplyDimensions<<DimInvert as DimTransform<DS>>::Output>, // Update the MultiplyDimensions bound to use the actual dimension from inv()
    <D as MultiplyDimensions<<DimInvert as DimTransform<DS>>::Output>>::Output:, // Ensure the final output dimension type exists
    [(); LAYERS * ROWS * COLS]:,
    [(); 1]:,
{
    type Output = Tensor<
        E,
        <D as MultiplyDimensions<<DimInvert as DimTransform<DS>>::Output>>::Output,
        LAYERS,
        ROWS,
        COLS
    >;

    fn div(self, rhs: Tensor<E, DS, 1, 1, 1>) -> Self::Output {
        self.scale(rhs.inv())
    }
}

// --- Scalar Inverse ---
impl<E: TensorElement + Div<Output = E> + Copy + PartialEq, D> Tensor<E, D, 1, 1, 1>
where
    [(); 1]:,
    D: InvertDimension, // Still needed conceptually and for the DimTransform impl
    DimInvert: DimTransform<D>, // Required by Morph
{
    // Change the return type dimension to match apply_morph's output
    pub fn inv(self) -> Tensor<E, <DimInvert as DimTransform<D>>::Output, 1, 1, 1> {
        let reciprocal_morph = Morph::<E, _, D, DimInvert>::new(|v| E::one() / v);
        // The return type of apply_morph now directly matches the function signature
        self.apply_morph(reciprocal_morph)
    }
}

// --- Negation ---
impl<E: TensorElement + Neg<Output = E> + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Neg for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy,
{
    type Output = Self;

    fn neg(self) -> Self {
        let neg_morph = Morph::<E, _, D, DimIdentity>::new(|v| -v);
        self.apply_morph(neg_morph)
    }
}

// --- Type Conversion ---
impl<E, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    E: TensorElement + Into<c64> + Copy,
    D: Copy,
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn to_c64(&self) -> Tensor<c64, D, LAYERS, ROWS, COLS> {
        let data: [c64; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| v.into())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Tensor {
            data,
            _phantom: self._phantom,
        }
    }
}

// --- Conjugate ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy,
{
    pub fn conjugate(self) -> Self {
        let conj_morph = Morph::<E, _, D, DimIdentity>::new(|v| v.conjugate());
        self.apply_morph(conj_morph)
    }

    pub fn conjugate_transpose(self) -> Tensor<E,D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        self.transpose().conjugate()
    }
}

// --- Transpose ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy,
{
    pub fn transpose(self) -> Tensor<E,D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        let mut transposed = [E::zero(); LAYERS * COLS * ROWS];
        for l in 0..LAYERS {
            for i in 0..ROWS {
                for j in 0..COLS {
                    let src = l * (ROWS * COLS) + i * COLS + j;
                    let dst = l * (COLS * ROWS) + j * ROWS + i;
                    transposed[dst] = self.data[src];
                }
            }
        }
        Tensor::<E,D, LAYERS, COLS, ROWS> {
            data: transposed,
            _phantom: self._phantom,
        }
    }
}

// --- Dot Product and Inner Product for Column Vectors ---
impl<
    E: TensorElement + Mul<Output = E> + Add<Output = E> + Copy,
    const L1: i32, const M1: i32, const T1: i32, const Θ1: i32, const I1: i32, const N1: i32, const J1: i32,
    const LEN: usize // RENAMED from N: The length of the column vectors
>
Tensor<E, Dimension<L1, M1, T1, Θ1, I1, N1, J1>, 1, LEN, 1> // Column Vector (1, LEN, 1)
where
    [(); LEN]:, // UPDATED Bound for the vector itself
    [(); 1 * 1 * LEN]:, // UPDATED Bound for the transposed vector (1, 1, LEN)
    [(); 1 * 1 * 1]:, // Bound for the output scalar (1, 1, 1)
    [(); 1 * LEN * 1]:, // UPDATED Bound for the other vector (1, LEN, 1)
    Dimension<L1, M1, T1, Θ1, I1, N1, J1>: Copy, // Needed for transpose/conj_transpose
{
    /// Calculates the dot product: self^T * other
    pub fn dot_product<
        const L2: i32, const M2: i32, const T2: i32, const Θ2: i32, const I2: i32, const N2: i32, const J2: i32
    >(
        self,
        other: Tensor<E, Dimension<L2, M2, T2, Θ2, I2, N2, J2>, 1, LEN, 1> // UPDATED Other Column Vector (1, LEN, 1)
    ) -> Tensor<E, <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>>::Output, 1, 1, 1>
    where
        Dimension<L1, M1, T1, Θ1, I1, N1, J1>: MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>>::Output:,
        Dimension<L2, M2, T2, Θ2, I2, N2, J2>: Copy, // Required by matmul
        [(); 1 * LEN * 1]:, // UPDATED Bound for other vector
        [(); 1 * 1 * 1]:, // Bound for output scalar
    {
        let self_t = self.transpose(); // (1, LEN, 1) -> (1, 1, LEN)
        // matmul: (1, 1, LEN) x (1, LEN, 1) -> (1, 1, 1)
        self_t.matmul(other)
    }

    /// Calculates the inner product: self^† * other (conjugate transpose)
    pub fn inner_product<
        const L2: i32, const M2: i32, const T2: i32, const Θ2: i32, const I2: i32, const N2: i32, const J2: i32
    >(
        self,
        other: Tensor<E, Dimension<L2, M2, T2, Θ2, I2, N2, J2>, 1, LEN, 1> // UPDATED Other Column Vector (1, LEN, 1)
    ) -> Tensor<E, <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>>::Output, 1, 1, 1>
    where
        Dimension<L1, M1, T1, Θ1, I1, N1, J1>: MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>,
        <Dimension<L1, M1, T1, Θ1, I1, N1, J1> as MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>>::Output:,
        Dimension<L2, M2, T2, Θ2, I2, N2, J2>: Copy, // Required by matmul
        [(); 1 * LEN * 1]:, // UPDATED Bound for other vector
        [(); 1 * 1 * 1]:, // Bound for output scalar
    {
        let self_ct = self.conjugate_transpose(); // (1, LEN, 1) -> (1, 1, LEN)
        // matmul: (1, 1, LEN) x (1, LEN, 1) -> (1, 1, 1)
        self_ct.matmul(other)
    }
}


// ip! macro
#[macro_export]
macro_rules! ip {
    ($a:expr, $b:expr) => {
        $a.inner_product($b)
    };
}

#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {
        $a.dot_product($b)
    };
}


// --Boolean AND ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy implicitly via combine_with_transform
{
    pub fn and(self, other: Self) -> Self {
        let and_fn = |a: E, b: E| if a != E::zero() && b != E::zero() { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation (requires D == D, outputs D)
        self.combine_with_transform::<_, _, DimSame>(other, and_fn)
    }
}

// --- Boolean OR ---
impl<E: TensorElement + PartialEq + Copy, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn or(self, other: Self) -> Self {
        let or_fn = |a: E, b: E| if a != E::zero() || b != E::zero() { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, or_fn)
    }
}

// --- Comparison EQ ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn eq(self, other: Self) -> Self {
        let eq_fn = |a: E, b: E| if a == b { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, eq_fn)
    }
}

// --- Comparison NE ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn ne(self, other: Self) -> Self {
        let ne_fn = |a: E, b: E| if a != b { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, ne_fn)
    }
}

// --- Comparison GT ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn gt(self, other: Self) -> Self {
        let gt_fn = |a: E, b: E| if a > b { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, gt_fn)
    }
}

// --- Comparison GE ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn ge(self, other: Self) -> Self {
        let ge_fn = |a: E, b: E| if a >= b { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, ge_fn)
    }
}

// --- Comparison LT ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn lt(self, other: Self) -> Self {
        let lt_fn = |a: E, b: E| if a < b { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, lt_fn)
    }
}

// --- Comparison LE ---
impl<E: TensorElement,D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E,D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    D: Copy, // DimSame requires D: Copy
{
    pub fn le(self, other: Self) -> Self {
        let le_fn = |a: E, b: E| if a <= b { E::one() } else { E::zero() };
        // Use DimSame for dimension preservation
        self.combine_with_transform::<_, _, DimSame>(other, le_fn)
    }
}
