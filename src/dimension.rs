use crate::units::*;
use crate::si::*;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
pub struct Dimension<const L: i32, const M: i32, const T: i32, const Θ: i32 = 0, const I: i32 = 0, const N: i32 = 0, const J: i32 = 0>;

// A helper trait for compile‑time addition.
pub trait ConstAdd<const A: i32, const B: i32> {
    const OUTPUT: i32;
}
impl<const A: i32, const B: i32> ConstAdd<A, B> for () {
    const OUTPUT: i32 = A + B;
}

// Helper trait for negation.
pub trait ConstNeg<const N: i32> {
    const OUTPUT: i32;
}
impl<const N: i32> ConstNeg<N> for () {
    const OUTPUT: i32 = -N;
}

// helper trait for integer division
pub trait ConstDiv<const A: i32, const B: i32> {
    const OUTPUT: i32;
}

impl<const A: i32, const B: i32> ConstDiv<A, B> for () {
    const OUTPUT: i32 = A / B;
}

// Dummy constraint to force computed constants to be used.
pub trait ConstCheck<const N: i32> {}
impl<const N: i32> ConstCheck<N> for () {}

// The new generic dimension transformation trait
pub trait DimTransform<Source> {
    type Output;

    // Optional: Provides a name for debugging
    fn name() -> &'static str {
        "generic transform"
    }
}

// Multiplier type to multiply dimensions
pub struct DimMultiply<D>(PhantomData<D>);

// Squaring transformation
pub struct DimSquare;

// Square root transformation
pub struct DimSqrt;

// Inversion transformation
pub struct DimInvert;

// Implementation for multiplication
impl<
    const L1: i32, const M1: i32, const T1: i32,
    const Θ1: i32, const I1: i32, const N1: i32, const J1: i32,
    const L2: i32, const M2: i32, const T2: i32,
    const Θ2: i32, const I2: i32, const N2: i32, const J2: i32,
> DimTransform<Dimension<L1, M1, T1, Θ1, I1, N1, J1>> for DimMultiply<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>
where
    (): ConstCheck<{ <() as ConstAdd<L1, L2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<M1, M2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<T1, T2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<Θ1, Θ2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<I1, I2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<N1, N2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<J1, J2>>::OUTPUT }>,
{
    type Output = Dimension<
        { <() as ConstAdd<L1, L2>>::OUTPUT },
        { <() as ConstAdd<M1, M2>>::OUTPUT },
        { <() as ConstAdd<T1, T2>>::OUTPUT },
        { <() as ConstAdd<Θ1, Θ2>>::OUTPUT },
        { <() as ConstAdd<I1, I2>>::OUTPUT },
        { <() as ConstAdd<N1, N2>>::OUTPUT },
        { <() as ConstAdd<J1, J2>>::OUTPUT }
    >;

    fn name() -> &'static str {
        "multiply"
    }
}

// Implementation for squaring
impl<
    const L: i32, const M: i32, const T: i32,
    const Θ: i32, const I: i32, const N: i32, const J: i32
> DimTransform<Dimension<L, M, T, Θ, I, N, J>> for DimSquare
where
    (): ConstCheck<{ <() as ConstAdd<L, L>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<M, M>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<T, T>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<Θ, Θ>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<I, I>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<N, N>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstAdd<J, J>>::OUTPUT }>,
{
    type Output = Dimension<
        { <() as ConstAdd<L, L>>::OUTPUT },
        { <() as ConstAdd<M, M>>::OUTPUT },
        { <() as ConstAdd<T, T>>::OUTPUT },
        { <() as ConstAdd<Θ, Θ>>::OUTPUT },
        { <() as ConstAdd<I, I>>::OUTPUT },
        { <() as ConstAdd<N, N>>::OUTPUT },
        { <() as ConstAdd<J, J>>::OUTPUT }
    >;

    fn name() -> &'static str {
        "square"
    }
}

// Implementation for square root
impl<
    const L: i32, const M: i32, const T: i32,
    const Θ: i32, const I: i32, const N: i32, const J: i32
> DimTransform<Dimension<L, M, T, Θ, I, N, J>> for DimSqrt
where
    (): ConstCheck<{ <() as ConstDiv<L, 2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstDiv<M, 2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstDiv<T, 2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstDiv<Θ, 2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstDiv<I, 2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstDiv<N, 2>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstDiv<J, 2>>::OUTPUT }>,
{
    type Output = Dimension<
        { <() as ConstDiv<L, 2>>::OUTPUT },
        { <() as ConstDiv<M, 2>>::OUTPUT },
        { <() as ConstDiv<T, 2>>::OUTPUT },
        { <() as ConstDiv<Θ, 2>>::OUTPUT },
        { <() as ConstDiv<I, 2>>::OUTPUT },
        { <() as ConstDiv<N, 2>>::OUTPUT },
        { <() as ConstDiv<J, 2>>::OUTPUT }
    >;

    fn name() -> &'static str {
        "sqrt"
    }
}

// Implementation for inversion
impl<
    const L: i32, const M: i32, const T: i32,
    const Θ: i32, const I: i32, const N: i32, const J: i32
> DimTransform<Dimension<L, M, T, Θ, I, N, J>> for DimInvert
where
    (): ConstCheck<{ <() as ConstNeg<L>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstNeg<M>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstNeg<T>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstNeg<Θ>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstNeg<I>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstNeg<N>>::OUTPUT }>,
    (): ConstCheck<{ <() as ConstNeg<J>>::OUTPUT }>,
{
    type Output = Dimension<
        { <() as ConstNeg<L>>::OUTPUT },
        { <() as ConstNeg<M>>::OUTPUT },
        { <() as ConstNeg<T>>::OUTPUT },
        { <() as ConstNeg<Θ>>::OUTPUT },
        { <() as ConstNeg<I>>::OUTPUT },
        { <() as ConstNeg<N>>::OUTPUT },
        { <() as ConstNeg<J>>::OUTPUT }
    >;

    fn name() -> &'static str {
        "invert"
    }
}

// Keep old traits for backward compatibility, but implement them using DimTransform

// MultiplyDimensions using the new DimTransform trait
pub trait MultiplyDimensions<Other> {
    type Output;
}
impl<D1, D2> MultiplyDimensions<D2> for D1
where
    DimMultiply<D2>: DimTransform<D1>,
{
    type Output = <DimMultiply<D2> as DimTransform<D1>>::Output;
}

pub trait SquareDimension {
    type Output;
}

// square the dimension.
impl<D> SquareDimension for D
where
    DimSquare: DimTransform<D>,
{
    type Output = <DimSquare as DimTransform<D>>::Output;
}

pub trait SqrtDimension {
    type Output;
}

// square root the dimension.
impl<D> SqrtDimension for D
where
    DimSqrt: DimTransform<D>,
{
    type Output = <DimSqrt as DimTransform<D>>::Output;
}

// InvertDimension using the new DimTransform trait.
pub trait InvertDimension {
    type Output;
}
impl<D> InvertDimension for D
where
    DimInvert: DimTransform<D>,
{
    type Output = <DimInvert as DimTransform<D>>::Output;
}

// Since Dimension is zero‑sized, we can implement Default.
impl<const L: i32, const M: i32, const T: i32, const Θ: i32, const I: i32, const N: i32, const J: i32>
    Default for Dimension<L, M, T, Θ, I, N, J>
{
    fn default() -> Self {
        Self
    }
}

// Some type aliases for common dimensions.
pub type Dimensionless = Dimension<0, 0, 0, 0, 0, 0, 0>;
// L
pub type Length = Dimension<1, 0, 0, 0, 0, 0, 0>;
// M
pub type Mass = Dimension<0, 1, 0, 0, 0, 0, 0>;
// T
pub type Time = Dimension<0, 0, 1, 0, 0, 0, 0>;
// Θ
pub type Temperature = Dimension<0, 0, 0, 1, 0, 0, 0>;
// I
pub type Current = Dimension<0, 0, 0, 0, 1, 0, 0>;
// N
pub type Amount = Dimension<0, 0, 0, 0, 0, 1, 0>;
// J
pub type LuminousIntensity = Dimension<0, 0, 0, 0, 0, 0, 1>;

pub trait BaseUnitForDim {
    type Unit: crate::units::Unit<Dimension = Self>;
    fn base_unit() -> Self::Unit;
}

impl BaseUnitForDim for Dimensionless {
    type Unit = Unitless;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for Length {
    type Unit = Meter;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for Mass {
    type Unit = Kilogram;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for Time {
    type Unit = Second;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for Temperature {
    type Unit = Kelvin;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for Current {
    type Unit = Ampere;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for Amount {
    type Unit = Mole;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

impl BaseUnitForDim for LuminousIntensity {
    type Unit = Candela;
    fn base_unit() -> Self::Unit {
        Self::Unit::default()
    }
}

// base unit for dimension macro
#[macro_export]
macro_rules! base_unit_dim {
    ($dim:ty) => {
        <$dim as BaseUnitForDim>::Unit
    };
}

#[macro_export]
macro_rules! dim_inv {
    ($dim:ty) => {
        <$dim as $crate::dimension::InvertDimension>::Output
    };
}

#[macro_export]
macro_rules! dim_mul {
    ($lhs:ty, $rhs:ty) => {
        <$lhs as $crate::dimension::MultiplyDimensions<$rhs>>::Output
    };
}

#[macro_export]
macro_rules! dim_div {
    ($lhs:ty, $rhs:ty) => {
        <$crate::dimension::DimMultiply<<$crate::dimension::DimInvert as $crate::dimension::DimTransform<$rhs>>::Output> as $crate::dimension::DimTransform<$lhs>>::Output
    };
}

/// Asserts that the type of `$value` is a Tensor with the expected dimension type.
/// Returns `$value` on success so that the macro can be used inline.
#[macro_export]
macro_rules! assert_dimension {
    ($value:expr, $expected:ty) => {{
        let _: Tensor<_,$expected, _,_, _> = $value;
        $value
    }};
}