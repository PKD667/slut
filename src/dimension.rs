#[derive(Clone, Copy)]
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

// Dummy constraint to force computed constants to be used.
pub trait ConstCheck<const N: i32> {}
impl<const N: i32> ConstCheck<N> for () {}

// MultiplyDimensions using the dummy ConstCheck bounds.
pub trait MultiplyDimensions<Other> {
    type Output;
}
impl<
    const L1: i32, const M1: i32, const T1: i32,
    const Θ1: i32, const I1: i32, const N1: i32, const J1: i32,
    const L2: i32, const M2: i32, const T2: i32,
    const Θ2: i32, const I2: i32, const N2: i32, const J2: i32,
> MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>>
    for Dimension<L1, M1, T1, Θ1, I1, N1, J1>
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
}

// InvertDimension using the dummy ConstCheck bounds.
pub trait InvertDimension {
    type Output;
}
impl<
    const L: i32, const M: i32, const T: i32,
    const Θ: i32, const I: i32, const N: i32, const J: i32
> InvertDimension for Dimension<L, M, T, Θ, I, N, J>
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
}

// Since Dimension is zero‑sized, we can implement Default.
impl<const L: i32, const M: i32, const T: i32, const Θ: i32, const I: i32, const N: i32, const J: i32>
    Default for Dimension<L, M, T, Θ, I, N, J>
{
    fn default() -> Self {
        Self
    }
}

// Removed the NormalizeDimension and AutoNormalize traits.

// Some type aliases for common dimensions.
pub type Dimensionless = Dimension<0, 0, 0, 0, 0, 0, 0>;
pub type Length = Dimension<1, 0, 0, 0, 0, 0, 0>;
pub type Area = Dimension<2, 0, 0, 0, 0, 0, 0>;
pub type Volume = Dimension<3, 0, 0, 0, 0, 0, 0>;
pub type Mass = Dimension<0, 1, 0, 0, 0, 0, 0>;
pub type Time = Dimension<0, 0, 1, 0, 0, 0, 0>;
pub type Frequency = Dimension<0, 0, -1, 0, 0, 0, 0>;
pub type Current = Dimension<0, 0, 0, 0, 1, 0, 0>;
pub type Temperature = Dimension<0, 0, 0, 1, 0, 0, 0>;
pub type Amount = Dimension<0, 0, 0, 0, 0, 1, 0>;
pub type Intensity = Dimension<0, 0, 0, 0, 0, 0, 1>;
pub type Velocity = Dimension<1, 0, -1, 0, 0, 0, 0>;
pub type Acceleration = Dimension<1, 0, -2, 0, 0, 0, 0>;
pub type Force = Dimension<1, 1, -2, 0, 0, 0, 0>;
pub type Energy = Dimension<2, 1, -2, 0, 0, 0, 0>;
pub type Power = Dimension<2, 1, -3, 0, 0, 0, 0>;
pub type Pressure = Dimension<-1, 1, -2, 0, 0, 0, 0>;
pub type ThermalConductivity = Dimension<-1, 1, -3, 0, 0, 0, 1>;