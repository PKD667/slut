// First, our dimension type (simplified version)
#[derive(Clone, Copy)]
pub struct Dimension<const L: i32, const M: i32, const T: i32, const Θ: i32 = 0, const I: i32 = 0, const N: i32 = 0, const J: i32 = 0>;

pub trait NormalizeDimension {
    type Output;
}

impl<const L: i32, const M: i32, const T: i32, const Θ: i32, const I: i32, const N: i32, const J: i32>
    NormalizeDimension for Dimension<L, M, T, Θ, I, N, J>
{
    type Output = Dimension<{ L }, { M }, { T }, { Θ }, { I }, { N }, { J }>;
}

pub trait MultiplyDimensions<Other> {
    type Output;
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
> MultiplyDimensions<Dimension<L2, M2, T2, Θ2, I2, N2, J2>> for Dimension<L1, M1, T1, Θ1, I1, N1, J1>
where
    [(); (L1 + L2) as usize]:,
    [(); (M1 + M2) as usize]:,
    [(); (T1 + T2) as usize]:,
    [(); (Θ1 + Θ2) as usize]:,
    [(); (I1 + I2) as usize]:,
    [(); (N1 + N2) as usize]:,
    [(); (J1 + J2) as usize]:,
{
    type Output = <Dimension<
        { L1 + L2 },
        { M1 + M2 },
        { T1 + T2 },
        { Θ1 + Θ2 },
        { I1 + I2 },
        { N1 + N2 },
        { J1 + J2 }
    > as NormalizeDimension>::Output;
}


pub trait InvertDimension {
    type Output;
}

impl<const L: i32, const M: i32, const T: i32, const Θ: i32, const I: i32, const N: i32, const J: i32>
    InvertDimension for Dimension<L, M, T, Θ, I, N, J>
where 
    [(); {-L} as usize]:,
    [(); {-M} as usize]:,
    [(); {-T} as usize]:,
    [(); {-Θ} as usize]:,
    [(); {-I} as usize]:,
    [(); {-N} as usize]:,
    [(); {-J} as usize]:,
{
    type Output = Dimension<
        {-L},
        {-M},
        {-T},
        {-Θ},
        {-I},
        {-N},
        {-J}
    >;
}



// Some type aliases for common dimensions
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


