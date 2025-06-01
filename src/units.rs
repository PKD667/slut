use crate::dimension::*;
use crate::*;

use crate::complex::c64;
use crate::tensor::element::*;

// define units for our dimensions
// units will be used as interfaces to our dimensions
#[derive(Clone, Copy)]
pub struct UnitParameters {
    pub(crate) scale: f64,
    pub(crate) offset: f64,
    pub(crate) symbol: &'static str,
    pub(crate) name: &'static str,
}

pub trait Unit {
    type Dimension; // Associated dimension type
    
    fn parameters() -> UnitParameters;

    fn to<S: Unit<Dimension = Self::Dimension>, E: TensorElement>(value: E) -> E
    {
        let base = Self::to_base(value.into());
        S::from_base(base).into()
    }

    fn from<S: Unit<Dimension = Self::Dimension>, E: TensorElement>(value: E) -> E
    {
        let base = S::to_base(value.into());
        Self::from_base(base).into()
    }

    fn to_base(value: c64) -> c64
    {
        let params = Self::parameters();
        c64::from(( ((value.a * params.scale) + params.offset) as f64, ((value.b * params.scale) + params.offset) as f64))
    }

    fn from_base(value: c64) -> c64
    {
        let params = Self::parameters();
        c64::from(( ((value.a / params.scale) - params.offset) as f64, ((value.b / params.scale) - params.offset) as f64))
    }

    // print
    fn symbol() -> &'static str {
        Self::parameters().symbol
    }
    fn name() -> &'static str {
        Self::parameters().name
    }

    /// Returns the conversion ratio from the current unit into unit T.
    ///
    /// # Example
    /// 
    /// ```
    /// use slut::units::Unit;
    /// use slut::si::{Meter, Centimeter};
    /// 
    /// // Get the ratio from meters to centimeters
    /// let ratio = Meter::ratio::<Centimeter>();
    /// assert_eq!(ratio, 100.0); // 1 meter = 100 centimeters
    /// ```
    ///
    /// This function assumes that both units share the same Dimension.
    fn ratio<T: Unit<Dimension = Self::Dimension>>() -> f64 {
        let self_params = Self::parameters();
        let other_params = T::parameters();
        self_params.scale / other_params.scale
    }
}
// Unitless trait unit
pub struct Unitless;
impl Unit for Unitless {
    type Dimension = Dimensionless;
    fn parameters() -> UnitParameters {
        UnitParameters {
            scale: 1.0,
            offset: 0.0,
            symbol: "",
            name: "Unitless",
        }
    }
}
// implement default for Unitless
impl Default for Unitless {
    fn default() -> Self {
        Unitless
    }
}

// ---------- MACROS ----------

use std::marker::PhantomData;

// --- UnitMul definition and impl ---

pub struct UnitMul<L: Unit, R: Unit>(PhantomData<(L, R)>);

impl<L: Unit, R: Unit> Unit for UnitMul<L, R>
where
    L::Dimension: MultiplyDimensions<R::Dimension>,
{
    type Dimension = dim_mul!((<L as Unit>::Dimension), (<R as Unit>::Dimension));
    fn parameters() -> UnitParameters {
        let lhs_params = L::parameters();
        let rhs_params = R::parameters();
        UnitParameters {
            scale: lhs_params.scale * rhs_params.scale,
            offset: 0.0,
            symbol: "", // Optionally, combine lhs_params.symbol and rhs_params.symbol
            name: "",   // Optionally, combine lhs_params.name and rhs_params.name
        }
    }
}

// --- UnitInv definition and impl ---

pub struct UnitInv<T: Unit>(PhantomData<T>);

impl<T: Unit> Unit for UnitInv<T>
where
    T::Dimension: InvertDimension,
{
    // Wrap the type expression in parentheses.
    type Dimension = dim_inv!((<T as Unit>::Dimension));
    fn parameters() -> UnitParameters {
        let params = T::parameters();
        UnitParameters {
            scale: 1.0 / params.scale,
            offset: 0.0,
            symbol: "", // Optionally, adjust to display inversion (e.g., "1/<symbol>")
            name: "",   // Optionally, adjust to display inversion (e.g., "per <name>")
        }
    }
}

// --- Macros for unit multiplication, inversion, division, and ratio ---

#[macro_export]
macro_rules! unit_mul {
    ($lhs:ty, $rhs:ty) => {
        UnitMul<$lhs, $rhs>
    };
}

#[macro_export]
macro_rules! unit_inv {
    ($unit:ty) => {
        UnitInv<$unit>
    };
}

#[macro_export]
macro_rules! unit_div {
    ($lhs:ty, $rhs:ty) => {
        UnitMul<$lhs, UnitInv<$rhs>>
    };
}

