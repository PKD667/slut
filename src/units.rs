pub type STYPE = f32;

use crate::dimension::*;
use crate::*;

// define units for our dimensions
// units will be used as interfaces to our dimensions
#[derive(Clone, Copy)]
pub struct UnitParameters {
    pub(crate) scale: STYPE,
    pub(crate) offset: STYPE,
    pub(crate) symbol: &'static str,
    pub(crate) name: &'static str,
}

// Unit trait - defines how to convert to/from base units
pub trait Unit {
    type Dimension;  // Associated dimension type
    fn parameters() -> UnitParameters;

    fn to<S: Unit<Dimension = Self::Dimension>>(value: STYPE) -> STYPE {
        let params = Self::parameters();
        S::from_base(params.scale * Self::to_base(value) + params.offset)
    }

    fn from<S: Unit<Dimension = Self::Dimension>>(value: STYPE) -> STYPE {
        let params = Self::parameters();
        Self::from_base((value - params.offset) / params.scale)
    }

    fn to_base(value: STYPE) -> STYPE {
        let params = Self::parameters();
        value * params.scale + params.offset
    }

    fn from_base(value: STYPE) -> STYPE {
        let params = Self::parameters();
        (value - params.offset) / params.scale
    }

    // print
    fn symbol() -> &'static str {
        Self::parameters().symbol
    }
    fn name() -> &'static str {
        Self::parameters().name
    }
}

// Unitless trait unit
pub struct Unitless;
impl Unit for Unitless {
    type Dimension = Dimensionless;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "", name: "Unitless" }
    }
}




// ---------- MACROS ----------

use std::marker::PhantomData;

// --- UnitMul definition and impl ---


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

// --- Macros for unit multiplication and inversion ---

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