use crate::*;
use crate::units::*;
use crate::tensor::*;
use crate::dimension::*;
use crate::complex::c64;

// ----------------------------------
// =========== SI DIMENSIONS =========
// ----------------------------------

pub type Area = Dimension<2, 0, 0, 0, 0, 0, 0>;
pub type Volume = Dimension<3, 0, 0, 0, 0, 0, 0>;
pub type Frequency = Dimension<0, 0, -1, 0, 0, 0, 0>;
pub type Intensity = Dimension<0, 0, 0, 0, 0, 0, 1>;
pub type Velocity = Dimension<1, 0, -1, 0, 0, 0, 0>;
pub type Acceleration = Dimension<1, 0, -2, 0, 0, 0, 0>;
pub type Force = Dimension<1, 1, -2, 0, 0, 0, 0>;
pub type Energy = Dimension<2, 1, -2, 0, 0, 0, 0>;
pub type Entropy = Dimension<2, 1, -2, -1, 0, 0, 0>;
pub type Power = Dimension<2, 1, -3, 0, 0, 0, 0>;
pub type Pressure = Dimension<-1, 1, -2, 0, 0, 0, 0>;
pub type Charge = Dimension<0, 0, 1, 0, 1, 0, 0>;
pub type HeatCapacity = Dimension<2, 1, -2, -1, 0, 0, 0>;

// ----------------------------------
// =========== SI UNITS =============
// ----------------------------------

// Macro to implement the Unit trait and Default for a unit type.
#[macro_export]
macro_rules! impl_unit {
    ($unit:ident, $dimension:ty, $scale:expr, $offset:expr, $symbol:expr, $name:expr) => {
        impl Default for $unit {
            fn default() -> Self {
                $unit
            }
        }
        impl Unit for $unit {
            type Dimension = $dimension;
            fn parameters() -> UnitParameters {
                UnitParameters {
                    scale: $scale,
                    offset: $offset,
                    symbol: $symbol,
                    name: $name,
                }
            }
        }
    };
}

// ---------- Length Units ----------
pub struct Nanometer;
pub struct Micrometer;
pub struct Millimeter;
pub struct Meter;
pub struct Kilometer;
pub struct Centimeter;
pub struct LightYear;
pub struct Ångström;

impl_unit!(Nanometer, Length, 1.0e-9, 0.0, "nm", "Nanometer");
impl_unit!(Micrometer, Length, 1.0e-6, 0.0, "µm", "Micrometer");
impl_unit!(Millimeter, Length, 1.0e-3, 0.0, "mm", "Millimeter");
impl_unit!(Meter, Length, 1.0, 0.0, "m", "Meter");
impl_unit!(Kilometer, Length, 1000.0, 0.0, "km", "Kilometer");
impl_unit!(Centimeter, Length, 0.01, 0.0, "cm", "Centimeter");
impl_unit!(LightYear, Length, 9.461e15, 0.0, "ly", "LightYear");
impl_unit!(Ångström, Length, 1.0e-10, 0.0, "Å", "Ångström");

// ---------- Time Units ----------
pub struct Nanosecond;
pub struct Microsecond;
pub struct Millisecond;
pub struct Second;
pub struct Minute;
pub struct Hour;
pub struct Day;
pub struct Year;
pub struct Century;

impl_unit!(Nanosecond, Time, 1.0e-9, 0.0, "ns", "Nanosecond");
impl_unit!(Microsecond, Time, 1.0e-6, 0.0, "µs", "Microsecond");
impl_unit!(Millisecond, Time, 1.0e-3, 0.0, "ms", "Millisecond");
impl_unit!(Second, Time, 1.0, 0.0, "s", "Second");
impl_unit!(Minute, Time, 60.0, 0.0, "min", "Minute");
impl_unit!(Hour, Time, 3600.0, 0.0, "h", "Hour");
impl_unit!(Day, Time, 86400.0, 0.0, "day", "Day");
impl_unit!(Year, Time, 365.25 * 86400.0, 0.0, "year", "Year");
impl_unit!(Century, Time, 100.0 * 365.25 * 86400.0, 0.0, "century", "Century");

// ---------- Force Units ----------
pub struct Newton;
pub struct Lbf;

impl_unit!(Newton, Force, 1.0, 0.0, "N", "Newton");
impl_unit!(Lbf, Force, 4.44822, 0.0, "lbf", "Pound-force");

// ---------- Velocity Units ----------
pub struct MetersPerSecond;
pub struct KilometersPerHour;

impl_unit!(MetersPerSecond, Velocity, 1.0, 0.0, "m/s", "MetersPerSecond");
impl_unit!(KilometersPerHour, Velocity, 1.0 / 3.6, 0.0, "km/h", "KilometersPerHour");

// ---------- Acceleration Unit ----------
pub struct MetersPerSecondSquared;

impl_unit!(MetersPerSecondSquared, Acceleration, 1.0, 0.0, "m/s²", "MeterPerSecondSquared");

// ---------- Temperature Units ----------
pub struct Celsius;
pub struct Kelvin;
pub struct Fahrenheit;

impl_unit!(Kelvin, Temperature, 1.0, 0.0, "K", "Kelvin");
impl_unit!(Celsius, Temperature, 1.0, 273.15, "°C", "Celsius");
impl_unit!(Fahrenheit, Temperature, 5.0/9.0, 255.372222, "°F", "Fahrenheit");

// ---------- Energy Units ----------
pub struct Joule;
pub struct ElectronVolt;

impl_unit!(Joule, Energy, 1.0, 0.0, "J", "Joule");
impl_unit!(ElectronVolt, Energy, 1.602176634e-19, 0.0, "eV", "ElectronVolt");

// ---------- Mass Units ----------
pub struct Kilogram;
pub struct Gram;
pub struct Pound;
pub struct Dalton;

impl_unit!(Kilogram, Mass, 1.0, 0.0, "kg", "Kilogram");
impl_unit!(Gram, Mass, 1.0e-3, 0.0, "g", "Gram");
impl_unit!(Pound, Mass, 0.45359237, 0.0, "lb", "Pound");
impl_unit!(Dalton, Mass, 1.66053906660e-27, 0.0, "Da", "Dalton");

// ---------- Luminous Intensity Units ----------
pub struct Candela;

impl_unit!(Candela, LuminousIntensity, 1.0, 0.0, "cd", "Candela");

// ---------- Amount Units ----------
pub struct Mole;

impl_unit!(Mole, Amount, 1.0, 0.0, "mol", "Mole");

// ---------- Pressure Units ----------
pub struct Pascal;
pub struct Bar;
pub struct Atmosphere;

impl_unit!(Pascal, Pressure, 1.0, 0.0, "Pa", "Pascal");
impl_unit!(Bar, Pressure, 100000.0, 0.0, "bar", "Bar");
impl_unit!(Atmosphere, Pressure, 101325.0, 0.0, "atm", "Atmosphere");

// ---------- Power Units ----------
pub struct Watt;
pub struct Horsepower;

impl_unit!(Watt, Power, 1.0, 0.0, "W", "Watt");
impl_unit!(Horsepower, Power, 745.7, 0.0, "hp", "Horsepower");

// ---------- Area Units ----------
pub struct SquareMeter;

impl_unit!(SquareMeter, Area, 1.0, 0.0, "m²", "SquareMeter");

// ---------- Volume Units ----------
pub struct CubicMeter;
pub struct Liter;
pub struct Milliliter;
pub struct Gallon;

impl_unit!(CubicMeter, Volume, 1.0, 0.0, "m³", "CubicMeter");
impl_unit!(Liter, Volume, 1.0e-3, 0.0, "L", "Liter");
impl_unit!(Milliliter, Volume, 1.0e-6, 0.0, "mL", "Milliliter");
impl_unit!(Gallon, Volume, 3.78541, 0.0, "gal", "Gallon");

// ---------- Frequency Units ----------
pub struct Hertz;

impl_unit!(Hertz, Frequency, 1.0, 0.0, "Hz", "Hertz");

// ---------- Electric Current Units ----------
pub struct Ampere;

impl_unit!(Ampere, Current, 1.0, 0.0, "A", "Ampere");

// -------------------------------------------------------------------
// SI Constants (using Scalar type)
// -------------------------------------------------------------------
use std::marker::PhantomData;

pub const C: Scalar<Velocity> = Scalar {
    data: [ c64 { a: 299_792_458.0, b: 0.0}],
    _phantom: PhantomData
};

pub const G: Scalar<Acceleration> = Scalar {
    data: [c64 { a: 9.80665, b: 0.0}],
    _phantom: PhantomData
};

pub const H: Scalar<dim_mul!(Energy,Time)> = Scalar {
    data: [c64 { a: 6.62607015e-34, b: 0.0}],
    _phantom: PhantomData
};

pub const K: Scalar<Entropy> = Scalar {
    data: [c64 { a: 1.380649e-23, b: 0.0}],
    _phantom: PhantomData
};

pub const E: Scalar<Charge> = Scalar {
    data: [c64 { a: 1.602176634e-19, b: 0.0}],
    _phantom: PhantomData
};

pub const M_E: Scalar<Mass> = Scalar {
    data: [c64 { a: 9.10938356e-31, b: 0.0}],
    _phantom: PhantomData
};

pub const M_P: Scalar<Mass> = Scalar {
    data: [c64 { a: 1.6726219e-27, b: 0.0}],
    _phantom: PhantomData
};

pub const N_A: Scalar<Amount> = Scalar {
    data: [c64 { a: 6.02214076e23, b: 0.0}],
    _phantom: PhantomData
};