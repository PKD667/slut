use crate::*;
use crate::units::*;
use crate::tensor::*;
use crate::dimension::*;


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


// A simple macro to implement Default for zero‐sized unit types.
#[macro_export]
macro_rules! default_unit {
    ($unit:ident) => {
        impl Default for $unit {
            fn default() -> Self {
                $unit
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

default_unit!(Nanometer);
default_unit!(Micrometer);
default_unit!(Millimeter);
default_unit!(Meter);
default_unit!(Kilometer);
default_unit!(Centimeter);
default_unit!(LightYear);
default_unit!(Ångström);

impl Unit for Nanometer {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-9, offset: 0.0, symbol: "nm", name: "Nanometer" }
    }
}

impl Unit for Micrometer {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-6, offset: 0.0, symbol: "µm", name: "Micrometer" }
    }
}

impl Unit for Millimeter {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-3, offset: 0.0, symbol: "mm", name: "Millimeter" }
    }
}

impl Unit for Meter {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m", name: "Meter" }
    }
}

impl Unit for Kilometer {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1000.0, offset: 0.0, symbol: "km", name: "Kilometer" }
    }
}

impl Unit for Centimeter {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 0.01, offset: 0.0, symbol: "cm", name: "Centimeter" }
    }
}

impl Unit for LightYear {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        // 1 light year ≈ 9.461e15 meters
        UnitParameters { scale: 9.461e15, offset: 0.0, symbol: "ly", name: "LightYear" }
    }
}

impl Unit for Ångström {
    type Dimension = Length;
    fn parameters() -> UnitParameters {
        // 1 Ångström = 1e-10 meters
        UnitParameters { scale: 1.0e-10, offset: 0.0, symbol: "Å", name: "Ångström" }
    }
}

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

default_unit!(Nanosecond);
default_unit!(Microsecond);
default_unit!(Millisecond);
default_unit!(Second);
default_unit!(Minute);
default_unit!(Hour);
default_unit!(Day);
default_unit!(Year);
default_unit!(Century);

impl Unit for Nanosecond {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-9, offset: 0.0, symbol: "ns", name: "Nanosecond" }
    }
}

impl Unit for Microsecond {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-6, offset: 0.0, symbol: "µs", name: "Microsecond" }
    }
}

impl Unit for Millisecond {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-3, offset: 0.0, symbol: "ms", name: "Millisecond" }
    }
}

impl Unit for Second {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "s", name: "Second" }
    }
}

impl Unit for Minute {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 60.0, offset: 0.0, symbol: "min", name: "Minute" }
    }
}

impl Unit for Hour {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 3600.0, offset: 0.0, symbol: "h", name: "Hour" }
    }
}

impl Unit for Day {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 86400.0, offset: 0.0, symbol: "day", name: "Day" }
    }
}

impl Unit for Year {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        // using average year = 365.25 days
        UnitParameters { scale: 365.25 * 86400.0, offset: 0.0, symbol: "year", name: "Year" }
    }
}

impl Unit for Century {
    type Dimension = Time;
    fn parameters() -> UnitParameters {
        // 100 years = 100 * Year
        UnitParameters { scale: 100.0 * 365.25 * 86400.0, offset: 0.0, symbol: "century", name: "Century" }
    }
}

// ---------- Force Units ----------
pub struct Newton;
pub struct Lbf;

default_unit!(Newton);
default_unit!(Lbf);

impl Unit for Newton {
    type Dimension = Force;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "N", name: "Newton" }
    }
}

impl Unit for Lbf {
    type Dimension = Force;
    fn parameters() -> UnitParameters {
        // 1 lbf ≈ 4.44822 newtons
        UnitParameters { scale: 4.44822, offset: 0.0, symbol: "lbf", name: "Pound-force" }
    }
}

// ---------- Velocity Units ----------
pub struct MetersPerSecond;
pub struct KilometersPerHour;

default_unit!(MetersPerSecond);
default_unit!(KilometersPerHour);

impl Unit for MetersPerSecond {
    type Dimension = Velocity;
    fn parameters() -> UnitParameters {
        // Base velocity unit: m/s.
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m/s", name: "MetersPerSecond" }
    }
}

impl Unit for KilometersPerHour {
    type Dimension = Velocity;
    fn parameters() -> UnitParameters {
        // 1 km/h = 1000 m / 3600 s = 1/3.6 m/s
        UnitParameters { scale: 1.0 / 3.6, offset: 0.0, symbol: "km/h", name: "KilometersPerHour" }
    }
}

// ---------- Acceleration Unit ----------
pub struct MetersPerSecondSquared;

default_unit!(MetersPerSecondSquared);

impl Unit for MetersPerSecondSquared {
    type Dimension = Acceleration;
    fn parameters() -> UnitParameters {
        // Base acceleration unit: m/s².
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m/s²", name: "MeterPerSecondSquared" }
    }
}

// ---------- Temperature Units ----------
pub struct Celsius;
pub struct Kelvin;
pub struct Fahrenheit;

default_unit!(Celsius);
default_unit!(Kelvin);
default_unit!(Fahrenheit);

impl Unit for Kelvin {
    type Dimension = Temperature;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "K", name: "Kelvin" }
    }
}

impl Unit for Celsius {
    type Dimension = Temperature;
    fn parameters() -> UnitParameters {
        // 0°C = 273.15K
        UnitParameters { scale: 1.0, offset: 273.15, symbol: "°C", name: "Celsius" }
    }
}

impl Unit for Fahrenheit {
    type Dimension = Temperature;
    fn parameters() -> UnitParameters {
        // Kelvin = (Fahrenheit * 5/9) + 255.372222...
        UnitParameters { scale: 5.0/9.0, offset: 255.372222, symbol: "°F", name: "Fahrenheit" }
    }
}

// ---------- Energy Units ----------
pub struct Joule;
pub struct ElectronVolt;

default_unit!(Joule);
default_unit!(ElectronVolt);

impl Unit for Joule {
    type Dimension = Energy;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "J", name: "Joule" }
    }
}

impl Unit for ElectronVolt {
    type Dimension = Energy;
    fn parameters() -> UnitParameters {
        // 1 eV = 1.602176634e-19 Joules
        UnitParameters { scale: 1.602176634e-19, offset: 0.0, symbol: "eV", name: "ElectronVolt" }
    }
}

// ---------- Mass Units ----------
pub struct Kilogram;
pub struct Gram;
pub struct Pound;

default_unit!(Kilogram);
default_unit!(Gram);
default_unit!(Pound);

impl Unit for Kilogram {
    type Dimension = Mass;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "kg", name: "Kilogram" }
    }
}

impl Unit for Gram {
    type Dimension = Mass;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-3, offset: 0.0, symbol: "g", name: "Gram" }
    }
}

impl Unit for Pound {
    type Dimension = Mass;
    fn parameters() -> UnitParameters {
        // 1 lb ≈ 0.45359237 kg
        UnitParameters { scale: 0.45359237, offset: 0.0, symbol: "lb", name: "Pound" }
    }
}

// ---------- Lumious Intensity Units ----------
pub struct Candela;

default_unit!(Candela);

impl Unit for Candela {
    type Dimension = LuminousIntensity;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "cd", name: "Candela" }
    }
}

// ---------- Amount Units ----------

pub struct Mole;

default_unit!(Mole);

impl Unit for Mole {
    type Dimension = Amount;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "mol", name: "Mole" }
    }
}

// ---------- Pressure Units ----------
pub struct Pascal;
pub struct Bar;
pub struct Atmosphere;

default_unit!(Pascal);
default_unit!(Bar);
default_unit!(Atmosphere);

impl Unit for Pascal {
    type Dimension = Pressure;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "Pa", name: "Pascal" }
    }
}

impl Unit for Bar {
    type Dimension = Pressure;
    fn parameters() -> UnitParameters {
        // 1 bar = 100000 Pa
        UnitParameters { scale: 100000.0, offset: 0.0, symbol: "bar", name: "Bar" }
    }
}

impl Unit for Atmosphere {
    type Dimension = Pressure;
    fn parameters() -> UnitParameters {
        // 1 atm = 101325 Pa
        UnitParameters { scale: 101325.0, offset: 0.0, symbol: "atm", name: "Atmosphere" }
    }
}

// ---------- Power Units ----------
pub struct Watt;
pub struct Horsepower;

default_unit!(Watt);
default_unit!(Horsepower);

impl Unit for Watt {
    type Dimension = Power;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "W", name: "Watt" }
    }
}

impl Unit for Horsepower {
    type Dimension = Power;
    fn parameters() -> UnitParameters {
        // 1 hp ≈ 745.7 W
        UnitParameters { scale: 745.7, offset: 0.0, symbol: "hp", name: "Horsepower" }
    }
}

// ---------- Area Units ----------
pub struct SquareMeter;
default_unit!(SquareMeter);
impl Unit for SquareMeter {
    type Dimension = Area;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m²", name: "SquareMeter" }
    }
}

// ---------- Volume Units ----------
pub struct CubicMeter;
pub struct Liter;
pub struct Milliliter;
pub struct Gallon;

default_unit!(CubicMeter);
default_unit!(Liter);
default_unit!(Milliliter);
default_unit!(Gallon);

impl Unit for CubicMeter {
    type Dimension = Volume;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m³", name: "CubicMeter" }
    }
}

impl Unit for Liter {
    type Dimension = Volume;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-3, offset: 0.0, symbol: "L", name: "Liter" }
    }
}

impl Unit for Milliliter {
    type Dimension = Volume;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0e-6, offset: 0.0, symbol: "mL", name: "Milliliter" }
    }
}

impl Unit for Gallon {
    type Dimension = Volume;
    fn parameters() -> UnitParameters {
        // 1 gallon ≈ 3.78541 liters
        UnitParameters { scale: 3.78541, offset: 0.0, symbol: "gal", name: "Gallon" }
    }
}

// ---------- Frequency Units ----------
pub struct Hertz;
default_unit!(Hertz);
impl Unit for Hertz {
    type Dimension = Frequency;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "Hz", name: "Hertz" }
    }
}

// ---------- Electric Current Units ----------
pub struct Ampere;
default_unit!(Ampere);
impl Unit for Ampere {
    type Dimension = Current;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "A", name: "Ampere" }
    }
}

// -------------------------------------------------------------------
// SI Constants (using Scalar type)
// -------------------------------------------------------------------
use std::marker::PhantomData;

pub const C: Scalar<Velocity> = Scalar {
    data: [299_792_458.0],
    _phantom: PhantomData
};

pub const G: Scalar<Acceleration> = Scalar {
    data: [9.81],
    _phantom: PhantomData
};

pub const H: Scalar<dim_mul!(Energy,Time)> = Scalar {
    data: [6.62607015e-34],
    _phantom: PhantomData
};

pub const K: Scalar<Temperature> = Scalar {
    data: [1.380649e-23],
    _phantom: PhantomData
};

pub const E: Scalar<Charge> = Scalar {
    data: [1.602176634e-19],
    _phantom: PhantomData
};

pub const M_E: Scalar<Mass> = Scalar {
    data: [9.10938356e-31],
    _phantom: PhantomData
};

pub const M_P: Scalar<Mass> = Scalar {
    data: [1.6726219e-27],
    _phantom: PhantomData
};

pub const N_A: Scalar<Amount> = Scalar {
    data: [6.02214076e23],
    _phantom: PhantomData
};