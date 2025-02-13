pub type STYPE = f32;

use crate::dimension::*;

// define units for our dimensions
// units will be used as interfaces to our dimensions
#[derive(Clone, Copy)]
pub struct UnitParameters {
    scale: STYPE,
    offset: STYPE,
    symbol: &'static str,
    name: &'static str,
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


// ---------- Length Units ----------

// Existing length units
pub struct Meter;
pub struct Kilometer;
pub struct Centimeter;

// New length units
pub struct Nanometer;
pub struct Micrometer;
pub struct Millimeter;
pub struct LightYear;

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

// ---------- Velocity Unit ----------

pub struct MetersPerSecond;
pub struct KilometersPerHour;

impl Unit for MetersPerSecond {
    // Assuming that the Velocity dimension is defined (derived as Length/Time)
    type Dimension = Velocity;
    fn parameters() -> UnitParameters {
        // Base velocity unit: m/s.
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m/s", name: "MetersPerSecond" }
    }
}

impl Unit for KilometersPerHour {
    type Dimension = Velocity;
    fn parameters() -> UnitParameters {
        // Conversion: 1 km/h = 1000 m / 3600 s = 1/3.6 m/s
        UnitParameters { scale: 1.0 / 3.6, offset: 0.0, symbol: "km/h", name: "KilometersPerHour" }
    }
}



// ---------- Acceleration Unit ----------

pub struct MetersPerSecondSquared;

impl Unit for MetersPerSecondSquared {
    // Assuming that the Acceleration dimension is defined (derived as Length/Time²)
    type Dimension = Acceleration;
    fn parameters() -> UnitParameters {
        // Base acceleration unit: m/s².
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m/s²", name: "MeterPerSecondSquared" }
    }
}

// ---------- Additional Example: Temperature Units ----------

pub struct Celsius;
pub struct Kelvin;
pub struct Fahrenheit;

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
        // Conversion: (F - 32) * 5/9 + 273.15 = Kelvin
        // Represent as scale and offset such that Kelvin = scale * Fahrenheit + offset.
        // scale = 5/9, offset = 255.372222...
        UnitParameters { scale: 5.0/9.0, offset: 255.372222, symbol: "°F", name: "Fahrenheit" }
    }
}

// ---------- Additional Example: Energy Units ----------
pub struct Joule;
pub struct ElectronVolt;

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

// ---------- Additional Example: Mass Units ----------
pub struct Kilogram;
pub struct Gram;
pub struct Pound;

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

// ---------- Additional Example: Pressure Units ----------

pub struct Pascal;
pub struct Bar;
pub struct Atmosphere;

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

// ---------- Additional Example: Power Units ----------

pub struct Watt;
pub struct Horsepower;

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

// ---------- Additional Example: Area Units ----------
pub struct SquareMeter;

impl Unit for SquareMeter {
    type Dimension = Area;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "m²", name: "SquareMeter" }
    }
}

// ---------- Additional Example: Volume Units ----------

pub struct CubicMeter;
pub struct Liter;
pub struct Milliliter;
pub struct Gallon;

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

// ---------- Additional Example: Frequency Units ----------

pub struct Hertz;

impl Unit for Hertz {
    type Dimension = Frequency;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "Hz", name: "Hertz" }
    }
}

// ---------- Additional Example: Electric Current Units ----------

pub struct Ampere;

impl Unit for Ampere {
    type Dimension = Current;
    fn parameters() -> UnitParameters {
        UnitParameters { scale: 1.0, offset: 0.0, symbol: "A", name: "Ampere" }
    }
}


