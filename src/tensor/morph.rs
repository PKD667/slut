// define a morph struct/trait
// It allows applying a numerical Op to a TensorElement
// with an internalized dimension transform

// Use necessary dimension components
use crate::dimension::{Dimension, DimTransform, DimMultiply, DimInvert, MultiplyDimensions, InvertDimension, SqrtDimension}; // Added SqrtDimension
use crate::dimension::{ConstAdd, ConstNeg, ConstCheck}; // Import helpers if needed for new transforms

use crate::tensor::element::TensorElement;
use std::marker::PhantomData;

// Morph represents a unary operation with its dimension transformation
pub struct Morph<T, F, D, DT>
where
    T: TensorElement,
    F: Fn(T) -> T,
    DT: DimTransform<D>,
    // D is the source dimension type, DT::Output will be the target
{
    // No tensor field needed here, Morph represents the operation itself
    pub func: F,
    pub dim_transform: PhantomData<DT>, // Store the transformer type, not an instance
    // marker for the unused type parameter D (source dimension)
    phantom_d: PhantomData<D>,
    // marker for the unused type parameter T (tensor element)
    phantom_t: PhantomData<T>,
}

// Example of how you might construct or use Morph (adjust as needed)
impl<T, F, D, DT> Morph<T, F, D, DT>
where
    T: TensorElement,
    F: Fn(T) -> T,
    DT: DimTransform<D>,
{
    // Constructor now only takes the function
    pub fn new(func: F) -> Self {
        Morph {
            func,
            dim_transform: PhantomData,
            phantom_d: PhantomData,
            phantom_t: PhantomData,
        }
    }

    // Apply the function to a single element
    pub fn apply_to_element(&self, element: T) -> T {
        (self.func)(element)
    }

    // Function to get the output dimension type marker
    pub fn output_dimension_type(&self) -> PhantomData<DT::Output> {
        PhantomData
    }
}

// --- Unary Operation Dimension Transformations ---

// Identity transform (Dimension remains unchanged)
pub struct DimIdentity;
impl<D> DimTransform<D> for DimIdentity {
    type Output = D;
    fn name() -> &'static str { "identity transform" }
}

// Square Root transform (Requires SqrtDimension trait)
pub struct DimSqrt;
impl<D> DimTransform<D> for DimSqrt
where
    D: SqrtDimension,
{
    type Output = <D as SqrtDimension>::Output;
    fn name() -> &'static str { "sqrt transform" }
}

// --- Binary Operation Dimension Transformations ---

/// Trait for defining how dimensions combine in a binary operation.
pub trait DimCombineTransform<Source1, Source2> {
    type Output;
    fn name() -> &'static str { "generic combine transform" }
}

// --- Implementations for DimCombineTransform ---

// 1. Multiplication
// We can reuse DimMultiply struct for the transformation logic.
// D1 * D2 -> Output
impl<D1, D2> DimCombineTransform<D1, D2> for DimMultiply<D2>
where
    // We rely on the existing MultiplyDimensions trait which uses DimTransform internally
    D1: MultiplyDimensions<D2>,
{
    type Output = <D1 as MultiplyDimensions<D2>>::Output;
    fn name() -> &'static str { "multiply combine" }
}

// 2. Addition / Subtraction
// Requires dimensions to be the same. Output dimension is also the same.
// Used for actual Tensor Add/Sub operations.
pub struct DimAdd; // Marker struct for Add/Sub transform
impl<D> DimCombineTransform<D, D> for DimAdd
// No complex bounds needed, just requires D1 == D2
{
    type Output = D; // Output dimension is the same as input
    fn name() -> &'static str { "add/subtract combine" }
}

// 3. Require Same Dimension (for Comparisons, Boolean Logic, etc.)
// Requires dimensions to be the same. Output dimension is also the same.
pub struct DimSame; // Marker struct for requiring identical dimensions
impl<D> DimCombineTransform<D, D> for DimSame
// No complex bounds needed, just requires D1 == D2
{
    type Output = D; // Output dimension is the same as input
    fn name() -> &'static str { "require same dimension combine" }
}

// 4. Division (Renumbered from 3)
// D1 / D2 is equivalent to D1 * Invert(D2)
pub struct DimDivide<D2>(PhantomData<D2>); // Marker struct for Division transform
impl<D1, D2> DimCombineTransform<D1, D2> for DimDivide<D2>
where
    D2: InvertDimension,
    D1: MultiplyDimensions<<D2 as InvertDimension>::Output>, // D1 * Invert(D2)
{
    type Output = <D1 as MultiplyDimensions<<D2 as InvertDimension>::Output>>::Output;
    fn name() -> &'static str { "divide combine" }
}

