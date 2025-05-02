use crate::tensor::Tensor;
use crate::tensor::element::TensorElement;
use std::ops::{Add, Mul};

use crate::dimension::*;
use crate::dless;


// Implement generic tensor functions
// we want to be able to define closures 
// that can be used to apply function that take and return a tensor

// a tensor function must respect the dimensional rules
// It can only be composed of tensor Ops