use crate::tensor::element::TensorElement;
use crate::tensor::base::Tensor;
use std::marker::PhantomData;

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Returns the total number of elements in the tensor
    pub fn size(&self) -> usize {
        LAYERS * ROWS * COLS
    }

    /// Returns the shape as (layers, rows, cols)
    pub fn shape(&self) -> (usize, usize, usize) {
        (LAYERS, ROWS, COLS)
    }

    /// Returns the number of layers
    pub fn layers(&self) -> usize {
        LAYERS
    }

    /// Returns the number of rows
    pub fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns
    pub fn cols(&self) -> usize {
        COLS
    }

    /// Returns the data type name as a string
    pub fn dtype(&self) -> &'static str {
        std::any::type_name::<E>()
    }

    /// Flattens the tensor to a 1D tensor using reshape
    pub fn flatten(&self) -> Tensor<E, D, 1, 1, { LAYERS * ROWS * COLS }>
    where
        [(); 1 * 1 * (LAYERS * ROWS * COLS)]:,
    {
        self.reshape::<1, 1, { LAYERS * ROWS * COLS }>()
    }

    /// Casts the tensor to a different element type using apply
    pub fn cast<T: TensorElement>(&self) -> Tensor<T, D, LAYERS, ROWS, COLS>
    where
        T: TensorElement,
    {
        self.apply_with_dimension(|v| T::from(v.into()))
    }

    /// Converts to raw vector using data()
    pub fn raw_vec(&self) -> Vec<E> {
        self.data().to_vec()
    }

    /// Converts to raw vector with type conversion using apply
    pub fn raw_vec_as<T: From<E> + TensorElement>(&self) -> Vec<T> {
        self.apply_with_dimension::<_, T, D>(|x| T::from(x)).data().to_vec()
    }
} 