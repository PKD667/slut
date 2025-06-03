use crate::tensor::element::TensorElement;
use crate::tensor::base::Tensor;
use std::marker::PhantomData;

impl<E: TensorElement, D: Clone, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    /// Returns the total number of elements in the tensor
    pub fn size(&self) -> usize {
        LAYERS * ROWS * COLS
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

    /// Cast to another type
    pub fn cast<T: TensorElement>(&self) -> Tensor<T, D, LAYERS, ROWS, COLS>
    where
        T: TensorElement,
    {
        // For casting, we create a new tensor with converted data
        let current_data = self.realize();
        let new_data: [T; LAYERS * ROWS * COLS] = current_data.map(|v| T::from(v.into()));
        Tensor::default(new_data)
    }

    /// Converts to raw vector using data()
    pub fn raw_vec(&self) -> Vec<E> {
        self.data().to_vec()
    }

    /// Get raw data as vector of target type
    pub fn raw_vec_as<T: From<E> + TensorElement>(&self) -> Vec<T> {
        let current_data = self.realize();
        current_data.iter().map(|&x| T::from(x)).collect::<Vec<_>>()
    }
} 