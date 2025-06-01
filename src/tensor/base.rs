use crate::tensor::scalar::Scalar;
use crate::units::Unit;
use std::marker::PhantomData;
use std::{ops::*, usize};

use crate::tensor::element::*;

use crate::dimension::{MultiplyDimensions};

#[derive(Copy, Clone)]
pub struct Tensor<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    data: [E; LAYERS * ROWS * COLS],
    pub _phantom: PhantomData<D>,
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
    Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn new<U: Unit<Dimension = D>>(values: [E; LAYERS * ROWS * COLS]) -> Self {
        let data: [E; LAYERS * ROWS * COLS] = values
            .iter()
            .map(|&v| E::from(U::to_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            data,
            _phantom: PhantomData,
        }
    }

    pub const fn default(data: [E; LAYERS * ROWS * COLS]) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn zero() -> Self {
        let data: [E; LAYERS * ROWS * COLS] = [E::zero(); LAYERS * ROWS * COLS];

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn random<U: Unit<Dimension = D>>(min: E, max: E) -> Self {
        let base_min: E = E::from(U::to_base(min.into()));
        let base_max: E = E::from(U::to_base(max.into()));
        let data: [E; LAYERS * ROWS * COLS] = (0..LAYERS * ROWS * COLS)
            .map(|_| {
                E::from(U::from_base(
                    ((base_max - base_min) + base_min)
                        .weak_mul(rand::random::<f64>())
                        .into(),
                ))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn apply<F>(&self, f: F) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        F: Fn(E) -> E,
    {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| f(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    /// Apply a function to each element, allowing dimension changes (use with caution!)
    pub fn apply_with_dimension<F, EO: TensorElement, DO>(&self, f: F) -> Tensor<EO, DO, LAYERS, ROWS, COLS>
    where
        F: Fn(E) -> EO,
    {
        let data: [EO; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .map(|&v| f(v))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<DO>,
        }
    }

    // init a tensor with a function that takes indices
    pub fn init(f: impl Fn(usize, usize, usize) -> E) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        [(); LAYERS * ROWS * COLS]:,
    {
        let mut data = [E::zero(); LAYERS * ROWS * COLS];
        for l in 0..LAYERS {
            for r in 0..ROWS {
                for c in 0..COLS {
                    let idx = l * (ROWS * COLS) + r * COLS + c;
                    data[idx] = f(l, r, c);
                }
            }
        }
        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn init_2d(f: impl Fn(usize, usize) -> E) -> Tensor<E, D, 1, ROWS, COLS>
    where
        [(); 1 * ROWS * COLS]:,
    {
        Tensor::<E, D, 1, ROWS, COLS>::init(|_, r, c| f(r, c))
    }

    /// Combine two tensors element-wise with a function, preserving dimensions
    pub fn combine<F>(
        &self,
        other: &Tensor<E, D, LAYERS, ROWS, COLS>,
        f: F,
    ) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        F: Fn(E, E) -> E,
    {
        let data: [E; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&v1, &v2)| f(v1, v2))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    /// Combine two tensors element-wise with a function, allowing dimension changes (use with caution!)
    pub fn combine_with_dimension<F, EO: TensorElement, DO>(
        &self,
        other: &Tensor<E, D, LAYERS, ROWS, COLS>,
        f: F,
    ) -> Tensor<EO, DO, LAYERS, ROWS, COLS>
    where
        F: Fn(E, E) -> EO,
    {
        let data: [EO; LAYERS * ROWS * COLS] = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&v1, &v2)| f(v1, v2))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData::<DO>,
        }
    }

    pub fn get<S: Unit<Dimension = D>>(&self) -> [E; LAYERS * ROWS * COLS] {
        self.data
            .iter()
            .map(|&v| E::from(S::from_base(v.into())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn get_at(&self, layer: usize, row: usize, col: usize) -> Scalar<E, D> {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        Scalar::<E, D> {
            data: [self.data[idx]],
            _phantom: PhantomData,
        }
    }

    pub fn set_at(&mut self, layer: usize, row: usize, col: usize, value: Scalar<E, D>) {
        assert!(layer < LAYERS && row < ROWS && col < COLS);
        let idx = layer * (ROWS * COLS) + row * COLS + col;
        self.data[idx] = value.data[0];
    }

    pub fn data(&self) -> &[E] {
        &self.data
    }

    pub fn transpose(self) -> Tensor<E, D, LAYERS, COLS, ROWS>
    where
        [(); LAYERS * COLS * ROWS]:,
    {
        let mut transposed = [E::zero(); LAYERS * COLS * ROWS];
        for l in 0..LAYERS {
            for i in 0..ROWS {
                for j in 0..COLS {
                    let src_idx = l * (ROWS * COLS) + i * COLS + j;
                    let dst_idx = l * (COLS * ROWS) + j * ROWS + i;
                    transposed[dst_idx] = self.data[src_idx];
                }
            }
        }
        Tensor::<E, D, LAYERS, COLS, ROWS> {
            data: transposed,
            _phantom: PhantomData,
        }
    }

    pub fn reshape<const L: usize, const R: usize, const C: usize>(&self) -> Tensor<E, D, L, R, C>
    where
        [(); L * R * C]:,
    {
        assert_eq!(
            LAYERS * ROWS * COLS,
            L * R * C,
            "Cannot reshape: sizes must match."
        );
        let mut new_data = [E::zero(); L * R * C];
        new_data.copy_from_slice(&self.data);

        Tensor {
            data: new_data,
            _phantom: PhantomData,
        }
    }

    // A compile-time length version of `cut`.
    // The caller supplies the desired output length as a const generic.
    pub fn cut<const LEN: usize>(&self, start: usize) -> Tensor<E, D, 1, 1, LEN>
    where
        [(); 1 * 1 * LEN]:,
        [(); LAYERS * ROWS * COLS]:,
    {
        // Ensure the requested slice is within bounds.
        assert!(start + LEN <= LAYERS * ROWS * COLS);

        // SAFETY: The bounds check above guarantees that the slice has LEN elements.
        let mut cut_data = [E::zero(); LEN];
        cut_data.copy_from_slice(&self.data[start..start + LEN]);

        Tensor::<E, D, 1, 1, LEN>::from_array(cut_data)
    }

    /// Returns the elements selected by `mask` (given as another tensor) as a new 1-D tensor.
    /// An element in `self` is retained iff the corresponding element in `mask` has magnitude `> 0`.
    /// The number of retained elements **must** equal the const generic `LEN`.
    pub fn masked_slice<const LEN: usize, EM: TensorElement, DM>(
        &self,
        mask: &Tensor<EM, DM, LAYERS, ROWS, COLS>,
    ) -> Tensor<E, D, 1, 1, LEN>
    where
        [(); 1 * 1 * LEN]:,
        [(); LAYERS * ROWS * COLS]:,
    {
        // Collect values whose mask magnitude is non-zero.
        let filtered: Vec<E> = self
            .data
            .iter()
            .zip(mask.data().iter())
            .filter_map(|(&value, &m)| if m.mag() > 0.0 { Some(value) } else { None })
            .collect();

        assert_eq!(
            filtered.len(),
            LEN,
            "The number of active mask elements must equal the const generic LEN."
        );

        // Convert the Vec<E> into a fixed-size array `[E; LEN]`.
        let data: [E; LEN] = filtered
            .try_into()
            .expect("Length already validated to equal LEN");

        Tensor::<E, D, 1, 1, LEN>::from_array(data)
    }

    /// Reduce along the row dimension: for every layer/column pair all rows
    /// are aggregated into a single value using the commutative/associative
    /// operation `f` (e.g. sum, product, max, ...).
    ///
    /// Result shape: `LAYERS × 1 × COLS` (i.e. all rows collapsed).
    pub fn reduce<F, EO: TensorElement, DO>(&self, f: F) -> Tensor<EO, DO, LAYERS, 1, COLS>
    where
        [(); LAYERS * 1 * COLS]:,
        F: Fn(EO, E) -> EO,
    {
        let mut out = [EO::zero(); LAYERS * 1 * COLS];

        for l in 0..LAYERS {
            for c in 0..COLS {
                let mut acc: EO = EO::zero();
                for r in 0..ROWS {
                    let idx = l * (ROWS * COLS) + r * COLS + c;
                    acc = f(acc, self.data[idx]);
                }
                out[l * COLS + c] = acc;
            }
        }

        Tensor::<EO, DO, LAYERS, 1, COLS> {
            data: out,
            _phantom: PhantomData,
        }
    }

    /// Cuts the tensor into an array of column tensors.
    pub fn get_cols(&self) -> [Tensor<E, D, LAYERS, ROWS, 1>; COLS]
    where
        [(); LAYERS * ROWS * 1]:, // Constraint for the shape of each column tensor
        D: Copy, // D must be Copy for Tensor<...,D,...> to be Copy for array initialization.
    {
        // Initialize an array. Tensor::zero() is available.
        // Requires that Tensor<E, D, LAYERS, ROWS, 1> itself is Copy.
        // Since E is Copy (from TensorElement) and D is now bound to Copy,
        // and PhantomData<D> is Copy if D is Copy, Tensor is Copy.
        let mut result_array = [Tensor::<E, D, LAYERS, ROWS, 1>::zero(); COLS];

        for c_idx in 0..COLS {
            let mut column_data = [E::zero(); LAYERS * ROWS * 1];
            for l in 0..LAYERS {
                for r in 0..ROWS {
                    let src_idx = l * (ROWS * COLS) + r * COLS + c_idx;
                    let dest_idx_in_col_data = l * ROWS + r;
                    column_data[dest_idx_in_col_data] = self.data[src_idx];
                }
            }
            // When constructing a new tensor, it should generally take PhantomData for its dimension type D.
            // If self._phantom implies a specific instance of D that should be copied, D:Copy is essential.
            result_array[c_idx] = Tensor {
                data: column_data,
                _phantom: PhantomData, // Standard practice for new Tensor, D is type, not instance from self.
            };
        }
        result_array
    }

    pub fn get_rows(&self) -> [Tensor<E, D, LAYERS, 1, COLS>; ROWS]
    where
        [(); LAYERS * 1 * COLS]:,
        [(); LAYERS * COLS * ROWS]:,  // Constraint needed for transpose()
        [(); LAYERS * COLS * 1]:,     // Constraint needed for get_cols() on transposed tensor
        [(); LAYERS * 1 * COLS]:,     // Constraint needed for transposing each column
        D: Copy,
    {
        self.transpose().get_cols().map(|col| col.transpose())
    }
    

    /// Element-wise multiplication (Hadamard product) with another tensor
    /// Results in a tensor with multiplied dimensions
    pub fn hadamard<DO>(
        &self, 
        other: &Tensor<E, DO, LAYERS, ROWS, COLS>
    ) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS>
    where
        E: std::ops::Mul<Output = E>,
        D: MultiplyDimensions<DO>,
    {
        let mut result_data = [E::zero(); LAYERS * ROWS * COLS];
        for (i, (a, b)) in self.data.iter().zip(other.data().iter()).enumerate() {
            result_data[i] = *a * *b;
        }
        Tensor::<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, ROWS, COLS> {
            data: result_data,
            _phantom: PhantomData,
        }
    }

    /// Dot product operation for vectors (1×1×N tensors)
    /// Returns a scalar result (1×1×1 tensor) with multiplied dimensions
    pub fn dot<DO>(&self, other: &Tensor<E, DO, LAYERS, ROWS, COLS>) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, 1, 1, 1>
    where
        E: std::ops::Mul<Output = E> + std::ops::Add<Output = E>,
        D: MultiplyDimensions<DO>,
        [(); 1 * 1 * 1]:,
    {
        self.hadamard(other).sum()
    }

    /// Sum all elements in the tensor to a scalar
    pub fn sum(&self) -> Tensor<E, D, 1, 1, 1>
    where
        E: std::ops::Add<Output = E>,
        [(); 1 * 1 * 1]:,
    {
        let mut total = E::zero();
        for &val in self.data.iter() {
            total = total + val;
        }
        Tensor::<E, D, 1, 1, 1> {
            data: [total],
            _phantom: PhantomData,
        }
    }

    /// Outer product of a column vector and row vector
    /// col_vec: (LAYERS, M, 1) × row_vec: (LAYERS, 1, N) -> (LAYERS, M, N)
    pub fn outer_product<const M: usize, const N: usize, DO>(
        col_vec: &Tensor<E, D, LAYERS, M, 1>,
        row_vec: &Tensor<E, DO, LAYERS, 1, N>,
    ) -> Tensor<E, <D as MultiplyDimensions<DO>>::Output, LAYERS, M, N>
    where
        E: std::ops::Mul<Output = E>,
        D: MultiplyDimensions<DO>,
        [(); LAYERS * M * 1]:,
        [(); LAYERS * 1 * N]:,
        [(); LAYERS * M * N]:,
    {
        let broadcasted_col = col_vec.broadcast_to::<LAYERS, M, N>();
        let broadcasted_row = row_vec.broadcast_to::<LAYERS, M, N>();
        broadcasted_col.hadamard(&broadcasted_row)
    }

    /// Broadcast a tensor to a larger shape by repeating elements
    /// Source: (SL, SR, SC) -> Target: (TL, TR, TC)
    /// Each dimension in source must divide evenly into the corresponding target dimension
    pub fn broadcast<const SL: usize, const SR: usize, const SC: usize, 
                     const TL: usize, const TR: usize, const TC: usize>(
        source: &Tensor<E, D, SL, SR, SC>,
    ) -> Tensor<E, D, TL, TR, TC>
    where
        [(); SL * SR * SC]:,
        [(); TL * TR * TC]:,
    {
        assert!(TL % SL == 0, "Target layers must be divisible by source layers");
        assert!(TR % SR == 0, "Target rows must be divisible by source rows");
        assert!(TC % SC == 0, "Target cols must be divisible by source cols");
        
        Tensor::<E, D, TL, TR, TC>::init(|l, r, c| {
            let src_l = l % SL;
            let src_r = r % SR;
            let src_c = c % SC;
            let src_idx = src_l * (SR * SC) + src_r * SC + src_c;
            source.data()[src_idx]
        })
    }

    /// Broadcast this tensor to a larger shape by repeating elements
    /// Self: (LAYERS, ROWS, COLS) -> Target: (TL, TR, TC)
    /// Each dimension in self must divide evenly into the corresponding target dimension
    pub fn broadcast_to<const TL: usize, const TR: usize, const TC: usize>(
        &self,
    ) -> Tensor<E, D, TL, TR, TC>
    where
        [(); TL * TR * TC]:,
    {
        assert!(TL % LAYERS == 0, "Target layers must be divisible by source layers");
        assert!(TR % ROWS == 0, "Target rows must be divisible by source rows");
        assert!(TC % COLS == 0, "Target cols must be divisible by source cols");
        
        Tensor::<E, D, TL, TR, TC>::init(|l, r, c| {
            let src_l = l % LAYERS;
            let src_r = r % ROWS;
            let src_c = c % COLS;
            let src_idx = src_l * (ROWS * COLS) + src_r * COLS + src_c;
            self.data[src_idx]
        })
    }
}

impl<E: TensorElement, D, const COLS_USIZE: usize> Tensor<E, D, 1, 1, COLS_USIZE>
where
    [(); 1 * 1 * COLS_USIZE]:,
{
    /// Construct a `1×1×COLS_USIZE` tensor directly from the given data array.
    pub fn from_array(data: [E; COLS_USIZE]) -> Self {
        // Safety: `[E; COLS_USIZE]` and `[E; 1 * 1 * COLS_USIZE]` have the same size.
        let coerced: [E; 1 * 1 * COLS_USIZE] = unsafe { std::mem::transmute_copy(&data) };
        Self {
            data: coerced,
            _phantom: PhantomData,
        }
    }
}

// This impl block is for methods that don't depend on specific LAYERS, ROWS, COLS of the Tensor struct itself,
// but rather define their own dimensions, like a constructor or a static factory method.
impl<E: TensorElement, D> Tensor<E, D, 0, 0, 0> // Dummy consts for the impl, method will define actuals
{
    pub fn merge_cols<const LAYERS: usize, const ROWS: usize, const COL_DIM_SINGLE: usize, const NUM_COLS: usize>(
        columns: [Tensor<E, D, LAYERS, ROWS, COL_DIM_SINGLE>; NUM_COLS],
    ) -> Tensor<E, D, LAYERS, ROWS, NUM_COLS>
    where
        // E is already bound by the impl block
        // D is already generic in the impl block
        [(); LAYERS * ROWS * COL_DIM_SINGLE]:, // Constraint for individual column Tensors
        [(); LAYERS * ROWS * NUM_COLS]:,     // Constraint for the output Tensor data array
    {
        assert_eq!(COL_DIM_SINGLE, 1, "Input tensors must be single columns (COL_DIM_SINGLE must be 1)");

        let mut merged_data = [E::zero(); LAYERS * ROWS * NUM_COLS];

        for l in 0..LAYERS {
            for r in 0..ROWS {
                for col_idx in 0..NUM_COLS {
                    let val = columns[col_idx].data[l * ROWS + r];
                    let dst_idx = l * (ROWS * NUM_COLS) + r * NUM_COLS + col_idx;
                    merged_data[dst_idx] = val;
                }
            }
        }

        Tensor::<E, D, LAYERS, ROWS, NUM_COLS> {
            data: merged_data,
            _phantom: PhantomData,
        }
    }
    
    pub fn merge<const LAYERS: usize, const ROWS: usize, const COLS: usize, const NUM_TENSORS: usize>(
        tensors: [Tensor<E, D, 1, 1, 1>; NUM_TENSORS],
    ) -> Tensor<E, D, LAYERS, ROWS, COLS>
    where
        [(); LAYERS * ROWS * COLS]:,
    {
        let mut merged_data = [E::zero(); LAYERS * ROWS * COLS];
        for (i, tensor) in tensors.iter().enumerate() {
            for l in 0..LAYERS {
                for r in 0..ROWS {
                    for c in 0..COLS {
                        merged_data[l * (ROWS * COLS) + r * COLS + c] = tensor.data[i];
                    }
                }
            }
        }
        Tensor::<E, D, LAYERS, ROWS, COLS> {
            data: merged_data,
            _phantom: PhantomData,
        }
    }
}
