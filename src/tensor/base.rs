use crate::tensor::scalar::Scalar;
use crate::units::Unit;
use std::marker::PhantomData;
use std::ops::*;

use crate::complex::c64;
use crate::tensor::element::*;

use crate::dimension::Dimension;

#[derive(Copy, Clone)]
pub struct Tensor<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize>
where
    [(); LAYERS * ROWS * COLS]:,
{
    data: [E; LAYERS * ROWS * COLS],
    pub _phantom: PhantomData<D>,
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
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
                E::from(U::from_base(((base_max - base_min) + base_min).weak_mul(rand::random::<f64>()).into()))
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Tensor {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn apply<F, EO: TensorElement, DO>(&self, f: F) -> Tensor<EO, DO, LAYERS, ROWS, COLS>
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
    pub fn init(
        f: impl Fn(usize, usize, usize) -> E,
    ) -> Tensor<E, D, LAYERS, ROWS, COLS>
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

    pub fn init_2d(
        f: impl Fn(usize, usize) -> E,
    ) -> Tensor<E, D, 1, ROWS, COLS>
    where
        [(); 1 * ROWS * COLS]:,
    {
        Tensor::<E, D, 1, ROWS, COLS>::init(|_, r, c| f(r, c))
    }


    pub fn combine<F, EO: TensorElement, DO>(&self, other: &Tensor<E, D, LAYERS, ROWS, COLS>, f: F) -> Tensor<EO, DO, LAYERS, ROWS, COLS>
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

    pub fn size(&self) -> usize {
        LAYERS * ROWS * COLS
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (LAYERS, ROWS, COLS)
    }

    pub fn layers(&self) -> usize {
        LAYERS
    }

    pub fn rows(&self) -> usize {
        ROWS
    }

    pub fn cols(&self) -> usize {
        COLS
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
        assert_eq!(LAYERS * ROWS * COLS, L * R * C, "Cannot reshape: sizes must match.");
        let mut new_data = [E::zero(); L * R * C];
        new_data.copy_from_slice(&self.data);

        Tensor {
            data: new_data,
            _phantom: PhantomData,
        }
    }

    pub fn flatten(&self) -> Tensor<E, D, 1, 1, { LAYERS * ROWS * COLS }>
    where
        [(); 1 * 1 * (LAYERS * ROWS * COLS)]:,
    {
        self.reshape::<1, 1, { LAYERS * ROWS * COLS }>()
    }
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    pub fn dtype(&self) -> &'static str {
        std::any::type_name::<E>()
    }

    pub fn cast<T: TensorElement>(&self) -> Tensor<T, D, LAYERS, ROWS, COLS>
    where
        T: TensorElement,
    {
        self.apply(|v| T::from(v.into()))
    }
}

pub type Vector<E: TensorElement, D, const N: usize> = Tensor<E, D, 1, N, 1>;

pub type Matrix<E: TensorElement, D, const N: usize, const M: usize> = Tensor<E, D, 1, N, M>;

pub type Vec2<E: TensorElement, D> = Vector<E, D, 2>;

pub type Vec3<E: TensorElement, D> = Vector<E, D, 3>;

pub type Vec4<E: TensorElement, D> = Vector<E, D, 4>;

pub type Mat2<E: TensorElement, D> = Matrix<E, D, 2, 2>;

pub type Mat3<E: TensorElement, D> = Matrix<E, D, 3, 3>;

pub type Mat4<E: TensorElement, D> = Matrix<E, D, 4, 4>;

impl<E: TensorElement, D> Vec2<E, D> {
    pub fn raw_tuple(&self) -> (E, E)
    where
        E: TensorElement,
    {
        (self.data[0], self.data[1])
    }

    pub fn raw_tuple_as<T: From<E>>(&self) -> (T, T)
    where
        E: TensorElement,
    {
        (T::from(self.data[0]), T::from(self.data[1]))
    }
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
    E: TensorElement,
{
    pub fn raw_vec(&self) -> Vec<E> {
        self.data.to_vec()
    }

    pub fn raw_vec_as<T: From<E> + TensorElement>(&self) -> Vec<T> {
        self.apply::<_, T, D>(|x| T::from(x)).data.to_vec()
    }
}

impl<E: TensorElement, D> Vec2<E, D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E, D> {
        Scalar::<E, D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<E, D> {
        Scalar::<E, D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement, D> Vec3<E, D>
where
    E: TensorElement,
{
    pub fn x(&self) -> Scalar<E, D> {
        Scalar::<E, D> {
            data: [self.data[0]],
            _phantom: PhantomData,
        }
    }

    pub fn y(&self) -> Scalar<E, D> {
        Scalar::<E, D> {
            data: [self.data[1]],
            _phantom: PhantomData,
        }
    }

    pub fn z(&self) -> Scalar<E, D> {
        Scalar::<E, D> {
            data: [self.data[2]],
            _phantom: PhantomData,
        }
    }
}

impl<E: TensorElement, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> PartialEq for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<E: TensorElement, D> PartialOrd for Tensor<E, D, 1, 1, 1>
where
    [(); 1]:,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data[0].partial_cmp(&other.data[0])
    }
}

impl<E: TensorElement + AddAssign, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> AddAssign for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }
}

impl<E: TensorElement + SubAssign, D, const LAYERS: usize, const ROWS: usize, const COLS: usize> SubAssign for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= *b;
        }
    }
}

