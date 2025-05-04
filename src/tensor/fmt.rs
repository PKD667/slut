use crate::dimension::Dimension;
use crate::tensor::element::TensorElement;
use crate::tensor::Tensor;

impl<const L: i32, const M: i32, const T: i32, const Θ: i32, const I: i32, const N: i32, const J: i32>
    std::fmt::Display for Dimension<L, M, T, Θ, I, N, J>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut parts = Vec::new();
        if L != 0 {
            parts.push(format!("L^{}", L));
        }
        if M != 0 {
            parts.push(format!("M^{}", M));
        }
        if T != 0 {
            parts.push(format!("T^{}", T));
        }
        if Θ != 0 {
            parts.push(format!("Θ^{}", Θ));
        }
        if I != 0 {
            parts.push(format!("I^{}", I));
        }
        if N != 0 {
            parts.push(format!("N^{}", N));
        }
        if J != 0 {
            parts.push(format!("J^{}", J));
        }
        if parts.is_empty() {
            write!(f, "Dimensionless")
        } else {
            write!(f, "{}", parts.join(" * "))
        }
    }
}

impl<E: TensorElement, D: std::fmt::Display + Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> std::fmt::Display
    for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        let data: &[E] = self.data();

        writeln!(f, "Tensor [{}x{}x{}]: {}", LAYERS, ROWS, COLS, D::default())?;
        for l in 0..LAYERS {
            writeln!(f, "-- Layer {} --", l)?;
            for i in 0..ROWS {
                write!(f, "(")?;
                for j in 0..COLS {
                    let idx = l * (ROWS * COLS) + i * COLS + j;
                    write!(f, " {} ", data[idx]);
                }
                writeln!(f, ")")?;
            }
        }
        Ok(())
    }
}

impl<E: TensorElement, D: std::fmt::Debug + Default, const LAYERS: usize, const ROWS: usize, const COLS: usize> std::fmt::Debug
    for Tensor<E, D, LAYERS, ROWS, COLS>
where
    [(); LAYERS * ROWS * COLS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dimension", &D::default())
            .field("shape", &format!("{}x{}x{}", LAYERS, ROWS, COLS))
            .field("data", &self.data())
            .finish()
    }
}

