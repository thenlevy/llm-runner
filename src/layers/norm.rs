use crate::{Error, layers::Vector};

use {
    nalgebra::{DMatrix, DVector, DVectorViewMut},
    safetensors::tensor::TensorView,
};

pub struct Norm {
    bias: Vector,
    weight: Vector,
    epsilon: f32,
}

impl Norm {
    pub fn try_from_views(
        bias: TensorView<'_>,
        weights: TensorView<'_>,
        epsilon: f32,
    ) -> Result<Self, Error> {
        let bias = Vector::try_from_view(bias, None)?;
        let len = bias.len();
        let weight = Vector::try_from_view(weights, Some(len))?;

        Ok(Self {
            bias,
            weight,
            epsilon,
        })
    }

    pub fn shape(&self) -> usize {
        self.bias.len()
    }

    pub fn normalize_row(&self, row: &mut DVectorViewMut<f32>) -> Result<(), Error> {
        let n = row.len();
        if n != self.shape() {
            return Err(Error::InconsistentShape);
        }

        let mean = row.mean();
        let inv_std = 1.0 / (row.variance() + self.epsilon).sqrt();

        *row -= &DVector::from_element(n, mean);

        *row *= inv_std;

        row.component_mul_assign(&self.weight);

        *row += &*self.bias;

        Ok(())
    }

    pub fn normalize_rows(&self, rows: &mut DMatrix<f32>) -> Result<(), Error> {
        let (nrows, n_cols) = rows.shape();

        if n_cols != self.shape() {
            return Err(Error::InconsistentShape);
        }

        let mut buf = vec![0.0f32; n_cols];
        for i in 0..nrows {
            for j in 0..n_cols {
                buf[j] = rows[(i, j)];
            }
            let mut view = DVectorViewMut::from_slice(&mut buf, n_cols);
            self.normalize_row(&mut view)?;
            for j in 0..n_cols {
                rows[(i, j)] = buf[j];
            }
        }
        Ok(())
    }
}
