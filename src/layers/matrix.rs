//! General purpose matrix layer

use crate::Error;

use {
    nalgebra::DMatrix,
    safetensors::tensor::TensorView,
    std::ops::{Deref, DerefMut},
};

pub struct Matrix {
    inner: DMatrix<f32>,
}

impl Deref for Matrix {
    type Target = DMatrix<f32>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Matrix {
    pub fn try_from_view(
        view: TensorView<'_>,
        expected_shape: [Option<usize>; 2],
    ) -> Result<Self, Error> {
        let [rows, cols] = view.shape() else {
            return Err(Error::InconsistentShape);
        };

        if expected_shape[0].is_some_and(|r| *rows != r) {
            return Err(Error::InconsistentShape);
        }

        if expected_shape[1].is_some_and(|c| *cols != c) {
            return Err(Error::InconsistentShape);
        }

        if view.data().len() != rows * cols * 4 {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            inner: DMatrix::from_row_iterator(
                *rows,
                *cols,
                view.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
        })
    }

    pub fn try_from_bytes(bytes: &[u8], shape: [usize; 2]) -> Result<Self, Error> {
        if bytes.len() != shape[0] * shape[1] * 4 {
            eprintln!(
                "Invalid data length: {} != {}",
                bytes.len(),
                shape[0] * shape[1] * 4
            );
            return Err(Error::InvalidData);
        }

        Ok(Self {
            inner: DMatrix::from_row_iterator(
                shape[0],
                shape[1],
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.inner.nrows(), self.inner.ncols()]
    }

    pub(crate) fn from_dmatrix(inner: DMatrix<f32>) -> Self {
        Self { inner }
    }

    pub fn transposed(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }
}
