//! General purpose vector layer

use crate::Error;

use {
    nalgebra::DVector,
    safetensors::tensor::TensorView,
    std::ops::{Deref, DerefMut},
};

pub struct Vector {
    inner: DVector<f32>,
}

impl Deref for Vector {
    type Target = DVector<f32>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Vector {
    pub fn try_from_view(
        view: TensorView<'_>,
        expected_length: Option<usize>,
    ) -> Result<Self, Error> {
        let [len] = view.shape() else {
            return Err(Error::InconsistentShape);
        };

        if expected_length.is_some_and(|l| l != *len) {
            return Err(Error::InconsistentShape);
        }

        if view.data().len() != len * 4 {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            inner: DVector::from_iterator(
                *len,
                view.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
        })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn try_from_f32_le_bytes(bytes: &[u8], expected_length: usize) -> Result<Self, Error> {
        if bytes.len() % 4 != 0 {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            inner: DVector::from_iterator(
                expected_length,
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
        })
    }
}
