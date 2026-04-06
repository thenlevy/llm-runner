use crate::Error;

use {nalgebra::DVector, safetensors::tensor::TensorView};

pub struct Norm {
    bias: DVector<f32>,
    weight: DVector<f32>,
    espilon: f32,
}

impl Norm {
    pub fn try_from_views(
        bias: TensorView<'_>,
        weights: TensorView<'_>,
        espilon: f32,
    ) -> Result<Self, Error> {
        let ([bias_dim], [weights_dim]) = (bias.shape(), weights.shape()) else {
            return Err(Error::InconsistentShape);
        };

        if bias_dim != weights_dim {
            return Err(Error::InconsistentShape);
        }

        if bias.data().len() != bias_dim * 4 {
            return Err(Error::InvalidData);
        }

        if weights.data().len() != weights_dim * 4 {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            bias: DVector::from_iterator(
                *bias_dim,
                bias.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
            weight: DVector::from_iterator(
                *weights_dim,
                weights
                    .data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
            espilon,
        })
    }

    pub fn shape(&self) -> usize {
        self.bias.len()
    }
}
