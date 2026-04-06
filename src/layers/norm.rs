use crate::{Error, layers::Vector};

use safetensors::tensor::TensorView;

pub struct Norm {
    bias: Vector,
    weight: Vector,
    espilon: f32,
}

impl Norm {
    pub fn try_from_views(
        bias: TensorView<'_>,
        weights: TensorView<'_>,
        espilon: f32,
    ) -> Result<Self, Error> {
        let bias = Vector::try_from_view(bias, None)?;
        let len = bias.len();
        let weight = Vector::try_from_view(weights, Some(len))?;

        Ok(Self {
            bias,
            weight,
            espilon,
        })
    }

    pub fn shape(&self) -> usize {
        self.bias.len()
    }
}
