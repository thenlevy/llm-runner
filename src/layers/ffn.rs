//! Feed-forward network layer

use crate::{
    Error,
    layers::{Matrix, Vector},
};

use safetensors::tensor::TensorView;

pub struct Ffn {
    pub linear_1: Matrix,
    pub linear_2: Matrix,
    pub bias_1: Vector,
    pub bias_2: Vector,
}

pub struct FfnViews<'a> {
    pub linear_1: TensorView<'a>,
    pub linear_2: TensorView<'a>,
    pub bias_1: TensorView<'a>,
    pub bias_2: TensorView<'a>,
}

impl Ffn {
    pub fn try_from_views(views: FfnViews, d_model: usize) -> Result<Self, Error> {
        let linear_1 = Matrix::try_from_view(views.linear_1, [None, Some(d_model)])?;
        let hidden_dimension = Some(linear_1.shape()[0]);

        Ok(Self {
            linear_1,
            linear_2: Matrix::try_from_view(views.linear_2, [Some(d_model), hidden_dimension])?,
            bias_1: Vector::try_from_view(views.bias_1, hidden_dimension)?,
            bias_2: Vector::try_from_view(views.bias_2, Some(d_model))?,
        })
    }
}
