//! Attention layer

use crate::{
    Error,
    layers::{Matrix, Vector},
};

use safetensors::tensor::TensorView;

pub struct Attention {
    pub q_weights: Matrix,
    pub q_bias: Vector,
    pub k_weights: Matrix,
    pub k_bias: Vector,
    pub v_weights: Matrix,
    pub v_bias: Vector,
}

pub struct AttentionViews<'a> {
    pub q_weights: TensorView<'a>,
    pub k_weights: TensorView<'a>,
    pub v_weights: TensorView<'a>,
    pub q_bias: TensorView<'a>,
    pub k_bias: TensorView<'a>,
    pub v_bias: TensorView<'a>,
}

impl Attention {
    pub fn try_from_views(views: AttentionViews, d_model: usize) -> Result<Self, Error> {
        Ok(Self {
            q_bias: Vector::try_from_view(views.q_bias, Some(d_model))?,
            k_bias: Vector::try_from_view(views.k_bias, Some(d_model))?,
            v_bias: Vector::try_from_view(views.v_bias, Some(d_model))?,
            q_weights: Matrix::try_from_view(views.q_weights, [Some(d_model), Some(d_model)])?,
            k_weights: Matrix::try_from_view(views.k_weights, [Some(d_model), Some(d_model)])?,
            v_weights: Matrix::try_from_view(views.v_weights, [Some(d_model), Some(d_model)])?,
        })
    }
}
