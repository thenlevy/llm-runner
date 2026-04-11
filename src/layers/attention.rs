//! Attention layer

use crate::{
    Error,
    layers::{Matrix, Vector, add_bias_rows, linear},
};

use {nalgebra::DMatrix, safetensors::tensor::TensorView};

pub struct Attention {
    pub q_weights: Matrix,
    pub q_bias: Vector,
    pub k_weights: Matrix,
    pub k_bias: Vector,
    pub v_weights: Matrix,
    pub v_bias: Vector,
    pub out_weights: Matrix,
    pub out_bias: Vector,
}

pub struct AttentionViews<'a> {
    pub q_weights: TensorView<'a>,
    pub k_weights: TensorView<'a>,
    pub v_weights: TensorView<'a>,
    pub out_weights: TensorView<'a>,
    pub q_bias: TensorView<'a>,
    pub k_bias: TensorView<'a>,
    pub v_bias: TensorView<'a>,
    pub out_bias: TensorView<'a>,
}

pub struct FusedAttentionViews<'a> {
    pub c_attn_weight: TensorView<'a>,
    pub c_attn_bias: TensorView<'a>,
    pub c_proj_weight: TensorView<'a>,
    pub c_proj_bias: TensorView<'a>,
}

impl Attention {
    pub fn try_from_fused_views(
        views: FusedAttentionViews<'_>,
        d_model: usize,
    ) -> Result<Self, Error> {
        let fused = Matrix::try_from_view(views.c_attn_weight, [Some(d_model), Some(3 * d_model)])?;

        let bias_bytes = views.c_attn_bias.data();
        if bias_bytes.len() != 3 * d_model * 4 {
            return Err(Error::InvalidData);
        }

        // The weights W_q, W_k, W_v are stored in a single W_fused matix so that
        // [K, V, Q] = x @ W_fused.
        // Then, in GPT-2 implementation the attention scores are computed "Conv1D-style":
        // y = x @ W + b.
        // Since in our implementation we use a PyTorch-style linear layer
        //(y = x * Wᵀ + b), we need to transpose the weights here for consistency.
        let q_weights = {
            let block = fused.view((0, 0), (d_model, d_model));
            Matrix::from_dmatrix(block.transpose())
        };
        let k_weights = {
            let block = fused.view((0, d_model), (d_model, d_model));
            Matrix::from_dmatrix(block.transpose())
        };
        let v_weights = {
            let block = fused.view((0, 2 * d_model), (d_model, d_model));
            Matrix::from_dmatrix(block.transpose())
        };

        let q_bias = Vector::try_from_f32_le_bytes(&bias_bytes[0..d_model * 4], d_model)?;
        let k_bias =
            Vector::try_from_f32_le_bytes(&bias_bytes[d_model * 4..2 * d_model * 4], d_model)?;
        let v_bias =
            Vector::try_from_f32_le_bytes(&bias_bytes[2 * d_model * 4..3 * d_model * 4], d_model)?;

        let c_proj = Matrix::try_from_view(views.c_proj_weight, [Some(d_model), Some(d_model)])?;

        Ok(Self {
            q_weights,
            k_weights,
            v_weights,
            q_bias,
            k_bias,
            v_bias,
            out_weights: c_proj.transposed(),
            out_bias: Vector::try_from_view(views.c_proj_bias, Some(d_model))?,
        })
    }

    pub fn try_from_views(views: AttentionViews, d_model: usize) -> Result<Self, Error> {
        Ok(Self {
            q_bias: Vector::try_from_view(views.q_bias, Some(d_model))?,
            k_bias: Vector::try_from_view(views.k_bias, Some(d_model))?,
            v_bias: Vector::try_from_view(views.v_bias, Some(d_model))?,
            out_bias: Vector::try_from_view(views.out_bias, Some(d_model))?,
            q_weights: Matrix::try_from_view(views.q_weights, [Some(d_model), Some(d_model)])?,
            k_weights: Matrix::try_from_view(views.k_weights, [Some(d_model), Some(d_model)])?,
            v_weights: Matrix::try_from_view(views.v_weights, [Some(d_model), Some(d_model)])?,
            out_weights: Matrix::try_from_view(views.out_weights, [Some(d_model), Some(d_model)])?,
        })
    }

    pub fn forward_multi_headed(
        &self,
        x: DMatrix<f32>,
        n_heads: usize,
    ) -> Result<DMatrix<f32>, Error> {
        let (seq, d_model) = x.shape();
        if d_model != self.q_weights.shape()[1] {
            return Err(Error::InconsistentShape);
        }
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(Error::InconsistentShape);
        }
        let d_head = d_model / n_heads;
        let scale = 1.0 / (d_head as f32).sqrt();

        let q = linear(&x, &self.q_weights, &self.q_bias);
        let k = linear(&x, &self.k_weights, &self.k_bias);
        let v = linear(&x, &self.v_weights, &self.v_bias);

        let mut attended = DMatrix::zeros(seq, d_model);
        for h in 0..n_heads {
            let c0 = h * d_head;
            let qh = q.view((0, c0), (seq, d_head));
            let kh = k.view((0, c0), (seq, d_head));
            let vh = v.view((0, c0), (seq, d_head));

            let mut scores = &qh * &kh.transpose();
            scores.scale_mut(scale);
            softmax_rows(&mut scores);

            let ctx = &scores * &vh;
            attended.view_mut((0, c0), (seq, d_head)).copy_from(&ctx);
        }

        let mut out = attended * self.out_weights.transpose();
        add_bias_rows(&mut out, &self.out_bias);
        Ok(out)
    }
}

fn softmax_rows(mat: &mut DMatrix<f32>) {
    let (nrows, ncols) = mat.shape();
    let mut buf = vec![0.0f32; ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            buf[j] = mat[(i, j)];
        }
        let max_v = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = buf.iter().map(|&x| (x - max_v).exp()).sum();
        let scale = 1.0 / sum;
        for j in 0..ncols {
            mat[(i, j)] = (buf[j] - max_v).exp() * scale;
        }
    }
}
