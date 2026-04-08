mod attention;
mod embeddings;
mod ffn;
mod matrix;
mod norm;
mod vector;

pub use {
    attention::{Attention, AttentionViews},
    embeddings::Embeddings,
    ffn::{Ffn, FfnViews},
    matrix::Matrix,
    norm::Norm,
    vector::Vector,
};

use nalgebra::{DMatrix, DVector};

pub(crate) fn linear(x: &DMatrix<f32>, w: &Matrix, b: &Vector) -> DMatrix<f32> {
    let mut y = x * w.transpose();
    add_bias_rows(&mut y, b);
    y
}

pub(crate) fn add_bias_rows(m: &mut DMatrix<f32>, bias: &DVector<f32>) {
    for (mut col, &b) in m.column_iter_mut().zip(bias.iter()) {
        for x in col.iter_mut() {
            *x += b;
        }
    }
}

pub(crate) fn apply_gelu(x: &mut DMatrix<f32>) {
    for x in x.iter_mut() {
        let u = std::f32::consts::FRAC_2_PI.sqrt() * (*x + 0.044715 * x.powi(3));
        *x = 0.5 * *x * (1.0 + u.tanh())
    }
}
