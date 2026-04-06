//! Structures for the DistilBERT model.

use crate::{
    Error,
    layers::{Matrix, Norm},
};

use {nalgebra::DMatrix, safetensors::tensor::TensorView};

pub struct DistilBert {
    pub embeddings: Embeddings,
    pub d_model: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}

pub struct Embeddings {
    pub norm: Norm,
    pub positions: Matrix,
    pub words: Matrix,
}
