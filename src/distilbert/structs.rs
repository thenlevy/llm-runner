//! Structures for the DistilBERT model.

use crate::{
    Error,
    layers::{Attention, Ffn, Matrix, Norm},
};

use {nalgebra::DMatrix, safetensors::tensor::TensorView};

pub struct DistilBert {
    pub embeddings: Embeddings,
    pub transformers: Vec<Transformer>,
    pub d_model: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}

pub struct Transformer {
    pub attention: Attention,
    pub attention_norm: Norm,
    pub ffn: Ffn,
    pub output_norm: Norm,
}

pub struct Embeddings {
    pub norm: Norm,
    pub positions: Matrix,
    pub words: Matrix,
}
