//! Structures for the GPT-2 model.

use crate::{
    Error,
    layers::{Attention, Embeddings, Ffn, Norm},
};

use nalgebra::DMatrix;

pub struct Gpt2 {
    pub embeddings: Embeddings,
    pub blocks: Vec<Block>,
    pub ln_f: Norm,
    pub d_model: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub n_heads: usize,
}

pub struct Block {
    pub ln_1: Norm,
    pub attention: Attention,
    pub ln_2: Norm,
    pub mlp: Ffn,
}

impl Gpt2 {
    pub fn evaluate(&self, input: &[u32]) -> Result<DMatrix<f32>, Error> {
        let mut x = self.embeddings.embed(input)?;

        for block in &self.blocks {
            let residual = x.clone();
            block.ln_1.normalize_rows(&mut x)?;
            let attn_out = block.attention.forward_multi_head_causal(x, self.n_heads)?;
            x = attn_out + residual;

            let residual = x.clone();
            block.ln_2.normalize_rows(&mut x)?;
            let mlp_out = block.mlp.forward(x)?;
            x = mlp_out + residual;
        }

        self.ln_f.normalize_rows(&mut x)?;

        let logits = &x * self.embeddings.words.transpose();
        Ok(logits)
    }
}
