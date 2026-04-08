//! Structures for the DistilBERT model.

use crate::{
    Error,
    layers::{Attention, Embeddings, Ffn, Matrix, Norm, Vector, apply_gelu, linear},
};

use nalgebra::DMatrix;

pub struct DistilBert {
    pub embeddings: Embeddings,
    pub encoder: Vec<Stack>,
    pub vocab_layer: VocabLayer,
    pub d_model: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    /// Must match the checkpoint (e.g. 12 for `distilbert-base-uncased`).
    pub n_heads: usize,
}

pub struct Stack {
    pub attention: Attention,
    pub attention_norm: Norm,
    pub ffn: Ffn,
    pub output_norm: Norm,
}

pub struct VocabLayer {
    pub norm: Norm,
    pub transform: Matrix,
    pub transform_bias: Vector,
    pub project: Matrix,
    pub project_bias: Vector,
}

impl DistilBert {
    pub fn evaluate(&self, input: &[u32]) -> Result<DMatrix<f32>, Error> {
        let mut output = self.embeddings.embed(input)?;
        for stack in &self.encoder {
            // TransformerBlock: sa_layer_norm(attn(x) + x), then output_layer_norm(ffn(h) + h)
            let residual = output.clone();
            let attn_out = stack
                .attention
                .forward_multi_headed(output, self.n_heads)?;
            let mut h = attn_out + residual;
            stack.attention_norm.normalize_rows(&mut h)?;

            let residual = h.clone();
            let ffn_out = stack.ffn.forward(h)?;
            let mut h = ffn_out + residual;
            stack.output_norm.normalize_rows(&mut h)?;
            output = h;
        }

        // DistilBertForMaskedLM: transform → activation → vocab_layer_norm → projector
        let mut h = linear(
            &output,
            &self.vocab_layer.transform,
            &self.vocab_layer.transform_bias,
        );
        apply_gelu(&mut h);
        self.vocab_layer.norm.normalize_rows(&mut h)?;
        let logits = linear(
            &h,
            &self.vocab_layer.project,
            &self.vocab_layer.project_bias,
        );

        Ok(logits)
    }
}
