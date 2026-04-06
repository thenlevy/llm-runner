//! Build the DistilBERT model from the safetensors format

use {
    super::structs::*,
    crate::{
        Error,
        layers::{Matrix, Norm},
    },
};

use safetensors::SafeTensors;

impl DistilBert {
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let safe_tensors = SafeTensors::deserialize(bytes)?;

        let mut path = vec!["distilbert"];
        let seq_len;
        let d_model;
        //let hidden_size;
        let vocab_size;

        let embedding;
        {
            path.push("embeddings");
            let norm;
            {
                path.push("LayerNorm");
                path.push("bias");
                let bias_view = safe_tensors.tensor(&path.join("."))?;
                path.pop();
                path.push("weight");
                let weight_view = safe_tensors.tensor(&path.join("."))?;
                path.pop();
                norm = Norm::try_from_views(bias_view, weight_view, 1e-12)?;
                d_model = norm.shape();
                path.pop();
            }
            path.push("position_embeddings.weight");
            let positions = Matrix::try_from_view(
                safe_tensors.tensor(&path.join("."))?,
                [None, Some(d_model)],
            )?;
            seq_len = positions.shape()[0];
            path.pop();
            path.push("word_embeddings.weight");
            let words = Matrix::try_from_view(
                safe_tensors.tensor(&path.join("."))?,
                [None, Some(d_model)],
            )?;
            vocab_size = words.shape()[0];

            path.pop();
            embedding = Embeddings {
                norm,
                positions,
                words,
            };
            path.pop();
        }

        Ok(Self {
            embeddings: embedding,
            d_model,
            seq_len,
            vocab_size,
        })
    }
}
