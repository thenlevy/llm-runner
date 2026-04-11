//! Build the GPT-2 model from the safetensors format (Hugging Face tensor names).

use {
    super::structs::*,
    crate::{
        Error,
        layers::{
            Attention, Embeddings, Ffn, FfnViews, FusedAttentionViews, Matrix, Norm,
        },
    },
};

use safetensors::SafeTensors;

impl Gpt2 {
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let safe_tensors = SafeTensors::deserialize(bytes)?;

        let epsilon = 1e-5;

        let d_model;
        let seq_len;
        let vocab_size;

        let embeddings;
        {
            let words = Matrix::try_from_view(safe_tensors.tensor("wte.weight")?, [None, None])?;
            vocab_size = words.shape()[0];
            d_model = words.shape()[1];

            let positions = Matrix::try_from_view(
                safe_tensors.tensor("wpe.weight")?,
                [None, Some(d_model)],
            )?;
            seq_len = positions.shape()[0];

            embeddings = Embeddings {
                norm: None,
                positions,
                words,
            };
        }

        let n_heads = d_model / 64;
        if n_heads == 0 || d_model % 64 != 0 {
            return Err(Error::InconsistentShape);
        }

        let mut blocks = vec![];
        {
            let mut layer = 0usize;
            loop {
                let prefix = format!("h.{layer}.");
                let path_c_attn = format!("{prefix}attn.c_attn.weight");
                if safe_tensors.tensor(&path_c_attn).is_err() {
                    break;
                }

                let attention = Attention::try_from_fused_views(
                    FusedAttentionViews {
                        c_attn_weight: safe_tensors.tensor(&path_c_attn)?,
                        c_attn_bias: safe_tensors.tensor(&format!("{prefix}attn.c_attn.bias"))?,
                        c_proj_weight: safe_tensors.tensor(&format!("{prefix}attn.c_proj.weight"))?,
                        c_proj_bias: safe_tensors.tensor(&format!("{prefix}attn.c_proj.bias"))?,
                    },
                    d_model,
                )?;

                let ln_1 = {
                    let bias = safe_tensors.tensor(&format!("{prefix}ln_1.bias"))?;
                    let weight = safe_tensors.tensor(&format!("{prefix}ln_1.weight"))?;
                    Norm::try_from_views(bias, weight, epsilon)?
                };

                let ln_2 = {
                    let bias = safe_tensors.tensor(&format!("{prefix}ln_2.bias"))?;
                    let weight = safe_tensors.tensor(&format!("{prefix}ln_2.weight"))?;
                    Norm::try_from_views(bias, weight, epsilon)?
                };

                let mlp = Ffn::try_from_transposed_views(
                    FfnViews {
                        linear_1: safe_tensors.tensor(&format!("{prefix}mlp.c_fc.weight"))?,
                        linear_2: safe_tensors.tensor(&format!("{prefix}mlp.c_proj.weight"))?,
                        bias_1: safe_tensors.tensor(&format!("{prefix}mlp.c_fc.bias"))?,
                        bias_2: safe_tensors.tensor(&format!("{prefix}mlp.c_proj.bias"))?,
                    },
                    d_model,
                )?;

                blocks.push(Block {
                    ln_1,
                    attention,
                    ln_2,
                    mlp,
                });

                layer += 1;
            }
        }

        let ln_f = {
            let bias = safe_tensors.tensor("ln_f.bias")?;
            let weight = safe_tensors.tensor("ln_f.weight")?;
            Norm::try_from_views(bias, weight, epsilon)?
        };

        Ok(Self {
            embeddings,
            blocks,
            ln_f,
            d_model,
            seq_len,
            vocab_size,
            n_heads,
        })
    }
}
