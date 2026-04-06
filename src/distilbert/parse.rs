//! Build the DistilBERT model from the safetensors format

use {
    super::structs::*,
    crate::{
        Error,
        layers::{Attention, AttentionViews, Ffn, FfnViews, Matrix, Norm},
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

        let mut transformers = vec![];
        {
            path.push("transformer");
            path.push("layer");
            let mut layer = 0;
            loop {
                path.push(layer.to_string().leak());
                let attention;
                {
                    path.push("attention");
                    path.push("q_lin.weight");
                    let Ok(q_weights_view) = safe_tensors.tensor(&path.join(".")) else {
                        break;
                    };
                    path.pop();

                    path.push("q_lin.bias");
                    let q_bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("k_lin.weight");
                    let k_weights_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("k_lin.bias");
                    let k_bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("v_lin.weight");
                    let v_weights_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("v_lin.bias");
                    let v_bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    attention = Attention::try_from_views(
                        AttentionViews {
                            q_weights: q_weights_view,
                            q_bias: q_bias_view,
                            k_weights: k_weights_view,
                            k_bias: k_bias_view,
                            v_weights: v_weights_view,
                            v_bias: v_bias_view,
                        },
                        d_model,
                    )?;
                    path.pop();
                }
                let ffn;
                {
                    path.push("ffn");
                    path.push("lin1.weight");
                    let lin_1_weights_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("lin1.bias");
                    let lin_1_bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("lin2.weight");
                    let lin_2_weights_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("lin2.bias");
                    let lin_2_bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    ffn = Ffn::try_from_views(
                        FfnViews {
                            linear_1: lin_1_weights_view,
                            bias_1: lin_1_bias_view,
                            linear_2: lin_2_weights_view,
                            bias_2: lin_2_bias_view,
                        },
                        d_model,
                    )?;
                    path.pop();
                }
                let attention_norm;
                {
                    path.push("sa_layer_norm");

                    path.push("bias");
                    let bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("weight");
                    let weight_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    attention_norm = Norm::try_from_views(bias_view, weight_view, 1e-12)?;
                    path.pop();
                }
                let output_norm;
                {
                    path.push("output_layer_norm");

                    path.push("bias");
                    let bias_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    path.push("weight");
                    let weight_view = safe_tensors.tensor(&path.join("."))?;
                    path.pop();

                    output_norm = Norm::try_from_views(bias_view, weight_view, 1e-12)?;
                    path.pop();
                }
                transformers.push(Transformer {
                    attention,
                    attention_norm,
                    ffn,
                    output_norm,
                });
                path.pop();
                layer += 1;
            }
        }

        Ok(Self {
            embeddings: embedding,
            transformers,
            d_model,
            seq_len,
            vocab_size,
        })
    }
}
