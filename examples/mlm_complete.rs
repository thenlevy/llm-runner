//! Append `[MASK].` to a prompt and print top MLM predictions for the masked position.
//!
//! Expects HuggingFace `distilbert-base-uncased` next to this crate (same layout as the library
//! test):
//! - `../distilbert-base-uncased/model.safetensors`
//! - `../distilbert-base-uncased/tokenizer.json`

use {llm_runner::DistilBert, tokenizers::Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_dir = root.join("../distilbert-base-uncased");
    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let user_prompt: String = std::env::args()
        .skip(1)
        .fold(String::new(), |acc, arg| acc + " " + arg.as_str());

    if user_prompt.is_empty() {
        return Err("No prompt provided".into());
    }

    let text = format!("{user_prompt} [MASK].");

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
    let mask_id = tokenizer
        .token_to_id("[MASK]")
        .ok_or("tokenizer has no [MASK] token")? as u32;

    // Here, the tokenizer will add the special tokens `[CLS]` and `[SEP]` at the beginning and end
    // of the text. These tokens indicates the begining and end of the sequence to process to the
    // DistilBERT model.
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| e.to_string())?;
    let input_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();

    let mask_pos = input_ids
        .iter()
        .position(|&id| id == mask_id)
        .ok_or("no [MASK] in tokenized input (check spelling / tokenizer)")?;

    let model_bytes = std::fs::read(model_path)?;
    let distilbert = DistilBert::try_from_bytes(&model_bytes).map_err(|e| format!("{e:?}"))?;

    if input_ids.len() > distilbert.seq_len {
        return Err(format!(
            "sequence length {} exceeds model max {}",
            input_ids.len(),
            distilbert.seq_len
        )
        .into());
    }

    let logits = distilbert
        .evaluate(&input_ids)
        .map_err(|e| format!("{e:?}"))?;
    let row = logits.row(mask_pos);

    let mut scored: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Prompt + mask: {text}");
    println!("Token IDs: {input_ids:?}");
    println!("MLM at position {mask_pos} — top predictions:");
    const TOP: usize = 5;
    for (rank, (tid, logit)) in scored.iter().take(TOP).enumerate() {
        let piece = tokenizer
            .id_to_token(*tid as u32)
            .unwrap_or_else(|| "?".to_string());
        println!(
            "  {}. id={} logit={:.3} token={piece:?}",
            rank + 1,
            tid,
            logit
        );
    }

    Ok(())
}
