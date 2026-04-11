//! Autoregressive completion using GPT-2.
//!
//! Expected files (same layout as `mlm_complete.rs`):
//! - `../gpt2-medium/model.safetensors` (override with `GPT2_MODEL_DIR`)
//! - `../gpt2-medium/tokenizer.json`
//!
//! ```text
//! MAX_NEW_TOKENS=64 SEED=42 cargo run --example gpt2_complete -- "The planet of the Solar System are Mercury,"
//! ```

use {
    llm_runner::Gpt2,
    nalgebra::DMatrix,
    rand::{Rng, SeedableRng, rngs::StdRng},
    std::io::Write,
    tokenizers::Tokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_dir = std::env::var("GPT2_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| root.join("../gpt2-medium"));
    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let user_prompt: String = std::env::args().skip(1).fold(String::new(), |acc, arg| {
        if acc.is_empty() {
            arg
        } else {
            acc + " " + arg.as_str()
        }
    });

    if user_prompt.is_empty() {
        return Err("usage: gpt2_complete <prompt words...>\n\
             env: GPT2_MODEL_DIR, MAX_NEW_TOKENS (default 32), TEMPERATURE (default 1.0), SEED (optional)"
            .into());
    }

    let max_new_tokens: usize = std::env::var("MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let mut rng = rng_from_env();

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
    let encoding = tokenizer
        .encode(user_prompt.as_str(), false)
        .map_err(|e| e.to_string())?;
    let mut ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();

    let model_bytes = std::fs::read(&model_path)?;
    let gpt2 = Gpt2::try_from_bytes(&model_bytes).map_err(|e| format!("{e:?}"))?;

    if ids.is_empty() {
        return Err("empty tokenization".into());
    }

    print!("{user_prompt}");
    std::io::stdout().flush()?;

    for _ in 0..max_new_tokens {
        if ids.len() >= gpt2.seq_len {
            return Err(format!(
                "sequence length {} reached model max {}",
                ids.len(),
                gpt2.seq_len
            )
            .into());
        }

        let logits = gpt2.evaluate(&ids).map_err(|e| format!("{e:?}"))?;
        let next_id = sample_last(&logits, &mut rng);
        ids.push(next_id);

        let piece = tokenizer
            .decode(&[next_id], true)
            .map_err(|e| e.to_string())?;
        print!("{piece}");
        std::io::stdout().flush()?;
    }

    println!();

    Ok(())
}

fn rng_from_env() -> StdRng {
    match std::env::var("SEED").map(|s| s.parse().ok()) {
        Ok(Some(s)) => StdRng::seed_from_u64(s),
        Ok(None) => {
            eprintln!("provided SEED is not a valid u64, using entropy");
            StdRng::from_entropy()
        }
        Err(_) => StdRng::from_entropy(),
    }
}

fn sample_last(logits: &DMatrix<f32>, rng: &mut impl Rng) -> u32 {
    let row = logits.nrows().saturating_sub(1);
    let cols = logits.ncols();

    let mut buf: Vec<f32> = (0..cols).map(|i| logits[(row, i)]).collect();

    let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    buf.iter_mut().for_each(|x| *x = (*x - max).exp());

    let sum: f32 = buf.iter().sum();

    let u = rng.gen_range(0.0f32..1.0f32) * sum;
    let mut acc = 0.0f32;
    for (i, x) in buf.iter().enumerate() {
        acc += *x;
        if u < acc {
            return i as u32;
        }
    }
    unreachable!()
}
