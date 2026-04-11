# LLM Runner, a library for running LLMs in Rust

This library provides an interface to parse models from the Hugging Face `safetensors` format and evaluate them.

## Usage and supported models

The supported models are:

- `distilbert-base-uncased` (required for `examples/mlm_complete.rs`)
- `gpt2-medium` (required for `examples/gpt2_complete.rs`)

To run the examples, you need to have the models in the same directory as the library.
The models can be downloaded from Hugging Face:

1. Install `git-lfs` if you do not have it already: `git lfs install`
2. `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{model_name}`
3. `cd {model_name} && git lfs pull -I *.safetensors`
