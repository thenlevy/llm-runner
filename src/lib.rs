mod distilbert;
mod gpt;
mod layers;

pub use distilbert::DistilBert;

#[derive(Debug)]
pub enum Error {
    InconsistentShape,
    InvalidData,
    Io(std::io::Error),
    DeserializationError(safetensors::SafeTensorError),
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(error: safetensors::SafeTensorError) -> Self {
        Error::DeserializationError(error)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn parse_distilbert() {
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../distilbert-base-uncased/model.safetensors");
        let bytes = std::fs::read(model_path).unwrap();
        let distilbert = crate::distilbert::DistilBert::try_from_bytes(&bytes).unwrap();
        assert_eq!(distilbert.d_model, 768);
        assert_eq!(distilbert.seq_len, 512);
        assert_eq!(distilbert.vocab_size, 30522);
        assert_eq!(distilbert.encoder.len(), 6);
    }
}
