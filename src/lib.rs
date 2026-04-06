mod distilbert;
mod layers;

#[derive(Debug)]
pub enum Error {
    InconsistentShape,
    InvalidData,
    DeserializationError(safetensors::SafeTensorError),
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(error: safetensors::SafeTensorError) -> Self {
        Error::DeserializationError(error)
    }
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let bytes = include_bytes!("../../distilbert-base-uncased/model.safetensors");
        let distilbert = crate::distilbert::DistilBert::try_from_bytes(bytes).unwrap();
        assert_eq!(distilbert.d_model, 768);
        assert_eq!(distilbert.seq_len, 512);
        assert_eq!(distilbert.vocab_size, 30522);
    }
}
