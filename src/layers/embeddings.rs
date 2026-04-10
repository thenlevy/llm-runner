//! Embeddings layer

use crate::{
    Error,
    layers::{Matrix, Norm},
};

use nalgebra::DMatrix;

pub struct Embeddings {
    pub norm: Norm,
    pub positions: Matrix,
    pub words: Matrix,
}

impl Embeddings {
    pub fn embed(&self, input: &[u32]) -> Result<DMatrix<f32>, Error> {
        let [vocab_size, d_model] = self.words.shape();

        let mut embeddings = DMatrix::zeros(input.len(), d_model);

        for (i, token) in input.iter().enumerate() {
            let t_id = *token as usize;
            if t_id >= vocab_size {
                return Err(Error::InconsistentShape);
            }

            embeddings
                .row_mut(i)
                .copy_from(&(self.words.row(t_id) + self.positions.row(i)));
        }

        self.norm.normalize_rows(&mut embeddings)?;

        Ok(embeddings)
    }
}
