use safetensors::{Dtype, slice::TensorIndexer};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::loader::{Error, Loader};

pub struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
pub struct Plan {
    slices: HashMap<String, Vec<TensorIndexer>>,
}

impl Plan {
    pub fn new() -> Self {
        Plan {
            slices: HashMap::new(),
        }
    }

    pub async fn get_slice(
        &mut self,
        tensor_name: &str,
        slices: Vec<TensorIndexer>,
    ) -> Result<(), Error> {
        match self.slices.entry(tensor_name.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(slices);
            }
            Entry::Occupied(_entry) => {
                return Err(Error::AlreadyExists(tensor_name.to_string()));
            }
        }
        Ok(())
    }

    async fn execute(&self, loader: &mut Loader) -> Result<HashMap<String, Tensor>, Error> {
        todo!()
    }
}
