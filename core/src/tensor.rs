use safetensors::{Dtype, View};
use std::borrow::Cow;

/// A generic tensor data container that owns its data
/// This can be used across the codebase instead of duplicating tensor structures
#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorData {
    /// Create a new TensorData with the given dtype, shape, and data
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }
    
    /// Create an empty TensorData with zeros
    pub fn zeros(dtype: Dtype, shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0u8; total_elements * dtype.size()];
        Self { dtype, shape, data }
    }
    
    /// Get the number of elements in this tensor
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl View for TensorData {
    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }
    
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl<'a> From<safetensors::tensor::TensorView<'a>> for TensorData {
    fn from(tensor_view: safetensors::tensor::TensorView<'a>) -> Self {
        Self {
            dtype: tensor_view.dtype(),
            shape: tensor_view.shape().to_vec(),
            data: tensor_view.data().to_vec(),
        }
    }
} 