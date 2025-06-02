use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tensor {
    Distributed(DistributedInfo),
    Shared(SharedInfo),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedInfo {
    shape: Vec<usize>,
    dtype: Dtype,
    filename: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedInfo {
    shape: Vec<usize>,
    dtype: Dtype,
    chunks: Vec<Chunk>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    offsets: Vec<usize>,
    shape: Vec<usize>,
    filename: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Topology {
    tensors: HashMap<String, Tensor>,
    n_ranks: usize,
}

/// Error type for topology operations
#[derive(Debug, thiserror::Error)]
pub enum TopologyError {
    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid tensor name: {0}")]
    InvalidTensorName(String),

    #[error("Missing metadata for tensor: {0}")]
    MissingMetadata(String),
}

impl Topology {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_serialization() {
        let topology = Topology::new();
        assert_eq!(
            serde_json::to_string(&topology).unwrap(),
            r#"{"tensors":{}}"#
        );
    }
}

