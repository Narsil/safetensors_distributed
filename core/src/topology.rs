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
    filename_index: usize,
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
    filename_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Topology {
    tensors: HashMap<String, Tensor>,
    filenames: Vec<String>,
    n_ranks: usize,
}

/// Error type for topology operations
#[derive(Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum TopologyError {
    #[error("Invalid tensor name: {0}")]
    InvalidTensorName(String),

    #[error("Missing metadata for tensor: {0}")]
    MissingMetadata(String),

    #[error("Invalid number of chunks for tensor {0}: expected {1}, got {2}")]
    InvalidChunkCount(String, usize, usize),

    #[error("Invalid filename index {1} for tensor {0}: index out of bounds (max: {2})")]
    InvalidFilenameIndex(String, usize, usize),

    #[error("Overlapping chunks detected in tensor {0}")]
    OverlappingChunks(String),

    #[error("Non-covering chunks detected in tensor {0}")]
    NonCoveringChunks(String),
}

impl Topology {
    pub fn new(n_ranks: usize) -> Self {
        Self {
            tensors: HashMap::new(),
            filenames: vec![],
            n_ranks,
        }
    }

    /// Validates that all distributed tensors have the correct number of chunks
    /// matching the number of ranks and that all filename indices are valid.
    /// Also checks that chunks are non-overlapping and form a complete covering set.
    pub fn validate(&self) -> Result<(), TopologyError> {
        for (name, tensor) in &self.tensors {
            match tensor {
                Tensor::Distributed(info) => {
                    // Check number of chunks
                    if info.chunks.len() != self.n_ranks {
                        return Err(TopologyError::InvalidChunkCount(
                            name.clone(),
                            self.n_ranks,
                            info.chunks.len(),
                        ));
                    }

                    // Check filename indices
                    for chunk in &info.chunks {
                        if chunk.filename_index >= self.filenames.len() {
                            return Err(TopologyError::InvalidFilenameIndex(
                                name.clone(),
                                chunk.filename_index,
                                self.filenames.len(),
                            ));
                        }
                    }

                    let ndim = info.shape.len();
                    let total_elements: usize = info.shape.iter().product();
                    let mut covered = Vec::with_capacity(total_elements);
                    for chunk in &info.chunks {
                        if chunk.offsets.len() != ndim || chunk.shape.len() != ndim {
                            return Err(TopologyError::InvalidChunkCount(
                                name.clone(),
                                ndim,
                                chunk.offsets.len(),
                            ));
                        }
                        for d in 0..ndim {
                            if chunk.offsets[d] + chunk.shape[d] > info.shape[d] {
                                return Err(TopologyError::InvalidChunkCount(
                                    name.clone(),
                                    info.shape[d],
                                    chunk.offsets[d] + chunk.shape[d],
                                ));
                            }
                        }
                        // Enumerate all indices covered by this chunk
                        let mut idx = vec![0; ndim];
                        loop {
                            let global_idx: Vec<usize> = idx
                                .iter()
                                .enumerate()
                                .map(|(d, &i)| chunk.offsets[d] + i)
                                .collect();
                            covered.push(global_idx.clone());
                            // Increment idx
                            let mut dim = ndim;
                            while dim > 0 {
                                dim -= 1;
                                idx[dim] += 1;
                                if idx[dim] < chunk.shape[dim] {
                                    break;
                                } else {
                                    idx[dim] = 0;
                                }
                            }
                            if dim == 0 && idx[0] == 0 {
                                break;
                            }
                        }
                    }
                    covered.sort();
                    // Check for overlaps and coverage
                    let mut prev: Option<&Vec<usize>> = None;
                    for idx in &covered {
                        if let Some(p) = prev {
                            if p == idx {
                                return Err(TopologyError::OverlappingChunks(name.clone()));
                            }
                        }
                        prev = Some(idx);
                    }
                    // Check for complete coverage
                    if covered.len() != total_elements {
                        return Err(TopologyError::NonCoveringChunks(name.clone()));
                    }
                    // Check that all indices are present
                    let mut expected = vec![0; ndim];
                    for idx in &covered {
                        if &expected != idx {
                            return Err(TopologyError::NonCoveringChunks(name.clone()));
                        }
                        // Increment expected
                        let mut dim = ndim;
                        while dim > 0 {
                            dim -= 1;
                            expected[dim] += 1;
                            if expected[dim] < info.shape[dim] {
                                break;
                            } else {
                                expected[dim] = 0;
                            }
                        }
                    }
                }
                Tensor::Shared(info) => {
                    if info.filename_index >= self.filenames.len() {
                        return Err(TopologyError::InvalidFilenameIndex(
                            name.clone(),
                            info.filename_index,
                            self.filenames.len(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, extract::State, routing::get};
    use reqwest;
    use safetensors::SafeTensors;
    use safetensors::tensor::{TensorView, serialize};
    use std::fs;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::oneshot;

    // Helper function to convert &[f32] to Vec<u8> using to_le_bytes
    fn f32s_to_le_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // Helper function to convert &[u8] to Vec<f32> using from_le_bytes
    fn le_bytes_to_f32s(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    // Helper function to reconstruct a tensor from its chunks
    fn reconstruct_tensor(topology: &Topology, tensor_name: &str) -> Vec<f32> {
        if let Some(Tensor::Distributed(info)) = topology.tensors.get(tensor_name) {
            let mut result = vec![0.0; info.shape.iter().product()];
            let ndim = info.shape.len();
            for chunk in &info.chunks {
                let data = fs::read(&topology.filenames[chunk.filename_index]).unwrap();
                let file = SafeTensors::deserialize(&data).unwrap();
                let tensor = file.tensor(tensor_name).unwrap();
                let bytes = tensor.data();
                let data: &[f32] = unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
                };
                // Compute strides for the full tensor
                let mut full_strides = vec![1; ndim];
                for i in (0..ndim - 1).rev() {
                    full_strides[i] = full_strides[i + 1] * info.shape[i + 1];
                }
                // Compute strides for the chunk
                let mut chunk_strides = vec![1; ndim];
                for i in (0..ndim - 1).rev() {
                    chunk_strides[i] = chunk_strides[i + 1] * chunk.shape[i + 1];
                }
                // For each element in the chunk, compute its index in the full tensor
                for idx in 0..data.len() {
                    // Convert flat idx to multi-dimensional index for the chunk
                    let mut chunk_multi_idx = vec![0; ndim];
                    let mut remaining = idx;
                    for d in 0..ndim {
                        chunk_multi_idx[d] = remaining / chunk_strides[d];
                        remaining %= chunk_strides[d];
                    }
                    // Compute the corresponding index in the full tensor
                    let mut full_idx = 0;
                    for d in 0..ndim {
                        let pos = chunk.offsets[d] + chunk_multi_idx[d];
                        full_idx += pos * full_strides[d];
                    }
                    result[full_idx] = data[idx];
                }
            }
            result
        } else {
            panic!("Tensor {} not found or not distributed", tensor_name);
        }
    }

    #[test]
    fn test_distributed_checkpoint() {
        // Create a temporary directory for our test files
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create the safetensors files for each rank
        let rank0_path = dir_path.join("rank0.safetensors");
        let rank1_path = dir_path.join("rank1.safetensors");

        // Create test data: a 4x4 tensor with values 0..15
        let full_tensor: Vec<f32> = (0..16).map(|x| x as f32).collect();

        // First tensor: sharded along first dimension
        // rank0: [:2, :] -> [[0,1,2,3], [4,5,6,7]]
        // rank1: [2:, :] -> [[8,9,10,11], [12,13,14,15]]
        let tensor1_rank0 = &full_tensor[0..8];
        let tensor1_rank1 = &full_tensor[8..16];

        // Second tensor: sharded along second dimension
        // rank0: [:, :2] -> [[0,1], [4,5], [8,9], [12,13]]
        // rank1: [:, 2:] -> [[2,3], [6,7], [10,11], [14,15]]
        let tensor2_rank0: Vec<f32> = vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0];
        let tensor2_rank1: Vec<f32> = vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0];

        // Write both tensors to rank0 file
        let mut tensors0 = std::collections::HashMap::new();
        let tensor1_rank0_bytes = f32s_to_le_bytes(tensor1_rank0);
        let tensor2_rank0_bytes = f32s_to_le_bytes(&tensor2_rank0);
        tensors0.insert(
            "tensor1".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![2, 4], &tensor1_rank0_bytes).unwrap(),
        );
        tensors0.insert(
            "tensor2".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![4, 2], &tensor2_rank0_bytes).unwrap(),
        );
        let bytes0 = serialize(&tensors0, &None).unwrap();
        fs::write(&rank0_path, bytes0).unwrap();

        // Write both tensors to rank1 file
        let mut tensors1 = std::collections::HashMap::new();
        let tensor1_rank1_bytes = f32s_to_le_bytes(tensor1_rank1);
        let tensor2_rank1_bytes = f32s_to_le_bytes(&tensor2_rank1);
        tensors1.insert(
            "tensor1".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![2, 4], &tensor1_rank1_bytes).unwrap(),
        );
        tensors1.insert(
            "tensor2".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![4, 2], &tensor2_rank1_bytes).unwrap(),
        );
        let bytes1 = serialize(&tensors1, &None).unwrap();
        fs::write(&rank1_path, bytes1).unwrap();

        // Create the topology
        let mut topology = Topology::new(2);
        topology.filenames = vec![
            rank0_path.to_str().unwrap().to_string(),
            rank1_path.to_str().unwrap().to_string(),
        ];

        // Add tensor1 (sharded along first dimension)
        topology.tensors.insert(
            "tensor1".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![2, 4],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![2, 0],
                        shape: vec![2, 4],
                        filename_index: 1,
                    },
                ],
            }),
        );

        // Add tensor2 (sharded along second dimension)
        topology.tensors.insert(
            "tensor2".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![4, 2],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![0, 2],
                        shape: vec![4, 2],
                        filename_index: 1,
                    },
                ],
            }),
        );

        // Verify the topology
        assert!(topology.validate().is_ok());

        // Reconstruct and verify tensor1
        let reconstructed_tensor1 = reconstruct_tensor(&topology, "tensor1");
        assert_eq!(reconstructed_tensor1, full_tensor);

        // Reconstruct and verify tensor2
        let reconstructed_tensor2 = reconstruct_tensor(&topology, "tensor2");
        assert_eq!(reconstructed_tensor2, full_tensor);
    }

    #[test]
    fn test_topology_serialization() {
        let topology = Topology::new(4);
        assert_eq!(
            serde_json::to_string(&topology).unwrap(),
            r#"{"tensors":{},"filenames":[],"n_ranks":4}"#
        );
    }

    #[test]
    fn test_validate() {
        let mut topology = Topology::new(4);

        // Add some filenames
        topology.filenames = vec![
            "file1.safetensors".to_string(),
            "file2.safetensors".to_string(),
            "file3.safetensors".to_string(),
            "file4.safetensors".to_string(),
        ];

        // Test with no tensors (should pass)
        assert!(topology.validate().is_ok());

        // Test with shared tensor (should pass)
        topology.tensors.insert(
            "shared".to_string(),
            Tensor::Shared(SharedInfo {
                shape: vec![10, 10],
                dtype: Dtype::F32,
                filename_index: 0,
            }),
        );
        assert_eq!(topology.validate(), Ok(()));

        // Test with valid distributed tensor
        topology.tensors.insert(
            "valid_dist".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![16, 16],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![4, 16],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![4, 0],
                        shape: vec![4, 16],
                        filename_index: 1,
                    },
                    Chunk {
                        offsets: vec![8, 0],
                        shape: vec![4, 16],
                        filename_index: 2,
                    },
                    Chunk {
                        offsets: vec![12, 0],
                        shape: vec![4, 16],
                        filename_index: 3,
                    },
                ],
            }),
        );
        assert_eq!(topology.validate(), Ok(()));

        // Test with valid distributed tensor
        topology.tensors.insert(
            "valid_dist_non_connex".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![16, 16],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![16, 4],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![0, 4],
                        shape: vec![16, 4],
                        filename_index: 1,
                    },
                    Chunk {
                        offsets: vec![0, 8],
                        shape: vec![16, 4],
                        filename_index: 2,
                    },
                    Chunk {
                        offsets: vec![0, 12],
                        shape: vec![16, 4],
                        filename_index: 3,
                    },
                ],
            }),
        );
        assert_eq!(topology.validate(), Ok(()));

        // Test with invalid distributed tensor (wrong number of chunks)
        topology.tensors.insert(
            "invalid_dist".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![10, 10],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![5, 10],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![5, 0],
                        shape: vec![5, 10],
                        filename_index: 1,
                    },
                ],
            }),
        );
        let err = topology.validate().unwrap_err();
        assert_eq!(
            err,
            TopologyError::InvalidChunkCount("invalid_dist".to_string(), 4, 2)
        );

        // Remove the invalid tensor
        topology.tensors.remove("invalid_dist");

        // Test with invalid filename index
        topology.tensors.insert(
            "invalid_filename".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![16, 16],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![4, 16],
                        filename_index: 8, // Invalid index
                    },
                    Chunk {
                        offsets: vec![4, 0],
                        shape: vec![4, 16],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![8, 0],
                        shape: vec![4, 16],
                        filename_index: 1,
                    },
                    Chunk {
                        offsets: vec![12, 0],
                        shape: vec![4, 16],
                        filename_index: 0,
                    },
                ],
            }),
        );
        let err = topology.validate().unwrap_err();
        assert_eq!(
            err,
            TopologyError::InvalidFilenameIndex("invalid_filename".to_string(), 8, 4)
        );
    }

    #[test]
    fn test_validate_chunk_coverage() {
        let mut topology = Topology::new(2);
        topology.filenames = vec![
            "file1.safetensors".to_string(),
            "file2.safetensors".to_string(),
        ];

        // Test case 1: Valid non-overlapping chunks with complete coverage
        topology.tensors.insert(
            "valid_coverage".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![2, 4],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![2, 0],
                        shape: vec![2, 4],
                        filename_index: 1,
                    },
                ],
            }),
        );
        assert!(topology.validate().is_ok());

        // Test case 2: Overlapping chunks
        topology.tensors.insert(
            "overlapping".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![3, 4], // Overlaps with second chunk
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![2, 0],
                        shape: vec![2, 4],
                        filename_index: 1,
                    },
                ],
            }),
        );
        let err = topology.validate().unwrap_err();
        assert_eq!(
            err,
            TopologyError::OverlappingChunks("overlapping".to_string())
        );

        // Remove the invalid tensor
        topology.tensors.remove("overlapping");

        // Test case 3: Non-covering chunks
        topology.tensors.insert(
            "non_covering".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![1, 4], // Only covers first row
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![2, 0],
                        shape: vec![1, 4], // Only covers third row
                        filename_index: 1,
                    },
                ],
            }),
        );
        let err = topology.validate().unwrap_err();
        assert_eq!(
            err,
            TopologyError::NonCoveringChunks("non_covering".to_string())
        );
    }

    #[tokio::test]
    async fn test_distributed_checkpoint_http() {
        // Create a temporary directory for our test files
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // Create the safetensors files for each rank
        let rank0_path = dir_path.join("rank0.safetensors");
        let rank1_path = dir_path.join("rank1.safetensors");

        // Create test data: a 4x4 tensor with values 0..15
        let full_tensor: Vec<f32> = (0..16).map(|x| x as f32).collect();

        // First tensor: sharded along first dimension
        let tensor1_rank0 = &full_tensor[0..8];
        let tensor1_rank1 = &full_tensor[8..16];

        // Second tensor: sharded along second dimension
        let tensor2_rank0: Vec<f32> = vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0];
        let tensor2_rank1: Vec<f32> = vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0];

        // Write both tensors to rank0 file
        let mut tensors0 = std::collections::HashMap::new();
        let tensor1_rank0_bytes = f32s_to_le_bytes(tensor1_rank0);
        let tensor2_rank0_bytes = f32s_to_le_bytes(&tensor2_rank0);
        tensors0.insert(
            "tensor1".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![2, 4], &tensor1_rank0_bytes).unwrap(),
        );
        tensors0.insert(
            "tensor2".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![4, 2], &tensor2_rank0_bytes).unwrap(),
        );
        let bytes0 = serialize(&tensors0, &None).unwrap();
        fs::write(&rank0_path, bytes0).unwrap();

        // Write both tensors to rank1 file
        let mut tensors1 = std::collections::HashMap::new();
        let tensor1_rank1_bytes = f32s_to_le_bytes(tensor1_rank1);
        let tensor2_rank1_bytes = f32s_to_le_bytes(&tensor2_rank1);
        tensors1.insert(
            "tensor1".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![2, 4], &tensor1_rank1_bytes).unwrap(),
        );
        tensors1.insert(
            "tensor2".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![4, 2], &tensor2_rank1_bytes).unwrap(),
        );
        let bytes1 = serialize(&tensors1, &None).unwrap();
        fs::write(&rank1_path, bytes1).unwrap();

        // Create the topology
        let mut topology = Topology::new(2);
        topology.filenames = vec![
            "rank0.safetensors".to_string(),
            "rank1.safetensors".to_string(),
        ];

        // Add tensor1 (sharded along first dimension)
        topology.tensors.insert(
            "tensor1".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![2, 4],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![2, 0],
                        shape: vec![2, 4],
                        filename_index: 1,
                    },
                ],
            }),
        );

        // Add tensor2 (sharded along second dimension)
        topology.tensors.insert(
            "tensor2".to_string(),
            Tensor::Distributed(DistributedInfo {
                shape: vec![4, 4],
                dtype: Dtype::F32,
                chunks: vec![
                    Chunk {
                        offsets: vec![0, 0],
                        shape: vec![4, 2],
                        filename_index: 0,
                    },
                    Chunk {
                        offsets: vec![0, 2],
                        shape: vec![4, 2],
                        filename_index: 1,
                    },
                ],
            }),
        );

        // Write topology to JSON file
        let topology_json = serde_json::to_string_pretty(&topology).unwrap();
        fs::write(dir_path.join("topology.json"), topology_json).unwrap();

        // Create a shared state for the server
        let dir_path = Arc::new(dir_path.to_path_buf());

        // Create the router
        let app = Router::new()
            .route("/topology.json", get(move |State(path): State<Arc<PathBuf>>| async move {
                let content = fs::read_to_string(path.join("topology.json")).unwrap();
                content
            }))
            .route("/:filename", get(move |State(path): State<Arc<PathBuf>>, axum::extract::Path(filename): axum::extract::Path<String>| async move {
                let content = fs::read(path.join(filename)).unwrap();
                content
            }))
            .with_state(dir_path.clone());

        // Start the server
        let addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let server = axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app);

        // Get the actual port
        let port = server
            .local_addr()
            .expect("Failed to get local_addr")
            .port();

        // Spawn the server
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server_handle = tokio::spawn(async move {
            server
                .with_graceful_shutdown(async {
                    shutdown_rx.await.ok();
                })
                .await
                .unwrap();
        });

        // Create a client
        let client = reqwest::Client::new();
        let base_url = format!("http://127.0.0.1:{}", port);

        // Fetch and reconstruct tensor1
        let reconstructed_tensor1 = reconstruct_tensor_http(&client, &base_url, "tensor1").await;
        assert_eq!(reconstructed_tensor1, full_tensor);

        // Fetch and reconstruct tensor2
        let reconstructed_tensor2 = reconstruct_tensor_http(&client, &base_url, "tensor2").await;
        assert_eq!(reconstructed_tensor2, full_tensor);

        // Shutdown the server
        shutdown_tx.send(()).unwrap();
        server_handle.await.unwrap();
    }

    async fn reconstruct_tensor_http(
        client: &reqwest::Client,
        base_url: &str,
        tensor_name: &str,
    ) -> Vec<f32> {
        // Fetch the topology
        let topology_json = client
            .get(&format!("{}/topology.json", base_url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let topology: Topology = serde_json::from_str(&topology_json).unwrap();

        if let Some(Tensor::Distributed(info)) = topology.tensors.get(tensor_name) {
            let mut result = vec![0.0; info.shape.iter().product()];
            let ndim = info.shape.len();

            for chunk in &info.chunks {
                // Fetch the safetensors file
                let filename = &topology.filenames[chunk.filename_index];
                let data = client
                    .get(&format!("{}/{}", base_url, filename))
                    .send()
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap();

                let file = SafeTensors::deserialize(&data).unwrap();
                let tensor = file.tensor(tensor_name).unwrap();
                let bytes = tensor.data();
                let data = le_bytes_to_f32s(bytes);

                // Compute strides for the full tensor
                let mut full_strides = vec![1; ndim];
                for i in (0..ndim - 1).rev() {
                    full_strides[i] = full_strides[i + 1] * info.shape[i + 1];
                }

                // Compute strides for the chunk
                let mut chunk_strides = vec![1; ndim];
                for i in (0..ndim - 1).rev() {
                    chunk_strides[i] = chunk_strides[i + 1] * chunk.shape[i + 1];
                }

                // For each element in the chunk, compute its index in the full tensor
                for idx in 0..data.len() {
                    // Convert flat idx to multi-dimensional index for the chunk
                    let mut chunk_multi_idx = vec![0; ndim];
                    let mut remaining = idx;
                    for d in 0..ndim {
                        chunk_multi_idx[d] = remaining / chunk_strides[d];
                        remaining %= chunk_strides[d];
                    }

                    // Compute the corresponding index in the full tensor
                    let mut full_idx = 0;
                    for d in 0..ndim {
                        let pos = chunk.offsets[d] + chunk_multi_idx[d];
                        full_idx += pos * full_strides[d];
                    }

                    result[full_idx] = data[idx];
                }
            }
            result
        } else {
            panic!("Tensor {} not found or not distributed", tensor_name);
        }
    }
}
