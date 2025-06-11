use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeMap};
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
    filename_indices: Vec<usize>,
}

impl SharedInfo {
    pub fn new(shape: Vec<usize>, dtype: Dtype, filename_indices: Vec<usize>) -> Self {
        Self {
            shape,
            dtype,
            filename_indices,
        }
    }
    /// Returns the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the data type of the tensor
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Returns the index of the file containing this tensor
    pub fn filename_indices(&self) -> &[usize] {
        &self.filename_indices
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedInfo {
    shape: Vec<usize>,
    dtype: Dtype,
    chunks: Vec<Chunk>,
}

impl DistributedInfo {
    /// Creates a new distributed tensor info with the given shape, dtype, and chunks
    pub fn new(shape: Vec<usize>, dtype: Dtype, chunks: Vec<Chunk>) -> Self {
        Self {
            shape,
            dtype,
            chunks,
        }
    }

    /// Returns the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the data type of the tensor
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Returns the chunks that make up this distributed tensor
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    offsets: Vec<usize>,
    shape: Vec<usize>,
    filename_index: usize,
}

impl Chunk {
    /// Creates a new chunk with the given offsets, shape, and filename index
    pub fn new(offsets: Vec<usize>, shape: Vec<usize>, filename_index: usize) -> Self {
        Self {
            offsets,
            shape,
            filename_index,
        }
    }

    /// Returns the offsets of this chunk in the full tensor
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns the shape of this chunk
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the index of the file containing this chunk
    pub fn filename_index(&self) -> usize {
        self.filename_index
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimpleTopo {
    tensors: BTreeMap<String, Tensor>,
    filenames: Vec<String>,
    world_size: usize,
}

impl SimpleTopo {
    /// Creates a new simple topology with the given tensors, filenames, and number of ranks
    pub fn new(
        tensors: BTreeMap<String, Tensor>,
        filenames: Vec<String>,
        world_size: usize,
    ) -> Self {
        Self {
            tensors,
            filenames,
            world_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SimpleTopo")]
pub struct Topology {
    tensors: BTreeMap<String, Tensor>,
    filenames: Vec<String>,
    world_size: usize,
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

    #[error("Missing filename indices {1:?} is missing for tensor {0}")]
    FilenameIndexMissing(String, Vec<bool>),

    #[error("Overlapping chunks detected in tensor {0}")]
    OverlappingChunks(String),

    #[error("Non-covering chunks detected in tensor {0}")]
    NonCoveringChunks(String),

    #[error("Invalid chunk dimensions for tensor {0}: expected {1} dimensions, got {2}")]
    InvalidChunkDimensions(String, usize, usize),

    #[error("Chunk out of bounds for tensor {0}: dimension {1} exceeds tensor shape {2}")]
    ChunkOutOfBounds(String, usize, usize),
}

/// Checks if there are any overlapping chunks in a distributed tensor.
/// Returns Ok(()) if there are no overlaps, or an error describing the overlap.
fn check_overlapping_chunks(
    info: &DistributedInfo,
    tensor_name: &str,
) -> Result<(), TopologyError> {
    let ndim = info.shape.len();
    let nelements: usize = info.shape.iter().product();
    let mut covered = Vec::new();

    // Compute strides for the full tensor
    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * info.shape[i + 1];
    }

    for chunk in &info.chunks {
        // Check chunk dimensions
        if chunk.offsets.len() != ndim || chunk.shape.len() != ndim {
            return Err(TopologyError::InvalidChunkDimensions(
                tensor_name.to_string(),
                ndim,
                chunk.offsets.len(),
            ));
        }

        // Check chunk bounds
        for d in 0..ndim {
            if chunk.offsets[d] + chunk.shape[d] > info.shape[d] {
                return Err(TopologyError::ChunkOutOfBounds(
                    tensor_name.to_string(),
                    d,
                    info.shape[d],
                ));
            }
        }

        let intervals = get_intervals(chunk, &strides, info.shape());
        covered.extend(intervals);
    }

    // Sort intervals by start index
    covered.sort_by_key(|&(start, _)| start);

    // Check for overlaps
    let mut prev_end = 0;
    for &(start, end) in &covered {
        match start.cmp(&prev_end) {
            Ordering::Less => {
                return Err(TopologyError::OverlappingChunks(tensor_name.to_string()));
            }
            Ordering::Greater => {
                return Err(TopologyError::NonCoveringChunks(tensor_name.to_string()));
            }
            Ordering::Equal => {}
        }
        prev_end = end;
    }
    match nelements.cmp(&prev_end) {
        Ordering::Less => {
            return Err(TopologyError::OverlappingChunks(tensor_name.to_string()));
        }
        Ordering::Greater => {
            return Err(TopologyError::NonCoveringChunks(tensor_name.to_string()));
        }
        Ordering::Equal => {}
    }
    Ok(())
}

pub fn get_intervals(chunk: &Chunk, strides: &[usize], shape: &[usize]) -> Vec<(usize, usize)> {
    // Pre-allocate with estimated capacity
    let estimated_capacity = chunk.shape.iter().product();
    let mut intervals = Vec::with_capacity(estimated_capacity);
    let mut buffer = Vec::with_capacity(estimated_capacity);
    
    let mut span = 0;
    for ((i, d), total_d) in chunk.shape.iter().enumerate().rev().zip(shape.iter().rev()) {
        if d == total_d && span == 0 {
            continue;
        } else if span == 0 {
            span = strides[i];
            let off = chunk.offsets[i];
            let start = off * strides[i];
            let stop = start + span * d;
            intervals.push((start, stop));
        } else {
            let stride = strides[i];
            let off = chunk.offsets[i];

            // Swap intervals and buffer instead of draining
            std::mem::swap(&mut intervals, &mut buffer);
            intervals.clear();

            for dd in 0..*d {
                for (old_start, old_stop) in &buffer {
                    let new_start = (off + dd) * stride + old_start;
                    let new_stop = (off + dd) * stride + old_stop;
                    intervals.push((new_start, new_stop));
                }
            }
        }
    }
    if intervals.is_empty() {
        let n: usize = chunk.shape().iter().product();
        intervals.push((0, n));
    }
    intervals
}

impl TryFrom<SimpleTopo> for Topology {
    type Error = TopologyError;
    fn try_from(value: SimpleTopo) -> Result<Self, Self::Error> {
        let topo = Topology {
            filenames: value.filenames,
            world_size: value.world_size,
            tensors: value.tensors,
        };
        topo.validate()?;
        Ok(topo)
    }
}
impl Topology {
    pub fn new(
        tensors: BTreeMap<String, Tensor>,
        filenames: Vec<String>,
        world_size: usize,
    ) -> Result<Self, TopologyError> {
        let topo = Self {
            tensors,
            filenames,
            world_size,
        };
        topo.validate()?;
        Ok(topo)
    }
    #[cfg(test)]
    fn empty(world_size: usize) -> Self {
        Self {
            tensors: BTreeMap::new(),
            filenames: vec![],
            world_size,
        }
    }

    /// Returns the number of ranks in the topology
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Returns an iterator over the tensor names and their corresponding tensors
    pub fn tensors(&self) -> &BTreeMap<String, Tensor> {
        &self.tensors
    }

    /// Returns a reference to a specific tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Returns the list of filenames
    pub fn filenames(&self) -> &[String] {
        &self.filenames
    }

    /// Validates that all distributed tensors have the correct number of chunks
    /// matching the number of ranks and that all filename indices are valid.
    /// Also checks that chunks are non-overlapping and form a complete covering set.
    fn validate(&self) -> Result<(), TopologyError> {
        for (name, tensor) in &self.tensors {
            match tensor {
                Tensor::Distributed(info) => {
                    // Check number of chunks
                    if info.chunks.len() != self.world_size {
                        return Err(TopologyError::InvalidChunkCount(
                            name.clone(),
                            self.world_size,
                            info.chunks.len(),
                        ));
                    }

                    let mut set = vec![false; self.filenames.len()];
                    // Check filename indices
                    for chunk in &info.chunks {
                        if chunk.filename_index >= self.filenames.len() {
                            return Err(TopologyError::InvalidFilenameIndex(
                                name.clone(),
                                chunk.filename_index,
                                self.filenames.len(),
                            ));
                        }
                        set[chunk.filename_index] = true;
                    }
                    let missing_indices: Vec<bool> = set.into_iter().filter(|f| !*f).collect();
                    if !missing_indices.is_empty() {
                        return Err(TopologyError::FilenameIndexMissing(
                            name.clone(),
                            missing_indices,
                        ));
                    }

                    // Check for overlaps
                    check_overlapping_chunks(info, name)?;
                }
                Tensor::Shared(info) => {
                    for filename_index in &info.filename_indices {
                        if *filename_index >= self.filenames.len() {
                            return Err(TopologyError::InvalidFilenameIndex(
                                name.clone(),
                                *filename_index,
                                self.filenames.len(),
                            ));
                        }
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
        let mut topology = Topology::empty(2);
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
        let topology = Topology::empty(4);
        assert_eq!(
            serde_json::to_string(&topology).unwrap(),
            r#"{"tensors":{},"filenames":[],"world_size":4}"#
        );
    }

    #[test]
    fn test_validate() {
        let mut topology = Topology::empty(4);

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
                filename_indices: vec![0],
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
        let mut topology = Topology::empty(2);
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
        // Test case 1: Valid non-overlapping chunks with complete coverage
        topology.tensors.insert(
            "valid_coverage2".to_string(),
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
        assert_eq!(topology.validate(), Ok(()));

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

    #[test]
    fn test_check_overlapping_chunks() {
        // Test case 1: No overlapping chunks
        let info1 = DistributedInfo {
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
        };
        assert_eq!(check_overlapping_chunks(&info1, "test_tensor1"), Ok(()));

        // Test case 1: No overlapping chunks last dim
        let info1 = DistributedInfo {
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
        };
        assert_eq!(check_overlapping_chunks(&info1, "test_tensor2"), Ok(()));

        // Test case 2: Overlapping chunks in first dimension
        let info2 = DistributedInfo {
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
        };
        assert_eq!(
            check_overlapping_chunks(&info2, "overlapping_tensor"),
            Err(TopologyError::OverlappingChunks(
                "overlapping_tensor".to_string()
            ))
        );

        // Test case 3: Invalid chunk dimensions
        let info3 = DistributedInfo {
            shape: vec![4, 4],
            dtype: Dtype::F32,
            chunks: vec![
                Chunk {
                    offsets: vec![0], // Wrong number of dimensions
                    shape: vec![4, 4],
                    filename_index: 0,
                },
                Chunk {
                    offsets: vec![2, 0],
                    shape: vec![2, 4],
                    filename_index: 1,
                },
            ],
        };
        assert_eq!(
            check_overlapping_chunks(&info3, "invalid_dim_tensor"),
            Err(TopologyError::InvalidChunkDimensions(
                "invalid_dim_tensor".to_string(),
                2,
                1
            ))
        );

        // Test case 4: Chunk out of bounds
        let info4 = DistributedInfo {
            shape: vec![4, 4],
            dtype: Dtype::F32,
            chunks: vec![
                Chunk {
                    offsets: vec![0, 0],
                    shape: vec![5, 4], // Exceeds tensor shape
                    filename_index: 0,
                },
                Chunk {
                    offsets: vec![2, 0],
                    shape: vec![2, 4],
                    filename_index: 1,
                },
            ],
        };
        assert_eq!(
            check_overlapping_chunks(&info4, "out_of_bounds_tensor"),
            Err(TopologyError::ChunkOutOfBounds(
                "out_of_bounds_tensor".to_string(),
                0,
                4
            ))
        );
    }

    #[test]
    fn test_check_overlapping_chunks_3d() {
        // Create a 3D tensor with overlapping chunks in the middle dimension
        let info = DistributedInfo {
            shape: vec![2, 4, 2], // 2x4x2 tensor
            dtype: Dtype::F32,
            chunks: vec![
                Chunk {
                    offsets: vec![0, 0, 0],
                    shape: vec![2, 2, 2], // First chunk: full height, first half of width
                    filename_index: 0,
                },
                Chunk {
                    offsets: vec![0, 2, 0], // Overlaps in middle dimension
                    shape: vec![2, 2, 2],   // Second chunk: full height, second half of width
                    filename_index: 1,
                },
            ],
        };
        assert_eq!(
            check_overlapping_chunks(&info, "overlapping_3d_tensor"),
            Ok(())
        );
    }

    #[test]
    fn test_check_overlapping_chunks_3d_fail() {
        // Create a 3D tensor with overlapping chunks in the middle dimension
        let info = DistributedInfo {
            shape: vec![2, 4, 2], // 2x4x2 tensor
            dtype: Dtype::F32,
            chunks: vec![
                Chunk {
                    offsets: vec![0, 0, 0],
                    shape: vec![2, 2, 2], // First chunk: full height, first half of width
                    filename_index: 0,
                },
                Chunk {
                    offsets: vec![0, 1, 0], // Overlaps in middle dimension
                    shape: vec![2, 2, 2],   // Second chunk: full height, second half of width
                    filename_index: 1,
                },
            ],
        };
        assert_eq!(
            check_overlapping_chunks(&info, "overlapping_3d_tensor"),
            Err(TopologyError::OverlappingChunks(
                "overlapping_3d_tensor".to_string()
            ))
        );
    }

    #[test]
    fn test_check_overlapping_chunks_quadrants() {
        // Create a 4x4 tensor split into 4 quadrants
        let info = DistributedInfo {
            shape: vec![4, 4], // 4x4 tensor
            dtype: Dtype::F32,
            chunks: vec![
                Chunk {
                    offsets: vec![0, 0],
                    shape: vec![2, 2], // Top-left quadrant
                    filename_index: 0,
                },
                Chunk {
                    offsets: vec![0, 2],
                    shape: vec![2, 2], // Top-right quadrant
                    filename_index: 1,
                },
                Chunk {
                    offsets: vec![2, 0],
                    shape: vec![2, 2], // Bottom-left quadrant
                    filename_index: 2,
                },
                Chunk {
                    offsets: vec![2, 2],
                    shape: vec![2, 2], // Bottom-right quadrant
                    filename_index: 3,
                },
            ],
        };
        assert_eq!(check_overlapping_chunks(&info, "quadrant_tensor"), Ok(()));
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
        let mut topology = Topology::empty(2);
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
            .route("/{filename}", get(move |State(path): State<Arc<PathBuf>>, axum::extract::Path(filename): axum::extract::Path<String>| async move {
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
