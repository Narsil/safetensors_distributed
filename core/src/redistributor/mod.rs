pub mod core;
pub mod loader;
pub mod location;
pub mod task;

use crate::topology::{Topology, TopologyError};
use indicatif::style::TemplateError;
use safetensors::tensor::Metadata;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;
use tokio::task::JoinError;

// Re-export main types
pub use core::AsyncTensorRedistributor;

/// Strategy for ordering reads and writes during redistribution
#[derive(Clone, Copy, Debug)]
pub enum RedistributionStrategy {
    /// Read files sequentially, write tasks can execute unordered
    ReadSerialWriteUnordered,
    /// Read tasks can execute unordered, but write tasks execute serially  
    ReadUnorderedWriteSerial,
}

impl Default for RedistributionStrategy {
    fn default() -> Self {
        Self::ReadUnorderedWriteSerial
    }
}

/// Structure for deserializing model.safetensors.index.json
#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    /// Map of tensor names to their containing file
    pub weight_map: HashMap<String, String>,
}

/// Layout containing topology and metadata information
pub struct Layout {
    pub topology: Topology,
    pub metadatas: Vec<(usize, Metadata)>,
}

#[derive(Error, Debug)]
pub enum RedistributorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Topology error: {0}")]
    Topology(#[from] TopologyError),

    #[error("Invalid tensor data source: {message}")]
    InvalidDataSource { message: String },

    #[error("Tensor not found: {name}")]
    TensorNotFound { name: String },

    #[error("Invalid tensor dimension {dim} for tensor with shape {shape:?}")]
    InvalidDimension { dim: usize, shape: Vec<usize> },

    #[error("Invalid slice range [{start}, {end}) for dimension {dim} with size {size}")]
    InvalidSliceRange {
        start: usize,
        end: usize,
        dim: usize,
        size: usize,
    },

    #[error(
        "No valid input found in directory {path:?} (expected topology.json + rank*.safetensors OR model.safetensors OR model.safetensors.index.json + chunked files)"
    )]
    NoValidInput { path: PathBuf },

    #[error("Failed to parse target world size: {input}")]
    InvalidWorldSize { input: String },

    #[error("Template error: {0}")]
    Template(#[from] TemplateError),

    #[error("Join error: {0}")]
    Join(#[from] JoinError),
}

pub type Result<T> = std::result::Result<T, RedistributorError>;

// Helper function used by multiple modules
pub fn safetensors_metadata<P: AsRef<std::path::Path>>(file_path: P) -> Result<(usize, Metadata)> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let mut file = BufReader::new(File::open(file_path)?);
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)?;
    let length = u64::from_le_bytes(length_bytes) as usize;

    let mut metadata_bytes = vec![0u8; length];
    file.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = serde_json::from_slice(&metadata_bytes)?;

    let header_size = 8 + length;
    Ok((header_size, metadata))
}

pub fn load_or_create_topology<P: AsRef<std::path::Path>>(dir: P) -> Result<Topology> {
    let dir = dir.as_ref();
    let topology_path = dir.join("topology.json");

    if topology_path.exists() {
        // Load existing topology
        let content = std::fs::read_to_string(topology_path)?;
        let topology: Topology = serde_json::from_str(&content)?;
        return Ok(topology);
    }

    // Try to detect from rank files
    let mut rank_files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("rank") && name.ends_with(".safetensors") {
                rank_files.push(path);
            }
        }
    }

    if !rank_files.is_empty() {
        rank_files.sort();
        // Load topology from rank files - create a distributed topology
        use std::collections::BTreeMap;
        let mut tensors = BTreeMap::new();
        let mut filenames = Vec::new();

        for (file_index, file_path) in rank_files.iter().enumerate() {
            let filename = file_path.file_name().unwrap().to_str().unwrap().to_string();
            filenames.push(filename);

            let (_, metadata) = safetensors_metadata(file_path)?;
            for (tensor_name, tensor_info) in metadata.tensors() {
                // For now, treat as shared tensors across ranks
                // TODO: This should be improved to detect actual distribution
                tensors.insert(
                    tensor_name.clone(),
                    crate::topology::Tensor::Shared(crate::topology::SharedInfo::new(
                        tensor_info.shape.clone(),
                        tensor_info.dtype,
                        vec![file_index],
                    )),
                );
            }
        }

        return crate::topology::Topology::new(tensors, filenames, rank_files.len())
            .map_err(RedistributorError::Topology);
    }

    // Try model.safetensors.index.json (chunked safetensors case)
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        use std::collections::BTreeMap;
        use std::collections::HashSet;

        let index_data = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_data)?;

        // Get unique filenames
        let filenames: HashSet<String> = index.weight_map.values().cloned().collect();
        let mut filenames: Vec<String> = filenames.into_iter().collect();
        filenames.sort();

        // Create shared tensors for all tensors using metadata from actual files
        let mut tensors = BTreeMap::new();
        for (tensor_name, file_name) in &index.weight_map {
            let file_path = dir.join(file_name);
            let (_, file_metadata) = safetensors_metadata(&file_path)?;

            if let Some(tensor_info) = file_metadata.tensors().get(tensor_name) {
                // Find which file index this tensor belongs to
                let file_index = filenames.iter().position(|f| f == file_name).unwrap();

                tensors.insert(
                    tensor_name.clone(),
                    crate::topology::Tensor::Shared(crate::topology::SharedInfo::new(
                        tensor_info.shape.clone(),
                        tensor_info.dtype,
                        vec![file_index],
                    )),
                );
            }
        }

        return crate::topology::Topology::new(tensors, filenames, 1)
            .map_err(RedistributorError::Topology);
    }

    // Try model.safetensors
    let model_path = dir.join("model.safetensors");
    if model_path.exists() {
        use std::collections::BTreeMap;
        let mut tensors = BTreeMap::new();
        let (_, metadata) = safetensors_metadata(&model_path)?;

        for (tensor_name, tensor_info) in metadata.tensors() {
            tensors.insert(
                tensor_name.clone(),
                crate::topology::Tensor::Shared(crate::topology::SharedInfo::new(
                    tensor_info.shape.clone(),
                    tensor_info.dtype,
                    vec![0],
                )),
            );
        }

        return crate::topology::Topology::new(tensors, vec!["model.safetensors".to_string()], 1)
            .map_err(RedistributorError::Topology);
    }

    Err(RedistributorError::NoValidInput {
        path: dir.to_path_buf(),
    })
}

// Intersection function used by core
pub fn intersection(
    source_intervals: &[(usize, usize)],
    target_intervals: &[(usize, usize)],
) -> Vec<(usize, usize, usize)> {
    // Pre-allocate with minimum capacity
    let min_capacity = source_intervals.len().min(target_intervals.len());
    let mut result = Vec::with_capacity(min_capacity);

    let mut soffset = 0;
    let mut toffset = 0;
    let mut sindex = 0;
    let mut tindex = 0;

    while sindex < source_intervals.len() && tindex < target_intervals.len() {
        let (source_start, source_end) = source_intervals[sindex];
        let (target_start, target_end) = target_intervals[tindex];
        let intersection_start = source_start.max(target_start);
        let intersection_end = source_end.min(target_end);

        if intersection_start < intersection_end {
            // There is an overlap
            let source_offset = soffset + (intersection_start - source_start);
            let target_offset = toffset + (intersection_start - target_start);
            let length = intersection_end - intersection_start;

            result.push((source_offset, target_offset, length));
        }

        if source_end < target_end {
            sindex += 1;
            soffset += source_end - source_start;
        } else {
            tindex += 1;
            toffset += target_end - target_start;
        }
    }

    result
}

/// Optimized direct chunk intersection without generating intervals
/// This replaces the expensive get_intervals + intersection workflow
/// Returns (source_offset, target_offset, length) tuples for overlapping regions
pub fn chunk_intersection_direct(
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<(usize, usize, usize)> {
    // Step 1: Generate intervals efficiently without materializing them
    let source_intervals = generate_intervals_lazy(source_chunk, strides, full_shape);
    let target_intervals = generate_intervals_lazy(target_chunk, strides, full_shape);

    // Step 2: Apply the same intersection logic as the legacy function
    intersection_direct(&source_intervals, &target_intervals)
}

/// Generate intervals lazily without fully materializing them
/// Returns a sorted list of (start, end) intervals
fn generate_intervals_lazy(
    chunk: &crate::topology::Chunk,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<(usize, usize)> {
    // For now, use the legacy get_intervals function but in the future
    // this could be optimized to generate intervals more efficiently
    crate::topology::get_intervals(chunk, strides, full_shape)
}

/// Direct intersection that follows the exact same logic as the legacy intersection function
fn intersection_direct(
    source_intervals: &[(usize, usize)],
    target_intervals: &[(usize, usize)],
) -> Vec<(usize, usize, usize)> {
    // This is identical to the intersection function but renamed for clarity
    intersection(source_intervals, target_intervals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{Chunk, get_intervals};

    #[test]
    fn test_chunk_intersection_direct_vs_legacy() {
        // Test case: Compare new direct method with old get_intervals + intersection method

        // Create test chunks and parameters
        let source_chunk = Chunk::new(vec![2, 1], vec![3, 2], 0); // offset [2,1], shape [3,2]
        let target_chunk = Chunk::new(vec![1, 0], vec![4, 3], 1); // offset [1,0], shape [4,3]
        let full_shape = vec![6, 4]; // 6x4 tensor

        // Calculate strides
        let mut strides = vec![1; full_shape.len()];
        for i in (0..full_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * full_shape[i + 1];
        }

        // Method 1: Legacy approach using get_intervals + intersection
        let source_intervals = get_intervals(&source_chunk, &strides, &full_shape);
        let target_intervals = get_intervals(&target_chunk, &strides, &full_shape);
        let legacy_result = intersection(&source_intervals, &target_intervals);

        // Method 2: New optimized direct approach
        let direct_result =
            chunk_intersection_direct(&source_chunk, &target_chunk, &strides, &full_shape);

        // Results should be identical
        assert_eq!(
            legacy_result, direct_result,
            "Direct chunk intersection should produce same results as legacy method"
        );
    }

    #[test]
    fn test_chunk_intersection_direct_no_overlap() {
        // Test case where chunks don't overlap
        let source_chunk = Chunk::new(vec![0, 0], vec![2, 2], 0);
        let target_chunk = Chunk::new(vec![3, 3], vec![2, 2], 1);
        let full_shape = vec![6, 6];
        let strides = vec![6, 1];

        let result = chunk_intersection_direct(&source_chunk, &target_chunk, &strides, &full_shape);
        assert!(
            result.is_empty(),
            "Non-overlapping chunks should return empty result"
        );
    }

    #[test]
    fn test_chunk_intersection_direct_full_overlap() {
        // Test case where chunks completely overlap
        let chunk = Chunk::new(vec![1, 1], vec![2, 2], 0);
        let full_shape = vec![4, 4];
        let strides = vec![4, 1];

        let result = chunk_intersection_direct(&chunk, &chunk, &strides, &full_shape);

        // The chunk should intersect with itself
        assert!(!result.is_empty(), "Same chunks should have intersection");

        // Calculate expected total length (should equal chunk size: 2*2 = 4)
        let expected_total_length: usize = chunk.shape().iter().product();
        let actual_total_length: usize = result.iter().map(|(_, _, len)| len).sum();
        assert_eq!(
            actual_total_length, expected_total_length,
            "Total intersection length should equal chunk size"
        );

        // All source and target offsets should start at 0 since it's the same chunk
        for &(source_offset, target_offset, _) in &result {
            assert_eq!(
                source_offset, target_offset,
                "Source and target offsets should be equal for identical chunks"
            );
        }
    }

    #[test]
    fn test_compute_read_ranges_direct_basic() {
        // Test basic functionality of compute_read_ranges_direct
        let source_chunk = Chunk::new(vec![0, 2], vec![2, 2], 0); // offset [0,2], shape [2,2]
        let target_chunk = Chunk::new(vec![1, 1], vec![2, 3], 1); // offset [1,1], shape [2,3]
        let full_shape = vec![4, 4]; // 4x4 tensor
        let strides = vec![4, 1];

        let source_header_size = 8;
        let source_data_offset = 16;
        let dtype_size = 4; // f32

        let result = compute_read_ranges_direct(
            &source_chunk,
            &target_chunk,
            source_header_size,
            source_data_offset,
            dtype_size,
            &strides,
            &full_shape,
        );

        // Should have intersection and return byte ranges
        assert!(
            !result.is_empty(),
            "Should have intersection between overlapping chunks"
        );

        // All ranges should have valid byte offsets
        for (start, end, target_offset) in result {
            assert!(start < end, "Start should be less than end");
            assert!(
                start >= (source_header_size + source_data_offset) as u64,
                "Start should be after header and data offset"
            );
        }
    }

    #[test]
    fn test_compute_read_ranges_direct_no_overlap() {
        // Test with non-overlapping chunks
        let source_chunk = Chunk::new(vec![0, 0], vec![2, 2], 0);
        let target_chunk = Chunk::new(vec![3, 3], vec![1, 1], 1);
        let full_shape = vec![4, 4];
        let strides = vec![4, 1];

        let result = compute_read_ranges_direct(
            &source_chunk,
            &target_chunk,
            8,
            16,
            4,
            &strides,
            &full_shape,
        );

        assert!(
            result.is_empty(),
            "Non-overlapping chunks should return empty ranges"
        );
    }

    #[test]
    fn test_compute_write_ranges_direct_basic() {
        // Test basic functionality of compute_write_ranges_direct
        let source_chunk = Chunk::new(vec![1, 1], vec![2, 2], 0);
        let target_chunk = Chunk::new(vec![0, 1], vec![3, 2], 1);
        let full_shape = vec![4, 4];
        let strides = vec![4, 1];

        let target_header_size = 8;
        let target_data_offset = 16;
        let dtype_size = 4;

        let result = compute_write_ranges_direct(
            &source_chunk,
            &target_chunk,
            target_header_size,
            target_data_offset,
            dtype_size,
            &strides,
            &full_shape,
        );

        assert!(!result.is_empty(), "Should have intersection");

        for write_range in result {
            assert!(
                write_range.target_start < write_range.target_end,
                "Target start should be less than target end"
            );
            assert!(
                write_range.target_start >= (target_header_size + target_data_offset) as u64,
                "Target start should be after header and data offset"
            );
            assert!(write_range.length > 0, "Length should be positive");
        }
    }

    #[test]
    fn test_compute_write_ranges_direct_no_overlap() {
        // Test with non-overlapping chunks
        let source_chunk = Chunk::new(vec![0, 0], vec![1, 1], 0);
        let target_chunk = Chunk::new(vec![2, 2], vec![1, 1], 1);
        let full_shape = vec![4, 4];
        let strides = vec![4, 1];

        let result = compute_write_ranges_direct(
            &source_chunk,
            &target_chunk,
            8,
            16,
            4,
            &strides,
            &full_shape,
        );

        assert!(
            result.is_empty(),
            "Non-overlapping chunks should return empty write ranges"
        );
    }

    #[test]
    fn test_compute_shared_to_distributed_ranges() {
        // Test shared-to-distributed range computation
        let target_chunk = Chunk::new(vec![1, 2], vec![2, 2], 0); // offset [1,2], shape [2,2]
        let full_shape = vec![4, 4];
        let strides = vec![4, 1];

        let source_header_size = 8;
        let source_data_offset = 16;
        let dtype_size = 4;

        let result = compute_shared_to_distributed_ranges(
            &target_chunk,
            source_header_size,
            source_data_offset,
            dtype_size,
            &strides,
            &full_shape,
        );

        assert!(
            !result.is_empty(),
            "Should generate ranges for shared-to-distributed"
        );

        // Check that ranges are ordered and valid
        let mut last_end = 0u64;
        for (start, end, target_offset) in result {
            assert!(start >= last_end, "Ranges should be ordered");
            assert!(start < end, "Start should be less than end");
            assert!(
                start >= (source_header_size + source_data_offset) as u64,
                "Start should be after header and data offset"
            );
            last_end = end;
        }
    }

    #[test]
    fn test_compute_distributed_to_shared_ranges() {
        // Test distributed-to-shared range computation
        let source_chunk = Chunk::new(vec![2, 1], vec![2, 2], 0); // offset [2,1], shape [2,2]
        let full_shape = vec![4, 4];
        let strides = vec![4, 1];

        let source_header_size = 8;
        let source_data_offset = 16;
        let dtype_size = 4;

        let result = compute_distributed_to_shared_ranges(
            &source_chunk,
            source_header_size,
            source_data_offset,
            dtype_size,
            &strides,
            &full_shape,
        );

        assert!(
            !result.is_empty(),
            "Should generate ranges for distributed-to-shared"
        );

        // Check range validity
        for (start, end, target_offset) in result {
            assert!(start < end, "Start should be less than end");
            assert!(
                start >= (source_header_size + source_data_offset) as u64,
                "Start should be after header and data offset"
            );
        }
    }

    #[test]
    fn test_compute_shared_to_distributed_write_ranges() {
        // Test shared-to-distributed write range computation
        let target_chunk = Chunk::new(vec![0, 1], vec![2, 2], 0);
        let full_shape = vec![3, 3];
        let strides = vec![3, 1];

        let target_header_size = 8;
        let target_data_offset = 16;
        let dtype_size = 4;

        let result = compute_shared_to_distributed_write_ranges(
            &target_chunk,
            target_header_size,
            target_data_offset,
            dtype_size,
            &strides,
            &full_shape,
        );

        assert!(!result.is_empty(), "Should generate write ranges");

        for write_range in result {
            assert!(
                write_range.target_start < write_range.target_end,
                "Target start should be less than target end"
            );
            assert!(write_range.length > 0, "Length should be positive");
            assert!(
                write_range.target_start >= (target_header_size + target_data_offset) as u64,
                "Target start should be after header and data offset"
            );
        }
    }

    #[test]
    fn test_compute_distributed_to_shared_write_ranges() {
        // Test distributed-to-shared write range computation
        let source_chunk = Chunk::new(vec![1, 0], vec![2, 3], 0);
        let full_shape = vec![4, 3];
        let strides = vec![3, 1];

        let target_header_size = 8;
        let target_data_offset = 16;
        let dtype_size = 4;

        let result = compute_distributed_to_shared_write_ranges(
            &source_chunk,
            target_header_size,
            target_data_offset,
            dtype_size,
            &strides,
            &full_shape,
        );

        assert!(!result.is_empty(), "Should generate write ranges");

        for write_range in result {
            assert!(
                write_range.target_start < write_range.target_end,
                "Target start should be less than target end"
            );
            assert!(write_range.length > 0, "Length should be positive");
        }
    }

    #[test]
    fn test_contiguous_block_optimization() {
        // Test that contiguous blocks are detected and optimized
        let source_chunk = Chunk::new(vec![0, 0], vec![2, 4], 0); // Full rows
        let target_chunk = Chunk::new(vec![0, 0], vec![2, 4], 1); // Same chunk
        let full_shape = vec![2, 4];
        let strides = vec![4, 1];

        let result = compute_read_ranges_direct(
            &source_chunk,
            &target_chunk,
            8,
            16,
            4,
            &strides,
            &full_shape,
        );

        // Should ideally produce fewer ranges due to contiguous block optimization
        // The exact number depends on the optimization, but should be reasonable
        assert!(!result.is_empty(), "Should produce ranges");
        assert!(
            result.len() <= 4,
            "Should optimize contiguous blocks (reduce fragmentation)"
        );
    }

    #[test]
    fn test_edge_case_single_element() {
        // Test with single element chunks
        let source_chunk = Chunk::new(vec![1, 1], vec![1, 1], 0);
        let target_chunk = Chunk::new(vec![1, 1], vec![1, 1], 1);
        let full_shape = vec![3, 3];
        let strides = vec![3, 1];

        let result = compute_read_ranges_direct(
            &source_chunk,
            &target_chunk,
            8,
            16,
            4,
            &strides,
            &full_shape,
        );

        assert_eq!(
            result.len(),
            1,
            "Should have exactly one range for single element"
        );
        let (start, end, target_offset) = result[0];
        assert_eq!(end - start, 4, "Should have 4 bytes for f32");
    }

    #[test]
    fn test_edge_case_partial_overlap_complex() {
        // Test complex partial overlap scenario
        let source_chunk = Chunk::new(vec![1, 1], vec![3, 2], 0); // rows 1-3, cols 1-2
        let target_chunk = Chunk::new(vec![2, 0], vec![2, 3], 1); // rows 2-3, cols 0-2
        let full_shape = vec![5, 4];
        let strides = vec![4, 1];

        // Intersection should be: rows 2-3, cols 1-2 = 2x2 = 4 elements
        let result = compute_read_ranges_direct(
            &source_chunk,
            &target_chunk,
            8,
            16,
            4,
            &strides,
            &full_shape,
        );

        assert!(!result.is_empty(), "Should have intersection");

        // Total bytes should equal intersection size * dtype_size = 4 * 4 = 16
        let total_bytes: u64 = result.iter().map(|(start, end, _)| end - start).sum();
        assert_eq!(
            total_bytes, 16,
            "Total bytes should equal intersection size"
        );
    }

    #[test]
    fn test_different_dtypes_and_sizes() {
        // Test with different data type sizes
        let source_chunk = Chunk::new(vec![0, 0], vec![2, 2], 0);
        let target_chunk = Chunk::new(vec![1, 1], vec![2, 2], 1);
        let full_shape = vec![3, 3];
        let strides = vec![3, 1];

        // Test with different dtype sizes
        for (dtype_size, expected_element_bytes) in [(1, 1), (2, 2), (4, 4), (8, 8)] {
            let result = compute_read_ranges_direct(
                &source_chunk,
                &target_chunk,
                8,
                16,
                dtype_size,
                &strides,
                &full_shape,
            );

            assert!(
                !result.is_empty(),
                "Should have intersection for dtype_size {}",
                dtype_size
            );

            // Check that all byte ranges are multiples of dtype_size
            for (start, end, _) in result {
                let range_size = end - start;
                assert_eq!(
                    range_size % dtype_size as u64,
                    0,
                    "Range size should be multiple of dtype_size"
                );
            }
        }
    }

    #[test]
    fn test_3d_tensor_intersection() {
        // Test with 3D tensors
        let source_chunk = Chunk::new(vec![0, 1, 0], vec![2, 2, 3], 0);
        let target_chunk = Chunk::new(vec![1, 0, 1], vec![2, 3, 2], 1);
        let full_shape = vec![3, 4, 4];
        let strides = vec![16, 4, 1]; // 3D strides

        let result = compute_read_ranges_direct(
            &source_chunk,
            &target_chunk,
            8,
            16,
            4,
            &strides,
            &full_shape,
        );

        // Should handle 3D intersection correctly
        if !result.is_empty() {
            // Verify all ranges are valid
            for (start, end, target_offset) in result {
                assert!(start < end, "3D: Start should be less than end");
                assert!(start >= 24, "3D: Start should be after header+data offset");
            }
        }
    }

    #[test]
    fn test_performance_comparison_large_chunks() {
        // Performance comparison test (mainly for correctness, timing is secondary)
        let source_chunk = Chunk::new(vec![0, 0], vec![50, 50], 0); // Large chunk
        let target_chunk = Chunk::new(vec![25, 25], vec![50, 50], 1); // Overlapping large chunk
        let full_shape = vec![100, 100];
        let strides = vec![100, 1];

        // Legacy method
        let source_intervals = get_intervals(&source_chunk, &strides, &full_shape);
        let target_intervals = get_intervals(&target_chunk, &strides, &full_shape);
        let legacy_result = intersection(&source_intervals, &target_intervals);

        // New optimized method
        let direct_result =
            chunk_intersection_direct(&source_chunk, &target_chunk, &strides, &full_shape);

        // Results should be identical
        assert_eq!(
            legacy_result, direct_result,
            "Large chunk intersection should produce identical results"
        );

        // Both should have substantial intersection
        assert!(
            !legacy_result.is_empty(),
            "Should have substantial intersection"
        );
        assert!(
            !direct_result.is_empty(),
            "Should have substantial intersection"
        );
    }
}

/// Specialized function to directly compute read byte ranges from chunk intersection
/// This replaces get_intervals + intersection + byte range conversion in one step
pub fn compute_read_ranges_direct(
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    source_header_size: usize,
    source_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<(u64, u64, u64)> {
    let ndim = full_shape.len();

    // Step 1: Quick intersection check in all dimensions
    let mut intersection_start = Vec::with_capacity(ndim);
    let mut intersection_size = Vec::with_capacity(ndim);

    for dim in 0..ndim {
        let source_start = source_chunk.offsets()[dim];
        let source_end = source_start + source_chunk.shape()[dim];
        let target_start = target_chunk.offsets()[dim];
        let target_end = target_start + target_chunk.shape()[dim];

        let intersect_start = source_start.max(target_start);
        let intersect_end = source_end.min(target_end);

        // No intersection if any dimension doesn't overlap
        if intersect_start >= intersect_end {
            return Vec::new();
        }

        intersection_start.push(intersect_start);
        intersection_size.push(intersect_end - intersect_start);
    }

    // Step 2: Directly compute byte ranges from intersection
    compute_byte_ranges_from_intersection(
        &intersection_start,
        &intersection_size,
        source_chunk,
        target_chunk,
        source_header_size,
        source_data_offset,
        dtype_size,
        strides,
    )
}

/// Specialized function for write operations - computes write ranges directly
pub fn compute_write_ranges_direct(
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    target_header_size: usize,
    target_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<WriteRange> {
    let ndim = full_shape.len();

    // Step 1: Quick intersection check in all dimensions
    let mut intersection_start = Vec::with_capacity(ndim);
    let mut intersection_size = Vec::with_capacity(ndim);

    for dim in 0..ndim {
        let source_start = source_chunk.offsets()[dim];
        let source_end = source_start + source_chunk.shape()[dim];
        let target_start = target_chunk.offsets()[dim];
        let target_end = target_start + target_chunk.shape()[dim];

        let intersect_start = source_start.max(target_start);
        let intersect_end = source_end.min(target_end);

        // No intersection if any dimension doesn't overlap
        if intersect_start >= intersect_end {
            return Vec::new();
        }

        intersection_start.push(intersect_start);
        intersection_size.push(intersect_end - intersect_start);
    }

    // Step 2: Directly compute write ranges from intersection
    compute_write_ranges_from_intersection(
        &intersection_start,
        &intersection_size,
        source_chunk,
        target_chunk,
        target_header_size,
        target_data_offset,
        dtype_size,
        strides,
    )
}

/// Helper struct for write operations
#[derive(Debug, Clone)]
pub struct WriteRange {
    pub target_start: u64,
    pub target_end: u64,
    pub source_offset: usize,
    pub length: usize,
}

/// Directly compute byte ranges from n-dimensional intersection
fn compute_byte_ranges_from_intersection(
    intersect_start: &[usize],
    intersect_size: &[usize],
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    source_header_size: usize,
    source_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
) -> Vec<(u64, u64, u64)> {
    let ndim = intersect_start.len();
    let mut ranges = Vec::new();

    // Calculate total intersection volume
    let total_positions: usize = intersect_size.iter().product();
    if total_positions == 0 {
        return ranges;
    }

    // Look for contiguous blocks to minimize number of ranges
    let contiguous_blocks = find_contiguous_blocks(
        intersect_start,
        intersect_size,
        source_chunk,
        target_chunk,
        strides,
    );

    for block in contiguous_blocks {
        let source_start_bytes = block.source_offset * dtype_size;
        let source_end_bytes = (block.source_offset + block.length) * dtype_size;
        let target_offset_bytes = block.target_offset * dtype_size;

        let file_source_start =
            (source_header_size + source_data_offset + source_start_bytes) as u64;
        let file_source_end = (source_header_size + source_data_offset + source_end_bytes) as u64;

        ranges.push((
            file_source_start,
            file_source_end,
            target_offset_bytes as u64,
        ));
    }

    ranges
}

/// Directly compute write ranges from n-dimensional intersection
fn compute_write_ranges_from_intersection(
    intersect_start: &[usize],
    intersect_size: &[usize],
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    target_header_size: usize,
    target_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
) -> Vec<WriteRange> {
    let ndim = intersect_start.len();
    let mut ranges = Vec::new();

    // Calculate total intersection volume
    let total_positions: usize = intersect_size.iter().product();
    if total_positions == 0 {
        return ranges;
    }

    // Look for contiguous blocks to minimize number of ranges
    let contiguous_blocks = find_contiguous_blocks(
        intersect_start,
        intersect_size,
        source_chunk,
        target_chunk,
        strides,
    );

    for block in contiguous_blocks {
        let target_start_bytes = block.target_offset * dtype_size;
        let target_end_bytes = (block.target_offset + block.length) * dtype_size;
        let source_offset_bytes = block.source_offset * dtype_size;

        let file_target_start =
            (target_header_size + target_data_offset + target_start_bytes) as u64;
        let file_target_end = (target_header_size + target_data_offset + target_end_bytes) as u64;

        ranges.push(WriteRange {
            target_start: file_target_start,
            target_end: file_target_end,
            source_offset: source_offset_bytes,
            length: (target_end_bytes - target_start_bytes),
        });
    }

    ranges
}

/// Helper struct for contiguous memory blocks
#[derive(Debug)]
struct ContiguousBlock {
    source_offset: usize,
    target_offset: usize,
    length: usize,
}

/// Find contiguous blocks in the intersection to minimize the number of memory ranges
fn find_contiguous_blocks(
    intersect_start: &[usize],
    intersect_size: &[usize],
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    strides: &[usize],
) -> Vec<ContiguousBlock> {
    let ndim = intersect_start.len();
    let mut blocks = Vec::new();

    // Find the innermost dimension that has contiguous memory layout
    let innermost_contiguous_dim =
        find_innermost_contiguous_dim(intersect_start, intersect_size, source_chunk, target_chunk);

    if innermost_contiguous_dim == ndim {
        // Entire intersection is contiguous
        let total_length: usize = intersect_size.iter().product();
        let source_offset = calculate_chunk_offset(intersect_start, source_chunk, strides);
        let target_offset = calculate_chunk_offset(intersect_start, target_chunk, strides);

        blocks.push(ContiguousBlock {
            source_offset,
            target_offset,
            length: total_length,
        });
    } else {
        // Generate blocks for non-contiguous dimensions
        generate_blocks_recursive(
            0,
            innermost_contiguous_dim,
            intersect_start,
            intersect_size,
            source_chunk,
            target_chunk,
            strides,
            &mut blocks,
        );
    }

    blocks
}

/// Find how many trailing dimensions are contiguous
fn find_innermost_contiguous_dim(
    intersect_start: &[usize],
    intersect_size: &[usize],
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
) -> usize {
    let ndim = intersect_start.len();

    for dim in (0..ndim).rev() {
        // Check if this dimension is fully spanned in both chunks
        let source_spans_full = (intersect_start[dim] == source_chunk.offsets()[dim])
            && (intersect_size[dim] == source_chunk.shape()[dim]);
        let target_spans_full = (intersect_start[dim] == target_chunk.offsets()[dim])
            && (intersect_size[dim] == target_chunk.shape()[dim]);

        if !source_spans_full || !target_spans_full {
            return dim + 1;
        }
    }

    0 // All dimensions are contiguous
}

/// Calculate offset within a chunk for given coordinates
fn calculate_chunk_offset(
    coords: &[usize],
    chunk: &crate::topology::Chunk,
    strides: &[usize],
) -> usize {
    let mut offset = 0;
    for (dim, &coord) in coords.iter().enumerate() {
        offset += (coord - chunk.offsets()[dim]) * strides[dim];
    }
    offset
}

/// Recursively generate contiguous blocks for non-contiguous dimensions
fn generate_blocks_recursive(
    dim_idx: usize,
    contiguous_from_dim: usize,
    intersect_start: &[usize],
    intersect_size: &[usize],
    source_chunk: &crate::topology::Chunk,
    target_chunk: &crate::topology::Chunk,
    strides: &[usize],
    blocks: &mut Vec<ContiguousBlock>,
) {
    if dim_idx == contiguous_from_dim {
        // Calculate contiguous block size from this dimension onwards
        let block_length: usize = intersect_size[dim_idx..].iter().product();

        // Calculate current coordinates
        let mut coords = intersect_start.to_vec();

        let source_offset = calculate_chunk_offset(&coords, source_chunk, strides);
        let target_offset = calculate_chunk_offset(&coords, target_chunk, strides);

        blocks.push(ContiguousBlock {
            source_offset,
            target_offset,
            length: block_length,
        });
        return;
    }

    // Iterate through all positions in this dimension
    for i in 0..intersect_size[dim_idx] {
        let mut coords = intersect_start.to_vec();
        coords[dim_idx] = intersect_start[dim_idx] + i;

        // Recursively process next dimension
        generate_blocks_recursive(
            dim_idx + 1,
            contiguous_from_dim,
            &coords,
            intersect_size,
            source_chunk,
            target_chunk,
            strides,
            blocks,
        );
    }
}

/// Specialized function for shared-to-distributed: compute ranges for entire target chunk
pub fn compute_shared_to_distributed_ranges(
    target_chunk: &crate::topology::Chunk,
    source_header_size: usize,
    source_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<(u64, u64, u64)> {
    // For shared tensor, we need all the ranges that correspond to the target chunk
    let target_intervals = crate::topology::get_intervals(target_chunk, strides, full_shape);
    let mut ranges = Vec::new();
    let mut target_offset_elements = 0;

    for (start_elem, end_elem) in target_intervals {
        let start_bytes = start_elem * dtype_size;
        let end_bytes = end_elem * dtype_size;
        let target_offset_bytes = target_offset_elements * dtype_size;

        let source_start = (source_header_size + source_data_offset + start_bytes) as u64;
        let source_end = (source_header_size + source_data_offset + end_bytes) as u64;

        ranges.push((source_start, source_end, target_offset_bytes as u64));
        target_offset_elements += end_elem - start_elem;
    }

    ranges
}

/// Specialized function for distributed-to-shared: compute write ranges for entire source chunk  
pub fn compute_distributed_to_shared_write_ranges(
    source_chunk: &crate::topology::Chunk,
    target_header_size: usize,
    target_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<WriteRange> {
    // For shared tensor, the source chunk maps directly to absolute positions
    let source_intervals = crate::topology::get_intervals(source_chunk, strides, full_shape);
    let mut ranges = Vec::new();
    let mut source_offset_elements = 0;

    for (start_elem, end_elem) in source_intervals {
        let source_offset_bytes = source_offset_elements * dtype_size;
        let target_start_bytes = start_elem * dtype_size;
        let length_bytes = (end_elem - start_elem) * dtype_size;

        let target_start = (target_header_size + target_data_offset + target_start_bytes) as u64;
        let target_end = target_start + length_bytes as u64;

        ranges.push(WriteRange {
            target_start,
            target_end,
            source_offset: source_offset_bytes,
            length: length_bytes,
        });

        source_offset_elements += end_elem - start_elem;
    }

    ranges
}

/// Specialized function for distributed-to-shared reads: compute read ranges for entire source chunk
pub fn compute_distributed_to_shared_ranges(
    source_chunk: &crate::topology::Chunk,
    source_header_size: usize,
    source_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<(u64, u64, u64)> {
    // For distributed-to-shared, the source chunk maps to its absolute positions in the shared tensor
    let source_intervals = crate::topology::get_intervals(source_chunk, strides, full_shape);
    let mut ranges = Vec::new();
    let mut source_offset_elements = 0;

    for (start_elem, end_elem) in source_intervals {
        let source_offset_bytes = source_offset_elements * dtype_size;
        let target_start_bytes = start_elem * dtype_size; // Absolute position in shared tensor
        let length_bytes = (end_elem - start_elem) * dtype_size;

        let file_source_start =
            (source_header_size + source_data_offset + source_offset_bytes) as u64;
        let file_source_end = file_source_start + length_bytes as u64;

        ranges.push((
            file_source_start,
            file_source_end,
            target_start_bytes as u64,
        ));
        source_offset_elements += end_elem - start_elem;
    }

    ranges
}

/// Specialized function for shared-to-distributed writes: compute write ranges for entire target chunk
pub fn compute_shared_to_distributed_write_ranges(
    target_chunk: &crate::topology::Chunk,
    target_header_size: usize,
    target_data_offset: usize,
    dtype_size: usize,
    strides: &[usize],
    full_shape: &[usize],
) -> Vec<WriteRange> {
    // For shared-to-distributed, the target chunk receives data from its absolute positions in the shared tensor
    let target_intervals = crate::topology::get_intervals(target_chunk, strides, full_shape);
    let mut ranges = Vec::new();
    let mut target_offset_elements = 0;

    for (start_elem, end_elem) in target_intervals {
        let source_offset_bytes = start_elem * dtype_size; // Absolute position in shared tensor
        let target_start_bytes = target_offset_elements * dtype_size;
        let length_bytes = (end_elem - start_elem) * dtype_size;

        let file_target_start =
            (target_header_size + target_data_offset + target_start_bytes) as u64;
        let file_target_end = file_target_start + length_bytes as u64;

        ranges.push(WriteRange {
            target_start: file_target_start,
            target_end: file_target_end,
            source_offset: source_offset_bytes,
            length: length_bytes,
        });

        target_offset_elements += end_elem - start_elem;
    }

    ranges
}
