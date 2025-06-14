pub mod core;
pub mod location;
pub mod task;

use crate::topology::{Topology, TopologyError};
use indicatif::style::TemplateError;
use safetensors::tensor::Metadata;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

// Re-export main types
pub use core::Redistributor;

/// Structure for deserializing model.safetensors.index.json
#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    /// Map of tensor names to their containing file
    pub weight_map: HashMap<String, String>,
}

/// Layout containing topology and metadata information
#[derive(Clone)]
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

    #[error("Worker thread panicked")]
    ThreadPanic,
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

/// Given a directory, reads the topology.json file as a [`Topology`] or creates a "full" one from
/// the present safetensors file if any. All tensors will be considered [`crate::topology::Tensor::Shared`] then.
pub fn load_or_create_topology<P: AsRef<std::path::Path>>(dir: P) -> Result<Topology> {
    let dir = dir.as_ref();
    let topology_path = dir.join("topology.json");

    if topology_path.exists() {
        // Load existing topology
        let content = std::fs::read_to_string(topology_path)?;
        let topology: Topology = serde_json::from_str(&content)?;
        return Ok(topology);
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

    // Conservative fix: if intersection spans multiple elements in multiple dimensions,
    // be cautious about contiguity to avoid the bug where non-contiguous memory
    // regions are treated as contiguous blocks
    let multi_dim_intersection = intersect_size.iter().filter(|&&size| size > 1).count() > 1;

    if multi_dim_intersection {
        // Only treat the innermost dimension as potentially contiguous
        // This prevents incorrect merging of memory regions that have gaps
        // (like when splitting along dimension 1 in a 2D tensor)
        return ndim - 1;
    }

    // Original logic for simple cases (single dimension or single element intersections)
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
    _full_tensor_strides: &[usize],
) -> usize {
    let mut offset = 0;

    // Calculate strides for this specific chunk based on its shape
    let ndim = chunk.shape().len();
    let mut chunk_strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk.shape()[i + 1];
    }

    for (dim, &coord) in coords.iter().enumerate() {
        let chunk_coord = coord - chunk.offsets()[dim];
        let contribution = chunk_coord * chunk_strides[dim];
        offset += contribution;
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

        // Use the current coordinates (passed down from recursion)
        let coords = intersect_start;

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

        // Recursively process next dimension with updated coordinates
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Chunk;

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
        for (start, end, _target_offset) in result {
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
        for (start, end, _target_offset) in result {
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
        for (start, end, _target_offset) in result {
            assert!(start < end, "Start should be less than end");
            assert!(
                start >= (source_header_size + source_data_offset) as u64,
                "Start should be after header and data offset"
            );
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
        let (start, end, _target_offset) = result[0];
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
        for (dtype_size, _expected_element_bytes) in [(1, 1), (2, 2), (4, 4), (8, 8)] {
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
            for (start, end, _target_offset) in result {
                assert!(start < end, "3D: Start should be less than end");
                assert!(start >= 24, "3D: Start should be after header+data offset");
            }
        }
    }

    #[test]
    fn test_find_contiguous_blocks_dimension_1_split_bug() {
        // This test reproduces the exact bug: incorrect contiguous block calculation
        // for tensors split along dimension 1 (columns)

        use crate::topology::Chunk;

        // Tensor: 2x8 with strides [8, 1]
        let full_shape = [2, 8];
        let strides = [8, 1];

        // Source chunk: [0, 4] with shape [2, 4] (columns 4-7 of full tensor)
        let source_chunk = Chunk::new(vec![0, 4], vec![2, 4], 1);

        // Target chunk: [0, 0] with shape [2, 8] (full tensor)
        let target_chunk = Chunk::new(vec![0, 0], vec![2, 8], 0);

        // Intersection: starts at [0, 4], size [2, 4]
        let intersect_start = [0, 4];
        let intersect_size = [2, 4];

        println!("Test case:");
        println!("  Full shape: {:?}, strides: {:?}", full_shape, strides);
        println!(
            "  Source chunk: offsets={:?}, shape={:?}",
            source_chunk.offsets(),
            source_chunk.shape()
        );
        println!(
            "  Target chunk: offsets={:?}, shape={:?}",
            target_chunk.offsets(),
            target_chunk.shape()
        );
        println!(
            "  Intersection: start={:?}, size={:?}",
            intersect_start, intersect_size
        );

        let blocks = super::find_contiguous_blocks(
            &intersect_start,
            &intersect_size,
            &source_chunk,
            &target_chunk,
            &strides,
        );

        println!("  Computed blocks: {:?}", blocks);

        // The bug: it creates 1 block of length 8 instead of 2 blocks of length 4
        // Expected: 2 separate blocks (one per row)
        // - Block 1: Row 0, cols 4-7 → source_offset=0, target_offset=4, length=4
        // - Block 2: Row 1, cols 4-7 → source_offset=4, target_offset=12, length=4

        // Actual (buggy): 1 block → source_offset=0, target_offset=4, length=8
        // This causes row 1 data to overwrite part of row 0!

        assert_eq!(blocks.len(), 2, "Should have 2 blocks, one per row");

        // Block 0: Row 0, columns 4-7
        assert_eq!(blocks[0].source_offset, 0);
        assert_eq!(blocks[0].target_offset, 4);
        assert_eq!(blocks[0].length, 4);

        // Block 1: Row 1, columns 4-7
        assert_eq!(blocks[1].source_offset, 4);
        assert_eq!(blocks[1].target_offset, 12); // Row 1 starts at offset 8, plus 4 columns = 12
        assert_eq!(blocks[1].length, 4);
    }
}
