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
    let mut result = Vec::new();

    let mut soffset = 0;
    let mut toffset = 0;
    let mut sindex = 0;
    let mut tindex = 0;

    while sindex < source_intervals.len() && tindex < target_intervals.len() {
        let (source_start, source_end) = &source_intervals[sindex];
        let (target_start, target_end) = &target_intervals[tindex];
        let intersection_start = (*source_start).max(*target_start);
        let intersection_end = (*source_end).min(*target_end);

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
