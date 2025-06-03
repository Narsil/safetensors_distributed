use crate::loader::Loader;
use crate::topology::{Tensor, Topology, TopologyError};
use hf_hub::api::tokio::{Api, ApiRepo};
use reqwest::Url;
use std::ops::Deref;
use std::path::Path;
use thiserror::Error;
use tokio::fs;

/// Represents a HuggingFace Hub model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: String) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl Deref for ModelId {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

#[derive(Debug, Error)]
pub enum TopologyLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Topology validation error: {0}")]
    Topology(#[from] TopologyError),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("HF Hub API error: {0}")]
    HfHub(#[from] hf_hub::api::tokio::ApiError),

    #[error("Url parsing error: {0}")]
    UrlError(#[from] url::ParseError),

    #[error("Loader error: {0}")]
    Loader(#[from] crate::loader::Error),

    #[error("Invalid rank: {0}")]
    InvalidRank(usize),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Invalid tensor type for rank {0}: expected distributed tensor")]
    InvalidTensorType(usize),

    #[error("Remote topology has different tensor names: {0:?}")]
    DifferentTensorNames(Vec<String>),

    #[error("Tensor {0} has different shapes: local={1:?}, remote={2:?}")]
    DifferentTensorShapes(String, Vec<usize>, Vec<usize>),
}

/// Load a topology from a local JSON file
pub async fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Topology, TopologyLoadError> {
    let contents = fs::read_to_string(path).await?;
    let topology: Topology = serde_json::from_str(&contents)?;
    Ok(topology)
}

/// Loads a tensor's data for a specific rank from a repository
///
/// # Arguments
/// * `saved_topology` - The topology as saved in the repository
/// * `loading_topology` - The topology of the current loading process
/// * `rank` - The rank of the current process
/// * `tensor_name` - The name of the tensor to load
/// * `base_url` - The base URL of the repository
pub async fn load_tensor_for_rank(
    saved_topology: &Topology,
    loading_topology: &Topology,
    rank: usize,
    tensor_name: &str,
    base_url: &str,
) -> Result<Vec<u8>, TopologyLoadError> {
    // TODO: Implement the loading logic
    todo!()
}

/// Creates a local safetensors file containing the tensors for the current rank
///
/// # Arguments
/// * `filename` - The path where to save the local safetensors file
/// * `model_id` - The HuggingFace Hub model identifier
/// * `local_topology` - The topology of the current loading process
/// * `rank` - The rank of the current process
pub async fn create_local_rank_file<P: AsRef<Path>>(
    filename: P,
    model_id: ModelId,
    local_topology: &Topology,
    rank: usize,
    api: &Api,
) -> Result<(), TopologyLoadError> {
    // Validate rank and fetch remote topology
    let repo = api.model(model_id.as_str().to_string());
    let remote_topology = validate_topologies(local_topology, rank, &repo).await?;

    // Fetch all safetensors files reported in the remote topology
    let loaders = fetch_remote_files(&remote_topology, &repo).await?;

    // TODO: Implement the rest of the function
    todo!()
}

/// Private helper function to fetch metadata for all safetensors files reported in the remote topology.
async fn fetch_remote_files(
    remote_topology: &Topology,
    repo: &ApiRepo,
) -> Result<Vec<Loader>, TopologyLoadError> {
    remote_topology
        .filenames()
        .into_iter()
        .map(|filename| -> Result<Loader, TopologyLoadError> {
            let url = repo.url(filename);
            Ok(Loader::new(Url::parse(&url)?)?)
        })
        .collect::<Result<Vec<Loader>, TopologyLoadError>>()
}

/// Private helper function to validate the rank and check that the tensors match between local and remote topologies.
async fn validate_topologies(
    local_topology: &Topology,
    rank: usize,
    repo: &ApiRepo,
) -> Result<Topology, TopologyLoadError> {
    // Validate rank
    if rank >= local_topology.n_ranks() {
        return Err(TopologyLoadError::InvalidRank(rank));
    }

    // Fetch remote topology
    let filename = repo.get("topology.json").await?;
    let content = tokio::fs::read_to_string(filename).await?;
    let remote_topology: Topology = serde_json::from_str(&content)?;

    // Validate that tensors match between local and remote topologies
    let local_tensor_names: std::collections::HashSet<_> =
        local_topology.tensors().map(|(k, _)| k).collect();
    let remote_tensor_names: std::collections::HashSet<_> =
        remote_topology.tensors().map(|(k, _)| k).collect();

    // Check for missing or extra tensors
    if local_tensor_names != remote_tensor_names {
        let missing: Vec<_> = local_tensor_names
            .difference(&remote_tensor_names)
            .map(|s| s.to_string())
            .collect();
        let extra: Vec<_> = remote_tensor_names
            .difference(&local_tensor_names)
            .map(|s| s.to_string())
            .collect();
        let mut differences = Vec::new();
        if !missing.is_empty() {
            differences.push(format!("missing: {}", missing.join(", ")));
        }
        if !extra.is_empty() {
            differences.push(format!("extra: {}", extra.join(", ")));
        }
        return Err(TopologyLoadError::DifferentTensorNames(differences));
    }

    // Check that tensor shapes match
    for (name, local_tensor) in local_topology.tensors() {
        let remote_tensor = remote_topology
            .get_tensor(name)
            .ok_or_else(|| TopologyLoadError::TensorNotFound(name.clone()))?;

        let (local_shape, remote_shape) = match (local_tensor, remote_tensor) {
            (Tensor::Distributed(local), Tensor::Distributed(remote)) => {
                (local.shape(), remote.shape())
            }
            (Tensor::Shared(local), Tensor::Shared(remote)) => (local.shape(), remote.shape()),
            _ => return Err(TopologyLoadError::InvalidTensorType(rank)),
        };

        if local_shape != remote_shape {
            return Err(TopologyLoadError::DifferentTensorShapes(
                name.clone(),
                local_shape.to_vec(),
                remote_shape.to_vec(),
            ));
        }
    }

    Ok(remote_topology)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{Chunk, DistributedInfo, Tensor};
    use safetensors::Dtype;
    use std::collections::HashMap;

    // TODO: Add tests
}
