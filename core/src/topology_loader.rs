use crate::loader::Loader;
use crate::topology::{Chunk, Tensor, Topology, TopologyError, get_intervals};
use futures_util::stream::FuturesUnordered;
use hf_hub::api::tokio::ApiRepo;
use reqwest::header::RANGE;
use reqwest::{Client, Url};
use safetensors::{Dtype, SafeTensorError};
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use std::collections::HashMap;
use std::io::SeekFrom;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::fs;
use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;

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

    #[error("Channel send error: {0}")]
    SendError(#[from] tokio::sync::mpsc::error::SendError<Vec<u8>>),

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
    _saved_topology: &Topology,
    _loading_topology: &Topology,
    _rank: usize,
    _tensor_name: &str,
    _base_url: &str,
) -> Result<Vec<u8>, TopologyLoadError> {
    // TODO: Implement the loading logic
    todo!()
}

/// Creates a local safetensors file containing the tensors for the current rank
///
/// # Arguments
/// * `filename` - The path where to save the local safetensors file
/// * `local_topology` - The topology of the current loading process
/// * `rank` - The rank of the current process
pub async fn create_local_rank_file<P: AsRef<Path>>(
    filename: P,
    local_topology: &Topology,
    rank: usize,
    repo: &ApiRepo,
) -> Result<(), TopologyLoadError> {
    // Validate rank and fetch remote topology
    let remote_topology = validate_topologies(local_topology, rank, &repo).await?;

    // Fetch all safetensors files reported in the remote topology
    let loaders = fetch_remote_files(&remote_topology, &repo)?;

    // Spawn a future to generate the header and send it to the writer loop
    let _filename_clone = filename.as_ref().to_path_buf();
    let local_topology_clone = local_topology.clone();
    let (metadata, final_offset) = generate_header(&local_topology_clone, rank).await?;
    let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
    // Force alignment to 8 bytes.
    let extra = (8 - metadata_buf.len() % 8) % 8;
    metadata_buf.extend(vec![b' '; extra]);

    let n: u64 = metadata_buf.len() as u64;

    // Writer loop
    let total_length = 8 + metadata_buf.len() + final_offset;
    let mut file = tokio::fs::File::create(&filename).await?;
    file.set_len(total_length as u64).await?;
    file.flush().await?;
    file.write_all(n.to_le_bytes().as_ref()).await?;
    file.write_all(&metadata_buf).await?;

    let chunk_size = 10 * 1024 * 1024;
    let n_chunks = total_length / chunk_size;
    let mut handles = Vec::with_capacity(n_chunks);
    for request in fetch_requests(&local_topology, rank, &remote_topology, &loaders) {
        handles.push(tokio::spawn(async move {
            if let Err(err) = request.run().await {
                log::error!("Error fetching chunk {err}");
            }
        }));
    }
    let futures: FuturesUnordered<_> = handles.into_iter().collect();
    futures::future::join_all(futures).await;

    Ok(())
}

struct FetchRequest {
    url: Url,
    client: Client,
    range: (usize, usize),
    filename: PathBuf,
    writes: Vec<(usize, usize)>,
}

impl FetchRequest {
    async fn run(&self) -> Result<(), TopologyLoadError> {
        let response = self
            .client
            .get(self.url.clone())
            .header(RANGE, format!("{}-{}", self.range.0, self.range.1))
            .send()
            .await?;
        let bytes = response.bytes().await?;
        let data = bytes.to_vec();
        // pb.inc(data.len() as u64);
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(&self.filename)
            .await?;
        let mut byte_offset = 0;
        for (start, length) in &self.writes {
            file.seek(SeekFrom::Start(*start as u64)).await?;
            file.write_all(&data[byte_offset..byte_offset + length])
                .await?;
            byte_offset += length;
        }
        assert_eq!(byte_offset, data.len());
        Ok(())
    }
}

fn fetch_requests(
    local_topology: &Topology,
    rank: usize,
    remote_topology: &Topology,
    loaders: &[Loader],
) -> Vec<FetchRequest> {
    let mut requests = vec![];
    for (name, info) in local_topology.tensors() {
        match info {
            Tensor::Distributed(info) => {
                let chunk = &info.chunks()[rank];
                requests.extend(fetch_requests_single(
                    name,
                    info.shape(),
                    chunk,
                    remote_topology,
                    loaders,
                ));
            }
            _ => todo!(),
        }
    }
    requests
}

fn fetch_requests_single(
    name: &str,
    full_shape: &[usize],
    chunk: &Chunk,
    remote_topology: &Topology,
    loaders: &[Loader],
) -> Vec<FetchRequest> {
    let mut requests = vec![];

    let ndim = full_shape.len();
    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * full_shape[i + 1];
    }
    let local_intervals = get_intervals(chunk, &strides, full_shape);
    let remote_chunk = remote_topology.tensors().get(name).unwrap();
    match remote_chunk {
        Tensor::Distributed(info) => {
            for remote_chunk in info.chunks() {
                let remote_intervals = get_intervals(remote_chunk, &strides, full_shape);

                todo!("local {local_intervals:?} - remote {remote_intervals:?}");
            }
            assert_eq!(info.shape(), full_shape);
        }
        _ => todo!(),
    }

    requests
}

/// A single tensor information.
/// Endianness is assumed to be little endian
/// Ordering is assumed to be 'C'.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TensorInfo {
    /// The type of each element of the tensor
    pub dtype: Dtype,
    /// The shape of the tensor
    pub shape: Vec<usize>,
    /// The offsets to find the data within the byte-buffer array.
    pub data_offsets: (usize, usize),
}

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
#[derive(Debug, Clone)]
pub struct Metadata {
    metadata: Option<HashMap<String, String>>,
    tensors: Vec<TensorInfo>,
    index_map: HashMap<String, usize>,
}

/// Helper struct used only for serialization deserialization
#[derive(Serialize, Deserialize)]
struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hashdata: HashMetadata = HashMetadata::deserialize(deserializer)?;
        let (metadata, tensors) = (hashdata.metadata, hashdata.tensors);
        let mut tensors: Vec<_> = tensors.into_iter().collect();
        // We need to sort by offsets
        // Previous versions might have a different ordering
        // Than we expect (Not aligned ordered, but purely name ordered,
        // or actually any order).
        tensors.sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));
        Metadata::new(metadata, tensors).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Metadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut names = vec![""; self.index_map.len()];
        for (name, index) in &self.index_map {
            names[*index] = name;
        }

        let tensors: Vec<_> = names.iter().zip(self.tensors.iter()).collect();
        let length = if let Some(metadata) = &self.metadata {
            metadata.len()
        } else {
            0
        };
        let mut map = serializer.serialize_map(Some(tensors.len() + length))?;
        if let Some(metadata) = &self.metadata {
            map.serialize_entry("__metadata__", metadata)?;
        }
        for (name, info) in tensors {
            map.serialize_entry(&name, &info)?;
        }
        map.end()
    }
}

impl Metadata {
    fn new(
        metadata: Option<HashMap<String, String>>,
        tensors: Vec<(String, TensorInfo)>,
    ) -> Result<Self, SafeTensorError> {
        let mut index_map = HashMap::with_capacity(tensors.len());

        let tensors: Vec<_> = tensors
            .into_iter()
            .enumerate()
            .map(|(index, (k, tensor))| {
                index_map.insert(k, index);
                tensor
            })
            .collect();

        let metadata = Self {
            metadata,
            tensors,
            index_map,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<usize, SafeTensorError> {
        let mut start = 0;
        for (i, info) in self.tensors.iter().enumerate() {
            let (s, e) = info.data_offsets;
            if s != start || e < s {
                let tensor_name = self
                    .index_map
                    .iter()
                    .find_map(|(name, &index)| if index == i { Some(&name[..]) } else { None })
                    .unwrap_or("no_tensor");
                return Err(SafeTensorError::InvalidOffset(tensor_name.to_string()));
            }
            start = e;
            let nelements: usize = info
                .shape
                .iter()
                .cloned()
                .try_fold(1usize, usize::checked_mul)
                .ok_or(SafeTensorError::ValidationOverflow)?;
            let nbytes = nelements
                .checked_mul(info.dtype.size())
                .ok_or(SafeTensorError::ValidationOverflow)?;
            if (e - s) != nbytes {
                return Err(SafeTensorError::TensorInvalidInfo);
            }
        }
        Ok(start)
    }

    /// Gives back the tensor metadata
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let index = self.index_map.get(name)?;
        self.tensors.get(*index)
    }

    /// Gives back the tensor metadata
    pub fn tensors(&self) -> HashMap<String, &TensorInfo> {
        self.index_map
            .iter()
            .map(|(tensor_name, index)| (tensor_name.clone(), &self.tensors[*index]))
            .collect()
    }

    /// Gives back the tensor metadata
    pub fn metadata(&self) -> &Option<HashMap<String, String>> {
        &self.metadata
    }
}

/// Private helper function to generate the local safetensors header using the local topology.
async fn generate_header(
    local_topology: &Topology,
    rank: usize,
) -> Result<(Metadata, usize), TopologyLoadError> {
    let mut metadata = Metadata {
        metadata: None,
        tensors: Vec::with_capacity(local_topology.tensors().len()),
        index_map: HashMap::new(),
    };
    let mut current_index = 0;
    let mut current_offset = 0;
    for (name, tensor) in local_topology.tensors() {
        match tensor {
            Tensor::Distributed(info) => {
                let chunk = &info.chunks()[rank];
                metadata.index_map.insert(name.to_string(), current_index);
                current_index += 1;
                let dtype = info.dtype();
                let shape = chunk.shape().to_vec();

                let length = shape.iter().product::<usize>() * dtype.size();
                let data_offsets = (current_offset, current_offset + length);
                current_offset += length;
                metadata.tensors.push(TensorInfo {
                    dtype,
                    shape,
                    data_offsets,
                });
            }
            Tensor::Shared(info) => {
                metadata.index_map.insert(name.to_string(), current_index);
                current_index += 1;
                let dtype = info.dtype();
                let shape = info.shape().to_vec();
                let length = shape.iter().product::<usize>() * dtype.size();
                let data_offsets = (current_offset, current_offset + length);
                current_offset += length;
                metadata.tensors.push(TensorInfo {
                    dtype,
                    shape,
                    data_offsets,
                });
            }
        }
    }

    Ok((metadata, current_offset))
}

/// Private helper function to fetch metadata for all safetensors files reported in the remote topology.
fn fetch_remote_files(
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
        local_topology.tensors().iter().map(|(k, _)| k).collect();
    let remote_tensor_names: std::collections::HashSet<_> =
        remote_topology.tensors().iter().map(|(k, _)| k).collect();

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
    use crate::topology::{Chunk, DistributedInfo, SimpleTopo, Tensor};
    use hf_hub::api::tokio::ApiBuilder;
    use safetensors::Dtype;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_topology_merge() {
        // Create a temporary directory for our test
        let temp_dir = tempdir().unwrap();
        let output_file = temp_dir.path().join("output.safetensors");

        // Create a simple local topology with one distributed tensor
        let mut local_tensors = HashMap::new();
        let local_chunk1 = Chunk::new(vec![0, 0], vec![2, 4], 0);
        let local_chunk2 = Chunk::new(vec![2, 0], vec![2, 4], 1);
        let local_info =
            DistributedInfo::new(vec![4, 4], Dtype::F32, vec![local_chunk1, local_chunk2]);
        local_tensors.insert("tensor1".to_string(), Tensor::Distributed(local_info));
        let local_topology = Topology::try_from(SimpleTopo::new(
            local_tensors,
            vec![
                "local1.safetensors".to_string(),
                "local2.safetensors".to_string(),
            ],
            2,
        ))
        .unwrap();

        // Create a simple remote topology with the same tensor but different chunks
        let mut remote_tensors = HashMap::new();
        let remote_chunk1 = Chunk::new(vec![0, 0], vec![2, 4], 0);
        let remote_chunk2 = Chunk::new(vec![2, 0], vec![2, 4], 1);
        let remote_info =
            DistributedInfo::new(vec![4, 4], Dtype::F32, vec![remote_chunk1, remote_chunk2]);
        remote_tensors.insert("tensor1".to_string(), Tensor::Distributed(remote_info));
        let remote_topology = Topology::try_from(SimpleTopo::new(
            remote_tensors,
            vec![
                "remote1.safetensors".to_string(),
                "remote2.safetensors".to_string(),
            ],
            2,
        ))
        .unwrap();

        let api = ApiBuilder::new()
            .with_endpoint("http://localhost".to_string())
            .build()
            .unwrap();
        // Create a mock repo that returns our remote topology
        let repo = api.model("test/repo".to_string());

        // Try to create the local rank file
        let result = create_local_rank_file(&output_file, &local_topology, 0, &repo).await;
        assert!(
            result.is_ok(),
            "Failed to create local rank file: {:?}",
            result.err()
        );
    }
}
