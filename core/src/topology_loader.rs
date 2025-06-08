use crate::loader::Loader;
use crate::topology::{Chunk, Tensor, Topology, TopologyError, get_intervals};
use futures_util::stream::FuturesUnordered;
use hf_hub::api::tokio::ApiRepo;
use reqwest::header::RANGE;
use reqwest::{Client, Url};
use safetensors::SafeTensorError;
use safetensors::tensor::{Metadata, TensorInfo};
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

    #[error("Creating safetensors data : {0}")]
    SafeTensorError(#[from] SafeTensorError),
}

/// Load a topology from a local JSON file
pub async fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Topology, TopologyLoadError> {
    let contents = fs::read_to_string(path).await?;
    let topology: Topology = serde_json::from_str(&contents)?;
    Ok(topology)
}

async fn get_remote_topology(repo: &ApiRepo) -> Result<Topology, TopologyLoadError> {
    // Fetch remote topology
    let filename = repo.get("topology.json").await?;
    let content = tokio::fs::read_to_string(filename).await?;
    let remote_topology: Topology = serde_json::from_str(&content)?;
    Ok(remote_topology)
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
    let remote_topology = get_remote_topology(&repo).await?;
    validate_topologies(local_topology, &remote_topology)?;

    // Fetch all safetensors files reported in the remote topology
    let metadatas = fetch_remote_files(&remote_topology, &repo).await?;

    // Spawn a future to generate the header and send it to the writer loop
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
    for request in fetch_requests(&local_topology, rank, &remote_topology, &metadatas) {
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
    metadatas: &[(Metadata, usize)],
) -> Vec<FetchRequest> {
    let mut requests = vec![];
    let mut local_offset = 0;
    for (name, info) in local_topology.tensors() {
        match info {
            Tensor::Distributed(info) => {
                let chunk = &info.chunks()[rank];
                requests.extend(fetch_requests_single(
                    name,
                    info.shape(),
                    chunk,
                    remote_topology,
                    metadatas,
                    &mut local_offset,
                ));
            }
            Tensor::Shared(_info) => {
                requests.extend(fetch_shared(
                    name,
                    remote_topology,
                    metadatas,
                    &mut local_offset,
                ));
            }
        }
    }
    requests
}

fn fetch_shared(
    name: &str,
    remote_topology: &Topology,
    metadatas: &[(Metadata, usize)],
    local_offset: &mut usize,
) -> Vec<FetchRequest> {
    let mut requests = vec![];
    let info = remote_topology.tensors().get(name).unwrap();
    match info {
        Tensor::Distributed(_) => unreachable!(),
        Tensor::Shared(info) => {
            let (metadata, remote_offset) = &metadatas[0];
            let remote_tensor_info = (*metadata.tensors().get(name).unwrap()).clone();
            let (remote_start, remote_stop) = remote_tensor_info.data_offsets;
            // TODO
            let length = remote_stop - remote_start;
            let request = FetchRequest {
                client: Client::new(),
                // TODO we need the local filename
                filename: remote_topology.filenames()[info.filename_indices()[0]]
                    .clone()
                    .into(),
                range: (remote_start + remote_offset, remote_stop + remote_offset),
                writes: vec![(*local_offset, length + *local_offset)],
                url: Url::parse("http://localhost").unwrap(),
            };
            *local_offset += length;
            requests.push(request);
        }
    }
    requests
}

fn fetch_requests_single(
    name: &str,
    full_shape: &[usize],
    chunk: &Chunk,
    remote_topology: &Topology,
    metadatas: &[(Metadata, usize)],
    local_offset: &mut usize,
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
            for (remote_rank, remote_chunk) in info.chunks().iter().enumerate() {
                let remote_intervals = get_intervals(remote_chunk, &strides, full_shape);

                if local_intervals == remote_intervals {
                    for (start, stop) in local_intervals {
                        let (_metadata, remote_offset) = &metadatas[remote_rank];
                        let length = stop - start;
                        let request = FetchRequest {
                            client: Client::new(),
                            // TODO we need the local filename
                            filename: remote_topology.filenames()[remote_chunk.filename_index()]
                                .clone()
                                .into(),
                            range: (start + remote_offset, stop + remote_offset),
                            writes: vec![(start + *local_offset, stop + *local_offset)],
                            url: Url::parse("http://localhost").unwrap(),
                        };
                        *local_offset += length;
                        requests.push(request);
                    }
                    return requests;
                }
                todo!("local {local_intervals:?} - remote {remote_intervals:?}");
            }
            assert_eq!(info.shape(), full_shape);
        }
        Tensor::Shared(_info) => panic!("Topology of shared/distributed should be consistent"),
    }

    requests
}

/// Private helper function to generate the local safetensors header using the local topology.
async fn generate_header(
    local_topology: &Topology,
    rank: usize,
) -> Result<(Metadata, usize), TopologyLoadError> {
    let mut tensors = Vec::with_capacity(local_topology.tensors().len());
    let mut current_offset = 0;
    for (name, tensor) in local_topology.tensors() {
        match tensor {
            Tensor::Distributed(info) => {
                let chunk = &info.chunks()[rank];
                let dtype = info.dtype();
                let shape = chunk.shape().to_vec();

                let length = shape.iter().product::<usize>() * dtype.size();
                let data_offsets = (current_offset, current_offset + length);
                current_offset += length;
                tensors.push((
                    name.to_string(),
                    TensorInfo {
                        dtype,
                        shape,
                        data_offsets,
                    },
                ));
            }
            Tensor::Shared(info) => {
                let dtype = info.dtype();
                let shape = info.shape().to_vec();
                let length = shape.iter().product::<usize>() * dtype.size();
                let data_offsets = (current_offset, current_offset + length);
                current_offset += length;
                tensors.push((
                    name.to_string(),
                    TensorInfo {
                        dtype,
                        shape,
                        data_offsets,
                    },
                ));
            }
        }
    }
    let metadata = Metadata::new(None, tensors)?;

    Ok((metadata, current_offset))
}

/// Private helper function to fetch metadata for all safetensors files reported in the remote topology.
async fn fetch_remote_files(
    remote_topology: &Topology,
    repo: &ApiRepo,
) -> Result<Vec<(Metadata, usize)>, TopologyLoadError> {
    let mut metadatas = Vec::with_capacity(remote_topology.filenames().len());
    for filename in remote_topology.filenames() {
        let url = match Url::parse(&repo.url(filename)) {
            Ok(url) => url,
            Err(err) => return Err(TopologyLoadError::from(err)), // Or whatever your error type is
        };

        let mut loader = match Loader::new(url) {
            Ok(loader) => loader,
            Err(err) => return Err(TopologyLoadError::from(err)),
        };

        // The result is a future, which we now wrap in `Ok`
        metadatas.push(loader.metadata().await?.clone());
    }
    Ok(metadatas)
}

/// Private helper function to validate the rank and check that the tensors match between local and remote topologies.
fn validate_topologies(
    local_topology: &Topology,
    remote_topology: &Topology,
) -> Result<(), TopologyLoadError> {
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
            _ => todo!(),
        };

        if local_shape != remote_shape {
            return Err(TopologyLoadError::DifferentTensorShapes(
                name.clone(),
                local_shape.to_vec(),
                remote_shape.to_vec(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{Chunk, DistributedInfo, SimpleTopo, Tensor};
    use safetensors::Dtype;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use tokio::fs;

    #[tokio::test]
    async fn test_topology_merge() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dir = Arc::new(temp_dir);

        // Create a simple local topology
        let mut local_tensors = BTreeMap::new();
        let local_tensor = Tensor::Distributed(DistributedInfo::new(
            vec![4, 4],
            Dtype::F32,
            vec![Chunk::new(vec![0, 0], vec![4, 4], 0)], // One chunk covering the entire tensor
        ));
        local_tensors.insert("test_tensor".to_string(), local_tensor);
        let local_topology = Topology::try_from(SimpleTopo::new(
            local_tensors,
            vec!["local1.safetensors".to_string()],
            1, // Only one rank
        ))
        .unwrap();

        // Create a simple remote topology
        let mut remote_tensors = BTreeMap::new();
        let remote_tensor = Tensor::Distributed(DistributedInfo::new(
            vec![4, 4],
            Dtype::F32,
            vec![Chunk::new(vec![0, 0], vec![4, 4], 0)], // One chunk covering the entire tensor
        ));
        remote_tensors.insert("test_tensor".to_string(), remote_tensor);
        let remote_topology = Topology::try_from(SimpleTopo::new(
            remote_tensors,
            vec!["remote1.safetensors".to_string()],
            1, // Only one rank
        ))
        .unwrap();

        // Write the remote topology to a file
        let topology_path = temp_dir.path().join("topology.json");
        fs::write(
            &topology_path,
            serde_json::to_string(&remote_topology).unwrap(),
        )
        .await
        .unwrap();

        // Create dummy safetensors files
        let dummy_data = vec![0u8; 1024]; // 1KB of zeros
        fs::write(temp_dir.path().join("remote1.safetensors"), &dummy_data)
            .await
            .unwrap();
        fs::write(temp_dir.path().join("remote2.safetensors"), &dummy_data)
            .await
            .unwrap();

        // Wait for the server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Merge the topologies
        validate_topologies(&local_topology, &remote_topology).unwrap();

        // Verify the merged topology
        assert_eq!(remote_topology.tensors().len(), 1);
        let tensor = remote_topology.get_tensor("test_tensor").unwrap();
        match tensor {
            Tensor::Distributed(info) => {
                assert_eq!(info.shape(), &[4, 4]);
                assert_eq!(info.dtype(), Dtype::F32);
                assert_eq!(info.chunks().len(), 1);
                let chunk = &info.chunks()[0];
                assert_eq!(chunk.offsets(), &[0, 0]);
                assert_eq!(chunk.shape(), &[4, 4]);
                assert_eq!(chunk.filename_index(), 0);
            }
            _ => panic!("Expected distributed tensor"),
        }
    }
}
