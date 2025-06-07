use crate::tensor::TensorData;
use crate::topology::{SharedInfo, Tensor, Topology, TopologyError, get_intervals};
use futures::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use safetensors::tensor::{Metadata, TensorInfo};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Semaphore;

/// Structure for deserializing model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    /// Map of tensor names to their containing file
    weight_map: HashMap<String, String>,
}

/// Error type for redistributor operations
#[derive(Debug, Error)]
pub enum RedistributorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Topology error: {0}")]
    Topology(#[from] TopologyError),

    #[error("Tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),

    #[error("Semaphore acquire error: {0}")]
    SemaphoreAcquire(#[from] tokio::sync::AcquireError),

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
}

/// Result type for redistributor operations
pub type Result<T> = std::result::Result<T, RedistributorError>;

/// Represents a tensor processing and writing task
#[derive(Debug, Clone)]
pub struct TensorTask {
    pub tensor_name: String,
    pub rank: usize,
    pub file_offset: usize,
}

/// Pre-calculated rank file information
#[derive(Debug)]
pub struct RankFileInfo {
    pub rank: usize,
    pub header: Vec<u8>,
    pub total_size: usize,
    pub tensor_tasks: Vec<TensorTask>,
}

/// Data source for tensor redistribution
pub enum TensorDataSource {
    /// Source data comes from reconstructing distributed tensors from files
    Reconstruct {
        source_topology: Topology,
        input_dir: std::path::PathBuf,
    },
    /// Source data comes from already loaded tensor data (for create_distributed case)
    Loaded {
        tensors: HashMap<String, TensorData>,
    },
}

/// Async streaming tensor redistributor with parallel processing and pre-calculated offsets
pub struct AsyncTensorRedistributor {
    data_source: TensorDataSource,
    target_topology: Topology,
    file_semaphore: Arc<Semaphore>,
}

impl AsyncTensorRedistributor {
    /// Create a new redistributor for reconstruction from distributed files
    pub fn new_from_topology<P: AsRef<Path>>(
        input_dir: P,
        target_topology: Topology,
    ) -> Result<Self> {
        // Load the existing topology (or create from model.safetensors)
        let source_topology = load_or_create_topology(&input_dir)?;

        let input_dir = input_dir.as_ref().to_path_buf();
        let max_concurrent_files = 50;
        Ok(Self {
            data_source: TensorDataSource::Reconstruct {
                source_topology,
                input_dir,
            },
            target_topology,
            file_semaphore: Arc::new(Semaphore::new(max_concurrent_files)),
        })
    }

    /// Redistribute tensors to target directory and return list of created files
    pub async fn redistribute<P: AsRef<Path>>(&self, output_dir: P) -> Result<Vec<String>> {
        let output_dir = output_dir.as_ref();
        tokio::fs::create_dir_all(output_dir).await?;

        // Pre-calculate all file layouts and offsets
        let rank_file_infos = self.pre_calculate_file_layout()?;

        // Create initial files with headers
        self.create_files_with_headers(output_dir, &rank_file_infos)
            .await?;

        // Process and write all tensors in parallel
        self.process_all_tensors(output_dir, &rank_file_infos)
            .await?;

        // Collect created safetensors files
        let mut created_files = Vec::new();
        for filename in self.target_topology.filenames() {
            created_files.push(filename.clone());
        }

        // Write topology.json if needed (for multi-rank outputs)
        if self.target_topology.n_ranks() > 1 {
            let topology_json = serde_json::to_vec_pretty(&self.target_topology)?;
            tokio::fs::write(output_dir.join("topology.json"), &topology_json).await?;
            created_files.push("topology.json".to_string());
        }

        Ok(created_files)
    }

    /// Pre-calculate all headers, offsets, and file structures based on target topology
    fn pre_calculate_file_layout(&self) -> Result<Vec<RankFileInfo>> {
        let mut rank_tensor_info: Vec<Vec<(String, TensorInfo)>> =
            (0..self.target_topology.n_ranks())
                .map(|_| Vec::new())
                .collect();
        let mut rank_offsets = vec![0usize; self.target_topology.n_ranks()];
        let mut tensor_tasks = Vec::new();

        // Process each tensor according to the target topology
        for (tensor_name, target_tensor) in self.target_topology.tensors() {
            match target_tensor {
                Tensor::Distributed(dist_info) => {
                    // Process distributed tensor - each rank gets a chunk
                    for (rank, chunk) in dist_info.chunks().iter().enumerate() {
                        let chunk_shape = chunk.shape().to_vec();
                        let chunk_size =
                            chunk_shape.iter().product::<usize>() * dist_info.dtype().size();

                        let tensor_info = TensorInfo {
                            dtype: dist_info.dtype(),
                            shape: chunk_shape,
                            data_offsets: (rank_offsets[rank], rank_offsets[rank] + chunk_size),
                        };

                        rank_tensor_info[rank].push((tensor_name.clone(), tensor_info));

                        tensor_tasks.push(TensorTask {
                            tensor_name: tensor_name.clone(),
                            rank,
                            file_offset: rank_offsets[rank],
                        });

                        rank_offsets[rank] += chunk_size;
                    }
                }
                Tensor::Shared(shared_info) => {
                    // Process shared tensor - all ranks get the full tensor
                    let tensor_size =
                        shared_info.shape().iter().product::<usize>() * shared_info.dtype().size();

                    for rank in 0..self.target_topology.n_ranks() {
                        let tensor_info = TensorInfo {
                            dtype: shared_info.dtype(),
                            shape: shared_info.shape().to_vec(),
                            data_offsets: (rank_offsets[rank], rank_offsets[rank] + tensor_size),
                        };

                        rank_tensor_info[rank].push((tensor_name.clone(), tensor_info));

                        tensor_tasks.push(TensorTask {
                            tensor_name: tensor_name.clone(),
                            rank,
                            file_offset: rank_offsets[rank],
                        });

                        rank_offsets[rank] += tensor_size;
                    }
                }
            }
        }

        // Create headers and final file info
        let mut rank_file_infos = Vec::new();
        for rank in 0..self.target_topology.n_ranks() {
            let metadata = Metadata::new(
                None, // metadata_header
                rank_tensor_info[rank].iter().cloned().collect(),
            )?;
            let header = serde_json::to_vec(&metadata)?;
            let header_size = header.len();

            // Update file_offset in tensor tasks for this rank to account for header
            let rank_tensor_tasks: Vec<_> = tensor_tasks
                .iter()
                .map(|task| {
                    if task.rank == rank {
                        TensorTask {
                            tensor_name: task.tensor_name.clone(),
                            rank: task.rank,
                            file_offset: task.file_offset + header_size + 8, // +8 for header size prefix
                        }
                    } else {
                        task.clone()
                    }
                })
                .collect();

            rank_file_infos.push(RankFileInfo {
                rank,
                header,
                total_size: rank_offsets[rank] + header_size + 8,
                tensor_tasks: rank_tensor_tasks
                    .into_iter()
                    .filter(|t| t.rank == rank)
                    .collect(),
            });
        }

        Ok(rank_file_infos)
    }

    /// Create all files with headers
    async fn create_files_with_headers<P: AsRef<Path>>(
        &self,
        output_dir: P,
        rank_file_infos: &[RankFileInfo],
    ) -> Result<()> {
        let output_dir = output_dir.as_ref();

        let file_creation_futures: Vec<_> = rank_file_infos
            .iter()
            .enumerate()
            .map(|(_idx, info)| {
                let filename = if self.target_topology.n_ranks() == 1 {
                    "model.safetensors".to_string()
                } else {
                    format!("rank{}.safetensors", info.rank)
                };
                let file_path = output_dir.join(&filename);
                let header = info.header.clone();
                async move {
                    let mut file = File::create(&file_path).await?;

                    file.write_all(&header).await?;
                    file.flush().await?;

                    Ok(())
                }
            })
            .collect();

        // Wait for all files to be created
        let creation_results: Vec<Result<()>> = join_all(file_creation_futures).await;
        for result in creation_results {
            result?;
        }

        Ok(())
    }

    /// Process and write all tensors in parallel
    async fn process_all_tensors<P: AsRef<Path>>(
        &self,
        output_dir: P,
        rank_file_infos: &[RankFileInfo],
    ) -> Result<()> {
        let output_dir = output_dir.as_ref();
        let mut tensor_tasks = Vec::new();

        // Collect all tensor tasks
        for info in rank_file_infos {
            tensor_tasks.extend(info.tensor_tasks.clone());
        }

        // Calculate total bytes to write
        let mut total_bytes: u64 = 0;
        for task in &tensor_tasks {
            // Get tensor metadata to determine size
            let tensor_metadata = self.get_tensor_metadata(&task.tensor_name).await?;
            let split_dim = self.determine_split_dimension(&task.tensor_name, &tensor_metadata);
            let tensor_size =
                if tensor_metadata.shape[split_dim] % self.target_topology.n_ranks() == 0 {
                    // Distributed: get chunk size for this rank
                    let chunk_size_per_rank =
                        tensor_metadata.shape[split_dim] / self.target_topology.n_ranks();
                    let mut chunk_shape = tensor_metadata.shape.clone();
                    chunk_shape[split_dim] = chunk_size_per_rank;
                    chunk_shape.iter().product::<usize>() * tensor_metadata.dtype.size()
                } else {
                    // Shared: full tensor
                    tensor_metadata.shape.iter().product::<usize>() * tensor_metadata.dtype.size()
                };
            total_bytes += tensor_size as u64;
        }

        // Create progress bar for bytes written
        let pb = ProgressBar::new(total_bytes);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] [{bar:40}] {bytes}/{total_bytes} ({eta}) {bytes_per_sec}")
            .unwrap());

        // Process tensors sequentially, updating the progress bar by bytes written
        for task in tensor_tasks {
            self.process_and_write_tensor(task, output_dir, &pb).await?;
        }

        pb.finish_with_message("Finished processing tensors");
        Ok(())
    }

    /// Process a single tensor and write it directly to the file
    async fn process_and_write_tensor(
        &self,
        task: TensorTask,
        output_dir: &Path,
        pb: &ProgressBar,
    ) -> Result<()> {
        // Get the tensor data for this rank
        let _permit = self.file_semaphore.acquire().await?;
        let tensor_data = self
            .get_tensor_data_for_rank(&task.tensor_name, task.rank)
            .await?;

        // Open the file and write the tensor data at the correct offset
        let filename = if self.target_topology.n_ranks() == 1 {
            "model.safetensors".to_string()
        } else {
            format!("rank{}.safetensors", task.rank)
        };
        let file_path = output_dir.join(&filename);
        let mut file = File::options().write(true).open(&file_path).await.unwrap();

        file.seek(SeekFrom::Start(task.file_offset as u64)).await?;
        file.write_all(&tensor_data).await?;
        file.flush().await?;

        // Update progress bar with bytes written
        pb.inc(tensor_data.len() as u64);

        Ok(())
    }

    /// Get tensor metadata without loading full tensor data
    async fn get_tensor_metadata(&self, tensor_name: &str) -> Result<TensorData> {
        match &self.data_source {
            TensorDataSource::Reconstruct {
                source_topology,
                input_dir,
            } => {
                let source_tensor = source_topology.get_tensor(tensor_name).unwrap();

                match source_tensor {
                    Tensor::Distributed(dist_info) => {
                        // For distributed tensors, get the full shape from topology
                        Ok(TensorData {
                            dtype: dist_info.dtype(),
                            shape: dist_info.shape().to_vec(),
                            data: Vec::new(), // We only need metadata here
                        })
                    }
                    Tensor::Shared(shared_info) => {
                        // For shared tensors, read from the file to get metadata
                        let file_idx = shared_info.filename_index();
                        let source_filename = &source_topology.filenames()[file_idx];
                        let source_path = input_dir.join(source_filename);
                        let tensor_name = tensor_name.to_string();

                        tokio::task::spawn_blocking(move || {
                            let file = std::fs::File::open(&source_path)?;
                            let mmap = unsafe { Mmap::map(&file)? };
                            let safetensors = SafeTensors::deserialize(&mmap)?;
                            let tensor_view = safetensors.tensor(&tensor_name)?;

                            let result = Ok(TensorData {
                                dtype: tensor_view.dtype(),
                                shape: tensor_view.shape().to_vec(),
                                data: Vec::new(), // We only need metadata here
                            });
                            result
                        })
                        .await?
                    }
                }
            }
            TensorDataSource::Loaded { tensors } => {
                let tensor = tensors.get(tensor_name).unwrap();
                Ok(TensorData {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                    data: Vec::new(), // We only need metadata here
                })
            }
        }
    }

    /// Get tensor data for a specific rank
    async fn get_tensor_data_for_rank(
        &self,
        tensor_name: &str,
        target_rank: usize,
    ) -> Result<Vec<u8>> {
        match &self.data_source {
            TensorDataSource::Reconstruct {
                source_topology, ..
            } => {
                let source_tensor = source_topology.get_tensor(tensor_name).unwrap();

                match source_tensor {
                    Tensor::Distributed(_) => {
                        // Need to reconstruct full tensor and then extract the chunk for target_rank
                        let full_tensor_data = self.reconstruct_tensor_async(tensor_name).await?;
                        let split_dim =
                            self.determine_split_dimension(tensor_name, &full_tensor_data);

                        if full_tensor_data.shape[split_dim] % self.target_topology.n_ranks() == 0 {
                            let chunk_size_per_rank =
                                full_tensor_data.shape[split_dim] / self.target_topology.n_ranks();
                            let start = target_rank * chunk_size_per_rank;
                            let end = (target_rank + 1) * chunk_size_per_rank;

                            let chunk_data =
                                extract_tensor_slice(&full_tensor_data, split_dim, start, end)?;
                            Ok(chunk_data)
                        } else {
                            // If not evenly divisible, keep as shared tensor (all ranks get full tensor)
                            Ok(full_tensor_data.data)
                        }
                    }
                    Tensor::Shared(info) => {
                        // Shared tensors are available on ALL ranks
                        let file_idx = info.filename_index();
                        let filename = &source_topology.filenames()[file_idx];
                        let file_path = match &self.data_source {
                            TensorDataSource::Reconstruct { input_dir, .. } => {
                                input_dir.join(filename)
                            }
                            TensorDataSource::Loaded { .. } => {
                                return Err(RedistributorError::InvalidDataSource {
                                    message: "Cannot access files from loaded tensor data source"
                                        .to_string(),
                                });
                            }
                        };
                        let tensor_name = tensor_name.to_string();

                        tokio::task::spawn_blocking(move || {
                            let file = std::fs::File::open(&file_path)?;
                            let mmap = unsafe { Mmap::map(&file)? };
                            let safetensors = SafeTensors::deserialize(&mmap)?;
                            let tensor = safetensors.tensor(&tensor_name)?;
                            let tensor_data = tensor.data().to_vec();

                            Ok(tensor_data)
                        })
                        .await?
                    }
                }
            }
            TensorDataSource::Loaded { tensors } => {
                let tensor = tensors.get(tensor_name).unwrap();

                // Determine split strategy
                let split_dim = self.determine_split_dimension(tensor_name, tensor);

                if tensor.shape[split_dim] % self.target_topology.n_ranks() == 0 {
                    // Split the tensor
                    let chunk_size_per_rank =
                        tensor.shape[split_dim] / self.target_topology.n_ranks();
                    let start = target_rank * chunk_size_per_rank;
                    let end = (target_rank + 1) * chunk_size_per_rank;

                    let chunk_data = extract_tensor_slice(tensor, split_dim, start, end)?;
                    Ok(chunk_data)
                } else {
                    // Keep as shared
                    Ok(tensor.data.clone())
                }
            }
        }
    }

    /// Determine the best dimension to split a tensor based on its name and shape
    fn determine_split_dimension(&self, tensor_name: &str, tensor: &TensorData) -> usize {
        let shape = &tensor.shape;

        // Use the same logic as the original GPT-2 splitting
        if ["c_fc", "c_attn", "wpe", "wte"]
            .iter()
            .any(|f| tensor_name.contains(f))
            && !tensor_name.contains("bias")
        {
            1 // Split along dimension 1
        } else if ["c_attn", "c_proj"].iter().any(|f| tensor_name.contains(f))
            && !tensor_name.contains("bias")
        {
            0 // Split along dimension 0
        } else {
            // Default: split along the largest dimension
            shape
                .iter()
                .enumerate()
                .max_by_key(|(_, size)| *size)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Reconstruct a tensor by reading from source files asynchronously
    async fn reconstruct_tensor_async(&self, tensor_name: &str) -> Result<TensorData> {
        match &self.data_source {
            TensorDataSource::Reconstruct {
                source_topology,
                input_dir,
            } => {
                match source_topology.get_tensor(tensor_name).unwrap() {
                    Tensor::Distributed(info) => {
                        let shape = info.shape();
                        let dtype = info.dtype();
                        let total_elements: usize = shape.iter().product();
                        let mut result = vec![0u8; total_elements * dtype.size()];

                        let ndim = shape.len();
                        // Compute strides for the full tensor
                        let mut full_strides = vec![1; ndim];
                        for i in (0..ndim - 1).rev() {
                            full_strides[i] = full_strides[i + 1] * shape[i + 1];
                        }

                        // Read all chunks in parallel
                        let chunk_futures: Vec<_> = info
                            .chunks()
                            .iter()
                            .map(|chunk| {
                                let file_idx = chunk.filename_index();
                                let filename = &source_topology.filenames()[file_idx];
                                let file_path = input_dir.join(filename);
                                let tensor_name = tensor_name.to_string();
                                let chunk = chunk.clone();
                                let full_strides = full_strides.clone();
                                let shape = shape.to_vec();
                                let dtype = dtype;

                                async move {
                                    tokio::task::spawn_blocking(move || {
                                        let file = std::fs::File::open(&file_path)?;
                                        let mmap = unsafe { Mmap::map(&file)? };
                                        let safetensors = SafeTensors::deserialize(&mmap)?;
                                        let tensor = safetensors.tensor(&tensor_name)?;
                                        let chunk_data = tensor.data().to_vec();

                                        // Calculate intervals for this chunk
                                        let intervals =
                                            get_intervals(&chunk, &full_strides, &shape);

                                        Ok::<_, RedistributorError>((intervals, chunk_data, dtype))
                                    })
                                    .await?
                                }
                            })
                            .collect();

                        let chunk_results: Vec<Result<(Vec<(usize, usize)>, Vec<u8>, Dtype)>> =
                            join_all(chunk_futures).await;

                        // Assemble the full tensor from chunks
                        for chunk_result in chunk_results {
                            let (intervals, chunk_data, _dtype) = chunk_result?;
                            let mut chunk_offset = 0;

                            for (start, stop) in intervals {
                                let start_byte = start * dtype.size();
                                let stop_byte = stop * dtype.size();
                                let chunk_size = stop_byte - start_byte;

                                result[start_byte..stop_byte].copy_from_slice(
                                    &chunk_data[chunk_offset..chunk_offset + chunk_size],
                                );
                                chunk_offset += chunk_size;
                            }
                        }

                        Ok(TensorData {
                            dtype,
                            shape: shape.to_vec(),
                            data: result,
                        })
                    }
                    Tensor::Shared(info) => {
                        let file_idx = info.filename_index();
                        let filename = &source_topology.filenames()[file_idx];
                        let file_path = input_dir.join(filename);
                        let tensor_name = tensor_name.to_string();
                        let shape = info.shape().to_vec();
                        let dtype = info.dtype();
                        tokio::task::spawn_blocking(move || {
                            let file = std::fs::File::open(&file_path)?;
                            let mmap = unsafe { Mmap::map(&file)? };
                            let safetensors = SafeTensors::deserialize(&mmap)?;
                            let tensor = safetensors.tensor(&tensor_name)?;
                            let tensor_data = tensor.data().to_vec();
                            Ok(TensorData {
                                dtype,
                                shape,
                                data: tensor_data,
                            })
                        })
                        .await?
                    }
                }
            }
            TensorDataSource::Loaded { tensors } => {
                // For loaded tensors, just return a clone
                Ok(tensors.get(tensor_name).unwrap().clone())
            }
        }
    }
}

/// Extract a slice of tensor data along a specific dimension
fn extract_tensor_slice(
    tensor: &TensorData,
    dim: usize,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    let shape = &tensor.shape;
    let dtype = tensor.dtype;
    let dtype_size = dtype.size();

    if dim >= shape.len() {
        return Err(RedistributorError::InvalidDimension {
            dim,
            shape: shape.to_vec(),
        });
    }

    if start >= end || end > shape[dim] {
        return Err(RedistributorError::InvalidSliceRange {
            start,
            end,
            dim,
            size: shape[dim],
        });
    }

    // Calculate strides
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Calculate the size of the extracted slice
    let mut slice_shape = shape.clone();
    slice_shape[dim] = end - start;
    let slice_elements: usize = slice_shape.iter().product();
    let mut result = vec![0u8; slice_elements * dtype_size];

    // Calculate how many complete "blocks" we have before the split dimension
    let outer_size: usize = shape[0..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let slice_inner_size = (end - start) * inner_size;

    // Copy data block by block
    for outer_idx in 0..outer_size {
        let src_offset = (outer_idx * shape[dim] * inner_size + start * inner_size) * dtype_size;
        let dst_offset = (outer_idx * slice_inner_size) * dtype_size;
        let copy_size = slice_inner_size * dtype_size;

        result[dst_offset..dst_offset + copy_size]
            .copy_from_slice(&tensor.data[src_offset..src_offset + copy_size]);
    }

    Ok(result)
}

/// Read safetensors file efficiently using memory mapping
fn safetensors_metadata<P: AsRef<Path>>(file_path: P) -> Result<Metadata> {
    let file_path = file_path.as_ref().to_path_buf();
    let file = std::fs::File::open(&file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let (_n, metadata) = SafeTensors::read_metadata(&mmap)?;
    Ok(metadata)
}

/// Load topology from directory, or create a single-rank topology if topology.json doesn't exist
pub fn load_or_create_topology<P: AsRef<Path>>(input_dir: P) -> Result<Topology> {
    let input_dir = input_dir.as_ref();
    let topology_path = input_dir.join("topology.json");
    let model_path = input_dir.join("model.safetensors");
    let index_path = input_dir.join("model.safetensors.index.json");

    // Check if we have a distributed setup (topology.json + rank*.safetensors)
    if topology_path.exists() {
        // Load existing distributed topology
        let topology_data = std::fs::read_to_string(&topology_path)?;
        let topology: Topology = serde_json::from_str(&topology_data)?;
        return Ok(topology);
    }
    let filenames = if index_path.exists() {
        // Chunked safetensors case - read the index file to get tensor information
        let index_data = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_data)?;

        // Group tensors by their chunk file
        let mut filenames: HashSet<String> = HashSet::new();
        for (_tensor_name, file_name) in &index.weight_map {
            filenames.insert(file_name.clone());
        }
        let mut filenames = filenames.into_iter().collect::<Vec<_>>();
        filenames.sort();
        filenames
    } else if model_path.exists() {
        vec!["model.safetensors".to_string()]
    } else {
        return Err(RedistributorError::NoValidInput {
            path: input_dir.to_path_buf(),
        });
    };
    // Create a topology with a single rank
    let mut tensors = HashMap::new();

    // Read each chunk file to get tensor information
    for (file_index, file_name) in filenames.iter().enumerate() {
        let file_path = input_dir.join(&file_name);
        let safetensors = safetensors_metadata(&file_path)?;

        for (tensor_name, tensor_info) in safetensors.tensors() {
            tensors.insert(
                tensor_name,
                Tensor::Shared(SharedInfo::new(
                    tensor_info.shape.to_vec(),
                    tensor_info.dtype,
                    file_index, // Use the correct file index
                )),
            );
        }
    }

    // Create topology with all chunk files
    let topology = Topology::new(tensors, filenames, 1)?;
    Ok(topology)
}
