use safetensors::{Dtype, SafeTensors};
use safetensors::tensor::{TensorInfo, Metadata};
use crate::topology::{
    Chunk, DistributedInfo, SharedInfo, Tensor, Topology, TopologyError, get_intervals,
};
use crate::tensor::TensorData;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncWriteExt, AsyncSeekExt, SeekFrom};
use tokio::fs::File;
use tokio::sync::Semaphore;
use futures::future::join_all;
use memmap2::Mmap;
use thiserror::Error;

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
    InvalidSliceRange { start: usize, end: usize, dim: usize, size: usize },

    #[error("No valid input found in directory {path:?} (expected topology.json + rank*.safetensors OR model.safetensors)")]
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

/// Configuration for the redistributor
#[derive(Debug, Clone)]
pub struct RedistributorConfig {
    /// Maximum number of concurrent file operations
    pub max_concurrent_files: usize,
    /// Whether to create model.safetensors for single rank outputs
    pub use_model_filename_for_single_rank: bool,
}

impl Default for RedistributorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_files: 50,
            use_model_filename_for_single_rank: true,
        }
    }
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
    target_world_size: usize,
    config: RedistributorConfig,
    file_semaphore: Arc<Semaphore>,
}

impl AsyncTensorRedistributor {
    /// Create a new redistributor for reconstruction from distributed files
    pub fn new_from_topology<P: AsRef<Path>>(
        source_topology: Topology,
        input_dir: P,
        target_world_size: usize,
        config: RedistributorConfig,
    ) -> Self {
        let input_dir = input_dir.as_ref().to_path_buf();
        
        Self {
            data_source: TensorDataSource::Reconstruct {
                source_topology,
                input_dir,
            },
            target_world_size,
            file_semaphore: Arc::new(Semaphore::new(config.max_concurrent_files)),
            config,
        }
    }
    
    /// Create a new redistributor for loaded tensor data
    pub fn new_from_loaded_tensors(
        tensors: HashMap<String, TensorData>,
        target_world_size: usize,
        config: RedistributorConfig,
    ) -> Self {
        Self {
            data_source: TensorDataSource::Loaded { tensors },
            target_world_size,
            file_semaphore: Arc::new(Semaphore::new(config.max_concurrent_files)),
            config,
        }
    }
    
    /// Redistribute tensors to target directory and return list of created files
    pub async fn redistribute<P: AsRef<Path>>(&self, output_dir: P) -> Result<Vec<String>> {
        let output_dir = output_dir.as_ref();
        tokio::fs::create_dir_all(output_dir).await?;
        
        println!("Starting async parallel redistribution with pre-calculated layout...");
        
        // Pre-calculate all file layouts and offsets
        let (new_topology, rank_file_infos) = self.pre_calculate_file_layout().await?;
        
        // Create initial files with headers
        self.create_files_with_headers(output_dir, &rank_file_infos).await?;
        
        // Process and write all tensors in parallel
        self.process_all_tensors(output_dir, &rank_file_infos).await?;
        
        // Collect created safetensors files
        let mut created_files = Vec::new();
        for filename in new_topology.filenames() {
            created_files.push(filename.clone());
        }
        
        // Write topology.json if needed (for multi-rank outputs)
        if self.target_world_size > 1 {
            let topology_json = serde_json::to_vec_pretty(&new_topology)?;
            tokio::fs::write(output_dir.join("topology.json"), &topology_json).await?;
            created_files.push("topology.json".to_string());
        }
        
        println!("Async parallel redistribution completed successfully");
        Ok(created_files)
    }
    
    /// Pre-calculate all headers, offsets, and file structures
    async fn pre_calculate_file_layout(&self) -> Result<(Topology, Vec<RankFileInfo>)> {
        println!("Pre-calculating file layout and offsets...");
        
        // First pass: determine tensor distribution and calculate sizes
        let mut new_topology_tensors = HashMap::new();
        let mut rank_tensor_info: Vec<Vec<(String, TensorInfo)>> = (0..self.target_world_size).map(|_| Vec::new()).collect();
        let mut rank_offsets = vec![0usize; self.target_world_size];
        let mut tensor_tasks = Vec::new();
        
        // Get tensor metadata and process each tensor
        let tensor_names = self.get_tensor_names().await?;
        for tensor_name in tensor_names {
            let tensor_metadata = self.get_tensor_metadata(&tensor_name).await?;
            
            // Determine how to redistribute this tensor
            let split_dim = self.determine_split_dimension(&tensor_name, &tensor_metadata);
            
            if tensor_metadata.shape[split_dim] % self.target_world_size == 0 {
                // Will be distributed - calculate chunk info
                let chunk_size_per_rank = tensor_metadata.shape[split_dim] / self.target_world_size;
                let mut chunks = Vec::new();
                
                for rank in 0..self.target_world_size {
                    let start = rank * chunk_size_per_rank;
                    let end = (rank + 1) * chunk_size_per_rank;
                    let mut chunk_shape = tensor_metadata.shape.clone();
                    chunk_shape[split_dim] = end - start;
                    let chunk_size = chunk_shape.iter().product::<usize>() * tensor_metadata.dtype.size();
                    
                    // Create offsets array for this chunk within the tensor
                    let mut chunk_offsets = vec![0; chunk_shape.len()];
                    chunk_offsets[split_dim] = start;
                    
                    let chunk = Chunk::new(
                        chunk_offsets,
                        chunk_shape.clone(),
                        rank,
                    );
                    chunks.push(chunk);
                    
                    let tensor_info = TensorInfo {
                        dtype: tensor_metadata.dtype,
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
                
                new_topology_tensors.insert(
                    tensor_name.clone(),
                    Tensor::Distributed(DistributedInfo::new(
                        tensor_metadata.shape.clone(),
                        tensor_metadata.dtype,
                        chunks,
                    )),
                );
            } else {
                // Keep as shared - add to ALL ranks
                let tensor_size = tensor_metadata.shape.iter().product::<usize>() * tensor_metadata.dtype.size();
                
                for rank in 0..self.target_world_size {
                    let tensor_info = TensorInfo {
                        dtype: tensor_metadata.dtype,
                        shape: tensor_metadata.shape.clone(),
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
                
                new_topology_tensors.insert(
                    tensor_name.clone(),
                    Tensor::Shared(SharedInfo::new(
                        tensor_metadata.shape.clone(),
                        tensor_metadata.dtype,
                        0,
                    )),
                );
            }
        }
        
        // Create headers and final file info
        let mut rank_file_infos = Vec::new();
        for rank in 0..self.target_world_size {
            let metadata = Metadata::new(
                None, // metadata_header
                rank_tensor_info[rank].iter().cloned().collect(),
            )?;
            let header = serde_json::to_vec(&metadata)?;
            let header_size = header.len();
            
            // Update file_offset in tensor tasks for this rank to account for header
            let rank_tensor_tasks: Vec<_> = tensor_tasks.iter().map(|task| {
                if task.rank == rank {
                    TensorTask {
                        tensor_name: task.tensor_name.clone(),
                        rank: task.rank,
                        file_offset: task.file_offset + header_size + 8, // +8 for header size prefix
                    }
                } else {
                    task.clone()
                }
            }).collect();
            
            rank_file_infos.push(RankFileInfo {
                rank,
                header,
                total_size: rank_offsets[rank] + header_size + 8,
                tensor_tasks: rank_tensor_tasks.into_iter().filter(|t| t.rank == rank).collect(),
            });
        }
        
        let new_filenames: Vec<_> = if self.target_world_size == 1 && self.config.use_model_filename_for_single_rank {
            vec!["model.safetensors".to_string()]
        } else {
            (0..self.target_world_size)
                .map(|rank| format!("rank{rank}.safetensors"))
                .collect()
        };
        
        let new_topology = Topology::new(new_topology_tensors, new_filenames, self.target_world_size)?;
        
        println!("Pre-calculation complete. File sizes: {:?}", 
                rank_file_infos.iter().map(|info| info.total_size).collect::<Vec<_>>());
        
        Ok((new_topology, rank_file_infos))
    }
    
    /// Create all files with headers
    async fn create_files_with_headers<P: AsRef<Path>>(&self, output_dir: P, rank_file_infos: &[RankFileInfo]) -> Result<()> {
        let output_dir = output_dir.as_ref();
        
        let file_creation_futures: Vec<_> = rank_file_infos.iter().enumerate().map(|(_idx, info)| {
            let filename = if self.target_world_size == 1 && self.config.use_model_filename_for_single_rank {
                "model.safetensors".to_string()
            } else {
                format!("rank{}.safetensors", info.rank)
            };
            let file_path = output_dir.join(&filename);
            let header = info.header.clone();
            let total_size = info.total_size;
            
            async move {
                let mut file = File::create(&file_path).await?;
                
                // Write header size (8 bytes)
                let header_size_bytes = (header.len() as u64).to_le_bytes();
                file.write_all(&header_size_bytes).await?;
                
                // Write header
                file.write_all(&header).await?;
                
                // Pre-allocate file to total size
                file.set_len(total_size as u64).await?;
                file.flush().await?;
                
                println!("Created {} with size {} bytes", filename, total_size);
                Ok(())
            }
        }).collect();
        
        // Wait for all files to be created
        let creation_results: Vec<Result<()>> = join_all(file_creation_futures).await;
        for result in creation_results {
            result?;
        }
        
        Ok(())
    }
    
    /// Process and write all tensors in parallel
    async fn process_all_tensors<P: AsRef<Path>>(&self, output_dir: P, rank_file_infos: &[RankFileInfo]) -> Result<()> {
        let output_dir = output_dir.as_ref();
        
        // Create futures for processing and writing each tensor
        let mut all_tensor_tasks = Vec::new();
        for rank_info in rank_file_infos {
            all_tensor_tasks.extend(rank_info.tensor_tasks.clone());
        }
        
        println!("Starting parallel tensor processing and writing for {} tensors...", all_tensor_tasks.len());
        
        let tensor_futures: Vec<_> = all_tensor_tasks.into_iter().map(|task| {
            let output_dir = output_dir.to_path_buf();
            
            async move {
                self.process_and_write_tensor(task, &output_dir).await
            }
        }).collect();
        
        // Wait for all tensor processing and writing to complete
        let tensor_results: Vec<Result<()>> = join_all(tensor_futures).await;
        for result in tensor_results {
            result?;
        }
        
        Ok(())
    }
    
    /// Process a single tensor and write it directly to the file
    async fn process_and_write_tensor(&self, task: TensorTask, output_dir: &Path) -> Result<()> {
        println!("Processing and writing tensor: {} to rank {} at offset {}", 
                task.tensor_name, task.rank, task.file_offset);
        
        // Get the tensor data for this rank
        let tensor_data = self.get_tensor_data_for_rank(&task.tensor_name, task.rank).await?;
        
        // Acquire semaphore before file operations to limit concurrent writes
        let _permit = self.file_semaphore.acquire().await?;
        
        // Open the file and write the tensor data at the correct offset
        let filename = if self.target_world_size == 1 && self.config.use_model_filename_for_single_rank {
            "model.safetensors".to_string()
        } else {
            format!("rank{}.safetensors", task.rank)
        };
        let file_path = output_dir.join(&filename);
        let mut file = File::options().write(true).open(&file_path).await?;
        
        file.seek(SeekFrom::Start(task.file_offset as u64)).await?;
        file.write_all(&tensor_data).await?;
        file.flush().await?;
        
        let target_file = if self.target_world_size == 1 && self.config.use_model_filename_for_single_rank {
            "model.safetensors".to_string()
        } else {
            format!("rank{}", task.rank)
        };
        println!("  ✓ Wrote {} bytes for {} to {}", tensor_data.len(), task.tensor_name, target_file);
        
        Ok(())
    }
    
    /// Get tensor names from the data source
    async fn get_tensor_names(&self) -> Result<Vec<String>> {
        match &self.data_source {
            TensorDataSource::Reconstruct { source_topology, .. } => {
                Ok(source_topology.tensors().keys().cloned().collect())
            }
            TensorDataSource::Loaded { tensors } => {
                Ok(tensors.keys().cloned().collect())
            }
        }
    }
    
    /// Get tensor metadata without loading full tensor data
    async fn get_tensor_metadata(&self, tensor_name: &str) -> Result<TensorData> {
        match &self.data_source {
            TensorDataSource::Reconstruct { source_topology, input_dir } => {
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
                            
                            Ok(TensorData {
                                dtype: tensor_view.dtype(),
                                shape: tensor_view.shape().to_vec(),
                                data: Vec::new(), // We only need metadata here
                            })
                        }).await?
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
    async fn get_tensor_data_for_rank(&self, tensor_name: &str, target_rank: usize) -> Result<Vec<u8>> {
        match &self.data_source {
            TensorDataSource::Reconstruct { source_topology, .. } => {
                let source_tensor = source_topology.get_tensor(tensor_name).unwrap();
                
                match source_tensor {
                    Tensor::Distributed(_) => {
                        // Need to reconstruct full tensor and then extract the chunk for target_rank
                        let full_tensor_data = self.reconstruct_tensor_async(tensor_name).await?;
                        let split_dim = self.determine_split_dimension(tensor_name, &full_tensor_data);
                        
                        if full_tensor_data.shape[split_dim] % self.target_world_size == 0 {
                            let chunk_size_per_rank = full_tensor_data.shape[split_dim] / self.target_world_size;
                            let start = target_rank * chunk_size_per_rank;
                            let end = (target_rank + 1) * chunk_size_per_rank;
                            
                            let chunk_data = extract_tensor_slice(&full_tensor_data, split_dim, start, end)?;
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
                            TensorDataSource::Reconstruct { input_dir, .. } => input_dir.join(filename),
                            TensorDataSource::Loaded { .. } => return Err(RedistributorError::InvalidDataSource { 
                message: "Cannot access files from loaded tensor data source".to_string() 
            }),
                        };
                        let tensor_name = tensor_name.to_string();
                        
                        tokio::task::spawn_blocking(move || {
                            let file = std::fs::File::open(&file_path)?;
                            let mmap = unsafe { Mmap::map(&file)? };
                            let safetensors = SafeTensors::deserialize(&mmap)?;
                            let tensor = safetensors.tensor(&tensor_name)?;
                            let tensor_data = tensor.data().to_vec();
                            
                            Ok(tensor_data)
                        }).await?
                    }
                }
            }
            TensorDataSource::Loaded { tensors } => {
                let tensor = tensors.get(tensor_name).unwrap();
                
                // Determine split strategy
                let split_dim = self.determine_split_dimension(tensor_name, tensor);
                
                if tensor.shape[split_dim] % self.target_world_size == 0 {
                    // Split the tensor
                    let chunk_size_per_rank = tensor.shape[split_dim] / self.target_world_size;
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
        if ["c_fc", "c_attn", "wpe", "wte"].iter().any(|f| tensor_name.contains(f))
            && !tensor_name.contains("bias")
        {
            1 // Split along dimension 1
        } else if ["c_attn", "c_proj"].iter().any(|f| tensor_name.contains(f))
            && !tensor_name.contains("bias")
        {
            0 // Split along dimension 0
        } else {
            // Default: split along the largest dimension
            shape.iter()
                .enumerate()
                .max_by_key(|(_, size)| *size)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }
    
    /// Reconstruct a tensor by reading from source files asynchronously
    async fn reconstruct_tensor_async(&self, tensor_name: &str) -> Result<TensorData> {
        match &self.data_source {
            TensorDataSource::Reconstruct { source_topology, input_dir } => {
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
                        let chunk_futures: Vec<_> = info.chunks().iter().map(|chunk| {
                            let file_idx = chunk.filename_index();
                            let filename = &source_topology.filenames()[file_idx];
                            let file_path = input_dir.join(filename);
                            let tensor_name = tensor_name.to_string();
                            let chunk = chunk.clone();
                            let full_strides = full_strides.clone();
                            let shape = shape.to_vec();
                            let dtype = dtype;
                            
                            async move {
                                // Use spawn_blocking for mmap operations
                                tokio::task::spawn_blocking(move || {
                                    let file = std::fs::File::open(&file_path)?;
                                    let mmap = unsafe { Mmap::map(&file)? };
                                    let safetensors = SafeTensors::deserialize(&mmap)?;
                                    let tensor = safetensors.tensor(&tensor_name)?;
                                    let chunk_data = tensor.data().to_vec();
                                    
                                    // Calculate intervals for this chunk
                                    let intervals = get_intervals(&chunk, &full_strides, &shape);
                                    
                                    Ok::<_, RedistributorError>((intervals, chunk_data, dtype))
                                }).await?
                            }
                        }).collect();
                        
                        let chunk_results: Vec<Result<(Vec<(usize, usize)>, Vec<u8>, Dtype)>> = join_all(chunk_futures).await;
                        
                        // Assemble the full tensor from chunks
                        for chunk_result in chunk_results {
                            let (intervals, chunk_data, _dtype) = chunk_result?;
                            let mut chunk_offset = 0;
                            
                            for (start, stop) in intervals {
                                let start_byte = start * dtype.size();
                                let stop_byte = stop * dtype.size();
                                let chunk_size = stop_byte - start_byte;
                                
                                result[start_byte..stop_byte]
                                    .copy_from_slice(&chunk_data[chunk_offset..chunk_offset + chunk_size]);
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
                        
                        tokio::task::spawn_blocking(move || {
                            let file = std::fs::File::open(&file_path)?;
                            let mmap = unsafe { Mmap::map(&file)? };
                            let safetensors = SafeTensors::deserialize(&mmap)?;
                            let tensor = safetensors.tensor(&tensor_name)?;
                            let tensor_data = tensor.data().to_vec();
                            
                            Ok(TensorData {
                                dtype: tensor.dtype(),
                                shape: tensor.shape().to_vec(),
                                data: tensor_data,
                            })
                        }).await?
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
        return Err(RedistributorError::InvalidDimension { dim, shape: shape.to_vec() });
    }
    
    if start >= end || end > shape[dim] {
                return Err(RedistributorError::InvalidSliceRange { 
            start, end, dim, size: shape[dim] 
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

/// Load topology from directory, or create a single-rank topology if topology.json doesn't exist
pub async fn load_or_create_topology<P: AsRef<Path>>(input_dir: P) -> Result<Topology> {
    let input_dir = input_dir.as_ref();
    let topology_path = input_dir.join("topology.json");
    
    if topology_path.exists() {
        // Load existing distributed topology
        println!("Loading distributed topology from {:?}", topology_path);
        let topology_data = tokio::fs::read_to_string(&topology_path).await?;
        let topology: Topology = serde_json::from_str(&topology_data)?;
        Ok(topology)
    } else {
        // Create a single-rank topology from model.safetensors
        let model_path = input_dir.join("model.safetensors");
        if !model_path.exists() {
            return Err(RedistributorError::NoValidInput { path: input_dir.to_path_buf() });
        }
        
        println!("No topology.json found, creating single-rank topology from {:?}", model_path);
        
        // Read the safetensors file to get tensor information
        let model_data = tokio::fs::read(&model_path).await?;
        let safetensors = SafeTensors::deserialize(&model_data)?;
        
        // Create topology with all tensors as shared
        let mut tensors = HashMap::new();
        for tensor_name in safetensors.names() {
            let tensor_info = safetensors.tensor(tensor_name)?;
            tensors.insert(
                tensor_name.to_string(),
                Tensor::Shared(SharedInfo::new(
                    tensor_info.shape().to_vec(),
                    tensor_info.dtype(),
                    0, // All tensors reference the single file
                )),
            );
        }
        
        let topology = Topology::new(
            tensors,
            vec!["model.safetensors".to_string()],
            1,
        )?;
        
        Ok(topology)
    }
} 