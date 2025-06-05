use anyhow::{Result, anyhow};
use safetensors::{Dtype, SafeTensors};
use safetensors::tensor::{TensorInfo, Metadata};
use safetensors_distributed::topology::{
    Chunk, DistributedInfo, SharedInfo, Tensor, Topology, get_intervals,
};
use safetensors_distributed::tensor::TensorData;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncWriteExt, AsyncSeekExt, SeekFrom};
use tokio::fs::File;
use tokio::sync::Semaphore;
use futures::future::join_all;
use memmap2::Mmap;

/// Async streaming tensor redistributor with parallel processing and pre-calculated offsets
struct AsyncStreamingRedistributor {
    source_topology: Topology,
    input_dir: std::path::PathBuf,
    target_world_size: usize,
    file_semaphore: Arc<Semaphore>,
}

/// Represents a tensor processing and writing task
#[derive(Debug, Clone)]
struct TensorTask {
    tensor_name: String,
    rank: usize,
    file_offset: usize,
}

/// Pre-calculated rank file information
#[derive(Debug)]
struct RankFileInfo {
    rank: usize,
    header: Vec<u8>,
    total_size: usize,
    tensor_tasks: Vec<TensorTask>,
}

impl AsyncStreamingRedistributor {
    fn new<P: AsRef<Path>>(
        source_topology: Topology,
        input_dir: P,
        target_world_size: usize,
    ) -> Self {
        let input_dir = input_dir.as_ref().to_path_buf();
        
        Self {
            source_topology,
            input_dir,
            target_world_size,
            file_semaphore: Arc::new(Semaphore::new(50)), // Limit concurrent file writes to avoid "too many files open"
        }
    }
    
    /// Pre-calculate all headers, offsets, and file structures
    async fn pre_calculate_file_layout(&self) -> Result<(Topology, Vec<RankFileInfo>)> {
        println!("Pre-calculating file layout and offsets...");
        
        // First pass: determine tensor distribution and calculate sizes
        let mut new_topology_tensors = HashMap::new();
        let mut rank_tensor_info: Vec<Vec<(String, TensorInfo)>> = (0..self.target_world_size).map(|_| Vec::new()).collect();
        let mut rank_offsets = vec![0usize; self.target_world_size];
        let mut tensor_tasks = Vec::new();
        
        for (tensor_name, source_tensor) in self.source_topology.tensors() {
            // Get tensor metadata by reading just the header from one source file
            let tensor_metadata = self.get_tensor_metadata(tensor_name).await?;
            
            match source_tensor {
                Tensor::Distributed(_) => {
                    let split_dim = determine_split_dimension(tensor_name, &tensor_metadata);
                    
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
                Tensor::Shared(_shared_info) => {
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
        
        let new_filenames: Vec<_> = if self.target_world_size == 1 {
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
    
    /// Process all tensors in parallel with pre-calculated offsets
    async fn redistribute_all_tensors<P: AsRef<Path>>(&self, output_dir: P) -> Result<Topology> {
        let output_dir = output_dir.as_ref();
        tokio::fs::create_dir_all(output_dir).await?;
        
        println!("Starting async parallel redistribution with pre-calculated layout...");
        
        // Pre-calculate all file layouts and offsets
        let (new_topology, rank_file_infos) = self.pre_calculate_file_layout().await?;
        
        // Create initial files with headers
        let file_creation_futures: Vec<_> = rank_file_infos.iter().enumerate().map(|(_idx, info)| {
            let filename = if self.target_world_size == 1 {
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
                anyhow::Ok(())
            }
        }).collect();
        
        // Wait for all files to be created
        let creation_results: Vec<Result<()>> = join_all(file_creation_futures).await;
        for result in creation_results {
            result?;
        }
        
        // Create futures for processing and writing each tensor
        let mut all_tensor_tasks = Vec::new();
        for rank_info in &rank_file_infos {
            all_tensor_tasks.extend(rank_info.tensor_tasks.clone());
        }
        
        println!("Starting parallel tensor processing and writing for {} tensors...", all_tensor_tasks.len());
        
        let tensor_futures: Vec<_> = all_tensor_tasks.into_iter().map(|task| {
            let redistributor = self;
            let output_dir = output_dir.to_path_buf();
            
            async move {
                redistributor.process_and_write_tensor(task, &output_dir).await
            }
        }).collect();
        
        // Wait for all tensor processing and writing to complete
        let tensor_results: Vec<Result<()>> = join_all(tensor_futures).await;
        for result in tensor_results {
            result?;
        }
        
        println!("Async parallel redistribution completed successfully");
        Ok(new_topology)
    }
    
    /// Process a single tensor and write it directly to the file
    async fn process_and_write_tensor(&self, task: TensorTask, output_dir: &Path) -> Result<()> {
        println!("Processing and writing tensor: {} to rank {} at offset {}", 
                task.tensor_name, task.rank, task.file_offset);
        
        // Reconstruct or extract the tensor data for this rank
        let tensor_data = self.get_tensor_data_for_rank(&task.tensor_name, task.rank).await?;
        
        // Acquire semaphore before file operations to limit concurrent writes
        let _permit = self.file_semaphore.acquire().await?;
        
        // Open the file and write the tensor data at the correct offset
        let filename = if self.target_world_size == 1 {
            "model.safetensors".to_string()
        } else {
            format!("rank{}.safetensors", task.rank)
        };
        let file_path = output_dir.join(&filename);
        let mut file = File::options().write(true).open(&file_path).await?;
        
        file.seek(SeekFrom::Start(task.file_offset as u64)).await?;
        file.write_all(&tensor_data).await?;
        file.flush().await?;
        
        let target_file = if self.target_world_size == 1 {
            "model.safetensors".to_string()
        } else {
            format!("rank{}", task.rank)
        };
        println!("  ✓ Wrote {} bytes for {} to {}", tensor_data.len(), task.tensor_name, target_file);
        
        Ok(())
    }
    
    /// Get tensor metadata without loading full tensor data
    async fn get_tensor_metadata(&self, tensor_name: &str) -> Result<TensorData> {
        let source_tensor = self.source_topology.get_tensor(tensor_name).unwrap();
        
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
                let source_filename = &self.source_topology.filenames()[file_idx];
                let source_path = self.input_dir.join(source_filename);
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
    
    /// Get tensor data for a specific rank
    async fn get_tensor_data_for_rank(&self, tensor_name: &str, target_rank: usize) -> Result<Vec<u8>> {
        let source_tensor = self.source_topology.get_tensor(tensor_name).unwrap();
        
        match source_tensor {
            Tensor::Distributed(_) => {
                // Need to reconstruct full tensor and then extract the chunk for target_rank
                let full_tensor_data = self.reconstruct_tensor_async(tensor_name).await?;
                let split_dim = determine_split_dimension(tensor_name, &full_tensor_data);
                
                if full_tensor_data.shape[split_dim] % self.target_world_size == 0 {
                    let chunk_size_per_rank = full_tensor_data.shape[split_dim] / self.target_world_size;
                    let start = target_rank * chunk_size_per_rank;
                    let end = (target_rank + 1) * chunk_size_per_rank;
                    
                    let chunk_data = extract_tensor_slice(&full_tensor_data, split_dim, start, end)?;
                    Ok(chunk_data)
                } else {
                    // If not evenly divisible and target_rank is 0, return full tensor
                    if target_rank == 0 {
                        Ok(full_tensor_data.data)
                    } else {
                        // This rank shouldn't have this tensor
                        Err(anyhow!("Tensor {} should not be present on rank {}", tensor_name, target_rank))
                    }
                }
            }
            Tensor::Shared(info) => {
                // Shared tensors are available on ALL ranks
                let file_idx = info.filename_index();
                let filename = &self.source_topology.filenames()[file_idx];
                let file_path = self.input_dir.join(filename);
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
    
    /// Reconstruct a tensor by reading from source files asynchronously
    async fn reconstruct_tensor_async(&self, tensor_name: &str) -> Result<TensorData> {
        match self.source_topology.get_tensor(tensor_name).unwrap() {
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
                    let filename = &self.source_topology.filenames()[file_idx];
                    let file_path = self.input_dir.join(filename);
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
                            
                            Ok::<_, anyhow::Error>((intervals, chunk_data, dtype))
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
                let filename = &self.source_topology.filenames()[file_idx];
                let file_path = self.input_dir.join(filename);
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
        return Err(anyhow!("Invalid dimension {} for tensor with shape {:?}", dim, shape));
    }
    
    if start >= end || end > shape[dim] {
        return Err(anyhow!("Invalid slice range [{}, {}) for dimension {} with size {}", 
                          start, end, dim, shape[dim]));
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

/// Determines the best dimension to split a tensor based on its name and shape
fn determine_split_dimension(tensor_name: &str, tensor: &TensorData) -> usize {
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

/// Load topology from directory, or create a single-rank topology if topology.json doesn't exist
async fn load_or_create_topology<P: AsRef<Path>>(input_dir: P) -> Result<Topology> {
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
            return Err(anyhow!("Neither topology.json nor model.safetensors found in {:?}", input_dir));
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

/// Main redistribution function using async parallel approach
async fn redistribute_model_async<P: AsRef<Path>>(
    input_dir: P,
    output_dir: P,
    target_world_size: usize,
) -> Result<()> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();
    
    println!("Reading model from {:?}", input_dir);
    
    // Load the existing topology (or create from model.safetensors)
    let source_topology = load_or_create_topology(input_dir).await?;
    
    println!("Source topology has {} ranks", source_topology.n_ranks());
    println!("Target topology will have {} ranks", target_world_size);
    
    // Create and run the async redistributor
    let redistributor = AsyncStreamingRedistributor::new(
        source_topology,
        input_dir,
        target_world_size,
    );
    
    let new_topology = redistributor.redistribute_all_tensors(output_dir).await?;
    
    // Save the new topology (skip if target_world_size is 1, as it's a non-sharded checkpoint)
    if target_world_size > 1 {
        let topology_json = serde_json::to_vec_pretty(&new_topology)?;
        tokio::fs::write(output_dir.join("topology.json"), &topology_json).await?;
    }
    
    // Print results
    if target_world_size == 1 {
        println!("Saved model.safetensors");
    } else {
        for rank in 0..target_world_size {
            println!("Saved rank{}.safetensors", rank);
        }
    }
    
    if target_world_size > 1 {
        println!("Model successfully redistributed from {} to {} ranks", 
                 new_topology.n_ranks(), target_world_size);
        println!("Output saved to {:?} (with topology.json)", output_dir);
    } else {
        println!("Model successfully redistributed from {} ranks to {} rank (non-sharded)", 
                 new_topology.n_ranks(), target_world_size);
        println!("Output saved to {:?} as model.safetensors (no topology.json)", output_dir);
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() != 4 {
        eprintln!("Usage: {} <input_dir> <output_dir> <target_world_size>", args[0]);
        eprintln!("Example: {} distributed_gpt2 redistributed_gpt2_async 2", args[0]);
        eprintln!("        {} single_model_dir redistributed_async 4", args[0]);
        eprintln!("Input: topology.json + rank*.safetensors OR model.safetensors");
        eprintln!("Note: target_world_size=1 creates model.safetensors (no topology.json)");
        std::process::exit(1);
    }
    
    let input_dir = &args[1];
    let output_dir = &args[2];
    let target_world_size: usize = args[3].parse()
        .map_err(|_| anyhow!("Invalid target world size: {}", args[3]))?;
    
    redistribute_model_async(input_dir, output_dir, target_world_size).await?;
    
    Ok(())
} 