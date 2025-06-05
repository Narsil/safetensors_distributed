use anyhow::{Result, anyhow};
use hf_hub::api::tokio::ApiBuilder;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use safetensors_distributed::topology::{
    Chunk, DistributedInfo, SharedInfo, Tensor, Topology, get_intervals,
};
use safetensors_distributed::tensor::TensorData;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, AsyncSeekExt, SeekFrom};
use futures::future::join_all;
use safetensors::tensor::{TensorInfo, Metadata};
use serde_json;

/// Represents a tensor processing and writing task
#[derive(Debug, Clone)]
struct TensorWriteTask {
    tensor_name: String,
    rank: usize,
    file_offset: usize,
    tensor_data: TensorData,
}

/// Pre-calculated rank file information
#[derive(Debug)]
struct RankFileInfo {
    filename: String,
    header: Vec<u8>,
    total_size: usize,
    tensor_tasks: Vec<TensorWriteTask>,
}

/// Async distributed model creator
struct AsyncDistributedCreator {
    file_semaphore: Arc<Semaphore>,
}

impl AsyncDistributedCreator {
    fn new() -> Self {
        Self {
            file_semaphore: Arc::new(Semaphore::new(50)), // Limit to 50 concurrent file operations
        }
    }
    
    /// Pre-calculate all file layouts, headers, and offsets
    async fn pre_calculate_layout(&self, tensors: &SafeTensors<'_>, world_size: usize) -> Result<(Topology, Vec<RankFileInfo>)> {
        println!("Pre-calculating file layout and offsets...");
        
        let mut new_topology_tensors = HashMap::new();
        let filenames: Vec<_> = (0..world_size)
            .map(|rank| format!("rank{rank}.safetensors"))
            .collect();
        
        // Track tensor data for each rank with offsets
        let mut rank_tensor_info: Vec<Vec<(String, TensorInfo)>> = (0..world_size).map(|_| Vec::new()).collect();
        let mut rank_offsets = vec![0usize; world_size];
        let mut tensor_tasks = Vec::new();
        
        for tensor_name in tensors.names() {
            let tensor = tensors.tensor(tensor_name)?;
            
            if ["c_fc", "c_attn", "wpe", "wte"]
                .iter()
                .any(|f| tensor_name.contains(f))
                && !tensor_name.contains("bias")
            {
                // Distributed tensor - split across ranks
                let (topo_tensor, split_data) = split(&tensor, 1, world_size)?;
                new_topology_tensors.insert(tensor_name.to_string(), topo_tensor);
                
                for (rank, tensor_data) in split_data.into_iter().enumerate() {
                    let tensor_size = tensor_data.data.len();
                    let tensor_info = TensorInfo {
                        dtype: tensor_data.dtype,
                        shape: tensor_data.shape.clone(),
                        data_offsets: (rank_offsets[rank], rank_offsets[rank] + tensor_size),
                    };
                    
                    rank_tensor_info[rank].push((tensor_name.to_string(), tensor_info));
                    
                    tensor_tasks.push(TensorWriteTask {
                        tensor_name: tensor_name.to_string(),
                        rank,
                        file_offset: rank_offsets[rank],
                        tensor_data,
                    });
                    
                    rank_offsets[rank] += tensor_size;
                }
            } else if ["c_attn", "c_proj"].iter().any(|f| tensor_name.contains(f))
                && !tensor_name.contains("bias")
            {
                // Distributed tensor - split across ranks (dim 0)
                let (topo_tensor, split_data) = split(&tensor, 0, world_size)?;
                new_topology_tensors.insert(tensor_name.to_string(), topo_tensor);
                
                for (rank, tensor_data) in split_data.into_iter().enumerate() {
                    let tensor_size = tensor_data.data.len();
                    let tensor_info = TensorInfo {
                        dtype: tensor_data.dtype,
                        shape: tensor_data.shape.clone(),
                        data_offsets: (rank_offsets[rank], rank_offsets[rank] + tensor_size),
                    };
                    
                    rank_tensor_info[rank].push((tensor_name.to_string(), tensor_info));
                    
                    tensor_tasks.push(TensorWriteTask {
                        tensor_name: tensor_name.to_string(),
                        rank,
                        file_offset: rank_offsets[rank],
                        tensor_data,
                    });
                    
                    rank_offsets[rank] += tensor_size;
                }
            } else if tensor_name.contains("ln_") || tensor_name.contains(".bias") {
                // Shared tensor - add to ALL ranks
                let topo_tensor = Tensor::Shared(SharedInfo::new(tensor.shape().to_vec(), tensor.dtype(), 0));
                new_topology_tensors.insert(tensor_name.to_string(), topo_tensor);
                
                let tensor_data: TensorData = tensor.into();
                let tensor_size = tensor_data.data.len();
                
                for rank in 0..world_size {
                    let tensor_info = TensorInfo {
                        dtype: tensor_data.dtype,
                        shape: tensor_data.shape.clone(),
                        data_offsets: (rank_offsets[rank], rank_offsets[rank] + tensor_size),
                    };
                    
                    rank_tensor_info[rank].push((tensor_name.to_string(), tensor_info));
                    
                    tensor_tasks.push(TensorWriteTask {
                        tensor_name: tensor_name.to_string(),
                        rank,
                        file_offset: rank_offsets[rank],
                        tensor_data: tensor_data.clone(),
                    });
                    
                    rank_offsets[rank] += tensor_size;
                }
            } else {
                return Err(anyhow!("Unhandled tensor: {}", tensor_name));
            }
        }
        
        // Create headers and final file info
        let mut rank_file_infos = Vec::new();
        for rank in 0..world_size {
            let metadata = Metadata::new(None, rank_tensor_info[rank].clone())?;
            let header = serde_json::to_vec(&metadata)?;
            let header_size = header.len();
            let total_size = header_size + 8 + rank_offsets[rank]; // +8 for header size prefix
            
            // Filter tasks for this rank
            let rank_tasks: Vec<_> = tensor_tasks.iter()
                .filter(|task| task.rank == rank)
                .cloned()
                .collect();
            
            rank_file_infos.push(RankFileInfo {
                filename: filenames[rank].clone(),
                header,
                total_size,
                tensor_tasks: rank_tasks,
            });
        }
        
        let topology = Topology::new(new_topology_tensors, filenames, world_size)?;
        
        println!("Pre-calculation complete. File sizes: {:?}", 
                rank_file_infos.iter().map(|info| info.total_size).collect::<Vec<_>>());
        
        Ok((topology, rank_file_infos))
    }
    
    /// Create all rank files and process tensors in parallel
    async fn create_all_files<P: AsRef<Path>>(&self, output_dir: P, rank_file_infos: Vec<RankFileInfo>) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        
        println!("Creating {} rank files and processing {} tensors in parallel...", 
                rank_file_infos.len(), 
                rank_file_infos.iter().map(|info| info.tensor_tasks.len()).sum::<usize>());
        
        // Create files with headers first
        for info in &rank_file_infos {
            let file_path = output_dir.join(&info.filename);
            let mut file = File::create(&file_path).await?;
            
            // Write header size (8 bytes)
            let header_size_bytes = (info.header.len() as u64).to_le_bytes();
            file.write_all(&header_size_bytes).await?;
            
            // Write header
            file.write_all(&info.header).await?;
            
            // Pre-allocate file to total size
            file.set_len(info.total_size as u64).await?;
            file.flush().await?;
            
            println!("Created {} with size {} bytes", info.filename, info.total_size);
        }
        
        // Collect all tensor writing tasks
        let mut tensor_futures = Vec::new();
        for info in rank_file_infos {
            for task in info.tensor_tasks {
                let output_dir = output_dir.to_path_buf();
                let semaphore = self.file_semaphore.clone();
                let header_prefix_size = 8 + info.header.len(); // 8 bytes for size + header
                
                let future = async move {
                    Self::write_tensor_to_file(task, output_dir, header_prefix_size, semaphore).await
                };
                tensor_futures.push(future);
            }
        }
        
        println!("Starting parallel tensor writing for {} tensors...", tensor_futures.len());
        
        // Execute all tensor writes in parallel
        let results = join_all(tensor_futures).await;
        
        // Check for errors
        for result in results {
            result?;
        }
        
        println!("All tensors written successfully!");
        Ok(())
    }
    
    /// Write a single tensor to its rank file with semaphore limiting
    async fn write_tensor_to_file(
        task: TensorWriteTask, 
        output_dir: std::path::PathBuf, 
        header_prefix_size: usize,
        semaphore: Arc<Semaphore>
    ) -> Result<()> {
        let _permit = semaphore.acquire().await?;
        
        let file_path = output_dir.join(format!("rank{}.safetensors", task.rank));
        let mut file = File::options().write(true).open(&file_path).await?;
        
        // Seek to the correct position (header + file offset)
        let file_position = header_prefix_size + task.file_offset;
        file.seek(SeekFrom::Start(file_position as u64)).await?;
        
        // Write tensor data
        file.write_all(&task.tensor_data.data).await?;
        file.flush().await?;
        
        println!("  ✓ Wrote {} bytes for {} to rank{}", 
                task.tensor_data.data.len(), task.tensor_name, task.rank);
        
        Ok(())
    }
}

fn split(tensor: &TensorView, dim: usize, world_size: usize) -> Result<(Tensor, Vec<TensorData>)> {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let mut local_shape = shape.to_vec();
    if let Some(dimsize) = local_shape.get(dim) {
        if dimsize % world_size != 0 {
            return Err(anyhow!(
                "Dim is not evenly divisible {dimsize} % {world_size} != 0"
            ));
        }
        local_shape[dim] = dimsize / world_size;
    } else {
        return Err(anyhow!("Invalid dim {dim}"));
    }
    let mut offsets = vec![0; shape.len()];
    let all_data = tensor.data();
    let (chunks, tensor_data) = (0..world_size)
        .map(|rank| {
            offsets[dim] = rank * local_shape[dim];
            let n: usize = local_shape.iter().product();
            let ndim = shape.len();
            // Compute strides for the full tensor
            let mut strides = vec![1; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            let chunk = Chunk::new(offsets.clone(), local_shape.clone(), rank);
            let intervals = get_intervals(&chunk, &strides, &shape);
            let data: Vec<u8> = intervals
                .into_iter()
                .map(|(start, stop)| &all_data[start * dtype.size()..stop * dtype.size()])
                .flatten()
                .cloned()
                .collect();
            assert_eq!(data.len(), n * dtype.size());
            (
                chunk,
                TensorData {
                    data,
                    dtype,
                    shape: local_shape.clone(),
                },
            )
        })
        .unzip();
    Ok((
        Tensor::Distributed(DistributedInfo::new(shape.to_vec(), dtype, chunks)),
        tensor_data,
    ))
}

/// Creates a distributed version of GPT-2 model with async parallel processing
async fn create_distributed_gpt2<P: AsRef<Path>>(filename: &Path, output_dir: P) -> Result<()> {
    // Load the original safetensors file
    let data = std::fs::read(filename)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let world_size = 4;
    let creator = AsyncDistributedCreator::new();
    
    println!("Starting async distributed model creation...");
    
    // Pre-calculate all file layouts and offsets
    let (topology, rank_file_infos) = creator.pre_calculate_layout(&tensors, world_size).await?;
    
    // Create all files and write tensors in parallel
    creator.create_all_files(&output_dir, rank_file_infos).await?;
    
    // Write topology file
    let topology_json = serde_json::to_vec_pretty(&topology)?;
    let topology_path = output_dir.as_ref().join("topology.json");
    tokio::fs::write(topology_path, topology_json).await?;
    
    println!("Async distributed model creation completed successfully!");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let output_dir = "distributed_gpt2";

    // Download the original GPT-2 model
    let api = ApiBuilder::from_env().build()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let filename = repo.get("model.safetensors").await?;
    create_distributed_gpt2(&filename, output_dir).await?;

    Ok(())
}
