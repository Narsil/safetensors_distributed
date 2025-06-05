use anyhow::{Result, anyhow};
use safetensors::{Dtype, SafeTensors, View, serialize_to_file};
use safetensors_distributed::topology::{
    Chunk, DistributedInfo, SharedInfo, Tensor, Topology, get_intervals,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

#[derive(Clone)]
struct LocalTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for LocalTensor {
    fn data(&self) -> std::borrow::Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn split(tensor: &LocalTensor, dim: usize, world_size: usize) -> Result<(Tensor, Vec<LocalTensor>)> {
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
                LocalTensor {
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



/// Redistributes a model from one topology to another
fn redistribute_model<P: AsRef<Path>>(
    input_dir: P,
    output_dir: P,
    target_world_size: usize,
) -> Result<()> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();
    
    println!("Reading existing distributed model from {:?}", input_dir);
    
    // Load the existing topology
    let topology_path = input_dir.join("topology.json");
    let topology_data = std::fs::read_to_string(&topology_path)?;
    let source_topology: Topology = serde_json::from_str(&topology_data)?;
    
    println!("Source topology has {} ranks", source_topology.n_ranks());
    println!("Target topology will have {} ranks", target_world_size);
    
    // Process each tensor immediately to avoid lifetime issues
    let mut reconstructed_tensors = HashMap::new();
    
    // Get all tensor names from the source topology
    let tensor_names: Vec<String> = source_topology.tensors().keys().cloned().collect();
    
    for tensor_name in &tensor_names {
        println!("Loading tensor: {}", tensor_name);
        
        // Reconstruct the full tensor immediately
        let full_tensor = reconstruct_tensor_from_files(&source_topology, tensor_name, input_dir)?;
        reconstructed_tensors.insert(tensor_name.clone(), full_tensor);
    }
    
    // Create output directory
    std::fs::create_dir_all(output_dir)?;
    
    // Redistribute each tensor
    let mut new_topology_tensors = HashMap::new();
    let new_filenames: Vec<_> = (0..target_world_size)
        .map(|rank| format!("rank{rank}.safetensors"))
        .collect();
    let mut new_tensor_data: Vec<Vec<(String, LocalTensor)>> =
        new_filenames.iter().map(|_| Vec::new()).collect();
    
    for tensor_name in tensor_names {
        println!("Redistributing tensor: {}", tensor_name);
        
        let full_tensor = &reconstructed_tensors[&tensor_name];
        
        // Determine how to redistribute this tensor
        match source_topology.get_tensor(&tensor_name).unwrap() {
            Tensor::Distributed(_) => {
                // Re-split the tensor for the new world size
                let split_dim = determine_split_dimension(&tensor_name, full_tensor);
                
                if full_tensor.shape()[split_dim] % target_world_size == 0 {
                    let (new_tensor_info, new_tensor_chunks) = split(full_tensor, split_dim, target_world_size)?;
                    new_topology_tensors.insert(tensor_name.clone(), new_tensor_info);
                    
                    for (rank_idx, chunk) in new_tensor_chunks.into_iter().enumerate() {
                        new_tensor_data[rank_idx].push((tensor_name.clone(), chunk));
                    }
                } else {
                    // If not evenly divisible, keep as shared
                    println!("  → Cannot split evenly, keeping as shared tensor");
                    new_topology_tensors.insert(
                        tensor_name.clone(),
                        Tensor::Shared(SharedInfo::new(
                            full_tensor.shape().to_vec(),
                            full_tensor.dtype(),
                            0,
                        )),
                    );
                    // Add shared tensor to ALL rank files
                    for rank_idx in 0..target_world_size {
                        new_tensor_data[rank_idx].push((tensor_name.clone(), full_tensor.clone()));
                    }
                }
            }
            Tensor::Shared(_) => {
                // Keep shared tensors as shared - add to ALL ranks
                new_topology_tensors.insert(
                    tensor_name.clone(),
                    Tensor::Shared(SharedInfo::new(
                        full_tensor.shape().to_vec(),
                        full_tensor.dtype(),
                        0,
                    )),
                );
                // Add shared tensor to ALL rank files
                for rank_idx in 0..target_world_size {
                    new_tensor_data[rank_idx].push((tensor_name.clone(), full_tensor.clone()));
                }
            }
        }
    }
    
    // Create the new topology
    let new_topology = Topology::new(new_topology_tensors, new_filenames.clone(), target_world_size)?;
    
    // Save the new topology
    let topology_json = serde_json::to_vec_pretty(&new_topology)?;
    let mut topology_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_dir.join("topology.json"))?;
    topology_file.write_all(&topology_json)?;
    
    // Save the new rank files
    for (filename, tensor_data) in new_filenames.iter().zip(new_tensor_data.into_iter()) {
        let file_path = output_dir.join(filename);
        serialize_to_file(tensor_data, &None, &file_path)?;
        println!("Saved {}", filename);
    }
    
    println!("Model successfully redistributed from {} to {} ranks", source_topology.n_ranks(), target_world_size);
    println!("Output saved to {:?}", output_dir);
    
    Ok(())
}

/// Reconstructs a tensor from files without lifetime issues
fn reconstruct_tensor_from_files(
    topology: &Topology,
    tensor_name: &str,
    input_dir: &Path,
) -> Result<LocalTensor> {
    match topology.get_tensor(tensor_name) {
        Some(Tensor::Distributed(info)) => {
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
            
            for chunk in info.chunks() {
                let file_idx = chunk.filename_index();
                let filename = &topology.filenames()[file_idx];
                let file_path = input_dir.join(filename);
                
                // Load the file each time to avoid lifetime issues
                let data = std::fs::read(&file_path)?;
                let safetensors = SafeTensors::deserialize(&data)?;
                let tensor = safetensors.tensor(tensor_name)?;
                let chunk_data = tensor.data();
                
                // Use the existing get_intervals function to determine which bytes to copy
                let intervals = get_intervals(chunk, &full_strides, shape);
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
            
            Ok(LocalTensor {
                dtype,
                shape: shape.to_vec(),
                data: result,
            })
        }
        Some(Tensor::Shared(info)) => {
            let file_idx = info.filename_index();
            let filename = &topology.filenames()[file_idx];
            let file_path = input_dir.join(filename);
            
            // Load the file
            let data = std::fs::read(&file_path)?;
            let safetensors = SafeTensors::deserialize(&data)?;
            let tensor = safetensors.tensor(tensor_name)?;
            
            Ok(LocalTensor {
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                data: tensor.data().to_vec(),
            })
        }
        None => Err(anyhow!("Tensor {} not found in topology", tensor_name)),
    }
}

/// Determines the best dimension to split a tensor based on its name and shape
fn determine_split_dimension(tensor_name: &str, tensor: &LocalTensor) -> usize {
    let shape = tensor.shape();
    
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() != 4 {
        eprintln!("Usage: {} <input_dir> <output_dir> <target_world_size>", args[0]);
        eprintln!("Example: {} distributed_gpt2 redistributed_gpt2 2", args[0]);
        std::process::exit(1);
    }
    
    let input_dir = &args[1];
    let output_dir = &args[2];
    let target_world_size: usize = args[3].parse()
        .map_err(|_| anyhow!("Invalid target world size: {}", args[3]))?;
    
    redistribute_model(input_dir, output_dir, target_world_size)?;
    
    Ok(())
} 