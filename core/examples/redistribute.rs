use anyhow::{Context, Result};
use safetensors_distributed::redistributor::{AsyncTensorRedistributor, load_or_create_topology};
use safetensors_distributed::topology::{Chunk, DistributedInfo, SharedInfo, Tensor, Topology};
use std::collections::BTreeMap;
use std::path::Path;

/// Create a target topology for redistribution based on source topology and target world size
fn create_target_topology(
    source_topology: &Topology,
    target_world_size: usize,
) -> Result<Topology> {
    let mut target_tensors = BTreeMap::new();

    // Generate target filenames
    let target_filenames = if target_world_size == 1 {
        vec!["model.safetensors".to_string()]
    } else {
        (0..target_world_size)
            .map(|rank| format!("rank{rank}.safetensors"))
            .collect()
    };

    // Process each tensor from the source topology
    for (tensor_name, source_tensor) in source_topology.tensors() {
        let (shape, dtype) = match source_tensor {
            Tensor::Distributed(info) => (info.shape().to_vec(), info.dtype()),
            Tensor::Shared(info) => (info.shape().to_vec(), info.dtype()),
        };

        // Determine the best dimension to split based on tensor name and shape
        let split_dim = determine_split_dimension(tensor_name, &shape);

        if shape[split_dim] % target_world_size == 0 {
            // Create distributed tensor
            let chunk_size_per_rank = shape[split_dim] / target_world_size;
            let mut chunks = Vec::new();

            for rank in 0..target_world_size {
                let start = rank * chunk_size_per_rank;
                let end = (rank + 1) * chunk_size_per_rank;
                let mut chunk_shape = shape.clone();
                chunk_shape[split_dim] = end - start;

                // Create offsets array for this chunk within the tensor
                let mut chunk_offsets = vec![0; chunk_shape.len()];
                chunk_offsets[split_dim] = start;

                let chunk = Chunk::new(chunk_offsets, chunk_shape, rank);
                chunks.push(chunk);
            }

            target_tensors.insert(
                tensor_name.clone(),
                Tensor::Distributed(DistributedInfo::new(shape, dtype, chunks)),
            );
        } else {
            // Keep as shared tensor
            target_tensors.insert(
                tensor_name.clone(),
                Tensor::Shared(SharedInfo::new(shape, dtype, vec![0])),
            );
        }
    }

    Ok(Topology::new(
        target_tensors,
        target_filenames,
        target_world_size,
    )?)
}

/// Determine the best dimension to split a tensor based on its name and shape
fn determine_split_dimension(tensor_name: &str, shape: &[usize]) -> usize {
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
    let source_topology = load_or_create_topology(input_dir)?;

    let source_ranks = source_topology.world_size();
    println!("Source topology has {} ranks", source_ranks);

    // Create target topology
    let target_topology = create_target_topology(&source_topology, target_world_size).unwrap();
    println!(
        "Target topology will have {} ranks",
        target_topology.world_size()
    );

    // Create and run the async redistributor
    let redistributor = AsyncTensorRedistributor::new(input_dir, output_dir, target_topology)?;

    let _created_files = redistributor.redistribute().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <input_dir> <output_dir> <target_world_size>",
            args[0]
        );
        eprintln!(
            "Example: {} distributed_gpt2 redistributed_gpt2_async 2",
            args[0]
        );
        eprintln!("        {} single_model_dir redistributed_async 4", args[0]);
        eprintln!("Input: topology.json + rank*.safetensors OR model.safetensors");
        eprintln!("Note: target_world_size=1 creates model.safetensors (no topology.json)");
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = &args[2];
    let target_world_size: usize = args[3]
        .parse()
        .with_context(|| format!("Invalid target world size: {}", args[3]))?;

    redistribute_model_async(input_dir, output_dir, target_world_size).await?;

    Ok(())
}
