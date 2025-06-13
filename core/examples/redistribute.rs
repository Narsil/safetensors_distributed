use anyhow::Result;
use clap::Parser;
use safetensors_distributed::{
    Chunk, DistributedInfo, Redistributor, SharedInfo, Tensor, Topology, load_or_create_topology,
};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Parser)]
#[command(name = "redistribute")]
#[command(about = "Redistribute safetensors models between different world sizes")]
struct Args {
    /// Input directory containing topology.json + rank*.safetensors OR model.safetensors
    #[arg(long)]
    input_dir: String,

    /// Output directory for redistributed files
    #[arg(long, short)]
    output_dir: String,

    /// Target world size (number of ranks)
    #[arg(long, short)]
    target_world_size: usize,
}

/// Create a target topology for redistribution based on source topology and target world size
fn create_target_topology(
    source_topology: &Topology,
    target_world_size: usize,
) -> Result<Topology> {
    let mut target_tensors = BTreeMap::new();

    let chunk_size: usize = 5 * 1024 * 1024 * 1204;
    let mut current_sizes = vec![0; target_world_size];
    // Process each tensor from the source topology
    for (tensor_name, source_tensor) in source_topology.tensors() {
        let (shape, dtype) = match source_tensor {
            Tensor::Distributed(info) => (info.shape().to_vec(), info.dtype()),
            Tensor::Shared(info) => (info.shape().to_vec(), info.dtype()),
        };

        // Determine the best dimension to split based on tensor name and shape
        let split_dim = determine_split_dimension(tensor_name);
        if let Some(split_dim) = split_dim {
            assert_eq!(
                shape[split_dim] % target_world_size,
                0,
                "{tensor_name}: {shape:?} is not divisible by {target_world_size} as location {split_dim}"
            );
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

                let local_size = chunk_shape.iter().product::<usize>() * dtype.size();
                current_sizes[rank] += local_size;
                let filename_index = current_sizes[rank] / chunk_size;
                let chunk = Chunk::new(chunk_offsets, chunk_shape, filename_index);

                chunks.push(chunk);
            }

            target_tensors.insert(
                tensor_name.clone(),
                Tensor::Distributed(DistributedInfo::new(shape, dtype, chunks)),
            );
        } else {
            // Keep as shared tensor

            let local_size = shape.iter().product::<usize>() * dtype.size();
            current_sizes.iter_mut().for_each(|s| *s += local_size);
            let filename_indices: Vec<_> = current_sizes
                .iter()
                .enumerate()
                .map(|(r, current_size)| (current_size / chunk_size * r))
                .collect();
            target_tensors.insert(
                tensor_name.clone(),
                Tensor::Shared(SharedInfo::new(shape, dtype, filename_indices)),
            );
        }
    }

    let target_filenames: Vec<_> = current_sizes
        .into_iter()
        .enumerate()
        .map(|(r, s)| {
            let nfiles = (s + chunk_size - 1) / chunk_size;

            (0..nfiles).map(move |x| format!("model_{r}_{x}.safetensors"))
        })
        .flatten()
        .collect();

    println!("n_files {:?}", target_filenames);

    // // Generate target filenames
    // let target_filenames = if target_world_size == 1 {
    //     vec!["model.safetensors".to_string()]
    // } else {
    //     (0..target_world_size)
    //         .map(|rank| format!("rank{rank}.safetensors"))
    //         .collect()
    // };

    Ok(Topology::new(
        target_tensors,
        target_filenames,
        target_world_size,
    )?)
}

/// Determine the best dimension to split a tensor based on its name and shape
fn determine_split_dimension(tensor_name: &str) -> Option<usize> {
    // Use the same logic as the original GPT-2 splitting
    if [
        "c_fc", "c_attn", "wpe", "wte", "lm_head", "q_proj", "k_proj", "v_proj", "up", "gate",
    ]
    .iter()
    .any(|f| tensor_name.contains(f))
        && !tensor_name.contains("bias")
    {
        Some(1) // Split along dimension 1
    } else if ["c_attn", "c_proj", "o_proj", "down"]
        .iter()
        .any(|f| tensor_name.contains(f))
        && !tensor_name.contains("bias")
    {
        Some(0) // Split along dimension 0
    // } else if tensor_name.contains("bias") {
    //     Some(0) // Split along dimension 0
    } else if ["norm", "bias"].iter().any(|f| tensor_name.contains(f)) {
        None // This is a shared tensor across ranks
    } else {
        // Default: split along the largest dimension
        Some(0)
    }
}

/// Main redistribution function for local input using synchronous approach
fn redistribute_model_from_local<P: AsRef<Path>>(
    input_dir: P,
    output_dir: P,
    target_world_size: usize,
) -> Result<()> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();

    println!("Reading model from local directory: {:?}", input_dir);

    // Load the existing topology (or create from model.safetensors)
    let source_topology = load_or_create_topology(input_dir)?;

    let source_ranks = source_topology.world_size();
    println!("Source topology has {} ranks", source_ranks);

    // Create target topology
    let target_topology = create_target_topology(&source_topology, target_world_size)?;
    println!(
        "Target topology will have {} ranks",
        target_topology.world_size()
    );

    // Create and run the redistributor
    let mut redistributor = Redistributor::from_local(input_dir, output_dir, target_topology)?;

    let _created_files = redistributor.redistribute()?;
    Ok(())
}

fn main() -> Result<()> {
    // Initialize env_logger to see reqwest debug logs
    env_logger::init();

    let args = Args::parse();

    redistribute_model_from_local(&args.input_dir, &args.output_dir, args.target_world_size)?;

    println!("Redistribution completed successfully!");
    Ok(())
}
