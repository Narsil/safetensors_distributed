use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{
    Repo, RepoType,
    api::tokio::Api,
};
use safetensors_distributed::redistributor::{
    AsyncTensorRedistributor, RedistributorConfig, load_or_create_topology,
};
use safetensors_distributed::topology::{Topology, Tensor, DistributedInfo, SharedInfo, Chunk};
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;

/// Create a target topology for redistribution based on source topology and target world size
fn create_target_topology(
    source_topology: &Topology,
    target_world_size: usize,
) -> Result<Topology> {
    let mut target_tensors = HashMap::new();
    
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
                Tensor::Shared(SharedInfo::new(shape, dtype, 0)),
            );
        }
    }

    Ok(Topology::new(target_tensors, target_filenames, target_world_size)?)
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The model repository to download from
    #[arg(short, long)]
    repo: String,

    /// The revision to download
    #[arg(long, default_value = "main")]
    revision: String,

    /// The output directory to save the redistributed files
    #[arg(short, long)]
    output_dir: PathBuf,

    /// The target world size (number of ranks)
    #[arg(short, long, default_value = "1")]
    world_size: usize,

    /// Maximum number of concurrent file operations
    #[arg(long, default_value = "50")]
    max_concurrent_files: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Downloading model {} (revision: {})...", args.repo, args.revision);

    // Create API and repo
    let api = Api::new().context("Failed to create HF API")?;
    let api_repo = api.repo(Repo::with_revision(
        args.repo.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).context("Failed to create output directory")?;

    // Check if the repository has a topology.json (distributed model)
    let topology_path = args.output_dir.join("topology.json");
    let has_topology = api_repo.get("topology.json").await.is_ok();

    if has_topology {
        println!("Found topology.json in repository, downloading distributed files...");
        // Load the topology to get the list of files we need
        let topology_data =
            fs::read_to_string(&topology_path).context("Failed to read topology.json")?;
        let topology: Topology =
            serde_json::from_str(&topology_data).context("Failed to parse topology.json")?;

        // Download all the safetensors files
        for filename in topology.filenames() {
            println!("Downloading {}...", filename);
            let file_path = args.output_dir.join(filename);
            let cached_path = api_repo
                .get(filename)
                .await
                .context(format!("Failed to fetch {}", filename))?;
            fs::copy(&cached_path, &file_path).context(format!("Failed to copy {}", filename))?;
        }
    } else {
        println!("No topology.json found, downloading single model.safetensors file...");
        // Download the single model file
        let model_path = args.output_dir.join("model.safetensors");
        let cached_path = api_repo
            .get("model.safetensors")
            .await
            .context("Failed to fetch model.safetensors")?;
        fs::copy(&cached_path, &model_path).context("Failed to copy model.safetensors")?;
    }

    // Load the topology and redistribute
    let source_topology = load_or_create_topology(&args.output_dir)
        .await
        .context("Failed to load topology")?;

    // Create target topology
    let target_topology = create_target_topology(&source_topology, args.world_size)?;

    let config = RedistributorConfig {
        max_concurrent_files: args.max_concurrent_files,
        use_model_filename_for_single_rank: true,
    };

    let redistributor = AsyncTensorRedistributor::new_from_topology(
        source_topology,
        &args.output_dir,
        target_topology,
        config,
    );

    println!("Redistributing tensors to {} ranks...", args.world_size);
    let created_files = redistributor
        .redistribute(&args.output_dir)
        .await
        .context("Failed to redistribute tensors")?;

    println!("Successfully redistributed tensors. Created files:");
    for file in created_files {
        println!("  - {}", file);
    }

    Ok(())
}

