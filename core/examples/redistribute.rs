use anyhow::Result;
use clap::Parser;
use reqwest::Client;
use reqwest::header::HeaderMap;
use safetensors_distributed::redistributor::{AsyncTensorRedistributor, load_or_create_topology};
use safetensors_distributed::topology::{Chunk, DistributedInfo, SharedInfo, Tensor, Topology};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;
use url::Url;

#[derive(Parser)]
#[command(name = "redistribute")]
#[command(about = "Redistribute safetensors models between different world sizes")]
struct Args {
    /// Input directory containing topology.json + rank*.safetensors OR model.safetensors
    #[arg(long, conflicts_with = "input_url")]
    input_dir: Option<String>,

    /// Input URL for remote model (HTTP/HTTPS)
    #[arg(long, conflicts_with = "input_dir")]
    input_url: Option<Url>,

    /// Output directory for redistributed files
    #[arg(long, short)]
    output_dir: String,

    /// Target world size (number of ranks)
    #[arg(long, short)]
    target_world_size: usize,
}

impl Args {
    fn validate(&self) -> Result<()> {
        match (&self.input_dir, &self.input_url) {
            (Some(_), Some(_)) => anyhow::bail!("Cannot specify both --input-dir and --input-url"),
            (None, None) => anyhow::bail!("Must specify either --input-dir or --input-url"),
            _ => Ok(()),
        }
    }
}

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

/// Main redistribution function for local input using async parallel approach
async fn redistribute_model_from_local<P: AsRef<Path>>(
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
    let target_topology = create_target_topology(&source_topology, target_world_size).unwrap();
    println!(
        "Target topology will have {} ranks",
        target_topology.world_size()
    );

    // Create and run the async redistributor
    let mut redistributor =
        AsyncTensorRedistributor::from_local(input_dir, output_dir, target_topology)?;

    let _created_files = redistributor.redistribute().await?;
    Ok(())
}

/// Main redistribution function for remote input using async parallel approach
async fn redistribute_model_from_url<P: AsRef<Path>>(
    input_url: &Url,
    output_dir: P,
    target_world_size: usize,
) -> Result<()> {
    let output_dir = output_dir.as_ref();

    println!("Reading model from remote URL: {}", input_url);

    let base_url = input_url.clone();
    
    // For now, use empty auth headers. In the future, this could be configurable
    let auth_headers = HeaderMap::new();
    
    // Create a single HTTP client for connection pooling with aggressive connection reuse
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(300)) // Keep connections alive much longer (5 minutes)
        .pool_max_idle_per_host(8) // Fewer connections but longer lived
        .http2_keep_alive_interval(Duration::from_secs(10)) // More frequent HTTP/2 keep-alive
        .http2_keep_alive_timeout(Duration::from_secs(30)) // Longer HTTP/2 keep-alive timeout
        .http2_keep_alive_while_idle(true) // Keep connections alive even when idle
        .tcp_keepalive(Duration::from_secs(30)) // More frequent TCP-level keep-alive
        .redirect(reqwest::redirect::Policy::default()) // Follow redirects (up to 10)
        .build()
        .expect("Failed to create HTTP client");

    // Load the source topology from the remote URL first
    println!("Loading remote topology...");
    let source_topology =
        AsyncTensorRedistributor::load_or_create_remote_topology(&client, &base_url, &auth_headers)
            .await?;

    let source_ranks = source_topology.world_size();
    println!("Source topology has {} ranks", source_ranks);

    // Create target topology from the source topology (same logic as local)
    let target_topology = create_target_topology(&source_topology, target_world_size)?;
    println!(
        "Target topology will have {} ranks",
        target_topology.world_size()
    );

    // Create and run the async redistributor from URL - pass the same client for connection reuse
    let mut redistributor =
        AsyncTensorRedistributor::from_url_with_client(client, base_url, auth_headers, output_dir, target_topology)
            .await?;
    println!("Initiated");

    let _created_files = redistributor.redistribute().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize env_logger to see reqwest debug logs
    env_logger::init();
    
    let args = Args::parse();

    // Validate that exactly one input source is provided
    args.validate()?;

    match (&args.input_dir, &args.input_url) {
        (Some(input_dir), None) => {
            redistribute_model_from_local(input_dir, &args.output_dir, args.target_world_size)
                .await?;
        }
        (None, Some(input_url)) => {
            redistribute_model_from_url(input_url, &args.output_dir, args.target_world_size)
                .await?;
        }
        _ => unreachable!("Validation should have caught this case"),
    }

    println!("Redistribution completed successfully!");
    Ok(())
}
