use anyhow::Result;
use clap::Parser;
use safetensors_distributed::{
    Redistributor, SharedInfo, Tensor, Topology, load_or_create_topology,
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
}

/// Create a target topology for redistribution based on source topology and target world size
fn create_target_topology(source_topology: &Topology) -> Result<Topology> {
    let mut target_tensors = BTreeMap::new();

    // Generate target filenames
    let target_filenames = vec!["model.safetensors".to_string()];

    // Process each tensor from the source topology
    for (tensor_name, source_tensor) in source_topology.tensors() {
        let (shape, dtype) = match source_tensor {
            Tensor::Distributed(info) => (info.shape().to_vec(), info.dtype()),
            Tensor::Shared(info) => (info.shape().to_vec(), info.dtype()),
        };

        // Keep as shared tensor
        target_tensors.insert(
            tensor_name.clone(),
            Tensor::Shared(SharedInfo::new(shape, dtype, vec![0])),
        );
    }

    Ok(Topology::new(target_tensors, target_filenames, 1)?)
}

/// Main redistribution function for local input using synchronous approach
fn redistribute_model_from_local<P: AsRef<Path>>(input_dir: P, output_dir: P) -> Result<()> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();

    println!("Reading model from local directory: {:?}", input_dir);

    // Load the existing topology (or create from model.safetensors)
    let source_topology = load_or_create_topology(input_dir)?;

    let source_ranks = source_topology.world_size();
    println!("Source topology has {} ranks", source_ranks);

    // Create target topology
    let target_topology = create_target_topology(&source_topology)?;
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

    redistribute_model_from_local(&args.input_dir, &args.output_dir)?;

    println!("Redistribution completed successfully!");
    Ok(())
}
