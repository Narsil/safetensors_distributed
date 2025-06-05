use anyhow::{Result, Context};
use safetensors_distributed::redistributor::{
    AsyncTensorRedistributor, RedistributorConfig, load_or_create_topology
};
use std::path::Path;





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
    
    let source_ranks = source_topology.n_ranks();
    println!("Source topology has {} ranks", source_ranks);
    println!("Target topology will have {} ranks", target_world_size);
    
    // Create and run the async redistributor
    let config = RedistributorConfig::default();
    let redistributor = AsyncTensorRedistributor::new_from_topology(
        source_topology,
        input_dir,
        target_world_size,
        config,
    );
    
    let created_files = redistributor.redistribute(output_dir).await?;
    
    // Print created files and success message
    for filename in &created_files {
        println!("Saved {}", filename);
    }
    
    println!("Model successfully redistributed from {} to {} ranks", 
             source_ranks, target_world_size);
    println!("Output saved to {:?}", output_dir);
    
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
        .with_context(|| format!("Invalid target world size: {}", args[3]))?;
    
    redistribute_model_async(input_dir, output_dir, target_world_size).await?;
    
    Ok(())
} 