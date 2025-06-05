use anyhow::Result;
use hf_hub::api::tokio::ApiBuilder;
use safetensors::SafeTensors;
use safetensors_distributed::redistributor::{
    AsyncTensorRedistributor, RedistributorConfig
};
use safetensors_distributed::tensor::TensorData;

use std::collections::HashMap;
use std::path::Path;

/// Load tensors from SafeTensors and convert to TensorData HashMap
fn load_tensors_as_hashmap(tensors: &SafeTensors<'_>) -> Result<HashMap<String, TensorData>> {
    let mut tensor_map = HashMap::new();
    
    for tensor_name in tensors.names() {
        let tensor = tensors.tensor(tensor_name)?;
        let tensor_data: TensorData = tensor.into();
        tensor_map.insert(tensor_name.to_string(), tensor_data);
    }
    
    Ok(tensor_map)
}



/// Creates a distributed version of GPT-2 model with async parallel processing
async fn create_distributed_gpt2<P: AsRef<Path>>(filename: &Path, output_dir: P) -> Result<()> {
    let output_dir = output_dir.as_ref();
    
    // Load the original safetensors file
    let data = std::fs::read(filename)?;
    let tensors = SafeTensors::deserialize(&data)?;

    // Load tensors into HashMap for redistributor
    let tensor_map = load_tensors_as_hashmap(&tensors)?;
    
    let world_size = 4;
    
    println!("Starting async distributed model creation...");
    
    // Configure redistributor to use rankN.safetensors filenames (not model.safetensors)
    let config = RedistributorConfig {
        max_concurrent_files: 50,
        use_model_filename_for_single_rank: false, // Force rank files even for world_size > 1
    };
    
    let redistributor = AsyncTensorRedistributor::new_from_loaded_tensors(
        tensor_map,
        world_size,
        config,
    );
    
    let created_files = redistributor.redistribute(output_dir).await?;
    
    println!("Created distributed model with {} ranks", world_size);
    for filename in &created_files {
        println!("Saved {}", filename);
    }
    
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
