//! Example demonstrating different redistribution strategies
//! 
//! This example shows how to use different redistribution strategies to optimize
//! performance for different scenarios:
//! 
//! - ReadSerialWriteUnordered: Best for local storage with slow random reads
//! - ReadUnorderedWriteSerial: Best for fast storage with slow random writes

use safetensors_distributed::redistributor::AsyncTensorRedistributor;
use safetensors_distributed::redistributor::RedistributionStrategy;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let source_dir = std::env::args().nth(1)
        .expect("Usage: strategy_comparison <source_dir> <target_dir>");
    let target_dir = std::env::args().nth(2)
        .expect("Usage: strategy_comparison <source_dir> <target_dir>");

    // Load source topology
    let source_topology = safetensors_distributed::redistributor::load_or_create_topology(&source_dir)?;
    println!("Source topology: {} files, {} ranks", 
             source_topology.filenames().len(), 
             source_topology.world_size());

    // Create target topology (single rank for this example)
    use safetensors_distributed::topology::{Topology, Tensor, SharedInfo};
    use std::collections::BTreeMap;
    
    let mut target_tensors = BTreeMap::new();
    for (tensor_name, tensor) in source_topology.tensors() {
        // Convert all tensors to shared on rank 0
        let (shape, dtype) = match tensor {
            Tensor::Distributed(dist_info) => (dist_info.shape(), dist_info.dtype()),
            Tensor::Shared(shared_info) => (shared_info.shape(), shared_info.dtype()),
        };
        target_tensors.insert(
            tensor_name.clone(),
            Tensor::Shared(SharedInfo::new(shape.to_vec(), dtype, vec![0])),
        );
    }
    
    let target_topology = Topology::new(
        target_tensors,
        vec!["consolidated.safetensors".to_string()],
        1,
    )?;

    // Test different strategies
    let strategies = vec![
        ("ReadSerial+WriteUnordered", RedistributionStrategy::ReadSerialWriteUnordered),
        ("ReadUnordered+WriteSerial", RedistributionStrategy::ReadUnorderedWriteSerial),
    ];

    for (name, strategy) in strategies {
        println!("\n=== Testing Strategy: {} ===", name);
        
        let strategy_dir = format!("{}_strategy_{}", target_dir, name.replace("+", "_"));
        std::fs::create_dir_all(&strategy_dir)?;
        
        let start = Instant::now();
        
        let mut redistributor = AsyncTensorRedistributor::from_local(
            &source_dir,
            &strategy_dir,
            target_topology.clone(),
        )?.with_strategy(strategy);

        let result = redistributor.redistribute().await;
        let duration = start.elapsed();
        
        match result {
            Ok(files) => {
                println!("✅ Success in {:?}", duration);
                println!("   Created files: {:?}", files);
                
                // Calculate total size
                let mut total_size = 0;
                for file in &files {
                    if let Ok(metadata) = std::fs::metadata(format!("{}/{}", strategy_dir, file)) {
                        total_size += metadata.len();
                    }
                }
                let throughput = total_size as f64 / duration.as_secs_f64() / 1024.0 / 1024.0;
                println!("   Throughput: {:.2} MB/s", throughput);
            }
            Err(e) => {
                println!("❌ Failed: {}", e);
            }
        }
    }

    Ok(())
} 