use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{
    Repo, RepoType,
    api::tokio::{Api, ApiRepo},
};
use safetensors_distributed::redistributor::{
    AsyncTensorRedistributor, RedistributorConfig, load_or_create_topology,
};
use safetensors_distributed::topology::Topology;
use std::fs;
use std::path::{Path, PathBuf};

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

    // Initialize the Hugging Face Hub API
    let api = ApiBuilder::from_env()
        .high()
        .build()
        .context("Failed to initialize Hugging Face Hub API")?;
    let repo = Repo::with_revision(args.repo.clone(), RepoType::Model, args.revision.clone());
    let api_repo = api.repo(repo);

    // Create output directory
    fs::create_dir_all(&args.output_dir).context("Failed to create output directory")?;

    // Try to get topology.json
    let topology_path = args.output_dir.join("topology.json");
    let has_topology = match api_repo.get("topology.json").await {
        Ok(cached_path) => {
            fs::copy(&cached_path, &topology_path).context("Failed to copy topology.json")?;
            true
        }
        Err(_) => false,
    };

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

    let config = RedistributorConfig {
        max_concurrent_files: args.max_concurrent_files,
        use_model_filename_for_single_rank: true,
    };

    let redistributor = AsyncTensorRedistributor::new_from_topology(
        source_topology,
        &args.output_dir,
        args.world_size,
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

