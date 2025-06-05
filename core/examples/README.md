# Safetensors Distributed Examples

This directory contains examples demonstrating how to work with distributed safetensors models.

## Examples

### 1. `create_distributed.rs`

Creates a distributed version of GPT-2 by downloading the model and splitting tensors across multiple ranks.

**Usage:**
```bash
cargo run --example create_distributed
```

**What it does:**
- Downloads GPT-2 model from Hugging Face Hub
- Analyzes each tensor and determines optimal splitting strategy
- Creates a 4-rank distributed model with:
  - `rank0.safetensors`, `rank1.safetensors`, `rank2.safetensors`, `rank3.safetensors`
  - `topology.json` describing how tensors are distributed
- Uses model-specific splitting rules:
  - Attention/FC weights split along dimension 1
  - Projection weights split along dimension 0  
  - Layer norms and biases kept as shared tensors

### 2. `redistribute.rs`

Redistributes an existing distributed model to a different number of ranks.

**Usage:**
```bash
cargo run --example redistribute <input_dir> <output_dir> <target_ranks>
```

**Examples:**
```bash
# Convert 4-rank model to 2-rank model
cargo run --example redistribute distributed_gpt2 redistributed_gpt2_2ranks 2

# Convert 2-rank model back to 4-rank model  
cargo run --example redistribute redistributed_gpt2_2ranks redistributed_gpt2_4ranks 4

# Convert to single rank (non-sharded checkpoint, no topology.json)
cargo run --example redistribute distributed_gpt2 redistributed_gpt2_single 1

# Convert from single model.safetensors to distributed model
cargo run --example redistribute single_model_dir distributed_4ranks 4

# Convert from single model.safetensors to single rank (creates model.safetensors)
cargo run --example redistribute single_model_dir redistributed_single 1

# Round-trip example: single → distributed → single
cargo run --example redistribute original_model distributed_temp 4
cargo run --example redistribute distributed_temp final_model 1
# original_model/model.safetensors → final_model/model.safetensors
```

**What it does:**
- Reads distributed models (with `topology.json`) OR single model files (`model.safetensors`)
- Reconstructs full tensors from their distributed chunks (or loads directly from single file)
- Re-splits tensors according to the new target rank count
- Preserves the original splitting strategy when possible
- Falls back to shared tensors if dimensions aren't evenly divisible
- **Special case:** When target rank count is 1, creates `model.safetensors` without `topology.json`

**Additional notes:**
- The redistribute example uses async streaming processing for efficiency
- Processes tensors in parallel and streams data directly to output files
- Memory-efficient approach inspired by the TopologyLoader's streaming design
- Pre-calculates file layouts and offsets for optimal performance

## Key Features

- **Intelligent Splitting:** Uses model-aware splitting strategies optimized for transformer architectures
- **Flexible Input:** Reads both distributed models (with topology.json) and single model files (model.safetensors)
- **Flexible Redistribution:** Can convert between any number of ranks (as long as tensor dimensions are divisible)
- **Topology Preservation:** Maintains consistent tensor distribution patterns
- **Memory Efficient:** Uses async streaming to process and write tensors on-the-fly for maximum memory efficiency
- **Error Handling:** Gracefully handles edge cases like non-divisible dimensions

## File Structure

After running the examples, you'll have:

```
single_model_dir/                    # Single model input (topology-unaware)
└── model.safetensors (524MB)        # Standard safetensors file

distributed_gpt2/                    # Original 4-rank model
├── rank0.safetensors (167MB)
├── rank1.safetensors (119MB) 
├── rank2.safetensors (119MB)
├── rank3.safetensors (119MB)
└── topology.json (60KB)

redistributed_gpt2_2ranks/           # 2-rank redistributed model
├── rank0.safetensors (286MB)
├── rank1.safetensors (237MB)
└── topology.json (42KB)

redistributed_gpt2_single/           # 1-rank non-sharded model
└── model.safetensors (524MB)        # No topology.json needed

distributed_4ranks/                  # Distributed from single model
├── rank0.safetensors (131MB)
├── rank1.safetensors (131MB)
├── rank2.safetensors (131MB)
├── rank3.safetensors (131MB)
└── topology.json (60KB)

redistributed_gpt2_4ranks/           # Back to 4-rank model
├── rank0.safetensors (167MB)
├── rank1.safetensors (119MB)
├── rank2.safetensors (119MB) 
├── rank3.safetensors (119MB)
└── topology.json (60KB)
```

## Architecture

All examples use:
- **TensorData**: A custom tensor type that owns its data, avoiding lifetime issues
- **Topology System**: Describes how tensors are distributed across ranks
- **Interval-based Reconstruction**: Uses the existing `get_intervals` function for efficient data copying
- **Model-specific Logic**: GPT-2 optimized splitting strategies 

The redistribute example features:
- **AsyncStreamingRedistributor**: Manages file handles and offset tracking across all ranks
- **Pre-calculated Layout**: Determines all file structures and offsets before processing
- **Parallel Tensor Processing**: Processes all tensors concurrently using async/await
- **On-the-fly Processing**: Reconstructs, splits, and writes tensors without intermediate storage
- **Header Management**: Reserves space for headers and writes them after data processing
- **Offset Tracking**: Maintains proper safetensors format with accurate data offsets
- **Tokio Runtime**: Uses async I/O for efficient resource management
- **Non-sharded Support**: Creates model.safetensors when target rank count is 1 (no topology.json)
- **Topology-unaware Input**: Auto-detects and handles single model.safetensors files
- **Round-trip Compatible**: Single model → distributed → single model preserves filename consistency

## Performance

The redistribute example uses async streaming processing for optimal performance:

- **High CPU Utilization**: Leverages multiple cores for parallel tensor processing (~4.7 cores)
- **Memory Efficient**: Processes tensors on-the-fly without loading all data into memory
- **Fast I/O**: Uses async file operations to maximize throughput
- **Pre-calculated Layout**: Determines all file structures upfront to minimize overhead
- **Parallel writes**: Executes all file writes simultaneously with futures

**Key Features:**
- Processes large models efficiently regardless of size
- Maintains low memory footprint through streaming
- Produces identical output to traditional approaches
- Supports both sharded (multi-rank) and non-sharded (single-rank) outputs
- Maintains filename consistency for round-trip compatibility 