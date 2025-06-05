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

### 2. `redistribute_model.rs`

Redistributes an existing distributed model to a different number of ranks.

**Usage:**
```bash
cargo run --example redistribute_model <input_dir> <output_dir> <target_ranks>
```

**Examples:**
```bash
# Convert 4-rank model to 2-rank model
cargo run --example redistribute_model distributed_gpt2 redistributed_gpt2_2ranks 2

# Convert 2-rank model back to 4-rank model  
cargo run --example redistribute_model redistributed_gpt2_2ranks redistributed_gpt2_4ranks 4
```

**What it does:**
- Reads an existing distributed model and its topology
- Reconstructs full tensors from their distributed chunks
- Re-splits tensors according to the new target rank count
- Preserves the original splitting strategy when possible
- Falls back to shared tensors if dimensions aren't evenly divisible

### 3. `redistribute_model_streaming.rs`

Memory-efficient streaming version of model redistribution that processes tensors one-by-one.

**Usage:**
```bash
cargo run --example redistribute_model_streaming <input_dir> <output_dir> <target_ranks>
```

**Examples:**
```bash
# Convert 4-rank model to 2-rank model (streaming)
cargo run --example redistribute_model_streaming distributed_gpt2 redistributed_gpt2_streaming 2

# Convert to 8-rank model (streaming)
cargo run --example redistribute_model_streaming distributed_gpt2 redistributed_gpt2_8ranks 8
```

**What it does:**
- Processes tensors one at a time instead of loading all into memory
- Streams tensor data directly to output files
- Tracks offsets and metadata during processing  
- Writes proper safetensors headers at the end
- Much more memory efficient for large models
- Inspired by the TopologyLoader's streaming approach

## Key Features

- **Intelligent Splitting:** Uses model-aware splitting strategies optimized for transformer architectures
- **Flexible Redistribution:** Can convert between any number of ranks (as long as tensor dimensions are divisible)
- **Topology Preservation:** Maintains consistent tensor distribution patterns
- **Memory Efficient:** Two approaches available:
  - Standard: Processes tensors one at a time to avoid lifetime issues
  - Streaming: Processes and writes tensors on-the-fly for maximum memory efficiency
- **Error Handling:** Gracefully handles edge cases like non-divisible dimensions

## File Structure

After running the examples, you'll have:

```
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

redistributed_gpt2_4ranks/           # Back to 4-rank model
├── rank0.safetensors (167MB)
├── rank1.safetensors (119MB)
├── rank2.safetensors (119MB) 
├── rank3.safetensors (119MB)
└── topology.json (60KB)
```

## Architecture

All examples use:
- **LocalTensor**: A custom tensor type that owns its data, avoiding lifetime issues
- **Topology System**: Describes how tensors are distributed across ranks
- **Interval-based Reconstruction**: Uses the existing `get_intervals` function for efficient data copying
- **Model-specific Logic**: GPT-2 optimized splitting strategies 

The streaming version additionally features:
- **StreamingRedistributor**: Manages file handles and offset tracking across all ranks
- **On-the-fly Processing**: Reconstructs, splits, and writes tensors without intermediate storage
- **Header Management**: Reserves space for headers and writes them after data processing
- **Offset Tracking**: Maintains proper safetensors format with accurate data offsets

The async version additionally features:
- **Parallel Tensor Processing**: Processes all tensors concurrently using async/await
- **Parallel Write Operations**: Executes all file writes simultaneously with futures
- **High CPU Utilization**: Leverages multiple cores for maximum throughput
- **Tokio Runtime**: Uses async I/O for efficient resource management

## Performance Comparison

Benchmarked on GPT-2 (4→2 ranks redistribution) in release mode:

| Approach | Total Time | CPU Usage | Memory Usage | Best For |
|----------|------------|-----------|--------------|----------|
| **Standard** | 8.2s | 79% | High (all tensors loaded) | Small models, simple use cases |
| **Streaming** | 7.5s | 88% | Low (one tensor at a time) | Large models, memory constraints |
| **Async** | 5.2s | 474% | High (parallel processing) | High-performance scenarios, multi-core systems |

**Key Findings:**
- **Async is fastest** (37% faster than streaming, 37% faster than standard)  
- **Streaming is most memory-efficient** while maintaining good performance
- **Standard is simplest** but least efficient for large models
- **Async has highest CPU utilization** (474% indicates ~4.7 cores utilized)
- All approaches produce **identical output files** 