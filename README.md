# Safetensors Distributed

A Python package for efficient sharding and de-sharding of safetensors model checkpoints.

## Overview

Safetensors Distributed provides tools to work with sharded model checkpoints that use the safetensors format. It enables you to:

- Download and fuse sharded checkpoints from distributed storage
- Prepare checkpoints for distributed training
- Work seamlessly with inference tools like TGI, vLLM, or SGLang

## Getting Started

Install the package using pip:

```bash
pip install safetensors_distributed
```

## Usage

### Working with Sharded Checkpoints

This package is particularly useful when you need to:

1. Download a sharded checkpoint created with DCP (Distributed Checkpoint) or similar tools
2. Fuse the sharded checkpoint for evaluation or inference
3. Work with inference tools that expect a single checkpoint file

The package supports checkpoints that include:
- Safetensors files
- Topology information (topology.json)

### Topology.json Format

The `topology.json` file describes how tensors are distributed across different ranks in a sharded checkpoint. It has the following structure:

```json
{
    "tensors": {
        "tensor_name": {
            "type": "Distributed",
            "shape": [dim1, dim2, ...],
            "dtype": "F32",
            "chunks": [
                {
                    "offsets": [offset1, offset2, ...],
                    "shape": [chunk_dim1, chunk_dim2, ...],
                    "filename_index": 0
                },
                // ... more chunks for other ranks
            ]
        },
        // ... more tensors
    },
    "filenames": ["rank0.safetensors", "rank1.safetensors", ...],
    "world_size": 2
}
```

Key components:
- `tensors`: A map of tensor names to their distribution information
  - `type`: Either "Distributed" (split across ranks) or "Shared" (replicated on all ranks)
  - `shape`: The full shape of the tensor
  - `dtype`: The data type (e.g., "F32", "BF16")
  - `chunks`: For distributed tensors, describes how the tensor is split
    - `offsets`: Starting position in each dimension
    - `shape`: Size of the chunk in each dimension
    - `filename_index`: Index into the filenames array
- `filenames`: List of safetensors files containing the tensor chunks
- `world_size`: Number of ranks in the distributed setup

#### Understanding Chunks

Each chunk represents a slice of the full tensor. The relationship between a chunk's `offsets` and `shape` and the corresponding slice of the full tensor follows this pattern:

For a chunk with `offsets: [x, y]` and `shape: [a, b]`, the chunk contains the data from the full tensor slice `tensor[x:x+a, y:y+b]`.

More generally, for an n-dimensional tensor, a chunk with `offsets: [o₁, o₂, ..., oₙ]` and `shape: [s₁, s₂, ..., sₙ]` corresponds to:
```
tensor[o₁:o₁+s₁, o₂:o₂+s₂, ..., oₙ:oₙ+sₙ]
```

This slice notation makes it clear which portion of the original tensor each chunk represents.

### Example Workflow

1. Create a distributed checkpoint using your preferred tool (e.g., DCP)
2. Use safetensors_distributed to download and fuse the checkpoint on a separate node
3. Use the fused checkpoint with inference tools like TGI, vLLM, or SGLang

## Features

- Efficient sharding and de-sharding of safetensors checkpoints
- Support for distributed storage systems
- Compatible with major inference frameworks
- Python-native implementation

## License

This project is licensed under both the MIT License and Apache License 2.0 at your option.

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)
