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

[Add your license information here]
