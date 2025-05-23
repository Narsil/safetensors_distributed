# SafeTensors Distributed

A Python API for loading SafeTensors files from URLs.

## Installation

```bash
pip install safetensors_distributed
```

## Usage

```python
from safetensors_distributed import SafeTensorsLoader, Plan

# Example URL for the SafeTensors file
url = "https://example.com/path/to/safetensors_file.safetensors"

# Open the SafeTensors file using the context manager
with SafeTensorsLoader(url) as loader:
    # Create a plan to request slices of tensors
    plan = loader.create_plan()
    
    # Add slices for different tensors
    plan.add_slice("tensor1", 0, 10)
    plan.add_slice("tensor2", 5, 15)
    
    # Execute the plan to fetch the tensors
    result = plan.execute(loader)
    
    # Print the result
    print(result)

# Test opening a local SafeTensors file
with SafeTensorsLoader("test.safetensors") as loader:
    metadata = loader.get_metadata()
    print("Metadata:", metadata)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 