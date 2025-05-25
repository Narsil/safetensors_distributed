from safetensors_distributed import dist_loader

# Example URL for the SafeTensors file
url = "https://huggingface.co/Qwen/Qwen3-32B/resolve/main/model-00001-of-00017.safetensors"

# Open the SafeTensors file using the context manager
with dist_loader(url) as loader:
    # Create a plan to request slices of tensors
    print(loader.metadata())
    plan = loader.create_plan()

    # # Add slices for different tensors
    # plan.add_slice("tensor1", 0, 10)
    # plan.add_slice("tensor2", 5, 15)

    # # Execute the plan to fetch the tensors
    # result = plan.execute(loader)

    # # Print the result
    # print(result)
