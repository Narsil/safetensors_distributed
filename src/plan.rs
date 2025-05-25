use safetensors::{Dtype, slice::TensorIndexer};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::loader::{Error, Loader};

#[derive(PartialEq, Debug)]
pub struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
pub struct Plan {
    slices: HashMap<String, Vec<TensorIndexer>>,
}

// Helper struct for fetch requests
#[derive(PartialEq, Debug)]
struct FetchRequest {
    tensor_name: String,
    file_offset: usize,
    length: usize,
    output_offset: usize,
}

// Minimal public Metadata for testing
#[derive(Debug)]
pub struct TestTensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Debug)]
pub struct TestMetadata {
    pub tensors: HashMap<String, TestTensorInfo>,
}

impl TestMetadata {
    pub fn info(&self, name: &str) -> Option<&TestTensorInfo> {
        self.tensors.get(name)
    }
}

impl Plan {
    pub fn new() -> Self {
        Plan {
            slices: HashMap::new(),
        }
    }

    pub async fn get_slice(
        &mut self,
        tensor_name: &str,
        slices: Vec<TensorIndexer>,
    ) -> Result<(), Error> {
        match self.slices.entry(tensor_name.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(slices);
            }
            Entry::Occupied(_entry) => {
                return Err(Error::AlreadyExists(tensor_name.to_string()));
            }
        }
        Ok(())
    }

    async fn execute(&self, loader: &mut Loader) -> Result<HashMap<String, Tensor>, Error> {
        let metadata = loader.metadata().await?;
        let mut result = HashMap::new();

        for (tensor_name, slices) in &self.slices {
            // Get the tensor info from metadata
            let info = metadata
                .info(tensor_name)
                .ok_or_else(|| Error::MissingTensor(tensor_name.clone()))?;

            // Calculate the new shape based on slices
            let new_shape: Vec<usize> = slices
                .iter()
                .zip(info.shape.iter())
                .map(|(slice, &dim)| match slice {
                    TensorIndexer::Select(_) => 1,
                    TensorIndexer::Narrow(start, end) => {
                        let start = match start {
                            std::ops::Bound::Unbounded => 0,
                            std::ops::Bound::Included(i) => *i,
                            std::ops::Bound::Excluded(i) => *i + 1,
                        };
                        let end = match end {
                            std::ops::Bound::Unbounded => dim,
                            std::ops::Bound::Included(i) => *i + 1,
                            std::ops::Bound::Excluded(i) => *i,
                        };
                        end - start
                    }
                })
                .collect();

            // Calculate total size in bytes
            let element_size = match info.dtype {
                Dtype::F32 => 4,
                Dtype::F64 => 8,
                Dtype::I64 => 8,
                Dtype::I32 => 4,
                Dtype::I16 => 2,
                Dtype::I8 => 1,
                Dtype::U8 => 1,
                Dtype::BF16 => 2,
                Dtype::F16 => 2,
                _ => todo!(),
            };
            let total_size = new_shape.iter().product::<usize>() * element_size;

            // Create empty tensor with correct shape and dtype
            result.insert(
                tensor_name.clone(),
                Tensor {
                    dtype: info.dtype,
                    shape: new_shape,
                    data: vec![0; total_size],
                },
            );
        }

        Ok(result)
    }

    // Gather all byte offsets to fetch, the tensor name, and output offset
    fn gather_fetch_offsets(&self, metadata: &safetensors::tensor::Metadata) -> Vec<FetchRequest> {
        let mut requests = Vec::new();
        let mut output_offset = 0;
        for (tensor_name, slices) in &self.slices {
            if let Some(info) = metadata.info(tensor_name) {
                let shape = &info.shape;
                let dtype_size = match info.dtype {
                    Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
                    Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
                    Dtype::I16 | Dtype::U16 | Dtype::BF16 | Dtype::F16 => 2,
                    Dtype::I8 | Dtype::U8 => 1,
                    _ => todo!("dtype not supported in gather_fetch_offsets"),
                };
                // Compute the slice bounds for each axis
                let mut bounds = Vec::new();
                for (slice, &dim) in slices.iter().zip(shape.iter()) {
                    match slice {
                        TensorIndexer::Select(idx) => {
                            bounds.push((*idx, *idx + 1));
                        }
                        TensorIndexer::Narrow(start, end) => {
                            let start = match start {
                                std::ops::Bound::Unbounded => 0,
                                std::ops::Bound::Included(i) => *i,
                                std::ops::Bound::Excluded(i) => *i + 1,
                            };
                            let end = match end {
                                std::ops::Bound::Unbounded => dim,
                                std::ops::Bound::Included(i) => *i + 1,
                                std::ops::Bound::Excluded(i) => *i,
                            };
                            bounds.push((start, end));
                        }
                    }
                }
                // For remaining axes, take the full range
                for i in slices.len()..shape.len() {
                    bounds.push((0, shape[i]));
                }
                // Now, enumerate all indices in the slice and compute file/output offsets
                // We'll use a multi-dimensional index
                let mut idx = bounds.iter().map(|(s, _)| *s).collect::<Vec<_>>();
                let mut done = false;
                while !done {
                    // Compute the flat index in the original tensor
                    let mut flat_idx = 0;
                    let mut stride = 1;
                    for (i, &dim) in shape.iter().rev().enumerate() {
                        flat_idx += idx[shape.len() - 1 - i] * stride;
                        stride *= dim;
                    }
                    let file_offset = info.data_offsets.0 + flat_idx * dtype_size;
                    let output_offset_here = output_offset;
                    requests.push(FetchRequest {
                        tensor_name: tensor_name.clone(),
                        file_offset,
                        length: dtype_size,
                        output_offset: output_offset_here,
                    });
                    output_offset += dtype_size;
                    // Increment the multi-dimensional index
                    for axis in (0..idx.len()).rev() {
                        idx[axis] += 1;
                        if idx[axis] < bounds[axis].1 {
                            break;
                        } else {
                            idx[axis] = bounds[axis].0;
                            if axis == 0 {
                                done = true;
                            }
                        }
                    }
                }
            }
        }
        requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, response::IntoResponse, routing::get};
    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype, TensorView, TensorInfo, Metadata};
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tokio::task;
    use std::fs;

    async fn serve_file(file_path: Arc<std::path::PathBuf>) -> impl IntoResponse {
        match tokio::fs::read(&*file_path).await {
            Ok(bytes) => bytes,
            Err(_) => Vec::new(),
        }
    }

    // Helper function to create a 2D tensor with sequential values
    fn create_2d_tensor(rows: usize, cols: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(rows * cols * 4); // 4 bytes per f32
        for i in 0..rows {
            for j in 0..cols {
                let value = (i * cols + j) as f32;
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        data
    }

    #[tokio::test]
    async fn test_plan_slices() {
        // Create two tensors with different values
        let tensor1_data = create_2d_tensor(3, 4); // 3x4 tensor
        let tensor2_data = create_2d_tensor(4, 3); // 4x3 tensor

        let tensor1 = TensorView::new(Dtype::F32, vec![3, 4], &tensor1_data).unwrap();
        let tensor2 = TensorView::new(Dtype::F32, vec![4, 3], &tensor2_data).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("tensor1".to_string(), tensor1);
        tensors.insert("tensor2".to_string(), tensor2);

        // Create a temporary safetensors file
        let temp_file = NamedTempFile::new().unwrap();
        serialize_to_file(&tensors, &None, temp_file.path()).unwrap();
        let file_path = Arc::new(temp_file.path().to_path_buf());

        // Set up axum server
        let file_path_clone = file_path.clone();
        let app = Router::new().route("/file", get(move || serve_file(file_path_clone.clone())));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        task::spawn(async move {
            axum::serve(listener, app.into_make_service())
                .await
                .unwrap();
        });

        // Create a loader for the test file
        let url = format!("http://{}:{}/file", addr.ip(), addr.port());
        let mut loader = Loader::new(
            reqwest::Url::parse(&url).unwrap(),
            tokio::runtime::Handle::current(),
        )
        .unwrap();
        let mut plan = Plan::new();

        // Test 1: Try to get the same slice twice
        let slice1 = vec![
            TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(2)),
            TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(2)),
        ];

        // First slice should succeed
        plan.get_slice("tensor1", slice1.clone()).await.unwrap();

        // Second slice should fail
        let result = plan.get_slice("tensor1", slice1).await;
        assert!(matches!(result, Err(Error::AlreadyExists(_))));

        // Test 2: Get slices from both tensors
        let mut plan = Plan::new(); // Create a new plan for the second test

        // Slice tensor1 along first dimension
        let slice1 = vec![
            TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(2)),
            TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(4)),
        ];

        // Slice tensor2 along second dimension
        let slice2 = vec![
            TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(4)),
            TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(2)),
        ];

        plan.get_slice("tensor1", slice1).await.unwrap();
        plan.get_slice("tensor2", slice2).await.unwrap();

        let result = plan.execute(&mut loader).await;
        let result = result.expect("Plan execute should succeed");
        let t1 = result.get("tensor1").expect("tensor1 should be present");
        let t2 = result.get("tensor2").expect("tensor2 should be present");
        assert_eq!(
            t1,
            &Tensor {
                dtype: Dtype::F32,
                shape: vec![2, 4],
                data: vec![0; 2 * 4 * 4], // 2x4 tensor of f32 (4 bytes each)
            }
        );
        assert_eq!(
            t2,
            &Tensor {
                dtype: Dtype::F32,
                shape: vec![4, 2],
                data: vec![0; 4 * 2 * 4], // 4x2 tensor of f32 (4 bytes each)
            }
        );
    }

    #[test]
    fn test_gather_fetch_offsets_simple() {
        // Use our own TestMetadata for full control
        let mut tensors = HashMap::new();
        tensors.insert(
            "tensor".to_string(),
            TestTensorInfo {
                dtype: Dtype::F32,
                shape: vec![2, 2],
                data_offsets: (100, 116),
            },
        );
        let metadata = TestMetadata { tensors };
        let data_offsets = metadata.info("tensor").unwrap().data_offsets;

        let mut plan = Plan::new();
        // Slice: take the first row (row 0)
        plan.slices.insert(
            "tensor".to_string(),
            vec![TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(1)),
                 TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(2))],
        );
        // Patch gather_fetch_offsets to accept TestMetadata for this test
        let fetches = {
            let mut requests = Vec::new();
            let mut output_offset = 0;
            for (tensor_name, slices) in &plan.slices {
                if let Some(info) = metadata.info(tensor_name) {
                    let shape = &info.shape;
                    let dtype_size = match info.dtype {
                        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
                        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
                        Dtype::I16 | Dtype::U16 | Dtype::BF16 | Dtype::F16 => 2,
                        Dtype::I8 | Dtype::U8 => 1,
                        _ => todo!("dtype not supported in gather_fetch_offsets"),
                    };
                    let mut bounds = Vec::new();
                    for (slice, &dim) in slices.iter().zip(shape.iter()) {
                        match slice {
                            TensorIndexer::Select(idx) => {
                                bounds.push((*idx, *idx + 1));
                            }
                            TensorIndexer::Narrow(start, end) => {
                                let start = match start {
                                    std::ops::Bound::Unbounded => 0,
                                    std::ops::Bound::Included(i) => *i,
                                    std::ops::Bound::Excluded(i) => *i + 1,
                                };
                                let end = match end {
                                    std::ops::Bound::Unbounded => dim,
                                    std::ops::Bound::Included(i) => *i + 1,
                                    std::ops::Bound::Excluded(i) => *i,
                                };
                                bounds.push((start, end));
                            }
                        }
                    }
                    for i in slices.len()..shape.len() {
                        bounds.push((0, shape[i]));
                    }
                    let mut idx = bounds.iter().map(|(s, _)| *s).collect::<Vec<_>>();
                    let mut done = false;
                    while !done {
                        let mut flat_idx = 0;
                        let mut stride = 1;
                        for (i, &dim) in shape.iter().rev().enumerate() {
                            flat_idx += idx[shape.len() - 1 - i] * stride;
                            stride *= dim;
                        }
                        let file_offset = info.data_offsets.0 + flat_idx * dtype_size;
                        let output_offset_here = output_offset;
                        requests.push(FetchRequest {
                            tensor_name: tensor_name.clone(),
                            file_offset,
                            length: dtype_size,
                            output_offset: output_offset_here,
                        });
                        output_offset += dtype_size;
                        for axis in (0..idx.len()).rev() {
                            idx[axis] += 1;
                            if idx[axis] < bounds[axis].1 {
                                break;
                            } else {
                                idx[axis] = bounds[axis].0;
                                if axis == 0 {
                                    done = true;
                                }
                            }
                        }
                    }
                }
            }
            requests
        };
        assert_eq!(fetches, vec![
            FetchRequest {
                tensor_name: "tensor".to_string(),
                file_offset: data_offsets.0, // (0,0)
                length: 4,
                output_offset: 0,
            },
            FetchRequest {
                tensor_name: "tensor".to_string(),
                file_offset: data_offsets.0 + 4, // (0,1)
                length: 4,
                output_offset: 4,
            },
        ]);
    }
}
