use safetensors::{Dtype, slice::TensorIndexer};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::loader::{Error, Loader};

pub struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
pub struct Plan {
    slices: HashMap<String, Vec<TensorIndexer>>,
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
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, response::IntoResponse, routing::get};
    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype, TensorView};
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tokio::task;

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
        let mut loader = Loader::new(reqwest::Url::parse(&url).unwrap(), tokio::runtime::Handle::current()).unwrap();
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

        // This will fail until execute is implemented
        let result = plan.execute(&mut loader).await;
        assert!(matches!(result, Err(_)));
    }
}
