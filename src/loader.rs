use futures_util::StreamExt;
use reqwest::Client;
use reqwest::Url;
use safetensors::SafeTensorError;
use safetensors::tensor::Metadata;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinError;

use crate::Plan;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("The future didn't exist")]
    NoFuture,
    #[error(transparent)]
    JoinError(#[from] JoinError),
    #[error(transparent)]
    Safetensor(#[from] SafeTensorError),
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
}

const MAX_HEADER_SIZE: usize = 100_000_000;

pub struct Loader {
    url: Url,
    client: Arc<Client>,
    metadata: Option<Metadata>,
}

impl Loader {
    pub fn new(url: Url) -> Result<Self, Error> {
        let client = Arc::new(Client::new());
        Ok(Self {
            url,
            client,
            metadata: None,
        })
    }

    pub async fn metadata(&mut self) -> Result<&Metadata, Error> {
        if self.metadata.is_none() {
            let metadata = Self::fetch_metadata(self.client.clone(), self.url.clone()).await?;
            self.metadata = Some(metadata);
        }
        Ok(self.metadata.as_ref().unwrap())
    }

    async fn fetch_metadata(client: Arc<Client>, url: Url) -> Result<Metadata, Error> {
        let response = client.get(url).send().await?;
        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();

        // Read initial chunk
        if let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if chunk.len() < 8 {
                return Err(Error::Safetensor(SafeTensorError::HeaderTooSmall));
            }

            // Get metadata length from first 8 bytes
            let arr: [u8; 8] = [
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ];
            let metadata_len: usize = u64::from_le_bytes(arr)
                .try_into()
                .map_err(|_| SafeTensorError::HeaderTooLarge)?;

            if metadata_len > MAX_HEADER_SIZE {
                return Err(Error::Safetensor(SafeTensorError::HeaderTooLarge));
            }

            // Calculate total bytes needed (metadata length + 8 bytes for the length)
            let total_bytes_needed = metadata_len + 8;

            // Add initial chunk to buffer
            buffer.extend_from_slice(&chunk);

            // If we need more data, read the remaining chunks
            while buffer.len() < total_bytes_needed {
                if let Some(chunk) = stream.next().await {
                    let chunk = chunk?;
                    buffer.extend_from_slice(&chunk);
                } else {
                    break;
                }
            }
        } else {
            return Err(Error::Safetensor(SafeTensorError::HeaderTooSmall));
        }

        Ok(get_metadata(buffer)?)
    }

    pub async fn execute(&self, plan: &Plan) -> Result<HashMap<String, Vec<u8>>, Error> {
        let client = Arc::clone(&self.client);
        let url = self.url.clone();
        let slices = plan.slices.clone();

        let response = client.get(url).send().await?;
        let bytes = response.bytes().await?;
        let mut result = HashMap::new();
        for (tensor_name, _) in slices {
            result.insert(tensor_name, bytes.to_vec());
        }
        Ok(result)
    }
}

fn get_metadata(buffer: Vec<u8>) -> Result<Metadata, SafeTensorError> {
    let buffer_len = buffer.len();
    if buffer_len < 8 {
        return Err(SafeTensorError::HeaderTooSmall);
    }
    let arr: [u8; 8] = [
        buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
    ];
    let n: usize = u64::from_le_bytes(arr)
        .try_into()
        .map_err(|_| SafeTensorError::HeaderTooLarge)?;
    if n > MAX_HEADER_SIZE {
        return Err(SafeTensorError::HeaderTooLarge);
    }

    let stop = n
        .checked_add(8)
        .ok_or(SafeTensorError::InvalidHeaderLength)?;
    if stop > buffer_len {
        return Err(SafeTensorError::InvalidHeaderLength);
    }
    let string =
        core::str::from_utf8(&buffer[8..stop]).map_err(|_| SafeTensorError::InvalidHeader)?;
    let metadata: Metadata =
        serde_json::from_str(string).map_err(|_| SafeTensorError::InvalidHeaderDeserialization)?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, response::IntoResponse, routing::get};
    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype, TensorView};
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tokio::task;

    async fn serve_file(file_path: Arc<std::path::PathBuf>) -> impl IntoResponse {
        match tokio::fs::read(&*file_path).await {
            Ok(bytes) => bytes,
            Err(_) => Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_loader_metadata() {
        // Create a temporary safetensors file using the library
        let temp_file = NamedTempFile::new().unwrap();
        // Create a dummy tensor
        let data: Vec<u8> = 0f32.to_le_bytes().into_iter().collect();

        let shape = vec![1];
        let tensor = TensorView::new(Dtype::F32, shape, &data).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("dummy".to_string(), tensor);

        // User metadata
        let mut user_metadata = HashMap::new();
        user_metadata.insert("test_key".to_string(), "test_value".to_string());
        let user_metadata = Some(user_metadata);

        serialize_to_file(&tensors, &user_metadata, temp_file.path()).unwrap();
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
        let mut loader = Loader::new(Url::parse(&url).unwrap()).unwrap();

        // Get the metadata
        let metadata = loader.metadata().await.unwrap();

        // Verify the metadata matches
        let user_metadata = metadata.metadata().as_ref().unwrap();
        assert_eq!(user_metadata.get("test_key").unwrap(), "test_value");
    }
}

