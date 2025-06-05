use indicatif::{ProgressBar, ProgressStyle};
use safetensors::tensor::Metadata;
use safetensors::{Dtype, slice::TensorIndexer};

use std::collections::hash_map::Entry;
use std::ops::Bound;
use std::collections::HashMap;
use tokio::sync::mpsc;

use crate::loader::{Error, Loader};
use crate::tensor::TensorData;

pub struct Plan<'a> {
    slices: HashMap<String, Vec<TensorIndexer>>,
    loader: &'a mut Loader,
}

// Helper struct for fetch requests
#[derive(PartialEq, Debug, Clone)]
struct FetchRequest {
    tensor_name: String,
    file_offset: usize,
    length: usize,
    output_offset: usize,
}

// New struct to represent a downloaded chunk
#[derive(Debug)]
struct DownloadedChunk {
    tensor_name: String,
    data: Vec<u8>,
    output_offset: usize,
}

impl<'a> Plan<'a> {
    pub fn new(loader: &'a mut Loader) -> Self {
        Plan {
            slices: HashMap::new(),
            loader,
        }
    }

    pub fn get_slice(
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

    pub async fn execute(&mut self) -> Result<HashMap<String, TensorData>, Error> {
        let (metadata, fetch_offsets) = self.gather_fetch_offsets().await?;
        // TODO Clean this clone up.
        let metadata = metadata.clone();
        let mut result = HashMap::new();

        // Calculate total size to download
        let total_size: usize = fetch_offsets.iter().map(|req| req.length).sum();

        // Create progress bar
        let pb = ProgressBar::new(total_size as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner} [{elapsed}] [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})",
                )
                .unwrap(),
        );
        // .progress_chars("#>-"));

        // First create all the empty tensors
        for (tensor_name, slices) in &self.slices {
            // Get the tensor info from metadata
            let info = metadata
                .info(tensor_name)
                .ok_or_else(|| Error::MissingTensor(tensor_name.clone()))?;

            // If we have any Select indices, we need to adjust the shape
            let mut final_shape = Vec::new();
            for (i, &dim) in info.shape.iter().enumerate() {
                if let Some(slice) = slices.get(i) {
                    match slice {
                        TensorIndexer::Select(_) => {
                            // Skip this dimension as it's been selected
                        }
                        TensorIndexer::Narrow(start, end) => {
                            let start = match start {
                                Bound::Unbounded => 0,
                                Bound::Included(i) => *i,
                                Bound::Excluded(i) => *i + 1,
                            };
                            let end = match end {
                                Bound::Unbounded => dim,
                                Bound::Included(i) => *i + 1,
                                Bound::Excluded(i) => *i,
                            };
                            final_shape.push(end - start);
                        }
                    }
                } else {
                    final_shape.push(dim);
                }
            }

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
            let total_size = final_shape.iter().product::<usize>() * element_size;

            // Create empty tensor with correct shape and dtype
            result.insert(
                tensor_name.clone(),
                TensorData {
                    dtype: info.dtype,
                    shape: final_shape,
                    data: vec![0; total_size],
                },
            );
        }

        // Create a channel for receiving downloaded chunks
        let (tx, mut rx) = mpsc::channel(100);
        let client = self.loader.client().clone();
        let url = self.loader.url().clone();

        // Group fetch requests by tensor and sort by offset
        let mut tensor_requests: HashMap<String, Vec<FetchRequest>> = HashMap::new();
        for request in &fetch_offsets {
            tensor_requests
                .entry(request.tensor_name.clone())
                .or_default()
                .push(request.clone());
        }

        // For each tensor, spawn download tasks
        for (tensor_name, mut requests) in tensor_requests {
            // Sort requests by file_offset to ensure correct chunking
            requests.sort_by_key(|r| r.file_offset);
            let mut current_chunk: Vec<FetchRequest> = Vec::new();
            let mut current_size = 0;
            const CHUNK_SIZE: usize = 10 * 1024 * 1024; // 10MB

            let tx = tx.clone();
            let client = client.clone();
            let url = url.clone();
            let pb = pb.clone();

            let mut prev_end: Option<usize> = None;
            for request in requests {
                let is_contiguous = match prev_end {
                    Some(end) => end == request.file_offset,
                    None => true,
                };
                if (!is_contiguous || current_size + request.length > CHUNK_SIZE)
                    && !current_chunk.is_empty()
                {
                    // Spawn task for current chunk
                    let chunk_requests = current_chunk.clone();
                    let tx = tx.clone();
                    let client = client.clone();
                    let url = url.clone();
                    let tensor_name = tensor_name.clone();
                    let pb = pb.clone();

                    tokio::spawn(async move {
                        let start = chunk_requests.first().unwrap().file_offset;
                        let end = chunk_requests.last().unwrap().file_offset
                            + chunk_requests.last().unwrap().length;
                        let range = format!("bytes={}-{}", start, end - 1);

                        match client.get(url).header("Range", range).send().await {
                            Ok(response) => {
                                if let Ok(bytes) = response.bytes().await {
                                    let data = bytes.to_vec();
                                    pb.inc(data.len() as u64);
                                    let mut effective_start = start;
                                    if data.len() > end - start {
                                        effective_start = 0;
                                    }
                                    for request in chunk_requests {
                                        let offset = request.file_offset - effective_start;
                                        if offset + request.length > data.len() {
                                            continue;
                                        }
                                        let chunk = DownloadedChunk {
                                            tensor_name: tensor_name.clone(),
                                            data: data[offset..offset + request.length].to_vec(),
                                            output_offset: request.output_offset,
                                        };
                                        if tx.send(chunk).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                // Handle error appropriately
                            }
                        }
                    });

                    current_chunk.clear();
                    current_size = 0;
                    // Immediately start new chunk with current request
                    current_chunk.push(request.clone());
                    current_size += request.length;
                    prev_end = Some(request.file_offset + request.length);
                    continue;
                }

                current_chunk.push(request.clone());
                current_size += request.length;
                prev_end = Some(request.file_offset + request.length);
            }

            // Flush any remaining chunk
            if !current_chunk.is_empty() {
                let chunk_requests = current_chunk;
                let tx = tx.clone();
                let client = client.clone();
                let url = url.clone();
                let tensor_name = tensor_name.clone();
                let pb = pb.clone();

                tokio::spawn(async move {
                    let start = chunk_requests.first().unwrap().file_offset;
                    let end = chunk_requests.last().unwrap().file_offset
                        + chunk_requests.last().unwrap().length;
                    let range = format!("bytes={}-{}", start, end - 1);

                    match client.get(url).header("Range", range).send().await {
                        Ok(response) => {
                            if let Ok(bytes) = response.bytes().await {
                                let data = bytes.to_vec();
                                pb.inc(data.len() as u64);
                                let mut effective_start = start;
                                if data.len() > end - start {
                                    effective_start = 0;
                                }
                                for request in chunk_requests {
                                    let offset = request.file_offset - effective_start;
                                    if offset + request.length > data.len() {
                                        continue;
                                    }
                                    let chunk = DownloadedChunk {
                                        tensor_name: tensor_name.clone(),
                                        data: data[offset..offset + request.length].to_vec(),
                                        output_offset: request.output_offset,
                                    };
                                    if tx.send(chunk).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Handle error appropriately
                        }
                    }
                });
            }
        }

        // Drop the original sender so the channel closes when all tasks are done
        drop(tx);

        // Process received chunks
        while let Some(chunk) = rx.recv().await {
            if let Some(tensor) = result.get_mut(&chunk.tensor_name) {
                let end = chunk.output_offset + chunk.data.len();
                if end <= tensor.data.len() {
                    tensor.data[chunk.output_offset..end].copy_from_slice(&chunk.data);
                }
            }
        }

        // Finish the progress bar
        pb.finish_with_message("Download complete");

        Ok(result)
    }

    // Gather all byte offsets to fetch, the tensor name, and output offset
    async fn gather_fetch_offsets(&mut self) -> Result<(&Metadata, Vec<FetchRequest>), Error> {
        let (metadata, offset) = self.loader.metadata().await?;
        let mut requests = Vec::new();
        for (tensor_name, slices) in &self.slices {
            let info = metadata.info(tensor_name).expect("Tensor should exist");
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
                            Bound::Unbounded => 0,
                            Bound::Included(i) => *i,
                            Bound::Excluded(i) => *i + 1,
                        };
                        let end = match end {
                            Bound::Unbounded => dim,
                            Bound::Included(i) => *i + 1,
                            Bound::Excluded(i) => *i,
                        };
                        bounds.push((start, end));
                    }
                }
            }
            // For remaining axes, take the full range
            for dim in shape.iter().skip(slices.len()) {
                bounds.push((0, *dim));
            }
            // Now, enumerate all indices in the slice and compute file/output offsets
            // We'll use a multi-dimensional index
            let mut idx = bounds.iter().map(|(s, _)| *s).collect::<Vec<_>>();
            let mut done = false;
            let mut output_offset = 0;
            while !done {
                // Compute the flat index in the original tensor
                let mut flat_idx = 0;
                let mut stride = 1;
                for (i, &dim) in shape.iter().rev().enumerate() {
                    flat_idx += idx[shape.len() - 1 - i] * stride;
                    stride *= dim;
                }
                let file_offset = offset + info.data_offsets.0 + flat_idx * dtype_size;
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
        // Make the requests stable.
        requests.sort_by(|a, b| a.file_offset.cmp(&b.file_offset));
        Ok((metadata, requests))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router, http::HeaderMap, http::StatusCode, response::IntoResponse, response::Response,
        routing::get,
    };
    use reqwest::Url;
    use safetensors::serialize_to_file;
    use safetensors::tensor::Dtype;

    use tempfile::NamedTempFile;
    use tokio::task;

    async fn serve_file(file_path: std::path::PathBuf, headers: HeaderMap) -> Response {
        match tokio::fs::read(&*file_path).await {
            Ok(bytes) => {
                if let Some(range_header) = headers.get("range") {
                    if let Ok(range_str) = range_header.to_str() {
                        if let Some(range) = range_str.strip_prefix("bytes=") {
                            let mut parts = range.split('-');
                            if let (Some(start), Some(end)) = (parts.next(), parts.next()) {
                                if let (Ok(start), Ok(end)) =
                                    (start.parse::<usize>(), end.parse::<usize>())
                                {
                                    let end = end + 1; // end is inclusive in HTTP Range
                                    let end = end.min(bytes.len());
                                    let start = start.min(end);
                                    let response = bytes[start..end].to_vec();
                                    let resp = Response::builder()
                                        .status(StatusCode::PARTIAL_CONTENT)
                                        .header(
                                            "Content-Range",
                                            format!("bytes {}-{}/{}", start, end - 1, bytes.len()),
                                        )
                                        .header("Accept-Ranges", "bytes")
                                        .body(axum::body::Body::from(response))
                                        .unwrap();
                                    return resp;
                                }
                            }
                        }
                    }
                }
                (StatusCode::OK, bytes).into_response()
            }
            Err(_) => (StatusCode::NOT_FOUND, Vec::new()).into_response(),
        }
    }

    // Helper function to create a 2D tensor with sequential values
    fn create_2d_tensor(rows: usize, cols: usize) -> TensorData {
        let mut data = Vec::with_capacity(rows * cols * 4); // 4 bytes per f32
        for i in 0..rows {
            for j in 0..cols {
                let value = (i * cols + j) as f32;
                data.extend_from_slice(&value.to_le_bytes());
            }
        }

        TensorData {
            data,
            shape: vec![rows, cols],
            dtype: Dtype::F32,
        }
    }

    fn save_tensors(tensors: HashMap<&str, TensorData>) -> NamedTempFile {
        // Create a temporary safetensors file
        let temp_file = NamedTempFile::new().unwrap();
        serialize_to_file(tensors, &None, temp_file.path()).unwrap();
        temp_file
    }

    async fn create_test_file(tensors: HashMap<&str, TensorData>) -> (NamedTempFile, Url) {
        let named = save_tensors(tensors);
        let file_path = named.path();

        // Set up axum server
        let file_path_clone = file_path.to_path_buf();
        let app = Router::new().route(
            "/file",
            get(move |headers: HeaderMap| serve_file(file_path_clone, headers)),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        task::spawn(async move {
            axum::serve(listener, app.into_make_service())
                .await
                .unwrap();
        });

        // Create a loader for the test file
        let url = format!("http://{}:{}/file", addr.ip(), addr.port());

        (named, Url::parse(&url).unwrap())
    }

    #[tokio::test]
    async fn test_plan_already_exists() {
        let tensors = HashMap::from([
            ("tensor1", create_2d_tensor(3, 4)),
            ("tensor2", create_2d_tensor(4, 3)),
        ]);
        let (_temp_handle, url) = create_test_file(tensors).await;
        let mut loader = Loader::new(url).unwrap();
        let mut plan = loader.plan();

        // Test 1: Try to get the same slice twice
        let slice1 = vec![TensorIndexer::Narrow(
            Bound::Included(0),
            Bound::Excluded(2),
        )];

        // First slice should succeed
        plan.get_slice("tensor1", slice1.clone()).unwrap();

        // Second slice should fail
        let result = plan.get_slice("tensor1", slice1);
        assert!(matches!(result, Err(Error::AlreadyExists(_))));
    }

    #[tokio::test]
    async fn test_plan_slices() {
        let tensors = HashMap::from([
            ("tensor1", create_2d_tensor(3, 4)),
            ("tensor2", create_2d_tensor(4, 3)),
        ]);
        let (_temp_handle, url) = create_test_file(tensors).await;
        let mut loader = Loader::new(url).unwrap();
        let mut plan = loader.plan();

        // Slice tensor1 along first dimension
        let slice1 = vec![
            TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(2)),
            TensorIndexer::Narrow(Bound::Included(2), Bound::Excluded(4)),
        ];

        // Slice tensor2 along second dimension
        let slice2 = vec![
            TensorIndexer::Narrow(Bound::Included(2), Bound::Excluded(3)),
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3)),
        ];

        plan.get_slice("tensor1", slice1).unwrap();
        plan.get_slice("tensor2", slice2).unwrap();

        let result = plan.execute().await;
        let result = result.expect("Plan execute should succeed");
        assert_eq!(
            result,
            HashMap::from([
                (
                    "tensor1".to_string(),
                    TensorData {
                        dtype: Dtype::F32,
                        shape: vec![2, 2],
                        data: vec![2.0f32, 3.0, 6.0, 7.0]
                            .into_iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                ),
                (
                    "tensor2".to_string(),
                    TensorData {
                        dtype: Dtype::F32,
                        shape: vec![1, 2],
                        data: vec![7.0f32, 8.0]
                            .into_iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                )
            ])
        );
    }

    #[tokio::test]
    async fn test_plan_partial() {
        let tensors = HashMap::from([
            ("tensor1", create_2d_tensor(3, 4)),
            ("tensor2", create_2d_tensor(4, 3)),
        ]);
        let (_temp_handle, url) = create_test_file(tensors).await;
        let mut loader = Loader::new(url).unwrap();
        let mut plan = loader.plan(); // Create a new plan for the second test

        // Slice tensor1 along first dimension
        let slice1 = vec![TensorIndexer::Narrow(
            Bound::Included(0),
            Bound::Excluded(2),
        )];

        // Slice tensor2 along second dimension
        let slice2 = vec![
            TensorIndexer::Select(1),
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3)),
        ];

        plan.get_slice("tensor1", slice1).unwrap();
        plan.get_slice("tensor2", slice2).unwrap();

        let result = plan.execute().await;
        let result = result.expect("Plan execute should succeed");
        assert_eq!(
            result,
            HashMap::from([
                (
                    "tensor1".to_string(),
                    TensorData {
                        dtype: Dtype::F32,
                        shape: vec![2, 4],
                        data: vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
                            .into_iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                ),
                (
                    "tensor2".to_string(),
                    TensorData {
                        dtype: Dtype::F32,
                        shape: vec![2],
                        data: vec![4.0f32, 5.0]
                            .into_iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    },
                )
            ]),
            "{result:#?}"
        );
    }

    #[tokio::test]
    async fn test_gather_fetch_offsets_simple() {
        // Create two tensors with different values
        let tensors = HashMap::from([
            ("tensor1", create_2d_tensor(3, 4)),
            ("tensor2", create_2d_tensor(4, 3)),
        ]);

        let (_temp_handle, url) = create_test_file(tensors).await;
        let mut loader = Loader::new(url).unwrap();
        let mut plan = loader.plan();
        // Slice tensor1 along first dimension
        let slice1 = vec![
            TensorIndexer::Narrow(Bound::Included(0), Bound::Excluded(2)),
            TensorIndexer::Narrow(Bound::Included(2), Bound::Excluded(4)),
        ];

        // Slice tensor2 along second dimension
        let slice2 = vec![
            TensorIndexer::Narrow(Bound::Included(2), Bound::Excluded(3)),
            TensorIndexer::Narrow(Bound::Included(1), Bound::Excluded(3)),
        ];
        plan.get_slice("tensor1", slice1).unwrap();
        plan.get_slice("tensor2", slice2).unwrap();
        let (_, fetches) = plan.gather_fetch_offsets().await.unwrap();
        assert_eq!(
            fetches,
            vec![
                FetchRequest {
                    tensor_name: "tensor1".to_string(),
                    file_offset: 144,
                    length: 4,
                    output_offset: 0,
                },
                FetchRequest {
                    tensor_name: "tensor1".to_string(),
                    file_offset: 148,
                    length: 4,
                    output_offset: 4,
                },
                FetchRequest {
                    tensor_name: "tensor1".to_string(),
                    file_offset: 160,
                    length: 4,
                    output_offset: 8,
                },
                FetchRequest {
                    tensor_name: "tensor1".to_string(),
                    file_offset: 164,
                    length: 4,
                    output_offset: 12,
                },
                FetchRequest {
                    tensor_name: "tensor2".to_string(),
                    file_offset: 212,
                    length: 4,
                    output_offset: 0,
                },
                FetchRequest {
                    tensor_name: "tensor2".to_string(),
                    file_offset: 216,
                    length: 4,
                    output_offset: 4,
                },
            ]
        );
    }
}
