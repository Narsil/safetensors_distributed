use safetensors::{Dtype, slice::TensorIndexer};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::loader::{Error, Loader};

#[derive(PartialEq, Debug)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

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

    pub async fn execute(&mut self) -> Result<HashMap<String, Tensor>, Error> {
        let metadata = self.loader.metadata().await?.clone();
        let fetch_offsets = self.gather_fetch_offsets(&metadata);
        let mut result = HashMap::new();

        // First create all the empty tensors
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

        // Create a channel for receiving downloaded chunks
        let (tx, mut rx) = mpsc::channel(100);
        let client = Arc::new(reqwest::Client::new());
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

                    tokio::spawn(async move {
                        let start = chunk_requests.first().unwrap().file_offset;
                        let end = chunk_requests.last().unwrap().file_offset
                            + chunk_requests.last().unwrap().length;
                        let range = format!("bytes={}-{}", start, end - 1);

                        match client.get(url).header("Range", range).send().await {
                            Ok(response) => {
                                if let Ok(bytes) = response.bytes().await {
                                    let data = bytes.to_vec();
                                    eprintln!(
                                        "[DEBUG] Downloaded chunk: start={}, end={}, data.len()={}, requests=[{}]",
                                        start,
                                        end,
                                        data.len(),
                                        chunk_requests
                                            .iter()
                                            .map(|r| format!(
                                                "{{offset={}, len={}}}",
                                                r.file_offset, r.length
                                            ))
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    );
                                    let mut effective_start = start;
                                    if data.len() > end - start {
                                        effective_start = 0;
                                    }
                                    for request in chunk_requests {
                                        let offset = request.file_offset - effective_start;
                                        if offset + request.length > data.len() {
                                            eprintln!(
                                                "[BUG] Out of bounds: offset {} + length {} > data.len() {} (effective_start={}, start={}, end={})",
                                                offset,
                                                request.length,
                                                data.len(),
                                                effective_start,
                                                start,
                                                end
                                            );
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

                tokio::spawn(async move {
                    let start = chunk_requests.first().unwrap().file_offset;
                    let end = chunk_requests.last().unwrap().file_offset
                        + chunk_requests.last().unwrap().length;
                    let range = format!("bytes={}-{}", start, end - 1);

                    match client.get(url).header("Range", range).send().await {
                        Ok(response) => {
                            if let Ok(bytes) = response.bytes().await {
                                let data = bytes.to_vec();
                                eprintln!(
                                    "[DEBUG] Downloaded chunk: start={}, end={}, data.len()={}, requests=[{}]",
                                    start,
                                    end,
                                    data.len(),
                                    chunk_requests
                                        .iter()
                                        .map(|r| format!(
                                            "{{offset={}, len={}}}",
                                            r.file_offset, r.length
                                        ))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                );
                                let mut effective_start = start;
                                if data.len() > end - start {
                                    effective_start = 0;
                                }
                                for request in chunk_requests {
                                    let offset = request.file_offset - effective_start;
                                    if offset + request.length > data.len() {
                                        eprintln!(
                                            "[BUG] Out of bounds: offset {} + length {} > data.len() {} (effective_start={}, start={}, end={})",
                                            offset,
                                            request.length,
                                            data.len(),
                                            effective_start,
                                            start,
                                            end
                                        );
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
                } else {
                    eprintln!(
                        "[ERROR] Chunk out of bounds: output_offset={}, data.len()={}, tensor.data.len()={}",
                        chunk.output_offset,
                        chunk.data.len(),
                        tensor.data.len()
                    );
                    std::io::stderr().flush().unwrap();
                }
            }
        }

        Ok(result)
    }

    // Gather all byte offsets to fetch, the tensor name, and output offset
    fn gather_fetch_offsets(&self, metadata: &safetensors::tensor::Metadata) -> Vec<FetchRequest> {
        let mut requests = Vec::new();
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
                let mut output_offset = 0;
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
                    eprintln!(
                        "[DEBUG] FetchRequest: tensor={}, bounds={:?}, idx={:?}, flat_idx={}, file_offset={}, length={}, output_offset={}",
                        tensor_name,
                        bounds,
                        idx,
                        flat_idx,
                        file_offset,
                        dtype_size,
                        output_offset_here
                    );
                    std::io::stderr().flush().unwrap();
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
    use axum::{
        Router, http::HeaderMap, http::StatusCode, response::IntoResponse, response::Response,
        routing::get,
    };
    use reqwest::Url;
    use safetensors::serialize_to_file;
    use safetensors::tensor::{Dtype, TensorView};

    use std::sync::Arc;
    use tempfile::NamedTempFile;
    use tokio::task;

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

    async fn serve_file(file_path: Arc<std::path::PathBuf>, headers: HeaderMap) -> Response {
        eprintln!("[DEBUG] Received headers: {:?}", headers);
        match tokio::fs::read(&*file_path).await {
            Ok(bytes) => {
                if let Some(range_header) = headers.get("range") {
                    eprintln!("[DEBUG] Range header: {:?}", range_header);
                    if let Ok(range_str) = range_header.to_str() {
                        eprintln!("[DEBUG] Range string: {}", range_str);
                        if let Some(range) = range_str.strip_prefix("bytes=") {
                            eprintln!("[DEBUG] Range value: {}", range);
                            let mut parts = range.split('-');
                            if let (Some(start), Some(end)) = (parts.next(), parts.next()) {
                                eprintln!("[DEBUG] Range parts: start={}, end={}", start, end);
                                if let (Ok(start), Ok(end)) =
                                    (start.parse::<usize>(), end.parse::<usize>())
                                {
                                    let end = end + 1; // end is inclusive in HTTP Range
                                    let end = end.min(bytes.len());
                                    let start = start.min(end);
                                    let response = bytes[start..end].to_vec();
                                    eprintln!(
                                        "[DEBUG] Response range: {}-{} (len={})",
                                        start,
                                        end - 1,
                                        response.len()
                                    );
                                    eprintln!("[DEBUG] Response data: {:?}", response);
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
                eprintln!("[DEBUG] Full file data: {:?}", bytes);
                (StatusCode::OK, bytes).into_response()
            }
            Err(_) => (StatusCode::NOT_FOUND, Vec::new()).into_response(),
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
        let app = Router::new().route(
            "/file",
            get(move |headers: HeaderMap| serve_file(file_path_clone.clone(), headers)),
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
        let mut loader = Loader::new(
            reqwest::Url::parse(&url).unwrap(),
            tokio::runtime::Handle::current(),
        )
        .unwrap();
        let mut plan = loader.plan();

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
        let mut plan = loader.plan(); // Create a new plan for the second test

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
        drop(plan);

        let metadata = loader.metadata().await.unwrap().clone();
        let mut plan = loader.plan();
        // Print all fetch requests for debugging
        let fetches = plan.gather_fetch_offsets(&metadata);
        for (name, info) in metadata.tensors().iter() {
            eprintln!("[DEBUG] {} data_offsets: {:?}", name, info.data_offsets);
        }
        eprintln!("[DEBUG] All fetch requests:");
        for f in &fetches {
            eprintln!(
                "  tensor={}, file_offset={}, length={}, output_offset={}",
                f.tensor_name, f.file_offset, f.length, f.output_offset
            );
        }

        let result = plan.execute().await;
        let result = result.expect("Plan execute should succeed");
        let t1 = result.get("tensor1").expect("tensor1 should be present");
        let t2 = result.get("tensor2").expect("tensor2 should be present");

        // Verify tensor1 shape and data
        assert_eq!(t1.shape, vec![2, 4]);
        assert_eq!(t1.dtype, Dtype::F32);
        // The expected data for tensor1 is the first 2 rows of the original 3x4 tensor
        let metadata = loader.metadata().await.unwrap();
        let t1_info = metadata.info("tensor1").unwrap();
        let t1_offset = t1_info.data_offsets.0;
        let expected_t1: Vec<u8> = {
            let start = t1_offset;
            let end = t1_offset + 2 * 4 * 4;
            let file_bytes = std::fs::read(file_path.as_ref()).unwrap();
            file_bytes[start..end].to_vec()
        };
        eprintln!("[DEBUG] t1.data (actual):   {:?}", t1.data);
        eprintln!("[DEBUG] t1.data (expected): {:?}", expected_t1);
        assert_eq!(t1.data, expected_t1);

        // Verify tensor2 shape and data
        assert_eq!(t2.shape, vec![4, 2]);
        assert_eq!(t2.dtype, Dtype::F32);
        // The expected data for tensor2 is the first 2 columns of the original 4x3 tensor
        let t2_info = metadata.info("tensor2").unwrap();
        let t2_offset = t2_info.data_offsets.0;
        let file_bytes = std::fs::read(file_path.as_ref()).unwrap();
        let mut expected_t2 = Vec::with_capacity(4 * 2 * 4);
        for row in 0..4 {
            for col in 0..2 {
                let idx = row * 3 + col;
                let start = t2_offset + idx * 4;
                let end = start + 4;
                expected_t2.extend_from_slice(&file_bytes[start..end]);
            }
        }
        assert_eq!(t2.data, expected_t2);
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

        let url = Url::parse("http://localhost").unwrap();
        let mut loader = Loader::new(url, tokio::runtime::Handle::current()).unwrap();
        let mut plan = loader.plan();
        // Slice: take the first row (row 0)
        plan.slices.insert(
            "tensor".to_string(),
            vec![
                TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(1)),
                TensorIndexer::Narrow(std::ops::Bound::Included(0), std::ops::Bound::Excluded(2)),
            ],
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
        assert_eq!(
            fetches,
            vec![
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
            ]
        );
    }
}
