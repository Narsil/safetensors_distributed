use safetensors::{Dtype, slice::TensorIndexer};
use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;
use tokio::sync::mpsc;

mod loader;
mod plan;

pub use loader::{Error, Loader};
pub use plan::{Plan, Tensor};

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

#[derive(PartialEq, Debug)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
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

    pub async fn execute(&self, loader: &mut Loader) -> Result<HashMap<String, Tensor>, Error> {
        let metadata = loader.metadata().await?;
        let fetch_offsets = self.gather_fetch_offsets(metadata);
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
        let url = loader.url().clone();

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

                tokio::spawn(async move {
                    let start = chunk_requests.first().unwrap().file_offset;
                    let end = chunk_requests.last().unwrap().file_offset
                        + chunk_requests.last().unwrap().length;
                    let range = format!("bytes={}-{}", start, end - 1);

                    match client.get(url).header("Range", range).send().await {
                        Ok(response) => {
                            if let Ok(bytes) = response.bytes().await {
                                let data = bytes.to_vec();
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
