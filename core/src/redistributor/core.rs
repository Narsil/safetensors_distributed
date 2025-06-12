use super::location::{Source, SourceLocation, Target, WriteLocation};
use super::task::Task;
use super::{
    Layout, RedistributorError, Result, compute_distributed_to_shared_ranges,
    compute_shared_to_distributed_ranges,
};
use crate::topology::{Tensor, Topology};
use indicatif::{ProgressBar, ProgressStyle};
use log::error;
use memmap2::Mmap;
use safetensors::tensor::{Metadata, TensorInfo};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{Sender, channel};

pub struct Redistributor {
    source: Source,
    target: Target,
}

impl Redistributor {
    /// Create a new redistributor for reconstruction from local distributed files
    pub fn from_local<P: AsRef<Path>>(
        source_dir: P,
        target_dir: P,
        target_topology: Topology,
    ) -> Result<Self> {
        // Load the existing topology (or create from model.safetensors)
        let source_dir = source_dir.as_ref().to_path_buf();
        let target_dir = target_dir.as_ref().to_path_buf();

        let source_topology = super::load_or_create_topology(&source_dir)?;
        let source_metadatas: Result<Vec<(usize, Metadata)>> = source_topology
            .filenames()
            .iter()
            .map(|f| Ok(super::safetensors_metadata(source_dir.join(f))?))
            .collect();
        let source_metadatas = source_metadatas?;
        let target_metadatas = Self::pre_calculate_metadatas(&target_topology)?;

        let source_layout = Layout {
            topology: source_topology,
            metadatas: source_metadatas,
        };
        let target_layout = Layout {
            topology: target_topology,
            metadatas: target_metadatas,
        };
        let source_mmaps: Result<Vec<_>> = source_layout
            .topology
            .filenames()
            .iter()
            .map(|filename| {
                let filepath = source_dir.join(filename);
                let file = std::fs::File::open(&filepath)?;
                unsafe { Ok(Arc::new(Mmap::map(&file)?)) }
            })
            .collect();
        let source_mmaps = source_mmaps?;

        // Don't open target files yet - they might not exist
        Ok(Self {
            source: Source {
                layout: source_layout,
                location: SourceLocation::Local {
                    mmaps: source_mmaps,
                },
            },
            target: Target {
                layout: target_layout,
                location: WriteLocation {
                    dir: target_dir,
                    mmaps: None,
                },
            },
        })
    }

    pub async fn redistribute(&mut self) -> Result<Vec<String>> {
        let start = Instant::now();

        let init_start = Instant::now();
        self.target.location.init(&self.target.layout).await?;

        let (tx, mut rx) = channel::<Task>(10_000);

        let calc_start = Instant::now();
        // Estimate of the data needed to be copied.
        let mut total = 0u64;
        for (header_size, metadata) in &self.source.layout.metadatas {
            // Find the maximum data end offset in this file
            let max_data_end = metadata
                .tensors()
                .values()
                .map(|tensor_info| tensor_info.data_offsets.1)
                .max()
                .unwrap_or(0);

            // Total file size = header + data section
            total += (header_size + max_data_end) as u64;
        }

        let progress = ProgressBar::new(total);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{eta_precise}] [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})")
                .unwrap(),
        );

        let p = progress.clone();
        // let strategy = self.strategy;
        let handle = tokio::spawn(async move {
            // match strategy {
            //     RedistributionStrategy::ReadUnorderedWriteSerial => {
            // Execute write tasks as they come in
            while let Some(task) = rx.recv().await {
                // if let Task::Write(write_task) = task {
                let length = (task.target_end - task.target_start) as usize;
                if let Err(e) = task.run().await {
                    error!("Write task failed: {}", e);
                }
                p.inc(length as u64);
                // } else {
                //     error!("ReadUnorderedWriteSerial strategy received unexpected ReadSerial task");
                // }
            }
            // }

            // RedistributionStrategy::ReadSerialWriteUnordered => {
            //     // Execute read tasks as they come in
            //     while let Some(task) = rx.recv().await {
            //         if let Task::ReadSerial(read_task) = task {
            //             let total_writes_length: u64 = read_task
            //                 .writes
            //                 .iter()
            //                 .map(|w| w.target_end - w.target_start)
            //                 .sum();

            //             if let Err(e) = read_task.run().await {
            //                 error!("Read task failed: {}", e);
            //             }
            //             p.inc(total_writes_length);
            //         } else {
            //             error!(
            //                 "ReadSerialWriteUnordered strategy received unexpected Write task"
            //             );
            //         }
            //     }
            // }
            // }
        });

        let tasks_start = Instant::now();
        self.create_tasks(&tx).await?;

        drop(tx);
        handle.await?;

        progress.finish();
        println!("Done write, waiting for the kernel to sync on device...");
        // Force flush all memory-mapped target files and write topology
        self.target
            .location
            .save(&self.target.layout.topology)
            .await?;

        // Collect created safetensors files
        let mut created_files = Vec::new();
        for filename in self.target.layout.topology.filenames() {
            created_files.push(filename.clone());
        }

        // Add topology.json if it was written (for multi-rank outputs)
        if self.target.layout.topology.world_size() > 1 {
            created_files.push("topology.json".to_string());
        }

        Ok(created_files)
    }

    /// Pre-calculate all headers, offsets, and file structures based on target topology
    fn pre_calculate_metadatas(topology: &Topology) -> Result<Vec<(usize, Metadata)>> {
        let mut rank_tensor_info: Vec<Vec<(String, TensorInfo)>> =
            (0..topology.world_size()).map(|_| Vec::new()).collect();
        let mut rank_offsets = vec![0usize; topology.world_size()];
        // Process each tensor according to the target topology
        for (tensor_name, tensor) in topology.tensors() {
            match tensor {
                Tensor::Distributed(dist_info) => {
                    // Process distributed tensor - each rank gets a chunk
                    for (rank, chunk) in dist_info.chunks().iter().enumerate() {
                        let chunk_shape = chunk.shape().to_vec();
                        let chunk_size =
                            chunk_shape.iter().product::<usize>() * dist_info.dtype().size();

                        let data_offsets = (rank_offsets[rank], rank_offsets[rank] + chunk_size);
                        let tensor_info = TensorInfo {
                            dtype: dist_info.dtype(),
                            shape: chunk_shape,
                            data_offsets,
                        };

                        rank_tensor_info[rank].push((tensor_name.clone(), tensor_info));

                        rank_offsets[rank] += chunk_size;
                    }
                }
                Tensor::Shared(shared_info) => {
                    // Process shared tensor - all ranks get the full tensor
                    let chunk_size =
                        shared_info.shape().iter().product::<usize>() * shared_info.dtype().size();

                    for rank in 0..topology.world_size() {
                        let tensor_info = TensorInfo {
                            dtype: shared_info.dtype(),
                            shape: shared_info.shape().to_vec(),
                            data_offsets: (rank_offsets[rank], rank_offsets[rank] + chunk_size),
                        };

                        rank_tensor_info[rank].push((tensor_name.clone(), tensor_info));

                        rank_offsets[rank] += chunk_size;
                    }
                }
            }
        }
        // Create headers and final file info
        let metadatas: Result<Vec<_>> = (0..topology.world_size())
            .map(|rank| {
                let metadata = Metadata::new(
                    None, // metadata_header
                    rank_tensor_info[rank].iter().cloned().collect(),
                )?;
                let metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
                // Force alignment to 8 bytes.
                let extra = (8 - metadata_buf.len() % 8) % 8;
                Ok((metadata_buf.len() + 8 + extra, metadata))
            })
            .collect();
        let metadatas = metadatas?;

        Ok(metadatas)
    }

    /// Create write-focused tasks for ReadUnorderedWriteSerial strategy
    /// Iterates target files → target tensors by data_offset (current logic)
    async fn create_tasks(&self, tx: &Sender<Task>) -> Result<()> {
        // Process each target file in order
        for (target_file_index, filename) in
            self.target.layout.topology.filenames().iter().enumerate()
        {
            // Get the metadata for this target file
            let (header_size, metadata) = &self.target.layout.metadatas[target_file_index];

            // Collect tensors and sort by data offset for sequential writes
            let tensors = metadata.tensors();
            let mut tensor_entries: Vec<(&String, &TensorInfo)> =
                tensors.iter().map(|(k, v)| (k, *v)).collect();
            tensor_entries.sort_by_key(|(_, tensor_info)| tensor_info.data_offsets.0);

            let tensor_count = tensor_entries.len();

            // Process each tensor in write order
            for (i, (tensor_name, tensor_info)) in tensor_entries.iter().enumerate() {
                let tensor_start = Instant::now();
                if i < 5 || i % 20 == 0 || i == tensor_count - 1 {}

                self.create_write_task_for_tensor(
                    tx,
                    target_file_index,
                    *header_size,
                    tensor_name,
                    tensor_info,
                )
                .await?;

                if i < 5 || i % 20 == 0 || i == tensor_count - 1 {}
            }
        }

        Ok(())
    }

    async fn create_write_task_for_tensor(
        &self,
        tx: &Sender<Task>,
        target_file_index: usize,
        target_header_size: usize,
        tensor_name: &str,
        target_tensor_info: &TensorInfo,
    ) -> Result<()> {
        // Look up the tensor in source topology
        let source_tensor = self
            .source
            .layout
            .topology
            .tensors()
            .get(tensor_name)
            .ok_or_else(|| RedistributorError::TensorNotFound {
                name: tensor_name.to_string(),
            })?;

        let target_tensor = self
            .target
            .layout
            .topology
            .tensors()
            .get(tensor_name)
            .ok_or_else(|| RedistributorError::TensorNotFound {
                name: tensor_name.to_string(),
            })?;

        // Calculate target write parameters
        let target_start = (target_header_size + target_tensor_info.data_offsets.0) as u64;
        let target_end = (target_header_size + target_tensor_info.data_offsets.1) as u64;

        // Collect source reads needed to fulfill this target write
        // let mut task_source = match &self.source.location {
        //     SourceLocation::Local { .. } => TaskSource::new_local(),
        //     SourceLocation::Remote {
        //         client,
        //         base_url,
        //         auth_headers,
        //         http_semaphore,
        //         ..
        //     } => TaskSource::new_remote(
        //         client.clone(),
        //         base_url.clone(),
        //         auth_headers.clone(),
        //         http_semaphore.clone(),
        //     ),
        // };
        let mut task_source = Vec::new();
        let mut source_ranges = Vec::new();
        let mut ranges_per_file = Vec::new();

        match (source_tensor, target_tensor) {
            (Tensor::Distributed(source_info), Tensor::Distributed(target_info)) => {
                self.collect_reads_distributed_to_distributed(
                    tensor_name,
                    source_info,
                    target_info,
                    target_file_index,
                    &mut task_source,
                    &mut source_ranges,
                    &mut ranges_per_file,
                )?;
            }
            (Tensor::Shared(source_info), Tensor::Distributed(target_info)) => {
                self.collect_reads_shared_to_distributed(
                    tensor_name,
                    source_info,
                    target_info,
                    target_file_index,
                    &mut task_source,
                    &mut source_ranges,
                    &mut ranges_per_file,
                )?;
            }
            (Tensor::Shared(source_info), Tensor::Shared(target_info)) => {
                self.collect_reads_shared_to_shared(
                    tensor_name,
                    source_info,
                    target_info,
                    target_file_index,
                    &mut task_source,
                    &mut source_ranges,
                    &mut ranges_per_file,
                )?;
            }
            (Tensor::Distributed(source_info), Tensor::Shared(target_info)) => {
                self.collect_reads_distributed_to_shared(
                    tensor_name,
                    source_info,
                    target_info,
                    target_file_index,
                    &mut task_source,
                    &mut source_ranges,
                    &mut ranges_per_file,
                )?;
            }
        }

        // Create and send the task if we have any reads to do
        if !source_ranges.is_empty() {
            let source_count = task_source.len();

            if let Some(write_task) = self.target.location.create_write_task(
                target_file_index,
                target_start,
                target_end,
                // task_source.into_task_sources(),
                task_source,
                source_ranges,
                ranges_per_file,
            ) {
                // tx.send(Task::Write(write_task)).await.unwrap();
                tx.send(write_task).await.unwrap();
            }
        } else {
        }

        Ok(())
    }

    fn collect_reads_distributed_to_distributed(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::DistributedInfo,
        target_info: &crate::topology::DistributedInfo,
        target_file_index: usize,
        task_source: &mut Vec<Arc<Mmap>>,
        source_ranges: &mut Vec<(u64, u64, u64)>,
        ranges_per_file: &mut Vec<usize>,
    ) -> Result<()> {
        // Find the target chunk that belongs to this target file
        let target_chunk = target_info
            .chunks()
            .iter()
            .find(|chunk| chunk.filename_index() == target_file_index)
            .ok_or_else(|| RedistributorError::InvalidDataSource {
                message: format!(
                    "No chunk found for target file {} in tensor {}",
                    target_file_index, tensor_name
                ),
            })?;

        // Get tensor properties
        let full_shape = source_info.shape();
        let dtype_size = source_info.dtype().size();

        // Calculate strides for the full tensor
        let ndim = full_shape.len();
        let mut full_strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
        }

        // Process each source chunk to see if it overlaps with our target
        for source_chunk in source_info.chunks() {
            let source_file_index = source_chunk.filename_index();

            // Get source file metadata to calculate byte offsets
            let (source_header_size, source_metadata) =
                &self.source.layout.metadatas[source_file_index];
            let source_tensor_info = source_metadata.info(tensor_name).ok_or_else(|| {
                RedistributorError::TensorNotFound {
                    name: tensor_name.to_string(),
                }
            })?;
            let source_data_offset = source_tensor_info.data_offsets.0;

            // Use optimized direct computation instead of get_intervals + intersection
            let byte_ranges = super::compute_read_ranges_direct(
                source_chunk,
                target_chunk,
                *source_header_size,
                source_data_offset,
                dtype_size,
                &full_strides,
                full_shape,
            );

            if !byte_ranges.is_empty() {
                let ranges_for_this_file = byte_ranges.len();
                source_ranges.extend(byte_ranges);

                // Add the source mmap and record how many ranges belong to this file
                // task_source.add_from_location(&self.source.location, source_file_index);
                let SourceLocation::Local { mmaps } = &self.source.location else {
                    unreachable!();
                };
                task_source.push(Arc::clone(&mmaps[source_file_index]));
                ranges_per_file.push(ranges_for_this_file);
            }
            // println!("Loop {:?}", start.elapsed());
        }

        Ok(())
    }

    fn collect_reads_shared_to_distributed(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::SharedInfo,
        target_info: &crate::topology::DistributedInfo,
        target_file_index: usize,
        task_source: &mut Vec<Arc<Mmap>>,
        source_ranges: &mut Vec<(u64, u64, u64)>,
        ranges_per_file: &mut Vec<usize>,
    ) -> Result<()> {
        // Find the target chunk that belongs to this target file
        let target_chunk = target_info
            .chunks()
            .iter()
            .find(|chunk| chunk.filename_index() == target_file_index)
            .ok_or_else(|| RedistributorError::InvalidDataSource {
                message: format!(
                    "No chunk found for target file {} in tensor {}",
                    target_file_index, tensor_name
                ),
            })?;

        // Get tensor properties
        let full_shape = source_info.shape();
        let dtype_size = source_info.dtype().size();

        // Calculate strides for the full tensor
        let ndim = full_shape.len();
        let mut full_strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
        }

        // Pick the first source file that contains this tensor (they're all identical for shared tensors)
        let source_file_index = source_info.filename_indices()[0];

        // Get source file metadata to calculate byte offsets
        let (source_header_size, source_metadata) =
            &self.source.layout.metadatas[source_file_index];
        let source_tensor_info = source_metadata.info(tensor_name).ok_or_else(|| {
            RedistributorError::TensorNotFound {
                name: tensor_name.to_string(),
            }
        })?;
        let source_data_offset = source_tensor_info.data_offsets.0;

        // Use optimized direct computation for shared-to-distributed
        let byte_ranges = compute_shared_to_distributed_ranges(
            target_chunk,
            *source_header_size,
            source_data_offset,
            dtype_size,
            &full_strides,
            full_shape,
        );

        if !byte_ranges.is_empty() {
            let ranges_for_this_file = byte_ranges.len();
            source_ranges.extend(byte_ranges);

            // Add the source mmap and record how many ranges belong to this file
            let SourceLocation::Local { mmaps } = &self.source.location else {
                unreachable!();
            };
            task_source.push(Arc::clone(&mmaps[source_file_index]));
            ranges_per_file.push(ranges_for_this_file);
        }

        Ok(())
    }

    fn collect_reads_shared_to_shared(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::SharedInfo,
        _target_info: &crate::topology::SharedInfo,
        _target_file_index: usize,
        task_source: &mut Vec<Arc<Mmap>>,
        source_ranges: &mut Vec<(u64, u64, u64)>,
        ranges_per_file: &mut Vec<usize>,
    ) -> Result<()> {
        // Pick the first source file that contains this tensor
        let source_file_index = source_info.filename_indices()[0];

        // Get tensor properties
        let dtype_size = source_info.dtype().size();

        // Get source file metadata to calculate byte offsets
        let (source_header_size, source_metadata) =
            &self.source.layout.metadatas[source_file_index];
        let source_tensor_info = source_metadata.info(tensor_name).ok_or_else(|| {
            RedistributorError::TensorNotFound {
                name: tensor_name.to_string(),
            }
        })?;
        let source_data_offset = source_tensor_info.data_offsets.0;

        // For shared to shared, we copy the entire tensor
        let total_elements = source_info.shape().iter().product::<usize>();
        let total_bytes = total_elements * dtype_size;

        let source_start = (source_header_size + source_data_offset) as u64;
        let source_end = (source_header_size + source_data_offset + total_bytes) as u64;

        source_ranges.push((source_start, source_end, 0));

        // Add the source mmap and record how many ranges belong to this file
        let SourceLocation::Local { mmaps } = &self.source.location else {
            unreachable!();
        };
        task_source.push(Arc::clone(&mmaps[source_file_index]));
        ranges_per_file.push(1);

        Ok(())
    }

    fn collect_reads_distributed_to_shared(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::DistributedInfo,
        _target_info: &crate::topology::SharedInfo,
        _target_file_index: usize,
        task_source: &mut Vec<Arc<Mmap>>,
        source_ranges: &mut Vec<(u64, u64, u64)>,
        ranges_per_file: &mut Vec<usize>,
    ) -> Result<()> {
        // Get tensor properties
        let full_shape = source_info.shape();
        let dtype_size = source_info.dtype().size();

        // Calculate strides for the full tensor
        let ndim = full_shape.len();
        let mut full_strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
        }

        // The target is the full tensor, so target intervals cover everything
        let total_elements = full_shape.iter().product::<usize>();
        let _target_intervals = vec![(0, total_elements)];

        // Process each source chunk
        for source_chunk in source_info.chunks() {
            let source_file_index = source_chunk.filename_index();

            // Get source file metadata to calculate byte offsets
            let (source_header_size, source_metadata) =
                &self.source.layout.metadatas[source_file_index];
            let source_tensor_info = source_metadata.info(tensor_name).ok_or_else(|| {
                RedistributorError::TensorNotFound {
                    name: tensor_name.to_string(),
                }
            })?;
            let source_data_offset = source_tensor_info.data_offsets.0;

            // Use optimized direct computation for distributed-to-shared
            let byte_ranges = compute_distributed_to_shared_ranges(
                source_chunk,
                *source_header_size,
                source_data_offset,
                dtype_size,
                &full_strides,
                full_shape,
            );

            if !byte_ranges.is_empty() {
                let ranges_for_this_file = byte_ranges.len();
                source_ranges.extend(byte_ranges);

                // Add the source mmap and record how many ranges belong to this file
                let SourceLocation::Local { mmaps } = &self.source.location else {
                    unreachable!();
                };
                task_source.push(Arc::clone(&mmaps[source_file_index]));
                ranges_per_file.push(ranges_for_this_file);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redistributor::intersection;
    use crate::topology::{Chunk, DistributedInfo, SharedInfo};
    use safetensors::{Dtype, serialize, tensor::TensorView};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    // Helper function to convert f32 slice to little-endian bytes
    fn f32s_to_le_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // Helper function to create test data
    fn create_test_data_8x4() -> Vec<f32> {
        (0..32).map(|i| i as f32).collect()
    }

    fn create_test_data_4x8() -> Vec<f32> {
        (100..132).map(|i| i as f32).collect()
    }

    // Helper function to calculate SHA256 of a file
    async fn calculate_file_hash<P: AsRef<std::path::Path>>(path: P) -> String {
        let path = path.as_ref();
        let contents = tokio::fs::read(path).await.unwrap();

        if contents.len() >= 8 {
            // Read the header size (first 8 bytes)
            let header_size = u64::from_le_bytes([
                contents[0],
                contents[1],
                contents[2],
                contents[3],
                contents[4],
                contents[5],
                contents[6],
                contents[7],
            ]) as usize;

            if contents.len() >= 8 + header_size {
                // Extract and display the JSON header
                let header_bytes = &contents[8..8 + header_size];
                if let Ok(header_str) = std::str::from_utf8(header_bytes) {}

                // Show some info about the data section
                let data_start = 8 + header_size;
                let data_size = contents.len() - data_start;

                // Show first 32 bytes of data as hex
                if data_size > 0 {
                    let hex_bytes = std::cmp::min(32, data_size);
                }
            }
        }

        // Show entire file content as UTF-8 (lossy)
        let mut hasher = Sha256::new();
        hasher.update(&contents);
        let hash = format!("{:x}", hasher.finalize());
        hash
    }

    #[tokio::test]
    async fn test_redistribution_round_trip() {
        // Create temp directories
        let source_dir = TempDir::new().unwrap();
        let distributed_dir = TempDir::new().unwrap();
        let final_dir = TempDir::new().unwrap();

        // Step 1: Create original model.safetensors with 2 tensors
        let tensor1_data = create_test_data_8x4(); // 8x4 tensor = 32 elements
        let tensor2_data = create_test_data_4x8(); // 4x8 tensor = 32 elements

        let tensor1_bytes = f32s_to_le_bytes(&tensor1_data);
        let tensor2_bytes = f32s_to_le_bytes(&tensor2_data);

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "tensor1".to_string(),
            TensorView::new(Dtype::F32, vec![8, 4], &tensor1_bytes).unwrap(),
        );
        tensors.insert(
            "tensor2".to_string(),
            TensorView::new(Dtype::F32, vec![4, 8], &tensor2_bytes).unwrap(),
        );

        let original_bytes = serialize(&tensors, &None).unwrap();
        let original_path = source_dir.path().join("model.safetensors");
        tokio::fs::write(&original_path, &original_bytes)
            .await
            .unwrap();

        // Calculate hash of original file
        let original_hash = calculate_file_hash(&original_path).await;

        // Step 2: Create target topology for 2 ranks with different split dimensions
        let mut target_tensors = BTreeMap::new();

        // Split tensor1 (8x4) along first dimension (8/2 = 4 each)
        target_tensors.insert(
            "tensor1".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![8, 4],
                Dtype::F32,
                vec![
                    Chunk::new(vec![0, 0], vec![4, 4], 0), // First 4 rows to rank 0
                    Chunk::new(vec![4, 0], vec![4, 4], 1), // Last 4 rows to rank 1
                ],
            )),
        );

        // Split tensor2 (4x8) along second dimension (8/2 = 4 each)
        target_tensors.insert(
            "tensor2".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![4, 8],
                Dtype::F32,
                vec![
                    Chunk::new(vec![0, 0], vec![4, 4], 0), // First 4 columns to rank 0
                    Chunk::new(vec![0, 4], vec![4, 4], 1), // Last 4 columns to rank 1
                ],
            )),
        );

        let distributed_topology = Topology::new(
            target_tensors,
            vec![
                "rank0.safetensors".to_string(),
                "rank1.safetensors".to_string(),
            ],
            2,
        )
        .unwrap();

        // Step 3: Redistribute from single file to 2 ranks
        let mut redistributor1 = Redistributor::from_local(
            source_dir.path(),
            distributed_dir.path(),
            distributed_topology,
        )
        .unwrap();

        redistributor1.redistribute().await.unwrap();

        // Verify distributed files exist
        assert!(distributed_dir.path().join("rank0.safetensors").exists());
        assert!(distributed_dir.path().join("rank1.safetensors").exists());
        assert!(distributed_dir.path().join("topology.json").exists());

        // Step 4: Create target topology for reconstruction back to 1 rank
        let mut final_tensors = BTreeMap::new();
        final_tensors.insert(
            "tensor1".to_string(),
            Tensor::Shared(SharedInfo::new(vec![8, 4], Dtype::F32, vec![0])),
        );
        final_tensors.insert(
            "tensor2".to_string(),
            Tensor::Shared(SharedInfo::new(vec![4, 8], Dtype::F32, vec![0])),
        );

        let final_topology =
            Topology::new(final_tensors, vec!["model.safetensors".to_string()], 1).unwrap();

        // Step 5: Redistribute from 2 ranks back to single file
        let mut redistributor2 =
            Redistributor::from_local(distributed_dir.path(), final_dir.path(), final_topology)
                .unwrap();

        redistributor2.redistribute().await.unwrap();

        // Verify final file exists
        let final_path = final_dir.path().join("model.safetensors");
        assert!(final_path.exists());

        // Step 6: Calculate hash of final file and compare
        let final_hash = calculate_file_hash(&final_path).await;

        // The files should be identical
        assert_eq!(
            original_hash, final_hash,
            "Round-trip redistribution should preserve file integrity"
        );

        // Step 7: Additional verification - load and compare tensor data
        let final_bytes = tokio::fs::read(&final_path).await.unwrap();
        let final_safetensors = safetensors::SafeTensors::deserialize(&final_bytes).unwrap();

        // Verify tensor1
        let final_tensor1 = final_safetensors.tensor("tensor1").unwrap();
        assert_eq!(final_tensor1.shape(), vec![8, 4]);
        assert_eq!(final_tensor1.dtype(), Dtype::F32);
        let final_tensor1_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                final_tensor1.data().as_ptr() as *const f32,
                final_tensor1.data().len() / 4,
            )
        };
        assert_eq!(final_tensor1_data, tensor1_data.as_slice());

        // Verify tensor2
        let final_tensor2 = final_safetensors.tensor("tensor2").unwrap();
        assert_eq!(final_tensor2.shape(), vec![4, 8]);
        assert_eq!(final_tensor2.dtype(), Dtype::F32);
        let final_tensor2_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                final_tensor2.data().as_ptr() as *const f32,
                final_tensor2.data().len() / 4,
            )
        };
        assert_eq!(final_tensor2_data, tensor2_data.as_slice());
    }

    #[test]
    fn test_intersection_function() {
        // Test case: [8, 8] full array (64 elements total)
        // Source chunk: [4:, :] = rows 4-7, all columns = elements 32-63 in 1D
        // Target chunk: [:, :2] = all rows, columns 0-1

        // Source intervals: [4:, :] covers elements 32-63
        let source_intervals = vec![(32, 64)];

        // Target intervals: [:, :2] covers first 2 columns of each row
        // Row 0, cols 0-1: elements 0-1   = (0, 2)
        // Row 1, cols 0-1: elements 8-9   = (8, 10)
        // Row 2, cols 0-1: elements 16-17 = (16, 18)
        // Row 3, cols 0-1: elements 24-25 = (24, 26)
        // Row 4, cols 0-1: elements 32-33 = (32, 34)  <- intersects with source
        // Row 5, cols 0-1: elements 40-41 = (40, 42)  <- intersects with source
        // Row 6, cols 0-1: elements 48-49 = (48, 50)  <- intersects with source
        // Row 7, cols 0-1: elements 56-57 = (56, 58)  <- intersects with source
        let target_intervals = vec![
            (0, 2),
            (8, 10),
            (16, 18),
            (24, 26),
            (32, 34),
            (40, 42),
            (48, 50),
            (56, 58),
        ];

        let result = intersection(&source_intervals, &target_intervals);

        // Expected result: (source_offset, target_offset, length)
        // - (32, 34): source_offset=0 (32-32), target_offset=8 (4 intervals * 2 bytes each), length=2
        // - (40, 42): source_offset=8 (40-32), target_offset=10 (5 intervals * 2 bytes each), length=2
        // - (48, 50): source_offset=16 (48-32), target_offset=12 (6 intervals * 2 bytes each), length=2
        // - (56, 58): source_offset=24 (56-32), target_offset=14 (7 intervals * 2 bytes each), length=2
        let expected = vec![
            (0, 8, 2),   // (32,34) -> source offset 0, target offset 8, length 2
            (8, 10, 2),  // (40,42) -> source offset 8, target offset 10, length 2
            (16, 12, 2), // (48,50) -> source offset 16, target offset 12, length 2
            (24, 14, 2), // (56,58) -> source offset 24, target offset 14, length 2
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_intersection_no_overlap() {
        // Test case where there's no overlap
        let source_intervals = vec![(10, 20)];
        let target_intervals = vec![(0, 5), (25, 30)];

        let result = intersection(&source_intervals, &target_intervals);
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_intersection_partial_overlap() {
        // Test case with partial overlaps
        let source_intervals = vec![(5, 15)];
        let target_intervals = vec![(0, 8), (12, 20)];

        // Expected intersections:
        // (5, 8) with source_offset=0, target_offset=5 (within first target interval), length=3
        // (12, 15) with source_offset=7, target_offset=8 (8 bytes from first interval + 0 from second), length=3
        let expected = vec![
            (0, 5, 3), // intersection (5,8) - offset 5 within first target interval
            (7, 8, 3), // intersection (12,15) - offset 8 (cumulative: 8 from first interval + 0)
        ];

        let result = intersection(&source_intervals, &target_intervals);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_intersection_multiple_source_intervals() {
        // Test case with multiple source intervals
        let source_intervals = vec![(0, 5), (10, 15)];
        let target_intervals = vec![(3, 8), (12, 18)];

        // Expected intersections:
        // (3, 5) from first source with source_offset=3, target_offset=0 (start of first target interval), length=2
        // (12, 15) from second source with source_offset=2, target_offset=5 (5 bytes from first interval + 0 from second), length=3
        let expected = vec![
            (3, 0, 2), // intersection (3,5) from first source interval
            (7, 5, 3), // intersection (12,15) from second source interval - offset 5 (cumulative: 5 from first + 0)
        ];

        let result = intersection(&source_intervals, &target_intervals);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_intersection_target_spans_multiple_sources() {
        // Test case where a single target interval spans across multiple source intervals
        let source_intervals = vec![(10, 20), (30, 40)];
        let target_intervals = vec![(15, 35)];

        // Target interval (15, 35) intersects with:
        // - First source (10, 20) at (15, 20) - length 5, maps to target offsets 0-4
        // - Second source (30, 40) at (30, 35) - length 5, maps to target offsets 15-19 (position-based)
        // OR if sequential filling: target offsets 5-9
        //
        // Based on the other tests, target_offset should be position-based, not sequential
        let expected = vec![
            (5, 0, 5),   // intersection (15,20): position 15 in target → offset 0
            (10, 15, 5), // intersection (30,35): position 30 in target → offset 15 (30-15)
        ];

        let result = intersection(&source_intervals, &target_intervals);
        assert_eq!(result, expected);
    }
}
