use super::location::{Source, SourceLocation, Target, WriteLocation};
use super::task::Task;
use super::{
    Layout, RedistributorError, Result, compute_distributed_to_shared_ranges,
    compute_shared_to_distributed_ranges,
};
use crate::topology::{Tensor, Topology};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use num_cpus;
use safetensors::tensor::{Metadata, TensorInfo};
use std::path::Path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// The structure responsible for managing a redistribution
/// from one topology to another on disk.
/// The strategy is to read the topology (or assume rank-1 if missing)
/// and then all the header to know the tensors.
/// Then the redistributor will start fetching all the various parts of data it requires from the
/// original files before writing them on-disk.
/// The writes are ordered to help be faster on NVMe.
#[derive(Clone)]
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
                location: SourceLocation {
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

    /// Starts running the remapping of the files from the source topology to the target topology.
    /// It returns the list of files created (relative to the output_dir).
    pub fn redistribute(&mut self) -> Result<Vec<String>> {
        self.target.location.init(&self.target.layout)?;

        // Get number of CPUs
        let num_cpus = num_cpus::get();

        // Estimate of the data needed to be written
        let mut total = 0u64;
        for (header_size, metadata) in &self.target.layout.metadatas {
            let max_data_end = metadata
                .tensors()
                .values()
                .map(|tensor_info| tensor_info.data_offsets.1)
                .max()
                .unwrap_or(0);
            total += (header_size + max_data_end) as u64;
        }

        let progress = ProgressBar::new(total);
        progress.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{eta_precise}] {msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})",
                )
                .unwrap(),
        );

        // Calculate chunk size for task distribution
        let chunk_size = total / num_cpus as u64;

        // Spawn worker threads
        let mut handles = Vec::with_capacity(num_cpus);
        for rank in 0..num_cpus {
            let mut p = progress.clone();
            let redistributor = self.clone();
            let handle: JoinHandle<Result<()>> = thread::spawn(move || {
                redistributor.create_tasks_for_rank(rank, chunk_size, &mut p)?;
                Ok(())
            });
            handles.push(handle);
        }

        // Wait for all workers to complete
        for handle in handles {
            match handle.join() {
                Ok(res) => res?,
                Err(_) => return Err(super::RedistributorError::ThreadPanic),
            }
        }

        progress.set_message("Done, kernel flush...");

        self.target.location.save(&self.target.layout.topology)?;

        progress.finish_with_message("Done");

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

    /// Creates tasks for a specific rank based on the chunk size
    fn create_tasks_for_rank(
        &self,
        rank: usize,
        chunk_size: u64,
        progress: &mut ProgressBar,
    ) -> Result<()> {
        let mut current_size = 0u64;
        // Process each target file in order
        for (target_file_index, _filename) in
            self.target.layout.topology.filenames().iter().enumerate()
        {
            // Get the metadata for this target file
            let (header_size, metadata) = &self.target.layout.metadatas[target_file_index];

            // Collect tensors and sort by data offset for sequential writes
            let tensors = metadata.tensors();
            let mut tensor_entries: Vec<(&String, &TensorInfo)> =
                tensors.iter().map(|(k, v)| (k, *v)).collect();
            tensor_entries.sort_by_key(|(_, tensor_info)| tensor_info.data_offsets.0);

            // Process each tensor in write order
            for (tensor_name, tensor_info) in tensor_entries.iter() {
                if let Some(task) = self.create_write_task_for_tensor(
                    target_file_index,
                    *header_size,
                    tensor_name,
                    tensor_info,
                ) {
                    // Check if this task belongs to this rank
                    let task_size = task.target_end - task.target_start;
                    let task_rank = (current_size / chunk_size) as usize;
                    if task_rank == rank {
                        task.run()?;
                        progress.inc(task_size as u64);
                    } else if task_rank > rank {
                        return Ok(());
                    }
                    current_size += task_size;
                }
            }
        }
        Ok(())
    }

    fn create_write_task_for_tensor(
        &self,
        target_file_index: usize,
        target_header_size: usize,
        tensor_name: &str,
        target_tensor_info: &TensorInfo,
    ) -> Option<Task> {
        // Look up the tensor in source topology
        let source_tensor = self
            .source
            .layout
            .topology
            .tensors()
            .get(tensor_name)
            .ok_or_else(|| RedistributorError::TensorNotFound {
                name: tensor_name.to_string(),
            })
            .ok()?;

        let target_tensor = self
            .target
            .layout
            .topology
            .tensors()
            .get(tensor_name)
            .ok_or_else(|| RedistributorError::TensorNotFound {
                name: tensor_name.to_string(),
            })
            .ok()?;

        // Calculate target write parameters
        let target_start = (target_header_size + target_tensor_info.data_offsets.0) as u64;
        let target_end = (target_header_size + target_tensor_info.data_offsets.1) as u64;

        // Collect source reads needed to fulfill this target write
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
                )
                .ok()?;
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
                )
                .ok()?;
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
                )
                .ok()?;
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
                )
                .ok()?;
            }
        }

        // Create and return the task if we have any reads to do
        if !source_ranges.is_empty() {
            self.target.location.create_write_task(
                target_file_index,
                target_start,
                target_end,
                task_source,
                source_ranges,
                ranges_per_file,
            )
        } else {
            None
        }
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
                let mmaps = &self.source.location.mmaps;
                task_source.push(Arc::clone(&mmaps[source_file_index]));
                ranges_per_file.push(ranges_for_this_file);
            }
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
            let mmaps = &self.source.location.mmaps;
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
        let mmaps = &self.source.location.mmaps;
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
                let mmaps = &self.source.location.mmaps;
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
    fn calculate_file_hash<P: AsRef<std::path::Path>>(path: P) -> String {
        let contents = std::fs::read(path).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(&contents);
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn test_redistribution_round_trip() {
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
        std::fs::write(&original_path, &original_bytes).unwrap();

        // Calculate hash of original file
        let original_hash = calculate_file_hash(&original_path);

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

        redistributor1.redistribute().unwrap();

        // Verify distributed files exist
        assert!(distributed_dir.path().join("rank0.safetensors").exists());
        assert!(distributed_dir.path().join("rank1.safetensors").exists());
        assert!(distributed_dir.path().join("topology.json").exists());

        // Step 4: Create target topology for reconstruction back to 1 rank
        let mut final_tensors = BTreeMap::new();
        final_tensors.insert(
            "tensor1".to_string(),
            // Tensor::Shared(SharedInfo::new(vec![8, 4], Dtype::F32, vec![0])),
            Tensor::Distributed(
                // SharedInfo::new(vec![8, 4], Dtype::F32, vec![0])
                DistributedInfo::new(
                    vec![8, 4],
                    Dtype::F32,
                    vec![Chunk::new(vec![0, 0], vec![8, 4], 0)],
                ),
            ),
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

        redistributor2.redistribute().unwrap();

        // Verify final file exists
        let final_path = final_dir.path().join("model.safetensors");
        assert!(final_path.exists());

        // Step 6: Calculate hash of final file and compare
        let final_hash = calculate_file_hash(&final_path);

        // The files should be identical
        assert_eq!(
            original_hash, final_hash,
            "Round-trip redistribution should preserve file integrity"
        );

        // Step 7: Additional verification - load and compare tensor data
        let final_bytes = std::fs::read(&final_path).unwrap();
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
    fn test_redistribution_round_trip_bug_case() {
        // This test case replicates the exact tensor patterns that exposed the bug:
        // - 'down' tensor: 8x2 with values 0-15 (torch.arange(16).view(8,2))
        // - 'up' tensor: 2x8 with values 0,2,4,...,30 (torch.arange(16)*2).view(2,8))

        // Create temp directories
        let source_dir = TempDir::new().unwrap();
        let distributed_dir = TempDir::new().unwrap();
        let final_dir = TempDir::new().unwrap();

        // Step 1: Create original model.safetensors with the exact bug case tensors
        // 'down' tensor: 8x2 = 16 elements, values 0.0 to 15.0
        let down_data: Vec<f32> = (0..16).map(|i| i as f32).collect();

        // 'up' tensor: 2x8 = 16 elements, values 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
        let up_data: Vec<f32> = (0..16).map(|i| (i * 2) as f32).collect();

        let down_bytes = f32s_to_le_bytes(&down_data);
        let up_bytes = f32s_to_le_bytes(&up_data);

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "down".to_string(),
            TensorView::new(Dtype::F32, vec![8, 2], &down_bytes).unwrap(),
        );
        tensors.insert(
            "up".to_string(),
            TensorView::new(Dtype::F32, vec![2, 8], &up_bytes).unwrap(),
        );

        let original_bytes = serialize(&tensors, &None).unwrap();
        let original_path = source_dir.path().join("model.safetensors");
        std::fs::write(&original_path, &original_bytes).unwrap();

        // Calculate hash of original file
        let original_hash = calculate_file_hash(&original_path);

        // Step 2: Create target topology for 2 ranks
        // Split 'down' tensor (8x2) along first dimension: first 4 rows to rank0, last 4 rows to rank1
        // Split 'up' tensor (2x8) along second dimension: first 4 cols to rank0, last 4 cols to rank1
        let mut target_tensors = BTreeMap::new();

        target_tensors.insert(
            "down".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![8, 2],
                Dtype::F32,
                vec![
                    Chunk::new(vec![0, 0], vec![4, 2], 0), // First 4 rows to rank 0
                    Chunk::new(vec![4, 0], vec![4, 2], 1), // Last 4 rows to rank 1
                ],
            )),
        );

        target_tensors.insert(
            "up".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![2, 8],
                Dtype::F32,
                vec![
                    Chunk::new(vec![0, 0], vec![2, 4], 0), // First 4 columns to rank 0
                    Chunk::new(vec![0, 4], vec![2, 4], 1), // Last 4 columns to rank 1
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

        redistributor1.redistribute().unwrap();

        // Step 4: Create target topology for reconstruction back to 1 rank
        let mut final_tensors = BTreeMap::new();
        final_tensors.insert(
            "down".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![8, 2],
                Dtype::F32,
                vec![Chunk::new(vec![0, 0], vec![8, 2], 0)],
            )),
        );
        final_tensors.insert(
            "up".to_string(),
            Tensor::Shared(SharedInfo::new(vec![2, 8], Dtype::F32, vec![0])),
        );

        let final_topology =
            Topology::new(final_tensors, vec!["model.safetensors".to_string()], 1).unwrap();

        // Step 5: Redistribute from 2 ranks back to single file
        let mut redistributor2 =
            Redistributor::from_local(distributed_dir.path(), final_dir.path(), final_topology)
                .unwrap();

        redistributor2.redistribute().unwrap();

        // Verify final file exists
        let final_path = final_dir.path().join("model.safetensors");
        assert!(final_path.exists());

        // Step 6: Calculate hash of final file and compare
        let final_hash = calculate_file_hash(&final_path);

        // This assertion should FAIL and expose the bug
        assert_eq!(
            original_hash, final_hash,
            "BUG DETECTED: Round-trip redistribution corrupted file integrity for 8x2/2x8 tensors"
        );

        // Step 7: Additional verification - load and compare tensor data element by element
        let final_bytes = std::fs::read(&final_path).unwrap();
        let final_safetensors = safetensors::SafeTensors::deserialize(&final_bytes).unwrap();

        // Verify 'down' tensor
        let final_down = final_safetensors.tensor("down").unwrap();
        assert_eq!(final_down.shape(), vec![8, 2]);
        assert_eq!(final_down.dtype(), Dtype::F32);
        let final_down_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                final_down.data().as_ptr() as *const f32,
                final_down.data().len() / 4,
            )
        };
        assert_eq!(
            final_down_data,
            down_data.as_slice(),
            "BUG DETECTED: 'down' tensor values corrupted after round-trip redistribution"
        );

        // Verify 'up' tensor - this should expose the bug
        let final_up = final_safetensors.tensor("up").unwrap();
        assert_eq!(final_up.shape(), vec![2, 8]);
        assert_eq!(final_up.dtype(), Dtype::F32);
        let final_up_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                final_up.data().as_ptr() as *const f32,
                final_up.data().len() / 4,
            )
        };

        // This assertion should FAIL and show the exact bug
        assert_eq!(
            final_up_data,
            up_data.as_slice(),
            "BUG DETECTED: 'up' tensor values corrupted after round-trip redistribution.\nOriginal: {:?}\nReconstructed: {:?}",
            up_data,
            final_up_data
        );
    }

    #[test]
    fn test_redistribution_round_trip_example_logic() {
        // This test replicates the exact logic used by the redistribute example
        // which uses determine_split_dimension() to decide how to split tensors

        // Create temp directories
        let source_dir = TempDir::new().unwrap();
        let distributed_dir = TempDir::new().unwrap();
        let final_dir = TempDir::new().unwrap();

        // Step 1: Create original model.safetensors with the exact bug case tensors
        // 'down' tensor: 8x2 = 16 elements, values 0.0 to 15.0
        let down_data: Vec<f32> = (0..16).map(|i| i as f32).collect();

        // 'up' tensor: 2x8 = 16 elements, values 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
        let up_data: Vec<f32> = (0..16).map(|i| (i * 2) as f32).collect();

        let down_bytes = f32s_to_le_bytes(&down_data);
        let up_bytes = f32s_to_le_bytes(&up_data);

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "down".to_string(),
            TensorView::new(Dtype::F32, vec![8, 2], &down_bytes).unwrap(),
        );
        tensors.insert(
            "up".to_string(),
            TensorView::new(Dtype::F32, vec![2, 8], &up_bytes).unwrap(),
        );

        let original_bytes = serialize(&tensors, &None).unwrap();
        let original_path = source_dir.path().join("model.safetensors");
        std::fs::write(&original_path, &original_bytes).unwrap();

        // Calculate hash of original file
        let original_hash = calculate_file_hash(&original_path);

        // Step 2: Create target topology for 2 ranks using the SAME LOGIC as the redistribute example
        // "down" tensor: determine_split_dimension() returns Some(0) -> split along dimension 0
        // "up" tensor: determine_split_dimension() returns Some(1) -> split along dimension 1

        let mut target_tensors = BTreeMap::new();
        let target_world_size = 2;

        // 'down' tensor (8x2) split along dimension 0: 8/2 = 4 rows per rank
        let down_chunk_size_per_rank = 8 / target_world_size;
        let down_chunks = vec![
            Chunk::new(vec![0, 0], vec![down_chunk_size_per_rank, 2], 0), // Rows 0-3 to rank 0
            Chunk::new(
                vec![down_chunk_size_per_rank, 0],
                vec![down_chunk_size_per_rank, 2],
                1,
            ), // Rows 4-7 to rank 1
        ];
        target_tensors.insert(
            "down".to_string(),
            Tensor::Distributed(DistributedInfo::new(vec![8, 2], Dtype::F32, down_chunks)),
        );

        // 'up' tensor (2x8) split along dimension 1: 8/2 = 4 columns per rank
        let up_chunk_size_per_rank = 8 / target_world_size;
        let up_chunks = vec![
            Chunk::new(vec![0, 0], vec![2, up_chunk_size_per_rank], 0), // Cols 0-3 to rank 0
            Chunk::new(
                vec![0, up_chunk_size_per_rank],
                vec![2, up_chunk_size_per_rank],
                1,
            ), // Cols 4-7 to rank 1
        ];
        target_tensors.insert(
            "up".to_string(),
            Tensor::Distributed(DistributedInfo::new(vec![2, 8], Dtype::F32, up_chunks)),
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

        redistributor1.redistribute().unwrap();

        // Step 4: Create target topology for reconstruction back to 1 rank
        // This uses the EXACT same logic as the redistribute example for target_world_size = 1
        // Even with world_size=1, it still creates Distributed tensors with single chunks based on split dimensions
        let mut final_tensors = BTreeMap::new();

        // "down" tensor: determine_split_dimension("down") returns Some(0) -> still creates distributed with single chunk
        final_tensors.insert(
            "down".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![8, 2],
                Dtype::F32,
                vec![Chunk::new(vec![0, 0], vec![8, 2], 0)], // Single chunk covering full tensor
            )),
        );

        // "up" tensor: determine_split_dimension("up") returns Some(1) -> still creates distributed with single chunk
        final_tensors.insert(
            "up".to_string(),
            Tensor::Distributed(DistributedInfo::new(
                vec![2, 8],
                Dtype::F32,
                vec![Chunk::new(vec![0, 0], vec![2, 8], 0)], // Single chunk covering full tensor
            )),
        );

        let final_topology =
            Topology::new(final_tensors, vec!["model.safetensors".to_string()], 1).unwrap();

        // Step 5: Redistribute from 2 ranks back to single file
        let mut redistributor2 =
            Redistributor::from_local(distributed_dir.path(), final_dir.path(), final_topology)
                .unwrap();

        redistributor2.redistribute().unwrap();

        // Verify final file exists
        let final_path = final_dir.path().join("model.safetensors");
        assert!(final_path.exists());

        // Step 6: Calculate hash of final file and compare
        let final_hash = calculate_file_hash(&final_path);

        // This assertion should FAIL and expose the bug
        assert_eq!(
            original_hash, final_hash,
            "BUG DETECTED: Round-trip redistribution corrupted file integrity for redistribute example logic"
        );

        // Step 7: Additional verification - load and compare tensor data element by element
        let final_bytes = std::fs::read(&final_path).unwrap();
        let final_safetensors = safetensors::SafeTensors::deserialize(&final_bytes).unwrap();

        // Verify 'down' tensor
        let final_down = final_safetensors.tensor("down").unwrap();
        assert_eq!(final_down.shape(), vec![8, 2]);
        assert_eq!(final_down.dtype(), Dtype::F32);
        let final_down_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                final_down.data().as_ptr() as *const f32,
                final_down.data().len() / 4,
            )
        };
        assert_eq!(
            final_down_data,
            down_data.as_slice(),
            "BUG DETECTED: 'down' tensor values corrupted after round-trip redistribution.\nOriginal: {:?}\nReconstructed: {:?}",
            down_data,
            final_down_data
        );

        // Verify 'up' tensor - this should expose the bug
        let final_up = final_safetensors.tensor("up").unwrap();
        assert_eq!(final_up.shape(), vec![2, 8]);
        assert_eq!(final_up.dtype(), Dtype::F32);
        let final_up_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                final_up.data().as_ptr() as *const f32,
                final_up.data().len() / 4,
            )
        };

        // This assertion should FAIL and show the exact bug
        assert_eq!(
            final_up_data,
            up_data.as_slice(),
            "BUG DETECTED: 'up' tensor values corrupted after round-trip redistribution.\nOriginal: {:?}\nReconstructed: {:?}",
            up_data,
            final_up_data
        );
    }
}
