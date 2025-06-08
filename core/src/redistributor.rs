use crate::topology::{SharedInfo, Tensor, Topology, TopologyError, get_intervals};
use futures::future::join_all;
use indicatif::{ProgressBar, ProgressStyle, style::TemplateError};
use memmap2::{Mmap, MmapMut};
use safetensors::SafeTensors;
use safetensors::tensor::{Metadata, TensorInfo};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

struct WriteTask {
    source_filename: PathBuf,
    source_start: u64,
    target_filename: PathBuf,
    target_start: u64,
    length: usize,
}

// Memory-mapped task type for high performance file operations
struct MmapWriteTask {
    source_mmap: Arc<Mutex<Mmap>>,
    source_start: u64,
    target_mmap: Arc<Mutex<MmapMut>>,
    target_start: u64,
    length: usize,
}

impl MmapWriteTask {
    fn run(&self) -> Result<()> {
        let source_guard = self.source_mmap.lock().unwrap();
        let mut target_guard = self.target_mmap.lock().unwrap();

        let source_slice =
            &source_guard[self.source_start as usize..(self.source_start as usize + self.length)];
        let target_slice = &mut target_guard
            [self.target_start as usize..(self.target_start as usize + self.length)];

        target_slice.copy_from_slice(source_slice);

        Ok(())
    }
}

/// Structure for deserializing model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    /// Map of tensor names to their containing file
    weight_map: HashMap<String, String>,
}

/// Error type for redistributor operations
#[derive(Debug, Error)]
pub enum RedistributorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Topology error: {0}")]
    Topology(#[from] TopologyError),

    #[error("Invalid tensor data source: {message}")]
    InvalidDataSource { message: String },

    #[error("Tensor not found: {name}")]
    TensorNotFound { name: String },

    #[error("Invalid tensor dimension {dim} for tensor with shape {shape:?}")]
    InvalidDimension { dim: usize, shape: Vec<usize> },

    #[error("Invalid slice range [{start}, {end}) for dimension {dim} with size {size}")]
    InvalidSliceRange {
        start: usize,
        end: usize,
        dim: usize,
        size: usize,
    },

    #[error(
        "No valid input found in directory {path:?} (expected topology.json + rank*.safetensors OR model.safetensors OR model.safetensors.index.json + chunked files)"
    )]
    NoValidInput { path: PathBuf },

    #[error("Failed to parse target world size: {input}")]
    InvalidWorldSize { input: String },

    #[error("Template error: {0}")]
    Template(#[from] TemplateError),
}

/// Result type for redistributor operations
type Result<T> = std::result::Result<T, RedistributorError>;

/// Data source for tensor redistribution
struct Layout {
    topology: Topology,
    metadatas: Vec<(usize, Metadata)>,
    dir: PathBuf,
}

/// Async streaming tensor redistributor with parallel processing and pre-calculated offsets
pub struct AsyncTensorRedistributor {
    source: Layout,
    target: Layout,
}

impl AsyncTensorRedistributor {
    /// Create a new redistributor for reconstruction from distributed files
    pub fn new<P: AsRef<Path>>(
        source_dir: P,
        target_dir: P,
        target_topology: Topology,
    ) -> Result<Self> {
        // Load the existing topology (or create from model.safetensors)
        let source_dir = source_dir.as_ref().to_path_buf();
        let target_dir = target_dir.as_ref().to_path_buf();

        let source_topology = load_or_create_topology(&source_dir)?;
        let source_metadatas: Result<Vec<(usize, Metadata)>> = source_topology
            .filenames()
            .iter()
            .map(|f| Ok(safetensors_metadata(source_dir.join(f))?))
            .collect();
        let source_metadatas = source_metadatas?;
        let target_metadatas = Self::pre_calculate_metadatas(&target_topology)?;

        let source = Layout {
            dir: source_dir,
            topology: source_topology,
            metadatas: source_metadatas,
        };
        let target = Layout {
            dir: target_dir,
            topology: target_topology,
            metadatas: target_metadatas,
        };
        Ok(Self { source, target })
    }

    /// Redistribute tensors to target directory and return list of created files
    pub async fn redistribute(&self) -> Result<Vec<String>> {
        println!("Start");
        let start = Instant::now();
        tokio::fs::create_dir_all(&self.target.dir).await?;

        self.create_files_with_headers().await?;
        println!("Created headers {:?}", start.elapsed());

        let write_tasks = self.create_tasks()?;
        println!(
            "Created tasks {} in {:?}",
            write_tasks.len(),
            start.elapsed()
        );

        // Always use memory mapped approach (30x faster!)
        self.execute_mmap_tasks(write_tasks).await?;
        println!("Tasks done {:?}", start.elapsed());

        // Collect created safetensors files
        let mut created_files = Vec::new();
        for filename in self.target.topology.filenames() {
            created_files.push(filename.clone());
        }

        // Write topology.json if needed (for multi-rank outputs)
        if self.target.topology.world_size() > 1 {
            let topology_json = serde_json::to_vec_pretty(&self.target.topology)?;
            tokio::fs::write(self.target.dir.join("topology.json"), &topology_json).await?;
            created_files.push("topology.json".to_string());
        }

        Ok(created_files)
    }

    /// Execute tasks using memory mapped files for better performance
    async fn execute_mmap_tasks(&self, write_tasks: Vec<WriteTask>) -> Result<()> {
        println!("Setting up memory mapped files...");
        let setup_start = Instant::now();

        // Create mmap pool and open all files
        let mut mmap_pool = MmapPool::new();
        mmap_pool.open_source_files(&self.source.topology, &self.source.dir)?;
        mmap_pool.open_target_files(&self.target.topology, &self.target.dir)?;

        println!("Mmap setup took {:?}", setup_start.elapsed());

        // Convert WriteTask to MmapWriteTask
        let mut total = 0;
        let mmap_tasks: Result<Vec<_>> = write_tasks
            .into_iter()
            .map(|task| {
                let source_mmap = mmap_pool
                    .get_source_mmap(&task.source_filename)
                    .ok_or_else(|| RedistributorError::InvalidDataSource {
                        message: format!("Source file not found: {:?}", task.source_filename),
                    })?;
                let target_mmap = mmap_pool
                    .get_target_mmap(&task.target_filename)
                    .ok_or_else(|| RedistributorError::InvalidDataSource {
                        message: format!("Target file not found: {:?}", task.target_filename),
                    })?;
                total += task.length;

                Ok(MmapWriteTask {
                    source_mmap,
                    source_start: task.source_start,
                    target_mmap,
                    target_start: task.target_start,
                    length: task.length,
                })
            })
            .collect();

        let mmap_tasks = mmap_tasks?;
        println!("Converted {} tasks to mmap tasks", mmap_tasks.len());

        // Execute all mmap tasks in parallel
        let progress = ProgressBar::new(total as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})")
                .unwrap(),
        );
        let copy_start = Instant::now();
        let futures: Vec<_> = mmap_tasks
            .into_iter()
            .map(|task| {
                let p = progress.clone();
                async move {
                    p.inc(task.length as u64);
                    task.run()
                }
            })
            .collect();

        let results = join_all(futures).await;
        for result in results {
            result?;
        }
        progress.finish();

        println!("Memory copy operations took {:?}", copy_start.elapsed());

        // Flush all target mmaps
        let flush_start = Instant::now();
        for mmap in mmap_pool.target_mmaps.values() {
            let mmap_guard = mmap.lock().unwrap();
            mmap_guard.flush()?;
        }
        println!("Flush took {:?}", flush_start.elapsed());

        Ok(())
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

    fn create_tasks(&self) -> Result<Vec<WriteTask>> {
        let mut tasks = vec![];
        for (name, target_tensor) in self.target.topology.tensors() {
            let source_tensor = self
                .source
                .topology
                .tensors()
                .get(name)
                .expect("Both topology should contain the same tensors");

            match (source_tensor, target_tensor) {
                (Tensor::Distributed(sinfo), Tensor::Distributed(tinfo)) => {
                    assert_eq!(sinfo.shape(), tinfo.shape());
                    assert_eq!(sinfo.dtype(), tinfo.dtype());
                    let dtype_size = sinfo.dtype().size();
                    let ndim = sinfo.shape().len();
                    let full_shape = sinfo.shape();
                    let mut full_strides = vec![1; ndim];
                    for i in (0..ndim - 1).rev() {
                        full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
                    }
                    for schunk in sinfo.chunks() {
                        for tchunk in tinfo.chunks() {
                            let source_intervals = get_intervals(schunk, &full_strides, full_shape);
                            let target_intervals = get_intervals(tchunk, &full_strides, full_shape);
                            let sindex = schunk.filename_index();
                            let tindex = tchunk.filename_index();
                            let sheader_size = self.source.metadatas[tindex].0;
                            let theader_size = self.target.metadatas[tindex].0;
                            let sdata_offset = self.source.metadatas[tindex]
                                .1
                                .tensors()
                                .get(name)
                                .expect("Tensor missing from metadata")
                                .data_offsets
                                .0;
                            let tdata_offset = self.target.metadatas[tindex]
                                .1
                                .tensors()
                                .get(name)
                                .expect("Tensor missing from metadata")
                                .data_offsets
                                .0;
                            self.tasks_from_interval(
                                &source_intervals,
                                &target_intervals,
                                sindex,
                                tindex,
                                sheader_size,
                                theader_size,
                                sdata_offset,
                                tdata_offset,
                                dtype_size,
                                &mut tasks,
                            );
                        }
                    }
                }
                (Tensor::Shared(sinfo), Tensor::Distributed(tinfo)) => {
                    assert_eq!(sinfo.shape(), tinfo.shape());
                    assert_eq!(sinfo.dtype(), tinfo.dtype());
                    let dtype_size = sinfo.dtype().size();
                    let ndim = sinfo.shape().len();
                    let full_shape = sinfo.shape();
                    let mut full_strides = vec![1; ndim];
                    for i in (0..ndim - 1).rev() {
                        full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
                    }
                    let n: usize = sinfo.shape().iter().product();
                    let source_intervals = vec![(0, n)];
                    let sindex = sinfo.filename_indices()[0];
                    let sheader_size = self.source.metadatas[sindex].0;
                    let sdata_offset = self.source.metadatas[sindex]
                        .1
                        .tensors()
                        .get(name)
                        .expect("Tensor missing from metadata")
                        .data_offsets
                        .0;
                    for tchunk in tinfo.chunks() {
                        let target_intervals = get_intervals(tchunk, &full_strides, full_shape);
                        let tindex = tchunk.filename_index();
                        let theader_size = self.target.metadatas[tindex].0;
                        let tdata_offset = self.target.metadatas[tindex]
                            .1
                            .tensors()
                            .get(name)
                            .expect("Tensor missing from metadata")
                            .data_offsets
                            .0;
                        self.tasks_from_interval(
                            &source_intervals,
                            &target_intervals,
                            sindex,
                            tindex,
                            sheader_size,
                            theader_size,
                            sdata_offset,
                            tdata_offset,
                            dtype_size,
                            &mut tasks,
                        );
                    }
                }
                (Tensor::Shared(sinfo), Tensor::Shared(tinfo)) => {
                    assert_eq!(sinfo.shape(), tinfo.shape());
                    assert_eq!(sinfo.dtype(), tinfo.dtype());
                    let dtype_size = sinfo.dtype().size();
                    let ndim = sinfo.shape().len();
                    let full_shape = sinfo.shape();
                    let mut full_strides = vec![1; ndim];
                    for i in (0..ndim - 1).rev() {
                        full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
                    }
                    let n: usize = sinfo.shape().iter().product();
                    let source_intervals = vec![(0, n)];
                    let sindex = sinfo.filename_indices()[0];
                    let sheader_size = self.source.metadatas[sindex].0;
                    let sdata_offset = self.source.metadatas[sindex]
                        .1
                        .tensors()
                        .get(name)
                        .expect("Tensor missing from metadata")
                        .data_offsets
                        .0;
                    let target_intervals = vec![(0, n)];
                    for &tindex in tinfo.filename_indices() {
                        let theader_size = self.target.metadatas[tindex].0;
                        let tdata_offset = self.target.metadatas[tindex]
                            .1
                            .tensors()
                            .get(name)
                            .expect("Tensor missing from metadata")
                            .data_offsets
                            .0;
                        self.tasks_from_interval(
                            &source_intervals,
                            &target_intervals,
                            sindex,
                            tindex,
                            sheader_size,
                            theader_size,
                            sdata_offset,
                            tdata_offset,
                            dtype_size,
                            &mut tasks,
                        );
                    }
                }
                (Tensor::Distributed(sinfo), Tensor::Shared(tinfo)) => {
                    assert_eq!(sinfo.shape(), tinfo.shape());
                    assert_eq!(sinfo.dtype(), tinfo.dtype());
                    let dtype_size = sinfo.dtype().size();
                    let ndim = sinfo.shape().len();
                    let full_shape = sinfo.shape();
                    let mut full_strides = vec![1; ndim];
                    for i in (0..ndim - 1).rev() {
                        full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
                    }
                    let n: usize = sinfo.shape().iter().product();

                    let target_intervals = vec![(0, n)];
                    for &tindex in tinfo.filename_indices() {
                        let theader_size = self.target.metadatas[tindex].0;
                        let tdata_offset = self.target.metadatas[tindex]
                            .1
                            .tensors()
                            .get(name)
                            .expect("Tensor missing from metadata")
                            .data_offsets
                            .0;
                        for schunk in sinfo.chunks() {
                            let source_intervals = get_intervals(schunk, &full_strides, full_shape);
                            let sindex = schunk.filename_index();
                            let sheader_size = self.source.metadatas[sindex].0;
                            let sdata_offset = self.source.metadatas[sindex]
                                .1
                                .tensors()
                                .get(name)
                                .expect("Tensor missing from metadata")
                                .data_offsets
                                .0;
                            self.tasks_from_interval(
                                &source_intervals,
                                &target_intervals,
                                sindex,
                                tindex,
                                sheader_size,
                                theader_size,
                                sdata_offset,
                                tdata_offset,
                                dtype_size,
                                &mut tasks,
                            );
                        }
                    }
                }
            }
        }
        Ok(tasks)
    }

    fn tasks_from_interval(
        &self,
        source_intervals: &[(usize, usize)],
        target_intervals: &[(usize, usize)],
        sindex: usize,
        tindex: usize,
        sheader_size: usize,
        theader_size: usize,
        sdata_offset: usize,
        tdata_offset: usize,
        dtype_size: usize,
        tasks: &mut Vec<WriteTask>,
    ) {
        for (sstart, tstart, length) in intersection(&source_intervals, &target_intervals) {
            // Convert from offset to bytes
            let sstart = sstart * dtype_size;
            let tstart = tstart * dtype_size;
            let length = length * dtype_size;

            let target_start = (theader_size + tdata_offset + tstart) as u64;
            let source_start = (sheader_size + sdata_offset + sstart) as u64;
            let target_filename = self
                .target
                .dir
                .join(&self.target.topology.filenames()[tindex]);
            let source_filename = self
                .source
                .dir
                .join(&self.source.topology.filenames()[sindex]);
            tasks.push(WriteTask {
                source_filename,
                source_start,
                target_filename,
                target_start,
                length,
            });
        }
    }

    /// Create all files with headers
    async fn create_files_with_headers(&self) -> Result<()> {
        assert_eq!(
            self.target.topology.filenames().len(),
            self.target.metadatas.len()
        );
        let file_creation_futures: Vec<_> = self
            .target
            .metadatas
            .iter()
            .zip(self.target.topology.filenames())
            .map(|((_, metadata), filename)| async move {
                let data_len = metadata.validate()?;
                let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
                // Force alignment to 8 bytes.
                let extra = (8 - metadata_buf.len() % 8) % 8;
                metadata_buf.extend(vec![b' '; extra]);

                let n: u64 = metadata_buf.len() as u64;

                let mut f = File::options()
                    .write(true)
                    .create(true)
                    .open(self.target.dir.join(filename))
                    .await?;
                f.write_all(&n.to_le_bytes()).await?;
                f.write_all(&metadata_buf).await?;
                let total = data_len + metadata_buf.len() + 8;
                f.set_len(total as u64).await?;
                // println!("File {filename:?} should have size {total}");
                f.flush().await?;

                Ok(())
            })
            .collect();

        // Wait for all files to be created
        let creation_results: Vec<Result<_>> = join_all(file_creation_futures).await;
        for result in creation_results {
            result?;
        }

        Ok(())
    }
}

/// Read safetensors file efficiently using memory mapping
fn safetensors_metadata<P: AsRef<Path>>(file_path: P) -> Result<(usize, Metadata)> {
    let file_path = file_path.as_ref().to_path_buf();
    let file = std::fs::File::open(&file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let (n, metadata) = SafeTensors::read_metadata(&mmap)?;
    // n is the size of the header, but we need to account for the first 8 bytes.
    Ok((n + 8, metadata))
}

/// Load topology from directory, or create a single-rank topology if topology.json doesn't exist
pub fn load_or_create_topology<P: AsRef<Path>>(dir: P) -> Result<Topology> {
    let dir = dir.as_ref();
    let topology_path = dir.join("topology.json");
    let model_path = dir.join("model.safetensors");
    let index_path = dir.join("model.safetensors.index.json");

    // Check if we have a distributed setup (topology.json + rank*.safetensors)
    if topology_path.exists() {
        // Load existing distributed topology
        let topology_data = std::fs::read_to_string(&topology_path)?;
        let topology: Topology = serde_json::from_str(&topology_data)?;
        return Ok(topology);
    }
    let filenames = if index_path.exists() {
        // Chunked safetensors case - read the index file to get tensor information
        let index_data = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_data)?;

        // Group tensors by their chunk file
        let mut filenames: HashSet<String> = HashSet::new();
        for (_tensor_name, file_name) in &index.weight_map {
            filenames.insert(file_name.clone());
        }
        let mut filenames = filenames.into_iter().collect::<Vec<_>>();
        filenames.sort();
        filenames
    } else if model_path.exists() {
        vec!["model.safetensors".to_string()]
    } else {
        return Err(RedistributorError::NoValidInput {
            path: dir.to_path_buf(),
        });
    };
    // Create a topology with a single rank
    let mut tensors = BTreeMap::new();

    // Read each chunk file to get tensor information
    for (file_index, file_name) in filenames.iter().enumerate() {
        let file_path = dir.join(&file_name);
        let (_, safetensors) = safetensors_metadata(&file_path)?;

        for (tensor_name, tensor_info) in safetensors.tensors() {
            tensors.insert(
                tensor_name,
                Tensor::Shared(SharedInfo::new(
                    tensor_info.shape.to_vec(),
                    tensor_info.dtype,
                    vec![file_index], // Use the correct file index
                )),
            );
        }
    }

    // Create topology with all chunk files
    let topology = Topology::new(tensors, filenames, 1)?;
    Ok(topology)
}

fn intersection(
    source_intervals: &[(usize, usize)],
    target_intervals: &[(usize, usize)],
) -> Vec<(usize, usize, usize)> {
    let mut result = Vec::new();

    let mut soffset = 0;
    let mut toffset = 0;
    let mut sindex = 0;
    let mut tindex = 0;

    while sindex < source_intervals.len() && tindex < target_intervals.len() {
        let (source_start, source_end) = &source_intervals[sindex];
        let (target_start, target_end) = &target_intervals[tindex];
        let intersection_start = (*source_start).max(*target_start);
        let intersection_end = (*source_end).min(*target_end);

        if intersection_start < intersection_end {
            // There is an overlap
            let source_offset = soffset + (intersection_start - source_start);
            let target_offset = toffset + (intersection_start - target_start);
            let length = intersection_end - intersection_start;

            result.push((source_offset, target_offset, length));
        }

        if source_end < target_end {
            sindex += 1;
            soffset += source_end - source_start;
        } else {
            tindex += 1;
            toffset += target_end - target_start;
        }
    }

    result
}

// Memory map pool for managing file mappings
struct MmapPool {
    source_mmaps: HashMap<PathBuf, Arc<Mutex<Mmap>>>,
    target_mmaps: HashMap<PathBuf, Arc<Mutex<MmapMut>>>,
}

impl MmapPool {
    fn new() -> Self {
        Self {
            source_mmaps: HashMap::new(),
            target_mmaps: HashMap::new(),
        }
    }

    fn open_source_files(&mut self, topology: &Topology, dir: &Path) -> Result<()> {
        for filename in topology.filenames() {
            let filepath = dir.join(filename);
            let file = std::fs::File::open(&filepath)?;
            let mmap = unsafe { Mmap::map(&file)? };
            self.source_mmaps
                .insert(filepath, Arc::new(Mutex::new(mmap)));
        }
        Ok(())
    }

    fn open_target_files(&mut self, topology: &Topology, dir: &Path) -> Result<()> {
        for filename in topology.filenames() {
            let filepath = dir.join(filename);
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&filepath)?;
            let mmap = unsafe { MmapMut::map_mut(&file)? };
            self.target_mmaps
                .insert(filepath, Arc::new(Mutex::new(mmap)));
        }
        Ok(())
    }

    fn get_source_mmap(&self, path: &Path) -> Option<Arc<Mutex<Mmap>>> {
        self.source_mmaps.get(path).cloned()
    }

    fn get_target_mmap(&self, path: &Path) -> Option<Arc<Mutex<MmapMut>>> {
        self.target_mmaps.get(path).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
