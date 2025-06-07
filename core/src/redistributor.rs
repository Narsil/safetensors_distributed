use crate::tensor::TensorData;
use crate::topology::{
    Chunk, DistributedInfo, SharedInfo, Tensor, Topology, TopologyError, get_intervals,
};
use futures::future::join_all;
use indicatif::style::TemplateError;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use safetensors::tensor::{Metadata, TensorInfo};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::{Semaphore, TryAcquireError};

struct WriteTask {
    source_filename: PathBuf,
    source_start: u64,
    target_filename: PathBuf,
    target_start: u64,
    length: usize,
    semaphore: Arc<Semaphore>,
}

impl WriteTask {
    async fn run(&self) -> Result<()> {
        let _permit = self.semaphore.acquire().await?;
        let mut f = File::open(&self.source_filename).await?;
        f.seek(SeekFrom::Start(self.source_start)).await?;
        let mut buf = Vec::with_capacity(self.length);
        f.take(self.length as u64).read(&mut buf).await?;

        let mut f = File::options()
            .write(true)
            .open(&self.target_filename)
            .await?;
        f.seek(SeekFrom::Start(self.target_start)).await?;
        f.write(&buf).await?;
        f.flush().await?;
        println!(
            "Writing from {:?} to {:?}",
            self.source_filename, self.target_filename
        );
        println!("  Source: {}", self.source_start);
        println!("  Target: {}", self.target_start);
        println!("  Size: {}", self.length);
        Ok(())
    }
}

/// Structure for deserializing model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    /// Map of tensor names to their containing file
    weight_map: HashMap<String, String>,
}

fn human_prefix(mut value: u64, unit: &str) -> String {
    for prefix in ["k", "M", "G"] {
        if value > 1000 {
            value /= 1000;
            continue;
        }
        return format!("{value} {prefix}{unit}");
    }
    todo!("Implement more of this");
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

    #[error("Tokio join error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),

    #[error("Semaphore acquire error: {0}")]
    SemaphoreAcquire(#[from] tokio::sync::AcquireError),

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

    #[error("Template error: {0}")]
    Semaphore(#[from] TryAcquireError),
}

/// Result type for redistributor operations
type Result<T> = std::result::Result<T, RedistributorError>;

/// Represents a tensor processing and writing task
#[derive(Debug, Clone)]
pub struct TensorTask {
    tensor_name: String,
    rank: usize,
    file_offset: usize,
    chunk_size: usize,
}

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
    // source.dir: PathBuf,
    // source.topology: Topology,
    // source.metadatas: Vec<Metadata>,
    // target.dir: PathBuf,
    // target.topology: Topology,
    // target.metadatas: Vec<Metadata>,
    file_semaphore: Arc<Semaphore>,
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

        let max_concurrent_files = 100;
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
        Ok(Self {
            source,
            target,
            file_semaphore: Arc::new(Semaphore::new(max_concurrent_files)),
        })
    }

    /// Redistribute tensors to target directory and return list of created files
    pub async fn redistribute(&self) -> Result<Vec<String>> {
        tokio::fs::create_dir_all(&self.target.dir).await?;

        self.create_files_with_headers().await?;

        let write_tasks = self.create_tasks()?;
        let futures: Vec<_> = write_tasks
            .into_iter()
            .map(|t| async move { t.run().await })
            .collect();

        join_all(futures).await;

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
        for (target_name, target_tensor) in self.target.topology.tensors() {
            match target_tensor {
                Tensor::Distributed(info) => {
                    let ndim = info.shape().len();
                    let full_shape = info.shape();
                    let mut full_strides = vec![1; ndim];
                    for i in (0..ndim - 1).rev() {
                        full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
                    }
                    for chunk in info.chunks() {
                        tasks.extend(self.distributed_task(
                            target_name,
                            chunk,
                            &full_strides,
                            full_shape,
                        ));
                    }
                }
                Tensor::Shared(info) => {
                    todo!("{info:?}");
                }
            }
        }
        Ok(tasks)
    }

    fn distributed_task(
        &self,
        name: &str,
        target_chunk: &Chunk,
        full_strides: &[usize],
        full_shape: &[usize],
    ) -> Vec<WriteTask> {
        let mut tasks = vec![];
        let source_tensor = self
            .source
            .topology
            .tensors()
            .get(name)
            .expect("Both topology should contain the same tensors");
        let target_intervals = get_intervals(target_chunk, full_strides, full_shape);
        let tindex = target_chunk.filename_index();
        let theader_size = self.target.metadatas[tindex].0;
        let tdata_offset = self.target.metadatas[tindex]
            .1
            .tensors()
            .get(name)
            .expect("Tensor missing from metadata")
            .data_offsets
            .0;
        match source_tensor {
            Tensor::Distributed(sinfo) => {
                assert_eq!(sinfo.shape(), full_shape);
                for source_chunk in sinfo.chunks() {
                    let source_intervals = get_intervals(source_chunk, full_strides, full_shape);

                    for (sstart, tstart, length) in
                        intersection(&source_intervals, &target_intervals)
                    {
                        let sindex = source_chunk.filename_index();
                        let sheader_size = self.source.metadatas[sindex].0;
                        let sdata_offset = self.source.metadatas[sindex]
                            .1
                            .tensors()
                            .get(name)
                            .expect("Tensor missing from metadata")
                            .data_offsets
                            .0;

                        let target_start = (theader_size + tdata_offset + tstart) as u64;
                        let source_start = (sheader_size + sdata_offset + sstart) as u64;
                        let semaphore = Arc::clone(&self.file_semaphore);
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
                            semaphore,
                            length,
                        });
                    }
                }
            }
            Tensor::Shared(sinfo) => {
                let mut tstart = 0;
                for (sstart, sstop) in target_intervals {
                    let length = sstop - sstart;

                    let sindex = sinfo.filename_index();
                    let sheader_size = self.source.metadatas[sindex].0;
                    let sdata_offset = self.source.metadatas[sindex]
                        .1
                        .tensors()
                        .get(name)
                        .expect("Tensor missing from metadata")
                        .data_offsets
                        .0;

                    let target_start = (theader_size + tdata_offset + tstart) as u64;
                    let source_start = (sheader_size + sdata_offset + sstart) as u64;
                    println!("Tensor {name} {sstart} {theader_size} {tdata_offset}");
                    println!("T {target_start} S {source_start}");
                    println!("T {theader_size} O {tdata_offset} S {tstart}");

                    let semaphore = Arc::clone(&self.file_semaphore);
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
                        semaphore,
                        length,
                    });
                    tstart += length;
                }
            }
        }
        tasks
    }

    fn to_shared_task(&self, rank: usize, name: &str, info: &SharedInfo) -> Vec<WriteTask> {
        let source_tensor = self
            .source
            .topology
            .tensors()
            .get(name)
            .expect("tensor is missing in source");
        match source_tensor {
            Tensor::Distributed(_) => todo!("distributed"),
            Tensor::Shared(source_info) => {
                assert_eq!(info.shape(), source_info.shape());
                assert_eq!(info.dtype(), source_info.dtype());

                let length = info.shape().iter().product::<usize>() * info.dtype().size();
                let source_filename = self
                    .source
                    .dir
                    .join(&self.source.topology.filenames()[source_info.filename_index()]);
                let target_filename = self
                    .target
                    .dir
                    .join(&self.target.topology.filenames()[info.filename_index()]);
                let (target_offset, tmetadata) = &self.target.metadatas[info.filename_index()];
                let (tstart, tstop) = tmetadata
                    .tensors()
                    .get(name)
                    .expect("Tensor should exist")
                    .data_offsets;
                assert_eq!(tstop - tstart, length);
                let (source_offset, smetadata) =
                    &self.source.metadatas[source_info.filename_index()];
                let (sstart, sstop) = smetadata
                    .tensors()
                    .get(name)
                    .expect("Tensor should exist")
                    .data_offsets;
                assert_eq!(tstop - tstart, length);
                assert_eq!(sstop - sstart, length);
                let target_start = tstart + target_offset;
                let source_start = sstart + source_offset;

                vec![WriteTask {
                    source_filename,
                    source_start: source_start as u64,
                    target_filename,
                    target_start: target_start as u64,
                    length,
                    semaphore: Arc::clone(&self.file_semaphore),
                }]
            }
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
                let _permit = self.file_semaphore.acquire().await?;

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
                println!("File {filename:?} should have size {total}");
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
    let mut tensors = HashMap::new();

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
                    file_index, // Use the correct file index
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
    vec![]
}
