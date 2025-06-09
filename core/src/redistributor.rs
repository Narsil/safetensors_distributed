use crate::topology::{SharedInfo, Tensor, Topology, TopologyError, get_intervals};
use futures::future::join_all;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle, style::TemplateError};
use log::{error, trace};
use memmap2::{Mmap, MmapMut};
use reqwest::header::RANGE;
use reqwest::{Client, header::HeaderMap};
use safetensors::SafeTensors;
use safetensors::tensor::{Metadata, TensorInfo};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tokio::sync::mpsc::{Sender, channel};
use tokio::task::JoinError;
use url::Url;

/// Source data abstraction for tasks
enum TaskSources {
    Local(Vec<Arc<Mmap>>),
    Remote {
        client: Client,
        base_url: Url,
        auth_headers: HeaderMap,
        file_paths: Vec<String>,
        http_semaphore: Arc<Semaphore>, // Limit concurrent HTTP requests
    },
}

impl TaskSources {
    /// Read data from the specified file index and range
    async fn read(&self, file_index: usize, start: u64, end: u64) -> Result<Cow<'_, [u8]>> {
        match self {
            TaskSources::Local(mmaps) => {
                let mmap = &mmaps[file_index];
                Ok(Cow::Borrowed(&mmap[start as usize..end as usize]))
            }
            TaskSources::Remote {
                client,
                base_url,
                auth_headers,
                file_paths,
                http_semaphore,
            } => {
                // Acquire semaphore permit to limit concurrent HTTP requests
                let _permit = http_semaphore.acquire().await.map_err(|_| {
                    RedistributorError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to acquire HTTP semaphore permit",
                    ))
                })?;

                let url = base_url.join(&file_paths[file_index]).map_err(|e| {
                    RedistributorError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Invalid URL: {}", e),
                    ))
                })?;

                let range_header = format!("bytes={}-{}", start, end - 1);
                let expected_size = (end - start) as usize;

                trace!("Fetching range {} from {}", range_header, url);

                let response = client
                    .get(url.clone())
                    .headers(auth_headers.clone())
                    .header(RANGE, range_header.clone())
                    .timeout(Duration::from_secs(30)) // 30 second timeout
                    .send()
                    .await
                    .map_err(|e| {
                        RedistributorError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("HTTP request failed for {}: {}", url, e),
                        ))
                    })?;

                if !response.status().is_success() {
                    return Err(RedistributorError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "HTTP request failed with status {} for {}",
                            response.status(),
                            url
                        ),
                    )));
                }

                // Stream the response chunk by chunk to avoid loading everything into memory
                let mut stream = response.bytes_stream();
                let mut buffer = Vec::with_capacity(expected_size);

                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| {
                        RedistributorError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to read response chunk from {}: {}", url, e),
                        ))
                    })?;

                    buffer.extend_from_slice(&chunk);

                    // Safety check: if we're getting way more data than expected, stop
                    if buffer.len() > expected_size * 2 {
                        return Err(RedistributorError::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!(
                                "Response size {} exceeds expected {} for range {}",
                                buffer.len(),
                                expected_size,
                                range_header
                            ),
                        )));
                    }
                }

                trace!(
                    "Successfully fetched {} bytes (expected {})",
                    buffer.len(),
                    expected_size
                );
                Ok(Cow::Owned(buffer))
                // _permit automatically dropped here, releasing semaphore
            }
        }
    }
}

/// Builder for collecting task sources during task creation
enum TaskSource {
    Local {
        mmaps: Vec<Arc<Mmap>>,
    },
    Remote {
        client: Client,
        base_url: Url,
        auth_headers: HeaderMap,
        file_paths: Vec<String>,
        http_semaphore: Arc<Semaphore>,
    },
}

impl TaskSource {
    /// Create a new empty task source builder for local sources
    fn new_local() -> Self {
        Self::Local { mmaps: Vec::new() }
    }

    /// Create a new empty task source builder for remote sources
    fn new_remote(
        client: Client,
        base_url: Url,
        auth_headers: HeaderMap,
        http_semaphore: Arc<Semaphore>,
    ) -> Self {
        Self::Remote {
            client,
            base_url,
            auth_headers,
            file_paths: Vec::new(),
            http_semaphore,
        }
    }

    /// Add a source from the given location at the specified file index
    fn add_from_location(&mut self, location: &SourceLocation, file_index: usize) {
        match (self, location) {
            (
                TaskSource::Local { mmaps },
                SourceLocation::Local {
                    mmaps: source_mmaps,
                },
            ) => {
                mmaps.push(Arc::clone(&source_mmaps[file_index]));
            }
            (
                TaskSource::Remote { file_paths, .. },
                SourceLocation::Remote {
                    file_paths: source_paths,
                    ..
                },
            ) => {
                file_paths.push(source_paths[file_index].clone());
            }
            _ => panic!("Mismatched task source and location types"),
        }
    }

    /// Get the number of source files in this task source
    fn len(&self) -> usize {
        match self {
            TaskSource::Local { mmaps } => mmaps.len(),
            TaskSource::Remote { file_paths, .. } => file_paths.len(),
        }
    }

    /// Convert into TaskSources for execution
    fn into_task_sources(self) -> TaskSources {
        match self {
            TaskSource::Local { mmaps } => TaskSources::Local(mmaps),
            TaskSource::Remote {
                client,
                base_url,
                auth_headers,
                file_paths,
                http_semaphore,
            } => TaskSources::Remote {
                client,
                base_url,
                auth_headers,
                file_paths,
                http_semaphore,
            },
        }
    }
}

// Memory-mapped task type for high performance file operations
struct MmapWriteTask {
    // Target write info - single contiguous region
    target_mmap: Arc<MmapMut>,
    target_start: u64,
    target_end: u64,
    target_file_index: usize,

    // Source read info - multiple reads from potentially multiple files
    source: TaskSources,
    source_ranges: Vec<(u64, u64, u64)>, // Flat list of (start, end, target_offset) byte ranges
    ranges_per_file: Vec<usize>,         // How many ranges belong to each source file
}

impl MmapWriteTask {
    async fn run(&self) -> Result<()> {
        let target_length = (self.target_end - self.target_start) as usize;
        trace!("Writing {target_length} to {}", self.target_file_index);

        // SAFETY: This is safe because:
        // 1. All write ranges are pre-calculated and guaranteed non-overlapping
        // 2. Each task writes to a unique byte range within the memory-mapped file
        // 3. No two tasks will ever write to the same memory location
        // 4. The topology calculation ensures mutually exclusive write regions
        unsafe {
            let target_ptr = self.target_mmap.as_ptr().add(self.target_start as usize) as *mut u8;
            let target_slice = std::slice::from_raw_parts_mut(target_ptr, target_length);

            let mut range_idx = 0usize;

            let source_file_count = match &self.source {
                TaskSources::Local(mmaps) => mmaps.len(),
                TaskSources::Remote { file_paths, .. } => file_paths.len(),
            };

            trace!(
                "  Executing {} source files, {} total ranges",
                source_file_count,
                self.source_ranges.len()
            );

            // Process each source file
            for (file_idx, &num_ranges) in self.ranges_per_file.iter().enumerate() {
                trace!(
                    "    Processing source file {}: {} ranges",
                    file_idx, num_ranges
                );

                // Process all ranges for this source file
                for range_num in 0..num_ranges {
                    let (source_start, source_end, target_offset) = self.source_ranges[range_idx];
                    let range_length = (source_end - source_start) as usize;

                    trace!(
                        "      Range {}: source {}→{} ({} bytes) -> target offset {}",
                        range_num, source_start, source_end, range_length, target_offset
                    );

                    // Read from source using abstracted method
                    let source_data = self.source.read(file_idx, source_start, source_end).await?;

                    // Write to target at specified offset (not sequential)
                    let target_offset_usize = target_offset as usize;
                    target_slice[target_offset_usize..target_offset_usize + range_length]
                        .copy_from_slice(&source_data);

                    range_idx += 1;
                }
            }
        }

        // Flush just the range we wrote asynchronously to avoid accumulating dirty pages
        if let Err(e) = self
            .target_mmap
            .flush_async_range(self.target_start as usize, target_length)
        {
            error!(
                "Async flush failed for range {}:{}: {}",
                self.target_start, self.target_end, e
            );
            // Don't fail the task, just log the error and continue
        }

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

    #[error("Join error: {0}")]
    Join(#[from] JoinError),
}

/// Result type for redistributor operations
type Result<T> = std::result::Result<T, RedistributorError>;

/// Data source for tensor redistribution
struct Layout {
    topology: Topology,
    metadatas: Vec<(usize, Metadata)>,
}

/// Source location for input files
enum SourceLocation {
    Local {
        mmaps: Vec<Arc<Mmap>>,
    },
    Remote {
        client: Client,
        base_url: Url,
        auth_headers: HeaderMap,
        file_paths: Vec<String>,
        http_semaphore: Arc<Semaphore>,
    },
}

/// Write location for target files
struct WriteLocation {
    dir: PathBuf,
    mmaps: Option<Vec<Arc<MmapMut>>>,
}

impl WriteLocation {
    /// Initialize target files and memory maps
    async fn init(&mut self, layout: &Layout) -> Result<()> {
        println!("INIT: Starting target location initialization...");

        // Create directory
        println!("INIT: Creating directory...");
        let dir_start = Instant::now();
        tokio::fs::create_dir_all(&self.dir).await?;
        println!("INIT: Directory created in {:?}", dir_start.elapsed());

        // Create files with headers
        println!("INIT: Creating files with headers...");
        let files_start = Instant::now();
        self.create_files_with_headers(layout).await?;
        println!("INIT: Files created in {:?}", files_start.elapsed());

        // Initialize memory maps
        println!("INIT: Initializing memory maps...");
        let mmap_start = Instant::now();
        self.init_target_mmaps(layout)?;
        println!(
            "INIT: Memory maps initialized in {:?}",
            mmap_start.elapsed()
        );

        println!("INIT: Target location initialization complete");
        Ok(())
    }

    /// Save/flush all memory-mapped files and write topology
    async fn save(&self, topology: &Topology) -> Result<()> {
        if let Some(ref target_mmaps) = self.mmaps {
            for target_mmap in target_mmaps {
                target_mmap.flush()?;
            }
        }
        self.write_topology(topology).await?;
        Ok(())
    }

    /// Create a write task for the specified file index and parameters
    fn create_write_task(
        &self,
        target_file_index: usize,
        target_start: u64,
        target_end: u64,
        source: TaskSources,
        source_ranges: Vec<(u64, u64, u64)>,
        ranges_per_file: Vec<usize>,
    ) -> Option<MmapWriteTask> {
        let target_mmap = Arc::clone(self.mmaps.as_ref()?.get(target_file_index)?);

        Some(MmapWriteTask {
            target_mmap,
            target_start,
            target_end,
            target_file_index,
            source,
            source_ranges,
            ranges_per_file,
        })
    }

    /// Write topology.json file if needed
    async fn write_topology(&self, topology: &Topology) -> Result<()> {
        if topology.world_size() > 1 {
            let topology_json = serde_json::to_vec_pretty(topology)?;
            tokio::fs::write(self.dir.join("topology.json"), &topology_json).await?;
        }
        Ok(())
    }

    /// Create all files with headers
    async fn create_files_with_headers(&self, layout: &Layout) -> Result<()> {
        assert_eq!(layout.topology.filenames().len(), layout.metadatas.len());
        let file_creation_futures: Vec<_> = layout
            .metadatas
            .iter()
            .zip(layout.topology.filenames())
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
                    .open(self.dir.join(filename))
                    .await?;
                f.write_all(&n.to_le_bytes()).await?;
                f.write_all(&metadata_buf).await?;
                let total = data_len + metadata_buf.len() + 8;
                f.set_len(total as u64).await?;
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

    /// Initialize target memory maps after files have been created
    fn init_target_mmaps(&mut self, layout: &Layout) -> Result<()> {
        if self.mmaps.is_some() {
            return Ok(()); // Already initialized
        }

        let target_mmaps: Result<Vec<_>> = layout
            .topology
            .filenames()
            .iter()
            .map(|filename| {
                let filepath = self.dir.join(filename);
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&filepath)?;
                unsafe { Ok(Arc::new(MmapMut::map_mut(&file)?)) }
            })
            .collect();

        self.mmaps = Some(target_mmaps?);
        Ok(())
    }
}

/// Source configuration
struct Source {
    layout: Layout,
    location: SourceLocation,
}

/// Target configuration
struct Target {
    layout: Layout,
    location: WriteLocation,
}

/// Async streaming tensor redistributor with parallel processing and pre-calculated offsets
pub struct AsyncTensorRedistributor {
    source: Source,
    target: Target,
}

impl AsyncTensorRedistributor {
    /// Create a new redistributor for reconstruction from local distributed files
    pub fn from_local<P: AsRef<Path>>(
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

    /// Create a new redistributor for reconstruction from remote distributed files
    pub async fn from_url<P: AsRef<Path>>(
        base_url: Url,
        auth_headers: HeaderMap,
        target_dir: P,
        target_topology: Topology,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60)) // Overall request timeout
            .connect_timeout(Duration::from_secs(10)) // Connection timeout
            .pool_idle_timeout(Duration::from_secs(90)) // Keep connections alive longer
            .pool_max_idle_per_host(32) // Allow more concurrent connections per host
            .http2_keep_alive_interval(Duration::from_secs(30)) // HTTP/2 keep-alive
            .http2_keep_alive_timeout(Duration::from_secs(10)) // HTTP/2 keep-alive timeout
            .http2_keep_alive_while_idle(true) // Keep connections alive even when idle
            .tcp_keepalive(Duration::from_secs(60)) // TCP-level keep-alive
            .redirect(reqwest::redirect::Policy::default()) // Follow redirects (up to 10)
            .build()
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create HTTP client: {}", e),
                ))
            })?;

        Self::from_url_with_client(client, base_url, auth_headers, target_dir, target_topology)
            .await
    }

    /// Create redistributor from remote URL with provided HTTP client (for connection pooling)
    pub async fn from_url_with_client<P: AsRef<Path>>(
        client: Client,
        base_url: Url,
        auth_headers: HeaderMap,
        target_dir: P,
        target_topology: Topology,
    ) -> Result<Self> {
        let target_dir = target_dir.as_ref().to_path_buf();
        let target_metadatas = Self::pre_calculate_metadatas(&target_topology)?;

        let target_layout = Layout {
            topology: target_topology,
            metadatas: target_metadatas,
        };

        // Create semaphore with 100 permits for HTTP request throttling
        let http_semaphore = Arc::new(Semaphore::new(100));

        // Discover remote topology and metadata WITH CACHING to avoid duplicate requests
        let (source_topology, metadata_cache) =
            Self::load_or_create_remote_topology_with_cache(&client, &base_url, &auth_headers)
                .await?;
        let file_paths = source_topology.filenames().to_vec();

        // Fetch metadata for each remote file, using cache when available
        let mut source_metadatas = Vec::new();
        for file_path in &file_paths {
            if let Some(cached_metadata) = metadata_cache.get(file_path) {
                // Use cached metadata (avoids redundant request)
                trace!("Using cached metadata for {}", file_path);
                let metadata_buf = serde_json::to_string(cached_metadata)?.into_bytes();
                let extra = (8 - metadata_buf.len() % 8) % 8;
                let header_size = metadata_buf.len() + 8 + extra;
                source_metadatas.push((header_size, cached_metadata.clone()));
            } else {
                // Fetch metadata if not cached
                let file_url = base_url.join(file_path).map_err(|e| {
                    RedistributorError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Invalid file URL: {}", e),
                    ))
                })?;

                let metadata =
                    Self::fetch_remote_safetensors_metadata(&client, &file_url, &auth_headers)
                        .await?;

                let metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
                let extra = (8 - metadata_buf.len() % 8) % 8;
                let header_size = metadata_buf.len() + 8 + extra;
                source_metadatas.push((header_size, metadata));
            }
        }

        let source_layout = Layout {
            topology: source_topology,
            metadatas: source_metadatas,
        };

        Ok(Self {
            source: Source {
                layout: source_layout,
                location: SourceLocation::Remote {
                    client,
                    base_url,
                    auth_headers,
                    file_paths: file_paths.to_vec(),
                    http_semaphore: http_semaphore.clone(),
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

    /// Load topology from remote URL, equivalent of load_or_create_topology for remote sources
    /// Returns (topology, metadata_cache) to avoid duplicate metadata fetching
    pub async fn load_or_create_remote_topology_with_cache(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<(Topology, HashMap<String, Metadata>)> {
        let mut metadata_cache = HashMap::new();

        // Optimized approach: Use HEAD requests to check file existence first to avoid unnecessary downloads
        let topology_url = base_url.join("topology.json").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid topology URL: {}", e),
            ))
        })?;

        let index_url = base_url.join("model.safetensors.index.json").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid index URL: {}", e),
            ))
        })?;

        let model_url = base_url.join("model.safetensors").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid model URL: {}", e),
            ))
        })?;

        // Use HEAD requests to quickly check which files exist
        let topology_exists = Self::check_file_exists(client, &topology_url, auth_headers).await;
        let index_exists = Self::check_file_exists(client, &index_url, auth_headers).await;
        let model_exists = Self::check_file_exists(client, &model_url, auth_headers).await;

        // Try distributed setup first (topology.json)
        if topology_exists {
            if let Ok(topology) =
                Self::try_fetch_topology_json(client, base_url, auth_headers).await
            {
                return Ok((topology, metadata_cache));
            }
        }

        // Try chunked setup (index.json)
        if index_exists {
            if let Ok(topology) = Self::try_fetch_index_json(client, base_url, auth_headers).await {
                return Ok((topology, metadata_cache));
            }
        }

        // Try single file setup (model.safetensors)
        if model_exists {
            if let Ok((topology, metadata)) =
                Self::try_fetch_model_safetensors_with_cache(client, base_url, auth_headers).await
            {
                // Cache the metadata we just fetched to avoid redundant request
                metadata_cache.insert("model.safetensors".to_string(), metadata);
                return Ok((topology, metadata_cache));
            }
        }

        Err(RedistributorError::NoValidInput {
            path: PathBuf::from(base_url.as_str()),
        })
    }

    /// Check if a file exists using a HEAD request (fast and lightweight)
    async fn check_file_exists(client: &Client, url: &Url, auth_headers: &HeaderMap) -> bool {
        match client
            .head(url.clone())
            .headers(auth_headers.clone())
            .timeout(Duration::from_secs(5)) // Short timeout for existence checks
            .send()
            .await
        {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    /// Load topology from remote URL, equivalent of load_or_create_topology for remote sources
    pub async fn load_or_create_remote_topology(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Topology> {
        let (topology, _) =
            Self::load_or_create_remote_topology_with_cache(client, base_url, auth_headers).await?;
        Ok(topology)
    }

    /// Try to fetch and parse topology.json
    async fn try_fetch_topology_json(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Topology> {
        let topology_url = base_url.join("topology.json").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid topology URL: {}", e),
            ))
        })?;

        let response = client
            .get(topology_url)
            .headers(auth_headers.clone())
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch topology.json: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "topology.json not found",
            )));
        }

        let topology_data = response.text().await.map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read topology.json: {}", e),
            ))
        })?;

        let topology: Topology = serde_json::from_str(&topology_data)?;
        Ok(topology)
    }

    /// Try to fetch and parse model.safetensors.index.json
    async fn try_fetch_index_json(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Topology> {
        let index_url = base_url.join("model.safetensors.index.json").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid index URL: {}", e),
            ))
        })?;

        let response = client
            .get(index_url)
            .headers(auth_headers.clone())
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch index.json: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "model.safetensors.index.json not found",
            )));
        }

        let index_data = response.text().await.map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read index.json: {}", e),
            ))
        })?;

        let index: SafetensorsIndex = serde_json::from_str(&index_data)?;

        // Group tensors by their chunk file and create topology
        let mut filenames: HashSet<String> = HashSet::new();
        for (_tensor_name, file_name) in &index.weight_map {
            filenames.insert(file_name.clone());
        }
        let mut filenames = filenames.into_iter().collect::<Vec<_>>();
        filenames.sort();

        // Create a topology with shared tensors (chunked model)
        let mut tensors = BTreeMap::new();

        // TODO: We need to fetch metadata from each file to get tensor shapes and dtypes
        // For now, create placeholder shared tensors
        for (tensor_name, _file_name) in &index.weight_map {
            tensors.insert(
                tensor_name.clone(),
                Tensor::Shared(SharedInfo::new(
                    vec![1],                 // Placeholder shape
                    safetensors::Dtype::F32, // Placeholder dtype
                    vec![0],                 // Placeholder file indices
                )),
            );
        }

        let topology = Topology::new(tensors, filenames, 1)?;
        Ok(topology)
    }

    /// Try to fetch and create topology from model.safetensors, returning metadata for caching
    async fn try_fetch_model_safetensors_with_cache(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<(Topology, Metadata)> {
        let model_url = base_url.join("model.safetensors").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid model URL: {}", e),
            ))
        })?;

        // Try to fetch the metadata from the file
        let metadata =
            Self::fetch_remote_safetensors_metadata(client, &model_url, auth_headers).await?;

        // Create a topology with shared tensors (single file model)
        let mut tensors = BTreeMap::new();
        for (tensor_name, tensor_info) in metadata.tensors() {
            tensors.insert(
                tensor_name,
                Tensor::Shared(SharedInfo::new(
                    tensor_info.shape.to_vec(),
                    tensor_info.dtype,
                    vec![0], // Single file at index 0
                )),
            );
        }

        let topology = Topology::new(tensors, vec!["model.safetensors".to_string()], 1)?;
        Ok((topology, metadata))
    }

    /// Fetch safetensors metadata from remote file using streaming
    async fn fetch_remote_safetensors_metadata(
        client: &Client,
        file_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Metadata> {
        // Make a single request and stream the response, cutting it early once we have the header
        let response = client
            .get(file_url.clone())
            .headers(auth_headers.clone())
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch from {}: {}", file_url, e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Failed to fetch from {}: status {}",
                    file_url,
                    response.status()
                ),
            )));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut header_length: Option<u64> = None;

        // Stream chunks and process them
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to read chunk: {}", e),
                ))
            })?;

            buffer.extend_from_slice(&chunk);

            // Step 1: Once we have at least 8 bytes, parse the header length
            if header_length.is_none() && buffer.len() >= 8 {
                let header_length_bytes: [u8; 8] = buffer[..8].try_into().map_err(|_| {
                    RedistributorError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Failed to convert header length bytes to array",
                    ))
                })?;

                header_length = Some(u64::from_le_bytes(header_length_bytes));
            }

            // Step 2: Once we know the header length, check if we have the complete header
            if let Some(len) = header_length {
                let total_needed = 8 + len as usize; // 8 bytes for length + header JSON

                if buffer.len() >= total_needed {
                    // We have all the data we need, extract the header JSON and stop streaming
                    let header_bytes = &buffer[8..total_needed];

                    // Step 3: Parse the JSON as safetensors metadata
                    let metadata: Metadata = serde_json::from_slice(header_bytes)
                        .map_err(|e| RedistributorError::Json(e))?;

                    return Ok(metadata);
                }
            }
        }

        // If we get here, the stream ended before we got all the data we needed
        Err(RedistributorError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "Stream ended before complete header was received",
        )))
    }

    /// Redistribute tensors to target directory and return list of created files
    pub async fn redistribute(&mut self) -> Result<Vec<String>> {
        trace!("Start");
        let start = Instant::now();
        println!("REDISTRIBUTION: Starting redistribute process...");

        println!("REDISTRIBUTION: Initializing target location...");
        let init_start = Instant::now();
        self.target.location.init(&self.target.layout).await?;
        println!(
            "REDISTRIBUTION: Target location initialized in {:?}",
            init_start.elapsed()
        );
        trace!("Created headers {:?}", start.elapsed());

        println!("REDISTRIBUTION: Setting up task channel...");
        let (tx, mut rx) = channel::<MmapWriteTask>(10_000);

        println!("REDISTRIBUTION: Calculating total size...");
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
        println!(
            "REDISTRIBUTION: Total size calculated: {} bytes in {:?}",
            total,
            calc_start.elapsed()
        );

        println!("REDISTRIBUTION: Setting up progress bar...");
        let progress = ProgressBar::new(total);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{eta_precise}] [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})")
                .unwrap(),
        );

        println!("REDISTRIBUTION: Starting progress task...");
        let p = progress.clone();
        let handle = tokio::spawn(async move {
            let mut batch = Vec::with_capacity(1024);

            loop {
                let received = rx.recv_many(&mut batch, 1024).await;
                if received == 0 {
                    break; // Channel closed
                }

                // Sort by target file index, then by target offset for sequential I/O
                batch.sort_by_key(|task| (task.target_file_index, task.target_start));

                // Execute batch in sorted order
                for task in batch.drain(..) {
                    let length = (task.target_end - task.target_start) as usize;
                    if let Err(e) = task.run().await {
                        error!("Write task failed: {}", e);
                        // Continue with other tasks rather than failing completely
                    }
                    p.inc(length as u64);
                }
            }
        });

        println!("REDISTRIBUTION: Creating tasks...");
        let tasks_start = Instant::now();
        self.create_tasks(&tx).await?;
        println!(
            "REDISTRIBUTION: Tasks created in {:?}",
            tasks_start.elapsed()
        );

        println!("REDISTRIBUTION: Starting task execution...");
        drop(tx);
        handle.await?;

        progress.finish();
        println!("Done write, waiting for the kernel to sync on device...");
        // Force flush all memory-mapped target files and write topology
        self.target
            .location
            .save(&self.target.layout.topology)
            .await?;

        trace!("Tasks done {:?}", start.elapsed());

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

    async fn create_tasks(&self, tx: &Sender<MmapWriteTask>) -> Result<()> {
        println!("TASKS: Starting task creation...");
        let start = Instant::now();

        trace!(
            "Creating tasks for {} target files",
            self.target.layout.topology.filenames().len()
        );

        let file_count = self.target.layout.topology.filenames().len();
        println!("TASKS: Processing {} target files", file_count);

        // Process each target file in order
        for (target_file_index, filename) in
            self.target.layout.topology.filenames().iter().enumerate()
        {
            let file_start = Instant::now();
            println!(
                "TASKS: Processing target file {}/{}: {}",
                target_file_index + 1,
                file_count,
                filename
            );

            trace!("Processing target file {}: {}", target_file_index, filename);

            // Get the metadata for this target file
            let (header_size, metadata) = &self.target.layout.metadatas[target_file_index];

            // Collect tensors and sort by data offset for sequential writes
            let tensors = metadata.tensors();
            let mut tensor_entries: Vec<(&String, &TensorInfo)> =
                tensors.iter().map(|(k, v)| (k, *v)).collect();
            tensor_entries.sort_by_key(|(_, tensor_info)| tensor_info.data_offsets.0);

            let tensor_count = tensor_entries.len();
            println!(
                "TASKS: Found {} tensors in target file {}",
                tensor_count, target_file_index
            );

            trace!(
                "  Found {} tensors in target file {}",
                tensor_entries.len(),
                target_file_index
            );

            // Process each tensor in write order
            for (i, (tensor_name, tensor_info)) in tensor_entries.iter().enumerate() {
                let tensor_start = Instant::now();
                if i < 5 || i % 20 == 0 || i == tensor_count - 1 {
                    println!(
                        "TASKS: Creating task for tensor {}/{}: '{}' (offset: {} -> {})",
                        i + 1,
                        tensor_count,
                        tensor_name,
                        tensor_info.data_offsets.0,
                        tensor_info.data_offsets.1
                    );
                }

                trace!(
                    "  Creating task for tensor '{}' (offset: {} -> {})",
                    tensor_name, tensor_info.data_offsets.0, tensor_info.data_offsets.1
                );

                self.create_task_for_tensor(
                    tx,
                    target_file_index,
                    *header_size,
                    tensor_name,
                    tensor_info,
                )
                .await?;

                if i < 5 || i % 20 == 0 || i == tensor_count - 1 {
                    println!(
                        "TASKS: Task created for tensor '{}' in {:?}",
                        tensor_name,
                        tensor_start.elapsed()
                    );
                }
            }

            println!(
                "TASKS: Completed target file {} in {:?}",
                target_file_index,
                file_start.elapsed()
            );
        }
        trace!("Finished creating tasks");
        println!("TASKS: All tasks created in {:?}", start.elapsed());
        Ok(())
    }

    async fn create_task_for_tensor(
        &self,
        tx: &Sender<MmapWriteTask>,
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
        let mut task_source = match &self.source.location {
            SourceLocation::Local { .. } => TaskSource::new_local(),
            SourceLocation::Remote {
                client,
                base_url,
                auth_headers,
                http_semaphore,
                ..
            } => TaskSource::new_remote(
                client.clone(),
                base_url.clone(),
                auth_headers.clone(),
                http_semaphore.clone(),
            ),
        };
        let mut source_ranges = Vec::new();
        let mut ranges_per_file = Vec::new();

        match (source_tensor, target_tensor) {
            (Tensor::Distributed(source_info), Tensor::Distributed(target_info)) => {
                trace!("    Pattern: Distributed -> Distributed");
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
                trace!("    Pattern: Shared -> Distributed");
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
                trace!("    Pattern: Shared -> Shared");
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
                trace!("    Pattern: Distributed -> Shared");
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

            trace!(
                "    Task created: {} source files, {} total ranges, writing {}->{}",
                source_count,
                source_ranges.len(),
                target_start,
                target_end
            );

            if let Some(task) = self.target.location.create_write_task(
                target_file_index,
                target_start,
                target_end,
                task_source.into_task_sources(),
                source_ranges,
                ranges_per_file,
            ) {
                tx.send(task).await.unwrap();
            }
        } else {
            trace!("    No task created (no source ranges)");
        }

        Ok(())
    }

    fn collect_reads_distributed_to_distributed(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::DistributedInfo,
        target_info: &crate::topology::DistributedInfo,
        target_file_index: usize,
        task_source: &mut TaskSource,
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

        // Get the intervals (element ranges) for this target chunk
        let target_intervals = get_intervals(target_chunk, &full_strides, full_shape);

        // Process each source chunk to see if it overlaps with our target
        for source_chunk in source_info.chunks() {
            let source_intervals = get_intervals(source_chunk, &full_strides, full_shape);

            // Find intersection between source and target intervals
            let intersections = intersection(&source_intervals, &target_intervals);

            if !intersections.is_empty() {
                let source_file_index = source_chunk.filename_index();

                // Get source file metadata to calculate byte offsets
                let (source_header_size, source_metadata) =
                    &self.source.layout.metadatas[source_file_index];
                let source_tensors = source_metadata.tensors();
                let source_tensor_info = source_tensors.get(tensor_name).ok_or_else(|| {
                    RedistributorError::TensorNotFound {
                        name: tensor_name.to_string(),
                    }
                })?;
                let source_data_offset = source_tensor_info.data_offsets.0;

                // Convert intersection results to byte ranges
                let mut ranges_for_this_file = 0;
                for (source_offset, target_offset, length) in intersections {
                    let start_bytes = source_offset * dtype_size;
                    let end_bytes = (source_offset + length) * dtype_size;
                    let target_offset_bytes = target_offset * dtype_size;

                    let source_start =
                        (source_header_size + source_data_offset + start_bytes) as u64;
                    let source_end = (source_header_size + source_data_offset + end_bytes) as u64;

                    source_ranges.push((source_start, source_end, target_offset_bytes as u64));
                    ranges_for_this_file += 1;
                }

                // Add the source mmap and record how many ranges belong to this file
                if ranges_for_this_file > 0 {
                    trace!(
                        "      Added {} ranges from source file {}",
                        ranges_for_this_file, source_file_index
                    );
                    task_source.add_from_location(&self.source.location, source_file_index);
                    ranges_per_file.push(ranges_for_this_file);
                }
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
        task_source: &mut TaskSource,
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

        // Get the intervals (element ranges) for this target chunk
        let target_intervals = get_intervals(target_chunk, &full_strides, full_shape);

        // Pick the first source file that contains this tensor (they're all identical for shared tensors)
        let source_file_index = source_info.filename_indices()[0];

        // Get source file metadata to calculate byte offsets
        let (source_header_size, source_metadata) =
            &self.source.layout.metadatas[source_file_index];
        let source_tensors = source_metadata.tensors();
        let source_tensor_info =
            source_tensors
                .get(tensor_name)
                .ok_or_else(|| RedistributorError::TensorNotFound {
                    name: tensor_name.to_string(),
                })?;
        let source_data_offset = source_tensor_info.data_offsets.0;

        // Convert element intervals to byte ranges
        let mut ranges_for_this_file = 0;
        let mut target_offset_elements = 0;
        for (start_elem, end_elem) in target_intervals {
            let start_bytes = start_elem * dtype_size;
            let end_bytes = end_elem * dtype_size;
            let target_offset_bytes = target_offset_elements * dtype_size;

            let source_start = (source_header_size + source_data_offset + start_bytes) as u64;
            let source_end = (source_header_size + source_data_offset + end_bytes) as u64;

            source_ranges.push((source_start, source_end, target_offset_bytes as u64));
            ranges_for_this_file += 1;
            target_offset_elements += end_elem - start_elem;
        }

        // Add the source mmap and record how many ranges belong to this file
        if ranges_for_this_file > 0 {
            trace!(
                "      Added {} ranges from source file {}",
                ranges_for_this_file, source_file_index
            );
            task_source.add_from_location(&self.source.location, source_file_index);
            ranges_per_file.push(ranges_for_this_file);
        }

        Ok(())
    }

    fn collect_reads_shared_to_shared(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::SharedInfo,
        target_info: &crate::topology::SharedInfo,
        target_file_index: usize,
        task_source: &mut TaskSource,
        source_ranges: &mut Vec<(u64, u64, u64)>,
        ranges_per_file: &mut Vec<usize>,
    ) -> Result<()> {
        // Check if this target file should contain this shared tensor
        if !target_info.filename_indices().contains(&target_file_index) {
            // This target file doesn't need this tensor
            return Ok(());
        }

        // Pick the first source file that contains this tensor (they're all identical for shared tensors)
        let source_file_index = source_info.filename_indices()[0];

        // Get source file metadata to calculate byte offsets
        let (source_header_size, source_metadata) =
            &self.source.layout.metadatas[source_file_index];
        let source_tensors = source_metadata.tensors();
        let source_tensor_info =
            source_tensors
                .get(tensor_name)
                .ok_or_else(|| RedistributorError::TensorNotFound {
                    name: tensor_name.to_string(),
                })?;
        let source_data_offset = source_tensor_info.data_offsets.0;

        // For shared to shared, we just copy the entire tensor
        let tensor_size_bytes =
            source_tensor_info.data_offsets.1 - source_tensor_info.data_offsets.0;

        let source_start = (source_header_size + source_data_offset) as u64;
        let source_end = (source_header_size + source_data_offset + tensor_size_bytes) as u64;

        source_ranges.push((source_start, source_end, 0));
        task_source.add_from_location(&self.source.location, source_file_index);
        ranges_per_file.push(1); // Single range for the complete tensor

        trace!(
            "      Added 1 range from source file {} (full tensor copy)",
            source_file_index
        );

        Ok(())
    }

    fn collect_reads_distributed_to_shared(
        &self,
        tensor_name: &str,
        source_info: &crate::topology::DistributedInfo,
        target_info: &crate::topology::SharedInfo,
        target_file_index: usize,
        task_source: &mut TaskSource,
        source_ranges: &mut Vec<(u64, u64, u64)>,
        ranges_per_file: &mut Vec<usize>,
    ) -> Result<()> {
        // Check if this target file should contain this shared tensor
        if !target_info.filename_indices().contains(&target_file_index) {
            // This target file doesn't need this tensor
            return Ok(());
        }

        // Get tensor properties
        let full_shape = source_info.shape();
        let dtype_size = source_info.dtype().size();

        // Calculate strides for the full tensor
        let ndim = full_shape.len();
        let mut full_strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
        }

        // For shared target, we need the complete tensor (all elements)
        let total_elements: usize = full_shape.iter().product();
        let target_intervals = vec![(0, total_elements)];

        // Process each source chunk and collect all the pieces
        for source_chunk in source_info.chunks() {
            let source_intervals = get_intervals(source_chunk, &full_strides, full_shape);

            // Find intersection between source chunk and the full target tensor
            let intersections = intersection(&source_intervals, &target_intervals);

            if !intersections.is_empty() {
                let source_file_index = source_chunk.filename_index();

                // Get source file metadata to calculate byte offsets
                let (source_header_size, source_metadata) =
                    &self.source.layout.metadatas[source_file_index];
                let source_tensors = source_metadata.tensors();
                let source_tensor_info = source_tensors.get(tensor_name).ok_or_else(|| {
                    RedistributorError::TensorNotFound {
                        name: tensor_name.to_string(),
                    }
                })?;
                let source_data_offset = source_tensor_info.data_offsets.0;

                // Convert intersection results to byte ranges
                let mut ranges_for_this_file = 0;
                for (source_offset, target_offset, length) in intersections {
                    let start_bytes = source_offset * dtype_size;
                    let end_bytes = (source_offset + length) * dtype_size;
                    let target_offset_bytes = target_offset * dtype_size;

                    let source_start =
                        (source_header_size + source_data_offset + start_bytes) as u64;
                    let source_end = (source_header_size + source_data_offset + end_bytes) as u64;

                    source_ranges.push((source_start, source_end, target_offset_bytes as u64));
                    ranges_for_this_file += 1;
                }

                // Add the source mmap and record how many ranges belong to this file
                if ranges_for_this_file > 0 {
                    trace!(
                        "      Added {} ranges from source file {}",
                        ranges_for_this_file, source_file_index
                    );
                    task_source.add_from_location(&self.source.location, source_file_index);
                    ranges_per_file.push(ranges_for_this_file);
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{Chunk, DistributedInfo, SharedInfo};
    use safetensors::{Dtype, serialize, tensor::TensorView};
    use sha2::{Digest, Sha256};
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
    async fn calculate_file_hash<P: AsRef<Path>>(path: P) -> String {
        let path = path.as_ref();
        let contents = tokio::fs::read(path).await.unwrap();

        trace!("\n=== FILE ANALYSIS: {} ===", path.display());
        trace!("File size: {} bytes", contents.len());

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

            trace!("Header size: {} bytes", header_size);

            if contents.len() >= 8 + header_size {
                // Extract and display the JSON header
                let header_bytes = &contents[8..8 + header_size];
                if let Ok(header_str) = std::str::from_utf8(header_bytes) {
                    trace!("Header JSON: {}", header_str);
                }

                // Show some info about the data section
                let data_start = 8 + header_size;
                let data_size = contents.len() - data_start;
                trace!(
                    "Data section starts at byte {}, size: {} bytes",
                    data_start, data_size
                );

                // Show first 32 bytes of data as hex
                if data_size > 0 {
                    let hex_bytes = std::cmp::min(32, data_size);
                    let hex_str: String = contents[data_start..data_start + hex_bytes]
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join(" ");
                    trace!("First {} bytes of data: {}", hex_bytes, hex_str);
                }
            }
        }

        // Show entire file content as UTF-8 (lossy)
        trace!("\n--- ENTIRE FILE CONTENT (UTF-8 lossy) ---");
        let content_str = String::from_utf8_lossy(&contents);
        trace!("{}", content_str);
        trace!("--- END ENTIRE FILE CONTENT ---\n");

        let mut hasher = Sha256::new();
        hasher.update(&contents);
        let hash = format!("{:x}", hasher.finalize());
        trace!("SHA256: {}", hash);
        trace!("=== END FILE ANALYSIS ===\n");

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
        trace!("Original file hash: {}", original_hash);

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
        let mut redistributor1 = AsyncTensorRedistributor::from_local(
            source_dir.path(),
            distributed_dir.path(),
            distributed_topology,
        )
        .unwrap();

        let created_files = redistributor1.redistribute().await.unwrap();
        trace!("Created distributed files: {:?}", created_files);

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
        let mut redistributor2 = AsyncTensorRedistributor::from_local(
            distributed_dir.path(),
            final_dir.path(),
            final_topology,
        )
        .unwrap();

        let final_files = redistributor2.redistribute().await.unwrap();
        trace!("Created final files: {:?}", final_files);

        // Verify final file exists
        let final_path = final_dir.path().join("model.safetensors");
        assert!(final_path.exists());

        // Step 6: Calculate hash of final file and compare
        let final_hash = calculate_file_hash(&final_path).await;
        trace!("Final file hash: {}", final_hash);

        // The files should be identical
        assert_eq!(
            original_hash, final_hash,
            "Round-trip redistribution should preserve file integrity"
        );

        // Step 7: Additional verification - load and compare tensor data
        let final_bytes = tokio::fs::read(&final_path).await.unwrap();
        let final_safetensors = SafeTensors::deserialize(&final_bytes).unwrap();

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

        trace!("✅ Round-trip redistribution test passed!");
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
