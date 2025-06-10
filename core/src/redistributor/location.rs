use super::{Result, RedistributorError, Layout};
use super::task::{TaskSources, MmapWriteTask};
use crate::topology::Topology;
use futures::future::join_all;
use memmap2::{Mmap, MmapMut};
use reqwest::{Client, header::HeaderMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use url::Url;
use log::trace;

pub enum SourceLocation {
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

pub struct WriteLocation {
    pub dir: PathBuf,
    pub mmaps: Option<Vec<Arc<MmapMut>>>,
}

impl WriteLocation {
    /// Initialize target files and memory maps
    pub async fn init(&mut self, layout: &Layout) -> Result<()> {
        trace!("INIT: Starting target location initialization...");

        // Create directory
        trace!("INIT: Creating directory...");
        let dir_start = Instant::now();
        tokio::fs::create_dir_all(&self.dir).await?;
        trace!("INIT: Directory created in {:?}", dir_start.elapsed());

        // Create files with headers
        trace!("INIT: Creating files with headers...");
        let files_start = Instant::now();
        self.create_files_with_headers(layout).await?;
        trace!("INIT: Files created in {:?}", files_start.elapsed());

        // Initialize memory maps
        trace!("INIT: Initializing memory maps...");
        let mmap_start = Instant::now();
        self.init_target_mmaps(layout)?;
        trace!(
            "INIT: Memory maps initialized in {:?}",
            mmap_start.elapsed()
        );

        trace!("INIT: Target location initialization complete");
        Ok(())
    }

    /// Save/flush all memory-mapped files and write topology
    pub async fn save(&self, topology: &Topology) -> Result<()> {
        if let Some(ref target_mmaps) = self.mmaps {
            for target_mmap in target_mmaps {
                target_mmap.flush()?;
            }
        }
        self.write_topology(topology).await?;
        Ok(())
    }

    /// Create a write task for the specified file index and parameters
    pub fn create_write_task(
        &self,
        target_file_index: usize,
        target_start: u64,
        target_end: u64,
        source: TaskSources,
        source_ranges: Vec<(u64, u64, u64)>,
        ranges_per_file: Vec<usize>,
    ) -> Option<MmapWriteTask> {
        if let Some(ref target_mmaps) = self.mmaps {
            Some(MmapWriteTask {
                target_mmap: Arc::clone(&target_mmaps[target_file_index]),
                target_start,
                target_end,
                target_file_index,
                source,
                source_ranges,
                ranges_per_file,
            })
        } else {
            None
        }
    }

    /// Write topology.json to the target directory
    async fn write_topology(&self, topology: &Topology) -> Result<()> {
        let topology_path = self.dir.join("topology.json");
        let topology_json = serde_json::to_string_pretty(topology)?;
        tokio::fs::write(topology_path, topology_json).await?;
        Ok(())
    }

    /// Create all target files with their headers
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
        use memmap2::MmapOptions;
        use std::fs::OpenOptions;

        let mut target_mmaps = Vec::new();

        for filename in layout.topology.filenames() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(self.dir.join(filename))?;

            let mmap = unsafe {
                MmapOptions::new()
                    .map_mut(&file)
                    .map_err(|e| {
                        RedistributorError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to create memory map for {}: {}", filename, e),
                        ))
                    })?
            };

            target_mmaps.push(Arc::new(mmap));
        }

        self.mmaps = Some(target_mmaps);
        Ok(())
    }
}

pub struct Source {
    pub layout: Layout,
    pub location: SourceLocation,
}

pub struct Target {
    pub layout: Layout,
    pub location: WriteLocation,
} 