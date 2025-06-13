use super::task::Task;
use super::{Layout, RedistributorError, Result};
use crate::topology::Topology;
use log::trace;
use memmap2::{Mmap, MmapMut};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use std::fs::{self, File};
use std::io::Write;

#[derive(Clone)]
pub struct SourceLocation {
    pub(crate) mmaps: Vec<Arc<Mmap>>,
}

#[derive(Clone)]
pub struct WriteLocation {
    pub dir: PathBuf,
    pub mmaps: Option<Vec<Arc<MmapMut>>>,
}

impl WriteLocation {
    /// Initialize target files and memory maps
    pub fn init(&mut self, layout: &Layout) -> Result<()> {
        trace!("INIT: Starting target location initialization...");

        // Create directory
        trace!("INIT: Creating directory...");
        let dir_start = Instant::now();
        fs::create_dir_all(&self.dir)?;
        trace!("INIT: Directory created in {:?}", dir_start.elapsed());

        // Create files with headers
        trace!("INIT: Creating files with headers...");
        let files_start = Instant::now();
        self.create_files_with_headers(layout)?;
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
    pub fn save(&self, topology: &Topology) -> Result<()> {
        if let Some(ref target_mmaps) = self.mmaps {
            for target_mmap in target_mmaps {
                target_mmap.flush()?;
            }
        }
        self.write_topology(topology)?;
        Ok(())
    }

    /// Create a write task for the specified file index and parameters
    pub fn create_write_task(
        &self,
        target_file_index: usize,
        target_start: u64,
        target_end: u64,
        source: Vec<Arc<Mmap>>,
        source_ranges: Vec<(u64, u64, u64)>,
        ranges_per_file: Vec<usize>,
    ) -> Option<Task> {
        if let Some(ref target_mmaps) = self.mmaps {
            Some(Task {
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
    fn write_topology(&self, topology: &Topology) -> Result<()> {
        let topology_path = self.dir.join("topology.json");
        let topology_json = serde_json::to_string_pretty(topology)?;
        fs::write(topology_path, topology_json)?;
        Ok(())
    }

    /// Create all target files with their headers
    fn create_files_with_headers(&self, layout: &Layout) -> Result<()> {
        assert_eq!(layout.topology.filenames().len(), layout.metadatas.len());
        
        for ((_, metadata), filename) in layout.metadatas.iter().zip(layout.topology.filenames()) {
            let data_len = metadata.validate()?;
            let mut metadata_buf = serde_json::to_string(&metadata)?.into_bytes();
            // Force alignment to 8 bytes.
            let extra = (8 - metadata_buf.len() % 8) % 8;
            metadata_buf.extend(vec![b' '; extra]);

            let n: u64 = metadata_buf.len() as u64;

            let mut f = File::options()
                .write(true)
                .create(true)
                .open(self.dir.join(filename))?;
            f.write_all(&n.to_le_bytes())?;
            f.write_all(&metadata_buf)?;
            let total = data_len + metadata_buf.len() + 8;
            f.set_len(total as u64)?;
            f.flush()?;
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
                MmapOptions::new().map_mut(&file).map_err(|e| {
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

#[derive(Clone)]
pub struct Source {
    pub layout: Layout,
    pub location: SourceLocation,
}

impl Source {
    pub fn new(layout: Layout, location: SourceLocation) -> Self {
        Self { layout, location }
    }
}

#[derive(Clone)]
pub struct Target {
    pub layout: Layout,
    pub location: WriteLocation,
}

impl Target {
    pub fn new(layout: Layout, location: WriteLocation) -> Self {
        Self { layout, location }
    }
}

/// Initialize source memory maps
pub fn init_source_mmaps(files: &[File]) -> Result<Vec<Arc<Mmap>>> {
    use memmap2::MmapOptions;

    let mut source_mmaps = Vec::new();

    for file in files {
        let mmap = unsafe {
            MmapOptions::new().map(file).map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create memory map: {}", e),
                ))
            })?
        };

        source_mmaps.push(Arc::new(mmap));
    }

    Ok(source_mmaps)
}
