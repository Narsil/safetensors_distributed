use super::location::SourceLocation;
use super::{RedistributorError, Result};
use futures_util::StreamExt;
use log::{error, trace};
use memmap2::{Mmap, MmapMut};
use reqwest::header::RANGE;
use reqwest::{Client, header::HeaderMap};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use url::Url;

/// Source data abstraction for tasks
pub enum TaskSources {
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
    pub async fn read(&self, file_index: usize, start: u64, end: u64) -> Result<Cow<'_, [u8]>> {
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
pub enum TaskSource {
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
    pub fn new_local() -> Self {
        Self::Local { mmaps: Vec::new() }
    }

    /// Create a new empty task source builder for remote sources
    pub fn new_remote(
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
    pub fn add_from_location(&mut self, location: &SourceLocation, file_index: usize) {
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
    pub fn len(&self) -> usize {
        match self {
            TaskSource::Local { mmaps } => mmaps.len(),
            TaskSource::Remote { file_paths, .. } => file_paths.len(),
        }
    }

    /// Check if this task source is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert this builder into task sources for execution
    pub fn into_task_sources(self) -> TaskSources {
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

/// Task type enum for different redistribution strategies
pub enum Task {
    /// Write-focused task: multiple reads → single write (for ReadUnorderedWriteSerial)
    Write(MmapWriteTask),
    /// Read-focused task: single read → multiple writes (for ReadSerialWriteUnordered)
    ReadSerial(ReadSerialTask),
}

/// Write operation for ReadSerialTask - one target write from read data
#[derive(Debug, Clone)]
pub struct WriteOperation {
    /// Target file index
    pub target_file_index: usize,
    /// Target memory-mapped file
    pub target_mmap: Arc<MmapMut>,
    /// Start position in target file
    pub target_start: u64,
    /// End position in target file
    pub target_end: u64,
    /// Offset within the read data where this write's data starts
    pub read_offset: usize,
}

/// Read-serial task type for ReadSerialWriteUnordered strategy
pub struct ReadSerialTask {
    // Source read info - single contiguous read
    pub source: TaskSources,
    pub source_file_index: usize,
    pub source_start: u64,
    pub source_end: u64,

    // Target write info - multiple writes to potentially multiple files
    pub writes: Vec<WriteOperation>,
}

// Memory-mapped task type for high performance file operations
pub struct MmapWriteTask {
    // Target write info - single contiguous region
    pub target_mmap: Arc<MmapMut>,
    pub target_start: u64,
    pub target_end: u64,
    pub target_file_index: usize,

    // Source read info - multiple reads from potentially multiple files
    pub source: TaskSources,
    pub source_ranges: Vec<(u64, u64, u64)>, // Flat list of (start, end, target_offset) byte ranges
    pub ranges_per_file: Vec<usize>,         // How many ranges belong to each source file
}

impl ReadSerialTask {
    pub async fn run(&self) -> Result<()> {
        let read_length = (self.source_end - self.source_start) as usize;
        trace!(
            "Reading {read_length} from source file {}",
            self.source_file_index
        );

        // Read once from source - large sequential read
        let source_data = self
            .source
            .read(0, self.source_start, self.source_end)
            .await?;

        trace!(
            "  Executing {} target writes from single read",
            self.writes.len()
        );

        // SAFETY: This is safe because:
        // 1. All write ranges are pre-calculated and guaranteed non-overlapping
        // 2. Each write operation writes to a unique byte range within memory-mapped files
        // 3. No two writes will ever write to the same memory location
        // 4. The topology calculation ensures mutually exclusive write regions
        unsafe {
            // Execute all writes from the read data
            for (write_idx, write_op) in self.writes.iter().enumerate() {
                let write_length = (write_op.target_end - write_op.target_start) as usize;

                trace!(
                    "    Write {}: {} bytes to target file {} offset {}→{}",
                    write_idx,
                    write_length,
                    write_op.target_file_index,
                    write_op.target_start,
                    write_op.target_end
                );

                // Get write data from read buffer
                let write_data =
                    &source_data[write_op.read_offset..write_op.read_offset + write_length];
                

                // Write to target at specified position
                let target_ptr = write_op
                    .target_mmap
                    .as_ptr()
                    .add(write_op.target_start as usize)
                    as *mut u8;
                let target_slice = std::slice::from_raw_parts_mut(target_ptr, write_length);
                target_slice.copy_from_slice(write_data);

                // Flush this write range asynchronously
                if let Err(e) = write_op
                    .target_mmap
                    .flush_async_range(write_op.target_start as usize, write_length)
                {
                    error!(
                        "Async flush failed for range {}:{}: {}",
                        write_op.target_start, write_op.target_end, e
                    );
                    // Don't fail the task, just log the error and continue
                }
            }
        }

        Ok(())
    }
}

impl MmapWriteTask {
    pub async fn run(&self) -> Result<()> {
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
            let mut range_idx = 0usize;
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
