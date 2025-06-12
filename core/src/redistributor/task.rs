use super::Result;
use log::{error, trace};
use memmap2::{Mmap, MmapMut};
use std::sync::Arc;

// Memory-mapped task type for high performance file operations
pub(crate) struct Task {
    // Target write info - single contiguous region
    pub target_mmap: Arc<MmapMut>,
    pub target_start: u64,
    pub target_end: u64,
    pub target_file_index: usize,

    // Source read info - multiple reads from potentially multiple files
    pub source: Vec<Arc<Mmap>>,
    pub source_ranges: Vec<(u64, u64, u64)>, // Flat list of (start, end, target_offset) byte ranges
    pub ranges_per_file: Vec<usize>,         // How many ranges belong to each source file
}

impl Task {
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

            let source_file_count = self.source.len();

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
                    let mmap = &self.source[file_idx];
                    let source_data = &mmap[source_start as usize..source_end as usize];
                    // let source_data = self.source.read(file_idx, source_start, source_end).await?;

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
