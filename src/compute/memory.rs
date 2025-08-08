use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use parking_lot::Mutex;
use tracing::debug;

use super::Device;

/// Memory allocation on a device
#[derive(Debug)]
pub struct MemoryAllocation {
    pub device_id: String,
    pub ptr: *mut u8,
    pub size: usize,
    pub offset: usize,
}

unsafe impl Send for MemoryAllocation {}
unsafe impl Sync for MemoryAllocation {}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    device: Device,
    total_size: usize,
    block_size: usize,
    free_blocks: Arc<Mutex<Vec<usize>>>,
    allocations: Arc<Mutex<HashMap<usize, AllocationInfo>>>,
}

#[derive(Debug)]
struct AllocationInfo {
    size: usize,
    blocks: Vec<usize>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(device: Device, total_size: usize) -> Self {
        let block_size = 1024 * 1024; // 1MB blocks
        let num_blocks = total_size / block_size;
        let free_blocks = (0..num_blocks).collect();

        Self {
            device,
            total_size,
            block_size,
            free_blocks: Arc::new(Mutex::new(free_blocks)),
            allocations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> Result<PoolAllocation> {
        let blocks_needed = (size + self.block_size - 1) / self.block_size;

        let mut free_blocks = self.free_blocks.lock();

        if free_blocks.len() < blocks_needed {
            anyhow::bail!(
                "Not enough memory in pool. Requested: {} MB, Available: {} MB",
                size / 1024 / 1024,
                free_blocks.len() * self.block_size / 1024 / 1024
            );
        }

        let mut allocated_blocks = Vec::with_capacity(blocks_needed);
        for _ in 0..blocks_needed {
            if let Some(block) = free_blocks.pop() {
                allocated_blocks.push(block);
            }
        }

        let allocation_id = allocated_blocks[0];
        let mut allocations = self.allocations.lock();
        allocations
            .insert(allocation_id, AllocationInfo { size, blocks: allocated_blocks.clone() });

        debug!(
            "Allocated {} MB ({} blocks) from pool on {}",
            size / 1024 / 1024,
            blocks_needed,
            self.device.name
        );

        Ok(PoolAllocation {
            pool: Arc::new(self.clone()),
            allocation_id,
            size,
            device_id: self.device.id.clone(),
        })
    }

    /// Free an allocation
    fn free(&self, allocation_id: usize) {
        let mut allocations = self.allocations.lock();
        if let Some(info) = allocations.remove(&allocation_id) {
            let mut free_blocks = self.free_blocks.lock();
            free_blocks.extend(info.blocks);

            debug!("Freed {} MB from pool on {}", info.size / 1024 / 1024, self.device.name);
        }
    }

    /// Get current usage statistics
    pub fn usage_stats(&self) -> MemoryUsageStats {
        let free_blocks = self.free_blocks.lock();
        let allocations = self.allocations.lock();

        let free_bytes = free_blocks.len() * self.block_size;
        let used_bytes = self.total_size - free_bytes;
        let allocation_count = allocations.len();

        MemoryUsageStats {
            total_bytes: self.total_size,
            used_bytes,
            free_bytes,
            allocation_count,
            fragmentation: self.calculate_fragmentation(&free_blocks),
        }
    }

    fn calculate_fragmentation(&self, free_blocks: &[usize]) -> f32 {
        if free_blocks.is_empty() {
            return 0.0;
        }

        // Simple fragmentation metric: ratio of non-contiguous blocks
        let mut sorted_blocks = free_blocks.to_vec();
        sorted_blocks.sort_unstable();

        let mut gaps = 0;
        for i in 1..sorted_blocks.len() {
            if sorted_blocks[i] != sorted_blocks[i - 1] + 1 {
                gaps += 1;
            }
        }

        gaps as f32 / sorted_blocks.len() as f32
    }
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            total_size: self.total_size,
            block_size: self.block_size,
            free_blocks: self.free_blocks.clone(),
            allocations: self.allocations.clone(),
        }
    }
}

/// A memory allocation from a pool
pub struct PoolAllocation {
    pool: Arc<MemoryPool>,
    allocation_id: usize,
    size: usize,
    device_id: String,
}

impl PoolAllocation {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }
}

impl Drop for PoolAllocation {
    fn drop(&mut self) {
        self.pool.free(self.allocation_id);
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub total_bytes: usize,
    pub used_bytes: usize,
    pub free_bytes: usize,
    pub allocation_count: usize,
    pub fragmentation: f32,
}

impl MemoryUsageStats {
    pub fn usage_percent(&self) -> f32 {
        (self.used_bytes as f32 / self.total_bytes as f32) * 100.0
    }

    pub fn format_summary(&self) -> String {
        format!(
            "Memory: {:.1}% used ({}/{} MB), {} allocations, {:.1}% fragmentation",
            self.usage_percent(),
            self.used_bytes / 1024 / 1024,
            self.total_bytes / 1024 / 1024,
            self.allocation_count,
            self.fragmentation * 100.0
        )
    }
}
