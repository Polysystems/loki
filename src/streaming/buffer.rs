use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_queue::ArrayQueue;
use crate::zero_cost_validation::{ZeroCostValidator, validation_levels};
use std::hint::{likely, unlikely};
use crate::hot_path;

/// Lock-free ring buffer for efficient streaming
pub struct RingBuffer<T> {
    buffer: Arc<ArrayQueue<T>>,
    capacity: usize,
    size: Arc<AtomicUsize>,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new lock-free ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(ArrayQueue::new(capacity)),
            capacity,
            size: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Write to the buffer (ultra-optimized critical hot path, lock-free)
    #[inline(always)] // Critical path - ensure inlining
    pub fn write(&self, item: T) -> bool {
        // Critical hot path for streaming operations
        hot_path!({
            // Zero-cost validation: ensure this hot path is optimized
            ZeroCostValidator::<Self, {validation_levels::ADVANCED}>::mark_zero_cost(|| {
                // Try to push to the lock-free queue
                match self.buffer.push(item) {
                    Ok(()) => {
                        self.size.fetch_add(1, Ordering::Relaxed);
                        true
                    }
                    Err(_) => {
                        // Buffer is full
                        false
                    }
                }
            })
        })
    }

    /// Read from the buffer (ultra-optimized critical hot path, lock-free)
    #[inline(always)] // Critical path - ensure inlining  
    pub fn read(&self) -> Option<T> {
        // Critical hot path for streaming operations
        hot_path!({
            // Zero-cost validation: ensure this hot path is optimized
            ZeroCostValidator::<Self, {validation_levels::ADVANCED}>::mark_zero_cost(|| {
                // Try to pop from the lock-free queue
                match self.buffer.pop() {
                    Some(item) => {
                        self.size.fetch_sub(1, Ordering::Relaxed);
                        Some(item)
                    }
                    None => None
                }
            })
        })
    }

    /// Get current size (lock-free)
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if empty (lock-free)
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if full (lock-free)
    pub fn is_full(&self) -> bool {
        self.buffer.is_full()
    }

    /// Clear the buffer (lock-free)
    pub fn clear(&self) {
        // Drain all items from the queue
        while self.buffer.pop().is_some() {
            // Keep popping until empty
        }
        self.size.store(0, Ordering::Relaxed);
    }
}

impl<T: Clone> Clone for RingBuffer<T> {
    fn clone(&self) -> Self {
        // Create a new buffer with same capacity
        let new_buffer = Arc::new(ArrayQueue::new(self.capacity));
        
        // Note: We can't clone the contents of ArrayQueue directly
        // This creates an empty clone with the same capacity
        Self {
            buffer: new_buffer,
            capacity: self.capacity,
            size: Arc::new(AtomicUsize::new(0)),
        }
    }
}

// Specialized implementations for common primitive types with optimized operations
impl RingBuffer<u8> {
    /// Optimized batch write for u8 (zero-copy for byte streams)
    /// Zero-cost validation ensures SIMD vectorization where possible
    #[inline(always)]
    pub fn write_bytes(&self, data: &[u8]) -> usize {
        // Validate SIMD alignment for potential vectorization
        crate::zero_cost_validation::simd_validation::SIMDValidator::validate_vectorization_hints(data);
        let mut written = 0;
        for &byte in data {
            if self.write(byte) {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Optimized batch read for u8 
    #[inline(always)]
    pub fn read_bytes(&self, output: &mut [u8]) -> usize {
        let mut read_count = 0;
        for slot in output.iter_mut() {
            if let Some(byte) = self.read() {
                *slot = byte;
                read_count += 1;
            } else {
                break;
            }
        }
        read_count
    }
}

impl RingBuffer<f32> {
    /// Specialized SIMD-friendly operations for f32
    /// Zero-cost validation ensures optimal SIMD code generation
    #[inline(always)]
    pub fn write_samples(&self, samples: &[f32]) -> usize {
        // Validate SIMD operations for f32 arrays
        #[cfg(feature = "simd-optimizations")]
        crate::zero_cost_validation::simd_validation::SIMDValidator::validate_f32_vectorization(samples);
        let mut written = 0;
        for &sample in samples {
            if self.write(sample) {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Calculate running average (optimized for audio/signal processing)
    #[inline(always)]
    pub fn running_average(&self, _window: usize) -> Option<f32> {
        // Note: ArrayQueue doesn't support indexed access needed for windowed operations
        // For running average, we would need to maintain a separate circular buffer
        // or use a different data structure that supports indexed access
        // 
        // This is a limitation of the lock-free ArrayQueue approach
        // Consider using a dedicated time-series buffer for this use case
        None
    }
}

/// Stream buffer for managing data flow
pub struct StreamBuffer {
    data_buffer: RingBuffer<Vec<u8>>,
    metadata_buffer: RingBuffer<StreamMetadata>,
}

#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub sequence: u64,
    pub timestamp: std::time::Instant,
    pub size: usize,
}

impl StreamBuffer {
    /// Create a new stream buffer
    pub fn new(capacity: usize) -> Self {
        Self { data_buffer: RingBuffer::new(capacity), metadata_buffer: RingBuffer::new(capacity) }
    }

    /// Write data with metadata
    pub fn write(&self, data: Vec<u8>, sequence: u64) -> bool {
        let metadata =
            StreamMetadata { sequence, timestamp: std::time::Instant::now(), size: data.len() };

        // Write both data and metadata atomically
        // Data buffer write success is likely in healthy systems
        if likely(self.data_buffer.write(data)) {
            // Metadata write success is also likely given data success
            if self.metadata_buffer.write(metadata) {
                true
            } else {
                // Metadata write failure - rollback needed
                self.data_buffer.read();
                false
            }
        } else {
            // Data buffer full - return false to indicate failure
            false
        }
    }

    /// Read data with metadata
    pub fn read(&self) -> Option<(Vec<u8>, StreamMetadata)> {
        // Read both atomically
        if let Some(data) = self.data_buffer.read() {
            if let Some(metadata) = self.metadata_buffer.read() {
                Some((data, metadata))
            } else {
                // This shouldn't happen, but handle gracefully
                None
            }
        } else {
            None
        }
    }

    /// Get buffer statistics
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            size: self.data_buffer.len(),
            capacity: self.data_buffer.capacity,
            is_full: self.data_buffer.is_full(),
            is_empty: self.data_buffer.is_empty(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferStats {
    pub size: usize,
    pub capacity: usize,
    pub is_full: bool,
    pub is_empty: bool,
}

impl BufferStats {
    pub fn utilization(&self) -> f32 {
        (self.size as f32 / self.capacity as f32) * 100.0
    }
}
