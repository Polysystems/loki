//! Lock-free indexed ring buffer with O(1) indexed access support
//! Combines ArrayQueue's lock-free properties with DashMap's indexed access

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use std::collections::VecDeque;

use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};

/// Lock-free ring buffer with indexed access capabilities
/// Provides both queue operations (push/pop) and indexed access (get/set)
pub struct IndexedRingBuffer<T>
where
    T: Clone + Send + Sync,
{
    /// Core lock-free queue for streaming operations
    queue: Arc<ArrayQueue<T>>,
    
    /// Index for O(1) random access by position
    index: Arc<DashMap<usize, T>>,
    
    /// Current head position (next write position)
    head: Arc<AtomicUsize>,
    
    /// Current tail position (next read position) 
    tail: Arc<AtomicUsize>,
    
    /// Buffer capacity
    capacity: usize,
    
    /// Current size
    size: Arc<AtomicUsize>,
    
    /// Statistics
    stats: Arc<IndexedBufferStats>,
}

impl<T> IndexedRingBuffer<T>
where
    T: Clone + Send + Sync,
{
    /// Create a new indexed ring buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
            index: Arc::new(DashMap::with_capacity(capacity)),
            head: Arc::new(AtomicUsize::new(0)),
            tail: Arc::new(AtomicUsize::new(0)),
            capacity,
            size: Arc::new(AtomicUsize::new(0)),
            stats: Arc::new(IndexedBufferStats::new()),
        }
    }
    
    /// Push item to the buffer (queue operation)
    pub fn push(&self, item: T) -> bool {
        self.stats.record_push();
        
        match self.queue.push(item.clone()) {
            Ok(()) => {
                let head_pos = self.head.fetch_add(1, Ordering::Relaxed) % self.capacity;
                
                // Update index for random access
                self.index.insert(head_pos, item);
                
                // Update size
                let old_size = self.size.load(Ordering::Relaxed);
                if old_size < self.capacity {
                    self.size.fetch_add(1, Ordering::Relaxed);
                }
                
                true
            }
            Err(_) => {
                self.stats.record_push_failure();
                false
            }
        }
    }
    
    /// Pop item from the buffer (queue operation)
    pub fn pop(&self) -> Option<T> {
        self.stats.record_pop();
        
        match self.queue.pop() {
            Some(item) => {
                let tail_pos = self.tail.fetch_add(1, Ordering::Relaxed) % self.capacity;
                
                // Remove from index
                self.index.remove(&tail_pos);
                
                // Update size
                if self.size.load(Ordering::Relaxed) > 0 {
                    self.size.fetch_sub(1, Ordering::Relaxed);
                }
                
                Some(item)
            }
            None => {
                self.stats.record_pop_failure();
                None
            }
        }
    }
    
    /// Get item by index (O(1) indexed access)
    pub fn get(&self, index: usize) -> Option<T> {
        self.stats.record_index_access();
        
        if index >= self.capacity {
            return None;
        }
        
        // Calculate actual position in the ring buffer
        let tail_pos = self.tail.load(Ordering::Relaxed);
        let actual_index = (tail_pos + index) % self.capacity;
        
        self.index.get(&actual_index).map(|entry| entry.value().clone())
    }
    
    /// Get item by absolute position in buffer
    pub fn get_absolute(&self, position: usize) -> Option<T> {
        self.stats.record_index_access();
        
        if position >= self.capacity {
            return None;
        }
        
        self.index.get(&position).map(|entry| entry.value().clone())
    }
    
    /// Get a window of items starting from index
    pub fn get_window(&self, start_index: usize, window_size: usize) -> Vec<T> {
        self.stats.record_window_access();
        
        let mut window = Vec::with_capacity(window_size);
        let current_size = self.size.load(Ordering::Relaxed);
        
        for i in 0..window_size {
            let index = start_index + i;
            if index >= current_size {
                break;
            }
            
            if let Some(item) = self.get(index) {
                window.push(item);
            } else {
                break;
            }
        }
        
        window
    }
    
    /// Get the last N items (most recent)
    pub fn get_last_n(&self, n: usize) -> Vec<T> {
        self.stats.record_window_access();
        
        let current_size = self.size.load(Ordering::Relaxed);
        if current_size == 0 || n == 0 {
            return Vec::new();
        }
        
        let start_index = if current_size >= n { current_size - n } else { 0 };
        self.get_window(start_index, n)
    }
    
    /// Get all items as a vector (for analysis operations)
    pub fn get_all(&self) -> Vec<T> {
        self.stats.record_bulk_access();
        
        let current_size = self.size.load(Ordering::Relaxed);
        self.get_window(0, current_size)
    }
    
    /// Get current size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size.load(Ordering::Relaxed) == 0
    }
    
    /// Check if full
    pub fn is_full(&self) -> bool {
        self.size.load(Ordering::Relaxed) >= self.capacity
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Clear the buffer
    pub fn clear(&self) {
        // Clear the queue
        while self.queue.pop().is_some() {}
        
        // Clear the index
        self.index.clear();
        
        // Reset positions
        self.head.store(0, Ordering::Relaxed);
        self.tail.store(0, Ordering::Relaxed);
        self.size.store(0, Ordering::Relaxed);
        
        self.stats.record_clear();
    }
    
    /// Get buffer statistics
    pub fn stats(&self) -> IndexedBufferStats {
        self.stats.snapshot()
    }
}

impl<T> Clone for IndexedRingBuffer<T>
where
    T: Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        // Create a new empty buffer with same capacity
        // Note: This doesn't clone the contents, similar to ArrayQueue
        Self::new(self.capacity)
    }
}

/// Specialized implementation for windowed operations on numeric types
impl IndexedRingBuffer<f32> {
    /// Calculate moving average over a window
    pub fn moving_average(&self, window_size: usize) -> Option<f32> {
        let current_size = self.len();
        if current_size == 0 || window_size == 0 {
            return None;
        }
        
        let actual_window = std::cmp::min(window_size, current_size);
        let window_data = self.get_last_n(actual_window);
        
        if window_data.is_empty() {
            return None;
        }
        
        let sum: f32 = window_data.iter().sum();
        Some(sum / window_data.len() as f32)
    }
    
    /// Calculate standard deviation over a window
    pub fn moving_std_dev(&self, window_size: usize) -> Option<f32> {
        let avg = self.moving_average(window_size)?;
        let current_size = self.len();
        if current_size == 0 || window_size == 0 {
            return None;
        }
        
        let actual_window = std::cmp::min(window_size, current_size);
        let window_data = self.get_last_n(actual_window);
        
        if window_data.len() < 2 {
            return Some(0.0);
        }
        
        let variance: f32 = window_data
            .iter()
            .map(|&x| {
                let diff = x - avg;
                diff * diff
            })
            .sum::<f32>() / window_data.len() as f32;
            
        Some(variance.sqrt())
    }
    
    /// Find min/max values in the last N items
    pub fn window_min_max(&self, window_size: usize) -> Option<(f32, f32)> {
        let window_data = self.get_last_n(window_size);
        
        if window_data.is_empty() {
            return None;
        }
        
        let min = window_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = window_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        Some((min, max))
    }
}

/// Specialized implementation for temporal data
impl<T> IndexedRingBuffer<T>
where
    T: Clone + Send + Sync + HasTimestamp,
{
    /// Get items within a time range
    pub fn get_time_range(&self, start: Instant, end: Instant) -> Vec<T> {
        self.stats.record_temporal_query();
        
        let all_items = self.get_all();
        all_items
            .into_iter()
            .filter(|item| {
                let timestamp = item.timestamp();
                timestamp >= start && timestamp <= end
            })
            .collect()
    }
    
    /// Get items newer than a specific time
    pub fn get_since(&self, since: Instant) -> Vec<T> {
        let now = Instant::now();
        self.get_time_range(since, now)
    }
}

/// Trait for items that have timestamps
pub trait HasTimestamp {
    fn timestamp(&self) -> Instant;
}

/// Statistics for the indexed buffer
#[derive(Debug, Clone)]
pub struct IndexedBufferStats {
    pub push_count: Arc<AtomicUsize>,
    pub pop_count: Arc<AtomicUsize>,
    pub push_failures: Arc<AtomicUsize>,
    pub pop_failures: Arc<AtomicUsize>,
    pub index_accesses: Arc<AtomicUsize>,
    pub window_accesses: Arc<AtomicUsize>,
    pub bulk_accesses: Arc<AtomicUsize>,
    pub temporal_queries: Arc<AtomicUsize>,
    pub clear_count: Arc<AtomicUsize>,
    pub created_at: Instant,
}

impl IndexedBufferStats {
    pub fn new() -> Self {
        Self {
            push_count: Arc::new(AtomicUsize::new(0)),
            pop_count: Arc::new(AtomicUsize::new(0)),
            push_failures: Arc::new(AtomicUsize::new(0)),
            pop_failures: Arc::new(AtomicUsize::new(0)),
            index_accesses: Arc::new(AtomicUsize::new(0)),
            window_accesses: Arc::new(AtomicUsize::new(0)),
            bulk_accesses: Arc::new(AtomicUsize::new(0)),
            temporal_queries: Arc::new(AtomicUsize::new(0)),
            clear_count: Arc::new(AtomicUsize::new(0)),
            created_at: Instant::now(),
        }
    }
    
    pub fn record_push(&self) {
        self.push_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_pop(&self) {
        self.pop_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_push_failure(&self) {
        self.push_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_pop_failure(&self) {
        self.pop_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_index_access(&self) {
        self.index_accesses.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_window_access(&self) {
        self.window_accesses.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_bulk_access(&self) {
        self.bulk_accesses.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_temporal_query(&self) {
        self.temporal_queries.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_clear(&self) {
        self.clear_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn snapshot(&self) -> Self {
        Self {
            push_count: Arc::new(AtomicUsize::new(self.push_count.load(Ordering::Relaxed))),
            pop_count: Arc::new(AtomicUsize::new(self.pop_count.load(Ordering::Relaxed))),
            push_failures: Arc::new(AtomicUsize::new(self.push_failures.load(Ordering::Relaxed))),
            pop_failures: Arc::new(AtomicUsize::new(self.pop_failures.load(Ordering::Relaxed))),
            index_accesses: Arc::new(AtomicUsize::new(self.index_accesses.load(Ordering::Relaxed))),
            window_accesses: Arc::new(AtomicUsize::new(self.window_accesses.load(Ordering::Relaxed))),
            bulk_accesses: Arc::new(AtomicUsize::new(self.bulk_accesses.load(Ordering::Relaxed))),
            temporal_queries: Arc::new(AtomicUsize::new(self.temporal_queries.load(Ordering::Relaxed))),
            clear_count: Arc::new(AtomicUsize::new(self.clear_count.load(Ordering::Relaxed))),
            created_at: self.created_at,
        }
    }
    
    /// Get success rate for push operations
    pub fn push_success_rate(&self) -> f64 {
        let total = self.push_count.load(Ordering::Relaxed);
        let failures = self.push_failures.load(Ordering::Relaxed);
        
        if total == 0 {
            1.0
        } else {
            (total - failures) as f64 / total as f64
        }
    }
    
    /// Get success rate for pop operations  
    pub fn pop_success_rate(&self) -> f64 {
        let total = self.pop_count.load(Ordering::Relaxed);
        let failures = self.pop_failures.load(Ordering::Relaxed);
        
        if total == 0 {
            1.0
        } else {
            (total - failures) as f64 / total as f64
        }
    }
    
    /// Get total operations count
    pub fn total_operations(&self) -> usize {
        self.push_count.load(Ordering::Relaxed) +
        self.pop_count.load(Ordering::Relaxed) +
        self.index_accesses.load(Ordering::Relaxed) +
        self.window_accesses.load(Ordering::Relaxed) +
        self.bulk_accesses.load(Ordering::Relaxed) +
        self.temporal_queries.load(Ordering::Relaxed)
    }
    
    /// Get operations per second since creation
    pub fn ops_per_second(&self) -> f64 {
        let elapsed = self.created_at.elapsed();
        let total_ops = self.total_operations();
        
        if elapsed.as_secs_f64() == 0.0 {
            0.0
        } else {
            total_ops as f64 / elapsed.as_secs_f64()
        }
    }
}

impl Default for IndexedBufferStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration helper to convert from VecDeque to IndexedRingBuffer
impl<T> From<VecDeque<T>> for IndexedRingBuffer<T>
where
    T: Clone + Send + Sync,
{
    fn from(deque: VecDeque<T>) -> Self {
        let capacity = std::cmp::max(deque.len(), 16); // Minimum capacity
        let buffer = Self::new(capacity);
        
        for item in deque {
            buffer.push(item);
        }
        
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_basic_operations() {
        let buffer = IndexedRingBuffer::new(10);
        
        // Test push/pop
        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(buffer.push(3));
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.len(), 2);
        
        // Test indexed access
        assert_eq!(buffer.get(0), Some(2));
        assert_eq!(buffer.get(1), Some(3));
        assert_eq!(buffer.get(2), None);
    }
    
    #[test]
    fn test_window_operations() {
        let buffer = IndexedRingBuffer::new(10);
        
        for i in 0..5 {
            buffer.push(i);
        }
        
        let window = buffer.get_window(1, 3);
        assert_eq!(window, vec![1, 2, 3]);
        
        let last_3 = buffer.get_last_n(3);
        assert_eq!(last_3, vec![2, 3, 4]);
    }
    
    #[test]
    fn test_moving_average() {
        let buffer = IndexedRingBuffer::new(10);
        
        for i in 1..=5 {
            buffer.push(i as f32);
        }
        
        let avg = buffer.moving_average(3).unwrap();
        assert_eq!(avg, 4.0); // Average of [3, 4, 5]
        
        let std_dev = buffer.moving_std_dev(3).unwrap();
        assert!(std_dev > 0.0);
    }
    
    #[test]
    fn test_concurrent_access() {
        let buffer = Arc::new(IndexedRingBuffer::new(1000));
        let num_threads = 4;
        let items_per_thread = 250;
        
        let mut handles = Vec::new();
        
        // Producer threads
        for thread_id in 0..num_threads {
            let buffer_clone = buffer.clone();
            let handle = thread::spawn(move || {
                for i in 0..items_per_thread {
                    let item = thread_id * items_per_thread + i;
                    while !buffer_clone.push(item) {
                        std::hint::spin_loop();
                    }
                }
            });
            handles.push(handle);
        }
        
        // Wait for producers
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all items were inserted
        assert_eq!(buffer.len(), num_threads * items_per_thread);
        
        // Test concurrent indexed access
        let mut handles = Vec::new();
        
        for _ in 0..num_threads {
            let buffer_clone = buffer.clone();
            let handle = thread::spawn(move || {
                let mut access_count = 0;
                for i in 0..100 {
                    if buffer_clone.get(i * 10).is_some() {
                        access_count += 1;
                    }
                }
                access_count
            });
            handles.push(handle);
        }
        
        let total_accesses: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert!(total_accesses > 0);
    }
}