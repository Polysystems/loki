//! Lock-free ring buffer implementation for zero-copy streaming

use bytes::Bytes;
use crossbeam_queue::ArrayQueue;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Zero-copy ring buffer using Bytes for efficient memory sharing
pub struct ZeroCopyRingBuffer {
    buffer: Arc<ArrayQueue<Bytes>>,
    capacity: usize,
    bytes_written: Arc<AtomicUsize>,
    bytes_read: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
}

impl ZeroCopyRingBuffer {
    /// Create new ring buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(ArrayQueue::new(capacity)),
            capacity,
            bytes_written: Arc::new(AtomicUsize::new(0)),
            bytes_read: Arc::new(AtomicUsize::new(0)),
            closed: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Write data to buffer (zero-copy via Bytes)
    pub fn write(&self, data: Bytes) -> Result<(), RingBufferError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(RingBufferError::Closed);
        }
        
        super::GLOBAL_STATS.record_operation();
        
        let data_len = data.len();
        match self.buffer.push(data) {
            Ok(()) => {
                self.bytes_written.fetch_add(data_len, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => {
                super::GLOBAL_STATS.record_contention();
                Err(RingBufferError::Full)
            }
        }
    }
    
    /// Read data from buffer (zero-copy)
    pub fn read(&self) -> Option<Bytes> {
        super::GLOBAL_STATS.record_operation();
        
        self.buffer.pop().map(|data| {
            self.bytes_read.fetch_add(data.len(), Ordering::Relaxed);
            data
        })
    }
    
    /// Try write with timeout
    pub fn write_timeout(&self, data: Bytes, timeout: Duration) -> Result<(), RingBufferError> {
        let deadline = Instant::now() + timeout;
        
        loop {
            match self.write(data.clone()) {
                Ok(()) => return Ok(()),
                Err(RingBufferError::Full) => {
                    if Instant::now() >= deadline {
                        return Err(RingBufferError::Timeout);
                    }
                    std::hint::spin_loop();
                }
                Err(e) => return Err(e),
            }
        }
    }
    
    /// Try read with timeout
    pub fn read_timeout(&self, timeout: Duration) -> Result<Bytes, RingBufferError> {
        let deadline = Instant::now() + timeout;
        
        loop {
            if let Some(data) = self.read() {
                return Ok(data);
            }
            
            if Instant::now() >= deadline {
                return Err(RingBufferError::Timeout);
            }
            
            if self.closed.load(Ordering::Acquire) && self.is_empty() {
                return Err(RingBufferError::Closed);
            }
            
            std::hint::spin_loop();
        }
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
    
    /// Get current size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Close the buffer
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
    
    /// Check if closed
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    
    /// Get statistics
    pub fn stats(&self) -> RingBufferStats {
        RingBufferStats {
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            current_size: self.len(),
            capacity: self.capacity,
        }
    }
}

impl Clone for ZeroCopyRingBuffer {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            capacity: self.capacity,
            bytes_written: self.bytes_written.clone(),
            bytes_read: self.bytes_read.clone(),
            closed: self.closed.clone(),
        }
    }
}

/// Multi-producer multi-consumer ring buffer
pub struct MPMCRingBuffer<T: Clone + Send> {
    queue: Arc<ArrayQueue<T>>,
    capacity: usize,
    producers: Arc<AtomicUsize>,
    consumers: Arc<AtomicUsize>,
}

impl<T: Clone + Send> MPMCRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
            capacity,
            producers: Arc::new(AtomicUsize::new(0)),
            consumers: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Register a producer
    pub fn add_producer(&self) -> ProducerHandle<T> {
        self.producers.fetch_add(1, Ordering::Relaxed);
        ProducerHandle {
            buffer: self.clone(),
        }
    }
    
    /// Register a consumer
    pub fn add_consumer(&self) -> ConsumerHandle<T> {
        self.consumers.fetch_add(1, Ordering::Relaxed);
        ConsumerHandle {
            buffer: self.clone(),
        }
    }
    
    /// Send item to buffer
    pub fn send(&self, item: T) -> Result<(), T> {
        super::GLOBAL_STATS.record_operation();
        self.queue.push(item)
    }
    
    /// Receive item from buffer
    pub fn recv(&self) -> Option<T> {
        super::GLOBAL_STATS.record_operation();
        self.queue.pop()
    }
    
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl<T: Clone + Send> Clone for MPMCRingBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
            capacity: self.capacity,
            producers: self.producers.clone(),
            consumers: self.consumers.clone(),
        }
    }
}

/// Producer handle for MPMC buffer
pub struct ProducerHandle<T: Clone + Send> {
    buffer: MPMCRingBuffer<T>,
}

impl<T: Clone + Send> ProducerHandle<T> {
    pub fn send(&self, item: T) -> Result<(), T> {
        self.buffer.send(item)
    }
}

impl<T: Clone + Send> Drop for ProducerHandle<T> {
    fn drop(&mut self) {
        self.buffer.producers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Consumer handle for MPMC buffer
pub struct ConsumerHandle<T: Clone + Send> {
    buffer: MPMCRingBuffer<T>,
}

impl<T: Clone + Send> ConsumerHandle<T> {
    pub fn recv(&self) -> Option<T> {
        self.buffer.recv()
    }
}

impl<T: Clone + Send> Drop for ConsumerHandle<T> {
    fn drop(&mut self) {
        self.buffer.consumers.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Ring buffer error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum RingBufferError {
    #[error("Ring buffer is full")]
    Full,
    
    #[error("Ring buffer is closed")]
    Closed,
    
    #[error("Operation timed out")]
    Timeout,
    
    #[error("Invalid capacity: {0}")]
    InvalidCapacity(usize),
}

/// Ring buffer statistics
#[derive(Debug, Clone)]
pub struct RingBufferStats {
    pub bytes_written: usize,
    pub bytes_read: usize,
    pub current_size: usize,
    pub capacity: usize,
}

impl RingBufferStats {
    pub fn throughput(&self) -> f64 {
        (self.bytes_written + self.bytes_read) as f64 / 2.0
    }
    
    pub fn utilization(&self) -> f64 {
        self.current_size as f64 / self.capacity as f64
    }
}

/// Bounded SPSC ring buffer for single-producer single-consumer scenarios
pub struct SPSCRingBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T> SPSCRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        
        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    
    /// Push item (single producer)
    pub fn push(&mut self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        
        let next_head = (head + 1) % self.capacity;
        if next_head == tail {
            return Err(item); // Buffer full
        }
        
        self.buffer[head] = Some(item);
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }
    
    /// Pop item (single consumer)
    pub fn pop(&mut self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Buffer empty
        }
        
        let item = self.buffer[tail].take();
        self.tail.store((tail + 1) % self.capacity, Ordering::Release);
        item
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_zero_copy_buffer() {
        let buffer = ZeroCopyRingBuffer::new(10);
        
        // Write some data
        let data = Bytes::from("Hello, World!");
        buffer.write(data.clone()).unwrap();
        
        // Read it back
        let read_data = buffer.read().unwrap();
        assert_eq!(read_data, data);
    }
    
    #[test]
    fn test_mpmc_concurrent() {
        let buffer = MPMCRingBuffer::new(100);
        
        // Spawn producers
        let producers: Vec<_> = (0..10).map(|i| {
            let handle = buffer.add_producer();
            thread::spawn(move || {
                for j in 0..10 {
                    handle.send(i * 10 + j).unwrap();
                }
            })
        }).collect();
        
        // Spawn consumers
        let consumers: Vec<_> = (0..5).map(|_| {
            let handle = buffer.add_consumer();
            thread::spawn(move || {
                let mut count = 0;
                while count < 20 {
                    if handle.recv().is_some() {
                        count += 1;
                    }
                }
                count
            })
        }).collect();
        
        // Wait for producers
        for p in producers {
            p.join().unwrap();
        }
        
        // Wait for consumers
        let total: usize = consumers.into_iter().map(|c| c.join().unwrap()).sum();
        assert_eq!(total, 100);
    }
    
    #[test]
    fn test_buffer_full() {
        let buffer = ZeroCopyRingBuffer::new(2);
        
        buffer.write(Bytes::from("1")).unwrap();
        buffer.write(Bytes::from("2")).unwrap();
        
        // Should fail - buffer full
        assert!(matches!(
            buffer.write(Bytes::from("3")),
            Err(RingBufferError::Full)
        ));
    }
}