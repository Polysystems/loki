//! Memory ordering optimization guide and utilities
//! 
//! This module provides guidance and utilities for choosing optimal memory
//! ordering for different lock-free operation patterns.

use std::sync::atomic::Ordering;

/// Memory ordering recommendations for different patterns
pub struct MemoryOrderingGuide;

impl MemoryOrderingGuide {
    /// For simple counters and statistics
    /// - Use Relaxed for increments that don't synchronize with other operations
    /// - No ordering guarantees needed
    pub const COUNTER_INCREMENT: Ordering = Ordering::Relaxed;
    pub const COUNTER_READ: Ordering = Ordering::Relaxed;
    
    /// For flag updates that signal state changes
    /// - Use Release when setting flags that other threads wait on
    /// - Use Acquire when reading flags to ensure visibility of prior writes
    pub const FLAG_SET: Ordering = Ordering::Release;
    pub const FLAG_READ: Ordering = Ordering::Acquire;
    
    /// For lock-free queue operations
    /// - Use Release when enqueuing to ensure item is visible
    /// - Use Acquire when dequeuing to see all writes to the item
    pub const QUEUE_PUSH: Ordering = Ordering::Release;
    pub const QUEUE_POP: Ordering = Ordering::Acquire;
    
    /// For sequence numbers and version counters
    /// - Use AcqRel for read-modify-write operations
    /// - Ensures both acquire and release semantics
    pub const VERSION_INCREMENT: Ordering = Ordering::AcqRel;
    pub const VERSION_READ: Ordering = Ordering::Acquire;
    
    /// For initialization and one-time setup
    /// - Use SeqCst for guaranteed ordering across all threads
    /// - Only for critical initialization paths
    pub const INIT_WRITE: Ordering = Ordering::SeqCst;
    pub const INIT_READ: Ordering = Ordering::SeqCst;
    
    /// For hot path optimizations
    /// - Use Relaxed when possible for maximum performance
    /// - Upgrade to Acquire/Release only when synchronization needed
    pub fn hot_path_read() -> Ordering {
        if cfg!(feature = "strict-ordering") {
            Ordering::Acquire
        } else {
            Ordering::Relaxed
        }
    }
    
    pub fn hot_path_write() -> Ordering {
        if cfg!(feature = "strict-ordering") {
            Ordering::Release
        } else {
            Ordering::Relaxed
        }
    }
}

/// Optimization patterns for common lock-free operations
pub mod patterns {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicBool};
    
    /// Optimized counter pattern
    pub struct OptimizedCounter {
        value: AtomicU64,
    }
    
    impl OptimizedCounter {
        pub fn new() -> Self {
            Self {
                value: AtomicU64::new(0),
            }
        }
        
        #[inline(always)]
        pub fn increment(&self) -> u64 {
            // Use Relaxed for pure counting
            self.value.fetch_add(1, MemoryOrderingGuide::COUNTER_INCREMENT)
        }
        
        #[inline(always)]
        pub fn get(&self) -> u64 {
            // Use Relaxed for reading counters
            self.value.load(MemoryOrderingGuide::COUNTER_READ)
        }
    }
    
    /// Optimized flag pattern
    pub struct OptimizedFlag {
        flag: AtomicBool,
    }
    
    impl OptimizedFlag {
        pub fn new() -> Self {
            Self {
                flag: AtomicBool::new(false),
            }
        }
        
        #[inline(always)]
        pub fn set(&self) {
            // Use Release to ensure visibility
            self.flag.store(true, MemoryOrderingGuide::FLAG_SET);
        }
        
        #[inline(always)]
        pub fn is_set(&self) -> bool {
            // Use Acquire to see all prior writes
            self.flag.load(MemoryOrderingGuide::FLAG_READ)
        }
        
        #[inline(always)]
        pub fn test_and_set(&self) -> bool {
            // Use AcqRel for read-modify-write
            self.flag.swap(true, Ordering::AcqRel)
        }
    }
    
    /// Optimized sequence number pattern
    pub struct OptimizedSequence {
        seq: AtomicU64,
    }
    
    impl OptimizedSequence {
        pub fn new() -> Self {
            Self {
                seq: AtomicU64::new(0),
            }
        }
        
        #[inline(always)]
        pub fn next(&self) -> u64 {
            // Use AcqRel for sequence generation
            self.seq.fetch_add(1, MemoryOrderingGuide::VERSION_INCREMENT)
        }
        
        #[inline(always)]
        pub fn current(&self) -> u64 {
            // Use Acquire to ensure ordering
            self.seq.load(MemoryOrderingGuide::VERSION_READ)
        }
    }
}

/// Memory fence utilities for cross-thread synchronization
pub mod fences {
    use std::sync::atomic::{fence, compiler_fence, Ordering};
    
    /// Insert a compiler fence to prevent reordering
    #[inline(always)]
    pub fn compiler_barrier() {
        compiler_fence(Ordering::SeqCst);
    }
    
    /// Insert an acquire fence
    #[inline(always)]
    pub fn acquire_fence() {
        fence(Ordering::Acquire);
    }
    
    /// Insert a release fence
    #[inline(always)]
    pub fn release_fence() {
        fence(Ordering::Release);
    }
    
    /// Insert a full memory fence
    #[inline(always)]
    pub fn full_fence() {
        fence(Ordering::SeqCst);
    }
}

/// Performance testing utilities
#[cfg(test)]
pub mod benchmarks {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;
    
    /// Benchmark different memory orderings
    pub fn benchmark_orderings() {
        let counter = AtomicU64::new(0);
        let iterations = 10_000_000;
        
        // Relaxed ordering
        let start = Instant::now();
        for _ in 0..iterations {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        let relaxed_time = start.elapsed();
        
        counter.store(0, Ordering::Relaxed);
        
        // Acquire-Release ordering
        let start = Instant::now();
        for _ in 0..iterations {
            counter.fetch_add(1, Ordering::AcqRel);
        }
        let acqrel_time = start.elapsed();
        
        counter.store(0, Ordering::Relaxed);
        
        // Sequential consistency
        let start = Instant::now();
        for _ in 0..iterations {
            counter.fetch_add(1, Ordering::SeqCst);
        }
        let seqcst_time = start.elapsed();
        
        println!("Memory Ordering Benchmark Results:");
        println!("  Relaxed: {:?}", relaxed_time);
        println!("  AcqRel:  {:?}", acqrel_time);
        println!("  SeqCst:  {:?}", seqcst_time);
        println!("  Speedup (Relaxed vs SeqCst): {:.2}x", 
                 seqcst_time.as_nanos() as f64 / relaxed_time.as_nanos() as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::patterns::*;
    
    #[test]
    fn test_optimized_counter() {
        let counter = OptimizedCounter::new();
        assert_eq!(counter.get(), 0);
        
        counter.increment();
        counter.increment();
        assert_eq!(counter.get(), 2);
    }
    
    #[test]
    fn test_optimized_flag() {
        let flag = OptimizedFlag::new();
        assert!(!flag.is_set());
        
        flag.set();
        assert!(flag.is_set());
        
        let was_set = flag.test_and_set();
        assert!(was_set);
    }
    
    #[test]
    fn test_optimized_sequence() {
        let seq = OptimizedSequence::new();
        assert_eq!(seq.current(), 0);
        
        let first = seq.next();
        assert_eq!(first, 0);
        assert_eq!(seq.current(), 1);
        
        let second = seq.next();
        assert_eq!(second, 1);
        assert_eq!(seq.current(), 2);
    }
}