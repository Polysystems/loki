//! Infrastructure modules for core system functionality

pub mod lockfree;

// Re-export commonly used lock-free types
pub use lockfree::{
    ConcurrentMap,
    AtomicConfig,
    ZeroCopyRingBuffer,
    SimdCacheLine,
    LockFreeEventQueue,
    Event,
    EventPriority,
};