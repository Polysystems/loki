//! Lock-free infrastructure components for high-performance concurrent operations
//! 
//! This module provides foundational lock-free data structures and utilities
//! that replace traditional mutex/rwlock-based synchronization primitives.

pub mod concurrent_map;
pub mod atomic_swap;
pub mod ring_buffer;
pub mod indexed_ring_buffer;
pub mod cross_scale_index;
pub mod atomic_context_analytics;
pub mod lockfree_context_learning;
pub mod simd_cache;
pub mod event_queue;
pub mod simd_pattern;
pub mod memory_ordering;
pub mod global_stats;

// Re-export commonly used types
pub use concurrent_map::{ConcurrentMap, ConcurrentMapRef};
pub use atomic_swap::{AtomicConfig, ConfigManager};
pub use ring_buffer::{ZeroCopyRingBuffer, RingBufferError};
pub use indexed_ring_buffer::{IndexedRingBuffer, IndexedBufferStats, HasTimestamp};
pub use cross_scale_index::{
    CrossScaleIndex, CrossScaleIndexConfig, ScaleIndexEntry, CorrelationData, CorrelationType,
    IndexEntryMetadata, CrossScaleIndexStats, current_timestamp_nanos
};
pub use atomic_context_analytics::{AtomicContextAnalytics, ContextAnalyticsSnapshot, ContextErrorType};
pub use lockfree_context_learning::{LockFreeContextLearningSystem, LockFreeLearningConfig, LearningSystemStats, TrainingExample};
pub use simd_cache::{SimdCacheLine, SimdAlignedData};
pub use event_queue::{LockFreeEventQueue, Event, EventPriority, EventRouter};
pub use simd_pattern::{SimdPatternMatcher, SimdStringOps};
pub use memory_ordering::{MemoryOrderingGuide, patterns, fences};
pub use global_stats::{GLOBAL_STATS, GlobalStats, StatsSnapshot, TimedOperation};