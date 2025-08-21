//! Global statistics collection for lock-free infrastructure
//! 
//! Provides centralized metrics collection for all lock-free components
//! with minimal performance overhead.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Global statistics collector
pub struct GlobalStats {
    // Operation counts
    total_operations: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    queue_pushes: AtomicU64,
    queue_pops: AtomicU64,
    map_inserts: AtomicU64,
    map_removes: AtomicU64,
    map_lookups: AtomicU64,
    
    // Performance metrics
    contentions: AtomicU64,
    retries: AtomicU64,
    failed_operations: AtomicU64,
    
    // SIMD operation counts
    simd_operations: AtomicU64,
    simd_pattern_matches: AtomicU64,
    
    // Component-specific stats
    component_stats: Arc<DashMap<String, ComponentStats>>,
    
    // Timing information
    start_time: Instant,
}

/// Per-component statistics
#[derive(Default)]
pub struct ComponentStats {
    operations: AtomicU64,
    successes: AtomicU64,
    failures: AtomicU64,
    total_latency_ns: AtomicU64,
    min_latency_ns: AtomicU64,
    max_latency_ns: AtomicU64,
}

impl GlobalStats {
    /// Create new global statistics collector
    pub fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            queue_pushes: AtomicU64::new(0),
            queue_pops: AtomicU64::new(0),
            map_inserts: AtomicU64::new(0),
            map_removes: AtomicU64::new(0),
            map_lookups: AtomicU64::new(0),
            contentions: AtomicU64::new(0),
            retries: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            simd_operations: AtomicU64::new(0),
            simd_pattern_matches: AtomicU64::new(0),
            component_stats: Arc::new(DashMap::new()),
            start_time: Instant::now(),
        }
    }
    
    /// Record a generic operation
    #[inline(always)]
    pub fn record_operation(&self) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record cache hit
    #[inline(always)]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record cache miss
    #[inline(always)]
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record queue push
    #[inline(always)]
    pub fn record_queue_push(&self) {
        self.queue_pushes.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record queue pop
    #[inline(always)]
    pub fn record_queue_pop(&self) {
        self.queue_pops.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record map insert
    #[inline(always)]
    pub fn record_map_insert(&self) {
        self.map_inserts.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record map remove
    #[inline(always)]
    pub fn record_map_remove(&self) {
        self.map_removes.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record map lookup
    #[inline(always)]
    pub fn record_map_lookup(&self) {
        self.map_lookups.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record contention event
    #[inline(always)]
    pub fn record_contention(&self) {
        self.contentions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record retry
    #[inline(always)]
    pub fn record_retry(&self) {
        self.retries.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record failed operation
    #[inline(always)]
    pub fn record_failure(&self) {
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record SIMD operation
    #[inline(always)]
    pub fn record_simd_operation(&self) {
        self.simd_operations.fetch_add(1, Ordering::Relaxed);
        self.record_operation();
    }
    
    /// Record SIMD pattern match
    #[inline(always)]
    pub fn record_simd_pattern_match(&self) {
        self.simd_pattern_matches.fetch_add(1, Ordering::Relaxed);
        self.record_simd_operation();
    }
    
    /// Record component-specific operation with latency
    pub fn record_component_operation(&self, component: &str, latency: Duration, success: bool) {
        let stats = self.component_stats
            .entry(component.to_string())
            .or_insert_with(|| ComponentStats::default());
        
        stats.operations.fetch_add(1, Ordering::Relaxed);
        
        if success {
            stats.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            stats.failures.fetch_add(1, Ordering::Relaxed);
        }
        
        let latency_ns = latency.as_nanos() as u64;
        stats.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        
        // Update min/max (non-atomic, best effort)
        let current_min = stats.min_latency_ns.load(Ordering::Relaxed);
        if current_min == 0 || latency_ns < current_min {
            stats.min_latency_ns.store(latency_ns, Ordering::Relaxed);
        }
        
        let current_max = stats.max_latency_ns.load(Ordering::Relaxed);
        if latency_ns > current_max {
            stats.max_latency_ns.store(latency_ns, Ordering::Relaxed);
        }
        
        self.record_operation();
    }
    
    /// Get current statistics snapshot
    pub fn snapshot(&self) -> StatsSnapshot {
        let uptime = self.start_time.elapsed();
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        
        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };
        
        let ops_per_sec = if uptime.as_secs() > 0 {
            total_ops as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };
        
        StatsSnapshot {
            uptime,
            total_operations: total_ops,
            operations_per_second: ops_per_sec,
            cache_hits,
            cache_misses,
            cache_hit_rate,
            queue_pushes: self.queue_pushes.load(Ordering::Relaxed),
            queue_pops: self.queue_pops.load(Ordering::Relaxed),
            map_inserts: self.map_inserts.load(Ordering::Relaxed),
            map_removes: self.map_removes.load(Ordering::Relaxed),
            map_lookups: self.map_lookups.load(Ordering::Relaxed),
            contentions: self.contentions.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            failed_operations: self.failed_operations.load(Ordering::Relaxed),
            simd_operations: self.simd_operations.load(Ordering::Relaxed),
            simd_pattern_matches: self.simd_pattern_matches.load(Ordering::Relaxed),
            component_stats: self.get_component_snapshots(),
        }
    }
    
    /// Get component statistics snapshots
    fn get_component_snapshots(&self) -> Vec<ComponentSnapshot> {
        self.component_stats
            .iter()
            .map(|entry| {
                let name = entry.key().clone();
                let stats = entry.value();
                let operations = stats.operations.load(Ordering::Relaxed);
                let total_latency = stats.total_latency_ns.load(Ordering::Relaxed);
                
                ComponentSnapshot {
                    name,
                    operations,
                    successes: stats.successes.load(Ordering::Relaxed),
                    failures: stats.failures.load(Ordering::Relaxed),
                    avg_latency_ns: if operations > 0 {
                        total_latency / operations
                    } else {
                        0
                    },
                    min_latency_ns: stats.min_latency_ns.load(Ordering::Relaxed),
                    max_latency_ns: stats.max_latency_ns.load(Ordering::Relaxed),
                }
            })
            .collect()
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        self.total_operations.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.queue_pushes.store(0, Ordering::Relaxed);
        self.queue_pops.store(0, Ordering::Relaxed);
        self.map_inserts.store(0, Ordering::Relaxed);
        self.map_removes.store(0, Ordering::Relaxed);
        self.map_lookups.store(0, Ordering::Relaxed);
        self.contentions.store(0, Ordering::Relaxed);
        self.retries.store(0, Ordering::Relaxed);
        self.failed_operations.store(0, Ordering::Relaxed);
        self.simd_operations.store(0, Ordering::Relaxed);
        self.simd_pattern_matches.store(0, Ordering::Relaxed);
        self.component_stats.clear();
    }
}

/// Statistics snapshot
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    pub uptime: Duration,
    pub total_operations: u64,
    pub operations_per_second: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub queue_pushes: u64,
    pub queue_pops: u64,
    pub map_inserts: u64,
    pub map_removes: u64,
    pub map_lookups: u64,
    pub contentions: u64,
    pub retries: u64,
    pub failed_operations: u64,
    pub simd_operations: u64,
    pub simd_pattern_matches: u64,
    pub component_stats: Vec<ComponentSnapshot>,
}

/// Component statistics snapshot
#[derive(Debug, Clone)]
pub struct ComponentSnapshot {
    pub name: String,
    pub operations: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_latency_ns: u64,
    pub min_latency_ns: u64,
    pub max_latency_ns: u64,
}

impl StatsSnapshot {
    /// Print formatted statistics
    pub fn print(&self) {
        println!("=== Lock-Free Infrastructure Statistics ===");
        println!("Uptime: {:?}", self.uptime);
        println!("Total Operations: {} ({:.2} ops/sec)", 
                 self.total_operations, self.operations_per_second);
        println!();
        
        println!("Cache Statistics:");
        println!("  Hits: {} ({:.2}%)", self.cache_hits, self.cache_hit_rate * 100.0);
        println!("  Misses: {}", self.cache_misses);
        println!();
        
        println!("Queue Operations:");
        println!("  Pushes: {}", self.queue_pushes);
        println!("  Pops: {}", self.queue_pops);
        println!();
        
        println!("Map Operations:");
        println!("  Inserts: {}", self.map_inserts);
        println!("  Removes: {}", self.map_removes);
        println!("  Lookups: {}", self.map_lookups);
        println!();
        
        println!("Performance:");
        println!("  Contentions: {}", self.contentions);
        println!("  Retries: {}", self.retries);
        println!("  Failed Operations: {}", self.failed_operations);
        println!();
        
        println!("SIMD Operations:");
        println!("  Total: {}", self.simd_operations);
        println!("  Pattern Matches: {}", self.simd_pattern_matches);
        println!();
        
        if !self.component_stats.is_empty() {
            println!("Component Statistics:");
            for comp in &self.component_stats {
                println!("  {}:", comp.name);
                println!("    Operations: {} (Success: {}, Failure: {})",
                         comp.operations, comp.successes, comp.failures);
                println!("    Latency: avg={:.2}µs, min={:.2}µs, max={:.2}µs",
                         comp.avg_latency_ns as f64 / 1000.0,
                         comp.min_latency_ns as f64 / 1000.0,
                         comp.max_latency_ns as f64 / 1000.0);
            }
        }
    }
}

use once_cell::sync::Lazy;

/// Global statistics instance
pub static GLOBAL_STATS: Lazy<GlobalStats> = Lazy::new(|| GlobalStats::new());

/// Timed operation helper
pub struct TimedOperation<'a> {
    component: &'a str,
    start: Instant,
}

impl<'a> TimedOperation<'a> {
    /// Start timing an operation
    pub fn start(component: &'a str) -> Self {
        Self {
            component,
            start: Instant::now(),
        }
    }
    
    /// Complete the operation successfully
    pub fn success(self) {
        let latency = self.start.elapsed();
        GLOBAL_STATS.record_component_operation(self.component, latency, true);
    }
    
    /// Complete the operation with failure
    pub fn failure(self) {
        let latency = self.start.elapsed();
        GLOBAL_STATS.record_component_operation(self.component, latency, false);
    }
}

/// Macro for timing operations
#[macro_export]
macro_rules! timed_operation {
    ($component:expr, $op:expr) => {{
        let timer = $crate::infrastructure::lockfree::global_stats::TimedOperation::start($component);
        let result = $op;
        if result.is_ok() {
            timer.success();
        } else {
            timer.failure();
        }
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_global_stats() {
        let stats = GlobalStats::new();
        
        stats.record_operation();
        stats.record_cache_hit();
        stats.record_cache_miss();
        
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_operations, 3);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.cache_hit_rate, 0.5);
    }
    
    #[test]
    fn test_component_stats() {
        let stats = GlobalStats::new();
        
        stats.record_component_operation("test_component", Duration::from_micros(100), true);
        stats.record_component_operation("test_component", Duration::from_micros(200), true);
        stats.record_component_operation("test_component", Duration::from_micros(150), false);
        
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.component_stats.len(), 1);
        
        let comp = &snapshot.component_stats[0];
        assert_eq!(comp.name, "test_component");
        assert_eq!(comp.operations, 3);
        assert_eq!(comp.successes, 2);
        assert_eq!(comp.failures, 1);
    }
}