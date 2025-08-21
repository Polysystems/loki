//! Lock-free atomic context analytics for high-performance metrics tracking
//! Replaces RwLock-based analytics with atomic operations for zero contention

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use serde::{Serialize, Deserialize};

/// Lock-free context analytics using atomic operations
#[derive(Debug)]
pub struct AtomicContextAnalytics {
    /// Core performance metrics
    metrics: Arc<AtomicPerformanceMetrics>,
    
    /// Quality tracking
    quality: Arc<AtomicQualityMetrics>,
    
    /// Cache performance
    cache: Arc<AtomicCacheMetrics>,
    
    /// Processing throughput
    throughput: Arc<AtomicThroughputMetrics>,
    
    /// Error tracking
    errors: Arc<AtomicErrorMetrics>,
    
    /// Creation timestamp
    created_at: Instant,
}

/// Atomic performance metrics
#[derive(Debug, Default)]
pub struct AtomicPerformanceMetrics {
    /// Total processing time in microseconds
    pub total_processing_time_us: AtomicU64,
    
    /// Number of processed items
    pub processed_count: AtomicU64,
    
    /// Average latency in microseconds (calculated as total_time / count)
    pub average_latency_us: AtomicU64,
    
    /// Peak processing latency in microseconds
    pub peak_latency_us: AtomicU64,
    
    /// Rolling window sum for recent average (last 1000 operations)
    pub rolling_sum_us: AtomicU64,
    
    /// Rolling window count for recent average
    pub rolling_count: AtomicUsize,
    
    /// Last update timestamp (nanos since epoch)
    pub last_update_nanos: AtomicU64,
}

/// Atomic quality metrics
#[derive(Debug, Default)]
pub struct AtomicQualityMetrics {
    /// Sum of quality scores (scaled by 10000 for precision)
    pub quality_score_sum_x10000: AtomicU64,
    
    /// Number of quality evaluations
    pub quality_evaluation_count: AtomicU64,
    
    /// Average quality score (scaled by 10000)
    pub average_quality_x10000: AtomicU64,
    
    /// Quality trend: 0=declining, 1=stable, 2=improving (rolling 100 samples)
    pub quality_trend: AtomicU64,
    
    /// Count of high quality items (score > 0.8)
    pub high_quality_count: AtomicU64,
    
    /// Count of low quality items (score < 0.4)
    pub low_quality_count: AtomicU64,
    
    /// Peak quality score seen (scaled by 10000)
    pub peak_quality_x10000: AtomicU64,
}

/// Atomic cache metrics
#[derive(Debug, Default)]
pub struct AtomicCacheMetrics {
    /// Cache hit count
    pub cache_hits: AtomicU64,
    
    /// Cache miss count
    pub cache_misses: AtomicU64,
    
    /// Cache eviction count
    pub cache_evictions: AtomicU64,
    
    /// Total cache lookup time in microseconds
    pub total_lookup_time_us: AtomicU64,
    
    /// Average lookup time in microseconds
    pub average_lookup_time_us: AtomicU64,
    
    /// Cache size (current entries)
    pub current_cache_size: AtomicUsize,
    
    /// Cache utilization percentage (scaled by 100)
    pub cache_utilization_x100: AtomicU64,
}

/// Atomic throughput metrics
#[derive(Debug, Default)]
pub struct AtomicThroughputMetrics {
    /// Items processed in current second
    pub current_second_count: AtomicU64,
    
    /// Peak throughput (items per second)
    pub peak_throughput: AtomicU64,
    
    /// Rolling average throughput (items per second, scaled by 100)
    pub rolling_avg_throughput_x100: AtomicU64,
    
    /// Current second timestamp
    pub current_second_timestamp: AtomicU64,
    
    /// Total bytes processed
    pub total_bytes_processed: AtomicU64,
    
    /// Bandwidth in bytes per second
    pub bandwidth_bytes_per_sec: AtomicU64,
}

/// Atomic error metrics
#[derive(Debug, Default)]
pub struct AtomicErrorMetrics {
    /// Total error count
    pub total_errors: AtomicU64,
    
    /// Processing errors
    pub processing_errors: AtomicU64,
    
    /// Timeout errors
    pub timeout_errors: AtomicU64,
    
    /// Memory errors
    pub memory_errors: AtomicU64,
    
    /// Network errors
    pub network_errors: AtomicU64,
    
    /// Validation errors
    pub validation_errors: AtomicU64,
    
    /// Error rate per thousand operations (scaled by 1000)
    pub error_rate_x1000: AtomicU64,
    
    /// Time since last error (seconds)
    pub seconds_since_last_error: AtomicU64,
}

impl AtomicContextAnalytics {
    /// Create a new atomic context analytics instance
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(AtomicPerformanceMetrics::default()),
            quality: Arc::new(AtomicQualityMetrics::default()),
            cache: Arc::new(AtomicCacheMetrics::default()),
            throughput: Arc::new(AtomicThroughputMetrics::default()),
            errors: Arc::new(AtomicErrorMetrics::default()),
            created_at: Instant::now(),
        }
    }
    
    /// Record processing latency
    pub fn record_processing_latency(&self, latency: Duration) {
        let latency_us = latency.as_micros() as u64;
        
        // Update total processing time and count
        self.metrics.total_processing_time_us.fetch_add(latency_us, Ordering::Relaxed);
        let count = self.metrics.processed_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Calculate and store average latency
        let total_time = self.metrics.total_processing_time_us.load(Ordering::Relaxed);
        let avg_latency = if count > 0 { total_time / count } else { 0 };
        self.metrics.average_latency_us.store(avg_latency, Ordering::Relaxed);
        
        // Update peak latency
        let current_peak = self.metrics.peak_latency_us.load(Ordering::Relaxed);
        if latency_us > current_peak {
            self.metrics.peak_latency_us.compare_exchange_weak(
                current_peak, latency_us, Ordering::Relaxed, Ordering::Relaxed
            ).ok(); // Ignore race condition - another thread may have updated it
        }
        
        // Update rolling window (last 1000 operations)
        const ROLLING_WINDOW_SIZE: usize = 1000;
        let rolling_count = self.metrics.rolling_count.load(Ordering::Relaxed);
        
        if rolling_count < ROLLING_WINDOW_SIZE {
            self.metrics.rolling_sum_us.fetch_add(latency_us, Ordering::Relaxed);
            self.metrics.rolling_count.fetch_add(1, Ordering::Relaxed);
        } else {
            // Reset rolling window periodically
            if rolling_count >= ROLLING_WINDOW_SIZE * 2 {
                self.metrics.rolling_sum_us.store(latency_us, Ordering::Relaxed);
                self.metrics.rolling_count.store(1, Ordering::Relaxed);
            } else {
                self.metrics.rolling_sum_us.fetch_add(latency_us, Ordering::Relaxed);
                self.metrics.rolling_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Update timestamp
        let now_nanos = current_timestamp_nanos();
        self.metrics.last_update_nanos.store(now_nanos, Ordering::Relaxed);
        
        // Update throughput
        self.update_throughput();
    }
    
    /// Record quality score
    pub fn record_quality_score(&self, score: f32) {
        let score_x10000 = (score.clamp(0.0, 1.0) * 10000.0) as u64;
        
        // Update sum and count
        self.quality.quality_score_sum_x10000.fetch_add(score_x10000, Ordering::Relaxed);
        let count = self.quality.quality_evaluation_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Calculate average
        let total_score = self.quality.quality_score_sum_x10000.load(Ordering::Relaxed);
        let avg_score = if count > 0 { total_score / count } else { 0 };
        self.quality.average_quality_x10000.store(avg_score, Ordering::Relaxed);
        
        // Update peak quality
        let current_peak = self.quality.peak_quality_x10000.load(Ordering::Relaxed);
        if score_x10000 > current_peak {
            self.quality.peak_quality_x10000.compare_exchange_weak(
                current_peak, score_x10000, Ordering::Relaxed, Ordering::Relaxed
            ).ok();
        }
        
        // Update quality category counts
        if score > 0.8 {
            self.quality.high_quality_count.fetch_add(1, Ordering::Relaxed);
        } else if score < 0.4 {
            self.quality.low_quality_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update quality trend (simplified: compare to running average)
        let current_avg = avg_score as f32 / 10000.0;
        if score > current_avg * 1.1 {
            self.quality.quality_trend.store(2, Ordering::Relaxed); // Improving
        } else if score < current_avg * 0.9 {
            self.quality.quality_trend.store(0, Ordering::Relaxed); // Declining  
        } else {
            self.quality.quality_trend.store(1, Ordering::Relaxed); // Stable
        }
    }
    
    /// Record cache hit
    pub fn record_cache_hit(&self, lookup_time: Duration) {
        self.cache.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.record_cache_lookup_time(lookup_time);
    }
    
    /// Record cache miss
    pub fn record_cache_miss(&self, lookup_time: Duration) {
        self.cache.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.record_cache_lookup_time(lookup_time);
    }
    
    /// Record cache eviction
    pub fn record_cache_eviction(&self) {
        self.cache.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update cache size
    pub fn update_cache_size(&self, size: usize, max_size: usize) {
        self.cache.current_cache_size.store(size, Ordering::Relaxed);
        let utilization = if max_size > 0 {
            ((size as f64 / max_size as f64) * 100.0) as u64
        } else {
            0
        };
        self.cache.cache_utilization_x100.store(utilization, Ordering::Relaxed);
    }
    
    /// Record different types of errors
    pub fn record_error(&self, error_type: ContextErrorType) {
        self.errors.total_errors.fetch_add(1, Ordering::Relaxed);
        
        match error_type {
            ContextErrorType::Processing => {
                self.errors.processing_errors.fetch_add(1, Ordering::Relaxed);
            }
            ContextErrorType::Timeout => {
                self.errors.timeout_errors.fetch_add(1, Ordering::Relaxed);
            }
            ContextErrorType::Memory => {
                self.errors.memory_errors.fetch_add(1, Ordering::Relaxed);
            }
            ContextErrorType::Network => {
                self.errors.network_errors.fetch_add(1, Ordering::Relaxed);
            }
            ContextErrorType::Validation => {
                self.errors.validation_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Update error rate (errors per thousand operations)
        let total_errors = self.errors.total_errors.load(Ordering::Relaxed);
        let total_operations = self.metrics.processed_count.load(Ordering::Relaxed);
        
        if total_operations > 0 {
            let error_rate = (total_errors * 1000) / total_operations;
            self.errors.error_rate_x1000.store(error_rate, Ordering::Relaxed);
        }
        
        // Reset seconds since last error
        self.errors.seconds_since_last_error.store(0, Ordering::Relaxed);
    }
    
    /// Record bytes processed
    pub fn record_bytes_processed(&self, bytes: usize) {
        self.throughput.total_bytes_processed.fetch_add(bytes as u64, Ordering::Relaxed);
        self.update_bandwidth();
    }
    
    /// Get current cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        }
    }
    
    /// Get current average quality score
    pub fn average_quality_score(&self) -> f32 {
        let avg_x10000 = self.quality.average_quality_x10000.load(Ordering::Relaxed);
        avg_x10000 as f32 / 10000.0
    }
    
    /// Get current average latency
    pub fn average_latency(&self) -> Duration {
        let avg_us = self.metrics.average_latency_us.load(Ordering::Relaxed);
        Duration::from_micros(avg_us)
    }
    
    /// Get recent average latency (rolling window)
    pub fn recent_average_latency(&self) -> Duration {
        let rolling_sum = self.metrics.rolling_sum_us.load(Ordering::Relaxed);
        let rolling_count = self.metrics.rolling_count.load(Ordering::Relaxed);
        
        if rolling_count > 0 {
            Duration::from_micros(rolling_sum / rolling_count as u64)
        } else {
            Duration::ZERO
        }
    }
    
    /// Get current throughput (items per second)
    pub fn current_throughput(&self) -> f64 {
        let throughput_x100 = self.throughput.rolling_avg_throughput_x100.load(Ordering::Relaxed);
        throughput_x100 as f64 / 100.0
    }
    
    /// Get error rate
    pub fn error_rate(&self) -> f32 {
        let rate_x1000 = self.errors.error_rate_x1000.load(Ordering::Relaxed);
        rate_x1000 as f32 / 1000.0
    }
    
    /// Get comprehensive statistics snapshot
    pub fn get_stats_snapshot(&self) -> ContextAnalyticsSnapshot {
        let elapsed = self.created_at.elapsed();
        
        ContextAnalyticsSnapshot {
            // Performance metrics
            total_processed: self.metrics.processed_count.load(Ordering::Relaxed),
            average_latency_ms: self.average_latency().as_millis() as f32,
            recent_latency_ms: self.recent_average_latency().as_millis() as f32,
            peak_latency_ms: Duration::from_micros(
                self.metrics.peak_latency_us.load(Ordering::Relaxed)
            ).as_millis() as f32,
            
            // Quality metrics
            average_quality: self.average_quality_score(),
            peak_quality: self.quality.peak_quality_x10000.load(Ordering::Relaxed) as f32 / 10000.0,
            high_quality_ratio: self.calculate_high_quality_ratio(),
            quality_trend: self.quality.quality_trend.load(Ordering::Relaxed),
            
            // Cache metrics  
            cache_hit_rate: self.cache_hit_rate(),
            cache_size: self.cache.current_cache_size.load(Ordering::Relaxed),
            cache_utilization: self.cache.cache_utilization_x100.load(Ordering::Relaxed) as f32 / 100.0,
            
            // Throughput metrics
            current_throughput: self.current_throughput(),
            peak_throughput: self.throughput.peak_throughput.load(Ordering::Relaxed) as f32,
            total_bytes: self.throughput.total_bytes_processed.load(Ordering::Relaxed),
            bandwidth_mbps: self.get_bandwidth_mbps(),
            
            // Error metrics
            total_errors: self.errors.total_errors.load(Ordering::Relaxed),
            error_rate: self.error_rate(),
            seconds_since_last_error: self.errors.seconds_since_last_error.load(Ordering::Relaxed),
            
            // System metrics
            uptime_seconds: elapsed.as_secs(),
        }
    }
    
    /// Private helper methods
    fn record_cache_lookup_time(&self, lookup_time: Duration) {
        let lookup_us = lookup_time.as_micros() as u64;
        self.cache.total_lookup_time_us.fetch_add(lookup_us, Ordering::Relaxed);
        
        let total_lookups = self.cache.cache_hits.load(Ordering::Relaxed) + 
                           self.cache.cache_misses.load(Ordering::Relaxed);
        
        if total_lookups > 0 {
            let total_time = self.cache.total_lookup_time_us.load(Ordering::Relaxed);
            let avg_time = total_time / total_lookups;
            self.cache.average_lookup_time_us.store(avg_time, Ordering::Relaxed);
        }
    }
    
    fn update_throughput(&self) {
        let now_secs = current_timestamp_nanos() / 1_000_000_000;
        let current_second = self.throughput.current_second_timestamp.load(Ordering::Relaxed);
        
        if now_secs != current_second {
            // New second - calculate throughput for previous second
            let count_in_second = self.throughput.current_second_count.load(Ordering::Relaxed);
            
            // Update peak throughput
            let current_peak = self.throughput.peak_throughput.load(Ordering::Relaxed);
            if count_in_second > current_peak {
                self.throughput.peak_throughput.compare_exchange_weak(
                    current_peak, count_in_second, Ordering::Relaxed, Ordering::Relaxed
                ).ok();
            }
            
            // Update rolling average (simple exponential moving average)
            let current_avg = self.throughput.rolling_avg_throughput_x100.load(Ordering::Relaxed);
            let alpha = 0.1; // Smoothing factor
            let new_avg = ((1.0 - alpha) * current_avg as f64 + 
                          alpha * (count_in_second as f64 * 100.0)) as u64;
            self.throughput.rolling_avg_throughput_x100.store(new_avg, Ordering::Relaxed);
            
            // Reset for new second
            self.throughput.current_second_timestamp.store(now_secs, Ordering::Relaxed);
            self.throughput.current_second_count.store(1, Ordering::Relaxed);
        } else {
            self.throughput.current_second_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    fn update_bandwidth(&self) {
        let elapsed_secs = self.created_at.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            let total_bytes = self.throughput.total_bytes_processed.load(Ordering::Relaxed);
            let bandwidth = (total_bytes as f64 / elapsed_secs) as u64;
            self.throughput.bandwidth_bytes_per_sec.store(bandwidth, Ordering::Relaxed);
        }
    }
    
    fn calculate_high_quality_ratio(&self) -> f32 {
        let high_quality = self.quality.high_quality_count.load(Ordering::Relaxed);
        let total = self.quality.quality_evaluation_count.load(Ordering::Relaxed);
        
        if total > 0 {
            high_quality as f32 / total as f32
        } else {
            0.0
        }
    }
    
    fn get_bandwidth_mbps(&self) -> f32 {
        let bandwidth_bps = self.throughput.bandwidth_bytes_per_sec.load(Ordering::Relaxed);
        (bandwidth_bps as f32) / (1024.0 * 1024.0)
    }
}

impl Clone for AtomicContextAnalytics {
    fn clone(&self) -> Self {
        // Clone creates a new instance with same configuration but reset metrics
        Self::new()
    }
}

impl Default for AtomicContextAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of context processing errors
#[derive(Debug, Clone, Copy)]
pub enum ContextErrorType {
    Processing,
    Timeout,
    Memory,
    Network,
    Validation,
}

/// Snapshot of analytics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalyticsSnapshot {
    // Performance
    pub total_processed: u64,
    pub average_latency_ms: f32,
    pub recent_latency_ms: f32,
    pub peak_latency_ms: f32,
    
    // Quality
    pub average_quality: f32,
    pub peak_quality: f32,
    pub high_quality_ratio: f32,
    pub quality_trend: u64, // 0=declining, 1=stable, 2=improving
    
    // Cache
    pub cache_hit_rate: f32,
    pub cache_size: usize,
    pub cache_utilization: f32,
    
    // Throughput
    pub current_throughput: f64,
    pub peak_throughput: f32,
    pub total_bytes: u64,
    pub bandwidth_mbps: f32,
    
    // Errors
    pub total_errors: u64,
    pub error_rate: f32,
    pub seconds_since_last_error: u64,
    
    // System
    pub uptime_seconds: u64,
}

/// Helper function to get current timestamp in nanoseconds
fn current_timestamp_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_performance_metrics() {
        let analytics = AtomicContextAnalytics::new();
        
        // Record some latencies
        analytics.record_processing_latency(Duration::from_millis(10));
        analytics.record_processing_latency(Duration::from_millis(20));
        analytics.record_processing_latency(Duration::from_millis(15));
        
        let avg = analytics.average_latency();
        assert!(avg.as_millis() >= 14 && avg.as_millis() <= 16);
        
        let snapshot = analytics.get_stats_snapshot();
        assert_eq!(snapshot.total_processed, 3);
        assert!(snapshot.peak_latency_ms >= 20.0);
    }
    
    #[test] 
    fn test_quality_metrics() {
        let analytics = AtomicContextAnalytics::new();
        
        // Record quality scores
        analytics.record_quality_score(0.8);
        analytics.record_quality_score(0.9);
        analytics.record_quality_score(0.3);
        
        let avg = analytics.average_quality_score();
        assert!(avg >= 0.66 && avg <= 0.68);
        
        let snapshot = analytics.get_stats_snapshot();
        assert_eq!(snapshot.peak_quality, 0.9);
        assert!(snapshot.high_quality_ratio >= 0.66); // 2 out of 3 are high quality
    }
    
    #[test]
    fn test_cache_metrics() {
        let analytics = AtomicContextAnalytics::new();
        
        // Record cache operations
        analytics.record_cache_hit(Duration::from_micros(100));
        analytics.record_cache_hit(Duration::from_micros(150));
        analytics.record_cache_miss(Duration::from_micros(1000));
        
        let hit_rate = analytics.cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01); // 2/3 hits
        
        analytics.update_cache_size(50, 100);
        let snapshot = analytics.get_stats_snapshot();
        assert_eq!(snapshot.cache_size, 50);
        assert_eq!(snapshot.cache_utilization, 0.5);
    }
    
    #[test]
    fn test_error_tracking() {
        let analytics = AtomicContextAnalytics::new();
        
        // Process some items and record errors
        for _ in 0..1000 {
            analytics.record_processing_latency(Duration::from_millis(10));
        }
        
        analytics.record_error(ContextErrorType::Processing);
        analytics.record_error(ContextErrorType::Timeout);
        
        let snapshot = analytics.get_stats_snapshot();
        assert_eq!(snapshot.total_errors, 2);
        assert_eq!(snapshot.error_rate, 0.002); // 2/1000
        assert_eq!(snapshot.seconds_since_last_error, 0);
    }
    
    #[test]
    fn test_concurrent_access() {
        let analytics = Arc::new(AtomicContextAnalytics::new());
        let num_threads = 4;
        let operations_per_thread = 1000;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..num_threads {
            let analytics_clone = analytics.clone();
            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    let latency = Duration::from_micros(10 + (i % 100));
                    analytics_clone.record_processing_latency(latency);
                    
                    let quality = 0.5 + ((thread_id + i) % 5) as f32 / 10.0;
                    analytics_clone.record_quality_score(quality);
                    
                    if i % 10 == 0 {
                        analytics_clone.record_cache_hit(Duration::from_micros(50));
                    } else {
                        analytics_clone.record_cache_miss(Duration::from_micros(200));
                    }
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let snapshot = analytics.get_stats_snapshot();
        assert_eq!(snapshot.total_processed, (num_threads * operations_per_thread) as u64);
        
        // Verify metrics are reasonable
        assert!(snapshot.average_latency_ms > 0.0);
        assert!(snapshot.average_quality > 0.0);
        assert!(snapshot.cache_hit_rate > 0.0);
    }
}