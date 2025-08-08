//! Adaptive Cache Controller
//!
//! Dynamically adjusts cache sizes based on workload patterns,
//! memory pressure, and performance metrics.

use std::sync::atomic::{AtomicUsize, AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{info, debug};
use serde::{Serialize, Deserialize};

/// Cache performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub level: usize,
    pub hit_rate: f32,
    pub miss_rate: f32,
    pub eviction_rate: f32,
    pub avg_access_time_us: f64,
    pub memory_usage_mb: usize,
    pub capacity_mb: usize,
    pub pressure_score: f32,
}

/// Historical metrics for trend analysis
#[derive(Clone, Debug)]
struct MetricsHistory {
    timestamps: VecDeque<Instant>,
    hit_rates: VecDeque<f32>,
    miss_rates: VecDeque<f32>,
    eviction_rates: VecDeque<f32>,
    access_times: VecDeque<f64>,
}

impl MetricsHistory {
    fn new(capacity: usize) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(capacity),
            hit_rates: VecDeque::with_capacity(capacity),
            miss_rates: VecDeque::with_capacity(capacity),
            eviction_rates: VecDeque::with_capacity(capacity),
            access_times: VecDeque::with_capacity(capacity),
        }
    }
    
    fn add_sample(&mut self, metrics: &CacheMetrics) {
        // Remove old samples if at capacity
        if self.timestamps.len() >= self.timestamps.capacity() {
            self.timestamps.pop_front();
            self.hit_rates.pop_front();
            self.miss_rates.pop_front();
            self.eviction_rates.pop_front();
            self.access_times.pop_front();
        }
        
        self.timestamps.push_back(Instant::now());
        self.hit_rates.push_back(metrics.hit_rate);
        self.miss_rates.push_back(metrics.miss_rate);
        self.eviction_rates.push_back(metrics.eviction_rate);
        self.access_times.push_back(metrics.avg_access_time_us);
    }
    
    fn get_trend(&self, values: &VecDeque<f32>) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        // Simple linear regression for trend
        let n = values.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        
        for (i, y) in values.iter().enumerate() {
            let x = i as f32;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        slope
    }
}

/// Adaptive sizing strategy
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SizingStrategy {
    /// Conservative - slow adjustments
    Conservative,
    /// Balanced - moderate adjustments  
    Balanced,
    /// Aggressive - fast adjustments
    Aggressive,
    /// Fixed - no adjustments
    Fixed,
}

/// Cache level configuration
#[derive(Clone, Debug)]
pub struct CacheLevelConfig {
    pub level: usize,
    pub min_size_mb: usize,
    pub max_size_mb: usize,
    pub target_hit_rate: f32,
    pub max_eviction_rate: f32,
}

/// Adaptive Cache Controller
pub struct AdaptiveCacheController {
    /// Current sizes for each cache level
    l1_size: AtomicUsize,
    l2_size: AtomicUsize,
    l3_size: AtomicUsize,
    
    /// Performance counters
    l1_hits: AtomicU64,
    l1_misses: AtomicU64,
    l1_evictions: AtomicU64,
    
    l2_hits: AtomicU64,
    l2_misses: AtomicU64,
    l2_evictions: AtomicU64,
    
    l3_hits: AtomicU64,
    l3_misses: AtomicU64,
    l3_evictions: AtomicU64,
    
    /// Access time tracking (in microseconds)
    l1_total_access_time: AtomicU64,
    l2_total_access_time: AtomicU64,
    l3_total_access_time: AtomicU64,
    
    /// Configuration
    l1config: CacheLevelConfig,
    l2config: CacheLevelConfig,
    l3config: CacheLevelConfig,
    
    /// Sizing strategy
    strategy: RwLock<SizingStrategy>,
    
    /// Historical metrics
    l1_history: RwLock<MetricsHistory>,
    l2_history: RwLock<MetricsHistory>,
    l3_history: RwLock<MetricsHistory>,
    
    /// Adaptation control
    last_adaptation: RwLock<Instant>,
    adaptation_interval: Duration,
    enabled: AtomicBool,
}

impl AdaptiveCacheController {
    pub fn new(
        l1config: CacheLevelConfig,
        l2config: CacheLevelConfig,
        l3config: CacheLevelConfig,
    ) -> Self {
        Self {
            l1_size: AtomicUsize::new(l1config.min_size_mb * 1024 * 1024),
            l2_size: AtomicUsize::new(l2config.min_size_mb * 1024 * 1024),
            l3_size: AtomicUsize::new(l3config.min_size_mb * 1024 * 1024),
            
            l1_hits: AtomicU64::new(0),
            l1_misses: AtomicU64::new(0),
            l1_evictions: AtomicU64::new(0),
            
            l2_hits: AtomicU64::new(0),
            l2_misses: AtomicU64::new(0),
            l2_evictions: AtomicU64::new(0),
            
            l3_hits: AtomicU64::new(0),
            l3_misses: AtomicU64::new(0),
            l3_evictions: AtomicU64::new(0),
            
            l1_total_access_time: AtomicU64::new(0),
            l2_total_access_time: AtomicU64::new(0),
            l3_total_access_time: AtomicU64::new(0),
            
            l1config,
            l2config,
            l3config,
            
            strategy: RwLock::new(SizingStrategy::Balanced),
            
            l1_history: RwLock::new(MetricsHistory::new(100)),
            l2_history: RwLock::new(MetricsHistory::new(100)),
            l3_history: RwLock::new(MetricsHistory::new(100)),
            
            last_adaptation: RwLock::new(Instant::now()),
            adaptation_interval: Duration::from_secs(30),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Record a cache hit
    pub fn record_hit(&self, level: usize, access_time_us: u64) {
        match level {
            1 => {
                self.l1_hits.fetch_add(1, Ordering::Relaxed);
                self.l1_total_access_time.fetch_add(access_time_us, Ordering::Relaxed);
            }
            2 => {
                self.l2_hits.fetch_add(1, Ordering::Relaxed);
                self.l2_total_access_time.fetch_add(access_time_us, Ordering::Relaxed);
            }
            3 => {
                self.l3_hits.fetch_add(1, Ordering::Relaxed);
                self.l3_total_access_time.fetch_add(access_time_us, Ordering::Relaxed);
            }
            _ => {}
        }
    }
    
    /// Record a cache miss
    pub fn record_miss(&self, level: usize) {
        match level {
            1 => { self.l1_misses.fetch_add(1, Ordering::Relaxed); }
            2 => { self.l2_misses.fetch_add(1, Ordering::Relaxed); }
            3 => { self.l3_misses.fetch_add(1, Ordering::Relaxed); }
            _ => {}
        }
    }
    
    /// Record an eviction
    pub fn record_eviction(&self, level: usize) {
        match level {
            1 => { self.l1_evictions.fetch_add(1, Ordering::Relaxed); }
            2 => { self.l2_evictions.fetch_add(1, Ordering::Relaxed); }
            3 => { self.l3_evictions.fetch_add(1, Ordering::Relaxed); }
            _ => {}
        }
    }
    
    /// Get current metrics for a cache level
    pub fn get_metrics(&self, level: usize) -> CacheMetrics {
        let (hits, misses, evictions, total_time, size, config) = match level {
            1 => (
                self.l1_hits.load(Ordering::Relaxed),
                self.l1_misses.load(Ordering::Relaxed),
                self.l1_evictions.load(Ordering::Relaxed),
                self.l1_total_access_time.load(Ordering::Relaxed),
                self.l1_size.load(Ordering::Relaxed),
                &self.l1config,
            ),
            2 => (
                self.l2_hits.load(Ordering::Relaxed),
                self.l2_misses.load(Ordering::Relaxed),
                self.l2_evictions.load(Ordering::Relaxed),
                self.l2_total_access_time.load(Ordering::Relaxed),
                self.l2_size.load(Ordering::Relaxed),
                &self.l2config,
            ),
            3 => (
                self.l3_hits.load(Ordering::Relaxed),
                self.l3_misses.load(Ordering::Relaxed),
                self.l3_evictions.load(Ordering::Relaxed),
                self.l3_total_access_time.load(Ordering::Relaxed),
                self.l3_size.load(Ordering::Relaxed),
                &self.l3config,
            ),
            _ => return CacheMetrics {
                level,
                hit_rate: 0.0,
                miss_rate: 0.0,
                eviction_rate: 0.0,
                avg_access_time_us: 0.0,
                memory_usage_mb: 0,
                capacity_mb: 0,
                pressure_score: 0.0,
            },
        };
        
        let total_accesses = hits + misses;
        let hit_rate = if total_accesses > 0 {
            hits as f32 / total_accesses as f32
        } else {
            0.0
        };
        
        let miss_rate = 1.0 - hit_rate;
        
        let eviction_rate = if total_accesses > 0 {
            evictions as f32 / total_accesses as f32
        } else {
            0.0
        };
        
        let avg_access_time_us = if hits > 0 {
            total_time as f64 / hits as f64
        } else {
            0.0
        };
        
        // Calculate pressure score (0-100)
        let hit_rate_pressure = ((config.target_hit_rate - hit_rate) * 100.0).max(0.0);
        let eviction_pressure = (eviction_rate / config.max_eviction_rate * 100.0).min(100.0);
        let pressure_score = (hit_rate_pressure + eviction_pressure) / 2.0;
        
        CacheMetrics {
            level,
            hit_rate,
            miss_rate,
            eviction_rate,
            avg_access_time_us,
            memory_usage_mb: size / 1024 / 1024,
            capacity_mb: size / 1024 / 1024,
            pressure_score,
        }
    }
    
    /// Get optimal cache size based on current metrics
    pub async fn get_optimal_size(&self, level: usize) -> usize {
        if !self.enabled.load(Ordering::Relaxed) {
            return self.get_current_size(level);
        }
        
        let metrics = self.get_metrics(level);
        let strategy = *self.strategy.read().await;
        
        // Get historical trend
        let trend = match level {
            1 => {
                let history = self.l1_history.read().await;
                history.get_trend(&history.hit_rates)
            }
            2 => {
                let history = self.l2_history.read().await;
                history.get_trend(&history.hit_rates)
            }
            3 => {
                let history = self.l3_history.read().await;
                history.get_trend(&history.hit_rates)
            }
            _ => 0.0,
        };
        
        let config = match level {
            1 => &self.l1config,
            2 => &self.l2config,
            3 => &self.l3config,
            _ => return self.get_current_size(level),
        };
        
        let current_size = self.get_current_size(level);
        let mut new_size = current_size;
        
        // Calculate adjustment factor based on strategy
        let adjustment_factor = match strategy {
            SizingStrategy::Conservative => 0.05,
            SizingStrategy::Balanced => 0.10,
            SizingStrategy::Aggressive => 0.20,
            SizingStrategy::Fixed => return current_size,
        };
        
        // Adjust based on pressure and trend
        if metrics.pressure_score > 50.0 {
            // High pressure - increase size
            let increase = (current_size as f32 * adjustment_factor * 
                           (metrics.pressure_score / 100.0)) as usize;
            new_size = current_size + increase;
        } else if metrics.pressure_score < 20.0 && trend > 0.0 {
            // Low pressure and improving trend - decrease size
            let decrease = (current_size as f32 * adjustment_factor * 0.5) as usize;
            new_size = current_size.saturating_sub(decrease);
        }
        
        // Apply bounds
        let min_size = config.min_size_mb * 1024 * 1024;
        let max_size = config.max_size_mb * 1024 * 1024;
        new_size = new_size.clamp(min_size, max_size);
        
        // Update stored size
        match level {
            1 => self.l1_size.store(new_size, Ordering::Relaxed),
            2 => self.l2_size.store(new_size, Ordering::Relaxed),
            3 => self.l3_size.store(new_size, Ordering::Relaxed),
            _ => {}
        }
        
        if new_size != current_size {
            info!(
                "Cache L{} size adjusted: {} MB -> {} MB (pressure: {:.1})",
                level,
                current_size / 1024 / 1024,
                new_size / 1024 / 1024,
                metrics.pressure_score
            );
        }
        
        new_size
    }
    
    /// Get current cache size
    pub fn get_current_size(&self, level: usize) -> usize {
        match level {
            1 => self.l1_size.load(Ordering::Relaxed),
            2 => self.l2_size.load(Ordering::Relaxed),
            3 => self.l3_size.load(Ordering::Relaxed),
            _ => 0,
        }
    }
    
    /// Run periodic adaptation
    pub async fn adapt(&self) -> Result<()> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(());
        }
        
        let mut last_adaptation = self.last_adaptation.write().await;
        
        if last_adaptation.elapsed() < self.adaptation_interval {
            return Ok(());
        }
        
        // Collect current metrics
        let l1_metrics = self.get_metrics(1);
        let l2_metrics = self.get_metrics(2);
        let l3_metrics = self.get_metrics(3);
        
        // Update history
        self.l1_history.write().await.add_sample(&l1_metrics);
        self.l2_history.write().await.add_sample(&l2_metrics);
        self.l3_history.write().await.add_sample(&l3_metrics);
        
        // Adapt sizes
        self.get_optimal_size(1).await;
        self.get_optimal_size(2).await;
        self.get_optimal_size(3).await;
        
        // Check for strategy adjustment
        self.adjust_strategy(&l1_metrics, &l2_metrics, &l3_metrics).await?;
        
        *last_adaptation = Instant::now();
        
        debug!(
            "Cache adaptation complete - L1: {:.1}% hit, L2: {:.1}% hit, L3: {:.1}% hit",
            l1_metrics.hit_rate * 100.0,
            l2_metrics.hit_rate * 100.0,
            l3_metrics.hit_rate * 100.0
        );
        
        Ok(())
    }
    
    /// Adjust sizing strategy based on overall performance
    async fn adjust_strategy(
        &self,
        l1: &CacheMetrics,
        l2: &CacheMetrics,
        l3: &CacheMetrics,
    ) -> Result<()> {
        let mut strategy = self.strategy.write().await;
        
        // Calculate overall system pressure
        let avg_pressure = (l1.pressure_score + l2.pressure_score + l3.pressure_score) / 3.0;
        
        // Adjust strategy based on pressure
        let new_strategy = if avg_pressure > 70.0 {
            SizingStrategy::Aggressive
        } else if avg_pressure > 40.0 {
            SizingStrategy::Balanced
        } else {
            SizingStrategy::Conservative
        };
        
        if *strategy != new_strategy {
            info!("Sizing strategy changed: {:?} -> {:?}", strategy, new_strategy);
            *strategy = new_strategy;
        }
        
        Ok(())
    }
    
    /// Reset all metrics
    pub async fn reset_metrics(&self) {
        self.l1_hits.store(0, Ordering::Relaxed);
        self.l1_misses.store(0, Ordering::Relaxed);
        self.l1_evictions.store(0, Ordering::Relaxed);
        self.l1_total_access_time.store(0, Ordering::Relaxed);
        
        self.l2_hits.store(0, Ordering::Relaxed);
        self.l2_misses.store(0, Ordering::Relaxed);
        self.l2_evictions.store(0, Ordering::Relaxed);
        self.l2_total_access_time.store(0, Ordering::Relaxed);
        
        self.l3_hits.store(0, Ordering::Relaxed);
        self.l3_misses.store(0, Ordering::Relaxed);
        self.l3_evictions.store(0, Ordering::Relaxed);
        self.l3_total_access_time.store(0, Ordering::Relaxed);
        
        self.l1_history.write().await.timestamps.clear();
        self.l2_history.write().await.timestamps.clear();
        self.l3_history.write().await.timestamps.clear();
        
        info!("Cache metrics reset");
    }
    
    /// Enable or disable adaptive sizing
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        info!("Adaptive cache sizing {}", if enabled { "enabled" } else { "disabled" });
    }
    
    /// Set sizing strategy
    pub async fn set_strategy(&self, strategy: SizingStrategy) {
        *self.strategy.write().await = strategy;
        info!("Cache sizing strategy set to {:?}", strategy);
    }
    
    /// Get comprehensive statistics
    pub async fn get_stats(&self) -> CacheControllerStats {
        let last_adaptation = self.last_adaptation.read().await;
        CacheControllerStats {
            l1_metrics: self.get_metrics(1),
            l2_metrics: self.get_metrics(2),
            l3_metrics: self.get_metrics(3),
            strategy: *self.strategy.read().await,
            enabled: self.enabled.load(Ordering::Relaxed),
            seconds_since_last_adaptation: last_adaptation.elapsed().as_secs_f64(),
        }
    }
}

/// Cache controller statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheControllerStats {
    pub l1_metrics: CacheMetrics,
    pub l2_metrics: CacheMetrics,
    pub l3_metrics: CacheMetrics,
    pub strategy: SizingStrategy,
    pub enabled: bool,
    /// Seconds since last adaptation
    pub seconds_since_last_adaptation: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_adaptive_sizing() {
        let l1config = CacheLevelConfig {
            level: 1,
            min_size_mb: 32,
            max_size_mb: 128,
            target_hit_rate: 0.90,
            max_eviction_rate: 0.05,
        };
        
        let l2config = CacheLevelConfig {
            level: 2,
            min_size_mb: 256,
            max_size_mb: 1024,
            target_hit_rate: 0.80,
            max_eviction_rate: 0.10,
        };
        
        let l3config = CacheLevelConfig {
            level: 3,
            min_size_mb: 1024,
            max_size_mb: 4096,
            target_hit_rate: 0.70,
            max_eviction_rate: 0.15,
        };
        
        let controller = AdaptiveCacheController::new(l1config, l2config, l3config);
        
        // Simulate poor hit rate
        for _ in 0..100 {
            controller.record_miss(1);
        }
        for _ in 0..50 {
            controller.record_hit(1, 10);
        }
        
        // Get optimal size - should increase due to low hit rate
        let initial_size = controller.get_current_size(1);
        let optimal_size = controller.get_optimal_size(1).await;
        
        assert!(optimal_size > initial_size);
    }
    
    #[test]
    fn test_metrics_calculation() {
        let l1config = CacheLevelConfig {
            level: 1,
            min_size_mb: 32,
            max_size_mb: 128,
            target_hit_rate: 0.90,
            max_eviction_rate: 0.05,
        };
        
        let l2config = CacheLevelConfig {
            level: 2,
            min_size_mb: 256,
            max_size_mb: 1024,
            target_hit_rate: 0.80,
            max_eviction_rate: 0.10,
        };
        
        let l3config = CacheLevelConfig {
            level: 3,
            min_size_mb: 1024,
            max_size_mb: 4096,
            target_hit_rate: 0.70,
            max_eviction_rate: 0.15,
        };
        
        let controller = AdaptiveCacheController::new(l1config, l2config, l3config);
        
        // Record some activity
        controller.record_hit(1, 5);
        controller.record_hit(1, 7);
        controller.record_hit(1, 6);
        controller.record_miss(1);
        controller.record_eviction(1);
        
        let metrics = controller.get_metrics(1);
        
        assert_eq!(metrics.hit_rate, 0.75); // 3 hits out of 4 accesses
        assert_eq!(metrics.miss_rate, 0.25);
        assert!((metrics.avg_access_time_us - 6.0).abs() < 0.1);
    }
} 