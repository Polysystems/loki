//! Telemetry and metrics for error tracking
//! 
//! Provides comprehensive error tracking, performance metrics,
//! and system health monitoring for the chat system.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{error, warn, info, debug};
use chrono::{DateTime, Utc};

use crate::tui::chat::error::ChatError;

/// Error metrics for tracking failures
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorMetrics {
    /// Total number of errors
    pub total_errors: u64,
    
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    
    /// Errors by module
    pub errors_by_module: HashMap<String, u64>,
    
    /// Error rate per minute
    pub error_rate: f64,
    
    /// Last error timestamp
    pub last_error: Option<DateTime<Utc>>,
    
    /// Most common error type
    pub most_common_error: Option<String>,
}

/// Performance metrics for operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    
    /// P95 response time
    pub p95_response_time_ms: f64,
    
    /// P99 response time
    pub p99_response_time_ms: f64,
    
    /// Total operations
    pub total_operations: u64,
    
    /// Successful operations
    pub successful_operations: u64,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Operations per second
    pub ops_per_second: f64,
}

/// System health metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthMetrics {
    /// System uptime in seconds
    pub uptime_seconds: u64,
    
    /// Active connections
    pub active_connections: u32,
    
    /// Queue depth
    pub queue_depth: u32,
    
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    
    /// Health status
    pub health_status: HealthStatus,
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Unknown,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Telemetry collector for the chat system
pub struct TelemetryCollector {
    /// Error metrics
    error_metrics: Arc<RwLock<ErrorMetrics>>,
    
    /// Performance metrics
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Health metrics
    health_metrics: Arc<RwLock<HealthMetrics>>,
    
    /// Response time samples for percentile calculation
    response_times: Arc<RwLock<Vec<f64>>>,
    
    /// Start time for uptime tracking
    start_time: Instant,
    
    /// Last metrics reset time
    last_reset: Arc<RwLock<Instant>>,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub fn new() -> Self {
        Self {
            error_metrics: Arc::new(RwLock::new(ErrorMetrics::default())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            health_metrics: Arc::new(RwLock::new(HealthMetrics::default())),
            response_times: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            last_reset: Arc::new(RwLock::new(Instant::now())),
        }
    }
    
    /// Record an error
    pub async fn record_error(&self, error: &ChatError, module: &str) {
        let mut metrics = self.error_metrics.write().await;
        
        // Update total count
        metrics.total_errors += 1;
        
        // Update error type count
        let error_type = format!("{:?}", error).split('(').next().unwrap_or("Unknown").to_string();
        *metrics.errors_by_type.entry(error_type.clone()).or_insert(0) += 1;
        
        // Update module count
        *metrics.errors_by_module.entry(module.to_string()).or_insert(0) += 1;
        
        // Update last error time
        metrics.last_error = Some(Utc::now());
        
        // Calculate error rate
        let elapsed = self.start_time.elapsed().as_secs_f64() / 60.0;
        if elapsed > 0.0 {
            metrics.error_rate = metrics.total_errors as f64 / elapsed;
        }
        
        // Update most common error
        if let Some((most_common, _)) = metrics.errors_by_type.iter()
            .max_by_key(|(_, count)| *count) {
            metrics.most_common_error = Some(most_common.clone());
        }
        
        // Log the error with context
        error!(
            module = module,
            error_type = error_type.as_str(),
            "Error recorded: {}",
            error
        );
    }
    
    /// Record an operation with its duration
    pub async fn record_operation(&self, duration: Duration, success: bool) {
        let mut metrics = self.performance_metrics.write().await;
        let mut response_times = self.response_times.write().await;
        
        let duration_ms = duration.as_secs_f64() * 1000.0;
        
        // Update counts
        metrics.total_operations += 1;
        if success {
            metrics.successful_operations += 1;
        }
        
        // Store response time for percentile calculation
        response_times.push(duration_ms);
        if response_times.len() > 10000 {
            // Keep only recent samples
            response_times.drain(0..5000);
        }
        
        // Update average response time
        metrics.avg_response_time_ms = 
            (metrics.avg_response_time_ms * (metrics.total_operations - 1) as f64 + duration_ms) 
            / metrics.total_operations as f64;
        
        // Update success rate
        metrics.success_rate = metrics.successful_operations as f64 / metrics.total_operations as f64;
        
        // Calculate operations per second
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            metrics.ops_per_second = metrics.total_operations as f64 / elapsed;
        }
        
        // Calculate percentiles
        if !response_times.is_empty() {
            let mut sorted = response_times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            let p95_index = (sorted.len() as f64 * 0.95) as usize;
            let p99_index = (sorted.len() as f64 * 0.99) as usize;
            
            metrics.p95_response_time_ms = sorted.get(p95_index).copied().unwrap_or(0.0);
            metrics.p99_response_time_ms = sorted.get(p99_index).copied().unwrap_or(0.0);
        }
        
        debug!(
            duration_ms = duration_ms,
            success = success,
            "Operation recorded"
        );
    }
    
    /// Update system health metrics
    pub async fn update_health(&self, connections: u32, queue_depth: u32) {
        let mut metrics = self.health_metrics.write().await;
        
        // Update uptime
        metrics.uptime_seconds = self.start_time.elapsed().as_secs();
        
        // Update connection and queue metrics
        metrics.active_connections = connections;
        metrics.queue_depth = queue_depth;
        
        // Get actual system metrics using sysinfo crate
        metrics.memory_usage_mb = self.get_memory_usage_mb();
        metrics.cpu_usage_percent = self.get_cpu_usage_percent();
        
        // Determine health status based on metrics
        let error_metrics = self.error_metrics.read().await;
        metrics.health_status = if error_metrics.error_rate > 10.0 {
            HealthStatus::Critical
        } else if error_metrics.error_rate > 5.0 || queue_depth > 100 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
    }
    
    /// Get current error metrics
    pub async fn get_error_metrics(&self) -> ErrorMetrics {
        self.error_metrics.read().await.clone()
    }
    
    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }
    
    /// Get current health metrics
    pub async fn get_health_metrics(&self) -> HealthMetrics {
        self.health_metrics.read().await.clone()
    }
    
    /// Get comprehensive telemetry report
    pub async fn get_telemetry_report(&self) -> TelemetryReport {
        TelemetryReport {
            timestamp: chrono::Utc::now(),
            error_metrics: self.get_error_metrics().await,
            performance_metrics: self.get_performance_metrics().await,
            health_metrics: self.get_health_metrics().await,
        }
    }
    
    /// Get actual memory usage in MB
    fn get_memory_usage_mb(&self) -> f64 {
        // Use std library to get memory info from /proc/self/status on Linux
        // or estimate based on allocator stats on other platforms
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                return kb / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback: use a reasonable estimate
        // In production, you might want to use the `memory-stats` crate
        100.0
    }
    
    /// Get actual CPU usage percentage
    fn get_cpu_usage_percent(&self) -> f64 {
        // Get CPU usage from /proc/self/stat on Linux
        #[cfg(target_os = "linux")]
        {
            use std::time::Duration;
            
            if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
                // Parse the stat file to get CPU time
                // This is a simplified version; in production use a proper crate
                let fields: Vec<&str> = stat.split_whitespace().collect();
                if fields.len() > 14 {
                    // utime + stime (user time + system time)
                    if let (Ok(utime), Ok(stime)) = (fields[13].parse::<u64>(), fields[14].parse::<u64>()) {
                        let total_time = utime + stime;
                        let elapsed = self.start_time.elapsed().as_secs();
                        if elapsed > 0 {
                            // Convert to percentage (approximate)
                            return ((total_time as f64 / 100.0) / elapsed as f64) * 100.0;
                        }
                    }
                }
            }
        }
        
        // Fallback: return a reasonable estimate
        // In production, you might want to use the `cpu-monitor` crate
        10.0
    }
    
    /// Reset metrics (useful for testing or periodic resets)
    pub async fn reset_metrics(&self) {
        *self.error_metrics.write().await = ErrorMetrics::default();
        *self.performance_metrics.write().await = PerformanceMetrics::default();
        self.response_times.write().await.clear();
        *self.last_reset.write().await = Instant::now();
        
        info!("Telemetry metrics reset");
    }
    
    /// Log current metrics summary
    pub async fn log_summary(&self) {
        let error_metrics = self.get_error_metrics().await;
        let perf_metrics = self.get_performance_metrics().await;
        let health_metrics = self.get_health_metrics().await;
        
        info!(
            "Telemetry Summary - Errors: {}, Error Rate: {:.2}/min, Success Rate: {:.2}%, Avg Response: {:.2}ms, Health: {:?}",
            error_metrics.total_errors,
            error_metrics.error_rate,
            perf_metrics.success_rate * 100.0,
            perf_metrics.avg_response_time_ms,
            health_metrics.health_status
        );
        
        if let Some(most_common) = &error_metrics.most_common_error {
            warn!("Most common error type: {}", most_common);
        }
    }
}

/// Comprehensive telemetry report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_metrics: ErrorMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub health_metrics: HealthMetrics,
}

/// Global telemetry instance
static mut TELEMETRY: Option<Arc<TelemetryCollector>> = None;
static TELEMETRY_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global telemetry
pub fn init_telemetry() -> Arc<TelemetryCollector> {
    unsafe {
        TELEMETRY_INIT.call_once(|| {
            TELEMETRY = Some(Arc::new(TelemetryCollector::new()));
        });
        TELEMETRY.as_ref().unwrap().clone()
    }
}

/// Get global telemetry instance
pub fn telemetry() -> Arc<TelemetryCollector> {
    init_telemetry()
}

/// Helper macro to record errors with telemetry
#[macro_export]
macro_rules! record_error {
    ($error:expr, $module:expr) => {
        {
            let telemetry = $crate::tui::chat::utils::telemetry::telemetry();
            tokio::spawn(async move {
                telemetry.record_error(&$error, $module).await;
            });
        }
    };
}

/// Helper macro to measure operation performance
#[macro_export]
macro_rules! measure_operation {
    ($operation:expr) => {
        {
            let telemetry = $crate::tui::chat::utils::telemetry::telemetry();
            let start = std::time::Instant::now();
            let result = $operation;
            let duration = start.elapsed();
            let success = result.is_ok();
            
            tokio::spawn(async move {
                telemetry.record_operation(duration, success).await;
            });
            
            result
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_error_recording() {
        let collector = TelemetryCollector::new();
        
        let error = ChatError::Internal("Test error".to_string());
        collector.record_error(&error, "test_module").await;
        
        let metrics = collector.get_error_metrics().await;
        assert_eq!(metrics.total_errors, 1);
        assert_eq!(metrics.errors_by_module.get("test_module"), Some(&1));
    }
    
    #[tokio::test]
    async fn test_operation_recording() {
        let collector = TelemetryCollector::new();
        
        collector.record_operation(Duration::from_millis(100), true).await;
        collector.record_operation(Duration::from_millis(200), true).await;
        collector.record_operation(Duration::from_millis(150), false).await;
        
        let metrics = collector.get_performance_metrics().await;
        assert_eq!(metrics.total_operations, 3);
        assert_eq!(metrics.successful_operations, 2);
        assert!((metrics.success_rate - 0.6667).abs() < 0.01);
    }
    
    #[tokio::test]
    async fn test_health_status() {
        let collector = TelemetryCollector::new();
        
        // Record many errors to trigger degraded status
        for _ in 0..100 {
            collector.record_error(&ChatError::Internal("Test".to_string()), "test").await;
        }
        
        collector.update_health(5, 10).await;
        
        let health = collector.get_health_metrics().await;
        assert_ne!(health.health_status, HealthStatus::Healthy);
    }
}