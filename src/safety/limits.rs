//! Resource Limits and Monitoring
//!
//! Controls and monitors resource usage to prevent exhaustion

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sysinfo::ProcessesToUpdate;
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{error, info};

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum number of requests
    pub max_requests: u32,

    /// Time window for the limit
    pub window: Duration,

    /// Current request count
    #[serde(skip)]
    pub current: u32,

    /// Window start time
    #[serde(skip)]
    pub window_start: DateTime<Utc>,
}

impl RateLimit {
    /// Create a new rate limit
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self { max_requests, window, current: 0, window_start: Utc::now() }
    }

    /// Check if request is allowed
    pub fn check(&mut self) -> bool {
        // Reset window if expired
        let elapsed = Utc::now().signed_duration_since(self.window_start);
        if let Ok(window_duration) = chrono::Duration::from_std(self.window) {
            if elapsed > window_duration {
                self.current = 0;
                self.window_start = Utc::now();
            }
        }

        if self.current < self.max_requests {
            self.current += 1;
            true
        } else {
            false
        }
    }

    /// Get remaining requests in current window
    pub fn remaining(&self) -> u32 {
        let elapsed = Utc::now().signed_duration_since(self.window_start);
        if let Ok(window_duration) = chrono::Duration::from_std(self.window) {
            if elapsed > window_duration {
                return self.max_requests;
            }
        }
        self.max_requests.saturating_sub(self.current)
    }
}

/// Token budget for AI providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Maximum tokens per hour
    pub max_tokens_per_hour: u32,

    /// Maximum tokens per day
    pub max_tokens_per_day: u32,

    /// Current hour's usage
    #[serde(skip)]
    pub hourly_usage: u32,

    /// Current day's usage
    #[serde(skip)]
    pub daily_usage: u32,

    /// Hour start time
    #[serde(skip)]
    pub hour_start: DateTime<Utc>,

    /// Day start time
    #[serde(skip)]
    pub day_start: DateTime<Utc>,
}

impl TokenBudget {
    /// Create a new token budget
    pub fn new(max_tokens_per_hour: u32, max_tokens_per_day: u32) -> Self {
        Self {
            max_tokens_per_hour,
            max_tokens_per_day,
            hourly_usage: 0,
            daily_usage: 0,
            hour_start: Utc::now(),
            day_start: Utc::now(),
        }
    }

    /// Check if tokens are available
    pub fn check(&mut self, tokens: u32) -> bool {
        // Reset hourly window if needed
        let hour_elapsed = Utc::now().signed_duration_since(self.hour_start);
        if hour_elapsed > chrono::Duration::seconds(3600) {
            self.hourly_usage = 0;
            self.hour_start = Utc::now();
        }

        // Reset daily window if needed
        let day_elapsed = Utc::now().signed_duration_since(self.day_start);
        if day_elapsed > chrono::Duration::seconds(86400) {
            self.daily_usage = 0;
            self.day_start = Utc::now();
        }

        // Check both limits
        if self.hourly_usage + tokens <= self.max_tokens_per_hour
            && self.daily_usage + tokens <= self.max_tokens_per_day
        {
            self.hourly_usage += tokens;
            self.daily_usage += tokens;
            true
        } else {
            false
        }
    }

    /// Get remaining tokens
    pub fn remaining_hourly(&self) -> u32 {
        let hour_elapsed = Utc::now().signed_duration_since(self.hour_start);
        if hour_elapsed > chrono::Duration::seconds(3600) {
            self.max_tokens_per_hour
        } else {
            self.max_tokens_per_hour.saturating_sub(self.hourly_usage)
        }
    }

    pub fn remaining_daily(&self) -> u32 {
        let day_elapsed = Utc::now().signed_duration_since(self.day_start);
        if day_elapsed > chrono::Duration::seconds(86400) {
            self.max_tokens_per_day
        } else {
            self.max_tokens_per_day.saturating_sub(self.daily_usage)
        }
    }
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,

    /// Maximum CPU usage percentage
    pub max_cpu_percent: f32,

    /// API rate limits by provider
    pub api_rate_limits: HashMap<String, RateLimit>,

    /// Token budgets by provider
    pub token_budgets: HashMap<String, TokenBudget>,

    /// Maximum file handles
    pub max_file_handles: usize,

    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let mut api_limits = HashMap::new();
        api_limits.insert(
            "openai".to_string(),
            RateLimit::new(500, Duration::from_secs(60)), // 500 req/min
        );
        api_limits.insert(
            "anthropic".to_string(),
            RateLimit::new(1000, Duration::from_secs(60)), // 1000 req/min
        );
        api_limits.insert(
            "x_twitter".to_string(),
            RateLimit::new(300, Duration::from_secs(900)), // 300 req/15min
        );

        let mut token_budgets = HashMap::new();
        token_budgets.insert(
            "openai".to_string(),
            TokenBudget::new(100_000, 1_000_000), // 100k/hr, 1M/day
        );
        token_budgets.insert("anthropic".to_string(), TokenBudget::new(100_000, 1_000_000));

        Self {
            // Increased to accommodate systems with high memory usage
            // Can be overridden via LOKI_MAX_MEMORY_MB environment variable
            max_memory_mb: std::env::var("LOKI_MAX_MEMORY_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(102400), // Default 100GB
            max_cpu_percent: 80.0,
            api_rate_limits: api_limits,
            token_budgets,
            max_file_handles: 1000,
            max_concurrent_ops: 100,
        }
    }
}

/// Current resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current memory usage in MB
    pub memory_mb: usize,

    /// Current CPU usage percentage
    pub cpu_percent: f32,

    /// Active file handles
    pub file_handles: usize,

    /// Active operations
    pub active_operations: usize,

    /// Timestamp
    #[serde(skip)]
    pub timestamp: DateTime<Utc>,
}

/// Resource limit exceeded error
#[derive(Debug, thiserror::Error)]
pub enum LimitExceeded {
    #[error("Memory limit exceeded: {current}MB > {limit}MB")]
    Memory { current: usize, limit: usize },

    #[error("CPU limit exceeded: {current}% > {limit}%")]
    Cpu { current: f32, limit: f32 },

    #[error("Rate limit exceeded for {service}: {remaining} remaining")]
    RateLimit { service: String, remaining: u32 },

    #[error(
        "Token budget exceeded for {provider}: {remaining_hourly} hourly, {remaining_daily} daily"
    )]
    TokenBudget { provider: String, remaining_hourly: u32, remaining_daily: u32 },

    #[error("File handle limit exceeded: {current} > {limit}")]
    FileHandles { current: usize, limit: usize },

    #[error("Concurrent operations limit exceeded: {current} > {limit}")]
    ConcurrentOps { current: usize, limit: usize },
}

/// Resource monitor that tracks usage
pub struct ResourceMonitor {
    limits: Arc<RwLock<ResourceLimits>>,
    current_usage: Arc<RwLock<ResourceUsage>>,
    history: Arc<RwLock<Vec<ResourceUsage>>>,
    alert_tx: mpsc::Sender<LimitExceeded>,
    shutdown_tx: tokio::sync::broadcast::Sender<()>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(limits: ResourceLimits, alert_tx: mpsc::Sender<LimitExceeded>) -> Self {
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);

        Self {
            limits: Arc::new(RwLock::new(limits)),
            current_usage: Arc::new(RwLock::new(ResourceUsage {
                memory_mb: 0,
                cpu_percent: 0.0,
                file_handles: 0,
                active_operations: 0,
                timestamp: Utc::now(),
            })),
            history: Arc::new(RwLock::new(Vec::new())),
            alert_tx,
            shutdown_tx,
        }
    }

    /// Start monitoring
    pub async fn start(&self) -> Result<()> {
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.monitoring_loop().await;
        });

        Ok(())
    }

    /// Monitoring loop
    async fn monitoring_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut check_interval = interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = check_interval.tick() => {
                    if let Err(e) = self.check_resources().await {
                        error!("Resource check error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Resource monitor shutting down");
                    break;
                }
            }
        }
    }

    /// Check current resource usage
    async fn check_resources(&self) -> Result<()> {
        // Get system stats (would use sysinfo crate in real implementation)
        let memory_mb = self.get_memory_usage_mb();
        let cpu_percent = self.get_cpu_usage_percent();
        let file_handles = self.get_file_handles();

        // Update current usage
        let mut usage = self.current_usage.write().await;
        usage.memory_mb = memory_mb;
        usage.cpu_percent = cpu_percent;
        usage.file_handles = file_handles;
        usage.timestamp = Utc::now();

        // Check limits
        let limits = self.limits.read().await;

        if memory_mb > limits.max_memory_mb {
            let _ = self
                .alert_tx
                .send(LimitExceeded::Memory { current: memory_mb, limit: limits.max_memory_mb })
                .await;
        }

        if cpu_percent > limits.max_cpu_percent {
            let _ = self
                .alert_tx
                .send(LimitExceeded::Cpu { current: cpu_percent, limit: limits.max_cpu_percent })
                .await;
        }

        if file_handles > limits.max_file_handles {
            let _ = self
                .alert_tx
                .send(LimitExceeded::FileHandles {
                    current: file_handles,
                    limit: limits.max_file_handles,
                })
                .await;
        }

        // Record history
        let usage_snapshot = usage.clone();
        drop(usage);

        let mut history = self.history.write().await;
        history.push(usage_snapshot);

        // Keep only last hour of history
        if history.len() > 3600 {
            history.drain(0..100);
        }

        Ok(())
    }

    /// Check if API call is allowed
    pub async fn check_api_limit(&self, provider: &str) -> Result<(), LimitExceeded> {
        let mut limits = self.limits.write().await;

        if let Some(rate_limit) = limits.api_rate_limits.get_mut(provider) {
            if !rate_limit.check() {
                return Err(LimitExceeded::RateLimit {
                    service: provider.to_string(),
                    remaining: rate_limit.remaining(),
                });
            }
        }

        Ok(())
    }

    /// Check if tokens are available
    pub async fn check_token_budget(
        &self,
        provider: &str,
        tokens: u32,
    ) -> Result<(), LimitExceeded> {
        let mut limits = self.limits.write().await;

        if let Some(budget) = limits.token_budgets.get_mut(provider) {
            if !budget.check(tokens) {
                return Err(LimitExceeded::TokenBudget {
                    provider: provider.to_string(),
                    remaining_hourly: budget.remaining_hourly(),
                    remaining_daily: budget.remaining_daily(),
                });
            }
        }

        Ok(())
    }

    /// Increment active operations
    pub async fn start_operation(&self) -> Result<(), LimitExceeded> {
        let mut usage = self.current_usage.write().await;
        let limits = self.limits.read().await;

        if usage.active_operations >= limits.max_concurrent_ops {
            return Err(LimitExceeded::ConcurrentOps {
                current: usage.active_operations,
                limit: limits.max_concurrent_ops,
            });
        }

        usage.active_operations += 1;
        Ok(())
    }

    /// Decrement active operations
    pub async fn end_operation(&self) {
        let mut usage = self.current_usage.write().await;
        usage.active_operations = usage.active_operations.saturating_sub(1);
    }

    /// Get current usage
    pub async fn get_usage(&self) -> ResourceUsage {
        self.current_usage.read().await.clone()
    }

    /// Get usage history
    pub async fn get_history(&self) -> Vec<ResourceUsage> {
        self.history.read().await.clone()
    }

    /// Update limits
    pub async fn update_limits(&self, new_limits: ResourceLimits) {
        *self.limits.write().await = new_limits;
    }

    /// Shutdown monitor
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Get actual memory usage using sysinfo
    fn get_memory_usage_mb(&self) -> usize {
        use sysinfo::{Pid, System};

        // Get current process info
        if let Ok(pid) = sysinfo::get_current_pid() {
            let mut system = System::new();
            system.refresh_processes(ProcessesToUpdate::Some(&[Pid::from_u32(pid.as_u32())]), false);

            if let Some(process) = system.process(Pid::from_u32(pid.as_u32())) {
                // Convert bytes to MB
                return (process.memory() / (1024 * 1024)) as usize;
            }
        }

        // Fallback: try to get system memory usage
        let mut system = System::new_all();
        system.refresh_memory();
        let used_memory = system.used_memory();
        (used_memory / 1024 / 1024) as usize
    }

    /// Get actual CPU usage using sysinfo
    fn get_cpu_usage_percent(&self) -> f32 {
        use sysinfo::{Pid, System};

        if let Ok(pid) = sysinfo::get_current_pid() {
            let mut system = System::new();
            system.refresh_processes(ProcessesToUpdate::Some(&[Pid::from_u32(pid.as_u32())]), false);

            if let Some(process) = system.process(Pid::from_u32(pid.as_u32())) {
                return process.cpu_usage();
            }
        }

        // Fallback: global CPU usage
        let mut system = System::new_all();
        system.refresh_cpu_all();

        // Get average CPU usage across all cores
        let total_usage: f32 = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum();

        if !system.cpus().is_empty() { total_usage / system.cpus().len() as f32 } else { 0.0 }
    }

    /// Get file handle count (platform-specific implementation)
    fn get_file_handles(&self) -> usize {
        // Linux/Unix implementation
        #[cfg(unix)]
        {
            use std::fs;
            if let Ok(entries) = fs::read_dir("/proc/self/fd") {
                return entries.count();
            }
        }

        // Windows implementation
        #[cfg(windows)]
        {
            use std::process;
            if let Ok(pid) = sysinfo::get_current_pid() {
                // On Windows, we can approximate by checking process handles
                // This is a simplified implementation
                use sysinfo::{Pid, System};
                let mut system = System::new();
                system.refresh_process(Pid::from_u32(pid.as_u32()));

                if let Some(process) = system.process(Pid::from_u32(pid.as_u32())) {
                    // Rough estimate based on memory usage
                    return (process.memory_kb() as usize).max(10);
                }
            }
        }

        // Fallback estimate based on common file usage patterns
        let base_handles = 10; // stdin, stdout, stderr, config files, etc.
        let estimated_handles = match std::env::var("RUST_LOG") {
            Ok(_) => base_handles + 5, // Additional logging handles
            Err(_) => base_handles,
        };

        estimated_handles
    }

    /// Get current memory usage in bytes
    pub async fn get_memory_usage(&self) -> u64 {
        let usage = self.current_usage.read().await;
        (usage.memory_mb as u64) * 1024 * 1024
    }

    /// Get current CPU usage percentage
    pub async fn get_cpu_usage(&self) -> f32 {
        let usage = self.current_usage.read().await;
        usage.cpu_percent
    }

    /// Get resource usage ratio (0.0 - 1.0)
    pub async fn get_usage_ratio(&self) -> f32 {
        // Return average of CPU and memory usage as a simple metric
        let usage = self.current_usage.read().await;
        let limits = self.limits.read().await;
        
        let cpu_ratio = usage.cpu_percent / 100.0;
        let memory_ratio = (usage.memory_mb as f32) / (limits.max_memory_mb as f32);
        
        (cpu_ratio + memory_ratio) / 2.0
    }
}

// Make ResourceMonitor cloneable for the monitoring loop
impl Clone for ResourceMonitor {
    fn clone(&self) -> Self {
        Self {
            limits: Arc::clone(&self.limits),
            current_usage: Arc::clone(&self.current_usage),
            history: Arc::clone(&self.history),
            alert_tx: self.alert_tx.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limit() {
        let mut limit = RateLimit::new(5, Duration::from_secs(1));

        // Should allow 5 requests
        for _ in 0..5 {
            assert!(limit.check());
        }

        // 6th should fail
        assert!(!limit.check());

        // Wait for window to reset
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should work again
        assert!(limit.check());
    }

    #[tokio::test]
    async fn test_token_budget() {
        let mut budget = TokenBudget::new(1000, 10000);

        // Should allow within budget
        assert!(budget.check(500));
        assert!(budget.check(400));

        // Should fail when exceeding hourly
        assert!(!budget.check(200));

        assert_eq!(budget.remaining_hourly(), 100);
    }
}
