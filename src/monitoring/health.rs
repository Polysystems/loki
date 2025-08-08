use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};
use tokio::time::interval;
use tracing::{error, info};

use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Health monitor for system monitoring
#[derive(Debug)]
pub struct HealthMonitor {
    /// System info
    system: Arc<RwLock<System>>,

    /// Current health status
    status: Arc<RwLock<HealthStatus>>,

    /// Historical metrics
    metrics_history: Arc<RwLock<Vec<SystemMetrics>>>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Health thresholds
    thresholds: HealthThresholds,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing health monitor");

        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self {
            system: Arc::new(RwLock::new(system)),
            status: Arc::new(RwLock::new(HealthStatus::Healthy)),
            metrics_history: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            memory,
            thresholds: HealthThresholds::default(),
        })
    }

    /// Start monitoring
    pub async fn start_monitoring(self: Arc<Self>) {
        info!("Starting health monitoring");

        let monitor = self.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;
                if let Err(e) = monitor.check_health().await {
                    error!("Health check error: {}", e);
                }
            }
        });
    }

    /// Check system health
    async fn check_health(&self) -> Result<()> {
        // Refresh system info
        {
            let mut system = self.system.write();
            system.refresh_all();
        }

        // Collect metrics
        let metrics = self.collect_metrics()?;

        // Determine health status
        let new_status = self.evaluate_health(&metrics);

        // Update status if changed
        let status_changed = {
            let mut status = self.status.write();
            let changed = *status != new_status;
            *status = new_status;
            changed
        };

        if status_changed {
            info!("Health status changed to: {:?}", new_status);

            // Store in memory
            self.memory
                .store(
                    format!("Health status: {:?}", new_status),
                    vec![format!("{:?}", metrics)],
                    MemoryMetadata {
                        source: "health_monitor".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                        tags: vec!["health".to_string(), format!("{:?}", new_status)],
                        importance: match new_status {
                            HealthStatus::Critical => 1.0,
                            HealthStatus::Unhealthy => 0.8,
                            HealthStatus::Degraded => 0.6,
                            HealthStatus::Healthy => 0.3,
                        },
                        associations: vec![],
                        context: Some(format!("Health status change to: {:?}", new_status)),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "monitoring".to_string(),
                    },
                )
                .await?;
        }

        // Store metrics
        {
            let mut history = self.metrics_history.write();
            history.push(metrics);

            // Keep only recent history
            while history.len() > 1000 {
                history.remove(0);
            }
        }

        Ok(())
    }

    /// Collect current system metrics
    fn collect_metrics(&self) -> Result<SystemMetrics> {
        let system = self.system.read();

        // CPU usage
        let cpu_usage = system.global_cpu_usage();

        // Memory usage
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let memory_usage =
            if total_memory > 0 { (used_memory as f32 / total_memory as f32) * 100.0 } else { 0.0 };

        // Disk usage
        let mut disk_usage = 0.0;
        let disks = Disks::new_with_refreshed_list();
        for disk in disks.list() {
            let total = disk.total_space();
            let available = disk.available_space();
            if total > 0 {
                disk_usage = ((total - available) as f32 / total as f32) * 100.0;
                break; // Just use first disk for now
            }
        }

        // Process info
        let process_count = system.processes().len();

        // Network (simplified)
        let network_rx = 0;
        let network_tx = 0;

        Ok(SystemMetrics {
            timestamp: Utc::now(),
            cpu_usage,
            memory_usage,
            disk_usage,
            memory_total: total_memory,
            memory_used: used_memory,
            process_count,
            network_rx_bytes: network_rx,
            network_tx_bytes: network_tx,
            error_count: 0,
            warning_count: 0,
        })
    }

    /// Evaluate health based on metrics
    fn evaluate_health(&self, metrics: &SystemMetrics) -> HealthStatus {
        let mut issues = Vec::new();

        // Check CPU
        if metrics.cpu_usage > self.thresholds.cpu_critical {
            issues.push("Critical CPU usage");
        } else if metrics.cpu_usage > self.thresholds.cpu_warning {
            issues.push("High CPU usage");
        }

        // Check memory
        if metrics.memory_usage > self.thresholds.memory_critical {
            issues.push("Critical memory usage");
        } else if metrics.memory_usage > self.thresholds.memory_warning {
            issues.push("High memory usage");
        }

        // Check disk
        if metrics.disk_usage > self.thresholds.disk_critical {
            issues.push("Critical disk usage");
        } else if metrics.disk_usage > self.thresholds.disk_warning {
            issues.push("High disk usage");
        }

        // Determine overall status
        if issues.iter().any(|i| i.contains("Critical")) {
            HealthStatus::Critical
        } else if !issues.is_empty() {
            HealthStatus::Unhealthy
        } else if metrics.cpu_usage > 50.0 || metrics.memory_usage > 70.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    /// Get current health status
    pub fn get_status(&self) -> HealthStatus {
        *self.status.read()
    }

    /// Get current metrics
    pub fn get_current_metrics(&self) -> Option<SystemMetrics> {
        self.metrics_history.read().last().cloned()
    }

    /// Get metrics history
    pub fn get_metrics_history(&self, count: usize) -> Vec<SystemMetrics> {
        let history = self.metrics_history.read();
        let start = history.len().saturating_sub(count);
        history[start..].to_vec()
    }

    /// Check if system is healthy enough for operation
    pub fn is_operational(&self) -> bool {
        matches!(*self.status.read(), HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Get system status (stub implementation)
    pub fn get_system_status(&self) -> Result<serde_json::Value> {
        let status = *self.status.read();
        let latest_metrics = self.metrics_history.read()
            .last()
            .cloned()
            .unwrap_or_default();

        Ok(serde_json::json!({
            "status": match status {
                HealthStatus::Healthy => "healthy",
                HealthStatus::Degraded => "degraded", 
                HealthStatus::Unhealthy => "unhealthy",
                HealthStatus::Critical => "critical"
            },
            "cpu_usage": latest_metrics.cpu_usage,
            "memory_usage": latest_metrics.memory_usage,
            "disk_usage": latest_metrics.disk_usage,
            "network_rx_bytes": latest_metrics.network_rx_bytes,
            "network_tx_bytes": latest_metrics.network_tx_bytes,
            "timestamp": latest_metrics.timestamp
        }))
    }
}

/// Health status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "Healthy"),
            HealthStatus::Degraded => write!(f, "Degraded"),
            HealthStatus::Unhealthy => write!(f, "Unhealthy"),
            HealthStatus::Critical => write!(f, "Critical"),
        }
    }
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub disk_usage: f32,
    pub memory_total: u64,
    pub memory_used: u64,
    pub process_count: usize,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub error_count: u32,
    pub warning_count: u32,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            memory_total: 0,
            memory_used: 0,
            process_count: 0,
            network_rx_bytes: 0,
            network_tx_bytes: 0,
            error_count: 0,
            warning_count: 0,
        }
    }
}

/// Health thresholds
#[derive(Debug, Clone)]
struct HealthThresholds {
    cpu_warning: f32,
    cpu_critical: f32,
    memory_warning: f32,
    memory_critical: f32,
    disk_warning: f32,
    disk_critical: f32,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 80.0,
            cpu_critical: 95.0,
            memory_warning: 85.0,
            memory_critical: 95.0,
            disk_warning: 85.0,
            disk_critical: 95.0,
        }
    }
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub resource_type: ResourceType,
    pub current_usage: f32,
    pub limit: f32,
    pub trend: UsageTrend,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    Memory,
    Disk,
    Network,
    GPU,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UsageTrend {
    Stable,
    Increasing,
    Decreasing,
    Spiking,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub component: String,
    pub status: ComponentStatus,
    pub message: Option<String>,
    pub last_checked: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComponentStatus {
    Ok,
    Warning,
    Error,
    Unknown,
}

impl std::fmt::Display for ComponentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentStatus::Ok => write!(f, "Ok"),
            ComponentStatus::Warning => write!(f, "Warning"),
            ComponentStatus::Error => write!(f, "Error"),
            ComponentStatus::Unknown => write!(f, "Unknown"),
        }
    }
}
