use std::time::Duration;

use tracing::{debug, warn};

use super::ClusterNode;

/// Health status of a node
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Health monitor for cluster nodes
pub struct HealthMonitor {
    check_interval: Duration,
    unhealthy_threshold: f32,
    degraded_threshold: f32,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(check_interval: Duration) -> Self {
        Self {
            check_interval,
            unhealthy_threshold: 0.9, // 90% resource usage
            degraded_threshold: 0.7,  // 70% resource usage
        }
    }

    /// Check the health of a node
    pub async fn check_node_health(&self, node: &ClusterNode) -> HealthStatus {
        let stats = node.stats.load();

        // Check memory usage
        if stats.memory_usage_percent > self.unhealthy_threshold * 100.0 {
            warn!(
                "Node {} has high memory usage: {:.1}%",
                node.device.id, stats.memory_usage_percent
            );
            return HealthStatus::Unhealthy;
        }

        // Check compute usage
        if stats.compute_usage_percent > self.unhealthy_threshold * 100.0 {
            warn!(
                "Node {} has high compute usage: {:.1}%",
                node.device.id, stats.compute_usage_percent
            );
            return HealthStatus::Unhealthy;
        }

        // Check error rate
        if stats.total_requests > 100 {
            let error_rate = stats.failed_requests as f64 / stats.total_requests as f64;
            if error_rate > 0.1 {
                // 10% error rate
                warn!("Node {} has high error rate: {:.1}%", node.device.id, error_rate * 100.0);
                return HealthStatus::Degraded;
            }
        }

        // Check if degraded
        if stats.memory_usage_percent > self.degraded_threshold * 100.0
            || stats.compute_usage_percent > self.degraded_threshold * 100.0
        {
            debug!(
                "Node {} is degraded (memory: {:.1}%, compute: {:.1}%)",
                node.device.id, stats.memory_usage_percent, stats.compute_usage_percent
            );
            return HealthStatus::Degraded;
        }

        // Check if we have recent updates
        if let Some(last_update) = stats.last_update {
            if last_update.elapsed() > Duration::from_secs(300) {
                // 5 minutes
                debug!("Node {} has stale stats", node.device.id);
                return HealthStatus::Unknown;
            }
        }

        HealthStatus::Healthy
    }

    /// Get the check interval
    pub fn check_interval(&self) -> Duration {
        self.check_interval
    }

    /// Set thresholds
    pub fn set_thresholds(&mut self, unhealthy: f32, degraded: f32) {
        self.unhealthy_threshold = unhealthy;
        self.degraded_threshold = degraded;
    }
}
