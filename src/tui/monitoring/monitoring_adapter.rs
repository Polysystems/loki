//! Adapter to bridge RealTimeMetricsAggregator to RealTimeMonitor interface
//!
//! This adapter allows the TUI's real-time metrics aggregator to be used
//! where a RealTimeMonitor is expected.

use std::sync::Arc;

use anyhow::Result;
use crate::monitoring::real_time::{ SystemMetrics};
use crate::tui::real_time_integration::RealTimeMetricsAggregator;

/// Adapter that implements RealTimeMonitor interface for RealTimeMetricsAggregator
pub struct RealTimeMonitorAdapter {
    aggregator: Arc<RealTimeMetricsAggregator>,
}

impl RealTimeMonitorAdapter {
    /// Create a new adapter
    pub fn new(aggregator: Arc<RealTimeMetricsAggregator>) -> Self {
        Self { aggregator }
    }
}

/// Implement the required methods for RealTimeMonitor compatibility
impl RealTimeMonitorAdapter {
    /// Get current system metrics
    pub async fn get_current_metrics(&self) -> Result<SystemMetrics> {
        // Use the aggregator's method to get current metrics
        self.aggregator
            .get_current_metrics()
            .await
            .ok_or_else(|| anyhow::anyhow!("No metrics available"))
    }
    
    /// Get metrics history
    pub async fn get_metrics_history(&self) -> Vec<SystemMetrics> {
        self.aggregator.get_metrics_history().await
    }
}