//! Chat statistics and analytics module
//! 
//! Provides comprehensive statistics and insights about chat usage

pub mod dashboard;
pub mod metrics;
pub mod analyzer;
pub mod visualizer;

// Re-export main components
pub use dashboard::{StatisticsDashboard, DashboardConfig};
pub use metrics::{ChatMetrics, MetricType, TimeRange, MetricsCalculator};
pub use analyzer::{ChatAnalyzer, AnalysisResult};
pub use visualizer::{MetricsVisualizer, ChartType};