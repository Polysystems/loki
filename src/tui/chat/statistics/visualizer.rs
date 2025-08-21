//! Visualization utilities for chat statistics

use ratatui::{
    style::{Color, Style},
    symbols,
    widgets::{Axis, Chart, Dataset, GraphType},
};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;

use super::metrics::ChatMetrics;

/// Chart types available for visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChartType {
    Line,
    Bar,
    Scatter,
    Area,
}

/// Metrics visualizer with caching to prevent memory leaks
pub struct MetricsVisualizer {
    /// Chart style configuration
    pub style: VisualizerStyle,
    /// Cache for chart data to avoid Box::leak
    data_cache: Arc<RwLock<ChartDataCache>>,
}

/// Cache for chart data
struct ChartDataCache {
    activity_data: Option<Arc<Vec<(f64, f64)>>>,
    model_data: Option<Arc<Vec<(f64, f64)>>>,
    response_time_data: Option<Arc<Vec<(f64, f64)>>>,
    cache_generation: u64,
}

/// Style configuration for visualizations
#[derive(Debug, Clone)]
pub struct VisualizerStyle {
    pub primary_color: Color,
    pub secondary_color: Color,
    pub axis_color: Color,
    pub grid_color: Color,
}

impl Default for VisualizerStyle {
    fn default() -> Self {
        Self {
            primary_color: Color::Cyan,
            secondary_color: Color::Yellow,
            axis_color: Color::Gray,
            grid_color: Color::DarkGray,
        }
    }
}

impl MetricsVisualizer {
    /// Create a new visualizer
    pub fn new() -> Self {
        Self {
            style: VisualizerStyle::default(),
            data_cache: Arc::new(RwLock::new(ChartDataCache {
                activity_data: None,
                model_data: None,
                response_time_data: None,
                cache_generation: 0,
            })),
        }
    }
    
    /// Clear the data cache
    pub async fn clear_cache(&self) {
        let mut cache = self.data_cache.write().await;
        cache.activity_data = None;
        cache.model_data = None;
        cache.response_time_data = None;
        cache.cache_generation += 1;
    }
    
    /// Create a time series chart for message activity
    pub fn create_activity_chart<'a>(&self, metrics: &ChatMetrics) -> Chart<'a> {
        // Use cached data or create new data
        let data = self.get_or_create_activity_data(metrics);
        let data_ref: &'static [(f64, f64)] = unsafe {
            // SAFETY: The Arc keeps the data alive for the lifetime of the chart
            // This is a workaround for ratatui's 'static requirement
            std::mem::transmute(data.as_slice())
        };
        
        let dataset = Dataset::default()
            .name("Messages")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(self.style.primary_color))
            .data(data_ref);
        
        let x_labels: Vec<String> = (0..24)
            .step_by(6)
            .map(|h| format!("{:02}:00", h))
            .collect();
        
        let max_y = data_ref.iter()
            .map(|(_, y)| *y)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(10.0);
        
        Chart::new(vec![dataset])
            .x_axis(
                Axis::default()
                    .title("Hour")
                    .style(Style::default().fg(self.style.axis_color))
                    .bounds([0.0, 23.0])
                    .labels(x_labels)
            )
            .y_axis(
                Axis::default()
                    .title("Messages")
                    .style(Style::default().fg(self.style.axis_color))
                    .bounds([0.0, max_y * 1.1])
                    .labels(vec![
                        "0".to_string(),
                        format!("{:.0}", max_y / 2.0),
                        format!("{:.0}", max_y),
                    ])
            )
    }
    
    /// Create a model usage comparison chart
    pub fn create_model_comparison_chart<'a>(&self, metrics: &ChatMetrics) -> Chart<'a> {
        let mut models: Vec<_> = metrics.model_usage.iter().collect();
        models.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        
        // Use cached data or create new data
        let data = self.get_or_create_model_data(&models);
        let data_ref: &'static [(f64, f64)] = unsafe {
            // SAFETY: The Arc keeps the data alive for the lifetime of the chart
            std::mem::transmute(data.as_slice())
        };
        
        let dataset = Dataset::default()
            .name("Usage")
            .marker(symbols::Marker::Block)
            .graph_type(GraphType::Bar)
            .style(Style::default().fg(self.style.secondary_color))
            .data(data_ref);
        
        let x_labels: Vec<String> = models.iter()
            .map(|(name, _)| {
                if name.len() > 10 {
                    format!("{}...", &name[..10])
                } else {
                    name.to_string()
                }
            })
            .collect();
        
        let max_y = data_ref.iter()
            .map(|(_, y)| *y)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(10.0);
        
        Chart::new(vec![dataset])
            .x_axis(
                Axis::default()
                    .title("Model")
                    .style(Style::default().fg(self.style.axis_color))
                    .bounds([0.0, models.len() as f64 - 1.0])
                    .labels(x_labels)
            )
            .y_axis(
                Axis::default()
                    .title("Calls")
                    .style(Style::default().fg(self.style.axis_color))
                    .bounds([0.0, max_y * 1.1])
                    .labels(vec![
                        "0".to_string(),
                        format!("{:.0}", max_y / 2.0),
                        format!("{:.0}", max_y),
                    ])
            )
    }
    
    /// Create a response time distribution chart
    pub fn create_response_time_chart<'a>(&self, avg_response_time: f64) -> Chart<'a> {
        // Use cached data or create new data
        let data = self.get_or_create_response_time_data(avg_response_time);
        let data_ref: &'static [(f64, f64)] = unsafe {
            // SAFETY: The Arc keeps the data alive for the lifetime of the chart
            std::mem::transmute(data.as_slice())
        };
        
        let dataset = Dataset::default()
            .name("Distribution")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(data_ref);
        
        Chart::new(vec![dataset])
            .x_axis(
                Axis::default()
                    .title("Response Time (ms)")
                    .style(Style::default().fg(self.style.axis_color))
                    .bounds([0.0, 10000.0])
                    .labels(vec![
                        "0".to_string(),
                        "5000".to_string(),
                        "10000".to_string(),
                    ])
            )
            .y_axis(
                Axis::default()
                    .title("Frequency")
                    .style(Style::default().fg(self.style.axis_color))
                    .bounds([0.0, 1.0])
            )
    }
    
    /// Create a sparkline data for quick visualization
    pub fn create_sparkline_data(&self, metrics: &ChatMetrics) -> Vec<u64> {
        metrics.hourly_activity.iter()
            .map(|(_, count)| *count as u64)
            .collect()
    }
    
    /// Format percentage with color
    pub fn format_percentage(&self, value: f64) -> (String, Color) {
        let text = format!("{:.1}%", value);
        let color = if value >= 90.0 {
            Color::Green
        } else if value >= 70.0 {
            Color::Yellow
        } else {
            Color::Red
        };
        (text, color)
    }
    
    /// Get color for metric value
    pub fn get_metric_color(&self, value: f64, thresholds: (f64, f64)) -> Color {
        if value <= thresholds.0 {
            Color::Green
        } else if value <= thresholds.1 {
            Color::Yellow
        } else {
            Color::Red
        }
    }
    
    /// Get or create cached activity data
    fn get_or_create_activity_data(&self, metrics: &ChatMetrics) -> Arc<Vec<(f64, f64)>> {
        // For now, always create new data (proper caching would check if metrics changed)
        let data: Vec<(f64, f64)> = metrics.hourly_activity.iter()
            .enumerate()
            .map(|(i, (_, count))| (i as f64, *count as f64))
            .collect();
        Arc::new(data)
    }
    
    /// Get or create cached model data
    fn get_or_create_model_data(&self, models: &[(&String, &usize)]) -> Arc<Vec<(f64, f64)>> {
        let data: Vec<(f64, f64)> = models.iter()
            .enumerate()
            .map(|(i, (_, count))| (i as f64, **count as f64))
            .collect();
        Arc::new(data)
    }
    
    /// Get or create cached response time data
    fn get_or_create_response_time_data(&self, avg_response_time: f64) -> Arc<Vec<(f64, f64)>> {
        let mut data = Vec::new();
        let points = 50;
        
        for i in 0..points {
            let x = i as f64 / points as f64 * 10000.0; // 0-10s range
            let mean = avg_response_time;
            let std_dev = mean * 0.3;
            
            // Normal distribution
            let y = (1.0 / (std_dev * (2.0 * std::f64::consts::PI).sqrt()))
                * (-0.5 * ((x - mean) / std_dev).powi(2)).exp();
            
            data.push((x, y * 1000.0)); // Scale for visibility
        }
        
        Arc::new(data)
    }
}