//! Monitoring tab - System metrics and performance monitoring

use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, List, ListItem, Paragraph, Gauge, Sparkline,
    Table, Row, Cell, Chart, Axis, Dataset, GraphType, Wrap
};
use crossterm::event::{KeyEvent, KeyCode};
use tokio::sync::RwLock;
use tracing::debug;
use std::collections::VecDeque;

use crate::tui::utilities::state::UtilitiesState;
use crate::tui::utilities::types::{UtilitiesAction, NotificationType};
use crate::tui::utilities::metrics::{SystemMetrics, format_bytes, format_percentage, format_uptime};
use super::UtilitiesSubtabController;

/// View modes for the monitoring tab
#[derive(Debug, Clone, PartialEq)]
enum ViewMode {
    Overview,
    Metrics,
    Logs,
    Alerts,
    Performance,
}

/// Monitoring data point for time series
#[derive(Debug, Clone)]
struct MetricPoint {
    timestamp: f64,
    value: f64,
}

/// Monitoring tab with system metrics visualization
pub struct MonitoringTab {
    /// Shared state
    state: Arc<RwLock<UtilitiesState>>,
    
    /// System metrics collector
    metrics_collector: Arc<SystemMetrics>,
    
    /// Current view mode
    view_mode: ViewMode,
    
    /// CPU usage history
    cpu_history: VecDeque<f64>,
    
    /// Memory usage history
    memory_history: VecDeque<f64>,
    
    /// Network I/O history
    network_history: VecDeque<f64>,
    
    /// Disk I/O history
    disk_history: VecDeque<f64>,
    
    /// Selected metric for detailed view
    selected_metric: usize,
    
    /// Alert list
    alerts: Vec<String>,
    
    /// Performance metrics
    performance_data: Vec<(String, f64)>,
    
    /// Log entries
    log_entries: VecDeque<String>,
    
    /// Maximum history size
    max_history: usize,
    
    /// Last update timestamp
    last_update: std::time::Instant,
}

impl MonitoringTab {
    pub fn new(
        state: Arc<RwLock<UtilitiesState>>,
    ) -> Self {
        let metrics_collector = Arc::new(SystemMetrics::new());
        
        let mut tab = Self {
            state,
            metrics_collector,
            view_mode: ViewMode::Overview,
            cpu_history: VecDeque::with_capacity(60),
            memory_history: VecDeque::with_capacity(60),
            network_history: VecDeque::with_capacity(60),
            disk_history: VecDeque::with_capacity(60),
            selected_metric: 0,
            alerts: Vec::new(),
            performance_data: Vec::new(),
            log_entries: VecDeque::with_capacity(100),
            max_history: 60,
            last_update: std::time::Instant::now(),
        };
        
        // Initialize with demo data
        tab.initialize_demo_data();
        
        tab
    }
    
    /// Get metrics collector for external use
    pub fn metrics_collector(&self) -> Arc<SystemMetrics> {
        self.metrics_collector.clone()
    }
    
    /// Initialize with default data (will be replaced by real metrics)
    fn initialize_demo_data(&mut self) {
        // Initialize with zeros - will be populated with real data on refresh
        for _ in 0..self.max_history {
            self.cpu_history.push_back(0.0);
            self.memory_history.push_back(0.0);
            self.network_history.push_back(0.0);
            self.disk_history.push_back(0.0);
        }
        
        // Alerts will be populated from actual system status
        self.alerts = vec![];
        
        // Performance data will be populated from real metrics
        self.performance_data = vec![];
        
        // Initial log entry
        self.log_entries.push_back("[INFO] System monitoring initialized".to_string());
    }
}

#[async_trait]
impl UtilitiesSubtabController for MonitoringTab {
    fn name(&self) -> &str {
        "Monitoring"
    }
    
    fn render(&mut self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Content
                Constraint::Length(3), // Footer
            ])
            .split(area);
        
        // Render header with view tabs
        self.render_header(f, chunks[0]);
        
        // Render content based on view mode
        match self.view_mode {
            ViewMode::Overview => self.render_overview(f, chunks[1]),
            ViewMode::Metrics => self.render_metrics(f, chunks[1]),
            ViewMode::Logs => self.render_logs(f, chunks[1]),
            ViewMode::Alerts => self.render_alerts(f, chunks[1]),
            ViewMode::Performance => self.render_performance(f, chunks[1]),
        }
        
        // Render controls
        self.render_controls(f, chunks[2]);
    }
    
    async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool> {
        match event.code {
            // View switching
            KeyCode::Char('1') => {
                self.view_mode = ViewMode::Overview;
                Ok(true)
            }
            KeyCode::Char('2') => {
                self.view_mode = ViewMode::Metrics;
                Ok(true)
            }
            KeyCode::Char('3') => {
                self.view_mode = ViewMode::Logs;
                Ok(true)
            }
            KeyCode::Char('4') => {
                self.view_mode = ViewMode::Alerts;
                Ok(true)
            }
            KeyCode::Char('5') => {
                self.view_mode = ViewMode::Performance;
                Ok(true)
            }
            // Navigation
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected_metric > 0 {
                    self.selected_metric -= 1;
                }
                Ok(true)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_metric < 3 {
                    self.selected_metric += 1;
                }
                Ok(true)
            }
            // Actions
            KeyCode::Char('r') => {
                self.refresh().await?;
                Ok(true)
            }
            KeyCode::Char('c') => {
                // Clear alerts
                self.alerts.clear();
                Ok(true)
            }
            KeyCode::Char('e') => {
                // Export metrics
                debug!("Exporting metrics to file");
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()> {
        match action {
            UtilitiesAction::RefreshMonitoring => {
                self.refresh().await?;
            }
            UtilitiesAction::ShowNotification(msg, notification_type) => {
                // Add to alerts
                let prefix = match notification_type {
                    NotificationType::Info => "ℹ️",
                    NotificationType::Success => "✅",
                    NotificationType::Warning => "⚠️",
                    NotificationType::Error => "❌",
                };
                self.alerts.push(format!("{} {}", prefix, msg));
            }
            _ => {}
        }
        Ok(())
    }
    
    async fn refresh(&mut self) -> Result<()> {
        debug!("Refreshing monitoring data");
        
        // Only update if enough time has passed (1 second)
        let now = std::time::Instant::now();
        if now.duration_since(self.last_update).as_secs() < 1 {
            return Ok(());
        }
        self.last_update = now;
        
        // Fetch real system metrics
        let cpu_usage = self.metrics_collector.get_cpu_usage().await;
        let (mem_used, mem_total) = self.metrics_collector.get_memory_usage().await;
        let memory_usage = if mem_total > 0 {
            (mem_used as f64 / mem_total as f64) * 100.0
        } else {
            0.0
        };
        
        let (net_rx, net_tx) = self.metrics_collector.get_network_stats().await;
        // Convert to MB/s (approximate rate calculation)
        let network_rate = ((net_rx + net_tx) as f64 / 1_000_000.0).min(100.0);
        
        let disk_usage = self.metrics_collector.get_disk_usage().await;
        let disk_io = if !disk_usage.is_empty() {
            let total_used: u64 = disk_usage.iter().map(|d| d.used_space).sum();
            let total_space: u64 = disk_usage.iter().map(|d| d.total_space).sum();
            if total_space > 0 {
                (total_used as f64 / total_space as f64) * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // Add new data points
        self.cpu_history.push_back(cpu_usage as f64);
        self.memory_history.push_back(memory_usage);
        self.network_history.push_back(network_rate);
        self.disk_history.push_back(disk_io);
        
        // Remove old data points if needed
        while self.cpu_history.len() > self.max_history {
            self.cpu_history.pop_front();
            self.memory_history.pop_front();
            self.network_history.pop_front();
            self.disk_history.pop_front();
        }
        
        // Update cached metrics in ModularUtilities
        {
            let mut state = self.state.write().await;
            // Update state cache if needed
            // state.cache could be updated here with monitoring data
        }
        
        Ok(())
    }
}

// Rendering methods
impl MonitoringTab {
    /// Render the header with view tabs
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let titles = vec![
            if self.view_mode == ViewMode::Overview { "[ Overview ]" } else { "Overview" },
            if self.view_mode == ViewMode::Metrics { "[ Metrics ]" } else { "Metrics" },
            if self.view_mode == ViewMode::Logs { "[ Logs ]" } else { "Logs" },
            if self.view_mode == ViewMode::Alerts { "[ Alerts ]" } else { "Alerts" },
            if self.view_mode == ViewMode::Performance { "[ Performance ]" } else { "Performance" },
        ];
        
        let header = Paragraph::new(titles.join(" │ "))
            .block(Block::default().borders(Borders::ALL).title("📊 System Monitoring"))
            .style(Style::default().fg(Color::White))
            .alignment(Alignment::Center);
        
        f.render_widget(header, area);
    }
    
    /// Render overview with gauges and sparklines
    fn render_overview(&mut self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // Gauges
                Constraint::Min(0),     // Sparklines
                Constraint::Length(5),  // Alerts
            ])
            .split(area);
        
        // Gauges section
        let gauge_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(chunks[0]);
        
        // CPU gauge
        let cpu_val = self.cpu_history.back().unwrap_or(&0.0);
        let cpu_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("CPU"))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent(*cpu_val as u16)
            .label(format!("{:.1}%", cpu_val));
        f.render_widget(cpu_gauge, gauge_chunks[0]);
        
        // Memory gauge
        let mem_val = self.memory_history.back().unwrap_or(&0.0);
        let mem_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Memory"))
            .gauge_style(Style::default().fg(Color::Green))
            .percent(*mem_val as u16)
            .label(format!("{:.1}%", mem_val));
        f.render_widget(mem_gauge, gauge_chunks[1]);
        
        // Network gauge
        let net_val = self.network_history.back().unwrap_or(&0.0);
        let net_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Network"))
            .gauge_style(Style::default().fg(Color::Yellow))
            .percent((*net_val * 5.0) as u16)  // Scale for display
            .label(format!("{:.1} MB/s", net_val));
        f.render_widget(net_gauge, gauge_chunks[2]);
        
        // Disk gauge
        let disk_val = self.disk_history.back().unwrap_or(&0.0);
        let disk_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Disk I/O"))
            .gauge_style(Style::default().fg(Color::Magenta))
            .percent((*disk_val * 10.0) as u16)  // Scale for display
            .label(format!("{:.1} MB/s", disk_val));
        f.render_widget(disk_gauge, gauge_chunks[3]);
        
        // Sparklines section
        let spark_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(chunks[1]);
        
        // CPU/Memory sparklines
        let cpu_data: Vec<u64> = self.cpu_history.iter().map(|v| *v as u64).collect();
        let cpu_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("CPU History"))
            .data(&cpu_data)
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(cpu_sparkline, spark_chunks[0]);
        
        let mem_data: Vec<u64> = self.memory_history.iter().map(|v| *v as u64).collect();
        let mem_sparkline = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title("Memory History"))
            .data(&mem_data)
            .style(Style::default().fg(Color::Green));
        f.render_widget(mem_sparkline, spark_chunks[1]);
        
        // Recent alerts
        self.render_alert_summary(f, chunks[2]);
    }
    
    /// Render detailed metrics view
    fn render_metrics(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
            .split(area);
        
        // Metric list
        let metrics = vec![
            ListItem::new("CPU Usage").style(if self.selected_metric == 0 {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            }),
            ListItem::new("Memory Usage").style(if self.selected_metric == 1 {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            }),
            ListItem::new("Network I/O").style(if self.selected_metric == 2 {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            }),
            ListItem::new("Disk I/O").style(if self.selected_metric == 3 {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            }),
        ];
        
        let list = List::new(metrics)
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));
        
        f.render_widget(list, chunks[0]);
        
        // Chart for selected metric
        let data = match self.selected_metric {
            0 => &self.cpu_history,
            1 => &self.memory_history,
            2 => &self.network_history,
            3 => &self.disk_history,
            _ => &self.cpu_history,
        };
        
        let dataset_points: Vec<(f64, f64)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64, *v))
            .collect();
        
        let datasets = vec![Dataset::default()
            .name(match self.selected_metric {
                0 => "CPU %",
                1 => "Memory %",
                2 => "Network MB/s",
                3 => "Disk MB/s",
                _ => "Unknown",
            })
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&dataset_points)];
        
        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title("Metric Details"))
            .x_axis(
                Axis::default()
                    .title("Time")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, self.max_history as f64]),
            )
            .y_axis(
                Axis::default()
                    .title("Value")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0]),
            );
        
        f.render_widget(chart, chunks[1]);
    }
    
    /// Render logs view
    fn render_logs(&self, f: &mut Frame, area: Rect) {
        let logs: Vec<ListItem> = self.log_entries
            .iter()
            .map(|entry| {
                let style = if entry.contains("[ERROR]") {
                    Style::default().fg(Color::Red)
                } else if entry.contains("[WARN]") {
                    Style::default().fg(Color::Yellow)
                } else if entry.contains("[INFO]") {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Gray)
                };
                ListItem::new(entry.as_str()).style(style)
            })
            .collect();
        
        let list = List::new(logs)
            .block(Block::default().borders(Borders::ALL).title("System Logs"));
        
        f.render_widget(list, area);
    }
    
    /// Render alerts view
    fn render_alerts(&self, f: &mut Frame, area: Rect) {
        let alerts: Vec<ListItem> = self.alerts
            .iter()
            .map(|alert| ListItem::new(alert.as_str()))
            .collect();
        
        let list = List::new(alerts)
            .block(Block::default().borders(Borders::ALL).title("System Alerts"));
        
        f.render_widget(list, area);
    }
    
    /// Render performance metrics
    fn render_performance(&self, f: &mut Frame, area: Rect) {
        let rows: Vec<Row> = self.performance_data
            .iter()
            .map(|(name, value)| {
                Row::new(vec![
                    Cell::from(name.clone()),
                    Cell::from(format!("{:.2} ms", value)),
                    Cell::from(if *value < 50.0 {
                        "🟢 Good"
                    } else if *value < 200.0 {
                        "🟡 Fair"
                    } else {
                        "🔴 Slow"
                    }),
                ])
            })
            .collect();
        
        let widths = [
            Constraint::Length(20),
            Constraint::Length(15),
            Constraint::Length(10),
        ];
        
        let table = Table::new(rows, widths)
            .header(Row::new(vec!["Operation", "Latency", "Status"])
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)))
            .block(Block::default().borders(Borders::ALL).title("Performance Metrics"));
        
        f.render_widget(table, area);
    }
    
    /// Render alert summary
    fn render_alert_summary(&self, f: &mut Frame, area: Rect) {
        let recent_alerts = self.alerts
            .iter()
            .take(3)
            .map(|a| a.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        let alerts = Paragraph::new(recent_alerts)
            .block(Block::default().borders(Borders::ALL).title("Recent Alerts"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(alerts, area);
    }
    
    /// Render controls footer
    fn render_controls(&self, f: &mut Frame, area: Rect) {
        let controls = match self.view_mode {
            ViewMode::Overview => "1-5: Switch View | r: Refresh | q: Back",
            ViewMode::Metrics => "↑↓: Select Metric | r: Refresh | 1-5: Switch View",
            ViewMode::Logs => "1-5: Switch View | r: Refresh",
            ViewMode::Alerts => "c: Clear Alerts | 1-5: Switch View",
            ViewMode::Performance => "e: Export | r: Refresh | 1-5: Switch View",
        };
        
        let controls_widget = Paragraph::new(controls)
            .block(Block::default().borders(Borders::ALL).title("Controls"))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        
        f.render_widget(controls_widget, area);
    }
}