//! Statistics dashboard for chat interface

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Wrap},
};
use crossterm::event::{KeyCode, KeyEvent};
use anyhow::Result;

use super::metrics::{ChatMetrics, MetricsCalculator, TimeRange};
use super::visualizer::{MetricsVisualizer, ChartType};
use crate::tui::chat::ChatState;

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Show real-time updates
    pub real_time: bool,
    
    /// Update interval in seconds
    pub update_interval: u64,
    
    /// Default time range
    pub default_time_range: TimeRange,
    
    /// Show advanced metrics
    pub show_advanced: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            real_time: true,
            update_interval: 5,
            default_time_range: TimeRange::Today,
            show_advanced: false,
        }
    }
}

/// Statistics dashboard
pub struct StatisticsDashboard {
    /// Chat state reference
    chat_state: Arc<RwLock<ChatState>>,
    
    /// Dashboard configuration
    config: DashboardConfig,
    
    /// Current metrics
    metrics: Option<ChatMetrics>,
    
    /// Selected time range
    time_range: TimeRange,
    
    /// Currently selected metric
    selected_metric: usize,
    
    /// Available metric views
    metric_views: Vec<MetricView>,
    
    /// Last update time
    last_update: std::time::Instant,
    
    /// Metrics visualizer
    visualizer: MetricsVisualizer,
}

/// Different metric views available
#[derive(Debug, Clone)]
enum MetricView {
    Overview,
    MessageAnalysis,
    ModelUsage,
    Performance,
    Topics,
    Timeline,
}

impl StatisticsDashboard {
    /// Create a new statistics dashboard
    pub fn new(chat_state: Arc<RwLock<ChatState>>, config: DashboardConfig) -> Self {
        let metric_views = vec![
            MetricView::Overview,
            MetricView::MessageAnalysis,
            MetricView::ModelUsage,
            MetricView::Performance,
            MetricView::Topics,
            MetricView::Timeline,
        ];
        
        Self {
            chat_state,
            time_range: config.default_time_range,
            config,
            metrics: None,
            selected_metric: 0,
            metric_views,
            last_update: std::time::Instant::now(),
            visualizer: MetricsVisualizer::new(),
        }
    }
    
    /// Update metrics if needed
    pub async fn update(&mut self) -> Result<()> {
        if self.config.real_time {
            let elapsed = self.last_update.elapsed().as_secs();
            if elapsed >= self.config.update_interval {
                self.refresh_metrics().await?;
                self.last_update = std::time::Instant::now();
            }
        }
        Ok(())
    }
    
    /// Refresh metrics from chat state
    pub async fn refresh_metrics(&mut self) -> Result<()> {
        let state = self.chat_state.read().await;
        self.metrics = Some(MetricsCalculator::calculate(&state, self.time_range));
        Ok(())
    }
    
    /// Handle keyboard input
    pub fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Left => self.previous_view(),
            KeyCode::Right => self.next_view(),
            KeyCode::Up => self.change_time_range(-1),
            KeyCode::Down => self.change_time_range(1),
            KeyCode::Char('r') => {
                // Force refresh
                self.last_update = std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(self.config.update_interval + 1))
                    .unwrap_or(std::time::Instant::now());
            }
            KeyCode::Char('a') => {
                self.config.show_advanced = !self.config.show_advanced;
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Go to previous metric view
    fn previous_view(&mut self) {
        if self.selected_metric > 0 {
            self.selected_metric -= 1;
        } else {
            self.selected_metric = self.metric_views.len() - 1;
        }
    }
    
    /// Go to next metric view
    fn next_view(&mut self) {
        self.selected_metric = (self.selected_metric + 1) % self.metric_views.len();
    }
    
    /// Change time range
    fn change_time_range(&mut self, delta: i32) {
        let ranges = vec![
            TimeRange::LastHour,
            TimeRange::Today,
            TimeRange::LastWeek,
            TimeRange::LastMonth,
            TimeRange::AllTime,
        ];
        
        let current_idx = ranges.iter()
            .position(|&r| std::mem::discriminant(&r) == std::mem::discriminant(&self.time_range))
            .unwrap_or(1);
        
        let new_idx = (current_idx as i32 + delta).clamp(0, ranges.len() as i32 - 1) as usize;
        self.time_range = ranges[new_idx];
        
        // Force refresh with new time range
        self.last_update = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(self.config.update_interval + 1))
            .unwrap_or(std::time::Instant::now());
    }
    
    /// Render the dashboard
    pub fn render(&self, f: &mut Frame, area: Rect) {
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(20),    // Content
                Constraint::Length(3),  // Footer
            ])
            .split(area);
        
        // Render header
        self.render_header(f, chunks[0]);
        
        // Render current metric view
        if let Some(metrics) = &self.metrics {
            match self.metric_views[self.selected_metric] {
                MetricView::Overview => self.render_overview(f, chunks[1], metrics),
                MetricView::MessageAnalysis => self.render_message_analysis(f, chunks[1], metrics),
                MetricView::ModelUsage => self.render_model_usage(f, chunks[1], metrics),
                MetricView::Performance => self.render_performance(f, chunks[1], metrics),
                MetricView::Topics => self.render_topics(f, chunks[1], metrics),
                MetricView::Timeline => self.render_timeline(f, chunks[1], metrics),
            }
        } else {
            self.render_loading(f, chunks[1]);
        }
        
        // Render footer
        self.render_footer(f, chunks[2]);
    }
    
    /// Render header
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let title = format!(
            " 📊 Chat Statistics - {} - {} ",
            match self.metric_views[self.selected_metric] {
                MetricView::Overview => "Overview",
                MetricView::MessageAnalysis => "Messages",
                MetricView::ModelUsage => "Models",
                MetricView::Performance => "Performance",
                MetricView::Topics => "Topics",
                MetricView::Timeline => "Timeline",
            },
            match self.time_range {
                TimeRange::LastHour => "Last Hour",
                TimeRange::Today => "Today",
                TimeRange::LastWeek => "Last Week",
                TimeRange::LastMonth => "Last Month",
                TimeRange::AllTime => "All Time",
                TimeRange::Custom { .. } => "Custom Range",
            }
        );
        
        let header = Paragraph::new(title)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded));
            
        f.render_widget(header, area);
    }
    
    /// Render footer with controls
    fn render_footer(&self, f: &mut Frame, area: Rect) {
        let controls = vec![
            Span::raw(" "),
            Span::styled("←→", Style::default().fg(Color::Yellow)),
            Span::raw(" Switch View | "),
            Span::styled("↑↓", Style::default().fg(Color::Yellow)),
            Span::raw(" Time Range | "),
            Span::styled("R", Style::default().fg(Color::Yellow)),
            Span::raw(" Refresh | "),
            Span::styled("A", Style::default().fg(Color::Yellow)),
            Span::raw(" Advanced"),
            if self.config.show_advanced {
                Span::styled(" ✓", Style::default().fg(Color::Green))
            } else {
                Span::raw("")
            },
            Span::raw(" "),
        ];
        
        let footer = Paragraph::new(Line::from(controls))
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);
            
        f.render_widget(footer, area);
    }
    
    /// Render loading state
    fn render_loading(&self, f: &mut Frame, area: Rect) {
        let loading = Paragraph::new("Loading statistics...")
            .style(Style::default().fg(Color::Gray))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
            
        f.render_widget(loading, area);
    }
    
    /// Render overview metrics
    fn render_overview(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),   // Key metrics
                Constraint::Length(10),  // Quality metrics
                Constraint::Min(10),     // Activity chart
            ])
            .split(area);
        
        // Key metrics
        self.render_key_metrics(f, chunks[0], metrics);
        
        // Quality metrics
        self.render_quality_metrics(f, chunks[1], metrics);
        
        // Activity chart
        self.render_activity_chart(f, chunks[2], metrics);
    }
    
    /// Render key metrics cards
    fn render_key_metrics(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area);
        
        // Total messages
        let messages_value = metrics.total_messages.to_string();
        let messages_card = self.create_metric_card(
            "Total Messages",
            &messages_value,
            Color::Cyan,
            "💬",
        );
        f.render_widget(messages_card, chunks[0]);
        
        // Total tokens
        let tokens_value = format_number(metrics.total_tokens);
        let tokens_card = self.create_metric_card(
            "Tokens Used",
            &tokens_value,
            Color::Green,
            "🪙",
        );
        f.render_widget(tokens_card, chunks[1]);
        
        // Error rate
        let error_rate = if metrics.total_messages > 0 {
            metrics.error_count as f64 / metrics.total_messages as f64 * 100.0
        } else {
            0.0
        };
        let error_value = format!("{:.1}%", error_rate);
        let error_card = self.create_metric_card(
            "Error Rate",
            &error_value,
            if error_rate > 5.0 { Color::Red } else { Color::Green },
            "⚠️",
        );
        f.render_widget(error_card, chunks[2]);
        
        // Avg response time
        let response_value = format!("{:.0}ms", metrics.avg_response_time);
        let response_card = self.create_metric_card(
            "Avg Response",
            &response_value,
            Color::Yellow,
            "⚡",
        );
        f.render_widget(response_card, chunks[3]);
    }
    
    /// Create a metric card widget
    fn create_metric_card<'a>(&self, title: &'a str, value: &'a str, color: Color, icon: &'a str) -> Paragraph<'a> {
        let content = vec![
            Line::from(vec![
                Span::raw(icon),
                Span::raw(" "),
                Span::styled(title, Style::default().fg(Color::Gray)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(value, Style::default().fg(color).add_modifier(Modifier::BOLD)),
            ]),
        ];
        
        Paragraph::new(content)
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded))
            .alignment(Alignment::Center)
    }
    
    /// Render quality metrics
    fn render_quality_metrics(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(area);
        
        // Success rate gauge
        let success_gauge = Gauge::default()
            .block(Block::default().title("Success Rate").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Green))
            .percent((metrics.quality_metrics.success_rate * 100.0) as u16)
            .label(format!("{:.1}%", metrics.quality_metrics.success_rate * 100.0));
        f.render_widget(success_gauge, chunks[0]);
        
        // Confidence gauge
        let confidence_gauge = Gauge::default()
            .block(Block::default().title("Avg Confidence").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent((metrics.quality_metrics.avg_confidence * 100.0) as u16)
            .label(format!("{:.1}%", metrics.quality_metrics.avg_confidence * 100.0));
        f.render_widget(confidence_gauge, chunks[1]);
        
        // Conversation length
        let conv_info = Paragraph::new(vec![
            Line::from("Avg Conversation"),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    format!("{:.1}", metrics.avg_conversation_length),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                ),
                Span::raw(" messages"),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
        f.render_widget(conv_info, chunks[2]);
    }
    
    /// Render activity chart
    fn render_activity_chart(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        // Convert hourly activity to sparkline data
        let data: Vec<u64> = (0..24)
            .map(|hour| {
                let hour_str = format!("{:02}:00", hour);
                metrics.hourly_activity.iter()
                    .find(|(h, _)| h == &hour_str)
                    .map(|(_, count)| *count as u64)
                    .unwrap_or(0)
            })
            .collect();
        
        let max = data.iter().max().copied().unwrap_or(1);
        
        let sparkline = Sparkline::default()
            .block(Block::default()
                .title("24-Hour Activity")
                .borders(Borders::ALL))
            .data(&data)
            .max(max)
            .style(Style::default().fg(Color::Cyan));
            
        f.render_widget(sparkline, area);
    }
    
    /// Render message analysis view
    fn render_message_analysis(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(area);
        
        // Message type breakdown
        // Sort the data before creating ListItems
        let mut type_data: Vec<_> = metrics.messages_by_type.iter().collect();
        type_data.sort_by_key(|(_, count)| *count);
        type_data.reverse();
        
        let type_items: Vec<ListItem> = type_data.into_iter()
            .map(|(msg_type, count)| {
                let percentage = *count as f64 / metrics.total_messages as f64 * 100.0;
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<10}", msg_type),
                        Style::default().fg(Color::Cyan)
                    ),
                    Span::raw(format!("{:>6} ", count)),
                    Span::styled(
                        format!("({:>5.1}%)", percentage),
                        Style::default().fg(Color::Gray)
                    ),
                ]))
            })
            .collect();
        
        let type_list = List::new(type_items)
            .block(Block::default()
                .title("Message Types")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(type_list, chunks[0]);
        
        // Peak hours
        let peak_hours_text = if metrics.peak_hours.is_empty() {
            vec![Line::from("No activity data")]
        } else {
            vec![
                Line::from("Most Active Hours:"),
                Line::from(""),
                Line::from(metrics.peak_hours.iter()
                    .map(|h| format!("{:02}:00", h))
                    .collect::<Vec<_>>()
                    .join(", ")),
                Line::from(""),
                Line::from(vec![
                    Span::raw("Total conversations: "),
                    Span::styled(
                        format!("{}", (metrics.total_messages as f64 / metrics.avg_conversation_length.max(1.0)) as usize),
                        Style::default().fg(Color::Yellow)
                    ),
                ]),
            ]
        };
        
        let peak_info = Paragraph::new(peak_hours_text)
            .block(Block::default()
                .title("Activity Insights")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded))
            .wrap(Wrap { trim: true });
        f.render_widget(peak_info, chunks[1]);
    }
    
    /// Render model usage view
    fn render_model_usage(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        if metrics.model_usage.is_empty() {
            let no_data = Paragraph::new("No model usage data available")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(no_data, area);
            return;
        }
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),    // Model list
                Constraint::Length(5),  // Total usage
            ])
            .split(area);
        
        // Model usage list
        // Sort the data before creating ListItems
        let mut model_data: Vec<_> = metrics.model_usage.iter().collect();
        model_data.sort_by_key(|(_, count)| *count);
        model_data.reverse();
        
        let total_model_usage: usize = metrics.model_usage.values().sum();
        let model_items: Vec<ListItem> = model_data.into_iter()
            .map(|(model, count)| {
                let percentage = *count as f64 / total_model_usage as f64 * 100.0;
                
                // Create a simple bar chart
                let bar_width = (percentage / 5.0) as usize;
                let bar = "█".repeat(bar_width);
                
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<20}", model),
                        Style::default().fg(Color::Cyan)
                    ),
                    Span::styled(bar, Style::default().fg(Color::Green)),
                    Span::raw(format!(" {:>6} ({:>5.1}%)", count, percentage)),
                ]))
            })
            .collect();
        
        let model_list = List::new(model_items)
            .block(Block::default()
                .title("Model Usage")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(model_list, chunks[0]);
        
        // Total model calls
        let total_calls: usize = metrics.model_usage.values().sum();
        let total_info = Paragraph::new(vec![
            Line::from(vec![
                Span::raw("Total API Calls: "),
                Span::styled(
                    format_number(total_calls),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
        f.render_widget(total_info, chunks[1]);
    }
    
    /// Render performance metrics view
    fn render_performance(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10),  // Response time info
                Constraint::Length(10),  // Token efficiency
                Constraint::Min(5),      // Additional metrics
            ])
            .split(area);
        
        // Response time analysis
        let response_analysis = vec![
            Line::from("Response Time Analysis"),
            Line::from(""),
            Line::from(vec![
                Span::raw("Average: "),
                Span::styled(
                    format!("{:.0}ms", metrics.avg_response_time),
                    Style::default().fg(Color::Yellow)
                ),
            ]),
            Line::from(vec![
                Span::raw("Quality: "),
                Span::styled(
                    if metrics.avg_response_time < 1000.0 { "Excellent" }
                    else if metrics.avg_response_time < 3000.0 { "Good" }
                    else if metrics.avg_response_time < 5000.0 { "Fair" }
                    else { "Needs Improvement" },
                    Style::default().fg(
                        if metrics.avg_response_time < 1000.0 { Color::Green }
                        else if metrics.avg_response_time < 3000.0 { Color::Yellow }
                        else { Color::Red }
                    )
                ),
            ]),
        ];
        
        let response_widget = Paragraph::new(response_analysis)
            .block(Block::default()
                .title("Response Performance")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(response_widget, chunks[0]);
        
        // Token efficiency
        let tokens_per_message = if metrics.total_messages > 0 {
            metrics.total_tokens as f64 / metrics.total_messages as f64
        } else {
            0.0
        };
        
        let token_analysis = vec![
            Line::from("Token Efficiency"),
            Line::from(""),
            Line::from(vec![
                Span::raw("Total Tokens: "),
                Span::styled(
                    format_number(metrics.total_tokens),
                    Style::default().fg(Color::Cyan)
                ),
            ]),
            Line::from(vec![
                Span::raw("Per Message: "),
                Span::styled(
                    format!("{:.0}", tokens_per_message),
                    Style::default().fg(Color::Green)
                ),
            ]),
            Line::from(vec![
                Span::raw("Estimated Cost: "),
                Span::styled(
                    format!("${:.2}", estimate_cost(metrics.total_tokens)),
                    Style::default().fg(Color::Yellow)
                ),
            ]),
        ];
        
        let token_widget = Paragraph::new(token_analysis)
            .block(Block::default()
                .title("Token Usage")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
        f.render_widget(token_widget, chunks[1]);
        
        // Additional metrics
        if self.config.show_advanced {
            let advanced_metrics = vec![
                Line::from(vec![
                    Span::raw("Regeneration Rate: "),
                    Span::styled(
                        format!("{:.1}%", metrics.quality_metrics.regeneration_rate * 100.0),
                        Style::default().fg(Color::Yellow)
                    ),
                ]),
                Line::from(vec![
                    Span::raw("Error Recovery: "),
                    Span::styled(
                        if metrics.error_count == 0 { "Perfect" } else { "Active" },
                        Style::default().fg(if metrics.error_count == 0 { Color::Green } else { Color::Yellow })
                    ),
                ]),
            ];
            
            let advanced_widget = Paragraph::new(advanced_metrics)
                .block(Block::default()
                    .title("Advanced Metrics")
                    .borders(Borders::ALL));
            f.render_widget(advanced_widget, chunks[2]);
        }
    }
    
    /// Render topics view
    fn render_topics(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        if metrics.topic_distribution.is_empty() {
            let no_topics = Paragraph::new("No topic data available")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(no_topics, area);
            return;
        }
        
        // Create topic visualization
        let mut topics: Vec<_> = metrics.topic_distribution.iter().collect();
        topics.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        
        let total_topic_mentions: usize = topics.iter().map(|(_, c)| **c).sum();
        
        let topic_items: Vec<ListItem> = topics.into_iter()
            .take(10) // Top 10 topics
            .map(|(topic, count)| {
                let percentage = *count as f64 / total_topic_mentions as f64 * 100.0;
                let bar_width = (percentage / 2.0) as usize;
                let bar = "▓".repeat(bar_width);
                
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<15}", topic),
                        Style::default().fg(Color::Cyan)
                    ),
                    Span::styled(bar, Style::default().fg(Color::Blue)),
                    Span::raw(format!(" {} ({:.1}%)", count, percentage)),
                ]))
            })
            .collect();
        
        let topic_list = List::new(topic_items)
            .block(Block::default()
                .title("Topic Distribution")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
                
        f.render_widget(topic_list, area);
    }
    
    /// Render timeline view
    fn render_timeline(&self, f: &mut Frame, area: Rect, metrics: &ChatMetrics) {
        // Hourly activity timeline
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(15),    // Activity graph
                Constraint::Length(5),  // Summary
            ])
            .split(area);
        
        // Create hourly bars
        let mut lines = vec![
            Line::from("24-Hour Message Timeline"),
            Line::from(""),
        ];
        
        // Find max for scaling
        let max_count = metrics.hourly_activity.iter()
            .map(|(_, count)| *count)
            .max()
            .unwrap_or(1);
        
        for hour in 0..24 {
            let hour_str = format!("{:02}:00", hour);
            let count = metrics.hourly_activity.iter()
                .find(|(h, _)| h == &hour_str)
                .map(|(_, c)| *c)
                .unwrap_or(0);
            
            let bar_width = if max_count > 0 {
                (count as f64 / max_count as f64 * 40.0) as usize
            } else {
                0
            };
            
            let bar = "█".repeat(bar_width);
            let color = if metrics.peak_hours.contains(&(hour as u32)) {
                Color::Yellow
            } else {
                Color::Cyan
            };
            
            lines.push(Line::from(vec![
                Span::raw(format!("{:02} ", hour)),
                Span::styled(bar, Style::default().fg(color)),
                Span::raw(format!(" {}", count)),
            ]));
        }
        
        let timeline = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded));
                
        f.render_widget(timeline, chunks[0]);
        
        // Summary
        let summary = Paragraph::new(vec![
            Line::from(vec![
                Span::raw("Total Messages: "),
                Span::styled(
                    metrics.total_messages.to_string(),
                    Style::default().fg(Color::Yellow)
                ),
                Span::raw(" | Time Range: "),
                Span::styled(
                    &metrics.time_range,
                    Style::default().fg(Color::Cyan)
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
        
        f.render_widget(summary, chunks[1]);
    }
}

/// Format large numbers with commas
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;
    
    for c in s.chars().rev() {
        if count == 3 {
            result.push(',');
            count = 0;
        }
        result.push(c);
        count += 1;
    }
    
    result.chars().rev().collect()
}

/// Estimate cost based on token usage (rough estimate)
fn estimate_cost(tokens: usize) -> f64 {
    // Rough estimate: $0.002 per 1K tokens (GPT-3.5 pricing)
    tokens as f64 * 0.002 / 1000.0
}