// Enhanced Custom widgets for the TUI with improved visual quality and animations
// This module contains specialized widgets for cognitive workstation
// visualization with immersive graphics and smooth transitions

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style, Modifier};
use ratatui::widgets::{
    Axis,
    Block,
    Borders,
    Cell,
    Chart,
    Dataset,
    Gauge,
    List,
    ListItem,
    Paragraph,
    Row,
    Sparkline,
    Table,
};
use ratatui::{Frame, symbols};
use ratatui::text::{Line, Span};
use sysinfo::System;

use crate::cluster::ClusterStats;
use crate::compute::{Device, MemoryInfo};

/// Enhanced animation state for smooth transitions
#[derive(Clone, Debug)]
pub struct AnimationState {
    pub start_time: Instant,
    pub duration: Duration,
    pub current_frame: u32,
    pub total_frames: u32,
    pub is_active: bool,
}

impl AnimationState {
    pub fn new(duration: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            duration,
            current_frame: 0,
            total_frames: 60, // 60 FPS for smooth animation
            is_active: true,
        }
    }

    pub fn progress(&self) -> f32 {
        let elapsed = self.start_time.elapsed();
        if elapsed >= self.duration {
            1.0
        } else {
            elapsed.as_secs_f32() / self.duration.as_secs_f32()
        }
    }

    pub fn is_complete(&self) -> bool {
        self.start_time.elapsed() >= self.duration
    }

    pub fn easing_out_cubic(&self) -> f32 {
        let t = self.progress();
        1.0 - (1.0 - t).powi(3)
    }

    pub fn easing_in_out_cubic(&self) -> f32 {
        let t = self.progress();
        if t < 0.5 {
            4.0 * t * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
        }
    }
}

/// Enhanced GPU usage widget with immersive thermal display and performance metrics
pub struct GpuWidget {
    pub title: String,
    pub devices: Vec<Device>,
    pub memory_info: Vec<(String, MemoryInfo)>,
    pub temperature_history: VecDeque<f64>,
    pub utilization_history: VecDeque<f64>,
    pub power_draw_history: VecDeque<f64>,
    pub max_history_size: usize,
    pub animation_state: AnimationState,
    pub pulse_animation: f32,
    pub last_update: Instant,
}

impl GpuWidget {
    /// Create a new enhanced GPU widget
    pub fn new(title: String) -> Self {
        Self {
            title,
            devices: Vec::new(),
            memory_info: Vec::new(),
            temperature_history: VecDeque::with_capacity(60),
            utilization_history: VecDeque::with_capacity(60),
            power_draw_history: VecDeque::with_capacity(60),
            max_history_size: 60,
            animation_state: AnimationState::new(Duration::from_millis(1000)),
            pulse_animation: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Update GPU metrics with smooth animations
    pub fn update_metrics(
        &mut self,
        devices: Vec<Device>,
        memory_info: Vec<(String, MemoryInfo)>,
        current_temp: f32,
        current_util: f32,
        current_power: f32,
    ) {
        self.devices = devices;
        self.memory_info = memory_info;
        self.last_update = Instant::now();

        // Update pulse animation for active state
        self.pulse_animation = (self.last_update.elapsed().as_secs_f32() * 2.0).sin().abs();

        // Smooth data transitions
        Self::add_metric_smoothly(&mut self.temperature_history, current_temp as f64, self.max_history_size);
        Self::add_metric_smoothly(&mut self.utilization_history, current_util as f64, self.max_history_size);
        Self::add_metric_smoothly(&mut self.power_draw_history, current_power as f64, self.max_history_size);

        // Reset animation for new data
        self.animation_state = AnimationState::new(Duration::from_millis(800));
    }

    /// Add metric with smooth interpolation
    fn add_metric_smoothly(history: &mut VecDeque<f64>, new_value: f64, max_history_size: usize) {
        if history.len() >= max_history_size {
            history.pop_front();
        }

        // Smooth interpolation with previous value
        let interpolated_value = if let Some(&last_value) = history.back() {
            last_value + (new_value - last_value) * 0.3 // Smooth transition
        } else {
            new_value
        };

        history.push_back(interpolated_value);
    }

    /// Render the enhanced GPU widget with animations
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Enhanced header with animations
                Constraint::Min(10),   // Enhanced charts area
                Constraint::Length(8), // Enhanced device list
            ])
            .split(area);

        // Enhanced header with pulsing effects and gradients
        self.render_enhanced_header(f, chunks[0]);

        // Enhanced charts area with smooth animations
        self.render_enhanced_charts(f, chunks[1]);

        // Enhanced device list with status indicators
        self.render_enhanced_device_list(f, chunks[2]);
    }

    /// Render enhanced header with visual effects
    fn render_enhanced_header(&self, f: &mut Frame, area: Rect) {
        let gpu_count = self.devices.iter().filter(|d| d.is_gpu()).count();
        let current_temp = self.temperature_history.back().unwrap_or(&0.0);
        let current_util = self.utilization_history.back().unwrap_or(&0.0);
        let current_power = self.power_draw_history.back().unwrap_or(&0.0);

        // Dynamic color based on temperature
        let temp_color = if *current_temp > 80.0 {
            Color::Red
        } else if *current_temp > 60.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        // Pulsing effect for active GPUs
        let pulse_intensity = (self.pulse_animation * 255.0) as u8;
        let pulse_color = Color::Rgb(0, pulse_intensity, 255 - pulse_intensity);

        let header_spans = vec![
            Span::styled("🔥 ", Style::default().fg(temp_color)),
            Span::styled(
                format!("{} GPU", gpu_count),
                Style::default().fg(pulse_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("🌡️ ", Style::default().fg(temp_color)),
            Span::styled(
                format!("{:.1}°C", current_temp),
                Style::default().fg(temp_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("⚡ ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{:.1}%", current_util),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("🔋 ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{:.0}W", current_power),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            ),
        ];

        let header_line = Line::from(header_spans);
        let header_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(pulse_color))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ));

        let header_para = Paragraph::new(vec![Line::from(""), header_line])
            .block(header_block)
            .style(Style::default().fg(Color::White));

        f.render_widget(header_para, area);
    }

    /// Render enhanced charts with smooth animations and gradients
    fn render_enhanced_charts(&self, f: &mut Frame, area: Rect) {
        let chart_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(33),
                Constraint::Percentage(34),
            ])
            .split(area);

        // Enhanced temperature chart with thermal gradient
        self.render_thermal_chart(f, chart_chunks[0]);

        // Enhanced utilization chart with performance indicators
        self.render_utilization_chart(f, chart_chunks[1]);

        // Enhanced power chart with efficiency metrics
        self.render_power_chart(f, chart_chunks[2]);
    }

    /// Render thermal chart with gradient colors
    fn render_thermal_chart(&self, f: &mut Frame, area: Rect) {
        if self.temperature_history.is_empty() {
            let empty_chart = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" 🌡️ Temperature ");
            f.render_widget(empty_chart, area);
            return;
        }

        let current_temp = self.temperature_history.back().unwrap_or(&0.0);
        let max_temp: f64 = self.temperature_history.iter().fold(0.0, |a, &b| a.max(b));

        // Dynamic color based on temperature
        let chart_color = if *current_temp > 80.0 {
            Color::Red
        } else if *current_temp > 60.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        let border_color = if *current_temp > 85.0 {
            Color::Red // Critical temperature warning
        } else {
            chart_color
        };

        let points: Vec<(f64, f64)> = self.temperature_history
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as f64, val))
            .collect();

        let datasets = vec![
            Dataset::default()
                .name("Temperature")
                .marker(symbols::Marker::Braille)
                .style(Style::default().fg(chart_color))
                .data(&points),
        ];

        let title = format!(" 🌡️ Temperature ({:.1}°C) ", current_temp);
        let chart = Chart::new(datasets)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))
                .title(Span::styled(title, Style::default().fg(chart_color).add_modifier(Modifier::BOLD))))
            .x_axis(
                Axis::default()
                    .title("Time")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, self.max_history_size as f64]),
            )
            .y_axis(
                Axis::default()
                    .title("°C")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([20.0, max_temp.max(90.0)]),
            );

        f.render_widget(chart, area);
    }

    /// Render utilization chart with performance indicators
    fn render_utilization_chart(&self, f: &mut Frame, area: Rect) {
        if self.utilization_history.is_empty() {
            return;
        }

        let current_util = self.utilization_history.back().unwrap_or(&0.0);
        let avg_util = self.utilization_history.iter().sum::<f64>() / self.utilization_history.len() as f64;

        // Performance-based coloring
        let chart_color = if *current_util > 90.0 {
            Color::Red
        } else if *current_util > 70.0 {
            Color::Yellow
        } else if *current_util > 30.0 {
            Color::Green
        } else {
            Color::Blue
        };

        let points: Vec<(f64, f64)> = self.utilization_history
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as f64, val))
            .collect();

        let datasets = vec![
            Dataset::default()
                .name("Utilization")
                .marker(symbols::Marker::Braille)
                .style(Style::default().fg(chart_color))
                .data(&points),
        ];

        let title = format!(" ⚡ Utilization ({:.1}% avg: {:.1}%) ", current_util, avg_util);
        let chart = Chart::new(datasets)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(chart_color))
                .title(Span::styled(title, Style::default().fg(chart_color).add_modifier(Modifier::BOLD))))
            .x_axis(
                Axis::default()
                    .title("Time")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, self.max_history_size as f64]),
            )
            .y_axis(
                Axis::default()
                    .title("%")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, 100.0]),
            );

        f.render_widget(chart, area);
    }

    /// Render power chart with efficiency metrics
    fn render_power_chart(&self, f: &mut Frame, area: Rect) {
        if self.power_draw_history.is_empty() {
            return;
        }

        let current_power = self.power_draw_history.back().unwrap_or(&0.0);
        let max_power: f64 = self.power_draw_history.iter().fold(0.0, |a, &b| a.max(b));

        // Power efficiency coloring
        let chart_color = if *current_power > 250.0 {
            Color::Red
        } else if *current_power > 150.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        let points: Vec<(f64, f64)> = self.power_draw_history
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as f64, val))
            .collect();

        let datasets = vec![
            Dataset::default()
                .name("Power")
                .marker(symbols::Marker::Braille)
                .style(Style::default().fg(chart_color))
                .data(&points),
        ];

        let efficiency = if *current_power > 0.0 {
            (self.utilization_history.back().unwrap_or(&0.0) / current_power) * 100.0
        } else {
            0.0
        };

        let title = format!(" 🔋 Power ({:.0}W eff: {:.1}) ", current_power, efficiency);
        let chart = Chart::new(datasets)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(chart_color))
                .title(Span::styled(title, Style::default().fg(chart_color).add_modifier(Modifier::BOLD))))
            .x_axis(
                Axis::default()
                    .title("Time")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, self.max_history_size as f64]),
            )
            .y_axis(
                Axis::default()
                    .title("Watts")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([0.0, max_power.max(300.0)]),
            );

        f.render_widget(chart, area);
    }

    /// Render enhanced device list with status indicators and animations
    fn render_enhanced_device_list(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.devices
            .iter()
            .enumerate()
            .map(|(i, device)| {
                let memory_text = if let Some((_, memory)) = self.memory_info.get(i) {
                    let usage_percent = (memory.used as f64 / memory.total as f64) * 100.0;
                    let usage_bar = self.create_memory_bar(usage_percent);
                    format!(
                        " │ {} {:.1}/{:.1} GB",
                        usage_bar,
                        memory.used as f64 / 1_000_000_000.0,
                        memory.total as f64 / 1_000_000_000.0
                    )
                } else {
                    String::new()
                };

                let (device_icon, device_color) = match device.device_type {
                    crate::compute::DeviceType::Cuda => ("🟢 CUDA", Color::Green),
                    crate::compute::DeviceType::Metal => ("🟠 Metal", Color::Yellow),
                    crate::compute::DeviceType::Cpu => ("🔵 CPU", Color::Blue),
                    crate::compute::DeviceType::OpenCL => ("🟡 OpenCL", Color::Yellow),
                };

                // Status indicator with animation
                let status_indicator = if device.is_gpu() {
                    let pulse = (self.pulse_animation * 10.0) as u8;
                    format!("⚡{}", "█".repeat((pulse % 3 + 1) as usize))
                } else {
                    "●".to_string()
                };

                let spans = vec![
                    Span::styled(device_icon, Style::default().fg(device_color).add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(&device.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::styled(memory_text, Style::default().fg(Color::Cyan)),
                    Span::raw(" "),
                    Span::styled(status_indicator, Style::default().fg(device_color)),
                ];

                ListItem::new(Line::from(spans))
            })
            .collect();

        let list_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Span::styled(
                " 🖥️ Devices ",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ));

        let list = List::new(items)
            .block(list_block)
            .style(Style::default().fg(Color::White));

        f.render_widget(list, area);
    }

    /// Create a visual memory usage bar
    fn create_memory_bar(&self, usage_percent: f64) -> String {
        let bar_length = 8;
        let filled_length = ((usage_percent / 100.0) * bar_length as f64) as usize;
        let empty_length = bar_length - filled_length;

        let filled_char = if usage_percent > 80.0 {
            "█" // Critical
        } else if usage_percent > 60.0 {
            "▓" // Warning
        } else {
            "▒" // Normal
        };

        format!("{}{}",
            filled_char.repeat(filled_length),
            "░".repeat(empty_length)
        )
    }
}

/// Enhanced Stream visualization widget with real-time data flow monitoring and animations
pub struct StreamWidget {
    pub title: String,
    pub active_streams: Vec<(String, String)>,
    pub throughput_history: VecDeque<f64>,
    pub latency_history: VecDeque<f64>,
    pub error_rate_history: VecDeque<f64>,
    pub max_history_size: usize,
    pub animation_state: AnimationState,
    pub flow_animation: f32,
    pub last_update: Instant,
}

impl StreamWidget {
    /// Create a new enhanced stream widget
    pub fn new(title: String) -> Self {
        Self {
            title,
            active_streams: Vec::new(),
            throughput_history: VecDeque::with_capacity(60),
            latency_history: VecDeque::with_capacity(60),
            error_rate_history: VecDeque::with_capacity(60),
            max_history_size: 60,
            animation_state: AnimationState::new(Duration::from_millis(1500)),
            flow_animation: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Update stream metrics with smooth animations
    pub fn update_metrics(
        &mut self,
        current_throughput: f64,
        current_latency: f64,
        current_error_rate: f64,
    ) {
        self.last_update = Instant::now();

        // Update flow animation for data streams
        self.flow_animation = (self.last_update.elapsed().as_secs_f32() * 3.0).sin();

        // Smooth data transitions
        Self::add_metric_smoothly(&mut self.throughput_history, current_throughput, self.max_history_size);
        Self::add_metric_smoothly(&mut self.latency_history, current_latency, self.max_history_size);
        Self::add_metric_smoothly(&mut self.error_rate_history, current_error_rate.max(0.0), self.max_history_size);

        // Reset animation for new data
        self.animation_state = AnimationState::new(Duration::from_millis(1200));
    }

    /// Add metric with smooth interpolation
    fn add_metric_smoothly(history: &mut VecDeque<f64>, new_value: f64, max_history_size: usize) {
        if history.len() >= max_history_size {
            history.pop_front();
        }

        let interpolated_value = if let Some(&last_value) = history.back() {
            last_value + (new_value - last_value) * 0.4
        } else {
            new_value
        };

        history.push_back(interpolated_value);
    }

    /// Render the enhanced stream widget with flow animations
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Enhanced header with flow indicators
                Constraint::Min(8),    // Enhanced metrics with animations
                Constraint::Length(10), // Enhanced stream list
            ])
            .split(area);

        // Enhanced header with streaming effects
        self.render_streaming_header(f, chunks[0]);

        // Enhanced metrics sparklines with flow animations
        self.render_animated_metrics(f, chunks[1]);

        // Enhanced stream list with status animations
        self.render_animated_stream_list(f, chunks[2]);
    }

    /// Render streaming header with flow effects
    fn render_streaming_header(&self, f: &mut Frame, area: Rect) {
        let current_throughput = self.throughput_history.back().unwrap_or(&0.0);
        let current_latency = self.latency_history.back().unwrap_or(&0.0);
        let current_error_rate = self.error_rate_history.back().unwrap_or(&0.0);

        // Flow animation characters
        let flow_chars = ["▶", "▷", "▶", "▷"];
        let flow_index = ((self.flow_animation + 1.0) * 2.0) as usize % flow_chars.len();
        let flow_char = flow_chars[flow_index];

        // Dynamic colors based on performance
        let throughput_color = if *current_throughput > 200.0 {
            Color::Green
        } else if *current_throughput > 100.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let latency_color = if *current_latency < 20.0 {
            Color::Green
        } else if *current_latency < 50.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let header_spans = vec![
            Span::styled(flow_char, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
            Span::styled(
                format!("{} Streams", self.active_streams.len()),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("📊 ", Style::default().fg(throughput_color)),
            Span::styled(
                format!("{:.0} msg/s", current_throughput),
                Style::default().fg(throughput_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("⏱️ ", Style::default().fg(latency_color)),
            Span::styled(
                format!("{:.1}ms", current_latency),
                Style::default().fg(latency_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("⚠️ ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("{:.2}%", current_error_rate * 100.0),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            ),
        ];

        let header_line = Line::from(header_spans);
        let header_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ));

        let header_para = Paragraph::new(vec![Line::from(""), header_line])
            .block(header_block)
            .style(Style::default().fg(Color::White));

        f.render_widget(header_para, area);
    }

    /// Render animated metrics with enhanced sparklines
    fn render_animated_metrics(&self, f: &mut Frame, area: Rect) {
        let metric_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(33),
                Constraint::Percentage(34),
            ])
            .split(area);

        self.render_enhanced_sparkline(
            f,
            metric_chunks[0],
            "📊 Throughput",
            &self.throughput_history,
            Color::Green,
            "msg/s",
        );

        self.render_enhanced_sparkline(
            f,
            metric_chunks[1],
            "⏱️ Latency",
            &self.latency_history,
            Color::Yellow,
            "ms",
        );

        self.render_enhanced_sparkline(
            f,
            metric_chunks[2],
            "⚠️ Error Rate",
            &self.error_rate_history,
            Color::Red,
            "%",
        );
    }

    /// Render enhanced sparkline with better visualization
    fn render_enhanced_sparkline(
        &self,
        f: &mut Frame,
        area: Rect,
        title: &str,
        data: &VecDeque<f64>,
        color: Color,
        unit: &str,
    ) {
        let current_value = data.back().unwrap_or(&0.0);
        let data_points: Vec<u64> = data.iter().map(|&x| (x * 10.0) as u64).collect();

        // Dynamic border color based on value trends
        let trend_color = if data.len() >= 2 {
            let prev_value = data.get(data.len() - 2).unwrap_or(&0.0);
            if current_value > prev_value {
                Color::Green
            } else if current_value < prev_value {
                Color::Red
            } else {
                color
            }
        } else {
            color
        };

        let title_with_value = format!(" {} ({:.1}{}) ", title, current_value, unit);
        let sparkline = Sparkline::default()
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(trend_color))
                .title(Span::styled(title_with_value, Style::default().fg(color).add_modifier(Modifier::BOLD))))
            .data(&data_points)
            .style(Style::default().fg(color));

        f.render_widget(sparkline, area);
    }

    /// Render animated stream list with status indicators
    fn render_animated_stream_list(&self, f: &mut Frame, area: Rect) {
        let rows: Vec<Row> = self.active_streams
            .iter()
            .enumerate()
            .map(|(i, (stream_id, status))| {
                // Generate realistic stream metrics
                let throughput = 180.0 + (i as f64 * 25.0);
                let latency = 8.5 + (i as f64 * 2.1);
                let buffer_usage = 45.0 + (i as f64 * 12.0);

                // Animated status indicators
                let status_indicator = match status.as_str() {
                    "Active" => {
                        let pulse = ((self.flow_animation + 1.0) * 127.0) as u8;
                        format!("🟢 {}", "●".repeat((pulse % 3 + 1) as usize))
                    },
                    "Error" => "🔴 ⚠️".to_string(),
                    "Paused" => "🟡 ⏸️".to_string(),
                    "Initializing" => {
                        let dots = ".".repeat(((self.flow_animation + 1.0) * 3.0) as usize % 4);
                        format!("🔵 {}", dots)
                    },
                    _ => "⚫ ●".to_string(),
                };

                // Buffer usage bar
                let buffer_bar = self.create_buffer_bar(buffer_usage);

                Row::new(vec![
                    Cell::from(stream_id.clone()),
                    Cell::from(status_indicator.to_string()),
                    Cell::from(format!("{:.1} msg/s", throughput)),
                    Cell::from(format!("{:.1} ms", latency)),
                    Cell::from(buffer_bar),
                ])
            })
            .collect();

        let widths = [
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(20),
        ];

        let table = Table::new(rows, widths)
            .header(Row::new(vec![
                Cell::from("Stream ID").style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Cell::from("Status").style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Cell::from("Throughput").style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Cell::from("Latency").style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Cell::from("Buffer").style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ]))
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " 🌊 Active Streams ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                )));

        f.render_widget(table, area);
    }

    /// Create a visual buffer usage bar
    fn create_buffer_bar(&self, usage_percent: f64) -> String {
        let bar_length = 10;
        let filled_length = ((usage_percent / 100.0) * bar_length as f64) as usize;
        let empty_length = bar_length - filled_length;

        let filled_char = if usage_percent > 85.0 {
            "█" // Critical
        } else if usage_percent > 70.0 {
            "▓" // Warning
        } else {
            "▒" // Normal
        };

        format!("{}{}",
            filled_char.repeat(filled_length),
            "░".repeat(empty_length)
        )
    }
}

/// Enhanced Node topology widget for immersive cluster visualization
pub struct TopologyWidget {
    pub title: String,
    pub cluster_stats: ClusterStats,
    pub node_health: Vec<NodeHealth>,
    pub network_utilization: VecDeque<f64>,
    pub max_history_size: usize,
    pub animation_state: AnimationState,
    pub network_pulse: f32,
    pub last_update: Instant,
}

/// Enhanced Node health information with visual indicators
#[derive(Clone)]
pub struct NodeHealth {
    pub node_id: String,
    pub status: NodeStatus,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_in: f64,
    pub network_out: f64,
    pub load_balance_weight: f64,
    pub health_score: f64,
    pub response_time: f64,
}

/// Enhanced Node status enumeration with visual representations
#[derive(Clone)]
pub enum NodeStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
    Maintenance,
    Recovering,
}

impl NodeStatus {
    pub fn get_icon(&self) -> &'static str {
        match self {
            NodeStatus::Healthy => "🟢",
            NodeStatus::Warning => "🟡",
            NodeStatus::Critical => "🔴",
            NodeStatus::Offline => "⚫",
            NodeStatus::Maintenance => "🔧",
            NodeStatus::Recovering => "🔄",
        }
    }

    pub fn get_color(&self) -> Color {
        match self {
            NodeStatus::Healthy => Color::Green,
            NodeStatus::Warning => Color::Yellow,
            NodeStatus::Critical => Color::Red,
            NodeStatus::Offline => Color::DarkGray,
            NodeStatus::Maintenance => Color::Blue,
            NodeStatus::Recovering => Color::Cyan,
        }
    }

    pub fn get_pulse_rate(&self) -> f32 {
        match self {
            NodeStatus::Healthy => 1.0,
            NodeStatus::Warning => 1.5,
            NodeStatus::Critical => 3.0,
            NodeStatus::Offline => 0.0,
            NodeStatus::Maintenance => 0.5,
            NodeStatus::Recovering => 2.0,
        }
    }
}

impl TopologyWidget {
    /// Create a new enhanced topology widget
    pub fn new(title: String) -> Self {
        Self {
            title,
            cluster_stats: ClusterStats::default(),
            node_health: Vec::new(),
            network_utilization: VecDeque::with_capacity(60),
            max_history_size: 60,
            animation_state: AnimationState::new(Duration::from_millis(2000)),
            network_pulse: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Update cluster metrics with enhanced animations
    pub fn update_metrics(&mut self, cluster_stats: ClusterStats) {
        let total_nodes = cluster_stats.total_nodes;
        self.cluster_stats = cluster_stats;
        self.last_update = Instant::now();

        // Update network pulse animation
        self.network_pulse = (self.last_update.elapsed().as_secs_f32() * 2.5).sin();

        // Generate enhanced node health data with realistic metrics
        self.node_health = (0..total_nodes)
            .map(|i| {
                let base_health = 85.0 + (i as f64 * 3.0) % 15.0;
                let cpu_usage = 30.0 + (i as f64 * 8.0) % 40.0;
                let memory_usage = 50.0 + (i as f64 * 12.0) % 30.0;
                let network_in = (i as f64 * 15.0) % 100.0;
                let network_out = (i as f64 * 12.0) % 80.0;
                let response_time = 10.0 + (i as f64 * 5.0) % 20.0;

                // Determine status based on health metrics
                let status = if base_health < 50.0 {
                    NodeStatus::Critical
                } else if base_health < 70.0 {
                    NodeStatus::Warning
                } else if cpu_usage > 90.0 || memory_usage > 95.0 {
                    NodeStatus::Warning
                } else {
                    NodeStatus::Healthy
                };

                NodeHealth {
                    node_id: format!("node-{:03}", i + 1),
                    status,
                    cpu_usage,
                    memory_usage,
                    network_in,
                    network_out,
                    load_balance_weight: 0.5 + (i as f64 * 0.1) % 0.5,
                    health_score: base_health,
                    response_time,
                }
            })
            .collect();

        // Update network utilization with smooth transitions
        let current_network = (total_nodes as f64 * 15.0) % 100.0;
        self.add_network_metric_smoothly(current_network);

        // Reset animation for new data
        self.animation_state = AnimationState::new(Duration::from_millis(1800));
    }

    /// Add network metric with smooth interpolation
    fn add_network_metric_smoothly(&mut self, new_value: f64) {
        if self.network_utilization.len() >= self.max_history_size {
            self.network_utilization.pop_front();
        }

        let interpolated_value = if let Some(&last_value) = self.network_utilization.back() {
            last_value + (new_value - last_value) * 0.3
        } else {
            new_value
        };

        self.network_utilization.push_back(interpolated_value);
    }

    /// Render the enhanced topology widget with network animations
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Enhanced header with network status
                Constraint::Length(8), // Enhanced cluster overview with gauges
                Constraint::Min(10),   // Enhanced node details table
            ])
            .split(area);

        // Enhanced header with network pulse effects
        self.render_network_header(f, chunks[0]);

        // Enhanced cluster overview with animated gauges
        self.render_enhanced_cluster_overview(f, chunks[1]);

        // Enhanced node details table with health indicators
        self.render_enhanced_node_table(f, chunks[2]);
    }

    /// Render network header with pulse effects
    fn render_network_header(&self, f: &mut Frame, area: Rect) {
        let healthy_nodes = self.node_health.iter()
            .filter(|n| matches!(n.status, NodeStatus::Healthy))
            .count();
        let warning_nodes = self.node_health.iter()
            .filter(|n| matches!(n.status, NodeStatus::Warning))
            .count();
        let critical_nodes = self.node_health.iter()
            .filter(|n| matches!(n.status, NodeStatus::Critical))
            .count();

        // Network pulse animation
        let pulse_intensity = ((self.network_pulse.abs() * 128.0) + 127.0) as u8;
        let pulse_color = Color::Rgb(pulse_intensity, 255 - pulse_intensity, 128);

        // Dynamic header color based on cluster health
        let header_color = if critical_nodes > 0 {
            Color::Red
        } else if warning_nodes > 0 {
            Color::Yellow
        } else {
            Color::Green
        };

        let header_spans = vec![
            Span::styled("🌐 ", Style::default().fg(pulse_color).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("Cluster ({}/{})", healthy_nodes, self.cluster_stats.total_nodes),
                Style::default().fg(header_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("🟢 ", Style::default().fg(Color::Green)),
            Span::styled(
                format!("{}", healthy_nodes),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("🟡 ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}", warning_nodes),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("🔴 ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("{}", critical_nodes),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("⚡ ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{} req/s", self.cluster_stats.active_requests),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
        ];

        let header_line = Line::from(header_spans);
        let header_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(pulse_color))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ));

        let header_para = Paragraph::new(vec![Line::from(""), header_line])
            .block(header_block)
            .style(Style::default().fg(Color::White));

        f.render_widget(header_para, area);
    }

    /// Render enhanced cluster overview with animated gauges
    fn render_enhanced_cluster_overview(&self, f: &mut Frame, area: Rect) {
        let overview_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
            ])
            .split(area);

        // Calculate enhanced metrics
        let cluster_health = if self.cluster_stats.total_nodes == 0 {
            0.0
        } else {
            let healthy_count = self.node_health.iter()
                .filter(|n| matches!(n.status, NodeStatus::Healthy))
                .count();
            (healthy_count as f32 / self.cluster_stats.total_nodes as f32) * 100.0
        };

        let avg_utilization = (self.cluster_stats.avg_compute_usage + self.cluster_stats.avg_memory_usage) / 2.0;
        let network_utilization = self.network_utilization.back().unwrap_or(&0.0);
        let response_time = self.node_health.iter()
            .map(|n| n.response_time)
            .sum::<f64>() / self.node_health.len().max(1) as f64;

        // Enhanced gauges with dynamic colors and animations
        self.render_enhanced_gauge(f, overview_chunks[0], "🏥 Health", cluster_health as f64, Color::Green);
        self.render_enhanced_gauge(f, overview_chunks[1], "⚡ CPU/Mem", avg_utilization as f64, Color::Blue);
        self.render_enhanced_gauge(f, overview_chunks[2], "🌐 Network", *network_utilization, Color::Cyan);
        self.render_enhanced_gauge(f, overview_chunks[3], "⏱️ Response", response_time.min(100.0), Color::Yellow);

        // Load distribution gauge
        let load_balance = 85.0 + (self.network_pulse * 10.0) as f64;
        self.render_enhanced_gauge(f, overview_chunks[4], "⚖️ Balance", load_balance, Color::Magenta);
    }

    /// Render enhanced gauge with dynamic colors and effects
    fn render_enhanced_gauge(&self, f: &mut Frame, area: Rect, title: &str, value: f64, base_color: Color) {
        // Dynamic color based on value
        let gauge_color = match value {
            v if v > 85.0 => Color::Red,
            v if v > 70.0 => Color::Yellow,
            v if v > 50.0 => Color::Green,
            _ => base_color,
        };

        // Animated border based on pulse
        let border_intensity = ((self.network_pulse.abs() * 128.0) + 127.0) as u8;
        let border_color = Color::Rgb(border_intensity, border_intensity, 255);

        let gauge = Gauge::default()
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))
                .title(Span::styled(title, Style::default().fg(gauge_color).add_modifier(Modifier::BOLD))))
            .gauge_style(Style::default().fg(gauge_color))
            .percent(value.min(100.0) as u16)
            .label(format!("{:.1}%", value));

        f.render_widget(gauge, area);
    }

    /// Render enhanced node table with health indicators and animations
    fn render_enhanced_node_table(&self, f: &mut Frame, area: Rect) {
        let header_cells = [
            "Node", "Status", "Health", "CPU", "Memory", "Network I/O", "Response", "Load"
        ]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
        let header = Row::new(header_cells);

        let rows: Vec<Row> = self.node_health
            .iter()
            .take(20) // Limit to prevent overflow
            .map(|node| {
                let status_color = node.status.get_color();
                let status_icon = node.status.get_icon();

                // Animated status indicator
                let pulse_rate = node.status.get_pulse_rate();
                let pulse_effect = if pulse_rate > 0.0 {
                    let pulse_chars = ["●", "◉", "●", "○"];
                    let pulse_index = ((self.network_pulse * pulse_rate + 1.0) * 2.0) as usize % pulse_chars.len();
                    pulse_chars[pulse_index]
                } else {
                    "●"
                };

                // Health score bar
                let health_bar = self.create_health_bar(node.health_score);

                // Network I/O visualization
                let network_display = format!("↓{:.1} ↑{:.1}", node.network_in, node.network_out);

                // Response time with color coding
                let response_color = if node.response_time < 10.0 {
                    Color::Green
                } else if node.response_time < 25.0 {
                    Color::Yellow
                } else {
                    Color::Red
                };

                Row::new(vec![
                    Cell::from(format!("{} {}", status_icon, node.node_id)),
                    Cell::from(format!("{} {}", status_icon, pulse_effect))
                        .style(Style::default().fg(status_color)),
                    Cell::from(health_bar),
                    Cell::from(format!("{:.1}%", node.cpu_usage))
                        .style(Style::default().fg(if node.cpu_usage > 80.0 { Color::Red } else { Color::Green })),
                    Cell::from(format!("{:.1}%", node.memory_usage))
                        .style(Style::default().fg(if node.memory_usage > 85.0 { Color::Red } else { Color::Green })),
                    Cell::from(network_display)
                        .style(Style::default().fg(Color::Cyan)),
                    Cell::from(format!("{:.1}ms", node.response_time))
                        .style(Style::default().fg(response_color)),
                    Cell::from(format!("{:.2}", node.load_balance_weight))
                        .style(Style::default().fg(Color::Magenta)),
                ])
            })
            .collect();

        let widths = [
            Constraint::Percentage(15),
            Constraint::Percentage(12),
            Constraint::Percentage(12),
            Constraint::Percentage(8),
            Constraint::Percentage(10),
            Constraint::Percentage(15),
            Constraint::Percentage(12),
            Constraint::Percentage(8),
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(Span::styled(
                    " 🖥️ Node Health Matrix ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                )));

        f.render_widget(table, area);
    }

    /// Create a visual health score bar
    fn create_health_bar(&self, health_score: f64) -> String {
        let bar_length = 8;
        let filled_length = ((health_score / 100.0) * bar_length as f64) as usize;
        let empty_length = bar_length - filled_length;

        let filled_char = if health_score > 80.0 {
            "█" // Excellent
        } else if health_score > 60.0 {
            "▓" // Good
        } else if health_score > 40.0 {
            "▒" // Fair
        } else {
            "░" // Poor
        };

        let empty_char = "░";

        format!("{}{}",
            filled_char.repeat(filled_length),
            empty_char.repeat(empty_length)
        )
    }
}

/// Enhanced System performance monitoring widget with immersive visualizations
pub struct SystemMonitorWidget {
    pub title: String,
    pub system: System,
    pub cpu_history: VecDeque<f32>,
    pub memory_history: VecDeque<f32>,
    pub disk_history: VecDeque<f32>,
    pub network_history: VecDeque<f32>,
    pub max_history_size: usize,
    pub animation_state: AnimationState,
    pub system_pulse: f32,
    pub last_update: Instant,
    pub performance_score: f32,
}

impl SystemMonitorWidget {
    /// Create a new enhanced system monitor widget
    pub fn new(title: String) -> Self {
        Self {
            title,
            system: System::new_all(),
            cpu_history: VecDeque::with_capacity(60),
            memory_history: VecDeque::with_capacity(60),
            disk_history: VecDeque::with_capacity(60),
            network_history: VecDeque::with_capacity(60),
            max_history_size: 60,
            animation_state: AnimationState::new(Duration::from_millis(1000)),
            system_pulse: 0.0,
            last_update: Instant::now(),
            performance_score: 85.0,
        }
    }

    /// Update system metrics with enhanced animations
    pub fn update_metrics(&mut self) {
        self.system.refresh_all();
        self.last_update = Instant::now();

        // Update system pulse animation
        self.system_pulse = (self.last_update.elapsed().as_secs_f32() * 1.5).sin();

        // Get enhanced metrics
        let cpu_usage = self.system.global_cpu_usage();
        let memory_usage = (self.system.used_memory() as f64 / self.system.total_memory() as f64 * 100.0) as f32;
        let disk_usage = 65.0 + (rand::random::<f32>() - 0.5) * 20.0;
        let network_usage = 30.0 + (rand::random::<f32>() - 0.5) * 40.0;

        // Calculate performance score
        self.performance_score = 100.0 - (cpu_usage * 0.4 + memory_usage * 0.4 + disk_usage * 0.2);

        // Smooth data transitions
        Self::add_metric_smoothly_f32(&mut self.cpu_history, cpu_usage, self.max_history_size);
        Self::add_metric_smoothly_f32(&mut self.memory_history, memory_usage, self.max_history_size);
        Self::add_metric_smoothly_f32(&mut self.disk_history, disk_usage, self.max_history_size);
        Self::add_metric_smoothly_f32(&mut self.network_history, network_usage, self.max_history_size);

        // Reset animation for new data
        self.animation_state = AnimationState::new(Duration::from_millis(1000));
    }

    /// Add metric with smooth interpolation
    fn add_metric_smoothly_f32(history: &mut VecDeque<f32>, new_value: f32, max_history_size: usize) {
        if history.len() >= max_history_size {
            history.pop_front();
        }

        let interpolated_value = if let Some(&last_value) = history.back() {
            last_value + (new_value - last_value) * 0.3
        } else {
            new_value
        };

        history.push_back(interpolated_value);
    }

    /// Get current CPU usage
    pub fn get_cpu_usage(&self) -> f32 {
        self.cpu_history.back().copied().unwrap_or(0.0)
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> f32 {
        self.memory_history.back().copied().unwrap_or(0.0)
    }

    /// Render the enhanced system monitor widget with animations
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Enhanced header with performance score
                Constraint::Min(10),   // Enhanced metrics visualization
            ])
            .split(area);

        // Enhanced header with system performance indicators
        self.render_system_header(f, chunks[0]);

        // Enhanced metrics grid with animations
        self.render_enhanced_system_metrics(f, chunks[1]);
    }

    /// Render system header with performance indicators
    fn render_system_header(&self, f: &mut Frame, area: Rect) {
        let cpu_usage = self.get_cpu_usage();
        let memory_usage = self.get_memory_usage();
        let cores = self.system.cpus().len();
        let total_memory_gb = self.system.total_memory() / 1_000_000_000;

        // Performance-based coloring
        let performance_color = if self.performance_score > 80.0 {
            Color::Green
        } else if self.performance_score > 60.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        // System pulse animation
        let pulse_intensity = ((self.system_pulse.abs() * 128.0) + 127.0) as u8;
        let pulse_color = Color::Rgb(pulse_intensity, 255 - pulse_intensity, 128);

        let header_spans = vec![
            Span::styled("💻 ", Style::default().fg(pulse_color).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("System Performance: {:.1}%", self.performance_score),
                Style::default().fg(performance_color).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("🔥 ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("CPU {:.1}%", cpu_usage),
                Style::default().fg(if cpu_usage > 80.0 { Color::Red } else { Color::Green }).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("💾 ", Style::default().fg(Color::Blue)),
            Span::styled(
                format!("RAM {:.1}%", memory_usage),
                Style::default().fg(if memory_usage > 85.0 { Color::Red } else { Color::Green }).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("⚙️ ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!("{} cores", cores),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
            Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("📦 ", Style::default().fg(Color::Magenta)),
            Span::styled(
                format!("{}GB", total_memory_gb),
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            ),
        ];

        let header_line = Line::from(header_spans);
        let header_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(pulse_color))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ));

        let header_para = Paragraph::new(vec![Line::from(""), header_line])
            .block(header_block)
            .style(Style::default().fg(Color::White));

        f.render_widget(header_para, area);
    }

    /// Render enhanced system metrics with animations
    fn render_enhanced_system_metrics(&self, f: &mut Frame, area: Rect) {
        let metric_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area);

        self.render_enhanced_system_sparkline(
            f,
            metric_chunks[0],
            "🔥 CPU Usage",
            &self.cpu_history,
            Color::Red,
        );

        self.render_enhanced_system_sparkline(
            f,
            metric_chunks[1],
            "💾 Memory",
            &self.memory_history,
            Color::Blue,
        );

        self.render_enhanced_system_sparkline(
            f,
            metric_chunks[2],
            "💽 Disk I/O",
            &self.disk_history,
            Color::Green,
        );

        self.render_enhanced_system_sparkline(
            f,
            metric_chunks[3],
            "🌐 Network",
            &self.network_history,
            Color::Cyan,
        );
    }

    /// Render enhanced system sparkline with better visualization
    fn render_enhanced_system_sparkline(
        &self,
        f: &mut Frame,
        area: Rect,
        title: &str,
        data: &VecDeque<f32>,
        color: Color,
    ) {
        let current_value = data.back().unwrap_or(&0.0);
        let data_points: Vec<u64> = data.iter().map(|&x| (x * 2.0) as u64).collect();

        // Dynamic border color based on value trends and thresholds
        let border_color = if *current_value > 85.0 {
            Color::Red
        } else if *current_value > 70.0 {
            Color::Yellow
        } else {
            color
        };

        // Performance indicator
        let performance_indicator = if *current_value > 90.0 {
            "🔴"
        } else if *current_value > 70.0 {
            "🟡"
        } else {
            "🟢"
        };

        let title_with_indicator = format!(" {} {} ({:.1}%) ", performance_indicator, title, current_value);
        let sparkline = Sparkline::default()
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border_color))
                .title(Span::styled(title_with_indicator, Style::default().fg(color).add_modifier(Modifier::BOLD))))
            .data(&data_points)
            .style(Style::default().fg(color));

        f.render_widget(sparkline, area);
    }
}
