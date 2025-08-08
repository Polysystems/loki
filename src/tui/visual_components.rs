//! Visual components library for enhanced TUI
//!
//! This module provides reusable, animated UI components with a consistent
//! design language for the Loki TUI.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Block, Borders, Gauge, LineGauge, List, ListItem, Paragraph,
        Sparkline,
        canvas::Canvas,
    },
    Frame,
};
use std::time::Instant;

/// Color palette for consistent theming
pub struct ColorPalette {
    pub primary: Color,
    pub secondary: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    pub background: Color,
    pub surface: Color,
    pub text: Color,
    pub text_dim: Color,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Rgb(255, 165, 0), // Orange
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            info: Color::Blue,
            background: Color::Black,
            surface: Color::Rgb(20, 20, 20),
            text: Color::White,
            text_dim: Color::Gray,
        }
    }
}

/// Animation state for smooth transitions
#[derive(Debug, Clone)]
pub struct AnimationState {
    pub start_time: Instant,
    pub duration_ms: u64,
    pub easing: EasingFunction,
}

#[derive(Debug, Clone)]
pub enum EasingFunction {
    Linear,
    EaseInOut,
    EaseOut,
    Bounce,
}

impl AnimationState {
    pub fn new(duration_ms: u64) -> Self {
        Self {
            start_time: Instant::now(),
            duration_ms,
            easing: EasingFunction::EaseInOut,
        }
    }

    pub fn progress(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_millis() as f32;
        let duration = self.duration_ms as f32;
        (elapsed / duration).min(1.0)
    }

    pub fn value(&self) -> f32 {
        let t = self.progress();
        match self.easing {
            EasingFunction::Linear => t,
            EasingFunction::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            EasingFunction::EaseOut => t * (2.0 - t),
            EasingFunction::Bounce => {
                if t < 1.0 / 2.75 {
                    7.5625 * t * t
                } else if t < 2.0 / 2.75 {
                    let t = t - 1.5 / 2.75;
                    7.5625 * t * t + 0.75
                } else if t < 2.5 / 2.75 {
                    let t = t - 2.25 / 2.75;
                    7.5625 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / 2.75;
                    7.5625 * t * t + 0.984375
                }
            }
        }
    }
}

/// Animated gauge with gradient effect
pub struct AnimatedGauge {
    pub label: String,
    pub value: f32,
    pub max_value: f32,
    pub color_start: Color,
    pub color_end: Color,
    pub animation: Option<AnimationState>,
}

impl AnimatedGauge {
    pub fn new(label: String, value: f32, max_value: f32) -> Self {
        Self {
            label,
            value,
            max_value,
            color_start: Color::Cyan,
            color_end: Color::Blue,
            animation: Some(AnimationState::new(800)),
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let base_ratio = self.value / self.max_value;
        let display_ratio = if let Some(anim) = &self.animation {
            // Animate towards the actual value, not multiply by animation value
            let progress = anim.value();
            base_ratio * progress.max(0.1) // Ensure minimum visibility
        } else {
            base_ratio
        };
        
        // Use the actual value for the label, not the animated value
        let gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(self.color_start)),
            )
            .gauge_style(Style::default().fg(self.color_start).bg(self.color_end))
            .ratio(display_ratio.into())
            .label(format!("{}: {:.1}%", self.label, base_ratio * 100.0));

        f.render_widget(gauge, area);
    }
}

/// Sparkline with glow effect
pub struct GlowingSparkline {
    pub data: Vec<u64>,
    pub color: Color,
    pub title: String,
}

impl GlowingSparkline {
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let max_value = *self.data.iter().max().unwrap_or(&1) as f64;
        let min_value = *self.data.iter().min().unwrap_or(&0) as f64;
        let range = max_value - min_value;
        
        // Create a more sophisticated visualization with gradient background
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // Title
                Constraint::Min(3),     // Main sparkline
                Constraint::Length(1),  // Stats
            ])
            .split(area);
        
        // Render title with stats
        let title_line = Line::from(vec![
            Span::styled(
                format!(" {} ", self.title),
                Style::default().fg(self.color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                format!("[{}..{}]", min_value, max_value),
                Style::default().fg(Color::Gray),
            ),
        ]);
        let title_widget = Paragraph::new(title_line);
        f.render_widget(title_widget, chunks[0]);
        
        // Enhanced sparkline with interpolation
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
                    .border_style(Style::default().fg(self.color)),
            )
            .data(&self.data)
            .max(max_value as u64)
            .style(Style::default().fg(self.color));
        
        f.render_widget(sparkline, chunks[1]);
        
        // Add trend indicator
        if self.data.len() >= 2 {
            let last = self.data[self.data.len() - 1] as f64;
            let prev = self.data[self.data.len() - 2] as f64;
            let trend = if last > prev {
                "↑"
            } else if last < prev {
                "↓"
            } else {
                "→"
            };
            let trend_color = if last > prev {
                Color::Green
            } else if last < prev {
                Color::Red
            } else {
                Color::Yellow
            };
            
            let stats_line = Line::from(vec![
                Span::raw("Current: "),
                Span::styled(format!("{:.1}", last), Style::default().fg(self.color)),
                Span::raw(" Trend: "),
                Span::styled(trend, Style::default().fg(trend_color)),
            ]);
            let stats_widget = Paragraph::new(stats_line)
                .alignment(Alignment::Right);
            f.render_widget(stats_widget, chunks[2]);
        }
    }
}

/// Status indicator with pulse animation
pub struct PulsingStatus {
    pub status: SystemStatus,
    pub label: String,
    pub animation: AnimationState,
}

#[derive(Debug, Clone)]
pub enum SystemStatus {
    Online,
    Connecting,
    Offline,
    Error,
}

impl PulsingStatus {
    pub fn new(label: String, status: SystemStatus) -> Self {
        Self {
            status,
            label,
            animation: AnimationState::new(2000),
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let pulse = (self.animation.progress() * std::f32::consts::PI * 2.0).sin().abs();

        let (icon, base_color) = match self.status {
            SystemStatus::Online => ("●", Color::Green),
            SystemStatus::Connecting => ("◐", Color::Yellow),
            SystemStatus::Offline => ("○", Color::Gray),
            SystemStatus::Error => ("✕", Color::Red),
        };

        let color = match self.status {
            SystemStatus::Online => {
                let r = (0.0 + pulse * 128.0) as u8;
                let g = (255.0 - pulse * 55.0) as u8;
                let b = 0;
                Color::Rgb(r, g, b)
            }
            SystemStatus::Connecting => {
                let brightness = (128.0 + pulse * 127.0) as u8;
                Color::Rgb(brightness, brightness, 0)
            }
            _ => base_color,
        };

        let status_text = format!("{} {} {}", icon, self.label, icon);
        let paragraph = Paragraph::new(status_text)
            .style(Style::default().fg(color))
            .alignment(Alignment::Center);

        f.render_widget(paragraph, area);
    }
}

/// Metric card with animated border
pub struct MetricCard {
    pub title: String,
    pub value: String,
    pub subtitle: String,
    pub trend: TrendDirection,
    pub border_color: Color,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Up,
    Down,
    Stable,
}

impl MetricCard {
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let trend_icon = match self.trend {
            TrendDirection::Up => "↑",
            TrendDirection::Down => "↓",
            TrendDirection::Stable => "→",
        };

        let trend_color = match self.trend {
            TrendDirection::Up => Color::Green,
            TrendDirection::Down => Color::Red,
            TrendDirection::Stable => Color::Yellow,
        };

        let content = vec![
            Line::from(vec![
                Span::styled(&self.title, Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled(&self.value, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled(trend_icon, Style::default().fg(trend_color)),
            ]),
            Line::from(vec![
                Span::styled(&self.subtitle, Style::default().fg(Color::DarkGray)),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.border_color));

        let paragraph = Paragraph::new(content)
            .block(block)
            .alignment(Alignment::Center);

        f.render_widget(paragraph, area);
    }
}

/// Loading spinner animation
pub struct LoadingSpinner {
    pub message: String,
    pub animation: AnimationState,
}

impl LoadingSpinner {
    pub fn new(message: String) -> Self {
        Self {
            message,
            animation: AnimationState::new(1000),
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let frames = vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame_index = ((self.animation.progress() * frames.len() as f32) as usize) % frames.len();

        let spinner_text = format!("{} {}", frames[frame_index], self.message);
        let paragraph = Paragraph::new(spinner_text)
            .style(Style::default().fg(Color::Cyan))
            .alignment(Alignment::Center);

        f.render_widget(paragraph, area);
    }
}

/// Connection graph visualization
pub struct ConnectionGraph {
    pub nodes: Vec<GraphNode>,
    pub connections: Vec<(usize, usize)>,
    pub animation: AnimationState,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub connected: bool,
}

impl ConnectionGraph {
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("System Connections")
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .x_bounds([-1.0, 1.0])
            .y_bounds([-1.0, 1.0])
            .paint(|ctx| {
                let rotation = self.animation.progress() as f64 * std::f64::consts::PI * 2.0;

                // Draw connections
                for (from, to) in &self.connections {
                    if let (Some(node_from), Some(node_to)) = (self.nodes.get(*from), self.nodes.get(*to)) {
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: node_from.x,
                            y1: node_from.y,
                            x2: node_to.x,
                            y2: node_to.y,
                            color: if node_from.connected && node_to.connected {
                                Color::Cyan
                            } else {
                                Color::DarkGray
                            },
                        });
                    }
                }

                // Draw nodes
                for (i, node) in self.nodes.iter().enumerate() {
                    let offset = (rotation + (i as f64 * 0.5)).sin() * 0.05;
                    ctx.print(
                        node.x + offset,
                        node.y + offset,
                        node.name.clone(),
                    );
                }
            });

        f.render_widget(canvas, area);
    }
}

/// Progress bar with segments
pub struct SegmentedProgress {
    pub segments: Vec<ProgressSegment>,
    pub total: f32,
}

#[derive(Debug, Clone)]
pub struct ProgressSegment {
    pub label: String,
    pub value: f32,
    pub color: Color,
}

impl SegmentedProgress {
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let mut current_x = area.x;
        let width = area.width;

        for segment in &self.segments {
            let segment_width = ((segment.value / self.total) * width as f32) as u16;
            if segment_width > 0 && current_x < area.x + area.width {
                let segment_area = Rect {
                    x: current_x,
                    y: area.y,
                    width: segment_width.min(area.x + area.width - current_x),
                    height: area.height,
                };

                let gauge = LineGauge::default()
                    .ratio((segment.value / self.total).into())
                    .label(segment.label.as_str())
                    .style(Style::default().fg(segment.color))
                    .line_set(symbols::line::THICK);

                f.render_widget(gauge, segment_area);
                current_x += segment_width;
            }
        }
    }
}

/// Helper function to create a gradient color
pub fn gradient_color(start: Color, end: Color, progress: f32) -> Color {
    match (start, end) {
        (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
            let r = (r1 as f32 + (r2 as f32 - r1 as f32) * progress) as u8;
            let g = (g1 as f32 + (g2 as f32 - g1 as f32) * progress) as u8;
            let b = (b1 as f32 + (b2 as f32 - b1 as f32) * progress) as u8;
            Color::Rgb(r, g, b)
        }
        _ => start, // Fallback to start color if not RGB
    }
}

/// Create a centered modal/popup area
pub fn centered_modal(width_percent: u16, height_percent: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_percent) / 2),
            Constraint::Percentage(height_percent),
            Constraint::Percentage((100 - height_percent) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(vertical[1])[1]
}

/// Animated list with highlighted items
pub struct AnimatedList {
    pub items: Vec<AnimatedListItem>,
    pub title: String,
    pub border_color: Color,
    pub animation_speed: f32,
}

/// Individual animated list item
pub struct AnimatedListItem {
    pub content: Line<'static>,
    pub highlight_color: Color,
    pub animation_offset: f32,
}

impl AnimatedList {
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let list_items: Vec<ListItem> = self.items
            .iter()
            .map(|item| ListItem::new(item.content.clone()))
            .collect();
        
        let list = List::new(list_items)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(self.title.clone())
                .border_style(Style::default().fg(self.border_color)))
            .style(Style::default().fg(Color::White));
        
        f.render_widget(list, area);
    }
}

/// Pulsing status indicator with color animation
pub struct PulsingStatusIndicator {
    pub text: String,
    pub color: Color,
    pub pulse_speed: f32,
}

impl PulsingStatusIndicator {
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let time = Instant::now().elapsed().as_secs_f32();
        let pulse = ((time * self.pulse_speed).sin() + 1.0) / 2.0;

        let pulsing_color = match self.color {
            Color::Rgb(r, g, b) => Color::Rgb(
                (r as f32 * pulse) as u8,
                (g as f32 * pulse) as u8,
                (b as f32 * pulse) as u8,
            ),
            _ => self.color,
        };
        
        let status = Paragraph::new(self.text.clone())
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(pulsing_color)))
            .style(Style::default().fg(self.color))
            .alignment(Alignment::Center);
        
        f.render_widget(status, area);
    }
}
