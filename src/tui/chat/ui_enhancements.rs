//! Advanced UI/UX enhancements for the chat system
//! 
//! This module provides additional visual and interactive features
//! to enhance the user experience.

use std::time::{Duration, Instant};
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Gauge, Paragraph, Sparkline, Borders},
    Frame,
};

// Re-export keyboard shortcuts module
pub mod keyboard_shortcuts;
pub use keyboard_shortcuts::{KeyboardShortcutsOverlay, QuickReferenceCard, ShortcutCategory};

/// Animation state for smooth transitions
#[derive(Debug, Clone)]
pub struct AnimationState {
    start_time: Instant,
    duration: Duration,
    from_value: f32,
    to_value: f32,
    easing: EasingFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum EasingFunction {
    Linear,
    EaseInOut,
    EaseOut,
    Bounce,
}

impl AnimationState {
    pub fn new(from: f32, to: f32, duration: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            duration,
            from_value: from,
            to_value: to,
            easing: EasingFunction::EaseInOut,
        }
    }
    
    pub fn current_value(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let progress = (elapsed / self.duration.as_secs_f32()).min(1.0);
        
        let eased_progress = match self.easing {
            EasingFunction::Linear => progress,
            EasingFunction::EaseInOut => {
                if progress < 0.5 {
                    2.0 * progress * progress
                } else {
                    -1.0 + (4.0 - 2.0 * progress) * progress
                }
            }
            EasingFunction::EaseOut => 1.0 - (1.0 - progress).powi(2),
            EasingFunction::Bounce => {
                if progress < 0.5 {
                    8.0 * progress.powi(2)
                } else {
                    1.0 - 8.0 * (progress - 1.0).powi(2)
                }
            }
        };
        
        self.from_value + (self.to_value - self.from_value) * eased_progress
    }
    
    pub fn is_complete(&self) -> bool {
        self.start_time.elapsed() >= self.duration
    }
}

/// Typing indicator with animated dots
pub struct TypingIndicator {
    dots: usize,
    last_update: Instant,
    visible: bool,
}

impl TypingIndicator {
    pub fn new() -> Self {
        Self {
            dots: 1,
            last_update: Instant::now(),
            visible: false,
        }
    }
    
    pub fn show(&mut self) {
        self.visible = true;
    }
    
    pub fn hide(&mut self) {
        self.visible = false;
    }
    
    pub fn render(&mut self) -> String {
        if !self.visible {
            return String::new();
        }
        
        // Update dots every 500ms
        if self.last_update.elapsed() > Duration::from_millis(500) {
            self.dots = (self.dots % 3) + 1;
            self.last_update = Instant::now();
        }
        
        format!("AI is thinking{}", ".".repeat(self.dots))
    }
}

/// Progress indicator with smooth animation
pub struct SmoothProgress {
    current: f32,
    target: f32,
    animation: Option<AnimationState>,
}

impl SmoothProgress {
    pub fn new() -> Self {
        Self {
            current: 0.0,
            target: 0.0,
            animation: None,
        }
    }
    
    pub fn set_progress(&mut self, value: f32) {
        if (value - self.target).abs() > 0.01 {
            self.target = value;
            self.animation = Some(AnimationState::new(
                self.current,
                value,
                Duration::from_millis(300),
            ));
        }
    }
    
    pub fn render(&mut self, f: &mut Frame, area: Rect, label: &str) {
        // Update current value from animation
        if let Some(anim) = &self.animation {
            self.current = anim.current_value();
            if anim.is_complete() {
                self.animation = None;
            }
        }
        
        let gauge = Gauge::default()
            .block(Block::default().title(label))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent((self.current * 100.0) as u16)
            .label(format!("{:.1}%", self.current * 100.0));
        
        f.render_widget(gauge, area);
    }
}

/// Sparkline chart for real-time metrics
pub struct MetricsSparkline {
    data: Vec<u64>,
    max_points: usize,
    title: String,
}

impl MetricsSparkline {
    pub fn new(title: String, max_points: usize) -> Self {
        Self {
            data: Vec::with_capacity(max_points),
            max_points,
            title,
        }
    }
    
    pub fn add_point(&mut self, value: u64) {
        self.data.push(value);
        if self.data.len() > self.max_points {
            self.data.remove(0);
        }
    }
    
    pub fn render(&self, f: &mut Frame, area: Rect) {
        if self.data.is_empty() {
            return;
        }
        
        let sparkline = Sparkline::default()
            .block(Block::default().title(self.title.as_str()))
            .data(&self.data)
            .style(Style::default().fg(Color::Yellow));
        
        f.render_widget(sparkline, area);
    }
}

/// Status indicator with color and icon
#[derive(Debug, Clone)]
pub struct StatusIndicator {
    status: Status,
    message: String,
    icon_animation: Option<AnimationState>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Status {
    Idle,
    Processing,
    Success,
    Warning,
    Error,
}

impl StatusIndicator {
    pub fn new() -> Self {
        Self {
            status: Status::Idle,
            message: String::new(),
            icon_animation: None,
        }
    }
    
    pub fn set_status(&mut self, status: Status, message: String) {
        self.status = status;
        self.message = message;
        
        // Start icon animation for processing status
        if status == Status::Processing {
            self.icon_animation = Some(AnimationState::new(
                0.0,
                1.0,
                Duration::from_secs(2),
            ));
        } else {
            self.icon_animation = None;
        }
    }
    
    pub fn render(&mut self) -> Line {
        let (icon, color) = match self.status {
            Status::Idle => ("⭘", Color::Gray),
            Status::Processing => {
                // Animated spinner
                let frames = vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let index = if let Some(anim) = &self.icon_animation {
                    ((anim.current_value() * frames.len() as f32) as usize) % frames.len()
                } else {
                    0
                };
                (frames[index], Color::Cyan)
            }
            Status::Success => ("✓", Color::Green),
            Status::Warning => ("⚠", Color::Yellow),
            Status::Error => ("✗", Color::Red),
        };
        
        Line::from(vec![
            Span::styled(icon, Style::default().fg(color)),
            Span::raw(" "),
            Span::styled(&self.message, Style::default().fg(color)),
        ])
    }
}

/// Smooth scrolling state
pub struct SmoothScroll {
    current_offset: f32,
    target_offset: f32,
    animation: Option<AnimationState>,
}

impl SmoothScroll {
    pub fn new() -> Self {
        Self {
            current_offset: 0.0,
            target_offset: 0.0,
            animation: None,
        }
    }
    
    pub fn scroll_to(&mut self, offset: usize) {
        let target = offset as f32;
        if (target - self.target_offset).abs() > 0.1 {
            self.target_offset = target;
            self.animation = Some(AnimationState {
                start_time: Instant::now(),
                duration: Duration::from_millis(200),
                from_value: self.current_offset,
                to_value: target,
                easing: EasingFunction::EaseOut,
            });
        }
    }
    
    pub fn current_offset(&mut self) -> usize {
        if let Some(anim) = &self.animation {
            self.current_offset = anim.current_value();
            if anim.is_complete() {
                self.animation = None;
            }
        }
        self.current_offset as usize
    }
    
    /// Update position to a new target (same as scroll_to but for compatibility)
    pub fn update_position(&mut self, position: usize) {
        self.scroll_to(position);
    }
}

/// Toast notification system
#[derive(Debug, Clone)]
pub struct Toast {
    message: String,
    toast_type: ToastType,
    created_at: Instant,
    duration: Duration,
}

#[derive(Debug, Clone, Copy)]
pub enum ToastType {
    Info,
    Success,
    Warning,
    Error,
}

pub struct ToastManager {
    toasts: Vec<Toast>,
    max_toasts: usize,
}

impl ToastManager {
    pub fn new() -> Self {
        Self {
            toasts: Vec::new(),
            max_toasts: 5,
        }
    }
    
    pub fn add_toast(&mut self, message: String, toast_type: ToastType) {
        let toast = Toast {
            message,
            toast_type,
            created_at: Instant::now(),
            duration: Duration::from_secs(3),
        };
        
        self.toasts.push(toast);
        
        // Remove old toasts
        if self.toasts.len() > self.max_toasts {
            self.toasts.remove(0);
        }
    }
    
    pub fn update(&mut self) {
        // Remove expired toasts
        self.toasts.retain(|toast| {
            toast.created_at.elapsed() < toast.duration
        });
    }
    
    pub fn render(&self, f: &mut Frame, area: Rect) {
        for (i, toast) in self.toasts.iter().enumerate() {
            let y = area.y + i as u16;
            if y >= area.bottom() {
                break;
            }
            
            let (icon, color) = match toast.toast_type {
                ToastType::Info => ("ℹ", Color::Blue),
                ToastType::Success => ("✓", Color::Green),
                ToastType::Warning => ("⚠", Color::Yellow),
                ToastType::Error => ("✗", Color::Red),
            };
            
            let opacity = {
                let elapsed = toast.created_at.elapsed();
                let fade_start = toast.duration - Duration::from_millis(500);
                if elapsed > fade_start {
                    let fade_progress = (elapsed - fade_start).as_secs_f32() / 0.5;
                    1.0 - fade_progress
                } else {
                    1.0
                }
            };
            
            // Render with opacity (simulated with color intensity)
            let text = format!("{} {}", icon, toast.message);
            let style = if opacity < 1.0 {
                Style::default().fg(Color::Gray)
            } else {
                Style::default().fg(color)
            };
            
            let paragraph = Paragraph::new(text)
                .style(style)
                .alignment(Alignment::Right);
            
            let toast_area = Rect {
                x: area.x,
                y,
                width: area.width,
                height: 1,
            };
            
            f.render_widget(paragraph, toast_area);
        }
    }
}

/// Theme system for consistent styling
#[derive(Debug, Clone)]
pub struct Theme {
    pub primary: Color,
    pub secondary: Color,
    pub background: Color,
    pub text: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub border: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Magenta,
            background: Color::Black,
            text: Color::White,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::Gray,
        }
    }
}

impl Theme {
    pub fn dark() -> Self {
        Self::default()
    }
    
    pub fn light() -> Self {
        Self {
            primary: Color::Blue,
            secondary: Color::Magenta,
            background: Color::White,
            text: Color::Black,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::Gray,
        }
    }
    
    pub fn high_contrast() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Magenta,
            background: Color::Black,
            text: Color::White,
            success: Color::LightGreen,
            warning: Color::LightYellow,
            error: Color::LightRed,
            border: Color::White,
        }
    }
}

/// Streaming indicator states
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingState {
    Idle,
    Connecting,
    Streaming { 
        start_time: Instant,
        tokens_count: usize,
        current_chunk: String,
    },
    Completing,
    Error(String),
}

/// Real-time streaming indicator widget
pub struct StreamingIndicator {
    pub state: StreamingState,
    animation_frame: usize,
    last_update: Instant,
    pulse_intensity: f32,
    dots_count: usize,
}

impl StreamingIndicator {
    /// Create a new streaming indicator
    pub fn new() -> Self {
        Self {
            state: StreamingState::Idle,
            animation_frame: 0,
            last_update: Instant::now(),
            pulse_intensity: 0.0,
            dots_count: 0,
        }
    }
    
    /// Start streaming
    pub fn start_streaming(&mut self) {
        self.state = StreamingState::Streaming {
            start_time: Instant::now(),
            tokens_count: 0,
            current_chunk: String::new(),
        };
    }
    
    /// Update with new streaming chunk
    pub fn add_chunk(&mut self, chunk: &str) {
        if let StreamingState::Streaming { tokens_count, current_chunk, .. } = &mut self.state {
            *tokens_count += chunk.split_whitespace().count();
            *current_chunk = chunk.to_string();
        }
    }
    
    /// Set to completing state
    pub fn complete(&mut self) {
        self.state = StreamingState::Completing;
    }
    
    /// Set error state
    pub fn error(&mut self, message: String) {
        self.state = StreamingState::Error(message);
    }
    
    /// Reset to idle
    pub fn reset(&mut self) {
        self.state = StreamingState::Idle;
    }
    
    /// Update animation
    pub fn update(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_update) > Duration::from_millis(100) {
            self.animation_frame = (self.animation_frame + 1) % 60;
            self.last_update = now;
            
            // Update dots animation
            self.dots_count = (self.dots_count + 1) % 4;
            
            // Update pulse effect
            self.pulse_intensity = ((self.animation_frame as f32 * 0.1).sin() + 1.0) / 2.0;
        }
    }
    
    /// Render the streaming indicator
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let (icon, text, color) = match &self.state {
            StreamingState::Idle => {
                return; // Don't render when idle
            }
            StreamingState::Connecting => {
                let dots = ".".repeat(self.dots_count);
                ("🔌", format!("Connecting{}", dots), Color::Yellow)
            }
            StreamingState::Streaming { start_time, tokens_count, current_chunk } => {
                let elapsed = Instant::now().duration_since(*start_time).as_secs();
                let tokens_per_sec = if elapsed > 0 {
                    *tokens_count as f32 / elapsed as f32
                } else {
                    0.0
                };
                
                let animation_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
                let spinner = animation_chars[self.animation_frame % animation_chars.len()];
                
                (
                    "✨",
                    format!(
                        "{} Streaming... {} tokens ({:.1} tok/s) | {}",
                        spinner,
                        tokens_count,
                        tokens_per_sec,
                        truncate_chunk(current_chunk, 30)
                    ),
                    Color::Green
                )
            }
            StreamingState::Completing => {
                ("⏳", "Finalizing response...".to_string(), Color::Cyan)
            }
            StreamingState::Error(msg) => {
                ("❌", format!("Error: {}", msg), Color::Red)
            }
        };
        
        // Create pulsing effect for the border
        let border_color = if self.pulse_intensity > 0.5 {
            color
        } else {
            Color::DarkGray
        };
        
        let content = vec![
            Line::from(vec![
                Span::raw(icon),
                Span::raw(" "),
                Span::styled(text, Style::default().fg(color)),
            ])
        ];
        
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(" Streaming Status ");
            
        let paragraph = Paragraph::new(content)
            .block(block);
            
        f.render_widget(paragraph, area);
    }
    
    /// Render inline streaming indicator (for message list)
    pub fn render_inline(&self, f: &mut Frame, area: Rect) {
        if let StreamingState::Streaming { tokens_count, .. } = &self.state {
            let dots = ".".repeat(self.dots_count);
            let typing_indicator = vec![
                Span::styled("AI is typing", Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC)),
                Span::styled(dots, Style::default().fg(Color::Gray)),
                Span::raw(" "),
                Span::styled(
                    format!("({} tokens)", tokens_count),
                    Style::default().fg(Color::DarkGray)
                ),
            ];
            
            let paragraph = Paragraph::new(Line::from(typing_indicator));
            f.render_widget(paragraph, area);
        }
    }
}

/// Truncate chunk for display
fn truncate_chunk(chunk: &str, max_len: usize) -> String {
    let trimmed = chunk.trim();
    if trimmed.len() <= max_len {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..max_len-3])
    }
}

/// Streaming progress bar widget
pub struct StreamingProgressBar {
    progress: f32,
    estimated_total: usize,
    current: usize,
}

impl StreamingProgressBar {
    /// Create a new progress bar
    pub fn new() -> Self {
        Self {
            progress: 0.0,
            estimated_total: 100,
            current: 0,
        }
    }
    
    /// Update progress
    pub fn update(&mut self, current: usize, estimated_total: usize) {
        self.current = current;
        self.estimated_total = estimated_total;
        self.progress = (current as f32 / estimated_total.max(1) as f32).min(1.0);
    }
    
    /// Render the progress bar
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::NONE))
            .gauge_style(Style::default().fg(Color::Green))
            .percent((self.progress * 100.0) as u16)
            .label(format!("{}/{}", self.current, self.estimated_total));
            
        f.render_widget(gauge, area);
    }
}

/// Animated typing effect for streaming messages
pub struct TypewriterEffect {
    full_text: String,
    displayed_chars: usize,
    last_update: Instant,
    chars_per_second: usize,
}

impl TypewriterEffect {
    /// Create a new typewriter effect
    pub fn new(text: String, chars_per_second: usize) -> Self {
        Self {
            full_text: text,
            displayed_chars: 0,
            last_update: Instant::now(),
            chars_per_second,
        }
    }
    
    /// Update the animation
    pub fn update(&mut self) {
        let elapsed = self.last_update.elapsed().as_millis() as f32 / 1000.0;
        let chars_to_add = (elapsed * self.chars_per_second as f32) as usize;
        
        if chars_to_add > 0 {
            self.displayed_chars = (self.displayed_chars + chars_to_add).min(self.full_text.len());
            self.last_update = Instant::now();
        }
    }
    
    /// Get the current displayed text
    pub fn get_displayed_text(&self) -> &str {
        &self.full_text[..self.displayed_chars]
    }
    
    /// Check if animation is complete
    pub fn is_complete(&self) -> bool {
        self.displayed_chars >= self.full_text.len()
    }
}