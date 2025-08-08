//! Stream rendering for real-time chat updates
//! 
//! Provides smooth animations and progressive rendering for streaming
//! responses, typing indicators, and live updates.

use ratatui::{
    layout::Rect,
    style::{Color,  Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Stream renderer for real-time updates
#[derive(Clone)]
pub struct StreamRenderer {
    /// Active streams
    streams: Vec<ActiveStream>,
    
    /// Animation states
    animations: AnimationManager,
    
    /// Buffer for partial content
    partial_buffer: String,
    
    /// Typing indicators
    typing_indicators: Vec<TypingIndicator>,
}

/// Active stream state
#[derive(Debug, Clone)]
pub struct ActiveStream {
    pub id: String,
    pub state: StreamState,
    pub content: String,
    pub partial_content: String,
    pub started_at: Instant,
    pub last_update: Instant,
    pub metadata: StreamMetadata,
}

/// Stream state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamState {
    Connecting,
    Streaming,
    Buffering,
    Complete,
    Error,
}

/// Stream metadata
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub source: String,
    pub tokens_received: usize,
    pub chunks_received: usize,
    pub estimated_total: Option<usize>,
    pub speed_tokens_per_second: f32,
}

/// Typing indicator
#[derive(Debug, Clone)]
struct TypingIndicator {
    user: String,
    started_at: Instant,
    style: TypingStyle,
}

#[derive(Debug, Clone, Copy)]
enum TypingStyle {
    Dots,
    Pulse,
    Wave,
}

/// Animation manager
#[derive(Clone)]
struct AnimationManager {
    /// Text reveal animations
    text_reveals: Vec<TextRevealAnimation>,
    
    /// Cursor animations
    cursors: Vec<CursorAnimation>,
    
    /// Progress animations
    progress_bars: Vec<ProgressAnimation>,
    
    /// Frame counter
    frame_count: u64,
    
    /// Last frame time
    last_frame: Instant,
}

/// Text reveal animation
#[derive(Clone)]
struct TextRevealAnimation {
    id: String,
    text: String,
    revealed_chars: usize,
    chars_per_frame: f32,
    effect: RevealEffect,
}

#[derive(Debug, Clone, Copy)]
enum RevealEffect {
    Typewriter,
    FadeIn,
    SlideIn,
    Matrix,
}

/// Cursor animation
#[derive(Clone)]
struct CursorAnimation {
    id: String,
    position: usize,
    blink_state: bool,
    style: CursorStyle,
}

#[derive(Debug, Clone, Copy)]
enum CursorStyle {
    Block,
    Line,
    Underline,
    Custom(char),
}

/// Progress animation
#[derive(Clone)]
struct ProgressAnimation {
    id: String,
    progress: f32,
    target: f32,
    speed: f32,
    style: ProgressStyle,
}

#[derive(Debug, Clone, Copy)]
enum ProgressStyle {
    Smooth,
    Steps,
    Bounce,
}

impl StreamRenderer {
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
            animations: AnimationManager::new(),
            partial_buffer: String::new(),
            typing_indicators: Vec::new(),
        }
    }
    
    /// Start a new stream
    pub fn start_stream(&mut self, id: String, source: String) -> String {
        let stream = ActiveStream {
            id: id.clone(),
            state: StreamState::Connecting,
            content: String::new(),
            partial_content: String::new(),
            started_at: Instant::now(),
            last_update: Instant::now(),
            metadata: StreamMetadata {
                source,
                tokens_received: 0,
                chunks_received: 0,
                estimated_total: None,
                speed_tokens_per_second: 0.0,
            },
        };
        
        self.streams.push(stream);
        
        // Start animations
        self.animations.start_text_reveal(&id, RevealEffect::Typewriter);
        self.animations.start_cursor(&id, CursorStyle::Line);
        
        id
    }
    
    /// Update stream with new content
    pub fn update_stream(&mut self, id: &str, chunk: &str) {
        if let Some(stream) = self.streams.iter_mut().find(|s| s.id == id) {
            stream.partial_content.push_str(chunk);
            stream.state = StreamState::Streaming;
            stream.metadata.chunks_received += 1;
            
            // Process complete tokens/words
            if let Some(last_space) = stream.partial_content.rfind(' ') {
                let complete_part = stream.partial_content[..=last_space].to_string();
                stream.content.push_str(&complete_part);
                stream.partial_content = stream.partial_content[last_space + 1..].to_string();
                
                // Update metadata
                stream.metadata.tokens_received += complete_part.split_whitespace().count();
                let elapsed = stream.started_at.elapsed().as_secs_f32();
                if elapsed > 0.0 {
                    stream.metadata.speed_tokens_per_second = 
                        stream.metadata.tokens_received as f32 / elapsed;
                }
            }
            
            stream.last_update = Instant::now();
            
            // Update animations
            self.animations.update_text_reveal(id, &stream.content);
        }
    }
    
    /// Complete a stream
    pub fn complete_stream(&mut self, id: &str) {
        if let Some(stream) = self.streams.iter_mut().find(|s| s.id == id) {
            // Flush any remaining partial content
            if !stream.partial_content.is_empty() {
                stream.content.push_str(&stream.partial_content);
                stream.partial_content.clear();
            }
            
            stream.state = StreamState::Complete;
            
            // Stop animations
            self.animations.stop_cursor(id);
            self.animations.complete_text_reveal(id);
        }
    }
    
    /// Add typing indicator
    pub fn add_typing_indicator(&mut self, user: String) {
        self.typing_indicators.push(TypingIndicator {
            user,
            started_at: Instant::now(),
            style: TypingStyle::Dots,
        });
    }
    
    /// Remove typing indicator
    pub fn remove_typing_indicator(&mut self, user: &str) {
        self.typing_indicators.retain(|t| t.user != user);
    }
    
    /// Update animations
    pub fn update(&mut self) {
        self.animations.update();
        
        // Remove completed streams after a delay
        let now = Instant::now();
        self.streams.retain(|stream| {
            stream.state != StreamState::Complete || 
            now.duration_since(stream.last_update) < Duration::from_secs(2)
        });
    }
    
    /// Check if renderer has any active streams
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
    
    /// Check if there are active streams
    pub fn has_active_streams(&self) -> bool {
        !self.streams.is_empty()
    }
    
    /// Render active streams
    pub fn render(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme) {
        let chunks = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints(
                self.streams
                    .iter()
                    .map(|_| ratatui::layout::Constraint::Min(3))
                    .collect::<Vec<_>>()
            )
            .split(area);
        
        for (i, stream) in self.streams.iter().enumerate() {
            if i < chunks.len() {
                self.render_stream(f, chunks[i], stream, theme);
            }
        }
        
        // Render typing indicators
        if !self.typing_indicators.is_empty() {
            self.render_typing_indicators(f, area, theme);
        }
    }
    
    /// Render individual stream
    fn render_stream(&self, f: &mut Frame, area: Rect, stream: &ActiveStream, theme: &super::theme_engine::ChatTheme) {
        let status_icon = match stream.state {
            StreamState::Connecting => "⟳",
            StreamState::Streaming => "●",
            StreamState::Buffering => "◐",
            StreamState::Complete => "✓",
            StreamState::Error => "✗",
        };
        
        let status_color = match stream.state {
            StreamState::Connecting => Color::Yellow,
            StreamState::Streaming => Color::Green,
            StreamState::Buffering => Color::Yellow,
            StreamState::Complete => Color::Green,
            StreamState::Error => Color::Red,
        };
        
        let title = format!(
            " {} {} ({:.1} tokens/s) ",
            status_icon,
            stream.metadata.source,
            stream.metadata.speed_tokens_per_second
        );
        
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(status_color));
        
        // Get animated content
        let display_content = if let Some(reveal) = self.animations.get_text_reveal(&stream.id) {
            reveal.get_revealed_text()
        } else {
            stream.content.clone()
        };
        
        // Add cursor if streaming
        let mut content = display_content;
        if stream.state == StreamState::Streaming {
            if let Some(cursor) = self.animations.get_cursor(&stream.id) {
                content.push_str(&cursor.get_cursor_char());
            }
        }
        
        // Add partial content with dimmed style
        if !stream.partial_content.is_empty() {
            content.push_str(&stream.partial_content);
        }
        
        let paragraph = Paragraph::new(content)
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .style(theme.text_styles.normal);
        
        f.render_widget(paragraph, area);
        
        // Render progress bar if available
        if let Some(progress) = self.calculate_progress(stream) {
            self.render_progress_bar(f, area, progress, theme);
        }
    }
    
    /// Calculate stream progress
    fn calculate_progress(&self, stream: &ActiveStream) -> Option<f32> {
        stream.metadata.estimated_total.map(|total| {
            (stream.metadata.tokens_received as f32 / total as f32).min(1.0)
        })
    }
    
    /// Render progress bar
    fn render_progress_bar(&self, f: &mut Frame, area: Rect, progress: f32, theme: &super::theme_engine::ChatTheme) {
        if area.height < 3 {
            return;
        }
        
        let progress_area = Rect {
            x: area.x + 1,
            y: area.y + area.height - 2,
            width: area.width.saturating_sub(2),
            height: 1,
        };
        
        let filled = (progress * progress_area.width as f32) as u16;
        let empty = progress_area.width.saturating_sub(filled);
        
        let progress_text = format!(
            "{}{}",
            "█".repeat(filled as usize),
            "░".repeat(empty as usize)
        );
        
        let progress_line = Line::from(vec![
            Span::styled(
                progress_text,
                theme.component_styles.progress.bar_filled
            ),
        ]);
        
        let progress_widget = Paragraph::new(progress_line);
        f.render_widget(progress_widget, progress_area);
    }
    
    /// Render typing indicators
    fn render_typing_indicators(&self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme) {
        let indicators: Vec<String> = self.typing_indicators
            .iter()
            .map(|indicator| {
                let dots = self.get_typing_animation(indicator);
                format!("{} is typing{}", indicator.user, dots)
            })
            .collect();
        
        if indicators.is_empty() {
            return;
        }
        
        let text = indicators.join(", ");
        let typing_area = Rect {
            x: area.x,
            y: area.y + area.height.saturating_sub(1),
            width: area.width,
            height: 1,
        };
        
        let typing_widget = Paragraph::new(text)
            .style(theme.text_styles.dim);
        
        f.render_widget(typing_widget, typing_area);
    }
    
    /// Get typing animation dots
    fn get_typing_animation(&self, indicator: &TypingIndicator) -> String {
        let elapsed = indicator.started_at.elapsed().as_millis();
        let dots_count = ((elapsed / 400) % 4) as usize;
        ".".repeat(dots_count)
    }
}

impl AnimationManager {
    fn new() -> Self {
        Self {
            text_reveals: Vec::new(),
            cursors: Vec::new(),
            progress_bars: Vec::new(),
            frame_count: 0,
            last_frame: Instant::now(),
        }
    }
    
    fn update(&mut self) {
        self.frame_count += 1;
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        
        // Update text reveals
        for reveal in &mut self.text_reveals {
            reveal.update(delta);
        }
        
        // Update cursors
        for cursor in &mut self.cursors {
            cursor.update(self.frame_count);
        }
        
        // Update progress bars
        for progress in &mut self.progress_bars {
            progress.update(delta);
        }
        
        // Clean up completed animations
        self.text_reveals.retain(|r| r.revealed_chars < r.text.len());
    }
    
    fn start_text_reveal(&mut self, id: &str, effect: RevealEffect) {
        self.text_reveals.push(TextRevealAnimation {
            id: id.to_string(),
            text: String::new(),
            revealed_chars: 0,
            chars_per_frame: match effect {
                RevealEffect::Typewriter => 2.0,
                RevealEffect::FadeIn => 5.0,
                RevealEffect::SlideIn => 10.0,
                RevealEffect::Matrix => 3.0,
            },
            effect,
        });
    }
    
    fn update_text_reveal(&mut self, id: &str, text: &str) {
        if let Some(reveal) = self.text_reveals.iter_mut().find(|r| r.id == id) {
            reveal.text = text.to_string();
        }
    }
    
    fn complete_text_reveal(&mut self, id: &str) {
        if let Some(reveal) = self.text_reveals.iter_mut().find(|r| r.id == id) {
            reveal.revealed_chars = reveal.text.len();
        }
    }
    
    fn get_text_reveal(&self, id: &str) -> Option<&TextRevealAnimation> {
        self.text_reveals.iter().find(|r| r.id == id)
    }
    
    fn start_cursor(&mut self, id: &str, style: CursorStyle) {
        self.cursors.push(CursorAnimation {
            id: id.to_string(),
            position: 0,
            blink_state: true,
            style,
        });
    }
    
    fn stop_cursor(&mut self, id: &str) {
        self.cursors.retain(|c| c.id != id);
    }
    
    fn get_cursor(&self, id: &str) -> Option<&CursorAnimation> {
        self.cursors.iter().find(|c| c.id == id)
    }
}

impl TextRevealAnimation {
    fn update(&mut self, delta: f32) {
        let chars_to_reveal = (self.chars_per_frame * delta * 60.0) as usize;
        self.revealed_chars = (self.revealed_chars + chars_to_reveal).min(self.text.len());
    }
    
    fn get_revealed_text(&self) -> String {
        match self.effect {
            RevealEffect::Typewriter => {
                self.text.chars().take(self.revealed_chars).collect()
            }
            RevealEffect::FadeIn => {
                // Simplified - would use actual fading in a real implementation
                self.text.chars().take(self.revealed_chars).collect()
            }
            RevealEffect::SlideIn => {
                // Simplified - would use actual sliding in a real implementation
                self.text.chars().take(self.revealed_chars).collect()
            }
            RevealEffect::Matrix => {
                // Matrix-style reveal with random characters
                let mut result = String::new();
                for (i, ch) in self.text.chars().enumerate() {
                    if i < self.revealed_chars {
                        result.push(ch);
                    } else if i < self.revealed_chars + 5 {
                        // Show random characters for upcoming text
                        result.push(['0', '1', '█', '▓', '▒', '░'][i % 6]);
                    } else {
                        break;
                    }
                }
                result
            }
        }
    }
}

impl CursorAnimation {
    fn update(&mut self, frame: u64) {
        // Blink every 30 frames
        self.blink_state = (frame / 30) % 2 == 0;
    }
    
    fn get_cursor_char(&self) -> String {
        if self.blink_state {
            match self.style {
                CursorStyle::Block => "█",
                CursorStyle::Line => "│",
                CursorStyle::Underline => "_",
                CursorStyle::Custom(ch) => return ch.to_string(),
            }
        } else {
            " "
        }.to_string()
    }
}

impl ProgressAnimation {
    fn update(&mut self, delta: f32) {
        match self.style {
            ProgressStyle::Smooth => {
                let diff = self.target - self.progress;
                self.progress += diff * self.speed * delta;
            }
            ProgressStyle::Steps => {
                if self.progress < self.target {
                    self.progress = (self.progress + self.speed * delta).min(self.target);
                }
            }
            ProgressStyle::Bounce => {
                // Simplified bounce animation
                self.progress = self.target;
            }
        }
    }
}

/// Create a stream update channel
pub fn create_stream_channel() -> (mpsc::Sender<StreamUpdate>, mpsc::Receiver<StreamUpdate>) {
    mpsc::channel(100)
}

/// Stream update message
#[derive(Debug, Clone)]
pub enum StreamUpdate {
    Start { id: String, source: String },
    Chunk { id: String, content: String },
    Complete { id: String },
    Error { id: String, error: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stream_lifecycle() {
        let mut renderer = StreamRenderer::new();
        
        let id = renderer.start_stream("test".to_string(), "AI".to_string());
        assert_eq!(renderer.streams.len(), 1);
        
        renderer.update_stream(&id, "Hello ");
        renderer.update_stream(&id, "world!");
        
        let stream = &renderer.streams[0];
        assert_eq!(stream.content, "Hello ");
        assert_eq!(stream.partial_content, "world!");
        
        renderer.complete_stream(&id);
        assert_eq!(renderer.streams[0].state, StreamState::Complete);
        assert_eq!(renderer.streams[0].content, "Hello world!");
    }
    
    #[test]
    fn test_typing_indicators() {
        let mut renderer = StreamRenderer::new();
        
        renderer.add_typing_indicator("Alice".to_string());
        assert_eq!(renderer.typing_indicators.len(), 1);
        
        renderer.remove_typing_indicator("Alice");
        assert_eq!(renderer.typing_indicators.len(), 0);
    }
}