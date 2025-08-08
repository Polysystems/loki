//! Real-time Cognitive Activity Indicators for TUI
//!
//! This module provides visual indicators for cognitive processing,
//! consciousness activity, and active modalities in the chat interface.

use std::sync::Arc;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline},
    Frame,
};

use crate::tui::{
    chat::integrations::cognitive::CognitiveChatEnhancement,
    cognitive_stream_integration::{CognitiveActivity},
    cognitive::integration::main::CognitiveModality,
};

/// Cognitive activity display component
#[derive(Clone)]
pub struct CognitiveActivityIndicator {
    /// Reference to cognitive enhancement
    cognitive_enhancement: Option<Arc<CognitiveChatEnhancement>>,
    
    /// Historical awareness levels for sparkline
    awareness_history: Vec<u64>,
    
    /// Historical coherence levels
    coherence_history: Vec<u64>,
    
    /// Active modalities
    active_modalities: Vec<CognitiveModality>,
    
    /// Current processing depth
    processing_depth: f64,
    
    /// Animation frame counter
    animation_frame: usize,
}

impl CognitiveActivityIndicator {
    /// Create new cognitive indicator
    pub fn new() -> Self {
        Self {
            cognitive_enhancement: None,
            awareness_history: vec![0; 20],
            coherence_history: vec![0; 20],
            active_modalities: vec![],
            processing_depth: 0.0,
            animation_frame: 0,
        }
    }
    
    /// Set cognitive enhancement reference
    pub fn set_enhancement(&mut self, enhancement: Arc<CognitiveChatEnhancement>) {
        self.cognitive_enhancement = Some(enhancement);
    }
    
    /// Update indicators from cognitive state
    pub async fn update(&mut self) {
        if let Some(enhancement) = &self.cognitive_enhancement {
            // Get consciousness activity
            let activity = enhancement.get_cognitive_activity();
            {
                // Update history (convert to 0-100 scale for sparkline)
                self.awareness_history.push((activity.awareness_level * 100.0) as u64);
                if self.awareness_history.len() > 20 {
                    self.awareness_history.remove(0);
                }
                
                self.coherence_history.push((activity.gradient_coherence * 100.0) as u64);
                if self.coherence_history.len() > 20 {
                    self.coherence_history.remove(0);
                }
            }
            
            // Update animation
            self.animation_frame = (self.animation_frame + 1) % 8;
        }
    }
    
    /// Render compact status bar indicator
    pub fn render_status(&self, f: &mut Frame, area: Rect) {
        if self.cognitive_enhancement.is_some() {
            let status = self.create_status_text();
            let paragraph = Paragraph::new(status)
                .style(Style::default().fg(Color::Cyan))
                .alignment(Alignment::Left);
            f.render_widget(paragraph, area);
        }
    }
    
    /// Render detailed cognitive panel
    pub fn render_panel(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" 🧠 Cognitive Activity ")
            .style(Style::default().fg(Color::Cyan));
        
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        // Split into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Awareness gauge
                Constraint::Length(3),  // Coherence gauge
                Constraint::Length(5),  // Sparklines
                Constraint::Length(4),  // Active modalities
                Constraint::Min(3),     // Current focus
            ])
            .split(inner);
        
        // Render components
        self.render_awareness_gauge(f, chunks[0]);
        self.render_coherence_gauge(f, chunks[1]);
        self.render_sparklines(f, chunks[2]);
        self.render_modalities(f, chunks[3]);
        self.render_focus(f, chunks[4]);
    }
    
    /// Create status text for status bar
    fn create_status_text(&self) -> Text<'static> {
        let awareness = if !self.awareness_history.is_empty() {
            *self.awareness_history.last().unwrap() as f64
        } else {
            0.0
        };
        
        let icon = self.get_animated_icon();
        let mode_indicator = self.get_mode_indicator();
        
        Text::from(vec![
            Line::from(vec![
                Span::styled(format!("{} ", icon), Style::default().fg(Color::Yellow)),
                Span::raw("Cognitive: "),
                Span::styled(
                    format!("{:.0}%", awareness),
                    Style::default().fg(self.get_awareness_color(awareness))
                ),
                Span::raw(" | "),
                Span::styled(mode_indicator, Style::default().fg(Color::Magenta)),
            ]),
        ])
    }
    
    /// Render awareness gauge
    fn render_awareness_gauge(&self, f: &mut Frame, area: Rect) {
        let awareness = if !self.awareness_history.is_empty() {
            *self.awareness_history.last().unwrap() as f64 / 100.0
        } else {
            0.0
        };
        
        let gauge = Gauge::default()
            .block(Block::default().title("Awareness"))
            .gauge_style(Style::default().fg(self.get_awareness_color(awareness * 100.0)))
            .percent((awareness * 100.0) as u16)
            .label(format!("{:.0}%", awareness * 100.0));
        
        f.render_widget(gauge, area);
    }
    
    /// Render coherence gauge
    fn render_coherence_gauge(&self, f: &mut Frame, area: Rect) {
        let coherence = if !self.coherence_history.is_empty() {
            *self.coherence_history.last().unwrap() as f64 / 100.0
        } else {
            0.0
        };
        
        let gauge = Gauge::default()
            .block(Block::default().title("Coherence"))
            .gauge_style(Style::default().fg(self.get_coherence_color(coherence * 100.0)))
            .percent((coherence * 100.0) as u16)
            .label(format!("{:.0}%", coherence * 100.0));
        
        f.render_widget(gauge, area);
    }
    
    /// Render sparklines for history
    fn render_sparklines(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        
        // Awareness sparkline
        let awareness_sparkline = Sparkline::default()
            .block(Block::default().title("Awareness Trend"))
            .data(&self.awareness_history)
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(awareness_sparkline, chunks[0]);
        
        // Coherence sparkline
        let coherence_sparkline = Sparkline::default()
            .block(Block::default().title("Coherence Trend"))
            .data(&self.coherence_history)
            .style(Style::default().fg(Color::Magenta));
        f.render_widget(coherence_sparkline, chunks[1]);
    }
    
    /// Render active modalities
    fn render_modalities(&self, f: &mut Frame, area: Rect) {
        let modalities: Vec<ListItem> = self.active_modalities.iter()
            .map(|m| {
                let (icon, color) = self.get_modality_style(m);
                ListItem::new(format!("{} {:?}", icon, m))
                    .style(Style::default().fg(color))
            })
            .collect();
        
        let list = List::new(modalities)
            .block(Block::default().title("Active Modalities"));
        
        f.render_widget(list, area);
    }
    
    /// Render current focus
    fn render_focus(&self, f: &mut Frame, area: Rect) {
        let focus_text = if self.cognitive_enhancement.is_some() {
            "Processing input with deep cognition..."
        } else {
            "Cognitive system initializing..."
        };
        
        let paragraph = Paragraph::new(focus_text)
            .block(Block::default().title("Current Focus"))
            .style(Style::default().fg(Color::Green))
            .alignment(Alignment::Center);
        
        f.render_widget(paragraph, area);
    }
    
    /// Get animated icon based on frame
    fn get_animated_icon(&self) -> &'static str {
        match self.animation_frame {
            0 => "🧠",
            1 => "🔮",
            2 => "💫",
            3 => "✨",
            4 => "🌟",
            5 => "💡",
            6 => "🎯",
            _ => "🧩",
        }
    }
    
    /// Get mode indicator string
    fn get_mode_indicator(&self) -> &'static str {
        if self.processing_depth > 0.8 {
            "Deep"
        } else if self.processing_depth > 0.5 {
            "Standard"
        } else {
            "Minimal"
        }
    }
    
    /// Get color based on awareness level
    fn get_awareness_color(&self, level: f64) -> Color {
        if level > 80.0 {
            Color::Green
        } else if level > 50.0 {
            Color::Yellow
        } else if level > 20.0 {
            Color::Magenta
        } else {
            Color::Red
        }
    }
    
    /// Get color based on coherence level
    fn get_coherence_color(&self, level: f64) -> Color {
        if level > 70.0 {
            Color::Cyan
        } else if level > 40.0 {
            Color::Blue
        } else {
            Color::Red
        }
    }
    
    /// Get modality style
    fn get_modality_style(&self, modality: &CognitiveModality) -> (&'static str, Color) {
        match modality {
            CognitiveModality::Logical => ("🔍", Color::Blue),
            CognitiveModality::Creative => ("🎨", Color::Magenta),
            CognitiveModality::Emotional => ("💝", Color::Red),
            CognitiveModality::Social => ("👥", Color::Green),
            CognitiveModality::Abstract => ("🌌", Color::Cyan),
            CognitiveModality::Analytical => ("📊", Color::Yellow),
            CognitiveModality::Narrative => ("📖", Color::White),
            CognitiveModality::Intuitive => ("✨", Color::LightMagenta),
        }
    }
}

/// Mini indicator for embedding in other widgets
#[derive(Clone)]
pub struct CognitiveMiniIndicator {
    awareness: f64,
    coherence: f64,
    is_processing: bool,
    pulse_frame: usize,
}

impl CognitiveMiniIndicator {
    pub fn new() -> Self {
        Self {
            awareness: 0.0,
            coherence: 0.0,
            is_processing: false,
            pulse_frame: 0,
        }
    }
    
    /// Update from consciousness activity
    pub fn update(&mut self, activity: &CognitiveActivity) {
        self.awareness = activity.awareness_level;
        self.coherence = activity.gradient_coherence;
        self.pulse_frame = (self.pulse_frame + 1) % 4;
    }
    
    /// Set processing state
    pub fn set_processing(&mut self, processing: bool) {
        self.is_processing = processing;
    }
    
    /// Render as a single line
    pub fn render_line(&self) -> Line<'static> {
        let icon = if self.is_processing {
            match self.pulse_frame {
                0 => "◐",
                1 => "◓",
                2 => "◑",
                _ => "◒",
            }
        } else {
            "●"
        };
        
        let color = if self.awareness > 0.7 {
            Color::Green
        } else if self.awareness > 0.3 {
            Color::Yellow
        } else {
            Color::Red
        };
        
        Line::from(vec![
            Span::styled(icon, Style::default().fg(color)),
            Span::raw(" "),
            Span::styled(
                format!("{:.0}%", self.awareness * 100.0),
                Style::default().fg(color)
            ),
        ])
    }
}

/// Progress indicator for cognitive operations
pub struct CognitiveProgressIndicator {
    operation: String,
    progress: f64,
    sub_operations: Vec<(String, bool)>,
}

impl CognitiveProgressIndicator {
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            progress: 0.0,
            sub_operations: vec![],
        }
    }
    
    /// Add sub-operation
    pub fn add_sub_operation(&mut self, name: String) {
        self.sub_operations.push((name, false));
    }
    
    /// Mark sub-operation as complete
    pub fn complete_sub_operation(&mut self, index: usize) {
        if index < self.sub_operations.len() {
            self.sub_operations[index].1 = true;
            self.update_progress();
        }
    }
    
    /// Update overall progress
    fn update_progress(&mut self) {
        let completed = self.sub_operations.iter().filter(|(_, done)| *done).count();
        let total = self.sub_operations.len();
        if total > 0 {
            self.progress = completed as f64 / total as f64;
        }
    }
    
    /// Render progress widget
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Progress bar
                Constraint::Min(1),     // Sub-operations
            ])
            .split(area);
        
        // Progress bar
        let gauge = Gauge::default()
            .block(Block::default().title(format!("🧠 {}", self.operation)))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent((self.progress * 100.0) as u16)
            .label(format!("{:.0}%", self.progress * 100.0));
        f.render_widget(gauge, chunks[0]);
        
        // Sub-operations list
        let items: Vec<ListItem> = self.sub_operations.iter()
            .map(|(name, done)| {
                let icon = if *done { "✓" } else { "○" };
                let color = if *done { Color::Green } else { Color::Gray };
                ListItem::new(format!("{} {}", icon, name))
                    .style(Style::default().fg(color))
            })
            .collect();
        
        let list = List::new(items);
        f.render_widget(list, chunks[1]);
    }
}

/// Create cognitive status line for embedding
pub fn create_cognitive_status_line(
    enhancement: Option<&CognitiveChatEnhancement>,
    processing: bool,
) -> Line<'static> {
    if let Some(enhancement) = enhancement {
        // Access the actual deep_processing_enabled value through Arc<RwLock>
        let deep_processing = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                enhancement.deep_processing_enabled
            })
        });
        
        let mode = if deep_processing { "Deep" } else { "Standard" };
        let icon = if processing { "🔄" } else { "🧠" };
        
        Line::from(vec![
            Span::styled(format!("{} ", icon), Style::default().fg(Color::Cyan)),
            Span::raw("Cognitive: "),
            Span::styled(mode, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ])
    } else {
        Line::from(vec![
            Span::styled("🧠 ", Style::default().fg(Color::DarkGray)),
            Span::styled("Cognitive: Offline", Style::default().fg(Color::DarkGray)),
        ])
    }
}