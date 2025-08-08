//! Progress Display Widget for Command Execution
//!
//! This module provides a widget for displaying real-time progress
//! of command execution in the chat interface.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
    Frame,
};
use std::collections::VecDeque;
use crate::tui::chat::core::tool_executor::ExecutionProgress;

/// Progress display widget for showing command execution status
#[derive(Debug, Clone)]
pub struct ProgressDisplay {
    /// Current execution ID being tracked
    pub execution_id: Option<String>,
    
    /// Progress history
    pub progress_history: VecDeque<ExecutionProgress>,
    
    /// Maximum history size
    pub max_history: usize,
    
    /// Whether to show detailed progress
    pub show_details: bool,
    
    /// Animation frame for spinner
    pub animation_frame: usize,
}

impl Default for ProgressDisplay {
    fn default() -> Self {
        Self {
            execution_id: None,
            progress_history: VecDeque::with_capacity(10),
            max_history: 10,
            show_details: true,
            animation_frame: 0,
        }
    }
}

impl ProgressDisplay {
    /// Create a new progress display
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update with new progress
    pub fn update(&mut self, progress: ExecutionProgress) {
        // Update execution ID if new
        if self.execution_id.as_ref() != Some(&progress.execution_id) {
            self.execution_id = Some(progress.execution_id.clone());
            self.progress_history.clear();
        }
        
        // Add to history
        self.progress_history.push_back(progress);
        
        // Maintain max history
        while self.progress_history.len() > self.max_history {
            self.progress_history.pop_front();
        }
        
        // Update animation
        self.animation_frame = (self.animation_frame + 1) % 8;
    }
    
    /// Clear progress display
    pub fn clear(&mut self) {
        self.execution_id = None;
        self.progress_history.clear();
        self.animation_frame = 0;
    }
    
    /// Check if there's active progress
    pub fn is_active(&self) -> bool {
        self.execution_id.is_some() && 
        self.progress_history.back()
            .map(|p| p.progress < 1.0)
            .unwrap_or(false)
    }
    
    /// Render the progress display
    pub fn render(&self, f: &mut Frame, area: Rect) {
        if !self.is_active() && self.progress_history.is_empty() {
            return;
        }
        
        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Progress bar
                Constraint::Min(0),    // Details
            ])
            .split(area);
        
        // Get latest progress
        if let Some(latest) = self.progress_history.back() {
            // Render progress bar
            self.render_progress_bar(f, chunks[0], latest);
            
            // Render details if enabled
            if self.show_details && chunks[1].height > 0 {
                self.render_details(f, chunks[1]);
            }
        }
    }
    
    /// Render inline progress (for embedding in messages)
    pub fn render_inline(&self) -> String {
        if let Some(latest) = self.progress_history.back() {
            let spinner = self.get_spinner();
            let percentage = (latest.progress * 100.0) as u8;
            
            if latest.progress >= 1.0 {
                format!("✅ {} - Complete", latest.stage)
            } else {
                format!("{} {}% - {} - {}", spinner, percentage, latest.stage, latest.message)
            }
        } else {
            String::new()
        }
    }
    
    /// Render the progress bar
    fn render_progress_bar(&self, f: &mut Frame, area: Rect, progress: &ExecutionProgress) {
        let percentage = (progress.progress * 100.0) as u16;
        
        let label = if progress.progress >= 1.0 {
            format!("✅ Complete - {}", progress.stage)
        } else {
            format!("{} {}% - {}", self.get_spinner(), percentage, progress.stage)
        };
        
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Execution Progress"))
            .gauge_style(Style::default().fg(Color::Cyan))
            .percent(percentage)
            .label(label);
        
        f.render_widget(gauge, area);
    }
    
    /// Render progress details
    fn render_details(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.progress_history
            .iter()
            .map(|progress| {
                let percentage = (progress.progress * 100.0) as u8;
                let icon = if progress.progress >= 1.0 { "✓" } else { "→" };
                
                let content = if let Some(details) = &progress.details {
                    format!("{} [{}%] {} - {} ({})", 
                        icon, percentage, progress.stage, progress.message,
                        serde_json::to_string(details).unwrap_or_default()
                    )
                } else {
                    format!("{} [{}%] {} - {}", 
                        icon, percentage, progress.stage, progress.message
                    )
                };
                
                ListItem::new(content).style(
                    if progress.progress >= 1.0 {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::Yellow)
                    }
                )
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Execution Details"));
        
        f.render_widget(list, area);
    }
    
    /// Get spinner character based on animation frame
    fn get_spinner(&self) -> &'static str {
        match self.animation_frame {
            0 => "⠋",
            1 => "⠙",
            2 => "⠹",
            3 => "⠸",
            4 => "⠼",
            5 => "⠴",
            6 => "⠦",
            7 => "⠧",
            _ => "⠋",
        }
    }
}

/// Progress overlay for full-screen progress display
pub struct ProgressOverlay {
    /// Progress display widget
    pub progress: ProgressDisplay,
    
    /// Whether to show the overlay
    pub visible: bool,
    
    /// Title for the overlay
    pub title: String,
    
    /// Whether user can cancel
    pub cancellable: bool,
}

impl ProgressOverlay {
    /// Create a new progress overlay
    pub fn new(title: String) -> Self {
        Self {
            progress: ProgressDisplay::new(),
            visible: false,
            title,
            cancellable: true,
        }
    }
    
    /// Show the overlay
    pub fn show(&mut self) {
        self.visible = true;
    }
    
    /// Hide the overlay
    pub fn hide(&mut self) {
        self.visible = false;
    }
    
    /// Update progress
    pub fn update(&mut self, progress: ExecutionProgress) {
        self.progress.update(progress);
    }
    
    /// Render the overlay
    pub fn render(&self, f: &mut Frame) {
        if !self.visible {
            return;
        }
        
        let area = centered_rect(60, 40, f.area());
        
        // Clear background
        f.render_widget(
            Block::default()
                .style(Style::default().bg(Color::Black))
                .borders(Borders::NONE),
            area,
        );
        
        // Render progress with border
        let block = Block::default()
            .title(self.title.as_str())
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White));
        
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        // Add cancel hint if cancellable
        if self.cancellable {
            let cancel_hint = Paragraph::new("Press ESC to cancel")
                .style(Style::default().fg(Color::Gray))
                .alignment(Alignment::Center);
            
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(inner);
            
            self.progress.render(f, chunks[0]);
            f.render_widget(cancel_hint, chunks[1]);
        } else {
            self.progress.render(f, inner);
        }
    }
}

/// Helper function to create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}