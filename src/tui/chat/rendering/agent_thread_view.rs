//! Agent thread view for parallel conversations

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, BorderType, List, ListItem, Paragraph, Wrap, Gauge},
};
use std::collections::HashMap;

use super::get_current_render_state;

/// Agent thread information
#[derive(Debug, Clone)]
pub struct AgentThread {
    pub agent_name: String,
    pub agent_type: String,
    pub status: ThreadStatus,
    pub current_task: Option<String>,
    pub messages: Vec<ThreadMessage>,
    pub progress: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThreadStatus {
    Idle,
    Working,
    Waiting,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ThreadMessage {
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub is_from_agent: bool,
}

/// Renders agent threads showing parallel conversations
pub struct AgentThreadView {
    pub selected_thread: usize,
    pub show_all_threads: bool,
}

impl AgentThreadView {
    pub fn new() -> Self {
        Self {
            selected_thread: 0,
            show_all_threads: true,
        }
    }
    
    /// Calculate progress based on agent status and task info
    fn calculate_progress(&self, agent: &super::render_state::AgentInfo) -> u16 {
        // Dynamic progress calculation based on agent status
        match agent.status.as_str() {
            "Idle" => 0,
            "Starting" => 10,
            "Active" | "Working" => {
                // For active agents, estimate progress based on time or messages
                // In a real implementation, this would track actual task progress
                let render_state = get_current_render_state();
                
                // Base progress on message count (simple heuristic)
                let msg_count = render_state.messages.recent.len();
                let progress = match msg_count {
                    0..=2 => 20,
                    3..=5 => 40,
                    6..=10 => 60,
                    11..=15 => 80,
                    _ => 90,
                };
                
                // If streaming, add some progress
                if let Some(last_msg) = render_state.messages.recent.last() {
                    if last_msg.is_streaming {
                        std::cmp::min(progress + 5, 95)
                    } else {
                        progress
                    }
                } else {
                    progress
                }
            },
            "Completed" => 100,
            "Failed" => 0,
            _ => 50, // Unknown status
        }
    }
    
    /// Render the complete agent thread view
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        
        // Split into thread list and selected thread detail
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40),  // Thread list
                Constraint::Percentage(60),  // Thread detail
            ])
            .split(area);
            
        self.render_thread_list(f, chunks[0]);
        self.render_thread_detail(f, chunks[1]);
    }
    
    /// Render the list of active agent threads
    fn render_thread_list(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut items = Vec::new();
        
        for (i, agent) in render_state.agents.available.iter().enumerate() {
            let status_icon = match agent.status.as_str() {
                "Active" => "🟢",
                "Idle" => "⚫",
                "Working" => "🔄",
                "Waiting" => "⏳",
                _ => "❓",
            };
            
            let selected = i == self.selected_thread;
            let style = if selected {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };
            
            items.push(ListItem::new(vec![
                Line::from(vec![
                    Span::raw(format!("{} ", status_icon)),
                    Span::styled(&agent.name, 
                        style.fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("Type: {}", agent.description), 
                        style.fg(Color::Yellow)),
                ]),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("Status: {}", agent.status), 
                        style.fg(match agent.status.as_str() {
                            "Active" => Color::Green,
                            "Idle" => Color::DarkGray,
                            _ => Color::Yellow,
                        })),
                ]),
                if !agent.capabilities.is_empty() {
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(format!("Skills: {}", agent.capabilities.join(", ")), 
                            style.fg(Color::DarkGray)),
                    ])
                } else {
                    Line::from("")
                },
            ]));
        }
        
        if items.is_empty() {
            items.push(ListItem::new(Line::from(
                Span::styled("No active agents", 
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))
            )));
        }
        
        let thread_list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" 🤝 Agent Threads "))
            .highlight_style(Style::default().bg(Color::DarkGray));
            
        f.render_widget(thread_list, area);
    }
    
    /// Render the selected thread's conversation detail
    fn render_thread_detail(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        
        // Split into header, messages, and input areas
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),   // Header
                Constraint::Min(10),     // Messages
                Constraint::Length(3),   // Progress/Status
            ])
            .split(area);
            
        // Get selected agent
        if let Some(agent) = render_state.agents.available.get(self.selected_thread) {
            // Render header
            self.render_thread_header(f, chunks[0], agent);
            
            // Render conversation
            self.render_conversation(f, chunks[1], agent);
            
            // Render progress
            self.render_progress(f, chunks[2], agent);
        } else {
            // No agent selected
            let placeholder = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled("Select an agent thread to view conversation", 
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))),
                Line::from(""),
                Line::from(Span::styled("Use ↑/↓ to navigate threads", 
                    Style::default().fg(Color::DarkGray))),
            ])
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Thread Detail "))
            .alignment(Alignment::Center);
            
            f.render_widget(placeholder, area);
        }
    }
    
    /// Render thread header with agent info
    fn render_thread_header(&self, f: &mut Frame, area: Rect, agent: &super::render_state::AgentInfo) {
        let lines = vec![
            Line::from(vec![
                Span::styled("Agent: ", Style::default().fg(Color::DarkGray)),
                Span::styled(&agent.name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(" | "),
                Span::styled(&agent.description, Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("Capabilities: ", Style::default().fg(Color::DarkGray)),
                Span::raw(agent.capabilities.join(", ")),
            ]),
        ];
        
        let header = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)));
            
        f.render_widget(header, area);
    }
    
    /// Render the conversation messages
    fn render_conversation(&self, f: &mut Frame, area: Rect, agent: &super::render_state::AgentInfo) {
        let render_state = get_current_render_state();
        let mut lines = Vec::new();
        
        // Show current task if any
        if agent.status == "Working" || agent.status == "Active" {
            lines.push(Line::from(vec![
                Span::styled("🔄 ", Style::default().fg(Color::Yellow)),
                Span::styled("Current Task: ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!("Processing with {} agent...", agent.name)),
            ]));
            lines.push(Line::from(""));
        }
        
        // Show actual messages from the chat if available
        if !render_state.messages.recent.is_empty() {
            // Display recent messages (limit to last 10 for performance)
            let message_count = render_state.messages.recent.len();
            let start_idx = if message_count > 10 { message_count - 10 } else { 0 };
            
            for msg in &render_state.messages.recent[start_idx..] {
                let (prefix, style) = match msg.role.as_str() {
                    "user" => ("User: ".to_string(), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    "assistant" => (format!("{}: ", agent.name), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    "system" => ("System: ".to_string(), Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC)),
                    _ => ("Unknown: ".to_string(), Style::default().fg(Color::DarkGray)),
                };
                
                // Split long messages into multiple lines
                let content_lines: Vec<&str> = msg.content.lines().collect();
                if !content_lines.is_empty() {
                    // First line with role prefix
                    lines.push(Line::from(vec![
                        Span::styled(prefix, style),
                        Span::raw(content_lines[0]),
                    ]));
                    
                    // Subsequent lines with indentation
                    for line in &content_lines[1..] {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::raw(*line),
                        ]));
                    }
                }
                
                // Add timestamp if enabled
                if render_state.messages.show_timestamps {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(
                            format!("[{}]", msg.timestamp.format("%H:%M:%S")),
                            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)
                        ),
                    ]));
                }
                
                lines.push(Line::from(""));
            }
            
            // Show if a message is currently streaming
            if let Some(last_msg) = render_state.messages.recent.last() {
                if last_msg.is_streaming {
                    lines.push(Line::from(vec![
                        Span::styled("⏳ ", Style::default().fg(Color::Yellow)),
                        Span::styled("Streaming response...", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
                    ]));
                }
            }
        } else {
            // Fallback to showing agent is ready
            lines.push(Line::from(vec![
                Span::styled("💬 ", Style::default().fg(Color::Cyan)),
                Span::styled(format!("{} is ready to assist", agent.name), 
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
            ]));
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("No messages yet. Start a conversation!", 
                    Style::default().fg(Color::DarkGray)),
            ]));
        }
        
        // Show active status
        if agent.status == "Active" && !lines.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("⚡ ", Style::default().fg(Color::Green)),
                Span::styled("Agent is actively processing...", 
                    Style::default().fg(Color::Green).add_modifier(Modifier::ITALIC)),
            ]));
        }
        
        let conversation = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(format!(" 💬 Conversation with {} ", agent.name)))
            .wrap(Wrap { trim: false });
            
        f.render_widget(conversation, area);
    }
    
    /// Render progress bar for active tasks
    fn render_progress(&self, f: &mut Frame, area: Rect, agent: &super::render_state::AgentInfo) {
        if agent.status == "Active" || agent.status == "Working" {
            // Show progress gauge
            let progress = Gauge::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" Progress "))
                .gauge_style(Style::default().fg(Color::Green))
                .percent(self.calculate_progress(agent))
                .label("Processing...");
                
            f.render_widget(progress, area);
        } else {
            // Show status
            let status_text = match agent.status.as_str() {
                "Idle" => "Agent is idle and ready for tasks",
                "Completed" => "Last task completed successfully",
                "Failed" => "Last task failed - check logs",
                _ => "Unknown status",
            };
            
            let status = Paragraph::new(status_text)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" Status "))
                .style(Style::default().fg(match agent.status.as_str() {
                    "Idle" => Color::DarkGray,
                    "Completed" => Color::Green,
                    "Failed" => Color::Red,
                    _ => Color::Yellow,
                }))
                .alignment(Alignment::Center);
                
            f.render_widget(status, area);
        }
    }
}