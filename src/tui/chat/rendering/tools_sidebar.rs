//! Tools sidebar panel for displaying available and active tools

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, BorderType, List, ListItem, Paragraph, Wrap, Gauge},
};
use std::collections::HashMap;

use super::get_current_render_state;

/// Tool execution status
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_name: String,
    pub status: ExecutionStatus,
    pub progress: f32,
    pub result: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Success,
    Failed,
    Cancelled,
}

/// Renders a tools sidebar showing available tools and execution status
pub struct ToolsSidebar {
    pub selected_tool: usize,
    pub show_execution_log: bool,
}

impl ToolsSidebar {
    pub fn new() -> Self {
        Self {
            selected_tool: 0,
            show_execution_log: false,
        }
    }
    
    /// Render the tools sidebar
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        
        // Split into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Header
                Constraint::Min(10),     // Available tools
                Constraint::Length(10),  // Active executions
                Constraint::Min(5),      // Suggestions
            ])
            .split(area);
            
        self.render_header(f, chunks[0]);
        self.render_available_tools(f, chunks[1]);
        self.render_active_executions(f, chunks[2]);
        self.render_tool_suggestions(f, chunks[3]);
    }
    
    /// Render the header
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let active_count = render_state.tools.active_executions.len();
        
        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("🔧 Tools ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw("| "),
                Span::styled(format!("{} active", active_count), 
                    Style::default().fg(if active_count > 0 { Color::Green } else { Color::DarkGray })),
            ]),
        ])
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Thick));
            
        f.render_widget(header, area);
    }
    
    /// Render available tools list
    fn render_available_tools(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut items = Vec::new();
        
        // Group tools by category
        let tool_categories = vec![
            ("Development", vec!["GitHub", "Git", "Code Analysis", "Testing"]),
            ("Communication", vec!["Slack", "Email", "Discord"]),
            ("Research", vec!["Web Search", "Wikipedia", "ArXiv"]),
            ("Data", vec!["Database", "Analytics", "Monitoring"]),
            ("AI/ML", vec!["Model Training", "Inference", "Evaluation"]),
        ];
        
        for (category, tools) in tool_categories {
            // Category header
            items.push(ListItem::new(Line::from(vec![
                Span::styled(category, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ])));
            
            // Tools in category
            for tool in tools {
                let is_available = render_state.tools.available.iter().any(|t| t.name == tool);
                let is_active = render_state.tools.active_executions.contains(&tool.to_string());
                
                let (icon, color) = if is_active {
                    ("🟢", Color::Green)
                } else if is_available {
                    ("⚪", Color::White)
                } else {
                    ("⚫", Color::DarkGray)
                };
                
                items.push(ListItem::new(Line::from(vec![
                    Span::raw("  "),
                    Span::raw(icon),
                    Span::raw(" "),
                    Span::styled(tool, Style::default().fg(color)),
                ])));
            }
        }
        
        let tools_list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Available Tools "))
            .highlight_style(Style::default().bg(Color::DarkGray));
            
        f.render_widget(tools_list, area);
    }
    
    /// Render active tool executions
    fn render_active_executions(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        
        if render_state.tools.active_executions.is_empty() {
            let placeholder = Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled("No active tool executions", 
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))),
                Line::from(""),
                Line::from(Span::styled("Tools will appear here when running", 
                    Style::default().fg(Color::DarkGray))),
            ])
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Active Executions "))
            .alignment(Alignment::Center);
            
            f.render_widget(placeholder, area);
        } else {
            let mut lines = Vec::new();
            
            for tool in &render_state.tools.active_executions {
                lines.push(Line::from(vec![
                    Span::styled("▶ ", Style::default().fg(Color::Green)),
                    Span::styled(tool, Style::default().fg(Color::Cyan)),
                    Span::raw(" "),
                    Span::styled("[Running]", Style::default().fg(Color::Yellow)),
                ]));
                
                // Show progress bar inline
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("████████░░", Style::default().fg(Color::Green)),
                    Span::raw(" 80%"),
                ]));
            }
            
            let executions = Paragraph::new(lines)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(" Active Executions "));
                
            f.render_widget(executions, area);
        }
    }
    
    /// Render tool suggestions based on context
    fn render_tool_suggestions(&self, f: &mut Frame, area: Rect) {
        let render_state = get_current_render_state();
        let mut lines = Vec::new();
        
        lines.push(Line::from(vec![
            Span::styled("💡 Suggestions", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Line::from(""));
        
        // Context-based suggestions
        if render_state.messages.recent.len() > 0 {
            // Analyze last message for tool suggestions
            if let Some(last_msg) = render_state.messages.recent.last() {
                if last_msg.content.contains("code") || last_msg.content.contains("github") {
                    lines.push(Line::from(vec![
                        Span::raw("• Use "),
                        Span::styled("GitHub", Style::default().fg(Color::Cyan)),
                        Span::raw(" for repository operations"),
                    ]));
                }
                if last_msg.content.contains("search") || last_msg.content.contains("find") {
                    lines.push(Line::from(vec![
                        Span::raw("• Use "),
                        Span::styled("Web Search", Style::default().fg(Color::Cyan)),
                        Span::raw(" to find information"),
                    ]));
                }
                if last_msg.content.contains("test") || last_msg.content.contains("run") {
                    lines.push(Line::from(vec![
                        Span::raw("• Use "),
                        Span::styled("Testing", Style::default().fg(Color::Cyan)),
                        Span::raw(" to execute tests"),
                    ]));
                }
            }
        }
        
        if lines.len() == 2 { // Only header and empty line
            lines.push(Line::from(Span::styled("No suggestions available", 
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))));
        }
        
        let suggestions = Paragraph::new(lines)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Tool Suggestions "))
            .wrap(Wrap { trim: false });
            
        f.render_widget(suggestions, area);
    }
}

/// Render tools panel in chat view (can be used as a sidebar)
pub fn render_tools_panel(f: &mut Frame, area: Rect) {
    let sidebar = ToolsSidebar::new();
    sidebar.render(f, area);
}