//! Command palette component for utilities

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use std::collections::HashMap;

use crate::tui::utilities::types::UtilitiesAction;

/// Command definition
#[derive(Debug, Clone)]
pub struct Command {
    pub name: String,
    pub description: String,
    pub shortcut: Option<String>,
    pub action: UtilitiesAction,
}

/// Command palette component
pub struct CommandPalette {
    /// Available commands
    commands: Vec<Command>,
    
    /// Current filter query
    query: String,
    
    /// Filtered commands based on query
    filtered_commands: Vec<Command>,
    
    /// Selected command index
    selected_index: usize,
    
    /// Whether the palette is active
    is_active: bool,
}

impl CommandPalette {
    pub fn new() -> Self {
        let commands = Self::init_commands();
        let filtered_commands = commands.clone();
        
        Self {
            commands,
            query: String::new(),
            filtered_commands,
            selected_index: 0,
            is_active: false,
        }
    }
    
    /// Initialize available commands
    fn init_commands() -> Vec<Command> {
        vec![
            Command {
                name: "Refresh All".to_string(),
                description: "Refresh data in all tabs".to_string(),
                shortcut: Some("Ctrl+R".to_string()),
                action: UtilitiesAction::RefreshAll,
            },
            Command {
                name: "Refresh Tools".to_string(),
                description: "Refresh tools registry".to_string(),
                shortcut: None,
                action: UtilitiesAction::RefreshTools,
            },
            Command {
                name: "Refresh MCP Servers".to_string(),
                description: "Refresh MCP server list".to_string(),
                shortcut: None,
                action: UtilitiesAction::RefreshMcpServers,
            },
            Command {
                name: "Refresh Plugins".to_string(),
                description: "Refresh plugin list".to_string(),
                shortcut: None,
                action: UtilitiesAction::RefreshPlugins,
            },
            Command {
                name: "Refresh Monitoring".to_string(),
                description: "Refresh monitoring metrics".to_string(),
                shortcut: None,
                action: UtilitiesAction::RefreshMonitoring,
            },
        ]
    }
    
    /// Show the command palette
    pub fn show(&mut self) {
        self.is_active = true;
        self.query.clear();
        self.filtered_commands = self.commands.clone();
        self.selected_index = 0;
    }
    
    /// Hide the command palette
    pub fn hide(&mut self) {
        self.is_active = false;
        self.query.clear();
    }
    
    /// Check if palette is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }
    
    /// Update search query
    pub fn update_query(&mut self, query: String) {
        self.query = query;
        self.filter_commands();
    }
    
    /// Add character to query
    pub fn add_char(&mut self, c: char) {
        self.query.push(c);
        self.filter_commands();
    }
    
    /// Remove last character from query
    pub fn backspace(&mut self) {
        self.query.pop();
        self.filter_commands();
    }
    
    /// Filter commands based on query
    fn filter_commands(&mut self) {
        if self.query.is_empty() {
            self.filtered_commands = self.commands.clone();
        } else {
            let query_lower = self.query.to_lowercase();
            self.filtered_commands = self.commands
                .iter()
                .filter(|cmd| {
                    cmd.name.to_lowercase().contains(&query_lower)
                        || cmd.description.to_lowercase().contains(&query_lower)
                })
                .cloned()
                .collect();
        }
        
        // Reset selection if it's out of bounds
        if self.selected_index >= self.filtered_commands.len() {
            self.selected_index = 0;
        }
    }
    
    /// Move selection up
    pub fn select_previous(&mut self) {
        if !self.filtered_commands.is_empty() {
            if self.selected_index == 0 {
                self.selected_index = self.filtered_commands.len() - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }
    
    /// Move selection down
    pub fn select_next(&mut self) {
        if !self.filtered_commands.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.filtered_commands.len();
        }
    }
    
    /// Get the selected command action
    pub fn get_selected_action(&self) -> Option<UtilitiesAction> {
        self.filtered_commands
            .get(self.selected_index)
            .map(|cmd| cmd.action.clone())
    }
    
    /// Render the command palette
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        if !self.is_active {
            return;
        }
        
        // Calculate palette area (centered, 60% width, 50% height)
        let palette_area = centered_rect(60, 50, area);
        
        // Clear the area behind the palette
        f.render_widget(Clear, palette_area);
        
        // Split into search bar and results
        use ratatui::layout::{Constraint, Direction, Layout};
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Search bar
                Constraint::Min(0),     // Results
                Constraint::Length(2),  // Help
            ])
            .split(palette_area);
        
        // Search bar
        let search_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title("🔍 Command Palette");
        
        let search_text = Paragraph::new(format!("> {}", self.query))
            .block(search_block)
            .style(Style::default().fg(Color::White));
        
        f.render_widget(search_text, chunks[0]);
        
        // Command list
        let items: Vec<ListItem> = self.filtered_commands
            .iter()
            .enumerate()
            .map(|(i, cmd)| {
                let mut content = vec![
                    Span::styled(&cmd.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ];
                
                if let Some(shortcut) = &cmd.shortcut {
                    content.push(Span::raw(" "));
                    content.push(Span::styled(
                        format!("[{}]", shortcut),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                
                content.push(Span::raw(" - "));
                content.push(Span::styled(&cmd.description, Style::default().fg(Color::Gray)));
                
                let style = if i == self.selected_index {
                    Style::default().bg(Color::DarkGray)
                } else {
                    Style::default()
                };
                
                ListItem::new(Line::from(content)).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Commands"));
        
        f.render_widget(list, chunks[1]);
        
        // Help text
        let help = Paragraph::new("↑↓: Navigate | Enter: Execute | Esc: Cancel")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        
        f.render_widget(help, chunks[2]);
    }
}

/// Helper function to create centered rect
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