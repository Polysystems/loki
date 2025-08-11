//! Main chat content renderer

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

use crate::tui::chat::ChatManager;
use super::ChatRenderer;

/// Renders the main chat content area
pub struct ChatContentRenderer {
    // Configuration for rendering
    pub show_timestamps: bool,
    pub highlight_mentions: bool,
    pub theme: ChatTheme,
}

#[derive(Clone)]
pub struct ChatTheme {
    pub user_color: Color,
    pub assistant_color: Color,
    pub system_color: Color,
    pub error_color: Color,
    pub border_color: Color,
}

impl Default for ChatTheme {
    fn default() -> Self {
        Self {
            user_color: Color::Cyan,
            assistant_color: Color::Green,
            system_color: Color::Yellow,
            error_color: Color::Red,
            border_color: Color::DarkGray,
        }
    }
}

impl ChatContentRenderer {
    pub fn new() -> Self {
        Self {
            show_timestamps: true,
            highlight_mentions: true,
            theme: ChatTheme::default(),
        }
    }
    
    /// Render the chat messages area
    pub fn render_messages(&self, f: &mut Frame, area: Rect, chat_manager: &ChatManager) {
        use ratatui::widgets::{List, ListItem};
        use ratatui::text::{Line, Span};
        use ratatui::style::Modifier;
        
        // Create block with border
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.border_color))
            .title(format!("📝 Chat Messages"));
            
        // Get inner area for content
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        // Get the chat state
        if let Ok(state) = chat_manager.state.try_read() {
            let model_name = state.active_model
                .as_ref()
                .map(|m| m.as_str())
                .unwrap_or("No model");
            
            // Create a list of messages
            let messages: Vec<ListItem> = state.messages.iter()
                .flat_map(|msg_type| {
                    // Convert AssistantResponseType to display items
                    // For now, just create a simple text representation
                    vec![
                        ListItem::new(Line::from(vec![
                            Span::styled("Message", Style::default().fg(self.theme.user_color)),
                            Span::raw(": "),
                            Span::raw(format!("{:?}", msg_type)),
                        ]))
                    ]
                })
                .collect();
            
            // Add status line at the bottom
            let status_line = Line::from(vec![
                Span::raw("Model: "),
                Span::styled(model_name, Style::default().fg(Color::Cyan)),
                Span::raw(" | Messages: "),
                Span::styled(state.messages.len().to_string(), Style::default().fg(Color::Green)),
                Span::raw(" | Status: "),
                Span::styled("Ready", Style::default().fg(Color::Green)),
            ]);
            
            // Create the messages list widget
            let messages_widget = List::new(messages)
                .style(Style::default().fg(Color::White));
            
            // Split inner area for messages and status
            let inner_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(3),      // Messages
                    Constraint::Length(1),   // Status line
                ])
                .split(inner);
            
            // Render messages and status
            f.render_widget(messages_widget, inner_chunks[0]);
            f.render_widget(Paragraph::new(status_line), inner_chunks[1]);
        } else {
            // No active chat
            let no_chat = Paragraph::new("No active chat. Create a new chat to start messaging.")
                .style(Style::default().fg(Color::DarkGray))
                .block(Block::default());
            f.render_widget(no_chat, inner);
        }
    }
    
    /// Render the input area
    pub fn render_input(&self, f: &mut Frame, area: Rect, chat_manager: &ChatManager) {
        use ratatui::text::{Line, Span};
        
        // Get input state from chat manager
        let input_text = chat_manager.get_input_text();
        let cursor_pos = chat_manager.get_cursor_position();
        let is_typing = !input_text.is_empty();
        
        // Create input block with dynamic title
        let title = if is_typing {
            " Input (typing...) "
        } else {
            " Input (Press Tab to focus) "
        };
        
        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(
                if is_typing { Color::Yellow } else { self.theme.border_color }
            ))
            .title(title);
            
        let inner = input_block.inner(area);
        f.render_widget(input_block, area);
        
        // Create input content with cursor
        let mut input_line = vec![Span::raw(input_text.clone())];
        
        // Add blinking cursor
        if is_typing {
            input_line.push(Span::styled("│", Style::default().fg(Color::Yellow)));
        }
        
        // Add input hints on the right
        let hints = if is_typing {
            vec![
                Span::raw(" "),
                Span::styled("Enter", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw(" Send | "),
                Span::styled("Esc", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw(" Cancel"),
            ]
        } else {
            vec![
                Span::raw(" "),
                Span::styled("Tab", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(" to type | "),
                Span::styled("/", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::raw(" for commands"),
            ]
        };
        
        // Calculate available space for input
        let hints_width: usize = hints.iter().map(|s| s.content.len()).sum();
        let max_input_width = inner.width.saturating_sub(hints_width as u16 + 1);
        
        // Create the full line with input and hints
        let mut full_line = input_line;
        let padding = " ".repeat(max_input_width.saturating_sub(input_text.len() as u16) as usize);
        full_line.push(Span::raw(padding));
        full_line.extend(hints);
        
        let input_widget = Paragraph::new(Line::from(full_line));
        f.render_widget(input_widget, inner);
    }
    
    /// Render the context panel if visible
    pub fn render_context(&self, f: &mut Frame, area: Rect, chat_manager: &ChatManager) {
        // Get show_context_panel from chat state
        let show_panel = if let Ok(state) = chat_manager.state.try_read() {
            state.show_context_panel
        } else {
            false
        };
        
        if !show_panel {
            return;
        }
        
        let context_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.border_color))
            .title("Context");
            
        f.render_widget(context_block, area);
        
        // Context rendering is now part of the chat tab's enhanced features
        // The context panel shows thread information and cognitive insights
    }
}

impl ChatRenderer for ChatContentRenderer {
    fn render(&self, f: &mut Frame, area: Rect) {
        // Split the area into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),      // Messages area
                Constraint::Length(3),   // Input area
            ])
            .split(area);
            
        // Create a ChatManager with defaults for rendering
        let chat_manager = ChatManager::with_defaults();
        
        // Render messages area
        self.render_messages(f, chunks[0], &chat_manager);
        
        // Render input area
        self.render_input(f, chunks[1], &chat_manager);
    }
}