//! Complete chat content rendering implementation
//! Replaces the old draw_chat_content function

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::tui::App;
use crate::tui::chat::ModularChat;
use crate::tui::run::AssistantResponseType;
use super::render_state::get_current_render_state;

/// Render the complete chat content with multi-panel layout
pub fn render_chat_content(f: &mut Frame, app: &mut App, area: Rect) {
    let chat_manager = &app.state.chat;
    
    // Determine layout mode based on settings or keyboard shortcut
    let show_tools = app.state.show_tools_panel.unwrap_or(false);
    let show_insights = app.state.show_cognitive_panel.unwrap_or(false);
    
    if show_tools || show_insights {
        // Multi-panel layout
        render_multi_panel_layout(f, app, area, show_tools, show_insights);
    } else {
        // Standard layout
        render_standard_layout(f, app, area);
    }
}

/// Render standard chat layout
fn render_standard_layout(f: &mut Frame, app: &mut App, area: Rect) {
    let chat_manager = &app.state.chat;
    
    // Split into messages and input areas
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),    // Messages area
            Constraint::Length(3),  // Input area
        ])
        .split(area);
    
    render_messages_area(f, chat_manager, chunks[0]);
    render_input_area(f, app, chunks[1]);
}

/// Render multi-panel layout with tools and/or insights
fn render_multi_panel_layout(f: &mut Frame, app: &mut App, area: Rect, show_tools: bool, show_insights: bool) {
    let chat_manager = &app.state.chat;
    
    // Calculate panel constraints
    let mut h_constraints = vec![];
    if show_tools {
        h_constraints.push(Constraint::Percentage(25)); // Tools sidebar
    }
    h_constraints.push(Constraint::Min(30)); // Main chat
    if show_insights {
        h_constraints.push(Constraint::Percentage(30)); // Insights panel
    }
    
    // Split horizontally
    let h_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(&h_constraints)
        .split(area);
    
    let mut chunk_index = 0;
    
    // Render tools sidebar if enabled
    if show_tools {
        super::tools_sidebar::render_tools_panel(f, h_chunks[chunk_index]);
        chunk_index += 1;
    }
    
    // Render main chat area
    let chat_area = h_chunks[chunk_index];
    let chat_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),    // Messages
            Constraint::Length(3),  // Input
        ])
        .split(chat_area);
    
    render_messages_area(f, chat_manager, chat_chunks[0]);
    render_input_area(f, app, chat_chunks[1]);
    
    // Render insights panel if enabled
    if show_insights {
        chunk_index += 1;
        super::cognitive_insights::render_cognitive_panel(f, h_chunks[chunk_index], true);
    }
}

/// Render the messages area with proper styling
fn render_messages_area(f: &mut Frame, chat_manager: &ModularChat, area: Rect) {
    // Get real messages from cache
    let render_state = get_current_render_state();
    let messages = &render_state.messages;
    
    // If we have messages, show them
    if !messages.recent.is_empty() {
        let mut message_lines = Vec::new();
        
        for msg in &messages.recent {
            // Add timestamp if enabled
            let mut line_spans = Vec::new();
            
            if messages.show_timestamps {
                line_spans.push(Span::styled(
                    format!("[{}] ", msg.timestamp.format("%H:%M:%S")),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            
            // Add role with appropriate color
            let role_color = match msg.role.as_str() {
                "user" | "User" => Color::Cyan,
                "assistant" | "Assistant" => Color::Green,
                "system" => Color::Yellow,
                _ => Color::White,
            };
            
            line_spans.push(Span::styled(
                format!("{}: ", msg.role),
                Style::default().fg(role_color).add_modifier(Modifier::BOLD),
            ));
            
            // Add streaming indicator if streaming
            if msg.is_streaming {
                line_spans.push(Span::styled(
                    "⏳ ",
                    Style::default().fg(Color::Yellow),
                ));
            }
            
            message_lines.push(Line::from(line_spans));
            
            // Add content (split by lines)
            for content_line in msg.content.lines() {
                message_lines.push(Line::from(Span::raw(format!("  {}", content_line))));
            }
            
            // Add model info if available
            if let Some(model) = &msg.model {
                message_lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        format!("[{}]", model),
                        Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
            
            message_lines.push(Line::from(""));
        }
        
        // Add message count info
        if messages.total_count > messages.recent.len() {
            message_lines.push(Line::from(vec![
                Span::styled(
                    format!("... {} more messages above ...", messages.total_count - messages.recent.len()),
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                ),
            ]));
        }
        
        let messages_widget = Paragraph::new(message_lines)
            .style(Style::default())
            .alignment(Alignment::Left)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(format!(" Chat #{} ({} messages) ", messages.active_chat, messages.total_count)));
        
        f.render_widget(messages_widget, area);
    } else {
        // Show welcome message when no messages
        let info = Paragraph::new(vec![
            Line::from("💬 Chat Interface"),
            Line::from(""),
            Line::from("Welcome! Start a conversation by typing a message below."),
            Line::from(""),
            Line::from("Keyboard shortcuts:"),
            Line::from("  • Tab: Toggle between input and chat history"),
            Line::from("  • PageUp/PageDown: Scroll through messages"),
            Line::from("  • Ctrl+↑/↓: Fine scroll control"),
            Line::from("  • Enter: Send message"),
            Line::from("  • /help: Show available commands"),
        ])
            .style(Style::default().fg(Color::Cyan))
            .alignment(Alignment::Left)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Chat Overview "));
        f.render_widget(info, area);
    }
}

/// Format a single message for display
fn format_message(msg: &AssistantResponseType, index: usize, selected: Option<usize>) -> Text<'static> {
    let is_selected = selected == Some(index);
    
    let (author_style, content_style) = match msg {
        AssistantResponseType::Message { author, .. } => {
            if author == "You" || author == "user" {
                (Style::default().fg(Color::Cyan), Style::default())
            } else {
                (Style::default().fg(Color::Green), Style::default())
            }
        }
        AssistantResponseType::Error { .. } => {
            (Style::default().fg(Color::Red), Style::default().fg(Color::Red))
        }
        AssistantResponseType::Action { .. } => {
            (Style::default().fg(Color::Yellow), Style::default())
        }
        AssistantResponseType::Code { .. } => {
            (Style::default().fg(Color::Magenta), Style::default().fg(Color::Gray))
        }
        _ => (Style::default(), Style::default()),
    };
    
    let mut lines = vec![];
    
    // Add timestamp and author
    let timestamp = msg.get_timestamp();
    let author = msg.get_author();
    lines.push(Line::from(vec![
        Span::styled(format!("[{}] ", timestamp), Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}: ", author), author_style),
    ]));
    
    // Add content
    let content = msg.get_content();
    for line in content.lines() {
        lines.push(Line::from(Span::styled(line.to_string(), content_style)));
    }
    
    // Add selection indicator
    if is_selected {
        lines[0].spans.insert(0, Span::styled("▶ ", Style::default().fg(Color::Yellow)));
    }
    
    Text::from(lines)
}

/// Render the input area
fn render_input_area(f: &mut Frame, app: &App, area: Rect) {
    let input_text = app.chat_input.clone();
    let title = "Input";
    
    let input = Paragraph::new(input_text.as_str())
        .style(Style::default().fg(Color::White))
        .block(Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(if app.chat_input_focused { 
                Style::default().fg(Color::Yellow) 
            } else { 
                Style::default() 
            }));
    
    f.render_widget(input, area);
    
    // Position cursor
    if app.chat_input_focused {
        let cursor_pos = app.chat_input.len();

        #[allow(deprecated)]
        f.set_cursor(
            area.x + cursor_pos as u16 + 1,
            area.y + 1,
        );
    }
}

/// Render models subtab content
pub fn render_models_content(f: &mut Frame, _app: &App, area: Rect) {
    // Get real orchestration state from cache
    let render_state = get_current_render_state();
    let orch = &render_state.orchestration;
    
    let mut model_info = vec![
        Line::from("🤖 Model Configuration"),
        Line::from(""),
        Line::from("Enabled Models:"),
    ];
    
    // Add actual enabled models
    for model in &orch.enabled_models {
        let status_icon = if model.status == "Available" { "✓" } else { "✗" };
        model_info.push(Line::from(format!("  • {} - {} [{}]", 
            model.name, model.provider, status_icon)));
    }
    
    if orch.enabled_models.is_empty() {
        model_info.push(Line::from("  • No models enabled"));
    }
    
    // Add active model if set
    if let Some(active) = &orch.active_model {
        model_info.push(Line::from(""));
        model_info.push(Line::from(format!("Active Model: {}", active)));
    }
    
    model_info.push(Line::from(""));
    model_info.push(Line::from(format!("Strategy: {}", orch.strategy)));
    model_info.push(Line::from(format!("Fallback: {}", 
        if orch.fallback_enabled { "Enabled ✓" } else { "Disabled ✗" })));
    model_info.push(Line::from(format!("Context Window: {} tokens", orch.context_window)));
    model_info.push(Line::from(format!("Cost Limit: ${:.2}", orch.cost_limit)));
    model_info.push(Line::from(""));
    model_info.push(Line::from("Use the Settings tab to configure models."));
    
    let models_panel = Paragraph::new(model_info)
        .style(Style::default().fg(Color::Green))
        .alignment(Alignment::Left)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Model Configuration "));
    f.render_widget(models_panel, area);
}

/// Render history subtab content
pub fn render_history_content(f: &mut Frame, _app: &App, area: Rect) {
    // Get real state from cache
    let render_state = get_current_render_state();
    let messages = &render_state.messages;
    
    let history_info = vec![
        Line::from("📜 Chat History"),
        Line::from(""),
        Line::from("Recent Sessions:"),
        Line::from(format!("  • Current Session: Chat #{} (Active)", messages.active_chat)),
        Line::from("  • Previous sessions available via /load"),
        Line::from(""),
        Line::from("Statistics:"),
        Line::from(format!("  • Total Messages: {}", messages.total_count)),
        Line::from(format!("  • Active Session: #{}", messages.active_chat)),
        Line::from(format!("  • Recent Messages Cached: {}", messages.recent.len())),
        Line::from(""),
        Line::from("Commands:"),
        Line::from("  • /export - Save chat history"),
        Line::from("  • /load <file> - Load previous chat"),
        Line::from("  • /clear - Clear current chat"),
    ];
    
    let history_panel = Paragraph::new(history_info)
        .style(Style::default().fg(Color::Magenta))
        .alignment(Alignment::Left)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Chat History "));
    f.render_widget(history_panel, area);
}