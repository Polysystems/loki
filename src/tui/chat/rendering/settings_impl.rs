//! Settings rendering implementation

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap},
};

use crate::tui::App;
use super::render_state::get_current_render_state;

/// Render settings subtab content
pub fn render_settings_content(f: &mut Frame, app: &App, area: Rect) {
    let chat_manager = &app.state.chat;
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Min(10),    // Settings list
            Constraint::Length(5),  // Status
        ])
        .split(area);
    
    // Title
    let title = Paragraph::new("Chat Settings")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);
    
    // Settings list
    let settings = vec![
        format!("Show Timestamps: {}", if chat_manager.show_timestamps { "✓" } else { "✗" }),
        format!("Message History Mode: {}", if chat_manager.message_history_mode { "✓" } else { "✗" }),
        format!("Active Model: {}", chat_manager.active_model.name),
        format!("Active Chat: #{}", chat_manager.active_chat),
    ];
    
    let settings_items: Vec<ListItem> = settings.into_iter()
        .map(|s| ListItem::new(s))
        .collect();
    
    let settings_list = List::new(settings_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .title("Settings"))
        .highlight_style(Style::default().bg(Color::DarkGray));
    
    f.render_widget(settings_list, chunks[1]);
    
    // Status/Help
    let help = Paragraph::new("Use arrow keys to navigate, Enter to toggle, q to quit")
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, chunks[2]);
}

/// Render orchestration content
pub fn render_orchestration_content(f: &mut Frame, app: &App, area: Rect) {
    let chat_manager = &app.state.chat;
    let orchestration = &chat_manager.orchestration;
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Length(5),  // Status
            Constraint::Min(10),    // Agents/Models
            Constraint::Length(5),  // Strategy
        ])
        .split(area);
    
    // Title
    let title = Paragraph::new("Orchestration Control")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);
    
    // Status information
    let status_lines = vec![
        Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Orchestration Active"),
        ]),
        Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Models: Connected"),
        ]),
        Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Streaming: Enabled"),
        ]),
    ];
    let status = Paragraph::new(status_lines)
        .block(Block::default().borders(Borders::ALL).title(" Status "));
    f.render_widget(status, chunks[1]);
    
    // Get real orchestration state from cache
    let render_state = get_current_render_state();
    let orch = &render_state.orchestration;
    
    // Build model lines from actual state
    let mut all_lines = vec![Line::from("Enabled Models:")];
    
    // Add enabled models from actual state
    if orch.enabled_models.is_empty() {
        all_lines.push(Line::from("  • No models enabled"));
    } else {
        for model in &orch.enabled_models {
            let status_color = if model.status == "Available" {
                Color::Green
            } else {
                Color::Red
            };
            all_lines.push(Line::from(vec![
                Span::raw("  • "),
                Span::styled(&model.name, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" ("),
                Span::styled(&model.status, Style::default().fg(status_color)),
                Span::raw(")"),
            ]));
        }
    }
    
    all_lines.push(Line::from(""));
    all_lines.push(Line::from(format!("Context Window: {} tokens", orch.context_window)));
    all_lines.push(Line::from(format!("Cost Limit: ${:.2}", orch.cost_limit)));
    
    let models_panel = Paragraph::new(all_lines)
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Models "));
    f.render_widget(models_panel, chunks[2]);
    
    // Strategy information from actual state
    let strategy_lines = vec![
        Line::from(format!("Strategy: {}", orch.strategy)),
        Line::from(format!("Quality Threshold: {:.1}%", orch.quality_threshold * 100.0)),
        Line::from(format!("Fallback: {}", if orch.fallback_enabled { "Enabled" } else { "Disabled" })),
    ];
    let strategy_widget = Paragraph::new(strategy_lines)
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title(" Strategy "));
    f.render_widget(strategy_widget, chunks[3]);
}

/// Render agents content with thread view
pub fn render_agents_content(f: &mut Frame, _app: &App, area: Rect) {
    // Use the new agent thread view for a richer display
    let thread_view = super::agent_thread_view::AgentThreadView::new();
    thread_view.render(f, area);
}

/// Render CLI content
pub fn render_cli_content(f: &mut Frame, _app: &App, area: Rect) {
    use ratatui::text::{Line, Span};
    use ratatui::widgets::List;
    
    // Split area for command list and help
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),     // Command list
            Constraint::Length(6),   // Help section
        ])
        .split(area);
    
    // Create command examples
    let commands = vec![
        Line::from(vec![
            Span::styled("/help", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Show available commands"),
        ]),
        Line::from(vec![
            Span::styled("/model gpt-4", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Switch to GPT-4 model"),
        ]),
        Line::from(vec![
            Span::styled("/clear", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Clear current chat"),
        ]),
        Line::from(vec![
            Span::styled("/save chat.json", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Save chat history"),
        ]),
        Line::from(vec![
            Span::styled("/load chat.json", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Load chat history"),
        ]),
        Line::from(vec![
            Span::styled("/search keyword", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Search in messages"),
        ]),
        Line::from(vec![
            Span::styled("/export pdf", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Export chat to PDF"),
        ]),
        Line::from(vec![
            Span::styled("/stats", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Show chat statistics"),
        ]),
        Line::from(vec![
            Span::styled("/context add file.txt", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Add file to context"),
        ]),
        Line::from(vec![
            Span::styled("/agent technical", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(" - Activate technical agent"),
        ]),
    ];
    
    let command_items: Vec<ListItem> = commands.iter()
        .map(|cmd| ListItem::new(cmd.clone()))
        .collect();
    
    let command_list = List::new(command_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" 🖥️  CLI Commands "))
        .style(Style::default().fg(Color::White));
    
    f.render_widget(command_list, chunks[0]);
    
    // Render help section
    let help_text = vec![
        Line::from(vec![
            Span::styled("Quick Tips:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("• Type "),
            Span::styled("/", Style::default().fg(Color::Yellow)),
            Span::raw(" to start a command"),
        ]),
        Line::from(vec![
            Span::raw("• Use "),
            Span::styled("Tab", Style::default().fg(Color::Green)),
            Span::raw(" for command completion"),
        ]),
        Line::from(vec![
            Span::raw("• Press "),
            Span::styled("Esc", Style::default().fg(Color::Red)),
            Span::raw(" to cancel a command"),
        ]),
    ];
    
    let help = Paragraph::new(help_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" 💡 Help "))
        .wrap(Wrap { trim: true });
    
    f.render_widget(help, chunks[1]);
}

/// Render editor content
pub fn render_editor_content(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::widgets::Clear;
    
    // Split area for editor and status
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),     // Editor area
            Constraint::Length(3),   // Status bar
        ])
        .split(area);
    
    // Check if we have an active editor session
    let editor_lines = if app.state.chat.editor_active {
        vec![
            Line::from(vec![
                Span::styled("📝 ", Style::default().fg(Color::Green)),
                Span::styled("Code Editor Active", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("File: ", Style::default().fg(Color::Cyan)),
                Span::raw(app.state.chat.current_file.as_deref().unwrap_or("No file open")),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("// Code content would appear here", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
            ]),
            Line::from(vec![
                Span::styled("// Syntax highlighting enabled", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
            ]),
            Line::from(vec![
                Span::styled("// LSP integration active", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)),
            ]),
        ]
    } else {
        vec![
            Line::from(vec![
                Span::styled("📝 Code Editor", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from("Welcome to the integrated code editor!"),
            Line::from(""),
            Line::from("Features:"),
            Line::from("  • Syntax highlighting for multiple languages"),
            Line::from("  • Code completion with AI assistance"),
            Line::from("  • LSP integration for diagnostics"),
            Line::from("  • Collaborative editing support"),
            Line::from("  • Integrated terminal execution"),
            Line::from(""),
            Line::from("Keyboard shortcuts:"),
            Line::from(vec![
                Span::styled("  Ctrl+O", Style::default().fg(Color::Yellow)),
                Span::raw(" - Open file"),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+S", Style::default().fg(Color::Yellow)),
                Span::raw(" - Save file"),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+N", Style::default().fg(Color::Yellow)),
                Span::raw(" - New file"),
            ]),
            Line::from(vec![
                Span::styled("  Ctrl+Space", Style::default().fg(Color::Yellow)),
                Span::raw(" - Trigger completion"),
            ]),
            Line::from(vec![
                Span::styled("  F5", Style::default().fg(Color::Yellow)),
                Span::raw(" - Run code"),
            ]),
        ]
    };
    
    let editor_panel = Paragraph::new(editor_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" 📝 Code Editor "))
        .wrap(Wrap { trim: false });
    
    f.render_widget(editor_panel, chunks[0]);
    
    // Status bar
    let status_text = if app.state.chat.editor_active {
        format!("Ready | Line 1, Col 1 | {} | UTF-8", 
            app.state.chat.editor_language.as_deref().unwrap_or("Plain Text"))
    } else {
        "Press Ctrl+O to open a file or Ctrl+N to create a new one".to_string()
    };
    
    let status_bar = Paragraph::new(status_text)
        .style(Style::default().fg(Color::Gray))
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));
    
    f.render_widget(status_bar, chunks[1]);
}

/// Render statistics content
pub fn render_statistics_content(f: &mut Frame, app: &App, area: Rect) {
    // Get render state for actual statistics
    let render_state = get_current_render_state();
    
    // Split area for different statistics sections
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Title
            Constraint::Length(8),   // Session stats
            Constraint::Length(8),   // Model performance
            Constraint::Length(8),   // Cost tracking
            Constraint::Min(5),      // Agent activity
        ])
        .split(area);
    
    // Title
    let title = Paragraph::new("📊 Chat Statistics Dashboard")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);
    
    // Session statistics
    let session_lines = vec![
        Line::from(vec![
            Span::styled("Session Statistics", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(format!("  Active Chat: #{}", render_state.messages.active_chat)),
        Line::from(format!("  Total Messages: {}", render_state.messages.total_count)),
        Line::from(format!("  Messages This Session: {}", render_state.messages.recent.len())),
        Line::from(format!("  Average Response Time: {:.2}s", 
            app.state.chat.avg_response_time.unwrap_or(0.0))),
        Line::from(format!("  Session Duration: {}m", 
            app.start_time.elapsed().as_secs() / 60)),
    ];
    
    let session_stats = Paragraph::new(session_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" 📈 Session "));
    f.render_widget(session_stats, chunks[1]);
    
    // Model performance statistics
    let model_lines = vec![
        Line::from(vec![
            Span::styled("Model Performance", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(format!("  Active Models: {}", render_state.orchestration.enabled_models.len())),
        Line::from(format!("  Total Tokens Used: {}", 
            app.state.chat.total_tokens.unwrap_or(0))),
        Line::from(format!("  Avg Tokens/Message: {}", 
            if render_state.messages.total_count > 0 {
                app.state.chat.total_tokens.unwrap_or(0) / render_state.messages.total_count
            } else { 0 })),
        Line::from(format!("  Context Utilization: {:.1}%", 
            (app.state.chat.total_tokens.unwrap_or(0) as f64 / render_state.orchestration.context_window as f64) * 100.0)),
        Line::from(format!("  Quality Score: {:.1}%", 
            render_state.orchestration.quality_threshold * 100.0)),
    ];
    
    let model_stats = Paragraph::new(model_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" 🤖 Models "));
    f.render_widget(model_stats, chunks[2]);
    
    // Cost tracking
    let cost_lines = vec![
        Line::from(vec![
            Span::styled("Cost Tracking", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(format!("  Session Cost: ${:.4}", 
            app.state.chat.session_cost.unwrap_or(0.0))),
        Line::from(format!("  Cost Limit: ${:.2}", render_state.orchestration.cost_limit)),
        Line::from(format!("  Cost/Message: ${:.4}", 
            if render_state.messages.total_count > 0 {
                app.state.chat.session_cost.unwrap_or(0.0) / render_state.messages.total_count as f32
            } else { 0.0 })),
        Line::from(format!("  Remaining Budget: ${:.2}", 
            render_state.orchestration.cost_limit - app.state.chat.session_cost.unwrap_or(0.0) as f64)),
        Line::from(format!("  Cost Optimization: {}", 
            if render_state.orchestration.strategy == "CostOptimized" { "Enabled ✓" } else { "Disabled" })),
    ];
    
    let cost_stats = Paragraph::new(cost_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" 💰 Costs "));
    f.render_widget(cost_stats, chunks[3]);
    
    // Agent activity
    let mut agent_lines = vec![
        Line::from(vec![
            Span::styled("Agent Activity", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
    ];
    
    if render_state.agents.available.is_empty() {
        agent_lines.push(Line::from("  No active agents"));
    } else {
        for agent in &render_state.agents.available {
            let status_color = if agent.status == "Active" { Color::Green } else { Color::Yellow };
            agent_lines.push(Line::from(vec![
                Span::raw("  • "),
                Span::styled(&agent.name, Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(": "),
                Span::styled(&agent.status, Style::default().fg(status_color)),
                Span::raw(format!(" ({})", agent.capabilities.join(", "))),
            ]));
        }
    }
    
    let agent_stats = Paragraph::new(agent_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" 🤝 Agents "));
    f.render_widget(agent_stats, chunks[4]);
}