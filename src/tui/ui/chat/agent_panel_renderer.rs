//! Agent Panel Rendering for TUI Chat Interface
//!
//! Provides rendering functions for agent streams in dedicated panels,
//! showing real-time progress, messages, and status.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},

    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
    Frame,
};

use crate::tui::ui::chat::{
    AgentStream, AgentMessage, AgentMessageType, MessagePriority, AgentStatus,
    ChatTheme,
};

/// Render an agent panel with stream content
pub fn render_agent_panel(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    theme: &ChatTheme,
    is_focused: bool,
) {
    // Split area into header, messages, and status
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),    // Header with progress
            Constraint::Min(10),      // Messages
            Constraint::Length(2),    // Status bar
        ])
        .split(area);
    
    // Render header
    render_agent_header(f, chunks[0], agent_stream, theme, is_focused);
    
    // Render messages
    render_agent_messages(f, chunks[1], agent_stream, theme);
    
    // Render status bar
    render_agent_status(f, chunks[2], agent_stream, theme);
}

/// Render agent header with name, task, and progress
fn render_agent_header(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    theme: &ChatTheme,
    _is_focused: bool,
) {
    let header_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),  // Title
            Constraint::Length(1),  // Progress bar
            Constraint::Length(1),  // Separator
        ])
        .split(area);
    
    // Agent title with status icon
    let status_icon = match agent_stream.status {
        AgentStatus::Initializing => "🔄",
        AgentStatus::Running => "▶️",
        AgentStatus::Thinking => "🤔",
        AgentStatus::ExecutingTool => "🔧",
        AgentStatus::Paused => "⏸️",
        AgentStatus::Completed => "✅",
        AgentStatus::Failed => "❌",
        AgentStatus::Cancelled => "🚫",
    };
    
    let title = vec![
        Span::raw(status_icon),
        Span::raw(" "),
        Span::styled(
            &agent_stream.agent_name,
            Style::default()
                .fg(theme.colors.primary)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" - "),
        Span::styled(
            &agent_stream.agent_type,
            Style::default().fg(theme.colors.secondary),
        ),
    ];
    
    let title_widget = Paragraph::new(Line::from(title))
        .alignment(Alignment::Left);
    f.render_widget(title_widget, header_chunks[0]);
    
    // Progress bar
    let progress_label = format!(
        "{:.0}% - {}",
        agent_stream.progress * 100.0,
        truncate_string(&agent_stream.task_description, area.width as usize - 10)
    );
    
    let progress_color = match agent_stream.status {
        AgentStatus::Completed => Color::Green,
        AgentStatus::Failed => Color::Red,
        AgentStatus::Running | AgentStatus::Thinking | AgentStatus::ExecutingTool => Color::Cyan,
        _ => Color::Gray,
    };
    
    let progress = Gauge::default()
        .block(Block::default())
        .gauge_style(Style::default().fg(progress_color))
        .ratio(agent_stream.progress as f64)
        .label(progress_label);
    
    f.render_widget(progress, header_chunks[1]);
    
    // Separator
    let separator = Paragraph::new("─".repeat(area.width as usize))
        .style(Style::default().fg(theme.colors.border));
    f.render_widget(separator, header_chunks[2]);
}

/// Render agent messages
fn render_agent_messages(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    theme: &ChatTheme,
) {
    let messages: Vec<ListItem> = agent_stream.messages
        .iter()
        .rev()  // Show newest first
        .take(area.height as usize - 2)  // Limit to visible area
        .map(|msg| render_agent_message(msg, theme, area.width))
        .collect();
    
    let messages_list = List::new(messages)
        .block(
            Block::default()
                .borders(Borders::NONE)
                .padding(ratatui::widgets::Padding::horizontal(1))
        );
    
    f.render_widget(messages_list, area);
}

/// Render a single agent message
fn render_agent_message(
    message: &AgentMessage,
    theme: &ChatTheme,
    width: u16,
) -> ListItem<'static> {
    let timestamp = message.timestamp.format("%H:%M:%S").to_string();
    
    let (icon, color) = match message.message_type {
        AgentMessageType::Thought => ("💭", theme.colors.info),
        AgentMessageType::Action => ("⚡", theme.colors.warning),
        AgentMessageType::Observation => ("👁️", theme.colors.success),
        AgentMessageType::Progress => ("📊", theme.colors.primary),
        AgentMessageType::Error => ("❌", theme.colors.error),
        AgentMessageType::Result => ("✅", theme.colors.success),
        AgentMessageType::Debug => ("🐛", theme.colors.foreground_dim),
        AgentMessageType::ToolInvocation => ("🔧", theme.colors.secondary),
        AgentMessageType::ToolResult => ("📤", theme.colors.info),
    };
    
    let priority_indicator = match message.priority {
        MessagePriority::Critical => "‼️",
        MessagePriority::High => "❗",
        MessagePriority::Normal => "",
        MessagePriority::Low => "",
        MessagePriority::Debug => "🔍",
    };
    
    // Format message content
    let content = truncate_string(&message.content, width as usize - 15);
    
    let spans = vec![
        Span::styled(timestamp, Style::default().fg(theme.colors.foreground_dim)),
        Span::raw(" "),
        Span::raw(icon),
        Span::raw(priority_indicator),
        Span::raw(" "),
        Span::styled(content, Style::default().fg(color)),
    ];
    
    ListItem::new(Line::from(spans))
}

/// Render agent status bar
fn render_agent_status(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    theme: &ChatTheme,
) {
    let duration = agent_stream.get_duration();
    let duration_str = format!(
        "{}m {}s",
        duration.num_minutes(),
        duration.num_seconds() % 60
    );
    
    let message_stats = format!(
        "Messages: {} | Duration: {}",
        agent_stream.messages.len(),
        duration_str
    );
    
    let important_count = agent_stream.get_important_messages().len();
    let error_count = agent_stream.get_messages_by_type(&AgentMessageType::Error).len();
    
    let mut spans = vec![
        Span::styled(
            message_stats,
            Style::default().fg(theme.colors.foreground_dim),
        ),
    ];
    
    if important_count > 0 {
        spans.push(Span::raw(" | "));
        spans.push(Span::styled(
            format!("Important: {}", important_count),
            Style::default().fg(theme.colors.warning),
        ));
    }
    
    if error_count > 0 {
        spans.push(Span::raw(" | "));
        spans.push(Span::styled(
            format!("Errors: {}", error_count),
            Style::default().fg(theme.colors.error),
        ));
    }
    
    let status_line = Paragraph::new(Line::from(spans))
        .alignment(Alignment::Center);
    
    f.render_widget(status_line, area);
}

/// Render agent overview panel showing all active agents
pub fn render_agent_overview(
    f: &mut Frame,
    area: Rect,
    agents: &[AgentStream],
    theme: &ChatTheme,
    selected_index: Option<usize>,
) {
    let block = Block::default()
        .title(" 🤖 Active Agents ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.colors.border));
    
    let inner = block.inner(area);
    f.render_widget(block, area);
    
    if agents.is_empty() {
        let empty_msg = Paragraph::new("No active agents")
            .style(Style::default().fg(theme.colors.foreground_dim))
            .alignment(Alignment::Center);
        f.render_widget(empty_msg, inner);
        return;
    }
    
    // Create list items for each agent
    let items: Vec<ListItem> = agents
        .iter()
        .enumerate()
        .map(|(idx, agent)| {
            let is_selected = selected_index == Some(idx);
            render_agent_overview_item(agent, theme, is_selected)
        })
        .collect();
    
    let agents_list = List::new(items)
        .highlight_style(
            Style::default()
                .bg(theme.colors.overlay)
                .fg(theme.colors.foreground_bright)
                .add_modifier(Modifier::BOLD)
        );
    
    f.render_widget(agents_list, inner);
}

/// Render a single agent in the overview
fn render_agent_overview_item(
    agent: &AgentStream,
    theme: &ChatTheme,
    is_selected: bool,
) -> ListItem<'static> {
    let status_color = match agent.status {
        AgentStatus::Completed => theme.colors.success,
        AgentStatus::Failed => theme.colors.error,
        AgentStatus::Running | AgentStatus::Thinking | AgentStatus::ExecutingTool => theme.colors.info,
        _ => theme.colors.foreground_dim,
    };
    
    let spans = vec![
        Span::styled(
            format!("{}", agent.status),
            Style::default().fg(status_color),
        ),
        Span::raw(" "),
        Span::styled(
            agent.agent_name.clone(),
            Style::default()
                .fg(if is_selected { theme.colors.foreground_bright } else { theme.colors.primary })
                .add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() }),
        ),
        Span::raw(" - "),
        Span::styled(
            format!("{:.0}%", agent.progress * 100.0),
            Style::default().fg(theme.colors.secondary),
        ),
        Span::raw(" - "),
        Span::styled(
            truncate_string(&agent.task_description, 40),
            Style::default().fg(theme.colors.text),
        ),
    ];
    
    ListItem::new(Line::from(spans))
}

/// Helper function to truncate strings
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Render a mini agent status in the main chat
pub fn render_agent_status_inline(
    agent: &AgentStream,
    theme: &ChatTheme,
) -> Vec<Span<'static>> {
    vec![
        Span::raw("["),
        Span::styled(
            format!("{}", agent.status),
            Style::default().fg(match agent.status {
                AgentStatus::Completed => theme.colors.success,
                AgentStatus::Failed => theme.colors.error,
                _ => theme.colors.info,
            }),
        ),
        Span::raw("] "),
        Span::styled(
            agent.agent_name.clone(),
            Style::default().fg(theme.colors.primary),
        ),
        Span::raw(": "),
        Span::styled(
            format!("{:.0}%", agent.progress * 100.0),
            Style::default().fg(theme.colors.secondary),
        ),
    ]
}