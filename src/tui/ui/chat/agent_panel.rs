//! Agent Panel UI Component
//! 
//! Renders agent execution streams with task context in the chat interface.

use ratatui::{
    prelude::*,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};
use std::time::Duration;

use crate::tui::ui::chat::agent_stream_manager::{AgentStream, AgentStatus, AgentMessage, MessagePriority};

/// Agent panel configuration
pub struct AgentPanelConfig {
    pub show_task_context: bool,
    pub show_dependencies: bool,
    pub show_progress_bar: bool,
    pub max_visible_messages: usize,
    pub compact_mode: bool,
}

impl Default for AgentPanelConfig {
    fn default() -> Self {
        Self {
            show_task_context: true,
            show_dependencies: true,
            show_progress_bar: true,
            max_visible_messages: 20,
            compact_mode: false,
        }
    }
}

/// Render an agent panel
pub fn render_agent_panel(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    config: &AgentPanelConfig,
    scroll_offset: usize,
) {
    let layout = if config.show_task_context {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),    // Task context
                Constraint::Length(3), // Progress bar
                Constraint::Min(10),   // Messages
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Progress bar
                Constraint::Min(10),   // Messages
            ])
            .split(area)
    };
    
    let mut current_area = 0;
    
    // Render task context if enabled
    if config.show_task_context {
        render_task_context(f, layout[current_area], agent_stream, config);
        current_area += 1;
    }
    
    // Render progress bar
    if config.show_progress_bar {
        render_progress_bar(f, layout[current_area], agent_stream);
        current_area += 1;
    }
    
    // Render messages
    render_messages(f, layout[current_area], agent_stream, config, scroll_offset);
}

/// Render task context section
fn render_task_context(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    config: &AgentPanelConfig,
) {
    let mut lines = vec![];
    
    // Agent info
    lines.push(Line::from(vec![
        Span::styled("Agent: ", Style::default().fg(Color::DarkGray)),
        Span::styled(&agent_stream.agent_name, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled(format!("[{}]", agent_stream.agent_type), Style::default().fg(Color::DarkGray)),
    ]));
    
    // Parent task
    if let Some(parent_desc) = &agent_stream.parent_task_description {
        lines.push(Line::from(vec![
            Span::styled("Parent: ", Style::default().fg(Color::DarkGray)),
            Span::styled(parent_desc, Style::default().fg(Color::Yellow)),
        ]));
    }
    
    // Current subtask
    lines.push(Line::from(vec![
        Span::styled("Task: ", Style::default().fg(Color::DarkGray)),
        Span::styled(&agent_stream.task_description, Style::default().fg(Color::White)),
    ]));
    
    // Subtask type and effort
    if let (Some(subtask_type), Some(effort)) = (&agent_stream.subtask_type, &agent_stream.estimated_effort) {
        lines.push(Line::from(vec![
            Span::styled("Type: ", Style::default().fg(Color::DarkGray)),
            Span::styled(subtask_type, Style::default().fg(Color::Magenta)),
            Span::raw(" | "),
            Span::styled("Est: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format_duration(effort), Style::default().fg(Color::Blue)),
        ]));
    }
    
    // Dependencies
    if config.show_dependencies && !agent_stream.dependencies.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("Deps: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                agent_stream.dependencies.join(", "),
                Style::default().fg(Color::Gray)
            ),
        ]));
    }
    
    // Parallel group
    if let Some(group) = agent_stream.parallel_group {
        lines.push(Line::from(vec![
            Span::styled("Group: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("Parallel Group {}", group + 1), Style::default().fg(Color::Green)),
        ]));
    }
    
    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(format!(" {} {} ", get_status_icon(&agent_stream.status), agent_stream.status))
                .title_style(get_status_style(&agent_stream.status))
        )
        .wrap(Wrap { trim: false });
        
    f.render_widget(paragraph, area);
}

/// Render progress bar
fn render_progress_bar(f: &mut Frame, area: Rect, agent_stream: &AgentStream) {
    let progress_percent = (agent_stream.progress * 100.0) as u16;
    let duration = agent_stream.get_duration();
    
    let label = if agent_stream.status == AgentStatus::Completed {
        format!("{}% - Completed in {}", progress_percent, format_duration_chrono(&duration))
    } else {
        format!("{}% - Running for {}", progress_percent, format_duration_chrono(&duration))
    };
    
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::NONE))
        .gauge_style(Style::default().fg(get_progress_color(agent_stream.progress)))
        .percent(progress_percent)
        .label(label);
        
    f.render_widget(gauge, area);
}

/// Render messages section
fn render_messages(
    f: &mut Frame,
    area: Rect,
    agent_stream: &AgentStream,
    config: &AgentPanelConfig,
    scroll_offset: usize,
) {
    let messages: Vec<ListItem> = agent_stream.messages
        .iter()
        .skip(scroll_offset)
        .take(config.max_visible_messages)
        .map(|msg| format_message(msg, config.compact_mode))
        .collect();
        
    let messages_list = List::new(messages)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Messages ")
        )
        .highlight_style(Style::default().add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");
        
    f.render_widget(messages_list, area);
    
    // Render scrollbar if needed
    if agent_stream.messages.len() > config.max_visible_messages {
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));
            
        let mut scrollbar_state = ScrollbarState::new(agent_stream.messages.len())
            .position(scroll_offset);
            
        f.render_stateful_widget(
            scrollbar,
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

/// Format a message for display
fn format_message(msg: &AgentMessage, compact: bool) -> ListItem<'static> {
    let icon = format!("{}", msg.message_type);
    let timestamp = if compact {
        msg.timestamp.format("%H:%M:%S").to_string()
    } else {
        msg.timestamp.format("%H:%M:%S.%3f").to_string()
    };
    
    let style = match msg.priority {
        MessagePriority::Critical => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        MessagePriority::High => Style::default().fg(Color::Yellow),
        MessagePriority::Normal => Style::default().fg(Color::White),
        MessagePriority::Low => Style::default().fg(Color::Gray),
        MessagePriority::Debug => Style::default().fg(Color::DarkGray),
    };
    
    let content = if compact && msg.content.len() > 80 {
        format!("{} {} {}", icon, timestamp, &msg.content[..77])
    } else {
        format!("{} {} {}", icon, timestamp, msg.content)
    };
    
    ListItem::new(Line::from(vec![
        Span::styled(content, style)
    ]))
}

/// Get status icon
fn get_status_icon(status: &AgentStatus) -> &'static str {
    match status {
        AgentStatus::Initializing => "🔄",
        AgentStatus::Running => "▶️",
        AgentStatus::Thinking => "🤔",
        AgentStatus::ExecutingTool => "🔧",
        AgentStatus::Paused => "⏸️",
        AgentStatus::Completed => "✅",
        AgentStatus::Failed => "❌",
        AgentStatus::Cancelled => "🚫",
    }
}

/// Get status style
fn get_status_style(status: &AgentStatus) -> Style {
    match status {
        AgentStatus::Initializing => Style::default().fg(Color::Blue),
        AgentStatus::Running => Style::default().fg(Color::Green),
        AgentStatus::Thinking => Style::default().fg(Color::Yellow),
        AgentStatus::ExecutingTool => Style::default().fg(Color::Cyan),
        AgentStatus::Paused => Style::default().fg(Color::Gray),
        AgentStatus::Completed => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        AgentStatus::Failed => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        AgentStatus::Cancelled => Style::default().fg(Color::DarkGray),
    }
}

/// Get progress color
fn get_progress_color(progress: f32) -> Color {
    if progress < 0.3 {
        Color::Red
    } else if progress < 0.7 {
        Color::Yellow
    } else if progress < 1.0 {
        Color::Blue
    } else {
        Color::Green
    }
}

/// Format duration
fn format_duration(duration: &Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Format chrono duration
fn format_duration_chrono(duration: &chrono::Duration) -> String {
    let secs = duration.num_seconds();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Render multiple agent panels in a grid
pub fn render_agent_panels_grid(
    f: &mut Frame,
    area: Rect,
    agent_streams: &[AgentStream],
    config: &AgentPanelConfig,
    scroll_offsets: &[usize],
) {
    if agent_streams.is_empty() {
        return;
    }
    
    // Determine grid layout based on number of agents
    let layout = match agent_streams.len() {
        1 => {
            // Single agent - full width
            vec![area]
        }
        2 => {
            // Two agents - side by side
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area)
                .to_vec()
        }
        3..=4 => {
            // 3-4 agents - 2x2 grid
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(area);
                
            let mut areas = vec![];
            for row in rows.iter() {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(*row);
                areas.extend_from_slice(&cols);
            }
            areas
        }
        _ => {
            // More than 4 - use compact mode with list
            let agent_height = 8; // Compact height per agent
            let constraints: Vec<Constraint> = agent_streams.iter()
                .take(6) // Max 6 visible
                .map(|_| Constraint::Length(agent_height))
                .collect();
                
            Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(area)
                .to_vec()
        }
    };
    
    // Render each agent panel
    for (i, (agent_stream, area)) in agent_streams.iter().zip(layout.iter()).enumerate() {
        let scroll_offset = scroll_offsets.get(i).copied().unwrap_or(0);
        let mut panel_config = config.clone();
        
        // Use compact mode for many agents
        if agent_streams.len() > 4 {
            panel_config.compact_mode = true;
            panel_config.show_dependencies = false;
        }
        
        render_agent_panel(f, *area, agent_stream, &panel_config, scroll_offset);
    }
}

// Implement Clone for AgentPanelConfig
impl Clone for AgentPanelConfig {
    fn clone(&self) -> Self {
        Self {
            show_task_context: self.show_task_context,
            show_dependencies: self.show_dependencies,
            show_progress_bar: self.show_progress_bar,
            max_visible_messages: self.max_visible_messages,
            compact_mode: self.compact_mode,
        }
    }
}