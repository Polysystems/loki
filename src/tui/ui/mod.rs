pub mod auth;
pub mod chat;
// tabs module moved to /src/tui/tabs/
pub mod widgets;
pub mod cognitive_indicators;
pub mod story_visualization;
pub mod enhanced_cognitive_panel;
use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs, Wrap};
// Re-export from the new location for backwards compatibility
pub use crate::tui::tabs::*;

/// Notification types for UI feedback
#[derive(Debug, Clone, PartialEq)]
pub enum NotificationType {
    Info,
    Success,
    Warning,
    Error,
    Optimization,
}

/// A notification to display to the user
#[derive(Debug, Clone)]
pub struct Notification {
    pub notification_type: NotificationType,
    pub message: String,
    pub timestamp: Instant,
}

impl Notification {
    pub fn new(notification_type: NotificationType, message: String) -> Self {
        Self {
            notification_type,
            message,
            timestamp: Instant::now(),
        }
    }
    
    /// Check if the notification has expired (older than 5 seconds)
    pub fn is_expired(&self) -> bool {
        self.timestamp.elapsed().as_secs() > 5
    }
}

use super::app::App;
use super::state::ViewState;

/// Enhanced animation timing for smooth transitions
const PULSE_SPEED: f32 = 2.0;

/// Main enhanced draw function with visual improvements
pub fn draw(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Enhanced header
            Constraint::Min(0),    // Enhanced content
            Constraint::Length(1), // Enhanced status bar
        ])
        .split(f.area());

    draw_enhanced_header(f, app, chunks[0]);
    draw_enhanced_content(f, app, chunks[1]);
    draw_enhanced_status_bar(f, app, chunks[2]);

    if app.state.show_help {
        draw_enhanced_help_overlay(f, app);
    }
    
    // Draw notifications
    draw_notifications(f, app);
}

/// Enhanced header with animations and visual improvements
fn draw_enhanced_header(f: &mut Frame, app: &App, area: Rect) {
    let titles = vec![
        "Home [1]",
        "Chat [2]", 
        "Utilities [3]",
        "Memory [4]",
        "Cognitive [5]",
        "Social [6]",
        "Settings [7]",
    ];

    let selected = match app.state.current_view {
        ViewState::Dashboard => 0,
        ViewState::Chat => 1,
        ViewState::Utilities => 2,
        ViewState::Memory => 3,
        ViewState::Cognitive => 4,
        ViewState::Streams => 5,
        ViewState::Models => 6,
        ViewState::Collaborative => 8,
        ViewState::PluginEcosystem => 9,
    };

    // Enhanced tab styling with animations
    let current_time = Instant::now();
    let pulse = (current_time.elapsed().as_secs_f32() * PULSE_SPEED).sin().abs();
    let pulse_color = Color::Rgb((128.0 + pulse * 127.0) as u8, (200.0 + pulse * 55.0) as u8, 255);

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(pulse_color))
                .title(Span::styled(
                    " 🌀 Loki - The Shapeshifter ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                )),
        )
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED),
        )
        .select(selected);

    f.render_widget(tabs, area);
}

fn draw_enhanced_content(f: &mut Frame<'_>, app: &mut App, area: Rect) {
    match app.state.current_view {
        ViewState::Dashboard => draw_tab_home(f, app, area),
        ViewState::Chat => {
            // Use modular rendering system directly
            use crate::tui::chat::rendering::modular_renderer;
            modular_renderer::render_modular_chat(f, app, area);
        }
        ViewState::Utilities => {
            // Use the utilities manager from app state
            app.state.utilities_manager.render(f, area);
        }
        ViewState::Memory => draw_tab_memory(f, app, area),
        ViewState::Cognitive => draw_tab_cognitive(f, app, area),
        ViewState::Streams => draw_tab_social(f, app, area),
        ViewState::Models => draw_tab_settings(f, app, area),
        ViewState::Collaborative => {
            draw_collaborative_view(f, app, area)
        }
        ViewState::PluginEcosystem => {
            draw_plugin_ecosystem_view(f, app, area)
        }
    }
}

/// Draw collaborative view
fn draw_collaborative_view(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Main content
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    // Header
    let header = Paragraph::new("🤝 Collaborative Features")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // Main content
    let items = vec![
        ListItem::new("💬 Real-time Collaboration Sessions"),
        ListItem::new("🔄 Shared Model Training"),
        ListItem::new("📊 Team Analytics Dashboard"),
        ListItem::new("🎯 Goal Alignment System"),
        ListItem::new("🧠 Collective Intelligence Metrics"),
        ListItem::new("🔗 Cross-Instance Communication"),
        ListItem::new("📝 Shared Knowledge Base"),
        ListItem::new("🚀 Distributed Task Coordination"),
    ];

    let list = List::new(items)
        .block(Block::default()
            .title("Available Features")
            .borders(Borders::ALL))
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_widget(list, chunks[1]);

    // Footer with status
    let footer = Paragraph::new("Press 'c' to connect to collaboration network | 's' to start session")
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);
}

/// Draw plugin ecosystem view
fn draw_plugin_ecosystem_view(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Main content
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    // Header
    let header = Paragraph::new("🔌 Plugin Ecosystem")
        .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // Main content - split into two columns
    let content_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Installed plugins
    let installed_plugins = vec![
        ListItem::new("✓ Core Tools v1.0.0"),
        ListItem::new("✓ Memory Enhancement v0.9.5"),
        ListItem::new("✓ Cognitive Extensions v1.2.0"),
        ListItem::new("✓ Safety Monitor v1.1.0"),
    ];

    let installed_list = List::new(installed_plugins)
        .block(Block::default()
            .title("Installed Plugins")
            .borders(Borders::ALL))
        .style(Style::default().fg(Color::Green));

    f.render_widget(installed_list, content_chunks[0]);

    // Available plugins marketplace
    let available = vec![
        ListItem::new("🎨 Creative Tools Extension"),
        ListItem::new("📈 Advanced Analytics Pack"),
        ListItem::new("🔐 Security Enhancement Suite"),
        ListItem::new("🌐 Web Integration Tools"),
        ListItem::new("🤖 ML Model Extensions"),
        ListItem::new("📝 Documentation Generator"),
        ListItem::new("🎮 Game Development Kit"),
        ListItem::new("🎵 Audio Processing Plugin"),
    ];

    let available_list = List::new(available)
        .block(Block::default()
            .title("Plugin Marketplace")
            .borders(Borders::ALL))
        .style(Style::default().fg(Color::Yellow));

    f.render_widget(available_list, content_chunks[1]);

    // Footer
    let footer = Paragraph::new("Press 'i' to install | 'u' to uninstall | 'r' to reload plugins")
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);
}

fn draw_enhanced_status_bar(f: &mut Frame, _app: &App, area: Rect) {
    let current_time = Instant::now();
    let pulse = (current_time.elapsed().as_secs_f32() * PULSE_SPEED * 0.5).sin().abs();

    // Animated status bar with gradient effect
    let gradient_color = Color::Rgb(
        (64.0 + pulse * 64.0) as u8,
        (64.0 + pulse * 64.0) as u8,
        (64.0 + pulse * 64.0) as u8,
    );

    let status_spans = vec![
        Span::styled("Press ", Style::default().fg(Color::Gray)),
        Span::styled("F1", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(" for help", Style::default().fg(Color::Gray)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled("Tab", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled(" to switch views", Style::default().fg(Color::Gray)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled("Ctrl+1-9", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" for direct navigation", Style::default().fg(Color::Gray)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled("Ctrl+Q", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled(" to quit", Style::default().fg(Color::Gray)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled("⚡ ", Style::default().fg(Color::Cyan)),
        Span::styled(
            "Loki AI Ready",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
    ];

    let status_line = Line::from(status_spans);
    let para = Paragraph::new(status_line)
        .style(Style::default().bg(gradient_color))
        .alignment(Alignment::Center);

    f.render_widget(para, area);
}

fn draw_enhanced_help_overlay(f: &mut Frame, app: &App) {
    let area = centered_rect(85, 85, f.area());
    f.render_widget(Clear, area);

    // Enhanced help view with animations
    draw_enhanced_help_view(f, app, area);
}

pub(crate) fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
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

fn draw_enhanced_help_view(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Enhanced left column with visual improvements
    let left_help_text = vec![
        Line::from(vec![Span::styled(
            "🗣️ Natural Language Commands",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "📊 System Status:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  🖥️ show devices - Display compute devices"),
        Line::from("  📊 monitor gpu - Monitor GPU usage"),
        Line::from("  🌐 show cluster - Display cluster status"),
        Line::from("  🌊 list streams - Show active streams"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "🤖 Model Setup Templates:",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  ⚡ setup lightning fast - FREE local setup"),
        Line::from("  ⚖️ setup balanced pro - Local + API (~$0.10/hr)"),
        Line::from("  💎 setup premium quality - Best models (~$0.50/hr)"),
        Line::from("  🦾 setup research beast - 5-model ensemble (~$1.00/hr)"),
        Line::from("  👨‍💻 setup code master - Code completion (FREE)"),
        Line::from("  ✍️ setup writing pro - Professional writing (~$0.30/hr)"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "🎭 Session Management:",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  📋 show templates - List available setups"),
        Line::from("  🎯 show sessions - List active sessions"),
        Line::from("  💰 show costs - View cost analytics"),
        Line::from("  🛑 stop session <id> - Stop specific session"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "📖 Story Commands:",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  📖 /story analyze - Analyze code as narrative"),
        Line::from("  🎨 /story generate - Generate story-driven code"),
        Line::from("  📝 /story document - Create narrative docs"),
        Line::from("  🧠 /story learn - Learn from patterns"),
        Line::from("  🔍 /story review - Review as plot development"),
    ];

    // Enhanced right column with visual improvements
    let right_help_text = vec![
        Line::from(vec![Span::styled(
            "💻 CLI Commands & Shortcuts",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "🚀 Setup Templates:",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  📋 loki setup list - List all templates"),
        Line::from("  ⚡ loki setup lightning-fast - Quick launch FREE"),
        Line::from("  ⚖️ loki setup balanced-pro - Launch balanced setup"),
        Line::from("  ℹ️ loki setup info <template> - Template details"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "🎭 Session Management:",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  📋 loki session list - List active sessions"),
        Line::from("  ℹ️ loki session info <id> - Session details"),
        Line::from("  🛑 loki session stop <id> - Stop session"),
        Line::from("  💰 loki session costs - Cost analytics"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "⌨️ Keyboard Shortcuts:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  ⭐ Tab - Cycle through views"),
        Line::from("  🔢 1-9, S, P, E, N, H, Q, 0 - Direct navigation"),
        Line::from("  🎯 Ctrl+1-9, Ctrl+S/P/E/N/H/Q/0 - Navigate while typing"),
        Line::from("  ❓ F1 - Toggle this help"),
        Line::from("  🔄 ↑/↓ - Navigate items or command history"),
        Line::from("  ⏎ Enter - Execute command or launch template"),
        Line::from("  🚪 Ctrl+Q - Quit application"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "📊 Multi-Panel Controls:",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )]),
        Line::from("  🧠 F2 - Toggle Context/Reasoning panel"),
        Line::from("  🔧 F3 - Toggle Tool Output panel"),
        Line::from("  🔄 F4 - Toggle Workflow panel"),
        Line::from("  👁️ F5 - Toggle Preview panel"),
        Line::from("  📖 Ctrl+S - Toggle Story visualization panel"),
        Line::from("  🔀 Ctrl+Tab - Cycle panel focus"),
    ];

    let left_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            " 🗣️ TUI Commands ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    let right_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green))
        .title(Span::styled(
            " 💻 CLI & Shortcuts ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));

    let left_para = Paragraph::new(left_help_text).block(left_block).wrap(Wrap { trim: true });

    let right_para = Paragraph::new(right_help_text).block(right_block).wrap(Wrap { trim: true });

    f.render_widget(left_para, chunks[0]);
    f.render_widget(right_para, chunks[1]);
}

// Sub-tab system structures
#[derive(Debug, Clone, PartialEq)]
pub struct SubTab {
    pub name: String,
    pub key: String,
}

#[derive(Debug, Clone)]
pub struct SubTabManager {
    pub tabs: Vec<SubTab>,
    pub current_index: usize,
}

impl SubTabManager {
    pub fn new(tabs: Vec<SubTab>) -> Self {
        Self { tabs, current_index: 0 }
    }

    pub fn next(&mut self) {
        if !self.tabs.is_empty() {
            self.current_index = (self.current_index + 1) % self.tabs.len();
        }
    }

    pub fn previous(&mut self) {
        if !self.tabs.is_empty() {
            self.current_index =
                if self.current_index == 0 { self.tabs.len() - 1 } else { self.current_index - 1 };
        }
    }

    pub fn current_tab(&self) -> Option<&SubTab> {
        self.tabs.get(self.current_index)
    }

    pub fn current_key(&self) -> Option<&str> {
        self.current_tab().map(|t| t.key.as_str())
    }
    
    pub fn set_current_index(&mut self, index: usize) {
        if index < self.tabs.len() {
            self.current_index = index;
        }
    }
}

///
pub fn draw_sub_tab_navigation(f: &mut Frame, sub_tabs: &SubTabManager, area: Rect) {
    let mut spans = Vec::new();

    for (i, tab) in sub_tabs.tabs.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" │ ", Style::default().fg(Color::White)));
        }

        let style = if i == sub_tabs.current_index {
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
        } else {
            Style::default().fg(Color::Gray)
        };

        spans.push(Span::styled(tab.name.clone(), style));
    }

    let nav_line = Line::from(spans);
    let nav_widget = Paragraph::new(nav_line)
        .block(
            Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::White)),
        )
        .alignment(Alignment::Center);

    f.render_widget(nav_widget, area);
}


/// Draw notifications overlay
fn draw_notifications(f: &mut Frame, app: &App) {
    // Clean up expired notifications
    let active_notifications: Vec<_> = app.notifications
        .iter()
        .filter(|n| !n.is_expired())
        .collect();
    
    if active_notifications.is_empty() {
        return;
    }
    
    // Calculate notification area (top-right corner)
    let notification_width = 50;
    let notification_height = active_notifications.len().min(5) as u16 * 3; // Max 5 notifications
    
    let area = Rect {
        x: f.area().width.saturating_sub(notification_width + 2),
        y: 4, // Below the header
        width: notification_width,
        height: notification_height,
    };
    
    // Create notification lines
    let mut lines = Vec::new();
    
    for (i, notification) in active_notifications.iter().take(5).enumerate() {
        if i > 0 {
            lines.push(Line::from("")); // Empty line between notifications
        }
        
        let (icon, color) = match notification.notification_type {
            NotificationType::Info => ("ℹ️", Color::Cyan),
            NotificationType::Success => ("✅", Color::Green),
            NotificationType::Warning => ("⚠️", Color::Yellow),
            NotificationType::Error => ("❌", Color::Red),
            NotificationType::Optimization => ("⚡", Color::Magenta),
        };
        
        // Header line with icon and type
        lines.push(Line::from(vec![
            Span::raw(format!("{} ", icon)),
            Span::styled(
                format!("{:?}", notification.notification_type),
                Style::default().fg(color).add_modifier(Modifier::BOLD)
            ),
        ]));
        
        // Message line (truncate if too long)
        let message = if notification.message.len() > notification_width as usize - 4 {
            format!("{}...", &notification.message[..notification_width as usize - 7])
        } else {
            notification.message.clone()
        };
        lines.push(Line::from(Span::raw(format!("  {}", message))));
    }
    
    // Create the notification widget
    let notifications_widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White))
                .title(Span::styled(
                    " Notifications ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        )
        .wrap(Wrap { trim: true });
    
    // Clear the area first
    f.render_widget(Clear, area);
    
    // Render the notifications
    f.render_widget(notifications_widget, area);
}
