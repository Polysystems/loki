use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

use crate::tui::app::App;
use crate::tui::ui::draw_sub_tab_navigation;

#[derive(Debug, Clone)]
pub struct SettingItem {
    pub name: String,
    pub value: SettingValue,
    pub choices: Option<Vec<String>>, // for multiple choice
}

#[derive(Debug, Clone)]
pub enum SettingValue {
    Bool(bool),
    Choice(String),
}

#[derive(Debug, Clone)]
pub struct SettingsUI {
    pub items: Vec<SettingItem>,
    pub selected: usize,
    pub page: usize,
}

impl SettingsUI {
    pub fn new(items: Vec<SettingItem>) -> SettingsUI {
        Self { items, selected: 0, page: 0 }
    }

    pub fn page_count(&self) -> usize {
        (self.items.len() + 9) / 10
    }

    pub fn current_page_items(&self) -> &[SettingItem] {
        let start = self.page * 10;
        let end = (start + 10).min(self.items.len());
        &self.items[start..end]
    }
}

pub fn draw_tab_settings(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Sub-tab navigation
            Constraint::Min(0),     // Content
        ])
        .split(area);

    // Draw sub-tab navigation
    draw_sub_tab_navigation(f, &app.state.settings_tabs, chunks[0]);

    // Draw content based on current sub-tab
    match app.state.settings_tabs.current_key() {
        Some("general") => draw_general_settings(f, app, chunks[1]),
        Some("configuration") => draw_configuration_management(f, app, chunks[1]),
        Some("safety") => draw_safety_management(f, app, chunks[1]),
        _ => draw_general_settings(f, app, chunks[1]),
    }
}

fn draw_general_settings(f: &mut Frame, app: &App, area: Rect) {
    let settings = &app.state.settings_ui;
    let items = settings.current_page_items();

    let lines: Vec<Line> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let idx = i + settings.page * 10;
            let is_selected = idx == settings.selected;

            let indicator = if is_selected { "> " } else { "  " };
            let val_text = match &item.value {
                SettingValue::Bool(v) => format!("{}", if v.clone() { "On" } else { "Off" }),
                SettingValue::Choice(v) => v.clone(),
            };

            Line::from(vec![
                Span::styled(indicator, Style::default().fg(Color::Yellow)),
                Span::styled(
                    &item.name,
                    Style::default().fg(if is_selected { Color::Cyan } else { Color::White }),
                ),
                Span::raw(" = "),
                Span::styled(val_text, Style::default().fg(Color::Magenta)),
            ])
        })
        .collect();

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title("General Settings")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        )
        .wrap(Wrap { trim: true });

    f.render_widget(widget, area);
}

fn draw_configuration_management(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12), // Configuration overview
            Constraint::Min(0),     // Configuration details
        ])
        .split(area);

    // Configuration Overview
    let overview_lines = vec![
        Line::from(vec![
            Span::styled("⚙️ Configuration Management", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Config File: "),
            Span::styled("~/.config/loki/config.toml", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("Environment: "),
            Span::styled("Production", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Last Modified: "),
            Span::styled("2 hours ago", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Active Profiles: "),
            Span::styled("3", Style::default().fg(Color::Blue)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Auto-reload: "),
            Span::styled("Enabled", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Validation: "),
            Span::styled("✓ Passed", Style::default().fg(Color::Green)),
        ]),
    ];

    let overview_widget = Paragraph::new(overview_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title(Span::styled(
                    " Overview ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(overview_widget, chunks[0]);

    // Configuration Categories
    let config_categories = vec![
        ListItem::new(Line::from(vec![
            Span::styled("🧠 ", Style::default().fg(Color::Cyan)),
            Span::styled("Cognitive", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - AI system settings"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🔧 ", Style::default().fg(Color::Yellow)),
            Span::styled("Tools", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - External tool configurations"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🌐 ", Style::default().fg(Color::Green)),
            Span::styled("Network", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - API endpoints and connections"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("💾 ", Style::default().fg(Color::Blue)),
            Span::styled("Storage", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - Database and file paths"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🔒 ", Style::default().fg(Color::Red)),
            Span::styled("Security", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - Authentication and permissions"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("📊 ", Style::default().fg(Color::Magenta)),
            Span::styled("Monitoring", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - Logging and metrics"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🎨 ", Style::default().fg(Color::Cyan)),
            Span::styled("Interface", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(" - UI and TUI preferences"),
        ])),
    ];

    let categories_list = List::new(config_categories)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue))
                .title(Span::styled(
                    " Configuration Categories ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(categories_list, chunks[1]);
}

fn draw_safety_management(f: &mut Frame, _app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Safety status
            Constraint::Min(0),     // Safety rules
        ])
        .split(chunks[0]);

    // Safety Status Panel
    let status_lines = vec![
        Line::from(vec![
            Span::styled("🛡️ Safety System Status", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            Span::styled("● Active", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::raw("Validators: "),
            Span::styled("5/5 Online", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Pending Actions: "),
            Span::styled("3", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::raw("Safety Score: "),
            Span::styled("98.5%", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::raw("Last Incident: "),
            Span::styled("None", Style::default().fg(Color::Green)),
        ]),
    ];

    let status_widget = Paragraph::new(status_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " Status ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(status_widget, left_chunks[0]);

    // Safety Rules
    let rules = vec![
        ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Resource limits enforced"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Action validation enabled"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Anomaly detection active"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Audit logging enabled"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("⚠️ ", Style::default().fg(Color::Yellow)),
            Span::raw("Rate limiting: Medium"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("Sandboxing enabled"),
        ])),
    ];

    let rules_list = List::new(rules)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " Active Rules ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(rules_list, left_chunks[1]);

    // Recent Safety Events (Right side)
    let events = vec![
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:56]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("ALLOWED", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": File write to /tmp/cache"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:55]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("BLOCKED", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(": Excessive memory allocation"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:54]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("WARNING", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(": High CPU usage detected"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:53]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("ALLOWED", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(": API call to approved endpoint"),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("[12:34:52]", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("REVIEW", Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)),
            Span::raw(": New tool registration pending"),
        ])),
    ];

    let events_list = List::new(events)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " Recent Events ",
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ))
        );

    f.render_widget(events_list, chunks[1]);
}