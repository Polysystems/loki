//! Reusable UI components for utilities tabs

pub mod command_palette;
pub mod search_overlay;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Gauge, List, ListItem};

pub use command_palette::CommandPalette;
pub use search_overlay::{SearchOverlay, SearchResult};

/// Render a status indicator with icon and text
pub fn render_status_indicator(status: &str, active: bool) -> Span<'static> {
    let (icon, color) = if active {
        ("🟢", Color::Green)
    } else {
        ("🔴", Color::Red)
    };
    
    Span::styled(
        format!("{} {}", icon, status),
        Style::default().fg(color)
    )
}

/// Render a progress gauge
pub fn render_progress_gauge(f: &mut Frame, area: Rect, title: &str, percent: u16, color: Color) {
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .gauge_style(Style::default().fg(color))
        .percent(percent)
        .label(format!("{}%", percent));
    
    f.render_widget(gauge, area);
}

/// Render a key-value info panel
pub fn render_info_panel(
    f: &mut Frame,
    area: Rect,
    title: &str,
    items: Vec<(&str, String, Color)>,
) {
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Wrap;
    
    let lines: Vec<Line> = items
        .into_iter()
        .map(|(label, value, color)| {
            Line::from(vec![
                Span::styled(format!("{}: ", label), Style::default().fg(Color::Cyan)),
                Span::styled(value, Style::default().fg(color)),
            ])
        })
        .collect();
    
    let panel = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: true });
    
    f.render_widget(panel, area);
}

/// Render a simple list with highlighting
pub fn render_simple_list(
    f: &mut Frame,
    area: Rect,
    title: &str,
    items: Vec<String>,
    selected: Option<usize>,
) {
    let list_items: Vec<ListItem> = items
        .into_iter()
        .map(|item| ListItem::new(item))
        .collect();
    
    let mut list = List::new(list_items)
        .block(Block::default().borders(Borders::ALL).title(title));
    
    if selected.is_some() {
        list = list
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
            .highlight_symbol("► ");
    }
    
    f.render_widget(list, area);
}

/// Format bytes to human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Format duration to human readable string
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    let days = total_seconds / 86400;
    let hours = (total_seconds % 86400) / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    
    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Create a colored text based on threshold
pub fn colored_by_threshold(value: f32, low: f32, high: f32) -> Color {
    if value < low {
        Color::Green
    } else if value < high {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Render a header with connection status
pub fn render_connection_header(
    f: &mut Frame,
    area: Rect,
    title: &str,
    connected: bool,
    stats: Option<(usize, usize)>, // (total, active)
) {
    let status = if connected {
        "🟢 Connected"
    } else {
        "🟡 Demo Mode"
    };
    
    let header_text = if let Some((total, active)) = stats {
        format!("{} - {} total ({} active) | {}", title, total, active, status)
    } else {
        format!("{} | {}", title, status)
    };
    
    let header = Paragraph::new(header_text)
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    
    f.render_widget(header, area);
}

/// Standard control footer text for different view modes
pub fn get_standard_controls(mode: &str) -> &'static str {
    match mode {
        "list" => "↑↓/jk: Navigate | Enter: Select | Tab: Switch View | q: Back",
        "details" => "b: Back | Tab: Switch View | q: Back",
        "edit" => "Ctrl+S: Save | Esc: Cancel | Type to edit",
        _ => "q: Back",
    }
}