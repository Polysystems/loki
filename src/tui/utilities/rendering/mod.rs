//! Rendering utilities for the utilities module

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, List, ListItem, Table, Row, Cell};

/// Render a split view with two panels
pub fn render_split_view(
    f: &mut Frame,
    area: Rect,
    left_ratio: u16,
    render_left: impl FnOnce(&mut Frame, Rect),
    render_right: impl FnOnce(&mut Frame, Rect),
) {
    use ratatui::layout::{Constraint, Direction, Layout};
    
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(left_ratio),
            Constraint::Percentage(100 - left_ratio),
        ])
        .split(area);
    
    render_left(f, chunks[0]);
    render_right(f, chunks[1]);
}

/// Render a three-panel layout
pub fn render_three_panel(
    f: &mut Frame,
    area: Rect,
    render_top: impl FnOnce(&mut Frame, Rect),
    render_middle: impl FnOnce(&mut Frame, Rect),
    render_bottom: impl FnOnce(&mut Frame, Rect),
) {
    use ratatui::layout::{Constraint, Direction, Layout};
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(0),     // Content
            Constraint::Length(3),  // Footer
        ])
        .split(area);
    
    render_top(f, chunks[0]);
    render_middle(f, chunks[1]);
    render_bottom(f, chunks[2]);
}

/// Render a status badge
pub fn render_status_badge(status: &str, color: Color) -> Span<'static> {
    Span::styled(
        format!(" {} ", status),
        Style::default()
            .fg(Color::Black)
            .bg(color)
            .add_modifier(Modifier::BOLD),
    )
}

/// Render a key-value table
pub fn render_kv_table(
    f: &mut Frame,
    area: Rect,
    title: &str,
    items: Vec<(&str, String, Color)>,
) {
    let rows: Vec<Row> = items
        .into_iter()
        .map(|(key, value, color)| {
            Row::new(vec![
                Cell::from(key).style(Style::default().fg(Color::Cyan)),
                Cell::from(value).style(Style::default().fg(color)),
            ])
        })
        .collect();
    
    let widths = [Constraint::Length(20), Constraint::Min(20)];
    
    let table = Table::new(rows, widths)
        .block(Block::default().borders(Borders::ALL).title(title));
    
    f.render_widget(table, area);
}

/// Render a loading spinner
pub fn render_loading(f: &mut Frame, area: Rect, message: &str) {
    let loading = Paragraph::new(format!("⌛ {}", message))
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::SLOW_BLINK));
    
    f.render_widget(loading, area);
}

/// Render an error message
pub fn render_error(f: &mut Frame, area: Rect, error: &str) {
    let error_widget = Paragraph::new(format!("❌ Error: {}", error))
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Red)))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Red))
        .wrap(ratatui::widgets::Wrap { trim: true });
    
    f.render_widget(error_widget, area);
}

/// Render a success message
pub fn render_success(f: &mut Frame, area: Rect, message: &str) {
    let success_widget = Paragraph::new(format!("✅ {}", message))
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Green));
    
    f.render_widget(success_widget, area);
}

/// Create a styled list with icons
pub fn create_icon_list(items: Vec<(&str, &str, Color)>) -> List<'static> {
    let list_items: Vec<ListItem> = items
        .into_iter()
        .map(|(icon, text, color)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", icon), Style::default().fg(color)),
                Span::raw(text.to_string()),
            ]))
        })
        .collect();
    
    List::new(list_items)
}