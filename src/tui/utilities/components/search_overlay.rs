//! Search overlay component for utilities

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};

/// Search overlay component
pub struct SearchOverlay {
    /// Search query
    query: String,
    
    /// Whether the overlay is active
    is_active: bool,
    
    /// Search results
    results: Vec<SearchResult>,
    
    /// Selected result index
    selected_index: usize,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: Option<String>,
    pub score: f32,
}

impl SearchOverlay {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            is_active: false,
            results: Vec::new(),
            selected_index: 0,
        }
    }
    
    /// Activate the search overlay
    pub fn activate(&mut self) {
        self.is_active = true;
        self.query.clear();
        self.results.clear();
        self.selected_index = 0;
    }
    
    /// Deactivate the search overlay
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }
    
    /// Check if overlay is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }
    
    /// Update search query
    pub fn update_query(&mut self, query: String) {
        self.query = query;
    }
    
    /// Add character to query
    pub fn input_char(&mut self, c: char) {
        self.query.push(c);
    }
    
    /// Remove last character
    pub fn delete_char(&mut self) {
        self.query.pop();
    }
    
    /// Get current query
    pub fn get_query(&self) -> String {
        self.query.clone()
    }
    
    /// Set search results
    pub fn set_results(&mut self, results: Vec<SearchResult>) {
        self.results = results;
        self.selected_index = 0;
    }
    
    /// Move selection up
    pub fn previous(&mut self) {
        if !self.results.is_empty() {
            if self.selected_index == 0 {
                self.selected_index = self.results.len() - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }
    
    /// Move selection down
    pub fn next(&mut self) {
        if !self.results.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.results.len();
        }
    }
    
    /// Clear search results
    pub fn clear_results(&mut self) {
        self.results.clear();
        self.selected_index = 0;
    }
    
    /// Get selected index
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }
    
    /// Get the selected search result
    pub fn get_selected_result(&self) -> Option<SearchResult> {
        self.results.get(self.selected_index).cloned()
    }
    
    /// Render the search overlay
    pub fn render(&self, f: &mut Frame, area: Rect) {
        if !self.is_active {
            return;
        }
        
        // Calculate overlay area (centered, 60% width, 40% height)
        let overlay_area = centered_rect(60, 40, area);
        
        // Clear the area
        f.render_widget(Clear, overlay_area);
        
        // Split into search bar and results
        use ratatui::layout::{Constraint, Direction, Layout};
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Search bar
                Constraint::Min(0),     // Results
            ])
            .split(overlay_area);
        
        // Search bar
        let search_text = Paragraph::new(format!("🔍 {}", self.query))
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title("Search"))
            .style(Style::default().fg(Color::White));
        
        f.render_widget(search_text, chunks[0]);
        
        // Results list
        if self.results.is_empty() && !self.query.is_empty() {
            let no_results = Paragraph::new("No results found")
                .block(Block::default().borders(Borders::ALL))
                .alignment(Alignment::Center)
                .style(Style::default().fg(Color::DarkGray));
            
            f.render_widget(no_results, chunks[1]);
        } else {
            let items: Vec<ListItem> = self.results
                .iter()
                .enumerate()
                .map(|(i, result)| {
                    let mut content = vec![];
                    
                    if let Some(ref cat) = result.category {
                        content.push(Span::styled(
                            format!("[{}] ", cat),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    
                    content.push(Span::styled(
                        &result.title,
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                    ));
                    
                    if !result.description.is_empty() {
                        content.push(Span::raw(" - "));
                        content.push(Span::styled(&result.description, Style::default().fg(Color::Gray)));
                    }
                    
                    let style = if i == self.selected_index {
                        Style::default().bg(Color::DarkGray)
                    } else {
                        Style::default()
                    };
                    
                    ListItem::new(Line::from(content)).style(style)
                })
                .collect();
            
            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Results"));
            
            f.render_widget(list, chunks[1]);
        }
    }
}

/// Helper function to create centered rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    use ratatui::layout::{Constraint, Direction, Layout};
    
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