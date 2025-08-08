//! Chat history browser subtab

use std::sync::Arc;
use tokio::sync::RwLock;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};
use anyhow::Result;
use chrono::{DateTime, Local, Timelike};

use super::SubtabController;
use crate::tui::chat::state::ChatState;
use crate::tui::run::AssistantResponseType;
use crate::tui::chat::ui_enhancements::{SmoothScroll, StatusIndicator, Status, MetricsSparkline, ToastManager, ToastType};

/// Search mode for history
#[derive(Debug, Clone, PartialEq)]
enum SearchMode {
    None,
    Query,
    DateRange,
    Author,
}

/// Dialog state for author filter
#[derive(Debug, Clone)]
enum DialogState {
    None,
    AuthorFilter {
        input: String,
        available_authors: Vec<String>,
        selected_index: usize,
    },
}

/// History entry for display
#[derive(Debug, Clone)]
struct HistoryEntry {
    timestamp: DateTime<Local>,
    author: String,
    preview: String,
    message_type: String,
    index: usize,
}

/// Chat history browser tab
pub struct HistoryTab {
    /// Reference to chat state
    state: Arc<RwLock<ChatState>>,
    
    /// Filtered history entries
    history_entries: Vec<HistoryEntry>,
    
    /// Selected entry index
    selected_index: usize,
    
    /// Smooth scrolling
    smooth_scroll: SmoothScroll,
    
    /// Status indicator
    status_indicator: StatusIndicator,
    
    /// Metrics sparkline for message frequency
    message_frequency: MetricsSparkline,
    
    /// Search query
    search_query: String,
    
    /// Search mode
    search_mode: SearchMode,
    
    /// Show full message in preview
    show_full_message: bool,
    
    /// Filter by author
    author_filter: Option<String>,
    
    /// Filter by date (days back)
    days_filter: Option<u32>,
    
    /// Toast notifications
    toast_manager: ToastManager,
    
    /// Loaded message for retrieval
    loaded_message: Option<String>,
    
    /// Full messages cache
    full_messages: Vec<AssistantResponseType>,
    
    /// Dialog state
    dialog_state: DialogState,
}

impl HistoryTab {
    /// Create a new history tab
    pub fn new() -> Self {
        // Create a dummy state for now - will be properly initialized later
        let dummy_state = Arc::new(RwLock::new(ChatState::new(0, "History".to_string())));
        
        Self {
            state: dummy_state,
            history_entries: Vec::new(),
            selected_index: 0,
            smooth_scroll: SmoothScroll::new(),
            status_indicator: StatusIndicator::new(),
            message_frequency: MetricsSparkline::new("Message Frequency".to_string(), 50),
            search_query: String::new(),
            search_mode: SearchMode::None,
            show_full_message: false,
            author_filter: None,
            days_filter: None,
            toast_manager: ToastManager::new(),
            loaded_message: None,
            full_messages: Vec::new(),
            dialog_state: DialogState::None,
        }
    }
    
    /// Set the chat state reference
    pub fn set_state(&mut self, state: Arc<RwLock<ChatState>>) {
        self.state = state;
        self.refresh_history();
    }
    
    /// Get the loaded message if any
    pub fn take_loaded_message(&mut self) -> Option<String> {
        self.loaded_message.take()
    }
    
    /// Navigate up in the history list
    pub fn navigate_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
            self.smooth_scroll.update_position(self.selected_index);
        }
    }
    
    /// Navigate down in the history list
    pub fn navigate_down(&mut self) {
        if self.selected_index < self.history_entries.len().saturating_sub(1) {
            self.selected_index += 1;
            self.smooth_scroll.update_position(self.selected_index);
        }
    }
    
    /// Refresh history entries from state
    fn refresh_history(&mut self) {
        self.history_entries.clear();
        
        // Update status
        self.status_indicator.set_status(Status::Processing, "Loading history...".to_string());
        
        // Get messages from state
        let messages = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let state = self.state.read().await;
                state.messages.clone()
            })
        });
        
        // Cache full messages
        self.full_messages = messages.clone();
        
        // Track message frequency
        let mut hourly_counts = std::collections::HashMap::new();
        
        // Convert messages to history entries
        for (index, msg) in messages.iter().enumerate() {
            if let Some(entry) = self.create_history_entry(msg, index) {
                // Update frequency metrics
                let hour = entry.timestamp.hour();
                *hourly_counts.entry(hour).or_insert(0) += 1;
                
                // Apply filters
                if self.matches_filters(&entry) {
                    self.history_entries.push(entry);
                }
            }
        }
        
        // Update message frequency sparkline
        for hour in 0..24 {
            let count = hourly_counts.get(&hour).copied().unwrap_or(0);
            self.message_frequency.add_point(count as u64);
        }
        
        // Sort by timestamp (newest first)
        self.history_entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        // Update status
        self.status_indicator.set_status(
            Status::Success,
            format!("Loaded {} messages", self.history_entries.len())
        );
    }
    
    /// Create a history entry from a message
    fn create_history_entry(&self, msg: &AssistantResponseType, index: usize) -> Option<HistoryEntry> {
        match msg {
            AssistantResponseType::Message { author, message, timestamp, .. } => {
                let preview = if message.len() > 80 {
                    format!("{}...", &message[..77])
                } else {
                    message.clone()
                };
                
                let timestamp = DateTime::parse_from_rfc3339(timestamp)
                    .ok()?
                    .with_timezone(&Local);
                
                Some(HistoryEntry {
                    timestamp,
                    author: author.clone(),
                    preview,
                    message_type: "message".to_string(),
                    index,
                })
            }
            AssistantResponseType::Error { error_type, message, timestamp, .. } => {
                let preview = format!("Error: {}", message);
                let timestamp = DateTime::parse_from_rfc3339(timestamp)
                    .ok()?
                    .with_timezone(&Local);
                
                Some(HistoryEntry {
                    timestamp,
                    author: "System".to_string(),
                    preview,
                    message_type: error_type.clone(),
                    index,
                })
            }
            _ => None,
        }
    }
    
    /// Load the selected message into the chat
    fn load_selected_message(&mut self) {
        if let Some(entry) = self.history_entries.get(self.selected_index) {
            // Get the full message from the cache
            if let Some(full_msg) = self.full_messages.get(entry.index) {
                match full_msg {
                    AssistantResponseType::Message { message, author, .. } => {
                        // Only load user messages into the input
                        if author == "user" || author == "User" {
                            // Store the message for the chat tab to retrieve
                            self.loaded_message = Some(message.clone());
                            
                            self.toast_manager.add_toast(
                                "Message loaded! Switch to Chat tab to see it.".to_string(),
                                ToastType::Success
                            );
                        } else {
                            // For AI messages, copy to clipboard if possible
                            self.toast_manager.add_toast(
                                format!("AI message from {} (view-only)", author),
                                ToastType::Info
                            );
                        }
                    }
                    _ => {
                        self.toast_manager.add_toast(
                            "Cannot load this message type".to_string(),
                            ToastType::Warning
                        );
                    }
                }
            }
        }
    }
    
    /// Check if entry matches current filters
    fn matches_filters(&self, entry: &HistoryEntry) -> bool {
        // Search query filter
        if !self.search_query.is_empty() {
            let query = self.search_query.to_lowercase();
            if !entry.preview.to_lowercase().contains(&query) &&
               !entry.author.to_lowercase().contains(&query) {
                return false;
            }
        }
        
        // Author filter
        if let Some(author) = &self.author_filter {
            if !entry.author.eq_ignore_ascii_case(author) {
                return false;
            }
        }
        
        // Date filter
        if let Some(days) = self.days_filter {
            let cutoff = Local::now() - chrono::Duration::days(days as i64);
            if entry.timestamp < cutoff {
                return false;
            }
        }
        
        true
    }
    
    /// Get the selected message
    fn get_selected_message(&self) -> Option<AssistantResponseType> {
        if let Some(entry) = self.history_entries.get(self.selected_index) {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let state = self.state.read().await;
                    state.messages.get(entry.index).cloned()
                })
            })
        } else {
            None
        }
    }
    
    /// Open the author filter dialog
    fn open_author_filter_dialog(&mut self) {
        // Collect unique authors from messages
        let messages = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let state = self.state.read().await;
                state.messages.clone()
            })
        });
        
        let mut authors = std::collections::HashSet::new();
        for msg in &messages {
            match msg {
                AssistantResponseType::Message { author, .. } => {
                    authors.insert(author.clone());
                }
                AssistantResponseType::UserMessage { .. } => {
                    authors.insert("User".to_string());
                }
                AssistantResponseType::SystemMessage { .. } => {
                    authors.insert("System".to_string());
                }
                _ => {}
            }
        }
        
        let mut available_authors: Vec<String> = authors.into_iter().collect();
        available_authors.sort();
        
        // Add an "All" option at the beginning
        available_authors.insert(0, "All".to_string());
        
        self.dialog_state = DialogState::AuthorFilter {
            input: String::new(),
            available_authors,
            selected_index: 0,
        };
    }
    
    /// Update author filter list based on input
    fn update_author_filter_list(&mut self) {
        if let DialogState::AuthorFilter { ref input, ref mut available_authors, ref mut selected_index } = &mut self.dialog_state {
            // Re-collect all authors
            let messages = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let state = self.state.read().await;
                    state.messages.clone()
                })
            });
            
            let mut authors = std::collections::HashSet::new();
            authors.insert("All".to_string());
            
            for msg in &messages {
                match msg {
                    AssistantResponseType::Message { author, .. } => {
                        authors.insert(author.clone());
                    }
                    AssistantResponseType::UserMessage { .. } => {
                        authors.insert("User".to_string());
                    }
                    AssistantResponseType::SystemMessage { .. } => {
                        authors.insert("System".to_string());
                    }
                    _ => {}
                }
            }
            
            // Filter based on input
            let filter_text = input.to_lowercase();
            let mut filtered: Vec<String> = authors
                .into_iter()
                .filter(|author| author.to_lowercase().contains(&filter_text))
                .collect();
            filtered.sort();
            
            *available_authors = filtered;
            *selected_index = (*selected_index).min(available_authors.len().saturating_sub(1));
        }
    }
    
    /// Render the author filter dialog
    fn render_author_filter_dialog(&self, f: &mut Frame, area: Rect, input: &str, authors: &[String], selected_index: usize) {
        // Calculate dialog size
        let dialog_width = 40;
        let dialog_height = (authors.len() + 4).min(20) as u16; // +4 for borders and input
        
        let dialog_x = area.x + (area.width.saturating_sub(dialog_width)) / 2;
        let dialog_y = area.y + (area.height.saturating_sub(dialog_height)) / 2;
        
        let dialog_area = Rect {
            x: dialog_x,
            y: dialog_y,
            width: dialog_width,
            height: dialog_height,
        };
        
        // Clear background
        let clear_block = Block::default()
            .style(Style::default().bg(Color::Black));
        f.render_widget(clear_block, dialog_area);
        
        // Dialog layout
        let dialog_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Input field
                Constraint::Min(5),    // Author list
            ])
            .split(dialog_area);
        
        // Input field
        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Filter Authors ")
            .style(Style::default().fg(Color::Cyan));
        
        let input_widget = Paragraph::new(input)
            .style(Style::default().fg(Color::White))
            .block(input_block);
        f.render_widget(input_widget, dialog_chunks[0]);
        
        // Author list
        let list_items: Vec<ListItem> = authors
            .iter()
            .enumerate()
            .map(|(i, author)| {
                let style = if i == selected_index {
                    Style::default().bg(Color::DarkGray).fg(Color::White).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                };
                ListItem::new(author.as_str()).style(style)
            })
            .collect();
        
        let list_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Available Authors ")
            .style(Style::default().fg(Color::Cyan));
        
        let list = List::new(list_items)
            .block(list_block)
            .highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White));
        
        f.render_widget(list, dialog_chunks[1]);
        
        // Help text
        let help = Line::from(vec![
            Span::styled(" ↑↓ ", Style::default().fg(Color::Yellow)),
            Span::raw("Navigate  "),
            Span::styled(" Enter ", Style::default().fg(Color::Yellow)),
            Span::raw("Select  "),
            Span::styled(" Esc ", Style::default().fg(Color::Yellow)),
            Span::raw("Cancel"),
        ]);
        
        let help_y = dialog_y + dialog_height - 1;
        f.render_widget(help, Rect { x: dialog_x + 2, y: help_y, width: dialog_width - 4, height: 1 });
    }
}

impl SubtabController for HistoryTab {
    fn render(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),   // Title and status
                Constraint::Length(3),   // Search bar
                Constraint::Min(10),     // History list
                Constraint::Length(8),   // Preview pane
                Constraint::Length(4),   // Metrics
            ])
            .split(area);
        
        // Title with stats and status
        let title_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),     // Title
                Constraint::Length(40),  // Status
            ])
            .split(chunks[0]);
            
        let title_text = format!(
            "📜 Chat History - {} messages{}{}",
            self.history_entries.len(),
            if let Some(author) = &self.author_filter {
                format!(" | Author: {}", author)
            } else {
                String::new()
            },
            if let Some(days) = self.days_filter {
                format!(" | Last {} days", days)
            } else {
                String::new()
            }
        );
        let title = Paragraph::new(title_text)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center);
        f.render_widget(title, title_chunks[0]);
        
        // Status indicator
        let status_line = self.status_indicator.render();
        let status = Paragraph::new(status_line)
            .alignment(Alignment::Right);
        f.render_widget(status, title_chunks[1]);
        
        // Search bar
        let search_style = match self.search_mode {
            SearchMode::Query => Style::default().fg(Color::Yellow),
            _ => Style::default().fg(Color::DarkGray),
        };
        let search_bar = Paragraph::new(self.search_query.as_str())
            .style(search_style)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Search (/ to search, Esc to clear) ")
                .border_style(if self.search_mode == SearchMode::Query {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default()
                }));
        f.render_widget(search_bar, chunks[1]);
        
        // History list with smooth scrolling
        let scroll_offset = self.smooth_scroll.current_offset();
        let visible_height = chunks[2].height.saturating_sub(2) as usize;
        
        let items: Vec<ListItem> = self.history_entries
            .iter()
            .enumerate()
            .skip(scroll_offset)
            .take(visible_height)
            .map(|(i, entry)| {
                let selected = i == self.selected_index;
                let style = if selected {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                
                let content = vec![
                    Line::from(vec![
                        Span::styled(
                            entry.timestamp.format("%Y-%m-%d %H:%M").to_string(),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::raw(" "),
                        Span::styled(
                            format!("[{}]", entry.author),
                            Style::default().fg(Color::Cyan),
                        ),
                    ]),
                    Line::from(entry.preview.clone()),
                ];
                
                ListItem::new(content).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Messages (↑/↓ navigate, Enter to view, d for date filter, a for author filter) "));
        f.render_widget(list, chunks[2]);
        
        // Preview pane
        if self.show_full_message {
            if let Some(msg) = self.get_selected_message() {
                let preview_text = match &msg {
                    AssistantResponseType::Message { message, .. } => message.clone(),
                    AssistantResponseType::Error { message, .. } => message.clone(),
                    _ => "No preview available".to_string(),
                };
                
                let preview = Paragraph::new(preview_text)
                    .wrap(Wrap { trim: true })
                    .block(Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" Full Message (Space to toggle) "));
                f.render_widget(preview, chunks[3]);
            }
        } else {
            let help = Paragraph::new(vec![
                Line::from("Keyboard shortcuts:"),
                Line::from("  / - Search messages"),
                Line::from("  a - Filter by author"),
                Line::from("  d - Filter by date range"),
                Line::from("  Enter - Load user message into chat"),
                Line::from("  Space - Toggle full message preview"),
                Line::from("  Esc - Clear search/filters"),
            ])
            .block(Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Help "));
            f.render_widget(help, chunks[3]);
        }
        
        // Metrics sparkline
        self.message_frequency.render(f, chunks[4]);
        
        // Toast notifications (overlay in top-right corner)
        self.toast_manager.update();
        let toast_area = Rect {
            x: area.x + area.width.saturating_sub(50),
            y: area.y + 1,
            width: 48.min(area.width.saturating_sub(4)),
            height: 5.min(area.height.saturating_sub(2)),
        };
        self.toast_manager.render(f, toast_area);
        
        // Render dialog if active
        if let DialogState::AuthorFilter { ref input, ref available_authors, ref selected_index } = &self.dialog_state {
            self.render_author_filter_dialog(f, area, input, available_authors, *selected_index);
        }
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        // Handle dialog input first
        if let DialogState::AuthorFilter { ref mut input, ref available_authors, ref mut selected_index } = &mut self.dialog_state {
            match key.code {
                KeyCode::Esc => {
                    self.dialog_state = DialogState::None;
                }
                KeyCode::Enter => {
                    // Apply the selected author filter
                    if *selected_index < available_authors.len() {
                        let selected_author = &available_authors[*selected_index];
                        if selected_author == "All" {
                            self.author_filter = None;
                            self.toast_manager.add_toast(
                                "Showing all authors".to_string(),
                                ToastType::Success
                            );
                        } else {
                            self.author_filter = Some(selected_author.clone());
                            self.toast_manager.add_toast(
                                format!("Filtering by author: {}", selected_author),
                                ToastType::Success
                            );
                        }
                        self.refresh_history();
                    }
                    self.dialog_state = DialogState::None;
                }
                KeyCode::Up => {
                    if *selected_index > 0 {
                        *selected_index -= 1;
                    }
                }
                KeyCode::Down => {
                    if *selected_index < available_authors.len().saturating_sub(1) {
                        *selected_index += 1;
                    }
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    // Update filtered authors list based on input
                    self.update_author_filter_list();
                }
                KeyCode::Backspace => {
                    input.pop();
                    self.update_author_filter_list();
                }
                _ => {}
            }
            return Ok(());
        }
        
        match self.search_mode {
            SearchMode::Query => {
                match key.code {
                    KeyCode::Esc => {
                        self.search_mode = SearchMode::None;
                        self.search_query.clear();
                        self.refresh_history();
                    }
                    KeyCode::Enter => {
                        self.search_mode = SearchMode::None;
                        self.refresh_history();
                    }
                    KeyCode::Char(c) => {
                        self.search_query.push(c);
                        self.refresh_history();
                    }
                    KeyCode::Backspace => {
                        self.search_query.pop();
                        self.refresh_history();
                    }
                    _ => {}
                }
            }
            _ => {
                match key.code {
                    KeyCode::Char('/') => {
                        self.search_mode = SearchMode::Query;
                    }
                    KeyCode::Char('a') => {
                        // Open author filter dialog
                        self.open_author_filter_dialog();
                    }
                    KeyCode::Char('d') => {
                        // Simple date filter - toggle between 1, 7, 30 days, and all
                        self.days_filter = match self.days_filter {
                            None => Some(1),
                            Some(1) => Some(7),
                            Some(7) => Some(30),
                            Some(_) => None,
                        };
                        self.refresh_history();
                    }
                    KeyCode::Up => {
                        if self.selected_index > 0 {
                            self.selected_index -= 1;
                            // Smooth scroll to keep selection in view
                            let visible_start = self.smooth_scroll.current_offset();
                            if self.selected_index < visible_start {
                                self.smooth_scroll.scroll_to(self.selected_index);
                            }
                        }
                    }
                    KeyCode::Down => {
                        if self.selected_index < self.history_entries.len().saturating_sub(1) {
                            self.selected_index += 1;
                            // Smooth scroll to keep selection in view
                            let visible_height = 10; // Approximate visible items
                            let visible_end = self.smooth_scroll.current_offset() + visible_height;
                            if self.selected_index >= visible_end {
                                self.smooth_scroll.scroll_to(self.selected_index.saturating_sub(visible_height - 1));
                            }
                        }
                    }
                    KeyCode::PageUp => {
                        self.selected_index = self.selected_index.saturating_sub(10);
                        self.smooth_scroll.scroll_to(self.selected_index);
                    }
                    KeyCode::PageDown => {
                        self.selected_index = (self.selected_index + 10)
                            .min(self.history_entries.len().saturating_sub(1));
                        self.smooth_scroll.scroll_to(self.selected_index.saturating_sub(5));
                    }
                    KeyCode::Char(' ') => {
                        self.show_full_message = !self.show_full_message;
                    }
                    KeyCode::Enter => {
                        self.load_selected_message();
                    }
                    KeyCode::Esc => {
                        self.search_query.clear();
                        self.author_filter = None;
                        self.days_filter = None;
                        self.refresh_history();
                    }
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    fn update(&mut self) -> Result<()> {
        // Periodically refresh if needed
        Ok(())
    }
    
    fn name(&self) -> &str {
        "History"
    }
}