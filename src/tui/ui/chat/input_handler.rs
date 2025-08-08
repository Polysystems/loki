//! Advanced input handling for the chat interface
//! 
//! Provides multiline editing, command completion, history navigation,
//! and smart suggestions.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use std::collections::VecDeque;
use tokio::sync::mpsc;

/// Input handler with advanced features
#[derive(Clone)]
pub struct InputHandler {
    /// Reference to the shared command input buffer
    /// This is now managed externally by ChatManager
    
    /// Cursor position in the buffer
    pub cursor_position: usize,
    
    /// Input mode
    mode: InputMode,
    
    /// Command history
    history: InputHistory,
    
    /// Autocomplete engine
    autocomplete: AutocompleteEngine,
    
    /// Current suggestions
    suggestions: Vec<CompletionSuggestion>,
    
    /// Selected suggestion index
    selected_suggestion: Option<usize>,
    
    /// Multi-line editing state
    multiline_state: MultilineState,
    
    /// Event channel for input events
    event_tx: Option<mpsc::Sender<InputEvent>>,
    
    /// Horizontal scroll offset for long input
    scroll_offset: usize,
}

/// Input modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputMode {
    Normal,
    Command,
    Search,
    Multiline,
}

/// Input history manager
#[derive(Clone)]
struct InputHistory {
    entries: VecDeque<String>,
    max_size: usize,
    current_index: Option<usize>,
    temp_buffer: Option<String>,
}

/// Multiline editing state
#[derive(Clone)]
struct MultilineState {
    lines: Vec<String>,
    current_line: usize,
    column: usize,
}

/// Autocomplete engine
#[derive(Clone)]
struct AutocompleteEngine {
    /// Command completions
    commands: Vec<CommandCompletion>,
    
    /// Context-aware completions
    context_completions: Vec<String>,
    
    /// Recently used completions
    recent_completions: VecDeque<String>,
    
    /// Fuzzy matching enabled
    fuzzy_match: bool,
}

/// Command completion definition
#[derive(Debug, Clone)]
struct CommandCompletion {
    trigger: String,
    completions: Vec<String>,
    description: String,
}

/// Completion suggestion
#[derive(Debug, Clone)]
pub struct CompletionSuggestion {
    pub text: String,
    pub display: String,
    pub description: String,
    pub score: f32,
    pub category: SuggestionCategory,
}

/// Suggestion categories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SuggestionCategory {
    Command,
    Tool,
    File,
    History,
    Context,
    Emoji,
}

/// Input events
#[derive(Debug, Clone)]
pub enum InputEvent {
    Submit(String),
    CommandExecute(String, Vec<String>),
    SearchQuery(String),
    Cancel,
}

impl InputHandler {
    pub fn new() -> Self {
        Self {
            cursor_position: 0,
            mode: InputMode::Normal,
            history: InputHistory::new(1000),
            autocomplete: AutocompleteEngine::new(),
            suggestions: Vec::new(),
            selected_suggestion: None,
            multiline_state: MultilineState::new(),
            event_tx: None,
            scroll_offset: 0,
        }
    }
    
    /// Set event channel
    pub fn set_event_channel(&mut self, tx: mpsc::Sender<InputEvent>) {
        self.event_tx = Some(tx);
    }
    
    /// Handle keyboard input with external buffer
    pub async fn handle_key(&mut self, key: KeyEvent, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        tracing::debug!("InputHandler handling key: {:?}, current buffer: '{}', cursor: {}", key, buffer, self.cursor_position);
        
        match self.mode {
            InputMode::Normal => self.handle_normal_mode(key, buffer).await?,
            InputMode::Command => self.handle_command_mode(key, buffer).await?,
            InputMode::Search => self.handle_search_mode(key, buffer).await?,
            InputMode::Multiline => self.handle_multiline_mode(key, buffer).await?,
        }
        
        // Update suggestions after input change
        self.update_suggestions(buffer);
        
        tracing::debug!("After handling key, buffer: '{}', cursor: {}", buffer, self.cursor_position);
        
        Ok(())
    }
    
    /// Handle normal mode input
    async fn handle_normal_mode(&mut self, key: KeyEvent, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        match (key.code, key.modifiers) {
            // Text input - also handle empty() which is equivalent to NONE
            (KeyCode::Char(c), modifiers) if modifiers == KeyModifiers::NONE || 
                                               modifiers == KeyModifiers::SHIFT || 
                                               modifiers.is_empty() => {
                self.insert_char(c, buffer);
            }
            
            // Navigation
            (KeyCode::Left, _) => self.move_cursor_left(buffer),
            (KeyCode::Right, _) => self.move_cursor_right(buffer),
            (KeyCode::Home, _) => self.move_cursor_home(),
            (KeyCode::End, _) => self.move_cursor_end(buffer),
            
            // Word navigation
            //(KeyCode::Left, KeyModifiers::CONTROL) => self.move_cursor_word_left(buffer),
            //(KeyCode::Right, KeyModifiers::CONTROL) => self.move_cursor_word_right(buffer),
            
            // Deletion
            (KeyCode::Backspace, _) => self.delete_char_before_cursor(buffer),
            (KeyCode::Delete, _) => self.delete_char_at_cursor(buffer),
            //(KeyCode::Backspace, KeyModifiers::CONTROL) => self.delete_word_before_cursor(buffer),
            //(KeyCode::Delete, KeyModifiers::CONTROL) => self.delete_word_at_cursor(buffer),

            // History
            (KeyCode::Up, _) => self.history_previous(buffer),
            (KeyCode::Down, _) => self.history_next(buffer),
            
            // Autocomplete
            (KeyCode::Tab, _) => self.trigger_autocomplete(buffer),
           // (KeyCode::Tab, KeyModifiers::SHIFT) => self.autocomplete_previous(),

            // Submit
            (KeyCode::Enter, KeyModifiers::NONE) => {
                self.submit_input(buffer).await?;
            }
            
            // Multiline
            (KeyCode::Enter, KeyModifiers::SHIFT) => {
                self.enter_multiline_mode(buffer);
            }
            
            // Mode switches
            (KeyCode::Char('/'), KeyModifiers::CONTROL) => {
                self.mode = InputMode::Search;
                buffer.clear();
                self.cursor_position = 0;
            }
            
            (KeyCode::Char(':'), KeyModifiers::NONE) if buffer.is_empty() => {
                self.mode = InputMode::Command;
                self.insert_char(':', buffer);
            }
            
            // Clear
            (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.clear_input(buffer);
            }
            
            // Paste (simplified - real implementation would use clipboard)
            (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                // Would integrate with clipboard here
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle command mode input
    async fn handle_command_mode(&mut self, key: KeyEvent, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        match key.code {
            KeyCode::Enter => {
                let command = buffer[1..].to_string(); // Skip the ':'
                self.execute_command(command).await?;
                self.clear_input(buffer);
                self.mode = InputMode::Normal;
            }
            KeyCode::Esc => {
                self.clear_input(buffer);
                self.mode = InputMode::Normal;
            }
            _ => {
                // Delegate to normal mode handler for text input
                self.handle_normal_mode(key, buffer).await?;
            }
        }
        Ok(())
    }
    
    /// Handle search mode input
    async fn handle_search_mode(&mut self, key: KeyEvent, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        match key.code {
            KeyCode::Enter => {
                if let Some(tx) = &self.event_tx {
                    tx.send(InputEvent::SearchQuery(buffer.clone())).await?;
                }
                self.mode = InputMode::Normal;
            }
            KeyCode::Esc => {
                self.clear_input(buffer);
                self.mode = InputMode::Normal;
            }
            _ => {
                self.handle_normal_mode(key, buffer).await?;
            }
        }
        Ok(())
    }
    
    /// Handle multiline mode input
    async fn handle_multiline_mode(&mut self, key: KeyEvent, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        match (key.code, key.modifiers) {
            (KeyCode::Enter, KeyModifiers::CONTROL) => {
                // Submit multiline input
                let content = self.multiline_state.get_content();
                self.submit_multiline(content).await?;
                self.exit_multiline_mode(buffer);
            }
            (KeyCode::Esc, _) => {
                self.exit_multiline_mode(buffer);
            }
            (KeyCode::Enter, KeyModifiers::NONE) => {
                self.multiline_state.new_line();
            }
            (KeyCode::Up, _) => {
                self.multiline_state.move_up();
            }
            (KeyCode::Down, _) => {
                self.multiline_state.move_down();
            }
            _ => {
                // Handle as normal input within current line
                self.handle_normal_mode(key, buffer).await?;
            }
        }
        Ok(())
    }
    
    /// Insert character at cursor
    fn insert_char(&mut self, c: char, buffer: &mut String) {
        // Handle UTF-8 properly: cursor_position is in bytes, not chars
        if self.cursor_position <= buffer.len() {
            buffer.insert(self.cursor_position, c);
            self.cursor_position += c.len_utf8();
        }
    }
    
    /// Move cursor left
    fn move_cursor_left(&mut self, buffer: &String) {
        if self.cursor_position > 0 {
            // Move to the previous char boundary
            let mut new_pos = self.cursor_position - 1;
            while new_pos > 0 && !buffer.is_char_boundary(new_pos) {
                new_pos -= 1;
            }
            self.cursor_position = new_pos;
            
            // Adjust scroll if cursor moves out of view
            if self.cursor_position < self.scroll_offset {
                self.scroll_offset = self.cursor_position;
            }
        }
    }
    
    /// Move cursor right
    fn move_cursor_right(&mut self, buffer: &String) {
        if self.cursor_position < buffer.len() {
            // Move to the next char boundary
            let mut new_pos = self.cursor_position + 1;
            while new_pos < buffer.len() && !buffer.is_char_boundary(new_pos) {
                new_pos += 1;
            }
            self.cursor_position = new_pos;
            // Scroll adjustment will be handled in render method based on visible width
        }
    }
    
    /// Move cursor to start
    fn move_cursor_home(&mut self) {
        self.cursor_position = 0;
        self.scroll_offset = 0;
    }
    
    /// Move cursor to end
    fn move_cursor_end(&mut self, buffer: &String) {
        self.cursor_position = buffer.len();
        // Scroll adjustment will be handled in render method based on visible width
    }
    
    /// Move cursor word left
    fn move_cursor_word_left(&mut self, buffer: &String) {
        if self.cursor_position == 0 {
            return;
        }
        
        let bytes = buffer.as_bytes();
        let mut pos = self.cursor_position - 1;
        
        // Skip whitespace
        while pos > 0 && bytes[pos].is_ascii_whitespace() {
            pos -= 1;
        }
        
        // Skip word characters
        while pos > 0 && !bytes[pos - 1].is_ascii_whitespace() {
            pos -= 1;
        }
        
        self.cursor_position = pos;
    }
    
    /// Move cursor word right
    fn move_cursor_word_right(&mut self, buffer: &String) {
        let bytes = buffer.as_bytes();
        let len = bytes.len();
        
        if self.cursor_position >= len {
            return;
        }
        
        let mut pos = self.cursor_position;
        
        // Skip word characters
        while pos < len && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        
        // Skip whitespace
        while pos < len && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        
        self.cursor_position = pos;
    }
    
    /// Delete character before cursor
    fn delete_char_before_cursor(&mut self, buffer: &mut String) {
        if self.cursor_position > 0 && !buffer.is_empty() {
            // Find the char boundary before cursor
            let mut char_boundary = self.cursor_position;
            while char_boundary > 0 && !buffer.is_char_boundary(char_boundary) {
                char_boundary -= 1;
            }
            
            // Find the previous char boundary
            if char_boundary > 0 {
                let prev_char_start = buffer[..char_boundary]
                    .char_indices()
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                buffer.drain(prev_char_start..char_boundary);
                self.cursor_position = prev_char_start;
            }
        }
    }
    
    /// Delete character at cursor
    fn delete_char_at_cursor(&mut self, buffer: &mut String) {
        if self.cursor_position < buffer.len() {
            // Find the next char boundary
            let mut next_boundary = self.cursor_position + 1;
            while next_boundary < buffer.len() && !buffer.is_char_boundary(next_boundary) {
                next_boundary += 1;
            }
            buffer.drain(self.cursor_position..next_boundary);
        }
    }
    
    /// Delete word before cursor
    fn delete_word_before_cursor(&mut self, buffer: &mut String) {
        let start = self.cursor_position;
        self.move_cursor_word_left(buffer);
        buffer.drain(self.cursor_position..start);
    }
    
    /// Delete word at cursor
    fn delete_word_at_cursor(&mut self, buffer: &mut String) {
        let start = self.cursor_position;
        self.move_cursor_word_right(buffer);
        let end = self.cursor_position;
        self.cursor_position = start;
        buffer.drain(start..end);
    }
    
    /// Navigate to previous history entry
    fn history_previous(&mut self, buffer: &mut String) {
        if let Some(prev) = self.history.previous(buffer) {
            *buffer = prev;
            self.cursor_position = buffer.len();
        }
    }
    
    /// Navigate to next history entry
    fn history_next(&mut self, buffer: &mut String) {
        if let Some(next) = self.history.next() {
            *buffer = next;
            self.cursor_position = buffer.len();
        }
    }
    
    /// Trigger autocomplete
    fn trigger_autocomplete(&mut self, buffer: &mut String) {
        if self.suggestions.is_empty() {
            self.update_suggestions(buffer);
        }
        
        if !self.suggestions.is_empty() {
            if let Some(selected) = self.selected_suggestion {
                if selected < self.suggestions.len() {
                    let suggestion = self.suggestions[selected].clone();
                    self.apply_suggestion(&suggestion, buffer);
                }
            } else {
                self.selected_suggestion = Some(0);
            }
        }
    }
    
    /// Navigate to previous autocomplete suggestion
    fn autocomplete_previous(&mut self) {
        if let Some(selected) = self.selected_suggestion {
            if selected > 0 {
                self.selected_suggestion = Some(selected - 1);
            } else {
                self.selected_suggestion = Some(self.suggestions.len() - 1);
            }
        }
    }
    
    /// Apply a suggestion
    fn apply_suggestion(&mut self, suggestion: &CompletionSuggestion, buffer: &mut String) {
        // Find the start of the current word
        let word_start = buffer[..self.cursor_position]
            .rfind(char::is_whitespace)
            .map(|i| i + 1)
            .unwrap_or(0);
        
        // Replace current word with suggestion
        buffer.replace_range(word_start..self.cursor_position, &suggestion.text);
        self.cursor_position = word_start + suggestion.text.len();
        
        // Clear suggestions
        self.suggestions.clear();
        self.selected_suggestion = None;
    }
    
    /// Update autocomplete suggestions
    fn update_suggestions(&mut self, buffer: &String) {
        self.suggestions = self.autocomplete.get_suggestions(buffer, self.cursor_position);
        
        if self.suggestions.is_empty() {
            self.selected_suggestion = None;
        } else if self.selected_suggestion.map_or(true, |i| i >= self.suggestions.len()) {
            self.selected_suggestion = Some(0);
        }
    }
    
    /// Get current input buffer content (for external access)
    pub fn get_current_input(&self, buffer: &String) -> String {
        buffer.clone()
    }
    
    /// Get current cursor position
    pub fn get_cursor_position(&self) -> usize {
        self.cursor_position
    }
    
    /// Set input content
    pub fn set_input(&mut self, input: String) {
        // Note: This method needs external buffer access
        // The actual buffer modification should be done by the caller
        self.cursor_position = input.len();
    }
    
    /// Set cursor position
    pub fn set_cursor_position(&mut self, position: usize) {
        self.cursor_position = position;
    }
    
    /// Focus the input handler
    pub fn focus(&mut self) {
        self.mode = InputMode::Normal;
    }
    
    /// Insert text at cursor position
    pub fn insert_text(&mut self, text: &str, buffer: &mut String) {
        if self.cursor_position <= buffer.len() {
            buffer.insert_str(self.cursor_position, text);
            self.cursor_position += text.len();
        }
    }
    
    /// Submit input
    async fn submit_input(&mut self, buffer: &mut String) -> Result<(), Box<dyn std::error::Error>> {
        if !buffer.is_empty() {
            let input = buffer.clone();
            self.history.add(input.clone());
            
            if let Some(tx) = &self.event_tx {
                tx.send(InputEvent::Submit(input)).await?;
            }
            
            self.clear_input(buffer);
        }
        Ok(())
    }
    
    /// Submit multiline input
    async fn submit_multiline(&mut self, content: String) -> Result<(), Box<dyn std::error::Error>> {
        if !content.is_empty() {
            self.history.add(content.clone());
            
            if let Some(tx) = &self.event_tx {
                tx.send(InputEvent::Submit(content)).await?;
            }
        }
        Ok(())
    }
    
    /// Execute command
    async fn execute_command(&mut self, command: String) -> Result<(), Box<dyn std::error::Error>> {
        let parts: Vec<String> = command.split_whitespace().map(String::from).collect();
        if !parts.is_empty() {
            let cmd = parts[0].clone();
            let args = parts[1..].to_vec();
            
            if let Some(tx) = &self.event_tx {
                tx.send(InputEvent::CommandExecute(cmd, args)).await?;
            }
        }
        Ok(())
    }
    
    /// Clear input
    fn clear_input(&mut self, buffer: &mut String) {
        buffer.clear();
        self.cursor_position = 0;
        self.suggestions.clear();
        self.selected_suggestion = None;
    }
    
    /// Enter multiline mode
    fn enter_multiline_mode(&mut self, buffer: &String) {
        self.mode = InputMode::Multiline;
        self.multiline_state.start_with(buffer);
    }
    
    /// Exit multiline mode
    fn exit_multiline_mode(&mut self, buffer: &mut String) {
        self.mode = InputMode::Normal;
        *buffer = self.multiline_state.get_current_line();
        self.cursor_position = buffer.len();
        self.multiline_state.clear();
    }
    
    /// Render the input area with external buffer
    pub fn render(&mut self, f: &mut Frame, area: Rect, theme: &super::theme_engine::ChatTheme, buffer: &String) {
        let input_style = match self.mode {
            InputMode::Normal => theme.component_styles.input_box.normal,
            InputMode::Command => theme.component_styles.input_box.focused.fg(Color::Yellow),
            InputMode::Search => theme.component_styles.input_box.focused.fg(Color::Green),
            InputMode::Multiline => theme.component_styles.input_box.focused.fg(Color::Blue),
        };
        
        let title = match self.mode {
            InputMode::Normal => " Input ",
            InputMode::Command => " Command ",
            InputMode::Search => " Search ",
            InputMode::Multiline => " Multiline (Ctrl+Enter to submit) ",
        };
        
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(theme.component_styles.panel.border_focused);
        
        // Render based on mode
        match self.mode {
            InputMode::Multiline => {
                self.render_multiline_input(f, area, block, input_style);
            }
            _ => {
                self.render_single_line_input(f, area, block, input_style, buffer);
            }
        }
        
        // Render suggestions if available
        if !self.suggestions.is_empty() {
            self.render_suggestions(f, area, theme);
        }
    }
    
    /// Render single line input
    fn render_single_line_input(&mut self, f: &mut Frame, area: Rect, block: Block, style: Style, buffer: &String) {
        // Calculate visible area width (accounting for borders)
        let visible_width = area.width.saturating_sub(2) as usize;
        
        // Convert byte positions to character positions for proper UTF-8 handling
        let chars: Vec<char> = buffer.chars().collect();
        let char_count = chars.len();
        
        // Convert cursor position from bytes to characters
        let cursor_char_pos = buffer[..self.cursor_position.min(buffer.len())]
            .chars()
            .count();
        
        // Calculate scroll offset in characters (not bytes)
        let mut scroll_char_offset = if buffer.is_empty() {
            0
        } else {
            buffer[..self.scroll_offset.min(buffer.len())]
                .chars()
                .count()
        };
        
        // Adjust scroll offset to keep cursor visible
        if cursor_char_pos >= scroll_char_offset + visible_width {
            // Cursor is beyond the right edge, scroll right
            scroll_char_offset = cursor_char_pos.saturating_sub(visible_width - 1);
        } else if cursor_char_pos < scroll_char_offset {
            // Cursor is before the left edge, scroll left
            scroll_char_offset = cursor_char_pos;
        }
        
        // Update byte-based scroll_offset from character position
        self.scroll_offset = chars.iter()
            .take(scroll_char_offset)
            .map(|c| c.len_utf8())
            .sum();
        
        // Get the visible portion of text (in characters)
        let visible_chars: String = if scroll_char_offset < char_count {
            let end_char_pos = (scroll_char_offset + visible_width).min(char_count);
            chars[scroll_char_offset..end_char_pos].iter().collect()
        } else {
            String::new()
        };
        
        // Build display text with scroll indicators
        let mut display_text = String::new();
        if scroll_char_offset > 0 {
            display_text.push('…'); // Left scroll indicator
            // Take all but first character if there's room
            if visible_chars.chars().count() > 1 {
                display_text.push_str(&visible_chars.chars().skip(1).collect::<String>());
            }
        } else {
            display_text.push_str(&visible_chars);
        }
        
        // Add right scroll indicator if needed
        if scroll_char_offset + visible_width < char_count && !display_text.is_empty() {
            // Remove last character to make room for indicator
            let mut chars = display_text.chars().collect::<Vec<_>>();
            if chars.len() > 1 {
                chars.pop();
                display_text = chars.into_iter().collect();
            }
            display_text.push('…');
        }
        
        let input = Paragraph::new(display_text.as_str())
            .style(Style::default().fg(Color::White))
            .block(block.clone());
        
        f.render_widget(input, area);
        
        // Render visible cursor
        if area.width > 2 && area.height > 2 {
            // Calculate cursor position relative to visible area (in characters)
            let relative_cursor_pos = cursor_char_pos.saturating_sub(scroll_char_offset);
            
            // Adjust for left scroll indicator
            let cursor_offset = if scroll_char_offset > 0 { 1 } else { 0 };
            
            let cursor_x = area.x + 1 + (relative_cursor_pos + cursor_offset) as u16;
            let cursor_y = area.y + 1;
            
            if cursor_x < area.x + area.width - 1 && cursor_y < area.y + area.height - 1 {
                // Create a small rectangle for the cursor
                let cursor_area = Rect {
                    x: cursor_x,
                    y: cursor_y,
                    width: 1,
                    height: 1,
                };
                
                // Render a blue block as cursor
                let cursor_block = Block::default()
                    .style(Style::default().bg(Color::Blue));
                f.render_widget(cursor_block, cursor_area);
            }
        }
    }
    
    /// Render multiline input
    fn render_multiline_input(&self, f: &mut Frame, area: Rect, block: Block, style: Style) {
        let lines: Vec<Line> = self.multiline_state.lines
            .iter()
            .map(|line| Line::from(line.as_str()))
            .collect();
        
        let paragraph = Paragraph::new(lines)
            .style(style)
            .block(block);
        
        f.render_widget(paragraph, area);
    }
    
    /// Render autocomplete suggestions
    fn render_suggestions(&self, f: &mut Frame, input_area: Rect, theme: &super::theme_engine::ChatTheme) {
        let suggestions_height = (self.suggestions.len() as u16).min(5);
        if suggestions_height == 0 || input_area.y < suggestions_height + 1 {
            return;
        }
        
        let suggestions_area = Rect {
            x: input_area.x,
            y: input_area.y.saturating_sub(suggestions_height + 1),
            width: input_area.width,
            height: suggestions_height + 2,
        };
        
        let items: Vec<ListItem> = self.suggestions
            .iter()
            .enumerate()
            .map(|(i, suggestion)| {
                let selected = self.selected_suggestion == Some(i);
                let style = if selected {
                    theme.component_styles.menu.item_selected
                } else {
                    theme.component_styles.menu.item
                };
                
                let content = format!("{} - {}", suggestion.display, suggestion.description);
                ListItem::new(content).style(style)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Suggestions ")
                .border_style(theme.component_styles.panel.border));
        
        f.render_widget(list, suggestions_area);
    }
}

impl InputHistory {
    fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_size,
            current_index: None,
            temp_buffer: None,
        }
    }
    
    fn add(&mut self, entry: String) {
        if self.entries.front() != Some(&entry) {
            self.entries.push_front(entry);
            if self.entries.len() > self.max_size {
                self.entries.pop_back();
            }
        }
        self.current_index = None;
        self.temp_buffer = None;
    }
    
    fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        
        match self.current_index {
            None => {
                self.temp_buffer = Some(current.to_string());
                self.current_index = Some(0);
                self.entries.get(0).cloned()
            }
            Some(index) => {
                if index + 1 < self.entries.len() {
                    self.current_index = Some(index + 1);
                    self.entries.get(index + 1).cloned()
                } else {
                    None
                }
            }
        }
    }
    
    fn next(&mut self) -> Option<String> {
        match self.current_index {
            None => None,
            Some(0) => {
                self.current_index = None;
                self.temp_buffer.take()
            }
            Some(index) => {
                self.current_index = Some(index - 1);
                self.entries.get(index - 1).cloned()
            }
        }
    }
}

impl MultilineState {
    fn new() -> Self {
        Self {
            lines: vec![String::new()],
            current_line: 0,
            column: 0,
        }
    }
    
    fn start_with(&mut self, initial: &str) {
        self.lines = vec![initial.to_string()];
        self.current_line = 0;
        self.column = initial.len();
    }
    
    fn new_line(&mut self) {
        let current_line_content = self.lines[self.current_line].clone();
        let (before, after) = current_line_content.split_at(self.column);
        
        self.lines[self.current_line] = before.to_string();
        self.current_line += 1;
        self.lines.insert(self.current_line, after.to_string());
        self.column = 0;
    }
    
    fn move_up(&mut self) {
        if self.current_line > 0 {
            self.current_line -= 1;
            self.column = self.column.min(self.lines[self.current_line].len());
        }
    }
    
    fn move_down(&mut self) {
        if self.current_line < self.lines.len() - 1 {
            self.current_line += 1;
            self.column = self.column.min(self.lines[self.current_line].len());
        }
    }
    
    fn get_content(&self) -> String {
        self.lines.join("\n")
    }
    
    fn get_current_line(&self) -> String {
        self.lines.get(self.current_line).cloned().unwrap_or_default()
    }
    
    fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.current_line = 0;
        self.column = 0;
    }
}

impl AutocompleteEngine {
    fn new() -> Self {
        let commands = vec![
            CommandCompletion {
                trigger: ":help".to_string(),
                completions: vec!["help".to_string()],
                description: "Show help information".to_string(),
            },
            CommandCompletion {
                trigger: ":clear".to_string(),
                completions: vec!["clear".to_string()],
                description: "Clear the chat history".to_string(),
            },
            CommandCompletion {
                trigger: ":theme".to_string(),
                completions: vec!["theme dark".to_string(), "theme light".to_string(), "theme high_contrast".to_string()],
                description: "Change the theme".to_string(),
            },
            CommandCompletion {
                trigger: ":layout".to_string(),
                completions: vec!["layout side".to_string(), "layout full".to_string(), "layout grid".to_string()],
                description: "Change the layout".to_string(),
            },
        ];
        
        Self {
            commands,
            context_completions: Vec::new(),
            recent_completions: VecDeque::with_capacity(100),
            fuzzy_match: true,
        }
    }
    
    fn get_suggestions(&self, buffer: &str, cursor_pos: usize) -> Vec<CompletionSuggestion> {
        let mut suggestions = Vec::new();
        
        // Handle empty buffer case
        if buffer.is_empty() || cursor_pos == 0 {
            return suggestions;
        }
        
        // Ensure cursor_pos is at a valid UTF-8 boundary
        let safe_cursor_pos = if cursor_pos > buffer.len() {
            buffer.len()
        } else {
            // Find the nearest character boundary at or before cursor_pos
            let mut pos = cursor_pos;
            while pos > 0 && !buffer.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        };
        
        // Get the text up to the cursor
        let text_to_cursor = &buffer[..safe_cursor_pos];
        
        // Find the start of the current word
        let word_start_byte = text_to_cursor
            .rfind(char::is_whitespace)
            .map(|i| {
                // Move past the whitespace character
                let mut pos = i + 1;
                while pos < text_to_cursor.len() && !text_to_cursor.is_char_boundary(pos) {
                    pos += 1;
                }
                pos
            })
            .unwrap_or(0);
        
        let current_word = &text_to_cursor[word_start_byte..];
        
        // Command completions
        if buffer.starts_with(':') {
            for cmd in &self.commands {
                if cmd.trigger.starts_with(buffer) {
                    suggestions.push(CompletionSuggestion {
                        text: cmd.trigger[1..].to_string(),
                        display: cmd.trigger.clone(),
                        description: cmd.description.clone(),
                        score: 1.0,
                        category: SuggestionCategory::Command,
                    });
                }
            }
        }
        
        // Tool completions
        if current_word.starts_with("@") {
            let tools = vec!["@file", "@web", "@search", "@code", "@terminal"];
            for tool in tools {
                if tool.starts_with(current_word) {
                    suggestions.push(CompletionSuggestion {
                        text: tool[1..].to_string(),
                        display: tool.to_string(),
                        description: format!("Use {} tool", tool),
                        score: 0.9,
                        category: SuggestionCategory::Tool,
                    });
                }
            }
        }
        
        // Sort suggestions by score
        suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        suggestions.truncate(10);
        
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cursor_movement() {
        let mut handler = InputHandler::new();
        let mut buffer = "Hello World".to_string();
        handler.cursor_position = 5;
        
        handler.move_cursor_left(&buffer);
        assert_eq!(handler.cursor_position, 4);
        
        handler.move_cursor_word_left(&buffer);
        assert_eq!(handler.cursor_position, 0);
        
        handler.move_cursor_word_right(&buffer);
        assert_eq!(handler.cursor_position, 6);
    }
    
    #[test]
    fn test_history() {
        let mut history = InputHistory::new(10);
        history.add("first".to_string());
        history.add("second".to_string());
        
        assert_eq!(history.previous("current"), Some("second".to_string()));
        assert_eq!(history.previous("current"), Some("first".to_string()));
        assert_eq!(history.next(), Some("second".to_string()));
    }
}