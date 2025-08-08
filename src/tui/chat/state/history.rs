//! Chat history management

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Manager for command history and navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryManager {
    pub selected_index: usize,
    history: VecDeque<String>,
    max_history: usize,
    current_position: Option<usize>,
}

impl Default for HistoryManager {
    fn default() -> Self {
        Self {
            selected_index: 0,
            history: VecDeque::new(),
            max_history: 1000,
            current_position: None,
        }
    }
}

impl HistoryManager {
    /// Create a new history manager with specified max size
    pub fn new(max_history: usize) -> Self {
        Self {
            selected_index: 0,
            history: VecDeque::with_capacity(max_history),
            max_history,
            current_position: None,
        }
    }
    
    /// Add a command to history
    pub fn add_to_history(&mut self, command: &str) {
        // Don't add empty commands or duplicates of the last command
        if command.is_empty() {
            return;
        }
        
        if let Some(last) = self.history.back() {
            if last == command {
                return;
            }
        }
        
        // Add to history
        self.history.push_back(command.to_string());
        
        // Trim if exceeds max
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
        
        // Reset navigation position
        self.current_position = None;
    }
    
    /// Get previous command in history
    pub fn get_previous(&mut self) -> Option<String> {
        if self.history.is_empty() {
            return None;
        }
        
        match self.current_position {
            None => {
                // Start from the end
                self.current_position = Some(self.history.len() - 1);
                self.history.get(self.history.len() - 1).cloned()
            }
            Some(pos) if pos > 0 => {
                self.current_position = Some(pos - 1);
                self.history.get(pos - 1).cloned()
            }
            _ => self.history.get(0).cloned(),
        }
    }
    
    /// Get next command in history
    pub fn get_next(&mut self) -> Option<String> {
        match self.current_position {
            Some(pos) if pos < self.history.len() - 1 => {
                self.current_position = Some(pos + 1);
                self.history.get(pos + 1).cloned()
            }
            Some(pos) if pos == self.history.len() - 1 => {
                self.current_position = None;
                None // Return to current input
            }
            _ => None,
        }
    }
    
    /// Get all history
    pub fn get_history(&self) -> Vec<String> {
        self.history.iter().cloned().collect()
    }
    
    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
        self.current_position = None;
        self.selected_index = 0;
    }
    
    /// Search history for commands containing a query
    pub fn search(&self, query: &str) -> Vec<String> {
        self.history
            .iter()
            .filter(|cmd| cmd.contains(query))
            .cloned()
            .collect()
    }
}