//! Input processing for chat interface

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use anyhow::Result;
use crossterm::event::KeyCode;

// Define InputEvent locally until we can import from the main chat module
#[derive(Debug, Clone)]
pub enum InputEvent {
    Submit(String),
    CommandExecute(String, Vec<String>),
    SearchQuery(String),
    Cancel,
}

/// Handles input processing for the chat interface
pub struct InputProcessor {
    /// Channel for sending processed messages
    message_tx: mpsc::Sender<(String, usize)>,
    
    /// Current input buffer
    input_buffer: String,
    
    /// Command history
    history: Vec<String>,
    history_index: Option<usize>,
    
    /// Chat state reference
    state: Option<Arc<RwLock<crate::tui::chat::state::ChatState>>>,
}

impl InputProcessor {
    /// Create a new input processor
    pub fn new(message_tx: mpsc::Sender<(String, usize)>) -> Self {
        Self {
            message_tx,
            input_buffer: String::new(),
            history: Vec::new(),
            history_index: None,
            state: None,
        }
    }
    
    /// Set the chat state reference
    pub fn set_state(&mut self, state: Arc<RwLock<crate::tui::chat::state::ChatState>>) {
        self.state = Some(state);
    }
    
    /// Handle keyboard input
    pub async fn handle_key(&mut self, key: KeyCode) -> Result<()> {
        match key {
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::Enter => {
                if !self.input_buffer.is_empty() {
                    let message = self.input_buffer.clone();
                    tracing::info!("💬 InputProcessor sending message: {}", message);
                    self.history.push(message.clone());
                    self.input_buffer.clear();
                    self.history_index = None;
                    
                    // Send message for processing
                    match self.message_tx.send((message.clone(), 0)).await {
                        Ok(_) => tracing::info!("✅ Message sent on channel"),
                        Err(e) => tracing::error!("❌ Failed to send message: {}", e),
                    }
                }
            }
            KeyCode::Up => {
                self.navigate_history_up();
            }
            KeyCode::Down => {
                self.navigate_history_down();
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle input events from enhanced UI
    pub async fn handle_input_event(&mut self, event: InputEvent) -> Result<()> {
        match event {
            InputEvent::Submit(text) => {
                tracing::info!("📝 Received submit event: {}", text);
                self.message_tx.send((text, 0)).await?;
            }
            InputEvent::CommandExecute(cmd, args) => {
                tracing::info!("🔧 Received command: {} {:?}", cmd, args);
                let full_command = format!("/{} {}", cmd, args.join(" "));
                self.message_tx.send((full_command, 0)).await?;
            }
            InputEvent::SearchQuery(query) => {
                tracing::info!("🔍 Received search query: {}", query);
                
                // Create search engine and perform search if state is available
                if let Some(state) = &self.state {
                    let mut search_engine = crate::tui::chat::search::SearchEngine::with_chat_state(
                        50, // max results
                        state.clone()
                    );
                
                let filters = crate::tui::chat::search::ChatSearchFilters {
                    query: Some(query.clone()),
                    ..Default::default()
                };
                
                match search_engine.search(&filters).await {
                    Ok(results) => {
                        // Format search results as a message
                        let mut message = format!("🔍 Search results for '{}': Found {} matches\n\n", query, results.len());
                        
                        for (idx, result) in results.iter().take(10).enumerate() {
                            message.push_str(&format!(
                                "{}. [Message #{}] {}: {}\n",
                                idx + 1,
                                result.message_index,
                                result.author,
                                result.snippet
                            ));
                        }
                        
                        if results.len() > 10 {
                            message.push_str(&format!("\n... and {} more results", results.len() - 10));
                        }
                        
                        // Send search results as a system message
                        self.message_tx.send((format!("/system {}", message), 0)).await?;
                    }
                    Err(e) => {
                        tracing::error!("Search failed: {}", e);
                        self.message_tx.send((
                            format!("/system ❌ Search failed: {}", e), 
                            0
                        )).await?;
                    }
                }
                } else {
                    // No chat state available
                    self.message_tx.send((
                        "/system ❌ Search not available: No chat state".to_string(),
                        0
                    )).await?;
                }
            }
            InputEvent::Cancel => {
                tracing::info!("❌ Received cancel event");
                self.input_buffer.clear();
            }
        }
        Ok(())
    }
    
    /// Navigate up in history
    fn navigate_history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        
        match self.history_index {
            None => {
                self.history_index = Some(self.history.len() - 1);
                self.input_buffer = self.history[self.history.len() - 1].clone();
            }
            Some(idx) if idx > 0 => {
                self.history_index = Some(idx - 1);
                self.input_buffer = self.history[idx - 1].clone();
            }
            _ => {}
        }
    }
    
    /// Navigate down in history
    fn navigate_history_down(&mut self) {
        match self.history_index {
            Some(idx) if idx < self.history.len() - 1 => {
                self.history_index = Some(idx + 1);
                self.input_buffer = self.history[idx + 1].clone();
            }
            Some(_) => {
                self.history_index = None;
                self.input_buffer.clear();
            }
            _ => {}
        }
    }
    
    /// Get current input buffer
    pub fn get_buffer(&self) -> &str {
        &self.input_buffer
    }
    
    /// Clear input buffer
    pub fn clear_buffer(&mut self) {
        self.input_buffer.clear();
    }
    
    /// Set the input buffer to a specific value
    pub fn set_buffer(&mut self, content: String) {
        self.input_buffer = content;
        self.history_index = None;
    }
}