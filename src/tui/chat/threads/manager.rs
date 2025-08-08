//! Thread management for branching conversations

use std::collections::HashMap;
use anyhow::Result;
use uuid::Uuid;

use super::thread::MessageThread;

/// Manages conversation threads
#[derive(Debug, Default)]
pub struct ThreadManager {
    /// All threads
    threads: HashMap<String, MessageThread>,
    
    /// Active thread ID
    active_thread: Option<String>,
}

impl ThreadManager {
    /// Create a new thread manager
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a new thread
    pub fn create_thread(&mut self, name: String) -> String {
        let thread_id = Uuid::new_v4().to_string();
        let thread = MessageThread::new(thread_id.clone(), name);
        self.threads.insert(thread_id.clone(), thread);
        thread_id
    }
    
    /// Add message to thread
    pub fn add_to_thread(&mut self, thread_id: String, message: crate::tui::run::AssistantResponseType) -> Result<()> {
        if let Some(thread) = self.threads.get_mut(&thread_id) {
            thread.add_message(message);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Thread not found"))
        }
    }
    
    /// Get thread messages
    pub fn get_thread_messages(&self, thread_id: String) -> Vec<crate::tui::run::AssistantResponseType> {
        self.threads.get(&thread_id)
            .map(|t| t.messages.clone())
            .unwrap_or_default()
    }
    
    /// Switch active thread
    pub fn switch_thread(&mut self, thread_id: String) -> Result<()> {
        if self.threads.contains_key(&thread_id) {
            self.active_thread = Some(thread_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Thread not found"))
        }
    }
}