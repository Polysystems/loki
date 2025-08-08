//! Chat session management
//! 
//! Manages multiple chat sessions and active chat switching

use std::collections::HashMap;
use anyhow::Result;

use super::chat_state::ChatState;

/// Manages multiple chat sessions
#[derive(Debug)]
pub struct SessionManager {
    /// All chat sessions
    pub chats: HashMap<usize, ChatState>,
    
    /// Currently active chat ID
    pub active_chat: usize,
    
    /// Next chat ID to assign
    next_chat_id: usize,
}

impl Default for SessionManager {
    fn default() -> Self {
        let mut chats = HashMap::new();
        chats.insert(0, ChatState::new(0, "Main Chat".to_string()));
        
        Self {
            chats,
            active_chat: 0,
            next_chat_id: 1,
        }
    }
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a placeholder instance
    pub fn placeholder() -> Self {
        Self::default()
    }
    
    /// Create a new chat session
    pub fn create_chat(&mut self, name: String) -> usize {
        let chat_id = self.next_chat_id;
        self.next_chat_id += 1;
        
        self.chats.insert(chat_id, ChatState::new(chat_id, name));
        chat_id
    }
    
    /// Switch to a different chat
    pub fn switch_to_chat(&mut self, chat_id: usize) -> Result<()> {
        if self.chats.contains_key(&chat_id) {
            self.active_chat = chat_id;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Chat {} not found", chat_id))
        }
    }
    
    /// Get the active chat
    pub fn active_chat(&self) -> Option<&ChatState> {
        self.chats.get(&self.active_chat)
    }
    
    /// Get the active chat mutably
    pub fn active_chat_mut(&mut self) -> Option<&mut ChatState> {
        self.chats.get_mut(&self.active_chat)
    }
    
    /// Delete a chat session
    pub fn delete_chat(&mut self, chat_id: usize) -> Result<()> {
        if chat_id == 0 {
            return Err(anyhow::anyhow!("Cannot delete main chat"));
        }
        
        self.chats.remove(&chat_id);
        
        // If we deleted the active chat, switch to main
        if self.active_chat == chat_id {
            self.active_chat = 0;
        }
        
        Ok(())
    }
    
    /// Get all chat sessions
    pub fn all_chats(&self) -> Vec<(usize, &ChatState)> {
        let mut chats: Vec<_> = self.chats.iter()
            .map(|(id, chat)| (*id, chat))
            .collect();
        chats.sort_by_key(|(id, _)| *id);
        chats
    }
    
    /// Count total messages across all chats
    pub fn total_message_count(&self) -> usize {
        self.chats.values()
            .map(|chat| chat.total_message_count())
            .sum()
    }
}