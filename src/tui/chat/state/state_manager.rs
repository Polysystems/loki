//! Core state management functionality
//! 
//! Handles chat state updates, thread management, and model selection

use crate::tui::chat::ChatState;
use crate::tui::run::AssistantResponseType;

/// Manages chat state updates and transitions
pub struct StateManager;

impl StateManager {
    /// Update the active model for the chat
    pub fn update_active_model(chat: &mut ChatState, model: String) {
        // Note: active_model is stored on ChatManager, not ChatState
        // This is a placeholder that marks the chat as modified
        // The actual model update should be handled by ChatManager
        chat.is_modified = true;
    }
    
    /// Switch to a different chat thread
    pub fn switch_thread(chat: &mut ChatState, thread_index: usize) -> bool {
        if thread_index <= 2 {
            // Clear selection when switching threads
            chat.selected_message_index = None;
            chat.scroll_offset = 0;
            true
        } else {
            false
        }
    }
    
    /// Get messages for the current thread
    pub fn get_thread_messages(chat: &ChatState, thread_index: usize) -> &[AssistantResponseType] {
        match thread_index {
            0 => &chat.messages,
            1 => &chat.messages_1,
            2 => &chat.messages_2,
            _ => &chat.messages,
        }
    }
    
    /// Get mutable messages for the current thread
    pub fn get_thread_messages_mut(chat: &mut ChatState, thread_index: usize) -> &mut Vec<AssistantResponseType> {
        match thread_index {
            0 => &mut chat.messages,
            1 => &mut chat.messages_1,
            2 => &mut chat.messages_2,
            _ => &mut chat.messages,
        }
    }
    
    /// Update chat metadata after activity
    pub fn update_activity(chat: &mut ChatState) {
        chat.last_activity = chrono::Utc::now();
        chat.is_modified = true;
    }
    
    /// Mark a specific message as being edited
    pub fn mark_message_editing(chat: &mut ChatState, index: usize, editing: bool) {
        if let Some(message) = chat.messages.get_mut(index) {
            if let AssistantResponseType::Message { is_editing, .. } = message {
                *is_editing = editing;
                chat.is_modified = true;
            }
        }
    }
    
    /// Update message streaming state
    pub fn update_streaming_state(
        chat: &mut ChatState, 
        index: usize, 
        state: crate::tui::run::StreamingState
    ) {
        if let Some(message) = chat.messages.get_mut(index) {
            if let AssistantResponseType::Message { streaming_state, .. } = message {
                *streaming_state = state;
            }
        }
    }
    
    /// Append content to a streaming message
    pub fn append_to_streaming_message(
        chat: &mut ChatState,
        index: usize,
        content: &str
    ) {
        if let Some(message) = chat.messages.get_mut(index) {
            if let AssistantResponseType::Message { message: msg_content, .. } = message {
                msg_content.push_str(content);
                chat.is_modified = true;
            }
        }
    }
    
    /// Clear all messages in a specific thread
    pub fn clear_thread(chat: &mut ChatState, thread_index: usize) {
        match thread_index {
            0 => chat.messages.clear(),
            1 => chat.messages_1.clear(),
            2 => chat.messages_2.clear(),
            _ => {}
        }
        chat.selected_message_index = None;
        chat.scroll_offset = 0;
        chat.is_modified = true;
    }
    
    /// Export chat history to a formatted string
    pub fn export_chat_history(chat: &ChatState) -> String {
        let mut output = String::new();
        output.push_str(&format!("# Chat: {}\n", chat.title));
        output.push_str(&format!("Created: {}\n", chat.created_at.format("%Y-%m-%d %H:%M:%S")));
        output.push_str(&format!("Last Activity: {}\n\n", chat.last_activity.format("%Y-%m-%d %H:%M:%S")));
        
        // Export all threads
        for (thread_idx, (thread_name, messages)) in [
            ("Main Thread", &chat.messages),
            ("Thread 1", &chat.messages_1),
            ("Thread 2", &chat.messages_2),
        ].iter().enumerate() {
            if !messages.is_empty() {
                output.push_str(&format!("\n## {}\n\n", thread_name));
                for msg in messages.iter() {
                    output.push_str(&format!("{}\n\n", msg.get_content()));
                }
            }
        }
        
        output
    }
}

/// Extension trait for ChatState to add state management methods
pub trait StateManagementExt {
    fn update_model(&mut self, model: String);
    fn switch_to_thread(&mut self, thread: usize) -> bool;
    fn mark_activity(&mut self);
    fn export_history(&self) -> String;
}

impl StateManagementExt for ChatState {
    fn update_model(&mut self, model: String) {
        StateManager::update_active_model(self, model);
    }
    
    fn switch_to_thread(&mut self, thread: usize) -> bool {
        StateManager::switch_thread(self, thread)
    }
    
    fn mark_activity(&mut self) {
        StateManager::update_activity(self);
    }
    
    fn export_history(&self) -> String {
        StateManager::export_chat_history(self)
    }
}