//! Message editing functionality

use crate::tui::chat::ChatState;
use crate::tui::run::AssistantResponseType;

/// Handles message editing operations
pub struct EditHandler;

impl EditHandler {
    /// Start editing the selected message
    pub fn start_edit(chat: &mut ChatState) -> bool {
        if let Some(index) = chat.selected_message_index {
            if let Some(message) = chat.messages.get_mut(index) {
                if let AssistantResponseType::Message { author, .. } = message {
                    if author == "user" {
                        chat.edit_buffer = message.get_content().to_string();
                        return true;
                    }
                }
            }
        }
        false
    }
    
    /// Finish editing and update the message
    pub fn finish_edit(chat: &mut ChatState) {
        if let Some(index) = chat.selected_message_index {
            if let Some(message) = chat.messages.get_mut(index) {
                match message {
                    AssistantResponseType::Message { message: content, author, .. } if author == "user" => {
                        *content = chat.edit_buffer.clone();
                        chat.is_modified = true;
                    }
                    _ => {}
                }
            }
            chat.edit_buffer.clear();
        }
    }
    
    /// Cancel editing without saving changes
    pub fn cancel_edit(chat: &mut ChatState) {
        chat.edit_buffer.clear();
    }
    
    /// Check if currently editing
    pub fn is_editing(chat: &ChatState) -> bool {
        !chat.edit_buffer.is_empty() && chat.selected_message_index.is_some()
    }
    
    /// Get the current edit buffer
    pub fn get_edit_buffer(chat: &ChatState) -> &str {
        &chat.edit_buffer
    }
    
    /// Update the edit buffer
    pub fn update_edit_buffer(chat: &mut ChatState, content: String) {
        chat.edit_buffer = content;
        chat.is_modified = true;
    }
}

/// Extension trait for ChatState to add editing methods
pub trait EditingExt {
    fn start_edit(&mut self) -> bool;
    fn finish_edit(&mut self);
    fn cancel_edit(&mut self);
    fn is_editing(&self) -> bool;
}

impl EditingExt for ChatState {
    fn start_edit(&mut self) -> bool {
        EditHandler::start_edit(self)
    }
    
    fn finish_edit(&mut self) {
        EditHandler::finish_edit(self);
    }
    
    fn cancel_edit(&mut self) {
        EditHandler::cancel_edit(self);
    }
    
    fn is_editing(&self) -> bool {
        EditHandler::is_editing(self)
    }
}