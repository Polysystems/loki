//! Message handling and management
//! 
//! Handles message creation, deletion, and modification

use crate::tui::chat::ChatState;
use crate::tui::run::AssistantResponseType;
use crate::tui::chat::state::StateManagementExt;

/// Handles message operations
pub struct MessageHandler;

impl MessageHandler {
    /// Add a new user message to the chat
    pub fn add_user_message(chat: &mut ChatState, content: String, thread: usize) {
        let message = AssistantResponseType::new_user_message(content);
        chat.add_message_to_chat(message, thread);
        chat.mark_activity();
    }
    
    /// Add a new AI message to the chat
    pub fn add_ai_message(chat: &mut ChatState, content: String, model: Option<String>, thread: usize) {
        let message = AssistantResponseType::new_ai_message(content, model);
        chat.add_message_to_chat(message, thread);
        chat.mark_activity();
    }
    
    /// Add an error message to the chat
    pub fn add_error_message(chat: &mut ChatState, error: String, thread: usize) {
        let message = AssistantResponseType::new_error("Error".to_string(), error);
        chat.add_message_to_chat(message, thread);
        chat.mark_activity();
    }
    
    /// Add an action message to the chat
    pub fn add_action_message(chat: &mut ChatState, command: String, result: Option<String>, thread: usize) {
        let message = if let Some(res) = result {
            AssistantResponseType::new_action_completed(command, res)
        } else {
            AssistantResponseType::new_action_executing(command, 0.0)
        };
        chat.add_message_to_chat(message, thread);
        chat.mark_activity();
    }
    
    /// Add a code message to the chat
    pub fn add_code_message(chat: &mut ChatState, code: String, language: String, thread: usize) {
        let message = AssistantResponseType::new_code(language, code, "Loki".to_string());
        chat.add_message_to_chat(message, thread);
        chat.mark_activity();
    }
    
    /// Delete a message by index
    pub fn delete_message(chat: &mut ChatState, index: usize, thread: usize) -> bool {
        let messages = match thread {
            0 => &mut chat.messages,
            1 => &mut chat.messages_1,
            2 => &mut chat.messages_2,
            _ => return false,
        };
        
        if index < messages.len() {
            messages.remove(index);
            
            // Adjust selection if needed
            if let Some(selected) = chat.selected_message_index {
                if selected == index {
                    chat.selected_message_index = None;
                } else if selected > index {
                    chat.selected_message_index = Some(selected - 1);
                }
            }
            
            chat.is_modified = true;
            true
        } else {
            false
        }
    }
    
    /// Copy a message to clipboard (returns the message content)
    pub fn copy_message(chat: &ChatState, index: usize, thread: usize) -> Option<String> {
        let messages = match thread {
            0 => &chat.messages,
            1 => &chat.messages_1,
            2 => &chat.messages_2,
            _ => return None,
        };
        
        messages.get(index).map(|msg| msg.get_content().to_string())
    }
    
    /// Update message metadata (add tag)
    pub fn add_message_tag(
        chat: &mut ChatState,
        index: usize,
        thread: usize,
        tag: String
    ) -> bool {
        let messages = match thread {
            0 => &mut chat.messages,
            1 => &mut chat.messages_1,
            2 => &mut chat.messages_2,
            _ => return false,
        };
        
        if let Some(message) = messages.get_mut(index) {
            if let AssistantResponseType::Message { metadata, .. } = message {
                if !metadata.tags.contains(&tag) {
                    metadata.tags.push(tag);
                    chat.is_modified = true;
                    return true;
                }
            }
        }
        false
    }
    
    /// Count messages by type
    pub fn count_messages_by_type(chat: &ChatState, thread: usize) -> MessageTypeCount {
        let messages = match thread {
            0 => &chat.messages,
            1 => &chat.messages_1,
            2 => &chat.messages_2,
            _ => &chat.messages,
        };
        
        let mut count = MessageTypeCount::default();
        
        for msg in messages {
            match msg {
                AssistantResponseType::Message { author, .. } => {
                    if author == "You" || author == "user" {
                        count.user_messages += 1;
                    } else {
                        count.ai_messages += 1;
                    }
                }
                AssistantResponseType::Action { .. } => count.action_messages += 1,
                AssistantResponseType::Code { .. } => count.code_messages += 1,
                AssistantResponseType::Error { .. } => count.error_messages += 1,
                AssistantResponseType::ToolUse { .. } => count.action_messages += 1,
                AssistantResponseType::Stream { .. } => count.ai_messages += 1,
                AssistantResponseType::ChatMessage { .. } => count.ai_messages += 1,
                AssistantResponseType::UserMessage { .. } => count.user_messages += 1,
                AssistantResponseType::SystemMessage { .. } => count.system_messages += 1,
                AssistantResponseType::ToolExecution { .. } => count.action_messages += 1,
                AssistantResponseType::ThinkingMessage { .. } => count.ai_messages += 1,
            }
        }
        
        count
    }
}

#[derive(Debug, Default)]
pub struct MessageTypeCount {
    pub user_messages: usize,
    pub ai_messages: usize,
    pub action_messages: usize,
    pub code_messages: usize,
    pub error_messages: usize,
    pub system_messages: usize,
}

/// Extension trait for ChatState to add message handling methods
pub trait MessageHandlingExt {
    fn add_user_msg(&mut self, content: String, thread: usize);
    fn add_ai_msg(&mut self, content: String, model: Option<String>, thread: usize);
    fn add_error_msg(&mut self, error: String, thread: usize);
    fn delete_msg(&mut self, index: usize, thread: usize) -> bool;
    fn copy_msg(&self, index: usize, thread: usize) -> Option<String>;
}

impl MessageHandlingExt for ChatState {
    fn add_user_msg(&mut self, content: String, thread: usize) {
        MessageHandler::add_user_message(self, content, thread);
    }
    
    fn add_ai_msg(&mut self, content: String, model: Option<String>, thread: usize) {
        MessageHandler::add_ai_message(self, content, model, thread);
    }
    
    fn add_error_msg(&mut self, error: String, thread: usize) {
        MessageHandler::add_error_message(self, error, thread);
    }
    
    fn delete_msg(&mut self, index: usize, thread: usize) -> bool {
        MessageHandler::delete_message(self, index, thread)
    }
    
    fn copy_msg(&self, index: usize, thread: usize) -> Option<String> {
        MessageHandler::copy_message(self, index, thread)
    }
}