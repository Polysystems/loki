//! Navigation and scrolling functionality for chat messages

use crate::tui::chat::ChatState;
use crate::tui::run::AssistantResponseType;

/// Handles navigation and scrolling within chat messages
pub struct NavigationHandler;

impl NavigationHandler {
    /// Navigate up in the message list
    pub fn navigate_up(chat: &mut ChatState) {
        if let Some(current) = chat.selected_message_index {
            if current > 0 {
                chat.selected_message_index = Some(current - 1);
            }
        } else if !chat.messages.is_empty() {
            // If no selection, start from bottom
            chat.selected_message_index = Some(chat.messages.len() - 1);
        }
    }
    
    /// Navigate down in the message list
    pub fn navigate_down(chat: &mut ChatState) {
        let max_index = chat.messages.len().saturating_sub(1);
        
        if let Some(current) = chat.selected_message_index {
            if current < max_index {
                chat.selected_message_index = Some(current + 1);
            }
        } else if !chat.messages.is_empty() {
            // If no selection, start from top
            chat.selected_message_index = Some(0);
        }
    }
    
    /// Clear current selection
    pub fn clear_selection(chat: &mut ChatState) {
        chat.selected_message_index = None;
    }
    
    /// Scroll to the bottom of the chat
    pub fn scroll_to_bottom(chat: &mut ChatState) {
        if !chat.messages.is_empty() {
            chat.selected_message_index = Some(chat.messages.len() - 1);
            chat.scroll_offset = chat.messages.len().saturating_sub(10);
        }
    }
    
    /// Calculate total lines needed for all messages
    pub fn calculate_total_lines(messages: &[AssistantResponseType], width: usize) -> usize {
        messages.iter().map(|msg| {
            let content = msg.get_content();
            let lines = content.lines().count();
            let wrapped_lines: usize = content.lines().map(|line| {
                if line.is_empty() {
                    1
                } else {
                    (line.len() + width - 1) / width
                }
            }).sum();
            wrapped_lines.max(lines)
        }).sum()
    }
    
    /// Get range of visible messages
    pub fn get_visible_messages(
        messages: &[AssistantResponseType],
        scroll_offset: usize,
        max_visible: usize
    ) -> (usize, usize) {
        let start = scroll_offset;
        let end = (scroll_offset + max_visible).min(messages.len());
        (start, end)
    }
    
    /// Jump to a specific message
    pub fn jump_to_message(
        chat: &mut ChatState,
        message_index: usize
    ) {
        if message_index < chat.messages.len() {
            chat.selected_message_index = Some(message_index);
            // Adjust scroll to ensure message is visible
            if message_index < chat.scroll_offset {
                chat.scroll_offset = message_index;
            } else if message_index >= chat.scroll_offset + 10 {
                chat.scroll_offset = message_index.saturating_sub(5);
            }
        }
    }
    
    /// Ensure the selected message is visible
    pub fn ensure_visible(chat: &mut ChatState, viewport_height: usize) {
        if let Some(selected) = chat.selected_message_index {
            if selected < chat.scroll_offset {
                chat.scroll_offset = selected;
            } else if selected >= chat.scroll_offset + viewport_height {
                chat.scroll_offset = selected.saturating_sub(viewport_height - 1);
            }
        }
    }
}

/// Extension trait for ChatState to add navigation methods
pub trait NavigationExt {
    fn navigate_up(&mut self);
    fn navigate_down(&mut self);
    fn clear_selection(&mut self);
    fn scroll_to_bottom(&mut self);
    fn jump_to_message(&mut self, index: usize);
    fn ensure_message_visible(&mut self, viewport_height: usize);
}

impl NavigationExt for ChatState {
    fn navigate_up(&mut self) {
        NavigationHandler::navigate_up(self);
    }
    
    fn navigate_down(&mut self) {
        NavigationHandler::navigate_down(self);
    }
    
    fn clear_selection(&mut self) {
        NavigationHandler::clear_selection(self);
    }
    
    fn scroll_to_bottom(&mut self) {
        NavigationHandler::scroll_to_bottom(self);
    }
    
    fn jump_to_message(&mut self, index: usize) {
        NavigationHandler::jump_to_message(self, index);
    }
    
    fn ensure_message_visible(&mut self, viewport_height: usize) {
        NavigationHandler::ensure_visible(self, viewport_height);
    }
}