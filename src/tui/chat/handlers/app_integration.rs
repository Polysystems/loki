//! Integration handler for connecting app.rs key events to the chat bridge

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use anyhow::Result;

use crate::tui::App;

/// Handle key events for the chat system through the modular system
pub async fn handle_chat_key_event(
    app: &mut App,
    key: KeyCode,
    modifiers: KeyModifiers,
) -> Result<bool> {
    // Create KeyEvent from components
    let key_event = KeyEvent::new(key, modifiers);
    
    // Try direct integration first (preferred)
    {
        let mut manager = app.state.chat.subtab_manager.borrow_mut();
        
        // Special handling for number keys to switch subtabs
        match key {
            KeyCode::Char('1'..='7') if !modifiers.contains(KeyModifiers::CONTROL) => {
                let index = (key_event.code.to_string().chars().next().unwrap() as usize) - ('1' as usize);
                if index < manager.tab_count() {
                    app.state.chat_tabs.current_index = index;
                    manager.set_active(index);
                    return Ok(false);
                }
            }
            _ => {}
        }
        
        // Update manager to current subtab
        manager.set_active(app.state.chat_tabs.current_index);
        
        // Let the manager handle the input - this includes scrolling
        match manager.handle_input(key_event) {
            Ok(_) => {
                // Update manager state after handling input
                if let Err(e) = manager.update() {
                    tracing::warn!("Failed to update subtab manager after input: {}", e);
                }
                return Ok(false);
            }
            Err(e) => {
                tracing::warn!("Subtab manager input handling failed: {}", e);
                // Fall through to legacy handling only for specific cases
            }
        }
    }
    
    // Fall back to legacy input handling if bridge not available
    // But ONLY for text input, not navigation keys
    handle_legacy_chat_input(app, key, modifiers).await
}

/// Legacy input handling (to be removed after full migration)
async fn handle_legacy_chat_input(
    app: &mut App,
    key: KeyCode,
    modifiers: KeyModifiers,
) -> Result<bool> {
    match key {
        // Handle Enter key for sending messages
        KeyCode::Enter if !app.state.chat.message_history_mode => {
            if !app.state.chat.command_input.is_empty() {
                let command = app.state.chat.command_input.trim().to_string();
                
                // Build command with attachments if any
                let command_with_attachments = app.state.chat.build_message_with_attachments(&command);
                
                let user_message = crate::tui::run::AssistantResponseType::new_user_message(
                    command_with_attachments.clone()
                );
                app.state.chat.add_message(user_message, Some("0".to_string())).await;
                
                // Clear the chat's input buffer
                app.state.chat.command_input.clear();
                app.state.cursor_position = 0;
                app.state.history_index = None;
                
                // Process the message through the model
                let response = app.state.chat.handle_model_task(command_with_attachments, Some("0".to_string())).await;
                if let Ok(resp) = response {
                    app.state.chat.add_message(resp, Some("0".to_string())).await;
                }
            }
            Ok(false)
        }
        
        // Character input
        KeyCode::Char(c) if !modifiers.contains(KeyModifiers::CONTROL) => {
            app.state.chat.command_input.push(c);
            app.state.cursor_position += 1;
            Ok(false)
        }
        
        // Backspace
        KeyCode::Backspace => {
            if app.state.cursor_position > 0 {
                app.state.chat.command_input.remove(app.state.cursor_position - 1);
                app.state.cursor_position -= 1;
            }
            Ok(false)
        }
        
        // Navigation
        KeyCode::Left => {
            if app.state.cursor_position > 0 {
                app.state.cursor_position -= 1;
            }
            Ok(false)
        }
        KeyCode::Right => {
            if app.state.cursor_position < app.state.chat.command_input.len() {
                app.state.cursor_position += 1;
            }
            Ok(false)
        }
        KeyCode::Home => {
            app.state.cursor_position = 0;
            Ok(false)
        }
        KeyCode::End => {
            app.state.cursor_position = app.state.chat.command_input.len();
            Ok(false)
        }
        
        // History navigation - only handle if we're actually typing in the input field
        // Otherwise, let the chat tab handle scrolling
        KeyCode::Up => {
            // Check if we're in input mode - if not, don't handle history here
            // The chat tab will handle this as a scroll event
            if !app.state.chat.message_history_mode && !app.state.chat.command_input.is_empty() {
                if app.state.command_history.is_empty() {
                    return Ok(false);
                }
                
                match app.state.history_index {
                    None => {
                        // Starting history navigation
                        app.state.history_index = Some(app.state.command_history.len() - 1);
                        if let Some(cmd) = app.state.command_history.get(app.state.command_history.len() - 1) {
                            app.state.chat.command_input = cmd.clone();
                            app.state.cursor_position = cmd.len();
                        }
                    }
                    Some(index) if index > 0 => {
                        // Navigate to older history
                        app.state.history_index = Some(index - 1);
                        if let Some(cmd) = app.state.command_history.get(index - 1) {
                            app.state.chat.command_input = cmd.clone();
                            app.state.cursor_position = cmd.len();
                        }
                    }
                    _ => {} // Already at oldest entry
                }
                Ok(false)
            } else {
                // Not in input mode or no text - don't handle, let chat tab scroll
                Ok(false)
            }
        }
        KeyCode::Down => {
            // Only handle if we're navigating history
            if let Some(index) = app.state.history_index {
                if index < app.state.command_history.len() - 1 {
                    // Navigate to newer history
                    app.state.history_index = Some(index + 1);
                    if let Some(cmd) = app.state.command_history.get(index + 1) {
                        app.state.chat.command_input = cmd.clone();
                        app.state.cursor_position = cmd.len();
                    }
                } else {
                    // Exit history navigation
                    app.state.history_index = None;
                    app.state.chat.command_input.clear();
                    app.state.cursor_position = 0;
                }
                Ok(false)
            } else {
                // Not navigating history - don't handle, let chat tab scroll
                Ok(false)
            }
        }
        
        _ => Ok(false),
    }
}