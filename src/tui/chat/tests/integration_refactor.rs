//! Integration tests for the refactored chat system

use super::*;
use crate::tui::chat::ChatManager;
use crate::tui::chat::initialization::ChatInitState;

#[tokio::test]
async fn test_chat_manager_with_bridge_initialization() {
    // Create a new ChatManager
    let chat = ChatManager::new().await;
    
    // Verify it initializes successfully
    assert!(matches!(
        chat.init_state, 
        ChatInitState::Ready | ChatInitState::Degraded(_)
    ), "ChatManager should initialize to Ready or Degraded state");
    
    // Verify the bridge is initialized
    assert!(chat.subtab_bridge.is_some(), "Subtab bridge should be initialized");
}

#[tokio::test]
async fn test_bridge_subtab_count() {
    let chat = ChatManager::new().await;
    
    if let Some(bridge_ref) = &chat.subtab_bridge {
        let bridge = bridge_ref.borrow();
        assert_eq!(bridge.subtab_count(), 7, "Should have 7 subtabs");
        assert_eq!(bridge.current_index(), 0, "Should start at index 0");
    } else {
        panic!("Bridge should be initialized");
    }
}

#[tokio::test]
async fn test_bridge_subtab_titles() {
    let chat = ChatManager::new().await;
    
    if let Some(bridge_ref) = &chat.subtab_bridge {
        let bridge = bridge_ref.borrow();
        
        // Check all subtab titles
        assert_eq!(bridge.get_title(0), Some("Chat".to_string()));
        assert_eq!(bridge.get_title(1), Some("Models".to_string()));
        assert_eq!(bridge.get_title(2), Some("History".to_string()));
        assert_eq!(bridge.get_title(3), Some("Settings".to_string()));
        assert_eq!(bridge.get_title(4), Some("Orchestration".to_string()));
        assert_eq!(bridge.get_title(5), Some("Agents".to_string()));
        assert_eq!(bridge.get_title(6), Some("CLI".to_string()));
    } else {
        panic!("Bridge should be initialized");
    }
}

#[tokio::test]
async fn test_bridge_navigation() {
    let chat = ChatManager::new().await;
    
    if let Some(bridge_ref) = &chat.subtab_bridge {
        let mut bridge = bridge_ref.borrow_mut();
        
        // Test navigation
        bridge.set_active(3);
        assert_eq!(bridge.current_index(), 3);
        assert_eq!(bridge.current_name(), "Settings");
        
        // Test bounds checking
        bridge.set_active(10); // Out of bounds
        assert_eq!(bridge.current_index(), 3); // Should not change
    } else {
        panic!("Bridge should be initialized");
    }
}

#[tokio::test]
async fn test_chat_functionality_with_bridge() {
    let mut chat = ChatManager::new().await;
    
    // Add a test message
    chat.process_user_message_with_orchestration("Test message", 0).await;
    
    // Verify message was added
    if let Some(chat_state) = chat.chats.get(&0) {
        assert!(chat_state.messages.len() > 0, "Should have at least one message");
    }
    
    // Verify bridge is still functional
    assert!(chat.subtab_bridge.is_some(), "Bridge should remain initialized");
}