//! Tests for the chat bridge functionality

use super::*;
use crate::tui::chat::bridge::ChatSubtabBridge;
use crate::tui::chat::state::ChatState;
use crate::tui::chat::orchestration::OrchestrationManager;
use crate::tui::chat::agents::manager::AgentManager;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_bridge_creation() {
    // Create dependencies
    let chat_state = Arc::new(RwLock::new(ChatState::new(0, "Test Chat".to_string())));
    let orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
    let (tx, _rx) = tokio::sync::mpsc::channel(100);
    let available_models = vec![];
    
    // Create bridge
    let bridge = ChatSubtabBridge::new(chat_state, orchestration, tx, available_models);
    
    // Verify bridge has all subtabs
    assert_eq!(bridge.subtab_count(), 7);
    assert_eq!(bridge.current_index(), 0);
}

#[tokio::test]
async fn test_bridge_navigation() {
    // Create dependencies
    let chat_state = Arc::new(RwLock::new(ChatState::new(0, "Test Chat".to_string())));
    let orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
    let (tx, _rx) = tokio::sync::mpsc::channel(100);
    let available_models = vec![];
    
    // Create bridge
    let mut bridge = ChatSubtabBridge::new(chat_state, orchestration,  tx, available_models);
    
    // Test navigation
    bridge.set_active(1); // Models tab
    assert_eq!(bridge.current_index(), 1);
    
    bridge.set_active(4); // Orchestration tab
    assert_eq!(bridge.current_index(), 4);
    
    // Test bounds
    bridge.set_active(10); // Out of bounds
    assert_eq!(bridge.current_index(), 4); // Should not change
}

#[tokio::test]
async fn test_bridge_titles() {
    // Create dependencies
    let chat_state = Arc::new(RwLock::new(ChatState::new(0, "Test Chat".to_string())));
    let orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
    let (tx, _rx) = tokio::sync::mpsc::channel(100);
    let available_models = vec![];
    
    // Create bridge
    let bridge = ChatSubtabBridge::new(chat_state, orchestration,  tx, available_models);
    
    // Verify all subtab titles
    let expected_titles = vec![
        "Chat",
        "Models",
        "History",
        "Settings",
        "Orchestration",
        "Agents",
        "CLI",
    ];
    
    for (i, expected) in expected_titles.iter().enumerate() {
        assert_eq!(bridge.get_title(i), Some(expected.to_string()));
    }
}

#[test]
fn test_bridge_handle_key() {
    // This test would require mocking the Frame and other UI components
    // For now, we'll just verify the bridge responds to key events
    // In a real test, we'd use a test framework that can mock ratatui components
}