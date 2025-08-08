//! State management tests for chat refactoring

#[cfg(test)]
mod state_tests {
    use crate::tui::chat::{ChatState, ChatManager, ChatSettings};
    use crate::tui::run::AssistantResponseType;
    
    #[test]
    fn test_chat_state_creation() {
        let state = ChatState::new(0, "Test Chat".to_string());
        assert_eq!(state.id, "0");
        assert_eq!(state.name, "Test Chat");
        assert_eq!(state.messages.len(), 0);
        assert!(!state.is_persistent);
    }
    
    #[test]
    fn test_chat_state_add_message() {
        let mut state = ChatState::new(0, "Test".to_string());
        let msg = AssistantResponseType::new_user_message("Hello".to_string());
        
        state.add_message_to_chat(msg, 0);
        assert_eq!(state.messages.len(), 1);
    }
    
    #[test]
    fn test_chat_state_persistence() {
        let mut state = ChatState::new(0, "Persistent".to_string());
        state.make_persistent();
        
        assert!(state.is_persistent);
        assert!(state.memory_associations.is_empty());
    }
    
    #[test]
    fn test_chat_settings_default() {
        let settings = ChatSettings::default();
        assert!(!settings.save_history);
        assert_eq!(settings.threads, 1);
        assert!(!settings.enhanced_ui);
    }
    
    #[tokio::test]
    async fn test_chat_manager_state_operations() {
        let mut manager = ChatManager::new().await.unwrap();
        
        // Test creating new chat
        let chat_id = manager.create_new_chat("Test Chat");
        assert!(manager.chats.contains_key(&chat_id));
        
        // Test switching chats
        let new_id = manager.create_new_chat("Another Chat");
        manager.switch_to_chat(new_id);
        assert_eq!(manager.active_chat, new_id);
        
        // Test deleting chat
        manager.delete_chat(chat_id);
        assert!(!manager.chats.contains_key(&chat_id));
    }
    
    #[tokio::test]
    async fn test_history_manager() {
        let mut manager = ChatManager::new().await.unwrap();
        
        // Add to history
        manager.history_manager.add_to_history("test command");
        assert_eq!(manager.history_manager.get_history().len(), 1);
        
        // Navigate history
        assert_eq!(manager.history_manager.get_previous(), Some("test command".to_string()));
        assert_eq!(manager.history_manager.get_next(), None);
    }
    
    #[tokio::test]
    async fn test_context_management() {
        let mut manager = ChatManager::new().await.unwrap();
        
        // Test token estimation
        let estimate = manager.context_manager.estimate_tokens("Hello world");
        assert!(estimate > 0);
        
        // Test context suggestions
        manager.context_manager.add_context_chunk(
            "test".to_string(),
            "Test content".to_string(),
            100,
        );
        
        let suggestions = manager.context_manager.get_suggestions("test");
        assert!(suggestions.len() > 0);
    }
    
    #[tokio::test]
    async fn test_search_functionality() {
        let mut manager = ChatManager::new().await.unwrap();
        
        // Add test messages
        manager.process_user_message_with_orchestration("test message", 0).await;
        manager.process_user_message_with_orchestration("another test", 0).await;
        
        // Search messages
        let results = manager.search_messages("test", None);
        assert!(results.len() >= 2);
    }
    
    #[tokio::test]
    async fn test_thread_management() {
        let mut manager = ChatManager::new().await.unwrap();
        
        // Create thread
        let thread_id = manager.thread_manager.create_thread("Main");
        
        // Add message to thread
        let msg = AssistantResponseType::new_user_message("Thread message".to_string());
        manager.thread_manager.add_to_thread(thread_id, msg);
        
        // Get thread messages
        let messages = manager.thread_manager.get_thread_messages(thread_id);
        assert_eq!(messages.len(), 1);
    }
}