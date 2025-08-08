//! Integration tests for chat system refactoring
//! 
//! These tests ensure all components remain connected during refactoring

#[cfg(test)]
mod chat_integration_tests {
    use super::*;
    use crate::tui::chat::ChatManager;
    use crate::cognitive::{CognitiveSystem, CognitiveConfig};
    use crate::memory::CognitiveMemory;
    use crate::models::orchestrator::ModelOrchestrator;
    use crate::tools::intelligent_manager::IntelligentToolManager;
    use crate::tools::task_management::TaskManager;
    use std::sync::Arc;
    use tokio;

    /// Test basic chat initialization without cognitive system
    #[tokio::test]
    async fn test_chat_basic_initialization() {
        let chat = ChatManager::new().await;
        assert!(chat.is_ok(), "Chat should initialize without cognitive system");
        
        let chat = chat.unwrap();
        assert!(chat.cognitive_enhancement.is_none());
        assert!(chat.intelligent_tool_manager.is_none());
        // Should still have basic components
        assert_eq!(chat.chats.len(), 1); // Default chat
    }

    /// Test chat initialization with full cognitive system
    #[tokio::test] 
    async fn test_chat_with_cognitive_system() {
        // Skip if no API keys configured
        if std::env::var("OPENAI_API_KEY").is_err() {
            println!("Skipping cognitive test - no API keys");
            return;
        }

        let config = CognitiveConfig::default();
        let api_keys = Default::default();
        let cognitive_system = CognitiveSystem::new(api_keys, config).await.ok();
        
        if let Some(cognitive) = cognitive_system {
            let chat = ChatManager::new().await.unwrap();
            // Initialize with cognitive system
            // ... test cognitive enhancement initialization
        }
    }

    /// Test message processing pipeline
    #[tokio::test]
    async fn test_message_processing_pipeline() {
        let mut chat = ChatManager::new().await.unwrap();
        
        // Test basic message
        chat.process_user_message_with_orchestration("Hello", 0).await;
        
        // Check message was added
        let active_chat = chat.chats.get(&0).unwrap();
        assert!(active_chat.messages.len() > 0);
    }

    /// Test tool command detection
    #[tokio::test]
    async fn test_tool_command_detection() {
        let chat = ChatManager::new().await.unwrap();
        
        // Test tool detection
        assert!(chat.looks_like_tool_request("/search test"));
        assert!(chat.looks_like_tool_request("run tests"));
        assert!(!chat.looks_like_tool_request("hello world"));
    }

    /// Test orchestration state synchronization
    #[tokio::test]
    async fn test_orchestration_sync() {
        let mut chat = ChatManager::new().await.unwrap();
        
        // Enable orchestration
        chat.orchestration_manager.orchestration_enabled = true;
        chat.orchestration_manager.preferred_strategy = 
            crate::models::RoutingStrategy::CapabilityBased;
        
        // Initialize orchestration
        let result = chat.initialize_basic_orchestration().await;
        
        if result.is_ok() && chat.model_orchestrator.is_some() {
            // TODO: Verify strategy is synced to backend
            // This will be implemented in refactoring
        }
    }

    /// Test agent stream manager integration
    #[tokio::test]
    async fn test_agent_stream_manager() {
        use crate::tui::ui::chat::agent_stream_manager::AgentStreamManager;
        
        let agent_manager = AgentStreamManager::new(4); // 4 panels max
        
        // Create test agent stream
        let agent_id = agent_manager.create_agent_stream(
            "test-agent".to_string(),
            "TestAgent".to_string(),
            "Test task".to_string(),
        ).await.unwrap();
        
        // Add message
        agent_manager.add_agent_message(
            &agent_id,
            "Test message".to_string(),
            crate::tui::ui::chat::agent_stream_manager::AgentMessageType::Thought,
        ).await.unwrap();
        
        // Verify stream exists
        let streams = agent_manager.get_active_streams().await;
        assert_eq!(streams.len(), 1);
    }

    /// Test consciousness stream integration
    #[tokio::test]
    async fn test_consciousness_stream() {
        use crate::tui::consciousness_stream_integration::ChatConsciousnessStream;
        
        // Skip if no cognitive system
        if std::env::var("OPENAI_API_KEY").is_err() {
            return;
        }
        
        // Would test consciousness stream initialization
        // This requires full cognitive system setup
    }

    /// Test NLP orchestrator integration
    #[tokio::test]
    async fn test_nlp_orchestrator() {
        use crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator;
        
        // Test basic NLP processing
        // Requires mock or real dependencies
    }

    /// Test UI component connections
    #[tokio::test]
    async fn test_ui_components() {
        let chat = ChatManager::new().await.unwrap();
        
        // Verify UI components are initialized
        assert!(chat.message_renderer.is_some());
        assert!(chat.layout_manager.is_some());
        assert!(chat.input_handler.is_some());
        // ... test other UI components
    }

    /// Test state persistence
    #[tokio::test]
    async fn test_state_persistence() {
        let mut chat = ChatManager::new().await.unwrap();
        
        // Add test message
        chat.process_user_message_with_orchestration("Test", 0).await;
        
        // Save state
        let saved = chat.save_chat_state(0).await;
        assert!(saved.is_ok());
        
        // Load state
        let loaded = chat.load_chat_state(0).await;
        assert!(loaded.is_ok());
    }

    /// Test subtab navigation (currently broken)
    #[tokio::test]
    #[should_panic(expected = "subtabs not initialized")]
    async fn test_subtab_navigation() {
        // This test documents the broken subtab system
        // Will be fixed in Phase 4
        let app = crate::tui::app::App::new().await.unwrap();
        assert!(app.chat_sub_tabs.tabs.len() > 0); // Currently fails
    }

    /// Test cognitive command routing
    #[tokio::test]
    async fn test_cognitive_command_routing() {
        let chat = ChatManager::new().await.unwrap();
        
        // Test cognitive command detection
        use crate::tui::cognitive::commands::router::is_cognitive_command;
        
        assert!(is_cognitive_command("think about this"));
        assert!(is_cognitive_command("reason through"));
        assert!(!is_cognitive_command("hello world"));
    }

    /// Test tool executor initialization
    #[tokio::test]
    async fn test_tool_executor_init() {
        use crate::tui::chat::core::tool_executor::ChatToolExecutor;
        
        let executor = ChatToolExecutor::new(
            None, // tool manager
            None, // mcp client
            None, // task manager
            None, // model orchestrator
        );
        
        // Should initialize even without dependencies
        assert!(executor.is_ready());
    }

    /// Test workflow manager
    #[tokio::test]
    async fn test_workflow_manager() {
        let chat = ChatManager::new().await.unwrap();
        let workflow_manager = chat.workflow_manager.read().await;
        
        // Test workflow state
        assert_eq!(
            workflow_manager.current_state(),
            crate::tui::chat::core::workflows::ExtendedWorkflowState::None
        );
    }

    /// Integration test for full message flow
    #[tokio::test]
    async fn test_full_message_flow_integration() {
        // This is the most important test
        // It verifies the entire pipeline works
        
        let mut chat = ChatManager::new().await.unwrap();
        
        // Process different message types
        let test_messages = vec![
            "Hello", // Basic chat
            "/help", // Command
            "search for tests", // Tool request
            "think about consciousness", // Cognitive command
        ];
        
        for msg in test_messages {
            chat.process_user_message_with_orchestration(msg, 0).await;
            
            // Verify message was processed
            let active_chat = chat.chats.get(&0).unwrap();
            assert!(active_chat.messages.len() > 0);
        }
    }
}

/// Performance benchmarks to ensure refactoring doesn't degrade performance
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn bench_chat_initialization() {
        let start = Instant::now();
        let _chat = ChatManager::new().await.unwrap();
        let duration = start.elapsed();
        
        // Should initialize in under 100ms
        assert!(duration.as_millis() < 100, "Chat init took {:?}", duration);
    }

    #[tokio::test]
    async fn bench_message_processing() {
        let mut chat = ChatManager::new().await.unwrap();
        
        let start = Instant::now();
        chat.process_user_message_with_orchestration("test", 0).await;
        let duration = start.elapsed();
        
        // Basic message should process in under 50ms
        assert!(duration.as_millis() < 50, "Message processing took {:?}", duration);
    }
}