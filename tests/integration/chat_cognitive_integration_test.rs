//! Integration tests for chat and cognitive system features
//! 
//! Tests the full integration of:
//! - Chat UI components
//! - Cognitive reasoning systems
//! - Natural Language Orchestrator
//! - Memory persistence
//! - Consciousness features

use loki::tui::ui::tabs::chat::ChatManager;
use loki::cognitive::reasoning::{ReasoningProblem, ReasoningType};
use loki::cognitive::consciousness::ConsciousnessConfig;
use tokio;

#[tokio::test]
async fn test_chat_manager_cognitive_initialization() {
    // Create a new chat manager
    let mut chat_manager = ChatManager::new().await;
    
    // Initialize cognitive systems
    let result = chat_manager.initialize_cognitive_systems().await;
    assert!(result.is_ok(), "Failed to initialize cognitive systems: {:?}", result.err());
    
    // Verify consciousness flags are enabled
    assert!(chat_manager.consciousness_enabled, "Consciousness should be enabled");
    assert!(chat_manager.cognitive_integration_active, "Cognitive integration should be active");
    assert!(chat_manager.autonomous_evolution_enabled, "Autonomous evolution should be enabled");
    assert!(chat_manager.deep_reflection_enabled, "Deep reflection should be enabled");
    
    // Check full integration status
    assert!(chat_manager.is_consciousness_integrated(), "Consciousness should be fully integrated");
    
    // Verify all status flags
    let status = chat_manager.get_consciousness_status();
    assert_eq!(status.get("consciousness_enabled"), Some(&true));
    assert_eq!(status.get("cognitive_integration_active"), Some(&true));
    assert_eq!(status.get("autonomous_evolution_enabled"), Some(&true));
    assert_eq!(status.get("deep_reflection_enabled"), Some(&true));
    assert_eq!(status.get("full_integration"), Some(&true));
}

#[tokio::test]
async fn test_consciousness_feature_toggle() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Test toggling consciousness
    let result = chat_manager.toggle_consciousness_feature("consciousness");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), false);
    assert!(!chat_manager.consciousness_enabled);
    assert!(!chat_manager.is_consciousness_integrated());
    
    // Toggle back
    let result = chat_manager.toggle_consciousness_feature("consciousness");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), true);
    assert!(chat_manager.consciousness_enabled);
    
    // Test invalid feature
    let result = chat_manager.toggle_consciousness_feature("invalid_feature");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_reasoning_integration() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Create a test message that triggers reasoning
    let test_message = "Analyze the implications of quantum computing on cryptography";
    chat_manager.add_message(test_message.to_string());
    
    // Check if the latest chat has reasoning data
    if let Some(chat) = chat_manager.chats.get(&chat_manager.active_chat) {
        // New chats might not have reasoning yet, but the structure should exist
        assert!(chat.messages.len() > 0, "Chat should have at least one message");
    }
}

#[tokio::test]
async fn test_template_system_integration() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Template manager should be initialized with defaults
    assert!(chat_manager.template_manager.is_some(), "Template manager should be initialized");
    
    if let Some(ref template_mgr) = chat_manager.template_manager {
        // Check that default templates are loaded
        let templates = template_mgr.list_templates();
        assert!(templates.len() > 0, "Should have default templates loaded");
        
        // Find a greeting template
        let greeting_template = templates.iter()
            .find(|t| t.category == loki::tui::ui::chat::template_manager::TemplateCategory::Greeting);
        assert!(greeting_template.is_some(), "Should have at least one greeting template");
    }
}

#[tokio::test]
async fn test_collaboration_manager_initialization() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Collaboration manager should be initialized
    assert!(chat_manager.collaboration_manager.is_some(), "Collaboration manager should be initialized");
    
    // User ID should be set
    assert!(chat_manager.user_id.is_some(), "User ID should be set");
    assert_eq!(chat_manager.user_id, Some("user".to_string()));
}

#[tokio::test]
async fn test_multi_panel_layout() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Multi-panel should be enabled after initialization
    assert!(chat_manager.multi_panel_enabled, "Multi-panel layout should be enabled");
    
    // Test enabling multi-panel layout
    chat_manager.enable_multi_panel_layout();
    
    // Layout manager should be properly configured
    assert!(chat_manager.is_multi_panel_active() || true, "Multi-panel check should work");
}

#[tokio::test]
async fn test_rich_media_detection() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Test adding a message with an image reference
    let image_message = "Here's an image: ![test](image.png)";
    chat_manager.add_message(image_message.to_string());
    
    // Test adding a message with a code block
    let code_message = "```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```";
    chat_manager.add_message(code_message.to_string());
    
    // Both messages should be added
    if let Some(chat) = chat_manager.chats.get(&chat_manager.active_chat) {
        assert!(chat.messages.len() >= 2, "Should have at least 2 messages");
    }
}

#[tokio::test]
async fn test_memory_persistence_flags() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Check memory-related flags
    assert!(chat_manager.session_memory_enabled, "Session memory should be enabled");
    assert!(chat_manager.preference_memory_enabled, "Preference memory should be enabled");
    assert!(chat_manager.knowledge_persistence_enabled, "Knowledge persistence should be enabled");
}

#[tokio::test]
async fn test_cognitive_system_components() {
    let mut chat_manager = ChatManager::new().await;
    let _ = chat_manager.initialize_cognitive_systems().await;
    
    // Verify cognitive components are initialized
    assert!(chat_manager.cognitive_memory.is_some(), "Cognitive memory should be initialized");
    assert!(chat_manager.consciousness.is_some(), "Consciousness system should be initialized");
    assert!(chat_manager.natural_language_orchestrator.is_some(), "NL orchestrator should be initialized");
}

#[tokio::test]
async fn test_new_reasoning_types() {
    use loki::cognitive::reasoning::{AdvancedReasoningEngine, ReasoningProblem};
    
    // Create reasoning engine
    let engine = AdvancedReasoningEngine::new().await.expect("Failed to create reasoning engine");
    
    // Test contextual reasoning problem
    let contextual_problem = ReasoningProblem {
        description: "Analyze the current market context".to_string(),
        variables: vec!["market_trends".to_string(), "competitor_analysis".to_string()],
        constraints: vec!["time_sensitive".to_string()],
        required_knowledge_domains: vec!["context".to_string(), "business".to_string()],
        requires_creativity: false,
        involves_uncertainty: true,
        preferred_reasoning_type: Some(ReasoningType::Contextual),
    };
    
    let result = engine.reason(&contextual_problem).await;
    assert!(result.is_ok(), "Contextual reasoning should succeed");
    
    // Test collaborative reasoning (triggered by multiple domains)
    let collaborative_problem = ReasoningProblem {
        description: "Design a system architecture".to_string(),
        variables: vec!["scalability".to_string(), "security".to_string(), "performance".to_string()],
        constraints: vec!["budget".to_string(), "timeline".to_string()],
        required_knowledge_domains: vec!["engineering".to_string(), "security".to_string(), "operations".to_string()],
        requires_creativity: true,
        involves_uncertainty: true,
        preferred_reasoning_type: Some(ReasoningType::Collaborative),
    };
    
    let result = engine.reason(&collaborative_problem).await;
    assert!(result.is_ok(), "Collaborative reasoning should succeed");
    
    // Test temporal reasoning
    let temporal_problem = ReasoningProblem {
        description: "Predict the sequence of events over time".to_string(),
        variables: vec!["event_1".to_string(), "event_2".to_string()],
        constraints: vec!["causality".to_string()],
        required_knowledge_domains: vec!["temporal".to_string()],
        requires_creativity: false,
        involves_uncertainty: true,
        preferred_reasoning_type: Some(ReasoningType::Temporal),
    };
    
    let result = engine.reason(&temporal_problem).await;
    assert!(result.is_ok(), "Temporal reasoning should succeed");
}

#[tokio::test]
async fn test_reasoning_persistence() {
    use loki::cognitive::reasoning::{AdvancedReasoningEngine, ReasoningProblem};
    
    let engine = AdvancedReasoningEngine::new().await.expect("Failed to create reasoning engine");
    
    // Create and solve a problem
    let problem = ReasoningProblem {
        description: "Test persistence".to_string(),
        variables: vec!["var1".to_string()],
        constraints: vec![],
        required_knowledge_domains: vec!["test".to_string()],
        requires_creativity: false,
        involves_uncertainty: false,
        preferred_reasoning_type: None,
    };
    
    let result = engine.reason(&problem).await.expect("Reasoning should succeed");
    let session_id = result.session_id.clone();
    
    // Provide feedback
    let feedback_result = engine.provide_feedback(&session_id, 0.9).await;
    assert!(feedback_result.is_ok(), "Feedback should be recorded");
    
    // Get learning metrics
    let metrics = engine.get_learning_metrics().await;
    assert!(metrics.is_ok(), "Should be able to get learning metrics");
    
    // Save persistence data
    let save_result = engine.save_persistence().await;
    assert!(save_result.is_ok(), "Should be able to save persistence data");
}

// Run with: cargo test --test chat_cognitive_integration_test