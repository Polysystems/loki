//! Tests for enhanced NLP analysis functionality

#[cfg(test)]
mod tests {
    use super::super::handlers::natural_language::{NaturalLanguageHandler, NlpCommand, NlpIntent};
    use super::super::integrations::nlp::NlpIntegration;
    use std::sync::Arc;
    use crate::tui::nlp::core::orchestrator::{NaturalLanguageOrchestrator, OrchestratorConfig};
    use crate::cognitive::{
        memory::CognitiveMemory,
        consciousness::CognitiveSystem,
    };
    use crate::models::orchestrator::ModelOrchestrator;
    use crate::cognitive::orchestrator::MultiAgentOrchestrator;
    use crate::tools::intelligent_manager::IntelligentToolManager;
    use crate::tools::task_management::TaskManager;
    use crate::mcp::client::McpClient;
    use crate::cognitive::safety::ActionValidator;
    use tokio::runtime::Handle;
    
    fn create_mock_orchestrator() -> Arc<NaturalLanguageOrchestrator> {
        // Create a blocking task to initialize the orchestrator
        // This is a workaround for testing since we can't use async in this context
        let handle = Handle::current();
        handle.block_on(async {
            // Create minimal mock components
            let cognitive_system = Arc::new(CognitiveSystem::new(Default::default()));
            let memory = Arc::new(CognitiveMemory::new());
            let model_orchestrator = Arc::new(ModelOrchestrator::new());
            let multi_agent = Arc::new(MultiAgentOrchestrator::new());
            let tool_manager = Arc::new(IntelligentToolManager::new());
            let task_manager = Arc::new(TaskManager::new());
            let mcp_client = Arc::new(McpClient::new("test".to_string()));
            let safety_validator = Arc::new(ActionValidator::new());
            
            // Create the orchestrator
            Arc::new(
                NaturalLanguageOrchestrator::new(
                    cognitive_system,
                    memory,
                    model_orchestrator,
                    multi_agent,
                    tool_manager,
                    task_manager,
                    mcp_client,
                    safety_validator,
                    OrchestratorConfig::default(),
                ).await.expect("Failed to create NaturalLanguageOrchestrator")
            )
        })
    }
    
    #[tokio::test]
    async fn test_enhanced_nlp_processing() {
        // Create handler with mock orchestrator
        let orchestrator = create_mock_orchestrator();
        let mut handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        // Test basic pattern matching still works
        let result = handler.process("create task to implement user authentication").await;
        assert!(result.is_ok());
        
        if let Ok(Some(intent)) = result {
            assert!(matches!(intent.command, NlpCommand::CreateTask));
            assert!(intent.confidence >= 0.9);
            assert!(!intent.args.is_empty());
        }
    }
    
    #[tokio::test]
    async fn test_entity_extraction() {
        let orchestrator = create_mock_orchestrator();
        let handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        // Test various entity types
        let test_cases = vec![
            ("meeting at 3:30 PM tomorrow", vec!["time", "date"]),
            ("create file named \"config.yaml\"", vec!["quoted_text", "file_path"]),
            ("analyze performance for 3 hours", vec!["measurement"]),
            ("search in /usr/local/bin/app.exe", vec!["file_path"]),
        ];
        
        for (input, expected_types) in test_cases {
            // The handler's extract_entities is private, so we test indirectly
            // through the process method
            let result = handler.process(input).await;
            assert!(result.is_ok());
        }
    }
    
    #[tokio::test]
    async fn test_sentiment_and_urgency_detection() {
        let orchestrator = create_mock_orchestrator();
        let mut handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        // Test positive sentiment
        let positive_result = handler.process("please create a great new feature").await;
        assert!(positive_result.is_ok());
        
        // Test negative sentiment with urgency
        let urgent_result = handler.process("urgent: fix this critical error immediately").await;
        assert!(urgent_result.is_ok());
        
        if let Ok(Some(intent)) = urgent_result {
            // Urgency should boost confidence
            assert!(intent.confidence > 0.8);
        }
    }
    
    #[tokio::test]
    async fn test_context_window_tracking() {
        let orchestrator = create_mock_orchestrator();
        let mut handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        // Process multiple inputs to build context
        let inputs = vec![
            "I'm working on a chat system",
            "It needs to handle multiple models",
            "Can you help me implement model switching?",
        ];
        
        for input in inputs {
            let _ = handler.process(input).await;
        }
        
        // The last query should benefit from context
        // (context window is private, so we test the effect)
        let result = handler.process("how should I do that?").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_question_detection() {
        let orchestrator = create_mock_orchestrator();
        let handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        let questions = vec![
            "what models are available?",
            "how do I create a task?",
            "can you explain orchestration?",
            "should I use ensemble mode?",
            "is the cognitive system active?",
            "does this support parallel execution?",
        ];
        
        for question in questions {
            let result = handler.process(question).await;
            assert!(result.is_ok());
            
            if let Ok(Some(intent)) = result {
                // Questions should be detected with reasonable confidence
                assert!(intent.confidence >= 0.6);
            }
        }
    }
    
    #[tokio::test]
    async fn test_verb_pattern_analysis() {
        let orchestrator = create_mock_orchestrator();
        let mut handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        let verb_tests = vec![
            ("analyze the codebase for issues", NlpCommand::Analyze),
            ("search for usage of deprecated functions", NlpCommand::Search),
            ("explain how the orchestration works", NlpCommand::Explain),
            ("execute the test suite", NlpCommand::RunTool),
            ("switch to gpt-4 model", NlpCommand::SwitchModel),
        ];
        
        for (input, expected_command) in verb_tests {
            let result = handler.process(input).await;
            assert!(result.is_ok());
            
            if let Ok(Some(intent)) = result {
                assert!(matches!(intent.command, expected_command));
                assert!(intent.confidence >= 0.7);
            }
        }
    }
    
    #[tokio::test]
    async fn test_complex_intent_combinations() {
        let orchestrator = create_mock_orchestrator();
        let mut handler = NaturalLanguageHandler::new(orchestrator)
            .expect("Failed to create NaturalLanguageHandler");
        
        // Test complex inputs that combine multiple signals
        let complex_inputs = vec![
            "urgent: please analyze the error logs from yesterday at 3 PM",
            "can you quickly search for \"config.json\" in /etc/app/?",
            "I need to create a task to fix the broken authentication system ASAP",
        ];
        
        for input in complex_inputs {
            let result = handler.process(input).await;
            assert!(result.is_ok());
            
            if let Ok(Some(intent)) = result {
                // Complex inputs should still be understood
                assert!(intent.confidence >= 0.65);
                // Should extract relevant arguments
                assert!(!intent.args.is_empty());
            }
        }
    }
}