//! Tests for task priority mapping functionality

#[cfg(test)]
mod tests {
    use super::super::processing::message_processor::MessageProcessor;
    use super::super::state::ChatState;
    use super::super::orchestration::OrchestrationManager;
    use super::super::agents::AgentManager;
    use crate::tui::nlp::core::orchestrator::{ExtractedTask, ExtractedTaskType, TaskPriority as NlpTaskPriority};
    use crate::tools::task_management::TaskPriority as ToolTaskPriority;
    use std::sync::Arc;
    use tokio::sync::{RwLock, mpsc};
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_priority_mapping() {
        // Create components
        let chat_state = Arc::new(RwLock::new(ChatState::new()));
        let orchestration_manager = Arc::new(RwLock::new(OrchestrationManager::new()));
        let agent_manager = Arc::new(RwLock::new(AgentManager::new()));
        let (tx, _rx) = mpsc::channel(100);
        
        let processor = MessageProcessor::new(
            chat_state,
            orchestration_manager,
            agent_manager,
            tx,
        );
        
        // Test priority mappings
        let test_cases = vec![
            (NlpTaskPriority::Critical, ToolTaskPriority::Critical),
            (NlpTaskPriority::High, ToolTaskPriority::High),
            (NlpTaskPriority::Medium, ToolTaskPriority::Medium),
            (NlpTaskPriority::Low, ToolTaskPriority::Low),
        ];
        
        for (nlp_priority, expected_tool_priority) in test_cases {
            let mapped = processor.map_task_priority(&nlp_priority);
            assert_eq!(mapped, expected_tool_priority, 
                "Priority mapping failed for {:?}", nlp_priority);
        }
    }
    
    #[tokio::test]
    async fn test_urgency_detection() {
        let chat_state = Arc::new(RwLock::new(ChatState::new()));
        let orchestration_manager = Arc::new(RwLock::new(OrchestrationManager::new()));
        let agent_manager = Arc::new(RwLock::new(AgentManager::new()));
        let (tx, _rx) = mpsc::channel(100);
        
        let processor = MessageProcessor::new(
            chat_state,
            orchestration_manager,
            agent_manager,
            tx,
        );
        
        // Test urgency detection
        let test_cases = vec![
            ("this is urgent!", "high"),
            ("please do this asap", "high"),
            ("critical issue needs fixing immediately", "high"),
            ("this is important", "medium"),
            ("please handle this soon", "medium"),
            ("high priority task", "medium"),
            ("whenever you can", "low"),
            ("eventually we should", "low"),
            ("low priority item", "low"),
            ("normal request", "normal"),
        ];
        
        for (input, expected_urgency) in test_cases {
            let detected = processor.detect_urgency_level(&input.to_lowercase());
            assert_eq!(detected, expected_urgency, 
                "Urgency detection failed for: {}", input);
        }
    }
    
    #[tokio::test]
    async fn test_task_context_building() {
        let chat_state = Arc::new(RwLock::new(ChatState::new()));
        let orchestration_manager = Arc::new(RwLock::new(OrchestrationManager::new()));
        let agent_manager = Arc::new(RwLock::new(AgentManager::new()));
        let (tx, _rx) = mpsc::channel(100);
        
        let processor = MessageProcessor::new(
            chat_state,
            orchestration_manager,
            agent_manager,
            tx,
        );
        
        // Test task with high confidence and effort
        let task1 = ExtractedTask {
            description: "Implement user authentication".to_string(),
            priority: NlpTaskPriority::High,
            task_type: ExtractedTaskType::CodingTask,
            confidence: 0.95,
            estimated_effort: Some(Duration::from_secs(3600)), // 1 hour
            dependencies: vec!["database setup".to_string()],
        };
        
        let context1 = processor.build_task_context(&task1);
        assert!(context1.contains("Type: CodingTask"));
        assert!(context1.contains("✅ High confidence: 95%"));
        assert!(context1.contains("Estimated time: 60 minutes"));
        assert!(context1.contains("Dependencies: database setup"));
        
        // Test task with low confidence
        let task2 = ExtractedTask {
            description: "Maybe refactor something".to_string(),
            priority: NlpTaskPriority::Low,
            task_type: ExtractedTaskType::GeneralTask,
            confidence: 0.6,
            estimated_effort: Some(Duration::from_secs(7200)), // 2 hours
            dependencies: vec![],
        };
        
        let context2 = processor.build_task_context(&task2);
        assert!(context2.contains("⚠️ Low confidence: 60%"));
        assert!(context2.contains("Estimated time: 2.0 hours"));
        assert!(!context2.contains("Dependencies")); // No dependencies
    }
    
    #[tokio::test]
    async fn test_task_request_creation_with_urgency() {
        let chat_state = Arc::new(RwLock::new(ChatState::new()));
        let orchestration_manager = Arc::new(RwLock::new(OrchestrationManager::new()));
        let agent_manager = Arc::new(RwLock::new(AgentManager::new()));
        let (tx, _rx) = mpsc::channel(100);
        
        let mut processor = MessageProcessor::new(
            chat_state,
            orchestration_manager,
            agent_manager,
            tx,
        );
        
        // Test urgent code generation request
        let urgent_request = processor.create_task_request(
            "urgent: implement the authentication system immediately"
        ).await.unwrap();
        
        assert_eq!(urgent_request.constraints.priority, "high");
        
        // Test normal request
        let normal_request = processor.create_task_request(
            "please analyze the code structure"
        ).await.unwrap();
        
        assert_eq!(normal_request.constraints.priority, "normal");
        
        // Test low priority request
        let low_request = processor.create_task_request(
            "whenever you can, add some documentation"
        ).await.unwrap();
        
        assert_eq!(low_request.constraints.priority, "low");
    }
}