//! Unit tests for task extraction and multi-agent execution
//! 
//! Tests individual components of the multi-agent task system

use loki::tui::{
    task_decomposer::{TaskDecomposer, DecompositionStrategy, SubTask},
    agent_task_mapper::{AgentTaskMapper, AgentType},
    agent_stream::{AgentStreamManager, StreamStatus},
    task_progress_aggregator::TaskProgressAggregator,
};
use loki::tools::task_management::TaskPriority;

#[tokio::test]
async fn test_task_decomposer_feature() {
    let decomposer = TaskDecomposer::new();
    
    // Test feature decomposition
    let task = "Build a user authentication system with OAuth support";
    let subtasks = decomposer.decompose(task, DecompositionStrategy::Feature).await.unwrap();
    
    // Should decompose into multiple subtasks
    assert!(subtasks.len() >= 3);
    
    // Check subtask properties
    for subtask in &subtasks {
        assert!(!subtask.description.is_empty());
        assert!(!subtask.id.is_empty());
        assert_eq!(subtask.priority, TaskPriority::Medium);
    }
    
    // Should include OAuth-related tasks
    let has_oauth = subtasks.iter().any(|t| t.description.contains("OAuth"));
    assert!(has_oauth, "Should include OAuth-related subtasks");
}

#[tokio::test]
async fn test_task_decomposer_bugfix() {
    let decomposer = TaskDecomposer::new();
    
    // Test bug fix decomposition
    let task = "Fix memory leak in the background processor";
    let subtasks = decomposer.decompose(task, DecompositionStrategy::BugFix).await.unwrap();
    
    // Bug fixes should have investigation and fix steps
    assert!(subtasks.len() >= 2);
    
    // Should have investigation step
    let has_investigation = subtasks.iter().any(|t| 
        t.description.contains("Investigate") || 
        t.description.contains("Analyze") ||
        t.description.contains("Identify")
    );
    assert!(has_investigation, "Should have investigation step");
    
    // Should have fix step
    let has_fix = subtasks.iter().any(|t| 
        t.description.contains("Fix") || 
        t.description.contains("Apply") ||
        t.description.contains("Implement")
    );
    assert!(has_fix, "Should have fix step");
}

#[tokio::test]
async fn test_agent_mapper_basic() {
    let mapper = AgentTaskMapper::new();
    
    // Test basic mappings
    let test_cases = vec![
        ("Write unit tests for the API", AgentType::QA),
        ("Fix SQL injection vulnerability", AgentType::Security),
        ("Optimize database queries", AgentType::Performance),
        ("Update API documentation", AgentType::Documentation),
        ("Analyze user behavior data", AgentType::Data),
        ("Create React component", AgentType::Frontend),
        ("Implement REST API endpoint", AgentType::Backend),
    ];
    
    for (task, expected_type) in test_cases {
        let agents = mapper.map_task_to_agents(task).await.unwrap();
        assert!(!agents.is_empty(), "Should map to at least one agent for: {}", task);
        
        let primary = &agents[0];
        assert_eq!(primary.agent_type, expected_type, 
            "Task '{}' should map to {:?}", task, expected_type);
        assert!(primary.score.total() > 0.5, "Primary agent should have high confidence");
    }
}

#[tokio::test]
async fn test_agent_mapper_multiple_agents() {
    let mapper = AgentTaskMapper::new();
    
    // Test task that should map to multiple agents
    let task = "Create a secure API endpoint with proper tests and documentation";
    let agents = mapper.map_task_to_agents(task).await.unwrap();
    
    // Should map to multiple agents
    assert!(agents.len() >= 2, "Complex task should map to multiple agents");
    
    // Check agent types
    let agent_types: Vec<_> = agents.iter().map(|a| &a.agent_type).collect();
    assert!(agent_types.contains(&AgentType::Backend));
    assert!(agent_types.contains(&AgentType::QA) || agent_types.contains(&AgentType::Documentation));
}

#[tokio::test]
async fn test_progress_aggregator() {
    let aggregator = TaskProgressAggregator::new();
    
    // Register subtasks
    let parent_id = "main-task";
    let subtasks = vec![
        ("subtask-1", "Task 1", 0.3),
        ("subtask-2", "Task 2", 0.3),
        ("subtask-3", "Task 3", 0.4),
    ];
    
    for (id, desc, weight) in &subtasks {
        aggregator.register_subtask(parent_id, id.to_string(), desc.to_string(), *weight).await;
    }
    
    // Initial progress should be 0
    let progress = aggregator.get_total_progress().await;
    assert_eq!(progress, 0.0);
    
    // Update subtask progress
    aggregator.update_subtask_progress("subtask-1", 1.0).await;
    let progress = aggregator.get_total_progress().await;
    assert!((progress - 0.3).abs() < 0.01, "Progress should be 30%");
    
    // Update more subtasks
    aggregator.update_subtask_progress("subtask-2", 0.5).await;
    let progress = aggregator.get_total_progress().await;
    assert!((progress - 0.45).abs() < 0.01, "Progress should be 45%");
    
    // Complete all tasks
    aggregator.update_subtask_progress("subtask-2", 1.0).await;
    aggregator.update_subtask_progress("subtask-3", 1.0).await;
    let progress = aggregator.get_total_progress().await;
    assert!((progress - 1.0).abs() < 0.01, "Progress should be 100%");
}

#[tokio::test]
async fn test_agent_stream_lifecycle() {
    use tokio::sync::RwLock;
    use std::sync::Arc;
    
    let manager = Arc::new(RwLock::new(AgentStreamManager::new()));
    
    // Create stream
    let stream_id = {
        let mut mgr = manager.write().await;
        mgr.create_stream(
            "test-stream".to_string(),
            AgentType::Backend,
            "Test task".to_string(),
        ).await.unwrap()
    };
    
    // Check initial status
    {
        let mgr = manager.read().await;
        let stream = mgr.get_stream(&stream_id).unwrap();
        assert_eq!(stream.status, StreamStatus::Active);
        assert_eq!(stream.agent_type, AgentType::Backend);
    }
    
    // Send update
    {
        let mut mgr = manager.write().await;
        mgr.send_update(
            &stream_id,
            loki::tui::agent_stream::StreamUpdate::Progress {
                percentage: 0.5,
                message: "Halfway done".to_string(),
            },
        ).await.unwrap();
    }
    
    // Complete stream
    {
        let mut mgr = manager.write().await;
        mgr.complete_stream(&stream_id).await.unwrap();
    }
    
    // Check final status
    {
        let mgr = manager.read().await;
        let stream = mgr.get_stream(&stream_id).unwrap();
        assert_eq!(stream.status, StreamStatus::Completed);
    }
}