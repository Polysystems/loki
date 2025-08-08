//! Integration tests for multi-agent task execution
//! 
//! Tests the complete flow of:
//! 1. Natural language task extraction
//! 2. Task decomposition into subtasks
//! 3. Agent assignment and parallel execution
//! 4. Progress aggregation

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use loki::{
    tui::{
        natural_language_orchestrator::{NaturalLanguageOrchestrator, IntentType},
        task_decomposer::{TaskDecomposer, DecompositionStrategy},
        agent_task_mapper::{AgentTaskMapper, AgentType},
        agent_stream::{AgentStreamManager, AgentStream},
        task_progress_aggregator::TaskProgressAggregator,
    },
    cognitive::CognitiveSystem,
    memory::CognitiveMemory,
    models::{
        orchestrator::ModelOrchestrator,
        multi_agent_orchestrator::MultiAgentOrchestrator,
    },
    tools::{IntelligentToolManager, task_management::TaskManager},
    safety::ActionValidator,
};

#[tokio::test]
async fn test_extract_tasks_from_natural_language() {
    // Initialize components
    let cognitive_system = Arc::new(CognitiveSystem::new(Default::default()).await.unwrap());
    let memory = Arc::new(CognitiveMemory::new(None).await.unwrap());
    let model_orchestrator = Arc::new(ModelOrchestrator::new(Default::default()).await.unwrap());
    let multi_agent = Arc::new(MultiAgentOrchestrator::new(Default::default()).await.unwrap());
    let tool_manager = Arc::new(IntelligentToolManager::new(Default::default()).await.unwrap());
    let task_manager = Arc::new(TaskManager::new(Default::default()).await.unwrap());
    let mcp_client = Arc::new(Default::default());
    let safety_validator = Arc::new(ActionValidator::new(Default::default()).unwrap());
    
    let orchestrator = NaturalLanguageOrchestrator::new(
        cognitive_system,
        memory,
        model_orchestrator,
        multi_agent,
        tool_manager,
        task_manager,
        mcp_client,
        safety_validator,
        Default::default(),
    ).await.unwrap();
    
    // Test task extraction
    let test_cases = vec![
        (
            "Fix the bug in the authentication module and add tests",
            vec!["Fix the bug in the authentication module", "add tests"],
            IntentType::FixIssue,
        ),
        (
            "Analyze the performance of the database queries and optimize them",
            vec!["Analyze the performance of the database queries", "optimize them"],
            IntentType::AnalyzeData,
        ),
        (
            "Create a new feature for user notifications with email and SMS support",
            vec!["Create a new feature for user notifications", "email support", "SMS support"],
            IntentType::CreateFeature,
        ),
    ];
    
    for (input, expected_tasks, expected_intent) in test_cases {
        let extracted = orchestrator.extract_tasks(input).await;
        
        // Check intent
        assert_eq!(extracted.intent, expected_intent);
        
        // Check that we extracted at least the expected number of tasks
        assert!(extracted.subtasks.len() >= expected_tasks.len(), 
            "Expected at least {} tasks, got {}", expected_tasks.len(), extracted.subtasks.len());
        
        // Verify task content (roughly)
        for expected in expected_tasks {
            let found = extracted.subtasks.iter()
                .any(|task| task.description.to_lowercase().contains(&expected.to_lowercase()));
            assert!(found, "Expected to find task containing '{}' in {:?}", 
                expected, extracted.subtasks);
        }
    }
}

#[tokio::test]
async fn test_task_decomposition() {
    let decomposer = TaskDecomposer::new();
    
    // Test feature decomposition
    let feature_task = "Build a user authentication system with OAuth support";
    let subtasks = decomposer.decompose(feature_task, DecompositionStrategy::Feature).await.unwrap();
    
    assert!(subtasks.len() >= 3, "Feature should decompose into multiple subtasks");
    assert!(subtasks.iter().any(|t| t.description.contains("OAuth")));
    assert!(subtasks.iter().any(|t| t.description.contains("authentication")));
    
    // Test bug fix decomposition
    let bug_task = "Fix memory leak in the background processor";
    let subtasks = decomposer.decompose(bug_task, DecompositionStrategy::BugFix).await.unwrap();
    
    assert!(subtasks.len() >= 2, "Bug fix should have investigation and fix steps");
    assert!(subtasks.iter().any(|t| t.description.contains("Investigate") || t.description.contains("Analyze")));
    assert!(subtasks.iter().any(|t| t.description.contains("Fix") || t.description.contains("Apply")));
}

#[tokio::test]
async fn test_agent_task_mapping() {
    let mapper = AgentTaskMapper::new();
    
    // Test different task types
    let test_cases = vec![
        ("Write unit tests for the API endpoints", AgentType::QA),
        ("Fix the SQL injection vulnerability", AgentType::Security),
        ("Optimize the image processing algorithm", AgentType::Performance),
        ("Add documentation for the new API", AgentType::Documentation),
        ("Analyze user behavior patterns", AgentType::Data),
        ("Create a new dashboard component", AgentType::Frontend),
        ("Implement the payment processing service", AgentType::Backend),
    ];
    
    for (task, expected_agent) in test_cases {
        let agents = mapper.map_task_to_agents(task).await.unwrap();
        assert!(!agents.is_empty(), "Should map to at least one agent");
        
        let primary_agent = &agents[0];
        assert_eq!(primary_agent.agent_type, expected_agent, 
            "Task '{}' should map to {:?}", task, expected_agent);
        assert!(primary_agent.score.total() > 0.5, 
            "Primary agent should have high confidence");
    }
}

#[tokio::test]
async fn test_parallel_agent_execution() {
    let stream_manager = Arc::new(RwLock::new(AgentStreamManager::new()));
    let progress_aggregator = Arc::new(TaskProgressAggregator::new());
    
    // Create multiple agent streams
    let agents = vec![
        ("backend-agent", AgentType::Backend),
        ("frontend-agent", AgentType::Frontend),
        ("test-agent", AgentType::QA),
    ];
    
    let mut stream_ids = vec![];
    
    // Start streams
    {
        let mut manager = stream_manager.write().await;
        for (id, agent_type) in &agents {
            let stream_id = manager.create_stream(
                id.to_string(),
                agent_type.clone(),
                "Test task".to_string(),
            ).await.unwrap();
            stream_ids.push(stream_id);
        }
    }
    
    // Simulate parallel execution
    let mut handles = vec![];
    
    for (i, stream_id) in stream_ids.iter().enumerate() {
        let manager = stream_manager.clone();
        let aggregator = progress_aggregator.clone();
        let sid = stream_id.clone();
        let task_id = format!("task-{}", i);
        
        let handle = tokio::spawn(async move {
            // Simulate work
            for step in 0..5 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                
                // Update progress
                let progress = (step + 1) as f32 / 5.0;
                aggregator.update_subtask_progress(&task_id, progress).await;
                
                // Send update through stream
                let mut manager = manager.write().await;
                manager.send_update(
                    &sid,
                    loki::tui::agent_stream::StreamUpdate::Progress {
                        percentage: progress,
                        message: format!("Step {} of 5", step + 1),
                    },
                ).await.unwrap();
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Check aggregated progress
    let total_progress = progress_aggregator.get_total_progress().await;
    assert!((total_progress - 1.0).abs() < 0.01, "All tasks should be complete");
    
    // Check stream states
    {
        let manager = stream_manager.read().await;
        for stream_id in &stream_ids {
            let stream = manager.get_stream(stream_id).unwrap();
            assert_eq!(stream.status, loki::tui::agent_stream::StreamStatus::Completed);
        }
    }
}

#[tokio::test]
async fn test_end_to_end_multi_agent_task() {
    // This test simulates the complete flow from natural language to multi-agent execution
    
    // Initialize all components
    let cognitive_system = Arc::new(CognitiveSystem::new(Default::default()).await.unwrap());
    let memory = Arc::new(CognitiveMemory::new(None).await.unwrap());
    let model_orchestrator = Arc::new(ModelOrchestrator::new(Default::default()).await.unwrap());
    let multi_agent = Arc::new(MultiAgentOrchestrator::new(Default::default()).await.unwrap());
    let tool_manager = Arc::new(IntelligentToolManager::new(Default::default()).await.unwrap());
    let task_manager = Arc::new(TaskManager::new(Default::default()).await.unwrap());
    let mcp_client = Arc::new(Default::default());
    let safety_validator = Arc::new(ActionValidator::new(Default::default()).unwrap());
    
    let orchestrator = Arc::new(NaturalLanguageOrchestrator::new(
        cognitive_system,
        memory,
        model_orchestrator,
        multi_agent,
        tool_manager,
        task_manager,
        mcp_client,
        safety_validator,
        Default::default(),
    ).await.unwrap());
    
    let decomposer = Arc::new(TaskDecomposer::new());
    let mapper = Arc::new(AgentTaskMapper::new());
    let stream_manager = Arc::new(RwLock::new(AgentStreamManager::new()));
    let progress_aggregator = Arc::new(TaskProgressAggregator::new());
    
    // Input task
    let input = "Create a REST API with authentication, add comprehensive tests, and document the endpoints";
    
    // Step 1: Extract tasks
    let extracted = orchestrator.extract_tasks(input).await;
    assert!(!extracted.subtasks.is_empty());
    
    // Step 2: Decompose each task
    let mut all_subtasks = vec![];
    for task in &extracted.subtasks {
        let strategy = match extracted.intent {
            IntentType::CreateFeature => DecompositionStrategy::Feature,
            IntentType::FixIssue => DecompositionStrategy::BugFix,
            IntentType::AnalyzeData => DecompositionStrategy::Analysis,
            _ => DecompositionStrategy::Generic,
        };
        
        let subtasks = decomposer.decompose(&task.description, strategy).await.unwrap();
        all_subtasks.extend(subtasks);
    }
    
    assert!(all_subtasks.len() > extracted.subtasks.len(), 
        "Decomposition should create more granular tasks");
    
    // Step 3: Map to agents and create streams
    let mut agent_assignments = vec![];
    {
        let mut manager = stream_manager.write().await;
        
        for (i, subtask) in all_subtasks.iter().enumerate() {
            let agents = mapper.map_task_to_agents(&subtask.description).await.unwrap();
            if let Some(primary_agent) = agents.first() {
                let stream_id = manager.create_stream(
                    format!("stream-{}", i),
                    primary_agent.agent_type.clone(),
                    subtask.description.clone(),
                ).await.unwrap();
                
                agent_assignments.push((stream_id, subtask.clone(), primary_agent.clone()));
            }
        }
    }
    
    assert!(!agent_assignments.is_empty(), "Should have agent assignments");
    
    // Step 4: Simulate parallel execution with progress tracking
    let mut handles = vec![];
    
    for (stream_id, subtask, agent) in agent_assignments {
        let manager = stream_manager.clone();
        let aggregator = progress_aggregator.clone();
        let task_id = subtask.id.clone();
        
        let handle = tokio::spawn(async move {
            // Register task with aggregator
            aggregator.register_subtask(
                "main-task",
                task_id.clone(),
                subtask.description.clone(),
                1.0 / 10.0, // Equal weight for simplicity
            ).await;
            
            // Simulate execution
            for step in 0..10 {
                tokio::time::sleep(Duration::from_millis(5)).await;
                
                let progress = (step + 1) as f32 / 10.0;
                aggregator.update_subtask_progress(&task_id, progress).await;
                
                // Send stream update
                let mut manager = manager.write().await;
                let _ = manager.send_update(
                    &stream_id,
                    loki::tui::agent_stream::StreamUpdate::Progress {
                        percentage: progress,
                        message: format!("Processing with {:?} agent", agent.agent_type),
                    },
                ).await;
            }
            
            // Mark as complete
            let mut manager = manager.write().await;
            let _ = manager.complete_stream(&stream_id).await;
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify final state
    let total_progress = progress_aggregator.get_total_progress().await;
    assert!((total_progress - 1.0).abs() < 0.01, 
        "All tasks should be complete, got {}", total_progress);
    
    // Check all streams are completed
    {
        let manager = stream_manager.read().await;
        let all_completed = manager.get_all_streams().iter()
            .all(|s| s.status == loki::tui::agent_stream::StreamStatus::Completed);
        assert!(all_completed, "All streams should be completed");
    }
}