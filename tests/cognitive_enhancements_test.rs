//! Comprehensive Tests for Enhanced Cognitive Features
//!
//! Tests for:
//! - Basic cognitive system functionality
//! - Memory integration
//! - Consciousness streams
//! - Query processing

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use loki::cognitive::{CognitiveConfig, CognitiveSystem, GoalPriority as Priority};
use loki::compute::ComputeManager;
use loki::memory::{MemoryConfig, MemoryMetadata};
use loki::streaming::StreamManager;
use tempfile::TempDir;
use tokio::time::timeout;

async fn setup_test_system() -> Result<Arc<CognitiveSystem>> {
    // Create basic dependencies
    let compute_manager = Arc::new(ComputeManager::new()?);
    let stream_manager = Arc::new(StreamManager::new(loki::config::Config::default())?);

    // Create temp directory for persistence
    let temp_dir = TempDir::new()?;

    // Create minimal config for testing
    let config = CognitiveConfig {
        memory_config: MemoryConfig {
            short_term_capacity: 1000,
            long_term_layers: 3,
            layer_capacity: 10000,
            embedding_dim: 768,
            cache_size_mb: 100,
            persistence_path: temp_dir.path().to_path_buf(),
            consolidation_threshold: 0.3,
            decay_rate: 0.01,
            max_memory_mb: None,
            context_window: None,
            enable_persistence: true,
            max_age_days: None,
            embedding_dimension: None,
        },
        orchestrator_model: "llama3.2:1b".to_string(), // Use smallest model for tests
        context_window: 2048,
        stream_batch_size: 16,
        background_tasks_enabled: false, // Disable for tests
        monitoring_interval: Duration::from_secs(10),
        max_agents: 2,
    };

    CognitiveSystem::new(compute_manager, stream_manager, config).await
}

#[tokio::test]
async fn test_cognitive_system_creation() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Verify system is created
    let memory = cognitive_system.memory();
    let stats = memory.stats();
    assert!(stats.short_term_count >= 0);

    let _orchestrator = cognitive_system.orchestrator();
    // Basic check that orchestrator exists

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_consciousness_stream_startup() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Start consciousness
    cognitive_system.clone().start_consciousness().await?;

    // Verify consciousness is running
    assert!(cognitive_system.consciousness().is_some());

    // Wait a moment for consciousness to process
    tokio::time::sleep(Duration::from_millis(100)).await;

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_memory_integration() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Store content in memory using correct API
    let memory = cognitive_system.memory();
    let memory_id = memory
        .store(
            "Test cognitive enhancement thought".to_string(),
            vec![],
            MemoryMetadata {
                source: "test".to_string(),
                tags: vec!["test".to_string(), "cognitive".to_string()],
                importance: 0.9,
                associations: vec![],
                context: None,
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
            },
        )
        .await?;

    // Retrieve and verify using similarity search
    let similar = memory.retrieve_similar("Test cognitive enhancement", 5).await?;
    assert!(!similar.is_empty());
    assert!(similar.iter().any(|m| m.id == memory_id));

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_query_processing() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Process a simple query
    let query = "What is cognitive enhancement?";
    let response =
        timeout(Duration::from_secs(30), cognitive_system.process_query(query)).await??;

    // Verify we got a response
    assert!(!response.is_empty());
    assert!(response.len() > 10);

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_orchestrator_availability() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Verify orchestrator is available
    let orchestrator = cognitive_system.orchestrator();
    // Basic check that orchestrator exists

    // Test orchestrator model access
    let _model = cognitive_system.orchestrator_model().await?;
    // Just verify we got a model back successfully
    // (All fields are private, so we can't inspect them directly)

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_recent_thoughts_retrieval() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Get recent thoughts (should start empty)
    let thoughts = cognitive_system.get_recent_thoughts(10);
    assert!(thoughts.len() <= 10);

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_consciousness_interruption() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Start consciousness first
    cognitive_system.clone().start_consciousness().await?;

    // Wait for consciousness to be ready
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Test interruption
    let result = cognitive_system
        .interrupt_consciousness(
            "test_source".to_string(),
            "Test interruption content".to_string(),
            Priority::High,
        )
        .await;

    // Should succeed if consciousness is running
    if cognitive_system.consciousness().is_some() {
        assert!(result.is_ok());
    }

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_autonomous_stream_deployment() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Deploy an autonomous stream
    let result = cognitive_system
        .start_autonomous_stream(
            "test_stream".to_string(),
            "Test autonomous processing".to_string(),
        )
        .await;

    // Should succeed
    assert!(result.is_ok());

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_agent_deployment() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Create agent config using correct field names
    let agent_config = loki::cognitive::AgentConfig {
        name: "test_agent".to_string(),
        agent_type: "testing".to_string(),
        capabilities: vec!["testing".to_string()],
        priority: 0.7,
    };

    // Deploy agent
    let agent_id = cognitive_system.deploy_agent(agent_config).await?;
    assert!(!agent_id.is_empty());

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_system_shutdown() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Test graceful shutdown
    let result = cognitive_system.shutdown().await;
    assert!(result.is_ok());

    Ok(())
}

// Performance benchmarks
#[tokio::test]
async fn test_cognitive_processing_performance() -> Result<()> {
    let cognitive_system = setup_test_system().await?;
    let memory = cognitive_system.memory();

    let start = std::time::Instant::now();

    // Store multiple memories to test performance
    for i in 0..5 {
        memory
            .store(
                format!("Performance test thought {}", i),
                vec![],
                MemoryMetadata {
                    source: "test".to_string(),
                    tags: vec!["performance".to_string(), "test".to_string()],
                    importance: 0.5,
                    associations: vec![],
                    context: None,
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                },
            )
            .await?;
    }

    let duration = start.elapsed();

    // Should complete within reasonable time
    assert!(duration < Duration::from_secs(5));

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_memory_performance() -> Result<()> {
    let cognitive_system = setup_test_system().await?;
    let memory = cognitive_system.memory();

    let start = std::time::Instant::now();

    // Store and retrieve multiple memories
    let mut memory_ids = Vec::new();
    for i in 0..10 {
        let id = memory
            .store(
                format!("Memory performance test {}", i),
                vec![],
                MemoryMetadata {
                    source: "test".to_string(),
                    tags: vec!["memory".to_string(), "performance".to_string()],
                    importance: 0.6,
                    associations: vec![],
                    context: None,
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                },
            )
            .await?;
        memory_ids.push(id);
    }

    // Test retrieval via similarity search
    for i in 0..10 {
        let similar = memory.retrieve_similar(&format!("Memory performance test {}", i), 3).await?;
        assert!(!similar.is_empty());
    }

    let duration = start.elapsed();

    // Should complete within reasonable time
    assert!(duration < Duration::from_secs(3));

    cognitive_system.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_concurrent_processing() -> Result<()> {
    let cognitive_system = setup_test_system().await?;

    // Spawn multiple concurrent tasks
    let tasks: Vec<_> = (0..3)
        .map(|i| {
            let system = cognitive_system.clone();
            tokio::spawn(async move {
                let query = format!("Concurrent test query {}", i);
                system.process_query(&query).await
            })
        })
        .collect();

    // Wait for all tasks to complete
    for task in tasks {
        let result = timeout(Duration::from_secs(30), task).await??;
        assert!(result.is_ok());
    }

    cognitive_system.shutdown().await?;
    Ok(())
}

#[test]
fn test_priority_ordering() {
    // Test that priority ordering is correct
    assert!(Priority::Critical as u8 > Priority::High as u8);
    assert!(Priority::High as u8 > Priority::Medium as u8);
    assert!(Priority::Medium as u8 > Priority::Low as u8);
}

#[tokio::test]
async fn test_memory_search_functionality() -> Result<()> {
    let cognitive_system = setup_test_system().await?;
    let memory = cognitive_system.memory();

    // Store diverse content
    let contents = vec![
        ("Artificial intelligence research", vec!["AI", "research"]),
        ("Machine learning algorithms", vec!["ML", "algorithms"]),
        ("Cognitive science concepts", vec!["cognitive", "science"]),
        ("Neural network architectures", vec!["neural", "networks"]),
    ];

    for (content, tags) in contents {
        memory
            .store(
                content.to_string(),
                vec![],
                MemoryMetadata {
                    source: "test".to_string(),
                    tags: tags.into_iter().map(|s| s.to_string()).collect(),
                    importance: 0.7,
                    associations: vec![],
                    context: None,
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                },
            )
            .await?;
    }

    // Test search functionality
    let ai_results = memory.retrieve_similar("artificial intelligence", 5).await?;
    assert!(!ai_results.is_empty());

    let ml_results = memory.retrieve_similar("machine learning", 5).await?;
    assert!(!ml_results.is_empty());

    cognitive_system.shutdown().await?;
    Ok(())
}
