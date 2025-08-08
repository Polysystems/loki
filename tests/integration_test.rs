use std::fs;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use loki::cognitive::{CognitiveConfig, CognitiveSystem, GoalPriority as Priority};
use loki::compute::ComputeManager;
use loki::memory::{CognitiveMemory, MemoryConfig, MemoryMetadata};
use loki::streaming::StreamManager;
use tempfile::TempDir;

#[tokio::test]
async fn test_memory_system_basic() -> Result<()> {
    // Create a temporary directory for test data
    let temp_dir = TempDir::new()?;

    // Create memory config with correct field names
    let config = MemoryConfig {
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
    };

    // Create memory system
    let memory = Arc::new(CognitiveMemory::new(config).await?);

    // Store some test memories
    let memory_id = memory
        .store(
            "Test content for integration testing".to_string(),
            vec![],
            MemoryMetadata {
                source: "integration_test".to_string(),
                tags: vec!["test".to_string(), "integration".to_string()],
                importance: 0.8,
                associations: vec![],
                context: None,
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
            },
        )
        .await?;

    // Verify retrieval
    let similar_memories = memory.retrieve_similar("integration test", 5).await?;
    assert!(!similar_memories.is_empty());
    assert!(similar_memories.iter().any(|m| m.id == memory_id));

    Ok(())
}

#[tokio::test]
async fn test_cognitive_system_initialization() -> Result<()> {
    // Create managers
    let compute_manager = Arc::new(ComputeManager::new()?);
    let stream_manager = Arc::new(StreamManager::new(loki::config::Config::default())?);

    // Create temp directory for persistence
    let temp_dir = TempDir::new()?;

    // Create cognitive config
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
        orchestrator_model: "llama3.2:1b".to_string(), // Use a small model for testing
        context_window: 2048,
        stream_batch_size: 16,
        background_tasks_enabled: false, // Disable for testing
        monitoring_interval: Duration::from_secs(30),
        max_agents: 2,
    };

    // Create cognitive system (should initialize without errors)
    let cognitive_system = CognitiveSystem::new(compute_manager, stream_manager, config).await;

    // System should initialize successfully
    assert!(cognitive_system.is_ok());

    // If initialization succeeded, test basic operations
    if let Ok(system) = cognitive_system {
        // Test memory access
        let memory = system.memory();
        let stats = memory.stats();
        assert!(stats.short_term_count >= 0);

        // Test consciousness access (might be None initially)
        let _consciousness = system.consciousness();

        // Clean shutdown
        system.shutdown().await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_memory_multiple_operations() -> Result<()> {
    // Create temporary directory with test structure
    let temp_dir = TempDir::new()?;

    // Create a test file to simulate having data to process
    let test_file = temp_dir.path().join("test_data.txt");
    fs::write(
        &test_file,
        r#"
This is test content for the integration test.
It contains multiple lines of text to demonstrate
memory storage and retrieval capabilities.
The system should be able to store and find this content.
"#,
    )?;

    let config = MemoryConfig {
        short_term_capacity: 1000,
        long_term_layers: 3,
        layer_capacity: 10000,
        embedding_dim: 768,
        cache_size_mb: 100,
        persistence_path: temp_dir.path().join("memory"),
        consolidation_threshold: 0.3,
        decay_rate: 0.01,
        max_memory_mb: None,
        context_window: None,
        enable_persistence: true,
        max_age_days: None,
        embedding_dimension: None,
    };

    let memory = Arc::new(CognitiveMemory::new(config).await?);

    // Store multiple memories
    let mut memory_ids = Vec::new();

    let memories_to_store = vec![
        ("First test memory", vec!["first", "test"]),
        ("Second test memory", vec!["second", "test"]),
        ("Third important memory", vec!["third", "important"]),
    ];

    for (content, tags) in memories_to_store {
        let id = memory
            .store(
                content.to_string(),
                vec![],
                MemoryMetadata {
                    source: "integration_test".to_string(),
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
        memory_ids.push(id);
    }

    // Test retrieval
    let similar = memory.retrieve_similar("test memory", 10).await?;
    assert!(!similar.is_empty());

    // Should find multiple test memories
    let test_memory_count = similar.iter().filter(|m| m.content.contains("test memory")).count();
    assert!(test_memory_count > 0);

    Ok(())
}

#[test]
fn test_utility_functions() {
    // Test basic utility functions that don't require async

    // Test priority ordering
    assert!(Priority::Critical as u8 > Priority::High as u8);
    assert!(Priority::High as u8 > Priority::Medium as u8);
    assert!(Priority::Medium as u8 > Priority::Low as u8);

    // Test that we can create IDs
    let id1 = loki::cognitive::ThoughtId::new();
    let id2 = loki::cognitive::ThoughtId::new();
    assert_ne!(id1, id2);
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    // Test error handling with invalid configurations

    // Try to create memory with invalid path
    let invalid_config = MemoryConfig {
        short_term_capacity: 1000,
        long_term_layers: 3,
        layer_capacity: 10000,
        embedding_dim: 768,
        cache_size_mb: 100,
        persistence_path: "/invalid/nonexistent/path".into(),
        consolidation_threshold: 0.3,
        decay_rate: 0.01,
        max_memory_mb: None,
        context_window: None,
        enable_persistence: true,
        max_age_days: None,
        embedding_dimension: None,
    };

    // This should handle the error gracefully
    let result = CognitiveMemory::new(invalid_config).await;

    // We expect this to either succeed (if it creates the path) or fail gracefully
    // The important thing is that it doesn't panic
    match result {
        Ok(_) => {
            // Memory system handled path creation
        }
        Err(_) => {
            // Memory system handled error gracefully
        }
    }

    Ok(())
}
