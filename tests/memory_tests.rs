use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use loki::memory::{CognitiveMemory, LayerType, MemoryConfig, MemoryId, MemoryMetadata};
use tempfile::TempDir;
use tokio::time::sleep;

/// Create test memory configuration
fn create_test_config(temp_dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        short_term_capacity: 100,
        long_term_layers: 3,
        layer_capacity: 200,
        embedding_dim: 384,
        cache_size_mb: 64,
        persistence_path: temp_dir.path().join("memory"),
        consolidation_threshold: 0.8,
        decay_rate: 0.95,
        max_memory_mb: None,
        context_window: None,
        enable_persistence: true,
        max_age_days: None,
        embedding_dimension: None,
    }
}

#[tokio::test]
async fn test_memory_creation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);

    let memory = CognitiveMemory::new(config).await?;

    // Check initial stats
    let stats = memory.stats();
    assert_eq!(stats.short_term_count, 0);
    assert!(stats.long_term_counts.iter().all(|&count| count == 0));

    Ok(())
}

#[tokio::test]
async fn test_memory_storage_and_retrieval() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);
    let memory = CognitiveMemory::new(config).await?;

    // Store memories
    let metadata1 = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["science".to_string(), "physics".to_string()],
        importance: 0.8,
        associations: vec![], // Empty for now, would need actual MemoryIds
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory
        .store(
            "The Earth orbits around the Sun".to_string(),
            vec!["Space facts".to_string()],
            metadata1,
        )
        .await?;

    let metadata2 = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["science".to_string(), "biology".to_string()],
        importance: 0.7,
        associations: vec![], // Empty for now, would need actual MemoryIds
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory
        .store(
            "Photosynthesis converts light into energy".to_string(),
            vec!["Biology facts".to_string()],
            metadata2,
        )
        .await?;

    // Retrieve similar memories
    let results = memory.retrieve_similar("solar system", 5).await?;
    assert!(!results.is_empty());

    // The memory about Earth orbiting the Sun should be relevant
    let has_earth_memory =
        results.iter().any(|m| m.content.contains("Earth") && m.content.contains("Sun"));
    assert!(has_earth_memory);

    // Check stats
    let stats = memory.stats();
    let total_memories = stats.short_term_count + stats.long_term_counts.iter().sum::<usize>();
    assert_eq!(total_memories, 2);

    Ok(())
}

#[tokio::test]
async fn test_memory_layers() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);
    let memory = CognitiveMemory::new(config).await?;

    // Store memories with different importance levels
    for i in 0..10 {
        let importance = (i as f32) / 10.0;
        let metadata = MemoryMetadata {
            source: "test".to_string(),
            tags: vec![format!("level_{}", i)],
            importance,
            associations: vec![],
            context: None,
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
        };

        memory.store(format!("Memory with importance {}", importance), vec![], metadata).await?;
    }

    // Check that memories are stored (consolidation happens automatically)
    let stats = memory.stats();
    let total_memories = stats.short_term_count + stats.long_term_counts.iter().sum::<usize>();
    assert!(total_memories > 0);

    // After consolidation, important memories should be preserved
    let all_memories = memory.retrieve_similar("Memory", 20).await?;

    // Higher importance memories should be retained
    let high_importance_count =
        all_memories.iter().filter(|m| m.metadata.importance >= 0.7).count();

    assert!(high_importance_count > 0);

    Ok(())
}

#[tokio::test]
async fn test_memory_tags_and_associations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);
    let memory = CognitiveMemory::new(config).await?;

    // Store memories with specific tags
    let metadata = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["programming".to_string(), "rust".to_string()],
        importance: 0.9,
        associations: vec![], // Fixed: Empty associations for now
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory
        .store(
            "Rust provides memory safety without garbage collection".to_string(),
            vec![],
            metadata,
        )
        .await?;

    // Search by tag-related content
    let results = memory.retrieve_similar("programming language", 5).await?;

    // Should find the Rust memory
    let has_rust_memory = results.iter().any(|m| {
        m.content.contains("Rust") && m.metadata.tags.contains(&"programming".to_string())
    });
    assert!(has_rust_memory);

    Ok(())
}

#[tokio::test]
async fn test_memory_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);

    // First session - store memories
    {
        let memory = CognitiveMemory::new(config.clone()).await?;

        let metadata = MemoryMetadata {
            source: "test".to_string(),
            tags: vec!["persistent".to_string()],
            importance: 0.9,
            associations: vec![],
            context: None,
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
        };

        memory
            .store("This memory should persist across sessions".to_string(), vec![], metadata)
            .await?;

        // Memory persists automatically
    }

    // Second session - retrieve memories
    {
        let memory = CognitiveMemory::new(config).await?;

        // Should be able to retrieve the persisted memory
        let results = memory.retrieve_similar("persist", 5).await?;

        let has_persistent_memory =
            results.iter().any(|m| m.content.contains("persist across sessions"));
        assert!(has_persistent_memory);
    }

    Ok(())
}

#[tokio::test]
async fn test_memory_retrieval_by_key() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);
    let memory = CognitiveMemory::new(config).await?;

    // Store a memory with specific content
    let metadata = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["unique".to_string()],
        importance: 0.8,
        associations: vec![],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory
        .store("This is a unique test memory with specific keywords".to_string(), vec![], metadata)
        .await?;

    // Test key-based retrieval
    let result = memory.retrieve_by_key("unique test memory").await?;
    assert!(result.is_some());

    let retrieved = result.unwrap();
    assert!(retrieved.content.contains("unique test memory"));

    Ok(())
}

#[tokio::test]
async fn test_memory_importance_levels() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);

    let memory = CognitiveMemory::new(config).await?;

    // Store memories with varying importance
    let low_importance = MemoryMetadata {
        source: "test".to_string(),
        tags: vec![],
        importance: 0.2, // Below threshold
        associations: vec![],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory.store("Low importance memory".to_string(), vec![], low_importance).await?;

    let high_importance = MemoryMetadata {
        source: "test".to_string(),
        tags: vec![],
        importance: 0.8, // Above threshold
        associations: vec![],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory.store("High importance memory".to_string(), vec![], high_importance).await?;

    // Both memories should be stored regardless of importance

    let results = memory.retrieve_similar("memory", 10).await?;

    // High importance memory should definitely be there
    let has_high = results.iter().any(|m| m.content.contains("High importance"));
    assert!(has_high);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_memory_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);
    let memory = Arc::new(CognitiveMemory::new(config).await?);

    // Spawn multiple tasks storing memories concurrently
    let mut handles = vec![];

    for i in 0..5 {
        let memory_clone = memory.clone();
        let handle = tokio::spawn(async move {
            for j in 0..3 {
                let metadata = MemoryMetadata {
                    source: format!("task_{}", i),
                    tags: vec![format!("concurrent_{}", i)],
                    importance: 0.5,
                    associations: vec![],
                    context: None,
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                };

                memory_clone
                    .store(format!("Memory {} from task {}", j, i), vec![], metadata)
                    .await
                    .unwrap();

                sleep(Duration::from_millis(10)).await;
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await?;
    }

    // Verify all memories were stored
    let stats = memory.stats();
    let total_memories = stats.short_term_count + stats.long_term_counts.iter().sum::<usize>();
    assert_eq!(total_memories, 15); // 5 tasks * 3 memories each

    Ok(())
}

#[tokio::test]
async fn test_memory_search_accuracy() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);
    let memory = CognitiveMemory::new(config).await?;

    // Store diverse memories
    let memories = vec![
        ("The cat sat on the mat", vec!["animals", "furniture"]),
        ("Dogs are loyal companions", vec!["animals", "pets"]),
        ("The quick brown fox jumps", vec!["animals", "action"]),
        ("Programming requires logical thinking", vec!["technology", "skills"]),
        ("Rust is a systems programming language", vec!["technology", "rust"]),
    ];

    for (content, tags) in memories {
        let metadata = MemoryMetadata {
            source: "test".to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            importance: 0.7,
            associations: vec![],
            context: None,
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
        };

        memory.store(content.to_string(), vec![], metadata).await?;
    }

    // Test various searches
    let animal_results = memory.retrieve_similar("pets and animals", 3).await?;
    let animal_count =
        animal_results.iter().filter(|m| m.metadata.tags.contains(&"animals".to_string())).count();
    assert!(animal_count >= 2);

    let tech_results = memory.retrieve_similar("programming languages", 3).await?;
    let tech_count =
        tech_results.iter().filter(|m| m.metadata.tags.contains(&"technology".to_string())).count();
    assert!(tech_count >= 1);

    Ok(())
}

#[tokio::test]
async fn test_large_memory_storage() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config = create_test_config(&temp_dir);

    let memory = CognitiveMemory::new(config).await?;

    // Store a large memory
    let large_content = "Lorem ipsum ".repeat(1000); // ~11KB of text
    let metadata = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["large".to_string()],
        importance: 0.8,
        associations: vec![],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    memory.store(large_content.clone(), vec![], metadata).await?;

    // Retrieve and verify
    let results = memory.retrieve_similar("Lorem ipsum", 1).await?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content.len(), large_content.len());

    Ok(())
}

#[test]
fn test_memory_metadata_creation() {
    let metadata = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["tag1".to_string(), "tag2".to_string()],
        importance: 0.75,
        associations: vec![MemoryId::new()],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
    };

    assert_eq!(metadata.source, "test");
    assert_eq!(metadata.tags.len(), 2);
    assert_eq!(metadata.importance, 0.75);
    assert_eq!(metadata.associations.len(), 1);
}

#[test]
fn test_layer_type_creation() {
    // Ensure layer types work correctly
    let short_term = LayerType::ShortTerm;
    let long_term_0 = LayerType::LongTerm(0);
    let long_term_1 = LayerType::LongTerm(1);

    // Test equality
    assert_eq!(short_term, LayerType::ShortTerm);
    assert_eq!(long_term_0, LayerType::LongTerm(0));
    assert_ne!(long_term_0, long_term_1);
}
