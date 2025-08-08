use anyhow::{Context, Result};
use loki::memory::{CognitiveMemory, MemoryConfig, MemoryMetadata};
use loki::tools::{DiscordConfig, EmailConfig, SlackConfig};
use tempfile::TempDir;

/// Helper function to create a test memory system
async fn create_test_memory() -> Result<CognitiveMemory> {
    let temp_dir = TempDir::new().context("Failed to create temporary directory for test")?;
    let test_path = temp_dir.path().to_path_buf();

    let mut config = MemoryConfig::default();
    config.persistence_path = test_path.join("test_memory");
    config.cache_size_mb = 64; // Smaller cache for tests
    config.short_term_capacity = 10;
    config.long_term_layers = 2;
    config.layer_capacity = 20;

    let memory = CognitiveMemory::new(config).await?;

    // Keep temp_dir alive by forgetting it (similar to new_for_test)
    std::mem::forget(temp_dir);

    Ok(memory)
}

/// Test basic memory functionality
#[tokio::test]
async fn test_cognitive_memory_basic() -> Result<()> {
    let memory = create_test_memory().await?;

    // Store a simple memory
    let metadata = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["basic".to_string()],
        importance: 0.7,
        associations: vec![],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
        category: "test".to_string(),
        timestamp: chrono::Utc::now(),
        expiration: None,
    };

    let memory_id = memory
        .store(
            "This is a basic test memory".to_string(),
            vec!["test context".to_string()],
            metadata,
        )
        .await?;

    // Verify retrieval
    let results = memory.retrieve_similar("basic test", 5).await?;
    assert!(!results.is_empty());

    let found = results.iter().any(|m| m.content.contains("basic test memory"));
    assert!(found);

    Ok(())
}

/// Test memory statistics
#[tokio::test]
async fn test_memory_stats() -> Result<()> {
    let memory = create_test_memory().await?;

    // Initial stats should be empty
    let stats = memory.stats();
    assert_eq!(stats.short_term_count, 0);
    assert!(stats.long_term_counts.iter().all(|&count| count == 0));

    // Store some memories
    for i in 0..3 {
        let metadata = MemoryMetadata {
            source: "test".to_string(),
            tags: vec![format!("tag_{}", i)],
            importance: 0.5,
            associations: vec![],
            context: None,
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
            category: "test".to_string(),
            timestamp: chrono::Utc::now(),
            expiration: None,
        };

        memory.store(format!("Test memory {}", i), vec![], metadata).await?;
    }

    // Check updated stats
    let stats = memory.stats();
    let total = stats.short_term_count + stats.long_term_counts.iter().sum::<usize>();
    assert_eq!(total, 3);

    Ok(())
}

/// Test SlackConfig creation and defaults
#[test]
fn test_slack_config_creation() {
    let config = SlackConfig::default();

    // Check essential fields exist
    assert!(config.bot_token.is_empty()); // Should be empty in default
    assert_eq!(config.monitored_channels, vec!["general".to_string()]);
    assert!(config.enable_dms);
    assert_eq!(config.awareness_level, 0.7);
    assert!(config.enable_threads);
}

/// Test DiscordConfig creation and defaults
#[test]
fn test_discord_config_creation() {
    let config = DiscordConfig::default();

    // Check essential fields exist
    assert!(config.bot_token.is_empty()); // Should be empty in default
    assert!(config.application_id.is_empty()); // Should be empty in default
    assert!(config.monitored_guilds.is_empty());
    assert!(config.enable_dms);
    assert_eq!(config.awareness_level, 0.7);
    assert!(!config.enable_moderation); // Should be disabled by default
    assert_eq!(config.max_message_length, 2000); // Discord limit
}

/// Test EmailConfig creation and defaults
#[test]
fn test_email_config_creation() {
    let config = EmailConfig::default();

    // Check essential fields exist
    assert!(config.imap_server.is_empty()); // Should be empty in default
    assert_eq!(config.imap_port, 993);
    assert!(config.imap_use_tls);
    assert_eq!(config.smtp_port, 587);
    assert!(config.smtp_use_tls);
    assert_eq!(config.monitored_folders, vec!["INBOX".to_string()]);
    assert!(!config.auto_reply_enabled); // Should be disabled by default
    assert_eq!(config.cognitive_awareness_level, 0.8);
}

/// Test custom SlackConfig creation
#[test]
fn test_slack_config_custom() {
    let config = SlackConfig {
        bot_token: "test-token".to_string(),
        workspace_domain: "test-workspace".to_string(),
        monitored_channels: vec!["general".to_string(), "dev".to_string()],
        enable_dms: false,
        awareness_level: 0.9,
        ..SlackConfig::default()
    };

    assert_eq!(config.bot_token, "test-token");
    assert_eq!(config.workspace_domain, "test-workspace");
    assert_eq!(config.monitored_channels.len(), 2);
    assert!(!config.enable_dms);
    assert_eq!(config.awareness_level, 0.9);
}

/// Test custom DiscordConfig creation
#[test]
fn test_discord_config_custom() {
    let config = DiscordConfig {
        bot_token: "test-discord-token".to_string(),
        application_id: "123456789".to_string(),
        monitored_guilds: vec!["guild1".to_string(), "guild2".to_string()],
        enable_moderation: true,
        awareness_level: 0.5,
        ..DiscordConfig::default()
    };

    assert_eq!(config.bot_token, "test-discord-token");
    assert_eq!(config.application_id, "123456789");
    assert_eq!(config.monitored_guilds.len(), 2);
    assert!(config.enable_moderation);
    assert_eq!(config.awareness_level, 0.5);
}

/// Test custom EmailConfig creation
#[test]
fn test_email_config_custom() {
    let config = EmailConfig {
        imap_server: "imap.example.com".to_string(),
        imap_username: "test@example.com".to_string(),
        smtp_server: "smtp.example.com".to_string(),
        smtp_username: "test@example.com".to_string(),
        auto_reply_enabled: true,
        cognitive_awareness_level: 0.9,
        ..EmailConfig::default()
    };

    assert_eq!(config.imap_server, "imap.example.com");
    assert_eq!(config.imap_username, "test@example.com");
    assert_eq!(config.smtp_server, "smtp.example.com");
    assert_eq!(config.smtp_username, "test@example.com");
    assert!(config.auto_reply_enabled);
    assert_eq!(config.cognitive_awareness_level, 0.9);
}

/// Test memory retrieval by key
#[tokio::test]
async fn test_memory_key_retrieval() -> Result<()> {
    let memory = create_test_memory().await?;

    let metadata = MemoryMetadata {
        source: "test".to_string(),
        tags: vec!["searchable".to_string()],
        importance: 0.8,
        associations: vec![],
        context: None,
        created_at: chrono::Utc::now(),
        accessed_count: 0,
        last_accessed: None,
        version: 1,
        category: "test".to_string(),
        timestamp: chrono::Utc::now(),
        expiration: None,
    };

    memory
        .store(
            "This memory contains a unique_keyword that should be findable".to_string(),
            vec![],
            metadata,
        )
        .await?;

    // Test key-based retrieval
    let result = memory.retrieve_by_key("unique_keyword").await?;
    assert!(result.is_some());

    let retrieved = result.unwrap();
    assert!(retrieved.content.contains("unique_keyword"));

    Ok(())
}

/// Test memory with different importance levels
#[tokio::test]
async fn test_memory_importance_levels() -> Result<()> {
    let memory = create_test_memory().await?;

    // Store memories with different importance
    let importance_levels = vec![0.1, 0.5, 0.9];

    for (i, importance) in importance_levels.iter().enumerate() {
        let metadata = MemoryMetadata {
            source: "test".to_string(),
            tags: vec![format!("importance_{}", i)],
            importance: *importance,
            associations: vec![],
            context: None,
            created_at: chrono::Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
            category: "test".to_string(),
            timestamp: chrono::Utc::now(),
            expiration: None,
        };

        memory
            .store(format!("Memory {} with importance {}", i, importance), vec![], metadata)
            .await?;
    }

    // Retrieve and check they're all stored
    let results = memory.retrieve_similar("Memory", 10).await?;
    assert_eq!(results.len(), 3);

    // Check that high importance memories are included
    let high_importance = results.iter().any(|m| m.metadata.importance > 0.8);
    assert!(high_importance);

    Ok(())
}
