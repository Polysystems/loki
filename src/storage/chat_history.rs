//! Chat History Persistence
//!
//! This module provides persistent storage for chat conversations,
//! enabling history search, context retrieval, and continuity across sessions.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::{Pool, Sqlite, SqlitePool};
use std::path::PathBuf;
use tracing::{info, debug, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Chat history storage manager
pub struct ChatHistoryStorage {
    /// SQLite connection pool
    pool: Pool<Sqlite>,
    
    /// Storage configuration
    config: ChatHistoryConfig,
    
    /// In-memory cache of recent conversations
    cache: Vec<ConversationSummary>,
}

/// Configuration for chat history storage
#[derive(Debug, Clone)]
pub struct ChatHistoryConfig {
    /// Database file path
    pub db_path: PathBuf,
    
    /// Maximum cache size
    pub cache_size: usize,
    
    /// Auto-save interval in seconds
    pub auto_save_interval: u64,
    
    /// Maximum history retention in days
    pub retention_days: u32,
}

impl Default for ChatHistoryConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("loki")
            .join("chat");
        
        Self {
            db_path: data_dir.join("chat_history.db"),
            cache_size: 100,
            auto_save_interval: 30,
            retention_days: 365,
        }
    }
}

/// Conversation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Unique conversation ID
    pub id: String,
    
    /// Conversation title
    pub title: String,
    
    /// Start time
    pub started_at: DateTime<Utc>,
    
    /// Last updated time
    pub updated_at: DateTime<Utc>,
    
    /// Model used
    pub model: String,
    
    /// Conversation metadata
    pub metadata: serde_json::Value,
    
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
}

/// Individual chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message ID
    pub id: String,
    
    /// Conversation ID
    pub conversation_id: String,
    
    /// Message role (user, assistant, system)
    pub role: String,
    
    /// Message content
    pub content: String,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Token count
    pub token_count: Option<i32>,
    
    /// Message metadata
    pub metadata: Option<serde_json::Value>,
}

/// Conversation summary for quick access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: i32,
    pub model: String,
}

/// Search result for chat history
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub conversation_id: String,
    pub message_id: String,
    pub content: String,
    pub relevance_score: f32,
    pub timestamp: DateTime<Utc>,
}

impl ChatHistoryStorage {
    /// Create new chat history storage
    pub async fn new(config: ChatHistoryConfig) -> Result<Self> {
        info!("📝 Initializing chat history storage");
        
        // Ensure directory exists
        if let Some(parent) = config.db_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Failed to create chat directory")?;
        }
        
        // Create database connection
        let connection_string = format!("sqlite://{}?mode=rwc", config.db_path.display());
        let pool = SqlitePool::connect(&connection_string).await
            .context("Failed to connect to chat database")?;
        
        // Run migrations
        Self::run_migrations(&pool).await?;
        
        let mut storage = Self {
            pool,
            config,
            cache: Vec::new(),
        };
        
        // Load recent conversations into cache
        storage.refresh_cache().await?;
        
        info!("✅ Chat history storage initialized");
        Ok(storage)
    }
    
    /// Run database migrations
    async fn run_migrations(pool: &Pool<Sqlite>) -> Result<()> {
        debug!("Running chat history migrations");
        
        // Create conversations table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                model TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        // Create messages table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                token_count INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        // Create indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
            .execute(pool)
            .await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            .execute(pool)
            .await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)")
            .execute(pool)
            .await?;
        
        // Create FTS table for search
        sqlx::query(
            r#"
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                message_id,
                content,
                tokenize='trigram'
            )
            "#,
        )
        .execute(pool)
        .await?;
        
        debug!("Chat history migrations completed");
        Ok(())
    }
    
    /// Start a new conversation
    pub async fn start_conversation(
        &mut self,
        title: String,
        model: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        sqlx::query(
            r#"
            INSERT INTO conversations (id, title, started_at, updated_at, model, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&id)
        .bind(&title)
        .bind(now)
        .bind(now)
        .bind(&model)
        .bind(metadata.as_ref().map(|m| m.to_string()))
        .execute(&self.pool)
        .await?;
        
        // Refresh cache
        self.refresh_cache().await?;
        
        info!("Started new conversation: {} ({})", title, id);
        Ok(id)
    }
    
    /// Add a message to a conversation
    pub async fn add_message(
        &mut self,
        conversation_id: String,
        role: String,
        content: String,
        token_count: Option<i32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String> {
        let message_id = Uuid::new_v4().to_string();
        let timestamp = Utc::now();
        
        // Insert message
        sqlx::query(
            r#"
            INSERT INTO messages (id, conversation_id, role, content, timestamp, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&message_id)
        .bind(&conversation_id)
        .bind(&role)
        .bind(&content)
        .bind(timestamp)
        .bind(token_count)
        .bind(metadata.as_ref().map(|m| m.to_string()))
        .execute(&self.pool)
        .await?;
        
        // Update conversation timestamp
        sqlx::query(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
        )
        .bind(timestamp)
        .bind(&conversation_id)
        .execute(&self.pool)
        .await?;
        
        // Update FTS index
        sqlx::query(
            "INSERT INTO messages_fts (message_id, content) VALUES (?, ?)",
        )
        .bind(&message_id)
        .bind(&content)
        .execute(&self.pool)
        .await?;
        
        debug!("Added message {} to conversation {}", message_id, conversation_id);
        Ok(message_id)
    }
    
    /// Get a full conversation with all messages
    pub async fn get_conversation(&self, conversation_id: &str) -> Result<Option<Conversation>> {
        // Get conversation metadata
        let conv_row = sqlx::query_as::<_, (String, String, DateTime<Utc>, DateTime<Utc>, String, Option<String>)>(
            r#"
            SELECT id, title, started_at, updated_at, model, metadata
            FROM conversations
            WHERE id = ?
            "#,
        )
        .bind(conversation_id)
        .fetch_optional(&self.pool)
        .await?;
        
        let (id, title, started_at, updated_at, model, metadata_str) = match conv_row {
            Some(row) => row,
            None => return Ok(None),
        };
        
        // Get all messages
        let messages = sqlx::query_as::<_, (String, String, String, String, DateTime<Utc>, Option<i32>, Option<String>)>(
            r#"
            SELECT id, conversation_id, role, content, timestamp, token_count, metadata
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            "#,
        )
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        
        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(id, conv_id, role, content, timestamp, token_count, metadata_str)| {
                ChatMessage {
                    id,
                    conversation_id: conv_id,
                    role,
                    content,
                    timestamp,
                    token_count,
                    metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                }
            })
            .collect();
        
        let metadata = metadata_str
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        
        Ok(Some(Conversation {
            id,
            title,
            started_at,
            updated_at,
            model,
            metadata,
            messages: chat_messages,
        }))
    }
    
    /// List recent conversations
    pub async fn list_conversations(&self, limit: i64) -> Result<Vec<ConversationSummary>> {
        let summaries = sqlx::query_as::<_, (String, String, DateTime<Utc>, DateTime<Utc>, i32, String)>(
            r#"
            SELECT 
                c.id,
                c.title,
                c.started_at,
                c.updated_at,
                COUNT(m.id) as message_count,
                c.model
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        
        Ok(summaries
            .into_iter()
            .map(|(id, title, started_at, updated_at, message_count, model)| {
                ConversationSummary {
                    id,
                    title,
                    started_at,
                    updated_at,
                    message_count,
                    model,
                }
            })
            .collect())
    }
    
    /// Search through chat history
    pub async fn search(&self, query: &str, limit: i64) -> Result<Vec<SearchResult>> {
        let results = sqlx::query_as::<_, (String, String, String, f32)>(
            r#"
            SELECT 
                m.conversation_id,
                m.message_id,
                m.content,
                bm25(messages_fts) as score
            FROM messages_fts fts
            JOIN messages m ON fts.message_id = m.id
            WHERE messages_fts MATCH ?
            ORDER BY score
            LIMIT ?
            "#,
        )
        .bind(query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        
        let mut search_results = Vec::new();
        
        for (conv_id, msg_id, content, score) in results {
            // Get timestamp for the message
            let timestamp: DateTime<Utc> = sqlx::query_scalar(
                "SELECT timestamp FROM messages WHERE id = ?",
            )
            .bind(&msg_id)
            .fetch_one(&self.pool)
            .await?;
            
            search_results.push(SearchResult {
                conversation_id: conv_id,
                message_id: msg_id,
                content,
                relevance_score: score,
                timestamp,
            });
        }
        
        Ok(search_results)
    }
    
    /// Delete a conversation
    pub async fn delete_conversation(&mut self, conversation_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM conversations WHERE id = ?")
            .bind(conversation_id)
            .execute(&self.pool)
            .await?;
        
        // Refresh cache
        self.refresh_cache().await?;
        
        info!("Deleted conversation: {}", conversation_id);
        Ok(())
    }
    
    /// Clean up old conversations based on retention policy
    pub async fn cleanup_old_conversations(&mut self) -> Result<u64> {
        let cutoff_date = Utc::now() - chrono::Duration::days(self.config.retention_days as i64);
        
        let result = sqlx::query(
            "DELETE FROM conversations WHERE updated_at < ?",
        )
        .bind(cutoff_date)
        .execute(&self.pool)
        .await?;
        
        let deleted = result.rows_affected();
        
        if deleted > 0 {
            info!("Cleaned up {} old conversations", deleted);
            self.refresh_cache().await?;
        }
        
        Ok(deleted)
    }
    
    /// Export conversation to JSON
    pub async fn export_conversation(&self, conversation_id: &str) -> Result<String> {
        let conversation = self.get_conversation(conversation_id).await?
            .context("Conversation not found")?;
        
        Ok(serde_json::to_string_pretty(&conversation)?)
    }
    
    /// Import conversation from JSON
    pub async fn import_conversation(&mut self, json: &str) -> Result<String> {
        let conversation: Conversation = serde_json::from_str(json)?;
        
        // Insert conversation
        sqlx::query(
            r#"
            INSERT INTO conversations (id, title, started_at, updated_at, model, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&conversation.id)
        .bind(&conversation.title)
        .bind(conversation.started_at)
        .bind(conversation.updated_at)
        .bind(&conversation.model)
        .bind(serde_json::to_string(&conversation.metadata)?)
        .execute(&self.pool)
        .await?;
        
        // Insert messages
        for msg in &conversation.messages {
            sqlx::query(
                r#"
                INSERT INTO messages (id, conversation_id, role, content, timestamp, token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                "#,
            )
            .bind(&msg.id)
            .bind(&msg.conversation_id)
            .bind(&msg.role)
            .bind(&msg.content)
            .bind(msg.timestamp)
            .bind(msg.token_count)
            .bind(msg.metadata.as_ref().map(|m| m.to_string()))
            .execute(&self.pool)
            .await?;
            
            // Update FTS
            sqlx::query(
                "INSERT INTO messages_fts (message_id, content) VALUES (?, ?)",
            )
            .bind(&msg.id)
            .bind(&msg.content)
            .execute(&self.pool)
            .await?;
        }
        
        self.refresh_cache().await?;
        
        info!("Imported conversation: {} ({})", conversation.title, conversation.id);
        Ok(conversation.id)
    }
    
    /// Refresh the in-memory cache
    async fn refresh_cache(&mut self) -> Result<()> {
        self.cache = self.list_conversations(self.config.cache_size as i64).await?;
        debug!("Refreshed cache with {} conversations", self.cache.len());
        Ok(())
    }
    
    /// Get conversation summaries from cache
    pub fn get_cached_summaries(&self) -> &[ConversationSummary] {
        &self.cache
    }
    
    /// Get statistics about chat history
    pub async fn get_statistics(&self) -> Result<serde_json::Value> {
        let total_conversations: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM conversations",
        )
        .fetch_one(&self.pool)
        .await?;
        
        let total_messages: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM messages",
        )
        .fetch_one(&self.pool)
        .await?;
        
        let total_tokens: Option<i64> = sqlx::query_scalar(
            "SELECT SUM(token_count) FROM messages WHERE token_count IS NOT NULL",
        )
        .fetch_optional(&self.pool)
        .await?;
        
        let models: Vec<(String, i64)> = sqlx::query_as(
            r#"
            SELECT model, COUNT(*) as count
            FROM conversations
            GROUP BY model
            ORDER BY count DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(serde_json::json!({
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_tokens": total_tokens.unwrap_or(0),
            "models": models.into_iter().map(|(model, count)| {
                serde_json::json!({
                    "model": model,
                    "count": count
                })
            }).collect::<Vec<_>>(),
            "cache_size": self.cache.len(),
            "retention_days": self.config.retention_days,
        }))
    }
}