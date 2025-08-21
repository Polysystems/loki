//! Storage-aware context for chat system
//!
//! This module integrates the chat system with persistent storage,
//! providing context from API keys, chat history, and configuration.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result, Context as AnyhowContext};
use tracing::{debug, info, warn};
use std::collections::HashMap;

use crate::tui::bridges::StorageBridge;
use crate::storage::{ConversationSummary, StoryExecution};

/// Storage context for chat system
pub struct ChatStorageContext {
    /// Storage bridge for persistent data
    storage_bridge: Arc<StorageBridge>,
    
    /// Current conversation ID
    current_conversation_id: Arc<RwLock<Option<String>>>,
    
    /// Cached API keys for quick access
    cached_api_keys: Arc<RwLock<HashMap<String, String>>>,
    
    /// Recent conversation summaries
    recent_conversations: Arc<RwLock<Vec<ConversationSummary>>>,
    
    /// Story execution history
    story_executions: Arc<RwLock<Vec<StoryExecution>>>,
}

impl ChatStorageContext {
    /// Create a new storage context
    pub fn new(storage_bridge: Arc<StorageBridge>) -> Self {
        Self {
            storage_bridge,
            current_conversation_id: Arc::new(RwLock::new(None)),
            cached_api_keys: Arc::new(RwLock::new(HashMap::new())),
            recent_conversations: Arc::new(RwLock::new(Vec::new())),
            story_executions: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Initialize the storage context
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing chat storage context");
        
        // Load recent conversations
        self.refresh_recent_conversations().await?;
        
        // Load cached API keys if storage is unlocked
        self.refresh_api_keys().await?;
        
        info!("✅ Chat storage context initialized");
        Ok(())
    }
    
    /// Start a new conversation
    pub async fn start_conversation(&self, title: String, model: String) -> Result<String> {
        debug!("Starting new conversation: {}", title);
        
        let conversation_id = self.storage_bridge
            .start_conversation(title.clone(), model.clone())
            .await?;
        
        // Update current conversation
        let mut current = self.current_conversation_id.write().await;
        *current = Some(conversation_id.clone());
        
        // Refresh recent conversations
        self.refresh_recent_conversations().await?;
        
        info!("Started conversation: {} ({})", title, conversation_id);
        Ok(conversation_id)
    }
    
    /// Add a message to the current conversation
    pub async fn add_message(
        &self,
        role: String,
        content: String,
        token_count: Option<i32>,
    ) -> Result<()> {
        let current = self.current_conversation_id.read().await;
        
        if let Some(conversation_id) = current.as_ref() {
            let message_id = self.storage_bridge
                .add_chat_message(
                    conversation_id.clone(),
                    role.clone(),
                    content.clone(),
                    token_count,
                )
                .await?;
            
            debug!("Added message to conversation {}: {}", conversation_id, message_id);
        } else {
            // Auto-start a new conversation if none exists
            let conversation_id = self.start_conversation(
                "New Conversation".to_string(),
                "default".to_string(),
            ).await?;
            
            self.storage_bridge
                .add_chat_message(
                    conversation_id.clone(),
                    role,
                    content,
                    token_count,
                )
                .await?;
        }
        
        Ok(())
    }
    
    /// Get API key for a provider
    pub async fn get_api_key(&self, provider: &str) -> Result<Option<String>> {
        // Check cache first
        {
            let cache = self.cached_api_keys.read().await;
            if let Some(key) = cache.get(provider) {
                return Ok(Some(key.clone()));
            }
        }
        
        // Try to get from storage
        match self.storage_bridge.get_api_key(provider).await {
            Ok(Some(key)) => {
                // Update cache
                let mut cache = self.cached_api_keys.write().await;
                cache.insert(provider.to_string(), key.clone());
                Ok(Some(key))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                warn!("Failed to get API key for {}: {}", provider, e);
                Ok(None)
            }
        }
    }
    
    /// Get all available API keys
    pub async fn get_all_api_keys(&self) -> Result<HashMap<String, String>> {
        match self.storage_bridge.get_all_api_keys().await {
            Ok(keys) => {
                // Update cache
                let mut cache = self.cached_api_keys.write().await;
                *cache = keys.clone();
                Ok(keys)
            }
            Err(e) => {
                warn!("Failed to get API keys: {}", e);
                Ok(HashMap::new())
            }
        }
    }
    
    /// Refresh API keys cache
    async fn refresh_api_keys(&self) -> Result<()> {
        let keys = self.get_all_api_keys().await?;
        debug!("Refreshed {} API keys in cache", keys.len());
        Ok(())
    }
    
    /// Get recent conversations
    pub async fn get_recent_conversations(&self) -> Vec<ConversationSummary> {
        self.recent_conversations.read().await.clone()
    }
    
    /// Refresh recent conversations
    async fn refresh_recent_conversations(&self) -> Result<()> {
        let conversations = self.storage_bridge
            .get_recent_conversations(20)
            .await?;
        
        let mut recent = self.recent_conversations.write().await;
        *recent = conversations;
        
        Ok(())
    }
    
    /// Search chat history
    pub async fn search_history(&self, query: &str, limit: i64) -> Result<Vec<crate::storage::chat_history::SearchResult>> {
        self.storage_bridge.search_chat_history(query, limit).await
    }
    
    /// Get messages for a conversation
    pub async fn get_conversation_messages(&self, conversation_id: &str) -> Result<Vec<crate::storage::chat_history::ChatMessage>> {
        self.storage_bridge.get_conversation_messages(conversation_id).await
    }
    
    /// Switch to a different conversation
    pub async fn switch_conversation(&self, conversation_id: String) -> Result<()> {
        let mut current = self.current_conversation_id.write().await;
        *current = Some(conversation_id.clone());
        
        info!("Switched to conversation: {}", conversation_id);
        Ok(())
    }
    
    /// Get current conversation ID
    pub async fn get_current_conversation_id(&self) -> Option<String> {
        self.current_conversation_id.read().await.clone()
    }
    
    /// Clear conversation cache
    pub async fn clear_cache(&self) {
        let mut recent = self.recent_conversations.write().await;
        recent.clear();
        
        let mut cache = self.cached_api_keys.write().await;
        cache.clear();
        
        debug!("Cleared storage context cache");
    }
    
    /// Get database configuration for a backend
    pub async fn get_database_config(&self, backend: &str) -> Result<Option<HashMap<String, String>>> {
        self.storage_bridge.get_database_config(backend).await
    }
    
    /// Store database configuration
    pub async fn store_database_config(
        &self,
        backend: &str,
        config: HashMap<String, String>,
    ) -> Result<()> {
        self.storage_bridge.store_database_config(backend, config).await
    }
    
    /// Get storage status
    pub async fn get_storage_status(&self) -> crate::tui::bridges::StorageStatus {
        self.storage_bridge.get_status().await
    }
    
    /// Export chat context for backup
    pub async fn export_context(&self) -> Result<serde_json::Value> {
        let mut export = serde_json::Map::new();
        
        // Export current conversation ID
        let current = self.current_conversation_id.read().await;
        if let Some(id) = current.as_ref() {
            export.insert("current_conversation".to_string(), serde_json::Value::String(id.clone()));
        }
        
        // Export recent conversations
        let recent = self.recent_conversations.read().await;
        export.insert("recent_conversations".to_string(), serde_json::to_value(&*recent)?);
        
        // Export storage status
        let status = self.get_storage_status().await;
        export.insert("storage_status".to_string(), serde_json::to_value(&status)?);
        
        Ok(serde_json::Value::Object(export))
    }
    
    /// Import chat context from backup
    pub async fn import_context(&self, data: serde_json::Value) -> Result<()> {
        if let Some(obj) = data.as_object() {
            // Import current conversation ID
            if let Some(current_id) = obj.get("current_conversation")
                .and_then(|v| v.as_str()) {
                let mut current = self.current_conversation_id.write().await;
                *current = Some(current_id.to_string());
            }
            
            // Refresh data
            self.refresh_recent_conversations().await?;
            self.refresh_api_keys().await?;
        }
        
        Ok(())
    }
}

/// Builder for creating storage-aware chat context
pub struct ChatStorageContextBuilder {
    storage_bridge: Option<Arc<StorageBridge>>,
}

impl ChatStorageContextBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            storage_bridge: None,
        }
    }
    
    /// Set the storage bridge
    pub fn with_storage_bridge(mut self, bridge: Arc<StorageBridge>) -> Self {
        self.storage_bridge = Some(bridge);
        self
    }
    
    /// Build the storage context
    pub fn build(self) -> Result<ChatStorageContext> {
        let storage_bridge = self.storage_bridge
            .context("Storage bridge is required")?;
        
        Ok(ChatStorageContext::new(storage_bridge))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_chat_storage_context() {
        // This would require a mock storage bridge for testing
        // For now, just ensure the module compiles
    }
}