//! Storage Bridge - Connects secure storage to all tabs for persistent data
//!
//! This bridge enables secure storage and retrieval of API keys, database configs,
//! chat history, and other sensitive data across all TUI tabs.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result, Context};
use std::collections::HashMap;
use tracing::{info, debug, warn};

use crate::storage::{
    SecureStorage, SecureStorageConfig,
    ChatHistoryStorage, ChatHistoryConfig,
    ConversationSummary,
};

/// Storage bridge for cross-tab persistent data
pub struct StorageBridge {
    /// Event bridge for notifications
    event_bridge: Arc<super::EventBridge>,
    
    /// Secure storage for API keys and secrets
    secure_storage: Arc<RwLock<Option<SecureStorage>>>,
    
    /// Chat history storage
    chat_history: Arc<RwLock<Option<ChatHistoryStorage>>>,
    
    /// Database configurations cache
    db_configs: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
    
    /// Storage status
    status: Arc<RwLock<StorageStatus>>,
}

/// Storage system status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StorageStatus {
    pub secure_storage_unlocked: bool,
    pub chat_history_connected: bool,
    pub total_api_keys: usize,
    pub total_conversations: usize,
    pub last_error: Option<String>,
}

impl Default for StorageStatus {
    fn default() -> Self {
        Self {
            secure_storage_unlocked: false,
            chat_history_connected: false,
            total_api_keys: 0,
            total_conversations: 0,
            last_error: None,
        }
    }
}

impl StorageBridge {
    /// Create a new storage bridge
    pub fn new(event_bridge: Arc<super::EventBridge>) -> Self {
        Self {
            event_bridge,
            secure_storage: Arc::new(RwLock::new(None)),
            chat_history: Arc::new(RwLock::new(None)),
            db_configs: Arc::new(RwLock::new(HashMap::new())),
            status: Arc::new(RwLock::new(StorageStatus::default())),
        }
    }
    
    /// Initialize the storage bridge
    pub async fn initialize(&self) -> Result<()> {
        info!("🔧 Initializing storage bridge");
        
        // Initialize chat history (doesn't require password)
        self.initialize_chat_history().await?;
        
        // Try to initialize secure storage with default config
        let config = SecureStorageConfig::default();
        let storage = SecureStorage::new(config).await?;
        
        let mut secure_storage = self.secure_storage.write().await;
        *secure_storage = Some(storage);
        
        // Update status
        let mut status = self.status.write().await;
        status.chat_history_connected = true;
        
        info!("✅ Storage bridge initialized");
        Ok(())
    }
    
    /// Initialize chat history storage
    async fn initialize_chat_history(&self) -> Result<()> {
        debug!("Initializing chat history storage");
        
        let config = ChatHistoryConfig::default();
        let chat_storage = ChatHistoryStorage::new(config).await?;
        
        // Get initial statistics
        let stats = chat_storage.get_statistics().await?;
        let total_conversations = stats["total_conversations"].as_i64().unwrap_or(0) as usize;
        
        let mut history = self.chat_history.write().await;
        *history = Some(chat_storage);
        
        let mut status = self.status.write().await;
        status.chat_history_connected = true;
        status.total_conversations = total_conversations;
        
        debug!("Chat history initialized with {} conversations", total_conversations);
        Ok(())
    }
    
    /// Unlock secure storage with master password
    pub async fn unlock_storage(&self, password: &str) -> Result<()> {
        info!("🔓 Unlocking secure storage");
        
        let mut storage = self.secure_storage.write().await;
        let storage_instance = storage.as_mut()
            .context("Secure storage not initialized")?;
        
        storage_instance.unlock(password).await?;
        
        // Load API keys count
        let keys = storage_instance.list_secrets().await?;
        let api_key_count = keys.iter()
            .filter(|k| k.starts_with("api_key_"))
            .count();
        
        let mut status = self.status.write().await;
        status.secure_storage_unlocked = true;
        status.total_api_keys = api_key_count;
        
        info!("✅ Storage unlocked with {} API keys", api_key_count);
        
        // Notify other tabs
        self.event_bridge.publish(
            crate::tui::event_bus::SystemEvent::StorageUnlocked
        ).await?;
        
        Ok(())
    }
    
    /// Lock secure storage
    pub async fn lock_storage(&self) -> Result<()> {
        info!("🔒 Locking secure storage");
        
        let mut storage = self.secure_storage.write().await;
        if let Some(storage_instance) = storage.as_mut() {
            storage_instance.clear_cache();
        }
        
        let mut status = self.status.write().await;
        status.secure_storage_unlocked = false;
        status.total_api_keys = 0;
        
        // Notify other tabs
        self.event_bridge.publish(
            crate::tui::event_bus::SystemEvent::StorageLocked
        ).await?;
        
        Ok(())
    }
    
    /// Store an API key
    pub async fn store_api_key(&self, provider: &str, key: &str) -> Result<()> {
        let mut storage = self.secure_storage.write().await;
        let storage_instance = storage.as_mut()
            .context("Secure storage not initialized")?;
        
        storage_instance.store_api_key(provider, key).await?;
        
        let mut status = self.status.write().await;
        status.total_api_keys += 1;
        
        info!("Stored API key for {}", provider);
        Ok(())
    }
    
    /// Get an API key
    pub async fn get_api_key(&self, provider: &str) -> Result<Option<String>> {
        let storage = self.secure_storage.read().await;
        let storage_instance = storage.as_ref()
            .context("Secure storage not initialized")?;
        
        storage_instance.get_api_key(provider).await
    }
    
    /// Get all API keys
    pub async fn get_all_api_keys(&self) -> Result<HashMap<String, String>> {
        let storage = self.secure_storage.read().await;
        let storage_instance = storage.as_ref()
            .context("Secure storage not initialized")?;
        
        storage_instance.get_all_api_keys().await
    }
    
    /// Store database configuration
    pub async fn store_database_config(
        &self,
        backend: &str,
        config: HashMap<String, String>,
    ) -> Result<()> {
        // Store in secure storage
        let mut storage = self.secure_storage.write().await;
        if let Some(storage_instance) = storage.as_mut() {
            storage_instance.store_database_config(backend, config.clone()).await?;
        }
        
        // Update cache
        let mut configs = self.db_configs.write().await;
        configs.insert(backend.to_string(), config);
        
        info!("Stored database configuration for {}", backend);
        Ok(())
    }
    
    /// Get database configuration
    pub async fn get_database_config(
        &self,
        backend: &str,
    ) -> Result<Option<HashMap<String, String>>> {
        // Check cache first
        {
            let configs = self.db_configs.read().await;
            if let Some(config) = configs.get(backend) {
                return Ok(Some(config.clone()));
            }
        }
        
        // Try secure storage
        let storage = self.secure_storage.read().await;
        if let Some(storage_instance) = storage.as_ref() {
            if let Some(config) = storage_instance.get_database_config(backend).await? {
                // Update cache
                let mut configs = self.db_configs.write().await;
                configs.insert(backend.to_string(), config.clone());
                return Ok(Some(config));
            }
        }
        
        Ok(None)
    }
    
    /// Start a new chat conversation
    pub async fn start_conversation(
        &self,
        title: String,
        model: String,
    ) -> Result<String> {
        let mut history = self.chat_history.write().await;
        let chat_storage = history.as_mut()
            .context("Chat history not initialized")?;
        
        let id = chat_storage.start_conversation(title, model, None).await?;
        
        let mut status = self.status.write().await;
        status.total_conversations += 1;
        
        info!("Started new conversation: {}", id);
        Ok(id)
    }
    
    /// Add a message to current conversation
    pub async fn add_chat_message(
        &self,
        conversation_id: String,
        role: String,
        content: String,
        token_count: Option<i32>,
    ) -> Result<String> {
        let mut history = self.chat_history.write().await;
        let chat_storage = history.as_mut()
            .context("Chat history not initialized")?;
        
        chat_storage.add_message(
            conversation_id,
            role,
            content,
            token_count,
            None,
        ).await
    }
    
    /// Get recent conversations
    pub async fn get_recent_conversations(&self, limit: i64) -> Result<Vec<ConversationSummary>> {
        let history = self.chat_history.read().await;
        let chat_storage = history.as_ref()
            .context("Chat history not initialized")?;
        
        chat_storage.list_conversations(limit).await
    }
    
    /// Search chat history
    pub async fn search_chat_history(&self, query: &str, limit: i64) -> Result<Vec<crate::storage::chat_history::SearchResult>> {
        let history = self.chat_history.read().await;
        let chat_storage = history.as_ref()
            .context("Chat history not initialized")?;
        
        chat_storage.search(query, limit).await
    }
    
    /// Get messages for a conversation
    pub async fn get_conversation_messages(&self, conversation_id: &str) -> Result<Vec<crate::storage::chat_history::ChatMessage>> {
        let history = self.chat_history.read().await;
        let chat_storage = history.as_ref()
            .context("Chat history not initialized")?;
        
        chat_storage.get_conversation_messages(conversation_id).await
    }
    
    /// Get storage status
    pub async fn get_status(&self) -> StorageStatus {
        self.status.read().await.clone()
    }
    
    /// Export all data for backup
    pub async fn export_all_data(&self) -> Result<serde_json::Value> {
        let mut export = serde_json::Map::new();
        
        // Export API keys if unlocked
        let storage = self.secure_storage.read().await;
        if let Some(storage_instance) = storage.as_ref() {
            match storage_instance.get_all_api_keys().await {
                Ok(keys) => {
                    export.insert("api_keys".to_string(), serde_json::to_value(keys)?);
                }
                Err(e) => {
                    warn!("Failed to export API keys: {}", e);
                }
            }
        }
        
        // Export database configs
        let configs = self.db_configs.read().await;
        export.insert("database_configs".to_string(), serde_json::to_value(&*configs)?);
        
        // Export chat history statistics
        let history = self.chat_history.read().await;
        if let Some(chat_storage) = history.as_ref() {
            let stats = chat_storage.get_statistics().await?;
            export.insert("chat_statistics".to_string(), stats);
        }
        
        Ok(serde_json::Value::Object(export))
    }
    
    /// Import data from backup
    pub async fn import_data(&self, data: serde_json::Value) -> Result<()> {
        if let Some(obj) = data.as_object() {
            // Import API keys
            if let Some(api_keys) = obj.get("api_keys") {
                if let Ok(keys) = serde_json::from_value::<HashMap<String, String>>(api_keys.clone()) {
                    let mut storage = self.secure_storage.write().await;
                    if let Some(storage_instance) = storage.as_mut() {
                        storage_instance.store_all_api_keys(keys).await?;
                    }
                }
            }
            
            // Import database configs
            if let Some(db_configs) = obj.get("database_configs") {
                if let Ok(configs) = serde_json::from_value::<HashMap<String, HashMap<String, String>>>(db_configs.clone()) {
                    let mut cached_configs = self.db_configs.write().await;
                    *cached_configs = configs;
                }
            }
            
            info!("Data import completed");
        }
        
        Ok(())
    }
}