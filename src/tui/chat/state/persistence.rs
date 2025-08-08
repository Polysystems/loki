//! State persistence for chat sessions
//! 
//! Handles saving and loading chat state to disk

use std::path::{PathBuf};
use anyhow::{Result, Context};

use super::chat_state::ChatState;
use super::settings::ChatSettings;

/// Handles chat state persistence
pub struct StatePersistence {
    /// Base directory for chat storage
    storage_dir: PathBuf,
}

impl StatePersistence {
    /// Create a new persistence handler
    pub fn new(storage_dir: PathBuf) -> Result<Self> {
        // Ensure directory exists
        std::fs::create_dir_all(&storage_dir)
            .context("Failed to create chat storage directory")?;
        
        Ok(Self { storage_dir })
    }
    
    /// Get the path for a chat file
    fn chat_path(&self, chat_id: &str) -> PathBuf {
        self.storage_dir.join(format!("chat_{}.json", chat_id))
    }
    
    /// Save a chat state to disk
    pub async fn save_chat(&self, chat: &ChatState) -> Result<()> {
        let path = self.chat_path(&chat.id);
        let json = serde_json::to_string_pretty(chat)
            .context("Failed to serialize chat state")?;
        
        tokio::fs::write(&path, json)
            .await
            .context("Failed to write chat file")?;
        
        Ok(())
    }
    
    /// Load a chat state from disk
    pub async fn load_chat(&self, chat_id: &str) -> Result<ChatState> {
        let path = self.chat_path(chat_id);
        let json = tokio::fs::read_to_string(&path)
            .await
            .context("Failed to read chat file")?;
        
        let chat = serde_json::from_str(&json)
            .context("Failed to deserialize chat state")?;
        
        Ok(chat)
    }
    
    /// List all saved chat IDs
    pub async fn list_chats(&self) -> Result<Vec<String>> {
        let mut chats = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.storage_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("chat_") && name.ends_with(".json") {
                    let chat_id = name
                        .strip_prefix("chat_")
                        .and_then(|s| s.strip_suffix(".json"))
                        .unwrap_or("")
                        .to_string();
                    chats.push(chat_id);
                }
            }
        }
        
        Ok(chats)
    }
    
    /// Delete a saved chat
    pub async fn delete_chat(&self, chat_id: &str) -> Result<()> {
        let path = self.chat_path(chat_id);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .context("Failed to delete chat file")?;
        }
        Ok(())
    }
    
    /// Export chat to a specific format
    pub async fn export_chat(&self, chat: &ChatState, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(chat)
                    .context("Failed to export as JSON")
            }
            ExportFormat::Markdown => {
                Ok(self.export_as_markdown(chat))
            }
        }
    }
    
    /// Export chat as markdown
    fn export_as_markdown(&self, chat: &ChatState) -> String {
        let mut md = format!("# {}\n\n", chat.title);
        md.push_str(&format!("Created: {}\n", chat.created_at.format("%Y-%m-%d %H:%M:%S")));
        md.push_str(&format!("Last Activity: {}\n\n", chat.last_activity.format("%Y-%m-%d %H:%M:%S")));
        
        md.push_str("## Conversation\n\n");
        
        for msg in &chat.messages {
            // Format message based on type
            match msg {
                crate::tui::run::AssistantResponseType::Message { author, message, timestamp, .. } => {
                    md.push_str(&format!("**{}** ({}): {}\n\n", author, timestamp, message));
                }
                _ => {
                    // Handle other message types
                }
            }
        }
        
        md
    }
}

/// Export format options
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Markdown,
}

/// Alias for compatibility
pub type ChatPersistence = StatePersistence;

impl ChatPersistence {
    /// Create a new persistence handler with default directory
    pub fn with_default_dir() -> Result<Self> {
        let base_dir = if let Ok(data_dir) = std::env::var("LOKI_DATA_DIR") {
            PathBuf::from(data_dir)
        } else {
            dirs::data_dir()
                .context("Failed to get data directory")?
                .join("loki")
                .join("chats")
        };
        
        StatePersistence::new(base_dir)
    }
    
    /// Save a chat with filename (compatibility method)
    pub async fn save_chat_with_filename(&self, state: &ChatState, filename: &str) -> Result<()> {
        // Extract chat ID from filename if it has an extension
        let chat_id = if let Some(id) = filename.strip_suffix(".json") {
            id.to_string()
        } else {
            filename.to_string()
        };
        
        // Create a new state with the correct ID
        let mut chat = state.clone();
        chat.id = chat_id.clone();
        
        // Call the original save_chat method
        self.save_chat(&chat).await
    }
    
    /// Load a chat by filename (compatibility method)
    pub async fn load_chat_by_filename(&self, filename: &str) -> Result<ChatState> {
        // Extract chat ID from filename
        let chat_id = if let Some(id) = filename.strip_suffix(".json") {
            id.to_string()
        } else if filename.starts_with("chat_") {
            filename.strip_prefix("chat_").unwrap_or(filename).to_string()
        } else {
            filename.to_string()
        };
        
        // Call the original load_chat method
        self.load_chat(&chat_id).await
    }
}

/// Settings persistence functionality
impl StatePersistence {
    /// Get the path for settings file
    fn settings_path(&self) -> PathBuf {
        self.storage_dir.join("chat_settings.json")
    }
    
    /// Save chat settings to disk
    pub async fn save_settings(&self, settings: &ChatSettings) -> Result<()> {
        let path = self.settings_path();
        let json = serde_json::to_string_pretty(settings)
            .context("Failed to serialize chat settings")?;
        
        tokio::fs::write(&path, json)
            .await
            .context("Failed to write settings file")?;
        
        tracing::info!("💾 Saved chat settings to {:?}", path);
        Ok(())
    }
    
    /// Load chat settings from disk
    pub async fn load_settings(&self) -> Result<ChatSettings> {
        let path = self.settings_path();
        
        // If settings file doesn't exist, return defaults
        if !path.exists() {
            tracing::info!("Settings file not found, using defaults");
            return Ok(ChatSettings::default());
        }
        
        let json = tokio::fs::read_to_string(&path)
            .await
            .context("Failed to read settings file")?;
        
        let settings = serde_json::from_str(&json)
            .context("Failed to deserialize chat settings")?;
        
        tracing::info!("📂 Loaded chat settings from {:?}", path);
        Ok(settings)
    }
    
    /// Check if settings file exists
    pub async fn settings_exist(&self) -> bool {
        self.settings_path().exists()
    }
    
    /// Get settings file metadata
    pub async fn settings_metadata(&self) -> Result<Option<(std::time::SystemTime, u64)>> {
        let path = self.settings_path();
        if !path.exists() {
            return Ok(None);
        }
        
        let metadata = tokio::fs::metadata(&path)
            .await
            .context("Failed to read settings metadata")?;
        
        Ok(Some((metadata.modified()?, metadata.len())))
    }
}