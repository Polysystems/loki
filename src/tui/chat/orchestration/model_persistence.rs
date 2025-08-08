//! Model persistence for saving enabled/disabled states
//! 
//! Persists model configuration across sessions

use std::path::PathBuf;
use std::collections::HashMap;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use tokio::fs;

/// Model configuration that persists across sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedModelConfig {
    /// Enabled model IDs
    pub enabled_models: Vec<String>,
    
    /// Model-specific settings
    pub model_settings: HashMap<String, ModelSettings>,
    
    /// Default model
    pub default_model: Option<String>,
    
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Per-model settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Custom display name
    pub display_name: Option<String>,
    
    /// Temperature override
    pub temperature: Option<f32>,
    
    /// Max tokens override
    pub max_tokens: Option<usize>,
    
    /// Custom system prompt
    pub system_prompt: Option<String>,
    
    /// Priority for routing (higher = preferred)
    pub priority: i32,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            display_name: None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            priority: 0,
        }
    }
}

/// Model persistence manager
pub struct ModelPersistence {
    /// Configuration file path
    config_path: PathBuf,
}

impl ModelPersistence {
    /// Create a new persistence manager
    pub fn new() -> Self {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("loki");
        
        Self {
            config_path: config_dir.join("models.json"),
        }
    }
    
    /// Load persisted model configuration
    pub async fn load(&self) -> Result<Option<PersistedModelConfig>> {
        if !self.config_path.exists() {
            return Ok(None);
        }
        
        let content = fs::read_to_string(&self.config_path)
            .await
            .context("Failed to read model config")?;
        
        let config: PersistedModelConfig = serde_json::from_str(&content)
            .context("Failed to parse model config")?;
        
        Ok(Some(config))
    }
    
    /// Save model configuration
    pub async fn save(&self, config: &PersistedModelConfig) -> Result<()> {
        // Ensure config directory exists
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)
                .await
                .context("Failed to create config directory")?;
        }
        
        let content = serde_json::to_string_pretty(config)
            .context("Failed to serialize model config")?;
        
        fs::write(&self.config_path, content)
            .await
            .context("Failed to write model config")?;
        
        Ok(())
    }
    
    /// Update enabled models
    pub async fn update_enabled_models(&self, enabled_models: Vec<String>) -> Result<()> {
        let mut config = self.load().await?.unwrap_or_else(|| PersistedModelConfig {
            enabled_models: vec![],
            model_settings: HashMap::new(),
            default_model: None,
            last_updated: chrono::Utc::now(),
        });
        
        config.enabled_models = enabled_models;
        config.last_updated = chrono::Utc::now();
        
        self.save(&config).await
    }
    
    /// Update model settings
    pub async fn update_model_settings(
        &self,
        model_id: String,
        settings: ModelSettings,
    ) -> Result<()> {
        let mut config = self.load().await?.unwrap_or_else(|| PersistedModelConfig {
            enabled_models: vec![],
            model_settings: HashMap::new(),
            default_model: None,
            last_updated: chrono::Utc::now(),
        });
        
        config.model_settings.insert(model_id, settings);
        config.last_updated = chrono::Utc::now();
        
        self.save(&config).await
    }
    
    /// Set default model
    pub async fn set_default_model(&self, model_id: Option<String>) -> Result<()> {
        let mut config = self.load().await?.unwrap_or_else(|| PersistedModelConfig {
            enabled_models: vec![],
            model_settings: HashMap::new(),
            default_model: None,
            last_updated: chrono::Utc::now(),
        });
        
        config.default_model = model_id;
        config.last_updated = chrono::Utc::now();
        
        self.save(&config).await
    }
}