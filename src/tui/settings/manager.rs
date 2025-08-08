//! Settings manager for TUI
//! 
//! Provides centralized settings management with persistence and real-time updates.

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::PathBuf;

/// TUI Settings that can be modified at runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiSettings {
    /// Store chat history
    pub store_history: bool,
    
    /// Number of threads for processing
    pub threads: usize,
    
    /// Temperature for model responses
    pub temperature: f32,
    
    /// Maximum tokens for responses
    pub max_tokens: usize,
    
    /// Show timestamps in chat
    pub show_timestamps: bool,
    
    /// Enable message history mode
    pub message_history_mode: bool,
    
    /// Auto-scroll chat
    pub auto_scroll: bool,
    
    /// Theme preference
    pub theme: ThemePreference,
    
    /// Notification settings
    pub notifications: NotificationSettings,
    
    /// Keyboard shortcuts customization
    pub shortcuts: KeyboardShortcuts,
}

impl Default for TuiSettings {
    fn default() -> Self {
        Self {
            store_history: true,
            threads: 1,
            temperature: 0.7,
            max_tokens: 2048,
            show_timestamps: true,
            message_history_mode: false,
            auto_scroll: true,
            theme: ThemePreference::default(),
            notifications: NotificationSettings::default(),
            shortcuts: KeyboardShortcuts::default(),
        }
    }
}

/// Theme preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThemePreference {
    Light,
    Dark,
    Auto,
    Custom(String),
}

impl Default for ThemePreference {
    fn default() -> Self {
        Self::Dark
    }
}

/// Notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub enabled: bool,
    pub sound: bool,
    pub desktop: bool,
    pub in_app: bool,
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            sound: false,
            desktop: false,
            in_app: true,
        }
    }
}

/// Keyboard shortcuts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardShortcuts {
    pub quit: String,
    pub help: String,
    pub clear: String,
    pub save: String,
    pub load: String,
    pub search: String,
    pub toggle_timestamps: String,
    pub toggle_history_mode: String,
}

impl Default for KeyboardShortcuts {
    fn default() -> Self {
        Self {
            quit: "q".to_string(),
            help: "?".to_string(),
            clear: "c".to_string(),
            save: "s".to_string(),
            load: "l".to_string(),
            search: "/".to_string(),
            toggle_timestamps: "t".to_string(),
            toggle_history_mode: "h".to_string(),
        }
    }
}

/// Settings manager for centralized access
pub struct SettingsManager {
    settings: Arc<RwLock<TuiSettings>>,
    config_path: PathBuf,
    auto_save: bool,
}

impl SettingsManager {
    /// Create new settings manager
    pub fn new(config_path: Option<PathBuf>) -> Result<Self> {
        let config_path = config_path.unwrap_or_else(|| {
            dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("loki")
                .join("tui_settings.json")
        });
        
        // Load settings if file exists
        let settings = if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&contents).unwrap_or_default()
        } else {
            TuiSettings::default()
        };
        
        Ok(Self {
            settings: Arc::new(RwLock::new(settings)),
            config_path,
            auto_save: true,
        })
    }
    
    /// Get current settings (read-only)
    pub async fn get(&self) -> TuiSettings {
        self.settings.read().await.clone()
    }
    
    /// Update settings
    pub async fn update<F>(&self, update_fn: F) -> Result<()>
    where
        F: FnOnce(&mut TuiSettings),
    {
        {
            let mut settings = self.settings.write().await;
            update_fn(&mut settings);
        }
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(())
    }
    
    /// Toggle store history
    pub async fn toggle_store_history(&self) -> Result<bool> {
        let mut settings = self.settings.write().await;
        settings.store_history = !settings.store_history;
        let new_value = settings.store_history;
        drop(settings);
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(new_value)
    }
    
    /// Cycle threads (1 -> 2 -> 3 -> 1)
    pub async fn cycle_threads(&self) -> Result<usize> {
        let mut settings = self.settings.write().await;
        settings.threads = match settings.threads {
            1 => 2,
            2 => 3,
            _ => 1,
        };
        let new_value = settings.threads;
        drop(settings);
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(new_value)
    }
    
    /// Cycle temperature (0.3 -> 0.7 -> 1.0 -> 0.3)
    pub async fn cycle_temperature(&self) -> Result<f32> {
        let mut settings = self.settings.write().await;
        settings.temperature = match settings.temperature {
            x if x < 0.4 => 0.7,
            x if x < 0.8 => 1.0,
            _ => 0.3,
        };
        let new_value = settings.temperature;
        drop(settings);
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(new_value)
    }
    
    /// Cycle max tokens (1024 -> 2048 -> 4096 -> 8192 -> 1024)
    pub async fn cycle_max_tokens(&self) -> Result<usize> {
        let mut settings = self.settings.write().await;
        settings.max_tokens = match settings.max_tokens {
            x if x < 2048 => 2048,
            x if x < 4096 => 4096,
            x if x < 8192 => 8192,
            _ => 1024,
        };
        let new_value = settings.max_tokens;
        drop(settings);
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(new_value)
    }
    
    /// Toggle timestamps
    pub async fn toggle_timestamps(&self) -> Result<bool> {
        let mut settings = self.settings.write().await;
        settings.show_timestamps = !settings.show_timestamps;
        let new_value = settings.show_timestamps;
        drop(settings);
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(new_value)
    }
    
    /// Toggle message history mode
    pub async fn toggle_history_mode(&self) -> Result<bool> {
        let mut settings = self.settings.write().await;
        settings.message_history_mode = !settings.message_history_mode;
        let new_value = settings.message_history_mode;
        drop(settings);
        
        if self.auto_save {
            self.save().await?;
        }
        
        Ok(new_value)
    }
    
    /// Save settings to disk
    pub async fn save(&self) -> Result<()> {
        let settings = self.settings.read().await;
        let contents = serde_json::to_string_pretty(&*settings)?;
        
        // Ensure directory exists
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        std::fs::write(&self.config_path, contents)?;
        Ok(())
    }
    
    /// Load settings from disk
    pub async fn load(&self) -> Result<()> {
        if self.config_path.exists() {
            let contents = std::fs::read_to_string(&self.config_path)?;
            let loaded_settings: TuiSettings = serde_json::from_str(&contents)?;
            
            let mut settings = self.settings.write().await;
            *settings = loaded_settings;
        }
        Ok(())
    }
    
    /// Get settings for synchronous contexts (uses block_in_place)
    pub fn get_sync(&self) -> TuiSettings {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.get().await
            })
        })
    }
    
    /// Set auto-save behavior
    pub fn set_auto_save(&mut self, enabled: bool) {
        self.auto_save = enabled;
    }
}

/// Global settings manager instance
static SETTINGS_MANAGER: once_cell::sync::OnceCell<Arc<SettingsManager>> = once_cell::sync::OnceCell::new();

/// Initialize the global settings manager
pub fn initialize_settings_manager(config_path: Option<PathBuf>) -> Result<()> {
    let manager = Arc::new(SettingsManager::new(config_path)?);
    SETTINGS_MANAGER.set(manager).map_err(|_| anyhow::anyhow!("Settings manager already initialized"))?;
    Ok(())
}

/// Get the global settings manager
pub fn get_settings_manager() -> Option<Arc<SettingsManager>> {
    SETTINGS_MANAGER.get().cloned()
}

/// Get settings manager or create default
pub fn get_or_create_settings_manager() -> Arc<SettingsManager> {
    SETTINGS_MANAGER.get_or_init(|| {
        Arc::new(SettingsManager::new(None).expect("Failed to create settings manager"))
    }).clone()
}