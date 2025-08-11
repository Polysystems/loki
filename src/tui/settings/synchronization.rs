//! Settings Synchronization Module
//! 
//! Provides real-time synchronization of settings across all tabs,
//! ensuring consistent configuration and immediate propagation of changes.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, broadcast, watch};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use crate::tui::bridges::EventBridge;

/// Settings synchronizer for cross-tab consistency
pub struct SettingsSynchronizer {
    /// Current settings state
    settings_store: Arc<RwLock<SettingsStore>>,
    
    /// Settings watchers by key
    watchers: Arc<RwLock<HashMap<String, watch::Sender<SettingValue>>>>,
    
    /// Event bridge for cross-tab communication
    event_bridge: Option<Arc<EventBridge>>,
    
    /// Synchronization strategy
    sync_strategy: Arc<RwLock<SyncStrategy>>,
    
    /// Conflict resolution
    conflict_resolver: Arc<ConflictResolver>,
    
    /// Settings history
    history: Arc<RwLock<SettingsHistory>>,
    
    /// Validation rules
    validators: Arc<RwLock<HashMap<String, Box<dyn SettingValidator>>>>,
    
    /// Event channel
    event_tx: broadcast::Sender<SettingSyncEvent>,
    
    /// Configuration
    config: SyncConfig,
}

/// Settings store
#[derive(Debug, Clone)]
pub struct SettingsStore {
    /// Settings by category
    settings: HashMap<String, CategorySettings>,
    
    /// Global settings
    global: GlobalSettings,
    
    /// Tab-specific overrides
    tab_overrides: HashMap<TabId, HashMap<String, SettingValue>>,
    
    /// Last update timestamp
    last_updated: DateTime<Utc>,
    
    /// Version for conflict detection
    version: u64,
}

/// Category of settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorySettings {
    pub name: String,
    pub description: String,
    pub settings: HashMap<String, SettingValue>,
    pub locked: bool,
}

/// Global settings that affect all tabs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSettings {
    pub theme: String,
    pub language: String,
    pub auto_save: bool,
    pub sync_enabled: bool,
    pub debug_mode: bool,
    pub performance_mode: PerformanceMode,
}

impl Default for GlobalSettings {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            language: "en".to_string(),
            auto_save: true,
            sync_enabled: true,
            debug_mode: false,
            performance_mode: PerformanceMode::Balanced,
        }
    }
}

/// Performance modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceMode {
    PowerSaver,
    Balanced,
    HighPerformance,
    Maximum,
}

/// Setting value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SettingValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
    Null,
}

impl SettingValue {
    /// Get as string
    pub fn as_string(&self) -> Option<&String> {
        match self {
            SettingValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Get as number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            SettingValue::Number(n) => Some(*n),
            _ => None,
        }
    }
    
    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SettingValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Synchronization strategy
#[derive(Debug, Clone)]
pub struct SyncStrategy {
    /// Synchronization mode
    pub mode: SyncMode,
    
    /// Propagation delay in milliseconds
    pub propagation_delay_ms: u64,
    
    /// Batch updates
    pub batch_updates: bool,
    
    /// Priority settings that sync immediately
    pub priority_settings: Vec<String>,
    
    /// Excluded settings from sync
    pub excluded_settings: Vec<String>,
}

/// Synchronization modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncMode {
    /// Immediate synchronization
    Immediate,
    
    /// Debounced synchronization
    Debounced,
    
    /// Periodic synchronization
    Periodic,
    
    /// Manual synchronization
    Manual,
}

/// Conflict resolver for concurrent updates
#[derive(Debug)]
pub struct ConflictResolver {
    /// Resolution strategy
    strategy: ConflictResolutionStrategy,
    
    /// Conflict history
    conflicts: Arc<RwLock<Vec<ConflictRecord>>>,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy)]
pub enum ConflictResolutionStrategy {
    /// Last write wins
    LastWriteWins,
    
    /// First write wins
    FirstWriteWins,
    
    /// Merge values
    Merge,
    
    /// User prompt
    UserPrompt,
}

/// Conflict record
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    pub setting_key: String,
    pub conflicting_values: Vec<(TabId, SettingValue)>,
    pub resolution: SettingValue,
    pub timestamp: DateTime<Utc>,
}

/// Settings history
#[derive(Debug, Clone)]
pub struct SettingsHistory {
    /// History entries
    entries: Vec<HistoryEntry>,
    
    /// Maximum history size
    max_size: usize,
}

/// History entry
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub setting_key: String,
    pub old_value: SettingValue,
    pub new_value: SettingValue,
    pub changed_by: TabId,
    pub timestamp: DateTime<Utc>,
    pub reason: Option<String>,
}

/// Setting validator trait
#[async_trait::async_trait]
pub trait SettingValidator: Send + Sync {
    async fn validate(&self, value: &SettingValue) -> Result<ValidationResult>;
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub normalized_value: Option<SettingValue>,
}

/// Settings synchronization events
#[derive(Debug, Clone)]
pub enum SettingSyncEvent {
    SettingChanged {
        key: String,
        value: SettingValue,
        source: TabId,
    },
    SettingSynchronized {
        key: String,
        tabs: Vec<TabId>,
    },
    ConflictDetected {
        key: String,
        conflict: ConflictRecord,
    },
    ConflictResolved {
        key: String,
        resolution: SettingValue,
    },
    SyncError {
        key: String,
        error: String,
    },
}

/// Synchronization configuration
#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub enable_auto_sync: bool,
    pub sync_interval_ms: u64,
    pub max_retries: usize,
    pub conflict_strategy: ConflictResolutionStrategy,
    pub enable_validation: bool,
    pub enable_history: bool,
    pub history_limit: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enable_auto_sync: true,
            sync_interval_ms: 100,
            max_retries: 3,
            conflict_strategy: ConflictResolutionStrategy::LastWriteWins,
            enable_validation: true,
            enable_history: true,
            history_limit: 100,
        }
    }
}

impl SettingsSynchronizer {
    /// Create a new settings synchronizer
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(100);
        
        Self {
            settings_store: Arc::new(RwLock::new(SettingsStore {
                settings: HashMap::new(),
                global: GlobalSettings::default(),
                tab_overrides: HashMap::new(),
                last_updated: Utc::now(),
                version: 0,
            })),
            watchers: Arc::new(RwLock::new(HashMap::new())),
            event_bridge: None,
            sync_strategy: Arc::new(RwLock::new(SyncStrategy {
                mode: SyncMode::Immediate,
                propagation_delay_ms: 0,
                batch_updates: false,
                priority_settings: vec!["theme".to_string(), "language".to_string()],
                excluded_settings: Vec::new(),
            })),
            conflict_resolver: Arc::new(ConflictResolver {
                strategy: ConflictResolutionStrategy::LastWriteWins,
                conflicts: Arc::new(RwLock::new(Vec::new())),
            }),
            history: Arc::new(RwLock::new(SettingsHistory {
                entries: Vec::new(),
                max_size: 100,
            })),
            validators: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            config: SyncConfig::default(),
        }
    }
    
    /// Set event bridge for cross-tab communication
    pub fn set_event_bridge(&mut self, bridge: Arc<EventBridge>) {
        self.event_bridge = Some(bridge);
    }
    
    /// Update a setting
    pub async fn update_setting(
        &self,
        key: String,
        value: SettingValue,
        source: TabId,
        reason: Option<String>,
    ) -> Result<()> {
        // Validate if enabled
        if self.config.enable_validation {
            if let Some(validator) = self.validators.read().await.get(&key) {
                let validation = validator.validate(&value).await?;
                if !validation.valid {
                    return Err(anyhow::anyhow!(
                        "Validation failed: {}",
                        validation.errors.join(", ")
                    ));
                }
            }
        }
        
        // Get old value for history
        let old_value = self.get_setting(&key).await;
        
        // Update store
        let mut store = self.settings_store.write().await;
        
        // Find and update setting in appropriate category
        let mut updated = false;
        for (_, category) in store.settings.iter_mut() {
            if let Some(existing) = category.settings.get_mut(&key) {
                *existing = value.clone();
                updated = true;
                break;
            }
        }
        
        // If not found in categories, add to default category
        if !updated {
            store.settings.entry("general".to_string())
                .or_insert_with(|| CategorySettings {
                    name: "General".to_string(),
                    description: "General settings".to_string(),
                    settings: HashMap::new(),
                    locked: false,
                })
                .settings
                .insert(key.clone(), value.clone());
        }
        
        store.last_updated = Utc::now();
        store.version += 1;
        
        drop(store); // Release lock
        
        // Add to history
        if self.config.enable_history {
            if let Some(old) = old_value {
                self.add_to_history(HistoryEntry {
                    setting_key: key.clone(),
                    old_value: old,
                    new_value: value.clone(),
                    changed_by: source.clone(),
                    timestamp: Utc::now(),
                    reason,
                }).await;
            }
        }
        
        // Notify watchers
        if let Some(sender) = self.watchers.read().await.get(&key) {
            let _ = sender.send(value.clone());
        }
        
        // Send event
        let _ = self.event_tx.send(SettingSyncEvent::SettingChanged {
            key: key.clone(),
            value: value.clone(),
            source: source.clone(),
        });
        
        // Synchronize across tabs
        if self.config.enable_auto_sync {
            self.synchronize_setting(key, value, source).await?;
        }
        
        Ok(())
    }
    
    /// Get a setting value
    pub async fn get_setting(&self, key: &str) -> Option<SettingValue> {
        let store = self.settings_store.read().await;
        
        for (_, category) in &store.settings {
            if let Some(value) = category.settings.get(key) {
                return Some(value.clone());
            }
        }
        
        None
    }
    
    /// Get setting with tab override
    pub async fn get_setting_for_tab(&self, key: &str, tab: &TabId) -> Option<SettingValue> {
        let store = self.settings_store.read().await;
        
        // Check for tab override first
        if let Some(overrides) = store.tab_overrides.get(tab) {
            if let Some(value) = overrides.get(key) {
                return Some(value.clone());
            }
        }
        
        // Fall back to global setting
        drop(store);
        self.get_setting(key).await
    }
    
    /// Watch a setting for changes
    pub async fn watch_setting(&self, key: String) -> watch::Receiver<SettingValue> {
        let mut watchers = self.watchers.write().await;
        
        if let Some(sender) = watchers.get(&key) {
            sender.subscribe()
        } else {
            let initial_value = self.get_setting(&key).await
                .unwrap_or(SettingValue::Null);
            
            let (tx, rx) = watch::channel(initial_value);
            watchers.insert(key, tx);
            rx
        }
    }
    
    /// Synchronize setting across tabs
    async fn synchronize_setting(
        &self,
        key: String,
        value: SettingValue,
        source: TabId,
    ) -> Result<()> {
        let strategy = self.sync_strategy.read().await;
        
        // Check if setting is excluded
        if strategy.excluded_settings.contains(&key) {
            return Ok(());
        }
        
        // Determine sync timing
        let should_sync_immediately = strategy.mode == SyncMode::Immediate
            || strategy.priority_settings.contains(&key);
        
        if should_sync_immediately {
            self.propagate_setting(key, value, source).await?;
        } else {
            // Queue for later synchronization based on strategy
            match strategy.mode {
                SyncMode::Debounced => {
                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        strategy.propagation_delay_ms
                    )).await;
                    self.propagate_setting(key, value, source).await?;
                }
                SyncMode::Periodic => {
                    // Would be handled by periodic sync task
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Propagate setting to other tabs
    async fn propagate_setting(
        &self,
        key: String,
        value: SettingValue,
        source: TabId,
    ) -> Result<()> {
        if let Some(ref bridge) = self.event_bridge {
            // Send to all tabs except source
            let tabs = vec![
                TabId::Home,
                TabId::Chat,
                TabId::Utilities,
                TabId::Memory,
                TabId::Cognitive,
                TabId::Settings,
            ];
            
            for tab in tabs {
                if tab != source {
                    bridge.publish(SystemEvent::CrossTabMessage {
                        from: source.clone(),
                        to: tab.clone(),
                        message: serde_json::json!({
                            "type": "setting_sync",
                            "key": key,
                            "value": value,
                        }),
                    }).await?;
                }
            }
            
            // Send synchronized event
            let _ = self.event_tx.send(SettingSyncEvent::SettingSynchronized {
                key,
                tabs: tabs.into_iter().filter(|t| *t != source).collect(),
            });
        }
        
        Ok(())
    }
    
    /// Handle setting conflict
    pub async fn handle_conflict(
        &self,
        key: String,
        values: Vec<(TabId, SettingValue)>,
    ) -> Result<SettingValue> {
        let resolution = match self.conflict_resolver.strategy {
            ConflictResolutionStrategy::LastWriteWins => {
                values.last().map(|(_, v)| v.clone()).unwrap_or(SettingValue::Null)
            }
            ConflictResolutionStrategy::FirstWriteWins => {
                values.first().map(|(_, v)| v.clone()).unwrap_or(SettingValue::Null)
            }
            ConflictResolutionStrategy::Merge => {
                // Simple merge - take the first non-null value
                values.into_iter()
                    .find_map(|(_, v)| match v {
                        SettingValue::Null => None,
                        _ => Some(v),
                    })
                    .unwrap_or(SettingValue::Null)
            }
            ConflictResolutionStrategy::UserPrompt => {
                // Would prompt user for resolution
                values.last().map(|(_, v)| v.clone()).unwrap_or(SettingValue::Null)
            }
        };
        
        // Record conflict
        let conflict = ConflictRecord {
            setting_key: key.clone(),
            conflicting_values: values,
            resolution: resolution.clone(),
            timestamp: Utc::now(),
        };
        
        self.conflict_resolver.conflicts.write().await.push(conflict.clone());
        
        // Send events
        let _ = self.event_tx.send(SettingSyncEvent::ConflictDetected {
            key: key.clone(),
            conflict: conflict.clone(),
        });
        
        let _ = self.event_tx.send(SettingSyncEvent::ConflictResolved {
            key,
            resolution: resolution.clone(),
        });
        
        Ok(resolution)
    }
    
    /// Add to history
    async fn add_to_history(&self, entry: HistoryEntry) {
        let mut history = self.history.write().await;
        history.entries.push(entry);
        
        // Limit history size
        if history.entries.len() > history.max_size {
            history.entries.drain(0..history.entries.len() - history.max_size);
        }
    }
    
    /// Get settings history
    pub async fn get_history(&self) -> Vec<HistoryEntry> {
        self.history.read().await.entries.clone()
    }
    
    /// Register a setting validator
    pub async fn register_validator(&self, key: String, validator: Box<dyn SettingValidator>) {
        self.validators.write().await.insert(key, validator);
        info!("Registered validator for setting");
    }
    
    /// Export settings
    pub async fn export_settings(&self) -> Result<String> {
        let store = self.settings_store.read().await;
        let json = serde_json::to_string_pretty(&store.settings)?;
        Ok(json)
    }
    
    /// Import settings
    pub async fn import_settings(&self, json: &str) -> Result<()> {
        let imported: HashMap<String, CategorySettings> = serde_json::from_str(json)?;
        
        let mut store = self.settings_store.write().await;
        store.settings = imported;
        store.last_updated = Utc::now();
        store.version += 1;
        
        info!("Imported settings successfully");
        Ok(())
    }
    
    /// Subscribe to sync events
    pub fn subscribe(&self) -> broadcast::Receiver<SettingSyncEvent> {
        self.event_tx.subscribe()
    }
}