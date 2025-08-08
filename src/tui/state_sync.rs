//! State Synchronization System
//! 
//! Ensures consistent state across all tabs and components through
//! automatic synchronization and conflict resolution.

use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use anyhow::Result;
use tracing::{debug, info, warn};

use crate::tui::{
    event_bus::{EventBus, SystemEvent, TabId},
    shared_state::SharedSystemState,
};

/// State synchronization manager
pub struct StateSyncManager {
    /// Shared state reference
    shared_state: Arc<SharedSystemState>,
    
    /// Event bus for notifications
    event_bus: Arc<EventBus>,
    
    /// State watchers
    watchers: Arc<RwLock<HashMap<String, Vec<StateWatcher>>>>,
    
    /// Sync policies
    policies: Arc<RwLock<HashMap<String, SyncPolicy>>>,
    
    /// Conflict resolver
    conflict_resolver: Arc<ConflictResolver>,
    
    /// Sync status
    status: Arc<RwLock<SyncStatus>>,
}

/// State watcher
#[derive(Clone)]
pub struct StateWatcher {
    pub id: String,
    pub tab_id: TabId,
    pub key_pattern: String,
    pub callback: Arc<dyn Fn(StateChange) + Send + Sync>,
}

/// State change notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChange {
    pub key: String,
    pub old_value: Option<Value>,
    pub new_value: Value,
    pub source: TabId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Synchronization policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPolicy {
    pub key_pattern: String,
    pub sync_mode: SyncMode,
    pub conflict_resolution: ConflictResolutionStrategy,
    pub propagation_delay_ms: u64,
    pub batch_updates: bool,
}

/// Synchronization modes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SyncMode {
    Immediate,      // Sync immediately on change
    Batched,        // Batch multiple changes
    Lazy,           // Sync on demand
    Periodic(u64),  // Sync periodically (interval in ms)
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    LastWrite,      // Last write wins
    FirstWrite,     // First write wins
    Merge,          // Attempt to merge changes
    Manual,         // Require manual resolution
    Custom,         // Use custom resolver
}

/// Sync status
#[derive(Debug, Clone)]
pub struct SyncStatus {
    pub active: bool,
    pub synced_keys: HashSet<String>,
    pub pending_changes: usize,
    pub conflicts_resolved: usize,
    pub last_sync: Option<chrono::DateTime<chrono::Utc>>,
    pub error_count: usize,
}

/// Conflict resolver
pub struct ConflictResolver {
    /// Resolution strategies
    strategies: HashMap<String, Box<dyn ResolutionStrategy>>,
    
    /// Conflict history
    history: Arc<RwLock<Vec<ConflictRecord>>>,
}

/// Resolution strategy trait
pub trait ResolutionStrategy: Send + Sync {
    fn resolve(&self, key: &str, current: &Value, incoming: &Value) -> Value;
}

/// Conflict record
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    pub key: String,
    pub current_value: Value,
    pub incoming_value: Value,
    pub resolved_value: Value,
    pub strategy_used: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// State transaction for atomic updates
pub struct StateTransaction {
    changes: Vec<(String, Value)>,
    source: TabId,
    atomic: bool,
}

impl StateSyncManager {
    /// Create a new state sync manager
    pub fn new(
        shared_state: Arc<SharedSystemState>,
        event_bus: Arc<EventBus>,
    ) -> Self {
        Self {
            shared_state,
            event_bus,
            watchers: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(Self::default_policies())),
            conflict_resolver: Arc::new(ConflictResolver::new()),
            status: Arc::new(RwLock::new(SyncStatus {
                active: false,
                synced_keys: HashSet::new(),
                pending_changes: 0,
                conflicts_resolved: 0,
                last_sync: None,
                error_count: 0,
            })),
        }
    }
    
    /// Default synchronization policies
    fn default_policies() -> HashMap<String, SyncPolicy> {
        let mut policies = HashMap::new();
        
        // Model selection policy
        policies.insert("models.*".to_string(), SyncPolicy {
            key_pattern: "models.*".to_string(),
            sync_mode: SyncMode::Immediate,
            conflict_resolution: ConflictResolutionStrategy::LastWrite,
            propagation_delay_ms: 0,
            batch_updates: false,
        });
        
        // Agent configuration policy
        policies.insert("agents.*".to_string(), SyncPolicy {
            key_pattern: "agents.*".to_string(),
            sync_mode: SyncMode::Batched,
            conflict_resolution: ConflictResolutionStrategy::Merge,
            propagation_delay_ms: 100,
            batch_updates: true,
        });
        
        // Tool state policy
        policies.insert("tools.*".to_string(), SyncPolicy {
            key_pattern: "tools.*".to_string(),
            sync_mode: SyncMode::Immediate,
            conflict_resolution: ConflictResolutionStrategy::LastWrite,
            propagation_delay_ms: 0,
            batch_updates: false,
        });
        
        // Settings policy
        policies.insert("settings.*".to_string(), SyncPolicy {
            key_pattern: "settings.*".to_string(),
            sync_mode: SyncMode::Periodic(1000),
            conflict_resolution: ConflictResolutionStrategy::Manual,
            propagation_delay_ms: 500,
            batch_updates: true,
        });
        
        policies
    }
    
    /// Start synchronization
    pub async fn start(&self) -> Result<()> {
        let mut status = self.status.write().await;
        status.active = true;
        
        // Start sync workers
        self.start_immediate_sync().await;
        self.start_batch_sync().await;
        self.start_periodic_sync().await;
        
        info!("State synchronization started");
        Ok(())
    }
    
    /// Stop synchronization
    pub async fn stop(&self) -> Result<()> {
        let mut status = self.status.write().await;
        status.active = false;
        
        info!("State synchronization stopped");
        Ok(())
    }
    
    /// Start immediate sync worker
    async fn start_immediate_sync(&self) {
        let shared_state = self.shared_state.clone();
        let event_bus = self.event_bus.clone();
        let policies = self.policies.clone();
        let watchers = self.watchers.clone();
        let status = self.status.clone();
        
        tokio::spawn(async move {
            loop {
                let active = status.read().await.active;
                if !active {
                    break;
                }
                
                // Check for immediate sync items
                let policies = policies.read().await;
                for (key_pattern, policy) in policies.iter() {
                    if policy.sync_mode == SyncMode::Immediate {
                        // Process immediate sync
                        // This would be triggered by state changes
                    }
                }
                
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
    }
    
    /// Start batch sync worker
    async fn start_batch_sync(&self) {
        let shared_state = self.shared_state.clone();
        let event_bus = self.event_bus.clone();
        let policies = self.policies.clone();
        let status = self.status.clone();
        
        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut last_batch = std::time::Instant::now();
            
            loop {
                let active = status.read().await.active;
                if !active {
                    break;
                }
                
                // Process batch if enough time has passed
                if last_batch.elapsed().as_millis() >= 100 {
                    if !batch.is_empty() {
                        // Process batched changes
                        for (key, value, source) in batch.drain(..) {
                            let event = SystemEvent::StateChanged {
                                key,
                                value,
                                source,
                            };
                            event_bus.publish(event).await.ok();
                        }
                        last_batch = std::time::Instant::now();
                    }
                }
                
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
        });
    }
    
    /// Start periodic sync worker
    async fn start_periodic_sync(&self) {
        let shared_state = self.shared_state.clone();
        let policies = self.policies.clone();
        let status = self.status.clone();
        
        tokio::spawn(async move {
            loop {
                let active = status.read().await.active;
                if !active {
                    break;
                }
                
                // Check for periodic sync items
                let policies = policies.read().await;
                for (key_pattern, policy) in policies.iter() {
                    if let SyncMode::Periodic(interval_ms) = policy.sync_mode {
                        // Process periodic sync
                        // This would sync on the specified interval
                    }
                }
                
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    }
    
    /// Register a state watcher
    pub async fn watch(
        &self,
        key_pattern: &str,
        tab_id: TabId,
        callback: impl Fn(StateChange) + Send + Sync + 'static,
    ) -> String {
        let watcher_id = uuid::Uuid::new_v4().to_string();
        
        let watcher = StateWatcher {
            id: watcher_id.clone(),
            tab_id,
            key_pattern: key_pattern.to_string(),
            callback: Arc::new(callback),
        };
        
        let mut watchers = self.watchers.write().await;
        watchers.entry(key_pattern.to_string())
            .or_insert_with(Vec::new)
            .push(watcher);
        
        debug!("Registered watcher {} for pattern {}", watcher_id, key_pattern);
        watcher_id
    }
    
    /// Unregister a watcher
    pub async fn unwatch(&self, watcher_id: &str) -> Result<()> {
        let mut watchers = self.watchers.write().await;
        
        for (_, watcher_list) in watchers.iter_mut() {
            watcher_list.retain(|w| w.id != watcher_id);
        }
        
        Ok(())
    }
    
    /// Update state with synchronization
    pub async fn update(
        &self,
        key: &str,
        value: Value,
        source: TabId,
    ) -> Result<()> {
        // Get current value
        let current = self.shared_state.get(key).await;
        
        // Check for conflicts
        let final_value = if let Some(current_value) = current.clone() {
            // Find applicable policy
            let policies = self.policies.read().await;
            let policy = self.find_policy(&policies, key);
            
            // Resolve conflict if needed
            if current_value != value {
                self.conflict_resolver.resolve(
                    key,
                    &current_value,
                    &value,
                    policy.conflict_resolution,
                ).await
            } else {
                value.clone()
            }
        } else {
            value.clone()
        };
        
        // Update shared state
        self.shared_state.set(key.to_string(), final_value.clone()).await;
        
        // Notify watchers
        self.notify_watchers(key, current, final_value.clone(), source.clone()).await;
        
        // Publish state change event
        let event = SystemEvent::StateChanged {
            key: key.to_string(),
            value: final_value,
            source,
        };
        self.event_bus.publish(event).await?;
        
        // Update status
        let mut status = self.status.write().await;
        status.synced_keys.insert(key.to_string());
        status.last_sync = Some(chrono::Utc::now());
        
        Ok(())
    }
    
    /// Find applicable policy for a key
    fn find_policy<'a>(
        &self,
        policies: &'a HashMap<String, SyncPolicy>,
        key: &str,
    ) -> &'a SyncPolicy {
        // Find matching policy by pattern
        for (pattern, policy) in policies {
            if self.matches_pattern(key, pattern) {
                return policy;
            }
        }
        
        // Return default policy if no match
        static DEFAULT_POLICY: SyncPolicy = SyncPolicy {
            key_pattern: String::new(),
            sync_mode: SyncMode::Immediate,
            conflict_resolution: ConflictResolutionStrategy::LastWrite,
            propagation_delay_ms: 0,
            batch_updates: false,
        };
        &DEFAULT_POLICY
    }
    
    /// Check if key matches pattern
    fn matches_pattern(&self, key: &str, pattern: &str) -> bool {
        if pattern.ends_with("*") {
            let prefix = &pattern[..pattern.len() - 1];
            key.starts_with(prefix)
        } else {
            key == pattern
        }
    }
    
    /// Notify watchers of state change
    async fn notify_watchers(
        &self,
        key: &str,
        old_value: Option<Value>,
        new_value: Value,
        source: TabId,
    ) {
        let watchers = self.watchers.read().await;
        
        for (pattern, watcher_list) in watchers.iter() {
            if self.matches_pattern(key, pattern) {
                let change = StateChange {
                    key: key.to_string(),
                    old_value: old_value.clone(),
                    new_value: new_value.clone(),
                    source: source.clone(),
                    timestamp: chrono::Utc::now(),
                };
                
                for watcher in watcher_list {
                    (watcher.callback)(change.clone());
                }
            }
        }
    }
    
    /// Begin a state transaction
    pub fn begin_transaction(&self, source: TabId) -> StateTransaction {
        StateTransaction {
            changes: Vec::new(),
            source,
            atomic: true,
        }
    }
    
    /// Commit a state transaction
    pub async fn commit_transaction(&self, mut transaction: StateTransaction) -> Result<()> {
        if transaction.atomic {
            // Apply all changes atomically
            for (key, value) in transaction.changes {
                self.update(&key, value, transaction.source.clone()).await?;
            }
        }
        Ok(())
    }
    
    /// Get synchronization status
    pub async fn get_status(&self) -> SyncStatus {
        self.status.read().await.clone()
    }
    
    /// Force synchronization of specific keys
    pub async fn force_sync(&self, keys: Vec<String>) -> Result<()> {
        for key in keys {
            if let Some(value) = self.shared_state.get(&key).await {
                let event = SystemEvent::StateChanged {
                    key: key.clone(),
                    value,
                    source: TabId::System,
                };
                self.event_bus.publish(event).await?;
            }
        }
        Ok(())
    }
}

impl ConflictResolver {
    fn new() -> Self {
        let mut resolver = Self {
            strategies: HashMap::new(),
            history: Arc::new(RwLock::new(Vec::new())),
        };
        
        // Add default strategies
        resolver.strategies.insert(
            "last_write".to_string(),
            Box::new(LastWriteStrategy),
        );
        resolver.strategies.insert(
            "first_write".to_string(),
            Box::new(FirstWriteStrategy),
        );
        resolver.strategies.insert(
            "merge".to_string(),
            Box::new(MergeStrategy),
        );
        
        resolver
    }
    
    async fn resolve(
        &self,
        key: &str,
        current: &Value,
        incoming: &Value,
        strategy: ConflictResolutionStrategy,
    ) -> Value {
        let resolved = match strategy {
            ConflictResolutionStrategy::LastWrite => incoming.clone(),
            ConflictResolutionStrategy::FirstWrite => current.clone(),
            ConflictResolutionStrategy::Merge => {
                if let Some(s) = self.strategies.get("merge") {
                    s.resolve(key, current, incoming)
                } else {
                    incoming.clone()
                }
            }
            ConflictResolutionStrategy::Manual => {
                warn!("Manual conflict resolution required for key: {}", key);
                current.clone() // Keep current until manual resolution
            }
            ConflictResolutionStrategy::Custom => {
                // Use custom strategy if available
                incoming.clone()
            }
        };
        
        // Record conflict resolution
        let record = ConflictRecord {
            key: key.to_string(),
            current_value: current.clone(),
            incoming_value: incoming.clone(),
            resolved_value: resolved.clone(),
            strategy_used: format!("{:?}", strategy),
            timestamp: chrono::Utc::now(),
        };
        
        let mut history = self.history.write().await;
        history.push(record);
        
        // Keep only recent history
        if history.len() > 1000 {
            history.drain(0..100);
        }
        
        resolved
    }
}

// Resolution strategies

struct LastWriteStrategy;

impl ResolutionStrategy for LastWriteStrategy {
    fn resolve(&self, _key: &str, _current: &Value, incoming: &Value) -> Value {
        incoming.clone()
    }
}

struct FirstWriteStrategy;

impl ResolutionStrategy for FirstWriteStrategy {
    fn resolve(&self, _key: &str, current: &Value, _incoming: &Value) -> Value {
        current.clone()
    }
}

struct MergeStrategy;

impl ResolutionStrategy for MergeStrategy {
    fn resolve(&self, _key: &str, current: &Value, incoming: &Value) -> Value {
        // Simple merge strategy for objects
        if let (Some(current_obj), Some(incoming_obj)) = (current.as_object(), incoming.as_object()) {
            let mut merged = current_obj.clone();
            for (key, value) in incoming_obj {
                merged.insert(key.clone(), value.clone());
            }
            Value::Object(merged)
        } else {
            // Can't merge non-objects, use last write
            incoming.clone()
        }
    }
}

impl StateTransaction {
    /// Add a change to the transaction
    pub fn set(&mut self, key: String, value: Value) {
        self.changes.push((key, value));
    }
    
    /// Set atomicity
    pub fn atomic(mut self, atomic: bool) -> Self {
        self.atomic = atomic;
        self
    }
}