//! Orchestration Bridge - Synchronizes orchestration settings across tabs
//! 
//! This bridge ensures that orchestration configuration (parallel models,
//! load balancing, routing strategies) is synchronized between the orchestration
//! tab and the chat tab for seamless multi-model usage.

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::tui::event_bus::{EventBus, SystemEvent, TabId, Subscriber, create_handler};
use crate::tui::chat::orchestration::{
    OrchestrationManager, RoutingStrategy,
    ModelCallTracker,
};

/// Orchestration bridge for settings synchronization
pub struct OrchestrationBridge {
    /// Event bus for cross-tab communication
    event_bus: Arc<EventBus>,
    
    /// Shared orchestration manager
    orchestration_manager: Arc<RwLock<OrchestrationManager>>,
    
    /// Model call tracker
    call_tracker: Arc<ModelCallTracker>,
    
    /// Current orchestration configuration
    current_config: Arc<RwLock<OrchestrationConfig>>,
    
    /// Synchronization state
    sync_state: Arc<RwLock<SyncState>>,
}

/// Orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    pub enabled: bool,
    pub parallel_models: usize,
    pub routing_strategy: RoutingStrategy,
    pub load_balancing: LoadBalancingMode,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub consensus_threshold: f32,
    pub cost_optimization: bool,
    pub quality_threshold: f32,
    pub selected_models: Vec<String>,
    pub model_weights: std::collections::HashMap<String, f32>,
}

/// Load balancing modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LoadBalancingMode {
    RoundRobin,
    LeastLatency,
    CostOptimized,
    QualityFirst,
    Adaptive,
}

/// Synchronization state
#[derive(Debug, Clone)]
struct SyncState {
    last_sync: chrono::DateTime<chrono::Utc>,
    pending_changes: Vec<ConfigChange>,
    sync_errors: Vec<String>,
    active_tabs: Vec<TabId>,
}

/// Configuration change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source_tab: TabId,
    pub change_type: ChangeType,
    pub old_value: serde_json::Value,
    pub new_value: serde_json::Value,
}

/// Types of configuration changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    ModelSelection,
    ParallelCount,
    RoutingStrategy,
    LoadBalancing,
    Timeout,
    RetryPolicy,
    ConsensusThreshold,
    CostOptimization,
    QualitySettings,
    ModelWeights,
}

impl OrchestrationBridge {
    /// Create a new orchestration bridge
    pub fn new(
        event_bus: Arc<EventBus>,
        orchestration_manager: Arc<RwLock<OrchestrationManager>>,
    ) -> Self {
        let call_tracker = Arc::new(ModelCallTracker::new());
        
        Self {
            event_bus,
            orchestration_manager,
            call_tracker,
            current_config: Arc::new(RwLock::new(OrchestrationConfig::default())),
            sync_state: Arc::new(RwLock::new(SyncState {
                last_sync: chrono::Utc::now(),
                pending_changes: Vec::new(),
                sync_errors: Vec::new(),
                active_tabs: vec![TabId::Chat, TabId::Settings],
            })),
        }
    }
    
    /// Initialize the bridge and start listening for events
    pub async fn initialize(&self) -> Result<()> {
        // Subscribe to orchestration events
        self.subscribe_to_events().await?;
        
        // Load initial configuration
        self.load_configuration().await?;
        
        // Start synchronization
        self.start_sync_task().await;
        
        tracing::info!("Orchestration bridge initialized");
        Ok(())
    }
    
    /// Subscribe to relevant events
    async fn subscribe_to_events(&self) -> Result<()> {
        // Subscribe to orchestration configuration changes
        let bus = self.event_bus.clone();
        let bridge = self.clone();
        
        let subscriber = Subscriber {
            id: uuid::Uuid::new_v4().to_string(),
            tab_id: TabId::Settings,
            handler: create_handler(move |event| {
                if let SystemEvent::ConfigurationChanged { setting, new_value, .. } = event {
                    if setting.starts_with("orchestration.") {
                        let bridge = bridge.clone();
                        // Spawn async task to handle config change
                        tokio::spawn(async move {
                            let _ = bridge.handle_config_change(TabId::Settings, setting, new_value).await;
                        });
                    }
                }
                Ok(())
            }),
            filter: None,
        };
        
        bus.subscribe("ConfigurationChanged".to_string(), subscriber).await?;
        
        Ok(())
    }
    
    /// Load initial configuration from orchestration manager
    async fn load_configuration(&self) -> Result<()> {
        let manager = self.orchestration_manager.read().await;
        
        let config = OrchestrationConfig {
            enabled: manager.orchestration_enabled,
            parallel_models: manager.parallel_models,
            routing_strategy: manager.preferred_strategy.clone(),
            load_balancing: self.map_to_load_balancing(&manager.preferred_strategy),
            timeout_seconds: 30, // Default timeout
            retry_attempts: 3,
            consensus_threshold: 0.7, // Default value
            cost_optimization: manager.cost_threshold_cents > 0.0,
            quality_threshold: manager.quality_threshold,
            selected_models: manager.enabled_models.clone(),
            model_weights: std::collections::HashMap::new(), // Will be populated from manager when available
        };
        
        *self.current_config.write().await = config;
        
        Ok(())
    }
    
    /// Start background synchronization task
    async fn start_sync_task(&self) {
        let bridge = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = bridge.sync_configuration().await {
                    tracing::error!("Orchestration sync error: {}", e);
                    
                    let mut state = bridge.sync_state.write().await;
                    state.sync_errors.push(format!("Sync failed: {}", e));
                    if state.sync_errors.len() > 10 {
                        state.sync_errors.remove(0);
                    }
                }
            }
        });
    }
    
    /// Synchronize configuration across tabs
    async fn sync_configuration(&self) -> Result<()> {
        let mut state = self.sync_state.write().await;
        
        // Process pending changes
        let changes_to_apply = state.pending_changes.drain(..).collect::<Vec<_>>();
        let active_tabs = state.active_tabs.clone();
        drop(state); // Release lock before applying changes
        
        for change in changes_to_apply {
            self.apply_change(change.clone()).await?;
            
            // Notify all active tabs about the change
            for tab_id in &active_tabs {
                if tab_id != &change.source_tab {
                    self.notify_tab_of_change(tab_id.clone(), &change).await;
                }
            }
        }
        
        let mut state = self.sync_state.write().await;
        state.last_sync = chrono::Utc::now();
        
        Ok(())
    }
    
    /// Notify a specific tab about configuration change
    async fn notify_tab_of_change(&self, tab_id: TabId, change: &ConfigChange) {
        tracing::debug!("Notifying tab {:?} about config change: {:?}", tab_id, change.change_type);
        
        // Send event to specific tab about the configuration update
        let event = SystemEvent::ConfigurationChanged {
            setting: format!("orchestration.{:?}", change.change_type),
            old_value: change.old_value.clone(),
            new_value: change.new_value.clone(),
        };
        
        self.event_bus.publish(event).await;
    }
    
    /// Handle configuration change from a tab
    pub async fn handle_config_change(
        &self,
        source_tab: TabId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        let change_type = self.parse_change_type(&key)?;
        
        let old_value = {
            let config = self.current_config.read().await;
            self.get_config_value(&config, &change_type)?
        };
        
        let change = ConfigChange {
            timestamp: chrono::Utc::now(),
            source_tab: source_tab.clone(),
            change_type,
            old_value,
            new_value: value.clone(),
        };
        
        // Add to pending changes
        self.sync_state.write().await.pending_changes.push(change);
        
        // Broadcast change to other tabs
        self.broadcast_change(source_tab, key, value).await?;
        
        Ok(())
    }
    
    /// Apply a configuration change
    async fn apply_change(&self, change: ConfigChange) -> Result<()> {
        let mut config = self.current_config.write().await;
        let mut manager = self.orchestration_manager.write().await;
        
        match change.change_type {
            ChangeType::ModelSelection => {
                if let Ok(models) = serde_json::from_value::<Vec<String>>(change.new_value) {
                    config.selected_models = models.clone();
                    manager.enabled_models = models;
                }
            }
            ChangeType::ParallelCount => {
                if let Some(count) = change.new_value.as_u64() {
                    config.parallel_models = count as usize;
                    manager.parallel_models = count as usize;
                }
            }
            ChangeType::RoutingStrategy => {
                if let Ok(strategy) = serde_json::from_value::<RoutingStrategy>(change.new_value) {
                    config.routing_strategy = strategy.clone();
                    manager.preferred_strategy = strategy;
                }
            }
            ChangeType::LoadBalancing => {
                if let Ok(mode) = serde_json::from_value::<LoadBalancingMode>(change.new_value) {
                    config.load_balancing = mode;
                    // Apply load balancing mode to manager
                    self.apply_load_balancing(&mut manager, mode);
                }
            }
            ChangeType::Timeout => {
                if let Some(timeout) = change.new_value.as_u64() {
                    config.timeout_seconds = timeout;
                }
            }
            ChangeType::RetryPolicy => {
                if let Some(retries) = change.new_value.as_u64() {
                    config.retry_attempts = retries as u32;
                }
            }
            ChangeType::ConsensusThreshold => {
                if let Some(threshold) = change.new_value.as_f64() {
                    config.consensus_threshold = threshold as f32;
                    // Consensus threshold can be stored as part of voting config when implemented
                }
            }
            ChangeType::CostOptimization => {
                if let Some(enabled) = change.new_value.as_bool() {
                    config.cost_optimization = enabled;
                    // Set cost threshold based on optimization mode
                    manager.cost_threshold_cents = if enabled { 100.0 } else { 0.0 };
                }
            }
            ChangeType::QualitySettings => {
                if let Some(threshold) = change.new_value.as_f64() {
                    config.quality_threshold = threshold as f32;
                    manager.quality_threshold = threshold as f32;
                }
            }
            ChangeType::ModelWeights => {
                if let Ok(weights) = serde_json::from_value(change.new_value) {
                    config.model_weights = weights;
                    // Model weights will be stored when weight-based routing is implemented
                }
            }
        }
        
        tracing::debug!("Applied config change: {:?}", change.change_type);
        Ok(())
    }
    
    /// Broadcast configuration change to other tabs
    async fn broadcast_change(
        &self,
        source_tab: TabId,
        key: String,
        value: serde_json::Value,
    ) -> Result<()> {
        let event = SystemEvent::ConfigurationChanged {
            setting: key,
            old_value: serde_json::Value::Null,  // We don't track old value here
            new_value: value,
        };
        
        self.event_bus.publish(event).await?;
        Ok(())
    }
    
    /// Get current orchestration configuration
    pub async fn get_configuration(&self) -> OrchestrationConfig {
        self.current_config.read().await.clone()
    }
    
    /// Update configuration from chat tab
    pub async fn update_from_chat(&self, updates: OrchestrationUpdate) -> Result<()> {
        let source_tab = TabId::Chat;
        
        if let Some(parallel_models) = updates.parallel_models {
            self.handle_config_change(
                source_tab.clone(),
                "orchestration.parallel_models".to_string(),
                serde_json::json!(parallel_models),
            ).await?;
        }
        
        if let Some(routing_strategy) = updates.routing_strategy {
            self.handle_config_change(
                source_tab.clone(),
                "orchestration.routing_strategy".to_string(),
                serde_json::to_value(routing_strategy)?,
            ).await?;
        }
        
        if let Some(load_balancing) = updates.load_balancing {
            self.handle_config_change(
                source_tab.clone(),
                "orchestration.load_balancing".to_string(),
                serde_json::to_value(load_balancing)?,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Get model call tracker
    pub fn get_call_tracker(&self) -> Arc<ModelCallTracker> {
        self.call_tracker.clone()
    }
    
    /// Check if orchestration is enabled
    pub async fn is_enabled(&self) -> bool {
        self.current_config.read().await.enabled
    }
    
    /// Get parallel model count
    pub async fn get_parallel_count(&self) -> usize {
        self.current_config.read().await.parallel_models
    }
    
    /// Map routing strategy to load balancing mode
    fn map_to_load_balancing(&self, strategy: &RoutingStrategy) -> LoadBalancingMode {
        match strategy {
            RoutingStrategy::RoundRobin => LoadBalancingMode::RoundRobin,
            RoutingStrategy::LeastLatency => LoadBalancingMode::LeastLatency,
            RoutingStrategy::CostOptimized => LoadBalancingMode::CostOptimized,
            RoutingStrategy::QualityFirst => LoadBalancingMode::QualityFirst,
            _ => LoadBalancingMode::Adaptive,
        }
    }
    
    /// Apply load balancing mode to manager
    fn apply_load_balancing(&self, manager: &mut OrchestrationManager, mode: LoadBalancingMode) {
        let strategy = match mode {
            LoadBalancingMode::RoundRobin => RoutingStrategy::RoundRobin,
            LoadBalancingMode::LeastLatency => RoutingStrategy::LeastLatency,
            LoadBalancingMode::CostOptimized => RoutingStrategy::CostOptimized,
            LoadBalancingMode::QualityFirst => RoutingStrategy::QualityFirst,
            LoadBalancingMode::Adaptive => RoutingStrategy::Adaptive,
        };
        manager.preferred_strategy = strategy;
    }
    
    /// Parse change type from configuration key
    fn parse_change_type(&self, key: &str) -> Result<ChangeType> {
        let change_type = match key {
            "orchestration.models" | "orchestration.selected_models" => ChangeType::ModelSelection,
            "orchestration.parallel_models" | "orchestration.parallel_count" => ChangeType::ParallelCount,
            "orchestration.routing_strategy" => ChangeType::RoutingStrategy,
            "orchestration.load_balancing" => ChangeType::LoadBalancing,
            "orchestration.timeout" => ChangeType::Timeout,
            "orchestration.retry" | "orchestration.retry_attempts" => ChangeType::RetryPolicy,
            "orchestration.consensus_threshold" => ChangeType::ConsensusThreshold,
            "orchestration.cost_optimization" => ChangeType::CostOptimization,
            "orchestration.quality_threshold" => ChangeType::QualitySettings,
            "orchestration.model_weights" => ChangeType::ModelWeights,
            _ => return Err(anyhow::anyhow!("Unknown configuration key: {}", key)),
        };
        
        Ok(change_type)
    }
    
    /// Get configuration value for a change type
    fn get_config_value(
        &self,
        config: &OrchestrationConfig,
        change_type: &ChangeType,
    ) -> Result<serde_json::Value> {
        let value = match change_type {
            ChangeType::ModelSelection => serde_json::to_value(&config.selected_models)?,
            ChangeType::ParallelCount => serde_json::json!(config.parallel_models),
            ChangeType::RoutingStrategy => serde_json::to_value(&config.routing_strategy)?,
            ChangeType::LoadBalancing => serde_json::to_value(&config.load_balancing)?,
            ChangeType::Timeout => serde_json::json!(config.timeout_seconds),
            ChangeType::RetryPolicy => serde_json::json!(config.retry_attempts),
            ChangeType::ConsensusThreshold => serde_json::json!(config.consensus_threshold),
            ChangeType::CostOptimization => serde_json::json!(config.cost_optimization),
            ChangeType::QualitySettings => serde_json::json!(config.quality_threshold),
            ChangeType::ModelWeights => serde_json::to_value(&config.model_weights)?,
        };
        
        Ok(value)
    }
}

/// Updates from chat tab
#[derive(Debug, Clone)]
pub struct OrchestrationUpdate {
    pub parallel_models: Option<usize>,
    pub routing_strategy: Option<RoutingStrategy>,
    pub load_balancing: Option<LoadBalancingMode>,
}

impl Clone for OrchestrationBridge {
    fn clone(&self) -> Self {
        Self {
            event_bus: self.event_bus.clone(),
            orchestration_manager: self.orchestration_manager.clone(),
            call_tracker: self.call_tracker.clone(),
            current_config: self.current_config.clone(),
            sync_state: self.sync_state.clone(),
        }
    }
}

impl OrchestrationBridge {
    /// Register a tab as active for synchronization
    pub async fn register_tab(&self, tab_id: TabId) {
        let mut state = self.sync_state.write().await;
        if !state.active_tabs.contains(&tab_id) {
            state.active_tabs.push(tab_id.clone());
            tracing::info!("Registered tab {:?} for orchestration sync", tab_id);
        }
    }
    
    /// Unregister a tab from synchronization
    pub async fn unregister_tab(&self, tab_id: TabId) {
        let mut state = self.sync_state.write().await;
        state.active_tabs.retain(|t| t != &tab_id);
        tracing::info!("Unregistered tab {:?} from orchestration sync", tab_id);
    }
    
    /// Get list of active tabs
    pub async fn get_active_tabs(&self) -> Vec<TabId> {
        self.sync_state.read().await.active_tabs.clone()
    }
    
    /// Check if a tab is registered for sync
    pub async fn is_tab_active(&self, tab_id: TabId) -> bool {
        self.sync_state.read().await.active_tabs.contains(&tab_id)
    }
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            parallel_models: 1,
            routing_strategy: RoutingStrategy::RoundRobin,
            load_balancing: LoadBalancingMode::RoundRobin,
            timeout_seconds: 30,
            retry_attempts: 3,
            consensus_threshold: 0.7,
            cost_optimization: false,
            quality_threshold: 0.8,
            selected_models: Vec::new(),
            model_weights: std::collections::HashMap::new(),
        }
    }
}

