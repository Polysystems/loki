//! Smart Context Switching Module
//! 
//! Provides intelligent context switching between different operational modes,
//! tabs, and execution environments based on current activity and resource availability.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, broadcast, watch};
use anyhow::{Result, Context as AnyhowContext};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use crate::tui::bridges::EventBridge;
use crate::story::StoryContext;
use crate::cognitive::CognitiveState;
use super::context_aware_execution::{ExecutionContext, ResourceContext};

/// Smart context switching manager
pub struct SmartContextSwitcher {
    /// Current active context
    active_context: Arc<RwLock<ActiveContext>>,
    
    /// Context history for learning
    context_history: Arc<RwLock<Vec<ContextTransition>>>,
    
    /// Context predictor
    predictor: Arc<ContextPredictor>,
    
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Context cache
    context_cache: Arc<RwLock<ContextCache>>,
    
    /// Switching strategies
    switching_strategies: Arc<RwLock<HashMap<String, Box<dyn SwitchingStrategy>>>>,
    
    /// Performance optimizer
    performance_optimizer: Arc<PerformanceOptimizer>,
    
    /// Event bridge for notifications
    event_bridge: Option<Arc<EventBridge>>,
    
    /// Watch channel for context updates
    context_watch: (watch::Sender<ActiveContext>, watch::Receiver<ActiveContext>),
    
    /// Event channel
    event_tx: broadcast::Sender<ContextSwitchEvent>,
    
    /// Configuration
    config: ContextSwitchConfig,
}

/// Active context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveContext {
    /// Context identifier
    pub id: String,
    
    /// Context type
    pub context_type: ContextType,
    
    /// Active tab
    pub active_tab: TabId,
    
    /// Story context if available
    pub story_context: Option<StoryContext>,
    
    /// Cognitive state
    pub cognitive_state: Option<CognitiveState>,
    
    /// Resource allocation
    pub resources: ResourceAllocation,
    
    /// Priority queue
    pub priority_queue: Vec<PriorityItem>,
    
    /// Context metadata
    pub metadata: HashMap<String, Value>,
    
    /// Activation timestamp
    pub activated_at: DateTime<Utc>,
    
    /// Performance metrics
    pub performance_metrics: ContextMetrics,
}

/// Context types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextType {
    /// Chat interaction context
    ChatInteraction,
    
    /// Tool execution context
    ToolExecution,
    
    /// Story development context
    StoryDevelopment,
    
    /// Cognitive processing context
    CognitiveProcessing,
    
    /// Multi-tab orchestration context
    MultiTabOrchestration,
    
    /// Background task context
    BackgroundTask,
    
    /// Emergency response context
    EmergencyResponse,
    
    /// Learning mode context
    LearningMode,
}

/// Resource allocation for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub memory_mb: usize,
    pub cpu_cores: usize,
    pub gpu_available: bool,
    pub network_bandwidth: NetworkBandwidth,
    pub token_allocation: usize,
    pub priority_level: u8,
}

/// Network bandwidth allocation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NetworkBandwidth {
    Unlimited,
    High,
    Medium,
    Low,
    Minimal,
}

/// Priority item in queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityItem {
    pub id: String,
    pub item_type: String,
    pub priority: u8,
    pub deadline: Option<DateTime<Utc>>,
    pub estimated_duration_ms: u64,
}

/// Context transition record
#[derive(Debug, Clone)]
pub struct ContextTransition {
    pub from_context: ActiveContext,
    pub to_context: ActiveContext,
    pub trigger: TransitionTrigger,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub success: bool,
    pub performance_impact: f32,
}

/// Transition triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionTrigger {
    /// User initiated switch
    UserRequest(String),
    
    /// Resource constraint
    ResourceConstraint(String),
    
    /// Priority change
    PriorityChange(String),
    
    /// Scheduled transition
    Scheduled,
    
    /// Emergency override
    Emergency(String),
    
    /// Performance optimization
    PerformanceOptimization,
    
    /// Story progression
    StoryProgression,
    
    /// Automatic prediction
    Predicted,
}

/// Context predictor for intelligent switching
#[derive(Debug)]
pub struct ContextPredictor {
    /// Prediction model state
    model_state: Arc<RwLock<PredictionModel>>,
    
    /// Pattern recognition
    pattern_recognizer: Arc<PatternRecognizer>,
    
    /// Activity analyzer
    activity_analyzer: Arc<ActivityAnalyzer>,
    
    /// Prediction confidence threshold
    confidence_threshold: f32,
}

/// Prediction model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Activity patterns
    activity_patterns: HashMap<String, ActivityPattern>,
    
    /// Transition probabilities
    transition_matrix: HashMap<(ContextType, ContextType), f32>,
    
    /// Time-based patterns
    temporal_patterns: Vec<TemporalPattern>,
    
    /// Learning rate
    learning_rate: f32,
}

/// Activity pattern
#[derive(Debug, Clone)]
pub struct ActivityPattern {
    pub pattern_id: String,
    pub activities: Vec<String>,
    pub frequency: f32,
    pub average_duration_ms: u64,
    pub next_likely_context: ContextType,
}

/// Temporal pattern
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub hour_of_day: u8,
    pub day_of_week: u8,
    pub preferred_context: ContextType,
    pub confidence: f32,
}

/// Pattern recognizer
#[derive(Debug)]
pub struct PatternRecognizer {
    /// Recognized patterns
    patterns: Arc<RwLock<Vec<RecognizedPattern>>>,
    
    /// Pattern threshold
    threshold: f32,
}

/// Recognized pattern
#[derive(Debug, Clone)]
pub struct RecognizedPattern {
    pub pattern_type: String,
    pub confidence: f32,
    pub suggested_action: String,
}

/// Activity analyzer
#[derive(Debug)]
pub struct ActivityAnalyzer {
    /// Current activity metrics
    current_metrics: Arc<RwLock<ActivityMetrics>>,
    
    /// Historical data
    history: Arc<RwLock<Vec<ActivitySnapshot>>>,
}

/// Activity metrics
#[derive(Debug, Clone)]
pub struct ActivityMetrics {
    pub message_rate: f32,
    pub tool_usage_rate: f32,
    pub context_switch_rate: f32,
    pub error_rate: f32,
    pub response_time_ms: u64,
}

/// Activity snapshot
#[derive(Debug, Clone)]
pub struct ActivitySnapshot {
    pub timestamp: DateTime<Utc>,
    pub metrics: ActivityMetrics,
    pub context_type: ContextType,
}

/// Resource monitor
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Current resource usage
    current_usage: Arc<RwLock<ResourceUsage>>,
    
    /// Resource thresholds
    thresholds: ResourceThresholds,
    
    /// Alert channel
    alert_tx: mpsc::Sender<ResourceAlert>,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_used_mb: usize,
    pub memory_total_mb: usize,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: Option<f32>,
    pub network_throughput_mbps: f32,
    pub token_usage_rate: f32,
}

/// Resource thresholds
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    pub memory_warning_percent: f32,
    pub memory_critical_percent: f32,
    pub cpu_warning_percent: f32,
    pub cpu_critical_percent: f32,
}

/// Resource alert
#[derive(Debug, Clone)]
pub struct ResourceAlert {
    pub alert_type: AlertType,
    pub resource: String,
    pub current_value: f32,
    pub threshold: f32,
    pub suggested_action: String,
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    Warning,
    Critical,
    Recovery,
}

/// Context cache for fast switching
#[derive(Debug)]
pub struct ContextCache {
    /// Cached contexts
    contexts: HashMap<String, CachedContext>,
    
    /// Cache size limit
    max_size: usize,
    
    /// LRU tracking
    access_order: Vec<String>,
}

/// Cached context
#[derive(Debug, Clone)]
pub struct CachedContext {
    pub context: ActiveContext,
    pub cached_at: DateTime<Utc>,
    pub access_count: usize,
    pub last_accessed: DateTime<Utc>,
    pub precomputed_data: Option<Value>,
}

/// Trait for switching strategies
#[async_trait::async_trait]
pub trait SwitchingStrategy: Send + Sync {
    async fn evaluate(&self, current: &ActiveContext, candidate: &ContextType) -> SwitchDecision;
    async fn prepare_switch(&self, from: &ActiveContext, to: &ContextType) -> Result<SwitchPreparation>;
    async fn execute_switch(&self, preparation: &SwitchPreparation) -> Result<()>;
}

/// Switch decision
#[derive(Debug, Clone)]
pub struct SwitchDecision {
    pub should_switch: bool,
    pub confidence: f32,
    pub reasons: Vec<String>,
    pub estimated_benefit: f32,
    pub estimated_cost: f32,
}

/// Switch preparation
#[derive(Debug, Clone)]
pub struct SwitchPreparation {
    pub save_state: bool,
    pub transfer_data: HashMap<String, Value>,
    pub cleanup_tasks: Vec<String>,
    pub initialization_tasks: Vec<String>,
    pub resource_adjustments: ResourceAllocation,
}

/// Performance optimizer
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Optimization rules
    rules: Arc<RwLock<Vec<OptimizationRule>>>,
    
    /// Performance history
    history: Arc<RwLock<Vec<PerformanceRecord>>>,
    
    /// Current optimizations
    active_optimizations: Arc<RwLock<HashMap<String, Optimization>>>,
}

/// Optimization rule
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub rule_id: String,
    pub condition: String,
    pub action: OptimizationAction,
    pub priority: u8,
}

/// Optimization action
#[derive(Debug, Clone)]
pub enum OptimizationAction {
    SwitchContext(ContextType),
    AdjustResources(ResourceAllocation),
    EnableFeature(String),
    DisableFeature(String),
    Custom(String),
}

/// Active optimization
#[derive(Debug, Clone)]
pub struct Optimization {
    pub optimization_id: String,
    pub rule_id: String,
    pub started_at: DateTime<Utc>,
    pub expected_improvement: f32,
    pub actual_improvement: Option<f32>,
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub timestamp: DateTime<Utc>,
    pub context_type: ContextType,
    pub metrics: ContextMetrics,
}

/// Context performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetrics {
    pub response_time_ms: u64,
    pub throughput: f32,
    pub error_rate: f32,
    pub resource_efficiency: f32,
    pub user_satisfaction: f32,
}

impl Default for ContextMetrics {
    fn default() -> Self {
        Self {
            response_time_ms: 0,
            throughput: 0.0,
            error_rate: 0.0,
            resource_efficiency: 1.0,
            user_satisfaction: 1.0,
        }
    }
}

/// Context switch events
#[derive(Debug, Clone)]
pub enum ContextSwitchEvent {
    SwitchInitiated {
        from: ContextType,
        to: ContextType,
        trigger: TransitionTrigger,
    },
    SwitchCompleted {
        context_id: String,
        duration_ms: u64,
    },
    SwitchFailed {
        reason: String,
        fallback: Option<ContextType>,
    },
    PredictionMade {
        predicted_context: ContextType,
        confidence: f32,
    },
    ResourceAlert {
        alert: ResourceAlert,
    },
    OptimizationApplied {
        optimization: Optimization,
    },
}

/// Configuration
#[derive(Debug, Clone)]
pub struct ContextSwitchConfig {
    pub enable_prediction: bool,
    pub prediction_lookahead_ms: u64,
    pub enable_auto_switch: bool,
    pub switch_threshold: f32,
    pub cache_size: usize,
    pub history_limit: usize,
    pub enable_optimization: bool,
    pub resource_monitoring_interval_ms: u64,
}

impl Default for ContextSwitchConfig {
    fn default() -> Self {
        Self {
            enable_prediction: true,
            prediction_lookahead_ms: 5000,
            enable_auto_switch: true,
            switch_threshold: 0.7,
            cache_size: 10,
            history_limit: 100,
            enable_optimization: true,
            resource_monitoring_interval_ms: 1000,
        }
    }
}

impl SmartContextSwitcher {
    /// Create a new smart context switcher
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let (alert_tx, _) = mpsc::channel(100);
        
        let initial_context = ActiveContext {
            id: uuid::Uuid::new_v4().to_string(),
            context_type: ContextType::ChatInteraction,
            active_tab: TabId::Chat,
            story_context: None,
            cognitive_state: None,
            resources: ResourceAllocation {
                memory_mb: 512,
                cpu_cores: 2,
                gpu_available: false,
                network_bandwidth: NetworkBandwidth::Medium,
                token_allocation: 4000,
                priority_level: 5,
            },
            priority_queue: Vec::new(),
            metadata: HashMap::new(),
            activated_at: Utc::now(),
            performance_metrics: ContextMetrics::default(),
        };
        
        let (watch_tx, watch_rx) = watch::channel(initial_context.clone());
        
        Self {
            active_context: Arc::new(RwLock::new(initial_context)),
            context_history: Arc::new(RwLock::new(Vec::new())),
            predictor: Arc::new(ContextPredictor {
                model_state: Arc::new(RwLock::new(PredictionModel {
                    activity_patterns: HashMap::new(),
                    transition_matrix: HashMap::new(),
                    temporal_patterns: Vec::new(),
                    learning_rate: 0.1,
                })),
                pattern_recognizer: Arc::new(PatternRecognizer {
                    patterns: Arc::new(RwLock::new(Vec::new())),
                    threshold: 0.7,
                }),
                activity_analyzer: Arc::new(ActivityAnalyzer {
                    current_metrics: Arc::new(RwLock::new(ActivityMetrics {
                        message_rate: 0.0,
                        tool_usage_rate: 0.0,
                        context_switch_rate: 0.0,
                        error_rate: 0.0,
                        response_time_ms: 0,
                    })),
                    history: Arc::new(RwLock::new(Vec::new())),
                }),
                confidence_threshold: 0.7,
            }),
            resource_monitor: Arc::new(ResourceMonitor {
                current_usage: Arc::new(RwLock::new(ResourceUsage {
                    memory_used_mb: 0,
                    memory_total_mb: 8192,
                    cpu_usage_percent: 0.0,
                    gpu_usage_percent: None,
                    network_throughput_mbps: 0.0,
                    token_usage_rate: 0.0,
                })),
                thresholds: ResourceThresholds {
                    memory_warning_percent: 80.0,
                    memory_critical_percent: 95.0,
                    cpu_warning_percent: 80.0,
                    cpu_critical_percent: 95.0,
                },
                alert_tx,
            }),
            context_cache: Arc::new(RwLock::new(ContextCache {
                contexts: HashMap::new(),
                max_size: 10,
                access_order: Vec::new(),
            })),
            switching_strategies: Arc::new(RwLock::new(HashMap::new())),
            performance_optimizer: Arc::new(PerformanceOptimizer {
                rules: Arc::new(RwLock::new(Vec::new())),
                history: Arc::new(RwLock::new(Vec::new())),
                active_optimizations: Arc::new(RwLock::new(HashMap::new())),
            }),
            event_bridge: None,
            context_watch: (watch_tx, watch_rx),
            event_tx,
            config: ContextSwitchConfig::default(),
        }
    }
    
    /// Set event bridge
    pub fn set_event_bridge(&mut self, bridge: Arc<EventBridge>) {
        self.event_bridge = Some(bridge);
    }
    
    /// Switch to a new context
    pub async fn switch_context(
        &self,
        new_context_type: ContextType,
        trigger: TransitionTrigger,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Get current context
        let current = self.active_context.read().await.clone();
        
        // Check if switch is needed
        if current.context_type == new_context_type {
            debug!("Already in requested context: {:?}", new_context_type);
            return Ok(());
        }
        
        // Send initiation event
        let _ = self.event_tx.send(ContextSwitchEvent::SwitchInitiated {
            from: current.context_type.clone(),
            to: new_context_type.clone(),
            trigger: trigger.clone(),
        });
        
        // Evaluate switch decision
        let decision = self.evaluate_switch(&current, &new_context_type).await?;
        
        if !decision.should_switch {
            debug!("Switch not recommended: {:?}", decision.reasons);
            return Ok(());
        }
        
        // Prepare for switch
        let preparation = self.prepare_switch(&current, &new_context_type).await?;
        
        // Execute cleanup tasks
        for task in &preparation.cleanup_tasks {
            debug!("Executing cleanup: {}", task);
        }
        
        // Create new context
        let new_context = self.create_context(new_context_type.clone(), preparation.resource_adjustments).await?;
        
        // Execute initialization tasks
        for task in &preparation.initialization_tasks {
            debug!("Executing initialization: {}", task);
        }
        
        // Record transition
        let transition = ContextTransition {
            from_context: current.clone(),
            to_context: new_context.clone(),
            trigger,
            timestamp: Utc::now(),
            duration_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            performance_impact: decision.estimated_benefit - decision.estimated_cost,
        };
        
        self.context_history.write().await.push(transition);
        
        // Update active context
        *self.active_context.write().await = new_context.clone();
        let _ = self.context_watch.0.send(new_context.clone());
        
        // Cache old context if valuable
        if decision.estimated_benefit > 0.5 {
            self.cache_context(current).await?;
        }
        
        // Send completion event
        let _ = self.event_tx.send(ContextSwitchEvent::SwitchCompleted {
            context_id: new_context.id,
            duration_ms: start_time.elapsed().as_millis() as u64,
        });
        
        // Update predictor model
        if self.config.enable_prediction {
            self.update_prediction_model(&new_context_type).await?;
        }
        
        info!("Context switched to {:?} in {}ms", new_context_type, start_time.elapsed().as_millis());
        Ok(())
    }
    
    /// Evaluate switch decision
    async fn evaluate_switch(
        &self,
        current: &ActiveContext,
        target: &ContextType,
    ) -> Result<SwitchDecision> {
        let strategies = self.switching_strategies.read().await;
        
        // Use strategy if available
        if let Some(strategy) = strategies.get(&format!("{:?}", target)) {
            return Ok(strategy.evaluate(current, target).await);
        }
        
        // Default evaluation
        let resource_usage = self.resource_monitor.current_usage.read().await;
        let memory_pressure = resource_usage.memory_used_mb as f32 / resource_usage.memory_total_mb as f32;
        
        let mut reasons = Vec::new();
        let mut confidence = 0.8;
        
        if memory_pressure > 0.9 && target == &ContextType::BackgroundTask {
            reasons.push("High memory pressure favors background tasks".to_string());
            confidence = 0.9;
        }
        
        if current.performance_metrics.error_rate > 0.1 {
            reasons.push("High error rate suggests context switch".to_string());
            confidence = 0.85;
        }
        
        Ok(SwitchDecision {
            should_switch: confidence > self.config.switch_threshold,
            confidence,
            reasons,
            estimated_benefit: 0.7,
            estimated_cost: 0.2,
        })
    }
    
    /// Prepare for context switch
    async fn prepare_switch(
        &self,
        from: &ActiveContext,
        to: &ContextType,
    ) -> Result<SwitchPreparation> {
        let mut preparation = SwitchPreparation {
            save_state: true,
            transfer_data: HashMap::new(),
            cleanup_tasks: Vec::new(),
            initialization_tasks: Vec::new(),
            resource_adjustments: from.resources.clone(),
        };
        
        // Adjust resources based on target context
        match to {
            ContextType::CognitiveProcessing => {
                preparation.resource_adjustments.memory_mb = 1024;
                preparation.resource_adjustments.cpu_cores = 4;
                preparation.initialization_tasks.push("Load cognitive models".to_string());
            }
            ContextType::ToolExecution => {
                preparation.resource_adjustments.network_bandwidth = NetworkBandwidth::High;
                preparation.initialization_tasks.push("Initialize tool manager".to_string());
            }
            ContextType::BackgroundTask => {
                preparation.resource_adjustments.priority_level = 2;
                preparation.resource_adjustments.memory_mb = 256;
            }
            _ => {}
        }
        
        // Add common cleanup tasks
        preparation.cleanup_tasks.push("Save current state".to_string());
        preparation.cleanup_tasks.push("Clear temporary caches".to_string());
        
        Ok(preparation)
    }
    
    /// Create new context
    async fn create_context(
        &self,
        context_type: ContextType,
        resources: ResourceAllocation,
    ) -> Result<ActiveContext> {
        Ok(ActiveContext {
            id: uuid::Uuid::new_v4().to_string(),
            context_type: context_type.clone(),
            active_tab: self.get_preferred_tab(&context_type),
            story_context: None,
            cognitive_state: None,
            resources,
            priority_queue: Vec::new(),
            metadata: HashMap::new(),
            activated_at: Utc::now(),
            performance_metrics: ContextMetrics::default(),
        })
    }
    
    /// Get preferred tab for context type
    fn get_preferred_tab(&self, context_type: &ContextType) -> TabId {
        match context_type {
            ContextType::ChatInteraction => TabId::Chat,
            ContextType::ToolExecution => TabId::Utilities,
            ContextType::StoryDevelopment => TabId::Chat,
            ContextType::CognitiveProcessing => TabId::Cognitive,
            ContextType::MultiTabOrchestration => TabId::Home,
            _ => TabId::Chat,
        }
    }
    
    /// Cache context for fast switching
    async fn cache_context(&self, context: ActiveContext) -> Result<()> {
        let mut cache = self.context_cache.write().await;
        
        // Enforce cache size limit
        if cache.contexts.len() >= cache.max_size {
            // Remove least recently used
            if let Some(lru_id) = cache.access_order.first().cloned() {
                cache.contexts.remove(&lru_id);
                cache.access_order.remove(0);
            }
        }
        
        let cached = CachedContext {
            context: context.clone(),
            cached_at: Utc::now(),
            access_count: 0,
            last_accessed: Utc::now(),
            precomputed_data: None,
        };
        
        cache.contexts.insert(context.id.clone(), cached);
        cache.access_order.push(context.id);
        
        Ok(())
    }
    
    /// Update prediction model
    async fn update_prediction_model(&self, new_context: &ContextType) -> Result<()> {
        let mut model = self.predictor.model_state.write().await;
        
        // Update transition matrix
        let history = self.context_history.read().await;
        if let Some(last_transition) = history.last() {
            let key = (last_transition.from_context.context_type.clone(), new_context.clone());
            let learning_rate = model.learning_rate;
            let count = model.transition_matrix.entry(key).or_insert(0.0);
            *count = (*count * (1.0 - learning_rate)) + learning_rate;
        }
        
        Ok(())
    }
    
    /// Predict next context
    pub async fn predict_next_context(&self) -> Option<(ContextType, f32)> {
        if !self.config.enable_prediction {
            return None;
        }
        
        let current = self.active_context.read().await;
        let model = self.predictor.model_state.read().await;
        
        // Find most likely transition
        let mut best_prediction = None;
        let mut best_confidence = 0.0;
        
        for ((from, to), probability) in &model.transition_matrix {
            if from == &current.context_type && *probability > best_confidence {
                best_confidence = *probability;
                best_prediction = Some(to.clone());
            }
        }
        
        if best_confidence > self.predictor.confidence_threshold {
            best_prediction.map(|ctx| (ctx, best_confidence))
        } else {
            None
        }
    }
    
    /// Monitor and auto-switch if needed
    pub async fn monitor_and_switch(&self) -> Result<()> {
        if !self.config.enable_auto_switch {
            return Ok(());
        }
        
        // Check resource alerts
        let usage = self.resource_monitor.current_usage.read().await;
        let memory_percent = (usage.memory_used_mb as f32 / usage.memory_total_mb as f32) * 100.0;
        
        if memory_percent > self.resource_monitor.thresholds.memory_critical_percent {
            // Switch to emergency context
            self.switch_context(
                ContextType::EmergencyResponse,
                TransitionTrigger::Emergency("Critical memory pressure".to_string()),
            ).await?;
        } else if let Some((predicted, confidence)) = self.predict_next_context().await {
            // Switch to predicted context if confidence is high
            if confidence > 0.9 {
                self.switch_context(
                    predicted,
                    TransitionTrigger::Predicted,
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Get current context
    pub async fn get_current_context(&self) -> ActiveContext {
        self.active_context.read().await.clone()
    }
    
    /// Watch context changes
    pub fn watch_context(&self) -> watch::Receiver<ActiveContext> {
        self.context_watch.1.clone()
    }
    
    /// Register switching strategy
    pub async fn register_strategy(&self, name: String, strategy: Box<dyn SwitchingStrategy>) {
        self.switching_strategies.write().await.insert(name, strategy);
        info!("Registered switching strategy");
    }
    
    /// Get context history
    pub async fn get_history(&self) -> Vec<ContextTransition> {
        self.context_history.read().await.clone()
    }
    
    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<ContextSwitchEvent> {
        self.event_tx.subscribe()
    }
}