//! Context-Aware Task Execution
//! 
//! Provides intelligent task execution that adapts based on context,
//! including story state, agent capabilities, tool availability, and system resources.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, broadcast};
use anyhow::{Result, Context as AnyhowContext};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::story::{StoryContext, StoryEngine};
use crate::cognitive::CognitiveOrchestrator;
use crate::tui::chat::agents::StoryAgentOrchestrator;
use crate::tui::bridges::{ToolBridge, StoryBridge};
use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use super::todo_manager::{TodoManager, TodoItem};

/// Context-aware task executor
pub struct ContextAwareExecutor {
    /// Story engine for narrative context
    story_engine: Option<Arc<StoryEngine>>,
    
    /// Cognitive orchestrator for reasoning
    cognitive_orchestrator: Option<Arc<CognitiveOrchestrator>>,
    
    /// Agent orchestrator for collaboration
    agent_orchestrator: Option<Arc<StoryAgentOrchestrator>>,
    
    /// Tool bridge for cross-tab execution
    tool_bridge: Option<Arc<ToolBridge>>,
    
    /// Todo manager for task tracking
    todo_manager: Option<Arc<TodoManager>>,
    
    /// Execution contexts by task
    execution_contexts: Arc<RwLock<HashMap<String, ExecutionContext>>>,
    
    /// Context enrichment strategies
    enrichment_strategies: Arc<RwLock<Vec<EnrichmentStrategy>>>,
    
    /// Execution adapters
    execution_adapters: Arc<RwLock<HashMap<String, Box<dyn ExecutionAdapter>>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<ExecutionMetrics>>,
    
    /// Event channel
    event_tx: broadcast::Sender<ContextExecutionEvent>,
    
    /// Configuration
    config: ContextExecutionConfig,
}

/// Comprehensive execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Task identifier
    pub task_id: String,
    
    /// Story context if available
    pub story_context: Option<StoryContext>,
    
    /// Cognitive state
    pub cognitive_state: CognitiveState,
    
    /// Available resources
    pub available_resources: ResourceContext,
    
    /// Agent assignments
    pub agent_assignments: Vec<AgentAssignment>,
    
    /// Tool requirements
    pub tool_requirements: Vec<ToolRequirement>,
    
    /// Environmental factors
    pub environment: EnvironmentContext,
    
    /// Historical performance
    pub historical_data: HistoricalContext,
    
    /// Adaptation parameters
    pub adaptations: AdaptationContext,
    
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Cognitive state for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    pub reasoning_depth: u8,
    pub creativity_level: f32,
    pub risk_tolerance: f32,
    pub focus_intensity: f32,
    pub emotional_state: String,
    pub confidence_threshold: f32,
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self {
            reasoning_depth: 3,
            creativity_level: 0.5,
            risk_tolerance: 0.3,
            focus_intensity: 0.7,
            emotional_state: "neutral".to_string(),
            confidence_threshold: 0.7,
        }
    }
}

/// Resource context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContext {
    pub available_memory_mb: usize,
    pub cpu_cores: usize,
    pub gpu_available: bool,
    pub network_quality: NetworkQuality,
    pub time_budget_seconds: Option<u64>,
    pub token_budget: Option<usize>,
}

/// Network quality indicator
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NetworkQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Offline,
}

/// Agent assignment for task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAssignment {
    pub agent_id: String,
    pub role: String,
    pub specialization: String,
    pub availability: f32,
    pub confidence: f32,
}

/// Tool requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequirement {
    pub tool_id: String,
    pub required: bool,
    pub alternatives: Vec<String>,
    pub parameters: HashMap<String, Value>,
}

/// Environmental context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    pub current_tab: TabId,
    pub active_users: usize,
    pub system_load: f32,
    pub time_of_day: String,
    pub urgency_level: UrgencyLevel,
}

/// Urgency levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Low,
    Normal,
    High,
    Critical,
}

/// Historical execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalContext {
    pub similar_task_count: usize,
    pub average_success_rate: f32,
    pub average_duration_ms: u64,
    pub common_failures: Vec<String>,
    pub best_practices: Vec<String>,
}

/// Adaptation context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationContext {
    pub strategy: AdaptationStrategy,
    pub learning_enabled: bool,
    pub feedback_incorporated: bool,
    pub optimization_goals: Vec<OptimizationGoal>,
}

/// Adaptation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    Conservative,
    Balanced,
    Aggressive,
    Experimental,
    Learning,
}

/// Optimization goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationGoal {
    Speed,
    Quality,
    ResourceEfficiency,
    Reliability,
    Innovation,
}

/// Context enrichment strategy
#[derive(Debug, Clone)]
pub struct EnrichmentStrategy {
    pub name: String,
    pub priority: u8,
    pub enricher: Arc<dyn ContextEnricher>,
}

/// Trait for context enrichment
#[async_trait::async_trait]
pub trait ContextEnricher: Send + Sync + std::fmt::Debug {
    async fn enrich(&self, context: &mut ExecutionContext) -> Result<()>;
}

/// Trait for execution adaptation
#[async_trait::async_trait]
pub trait ExecutionAdapter: Send + Sync {
    async fn adapt(&self, context: &ExecutionContext) -> Result<ExecutionPlan>;
}

/// Execution plan after adaptation
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecutionStep>,
    pub estimated_duration_ms: u64,
    pub resource_requirements: ResourceContext,
    pub fallback_plans: Vec<FallbackPlan>,
}

/// Individual execution step
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub id: String,
    pub description: String,
    pub executor: ExecutorType,
    pub parameters: HashMap<String, Value>,
    pub dependencies: Vec<String>,
    pub timeout_ms: Option<u64>,
}

/// Executor types
#[derive(Debug, Clone)]
pub enum ExecutorType {
    Agent(String),
    Tool(String),
    Cognitive,
    Manual,
    Hybrid(Vec<ExecutorType>),
}

/// Fallback plan
#[derive(Debug, Clone)]
pub struct FallbackPlan {
    pub trigger_condition: String,
    pub alternative_steps: Vec<ExecutionStep>,
}

/// Execution metrics
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    pub total_executions: usize,
    pub successful_executions: usize,
    pub context_enrichments: usize,
    pub adaptations_applied: usize,
    pub average_context_score: f32,
    pub resource_efficiency: f32,
}

/// Context execution events
#[derive(Debug, Clone)]
pub enum ContextExecutionEvent {
    ContextEnriched {
        task_id: String,
        enrichments: Vec<String>,
    },
    ExecutionAdapted {
        task_id: String,
        strategy: AdaptationStrategy,
    },
    ExecutionStarted {
        task_id: String,
        plan: String,
    },
    ExecutionCompleted {
        task_id: String,
        success: bool,
        duration_ms: u64,
    },
    FallbackTriggered {
        task_id: String,
        reason: String,
    },
}

/// Configuration
#[derive(Debug, Clone)]
pub struct ContextExecutionConfig {
    pub enable_auto_enrichment: bool,
    pub max_enrichment_depth: u8,
    pub enable_adaptive_execution: bool,
    pub context_cache_duration_seconds: u64,
    pub parallel_execution_limit: usize,
    pub enable_learning: bool,
}

impl Default for ContextExecutionConfig {
    fn default() -> Self {
        Self {
            enable_auto_enrichment: true,
            max_enrichment_depth: 3,
            enable_adaptive_execution: true,
            context_cache_duration_seconds: 300,
            parallel_execution_limit: 5,
            enable_learning: true,
        }
    }
}

impl ContextAwareExecutor {
    /// Create a new context-aware executor
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(100);
        
        Self {
            story_engine: None,
            cognitive_orchestrator: None,
            agent_orchestrator: None,
            tool_bridge: None,
            todo_manager: None,
            execution_contexts: Arc::new(RwLock::new(HashMap::new())),
            enrichment_strategies: Arc::new(RwLock::new(Vec::new())),
            execution_adapters: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
            event_tx,
            config: ContextExecutionConfig::default(),
        }
    }
    
    /// Set story engine
    pub fn set_story_engine(&mut self, engine: Arc<StoryEngine>) {
        self.story_engine = Some(engine);
    }
    
    /// Set cognitive orchestrator
    pub fn set_cognitive_orchestrator(&mut self, orchestrator: Arc<CognitiveOrchestrator>) {
        self.cognitive_orchestrator = Some(orchestrator);
    }
    
    /// Execute task with full context awareness
    pub async fn execute_with_context(
        &self,
        task_id: String,
        initial_context: Option<ExecutionContext>,
    ) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        
        // Build or enrich context
        let mut context = if let Some(ctx) = initial_context {
            ctx
        } else {
            self.build_context(&task_id).await?
        };
        
        // Enrich context if enabled
        if self.config.enable_auto_enrichment {
            self.enrich_context(&mut context).await?;
        }
        
        // Store context
        self.execution_contexts.write().await.insert(task_id.clone(), context.clone());
        
        // Adapt execution based on context
        let execution_plan = if self.config.enable_adaptive_execution {
            self.adapt_execution(&context).await?
        } else {
            self.create_default_plan(&context).await?
        };
        
        // Send execution started event
        let _ = self.event_tx.send(ContextExecutionEvent::ExecutionStarted {
            task_id: task_id.clone(),
            plan: format!("{} steps", execution_plan.steps.len()),
        });
        
        // Execute the plan
        let result = self.execute_plan(execution_plan, &context).await?;
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_executions += 1;
        if result.success {
            metrics.successful_executions += 1;
        }
        
        // Send completion event
        let _ = self.event_tx.send(ContextExecutionEvent::ExecutionCompleted {
            task_id: task_id.clone(),
            success: result.success,
            duration_ms: start_time.elapsed().as_millis() as u64,
        });
        
        // Learn from execution if enabled
        if self.config.enable_learning {
            self.learn_from_execution(&context, &result).await?;
        }
        
        Ok(result)
    }
    
    /// Build initial context
    async fn build_context(&self, task_id: &str) -> Result<ExecutionContext> {
        let mut context = ExecutionContext {
            task_id: task_id.to_string(),
            story_context: None,
            cognitive_state: CognitiveState::default(),
            available_resources: self.assess_resources().await?,
            agent_assignments: Vec::new(),
            tool_requirements: Vec::new(),
            environment: self.assess_environment().await?,
            historical_data: self.gather_historical_data(task_id).await?,
            adaptations: AdaptationContext {
                strategy: AdaptationStrategy::Balanced,
                learning_enabled: self.config.enable_learning,
                feedback_incorporated: false,
                optimization_goals: vec![OptimizationGoal::Quality, OptimizationGoal::Speed],
            },
            metadata: HashMap::new(),
        };
        
        // Add story context if available
        if let Some(ref story_engine) = self.story_engine {
            context.story_context = Some(story_engine.get_current_context().await?.into());
        }
        
        // Add cognitive state if available
        if let Some(ref cognitive) = self.cognitive_orchestrator {
            // Get cognitive metrics and adjust state
            let metrics = cognitive.get_cognitive_metrics().await;
            context.cognitive_state.confidence_threshold = metrics.decision_quality as f32;
            context.cognitive_state.focus_intensity = metrics.overall_awareness as f32;
        }
        
        Ok(context)
    }
    
    /// Enrich context with additional information
    async fn enrich_context(&self, context: &mut ExecutionContext) -> Result<()> {
        let strategies = self.enrichment_strategies.read().await;
        
        // Sort by priority and apply enrichments
        let mut sorted_strategies = strategies.clone();
        sorted_strategies.sort_by_key(|s| std::cmp::Reverse(s.priority));
        
        for strategy in sorted_strategies.iter().take(self.config.max_enrichment_depth as usize) {
            if let Err(e) = strategy.enricher.enrich(context).await {
                warn!("Enrichment strategy {} failed: {}", strategy.name, e);
            }
        }
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.context_enrichments += 1;
        
        // Send enrichment event
        let _ = self.event_tx.send(ContextExecutionEvent::ContextEnriched {
            task_id: context.task_id.clone(),
            enrichments: sorted_strategies.iter().map(|s| s.name.clone()).collect(),
        });
        
        Ok(())
    }
    
    /// Adapt execution based on context
    async fn adapt_execution(&self, context: &ExecutionContext) -> Result<ExecutionPlan> {
        // Select adapter based on context
        let adapter_name = self.select_adapter(context).await?;
        
        let adapters = self.execution_adapters.read().await;
        
        if let Some(adapter) = adapters.get(&adapter_name) {
            let plan = adapter.adapt(context).await?;
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.adaptations_applied += 1;
            
            // Send adaptation event
            let _ = self.event_tx.send(ContextExecutionEvent::ExecutionAdapted {
                task_id: context.task_id.clone(),
                strategy: context.adaptations.strategy,
            });
            
            Ok(plan)
        } else {
            // Fallback to default plan
            self.create_default_plan(context).await
        }
    }
    
    /// Select appropriate adapter
    async fn select_adapter(&self, context: &ExecutionContext) -> Result<String> {
        // Logic to select adapter based on context
        // For now, use a simple selection
        
        if context.story_context.is_some() {
            Ok("story_aware".to_string())
        } else if !context.agent_assignments.is_empty() {
            Ok("agent_collaborative".to_string())
        } else if !context.tool_requirements.is_empty() {
            Ok("tool_focused".to_string())
        } else {
            Ok("default".to_string())
        }
    }
    
    /// Create default execution plan
    async fn create_default_plan(&self, context: &ExecutionContext) -> Result<ExecutionPlan> {
        Ok(ExecutionPlan {
            steps: vec![
                ExecutionStep {
                    id: "step_1".to_string(),
                    description: format!("Execute task {}", context.task_id),
                    executor: ExecutorType::Cognitive,
                    parameters: HashMap::new(),
                    dependencies: Vec::new(),
                    timeout_ms: Some(30000),
                },
            ],
            estimated_duration_ms: 30000,
            resource_requirements: context.available_resources.clone(),
            fallback_plans: Vec::new(),
        })
    }
    
    /// Execute the plan
    async fn execute_plan(
        &self,
        plan: ExecutionPlan,
        context: &ExecutionContext,
    ) -> Result<ExecutionResult> {
        let mut results = Vec::new();
        let mut overall_success = true;
        
        for step in plan.steps {
            let step_result = self.execute_step(step, context).await;
            
            match step_result {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    warn!("Step execution failed: {}", e);
                    
                    // Try fallback if available
                    if let Some(fallback) = plan.fallback_plans.first() {
                        let _ = self.event_tx.send(ContextExecutionEvent::FallbackTriggered {
                            task_id: context.task_id.clone(),
                            reason: e.to_string(),
                        });
                        
                        // Execute fallback steps
                        for fallback_step in &fallback.alternative_steps {
                            if let Ok(result) = self.execute_step(fallback_step.clone(), context).await {
                                results.push(result);
                            }
                        }
                    } else {
                        overall_success = false;
                        break;
                    }
                }
            }
        }
        
        Ok(ExecutionResult {
            success: overall_success,
            step_results: results,
            context_score: self.calculate_context_score(context),
            adaptations_used: vec![context.adaptations.strategy],
        })
    }
    
    /// Execute individual step
    async fn execute_step(
        &self,
        step: ExecutionStep,
        context: &ExecutionContext,
    ) -> Result<StepResult> {
        let start_time = std::time::Instant::now();
        
        let output = match step.executor {
            ExecutorType::Agent(agent_id) => {
                // Execute via agent
                serde_json::json!({
                    "agent": agent_id,
                    "executed": true,
                })
            }
            ExecutorType::Tool(tool_id) => {
                // Execute via tool
                if let Some(ref tool_bridge) = self.tool_bridge {
                    let result = tool_bridge.execute_from_chat(
                        tool_id,
                        serde_json::json!(step.parameters),
                    ).await?;
                    serde_json::to_value(result)?
                } else {
                    serde_json::json!({"error": "Tool bridge not available"})
                }
            }
            ExecutorType::Cognitive => {
                // Execute via cognitive orchestrator
                if let Some(ref cognitive) = self.cognitive_orchestrator {
                    let response = cognitive.process_with_story_context(&step.description).await?;
                    serde_json::json!({
                        "thought": response.thought.content,
                        "story_influenced": response.story_influenced,
                    })
                } else {
                    serde_json::json!({"error": "Cognitive orchestrator not available"})
                }
            }
            _ => serde_json::json!({"status": "simulated"}),
        };
        
        Ok(StepResult {
            step_id: step.id,
            success: true,
            duration_ms: start_time.elapsed().as_millis() as u64,
            output,
        })
    }
    
    /// Assess available resources
    async fn assess_resources(&self) -> Result<ResourceContext> {
        Ok(ResourceContext {
            available_memory_mb: 1024,
            cpu_cores: 4,
            gpu_available: false,
            network_quality: NetworkQuality::Good,
            time_budget_seconds: Some(60),
            token_budget: Some(4000),
        })
    }
    
    /// Assess environment
    async fn assess_environment(&self) -> Result<EnvironmentContext> {
        Ok(EnvironmentContext {
            current_tab: TabId::Chat,
            active_users: 1,
            system_load: 0.3,
            time_of_day: "afternoon".to_string(),
            urgency_level: UrgencyLevel::Normal,
        })
    }
    
    /// Gather historical data
    async fn gather_historical_data(&self, _task_id: &str) -> Result<HistoricalContext> {
        Ok(HistoricalContext {
            similar_task_count: 5,
            average_success_rate: 0.8,
            average_duration_ms: 5000,
            common_failures: vec!["timeout".to_string()],
            best_practices: vec!["validate input".to_string()],
        })
    }
    
    /// Calculate context score
    fn calculate_context_score(&self, context: &ExecutionContext) -> f32 {
        let mut score = 0.0;
        let mut factors = 0;
        
        // Story context contribution
        if context.story_context.is_some() {
            score += 0.2;
            factors += 1;
        }
        
        // Resource availability
        if context.available_resources.available_memory_mb > 512 {
            score += 0.2;
            factors += 1;
        }
        
        // Agent assignments
        if !context.agent_assignments.is_empty() {
            score += 0.2;
            factors += 1;
        }
        
        // Historical success rate
        if context.historical_data.average_success_rate > 0.7 {
            score += 0.2;
            factors += 1;
        }
        
        // Cognitive confidence
        if context.cognitive_state.confidence_threshold > 0.6 {
            score += 0.2;
            factors += 1;
        }
        
        if factors > 0 {
            score / factors as f32
        } else {
            0.5
        }
    }
    
    /// Learn from execution
    async fn learn_from_execution(
        &self,
        context: &ExecutionContext,
        result: &ExecutionResult,
    ) -> Result<()> {
        // Store learning data for future adaptations
        debug!(
            "Learning from execution: task={}, success={}, score={}",
            context.task_id, result.success, result.context_score
        );
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.average_context_score = 
            (metrics.average_context_score * metrics.total_executions as f32 + result.context_score) 
            / (metrics.total_executions + 1) as f32;
        
        Ok(())
    }
    
    /// Register enrichment strategy
    pub async fn register_enrichment(&self, strategy: EnrichmentStrategy) {
        let mut strategies = self.enrichment_strategies.write().await;
        strategies.push(strategy);
        info!("Registered enrichment strategy");
    }
    
    /// Register execution adapter
    pub async fn register_adapter(&self, name: String, adapter: Box<dyn ExecutionAdapter>) {
        let mut adapters = self.execution_adapters.write().await;
        adapters.insert(name.clone(), adapter);
        info!("Registered execution adapter: {}", name);
    }
    
    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<ContextExecutionEvent> {
        self.event_tx.subscribe()
    }
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub step_results: Vec<StepResult>,
    pub context_score: f32,
    pub adaptations_used: Vec<AdaptationStrategy>,
}

/// Step execution result
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: String,
    pub success: bool,
    pub duration_ms: u64,
    pub output: Value,
}