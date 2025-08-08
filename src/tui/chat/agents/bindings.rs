//! Agent Bindings System
//! 
//! Manages the associations between agents, models, and tools, including
//! dynamic binding, capability matching, and resource allocation.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{info, debug, warn};

use super::creation::AgentConfig;
use crate::tui::chat::models::catalog::ModelEntry;

/// Agent binding manager
pub struct BindingManager {
    /// Agent-model bindings
    agent_models: Arc<RwLock<HashMap<String, ModelBinding>>>,
    
    /// Agent-tool bindings
    agent_tools: Arc<RwLock<HashMap<String, ToolBinding>>>,
    
    /// Model capabilities
    model_capabilities: Arc<RwLock<HashMap<String, ModelCapabilities>>>,
    
    /// Tool capabilities
    tool_capabilities: Arc<RwLock<HashMap<String, ToolCapabilities>>>,
    
    /// Binding policies
    policies: Arc<RwLock<Vec<BindingPolicy>>>,
    
    /// Resource allocations
    allocations: Arc<RwLock<ResourceAllocations>>,
}

/// Model binding for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBinding {
    pub agent_id: String,
    pub primary_model: String,
    pub fallback_models: Vec<String>,
    pub model_selection_rules: Vec<SelectionRule>,
    pub usage_limits: UsageLimits,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}

/// Tool binding for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolBinding {
    pub agent_id: String,
    pub allowed_tools: HashSet<String>,
    pub blocked_tools: HashSet<String>,
    pub tool_permissions: HashMap<String, ToolPermission>,
    pub execution_rules: Vec<ExecutionRule>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub model_id: String,
    pub supports_streaming: bool,
    pub supports_functions: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub max_tokens: usize,
    pub context_window: usize,
    pub languages: Vec<String>,
    pub specializations: Vec<String>,
}

/// Tool capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapabilities {
    pub tool_id: String,
    pub category: ToolCategory,
    pub requires_auth: bool,
    pub supports_async: bool,
    pub input_types: Vec<DataType>,
    pub output_types: Vec<DataType>,
    pub rate_limits: RateLimits,
    pub dependencies: Vec<String>,
}

/// Tool categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolCategory {
    FileSystem,
    Network,
    Database,
    CodeExecution,
    DataAnalysis,
    Communication,
    Monitoring,
    Security,
    Custom(String),
}

/// Data types for tool I/O
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Text,
    Json,
    Binary,
    Image,
    Audio,
    Video,
    Structured(String),
}

/// Selection rule for model choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRule {
    pub rule_id: String,
    pub condition: SelectionCondition,
    pub preferred_model: String,
    pub priority: u8,
}

/// Selection conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionCondition {
    TaskType(String),
    TokenCount { min: usize, max: usize },
    Language(String),
    TimeOfDay { start_hour: u8, end_hour: u8 },
    CostThreshold(f64),
    PerformanceRequirement(PerformanceReq),
    Custom(String),
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReq {
    pub max_latency_ms: u32,
    pub min_throughput_tps: f64,
    pub min_accuracy: f64,
}

/// Tool permission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPermission {
    pub can_execute: bool,
    pub requires_approval: bool,
    pub auto_approve_conditions: Vec<String>,
    pub max_executions_per_hour: Option<u32>,
    pub allowed_operations: HashSet<String>,
}

/// Execution rule for tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRule {
    pub rule_id: String,
    pub tool_pattern: String,
    pub condition: ExecutionCondition,
    pub action: ExecutionAction,
}

/// Execution conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionCondition {
    Always,
    Never,
    RequiresApproval,
    TimeWindow { start: String, end: String },
    ResourceAvailable(String),
    Custom(String),
}

/// Execution actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionAction {
    Allow,
    Block,
    Queue,
    Redirect(String),
    Log,
    Notify(String),
}

/// Usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLimits {
    pub max_requests_per_minute: Option<u32>,
    pub max_tokens_per_day: Option<u32>,
    pub max_cost_per_hour: Option<f64>,
    pub priority_level: PriorityLevel,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum PriorityLevel {
    Low,
    Normal,
    High,
    Critical,
}

/// Rate limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_second: Option<f64>,
    pub requests_per_minute: Option<u32>,
    pub requests_per_hour: Option<u32>,
    pub concurrent_limit: Option<u32>,
}

/// Binding policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingPolicy {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub agent_pattern: String,
    pub model_requirements: Vec<String>,
    pub tool_requirements: Vec<String>,
    pub constraints: Vec<Constraint>,
    pub active: bool,
}

/// Constraint for bindings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    RequireModelCategory(String),
    RequireToolCategory(ToolCategory),
    MaxCostPerRequest(f64),
    MinPerformanceScore(f64),
    RequireLocalModel,
    RequireSecureConnection,
    Custom(String),
}

/// Resource allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocations {
    /// CPU allocations per agent
    pub cpu_allocations: HashMap<String, f32>,
    
    /// Memory allocations per agent (MB)
    pub memory_allocations: HashMap<String, u32>,
    
    /// Token allocations per agent
    pub token_allocations: HashMap<String, TokenAllocation>,
    
    /// Total available resources
    pub total_resources: SystemResources,
}

/// Token allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAllocation {
    pub daily_limit: u32,
    pub used_today: u32,
    pub reserved: u32,
    pub burst_allowed: bool,
}

/// System resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    pub total_cpu_cores: f32,
    pub total_memory_mb: u32,
    pub total_daily_tokens: u32,
    pub available_cpu: f32,
    pub available_memory: u32,
    pub available_tokens: u32,
}

impl BindingManager {
    /// Create a new binding manager
    pub fn new() -> Self {
        Self {
            agent_models: Arc::new(RwLock::new(HashMap::new())),
            agent_tools: Arc::new(RwLock::new(HashMap::new())),
            model_capabilities: Arc::new(RwLock::new(HashMap::new())),
            tool_capabilities: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(Vec::new())),
            allocations: Arc::new(RwLock::new(ResourceAllocations::default())),
        }
    }
    
    /// Bind a model to an agent
    pub async fn bind_model(
        &self,
        agent_id: String,
        primary_model: String,
        fallback_models: Vec<String>,
    ) -> Result<()> {
        let binding = ModelBinding {
            agent_id: agent_id.clone(),
            primary_model,
            fallback_models,
            model_selection_rules: Vec::new(),
            usage_limits: UsageLimits {
                max_requests_per_minute: Some(60),
                max_tokens_per_day: Some(100000),
                max_cost_per_hour: Some(1.0),
                priority_level: PriorityLevel::Normal,
            },
            created_at: chrono::Utc::now(),
            last_used: None,
        };
        
        self.agent_models.write().await.insert(agent_id, binding);
        info!("Model binding created for agent");
        Ok(())
    }
    
    /// Bind tools to an agent
    pub async fn bind_tools(
        &self,
        agent_id: String,
        allowed_tools: Vec<String>,
        blocked_tools: Vec<String>,
    ) -> Result<()> {
        let binding = ToolBinding {
            agent_id: agent_id.clone(),
            allowed_tools: allowed_tools.into_iter().collect(),
            blocked_tools: blocked_tools.into_iter().collect(),
            tool_permissions: HashMap::new(),
            execution_rules: Vec::new(),
            created_at: chrono::Utc::now(),
        };
        
        self.agent_tools.write().await.insert(agent_id, binding);
        info!("Tool binding created for agent");
        Ok(())
    }
    
    /// Get best model for agent and task
    pub async fn get_best_model(
        &self,
        agent_id: &str,
        task_context: &TaskContext,
    ) -> Result<String> {
        let bindings = self.agent_models.read().await;
        
        if let Some(binding) = bindings.get(agent_id) {
            // Check selection rules
            for rule in &binding.model_selection_rules {
                if self.matches_condition(&rule.condition, task_context) {
                    return Ok(rule.preferred_model.clone());
                }
            }
            
            // Return primary model by default
            Ok(binding.primary_model.clone())
        } else {
            Err(anyhow::anyhow!("No model binding found for agent"))
        }
    }
    
    /// Check if tool is allowed for agent
    pub async fn is_tool_allowed(
        &self,
        agent_id: &str,
        tool_id: &str,
    ) -> Result<bool> {
        let bindings = self.agent_tools.read().await;
        
        if let Some(binding) = bindings.get(agent_id) {
            if binding.blocked_tools.contains(tool_id) {
                return Ok(false);
            }
            
            if binding.allowed_tools.is_empty() {
                return Ok(true); // All tools allowed if no explicit list
            }
            
            Ok(binding.allowed_tools.contains(tool_id))
        } else {
            Ok(true) // No binding means no restrictions
        }
    }
    
    /// Add model capabilities
    pub async fn register_model_capabilities(
        &self,
        model_id: String,
        capabilities: ModelCapabilities,
    ) -> Result<()> {
        self.model_capabilities.write().await.insert(model_id, capabilities);
        Ok(())
    }
    
    /// Add tool capabilities
    pub async fn register_tool_capabilities(
        &self,
        tool_id: String,
        capabilities: ToolCapabilities,
    ) -> Result<()> {
        self.tool_capabilities.write().await.insert(tool_id, capabilities);
        Ok(())
    }
    
    /// Find compatible models for requirements
    pub async fn find_compatible_models(
        &self,
        requirements: &ModelRequirements,
    ) -> Vec<String> {
        let capabilities = self.model_capabilities.read().await;
        
        capabilities
            .iter()
            .filter(|(_, cap)| {
                cap.context_window >= requirements.min_context_window &&
                cap.max_tokens >= requirements.min_max_tokens &&
                (!requirements.requires_streaming || cap.supports_streaming) &&
                (!requirements.requires_functions || cap.supports_functions) &&
                (!requirements.requires_vision || cap.supports_vision)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Find compatible tools for requirements
    pub async fn find_compatible_tools(
        &self,
        requirements: &ToolRequirements,
    ) -> Vec<String> {
        let capabilities = self.tool_capabilities.read().await;
        
        capabilities
            .iter()
            .filter(|(_, cap)| {
                requirements.categories.is_empty() ||
                requirements.categories.contains(&cap.category)
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Allocate resources for agent
    pub async fn allocate_resources(
        &self,
        agent_id: String,
        cpu: f32,
        memory: u32,
        tokens: u32,
    ) -> Result<()> {
        let mut allocations = self.allocations.write().await;
        
        // Check if resources are available
        if allocations.total_resources.available_cpu < cpu {
            return Err(anyhow::anyhow!("Insufficient CPU resources"));
        }
        if allocations.total_resources.available_memory < memory {
            return Err(anyhow::anyhow!("Insufficient memory resources"));
        }
        if allocations.total_resources.available_tokens < tokens {
            return Err(anyhow::anyhow!("Insufficient token allocation"));
        }
        
        // Allocate resources
        allocations.cpu_allocations.insert(agent_id.clone(), cpu);
        allocations.memory_allocations.insert(agent_id.clone(), memory);
        allocations.token_allocations.insert(agent_id, TokenAllocation {
            daily_limit: tokens,
            used_today: 0,
            reserved: tokens / 10, // Reserve 10%
            burst_allowed: true,
        });
        
        // Update available resources
        allocations.total_resources.available_cpu -= cpu;
        allocations.total_resources.available_memory -= memory;
        allocations.total_resources.available_tokens -= tokens;
        
        Ok(())
    }
    
    /// Check condition match
    fn matches_condition(&self, condition: &SelectionCondition, context: &TaskContext) -> bool {
        match condition {
            SelectionCondition::TaskType(task_type) => context.task_type == *task_type,
            SelectionCondition::TokenCount { min, max } => {
                context.estimated_tokens >= *min && context.estimated_tokens <= *max
            }
            SelectionCondition::Language(lang) => context.language == *lang,
            SelectionCondition::CostThreshold(threshold) => context.max_cost <= *threshold,
            _ => false,
        }
    }
}

/// Task context for model selection
#[derive(Debug, Clone)]
pub struct TaskContext {
    pub task_type: String,
    pub estimated_tokens: usize,
    pub language: String,
    pub max_cost: f64,
    pub priority: PriorityLevel,
}

/// Model requirements
#[derive(Debug, Clone)]
pub struct ModelRequirements {
    pub min_context_window: usize,
    pub min_max_tokens: usize,
    pub requires_streaming: bool,
    pub requires_functions: bool,
    pub requires_vision: bool,
}

/// Tool requirements
#[derive(Debug, Clone)]
pub struct ToolRequirements {
    pub categories: Vec<ToolCategory>,
    pub requires_async: bool,
    pub input_types: Vec<DataType>,
}

impl Default for ResourceAllocations {
    fn default() -> Self {
        Self {
            cpu_allocations: HashMap::new(),
            memory_allocations: HashMap::new(),
            token_allocations: HashMap::new(),
            total_resources: SystemResources {
                total_cpu_cores: 8.0,
                total_memory_mb: 16384,
                total_daily_tokens: 1000000,
                available_cpu: 8.0,
                available_memory: 16384,
                available_tokens: 1000000,
            },
        }
    }
}

/// Dynamic binding optimizer
pub struct BindingOptimizer {
    /// Historical performance data
    performance_history: Arc<RwLock<HashMap<String, PerformanceHistory>>>,
    
    /// Optimization strategies
    strategies: Vec<OptimizationStrategy>,
}

/// Performance history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    pub agent_id: String,
    pub model_id: String,
    pub task_type: String,
    pub avg_latency_ms: f64,
    pub avg_tokens_per_second: f64,
    pub success_rate: f64,
    pub total_requests: u32,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    MinimizeLatency,
    MaximizeThroughput,
    MinimizeCost,
    BalancedPerformance,
    AdaptiveLoadBalancing,
}

impl BindingOptimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            strategies: vec![
                OptimizationStrategy::BalancedPerformance,
                OptimizationStrategy::AdaptiveLoadBalancing,
            ],
        }
    }
    
    /// Optimize bindings based on performance
    pub async fn optimize_bindings(
        &self,
        manager: &BindingManager,
        agent_id: &str,
    ) -> Result<()> {
        let history = self.performance_history.read().await;
        
        // Analyze performance history
        let agent_history: Vec<_> = history
            .values()
            .filter(|h| h.agent_id == agent_id)
            .collect();
        
        if agent_history.is_empty() {
            return Ok(()); // No history to optimize
        }
        
        // Find best performing model
        let best_model = agent_history
            .iter()
            .max_by(|a, b| {
                let a_score = a.success_rate * a.avg_tokens_per_second / (a.avg_latency_ms + 1.0);
                let b_score = b.success_rate * b.avg_tokens_per_second / (b.avg_latency_ms + 1.0);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .map(|h| h.model_id.clone());
        
        if let Some(model) = best_model {
            // Update binding to prefer best performing model
            let mut bindings = manager.agent_models.write().await;
            if let Some(binding) = bindings.get_mut(agent_id) {
                binding.primary_model = model;
                info!("Optimized model binding for agent {}", agent_id);
            }
        }
        
        Ok(())
    }
    
    /// Record performance metrics
    pub async fn record_performance(
        &self,
        agent_id: String,
        model_id: String,
        task_type: String,
        latency_ms: f64,
        tokens_per_second: f64,
        success: bool,
    ) {
        let key = format!("{}:{}:{}", agent_id, model_id, task_type);
        let mut history = self.performance_history.write().await;
        
        let entry = history.entry(key).or_insert_with(|| PerformanceHistory {
            agent_id,
            model_id,
            task_type,
            avg_latency_ms: 0.0,
            avg_tokens_per_second: 0.0,
            success_rate: 0.0,
            total_requests: 0,
            last_updated: chrono::Utc::now(),
        });
        
        // Update rolling averages
        let n = entry.total_requests as f64;
        entry.avg_latency_ms = (entry.avg_latency_ms * n + latency_ms) / (n + 1.0);
        entry.avg_tokens_per_second = (entry.avg_tokens_per_second * n + tokens_per_second) / (n + 1.0);
        entry.success_rate = (entry.success_rate * n + if success { 1.0 } else { 0.0 }) / (n + 1.0);
        entry.total_requests += 1;
        entry.last_updated = chrono::Utc::now();
    }
}
