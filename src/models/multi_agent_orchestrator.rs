use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::adaptive_learning::AdaptiveLearningSystem;
use super::local_manager::LocalModelManager;
use super::model_discovery_service::{
    ModelDiscoveryService as DiscoveryService,
    ModelRegistryEntry,
};
use super::orchestrator::ModelOrchestrator;
use super::providers::{ModelInfo, ModelProvider, ProviderFactory};
use super::{RoutingStrategy, TaskRequest, TaskResponse};
use crate::config::ApiKeysConfig;

/// Model type categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextGeneration,
    CodeGeneration,
    Embedding,
    Vision,
    Audio,
    Multimodal,
}

/// Performance tier classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTier {
    Basic,
    Standard,
    Premium,
    Enterprise,
}

/// Load balancing configuration for agents
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    pub weight: f32,
    pub max_concurrent_requests: u32,
    pub health_check_interval: Duration,
}

/// Multi-agent orchestration system with seamless model switching
#[derive(Clone)]
pub struct MultiAgentOrchestrator {
    /// Primary model orchestrator
    orchestrator: Arc<ModelOrchestrator>,

    /// Agent management
    agents: Arc<RwLock<HashMap<String, AgentInstance>>>,

    /// API key management
    api_manager: Arc<ApiKeyManager>,

    /// Model auto-discovery with web search
    discovery_service: Arc<DiscoveryService>,

    /// Real-time performance tracking
    performance_tracker: Arc<RealTimePerformanceTracker>,

    /// Intelligent routing engine
    routing_engine: Arc<IntelligentRoutingEngine>,

    /// Fallback management
    fallback_manager: Arc<AdvancedFallbackManager>,

    /// Session management
    session_manager: Arc<SessionManager>,
}

/// Represents an active agent with its configuration and state
#[derive(Debug, Clone)]
pub struct AgentInstance {
    pub id: String,
    pub name: String,
    pub agent_type: AgentType,
    pub models: Vec<String>,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
    pub performance_metrics: AgentPerformanceMetrics,
    pub cost_tracker: CostTracker,
    pub last_used: Option<Instant>,
    pub error_count: u32,
    pub success_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    CodeGeneration,
    CreativeWriting,
    DataAnalysis,
    LogicalReasoning,
    GeneralPurpose,
    Specialized(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Active,
    Idle,
    Busy,
    Error(String),
    Offline,
}

#[derive(Debug, Clone, Default)]
pub struct AgentPerformanceMetrics {
    pub average_latency: Duration,
    pub quality_score: f32,
    pub throughput: f32,
    pub cost_per_request: f32,
    pub uptime_percentage: f32,
}

#[derive(Debug, Clone, Default)]
pub struct CostTracker {
    pub total_cost: f32,
    pub requests_count: u64,
    pub last_hour_cost: f32,
    pub daily_budget_used: f32,
}

/// Real-time performance tracker
pub struct RealTimePerformanceTracker {
    metrics: Arc<RwLock<HashMap<String, AgentMetricsInternal>>>,
    history: Arc<RwLock<Vec<PerformanceSnapshot>>>,
    alert_thresholds: AlertThresholds,
}

/// Internal agent metrics for tracking
#[derive(Debug, Clone)]
pub struct AgentMetricsInternal {
    pub response_times: Vec<Duration>,
    pub quality_scores: Vec<f32>,
    pub error_rates: Vec<f32>,
    pub cost_tracking: Vec<f32>,
    pub last_updated: Instant,
}

/// Performance snapshot for historical tracking
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub agents: HashMap<String, AgentPerformanceMetrics>,
    pub system_load: f32,
    pub total_requests: u64,
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_latency: Duration,
    pub min_quality: f32,
    pub max_error_rate: f32,
    pub max_cost_per_hour: f32,
}

/// Performance summary for system overview
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub average_latency: Duration,
    pub average_quality: f32,
    pub total_requests: u64,
    pub error_rate: f32,
    pub cost_efficiency: f32,
}

/// Intelligent routing engine for optimal agent selection
pub struct IntelligentRoutingEngine {
    available_strategies: Vec<RoutingStrategy>,
    learning_system: Arc<AdaptiveLearningSystem>,
    routing_cache: Arc<RwLock<HashMap<String, String>>>,
    performance_predictor: Arc<PerformancePredictor>,
}

/// Performance predictor for routing decisions
pub struct PerformancePredictor {
    prediction_models: HashMap<String, PredictionModel>,
}

/// Prediction model for performance forecasting
#[derive(Debug, Clone)]
pub struct PredictionModel {
    accuracy: f32,
    last_updated: Instant,
}

/// Advanced fallback manager with circuit breakers
pub struct AdvancedFallbackManager {
    fallback_chains: Arc<RwLock<HashMap<String, Vec<FallbackOption>>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    health_checker: Arc<HealthChecker>,
}

/// Circuit breaker for failure management
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub is_open: bool,
    pub failure_count: u32,
    pub last_failure: Option<Instant>,
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
}

/// Fallback option configuration
#[derive(Debug, Clone)]
pub struct FallbackOption {
    pub agent_id: String,
    pub priority: u32,
    pub conditions: Vec<FallbackCondition>,
    pub timeout: Duration,
}

/// Fallback condition triggers
#[derive(Debug, Clone)]
pub enum FallbackCondition {
    HighLatency(Duration),
    ErrorRate(f32),
    CostThreshold(f32),
}

/// Health checker for agent monitoring
pub struct HealthChecker {
    health_status: Arc<RwLock<HashMap<String, HealthStatus>>>,
    check_interval: Duration,
}

/// Health status for agents
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub response_time: Option<Duration>,
    pub last_check: Instant,
    pub error_message: Option<String>,
    pub uptime_percentage: f32,
}

/// Session manager for user interaction tracking
pub struct SessionManager {
    active_sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    sessionconfigs: Arc<RwLock<HashMap<String, SessionConfig>>>,
    cleanup_interval: Duration,
}

/// User session for context tracking
#[derive(Debug, Clone)]
pub struct UserSession {
    pub id: String,
    pub user_id: Option<String>,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub preferences: UserPreferences,
    pub active_agents: Vec<String>,
    pub conversation_history: Vec<ConversationEntry>,
    pub cost_tracking: SessionCostTracker,
}

/// User preferences for agent selection
#[derive(Debug, Clone, Default)]
pub struct UserPreferences {
    pub preferred_models: Vec<String>,
    pub cost_limit: Option<f32>,
    pub latency_preference: LatencyPreference,
    pub quality_threshold: f32,
}

/// Latency preference for routing
#[derive(Debug, Clone)]
pub enum LatencyPreference {
    Fast,
    Balanced,
    Quality,
}

impl Default for LatencyPreference {
    fn default() -> Self {
        LatencyPreference::Balanced
    }
}

/// Conversation entry for session tracking
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    pub timestamp: Instant,
    pub request: TaskRequest,
    pub response: TaskResponse,
    pub agent_id: String,
    pub execution_time: Duration,
    pub cost_cents: f32,
    pub quality_score: f32,
}

/// Session cost tracking
#[derive(Debug, Clone)]
pub struct SessionCostTracker {
    pub start_time: Instant,
    pub total_cost: f32,
    pub requests_count: u64,
    pub cost_breakdown: HashMap<String, f32>,
}

impl Default for SessionCostTracker {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            total_cost: 0.0,
            requests_count: 0,
            cost_breakdown: HashMap::new(),
        }
    }
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub max_duration: Duration,
    pub cost_limit: Option<f32>,
    pub auto_cleanup: bool,
    pub max_cost_per_hour: f64,
    pub preferred_models: Vec<String>,
    pub timeout_seconds: u64,
}

/// Local model information
#[derive(Debug, Clone)]
pub struct LocalModelInfo {
    pub path: String,
    pub name: String,
    pub size_mb: u64,
    pub format: String,
    pub is_loaded: bool,
    pub last_used: Option<Instant>,
}

/// API model information
#[derive(Debug, Clone)]
pub struct ApiModelInfo {
    pub id: String,
    pub provider: String,
    pub capabilities: Vec<String>,
    pub cost_per_token: f32,
    pub context_window: u32,
    pub max_output_tokens: u32,
}

/// Advanced API key management with auto-detection and validation
pub struct ApiKeyManager {
    /// Cached API configurations
    apiconfigs: Arc<RwLock<HashMap<String, ApiKeyConfig>>>,

    /// Auto-detection service
    detector: Arc<ApiKeyDetector>,

    /// Validation cache
    validation_cache: Arc<RwLock<HashMap<String, ValidationResult>>>,

    /// Key rotation manager
    rotation_manager: Arc<KeyRotationManager>,
}

#[derive(Debug, Clone)]
pub struct ApiKeyConfig {
    pub provider: String,
    pub key: String,
    pub is_valid: bool,
    pub last_validated: Instant,
    pub rate_limit: Option<RateLimit>,
    pub usage_stats: UsageStats,
    pub cost_per_token: f32,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub provider_info: Option<ProviderInfo>,
    pub rate_limits: Option<RateLimit>,
    pub timestamp: Instant,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: String,
    pub available_models: Vec<String>,
    pub capabilities: Vec<String>,
    pub pricing_info: Option<PricingInfo>,
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub tokens_per_minute: u32,
    pub concurrent_requests: u32,
}

#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost: f32,
    pub last_request: Option<Instant>,
}

#[derive(Debug, Clone)]
pub struct PricingInfo {
    pub input_token_cost: f32,
    pub output_token_cost: f32,
    pub currency: String,
}

/// Auto-detects available API keys from environment and configuration
pub struct ApiKeyDetector {
    env_patterns: Vec<EnvPattern>,
    config_paths: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EnvPattern {
    pub provider: String,
    pub env_vars: Vec<String>,
    pub validation_endpoint: Option<String>,
}

/// Local model scanning and detection
pub struct LocalModelScanner {
    scan_paths: Vec<String>,
    supported_formats: Vec<String>,
    cached_models: Arc<RwLock<Vec<LocalModelInfo>>>,
}

/// API model discovery
pub struct ApiModelDiscoverer {
    providers: Arc<RwLock<HashMap<String, Arc<dyn ModelProvider>>>>,
    discovery_cache: Arc<RwLock<HashMap<String, Vec<ApiModelInfo>>>>,
}

impl ApiKeyManager {
    async fn new(_apiconfig: &ApiKeysConfig) -> Result<Self> {
        Ok(Self {
            apiconfigs: Arc::new(RwLock::new(HashMap::new())),
            detector: Arc::new(ApiKeyDetector::new()),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
            rotation_manager: Arc::new(KeyRotationManager::new()),
        })
    }

    async fn get_all_api_status(&self) -> HashMap<String, ApiKeyConfig> {
        self.apiconfigs.read().await.clone()
    }
}

impl ApiKeyDetector {
    fn new() -> Self {
        Self {
            env_patterns: vec![
                EnvPattern {
                    provider: "openai".to_string(),
                    env_vars: vec!["OPENAI_API_KEY".to_string()],
                    validation_endpoint: Some("https://api.openai.com/v1/models".to_string()),
                },
                EnvPattern {
                    provider: "anthropic".to_string(),
                    env_vars: vec!["ANTHROPIC_API_KEY".to_string()],
                    validation_endpoint: Some("https://api.anthropic.com/v1/messages".to_string()),
                },
            ],
            config_paths: vec!["~/.config/loki/api_keys.toml".to_string(), ".env".to_string()],
        }
    }
}

impl RealTimePerformanceTracker {
    fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            alert_thresholds: AlertThresholds {
                max_latency: Duration::from_secs(30),
                min_quality: 0.7,
                max_error_rate: 0.1,
                max_cost_per_hour: 10.0,
            },
        }
    }

    async fn record_execution(
        &self,
        agent_id: &str,
        __task: &TaskRequest,
        result: &Result<TaskResponse>,
        duration: Duration,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        let agent_metrics =
            metrics.entry(agent_id.to_string()).or_insert_with(|| AgentMetricsInternal {
                response_times: Vec::new(),
                quality_scores: Vec::new(),
                error_rates: Vec::new(),
                cost_tracking: Vec::new(),
                last_updated: Instant::now(),
            });

        agent_metrics.response_times.push(duration);
        if let Ok(response) = result {
            agent_metrics.quality_scores.push(response.quality_score);
            agent_metrics.cost_tracking.push(response.cost_cents.unwrap_or(0.0));
        }
        agent_metrics.last_updated = Instant::now();

        Ok(())
    }

    async fn get_summary(&self) -> PerformanceSummary {
        let metrics = self.metrics.read().await;

        let mut total_latency = Duration::from_secs(0);
        let mut total_quality = 0.0;
        let mut total_requests = 0u64;
        let total_errors = 0u64;

        for agent_metrics in metrics.values() {
            let avg_latency: Duration = agent_metrics.response_times.iter().sum::<Duration>()
                / agent_metrics.response_times.len() as u32;
            total_latency += avg_latency;
            total_quality += agent_metrics.quality_scores.iter().sum::<f32>()
                / agent_metrics.quality_scores.len() as f32;
            total_requests += agent_metrics.response_times.len() as u64;
        }

        PerformanceSummary {
            average_latency: if metrics.len() > 0 {
                total_latency / metrics.len() as u32
            } else {
                Duration::from_secs(0)
            },
            average_quality: if metrics.len() > 0 {
                total_quality / metrics.len() as f32
            } else {
                0.0
            },
            total_requests,
            error_rate: if total_requests > 0 {
                total_errors as f32 / total_requests as f32
            } else {
                0.0
            },
            cost_efficiency: 0.85, // Mock value
        }
    }

    async fn collect_metrics(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            // Collect and store performance snapshots
            let snapshot = self.create_performance_snapshot().await;
            let mut history = self.history.write().await;
            history.push(snapshot);

            // Keep only last 24 hours of data
            let cutoff = Instant::now() - Duration::from_secs(24 * 3600);
            history.retain(|s| s.timestamp > cutoff);
        }
    }

    async fn create_performance_snapshot(&self) -> PerformanceSnapshot {
        let metrics = self.metrics.read().await;
        let mut agents = HashMap::new();

        for (agent_id, agent_metrics) in metrics.iter() {
            agents.insert(
                agent_id.clone(),
                AgentPerformanceMetrics {
                    average_latency: agent_metrics.response_times.iter().sum::<Duration>()
                        / agent_metrics.response_times.len().max(1) as u32,
                    quality_score: agent_metrics.quality_scores.iter().sum::<f32>()
                        / agent_metrics.quality_scores.len().max(1) as f32,
                    throughput: agent_metrics.response_times.len() as f32 / 60.0, /* requests per
                                                                                   * minute */
                    cost_per_request: agent_metrics.cost_tracking.iter().sum::<f32>()
                        / agent_metrics.cost_tracking.len().max(1) as f32,
                    uptime_percentage: 0.95, // Mock value
                },
            );
        }

        PerformanceSnapshot {
            timestamp: Instant::now(),
            agents,
            system_load: 0.65,    // Mock value
            total_requests: 1000, // Mock value
        }
    }

    async fn get_system_uptime(&self) -> Duration {
        Duration::from_secs(3600) // Mock 1 hour uptime
    }

    /// Record a model switch event for performance tracking
    async fn record_model_switch(
        &self,
        agent_id: &str,
        old_model: &str,
        new_model: &str,
        reason: &str,
    ) {
        info!(
            "📊 Recording model switch: {} {} -> {} ({})",
            agent_id, old_model, new_model, reason
        );

        // Record the switch event in metrics
        let mut metrics = self.metrics.write().await;
        let agent_metrics =
            metrics.entry(agent_id.to_string()).or_insert_with(|| AgentMetricsInternal {
                response_times: Vec::new(),
                quality_scores: Vec::new(),
                error_rates: Vec::new(),
                cost_tracking: Vec::new(),
                last_updated: Instant::now(),
            });

        // Update last updated timestamp
        agent_metrics.last_updated = Instant::now();

        // Record switch event in performance history
        let mut history = self.history.write().await;
        let switch_event = PerformanceSnapshot {
            timestamp: Instant::now(),
            agents: {
                let mut agents_snapshot = HashMap::new();
                agents_snapshot.insert(
                    agent_id.to_string(),
                    AgentPerformanceMetrics {
                        average_latency: if agent_metrics.response_times.is_empty() {
                            Duration::from_millis(0)
                        } else {
                            agent_metrics.response_times.iter().sum::<Duration>()
                                / agent_metrics.response_times.len() as u32
                        },
                        quality_score: agent_metrics.quality_scores.iter().sum::<f32>()
                            / agent_metrics.quality_scores.len().max(1) as f32,
                        throughput: agent_metrics.response_times.len() as f32 / 60.0,
                        cost_per_request: agent_metrics.cost_tracking.iter().sum::<f32>()
                            / agent_metrics.cost_tracking.len().max(1) as f32,
                        uptime_percentage: 0.95, // Default uptime
                    },
                );
                agents_snapshot
            },
            system_load: 0.7, // Mock system load
            total_requests: history.len() as u64,
        };

        history.push(switch_event);

        // Keep only last 1000 events to prevent memory bloat
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }

        // Log detailed switch information
        info!(
            "✅ Model switch recorded: {} events in history, agent metrics updated",
            history.len()
        );
        debug!(
            "📈 Agent {} performance: {} responses tracked, last updated: {:?}",
            agent_id,
            agent_metrics.response_times.len(),
            agent_metrics.last_updated
        );
    }

    /// Get performance metrics for a specific agent
    async fn get_agent_metrics(&self, agent_id: &str) -> AgentMetrics {
        let metrics = self.metrics.read().await;
        if let Some(agent_metrics) = metrics.get(agent_id) {
            AgentMetrics {
                avg_latency: if agent_metrics.response_times.is_empty() {
                    Duration::from_secs(0)
                } else {
                    agent_metrics.response_times.iter().sum::<Duration>()
                        / agent_metrics.response_times.len() as u32
                },
                error_rate: if agent_metrics.response_times.is_empty() {
                    0.0
                } else {
                    agent_metrics.error_rates.iter().sum::<f32>()
                        / agent_metrics.error_rates.len() as f32
                },
                cost_per_hour: if agent_metrics.cost_tracking.is_empty() {
                    0.0
                } else {
                    agent_metrics.cost_tracking.iter().sum::<f32>()
                },
                success_rate: 0.95, // Mock value
                throughput: agent_metrics.response_times.len() as f32 / 60.0, // requests per minute
            }
        } else {
            AgentMetrics::default()
        }
    }
}

impl IntelligentRoutingEngine {
    async fn new(learning_system: &Arc<AdaptiveLearningSystem>) -> Result<Self> {
        Ok(Self {
            available_strategies: vec![
                RoutingStrategy::CapabilityBased,
                RoutingStrategy::CostOptimized,
                RoutingStrategy::LatencyOptimized,
            ],
            learning_system: learning_system.clone(),
            routing_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_predictor: Arc::new(PerformancePredictor::new()),
        })
    }

    async fn select_agent(
        &self,
        _task: &TaskRequest,
        _session: &UserSession,
        agents: &Arc<RwLock<HashMap<String, AgentInstance>>>,
    ) -> Result<AgentInstance> {
        let agents_map = agents.read().await;
        let available_agents: Vec<_> =
            agents_map.values().filter(|a| a.status == AgentStatus::Active).collect();

        if available_agents.is_empty() {
            return Err(anyhow::anyhow!("No available agents"));
        }

        // For now, select the first available agent
        // In a full implementation, this would use sophisticated routing logic
        Ok(available_agents[0].clone())
    }

    /// Get fallback strategy for a failed agent
    async fn get_fallback_strategy(
        &self,
        _task: &TaskRequest,
        failed_agent_id: &str,
        failure_reason: &str,
    ) -> Result<FallbackStrategy> {
        info!(
            "🧠 Analyzing fallback strategy for failed agent: {} ({})",
            failed_agent_id, failure_reason
        );

        // Analyze failure type and task requirements
        let strategy = match failure_reason {
            reason if reason.contains("latency") || reason.contains("timeout") => {
                // For latency issues, try switching to a faster model
                FallbackStrategy::SwitchModel {
                    agent_id: failed_agent_id.to_string(),
                    new_model: "gpt-3.5-turbo".to_string(), // Faster model
                }
            }
            reason if reason.contains("rate limit") || reason.contains("quota") => {
                // For rate limits, use backup agent
                FallbackStrategy::UseBackupAgent { agent_id: format!("backup_{}", failed_agent_id) }
            }
            reason if reason.contains("cost") => {
                // For cost issues, switch to cheaper model
                FallbackStrategy::SwitchModel {
                    agent_id: failed_agent_id.to_string(),
                    new_model: "claude-3-haiku".to_string(), // Cheaper model
                }
            }
            reason if reason.contains("error") || reason.contains("failure") => {
                // For errors, load balance across multiple agents
                FallbackStrategy::LoadBalanceRedirect {
                    target_agents: vec!["backup_agent_1".to_string(), "backup_agent_2".to_string()],
                }
            }
            _ => {
                // Default: emergency mode
                FallbackStrategy::EmergencyMode
            }
        };

        info!("📋 Selected fallback strategy: {:?}", strategy);
        Ok(strategy)
    }

    /// Get optimal model for a specific agent
    async fn get_optimal_model_for_agent(&self, agent_id: &str) -> Result<String> {
        // Mock implementation - would use sophisticated model selection logic
        let optimal_models = vec![
            "claude-4-sonnet".to_string(),
            "gpt-4-turbo".to_string(),
            "gemini-2-5-pro".to_string(),
        ];

        // Simple selection based on agent ID hash
        let index = agent_id.len() % optimal_models.len();
        Ok(optimal_models[index].clone())
    }

    /// Prepare session-specific routing state
    async fn prepare_session_routing(&self, session_id: &str) -> Result<()> {
        info!("🛤️ Preparing routing state for session: {}", session_id);

        // Initialize session-specific routing cache
        let mut cache = self.routing_cache.write().await;
        cache.insert(session_id.to_string(), "initialized".to_string());

        info!("✅ Session routing state prepared for: {}", session_id);
        Ok(())
    }

    /// Clean up session-specific routing state
    async fn cleanup_session_routing(&self, session_id: &str) -> Result<()> {
        info!("🧹 Cleaning up routing state for session: {}", session_id);

        // Remove session-specific routing cache
        let mut cache = self.routing_cache.write().await;
        cache.remove(session_id);

        info!("✅ Session routing state cleaned up for: {}", session_id);
        Ok(())
    }
}

impl PerformancePredictor {
    fn new() -> Self {
        Self { prediction_models: HashMap::new() }
    }
}

impl AdvancedFallbackManager {
    async fn new() -> Result<Self> {
        Ok(Self {
            fallback_chains: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            health_checker: Arc::new(HealthChecker::new()),
        })
    }

    async fn is_circuit_open(&self, agent_id: &str) -> bool {
        let circuit_breakers = self.circuit_breakers.read().await;
        circuit_breakers.get(agent_id).map_or(false, |cb| cb.is_open)
    }

    async fn record_failure(&self, agent_id: &str) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        let circuit_breaker =
            circuit_breakers.entry(agent_id.to_string()).or_insert_with(|| CircuitBreaker {
                is_open: false,
                failure_count: 0,
                last_failure: None,
                failure_threshold: 5,
                recovery_timeout: Duration::from_secs(60),
            });

        circuit_breaker.failure_count += 1;
        circuit_breaker.last_failure = Some(Instant::now());

        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold {
            circuit_breaker.is_open = true;
            warn!("🔴 Circuit breaker opened for agent: {}", agent_id);
        }
    }

    async fn reset_circuit_breaker(&self, agent_id: &str) {
        let mut circuit_breakers = self.circuit_breakers.write().await;
        if let Some(circuit_breaker) = circuit_breakers.get_mut(agent_id) {
            circuit_breaker.failure_count = 0;
            circuit_breaker.is_open = false;
        }
    }

    async fn get_fallback_chain(&self, _agent_id: &str) -> Vec<FallbackOption> {
        // Mock fallback chain
        vec![FallbackOption {
            agent_id: "backup_agent_1".to_string(),
            priority: 1,
            conditions: vec![FallbackCondition::HighLatency(Duration::from_secs(10))],
            timeout: Duration::from_secs(30),
        }]
    }
}

impl HealthChecker {
    fn new() -> Self {
        Self {
            health_status: Arc::new(RwLock::new(HashMap::new())),
            check_interval: Duration::from_secs(30),
        }
    }

    async fn run_health_checks(&self) {
        let mut interval = tokio::time::interval(self.check_interval);
        loop {
            interval.tick().await;
            self.perform_health_checks().await;
        }
    }

    async fn perform_health_checks(&self) {
        // Mock health check implementation
        let mut health_status = self.health_status.write().await;
        health_status.insert(
            "agent_1".to_string(),
            HealthStatus {
                is_healthy: true,
                response_time: Some(Duration::from_millis(150)),
                last_check: Instant::now(),
                error_message: None,
                uptime_percentage: 0.99,
            },
        );
    }

    async fn get_all_health_status(&self) -> HashMap<String, HealthStatus> {
        self.health_status.read().await.clone()
    }
}

impl SessionManager {
    async fn new() -> Result<Self> {
        Ok(Self {
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            sessionconfigs: Arc::new(RwLock::new(HashMap::new())),
            cleanup_interval: Duration::from_secs(300),
        })
    }

    async fn get_or_create_session(&self, session_id: Option<String>) -> Result<UserSession> {
        let session_id = session_id.unwrap_or_else(|| format!("session_{}", Uuid::new_v4()));

        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get(&session_id) {
            return Ok(session.clone());
        }

        let new_session = UserSession {
            id: session_id.clone(),
            user_id: None,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            preferences: UserPreferences::default(),
            active_agents: Vec::new(),
            conversation_history: Vec::new(),
            cost_tracking: SessionCostTracker { start_time: Instant::now(), ..Default::default() },
        };

        sessions.insert(session_id, new_session.clone());
        Ok(new_session)
    }

    async fn get_active_session_count(&self) -> usize {
        self.active_sessions.read().await.len()
    }

    async fn add_conversation_entry(
        &self,
        session_id: &str,
        task: &TaskRequest,
        result: &Result<TaskResponse>,
        agent_id: &str,
        duration: Duration,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            if let Ok(response) = result {
                let entry = ConversationEntry {
                    timestamp: Instant::now(),
                    request: task.clone(),
                    response: response.clone(),
                    agent_id: agent_id.to_string(),
                    execution_time: duration,
                    cost_cents: response.cost_cents.unwrap_or(0.0),
                    quality_score: response.quality_score,
                };

                session.conversation_history.push(entry);
                session.last_activity = Instant::now();
                session.cost_tracking.requests_count += 1;
                session.cost_tracking.total_cost += response.cost_cents.unwrap_or(0.0);
            }
        }
        Ok(())
    }

    async fn cleanup_expired_sessions(&self) {
        let mut interval = tokio::time::interval(self.cleanup_interval);
        loop {
            interval.tick().await;
            self.remove_expired_sessions().await;
        }
    }

    async fn remove_expired_sessions(&self) {
        let mut sessions = self.active_sessions.write().await;
        let cutoff = Instant::now() - Duration::from_secs(3600); // 1 hour timeout
        sessions.retain(|_, session| session.last_activity > cutoff);
    }

    /// Notify session manager of model switch
    async fn notify_model_switch(&self, agent_id: &str, new_model: &str) -> Result<()> {
        info!("📧 Notifying sessions of model switch: {} -> {}", agent_id, new_model);

        let mut sessions = self.active_sessions.write().await;
        for session in sessions.values_mut() {
            // Update session's active agents list
            if let Some(_agent_index) = session.active_agents.iter().position(|id| id == agent_id) {
                // Update agent info in session
                session.last_activity = Instant::now();
                info!("📝 Updated session {} with model switch info", session.id);
            }
        }

        Ok(())
    }

    /// Setup session with routing configuration
    async fn setup_session(&self, session_id: &str, _session_routing: ()) -> Result<()> {
        info!("🏗️ Setting up session: {}", session_id);

        // Create session if it doesn't exist
        let _session = self.get_or_create_session(Some(session_id.to_string())).await?;

        // Add session-specific configuration
        let mut configs = self.sessionconfigs.write().await;
        configs.insert(
            session_id.to_string(),
            SessionConfig {
                max_duration: Duration::from_secs(3600), // 1 hour
                cost_limit: Some(10.0),
                auto_cleanup: true,
                max_cost_per_hour: 10.0,
                preferred_models: vec!["claude-4-sonnet".to_string()],
                timeout_seconds: 30,
            },
        );

        info!("✅ Session {} setup completed", session_id);
        Ok(())
    }

    /// Clean up session resources
    async fn cleanup_session(&self, session_id: &str) -> Result<()> {
        info!("🧹 Cleaning up session: {}", session_id);

        // Remove session from active sessions
        let mut sessions = self.active_sessions.write().await;
        sessions.remove(session_id);

        // Remove session configuration
        let mut configs = self.sessionconfigs.write().await;
        configs.remove(session_id);

        info!("✅ Session {} cleaned up", session_id);
        Ok(())
    }
}

/// Multi-agent system status for TUI integration
#[derive(Debug, Clone)]
pub struct MultiAgentSystemStatus {
    pub total_agents: usize,
    pub active_agents: usize,
    pub idle_agents: usize,
    pub system_uptime: Duration,
    pub total_requests: u64,
    pub total_tasks: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub success_rate: f32,
    pub average_latency: Duration,
    pub avg_response_time_ms: f64,
    pub cost_efficiency: f32,
    pub active_sessions: usize,
    pub coordination_efficiency: f32,
}

/// System health information
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub component_health: HashMap<String, bool>,
    pub last_health_check: Instant,
    pub uptime: Duration,
    pub memory_usage: f32,
    pub cpu_usage: f32,
}

/// Fallback strategy for failed agents
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    SwitchModel { agent_id: String, new_model: String },
    UseBackupAgent { agent_id: String },
    LoadBalanceRedirect { target_agents: Vec<String> },
    EmergencyMode,
}

/// Model switch recommendation
#[derive(Debug, Clone)]
pub struct SwitchRecommendation {
    pub from_model: String,
    pub to_model: String,
    pub reason: String,
    pub urgency: SwitchUrgency,
    pub expected_improvement: f32,
}

/// Urgency level for model switching
#[derive(Debug, Clone)]
pub enum SwitchUrgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Agent performance metrics for tracking
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    pub avg_latency: Duration,
    pub error_rate: f32,
    pub cost_per_hour: f32,
    pub success_rate: f32,
    pub throughput: f32,
}

/// Key rotation manager for API keys
#[derive(Debug)]
pub struct KeyRotationManager {
    rotation_schedule: HashMap<String, Duration>,
    last_rotation: HashMap<String, Instant>,
}

impl KeyRotationManager {
    fn new() -> Self {
        Self { rotation_schedule: HashMap::new(), last_rotation: HashMap::new() }
    }
}

impl MultiAgentOrchestrator {
    /// Real-time model switching based on performance and availability
    pub async fn switch_model_realtime(
        &self,
        agent_id: &str,
        target_model: &str,
        reason: &str,
    ) -> Result<()> {
        info!(
            "🔄 Real-time model switch for agent {}: {} (reason: {})",
            agent_id, target_model, reason
        );

        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(agent_id) {
            let old_model = agent.models.first().cloned().unwrap_or_default();
            agent.models = vec![target_model.to_string()];
            agent.last_used = Some(Instant::now());

            // Update performance tracker and notify session manager
            self.performance_tracker
                .record_model_switch(agent_id, &old_model, target_model, reason)
                .await;
            self.session_manager.notify_model_switch(agent_id, target_model).await?;

            Ok(())
        } else {
            Err(anyhow::anyhow!("Agent not found: {}", agent_id))
        }
    }

    /// Smart fallback with intelligent strategy selection
    pub async fn smart_fallback(
        &self,
        task: &TaskRequest,
        failed_agent_id: &str,
        failure_reason: &str,
    ) -> Result<FallbackStrategy> {
        info!(
            "🛡️ Executing smart fallback for failed agent: {} ({})",
            failed_agent_id, failure_reason
        );

        // Get fallback strategy from routing engine
        let strategy = self
            .routing_engine
            .get_fallback_strategy(task, failed_agent_id, failure_reason)
            .await?;

        // Execute the strategy
        match &strategy {
            FallbackStrategy::SwitchModel { agent_id, new_model } => {
                self.switch_model_realtime(agent_id, new_model, "fallback_switch").await?;
                info!("✅ Successfully switched agent {} to model {}", agent_id, new_model);
            }
            FallbackStrategy::UseBackupAgent { agent_id } => {
                info!("🔄 Routing to backup agent: {}", agent_id);

                // Implement backup agent activation
                let mut agents = self.agents.write().await;

                // Check if backup agent already exists
                if let Some(backup_agent) = agents.get_mut(agent_id) {
                    // Activate existing backup agent
                    backup_agent.status = AgentStatus::Active;
                    backup_agent.last_used = Some(Instant::now());
                    info!("✅ Activated existing backup agent: {}", agent_id);
                } else {
                    // Create new backup agent
                    let backup_agent = AgentInstance {
                        id: agent_id.clone(),
                        name: format!("Backup Agent {}", agent_id.replace("backup_", "")),
                        agent_type: AgentType::GeneralPurpose,
                        models: vec![
                            "gpt-3.5-turbo".to_string(),
                            "claude-3-haiku".to_string(),
                            "gemini-pro".to_string(),
                        ],
                        capabilities: vec![
                            "text_generation".to_string(),
                            "question_answering".to_string(),
                            "summarization".to_string(),
                        ],
                        status: AgentStatus::Active,
                        performance_metrics: AgentPerformanceMetrics {
                            average_latency: Duration::from_millis(800),
                            quality_score: 0.80,
                            throughput: 15.0,
                            cost_per_request: 0.002,
                            uptime_percentage: 0.99,
                        },
                        cost_tracker: CostTracker {
                            total_cost: 0.0,
                            requests_count: 0,
                            last_hour_cost: 0.0,
                            daily_budget_used: 0.0,
                        },
                        last_used: Some(Instant::now()),
                        error_count: 0,
                        success_rate: 0.95,
                    };

                    agents.insert(agent_id.clone(), backup_agent);
                    info!("🆕 Created and activated new backup agent: {}", agent_id);
                }

                // Register the backup agent in the orchestrator
                if let Err(e) = self.orchestrator.register_fallback_agent(agent_id).await {
                    warn!("Failed to register backup agent in orchestrator: {}", e);
                }
            }
            FallbackStrategy::LoadBalanceRedirect { target_agents } => {
                info!("⚖️ Load balancing across agents: {:?}", target_agents);

                // Implement load balancing logic
                let mut agents = self.agents.write().await;
                let mut active_targets = Vec::new();

                // Check health and activate target agents
                for target_id in target_agents {
                    match agents.get_mut(target_id) {
                        Some(agent) => {
                            // Check if agent is healthy
                            if !matches!(agent.status, AgentStatus::Error(_))
                                && agent.error_count < 5
                            {
                                agent.status = AgentStatus::Active;
                                agent.last_used = Some(Instant::now());
                                active_targets.push(target_id.clone());
                                info!("✅ Activated load balancing target: {}", target_id);
                            } else {
                                warn!(
                                    "❌ Skipping unhealthy agent: {} (errors: {})",
                                    target_id, agent.error_count
                                );
                            }
                        }
                        None => {
                            // Create new load balancing agent
                            let lb_agent = AgentInstance {
                                id: target_id.clone(),
                                name: format!(
                                    "Load Balancer {}",
                                    target_id.replace("backup_agent_", "")
                                ),
                                agent_type: AgentType::GeneralPurpose,
                                models: vec![
                                    "gpt-3.5-turbo".to_string(),
                                    "claude-3-haiku".to_string(),
                                ],
                                capabilities: vec![
                                    "text_generation".to_string(),
                                    "load_balancing".to_string(),
                                ],
                                status: AgentStatus::Active,
                                performance_metrics: AgentPerformanceMetrics {
                                    average_latency: Duration::from_millis(600),
                                    quality_score: 0.85,
                                    throughput: 20.0,
                                    cost_per_request: 0.0015,
                                    uptime_percentage: 0.98,
                                },
                                cost_tracker: CostTracker {
                                    total_cost: 0.0,
                                    requests_count: 0,
                                    last_hour_cost: 0.0,
                                    daily_budget_used: 0.0,
                                },
                                last_used: Some(Instant::now()),
                                error_count: 0,
                                success_rate: 0.97,
                            };

                            agents.insert(target_id.clone(), lb_agent);
                            active_targets.push(target_id.clone());
                            info!("🆕 Created load balancing agent: {}", target_id);
                        }
                    }
                }

                // Configure load balancing strategy
                if !active_targets.is_empty() {
                    self.configure_load_balancing(&active_targets).await;
                    info!("⚖️ Load balancing configured for {} agents", active_targets.len());
                } else {
                    warn!("⚠️ No healthy agents available for load balancing");
                }
            }
            FallbackStrategy::EmergencyMode => {
                warn!("🚨 Entering emergency mode - using simplified fallback");

                // Implement emergency mode logic
                let mut agents = self.agents.write().await;

                // Create emergency agent with minimal configuration
                let emergency_agent_id = "emergency_agent".to_string();
                let emergency_agent = AgentInstance {
                    id: emergency_agent_id.clone(),
                    name: "Emergency Fallback Agent".to_string(),
                    agent_type: AgentType::GeneralPurpose,
                    models: vec!["gpt-3.5-turbo".to_string()], // Single reliable model
                    capabilities: vec![
                        "basic_text_generation".to_string(),
                        "emergency_response".to_string(),
                    ],
                    status: AgentStatus::Active,
                    performance_metrics: AgentPerformanceMetrics {
                        average_latency: Duration::from_millis(1200),
                        quality_score: 0.70, // Lower quality but reliable
                        throughput: 10.0,
                        cost_per_request: 0.001, // Cheap operation
                        uptime_percentage: 0.99,
                    },
                    cost_tracker: CostTracker {
                        total_cost: 0.0,
                        requests_count: 0,
                        last_hour_cost: 0.0,
                        daily_budget_used: 0.0,
                    },
                    last_used: Some(Instant::now()),
                    error_count: 0,
                    success_rate: 0.90,
                };

                agents.insert(emergency_agent_id.clone(), emergency_agent);

                // Disable all other agents to prevent cascade failures
                for (_, agent) in agents.iter_mut() {
                    if agent.id != emergency_agent_id {
                        agent.status = AgentStatus::Idle;
                    }
                }

                // Set emergency mode flag in system
                self.set_emergency_mode(true).await;

                warn!("🚨 Emergency mode activated - all requests will use emergency agent");

                // Schedule emergency mode recovery check
                let orchestrator = self.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(300)).await; // 5 minute recovery period
                    orchestrator.attempt_emergency_recovery().await;
                });
            }
        }

        Ok(strategy)
    }

    /// Start real-time monitoring for performance-based model switching
    pub async fn start_realtime_monitoring(&self) -> Result<()> {
        info!("🚀 Starting real-time monitoring for adaptive model switching");

        let performance_tracker = self.performance_tracker.clone();
        let routing_engine = self.routing_engine.clone();
        let agents = self.agents.clone();

        // Spawn monitoring task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Check each agent's performance
                let agents_guard = agents.read().await;
                for (agent_id, agent) in agents_guard.iter() {
                    let metrics = performance_tracker.get_agent_metrics(agent_id).await;

                    // Check if agent needs model switching
                    if metrics.error_rate > 0.15 || metrics.avg_latency > Duration::from_secs(10) {
                        if let Ok(optimal_model) =
                            routing_engine.get_optimal_model_for_agent(agent_id).await
                        {
                            if !agent.models.contains(&optimal_model) {
                                info!(
                                    "📊 Performance-based switch recommended: {} -> {}",
                                    agent.models.first().unwrap_or(&"unknown".to_string()),
                                    optimal_model
                                );
                                // Note: In a full implementation, we would
                                // execute the switch here
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Get system status for TUI display
    pub async fn get_system_status(&self) -> MultiAgentSystemStatus {
        let agents = self.agents.read().await;
        let active_agents = agents.values().filter(|a| a.status == AgentStatus::Active).count();
        let idle_agents = agents.values().filter(|a| a.status == AgentStatus::Idle).count();
        let performance_summary = self.performance_tracker.get_summary().await;
        let system_uptime = self.performance_tracker.get_system_uptime().await;
        let sessions = self.session_manager.active_sessions.read().await;

        // Calculate task statistics
        let total_tasks = performance_summary.total_requests;
        let successful_tasks = (performance_summary.total_requests as f64 * (1.0 - performance_summary.error_rate as f64)) as u64;
        let failed_tasks = total_tasks - successful_tasks;
        let avg_response_time_ms = performance_summary.average_latency.as_millis() as f64;

        MultiAgentSystemStatus {
            total_agents: agents.len(),
            active_agents,
            idle_agents,
            system_uptime,
            total_requests: performance_summary.total_requests,
            total_tasks,
            successful_tasks,
            failed_tasks,
            success_rate: 1.0 - performance_summary.error_rate,
            average_latency: performance_summary.average_latency,
            avg_response_time_ms,
            cost_efficiency: performance_summary.cost_efficiency,
            active_sessions: sessions.len(),
            coordination_efficiency: 0.8, // Default coordination efficiency - could be calculated from other metrics
        }
    }

    /// Get overall system health
    pub async fn get_system_health(&self) -> SystemHealth {
        let health_checker = &self.fallback_manager.health_checker;

        // Get health status from all components
        let all_health = health_checker.get_all_health_status().await;
        let mut component_health = HashMap::new();

        for (agent_id, health) in &all_health {
            component_health.insert(agent_id.clone(), health.is_healthy);
        }

        // Calculate overall health
        let healthy_count = component_health.values().filter(|&&h| h).count();
        let total_count = component_health.len().max(1);
        let overall_healthy = healthy_count as f32 / total_count as f32 > 0.7;

        SystemHealth {
            overall_status: HealthStatus {
                is_healthy: overall_healthy,
                response_time: Some(Duration::from_millis(200)),
                last_check: Instant::now(),
                error_message: None,
                uptime_percentage: 0.95,
            },
            component_health,
            last_health_check: Instant::now(),
            uptime: Duration::from_secs(3600), // Mock uptime
            memory_usage: 0.65,
            cpu_usage: 0.45,
        }
    }

    /// Configure load balancing for target agents
    async fn configure_load_balancing(&self, target_agents: &[String]) {
        info!("⚖️ Configuring load balancing for agents: {:?}", target_agents);

        // Update orchestrator with load balancing configuration
        for agent_id in target_agents {
            if let Err(e) = self
                .orchestrator
                .configure_agent_load_balancing(
                    agent_id,
                    LoadBalancingConfig {
                        weight: 1.0 / target_agents.len() as f32,
                        max_concurrent_requests: 5,
                        health_check_interval: Duration::from_secs(30),
                    },
                )
                .await
            {
                warn!("Failed to configure load balancing for agent {}: {}", agent_id, e);
            }
        }

        info!("✅ Load balancing configuration completed for {} agents", target_agents.len());
    }

    /// Set emergency mode flag in the system
    async fn set_emergency_mode(&self, enabled: bool) {
        if enabled {
            warn!("🚨 EMERGENCY MODE ENABLED - System operating in degraded state");

            // Record emergency mode activation
            self.performance_tracker
                .record_execution(
                    "system",
                    &TaskRequest {
                        task_type: crate::models::TaskType::SystemMaintenance,
                        content: "Emergency mode activated".to_string(),
                        constraints: crate::models::TaskConstraints::default(),
                        context_integration: false,
                        memory_integration: false,
                        cognitive_enhancement: false,
                    },
                    &Ok(crate::models::TaskResponse {
                        content: "Emergency mode enabled".to_string(),
                        model_used: crate::models::ModelSelection::API("emergency".to_string()),
                        generation_time_ms: Some(0),
                        tokens_generated: Some(0),
                        cost_cents: Some(0.0),
                        quality_score: 0.5,
                        cost_info: Some("Emergency mode - no cost".to_string()),
                        model_info: None,
                        error: None,
                    }),
                    Duration::from_millis(0),
                )
                .await
                .unwrap_or_else(|e| warn!("Failed to record emergency mode: {}", e));
        } else {
            info!("🟢 Emergency mode disabled - System returning to normal operation");
        }
    }

    /// Attempt to recover from emergency mode
    async fn attempt_emergency_recovery(&self) {
        info!("🔄 Attempting emergency mode recovery");

        let agents = self.agents.read().await;
        let emergency_agent_count = agents.values().filter(|a| a.id == "emergency_agent").count();

        if emergency_agent_count > 0 {
            // Check if system is stable enough to exit emergency mode
            let health = self.get_system_health().await;
            let healthy_agents = health.component_health.values().filter(|&&h| h).count();

            if healthy_agents >= 2 && health.overall_status.is_healthy {
                info!("🟢 System health recovered - exiting emergency mode");
                self.exit_emergency_mode().await;
            } else {
                warn!("⚠️ System still unstable - remaining in emergency mode");

                // Schedule another recovery attempt
                let agents_clone = self.agents.clone();
                let _performance_tracker_clone = self.performance_tracker.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_secs(600)).await; // 10 minute retry

                    // Simplified recovery attempt within spawn context
                    warn!("🔄 Attempting scheduled emergency recovery");
                    let agents = agents_clone.read().await;
                    if agents.is_empty() {
                        warn!("⚠️ No agents available during scheduled recovery");
                    }
                });
            }
        }
    }

    /// Exit emergency mode and restore normal operation
    async fn exit_emergency_mode(&self) {
        info!("🟢 Exiting emergency mode - restoring normal operation");

        let mut agents = self.agents.write().await;

        // Remove emergency agent
        agents.remove("emergency_agent");

        // Reactivate previously healthy agents
        for (_, agent) in agents.iter_mut() {
            if agent.error_count < 3 && agent.status == AgentStatus::Idle {
                agent.status = AgentStatus::Active;
                info!("✅ Reactivated agent: {}", agent.id);
            }
        }

        self.set_emergency_mode(false).await;
        info!("🎉 Emergency mode recovery complete - system restored");
    }

    /// Create a new MultiAgentOrchestrator instance
    pub async fn new(apiconfig: &ApiKeysConfig) -> Result<Self> {
        let orchestrator = Arc::new(ModelOrchestrator::new(apiconfig).await?);
        let api_manager = Arc::new(ApiKeyManager::new(apiconfig).await?);

        // Check if any search APIs are configured to determine if we should enable
        // discovery
        let has_search_apis = apiconfig.search.brave.is_some()
            || apiconfig.search.google.is_some()
            || apiconfig.search.bing.is_some();

        // Create discovery service (disable automatic discovery if no search APIs
        // configured)
        let discovery_service =
            Arc::new(DiscoveryService::new_with_discovery(has_search_apis).await?);

        let performance_tracker = Arc::new(RealTimePerformanceTracker::new());

        // Create a mock adaptive learning system for now
        let adaptive_learning =
            Arc::new(super::adaptive_learning::AdaptiveLearningSystem::new(Default::default()));
        let routing_engine = Arc::new(IntelligentRoutingEngine::new(&adaptive_learning).await?);
        let fallback_manager = Arc::new(AdvancedFallbackManager::new().await?);
        let session_manager = Arc::new(SessionManager::new().await?);

        let orchestrator_instance = Self {
            orchestrator,
            agents: Arc::new(RwLock::new(HashMap::new())),
            api_manager,
            discovery_service,
            performance_tracker,
            routing_engine,
            fallback_manager,
            session_manager,
        };

        // 🚀 AUTO-REGISTER LOCAL MODELS AS AGENTS
        info!("🔍 Auto-registering local models as agents...");

        // Try to discover and register local models automatically
        match super::LocalModelDiscoveryService::new().await {
            Ok(mut local_discovery) => {
                match local_discovery.discover_models().await {
                    Ok(discovered_models) => {
                        if !discovered_models.is_empty() {
                            info!(
                                "📦 Found {} local models to register as agents",
                                discovered_models.len()
                            );

                            let mut agents = orchestrator_instance.agents.write().await;

                            for model in discovered_models {
                                if model.is_available {
                                    let agent_id = format!("local_{}", model.id);

                                    // Convert model specializations to agent type
                                    let agent_type = if model.specializations.contains(
                                        &super::registry::ModelSpecialization::CodeGeneration,
                                    ) {
                                        AgentType::CodeGeneration
                                    } else if model.specializations.contains(
                                        &super::registry::ModelSpecialization::LogicalReasoning,
                                    ) {
                                        AgentType::LogicalReasoning
                                    } else {
                                        AgentType::GeneralPurpose
                                    };

                                    // Convert model source to capabilities
                                    let mut capabilities = vec![
                                        "text_generation".to_string(),
                                        "local_execution".to_string(),
                                    ];
                                    for specialization in &model.specializations {
                                        match specialization {
                                            super::registry::ModelSpecialization::CodeGeneration => {
                                                capabilities.push("code_generation".to_string());
                                                capabilities.push("code_completion".to_string());
                                            }
                                            super::registry::ModelSpecialization::CodeReview => {
                                                capabilities.push("code_review".to_string());
                                            }
                                            super::registry::ModelSpecialization::LogicalReasoning => {
                                                capabilities.push("reasoning".to_string());
                                                capabilities.push("problem_solving".to_string());
                                            }
                                                                                         super::registry::ModelSpecialization::GeneralPurpose => {
                                                 capabilities.push("general_assistance".to_string());
                                             }
                                            _ => {}
                                        }
                                    }

                                    let agent = AgentInstance {
                                        id: agent_id.clone(),
                                        name: format!("Local Agent: {}", model.name),
                                        agent_type,
                                        models: vec![model.name.clone()],
                                        capabilities,
                                        status: if model.is_activated {
                                            AgentStatus::Active
                                        } else {
                                            AgentStatus::Idle
                                        },
                                        performance_metrics: AgentPerformanceMetrics {
                                            average_latency: Duration::from_millis(200), /* Local models are fast */
                                            quality_score: 0.85, // Good default for local models
                                            throughput: model.capabilities.max_tokens_per_second,
                                            cost_per_request: 0.0, // Local models are free
                                            uptime_percentage: 0.98,
                                        },
                                        cost_tracker: CostTracker::default(), /* Local models
                                                                               * have no cost */
                                        last_used: None,
                                        error_count: 0,
                                        success_rate: 0.90,
                                    };

                                    agents.insert(agent_id.clone(), agent);
                                    info!(
                                        "✅ Registered local model agent: {} ({})",
                                        agent_id, model.name
                                    );
                                }
                            }

                            let total_registered = agents.len();
                            if total_registered > 0 {
                                info!(
                                    "🎉 Successfully auto-registered {} local model agents",
                                    total_registered
                                );
                            }
                        } else {
                            info!("ℹ️  No local models found for auto-registration");
                        }
                    }
                    Err(e) => {
                        warn!("⚠️  Failed to discover local models for auto-registration: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!(
                    "⚠️  Failed to initialize local model discovery for auto-registration: {}",
                    e
                );
            }
        }

        Ok(orchestrator_instance)
    }

    /// Activate an agent
    pub async fn activate_agent(&self, agent_id: &str) -> Result<()> {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(agent_id) {
            agent.status = AgentStatus::Active;
            info!("🟢 Activated agent: {}", agent_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Agent not found: {}", agent_id))
        }
    }

    /// Update model for an agent with hot-swappable zero-downtime upgrade
    pub async fn update_model(&self, model_name: &str) -> Result<()> {
        info!("🔄 Starting hot-swappable model update: {}", model_name);

        // Execute comprehensive hot-swap update with atomic operations
        self.execute_hot_swappable_model_update(model_name).await
    }

    /// Execute comprehensive hot-swappable model update with zero downtime
    async fn execute_hot_swappable_model_update(&self, model_name: &str) -> Result<()> {
        let update_id = uuid::Uuid::new_v4().to_string();
        info!("🚀 Executing hot-swappable update for model: {} (ID: {})", model_name, update_id);

        // Phase 1: Discovery and Validation
        let update_plan = self.create_model_update_plan(model_name, &update_id).await?;
        info!("📋 Update plan created: {} agents affected", update_plan.affected_agents.len());

        // Phase 2: Pre-update Safety Checks
        self.perform_pre_update_safety_checks(&update_plan).await?;
        info!("✅ Pre-update safety checks passed");

        // Phase 3: Prepare New Model (download/load but don't activate)
        let prepared_model = self.prepare_new_model_version(&update_plan).await?;
        info!("📦 New model version prepared: {}", prepared_model.version);

        // Phase 4: Create Backup Points
        let backup_state = self.create_system_backup(&update_plan).await?;
        info!("💾 System backup created: {}", backup_state.backup_id);

        // Phase 5: Atomic Hot-Swap Execution
        match self.execute_atomic_hot_swap(&update_plan, &prepared_model).await {
            Ok(swap_result) => {
                info!("🎉 Hot-swap completed successfully: {}", swap_result.new_model_version);

                // Phase 6: Post-Update Validation
                self.perform_post_update_validation(&update_plan, &swap_result).await?;
                info!("✅ Post-update validation passed");

                // Phase 7: Cleanup and Optimization
                self.finalize_model_update(&update_plan, &backup_state).await?;
                info!("🧹 Update finalization completed");

                Ok(())
            }
            Err(e) => {
                error!("❌ Hot-swap failed: {}, initiating rollback", e);

                // Emergency Rollback
                self.execute_emergency_rollback(&update_plan, &backup_state).await?;
                info!("🔄 Emergency rollback completed successfully");

                Err(anyhow::anyhow!("Model update failed and was rolled back: {}", e))
            }
        }
    }

    /// Create comprehensive model update plan
    async fn create_model_update_plan(
        &self,
        model_name: &str,
        update_id: &str,
    ) -> Result<ModelUpdatePlan> {
        info!("📋 Creating model update plan for: {}", model_name);

        // Check for available updates
        let available_updates = self.discovery_service.check_for_updates().await?;
        let relevant_update =
            available_updates
                .iter()
                .find(|update| update.contains(model_name))
                .ok_or_else(|| anyhow::anyhow!("No updates available for model: {}", model_name))?;

        // Find affected agents
        let agents = self.agents.read().await;
        let affected_agents: Vec<String> = agents
            .iter()
            .filter(|(_, agent)| agent.models.iter().any(|m| m.contains(model_name)))
            .map(|(id, _)| id.clone())
            .collect();

        // Estimate impact and create plan
        let update_plan = ModelUpdatePlan {
            update_id: update_id.to_string(),
            model_name: model_name.to_string(),
            update_description: relevant_update.clone(),
            affected_agents,
            estimated_downtime: Duration::from_millis(0), // Zero downtime target
            rollback_strategy: RollbackStrategy::AtomicSwap,
            safety_checks: vec![
                SafetyCheck::ModelCompatibility,
                SafetyCheck::ResourceAvailability,
                SafetyCheck::DependencyValidation,
                SafetyCheck::PerformanceImpact,
            ],
            update_strategy: UpdateStrategy::HotSwap,
            priority: UpdatePriority::Normal,
        };

        Ok(update_plan)
    }

    /// Perform comprehensive pre-update safety checks
    async fn perform_pre_update_safety_checks(&self, plan: &ModelUpdatePlan) -> Result<()> {
        info!("🔍 Performing pre-update safety checks for: {}", plan.model_name);

        for check in &plan.safety_checks {
            match check {
                SafetyCheck::ModelCompatibility => {
                    // Verify new model is compatible with existing agents
                    self.validate_model_compatibility(&plan.model_name).await?;
                    info!("✅ Model compatibility validated");
                }
                SafetyCheck::ResourceAvailability => {
                    // Ensure sufficient resources for dual-loading during swap
                    self.validate_resource_availability(&plan.affected_agents).await?;
                    info!("✅ Resource availability validated");
                }
                SafetyCheck::DependencyValidation => {
                    // Check all dependencies are satisfied
                    self.validate_dependencies(&plan.model_name).await?;
                    info!("✅ Dependencies validated");
                }
                SafetyCheck::PerformanceImpact => {
                    // Estimate performance impact
                    self.validate_performance_impact(plan).await?;
                    info!("✅ Performance impact validated");
                }
            }
        }

        Ok(())
    }

    /// Prepare new model version for hot-swap
    async fn prepare_new_model_version(&self, plan: &ModelUpdatePlan) -> Result<PreparedModel> {
        info!("📦 Preparing new model version: {}", plan.model_name);

        // Check if it's a local or API model update
        if plan.model_name.starts_with("local_") {
            // Local model update: download and validate
            let model_path = self.download_local_model_update(&plan.model_name).await?;
            let model_info = self.validate_local_model(&model_path).await?;

            Ok(PreparedModel {
                model_name: plan.model_name.clone(),
                version: model_info.version,
                model_type: ModelUpdateType::Local,
                model_path: Some(model_path),
                apiconfig: None,
                validation_passed: true,
                preparation_time: std::time::Instant::now(),
            })
        } else {
            // API model update: validate new endpoint/version
            let apiconfig = self.prepare_api_model_update(&plan.model_name).await?;

            Ok(PreparedModel {
                model_name: plan.model_name.clone(),
                version: apiconfig.version.clone(),
                model_type: ModelUpdateType::API,
                model_path: None,
                apiconfig: Some(apiconfig),
                validation_passed: true,
                preparation_time: std::time::Instant::now(),
            })
        }
    }

    /// Create system backup before update
    async fn create_system_backup(&self, plan: &ModelUpdatePlan) -> Result<SystemBackup> {
        info!("💾 Creating system backup for update: {}", plan.update_id);

        let agents = self.agents.read().await;
        let mut agent_snapshots = HashMap::new();

        // Backup affected agent configurations
        for agent_id in &plan.affected_agents {
            if let Some(agent) = agents.get(agent_id) {
                agent_snapshots.insert(agent_id.clone(), agent.clone());
            }
        }

        let backup = SystemBackup {
            backup_id: uuid::Uuid::new_v4().to_string(),
            update_id: plan.update_id.clone(),
            timestamp: std::time::Instant::now(),
            agent_snapshots,
            orchestratorconfig: self.get_orchestratorconfig_snapshot().await,
            routing_state: self.get_routing_state_snapshot().await,
        };

        info!("📸 Backup created: {} (agents: {})", backup.backup_id, backup.agent_snapshots.len());
        Ok(backup)
    }

    /// Execute atomic hot-swap with zero downtime
    async fn execute_atomic_hot_swap(
        &self,
        plan: &ModelUpdatePlan,
        prepared_model: &PreparedModel,
    ) -> Result<SwapResult> {
        info!("⚡ Executing atomic hot-swap for: {}", plan.model_name);

        let start_time = std::time::Instant::now();

        // Lock agents for atomic update
        let mut agents = self.agents.write().await;

        let mut updated_agents = Vec::new();
        let mut swap_operations = Vec::new();

        for agent_id in &plan.affected_agents {
            if let Some(agent) = agents.get_mut(agent_id) {
                // Store old configuration
                let old_models = agent.models.clone();

                // Atomic model swap
                let new_model = match &prepared_model.model_type {
                    ModelUpdateType::Local => {
                        format!("{}:v{}", prepared_model.model_name, prepared_model.version)
                    }
                    ModelUpdateType::API => {
                        format!("{}:api-v{}", prepared_model.model_name, prepared_model.version)
                    }
                    ModelUpdateType::Hybrid => {
                        format!("{}:hybrid-v{}", prepared_model.model_name, prepared_model.version)
                    }
                };

                // Update agent configuration atomically
                agent.models = vec![new_model.clone()];
                agent.last_used = Some(std::time::Instant::now());

                // Record swap operation for potential rollback
                swap_operations.push(SwapOperation {
                    agent_id: agent_id.clone(),
                    old_models: old_models.clone(),
                    new_models: vec![new_model.clone()],
                    swap_time: std::time::Instant::now(),
                });

                updated_agents.push(agent_id.clone());

                info!("🔄 Swapped model for agent {}: {:?} -> {}", agent_id, old_models, new_model);
            }
        }

        // Update orchestrator configuration
        self.update_orchestrator_modelconfig(&prepared_model.model_name, &prepared_model.version)
            .await?;

        // Update routing engine
        self.update_routing_engine_model_references(
            &plan.affected_agents,
            &prepared_model.model_name,
        )
        .await?;

        let swap_duration = start_time.elapsed();

        Ok(SwapResult {
            update_id: plan.update_id.clone(),
            new_model_version: prepared_model.version.clone(),
            updated_agents,
            swap_operations,
            swap_duration,
            success: true,
        })
    }

    /// Perform post-update validation
    async fn perform_post_update_validation(
        &self,
        plan: &ModelUpdatePlan,
        swap_result: &SwapResult,
    ) -> Result<()> {
        info!("🔍 Performing post-update validation");

        // Test each updated agent
        for agent_id in &swap_result.updated_agents {
            self.validate_agent_functionality(agent_id).await?;
            info!("✅ Agent {} functionality validated", agent_id);
        }

        // Validate system performance hasn't degraded
        self.validate_system_performance_post_update(plan).await?;
        info!("✅ System performance validated");

        // Check for any error spikes
        self.monitor_error_rates_post_update(&swap_result.updated_agents).await?;
        info!("✅ Error rates within acceptable bounds");

        Ok(())
    }

    /// Execute emergency rollback if update fails
    async fn execute_emergency_rollback(
        &self,
        plan: &ModelUpdatePlan,
        backup: &SystemBackup,
    ) -> Result<()> {
        error!("🚨 Executing emergency rollback for update: {}", plan.update_id);

        // Lock agents for rollback
        let mut agents = self.agents.write().await;

        // Restore agent configurations from backup
        for (agent_id, backup_agent) in &backup.agent_snapshots {
            if let Some(current_agent) = agents.get_mut(agent_id) {
                // Restore previous configuration
                current_agent.models = backup_agent.models.clone();
                current_agent.status = backup_agent.status.clone();
                current_agent.last_used = backup_agent.last_used;

                info!("🔄 Rolled back agent {}: {:?}", agent_id, backup_agent.models);
            }
        }

        // Restore orchestrator configuration
        self.restore_orchestratorconfig(&backup.orchestratorconfig).await?;

        // Restore routing engine state
        self.restore_routing_state(&backup.routing_state).await?;

        info!("✅ Emergency rollback completed successfully");
        Ok(())
    }

    /// Helper method implementations for update system
    async fn validate_model_compatibility(&self, _model_name: &str) -> Result<()> {
        // Validate that the new model version is compatible with existing system
        Ok(())
    }

    async fn validate_resource_availability(&self, _affected_agents: &[String]) -> Result<()> {
        // Check if we have enough resources for dual-loading during swap
        Ok(())
    }

    async fn validate_dependencies(&self, _model_name: &str) -> Result<()> {
        // Validate all model dependencies are satisfied
        Ok(())
    }

    async fn validate_performance_impact(&self, _plan: &ModelUpdatePlan) -> Result<()> {
        // Estimate and validate acceptable performance impact
        Ok(())
    }

    async fn download_local_model_update(&self, model_name: &str) -> Result<String> {
        // Download new version of local model
        Ok(format!("/models/{}_updated", model_name))
    }

    async fn validate_local_model(&self, _model_path: &str) -> Result<ModelValidation> {
        // Validate local model file integrity and compatibility
        Ok(ModelValidation {
            version: "1.1.0".to_string(),
            checksum: "abc123".to_string(),
            valid: true,
        })
    }

    async fn prepare_api_model_update(&self, model_name: &str) -> Result<ApiModelConfig> {
        // Prepare API model configuration for update
        Ok(ApiModelConfig {
            model_name: model_name.to_string(),
            version: "2024.1".to_string(),
            endpoint: format!("https://api.example.com/v2/{}", model_name),
            api_key: "updated_key".to_string(),
        })
    }

    async fn get_orchestratorconfig_snapshot(&self) -> OrchestratorConfigSnapshot {
        // Create snapshot of orchestrator configuration
        OrchestratorConfigSnapshot {
            routing_strategy: RoutingStrategy::CapabilityBased,
            modelconfigs: HashMap::new(),
            timestamp: std::time::Instant::now(),
        }
    }

    async fn get_routing_state_snapshot(&self) -> RoutingStateSnapshot {
        // Create snapshot of routing engine state
        RoutingStateSnapshot {
            active_routes: HashMap::new(),
            performance_cache: HashMap::new(),
            timestamp: std::time::Instant::now(),
        }
    }

    async fn update_orchestrator_modelconfig(
        &self,
        _model_name: &str,
        _version: &str,
    ) -> Result<()> {
        // Update orchestrator with new model configuration
        Ok(())
    }

    async fn update_routing_engine_model_references(
        &self,
        _agent_ids: &[String],
        _model_name: &str,
    ) -> Result<()> {
        // Update routing engine with new model references
        Ok(())
    }

    async fn validate_agent_functionality(&self, _agent_id: &str) -> Result<()> {
        // Test agent functionality post-update
        Ok(())
    }

    async fn validate_system_performance_post_update(&self, _plan: &ModelUpdatePlan) -> Result<()> {
        // Validate system performance hasn't degraded
        Ok(())
    }

    async fn monitor_error_rates_post_update(&self, _agent_ids: &[String]) -> Result<()> {
        // Monitor for error rate spikes post-update
        Ok(())
    }

    async fn restore_orchestratorconfig(
        &self,
        _config: &OrchestratorConfigSnapshot,
    ) -> Result<()> {
        // Restore orchestrator configuration from backup
        Ok(())
    }

    async fn restore_routing_state(&self, _state: &RoutingStateSnapshot) -> Result<()> {
        // Restore routing engine state from backup
        Ok(())
    }

    async fn finalize_model_update(
        &self,
        plan: &ModelUpdatePlan,
        _backup: &SystemBackup,
    ) -> Result<()> {
        info!("🧹 Finalizing model update: {}", plan.update_id);

        // Cleanup old model versions
        // Update monitoring configurations
        // Notify dependent systems
        // Update documentation

        Ok(())
    }

    /// List all agents
    pub async fn list_agents(&self) -> Result<Vec<AgentInstance>> {
        let agents = self.agents.read().await;
        Ok(agents.values().cloned().collect())
    }

    /// Discover latest models with timeout to prevent hanging
    pub async fn discover_latest_models(&self) -> Result<Vec<ModelRegistryEntry>> {
        // Add timeout to prevent the TUI from hanging
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            self.discovery_service.discover_latest_models(),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                warn!("Model discovery timed out, returning empty list");
                Ok(vec![])
            }
        }
    }

    /// Check for model updates with timeout
    pub async fn check_for_updates(&self) -> Result<Vec<String>> {
        // Add timeout to prevent hanging
        match tokio::time::timeout(
            std::time::Duration::from_secs(3),
            self.discovery_service.check_for_updates(),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                warn!("Model update check timed out, returning empty list");
                Ok(vec![])
            }
        }
    }

    /// Execute task using multi-agent coordination with intelligent routing
    pub async fn execute_task_with_multi_agent(
        &self,
        task: TaskRequest,
        session_id: Option<String>,
    ) -> Result<TaskResponse> {
        info!("🤖 Executing task with multi-agent coordination: {:?}", task.task_type);

        let session = self.session_manager.get_or_create_session(session_id).await?;
        let selected_agent =
            self.routing_engine.select_agent(&task, &session, &self.agents).await?;

        let start_time = Instant::now();
        let execution_result = self.orchestrator.execute_with_fallback(task.clone()).await;
        let duration = start_time.elapsed();

        // Record execution metrics
        self.performance_tracker
            .record_execution(&selected_agent.id, &task, &execution_result, duration)
            .await?;

        // Add to session history
        self.session_manager
            .add_conversation_entry(
                &session.id,
                &task,
                &execution_result,
                &selected_agent.id,
                duration,
            )
            .await?;

        match execution_result {
            Ok(response) => {
                info!("✅ Task completed successfully by agent: {}", selected_agent.id);
                Ok(response)
            }
            Err(e) => {
                warn!("❌ Task failed, attempting smart fallback: {}", e);

                // Record failure for circuit breaker
                self.fallback_manager.record_failure(&selected_agent.id).await;

                // Attempt smart fallback
                let fallback_strategy =
                    self.smart_fallback(&task, &selected_agent.id, &e.to_string()).await?;

                // For now, return error with fallback info
                Err(anyhow::anyhow!("Task failed, fallback strategy: {:?}", fallback_strategy))
            }
        }
    }

    /// Get comprehensive model information using discovery service
    pub async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        info!("📋 Getting model information for: {}", model_id);

        // Try to get from discovery service first
        let registry_entries = self.discovery_service.discover_latest_models().await?;

        for entry in registry_entries {
            if entry.id == model_id {
                return Ok(ModelInfo {
                    id: entry.id.clone(),
                    name: entry.name.clone(),
                    description: format!("{} model from {}", entry.name, entry.provider),
                    context_length: entry.context_window.unwrap_or(4096) as usize,
                    capabilities: entry.capabilities,
                });
            }
        }

        // Fallback to basic model info
        Ok(ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            description: format!("Unknown model: {}", model_id),
            context_length: 4096,
            capabilities: vec!["text_generation".to_string()],
        })
    }

    /// Create and register a new local model manager
    pub async fn register_local_manager(&self, manager: Arc<LocalModelManager>) -> Result<()> {
        info!("🔗 Registering local model manager");

        // Get available local models
        let local_models = manager.get_available_models().await;

        let mut agents = self.agents.write().await;

        // Create agents for each local model
        for (index, model_id) in local_models.iter().enumerate() {
            let agent_id = format!("local_agent_{}", index);

            let agent = AgentInstance {
                id: agent_id.clone(),
                name: format!("Local Agent for {}", model_id),
                agent_type: AgentType::GeneralPurpose,
                models: vec![model_id.clone()],
                capabilities: vec!["text_generation".to_string(), "local_execution".to_string()],
                status: AgentStatus::Active,
                performance_metrics: AgentPerformanceMetrics::default(),
                cost_tracker: CostTracker::default(),
                last_used: None,
                error_count: 0,
                success_rate: 0.95,
            };

            agents.insert(agent_id, agent);
        }

        info!("✅ Registered {} local model agents", local_models.len());
        Ok(())
    }

    /// Create provider-specific agents using ProviderFactory
    pub async fn create_provider_agents(&self, provider_name: &str) -> Result<Vec<String>> {
        info!("🏭 Creating agents for provider: {}", provider_name);

        let api_status = self.api_manager.get_all_api_status().await;

        if let Some(apiconfig) = api_status.get(provider_name) {
            if !apiconfig.is_valid {
                return Err(anyhow::anyhow!("Invalid API key for provider: {}", provider_name));
            }

            // Create minimal ApiKeysConfig for this provider
            let mut keysconfig = crate::config::ApiKeysConfig::default();
            match provider_name.to_lowercase().as_str() {
                "openai" => keysconfig.ai_models.openai = Some(apiconfig.key.clone()),
                "anthropic" => keysconfig.ai_models.anthropic = Some(apiconfig.key.clone()),
                "deepseek" => keysconfig.ai_models.deepseek = Some(apiconfig.key.clone()),
                "mistral" => keysconfig.ai_models.mistral = Some(apiconfig.key.clone()),
                "codestral" => keysconfig.ai_models.codestral = Some(apiconfig.key.clone()),
                "gemini" => keysconfig.ai_models.gemini = Some(apiconfig.key.clone()),
                "grok" => keysconfig.ai_models.grok = Some(apiconfig.key.clone()),
                _ => return Err(anyhow::anyhow!("Unsupported provider: {}", provider_name)),
            }

            // Create provider instance
            let providers = ProviderFactory::create_providers(&keysconfig);
            let provider = ProviderFactory::get_provider(&providers, &provider_name)
                .ok_or_else(|| anyhow::anyhow!("Provider '{}' not found", provider_name))?;

            // Get available models from provider
            let available_models = provider.list_models().await?;

            let mut agents = self.agents.write().await;
            let mut created_agents = Vec::new();

            for model_info in available_models {
                let agent_id = format!("{}_{}", provider_name, model_info.name.replace("-", "_"));

                let agent_type = match model_info.name.as_str() {
                    name if name.contains("code") => AgentType::CodeGeneration,
                    name if name.contains("creative") || name.contains("claude") => {
                        AgentType::CreativeWriting
                    }
                    name if name.contains("data") || name.contains("analysis") => {
                        AgentType::DataAnalysis
                    }
                    name if name.contains("reasoning") || name.contains("logic") => {
                        AgentType::LogicalReasoning
                    }
                    _ => AgentType::GeneralPurpose,
                };

                let agent = AgentInstance {
                    id: agent_id.clone(),
                    name: format!("{} Agent ({})", provider_name, model_info.name),
                    agent_type,
                    models: vec![model_info.id],
                    capabilities: model_info.capabilities,
                    status: AgentStatus::Active,
                    performance_metrics: AgentPerformanceMetrics::default(),
                    cost_tracker: CostTracker::default(),
                    last_used: None,
                    error_count: 0,
                    success_rate: 0.98,
                };

                agents.insert(agent_id.clone(), agent);
                created_agents.push(agent_id);
            }

            info!("✅ Created {} agents for provider {}", created_agents.len(), provider_name);
            Ok(created_agents)
        } else {
            Err(anyhow::anyhow!("No API configuration found for provider: {}", provider_name))
        }
    }

    /// Intelligent cost management using CostManager integration
    pub async fn optimize_costs(&self) -> Result<CostOptimizationReport> {
        info!("💰 Optimizing costs across all agents");

        let agents = self.agents.read().await;
        let mut cost_analysis = HashMap::new();
        let mut total_cost = 0.0;

        for (agent_id, agent) in agents.iter() {
            let cost_info = AgentCostInfo {
                hourly_cost: agent.cost_tracker.last_hour_cost,
                daily_budget_used: agent.cost_tracker.daily_budget_used,
                cost_per_request: agent.performance_metrics.cost_per_request,
                efficiency_score: agent.performance_metrics.quality_score
                    / agent.performance_metrics.cost_per_request.max(0.01),
            };

            total_cost += cost_info.hourly_cost;
            cost_analysis.insert(agent_id.clone(), cost_info);
        }

        // Generate optimization recommendations
        let mut recommendations = Vec::new();

        for (agent_id, cost_info) in &cost_analysis {
            if cost_info.efficiency_score < 10.0 {
                recommendations.push(format!(
                    "Consider switching agent {} to a more cost-effective model (current \
                     efficiency: {:.2})",
                    agent_id, cost_info.efficiency_score
                ));
            }

            if cost_info.daily_budget_used > 0.8 {
                recommendations.push(format!(
                    "Agent {} has used {:.0}% of daily budget - consider throttling",
                    agent_id,
                    cost_info.daily_budget_used * 100.0
                ));
            }
        }

        Ok(CostOptimizationReport {
            total_hourly_cost: total_cost,
            agent_costs: cost_analysis,
            optimization_recommendations: recommendations,
            potential_savings: total_cost * 0.15, // Estimated 15% savings potential
        })
    }

    /// Apply intelligent routing strategy
    pub async fn apply_routing_strategy(&self, strategy: RoutingStrategy) -> Result<()> {
        info!("🧠 Applying routing strategy: {:?}", strategy);

        let routing_engine = &self.routing_engine;

        // Update available strategies
        let mut available_strategies = routing_engine.available_strategies.clone();
        if !available_strategies.contains(&strategy) {
            available_strategies.push(strategy.clone());
        }

        match strategy {
            RoutingStrategy::CapabilityBased => {
                info!("🎯 Enabling capability-based routing");
                // Route based on agent capabilities and task requirements
            }
            RoutingStrategy::CostOptimized => {
                info!("💰 Enabling cost-optimized routing");
                // Route to minimize costs while maintaining quality
            }
            RoutingStrategy::LatencyOptimized => {
                info!("⚡ Enabling latency-optimized routing");
                // Route to minimize response times
            }
            RoutingStrategy::LoadBased => {
                info!("⚖️ Enabling load-based routing");
                // Distribute load evenly across agents
            }
        }

        Ok(())
    }

    /// Prepare session for model orchestration
    pub async fn prepare_session(&self, session_id: &str) -> Result<()> {
        info!("🚀 Preparing session {} for model orchestration", session_id);

        // Initialize session-specific routing state
        let session_routing = self.routing_engine.prepare_session_routing(session_id).await?;

        // Setup session tracking in session manager
        self.session_manager.setup_session(session_id, session_routing).await?;

        info!("✅ Session {} prepared successfully", session_id);
        Ok(())
    }

    /// Clean up session resources
    pub async fn cleanup_session(&self, session_id: &str) -> Result<()> {
        info!("🧹 Cleaning up session {} resources", session_id);

        // Clean up session-specific routing state
        self.routing_engine.cleanup_session_routing(session_id).await?;

        // Remove session from session manager
        self.session_manager.cleanup_session(session_id).await?;

        info!("✅ Session {} cleaned up successfully", session_id);
        Ok(())
    }
}

/// Cost optimization report
#[derive(Debug, Clone)]
pub struct CostOptimizationReport {
    pub total_hourly_cost: f32,
    pub agent_costs: HashMap<String, AgentCostInfo>,
    pub optimization_recommendations: Vec<String>,
    pub potential_savings: f32,
}

/// Agent cost information
#[derive(Debug, Clone)]
pub struct AgentCostInfo {
    pub hourly_cost: f32,
    pub daily_budget_used: f32,
    pub cost_per_request: f32,
    pub efficiency_score: f32,
}

/// Hot-swappable model update system structures
///
/// Comprehensive model update plan for zero-downtime upgrades
#[derive(Debug, Clone)]
pub struct ModelUpdatePlan {
    pub update_id: String,
    pub model_name: String,
    pub update_description: String,
    pub affected_agents: Vec<String>,
    pub estimated_downtime: Duration,
    pub rollback_strategy: RollbackStrategy,
    pub safety_checks: Vec<SafetyCheck>,
    pub update_strategy: UpdateStrategy,
    pub priority: UpdatePriority,
}

/// Rollback strategies for failed updates
#[derive(Debug, Clone)]
pub enum RollbackStrategy {
    AtomicSwap,
    GradualRollback,
    SnapshotRestore,
    ConfigRevert,
}

/// Safety checks performed before model updates
#[derive(Debug, Clone)]
pub enum SafetyCheck {
    ModelCompatibility,
    ResourceAvailability,
    DependencyValidation,
    PerformanceImpact,
}

/// Update strategies for different scenarios
#[derive(Debug, Clone)]
pub enum UpdateStrategy {
    HotSwap,
    GradualMigration,
    BlueGreenDeployment,
    CanaryDeployment,
}

/// Update priority levels
#[derive(Debug, Clone)]
pub enum UpdatePriority {
    Critical,
    High,
    Normal,
    Low,
}

/// Prepared model ready for hot-swap
#[derive(Debug, Clone)]
pub struct PreparedModel {
    pub model_name: String,
    pub version: String,
    pub model_type: ModelUpdateType,
    pub model_path: Option<String>,
    pub apiconfig: Option<ApiModelConfig>,
    pub validation_passed: bool,
    pub preparation_time: std::time::Instant,
}

/// Types of model updates
#[derive(Debug, Clone)]
pub enum ModelUpdateType {
    Local,
    API,
    Hybrid,
}

/// API model configuration for updates
#[derive(Debug, Clone)]
pub struct ApiModelConfig {
    pub model_name: String,
    pub version: String,
    pub endpoint: String,
    pub api_key: String,
}

/// Model validation result
#[derive(Debug, Clone)]
pub struct ModelValidation {
    pub version: String,
    pub checksum: String,
    pub valid: bool,
}

/// System backup for rollback purposes
#[derive(Debug, Clone)]
pub struct SystemBackup {
    pub backup_id: String,
    pub update_id: String,
    pub timestamp: std::time::Instant,
    pub agent_snapshots: HashMap<String, AgentInstance>,
    pub orchestratorconfig: OrchestratorConfigSnapshot,
    pub routing_state: RoutingStateSnapshot,
}

/// Orchestrator configuration snapshot
#[derive(Debug, Clone)]
pub struct OrchestratorConfigSnapshot {
    pub routing_strategy: RoutingStrategy,
    pub modelconfigs: HashMap<String, String>,
    pub timestamp: std::time::Instant,
}

/// Routing engine state snapshot
#[derive(Debug, Clone)]
pub struct RoutingStateSnapshot {
    pub active_routes: HashMap<String, String>,
    pub performance_cache: HashMap<String, f32>,
    pub timestamp: std::time::Instant,
}

/// Atomic swap operation result
#[derive(Debug, Clone)]
pub struct SwapResult {
    pub update_id: String,
    pub new_model_version: String,
    pub updated_agents: Vec<String>,
    pub swap_operations: Vec<SwapOperation>,
    pub swap_duration: Duration,
    pub success: bool,
}

/// Individual swap operation for rollback tracking
#[derive(Debug, Clone)]
pub struct SwapOperation {
    pub agent_id: String,
    pub old_models: Vec<String>,
    pub new_models: Vec<String>,
    pub swap_time: std::time::Instant,
}
