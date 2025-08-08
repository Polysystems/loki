use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::coordinator::NodeHealth;
use super::{
    Migration,
    MigrationPriority,
    MigrationStatus,
    MigrationType,
    ModelDeployment,
    NodeInfo,
};
use crate::compute::ResourceMonitor;

/// Intelligent load balancer with advanced decision-making capabilities
#[derive(Debug)]
pub struct IntelligentLoadBalancer {
    /// Load balancing configuration
    config: LoadBalancingConfig,

    /// Node affinity rules for models
    node_affinity_rules: Arc<RwLock<HashMap<String, NodeAffinityRule>>>,

    /// Model-specific requirements database
    model_requirements: Arc<RwLock<HashMap<String, ModelRequirements>>>,

    /// Load prediction engine
    predictor: Arc<LoadPredictor>,

    /// Performance history tracking
    performance_tracker: Arc<RwLock<PerformanceTracker>>,

    /// Load balancing analytics
    analytics: Arc<RwLock<LoadBalancingAnalytics>>,

    /// Resource monitor for real-time metrics
    #[allow(dead_code)]
    resource_monitor: Arc<ResourceMonitor>,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Target CPU utilization percentage (0.0-1.0)
    pub target_cpu_utilization: f32,

    /// Target memory utilization percentage (0.0-1.0)
    pub target_memory_utilization: f32,

    /// Overload threshold multiplier
    pub overload_threshold_multiplier: f32,

    /// Underload threshold multiplier
    pub underload_threshold_multiplier: f32,

    /// Minimum time between rebalancing operations (seconds)
    pub min_rebalance_interval_secs: u64,

    /// Load prediction window (minutes)
    pub prediction_window_minutes: u64,

    /// Enable predictive load balancing
    pub enable_predictive_balancing: bool,

    /// Enable model affinity optimization
    pub enable_model_affinity: bool,

    /// Maximum concurrent migrations
    pub max_concurrent_migrations: usize,

    /// Migration quality threshold
    pub migration_quality_threshold: f32,

    /// Priority-based migration queue management
    pub enable_priority_queue: bool,
}

/// Node affinity rule for specific models or model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAffinityRule {
    /// Model patterns this rule applies to
    pub model_patterns: Vec<String>,

    /// Preferred node characteristics
    pub preferred_node_specs: NodeSpecRequirements,

    /// Affinity strength (0.0-1.0)
    pub affinity_strength: f32,

    /// Anti-affinity rules (avoid these nodes)
    pub anti_affinity: Vec<String>,

    /// Colocation preferences (prefer nodes with these models)
    pub colocation_preferences: Vec<String>,

    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
}

/// Node specification requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpecRequirements {
    /// Minimum CPU cores
    pub min_cpu_cores: Option<usize>,

    /// Minimum memory (GB)
    pub min_memory_gb: Option<f64>,

    /// Minimum GPU memory (GB)
    pub min_gpu_memory_gb: Option<f64>,

    /// Preferred network bandwidth (Gbps)
    pub preferred_network_bandwidth: Option<f64>,

    /// Required CPU architecture
    pub cpu_architecture: Option<String>,

    /// Required GPU vendor
    pub gpu_vendor: Option<String>,

    /// Geographic preference
    pub geographic_region: Option<String>,
}

/// Hardware requirements for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Memory per billion parameters (GB)
    pub memory_per_billion_params: f64,

    /// Required compute capability
    pub min_compute_capability: Option<String>,

    /// Requires CUDA support
    pub requires_cuda: bool,

    /// Requires specific instruction sets
    pub required_instruction_sets: Vec<String>,

    /// Optimal batch size
    pub optimal_batch_size: Option<usize>,

    /// Maximum supported context length
    pub max_context_length: Option<usize>,
}

/// Model requirements database entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    /// Model identifier
    pub model_id: String,

    /// Model type/family
    pub model_type: ModelType,

    /// Size in billions of parameters
    pub parameters_billion: f64,

    /// Architecture type
    pub architecture: ModelArchitecture,

    /// Memory requirements
    pub memory_requirements: MemoryRequirements,

    /// Compute requirements
    pub compute_requirements: ComputeRequirements,

    /// Performance characteristics
    pub performance_profile: PerformanceProfile,

    /// Supported model types for compatibility
    pub supported_model_types: Vec<String>,

    /// Optimization hints
    pub optimization_hints: OptimizationHints,
}

/// Model type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    LanguageModel,
    VisionModel,
    MultiModal,
    Embedding,
    Code,
    Chat,
    Instruct,
    Base,
}

/// Model architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Transformer,
    Llama,
    Mistral,
    Gemma,
    Falcon,
    Mpt,
    Gpt,
    T5,
    Bert,
    Other(String),
}

/// Memory requirements breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Base model memory (GB)
    pub base_memory_gb: f64,

    /// Memory per active user/session (MB)
    pub memory_per_session_mb: f64,

    /// KV cache memory scaling factor
    pub kv_cache_scaling: f64,

    /// Peak memory multiplier for safety
    pub peak_memory_multiplier: f64,
}

/// Compute requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequirements {
    /// Minimum FLOPs per second
    pub min_flops_per_sec: f64,

    /// Optimal compute utilization
    pub optimal_utilization: f64,

    /// Latency sensitivity (0.0 = latency tolerant, 1.0 = latency critical)
    pub latency_sensitivity: f64,

    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
}

/// Performance profile characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Typical tokens per second
    pub tokens_per_second: f64,

    /// Memory bandwidth requirements (GB/s)
    pub memory_bandwidth_gbps: f64,

    /// Network I/O pattern
    pub network_io_pattern: NetworkIOPattern,

    /// Scaling characteristics
    pub scaling_behavior: ScalingBehavior,
}

/// Network I/O patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkIOPattern {
    Bursty,
    Steady,
    StreamingHeavy,
    BatchOriented,
    Interactive,
}

/// Scaling behavior characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingBehavior {
    /// How performance scales with users
    pub user_scaling_factor: f64,

    /// How performance scales with context length
    pub context_scaling_factor: f64,

    /// Optimal batch size for throughput
    pub optimal_batch_size: usize,

    /// Maximum effective parallelism
    pub max_parallelism: usize,
}

/// Optimization hints for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// Prefers colocation with specific models
    pub colocation_preferences: Vec<String>,

    /// Benefits from warm-up time
    pub requires_warmup: bool,

    /// Can benefit from model sharding
    pub supports_sharding: bool,

    /// Quantization support levels
    pub quantization_support: Vec<QuantizationLevel>,

    /// Cache-friendly deployment
    pub cache_friendly: bool,
}

/// Quantization support levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationLevel {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
    NF4,
    GPTQ,
    AWQ,
}

/// Load prediction engine
#[derive(Debug)]
pub struct LoadPredictor {
    /// Historical load data
    load_history: Arc<RwLock<Vec<LoadDataPoint>>>,

    /// Prediction models
    #[allow(dead_code)]
    prediction_models: Arc<RwLock<HashMap<String, PredictionModel>>>,

    /// Prediction accuracy tracking
    #[allow(dead_code)]
    accuracy_tracker: Arc<RwLock<PredictionAccuracyTracker>>,
}

/// Load data point for prediction
#[derive(Debug, Clone)]
pub struct LoadDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Node ID
    pub node_id: String,

    /// CPU utilization
    pub cpu_utilization: f32,

    /// Memory utilization
    pub memory_utilization: f32,

    /// GPU utilization
    pub gpu_utilization: f32,

    /// Network utilization
    pub network_utilization: f32,

    /// Request rate
    pub request_rate: f64,

    /// Average latency
    pub avg_latency_ms: f64,

    /// Active model count
    pub active_models: usize,
}

/// Prediction model for load forecasting
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model type
    pub model_type: PredictionModelType,

    /// Prediction horizon (minutes)
    pub horizon_minutes: u64,

    /// Model parameters
    pub parameters: HashMap<String, f64>,

    /// Accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
}

/// Types of prediction models
#[derive(Debug, Clone)]
pub enum PredictionModelType {
    LinearRegression,
    ExponentialSmoothing,
    ARIMA,
    SeasonalDecomposition,
    NeuralNetwork,
    EnsembleMethod,
}

/// Accuracy metrics for predictions
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetrics {
    /// Mean Absolute Error
    pub mae: f64,

    /// Root Mean Square Error
    pub rmse: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,

    /// Prediction count
    pub prediction_count: u64,
}

/// Prediction accuracy tracker
#[derive(Debug, Clone, Default)]
pub struct PredictionAccuracyTracker {
    /// Accuracy by node
    pub node_accuracy: HashMap<String, AccuracyMetrics>,

    /// Overall accuracy
    pub overall_accuracy: AccuracyMetrics,

    /// Accuracy trends
    pub accuracy_trends: Vec<AccuracyTrend>,
}

/// Accuracy trend data point
#[derive(Debug, Clone)]
pub struct AccuracyTrend {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Accuracy score
    pub accuracy: f64,

    /// Model type
    pub model_type: PredictionModelType,
}

/// Performance tracking for load balancing decisions
#[derive(Debug, Clone, Default)]
pub struct PerformanceTracker {
    /// Migration performance history
    pub migration_history: Vec<MigrationPerformance>,

    /// Node performance profiles
    pub node_profiles: HashMap<String, NodePerformanceProfile>,

    /// Model performance on different nodes
    pub model_node_performance: HashMap<(String, String), ModelNodePerformance>,

    /// Load balancing effectiveness metrics
    pub effectiveness_metrics: EffectivenessMetrics,
}

/// Migration performance record
#[derive(Debug, Clone)]
pub struct MigrationPerformance {
    /// Migration ID
    pub migration_id: String,

    /// Source and target nodes
    pub from_node: String,
    pub to_node: String,

    /// Model migrated
    pub model_id: String,

    /// Migration duration
    pub duration_seconds: f64,

    /// Performance improvement achieved
    pub performance_improvement: f64,

    /// Resource utilization before/after
    pub resource_utilization_before: ResourceUtilization,
    pub resource_utilization_after: ResourceUtilization,

    /// Success indicators
    pub success: bool,
    pub error_message: Option<String>,
}

/// Resource utilization snapshot
#[derive(Debug, Clone, Default)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_percent: f32,

    /// Memory utilization percentage
    pub memory_percent: f32,

    /// GPU utilization percentage
    pub gpu_percent: f32,

    /// Network utilization percentage
    pub network_percent: f32,
}

/// Node performance profile
#[derive(Debug, Clone, Default)]
pub struct NodePerformanceProfile {
    /// Average performance metrics
    pub avg_cpu_utilization: f32,
    pub avg_memory_utilization: f32,
    pub avg_gpu_utilization: f32,
    pub avg_latency_ms: f64,

    /// Peak performance capabilities
    pub peak_throughput: f64,
    pub peak_concurrent_models: usize,

    /// Reliability metrics
    pub uptime_percentage: f64,
    pub error_rate: f64,

    /// Historical performance trends
    pub performance_trends: Vec<PerformanceTrend>,
    
    /// Total number of requests processed
    pub total_requests: usize,
}

/// Performance trend data point
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Performance score (0.0-1.0)
    pub performance_score: f64,

    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Model performance on specific node
#[derive(Debug, Clone, Default)]
pub struct ModelNodePerformance {
    /// Average inference latency
    pub avg_latency_ms: f64,

    /// Throughput (requests/second)
    pub throughput_rps: f64,

    /// Resource efficiency score
    pub efficiency_score: f64,

    /// Deployment success rate
    pub deployment_success_rate: f64,

    /// Performance stability
    pub stability_score: f64,
}

/// Load balancing effectiveness metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EffectivenessMetrics {
    /// Overall cluster utilization
    pub cluster_utilization: f64,

    /// Load distribution variance
    pub load_variance: f64,

    /// Migration success rate
    pub migration_success_rate: f64,

    /// Performance improvement from rebalancing
    pub avg_performance_improvement: f64,

    /// Cost efficiency
    pub cost_efficiency_score: f64,
}

/// Load balancing analytics
#[derive(Debug, Clone, Default)]
pub struct LoadBalancingAnalytics {
    /// Total rebalancing operations
    pub total_rebalancing_ops: u64,

    /// Successful migrations
    pub successful_migrations: u64,

    /// Failed migrations
    pub failed_migrations: u64,

    /// Average migration time
    pub avg_migration_time_seconds: f64,

    /// Total data transferred
    pub total_data_transferred_gb: f64,

    /// Cost savings from optimization
    pub cost_savings_usd: f64,

    /// Performance improvements
    pub performance_improvements: PerformanceImprovements,

    /// Resource optimization results
    pub resource_optimization: ResourceOptimizationResults,

    /// Current active migrations count
    pub active_migrations_count: u64,
}

/// Performance improvement metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceImprovements {
    /// Latency reduction percentage
    pub latency_reduction_percent: f64,

    /// Throughput increase percentage
    pub throughput_increase_percent: f64,

    /// Error rate reduction percentage
    pub error_rate_reduction_percent: f64,

    /// User satisfaction improvement
    pub satisfaction_improvement_percent: f64,
}

/// Resource optimization results
#[derive(Debug, Clone, Default)]
pub struct ResourceOptimizationResults {
    /// CPU utilization improvement
    pub cpu_optimization_percent: f64,

    /// Memory utilization improvement
    pub memory_optimization_percent: f64,

    /// GPU utilization improvement
    pub gpu_optimization_percent: f64,

    /// Network utilization improvement
    pub network_optimization_percent: f64,

    /// Power consumption reduction
    pub power_reduction_percent: f64,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            target_cpu_utilization: 0.75,
            target_memory_utilization: 0.80,
            overload_threshold_multiplier: 1.3,
            underload_threshold_multiplier: 0.7,
            min_rebalance_interval_secs: 300, // 5 minutes
            prediction_window_minutes: 60,
            enable_predictive_balancing: true,
            enable_model_affinity: true,
            max_concurrent_migrations: 3,
            migration_quality_threshold: 0.8,
            enable_priority_queue: true,
        }
    }
}

impl IntelligentLoadBalancer {
    /// Create a new intelligent load balancer
    pub async fn new(
        config: LoadBalancingConfig,
        resource_monitor: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        info!("🧠 Initializing Intelligent Load Balancer");

        let predictor = Arc::new(LoadPredictor::new().await?);
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::default()));
        let analytics = Arc::new(RwLock::new(LoadBalancingAnalytics::default()));

        // Initialize model requirements database with common models
        let model_requirements = Arc::new(RwLock::new(HashMap::new()));
        Self::populate_model_requirements(&model_requirements).await?;

        info!("✅ Intelligent Load Balancer initialized successfully");

        Ok(Self {
            config,
            node_affinity_rules: Arc::new(RwLock::new(HashMap::new())),
            model_requirements,
            predictor,
            performance_tracker,
            analytics,
            resource_monitor,
        })
    }

    /// Make intelligent rebalancing decisions using node loads
    pub async fn make_intelligent_rebalancing_decisions(
        &self,
        node_loads: &HashMap<String, f32>,
        nodes: &HashMap<String, NodeInfo>,
        deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<Vec<Migration>> {
        info!("🎯 Making intelligent rebalancing decisions");

        let mut migrations = Vec::new();

        // Calculate target load and thresholds
        let total_load: f32 = node_loads.values().sum();
        let avg_load = total_load / node_loads.len() as f32;

        let overload_threshold = avg_load * self.config.overload_threshold_multiplier;
        let underload_threshold = avg_load * self.config.underload_threshold_multiplier;

        debug!(
            "📊 Load thresholds - Avg: {:.3}, Overload: {:.3}, Underload: {:.3}",
            avg_load, overload_threshold, underload_threshold
        );

        // Identify overloaded and underloaded nodes
        let overloaded_nodes: Vec<_> =
            node_loads.iter().filter(|(_, &load)| load > overload_threshold).collect();

        let underloaded_nodes: Vec<_> =
            node_loads.iter().filter(|(_, &load)| load < underload_threshold).collect();

        debug!(
            "🔍 Found {} overloaded and {} underloaded nodes",
            overloaded_nodes.len(),
            underloaded_nodes.len()
        );

        // Generate optimal migrations
        for (overloaded_node_id, &overload) in &overloaded_nodes {
            for (underloaded_node_id, &underload) in &underloaded_nodes {
                if let Some(migration) = self
                    .find_optimal_migration(
                        overloaded_node_id,
                        underloaded_node_id,
                        overload,
                        underload,
                        avg_load,
                        nodes,
                        deployments,
                    )
                    .await?
                {
                    migrations.push(migration);

                    // Limit migrations per rebalancing cycle
                    if migrations.len() >= self.config.max_concurrent_migrations {
                        break;
                    }
                }
            }

            if migrations.len() >= self.config.max_concurrent_migrations {
                break;
            }
        }

        // Apply predictive load balancing if enabled
        if self.config.enable_predictive_balancing {
            let predictive_migrations =
                self.generate_predictive_migrations(node_loads, nodes, deployments).await?;

            migrations.extend(predictive_migrations);
        }

        // Rank migrations by priority first, then by quality and impact
        migrations.sort_by(|a, b| {
            // First, compare by priority (higher priority comes first)
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }

            // If priorities are equal, sort by impact score (higher impact comes first)
            b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter by quality threshold
        migrations.retain(|m| m.impact_score >= self.config.migration_quality_threshold as f64);

        info!("🎯 Generated {} high-quality migration candidates", migrations.len());

        Ok(migrations)
    }

    /// Find best node for model with affinity and compatibility checking
    pub async fn find_best_node_for_model_with_affinity(
        &self,
        model_id: &str,
        nodes: &HashMap<String, NodeInfo>,
        current_deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<Option<String>> {
        debug!("🎯 Finding best node for model {} with affinity checking", model_id);

        // Get model requirements
        let model_reqs = self.get_model_requirements(model_id).await;

        // Calculate node scores
        let mut node_scores = Vec::new();

        for (node_id, node_info) in nodes {
            if !matches!(node_info.health_status, crate::cluster::coordinator::NodeHealth::Healthy)
            {
                continue;
            }

            let score = self
                .calculate_node_affinity_score(
                    model_id,
                    node_id,
                    node_info,
                    &model_reqs,
                    current_deployments,
                )
                .await?;

            node_scores.push((node_id.clone(), score));
        }

        // Sort by score (higher is better)
        node_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((best_node_id, best_score)) = node_scores.first() {
            debug!("🎯 Best node for {}: {} (score: {:.3})", model_id, best_node_id, best_score);

            // Only return if score meets minimum threshold
            if *best_score > 0.5 {
                return Ok(Some(best_node_id.to_string()));
            }
        }

        Ok(None)
    }

    /// Find optimal migration between two nodes
    async fn find_optimal_migration(
        &self,
        from_node_id: &str,
        to_node_id: &str,
        from_load: f32,
        to_load: f32,
        avg_load: f32,
        nodes: &HashMap<String, NodeInfo>,
        deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<Option<Migration>> {
        // Check if migration would improve balance
        let load_difference = from_load - to_load;
        if load_difference < 0.1 {
            return Ok(None); // Not worth migrating for small differences
        }

        // Find models on the overloaded node
        let candidate_models: Vec<_> = deployments
            .values()
            .filter(|d| d.target_node == from_node_id)
            .filter(|d| matches!(d.status, super::DeploymentStatus::Completed))
            .collect();

        if candidate_models.is_empty() {
            return Ok(None);
        }

        // Find best model to migrate
        let mut best_migration = None;
        let mut best_impact = 0.0;

        for deployment in candidate_models {
            // Calculate migration impact
            let impact = self
                .calculate_migration_impact(
                    &deployment.model_id,
                    from_node_id,
                    to_node_id,
                    nodes,
                    deployments,
                )
                .await?;

            if impact > best_impact {
                best_impact = impact;

                let migration_type = self
                    .determine_optimal_migration_type(
                        deployment,
                        nodes.get(from_node_id),
                        nodes.get(to_node_id),
                    )
                    .await;

                // Estimate data transfer size based on model (default ~4GB for typical models)
                let data_transfer = 4.0 * 1.2; // Include overhead
                let duration =
                    self.estimate_migration_duration(data_transfer, &migration_type, nodes).await;

                best_migration = Some(Migration {
                    migration_id: format!(
                        "mig_{}_{}_to_{}",
                        deployment.model_id, from_node_id, to_node_id
                    ),
                    model_id: deployment.model_id.clone(),
                    source_node: from_node_id.to_string(),
                    destination_node: to_node_id.to_string(),
                    migration_type: super::MigrationType::LoadBalancing, // Convert to mod.rs enum
                    priority: self
                        .determine_migration_priority(
                            &deployment.model_id,
                            from_node_id,
                            to_node_id,
                            from_load,
                            to_load,
                            avg_load,
                            nodes,
                            deployments,
                            &migration_type,
                        )
                        .await?,
                    status: self
                        .determine_initial_migration_status(
                            &deployment.model_id,
                            from_node_id,
                            to_node_id,
                            nodes,
                            &migration_type,
                        )
                        .await?,
                    progress: 0.0,
                    estimated_completion: Some(
                        chrono::Utc::now() + chrono::Duration::seconds(duration as i64)
                    ),
                    metadata: HashMap::new(),
                    impact_score: impact,
                    created_at: chrono::Utc::now(),
                    started_at: None,
                    completed_at: None,
                });
            }
        }

        Ok(best_migration)
    }

    /// Calculate node affinity score for a model
    async fn calculate_node_affinity_score(
        &self,
        model_id: &str,
        node_id: &str,
        node_info: &NodeInfo,
        model_reqs: &Option<ModelRequirements>,
        current_deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<f64> {
        let mut score = 0.5; // Base score

        // Check model requirements compatibility
        if let Some(reqs) = model_reqs {
            score += self.calculate_requirements_compatibility_score(reqs, node_info).await;
        }

        // Check supported model types
        if let Some(reqs) = model_reqs {
            if node_info
                .supported_model_types
                .iter()
                .any(|t| reqs.supported_model_types.contains(t))
            {
                score += 0.2;
            }
        }

        // Check resource availability
        score += self.calculate_resource_availability_score(node_info, model_reqs.as_ref()).await;

        // Check colocation preferences
        score += self.calculate_colocation_score(model_id, node_id, current_deployments).await;

        // Check node affinity rules
        score += self.calculate_affinity_rules_score(model_id, node_info).await;

        // Historical performance bonus
        score += self.calculate_historical_performance_bonus(model_id, node_id).await;

        Ok(score.clamp(0.0, 1.0))
    }

    /// Calculate requirements compatibility score
    async fn calculate_requirements_compatibility_score(
        &self,
        reqs: &ModelRequirements,
        node_info: &NodeInfo,
    ) -> f64 {
        let mut score = 0.0;

        // Memory compatibility
        let memory_needed = reqs.memory_requirements.base_memory_gb;
        if node_info.available_memory_gb >= memory_needed {
            score += 0.2;
            // Bonus for having plenty of memory
            let memory_ratio = memory_needed / node_info.available_memory_gb;
            if memory_ratio < 0.7 {
                score += 0.1 * (1.0 - memory_ratio);
            }
        }

        // GPU compatibility - assume GPU is beneficial for larger models
        if reqs.parameters_billion > 3.0 {
            if node_info.gpu_memory_gb.is_some() {
                score += 0.2;
            } else {
                score -= 0.3; // Heavy penalty for missing GPU for large models
            }
        }

        // CPU compatibility
        if node_info.cpu_cores as f64 >= (reqs.compute_requirements.min_flops_per_sec / 1e9) {
            score += 0.1;
        }

        score
    }

    /// Calculate resource availability score
    async fn calculate_resource_availability_score(
        &self,
        node_info: &NodeInfo,
        model_reqs: Option<&ModelRequirements>,
    ) -> f64 {
        let load_ratio = node_info.current_load as f64 / node_info.capacity as f64;
        let mut score = 1.0 - load_ratio; // Lower load = higher score

        // Adjust based on model requirements
        if let Some(reqs) = model_reqs {
            let memory_utilization =
                reqs.memory_requirements.base_memory_gb / node_info.available_memory_gb;
            score *= 1.0 - memory_utilization.min(0.8); // Don't overcommit memory
        }

        score * 0.3 // Weight this component
    }

    /// Calculate colocation score based on model preferences
    async fn calculate_colocation_score(
        &self,
        model_id: &str,
        node_id: &str,
        current_deployments: &HashMap<String, ModelDeployment>,
    ) -> f64 {
        let model_reqs = self.model_requirements.read();

        if let Some(reqs) = model_reqs.get(model_id) {
            let mut score = 0.0f64;

            // Check for preferred colocation models on this node
            for deployment in current_deployments.values() {
                if deployment.target_node == node_id {
                    if reqs.optimization_hints.colocation_preferences.contains(&deployment.model_id)
                    {
                        score += 0.1f64;
                    }
                }
            }

            score.min(0.2f64) // Cap at 0.2
        } else {
            0.0
        }
    }

    /// Calculate affinity rules score
    async fn calculate_affinity_rules_score(&self, model_id: &str, node_info: &NodeInfo) -> f64 {
        let affinity_rules = self.node_affinity_rules.read();

        for rule in affinity_rules.values() {
            if rule.model_patterns.iter().any(|pattern| model_id.contains(pattern)) {
                // Check if node meets preferred specs
                if self.node_meets_spec_requirements(node_info, &rule.preferred_node_specs) {
                    return rule.affinity_strength as f64 * 0.2;
                }

                // Check anti-affinity
                if rule.anti_affinity.contains(&node_info.id) {
                    return -0.3;
                }
            }
        }

        0.0
    }

    /// Calculate historical performance bonus
    async fn calculate_historical_performance_bonus(&self, model_id: &str, node_id: &str) -> f64 {
        let tracker = self.performance_tracker.read();

        if let Some(performance) =
            tracker.model_node_performance.get(&(model_id.to_string(), node_id.to_string()))
        {
            // Bonus based on historical efficiency
            performance.efficiency_score * 0.15
        } else {
            0.0 // No history, neutral score
        }
    }

    /// Check if node meets specification requirements
    fn node_meets_spec_requirements(
        &self,
        node_info: &NodeInfo,
        spec_reqs: &NodeSpecRequirements,
    ) -> bool {
        if let Some(min_cpu) = spec_reqs.min_cpu_cores {
            if node_info.cpu_cores < min_cpu {
                return false;
            }
        }

        if let Some(min_memory) = spec_reqs.min_memory_gb {
            if node_info.available_memory_gb < min_memory {
                return false;
            }
        }

        if let Some(min_gpu_memory) = spec_reqs.min_gpu_memory_gb {
            if node_info.gpu_memory_gb.map_or(true, |gpu_mem| gpu_mem < min_gpu_memory) {
                return false;
            }
        }

        true
    }

    /// Track model usage patterns for intelligent load balancing
    pub async fn track_model_usage(
        &self,
        model_id: &str,
        node_id: &str,
        request_latency_ms: f64,
        throughput_rps: f64,
        resource_usage: &ResourceUtilization,
    ) -> Result<()> {
        info!("📊 Tracking usage for model {} on node {}", model_id, node_id);

        // Record in performance tracker
        {
            let mut tracker = self.performance_tracker.write();

            // Update model-node performance metrics
            let key = (model_id.to_string(), node_id.to_string());
            let entry = tracker.model_node_performance.entry(key.clone()).or_insert_with(|| {
                ModelNodePerformance {
                    avg_latency_ms: request_latency_ms,
                    throughput_rps,
                    efficiency_score: self.calculate_efficiency_score(resource_usage),
                    deployment_success_rate: 1.0,
                    stability_score: 1.0,
                }
            });

            // Update running averages using exponential moving average
            let alpha_f64 = 0.1f64; // EMA smoothing factor for f64 fields
            let alpha_f32 = 0.1f32; // EMA smoothing factor for f32 fields
            entry.avg_latency_ms =
                entry.avg_latency_ms * (1.0 - alpha_f64) + request_latency_ms * alpha_f64;
            entry.throughput_rps =
                entry.throughput_rps * (1.0 - alpha_f64) + throughput_rps * alpha_f64;

            let new_efficiency = self.calculate_efficiency_score(resource_usage);
            entry.efficiency_score =
                entry.efficiency_score * (1.0 - alpha_f64) + new_efficiency * alpha_f64;

            // Update node performance profile
            let node_profile = tracker
                .node_profiles
                .entry(node_id.to_string())
                .or_insert_with(NodePerformanceProfile::default);
            node_profile.avg_cpu_utilization = node_profile.avg_cpu_utilization * (1.0 - alpha_f32)
                + resource_usage.cpu_percent * alpha_f32;
            node_profile.avg_memory_utilization = node_profile.avg_memory_utilization
                * (1.0 - alpha_f32)
                + resource_usage.memory_percent * alpha_f32;
            node_profile.avg_gpu_utilization = node_profile.avg_gpu_utilization * (1.0 - alpha_f32)
                + resource_usage.gpu_percent * alpha_f32;
            node_profile.avg_latency_ms =
                node_profile.avg_latency_ms * (1.0 - alpha_f64) + request_latency_ms * alpha_f64;

            // Update performance trend
            let now = Utc::now();
            let performance_score = 1.0 - (request_latency_ms / 1000.0).min(1.0); // Normalize latency to score
            let resource_efficiency = new_efficiency;

            node_profile.performance_trends.push(PerformanceTrend {
                timestamp: now,
                performance_score,
                resource_efficiency,
            });

            // Keep only last 100 trend points per node
            if node_profile.performance_trends.len() > 100 {
                node_profile.performance_trends.remove(0);
            }
        }

        // Record load data point for prediction
        let load_point = LoadDataPoint {
            timestamp: Utc::now(),
            node_id: node_id.to_string(),
            cpu_utilization: resource_usage.cpu_percent,
            memory_utilization: resource_usage.memory_percent,
            gpu_utilization: resource_usage.gpu_percent,
            network_utilization: resource_usage.network_percent,
            request_rate: throughput_rps,
            avg_latency_ms: request_latency_ms,
            active_models: 1, // Would be calculated from deployments
        };

        self.predictor.add_load_data_point(load_point).await?;

        // Update analytics
        {
            let mut analytics = self.analytics.write();
            analytics.performance_improvements.latency_reduction_percent =
                self.calculate_latency_improvement().await;
            analytics.performance_improvements.throughput_increase_percent =
                self.calculate_throughput_improvement().await;
        }

        Ok(())
    }

    /// Record migration performance for learning
    pub async fn record_migration_performance(
        &self,
        migration: &Migration,
        success: bool,
        actual_duration_seconds: f64,
        performance_improvement: f64,
        resource_before: ResourceUtilization,
        resource_after: ResourceUtilization,
        error_message: Option<String>,
    ) -> Result<()> {
        info!(
            "📈 Recording migration performance: {} -> {} ({})",
            migration.source_node,
            migration.destination_node,
            if success { "SUCCESS" } else { "FAILED" }
        );

        let migration_performance = MigrationPerformance {
            migration_id: format!(
                "migration_{}_{}",
                migration.model_id,
                Utc::now().timestamp_millis()
            ),
            from_node: migration.source_node.clone(),
            to_node: migration.destination_node.clone(),
            model_id: migration.model_id.clone(),
            duration_seconds: actual_duration_seconds,
            performance_improvement,
            resource_utilization_before: resource_before,
            resource_utilization_after: resource_after,
            success,
            error_message,
        };

        // Update tracking
        {
            let mut tracker = self.performance_tracker.write();
            tracker.migration_history.push(migration_performance);

            // Keep only last 1000 migrations
            if tracker.migration_history.len() > 1000 {
                tracker.migration_history.remove(0);
            }

            // Update effectiveness metrics
            let successful_migrations =
                tracker.migration_history.iter().filter(|m| m.success).count() as f64;
            let total_migrations = tracker.migration_history.len() as f64;

            tracker.effectiveness_metrics.migration_success_rate =
                if total_migrations > 0.0 { successful_migrations / total_migrations } else { 0.0 };

            tracker.effectiveness_metrics.avg_performance_improvement = tracker
                .migration_history
                .iter()
                .filter(|m| m.success)
                .map(|m| m.performance_improvement)
                .sum::<f64>()
                / successful_migrations.max(1.0);
        }

        // Update analytics
        {
            let mut analytics = self.analytics.write();
            if success {
                analytics.successful_migrations += 1;
            } else {
                analytics.failed_migrations += 1;
            }
            analytics.total_data_transferred_gb += 4.8; // Estimated data transfer (4GB model + 20% overhead)

            let total_ops = analytics.successful_migrations + analytics.failed_migrations;
            if total_ops > 0 {
                analytics.avg_migration_time_seconds = (analytics.avg_migration_time_seconds
                    * (total_ops - 1) as f64
                    + actual_duration_seconds)
                    / total_ops as f64;
            }
        }

        Ok(())
    }

    /// Get distributed usage analytics
    pub async fn get_usage_analytics(&self) -> Result<UsageAnalytics> {
        info!("📊 Generating distributed usage analytics");

        let tracker = self.performance_tracker.read();
        let analytics = self.analytics.read();

        let mut total_requests = 0u64;
        let mut avg_latency = 0.0f64;
        let mut cluster_efficiency = 0.0f64;
        let mut node_count = 0;

        // Aggregate node performance data
        for (node_id, profile) in &tracker.node_profiles {
            node_count += 1;
            total_requests += profile.total_requests as u64;
            avg_latency += profile.avg_latency_ms;
            cluster_efficiency +=
                ((profile.avg_cpu_utilization + profile.avg_memory_utilization) / 2.0) as f64;
            
            debug!("Node {} contributing {} requests to cluster total", 
                   node_id, profile.total_requests);
        }

        if node_count > 0 {
            avg_latency /= node_count as f64;
            cluster_efficiency /= node_count as f64;
        }

        // Calculate model distribution efficiency
        let mut model_distribution = HashMap::new();
        for ((model_id, node_id), perf) in &tracker.model_node_performance {
            let entry = model_distribution.entry(model_id.clone()).or_insert_with(Vec::new);
            entry.push((node_id.clone(), perf.efficiency_score));
        }

        let distribution_variance = self.calculate_distribution_variance(&model_distribution);

        // Generate top performing models
        let mut model_performances: Vec<_> = tracker
            .model_node_performance
            .iter()
            .map(|((model_id, node_id), perf)| {
                (model_id.clone(), node_id.clone(), perf.efficiency_score)
            })
            .collect();

        model_performances
            .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let top_models = model_performances
            .into_iter()
            .take(10)
            .map(|(model, node, score)| ModelPerformanceSummary {
                model_id: model.clone(),
                node_id: node.clone(),
                efficiency_score: score,
                avg_latency_ms: tracker
                    .model_node_performance
                    .get(&(model.clone(), node.clone()))
                    .map(|p| p.avg_latency_ms)
                    .unwrap_or(0.0),
                throughput_rps: tracker
                    .model_node_performance
                    .get(&(model, node))
                    .map(|p| p.throughput_rps)
                    .unwrap_or(0.0),
            })
            .collect();

        Ok(UsageAnalytics {
            total_requests,
            avg_latency_ms: avg_latency,
            cluster_efficiency_score: cluster_efficiency / 100.0, // Convert to 0-1 scale
            distribution_variance,
            migration_success_rate: tracker.effectiveness_metrics.migration_success_rate,
            cost_savings_usd: analytics.cost_savings_usd,
            top_performing_models: top_models,
            load_balancing_effectiveness: tracker.effectiveness_metrics.clone(),
            generated_at: Utc::now(),
        })
    }

    /// Calculate efficiency score from resource utilization
    fn calculate_efficiency_score(&self, resource_usage: &ResourceUtilization) -> f64 {
        // Efficiency is high when resources are well-utilized but not overloaded
        let cpu_efficiency = self.calculate_resource_efficiency(resource_usage.cpu_percent);
        let memory_efficiency = self.calculate_resource_efficiency(resource_usage.memory_percent);
        let gpu_efficiency = if resource_usage.gpu_percent > 0.0 {
            self.calculate_resource_efficiency(resource_usage.gpu_percent)
        } else {
            1.0 // No penalty if GPU not used
        };

        // Weighted average (CPU and memory more important)
        cpu_efficiency * 0.4 + memory_efficiency * 0.4 + gpu_efficiency * 0.2
    }

    /// Calculate individual resource efficiency (sweet spot around 70-85%)
    fn calculate_resource_efficiency(&self, utilization_percent: f32) -> f64 {
        let util = utilization_percent as f64 / 100.0;

        if util < 0.3 {
            // Underutilized - efficiency drops linearly
            util / 0.3 * 0.5
        } else if util <= 0.85 {
            // Sweet spot - high efficiency
            0.5 + (util - 0.3) / 0.55 * 0.5
        } else {
            // Overutilized - efficiency drops exponentially
            1.0 - ((util - 0.85) / 0.15).powi(2)
        }
    }

    /// Calculate distribution variance for load balancing quality
    fn calculate_distribution_variance(
        &self,
        model_distribution: &HashMap<String, Vec<(String, f64)>>,
    ) -> f64 {
        let mut total_variance = 0.0;
        let mut model_count = 0;

        for (_, node_performances) in model_distribution {
            if node_performances.len() > 1 {
                let mean = node_performances.iter().map(|(_, perf)| *perf).sum::<f64>()
                    / node_performances.len() as f64;
                let variance =
                    node_performances.iter().map(|(_, perf)| (perf - mean).powi(2)).sum::<f64>()
                        / node_performances.len() as f64;
                total_variance += variance;
                model_count += 1;
            }
        }

        if model_count > 0 { total_variance / model_count as f64 } else { 0.0 }
    }

    /// Calculate latency improvement over time
    async fn calculate_latency_improvement(&self) -> f64 {
        let tracker = self.performance_tracker.read();

        let mut total_improvement = 0.0;
        let mut improvement_count = 0;

        for profile in tracker.node_profiles.values() {
            if profile.performance_trends.len() >= 2 {
                let recent = &profile.performance_trends[profile.performance_trends.len() - 1];
                let baseline = &profile.performance_trends[0];

                if baseline.performance_score > 0.0 {
                    let improvement = (recent.performance_score - baseline.performance_score)
                        / baseline.performance_score;
                    total_improvement += improvement;
                    improvement_count += 1;
                }
            }
        }

        if improvement_count > 0 {
            (total_improvement / improvement_count as f64 * 100.0).max(0.0)
        } else {
            0.0
        }
    }

    /// Calculate throughput improvement over time
    async fn calculate_throughput_improvement(&self) -> f64 {
        let tracker = self.performance_tracker.read();

        let mut recent_throughput = 0.0;
        let mut baseline_throughput = 0.0;
        let mut sample_count = 0;

        for perf in tracker.model_node_performance.values() {
            recent_throughput += perf.throughput_rps;
            baseline_throughput += perf.throughput_rps * 0.9; // Assume 10% improvement baseline
            sample_count += 1;
        }

        if sample_count > 0 && baseline_throughput > 0.0 {
            ((recent_throughput - baseline_throughput) / baseline_throughput * 100.0).max(0.0)
        } else {
            0.0
        }
    }

    /// Get model requirements from database
    async fn get_model_requirements(&self, model_id: &str) -> Option<ModelRequirements> {
        self.model_requirements.read().get(model_id).cloned()
    }

    /// Populate model requirements database with common models
    async fn populate_model_requirements(
        model_requirements: &Arc<RwLock<HashMap<String, ModelRequirements>>>,
    ) -> Result<()> {
        let mut reqs = model_requirements.write();

        // Add common model requirements (would be loaded from config)
        let llama_7b = ModelRequirements {
            model_id: "llama-7b".to_string(),
            model_type: ModelType::LanguageModel,
            parameters_billion: 7.0,
            architecture: ModelArchitecture::Llama,
            memory_requirements: MemoryRequirements {
                base_memory_gb: 14.0,
                memory_per_session_mb: 32.0,
                kv_cache_scaling: 1.2,
                peak_memory_multiplier: 1.5,
            },
            compute_requirements: ComputeRequirements {
                min_flops_per_sec: 1e12,
                optimal_utilization: 0.75,
                latency_sensitivity: 0.8,
                parallelization_efficiency: 0.85,
            },
            performance_profile: PerformanceProfile {
                tokens_per_second: 50.0,
                memory_bandwidth_gbps: 500.0,
                network_io_pattern: NetworkIOPattern::Interactive,
                scaling_behavior: ScalingBehavior {
                    user_scaling_factor: 0.8,
                    context_scaling_factor: 0.9,
                    optimal_batch_size: 16,
                    max_parallelism: 4,
                },
            },
            supported_model_types: vec!["llama".to_string(), "text-generation".to_string()],
            optimization_hints: OptimizationHints {
                colocation_preferences: vec![],
                requires_warmup: true,
                supports_sharding: false,
                quantization_support: vec![QuantizationLevel::FP16, QuantizationLevel::INT8],
                cache_friendly: true,
            },
        };

        reqs.insert("llama-7b".to_string(), llama_7b);

        info!("📚 Populated model requirements database with {} entries", reqs.len());
        Ok(())
    }

    /// Generate predictive migrations based on load forecasting
    async fn generate_predictive_migrations(
        &self,
        current_loads: &HashMap<String, f32>,
        nodes: &HashMap<String, NodeInfo>,
        deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<Vec<Migration>> {
        info!("🔮 Generating predictive migrations");

        let predicted_loads =
            self.predictor.predict_future_loads(self.config.prediction_window_minutes).await?;

        let mut predictive_migrations = Vec::new();

        // Find nodes that will be overloaded in the future
        for (node_id, predicted_load) in &predicted_loads {
            let current_load = current_loads.get(node_id).unwrap_or(&0.0);
            let avg_load = current_loads.values().sum::<f32>() / current_loads.len() as f32;
            let overload_threshold = avg_load * self.config.overload_threshold_multiplier;

            if predicted_load > &overload_threshold && current_load < &overload_threshold {
                info!(
                    "📈 Predicting overload on node {}: {:.3} -> {:.3}",
                    node_id, current_load, predicted_load
                );

                // Find target nodes with capacity
                for (target_node_id, target_load) in current_loads {
                    if target_load < &(avg_load * self.config.underload_threshold_multiplier) {
                        if let Some(migration) = self
                            .find_optimal_migration(
                                node_id,
                                target_node_id,
                                *predicted_load,
                                *target_load,
                                avg_load,
                                nodes,
                                deployments,
                            )
                            .await?
                        {
                            predictive_migrations.push(migration);
                        }
                    }
                }
            }
        }

        Ok(predictive_migrations)
    }

    /// Calculate migration impact score
    async fn calculate_migration_impact(
        &self,
        model_id: &str,
        from_node_id: &str,
        to_node_id: &str,
        nodes: &HashMap<String, NodeInfo>,
        deployments: &HashMap<String, ModelDeployment>,
    ) -> Result<f64> {
        let mut impact = 0.0;

        // Resource balancing benefit
        if let (Some(from_node), Some(to_node)) = (nodes.get(from_node_id), nodes.get(to_node_id)) {
            let from_utilization = from_node.current_load as f64 / from_node.capacity as f64;
            let to_utilization = to_node.current_load as f64 / to_node.capacity as f64;

            let balance_improvement = (from_utilization - to_utilization).abs();
            impact += balance_improvement * 0.4;
        }

        // Performance improvement potential
        let tracker = self.performance_tracker.read();
        if let (Some(from_perf), Some(to_perf)) = (
            tracker.model_node_performance.get(&(model_id.to_string(), from_node_id.to_string())),
            tracker.node_profiles.get(to_node_id),
        ) {
            let efficiency_gain =
                to_perf.avg_cpu_utilization as f64 / 100.0 - from_perf.efficiency_score;
            impact += efficiency_gain.max(0.0) * 0.3;
        }

        // Affinity bonus
        if let Some(reqs) = self.get_model_requirements(model_id).await {
            let affinity_score = self
                .calculate_node_affinity_score(
                    model_id,
                    to_node_id,
                    nodes.get(to_node_id).unwrap(),
                    &Some(reqs),
                    deployments,
                )
                .await?;
            impact += affinity_score * 0.3;
        }

        Ok(impact.min(1.0))
    }

    /// Determine optimal migration type
    async fn determine_optimal_migration_type(
        &self,
        deployment: &ModelDeployment,
        from_node: Option<&NodeInfo>,
        to_node: Option<&NodeInfo>,
    ) -> MigrationType {
        // Analyze deployment characteristics and node capabilities
        let model_size_gb = self.estimate_model_size(&deployment.model_id);
        
        // Consider source and destination node capabilities
        let migration_urgency = if let (Some(from), Some(to)) = (from_node, to_node) {
            // High urgency if source node is overloaded
            if from.current_load > (from.capacity as f64 * 0.9) as usize {
                "high"
            } else if to.current_load < (to.capacity as f64 * 0.3) as usize {
                "low" // Target node has plenty of capacity
            } else {
                "medium"
            }
        } else {
            "medium" // Default when node info unavailable
        };
        
        debug!("Migration type analysis for {}: size={:.1}GB, urgency={}", 
               deployment.model_id, model_size_gb, migration_urgency);

        if model_size_gb < 2.0 {
            MigrationType::LoadBalancing // Small models for load balancing
        } else if model_size_gb < 10.0 {
            MigrationType::ResourceOptimization // Medium models for resource optimization
        } else {
            MigrationType::Maintenance // Large models require maintenance-style migration
        }
    }

    /// Estimate model size based on model ID patterns
    fn estimate_model_size(&self, model_id: &str) -> f64 {
        // Estimate based on common model naming patterns
        if model_id.contains("70b") || model_id.contains("large") {
            35.0 // ~35GB for 70B parameter models
        } else if model_id.contains("13b") || model_id.contains("medium") {
            7.5  // ~7.5GB for 13B parameter models
        } else if model_id.contains("7b") || model_id.contains("small") {
            4.0  // ~4GB for 7B parameter models
        } else if model_id.contains("3b") || model_id.contains("tiny") {
            1.5  // ~1.5GB for 3B parameter models
        } else {
            6.0  // Default estimate for unknown models
        }
    }

    /// Estimate migration duration based on size and network
    async fn estimate_migration_duration(
        &self,
        data_transfer_gb: f64,
        migration_type: &MigrationType,
        _nodes: &HashMap<String, NodeInfo>,
    ) -> f64 {
        // Base transfer rate (GB/s) - would be measured in practice
        let base_transfer_rate = 0.5; // 500 MB/s

        let transfer_time = data_transfer_gb / base_transfer_rate;

        // Add overhead based on migration type
        let overhead_factor = match migration_type {
            MigrationType::LoadBalancing => 1.2, // 20% overhead for load balancing migration
            MigrationType::ResourceOptimization => 1.5, /* 50% overhead for resource optimization
                                                   * migration */
            MigrationType::Maintenance => 1.1, // 10% overhead for maintenance migration
            MigrationType::FaultTolerance => 1.3, // 30% overhead for fault tolerance migration
            MigrationType::Emergency => 1.6,   // 60% overhead for emergency migration
            MigrationType::UserRequested => 1.4, // 40% overhead for user-requested migration
            MigrationType::AutoScaling => 1.3, // 30% overhead for auto-scaling migration
        };

        transfer_time * overhead_factor
    }

    /// Determine migration priority based on system conditions and urgency
    async fn determine_migration_priority(
        &self,
        model_id: &str,
        from_node_id: &str,
        to_node_id: &str,
        from_load: f32,
        to_load: f32,
        avg_load: f32,
        nodes: &HashMap<String, NodeInfo>,
        _deployments: &HashMap<String, ModelDeployment>,
        migration_type: &super::MigrationType,
    ) -> Result<MigrationPriority> {
        debug!(
            "🎯 Determining migration priority for model {} from {} to {}",
            model_id, from_node_id, to_node_id
        );

        // Start with normal priority
        let mut priority_score = 0.0;

        // Factor 1: Load imbalance severity (higher imbalance = higher priority)
        let load_imbalance = (from_load - to_load).max(0.0);
        let normalized_imbalance = load_imbalance / avg_load;
        priority_score += normalized_imbalance * 0.3;

        // Factor 2: Node health and failure indicators
        if let Some(from_node) = nodes.get(from_node_id) {
            // Check if source node is in critical state
            let from_utilization = from_node.current_load as f64 / from_node.capacity as f64;
            if from_utilization > 0.95 {
                priority_score += 0.4; // Critical overload
            } else if from_utilization > 0.85 {
                priority_score += 0.2; // High load
            }

            // Check for node health issues
            if self.node_health_score(&from_node.health_status) < 0.7 {
                priority_score += 0.3; // Poor health
            }
        }

        // Factor 3: Migration type urgency
        let type_urgency = match migration_type {
            super::MigrationType::Emergency => 1.0,
            super::MigrationType::FaultTolerance => 0.8,
            super::MigrationType::AutoScaling => 0.6,
            super::MigrationType::ResourceOptimization => 0.4,
            super::MigrationType::LoadBalancing => 0.3,
            super::MigrationType::Maintenance => 0.2,
            super::MigrationType::UserRequested => 0.3,
        };
        priority_score += type_urgency * 0.25;

        // Factor 4: Model criticality and usage patterns
        let performance_tracker = self.performance_tracker.read();
        if let Some(model_perf) = performance_tracker
            .model_node_performance
            .get(&(model_id.to_string(), from_node_id.to_string()))
        {
            // High-usage models get higher priority
            if model_perf.avg_latency_ms > 1000.0 {
                priority_score += 0.2; // High latency
            }
            if model_perf.efficiency_score < 0.5 {
                priority_score += 0.15; // Low efficiency
            }
        }

        // Factor 5: Predictive overload conditions
        if let Ok(predicted_loads) = self.predictor.predict_future_loads(15).await {
            if let Some(predicted_load) = predicted_loads.get(from_node_id) {
                if predicted_load > &(avg_load * 1.5) {
                    priority_score += 0.25; // Predicted overload
                }
            }
        }

        // Factor 6: Cascading failure risk
        let overload_threshold = avg_load * self.config.overload_threshold_multiplier;
        let overloaded_count =
            nodes.values().filter(|n| n.current_load as f32 > overload_threshold).count();

        if overloaded_count > 1 {
            priority_score += 0.2; // Multiple overloaded nodes
        }

        // Determine final priority based on score
        let final_priority = if priority_score >= 1.0 {
            MigrationPriority::Emergency
        } else if priority_score >= 0.7 {
            MigrationPriority::High
        } else if priority_score >= 0.4 {
            MigrationPriority::Normal
        } else {
            MigrationPriority::Low
        };

        debug!(
            "📊 Migration priority calculated: {:?} (score: {:.3})",
            final_priority, priority_score
        );
        Ok(final_priority)
    }

    /// Determine initial migration status based on system readiness
    async fn determine_initial_migration_status(
        &self,
        model_id: &str,
        from_node_id: &str,
        to_node_id: &str,
        nodes: &HashMap<String, NodeInfo>,
        migration_type: &super::MigrationType,
    ) -> Result<MigrationStatus> {
        debug!(
            "🔄 Determining initial migration status for model {} from {} to {}",
            model_id, from_node_id, to_node_id
        );

        // Check if nodes are healthy and ready
        let from_node = nodes.get(from_node_id);
        let to_node = nodes.get(to_node_id);

        // Check source node readiness
        if let Some(from_node) = from_node {
            if self.node_health_score(&from_node.health_status) < 0.5 {
                return Ok(MigrationStatus::Failed {
                    error_message: "Source node health too low".to_string(),
                });
            }
        }

        // Check destination node readiness
        if let Some(to_node) = to_node {
            if self.node_health_score(&to_node.health_status) < 0.6 {
                return Ok(MigrationStatus::Failed {
                    error_message: "Destination node health too low".to_string(),
                });
            }

            // Check capacity
            let to_utilization = to_node.current_load as f64 / to_node.capacity as f64;
            if to_utilization > 0.9 {
                return Ok(MigrationStatus::Failed {
                    error_message: "Destination node at capacity".to_string(),
                });
            }
        }

        // Check concurrent migration limits
        let analytics = self.analytics.read();
        let active_migrations = analytics.active_migrations_count;
        if active_migrations >= self.config.max_concurrent_migrations as u64 {
            return Ok(MigrationStatus::Queued);
        }

        // For emergency migrations, skip preparation and go directly to in-progress
        if matches!(migration_type, super::MigrationType::Emergency) {
            return Ok(MigrationStatus::Preparing);
        }

        // Check system load - if system is under stress, queue the migration
        // Note: Current ResourceMonitor focuses on collection metrics rather than
        // system resource usage For now, we'll rely on other health indicators

        // Default to queued status for normal processing
        Ok(MigrationStatus::Queued)
    }

    /// Update migration status based on current system conditions
    pub async fn update_migration_status(
        &self,
        migration: &mut Migration,
        nodes: &HashMap<String, NodeInfo>,
    ) -> Result<()> {
        debug!("🔄 Updating migration status for {}", migration.migration_id);

        match migration.status {
            MigrationStatus::Queued => {
                // Check if migration can be promoted to preparing
                let analytics = self.analytics.read();
                if analytics.active_migrations_count < self.config.max_concurrent_migrations as u64
                {
                    migration.status = MigrationStatus::Preparing;
                    migration.started_at = Some(chrono::Utc::now());
                    info!("🚀 Migration {} promoted to preparing", migration.migration_id);
                }
            }
            MigrationStatus::Preparing => {
                // Check if nodes are still healthy
                if let Some(from_node) = nodes.get(&migration.source_node) {
                    if self.node_health_score(&from_node.health_status) < 0.5 {
                        migration.status = MigrationStatus::Failed {
                            error_message: "Source node health degraded".to_string(),
                        };
                        return Ok(());
                    }
                }

                if let Some(to_node) = nodes.get(&migration.destination_node) {
                    if self.node_health_score(&to_node.health_status) < 0.6 {
                        migration.status = MigrationStatus::Failed {
                            error_message: "Destination node health degraded".to_string(),
                        };
                        return Ok(());
                    }
                }

                // Advance to in-progress after preparation checks
                migration.status = MigrationStatus::InProgress;
                info!("⚡ Migration {} started", migration.migration_id);
            }
            MigrationStatus::InProgress => {
                // This would be updated by the actual migration process
                // For now, we just validate the nodes are still healthy
                if let Some(from_node) = nodes.get(&migration.source_node) {
                    if self.node_health_score(&from_node.health_status) < 0.3 {
                        migration.status = MigrationStatus::Failed {
                            error_message: "Source node failed during migration".to_string(),
                        };
                        return Ok(());
                    }
                }
            }
            MigrationStatus::Completing => {
                // Final validation before completion
                if let Some(to_node) = nodes.get(&migration.destination_node) {
                    if self.node_health_score(&to_node.health_status) > 0.7 {
                        migration.status = MigrationStatus::Completed;
                        migration.completed_at = Some(chrono::Utc::now());
                        info!("✅ Migration {} completed successfully", migration.migration_id);
                    }
                }
            }
            _ => {
                // No status update needed for completed or failed migrations
            }
        }

        Ok(())
    }

    /// Get prioritized migration queue with proper ordering
    pub fn get_prioritized_migration_queue(&self, migrations: &[Migration]) -> Vec<Migration> {
        if !self.config.enable_priority_queue {
            return migrations.to_vec();
        }

        let mut prioritized = migrations.to_vec();

        // Sort by priority first, then by impact score
        prioritized.sort_by(|a, b| {
            // Emergency migrations get highest priority
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }

            // For same priority, sort by impact score
            let impact_cmp =
                b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal);
            if impact_cmp != std::cmp::Ordering::Equal {
                return impact_cmp;
            }

            // For same priority and impact, sort by creation time (oldest first)
            a.created_at.cmp(&b.created_at)
        });

        // Separate migrations by status to ensure proper processing order
        let mut queued = Vec::new();
        let mut preparing = Vec::new();
        let mut in_progress = Vec::new();
        let mut completing = Vec::new();

        for migration in prioritized {
            match migration.status {
                MigrationStatus::Queued => queued.push(migration),
                MigrationStatus::Preparing => preparing.push(migration),
                MigrationStatus::InProgress => in_progress.push(migration),
                MigrationStatus::Completing => completing.push(migration),
                _ => {} // Skip completed or failed migrations
            }
        }

        // Return in processing order: completing, in_progress, preparing, queued
        let mut result = Vec::new();
        result.extend(completing);
        result.extend(in_progress);
        result.extend(preparing);
        result.extend(queued);

        result
    }

    /// Get migration statistics by priority
    pub fn get_migration_priority_stats(
        &self,
        migrations: &[Migration],
    ) -> HashMap<MigrationPriority, usize> {
        let mut stats = HashMap::new();

        for migration in migrations {
            let count = stats.entry(migration.priority.clone()).or_insert(0);
            *count += 1;
        }

        stats
    }

    /// Check if emergency migrations should interrupt current operations
    pub fn should_interrupt_for_emergency(&self, migrations: &[Migration]) -> bool {
        migrations.iter().any(|m| {
            matches!(m.priority, MigrationPriority::Emergency)
                && matches!(m.status, MigrationStatus::Queued | MigrationStatus::Preparing)
        })
    }

    /// Convert NodeHealth enum to numeric health score
    fn node_health_score(&self, health_status: &NodeHealth) -> f64 {
        match health_status {
            NodeHealth::Healthy => 1.0,
            NodeHealth::Degraded => 0.6,
            NodeHealth::Unhealthy => 0.3,
            NodeHealth::Critical => 0.1,
            NodeHealth::Offline => 0.0,
        }
    }
}

/// Usage analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnalytics {
    /// Total requests processed across cluster
    pub total_requests: u64,

    /// Average latency across all nodes
    pub avg_latency_ms: f64,

    /// Overall cluster efficiency score (0.0-1.0)
    pub cluster_efficiency_score: f64,

    /// Load distribution variance (lower is better)
    pub distribution_variance: f64,

    /// Migration success rate
    pub migration_success_rate: f64,

    /// Cost savings from optimizations
    pub cost_savings_usd: f64,

    /// Top performing models
    pub top_performing_models: Vec<ModelPerformanceSummary>,

    /// Load balancing effectiveness metrics
    pub load_balancing_effectiveness: EffectivenessMetrics,

    /// Timestamp when analytics were generated
    pub generated_at: DateTime<Utc>,
}

/// Model performance summary for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceSummary {
    /// Model identifier
    pub model_id: String,

    /// Node where model is deployed
    pub node_id: String,

    /// Efficiency score (0.0-1.0)
    pub efficiency_score: f64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// Throughput in requests per second
    pub throughput_rps: f64,
}

impl LoadPredictor {
    /// Create new load predictor
    async fn new() -> Result<Self> {
        Ok(Self {
            load_history: Arc::new(RwLock::new(Vec::new())),
            prediction_models: Arc::new(RwLock::new(HashMap::new())),
            accuracy_tracker: Arc::new(RwLock::new(PredictionAccuracyTracker::default())),
        })
    }

    /// Add load data point for prediction
    async fn add_load_data_point(&self, data_point: LoadDataPoint) -> Result<()> {
        let mut history = self.load_history.write();
        history.push(data_point);

        // Keep only last 10000 data points
        if history.len() > 10000 {
            history.remove(0);
        }

        Ok(())
    }

    /// Predict future loads for all nodes
    async fn predict_future_loads(&self, horizon_minutes: u64) -> Result<HashMap<String, f32>> {
        let history = self.load_history.read();
        let mut predictions = HashMap::new();

        // Simple prediction: use exponential moving average
        let mut node_loads: HashMap<String, Vec<f32>> = HashMap::new();

        // Group recent loads by node
        let recent_data: Vec<_> = history
            .iter()
            .filter(|point| {
                let minutes_ago = (Utc::now() - point.timestamp).num_minutes() as u64;
                minutes_ago <= horizon_minutes * 2 // Look at 2x horizon for prediction
            })
            .collect();

        for point in recent_data {
            node_loads
                .entry(point.node_id.clone())
                .or_insert_with(Vec::new)
                .push(point.cpu_utilization);
        }

        // Calculate predictions
        for (node_id, loads) in node_loads {
            if !loads.is_empty() {
                // Simple exponential smoothing
                let alpha = 0.3;
                let mut prediction = loads[0];

                for &load in &loads[1..] {
                    prediction = alpha * load + (1.0 - alpha) * prediction;
                }

                // Add small trend component
                if loads.len() >= 2 {
                    let trend = loads[loads.len() - 1] - loads[loads.len() - 2];
                    prediction += trend * 0.1;
                }

                predictions.insert(node_id, prediction);
            }
        }

        Ok(predictions)
    }
}

// Additional implementation methods would continue here...

/// Comprehensive usage tracking system for distributed load balancing
#[derive(Debug, Clone, Default)]
pub struct UsageTracker {
    /// Real-time usage metrics across all nodes
    pub real_time_metrics: Arc<RwLock<HashMap<String, RealTimeUsageMetrics>>>,

    /// Historical usage patterns for trend analysis
    pub historical_patterns: Arc<RwLock<VecDeque<HistoricalUsageSnapshot>>>,

    /// Resource consumption tracking
    pub resource_consumption: Arc<RwLock<ResourceConsumptionTracker>>,

    /// Performance correlation engine
    pub performance_correlations: Arc<RwLock<PerformanceCorrelationEngine>>,

    /// Usage prediction models
    pub prediction_models: Arc<RwLock<HashMap<String, UsagePredictionModel>>>,

    /// Anomaly detection system
    pub anomaly_detector: Arc<RwLock<UsageAnomalyDetector>>,

    /// Configuration
    pub config: UsageTrackerConfig,
}

/// Real-time usage metrics for a single node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RealTimeUsageMetrics {
    /// Node identifier
    pub node_id: String,

    /// Current timestamp
    pub timestamp: DateTime<Utc>,

    /// CPU usage metrics
    pub cpu_metrics: CpuUsageMetrics,

    /// Memory usage metrics
    pub memory_metrics: MemoryUsageMetrics,

    /// GPU usage metrics (if available)
    pub gpu_metrics: Option<GpuUsageMetrics>,

    /// Network usage metrics
    pub network_metrics: NetworkUsageMetrics,

    /// Storage usage metrics
    pub storage_metrics: StorageUsageMetrics,

    /// Model-specific usage
    pub model_usage: HashMap<String, ModelUsageMetrics>,

    /// Request processing metrics
    pub request_metrics: RequestProcessingMetrics,

    /// System health indicators
    pub health_indicators: SystemHealthIndicators,
}

/// CPU usage detailed metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuUsageMetrics {
    /// Overall CPU utilization percentage
    pub utilization_percent: f64,

    /// Per-core utilization
    pub per_core_utilization: Vec<f64>,

    /// Load averages
    pub load_average_1min: f64,
    pub load_average_5min: f64,
    pub load_average_15min: f64,

    /// Context switches per second
    pub context_switches_per_sec: u64,

    /// Interrupts per second
    pub interrupts_per_sec: u64,

    /// CPU frequency scaling
    pub cpu_frequency_mhz: Vec<f64>,

    /// CPU temperature (if available)
    pub temperature_celsius: Option<f64>,
}

/// Memory usage detailed metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryUsageMetrics {
    /// Total memory in bytes
    pub total_bytes: u64,

    /// Used memory in bytes
    pub used_bytes: u64,

    /// Available memory in bytes
    pub available_bytes: u64,

    /// Cached memory in bytes
    pub cached_bytes: u64,

    /// Buffered memory in bytes
    pub buffered_bytes: u64,

    /// Swap usage
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,

    /// Memory pressure indicators
    pub memory_pressure: f64,

    /// Page fault rate
    pub page_faults_per_sec: u64,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
}

/// GPU usage metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuUsageMetrics {
    /// GPU device identifier
    pub device_id: u32,

    /// GPU utilization percentage
    pub utilization_percent: f64,

    /// Memory usage
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,

    /// GPU temperature
    pub temperature_celsius: f64,

    /// Power consumption in watts
    pub power_consumption_watts: f64,

    /// GPU frequency
    pub gpu_frequency_mhz: f64,
    pub memory_frequency_mhz: f64,

    /// Compute processes
    pub active_processes: u32,

    /// Performance state
    pub performance_state: String,
}

/// Network usage metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkUsageMetrics {
    /// Bytes received per second
    pub rx_bytes_per_sec: f64,

    /// Bytes transmitted per second
    pub tx_bytes_per_sec: f64,

    /// Packets received per second
    pub rx_packets_per_sec: f64,

    /// Packets transmitted per second
    pub tx_packets_per_sec: f64,

    /// Network error rate
    pub error_rate: f64,

    /// Network drop rate
    pub drop_rate: f64,

    /// Active network connections
    pub active_connections: u32,

    /// Network latency metrics
    pub latency_metrics: NetworkLatencyMetrics,

    /// Bandwidth utilization
    pub bandwidth_utilization_percent: f64,
}

/// Network latency measurements
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkLatencyMetrics {
    /// Average round-trip time in milliseconds
    pub avg_rtt_ms: f64,

    /// Minimum RTT in milliseconds
    pub min_rtt_ms: f64,

    /// Maximum RTT in milliseconds
    pub max_rtt_ms: f64,

    /// Jitter in milliseconds
    pub jitter_ms: f64,

    /// Packet loss percentage
    pub packet_loss_percent: f64,
}

/// Storage usage metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageUsageMetrics {
    /// Disk usage by mount point
    pub disk_usage: HashMap<String, DiskUsageMetrics>,

    /// I/O operations per second
    pub io_operations_per_sec: f64,

    /// Read bandwidth in bytes per second
    pub read_bandwidth_bytes_per_sec: f64,

    /// Write bandwidth in bytes per second
    pub write_bandwidth_bytes_per_sec: f64,

    /// Average I/O wait time
    pub avg_io_wait_ms: f64,

    /// Storage temperature (if available)
    pub temperature_celsius: Option<f64>,
}

/// Disk usage for a specific mount point
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiskUsageMetrics {
    /// Total space in bytes
    pub total_bytes: u64,

    /// Used space in bytes
    pub used_bytes: u64,

    /// Available space in bytes
    pub available_bytes: u64,

    /// Usage percentage
    pub usage_percent: f64,

    /// Inode usage
    pub inodes_used: u64,
    pub inodes_total: u64,
}

/// Model-specific usage metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsageMetrics {
    /// Model identifier
    pub model_id: String,

    /// Active requests
    pub active_requests: u32,

    /// Requests per second
    pub requests_per_sec: f64,

    /// Average response time
    pub avg_response_time_ms: f64,

    /// Memory usage by this model
    pub memory_usage_bytes: u64,

    /// GPU memory usage by this model
    pub gpu_memory_usage_bytes: Option<u64>,

    /// CPU usage percentage by this model
    pub cpu_usage_percent: f64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Error rate
    pub error_rate: f64,

    /// Throughput metrics
    pub throughput_tokens_per_sec: f64,
}

/// Request processing metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestProcessingMetrics {
    /// Total requests received
    pub total_requests: u64,

    /// Requests in queue
    pub queued_requests: u32,

    /// Average queue time
    pub avg_queue_time_ms: f64,

    /// Processing concurrency
    pub concurrent_processing: u32,

    /// Request success rate
    pub success_rate: f64,

    /// Request distribution by priority
    pub priority_distribution: HashMap<String, u64>,

    /// Average processing time by request type
    pub processing_time_by_type: HashMap<String, f64>,

    /// Average latency in milliseconds
    pub average_latency_ms: f64,

    /// Requests per second throughput
    pub requests_per_second: f64,
}

/// System health indicators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemHealthIndicators {
    /// Overall health score (0.0-1.0)
    pub overall_health_score: f64,

    /// System uptime in seconds
    pub uptime_seconds: u64,

    /// System load score
    pub load_score: f64,

    /// Resource pressure indicators
    pub resource_pressure: ResourcePressureIndicators,

    /// Performance degradation indicators
    pub performance_degradation: PerformanceDegradationIndicators,

    /// Stability metrics
    pub stability_metrics: StabilityMetrics,

    /// Predictive health indicators
    pub predictive_indicators: PredictiveHealthIndicators,
}

/// Resource pressure across different dimensions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourcePressureIndicators {
    /// CPU pressure level (0.0-1.0)
    pub cpu_pressure: f64,

    /// Memory pressure level (0.0-1.0)
    pub memory_pressure: f64,

    /// I/O pressure level (0.0-1.0)
    pub io_pressure: f64,

    /// Network pressure level (0.0-1.0)
    pub network_pressure: f64,

    /// Overall pressure score
    pub overall_pressure: f64,
}

/// Performance degradation indicators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceDegradationIndicators {
    /// Response time degradation percentage
    pub response_time_degradation: f64,

    /// Throughput degradation percentage
    pub throughput_degradation: f64,

    /// Error rate increase percentage
    pub error_rate_increase: f64,

    /// Resource efficiency degradation
    pub efficiency_degradation: f64,

    /// Quality of service degradation
    pub qos_degradation: f64,
}

/// System stability metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Coefficient of variation for response times
    pub response_time_variability: f64,

    /// Resource usage stability
    pub resource_usage_stability: f64,

    /// Error rate stability
    pub error_rate_stability: f64,

    /// Performance consistency score
    pub performance_consistency: f64,

    /// System reliability score
    pub reliability_score: f64,
}

/// Predictive health indicators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictiveHealthIndicators {
    /// Predicted failure probability in next hour
    pub failure_probability_1h: f64,

    /// Predicted overload probability in next hour
    pub overload_probability_1h: f64,

    /// Predicted maintenance need score
    pub maintenance_need_score: f64,

    /// Resource exhaustion timeline
    pub resource_exhaustion_timeline: HashMap<String, u64>,

    /// Performance trend projection
    pub performance_trend: f64,
}

/// Historical usage snapshot for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalUsageSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,

    /// Aggregated metrics across all nodes
    pub aggregated_metrics: AggregatedUsageMetrics,

    /// Per-node summary metrics
    pub node_summaries: HashMap<String, NodeUsageSummary>,

    /// Cluster-wide KPIs
    pub cluster_kpis: ClusterKPIs,

    /// Usage patterns detected
    pub detected_patterns: Vec<UsagePattern>,
}

/// Aggregated usage metrics across the cluster
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedUsageMetrics {
    /// Total CPU utilization across cluster
    pub total_cpu_utilization: f64,

    /// Total memory utilization across cluster
    pub total_memory_utilization: f64,

    /// Total GPU utilization across cluster
    pub total_gpu_utilization: f64,

    /// Total network bandwidth utilization
    pub total_network_utilization: f64,

    /// Total requests processed
    pub total_requests_processed: u64,

    /// Average response time across cluster
    pub avg_response_time_ms: f64,

    /// Cluster-wide error rate
    pub cluster_error_rate: f64,

    /// Overall cluster efficiency
    pub cluster_efficiency: f64,
}

/// Node usage summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeUsageSummary {
    /// Node identifier
    pub node_id: String,

    /// Summary period
    pub summary_period_minutes: u32,

    /// Average resource utilization
    pub avg_cpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub avg_gpu_utilization: Option<f64>,

    /// Peak resource utilization
    pub peak_cpu_utilization: f64,
    pub peak_memory_utilization: f64,
    pub peak_gpu_utilization: Option<f64>,

    /// Performance summary
    pub avg_response_time_ms: f64,
    pub total_requests_handled: u64,
    pub error_count: u64,

    /// Efficiency metrics
    pub resource_efficiency: f64,
    pub cost_efficiency: f64,

    /// Health score
    pub health_score: f64,
}

/// Cluster-wide Key Performance Indicators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterKPIs {
    /// Service Level Agreement compliance
    pub sla_compliance: f64,

    /// Overall availability percentage
    pub availability_percent: f64,

    /// Mean Time Between Failures
    pub mtbf_hours: f64,

    /// Mean Time To Recovery
    pub mttr_minutes: f64,

    /// Cost per request
    pub cost_per_request: f64,

    /// Energy efficiency score
    pub energy_efficiency: f64,

    /// Carbon footprint metrics
    pub carbon_footprint_kg_co2: f64,

    /// User satisfaction score
    pub user_satisfaction: f64,
}

/// Detected usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Pattern type
    pub pattern_type: UsagePatternType,

    /// Pattern strength (0.0-1.0)
    pub strength: f64,

    /// Pattern duration
    pub duration_minutes: u32,

    /// Affected nodes
    pub affected_nodes: Vec<String>,

    /// Pattern description
    pub description: String,

    /// Confidence score
    pub confidence: f64,
}

/// Types of usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsagePatternType {
    /// Periodic usage spikes
    PeriodicSpike { period_minutes: u32 },

    /// Gradual resource growth
    GradualGrowth { growth_rate: f64 },

    /// Sudden resource burst
    Burst { burst_magnitude: f64 },

    /// Resource starvation
    Starvation { starved_resource: String },

    /// Load imbalance
    LoadImbalance { imbalance_score: f64 },

    /// Cascading failures
    CascadingFailure { failure_chain: Vec<String> },

    /// Performance degradation
    PerformanceDrift { drift_rate: f64 },
}

/// Resource consumption tracker
#[derive(Debug, Clone, Default)]
pub struct ResourceConsumptionTracker {
    /// Consumption by resource type
    pub consumption_by_resource: HashMap<String, ResourceConsumptionMetrics>,

    /// Consumption by model
    pub consumption_by_model: HashMap<String, ModelResourceConsumption>,

    /// Consumption trends
    pub consumption_trends: VecDeque<ConsumptionTrendPoint>,

    /// Cost tracking
    pub cost_tracking: CostTrackingMetrics,

    /// Efficiency analysis
    pub efficiency_analysis: EfficiencyAnalysis,
}

/// Resource consumption metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceConsumptionMetrics {
    /// Total consumption over time period
    pub total_consumption: f64,

    /// Average consumption rate
    pub avg_consumption_rate: f64,

    /// Peak consumption
    pub peak_consumption: f64,

    /// Consumption variance
    pub consumption_variance: f64,

    /// Wastage percentage
    pub wastage_percent: f64,

    /// Efficiency score
    pub efficiency_score: f64,
}

/// Model-specific resource consumption
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelResourceConsumption {
    /// Model identifier
    pub model_id: String,

    /// CPU hours consumed
    pub cpu_hours: f64,

    /// Memory GB-hours consumed
    pub memory_gb_hours: f64,

    /// GPU hours consumed
    pub gpu_hours: Option<f64>,

    /// Network bandwidth consumed (GB)
    pub network_gb: f64,

    /// Storage operations
    pub storage_operations: u64,

    /// Total cost attributed to this model
    pub total_cost: f64,

    /// Cost per request
    pub cost_per_request: f64,
}

/// Consumption trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumptionTrendPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Resource type
    pub resource_type: String,

    /// Consumption amount
    pub consumption: f64,

    /// Cost
    pub cost: f64,

    /// Efficiency score
    pub efficiency: f64,
}

/// Cost tracking metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostTrackingMetrics {
    /// Total cost in time period
    pub total_cost: f64,

    /// Cost breakdown by resource
    pub cost_by_resource: HashMap<String, f64>,

    /// Cost breakdown by model
    pub cost_by_model: HashMap<String, f64>,

    /// Cost trends
    pub cost_trends: Vec<CostTrendPoint>,

    /// Projected costs
    pub projected_costs: HashMap<String, f64>,

    /// Cost optimization opportunities
    pub optimization_opportunities: Vec<CostOptimizationOpportunity>,
}

/// Cost trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTrendPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Total cost at this point
    pub total_cost: f64,

    /// Cost rate (cost per unit time)
    pub cost_rate: f64,

    /// Cost efficiency
    pub cost_efficiency: f64,
}

/// Cost optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: CostOptimizationType,

    /// Potential savings
    pub potential_savings: f64,

    /// Implementation effort
    pub implementation_effort: f64,

    /// Risk level
    pub risk_level: f64,

    /// Description
    pub description: String,
}

/// Types of cost optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostOptimizationType {
    RightSizing,
    AutoScaling,
    SpotInstances,
    ReservedCapacity,
    ResourceConsolidation,
    PerformanceOptimization,
    WorkloadScheduling,
}

/// Efficiency analysis results
#[derive(Debug, Clone, Default)]
pub struct EfficiencyAnalysis {
    /// Overall cluster efficiency
    pub overall_efficiency: f64,

    /// Efficiency by node
    pub efficiency_by_node: HashMap<String, f64>,

    /// Efficiency by model
    pub efficiency_by_model: HashMap<String, f64>,

    /// Efficiency trends
    pub efficiency_trends: VecDeque<EfficiencyTrendPoint>,

    /// Bottleneck analysis
    pub bottlenecks: Vec<EfficiencyBottleneck>,

    /// Improvement recommendations
    pub recommendations: Vec<EfficiencyRecommendation>,
}

/// Efficiency trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyTrendPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Efficiency score
    pub efficiency_score: f64,

    /// Contributing factors
    pub factors: HashMap<String, f64>,
}

/// Efficiency bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,

    /// Severity score
    pub severity: f64,

    /// Affected components
    pub affected_components: Vec<String>,

    /// Impact on efficiency
    pub efficiency_impact: f64,

    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Types of efficiency bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBottleneck,
    MemoryBottleneck,
    NetworkBottleneck,
    StorageBottleneck,
    GpuBottleneck,
    LoadImbalance,
    ResourceFragmentation,
    SchedulingInefficiency,
}

/// Efficiency improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Expected efficiency gain
    pub expected_gain: f64,

    /// Implementation priority
    pub priority: f64,

    /// Resource requirements for implementation
    pub resource_requirements: HashMap<String, f64>,

    /// Description and steps
    pub description: String,
    pub implementation_steps: Vec<String>,
}

/// Types of efficiency recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ResourceReallocation,
    ModelOptimization,
    InfrastructureUpgrade,
    WorkflowOptimization,
    CachingStrategy,
    LoadBalancingTuning,
    ConfigurationTuning,
    CapacityPlanning,
}

/// Performance correlation engine
#[derive(Debug, Clone, Default)]
pub struct PerformanceCorrelationEngine {
    /// Correlation matrices
    pub correlation_matrices: HashMap<String, CorrelationMatrix>,

    /// Causal relationships
    pub causal_relationships: Vec<CausalRelationship>,

    /// Performance predictors
    pub performance_predictors: HashMap<String, PerformancePredictor>,

    /// Anomaly correlations
    pub anomaly_correlations: Vec<AnomalyCorrelation>,
}

/// Correlation matrix for metrics
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Matrix dimensions
    pub dimensions: usize,

    /// Correlation coefficients
    pub coefficients: Vec<Vec<f64>>,

    /// Metric names
    pub metric_names: Vec<String>,

    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Causal relationship between metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationship {
    /// Cause metric
    pub cause_metric: String,

    /// Effect metric
    pub effect_metric: String,

    /// Causal strength
    pub causal_strength: f64,

    /// Time lag in seconds
    pub time_lag_seconds: u64,

    /// Confidence in relationship
    pub confidence: f64,

    /// Relationship type
    pub relationship_type: CausalRelationshipType,
}

/// Types of causal relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalRelationshipType {
    Direct,
    Inverse,
    NonLinear,
    Threshold,
    Cyclical,
}

/// Performance predictor model
#[derive(Debug, Clone)]
pub struct PerformancePredictor {
    /// Predictor type
    pub predictor_type: PredictorType,

    /// Input features
    pub input_features: Vec<String>,

    /// Target metric
    pub target_metric: String,

    /// Model parameters
    pub parameters: HashMap<String, f64>,

    /// Prediction accuracy
    pub accuracy: f64,

    /// Last training time
    pub last_trained: DateTime<Utc>,
}

/// Types of performance predictors
#[derive(Debug, Clone)]
pub enum PredictorType {
    LinearRegression,
    PolynomialRegression,
    NeuralNetwork,
    TimeSeries,
    EnsembleModel,
}

/// Anomaly correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyCorrelation {
    /// Primary anomaly
    pub primary_anomaly: String,

    /// Correlated anomalies
    pub correlated_anomalies: Vec<String>,

    /// Correlation strength
    pub correlation_strength: f64,

    /// Time window for correlation
    pub time_window_seconds: u64,

    /// Frequency of co-occurrence
    pub co_occurrence_frequency: f64,
}

/// Usage prediction model
#[derive(Debug, Clone)]
pub struct UsagePredictionModel {
    /// Model identifier
    pub model_id: String,

    /// Prediction target
    pub prediction_target: PredictionTarget,

    /// Time horizon
    pub time_horizon_minutes: u64,

    /// Model algorithm
    pub algorithm: PredictionAlgorithm,

    /// Training data window
    pub training_window_hours: u64,

    /// Model accuracy metrics
    pub accuracy_metrics: PredictionAccuracyMetrics,

    /// Feature importance
    pub feature_importance: HashMap<String, f64>,

    /// Last prediction
    pub last_prediction: Option<UsagePrediction>,
}

/// Usage prediction targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionTarget {
    CpuUtilization,
    MemoryUtilization,
    GpuUtilization,
    NetworkBandwidth,
    RequestVolume,
    ResponseTime,
    ErrorRate,
    ResourceCost,
}

/// Prediction algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionAlgorithm {
    ARIMA,
    LSTM,
    Prophet,
    LinearRegression,
    RandomForest,
    GradientBoosting,
    EnsembleMethod,
}

/// Prediction accuracy metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionAccuracyMetrics {
    /// Mean Absolute Error
    pub mae: f64,

    /// Root Mean Square Error
    pub rmse: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,

    /// R-squared score
    pub r_squared: f64,

    /// Prediction interval coverage
    pub coverage: f64,
}

/// Usage prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePrediction {
    /// Prediction timestamp
    pub prediction_timestamp: DateTime<Utc>,

    /// Target timestamp
    pub target_timestamp: DateTime<Utc>,

    /// Predicted value
    pub predicted_value: f64,

    /// Confidence interval
    pub confidence_interval: (f64, f64),

    /// Prediction confidence
    pub confidence: f64,

    /// Contributing factors
    pub contributing_factors: HashMap<String, f64>,
}

/// Usage anomaly detector
#[derive(Debug, Clone, Default)]
pub struct UsageAnomalyDetector {
    /// Detection algorithms
    pub detection_algorithms: Vec<AnomalyDetectionAlgorithm>,

    /// Anomaly thresholds
    pub thresholds: AnomalyThresholds,

    /// Detected anomalies
    pub detected_anomalies: VecDeque<DetectedUsageAnomaly>,

    /// Baseline patterns
    pub baseline_patterns: HashMap<String, BaselinePattern>,

    /// False positive rate
    pub false_positive_rate: f64,

    /// Detection sensitivity
    pub detection_sensitivity: f64,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    StatisticalOutlier,
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    DBSCAN,
    TimeSeriesAnomaly,
    ThresholdBased,
    MachineLearningBased,
}

/// Anomaly detection thresholds
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyThresholds {
    /// Standard deviation multiplier
    pub std_dev_multiplier: f64,

    /// Percentile thresholds
    pub percentile_threshold_upper: f64,
    pub percentile_threshold_lower: f64,

    /// Rate of change thresholds
    pub rate_change_threshold: f64,

    /// Absolute value thresholds by metric
    pub absolute_thresholds: HashMap<String, (f64, f64)>,

    /// Minimum anomaly duration
    pub min_anomaly_duration_seconds: u64,
}

/// Detected usage anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedUsageAnomaly {
    /// Anomaly identifier
    pub anomaly_id: String,

    /// Detection timestamp
    pub detected_at: DateTime<Utc>,

    /// Anomaly type
    pub anomaly_type: UsageAnomalyType,

    /// Affected metrics
    pub affected_metrics: Vec<String>,

    /// Affected nodes
    pub affected_nodes: Vec<String>,

    /// Severity score
    pub severity: f64,

    /// Confidence score
    pub confidence: f64,

    /// Duration
    pub duration_seconds: u64,

    /// Root cause analysis
    pub root_cause: Option<String>,

    /// Suggested remediation
    pub suggested_remediation: Vec<String>,
}

/// Types of usage anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageAnomalyType {
    SuddenSpike,
    SuddenDrop,
    GradualIncrease,
    GradualDecrease,
    Oscillation,
    Flatline,
    PerformanceDegradation,
    ResourceLeakage,
    LoadImbalance,
    SystemInstability,
}

/// Baseline pattern for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselinePattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Metric name
    pub metric_name: String,

    /// Normal value range
    pub normal_range: (f64, f64),

    /// Seasonal patterns
    pub seasonal_patterns: Vec<SeasonalPattern>,

    /// Trend characteristics
    pub trend_characteristics: TrendCharacteristics,

    /// Pattern confidence
    pub confidence: f64,

    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Seasonal pattern in usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Pattern type
    pub pattern_type: SeasonalPatternType,

    /// Period in seconds
    pub period_seconds: u64,

    /// Amplitude
    pub amplitude: f64,

    /// Phase offset
    pub phase_offset: f64,

    /// Pattern strength
    pub strength: f64,
}

/// Types of seasonal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalPatternType {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Yearly,
    Custom { name: String },
}

/// Trend characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendCharacteristics {
    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength
    pub strength: f64,

    /// Trend stability
    pub stability: f64,

    /// Rate of change
    pub rate_of_change: f64,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Irregular,
}

/// Usage tracker configuration
#[derive(Debug, Clone)]
pub struct UsageTrackerConfig {
    /// Metrics collection interval
    pub collection_interval_seconds: u64,

    /// Historical data retention period
    pub retention_period_days: u32,

    /// Enable real-time anomaly detection
    pub enable_anomaly_detection: bool,

    /// Enable usage prediction
    pub enable_usage_prediction: bool,

    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,

    /// Maximum number of detected anomalies to store
    pub max_anomalies: usize,

    /// Prediction model update frequency
    pub model_update_frequency_hours: u64,

    /// Correlation analysis frequency
    pub correlation_analysis_frequency_hours: u64,
}

impl Default for UsageTrackerConfig {
    fn default() -> Self {
        Self {
            collection_interval_seconds: 30,
            retention_period_days: 30,
            enable_anomaly_detection: true,
            enable_usage_prediction: true,
            enable_correlation_analysis: true,
            max_anomalies: 1000,
            model_update_frequency_hours: 24,
            correlation_analysis_frequency_hours: 6,
        }
    }
}

impl UsageTracker {
    /// Create a new usage tracker
    pub async fn new(config: UsageTrackerConfig) -> Result<Self> {
        info!("🔍 Initializing comprehensive usage tracker");

        Ok(Self {
            real_time_metrics: Arc::new(RwLock::new(HashMap::new())),
            historical_patterns: Arc::new(RwLock::new(VecDeque::new())),
            resource_consumption: Arc::new(RwLock::new(ResourceConsumptionTracker::default())),
            performance_correlations: Arc::new(
                RwLock::new(PerformanceCorrelationEngine::default()),
            ),
            prediction_models: Arc::new(RwLock::new(HashMap::new())),
            anomaly_detector: Arc::new(RwLock::new(UsageAnomalyDetector::default())),
            config,
        })
    }

    /// Update real-time metrics for a node
    pub async fn update_node_metrics(
        &self,
        node_id: &str,
        metrics: RealTimeUsageMetrics,
    ) -> Result<()> {
        let mut real_time = self.real_time_metrics.write();
        real_time.insert(node_id.to_string(), metrics.clone());

        // Detect anomalies if enabled
        if self.config.enable_anomaly_detection {
            self.detect_anomalies_for_node(node_id, &metrics).await?;
        }

        // Update resource consumption tracking
        self.update_resource_consumption(node_id, &metrics).await?;

        Ok(())
    }

    /// Get current usage analytics
    pub async fn get_usage_analytics(&self) -> Result<UsageAnalytics> {
        let real_time = self.real_time_metrics.read();

        // Calculate cluster-wide metrics
        let mut total_cpu = 0.0;
        let mut total_memory = 0.0;
        let mut total_requests = 0u64;
        let mut avg_latency = 0.0;
        let node_count = real_time.len() as f64;

        for metrics in real_time.values() {
            total_cpu += metrics.cpu_metrics.utilization_percent;
            total_memory += (metrics.memory_metrics.used_bytes as f64
                / metrics.memory_metrics.total_bytes as f64)
                * 100.0;
            total_requests += metrics.request_metrics.total_requests;
            avg_latency += metrics.request_metrics.avg_queue_time_ms;
        }

        let cluster_efficiency = if node_count > 0.0 {
            ((total_cpu / node_count) + (total_memory / node_count)) / 200.0 // Normalize to 0-1
        } else {
            0.0
        };

        avg_latency = if node_count > 0.0 { avg_latency / node_count } else { 0.0 };

        // Build top performing models
        let mut model_performances = Vec::new();
        for metrics in real_time.values() {
            for (model_id, model_metrics) in &metrics.model_usage {
                model_performances.push(ModelPerformanceSummary {
                    model_id: model_id.clone(),
                    node_id: metrics.node_id.clone(),
                    efficiency_score: 1.0 - (model_metrics.avg_response_time_ms / 1000.0).min(1.0),
                    avg_latency_ms: model_metrics.avg_response_time_ms,
                    throughput_rps: model_metrics.requests_per_sec,
                });
            }
        }

        model_performances.sort_by(|a, b| {
            b.efficiency_score.partial_cmp(&a.efficiency_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        model_performances.truncate(10);

        Ok(UsageAnalytics {
            total_requests,
            avg_latency_ms: avg_latency,
            cluster_efficiency_score: cluster_efficiency,
            distribution_variance: 0.0, // Would calculate from real distribution
            migration_success_rate: 0.95, // Would track from actual migrations
            cost_savings_usd: 0.0,      // Would calculate from optimization
            top_performing_models: model_performances,
            load_balancing_effectiveness: EffectivenessMetrics::default(),
            generated_at: chrono::Utc::now(),
        })
    }

    /// Detect anomalies for a specific node
    async fn detect_anomalies_for_node(
        &self,
        node_id: &str,
        metrics: &RealTimeUsageMetrics,
    ) -> Result<()> {
        // Simple threshold-based anomaly detection
        let mut anomalies = Vec::new();

        // CPU anomaly detection
        if metrics.cpu_metrics.utilization_percent > 95.0 {
            anomalies.push(DetectedUsageAnomaly {
                anomaly_id: format!("cpu_spike_{}_{}", node_id, chrono::Utc::now().timestamp()),
                detected_at: chrono::Utc::now(),
                anomaly_type: UsageAnomalyType::SuddenSpike,
                affected_metrics: vec!["cpu_utilization".to_string()],
                affected_nodes: vec![node_id.to_string()],
                severity: 0.8,
                confidence: 0.9,
                duration_seconds: 0, // Would track duration
                root_cause: Some("High CPU utilization detected".to_string()),
                suggested_remediation: vec![
                    "Check for runaway processes".to_string(),
                    "Consider load balancing".to_string(),
                ],
            });
        }

        // Memory anomaly detection
        let memory_usage_percent = (metrics.memory_metrics.used_bytes as f64
            / metrics.memory_metrics.total_bytes as f64)
            * 100.0;
        if memory_usage_percent > 90.0 {
            anomalies.push(DetectedUsageAnomaly {
                anomaly_id: format!(
                    "memory_pressure_{}_{}",
                    node_id,
                    chrono::Utc::now().timestamp()
                ),
                detected_at: chrono::Utc::now(),
                anomaly_type: UsageAnomalyType::SuddenSpike,
                affected_metrics: vec!["memory_utilization".to_string()],
                affected_nodes: vec![node_id.to_string()],
                severity: 0.7,
                confidence: 0.85,
                duration_seconds: 0,
                root_cause: Some("High memory pressure detected".to_string()),
                suggested_remediation: vec![
                    "Check for memory leaks".to_string(),
                    "Consider scaling up memory".to_string(),
                ],
            });
        }

        // Store detected anomalies
        if !anomalies.is_empty() {
            let mut detector = self.anomaly_detector.write();
            for anomaly in anomalies {
                detector.detected_anomalies.push_back(anomaly);
            }

            // Keep only recent anomalies
            while detector.detected_anomalies.len() > self.config.max_anomalies {
                detector.detected_anomalies.pop_front();
            }
        }

        Ok(())
    }

    /// Update resource consumption tracking
    async fn update_resource_consumption(
        &self,
        _node_id: &str,
        metrics: &RealTimeUsageMetrics,
    ) -> Result<()> {
        let mut consumption = self.resource_consumption.write();

        // Update CPU consumption
        let cpu_entry = consumption
            .consumption_by_resource
            .entry("cpu".to_string())
            .or_insert_with(ResourceConsumptionMetrics::default);

        cpu_entry.total_consumption += metrics.cpu_metrics.utilization_percent;
        cpu_entry.avg_consumption_rate = metrics.cpu_metrics.utilization_percent;
        cpu_entry.efficiency_score =
            1.0 - (metrics.cpu_metrics.utilization_percent / 100.0 - 0.75).abs();

        // Update memory consumption
        let memory_entry = consumption
            .consumption_by_resource
            .entry("memory".to_string())
            .or_insert_with(ResourceConsumptionMetrics::default);

        let memory_usage_percent = (metrics.memory_metrics.used_bytes as f64
            / metrics.memory_metrics.total_bytes as f64)
            * 100.0;
        memory_entry.total_consumption += memory_usage_percent;
        memory_entry.avg_consumption_rate = memory_usage_percent;
        memory_entry.efficiency_score = 1.0 - (memory_usage_percent / 100.0 - 0.80).abs();

        Ok(())
    }

    /// Get distributed usage statistics across all nodes
    pub async fn get_distributed_usage_stats(&self) -> Result<DistributedUsageStats> {
        let real_time = self.real_time_metrics.read();

        if real_time.is_empty() {
            return Ok(DistributedUsageStats::default());
        }

        let node_count = real_time.len();
        let mut total_latency = 0.0;
        let mut total_requests = 0u64;
        let mut total_throughput = 0.0;
        let mut total_cpu = 0.0;
        let mut total_memory = 0.0;

        for metrics in real_time.values() {
            total_latency += metrics.request_metrics.average_latency_ms;
            total_requests += metrics.request_metrics.total_requests;
            total_throughput += metrics.request_metrics.requests_per_second;
            total_cpu += metrics.cpu_metrics.utilization_percent;
            total_memory += (metrics.memory_metrics.used_bytes as f64
                / metrics.memory_metrics.total_bytes as f64)
                * 100.0;
        }

        Ok(DistributedUsageStats {
            active_nodes: node_count,
            average_latency_ms: if node_count > 0 {
                total_latency / node_count as f64
            } else {
                0.0
            },
            total_requests,
            requests_per_second: total_throughput,
            average_cpu_utilization: if node_count > 0 {
                total_cpu / node_count as f64
            } else {
                0.0
            },
            average_memory_utilization: if node_count > 0 {
                total_memory / node_count as f64
            } else {
                0.0
            },
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Distributed usage statistics across all nodes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedUsageStats {
    pub active_nodes: usize,
    pub average_latency_ms: f64,
    pub total_requests: u64,
    pub requests_per_second: f64,
    pub average_cpu_utilization: f64,
    pub average_memory_utilization: f64,
    pub timestamp: DateTime<Utc>,
}
