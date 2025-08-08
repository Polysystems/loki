use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{TaskRequest, TaskResponse, TaskType};

/// Comprehensive cost tracking and budget management system
#[derive(Debug)]
pub struct CostManager {
    /// Current budget configuration
    budgetconfig: Arc<RwLock<BudgetConfig>>,

    /// Cost tracking for different components
    cost_tracker: Arc<RwLock<CostTracker>>,

    /// Provider-specific pricing information
    pricing_info: Arc<RwLock<PricingDatabase>>,

    /// Usage analytics and optimization
    usage_analyzer: Arc<UsageAnalyzer>,

    /// Budget alerts and enforcement
    budget_enforcer: Arc<BudgetEnforcer>,

    /// Cost optimization recommendations
    optimizer: Arc<CostOptimizer>,
}

/// Budget configuration and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Daily budget limit in cents
    pub daily_limit_cents: f32,

    /// Weekly budget limit in cents
    pub weekly_limit_cents: f32,

    /// Monthly budget limit in cents
    pub monthly_limit_cents: f32,

    /// Emergency stop threshold (% of monthly budget)
    pub emergency_threshold: f32,

    /// Warning threshold (% of monthly budget)
    pub warning_threshold: f32,

    /// Model-specific budget allocations
    pub model_budgets: HashMap<String, ModelBudget>,

    /// Task-type specific budgets
    pub task_budgets: HashMap<String, f32>,

    /// Consciousness processing budget
    pub consciousness_budget_cents: f32,

    /// Streaming budget allocation
    pub streaming_budget_cents: f32,

    /// Budget period (daily, weekly, monthly)
    pub budget_period: BudgetPeriod,

    /// Auto-optimization enabled
    pub auto_optimization: bool,

    /// Cost tracking granularity
    pub tracking_granularity: TrackingGranularity,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            daily_limit_cents: 500.0,     // $5.00 per day
            weekly_limit_cents: 3000.0,   // $30.00 per week
            monthly_limit_cents: 10000.0, // $100.00 per month
            emergency_threshold: 90.0,    // Stop at 90% of monthly
            warning_threshold: 75.0,      // Warn at 75% of monthly
            model_budgets: HashMap::new(),
            task_budgets: HashMap::new(),
            consciousness_budget_cents: 2000.0, // $20.00 for consciousness
            streaming_budget_cents: 1500.0,     // $15.00 for streaming
            budget_period: BudgetPeriod::Monthly,
            auto_optimization: true,
            tracking_granularity: TrackingGranularity::Detailed,
        }
    }
}

/// Model-specific budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBudget {
    pub model_id: String,
    pub daily_limit_cents: f32,
    pub monthly_limit_cents: f32,
    pub priority: BudgetPriority,
    pub auto_scaling: bool,
    pub cost_per_token: Option<f32>,
}

/// Budget time periods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BudgetPeriod {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
}

/// Budget priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum BudgetPriority {
    Essential = 1, // Critical consciousness functions
    High = 2,      // Important decision making
    Medium = 3,    // General processing
    Low = 4,       // Non-critical tasks
    Optional = 5,  // Nice-to-have features
}

/// Cost tracking granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrackingGranularity {
    Basic,    // Track total costs only
    Moderate, // Track by model and task type
    Detailed, // Track every request with metadata
    Verbose,  // Track with full context and analytics
}

/// Comprehensive cost tracking
#[derive(Debug, Default)]
pub struct CostTracker {
    /// Total costs by time period
    pub total_costs: CostBreakdown,

    /// Model-specific costs
    pub model_costs: HashMap<String, CostBreakdown>,

    /// Task-type specific costs
    pub task_costs: HashMap<String, CostBreakdown>,

    /// Consciousness-specific costs
    pub consciousness_costs: CostBreakdown,

    /// Streaming costs
    pub streaming_costs: CostBreakdown,

    /// Individual request tracking
    pub request_history: Vec<CostEntry>,

    /// Cost trends and analytics
    pub cost_trends: CostTrends,

    /// Budget utilization
    pub budget_utilization: BudgetUtilization,
}

/// Cost breakdown by time periods
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub hourly_cents: f32,
    pub daily_cents: f32,
    pub weekly_cents: f32,
    pub monthly_cents: f32,
    pub total_cents: f32,
    pub request_count: u64,
    pub token_count: u64,
    pub average_cost_per_request: f32,
    pub average_cost_per_token: f32,
}

/// Individual cost entry for detailed tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    pub timestamp: SystemTime,
    pub model_id: String,
    pub task_type: String,
    pub cost_cents: f32,
    pub tokens_used: u32,
    pub execution_time_ms: u32,
    pub quality_score: f32,
    pub context: String,
    pub consciousness_enhanced: bool,
    pub streaming_enabled: bool,
    pub request_id: String,
}

/// Cost trends and analytics
#[derive(Debug, Default, Clone)]
pub struct CostTrends {
    pub daily_trend: TrendDirection,
    pub weekly_trend: TrendDirection,
    pub cost_efficiency_trend: TrendDirection,
    pub token_efficiency_trend: TrendDirection,
    pub projected_monthly_cost: f32,
    pub cost_per_quality_point: f32,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Trend direction indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

impl Default for TrendDirection {
    fn default() -> Self {
        TrendDirection::Stable
    }
}

/// Budget utilization tracking
#[derive(Debug, Default)]
pub struct BudgetUtilization {
    pub daily_utilization: f32, // 0.0 to 1.0
    pub weekly_utilization: f32,
    pub monthly_utilization: f32,
    pub consciousness_utilization: f32,
    pub streaming_utilization: f32,
    pub time_remaining_in_period: Duration,
    pub projected_period_usage: f32,
    pub budget_health: BudgetHealth,
}

/// Budget health indicators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BudgetHealth {
    Excellent,
    Good,
    Caution,
    Warning,
    Critical,
}

impl Default for BudgetHealth {
    fn default() -> Self {
        BudgetHealth::Good
    }
}

/// Pricing database for different providers and models
#[derive(Debug)]
pub struct PricingDatabase {
    /// Provider pricing information
    pub provider_pricing: HashMap<String, ProviderPricing>,

    /// Model-specific pricing overrides
    pub model_pricing: HashMap<String, ModelPricing>,

    /// Task-type pricing modifiers
    pub task_modifiers: HashMap<String, f32>,

    /// Dynamic pricing updates
    pub pricing_updates: Vec<PricingUpdate>,

    /// Last pricing update
    pub last_update: SystemTime,
}

/// Provider-specific pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPricing {
    pub provider_name: String,
    pub input_cost_per_1k_tokens: f32,
    pub output_cost_per_1k_tokens: f32,
    pub streaming_multiplier: f32,
    pub context_window_pricing: Vec<ContextPricing>,
    pub minimum_charge_cents: f32,
    pub bulk_discount_tiers: Vec<DiscountTier>,
}

/// Model-specific pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub model_id: String,
    pub input_cost_per_1k_tokens: f32,
    pub output_cost_per_1k_tokens: f32,
    pub quality_multiplier: f32,
    pub consciousness_surcharge: f32,
    pub effective_date: SystemTime,
}

/// Context window pricing tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPricing {
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub multiplier: f32,
}

/// Discount tiers for bulk usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscountTier {
    pub min_monthly_spend_cents: f32,
    pub discount_percentage: f32,
}

/// Pricing update events
#[derive(Debug, Clone)]
pub struct PricingUpdate {
    pub timestamp: SystemTime,
    pub provider: String,
    pub model: Option<String>,
    pub update_type: PricingUpdateType,
    pub old_price: f32,
    pub new_price: f32,
}

/// Types of pricing updates
#[derive(Debug, Clone)]
pub enum PricingUpdateType {
    InputTokenPrice,
    OutputTokenPrice,
    StreamingMultiplier,
    QualityMultiplier,
    NewModel,
    ModelDeprecated,
}

/// Usage analysis and optimization
#[derive(Debug)]
pub struct UsageAnalyzer {
    /// Usage patterns by time
    usage_patterns: Arc<RwLock<UsagePatterns>>,

    /// Efficiency metrics
    efficiency_metrics: Arc<RwLock<EfficiencyMetrics>>,

    /// Cost anomaly detection
    anomaly_detector: AnomalyDetector,
}

/// Usage patterns analysis
#[derive(Debug, Clone)]
pub struct UsagePatterns {
    pub peak_hours: Vec<u8>,
    pub usage_frequency: f32,
    pub request_size_trend: TrendDirection,
    pub model_preference_shift: HashMap<String, f32>,
    pub seasonal_patterns: Vec<String>,
}

impl Default for UsagePatterns {
    fn default() -> Self {
        Self {
            peak_hours: vec![9, 10, 11, 14, 15, 16],
            usage_frequency: 1.0,
            request_size_trend: TrendDirection::Stable,
            model_preference_shift: HashMap::new(),
            seasonal_patterns: Vec::new(),
        }
    }
}

/// Efficiency metrics
#[derive(Debug, Default)]
pub struct EfficiencyMetrics {
    pub cost_per_quality_point: f32,
    pub tokens_per_dollar: f32,
    pub response_quality_trend: f32,
    pub cost_trend: f32,
    pub efficiency_score: f32,
    pub optimization_potential: f32,
    pub roi_metrics: ROIMetrics,
}

/// Return on investment metrics
#[derive(Debug, Default)]
pub struct ROIMetrics {
    pub consciousness_roi: f32,
    pub streaming_roi: f32,
    pub ensemble_roi: f32,
    pub learning_roi: f32,
    pub overall_roi: f32,
}

/// Anomaly detection for cost patterns
#[derive(Debug)]
pub struct AnomalyDetector {
    baseline_patterns: HashMap<String, f32>,
    anomaly_threshold: f32,
    recent_anomalies: Vec<CostAnomaly>,
}

/// Cost anomaly events
#[derive(Debug, Clone)]
pub struct CostAnomaly {
    pub timestamp: SystemTime,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub cost_impact: f32,
    pub suggested_action: String,
}

/// Types of cost anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    SuddenCostSpike,
    UnusualModelUsage,
    QualityDropWithHighCost,
    BudgetBreach,
    EfficiencyDegradation,
    UnexpectedStreamingCosts,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Budget enforcement and alerts
#[derive(Debug)]
pub struct BudgetEnforcer {
    /// Alert thresholds
    alert_thresholds: Vec<AlertThreshold>,

    /// Active alerts
    active_alerts: Arc<RwLock<Vec<BudgetAlert>>>,

    /// Enforcement actions
    enforcement_actions: Vec<EnforcementAction>,
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThreshold {
    pub percentage: f32,
    pub alert_type: AlertType,
    pub action: AlertAction,
    pub cooldown_minutes: u32,
}

/// Types of budget alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    DailyBudgetWarning,
    WeeklyBudgetWarning,
    MonthlyBudgetWarning,
    ModelBudgetExceeded,
    ConsciousnessBudgetWarning,
    StreamingBudgetWarning,
    EmergencyStop,
}

/// Alert actions to take
#[derive(Debug, Clone)]
pub enum AlertAction {
    LogWarning,
    NotifyUser,
    ReduceQuality,
    DisableStreaming,
    SwitchToLocalModels,
    EmergencyStop,
    OptimizeRouting,
}

/// Budget alert instances
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    pub id: String,
    pub timestamp: SystemTime,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub current_usage: f32,
    pub budget_limit: f32,
    pub projected_overage: Option<f32>,
    pub recommended_actions: Vec<String>,
    pub acknowledged: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Budget enforcement actions
#[derive(Debug, Clone)]
pub struct EnforcementAction {
    pub trigger_percentage: f32,
    pub action_type: EnforcementActionType,
    pub duration_minutes: Option<u32>,
    pub priority_override: bool,
}

/// Types of enforcement actions
#[derive(Debug, Clone)]
pub enum EnforcementActionType {
    ThrottleRequests,
    ReduceQualityThreshold,
    DisableNonEssentialFeatures,
    ForceLocalModels,
    DisableEnsemble,
    DisableStreaming,
    HardStop,
}

/// Cost optimization engine
#[derive(Debug)]
pub struct CostOptimizer {
    /// Optimization strategies
    strategies: Vec<OptimizationStrategy>,

    /// Current optimizations
    active_optimizations: Arc<RwLock<Vec<ActiveOptimization>>>,

    /// Optimization history
    optimization_history: Vec<OptimizationResult>,
}

/// Optimization opportunities
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OptimizationType,
    pub potential_savings_cents: f32,
    pub confidence: f32,
    pub implementation_effort: ImplementationEffort,
    pub description: String,
    pub recommended_action: String,
    pub estimated_quality_impact: f32,
}

/// Types of cost optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    ModelSelection,
    QualityThresholdAdjustment,
    StreamingOptimization,
    ConsciousnessOptimization,
    BatchingRequests,
    CachingResponses,
    LoadBalancing,
    ProviderSwitching,
}

/// Implementation effort levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImplementationEffort {
    Automatic,
    Low,
    Medium,
    High,
}

/// Optimization strategies
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub name: String,
    pub strategy_type: OptimizationType,
    pub target_savings_percentage: f32,
    pub quality_impact_tolerance: f32,
    pub implementation_priority: u8,
    pub conditions: Vec<OptimizationCondition>,
}

/// Conditions for optimization activation
#[derive(Debug, Clone)]
pub struct OptimizationCondition {
    pub condition_type: ConditionType,
    pub threshold: f32,
    pub comparison: ComparisonOperator,
}

/// Types of optimization conditions
#[derive(Debug, Clone)]
pub enum ConditionType {
    BudgetUtilization,
    CostTrend,
    QualityScore,
    EfficiencyRatio,
    TimeRemaining,
}

/// Comparison operators for conditions
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterEqual,
    LessEqual,
}

/// Active optimization instances
#[derive(Debug, Clone)]
pub struct ActiveOptimization {
    pub id: String,
    pub strategy: OptimizationStrategy,
    pub started_at: SystemTime,
    pub estimated_completion: SystemTime,
    pub current_savings: f32,
    pub quality_impact: f32,
    pub status: OptimizationStatus,
}

/// Optimization status
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStatus {
    Planning,
    Active,
    Monitoring,
    Completed,
    Failed,
    Reverted,
}

/// Optimization results tracking
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimization_id: String,
    pub strategy_type: OptimizationType,
    pub duration: Duration,
    pub actual_savings: f32,
    pub predicted_savings: f32,
    pub quality_impact: f32,
    pub success_rate: f32,
    pub lessons_learned: Vec<String>,
}

impl CostManager {
    /// Create new cost manager
    pub async fn new(budgetconfig: BudgetConfig) -> Result<Self> {
        info!("💰 Initializing Cost Manager with budget tracking");

        let pricing_info = Arc::new(RwLock::new(Self::initialize_pricing_database().await?));
        let usage_analyzer = Arc::new(UsageAnalyzer::new());
        let budget_enforcer = Arc::new(BudgetEnforcer::new(&budgetconfig));
        let optimizer = Arc::new(CostOptimizer::new());

        Ok(Self {
            budgetconfig: Arc::new(RwLock::new(budgetconfig)),
            cost_tracker: Arc::new(RwLock::new(CostTracker::default())),
            pricing_info,
            usage_analyzer,
            budget_enforcer,
            optimizer,
        })
    }

    /// Record cost for a task execution
    pub async fn record_cost(
        &self,
        task: &TaskRequest,
        response: &TaskResponse,
        execution_time: Duration,
    ) -> Result<()> {
        let cost_entry = self.calculate_cost_entry(task, response, execution_time).await?;

        debug!(
            "💰 Recording cost: {:.3} cents for {} with {}",
            cost_entry.cost_cents, cost_entry.task_type, cost_entry.model_id
        );

        // Update cost tracking
        {
            let mut tracker = self.cost_tracker.write().await;
            self.update_cost_breakdown(&mut tracker, &cost_entry).await;

            // Store detailed entry if granularity is detailed or verbose
            let config = self.budgetconfig.read().await;
            if matches!(
                config.tracking_granularity,
                TrackingGranularity::Detailed | TrackingGranularity::Verbose
            ) {
                tracker.request_history.push(cost_entry.clone());

                // Limit history size to prevent memory bloat
                if tracker.request_history.len() > 10000 {
                    tracker.request_history.drain(0..1000);
                }
            }
        }

        // Check budget thresholds
        self.budget_enforcer.check_budget_thresholds(&cost_entry, &self.cost_tracker).await?;

        // Update usage analytics
        self.usage_analyzer.update_usage_patterns(&cost_entry).await;

        // Check for optimization opportunities
        self.optimizer.evaluate_optimization_opportunities(&cost_entry, &self.cost_tracker).await;

        Ok(())
    }

    /// Calculate cost for a specific task execution
    async fn calculate_cost_entry(
        &self,
        task: &TaskRequest,
        response: &TaskResponse,
        execution_time: Duration,
    ) -> Result<CostEntry> {
        let pricing = self.pricing_info.read().await;
        let model_id = response.model_used.model_id();

        // Get base cost from response or calculate from pricing
        let base_cost = if let Some(cost) = response.cost_cents {
            cost
        } else {
            self.estimate_cost_from_usage(&pricing, &model_id, response).await?
        };

        // Apply consciousness enhancement surcharge if applicable
        let consciousness_multiplier = if self.is_consciousness_enhanced(task) {
            1.2 // 20% surcharge for consciousness processing
        } else {
            1.0
        };

        // Apply streaming multiplier if applicable
        let streaming_multiplier = if task.constraints.require_streaming {
            1.15 // 15% surcharge for streaming
        } else {
            1.0
        };

        // Apply quality-based pricing adjustment
        let quality_multiplier = if response.quality_score > 0.9 {
            1.1 // Premium for high quality
        } else if response.quality_score < 0.5 {
            0.9 // Discount for lower quality
        } else {
            1.0
        };

        let final_cost =
            base_cost * consciousness_multiplier * streaming_multiplier * quality_multiplier;

        Ok(CostEntry {
            timestamp: SystemTime::now(),
            model_id,
            task_type: format!("{:?}", task.task_type),
            cost_cents: final_cost,
            tokens_used: response.tokens_generated.unwrap_or(0),
            execution_time_ms: execution_time.as_millis() as u32,
            quality_score: response.quality_score,
            context: task.content.chars().take(100).collect(),
            consciousness_enhanced: self.is_consciousness_enhanced(task),
            streaming_enabled: task.constraints.require_streaming,
            request_id: format!(
                "req_{}",
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
            ),
        })
    }

    /// Estimate cost from usage data
    async fn estimate_cost_from_usage(
        &self,
        pricing: &PricingDatabase,
        model_id: &str,
        response: &TaskResponse,
    ) -> Result<f32> {
        // Try model-specific pricing first
        if let Some(model_pricing) = pricing.model_pricing.get(model_id) {
            let tokens = response.tokens_generated.unwrap_or(100) as f32;
            return Ok((tokens / 1000.0) * model_pricing.output_cost_per_1k_tokens);
        }

        // Fall back to provider pricing
        for (provider_name, provider_pricing) in &pricing.provider_pricing {
            if model_id.contains(provider_name) || model_id.starts_with(provider_name) {
                let tokens = response.tokens_generated.unwrap_or(100) as f32;
                return Ok((tokens / 1000.0) * provider_pricing.output_cost_per_1k_tokens);
            }
        }

        // Default estimation if no specific pricing found
        let tokens = response.tokens_generated.unwrap_or(100) as f32;
        Ok((tokens / 1000.0) * 0.002) // Default: $0.002 per 1K tokens
    }

    /// Check if a task is consciousness enhanced
    fn is_consciousness_enhanced(&self, task: &TaskRequest) -> bool {
        task.constraints.quality_threshold.unwrap_or(0.7) > 0.8
            || matches!(task.task_type, TaskType::LogicalReasoning | TaskType::CreativeWriting)
    }

    /// Update cost breakdown with new entry
    async fn update_cost_breakdown(&self, tracker: &mut CostTracker, entry: &CostEntry) {
        // Update total costs
        tracker.total_costs.total_cents += entry.cost_cents;
        tracker.total_costs.daily_cents += entry.cost_cents;
        tracker.total_costs.weekly_cents += entry.cost_cents;
        tracker.total_costs.monthly_cents += entry.cost_cents;
        tracker.total_costs.request_count += 1;
        tracker.total_costs.token_count += entry.tokens_used as u64;

        // Update averages
        if tracker.total_costs.request_count > 0 {
            tracker.total_costs.average_cost_per_request =
                tracker.total_costs.total_cents / tracker.total_costs.request_count as f32;
        }

        if tracker.total_costs.token_count > 0 {
            tracker.total_costs.average_cost_per_token =
                tracker.total_costs.total_cents / tracker.total_costs.token_count as f32;
        }

        // Update model-specific costs
        let model_breakdown = tracker.model_costs.entry(entry.model_id.clone()).or_default();
        model_breakdown.total_cents += entry.cost_cents;
        model_breakdown.request_count += 1;
        model_breakdown.token_count += entry.tokens_used as u64;

        // Update task-specific costs
        let task_breakdown = tracker.task_costs.entry(entry.task_type.clone()).or_default();
        task_breakdown.total_cents += entry.cost_cents;
        task_breakdown.request_count += 1;
        task_breakdown.token_count += entry.tokens_used as u64;

        // Update consciousness costs if applicable
        if entry.consciousness_enhanced {
            tracker.consciousness_costs.total_cents += entry.cost_cents;
            tracker.consciousness_costs.request_count += 1;
            tracker.consciousness_costs.token_count += entry.tokens_used as u64;
        }

        // Update streaming costs if applicable
        if entry.streaming_enabled {
            tracker.streaming_costs.total_cents += entry.cost_cents;
            tracker.streaming_costs.request_count += 1;
            tracker.streaming_costs.token_count += entry.tokens_used as u64;
        }
    }

    /// Get current budget status
    pub async fn get_budget_status(&self) -> BudgetStatus {
        let config = self.budgetconfig.read().await;
        let tracker = self.cost_tracker.read().await;

        let current_period_cost = match config.budget_period {
            BudgetPeriod::Daily => tracker.total_costs.daily_cents,
            BudgetPeriod::Weekly => tracker.total_costs.weekly_cents,
            BudgetPeriod::Monthly => tracker.total_costs.monthly_cents,
            _ => tracker.total_costs.daily_cents,
        };

        let period_limit = match config.budget_period {
            BudgetPeriod::Daily => config.daily_limit_cents,
            BudgetPeriod::Weekly => config.weekly_limit_cents,
            BudgetPeriod::Monthly => config.monthly_limit_cents,
            _ => config.daily_limit_cents,
        };

        let utilization = if period_limit > 0.0 { current_period_cost / period_limit } else { 0.0 };

        let health = if utilization >= 0.9 {
            BudgetHealth::Critical
        } else if utilization >= 0.75 {
            BudgetHealth::Warning
        } else if utilization >= 0.5 {
            BudgetHealth::Good
        } else {
            BudgetHealth::Excellent
        };

        BudgetStatus {
            current_period_cost,
            period_limit,
            utilization,
            health,
            remaining_budget: period_limit - current_period_cost,
            usage_percentage: utilization * 100.0,
            days_until_reset: 0,
            projected_end_of_period_cost: self.calculate_projected_cost(&tracker, &config).await,
            consciousness_cost: tracker.consciousness_costs.total_cents,
            streaming_cost: tracker.streaming_costs.total_cents,
            top_cost_models: self.get_top_cost_models(&tracker, 5).await,
            optimization_savings: self.get_current_optimization_savings().await,
            last_update: SystemTime::now(),
        }
    }

    /// Get cost analytics and insights
    pub async fn get_cost_analytics(&self) -> CostAnalytics {
        let tracker = self.cost_tracker.read().await;
        let usage_patterns = self.usage_analyzer.usage_patterns.read().await;
        let efficiency_metrics = self.usage_analyzer.efficiency_metrics.read().await;

        CostAnalytics {
            total_costs: tracker.total_costs.clone(),
            model_breakdown: tracker.model_costs.clone(),
            task_breakdown: tracker.task_costs.clone(),
            consciousness_roi: efficiency_metrics.roi_metrics.consciousness_roi,
            streaming_roi: efficiency_metrics.roi_metrics.streaming_roi,
            cost_trends: tracker.cost_trends.clone(),
            usage_patterns: usage_patterns.clone(),
            optimization_opportunities: tracker.cost_trends.optimization_opportunities.clone(),
            efficiency_score: efficiency_metrics.efficiency_score,
            cost_per_quality: efficiency_metrics.cost_per_quality_point,
        }
    }

    /// Initialize pricing database with known provider pricing
    async fn initialize_pricing_database() -> Result<PricingDatabase> {
        let mut pricing_db = PricingDatabase {
            provider_pricing: HashMap::new(),
            model_pricing: HashMap::new(),
            task_modifiers: HashMap::new(),
            pricing_updates: Vec::new(),
            last_update: SystemTime::now(),
        };

        // OpenAI pricing (as of 2024)
        pricing_db.provider_pricing.insert(
            "openai".to_string(),
            ProviderPricing {
                provider_name: "openai".to_string(),
                input_cost_per_1k_tokens: 0.03, // $0.03 per 1K input tokens
                output_cost_per_1k_tokens: 0.06, // $0.06 per 1K output tokens
                streaming_multiplier: 1.0,
                context_window_pricing: vec![],
                minimum_charge_cents: 0.01,
                bulk_discount_tiers: vec![],
            },
        );

        // Anthropic pricing
        pricing_db.provider_pricing.insert(
            "anthropic".to_string(),
            ProviderPricing {
                provider_name: "anthropic".to_string(),
                input_cost_per_1k_tokens: 0.025, // $0.025 per 1K input tokens
                output_cost_per_1k_tokens: 0.125, // $0.125 per 1K output tokens
                streaming_multiplier: 1.0,
                context_window_pricing: vec![],
                minimum_charge_cents: 0.01,
                bulk_discount_tiers: vec![],
            },
        );

        // Mistral pricing
        pricing_db.provider_pricing.insert(
            "mistral".to_string(),
            ProviderPricing {
                provider_name: "mistral".to_string(),
                input_cost_per_1k_tokens: 0.02, // $0.02 per 1K input tokens
                output_cost_per_1k_tokens: 0.06, // $0.06 per 1K output tokens
                streaming_multiplier: 1.0,
                context_window_pricing: vec![],
                minimum_charge_cents: 0.01,
                bulk_discount_tiers: vec![],
            },
        );

        pricing_db.last_update = SystemTime::now();
        Ok(pricing_db)
    }

    /// Calculate projected cost for end of period
    async fn calculate_projected_cost(&self, tracker: &CostTracker, config: &BudgetConfig) -> f32 {
        // Simple projection based on current usage rate
        let current_cost = match config.budget_period {
            BudgetPeriod::Daily => tracker.total_costs.daily_cents,
            BudgetPeriod::Weekly => tracker.total_costs.weekly_cents,
            BudgetPeriod::Monthly => tracker.total_costs.monthly_cents,
            _ => tracker.total_costs.daily_cents,
        };

        // Assume linear projection for simplicity
        // In a real implementation, this would use more sophisticated forecasting
        current_cost * 1.2 // Project 20% increase
    }

    /// Get top cost models
    async fn get_top_cost_models(&self, tracker: &CostTracker, limit: usize) -> Vec<(String, f32)> {
        let mut model_costs: Vec<(String, f32)> = tracker
            .model_costs
            .iter()
            .map(|(model, breakdown)| (model.clone(), breakdown.total_cents))
            .collect();

        model_costs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        model_costs.truncate(limit);
        model_costs
    }

    /// Get current optimization savings
    async fn get_current_optimization_savings(&self) -> f32 {
        let active_optimizations = self.optimizer.active_optimizations.read().await;
        active_optimizations
            .iter()
            .filter(|opt| opt.status == OptimizationStatus::Active)
            .map(|opt| opt.current_savings)
            .sum()
    }

    /// Get cost manager configuration
    pub async fn getconfig(&self) -> BudgetConfig {
        self.budgetconfig.read().await.clone()
    }

    /// Update budget configuration
    pub async fn updateconfig(&self, newconfig: BudgetConfig) -> Result<()> {
        info!("💰 Updating cost manager configuration");
        *self.budgetconfig.write().await = newconfig;
        Ok(())
    }

    /// Check if a task is within budget constraints
    pub async fn check_budget_constraints(&self, _task: &TaskRequest) -> Result<bool> {
        let budget_status = self.get_budget_status().await;
        Ok(budget_status.usage_percentage < 95.0) // Allow if under 95% budget usage
    }
}

/// Budget status and monitoring
#[derive(Debug)]
pub struct BudgetStatus {
    pub current_period_cost: f32,
    pub period_limit: f32,
    pub utilization: f32,
    pub health: BudgetHealth,
    pub remaining_budget: f32,
    pub usage_percentage: f32,
    pub days_until_reset: u32,
    pub projected_end_of_period_cost: f32,
    pub consciousness_cost: f32,
    pub streaming_cost: f32,
    pub top_cost_models: Vec<(String, f32)>,
    pub optimization_savings: f32,
    pub last_update: SystemTime,
}

impl Default for BudgetStatus {
    fn default() -> Self {
        Self {
            current_period_cost: 0.0,
            period_limit: 1000.0,
            utilization: 0.0,
            health: BudgetHealth::Good,
            remaining_budget: 1000.0,
            usage_percentage: 0.0,
            days_until_reset: 30,
            projected_end_of_period_cost: 0.0,
            consciousness_cost: 0.0,
            streaming_cost: 0.0,
            top_cost_models: Vec::new(),
            optimization_savings: 0.0,
            last_update: SystemTime::now(),
        }
    }
}

/// Cost analytics summary
#[derive(Debug, Clone)]
pub struct CostAnalytics {
    pub total_costs: CostBreakdown,
    pub model_breakdown: HashMap<String, CostBreakdown>,
    pub task_breakdown: HashMap<String, CostBreakdown>,
    pub consciousness_roi: f32,
    pub streaming_roi: f32,
    pub cost_trends: CostTrends,
    pub usage_patterns: UsagePatterns,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub efficiency_score: f32,
    pub cost_per_quality: f32,
}

// Stub implementations for supporting structures
impl UsageAnalyzer {
    fn new() -> Self {
        Self {
            usage_patterns: Arc::new(RwLock::new(UsagePatterns::default())),
            efficiency_metrics: Arc::new(RwLock::new(EfficiencyMetrics::default())),
            anomaly_detector: AnomalyDetector {
                baseline_patterns: HashMap::new(),
                anomaly_threshold: 2.0,
                recent_anomalies: Vec::new(),
            },
        }
    }

    async fn update_usage_patterns(&self, _entry: &CostEntry) {
        // Implementation for updating usage patterns
    }
}

impl BudgetEnforcer {
    fn new(_config: &BudgetConfig) -> Self {
        Self {
            alert_thresholds: vec![
                AlertThreshold {
                    percentage: 75.0,
                    alert_type: AlertType::MonthlyBudgetWarning,
                    action: AlertAction::LogWarning,
                    cooldown_minutes: 60,
                },
                AlertThreshold {
                    percentage: 90.0,
                    alert_type: AlertType::EmergencyStop,
                    action: AlertAction::EmergencyStop,
                    cooldown_minutes: 30,
                },
            ],
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            enforcement_actions: Vec::new(),
        }
    }

    async fn check_budget_thresholds(
        &self,
        _entry: &CostEntry,
        _tracker: &Arc<RwLock<CostTracker>>,
    ) -> Result<()> {
        // Implementation for checking budget thresholds
        Ok(())
    }
}

impl CostOptimizer {
    fn new() -> Self {
        Self {
            strategies: Vec::new(),
            active_optimizations: Arc::new(RwLock::new(Vec::new())),
            optimization_history: Vec::new(),
        }
    }

    async fn evaluate_optimization_opportunities(
        &self,
        _entry: &CostEntry,
        _tracker: &Arc<RwLock<CostTracker>>,
    ) {
        // Implementation for evaluating optimization opportunities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cost_manager_creation() {
        let config = BudgetConfig::default();
        let cost_manager = CostManager::new(config).await.unwrap();

        let status = cost_manager.get_budget_status().await;
        assert_eq!(status.health, BudgetHealth::Good);
    }

    #[test]
    fn test_budgetconfig_defaults() {
        let config = BudgetConfig::default();
        assert_eq!(config.daily_limit_cents, 500.0);
        assert_eq!(config.monthly_limit_cents, 10000.0);
        assert!(config.auto_optimization);
    }
}
