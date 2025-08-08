//! Cost Analytics and Intelligent Provider Switching
//!
//! Provides real-time cost tracking, analytics, and automatic provider fallback
//! based on cost thresholds and usage patterns.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{debug, info, warn};

/// Cost tracking for API providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Provider-specific costs
    pub providers: HashMap<String, ProviderCost>,

    /// Total costs across all providers
    pub total: TotalCost,

    /// Cost breakdown by model
    pub models: HashMap<String, ModelCost>,

    /// Cost breakdown by task type
    pub tasks: HashMap<TaskType, TaskCost>,

    /// Current cost rates and trends
    pub trends: CostTrends,

    /// Timestamp of cost calculation
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCost {
    pub name: String,
    pub total_cost_usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub input_cost_usd: f64,
    pub output_cost_usd: f64,
    pub requests_count: u64,
    pub avg_cost_per_request: f64,
    pub daily_cost_usd: f64,
    pub hourly_cost_usd: f64,
    pub rate_limits_hit: u64,
    pub fallbacks_triggered: u64,
    pub efficiency_score: f64, // Cost per useful output
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TotalCost {
    pub today_usd: f64,
    pub this_hour_usd: f64,
    pub this_month_usd: f64,
    pub all_time_usd: f64,
    pub projected_monthly_usd: f64,
    pub cost_per_hour_avg: f64,
    pub savings_from_optimization_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub model_name: String,
    pub provider: String,
    pub total_cost_usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub requests_count: u64,
    pub avg_latency_ms: f64,
    pub cost_per_token: f64,
    pub quality_score: f64, // Output quality rating
    pub value_score: f64,   // Quality / Cost ratio
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    Reasoning,
    Coding,
    SocialPosting,
    Creative,
    Search,
    DataAnalysis,
    Memory,
    ToolUsage,
    Default,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCost {
    pub task_type: TaskType,
    pub total_cost_usd: f64,
    pub requests_count: u64,
    pub preferred_provider: String,
    pub fallback_providers: Vec<String>,
    pub avg_cost_per_task: f64,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTrends {
    pub hourly_spend_rate: f64,
    pub cost_acceleration: f64, // Change in spend rate
    pub predicted_daily_cost: f64,
    pub predicted_monthly_cost: f64,
    pub efficiency_trend: f64, // Cost per useful output over time
    pub savings_opportunity: f64,
}

/// Cost-based provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// Maximum daily spend across all providers
    pub max_daily_spend_usd: f64,

    /// Maximum hourly spend across all providers
    pub max_hourly_spend_usd: f64,

    /// Per-provider daily limits
    pub provider_daily_limits: HashMap<String, f64>,

    /// Per-provider hourly limits
    pub provider_hourly_limits: HashMap<String, f64>,

    /// Task-specific cost allocation
    pub task_budgets: HashMap<TaskType, f64>,

    /// Provider preferences with cost tiers
    pub provider_tiers: HashMap<TaskType, Vec<ProviderTier>>,

    /// Automatic fallback configuration
    pub auto_fallback: AutoFallbackConfig,

    /// Cost optimization settings
    pub optimization: CostOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderTier {
    pub provider: String,
    pub model: String,
    pub max_cost_per_request: f64,
    pub priority: u8, // 0 = highest priority
    pub quality_threshold: f64,
    pub use_when: TierCondition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TierCondition {
    Always,
    BudgetRemaining(f64), // Use when this much budget remains
    QualityRequired(f64), // Use when quality score above threshold
    LatencyRequired(u64), // Use when latency below threshold (ms)
    TokenCountBelow(u64), // Use when token count below threshold
    TokenCountAbove(u64), // Use when token count above threshold
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoFallbackConfig {
    pub enabled: bool,
    pub trigger_on_rate_limit: bool,
    pub trigger_on_cost_limit: bool,
    pub trigger_on_error: bool,
    pub fallback_delay_ms: u64,
    pub max_fallback_attempts: u8,
    pub preserve_quality_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimization {
    pub enabled: bool,
    pub target_monthly_budget: f64,
    pub optimize_for: OptimizationTarget,
    pub model_switching_enabled: bool,
    pub batch_requests_enabled: bool,
    pub cache_responses_enabled: bool,
    pub smart_routing_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    MinimizeCost,
    MaximizeQuality,
    BalanceCostQuality,
    MaximizeSpeed,
}

/// Cost alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAlert {
    pub alert_type: CostAlertType,
    pub threshold: f64,
    pub current_value: f64,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: u64,
    pub provider: Option<String>,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostAlertType {
    DailyBudgetExceeded,
    HourlyBudgetExceeded,
    ProviderBudgetExceeded,
    UnexpectedCostSpike,
    EfficiencyDegrading,
    MonthlyProjectionHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Provider performance metrics for intelligent switching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPerformance {
    pub provider: String,
    pub model: String,
    pub success_rate: f64,
    pub avg_latency_ms: f64,
    pub cost_per_token: f64,
    pub quality_score: f64,
    pub reliability_score: f64,
    pub recent_errors: u64,
    pub rate_limit_frequency: f64,
    pub availability: f64,
}

/// Intelligent cost analytics and provider switching system
pub struct CostAnalytics {
    /// Current cost metrics
    cost_metrics: Arc<RwLock<CostMetrics>>,

    /// Cost configuration
    costconfig: Arc<RwLock<CostConfig>>,

    /// Provider performance tracking
    provider_performance: Arc<RwLock<HashMap<String, ProviderPerformance>>>,

    /// Cost history for trend analysis
    cost_history: Arc<RwLock<Vec<CostMetrics>>>,

    /// Active cost alerts
    alerts: Arc<RwLock<Vec<CostAlert>>>,

    /// Cost broadcast sender
    cost_sender: broadcast::Sender<CostMetrics>,

    /// Alert broadcast sender
    alert_sender: broadcast::Sender<CostAlert>,

    /// Provider recommendation cache
    recommendations: Arc<RwLock<HashMap<TaskType, Vec<String>>>>,

    /// Collection interval
    collection_interval: Duration,

    /// Maximum history length
    max_history: usize,
}

impl Default for CostConfig {
    fn default() -> Self {
        let mut provider_tiers = HashMap::new();

        // Reasoning task tiers (quality-focused)
        provider_tiers.insert(
            TaskType::Reasoning,
            vec![
                ProviderTier {
                    provider: "anthropic".to_string(),
                    model: "claude-3-opus".to_string(),
                    max_cost_per_request: 0.50,
                    priority: 0,
                    quality_threshold: 0.9,
                    use_when: TierCondition::QualityRequired(0.85),
                },
                ProviderTier {
                    provider: "openai".to_string(),
                    model: "gpt-4".to_string(),
                    max_cost_per_request: 0.30,
                    priority: 1,
                    quality_threshold: 0.85,
                    use_when: TierCondition::BudgetRemaining(10.0),
                },
                ProviderTier {
                    provider: "deepseek".to_string(),
                    model: "deepseek-chat".to_string(),
                    max_cost_per_request: 0.05,
                    priority: 2,
                    quality_threshold: 0.75,
                    use_when: TierCondition::Always,
                },
            ],
        );

        // Coding task tiers (specialized models)
        provider_tiers.insert(
            TaskType::Coding,
            vec![
                ProviderTier {
                    provider: "deepseek".to_string(),
                    model: "deepseek-coder".to_string(),
                    max_cost_per_request: 0.05,
                    priority: 0,
                    quality_threshold: 0.85,
                    use_when: TierCondition::Always,
                },
                ProviderTier {
                    provider: "codestral".to_string(),
                    model: "codestral-latest".to_string(),
                    max_cost_per_request: 0.10,
                    priority: 1,
                    quality_threshold: 0.80,
                    use_when: TierCondition::BudgetRemaining(5.0),
                },
                ProviderTier {
                    provider: "openai".to_string(),
                    model: "gpt-4".to_string(),
                    max_cost_per_request: 0.30,
                    priority: 2,
                    quality_threshold: 0.90,
                    use_when: TierCondition::QualityRequired(0.85),
                },
            ],
        );

        // Social posting (cost-optimized)
        provider_tiers.insert(
            TaskType::SocialPosting,
            vec![
                ProviderTier {
                    provider: "ollama".to_string(),
                    model: "llama3.2:3b".to_string(),
                    max_cost_per_request: 0.0,
                    priority: 0,
                    quality_threshold: 0.7,
                    use_when: TierCondition::Always,
                },
                ProviderTier {
                    provider: "deepseek".to_string(),
                    model: "deepseek-chat".to_string(),
                    max_cost_per_request: 0.02,
                    priority: 1,
                    quality_threshold: 0.75,
                    use_when: TierCondition::QualityRequired(0.7),
                },
            ],
        );

        Self {
            max_daily_spend_usd: 50.0,
            max_hourly_spend_usd: 5.0,
            provider_daily_limits: HashMap::from([
                ("openai".to_string(), 20.0),
                ("anthropic".to_string(), 20.0),
                ("deepseek".to_string(), 5.0),
                ("codestral".to_string(), 5.0),
            ]),
            provider_hourly_limits: HashMap::from([
                ("openai".to_string(), 2.0),
                ("anthropic".to_string(), 2.0),
                ("deepseek".to_string(), 1.0),
                ("codestral".to_string(), 1.0),
            ]),
            task_budgets: HashMap::from([
                (TaskType::Reasoning, 15.0),
                (TaskType::Coding, 10.0),
                (TaskType::Creative, 10.0),
                (TaskType::SocialPosting, 2.0),
                (TaskType::Search, 3.0),
                (TaskType::DataAnalysis, 5.0),
                (TaskType::Memory, 2.0),
                (TaskType::ToolUsage, 3.0),
                (TaskType::Default, 5.0),
            ]),
            provider_tiers,
            auto_fallback: AutoFallbackConfig {
                enabled: true,
                trigger_on_rate_limit: true,
                trigger_on_cost_limit: true,
                trigger_on_error: true,
                fallback_delay_ms: 1000,
                max_fallback_attempts: 3,
                preserve_quality_threshold: 0.7,
            },
            optimization: CostOptimization {
                enabled: true,
                target_monthly_budget: 500.0,
                optimize_for: OptimizationTarget::BalanceCostQuality,
                model_switching_enabled: true,
                batch_requests_enabled: true,
                cache_responses_enabled: true,
                smart_routing_enabled: true,
            },
        }
    }
}

impl CostAnalytics {
    /// Create a new cost analytics system
    pub fn new(
        costconfig: Option<CostConfig>,
        collection_interval: Option<Duration>,
        max_history: Option<usize>,
    ) -> Self {
        let (cost_sender, _) = broadcast::channel(1000);
        let (alert_sender, _) = broadcast::channel(1000);

        Self {
            cost_metrics: Arc::new(RwLock::new(CostMetrics::default())),
            costconfig: Arc::new(RwLock::new(costconfig.unwrap_or_default())),
            provider_performance: Arc::new(RwLock::new(HashMap::new())),
            cost_history: Arc::new(RwLock::new(Vec::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            cost_sender,
            alert_sender,
            recommendations: Arc::new(RwLock::new(HashMap::new())),
            collection_interval: collection_interval.unwrap_or(Duration::from_secs(60)),
            max_history: max_history.unwrap_or(1440), // 24 hours at 1min intervals
        }
    }

    /// Start the cost monitoring loop
    pub async fn start(&self) -> Result<()> {
        info!("Starting cost analytics monitoring");

        let cost_metrics = self.cost_metrics.clone();
        let costconfig = self.costconfig.clone();
        let cost_history = self.cost_history.clone();
        let alerts = self.alerts.clone();
        let cost_sender = self.cost_sender.clone();
        let alert_sender = self.alert_sender.clone();
        let max_history = self.max_history;

        let mut interval = interval(self.collection_interval);

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                // Update cost metrics
                let current_metrics = cost_metrics.read().await.clone();

                // Store in history
                {
                    let mut history = cost_history.write().await;
                    history.push(current_metrics.clone());

                    if history.len() > max_history {
                        let drain_count = history.len() - max_history;
                        history.drain(0..drain_count);
                    }
                }

                // Check for cost alerts
                let config = costconfig.read().await;
                let new_alerts = Self::check_cost_alerts(&current_metrics, &config).await;

                if !new_alerts.is_empty() {
                    let mut alerts_guard = alerts.write().await;
                    for alert in new_alerts {
                        warn!("Cost alert: {}", alert.message);

                        // Send alert broadcast
                        let _ = alert_sender.send(alert.clone());

                        alerts_guard.push(alert);
                    }

                    // Limit alerts history
                    if alerts_guard.len() > 1000 {
                        let drain_count = alerts_guard.len() - 1000;
                        alerts_guard.drain(0..drain_count);
                    }
                }

                // Broadcast cost metrics
                let _ = cost_sender.send(current_metrics);
            }
        });

        Ok(())
    }

    /// Record a new API call cost
    pub async fn record_api_call(
        &self,
        provider: &str,
        model: &str,
        task_type: TaskType,
        input_tokens: u64,
        output_tokens: u64,
        cost_usd: f64,
        latency_ms: u64,
        quality_score: Option<f64>,
        success: bool,
    ) -> Result<()> {
        debug!(
            "Recording API call: {} {} {} tokens in:{} out:{} cost:${:.4}",
            provider,
            model,
            task_type.name(),
            input_tokens,
            output_tokens,
            cost_usd
        );

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let mut metrics = self.cost_metrics.write().await;

        // Update provider costs
        let provider_cost =
            metrics.providers.entry(provider.to_string()).or_insert_with(|| ProviderCost {
                name: provider.to_string(),
                total_cost_usd: 0.0,
                input_tokens: 0,
                output_tokens: 0,
                input_cost_usd: 0.0,
                output_cost_usd: 0.0,
                requests_count: 0,
                avg_cost_per_request: 0.0,
                daily_cost_usd: 0.0,
                hourly_cost_usd: 0.0,
                rate_limits_hit: 0,
                fallbacks_triggered: 0,
                efficiency_score: 1.0,
            });

        provider_cost.total_cost_usd += cost_usd;
        provider_cost.input_tokens += input_tokens;
        provider_cost.output_tokens += output_tokens;
        provider_cost.requests_count += 1;
        provider_cost.avg_cost_per_request =
            provider_cost.total_cost_usd / provider_cost.requests_count as f64;

        // Update model costs
        let model_key = format!("{}:{}", provider, model);
        let model_cost = metrics.models.entry(model_key.clone()).or_insert_with(|| ModelCost {
            model_name: model.to_string(),
            provider: provider.to_string(),
            total_cost_usd: 0.0,
            input_tokens: 0,
            output_tokens: 0,
            requests_count: 0,
            avg_latency_ms: 0.0,
            cost_per_token: 0.0,
            quality_score: 0.8,
            value_score: 0.8,
        });

        model_cost.total_cost_usd += cost_usd;
        model_cost.input_tokens += input_tokens;
        model_cost.output_tokens += output_tokens;
        model_cost.requests_count += 1;

        // Update average latency
        model_cost.avg_latency_ms = (model_cost.avg_latency_ms
            * (model_cost.requests_count - 1) as f64
            + latency_ms as f64)
            / model_cost.requests_count as f64;

        // Calculate cost per token
        let total_tokens = model_cost.input_tokens + model_cost.output_tokens;
        if total_tokens > 0 {
            model_cost.cost_per_token = model_cost.total_cost_usd / total_tokens as f64;
        }

        // Update quality score
        if let Some(quality) = quality_score {
            model_cost.quality_score = (model_cost.quality_score * 0.9) + (quality * 0.1);
        }

        // Calculate value score (quality / cost ratio)
        if model_cost.cost_per_token > 0.0 {
            model_cost.value_score = model_cost.quality_score / model_cost.cost_per_token;
        }

        // Update task costs
        let task_cost = metrics.tasks.entry(task_type.clone()).or_insert_with(|| TaskCost {
            task_type: task_type.clone(),
            total_cost_usd: 0.0,
            requests_count: 0,
            preferred_provider: provider.to_string(),
            fallback_providers: Vec::new(),
            avg_cost_per_task: 0.0,
            optimization_potential: 0.0,
        });

        task_cost.total_cost_usd += cost_usd;
        task_cost.requests_count += 1;
        task_cost.avg_cost_per_task = task_cost.total_cost_usd / task_cost.requests_count as f64;

        // Update provider performance tracking
        {
            let mut performance = self.provider_performance.write().await;
            let perf = performance.entry(model_key).or_insert_with(|| ProviderPerformance {
                provider: provider.to_string(),
                model: model.to_string(),
                success_rate: 1.0,
                avg_latency_ms: latency_ms as f64,
                cost_per_token: 0.0,
                quality_score: quality_score.unwrap_or(0.8),
                reliability_score: 1.0,
                recent_errors: 0,
                rate_limit_frequency: 0.0,
                availability: 1.0,
            });

            // Update success rate
            let total_requests = perf.success_rate * 100.0 + if success { 1.0 } else { 0.0 };
            perf.success_rate = total_requests / 101.0;

            // Update average latency
            perf.avg_latency_ms = (perf.avg_latency_ms * 0.9) + (latency_ms as f64 * 0.1);

            // Update cost per token
            if input_tokens + output_tokens > 0 {
                perf.cost_per_token = cost_usd / (input_tokens + output_tokens) as f64;
            }

            // Update quality score
            if let Some(quality) = quality_score {
                perf.quality_score = (perf.quality_score * 0.9) + (quality * 0.1);
            }

            // Update reliability score based on errors
            if !success {
                perf.recent_errors += 1;
                perf.reliability_score = (perf.reliability_score * 0.95).max(0.1);
            } else {
                perf.reliability_score = (perf.reliability_score * 1.01).min(1.0);
            }
        }

        // Update total costs
        metrics.total.today_usd += cost_usd;
        metrics.total.this_hour_usd += cost_usd;
        metrics.total.all_time_usd += cost_usd;

        metrics.timestamp = timestamp;

        info!(
            "Recorded ${:.4} cost for {} ({} tokens) - Total today: ${:.2}",
            cost_usd,
            provider,
            input_tokens + output_tokens,
            metrics.total.today_usd
        );

        Ok(())
    }

    /// Get intelligent provider recommendation for a task
    pub async fn get_provider_recommendation(
        &self,
        task_type: TaskType,
        estimated_tokens: Option<u64>,
        quality_requirement: Option<f64>,
        budget_remaining: Option<f64>,
    ) -> Result<Option<String>> {
        let config = self.costconfig.read().await;
        let performance = self.provider_performance.read().await;
        let metrics = self.cost_metrics.read().await;

        // Get provider tiers for this task
        let tiers = config.provider_tiers.get(&task_type).cloned().unwrap_or_default();

        if tiers.is_empty() {
            debug!("No provider tiers configured for task: {:?}", task_type);
            return Ok(None);
        }

        // Filter tiers based on conditions
        let mut eligible_tiers = Vec::new();

        for tier in tiers {
            let mut eligible = true;

            // Check tier conditions
            match &tier.use_when {
                TierCondition::Always => {}
                TierCondition::BudgetRemaining(required) => {
                    if budget_remaining.unwrap_or(0.0) < *required {
                        eligible = false;
                    }
                }
                TierCondition::QualityRequired(required) => {
                    if quality_requirement.unwrap_or(0.0) < *required {
                        eligible = false;
                    }
                }
                TierCondition::TokenCountBelow(max_tokens) => {
                    if estimated_tokens.unwrap_or(0) >= *max_tokens {
                        eligible = false;
                    }
                }
                TierCondition::TokenCountAbove(min_tokens) => {
                    if estimated_tokens.unwrap_or(0) < *min_tokens {
                        eligible = false;
                    }
                }
                TierCondition::LatencyRequired(_) => {
                    // Check provider performance for latency
                    let provider_key = format!("{}:{}", tier.provider, tier.model);
                    if let Some(perf) = performance.get(&provider_key) {
                        if perf.avg_latency_ms > 5000.0 {
                            // 5 second threshold
                            eligible = false;
                        }
                    }
                }
            }

            // Check cost limits
            if let Some(tokens) = estimated_tokens {
                let provider_key = format!("{}:{}", tier.provider, tier.model);
                if let Some(perf) = performance.get(&provider_key) {
                    let estimated_cost = perf.cost_per_token * tokens as f64;
                    if estimated_cost > tier.max_cost_per_request {
                        eligible = false;
                    }
                }
            }

            // Check provider daily/hourly limits
            if let Some(provider_cost) = metrics.providers.get(&tier.provider) {
                if let Some(daily_limit) = config.provider_daily_limits.get(&tier.provider) {
                    if provider_cost.daily_cost_usd >= *daily_limit {
                        eligible = false;
                    }
                }

                if let Some(hourly_limit) = config.provider_hourly_limits.get(&tier.provider) {
                    if provider_cost.hourly_cost_usd >= *hourly_limit {
                        eligible = false;
                    }
                }
            }

            if eligible {
                eligible_tiers.push(tier);
            }
        }

        if eligible_tiers.is_empty() {
            warn!("No eligible providers for task: {:?}", task_type);
            return Ok(None);
        }

        // Sort by priority and select best
        eligible_tiers.sort_by_key(|tier| tier.priority);

        let selected = &eligible_tiers[0];
        let provider_model = format!("{}:{}", selected.provider, selected.model);

        info!(
            "Selected provider {} for task {:?} (priority: {}, max_cost: ${:.4})",
            provider_model, task_type, selected.priority, selected.max_cost_per_request
        );

        Ok(Some(provider_model))
    }

    /// Record a fallback event
    pub async fn record_fallback(
        &self,
        from_provider: &str,
        to_provider: &str,
        reason: &str,
    ) -> Result<()> {
        info!("Provider fallback: {} -> {} (reason: {})", from_provider, to_provider, reason);

        let mut metrics = self.cost_metrics.write().await;

        // Update fallback count for the original provider
        if let Some(provider_cost) = metrics.providers.get_mut(from_provider) {
            provider_cost.fallbacks_triggered += 1;
        }

        // Update provider performance
        let mut performance = self.provider_performance.write().await;
        if let Some(perf) = performance.get_mut(from_provider) {
            perf.reliability_score = (perf.reliability_score * 0.9).max(0.1);

            if reason.contains("rate limit") {
                perf.rate_limit_frequency += 0.1;
            }

            if reason.contains("error") {
                perf.recent_errors += 1;
            }
        }

        Ok(())
    }

    /// Check for cost alerts
    async fn check_cost_alerts(metrics: &CostMetrics, config: &CostConfig) -> Vec<CostAlert> {
        let mut alerts = Vec::new();
        let timestamp = metrics.timestamp;

        // Daily budget alerts
        if metrics.total.today_usd >= config.max_daily_spend_usd {
            alerts.push(CostAlert {
                alert_type: CostAlertType::DailyBudgetExceeded,
                threshold: config.max_daily_spend_usd,
                current_value: metrics.total.today_usd,
                message: format!(
                    "Daily budget exceeded: ${:.2} / ${:.2}",
                    metrics.total.today_usd, config.max_daily_spend_usd
                ),
                severity: AlertSeverity::Critical,
                timestamp,
                provider: None,
                resolved: false,
            });
        } else if metrics.total.today_usd >= config.max_daily_spend_usd * 0.8 {
            alerts.push(CostAlert {
                alert_type: CostAlertType::DailyBudgetExceeded,
                threshold: config.max_daily_spend_usd * 0.8,
                current_value: metrics.total.today_usd,
                message: format!(
                    "Daily budget 80% reached: ${:.2} / ${:.2}",
                    metrics.total.today_usd, config.max_daily_spend_usd
                ),
                severity: AlertSeverity::Warning,
                timestamp,
                provider: None,
                resolved: false,
            });
        }

        // Hourly budget alerts
        if metrics.total.this_hour_usd >= config.max_hourly_spend_usd {
            alerts.push(CostAlert {
                alert_type: CostAlertType::HourlyBudgetExceeded,
                threshold: config.max_hourly_spend_usd,
                current_value: metrics.total.this_hour_usd,
                message: format!(
                    "Hourly budget exceeded: ${:.2} / ${:.2}",
                    metrics.total.this_hour_usd, config.max_hourly_spend_usd
                ),
                severity: AlertSeverity::Critical,
                timestamp,
                provider: None,
                resolved: false,
            });
        }

        // Provider-specific budget alerts
        for (provider_name, provider_cost) in &metrics.providers {
            if let Some(daily_limit) = config.provider_daily_limits.get(provider_name) {
                if provider_cost.daily_cost_usd >= *daily_limit {
                    alerts.push(CostAlert {
                        alert_type: CostAlertType::ProviderBudgetExceeded,
                        threshold: *daily_limit,
                        current_value: provider_cost.daily_cost_usd,
                        message: format!(
                            "{} daily budget exceeded: ${:.2} / ${:.2}",
                            provider_name, provider_cost.daily_cost_usd, daily_limit
                        ),
                        severity: AlertSeverity::Critical,
                        timestamp,
                        provider: Some(provider_name.clone()),
                        resolved: false,
                    });
                }
            }
        }

        // Monthly projection alerts
        if metrics.total.projected_monthly_usd >= config.optimization.target_monthly_budget {
            alerts.push(CostAlert {
                alert_type: CostAlertType::MonthlyProjectionHigh,
                threshold: config.optimization.target_monthly_budget,
                current_value: metrics.total.projected_monthly_usd,
                message: format!(
                    "Monthly projection high: ${:.2} / ${:.2}",
                    metrics.total.projected_monthly_usd, config.optimization.target_monthly_budget
                ),
                severity: AlertSeverity::Warning,
                timestamp,
                provider: None,
                resolved: false,
            });
        }

        alerts
    }

    /// Get current cost metrics
    pub async fn get_current_metrics(&self) -> CostMetrics {
        self.cost_metrics.read().await.clone()
    }

    /// Get cost history
    pub async fn get_cost_history(&self, limit: Option<usize>) -> Vec<CostMetrics> {
        let history = self.cost_history.read().await;
        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.clone(),
        }
    }

    /// Get active cost alerts
    pub async fn get_active_alerts(&self) -> Vec<CostAlert> {
        let alerts = self.alerts.read().await;
        alerts.iter().filter(|alert| !alert.resolved).cloned().collect()
    }

    /// Subscribe to cost updates
    pub fn subscribe_costs(&self) -> broadcast::Receiver<CostMetrics> {
        self.cost_sender.subscribe()
    }

    /// Subscribe to cost alerts
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<CostAlert> {
        self.alert_sender.subscribe()
    }

    /// Update cost configuration
    pub async fn updateconfig(&self, config: CostConfig) -> Result<()> {
        let mut currentconfig = self.costconfig.write().await;
        *currentconfig = config;
        info!("Updated cost configuration");
        Ok(())
    }

    /// Get provider performance metrics
    pub async fn get_provider_performance(&self) -> HashMap<String, ProviderPerformance> {
        self.provider_performance.read().await.clone()
    }

    /// Reset daily costs (called at midnight)
    pub async fn reset_daily_costs(&self) -> Result<()> {
        let mut metrics = self.cost_metrics.write().await;

        metrics.total.today_usd = 0.0;
        for provider_cost in metrics.providers.values_mut() {
            provider_cost.daily_cost_usd = 0.0;
        }

        info!("Reset daily cost counters");
        Ok(())
    }

    /// Reset hourly costs (called every hour)
    pub async fn reset_hourly_costs(&self) -> Result<()> {
        let mut metrics = self.cost_metrics.write().await;

        metrics.total.this_hour_usd = 0.0;
        for provider_cost in metrics.providers.values_mut() {
            provider_cost.hourly_cost_usd = 0.0;
        }

        info!("Reset hourly cost counters");
        Ok(())
    }
}

impl Default for CostMetrics {
    fn default() -> Self {
        Self {
            providers: HashMap::new(),
            total: TotalCost {
                today_usd: 0.0,
                this_hour_usd: 0.0,
                this_month_usd: 0.0,
                all_time_usd: 0.0,
                projected_monthly_usd: 0.0,
                cost_per_hour_avg: 0.0,
                savings_from_optimization_usd: 0.0,
            },
            models: HashMap::new(),
            tasks: HashMap::new(),
            trends: CostTrends {
                hourly_spend_rate: 0.0,
                cost_acceleration: 0.0,
                predicted_daily_cost: 0.0,
                predicted_monthly_cost: 0.0,
                efficiency_trend: 0.0,
                savings_opportunity: 0.0,
            },
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        }
    }
}

impl TaskType {
    /// Get the string name of the task type
    pub fn name(&self) -> &'static str {
        match self {
            TaskType::Reasoning => "reasoning",
            TaskType::Coding => "coding",
            TaskType::SocialPosting => "social_posting",
            TaskType::Creative => "creative",
            TaskType::Search => "search",
            TaskType::DataAnalysis => "data_analysis",
            TaskType::Memory => "memory",
            TaskType::ToolUsage => "tool_usage",
            TaskType::Default => "default",
        }
    }
}
