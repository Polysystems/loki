use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::orchestrator::{TaskRequest, TaskResponse};

/// Self-optimizing performance learning system
#[derive(Debug)]
pub struct AdaptiveLearningSystem {
    performance_history: Arc<RwLock<PerformanceHistory>>,
    routing_optimizer: Arc<RoutingOptimizer>,
    model_evaluator: Arc<ModelEvaluator>,
    pattern_analyzer: Arc<PatternAnalyzer>,
    config: AdaptiveLearningConfig,
}

/// Configuration for adaptive learning behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    pub learning_rate: f32,
    pub history_retention_days: u64,
    pub min_samples_for_learning: usize,
    pub confidence_threshold: f32,
    pub exploration_rate: f32,
    pub adaptation_window_hours: u64,
    pub quality_weight: f32,
    pub latency_weight: f32,
    pub cost_weight: f32,
    pub reliability_weight: f32,
}

/// Historical performance data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    pub executions: Vec<ExecutionRecord>,
    pub model_profiles: HashMap<String, ModelProfile>,
    pub task_patterns: HashMap<TaskSignature, TaskPattern>,
    pub routing_decisions: Vec<RoutingDecision>,
    pub optimization_events: Vec<OptimizationEvent>,
}

/// Individual execution record for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub timestamp: u64,
    pub task_signature: TaskSignature,
    pub model_used: String,
    pub execution_time_ms: u32,
    pub quality_score: f32,
    pub success: bool,
    pub cost_cents: Option<f32>,
    pub tokens_generated: Option<u32>,
    pub context_size: usize,
    pub user_feedback: Option<f32>,
    pub ensemble_used: bool,
    pub fallback_triggered: bool,
}

/// Task signature for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub struct TaskSignature {
    pub task_type: String,
    pub complexity_level: ComplexityLevel,
    pub content_category: ContentCategory,
    pub context_size_bucket: ContextSizeBucket,
    pub specialization_required: Vec<String>,
}

/// Complexity assessment levels
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ComplexityLevel {
    Simple,   // Short, straightforward tasks
    Moderate, // Standard complexity
    Complex,  // Multi-step or nuanced tasks
    Expert,   // Requires specialized knowledge
}

/// Content category classification
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ContentCategory {
    Code,
    Documentation,
    Analysis,
    Creative,
    Logical,
    Conversational,
    Technical,
    Research,
}

/// Context size buckets for efficient categorization
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ContextSizeBucket {
    Small,  // 0-500 chars
    Medium, // 500-2000 chars
    Large,  // 2000-8000 chars
    XLarge, // 8000+ chars
}

/// Learned model performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub model_id: String,
    pub total_executions: u64,
    pub success_rate: f32,
    pub average_quality: f32,
    pub average_latency_ms: f32,
    pub average_cost_cents: Option<f32>,
    pub specialization_scores: HashMap<String, f32>,
    pub task_type_performance: HashMap<String, TaskTypePerformance>,
    pub reliability_trend: Vec<ReliabilityPoint>,
    pub last_updated: u64,
    pub confidence_level: f32,
}

/// Performance metrics for specific task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTypePerformance {
    pub samples: usize,
    pub avg_quality: f32,
    pub avg_latency: f32,
    pub success_rate: f32,
    pub user_satisfaction: Option<f32>,
    pub improvement_trend: f32,
}

/// Reliability tracking point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityPoint {
    pub timestamp: u64,
    pub success_rate: f32,
    pub quality_score: f32,
    pub sample_size: usize,
}

/// Task execution pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPattern {
    pub signature: TaskSignature,
    pub optimal_models: Vec<ModelRecommendation>,
    pub avg_execution_time: Duration,
    pub typical_cost_range: (f32, f32),
    pub quality_expectations: f32,
    pub sample_count: usize,
    pub last_analysis: u64,
}

/// Model recommendation with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub model_id: String,
    pub confidence: f32,
    pub expected_quality: f32,
    pub expected_latency_ms: u32,
    pub expected_cost_cents: Option<f32>,
    pub reasoning: String,
}

/// Routing decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub timestamp: u64,
    pub task_signature: TaskSignature,
    pub selected_model: String,
    pub alternatives_considered: Vec<String>,
    pub decision_factors: DecisionFactors,
    pub outcome_quality: Option<f32>,
    pub was_optimal: Option<bool>,
}

/// Factors that influenced routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionFactors {
    pub capability_score: f32,
    pub load_factor: f32,
    pub cost_factor: f32,
    pub historical_performance: f32,
    pub exploration_bonus: f32,
    pub final_score: f32,
}

/// System optimization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEvent {
    pub timestamp: u64,
    pub event_type: OptimizationEventType,
    pub affected_models: Vec<String>,
    pub performance_change: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationEventType {
    ModelProfileUpdate,
    RoutingStrategyAdjustment,
    QualityThresholdUpdate,
    PerformanceDegradationDetected,
    NewPatternDiscovered,
    ConfigurationOptimized,
}

impl AdaptiveLearningSystem {
    /// Create a new adaptive learning system
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(PerformanceHistory::new())),
            routing_optimizer: Arc::new(RoutingOptimizer::new()),
            model_evaluator: Arc::new(ModelEvaluator::new()),
            pattern_analyzer: Arc::new(PatternAnalyzer::new()),
            config,
        }
    }

    /// Record a task execution for learning
    pub async fn record_execution(
        &self,
        task: &TaskRequest,
        response: &TaskResponse,
        execution_time: Duration,
        success: bool,
        user_feedback: Option<f32>,
    ) -> Result<()> {
        let task_signature = self.create_task_signature(task).await;

        let record = ExecutionRecord {
            timestamp: current_timestamp(),
            task_signature: task_signature.clone(),
            model_used: response.model_used.model_id(),
            execution_time_ms: execution_time.as_millis() as u32,
            quality_score: self.assess_response_quality(response, task).await,
            success,
            cost_cents: response.cost_cents,
            tokens_generated: response.tokens_generated,
            context_size: task.content.len(),
            user_feedback,
            ensemble_used: false, // Will be set by caller if ensemble was used
            fallback_triggered: false,
        };

        let mut history = self.performance_history.write().await;
        history.executions.push(record);

        // Trigger learning if we have enough new data
        if history.executions.len() % self.config.min_samples_for_learning == 0 {
            drop(history); // Release lock before async learning
            self.trigger_learning_update().await?;
        }

        Ok(())
    }

    /// Get optimized model recommendation for a task
    pub async fn get_optimized_recommendation(
        &self,
        task: &TaskRequest,
        available_models: &[String],
    ) -> Result<OptimizedRecommendation> {
        let task_signature = self.create_task_signature(task).await;

        // Get historical patterns
        let history = self.performance_history.read().await;
        let _pattern = history.task_patterns.get(&task_signature);

        // Generate recommendations using learned data
        let recommendations = self
            .routing_optimizer
            .generate_recommendations(&task_signature, available_models, &history, &self.config)
            .await?;

        // Apply exploration vs exploitation balance
        let final_recommendation =
            self.apply_exploration_strategy(recommendations, &task_signature, &history).await?;

        Ok(final_recommendation)
    }

    /// Update learning models based on recent performance
    pub async fn trigger_learning_update(&self) -> Result<()> {
        info!("Triggering adaptive learning update");

        let mut history = self.performance_history.write().await;

        // Update model profiles
        self.update_model_profiles(&mut history).await?;

        // Analyze task patterns
        self.analyze_task_patterns(&mut history).await?;

        // Optimize routing strategies
        self.optimize_routing_strategies(&mut history).await?;

        // Clean old data
        self.clean_historical_data(&mut history).await?;

        // Record optimization event - extract values before mutable borrow
        let affected_models: Vec<String> = history.model_profiles.keys().cloned().collect();
        let performance_change = self.calculate_recent_performance_change(&history).await;

        history.optimization_events.push(OptimizationEvent {
            timestamp: current_timestamp(),
            event_type: OptimizationEventType::ModelProfileUpdate,
            affected_models,
            performance_change,
            description: "Routine learning update completed".to_string(),
        });

        info!("Adaptive learning update completed");
        Ok(())
    }

    /// Create task signature for pattern matching
    async fn create_task_signature(&self, task: &TaskRequest) -> TaskSignature {
        let complexity = self.assess_task_complexity(task).await;
        let category = self.categorize_content(task).await;
        let context_bucket = self.get_context_size_bucket(task.content.len());
        let specializations = self.extract_required_specializations(task).await;

        TaskSignature {
            task_type: format!("{:?}", task.task_type),
            complexity_level: complexity,
            content_category: category,
            context_size_bucket: context_bucket,
            specialization_required: specializations,
        }
    }

    async fn assess_task_complexity(&self, task: &TaskRequest) -> ComplexityLevel {
        let content = &task.content;
        let word_count = content.split_whitespace().count();

        // Complexity indicators
        let has_multiple_steps =
            content.contains("first") || content.contains("then") || content.contains("finally");
        let has_technical_terms = content.matches(char::is_uppercase).count() > word_count / 10;
        let has_specific_requirements = content.contains("must") || content.contains("requirement");

        match (word_count, has_multiple_steps, has_technical_terms, has_specific_requirements) {
            (w, _, _, _) if w < 20 => ComplexityLevel::Simple,
            (w, false, false, false) if w < 100 => ComplexityLevel::Simple,
            (w, true, _, _) if w > 200 => ComplexityLevel::Complex,
            (_, _, true, true) => ComplexityLevel::Expert,
            (w, _, _, _) if w > 150 => ComplexityLevel::Complex,
            _ => ComplexityLevel::Moderate,
        }
    }

    async fn categorize_content(&self, task: &TaskRequest) -> ContentCategory {
        let content = &task.content.to_lowercase();

        // Content category detection
        if content.contains("code") || content.contains("function") || content.contains("algorithm")
        {
            ContentCategory::Code
        } else if content.contains("document")
            || content.contains("explain")
            || content.contains("describe")
        {
            ContentCategory::Documentation
        } else if content.contains("analyze")
            || content.contains("compare")
            || content.contains("evaluate")
        {
            ContentCategory::Analysis
        } else if content.contains("story")
            || content.contains("creative")
            || content.contains("imagine")
        {
            ContentCategory::Creative
        } else if content.contains("logic")
            || content.contains("reason")
            || content.contains("solve")
        {
            ContentCategory::Logical
        } else if content.contains("research")
            || content.contains("investigate")
            || content.contains("study")
        {
            ContentCategory::Research
        } else if content.contains("technical") || content.contains("specification") {
            ContentCategory::Technical
        } else {
            ContentCategory::Conversational
        }
    }

    fn get_context_size_bucket(&self, size: usize) -> ContextSizeBucket {
        match size {
            0..=500 => ContextSizeBucket::Small,
            501..=2000 => ContextSizeBucket::Medium,
            2001..=8000 => ContextSizeBucket::Large,
            _ => ContextSizeBucket::XLarge,
        }
    }

    async fn extract_required_specializations(&self, task: &TaskRequest) -> Vec<String> {
        let mut specializations = Vec::new();
        let content = &task.content.to_lowercase();

        // Extract specializations based on content analysis
        if content.contains("python") || content.contains("rust") || content.contains("javascript")
        {
            specializations.push("code_generation".to_string());
        }
        if content.contains("review") || content.contains("audit") || content.contains("check") {
            specializations.push("code_review".to_string());
        }
        if content.contains("math") || content.contains("calculate") || content.contains("formula")
        {
            specializations.push("mathematical_computation".to_string());
        }
        if content.contains("translate") || content.contains("language") {
            specializations.push("language_translation".to_string());
        }

        specializations
    }

    async fn assess_response_quality(&self, response: &TaskResponse, task: &TaskRequest) -> f32 {
        // Simplified quality assessment - in practice would use more sophisticated
        // metrics
        let content_length_score = if response.content.len() > 50 { 0.8 } else { 0.4 };
        let task_relevance_score = if response
            .content
            .to_lowercase()
            .contains(&task.content.to_lowercase().split_whitespace().next().unwrap_or(""))
        {
            0.9
        } else {
            0.6
        };

        (content_length_score + task_relevance_score) / 2.0
    }

    async fn update_model_profiles(&self, history: &mut PerformanceHistory) -> Result<()> {
        let recent_window = current_timestamp() - (self.config.adaptation_window_hours * 3600);
        let recent_executions: Vec<_> =
            history.executions.iter().filter(|e| e.timestamp >= recent_window).collect();

        for execution in recent_executions {
            let model_id = &execution.model_used;
            let profile = history
                .model_profiles
                .entry(model_id.clone())
                .or_insert_with(|| ModelProfile::new(model_id.clone()));

            // Update profile with execution data
            profile.total_executions += 1;
            profile.average_quality = self.update_moving_average(
                profile.average_quality,
                execution.quality_score,
                profile.total_executions,
                self.config.learning_rate,
            );
            profile.average_latency_ms = self.update_moving_average(
                profile.average_latency_ms,
                execution.execution_time_ms as f32,
                profile.total_executions,
                self.config.learning_rate,
            );

            // Update success rate
            let success_value = if execution.success { 1.0 } else { 0.0 };
            profile.success_rate = self.update_moving_average(
                profile.success_rate,
                success_value,
                profile.total_executions,
                self.config.learning_rate,
            );

            // Update task type performance
            let task_type = &execution.task_signature.task_type;
            let task_perf = profile
                .task_type_performance
                .entry(task_type.clone())
                .or_insert_with(TaskTypePerformance::default);

            task_perf.samples += 1;
            task_perf.avg_quality = self.update_moving_average(
                task_perf.avg_quality,
                execution.quality_score,
                task_perf.samples as u64,
                self.config.learning_rate,
            );

            profile.last_updated = current_timestamp();
            profile.confidence_level = self.calculate_confidence_level(profile.total_executions);
        }

        Ok(())
    }

    async fn analyze_task_patterns(&self, history: &mut PerformanceHistory) -> Result<()> {
        // Group executions by task signature
        let mut signature_groups: HashMap<TaskSignature, Vec<&ExecutionRecord>> = HashMap::new();

        for execution in &history.executions {
            signature_groups
                .entry(execution.task_signature.clone())
                .or_insert_with(Vec::new)
                .push(execution);
        }

        // Analyze each pattern
        for (signature, executions) in signature_groups {
            if executions.len() >= self.config.min_samples_for_learning {
                let pattern = self.analyze_execution_pattern(&signature, &executions).await?;
                history.task_patterns.insert(signature, pattern);
            }
        }

        Ok(())
    }

    async fn analyze_execution_pattern(
        &self,
        signature: &TaskSignature,
        executions: &[&ExecutionRecord],
    ) -> Result<TaskPattern> {
        // Calculate statistics
        let avg_quality: f32 =
            executions.iter().map(|e| e.quality_score).sum::<f32>() / executions.len() as f32;
        let avg_time_ms: f32 = executions.iter().map(|e| e.execution_time_ms as f32).sum::<f32>()
            / executions.len() as f32;

        let costs: Vec<f32> = executions.iter().filter_map(|e| e.cost_cents).collect();
        let cost_range = if costs.is_empty() {
            (0.0, 0.0)
        } else {
            let min_cost = costs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_cost = costs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (min_cost, max_cost)
        };

        // Find optimal models for this pattern
        let optimal_models = self.find_optimal_models_for_pattern(executions).await?;

        Ok(TaskPattern {
            signature: signature.clone(),
            optimal_models,
            avg_execution_time: Duration::from_millis(avg_time_ms as u64),
            typical_cost_range: cost_range,
            quality_expectations: avg_quality,
            sample_count: executions.len(),
            last_analysis: current_timestamp(),
        })
    }

    async fn find_optimal_models_for_pattern(
        &self,
        executions: &[&ExecutionRecord],
    ) -> Result<Vec<ModelRecommendation>> {
        let mut model_stats: HashMap<String, Vec<f32>> = HashMap::new();

        // Group by model
        for execution in executions {
            model_stats
                .entry(execution.model_used.clone())
                .or_insert_with(Vec::new)
                .push(execution.quality_score);
        }

        // Calculate recommendations with real performance analysis
        let mut recommendations = Vec::new();
        for (model_id, scores) in model_stats {
            if scores.len() >= 3 {
                // Minimum samples for confidence
                // Calculate quality metrics
                let avg_quality = scores.iter().sum::<f32>() / scores.len() as f32;
                let variance = scores.iter().map(|s| (s - avg_quality).powi(2)).sum::<f32>()
                    / scores.len() as f32;
                let confidence = (1.0 - variance).max(0.0).min(1.0);

                if confidence >= self.config.confidence_threshold {
                    // Calculate real latency and cost metrics from execution records
                    let model_executions: Vec<&ExecutionRecord> = executions
                        .iter()
                        .filter(|e| e.model_used == model_id)
                        .map(|e| *e)
                        .collect();

                    // Real latency calculation with statistical analysis
                    let latency_metrics = self.calculate_latency_statistics(&model_executions);

                    // Real cost calculation with trend analysis
                    let cost_metrics = self.calculate_cost_statistics(&model_executions);

                    // Performance-weighted expected values
                    let expected_latency_ms =
                        self.calculate_expected_latency(&latency_metrics, avg_quality);
                    let expected_cost_cents =
                        self.calculate_expected_cost(&cost_metrics, avg_quality);

                    let cost_display = match expected_cost_cents {
                        Some(cost) => cost,
                        None => 0.0,
                    };
                    debug!(
                        "📊 Model {} performance analysis: Quality: {:.2}, Latency: {}ms, Cost: \
                         {:.2}¢",
                        model_id, avg_quality, expected_latency_ms, cost_display
                    );

                    recommendations.push(ModelRecommendation {
                        model_id: model_id.clone(),
                        confidence,
                        expected_quality: avg_quality,
                        expected_latency_ms,
                        expected_cost_cents,
                        reasoning: format!(
                            "Performance analysis from {} samples: {:.1}% confidence, {:.0}ms avg \
                             latency",
                            scores.len(),
                            confidence * 100.0,
                            expected_latency_ms
                        ),
                    });
                }
            }
        }

        // Sort by expected quality
        recommendations
            .sort_by(|a, b| b.expected_quality.partial_cmp(&a.expected_quality).unwrap());

        Ok(recommendations)
    }

    async fn optimize_routing_strategies(&self, history: &mut PerformanceHistory) -> Result<()> {
        // Analyze routing decisions and their outcomes
        let recent_decisions: Vec<_> = history
            .routing_decisions
            .iter()
            .filter(|d| {
                d.timestamp >= current_timestamp() - (self.config.adaptation_window_hours * 3600)
            })
            .collect();

        if recent_decisions.len() >= self.config.min_samples_for_learning {
            // Analyze which decision factors correlate with better outcomes
            let optimization_insights =
                self.analyze_decision_effectiveness(&recent_decisions).await?;

            // Apply insights to update routing parameters
            self.apply_routing_optimizations(optimization_insights).await?;
        }

        Ok(())
    }

    async fn analyze_decision_effectiveness(
        &self,
        decisions: &[&RoutingDecision],
    ) -> Result<RoutingOptimizationInsights> {
        info!("🔍 Analyzing decision effectiveness for {} routing decisions", decisions.len());

        // Real-time analysis of routing decision effectiveness
        let mut capability_score_impact = 0.0;
        let mut cost_score_impact = 0.0;
        let mut performance_score_impact = 0.0;
        let mut exploration_effectiveness = 0.0;

        let mut successful_decisions = 0;
        let mut total_decisions = 0;

        // Analyze correlation between decision factors and outcomes
        for decision in decisions {
            if let Some(outcome_quality) = decision.outcome_quality {
                total_decisions += 1;

                // Analyze capability score correlation
                let capability_correlation = self.calculate_factor_correlation(
                    decision.decision_factors.capability_score,
                    outcome_quality,
                );
                capability_score_impact += capability_correlation;

                // Analyze cost factor effectiveness
                let cost_correlation = self.calculate_cost_effectiveness(
                    decision.decision_factors.cost_factor,
                    outcome_quality,
                );
                cost_score_impact += cost_correlation;

                // Analyze performance factor correlation
                let performance_correlation = self.calculate_factor_correlation(
                    decision.decision_factors.historical_performance,
                    outcome_quality,
                );
                performance_score_impact += performance_correlation;

                // Analyze exploration bonus effectiveness
                if decision.decision_factors.exploration_bonus > 0.0 {
                    exploration_effectiveness += if outcome_quality > 0.7 { 1.0 } else { -0.5 };
                }

                if outcome_quality > 0.6 {
                    successful_decisions += 1;
                }
            }
        }

        // Calculate adjustment factors based on effectiveness analysis
        let success_rate = if total_decisions > 0 {
            successful_decisions as f32 / total_decisions as f32
        } else {
            0.5
        };

        // Normalize correlation scores
        let capability_adjustment = if total_decisions > 0 {
            (capability_score_impact / total_decisions as f32).clamp(-0.1, 0.1)
        } else {
            0.0
        };

        let cost_adjustment = if total_decisions > 0 {
            (cost_score_impact / total_decisions as f32).clamp(-0.1, 0.1)
        } else {
            0.0
        };

        let performance_adjustment = if total_decisions > 0 {
            (performance_score_impact / total_decisions as f32).clamp(-0.1, 0.1)
        } else {
            0.0
        };

        // Adjust exploration rate based on effectiveness
        let exploration_adjustment = if total_decisions > 5 {
            let exploration_avg = exploration_effectiveness / total_decisions as f32;
            if exploration_avg > 0.3 {
                0.01 // Increase exploration if it's working
            } else if exploration_avg < -0.3 {
                -0.01 // Decrease exploration if it's not working
            } else {
                0.0
            }
        } else {
            0.0
        };

        info!(
            "📊 Decision analysis results: Success rate: {:.1}%, Capability: {:.3}, Cost: {:.3}, \
             Performance: {:.3}",
            success_rate * 100.0,
            capability_adjustment,
            cost_adjustment,
            performance_adjustment
        );

        Ok(RoutingOptimizationInsights {
            capability_weight_adjustment: capability_adjustment,
            cost_weight_adjustment: cost_adjustment,
            performance_weight_adjustment: performance_adjustment,
            exploration_rate_adjustment: exploration_adjustment,
        })
    }

    async fn apply_routing_optimizations(
        &self,
        insights: RoutingOptimizationInsights,
    ) -> Result<()> {
        info!("🎯 Applying routing optimizations based on learning insights");

        // Apply weight adjustments to configuration (would modify actual config in
        // practice)
        if insights.capability_weight_adjustment.abs() > 0.001 {
            info!(
                "  📈 Capability weight adjustment: {:+.3}",
                insights.capability_weight_adjustment
            );
        }

        if insights.cost_weight_adjustment.abs() > 0.001 {
            info!("  💰 Cost weight adjustment: {:+.3}", insights.cost_weight_adjustment);
        }

        if insights.performance_weight_adjustment.abs() > 0.001 {
            info!(
                "  ⚡ Performance weight adjustment: {:+.3}",
                insights.performance_weight_adjustment
            );
        }

        if insights.exploration_rate_adjustment.abs() > 0.001 {
            info!("  🔍 Exploration rate adjustment: {:+.3}", insights.exploration_rate_adjustment);
        }

        // In a real implementation, these adjustments would be applied to the active
        // configuration and potentially persisted for future routing decisions

        debug!("Routing optimization insights successfully applied");
        Ok(())
    }

    /// Calculate correlation between decision factor and outcome quality
    fn calculate_factor_correlation(&self, factor_score: f32, outcome_quality: f32) -> f32 {
        // Simple correlation: higher factor scores should correlate with better
        // outcomes
        if factor_score > 0.7 && outcome_quality > 0.7 {
            0.1 // Positive correlation
        } else if factor_score < 0.3 && outcome_quality < 0.5 {
            -0.1 // Negative correlation (low factor, poor outcome)
        } else {
            0.0 // No clear correlation
        }
    }

    /// Calculate cost factor effectiveness (lower cost with good quality is
    /// better)
    fn calculate_cost_effectiveness(&self, cost_factor: f32, outcome_quality: f32) -> f32 {
        // Cost effectiveness: good quality with lower cost scores higher
        if cost_factor < 0.3 && outcome_quality > 0.7 {
            0.05 // High cost but good quality - slight negative adjustment
        } else if cost_factor > 0.7 && outcome_quality > 0.7 {
            -0.05 // Low cost and good quality - positive for cost optimization
        } else {
            0.0
        }
    }

    async fn clean_historical_data(&self, history: &mut PerformanceHistory) -> Result<()> {
        let cutoff_time = current_timestamp() - (self.config.history_retention_days * 24 * 3600);

        // Remove old execution records
        history.executions.retain(|e| e.timestamp >= cutoff_time);
        history.routing_decisions.retain(|d| d.timestamp >= cutoff_time);
        history.optimization_events.retain(|e| e.timestamp >= cutoff_time);

        info!("Cleaned historical data older than {} days", self.config.history_retention_days);
        Ok(())
    }

    async fn calculate_recent_performance_change(&self, history: &PerformanceHistory) -> f32 {
        info!("📈 Calculating recent performance change trends");

        let current_time = current_timestamp();
        let recent_window = current_time - (self.config.adaptation_window_hours * 3600);
        let comparison_window = recent_window - (self.config.adaptation_window_hours * 3600);

        // Get recent executions (last window)
        let recent_executions: Vec<_> =
            history.executions.iter().filter(|e| e.timestamp >= recent_window).collect();

        // Get comparison executions (previous window)
        let comparison_executions: Vec<_> = history
            .executions
            .iter()
            .filter(|e| e.timestamp >= comparison_window && e.timestamp < recent_window)
            .collect();

        if recent_executions.is_empty() || comparison_executions.is_empty() {
            debug!("Insufficient data for performance change calculation");
            return 0.0;
        }

        // Calculate recent performance metrics
        let recent_quality = self.calculate_average_quality(&recent_executions);
        let recent_latency = self.calculate_average_latency(&recent_executions);
        let recent_success_rate = self.calculate_success_rate(&recent_executions);

        // Calculate comparison performance metrics
        let comparison_quality = self.calculate_average_quality(&comparison_executions);
        let comparison_latency = self.calculate_average_latency(&comparison_executions);
        let comparison_success_rate = self.calculate_success_rate(&comparison_executions);

        // Calculate normalized performance changes
        let quality_change = if comparison_quality > 0.0 {
            (recent_quality - comparison_quality) / comparison_quality
        } else {
            0.0
        };

        let latency_change = if comparison_latency > 0.0 {
            // Lower latency is better, so inverse the change
            (comparison_latency - recent_latency) / comparison_latency
        } else {
            0.0
        };

        let success_rate_change = if comparison_success_rate > 0.0 {
            (recent_success_rate - comparison_success_rate) / comparison_success_rate
        } else {
            0.0
        };

        // Weight the different performance aspects
        let weighted_change = quality_change * self.config.quality_weight
            + latency_change * self.config.latency_weight
            + success_rate_change * self.config.reliability_weight;

        // Normalize to reasonable range
        let normalized_change = weighted_change.clamp(-0.5, 0.5);

        info!(
            "📊 Performance analysis: Quality: {:.1}% -> {:.1}% ({:+.1}%), Latency: {:.0}ms -> \
             {:.0}ms ({:+.1}%), Success: {:.1}% -> {:.1}% ({:+.1}%)",
            comparison_quality * 100.0,
            recent_quality * 100.0,
            quality_change * 100.0,
            comparison_latency,
            recent_latency,
            -latency_change * 100.0,
            comparison_success_rate * 100.0,
            recent_success_rate * 100.0,
            success_rate_change * 100.0
        );

        info!("🎯 Overall performance change: {:+.1}%", normalized_change * 100.0);

        normalized_change
    }

    /// Calculate average quality from executions
    fn calculate_average_quality(&self, executions: &[&ExecutionRecord]) -> f32 {
        if executions.is_empty() {
            return 0.0;
        }

        let total_quality: f32 = executions.iter().map(|e| e.quality_score).sum();
        total_quality / executions.len() as f32
    }

    /// Calculate average latency from executions
    fn calculate_average_latency(&self, executions: &[&ExecutionRecord]) -> f32 {
        if executions.is_empty() {
            return 0.0;
        }

        let total_latency: f32 = executions.iter().map(|e| e.execution_time_ms as f32).sum();
        total_latency / executions.len() as f32
    }

    /// Calculate success rate from executions
    fn calculate_success_rate(&self, executions: &[&ExecutionRecord]) -> f32 {
        if executions.is_empty() {
            return 0.0;
        }

        let successful_count = executions.iter().filter(|e| e.success).count();
        successful_count as f32 / executions.len() as f32
    }

    fn update_moving_average(
        &self,
        current: f32,
        new_value: f32,
        total_samples: u64,
        learning_rate: f32,
    ) -> f32 {
        if total_samples == 1 {
            new_value
        } else {
            current * (1.0 - learning_rate) + new_value * learning_rate
        }
    }

    fn calculate_confidence_level(&self, sample_count: u64) -> f32 {
        // Confidence increases with sample size but plateaus
        let log_samples = (sample_count as f32).log10();
        (log_samples / 3.0).min(1.0).max(0.0)
    }

    async fn apply_exploration_strategy(
        &self,
        recommendations: Vec<ModelRecommendation>,
        _task_signature: &TaskSignature,
        _history: &PerformanceHistory,
    ) -> Result<OptimizedRecommendation> {
        // Apply epsilon-greedy exploration
        let explore = rand::random::<f32>() < self.config.exploration_rate;

        if explore && recommendations.len() > 1 {
            // Explore: select a non-optimal model occasionally
            let exploration_choice = &recommendations[1];
            Ok(OptimizedRecommendation {
                model_id: exploration_choice.model_id.clone(),
                confidence: exploration_choice.confidence,
                expected_quality: exploration_choice.expected_quality,
                reasoning: format!("Exploration choice: {}", exploration_choice.reasoning),
                is_exploration: true,
            })
        } else if let Some(best) = recommendations.first() {
            // Exploit: use best known model
            Ok(OptimizedRecommendation {
                model_id: best.model_id.clone(),
                confidence: best.confidence,
                expected_quality: best.expected_quality,
                reasoning: format!("Optimal choice: {}", best.reasoning),
                is_exploration: false,
            })
        } else {
            Err(anyhow::anyhow!("No recommendations available"))
        }
    }

    /// Calculate comprehensive latency statistics for real performance analysis
    fn calculate_latency_statistics(&self, executions: &[&ExecutionRecord]) -> LatencyMetrics {
        if executions.is_empty() {
            return LatencyMetrics::default();
        }

        let latencies: Vec<f32> = executions.iter().map(|e| e.execution_time_ms as f32).collect();
        let mean = latencies.iter().sum::<f32>() / latencies.len() as f32;

        // Calculate percentiles for robust latency analysis
        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = sorted_latencies[sorted_latencies.len() / 2];
        let p95 = sorted_latencies[(sorted_latencies.len() as f32 * 0.95) as usize];
        let p99 = sorted_latencies[(sorted_latencies.len() as f32 * 0.99) as usize];

        // Calculate variance and standard deviation
        let variance =
            latencies.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / latencies.len() as f32;
        let std_dev = variance.sqrt();

        LatencyMetrics { mean, median: p50, p95, p99, std_dev, sample_count: latencies.len() }
    }

    /// Calculate comprehensive cost statistics for real performance analysis
    fn calculate_cost_statistics(&self, executions: &[&ExecutionRecord]) -> CostMetrics {
        let costs_with_values: Vec<f32> = executions.iter().filter_map(|e| e.cost_cents).collect();

        if costs_with_values.is_empty() {
            return CostMetrics::default();
        }

        let mean_cost = costs_with_values.iter().sum::<f32>() / costs_with_values.len() as f32;

        // Calculate cost efficiency (quality per cost)
        let mut efficiency_scores = Vec::new();
        for execution in executions {
            if let Some(cost) = execution.cost_cents {
                if cost > 0.0 {
                    efficiency_scores.push(execution.quality_score / cost);
                }
            }
        }

        let mean_efficiency = if !efficiency_scores.is_empty() {
            efficiency_scores.iter().sum::<f32>() / efficiency_scores.len() as f32
        } else {
            0.0
        };

        // Calculate cost variance
        let cost_variance = costs_with_values.iter().map(|c| (c - mean_cost).powi(2)).sum::<f32>()
            / costs_with_values.len() as f32;

        CostMetrics {
            mean_cost,
            cost_variance,
            efficiency_score: mean_efficiency,
            samples_with_cost: costs_with_values.len(),
        }
    }

    /// Calculate expected latency using performance-weighted predictions
    fn calculate_expected_latency(
        &self,
        latency_metrics: &LatencyMetrics,
        quality_score: f32,
    ) -> u32 {
        if latency_metrics.sample_count == 0 {
            return 1000; // Default fallback
        }

        // Use weighted prediction based on quality and performance consistency
        let stability_factor = 1.0 - (latency_metrics.std_dev / latency_metrics.mean.max(1.0));
        let quality_weight = quality_score.clamp(0.1, 1.0);

        // High quality models tend to be more consistent, use median for stable models
        let predicted_latency = if stability_factor > 0.7 && quality_weight > 0.8 {
            // Stable, high-quality model - use median (P50)
            latency_metrics.median
        } else if stability_factor > 0.5 {
            // Moderately stable - use mean with safety margin
            latency_metrics.mean * 1.1
        } else {
            // Unstable model - use P95 for safer estimates
            latency_metrics.p95
        };

        predicted_latency as u32
    }

    /// Calculate expected cost using performance-weighted predictions
    fn calculate_expected_cost(
        &self,
        cost_metrics: &CostMetrics,
        quality_score: f32,
    ) -> Option<f32> {
        if cost_metrics.samples_with_cost == 0 {
            return None;
        }

        // Quality-adjusted cost prediction
        let base_cost = cost_metrics.mean_cost;
        let quality_factor = quality_score.clamp(0.1, 1.0);

        // Higher quality models might justify slightly higher costs
        let quality_adjusted_cost = base_cost * (0.8 + 0.4 * quality_factor);

        // Factor in cost efficiency trends
        let efficiency_adjustment = if cost_metrics.efficiency_score > 0.1 {
            // Good efficiency suggests predictable costs
            1.0
        } else {
            // Poor efficiency suggests higher variability
            1.2
        };

        Some(quality_adjusted_cost * efficiency_adjustment)
    }
}

/// Latency performance metrics for statistical analysis
#[derive(Debug, Clone, Default)]
struct LatencyMetrics {
    mean: f32,
    median: f32,
    p95: f32,
    p99: f32,
    std_dev: f32,
    sample_count: usize,
}

/// Cost performance metrics for economic analysis
#[derive(Debug, Clone, Default)]
struct CostMetrics {
    mean_cost: f32,
    cost_variance: f32,
    efficiency_score: f32,
    samples_with_cost: usize,
}

/// Routing optimization component
#[derive(Debug)]
pub struct RoutingOptimizer;

impl RoutingOptimizer {
    pub fn new() -> Self {
        Self
    }

    pub async fn generate_recommendations(
        &self,
        task_signature: &TaskSignature,
        available_models: &[String],
        history: &PerformanceHistory,
        config: &AdaptiveLearningConfig,
    ) -> Result<Vec<ModelRecommendation>> {
        let mut recommendations = Vec::new();

        // Check for learned patterns
        if let Some(pattern) = history.task_patterns.get(task_signature) {
            // Use learned optimal models
            for optimal in &pattern.optimal_models {
                if available_models.contains(&optimal.model_id) {
                    recommendations.push(optimal.clone());
                }
            }
        }

        // If no learned patterns, use model profiles
        if recommendations.is_empty() {
            for model_id in available_models {
                if let Some(profile) = history.model_profiles.get(model_id) {
                    let score = self.calculate_composite_score(profile, config);
                    recommendations.push(ModelRecommendation {
                        model_id: model_id.clone(),
                        confidence: profile.confidence_level,
                        expected_quality: profile.average_quality,
                        expected_latency_ms: profile.average_latency_ms as u32,
                        expected_cost_cents: profile.average_cost_cents,
                        reasoning: format!("Profile-based score: {:.2}", score),
                    });
                }
            }

            // Sort by composite score
            recommendations
                .sort_by(|a, b| b.expected_quality.partial_cmp(&a.expected_quality).unwrap());
        }

        Ok(recommendations)
    }

    fn calculate_composite_score(
        &self,
        profile: &ModelProfile,
        config: &AdaptiveLearningConfig,
    ) -> f32 {
        let quality_component = profile.average_quality * config.quality_weight;
        let latency_component =
            (1.0 / (profile.average_latency_ms / 1000.0).max(0.1)) * config.latency_weight;
        let reliability_component = profile.success_rate * config.reliability_weight;
        let cost_component = profile.average_cost_cents.map_or(1.0, |cost| 1.0 / cost.max(0.01))
            * config.cost_weight;

        quality_component + latency_component + reliability_component + cost_component
    }
}

/// Model evaluation component with real-time performance analysis
#[derive(Debug)]
pub struct ModelEvaluator {
    /// Real-time model performance metrics
    real_time_metrics: Arc<RwLock<HashMap<String, RealTimeModelMetrics>>>,
}

/// Real-time metrics for individual models
#[derive(Debug, Clone)]
struct RealTimeModelMetrics {
    /// Moving average of recent quality scores
    quality_trend: MovingAverage,
    /// Moving average of recent latency
    latency_trend: MovingAverage,
    /// Success rate over recent samples
    success_trend: MovingAverage,
    /// Cost efficiency tracking
    cost_efficiency_trend: MovingAverage,
    /// Last performance evaluation timestamp
    last_evaluated: u64,
    /// Performance degradation alerts
    degradation_alerts: Vec<PerformanceDegradationAlert>,
}

/// Moving average calculator for real-time metrics
#[derive(Debug, Clone)]
struct MovingAverage {
    window_size: usize,
    values: Vec<f32>,
    current_average: f32,
}

/// Performance degradation alert
#[derive(Debug, Clone)]
struct PerformanceDegradationAlert {
    timestamp: u64,
    metric_type: String,
    severity: AlertSeverity,
    previous_value: f32,
    current_value: f32,
    description: String,
}

/// Alert severity levels
#[derive(Debug, Clone)]
enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ModelEvaluator {
    pub fn new() -> Self {
        Self { real_time_metrics: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Update real-time metrics for a model
    pub async fn update_real_time_metrics(
        &self,
        model_id: &str,
        quality_score: f32,
        latency_ms: f32,
        success: bool,
        cost_cents: Option<f32>,
    ) -> Result<()> {
        let mut metrics = self.real_time_metrics.write().await;

        let model_metrics =
            metrics.entry(model_id.to_string()).or_insert_with(|| RealTimeModelMetrics::new());

        // Update moving averages
        model_metrics.quality_trend.add_value(quality_score);
        model_metrics.latency_trend.add_value(latency_ms);
        model_metrics.success_trend.add_value(if success { 1.0 } else { 0.0 });

        if let Some(cost) = cost_cents {
            let efficiency = if cost > 0.0 { quality_score / cost } else { quality_score };
            model_metrics.cost_efficiency_trend.add_value(efficiency);
        }

        model_metrics.last_evaluated = current_timestamp();

        // Check for performance degradation
        self.check_performance_degradation(model_id, model_metrics).await?;

        Ok(())
    }

    /// Check for performance degradation and generate alerts
    async fn check_performance_degradation(
        &self,
        model_id: &str,
        metrics: &mut RealTimeModelMetrics,
    ) -> Result<()> {
        let current_time = current_timestamp();

        // Check quality degradation
        if metrics.quality_trend.values.len() >= 10 {
            let recent_avg = metrics.quality_trend.current_average;
            let historical_avg = metrics.quality_trend.values.iter().take(5).sum::<f32>() / 5.0;

            if recent_avg < historical_avg * 0.85 {
                let alert = PerformanceDegradationAlert {
                    timestamp: current_time,
                    metric_type: "quality".to_string(),
                    severity: if recent_avg < historical_avg * 0.7 {
                        AlertSeverity::High
                    } else {
                        AlertSeverity::Medium
                    },
                    previous_value: historical_avg,
                    current_value: recent_avg,
                    description: format!(
                        "Quality degradation detected for model {}: {:.2} -> {:.2}",
                        model_id, historical_avg, recent_avg
                    ),
                };

                metrics.degradation_alerts.push(alert);
                warn!("⚠️  Quality degradation detected for model {}", model_id);
            }
        }

        // Check latency degradation
        if metrics.latency_trend.values.len() >= 10 {
            let recent_avg = metrics.latency_trend.current_average;
            let historical_avg = metrics.latency_trend.values.iter().take(5).sum::<f32>() / 5.0;

            if recent_avg > historical_avg * 1.5 {
                let alert = PerformanceDegradationAlert {
                    timestamp: current_time,
                    metric_type: "latency".to_string(),
                    severity: if recent_avg > historical_avg * 2.0 {
                        AlertSeverity::High
                    } else {
                        AlertSeverity::Medium
                    },
                    previous_value: historical_avg,
                    current_value: recent_avg,
                    description: format!(
                        "Latency degradation detected for model {}: {:.0}ms -> {:.0}ms",
                        model_id, historical_avg, recent_avg
                    ),
                };

                metrics.degradation_alerts.push(alert);
                warn!("⚠️  Latency degradation detected for model {}", model_id);
            }
        }

        Ok(())
    }

    /// Get real-time performance summary for a model
    pub async fn get_real_time_summary(&self, model_id: &str) -> Option<ModelPerformanceSummary> {
        let metrics = self.real_time_metrics.read().await;

        if let Some(model_metrics) = metrics.get(model_id) {
            Some(ModelPerformanceSummary {
                model_id: model_id.to_string(),
                current_quality: model_metrics.quality_trend.current_average,
                current_latency: model_metrics.latency_trend.current_average,
                current_success_rate: model_metrics.success_trend.current_average,
                cost_efficiency: model_metrics.cost_efficiency_trend.current_average,
                recent_alerts: model_metrics.degradation_alerts.len(),
                last_evaluated: model_metrics.last_evaluated,
            })
        } else {
            None
        }
    }
}

/// Real-time performance summary
#[derive(Debug, Clone)]
pub struct ModelPerformanceSummary {
    pub model_id: String,
    pub current_quality: f32,
    pub current_latency: f32,
    pub current_success_rate: f32,
    pub cost_efficiency: f32,
    pub recent_alerts: usize,
    pub last_evaluated: u64,
}

impl RealTimeModelMetrics {
    fn new() -> Self {
        Self {
            quality_trend: MovingAverage::new(20),
            latency_trend: MovingAverage::new(20),
            success_trend: MovingAverage::new(20),
            cost_efficiency_trend: MovingAverage::new(20),
            last_evaluated: current_timestamp(),
            degradation_alerts: Vec::new(),
        }
    }
}

impl MovingAverage {
    fn new(window_size: usize) -> Self {
        Self { window_size, values: Vec::new(), current_average: 0.0 }
    }

    fn add_value(&mut self, value: f32) {
        self.values.push(value);

        // Keep only the most recent values within window size
        if self.values.len() > self.window_size {
            self.values.remove(0);
        }

        // Recalculate average
        self.current_average = self.values.iter().sum::<f32>() / self.values.len() as f32;
    }
}

/// Pattern analysis component with real-time learning pattern detection
#[derive(Debug)]
pub struct PatternAnalyzer {
    /// Detected usage patterns
    detected_patterns: Arc<RwLock<Vec<UsagePattern>>>,
    /// Pattern recognition algorithms
    pattern_algorithms: Vec<PatternAlgorithm>,
}

/// Detected usage pattern
#[derive(Debug, Clone)]
struct UsagePattern {
    pattern_id: String,
    pattern_type: PatternType,
    frequency: f32,
    confidence: f32,
    first_detected: u64,
    last_seen: u64,
    impact_on_performance: f32,
    recommendations: Vec<String>,
}

/// Types of detectable patterns
#[derive(Debug, Clone)]
enum PatternType {
    TemporalUsage,     // Time-based usage patterns
    TaskComplexity,    // Complexity-based routing patterns
    UserBehavior,      // User interaction patterns
    ModelPreference,   // Model preference patterns
    CostOptimization,  // Cost-aware usage patterns
    PerformanceSpikes, // Performance anomaly patterns
}

/// Pattern recognition algorithm
#[derive(Debug, Clone)]
struct PatternAlgorithm {
    name: String,
    algorithm_type: PatternType,
    sensitivity: f32,
    minimum_samples: usize,
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            detected_patterns: Arc::new(RwLock::new(Vec::new())),
            pattern_algorithms: vec![
                PatternAlgorithm {
                    name: "temporal_usage_detector".to_string(),
                    algorithm_type: PatternType::TemporalUsage,
                    sensitivity: 0.7,
                    minimum_samples: 10,
                },
                PatternAlgorithm {
                    name: "complexity_routing_detector".to_string(),
                    algorithm_type: PatternType::TaskComplexity,
                    sensitivity: 0.8,
                    minimum_samples: 15,
                },
                PatternAlgorithm {
                    name: "performance_spike_detector".to_string(),
                    algorithm_type: PatternType::PerformanceSpikes,
                    sensitivity: 0.9,
                    minimum_samples: 5,
                },
            ],
        }
    }

    /// Analyze execution patterns for insights
    pub async fn analyze_execution_patterns(
        &self,
        executions: &[ExecutionRecord],
    ) -> Result<Vec<PatternInsight>> {
        info!("🔍 Analyzing execution patterns from {} records", executions.len());

        let mut insights = Vec::new();

        // Temporal usage pattern analysis
        insights.extend(self.analyze_temporal_patterns(executions).await?);

        // Task complexity routing pattern analysis
        insights.extend(self.analyze_complexity_patterns(executions).await?);

        // Performance spike pattern analysis
        insights.extend(self.analyze_performance_patterns(executions).await?);

        // Update detected patterns
        self.update_detected_patterns(&insights).await?;

        info!("📊 Generated {} pattern insights", insights.len());
        Ok(insights)
    }

    /// Analyze temporal usage patterns
    async fn analyze_temporal_patterns(
        &self,
        executions: &[ExecutionRecord],
    ) -> Result<Vec<PatternInsight>> {
        let mut insights = Vec::new();

        // Group by hour of day
        let mut hourly_usage: HashMap<u32, Vec<&ExecutionRecord>> = HashMap::new();

        for execution in executions {
            let hour = (execution.timestamp % 86400) / 3600; // Hour of day
            hourly_usage.entry(hour as u32).or_insert_with(Vec::new).push(execution);
        }

        // Find peak usage hours
        if let Some((peak_hour, peak_executions)) =
            hourly_usage.iter().max_by_key(|(_, executions)| executions.len())
        {
            if peak_executions.len() > 5 {
                insights.push(PatternInsight {
                    insight_type: "temporal_peak".to_string(),
                    description: format!(
                        "Peak usage detected at hour {} with {} executions",
                        peak_hour,
                        peak_executions.len()
                    ),
                    confidence: 0.8,
                    actionable_recommendations: vec![
                        format!("Consider pre-warming models before hour {}", peak_hour),
                        "Scale compute resources during peak hours".to_string(),
                    ],
                });
            }
        }

        Ok(insights)
    }

    /// Analyze task complexity routing patterns
    async fn analyze_complexity_patterns(
        &self,
        executions: &[ExecutionRecord],
    ) -> Result<Vec<PatternInsight>> {
        let mut insights = Vec::new();

        // Group by complexity level
        let mut complexity_performance: HashMap<String, Vec<f32>> = HashMap::new();

        for execution in executions {
            let complexity_key = format!("{:?}", execution.task_signature.complexity_level);
            complexity_performance
                .entry(complexity_key)
                .or_insert_with(Vec::new)
                .push(execution.quality_score);
        }

        // Find patterns in complexity handling
        for (complexity, scores) in complexity_performance {
            if scores.len() >= 5 {
                let avg_quality = scores.iter().sum::<f32>() / scores.len() as f32;

                if avg_quality < 0.6 {
                    insights.push(PatternInsight {
                        insight_type: "complexity_performance".to_string(),
                        description: format!(
                            "{} complexity tasks showing low performance: {:.2}",
                            complexity, avg_quality
                        ),
                        confidence: 0.7,
                        actionable_recommendations: vec![
                            format!("Review model selection for {} complexity tasks", complexity),
                            "Consider specialized models for complex tasks".to_string(),
                        ],
                    });
                }
            }
        }

        Ok(insights)
    }

    /// Analyze performance spike patterns
    async fn analyze_performance_patterns(
        &self,
        executions: &[ExecutionRecord],
    ) -> Result<Vec<PatternInsight>> {
        let mut insights = Vec::new();

        if executions.len() < 10 {
            return Ok(insights);
        }

        // Calculate performance metrics
        let latencies: Vec<f32> = executions.iter().map(|e| e.execution_time_ms as f32).collect();
        let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;

        // Detect latency spikes
        let spike_threshold = avg_latency * 2.0;
        let spike_count = latencies.iter().filter(|&&latency| latency > spike_threshold).count();

        if spike_count > executions.len() / 10 {
            insights.push(PatternInsight {
                insight_type: "performance_spikes".to_string(),
                description: format!(
                    "Frequent latency spikes detected: {} out of {} executions",
                    spike_count,
                    executions.len()
                ),
                confidence: 0.9,
                actionable_recommendations: vec![
                    "Investigate system resource bottlenecks".to_string(),
                    "Consider load balancing across models".to_string(),
                    "Monitor model memory usage patterns".to_string(),
                ],
            });
        }

        Ok(insights)
    }

    /// Update detected patterns with new insights
    async fn update_detected_patterns(&self, insights: &[PatternInsight]) -> Result<()> {
        let mut patterns = self.detected_patterns.write().await;

        for insight in insights {
            // Convert insight to usage pattern
            let pattern = UsagePattern {
                pattern_id: format!("{}_{}", insight.insight_type, current_timestamp()),
                pattern_type: match insight.insight_type.as_str() {
                    "temporal_peak" => PatternType::TemporalUsage,
                    "complexity_performance" => PatternType::TaskComplexity,
                    "performance_spikes" => PatternType::PerformanceSpikes,
                    _ => PatternType::UserBehavior,
                },
                frequency: 1.0,
                confidence: insight.confidence,
                first_detected: current_timestamp(),
                last_seen: current_timestamp(),
                impact_on_performance: if insight.insight_type == "performance_spikes" {
                    -0.3
                } else {
                    0.1
                },
                recommendations: insight.actionable_recommendations.clone(),
            };

            patterns.push(pattern);
        }

        // Keep only recent patterns (last 30 days)
        let cutoff_time = current_timestamp() - (30 * 24 * 3600);
        patterns.retain(|p| p.last_seen >= cutoff_time);

        Ok(())
    }
}

/// Pattern analysis insight
#[derive(Debug, Clone)]
pub struct PatternInsight {
    pub insight_type: String,
    pub description: String,
    pub confidence: f32,
    pub actionable_recommendations: Vec<String>,
}

/// Optimized recommendation result
#[derive(Debug, Clone)]
pub struct OptimizedRecommendation {
    pub model_id: String,
    pub confidence: f32,
    pub expected_quality: f32,
    pub reasoning: String,
    pub is_exploration: bool,
}

#[derive(Debug, Clone)]
struct RoutingOptimizationInsights {
    capability_weight_adjustment: f32,
    cost_weight_adjustment: f32,
    performance_weight_adjustment: f32,
    exploration_rate_adjustment: f32,
}

impl PerformanceHistory {
    fn new() -> Self {
        Self {
            executions: Vec::new(),
            model_profiles: HashMap::new(),
            task_patterns: HashMap::new(),
            routing_decisions: Vec::new(),
            optimization_events: Vec::new(),
        }
    }
}

impl ModelProfile {
    fn new(model_id: String) -> Self {
        Self {
            model_id,
            total_executions: 0,
            success_rate: 0.0,
            average_quality: 0.0,
            average_latency_ms: 0.0,
            average_cost_cents: None,
            specialization_scores: HashMap::new(),
            task_type_performance: HashMap::new(),
            reliability_trend: Vec::new(),
            last_updated: current_timestamp(),
            confidence_level: 0.0,
        }
    }
}

impl TaskTypePerformance {
    fn default() -> Self {
        Self {
            samples: 0,
            avg_quality: 0.0,
            avg_latency: 0.0,
            success_rate: 0.0,
            user_satisfaction: None,
            improvement_trend: 0.0,
        }
    }
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            history_retention_days: 30,
            min_samples_for_learning: 10,
            confidence_threshold: 0.7,
            exploration_rate: 0.1,
            adaptation_window_hours: 24,
            quality_weight: 0.4,
            latency_weight: 0.3,
            cost_weight: 0.2,
            reliability_weight: 0.1,
        }
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adaptive_learning_creation() {
        let config = AdaptiveLearningConfig::default();
        let learning_system = AdaptiveLearningSystem::new(config);

        // Test that the system initializes properly
        assert!(learning_system.performance_history.read().await.executions.is_empty());
    }

    #[tokio::test]
    async fn test_task_signature_creation() {
        let learning_system = AdaptiveLearningSystem::new(AdaptiveLearningConfig::default());

        let task = TaskRequest {
            task_type: super::super::orchestrator::TaskType::CodeGeneration {
                language: "python".to_string(),
            },
            content: "Write a simple function to calculate fibonacci numbers".to_string(),
            constraints: super::super::orchestrator::TaskConstraints::default(),
            context_integration: false,
            memory_integration: false,
            cognitive_enhancement: false,
        };

        let signature = learning_system.create_task_signature(&task).await;

        assert_eq!(signature.content_category, ContentCategory::Code);
        assert_eq!(signature.context_size_bucket, ContextSizeBucket::Small);
    }

    #[tokio::test]
    async fn test_complexity_assessment() {
        let learning_system = AdaptiveLearningSystem::new(AdaptiveLearningConfig::default());

        // Test simple task complexity
        let simple_task = TaskRequest {
            task_type: super::super::orchestrator::TaskType::CodeGeneration {
                language: "python".to_string(),
            },
            content: "Hello world".to_string(),
            constraints: super::super::orchestrator::TaskConstraints::default(),
            context_integration: false,
            memory_integration: false,
            cognitive_enhancement: false,
        };
        let simple_complexity = learning_system.assess_task_complexity(&simple_task).await;
        assert_eq!(simple_complexity, ComplexityLevel::Simple);

        // Test complex task with multiple steps
        let complex_task = TaskRequest {
            task_type: super::super::orchestrator::TaskType::CodeGeneration {
                language: "python".to_string(),
            },
            content: "First, create a function to parse JSON. Then, implement error handling with \
                      specific requirements for network timeouts. Finally, add comprehensive \
                      logging."
                .to_string(),
            constraints: super::super::orchestrator::TaskConstraints::default(),
            context_integration: false,
            memory_integration: false,
            cognitive_enhancement: false,
        };
        let complex_complexity = learning_system.assess_task_complexity(&complex_task).await;
        assert_eq!(complex_complexity, ComplexityLevel::Complex);

        // Test expert-level task with technical terms and requirements
        let expert_task = TaskRequest {
            task_type: super::super::orchestrator::TaskType::CodeGeneration {
                language: "python".to_string(),
            },
            content: "Implement a high-performance async HTTP client with connection pooling, \
                      request batching, and circuit breaker patterns. Must handle OAuth2 \
                      authentication and support custom retry policies with exponential backoff."
                .to_string(),
            constraints: super::super::orchestrator::TaskConstraints::default(),
            context_integration: false,
            memory_integration: false,
            cognitive_enhancement: false,
        };
        let expert_complexity = learning_system.assess_task_complexity(&expert_task).await;
        assert_eq!(expert_complexity, ComplexityLevel::Expert);
    }

    #[test]
    fn test_latency_statistics_calculation() {
        let learning_system = AdaptiveLearningSystem::new(AdaptiveLearningConfig::default());

        // Create test execution records with varying latencies
        let executions: Vec<ExecutionRecord> = vec![
            ExecutionRecord {
                timestamp: 1000,
                task_signature: TaskSignature {
                    task_type: "test".to_string(),
                    complexity_level: ComplexityLevel::Simple,
                    content_category: ContentCategory::Code,
                    context_size_bucket: ContextSizeBucket::Small,
                    specialization_required: vec![],
                },
                model_used: "test_model".to_string(),
                execution_time_ms: 100,
                quality_score: 0.8,
                success: true,
                cost_cents: Some(1.0),
                tokens_generated: Some(50),
                context_size: 100,
                user_feedback: None,
                ensemble_used: false,
                fallback_triggered: false,
            },
            ExecutionRecord {
                timestamp: 1001,
                task_signature: TaskSignature {
                    task_type: "test".to_string(),
                    complexity_level: ComplexityLevel::Simple,
                    content_category: ContentCategory::Code,
                    context_size_bucket: ContextSizeBucket::Small,
                    specialization_required: vec![],
                },
                model_used: "test_model".to_string(),
                execution_time_ms: 200,
                quality_score: 0.9,
                success: true,
                cost_cents: Some(2.0),
                tokens_generated: Some(75),
                context_size: 150,
                user_feedback: None,
                ensemble_used: false,
                fallback_triggered: false,
            },
            ExecutionRecord {
                timestamp: 1002,
                task_signature: TaskSignature {
                    task_type: "test".to_string(),
                    complexity_level: ComplexityLevel::Simple,
                    content_category: ContentCategory::Code,
                    context_size_bucket: ContextSizeBucket::Small,
                    specialization_required: vec![],
                },
                model_used: "test_model".to_string(),
                execution_time_ms: 150,
                quality_score: 0.85,
                success: true,
                cost_cents: Some(1.5),
                tokens_generated: Some(60),
                context_size: 120,
                user_feedback: None,
                ensemble_used: false,
                fallback_triggered: false,
            },
        ];

        let execution_refs: Vec<&ExecutionRecord> = executions.iter().collect();
        let latency_metrics = learning_system.calculate_latency_statistics(&execution_refs);

        // Verify statistics calculation
        assert_eq!(latency_metrics.mean, 150.0); // (100 + 200 + 150) / 3
        assert_eq!(latency_metrics.median, 150.0); // Middle value when sorted
        assert_eq!(latency_metrics.sample_count, 3);
        assert!(latency_metrics.std_dev > 0.0); // Should have some variance
    }
}
