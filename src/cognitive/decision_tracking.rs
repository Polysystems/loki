//! Comprehensive Decision Tracking System
//!
//! This module implements advanced decision tracking, analytics, monitoring,
//! and debugging capabilities for the cognitive decision-making system.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info};

use super::{ActualOutcome, Decision, DecisionId};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Comprehensive decision tracking system
#[derive(Debug)]
pub struct DecisionTracker {
    /// Core decision analytics engine
    analytics: Arc<DecisionAnalytics>,

    /// Real-time decision monitor
    monitor: Arc<DecisionMonitor>,

    /// Decision debugging and replay system
    debugger: Arc<DecisionDebugger>,

    /// Decision flow analyzer
    flow_analyzer: Arc<DecisionFlowAnalyzer>,

    /// Memory integration
    memory: Arc<CognitiveMemory>,

    /// Event channels
    event_tx: broadcast::Sender<DecisionTrackingEvent>,

    /// Configuration
    config: DecisionTrackingConfig,

    /// Active tracking sessions
    tracking_sessions: Arc<RwLock<HashMap<String, TrackingSession>>>,

    /// Performance metrics
    metrics: Arc<RwLock<DecisionTrackingMetrics>>,
}

/// Decision tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrackingConfig {
    /// Enable real-time monitoring
    pub enable_monitoring: bool,

    /// Enable decision flow analysis
    pub enable_flow_analysis: bool,

    /// Enable performance profiling
    pub enable_profiling: bool,

    /// Maximum decision history size
    pub max_history_size: usize,

    /// Analytics update interval
    pub analytics_interval_secs: u64,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Memory storage settings
    pub memory_integration: MemoryIntegrationConfig,
}

impl Default for DecisionTrackingConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            enable_flow_analysis: true,
            enable_profiling: true,
            max_history_size: 10000,
            analytics_interval_secs: 60,
            alert_thresholds: AlertThresholds::default(),
            memory_integration: MemoryIntegrationConfig::default(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Low confidence threshold
    pub low_confidence_threshold: f32,

    /// High decision time threshold (seconds)
    pub high_decision_time_threshold: f32,

    /// Low success rate threshold
    pub low_success_rate_threshold: f32,

    /// High error rate threshold
    pub high_error_rate_threshold: f32,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            low_confidence_threshold: 0.4,
            high_decision_time_threshold: 30.0,
            low_success_rate_threshold: 0.6,
            high_error_rate_threshold: 0.1,
        }
    }
}

/// Memory integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIntegrationConfig {
    /// Store decision context in memory
    pub store_context: bool,

    /// Store decision outcomes in memory
    pub store_outcomes: bool,

    /// Store decision patterns in memory
    pub store_patterns: bool,

    /// Memory importance threshold
    pub importance_threshold: f32,
}

impl Default for MemoryIntegrationConfig {
    fn default() -> Self {
        Self {
            store_context: true,
            store_outcomes: true,
            store_patterns: true,
            importance_threshold: 0.7,
        }
    }
}

/// Decision tracking events
#[derive(Debug, Clone)]
pub enum DecisionTrackingEvent {
    /// Decision initiated
    DecisionStarted {
        session_id: String,
        decision_id: DecisionId,
        context: String,
        timestamp: SystemTime,
    },

    /// Decision step completed
    DecisionStepCompleted {
        session_id: String,
        decision_id: DecisionId,
        step: DecisionStep,
        duration_ms: u64,
    },

    /// Decision completed
    DecisionCompleted {
        session_id: String,
        decision_id: DecisionId,
        decision: Decision,
        total_duration_ms: u64,
        context_captured: bool,
    },

    /// Decision outcome recorded
    OutcomeRecorded {
        decision_id: DecisionId,
        outcome: ActualOutcome,
        impact_analysis: ImpactAnalysis,
    },

    /// Pattern detected
    PatternDetected {
        pattern_type: PatternType,
        pattern_description: String,
        confidence: f32,
        related_decisions: Vec<DecisionId>,
    },

    /// Alert triggered
    AlertTriggered {
        alert_type: AlertType,
        message: String,
        severity: AlertSeverity,
        related_decisions: Vec<DecisionId>,
    },
}

/// Decision processing steps for tracking
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum DecisionStep {
    ContextAnalysis,
    OptionGeneration,
    CriteriaDefinition,
    OptionEvaluation,
    EmotionalInfluence,
    ConsequencePrediction,
    FinalSelection,
    SafetyValidation,
    Execution,
}

/// Decision pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Recurring decision context
    RecurringContext,

    /// Consistent option selection
    ConsistentSelection,

    /// Predictable criteria weighting
    CriteriaPattern,

    /// Emotional influence pattern
    EmotionalPattern,

    /// Time-based decision pattern
    TemporalPattern,

    /// Success/failure pattern
    OutcomePattern,

    /// Cross-system impact pattern
    SystemImpactPattern,
}

/// Alert types for decision monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    LowConfidenceDecision,
    SlowDecisionProcess,
    HighErrorRate,
    UnusualPattern,
    SystemImpact,
    ResourceExhaustion,
    SafetyViolation,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Tracking session for grouped decisions
#[derive(Debug, Clone)]
pub struct TrackingSession {
    pub session_id: String,
    pub started_at: SystemTime,
    pub context: String,
    pub decisions: Vec<DecisionId>,
    pub current_step: Option<DecisionStep>,
    pub step_timings: HashMap<DecisionStep, Duration>,
    pub metadata: HashMap<String, String>,
}

/// Decision tracking metrics
#[derive(Debug, Clone, Default)]
pub struct DecisionTrackingMetrics {
    pub total_decisions_tracked: u64,
    pub total_sessions: u64,
    pub avg_decision_time_ms: f64,
    pub avg_confidence: f32,
    pub pattern_detection_accuracy: f32,
    pub alert_count: HashMap<AlertType, u64>,
    pub system_impact_score: f32,
    pub tracking_overhead_ms: f64,
}

/// Impact analysis for decision outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    /// Direct impact on system performance
    pub performance_impact: f32,

    /// Impact on other systems
    pub cross_system_impact: HashMap<String, f32>,

    /// Long-term consequences observed
    pub long_term_effects: Vec<String>,

    /// Unexpected side effects
    pub side_effects: Vec<String>,

    /// Resource utilization impact
    pub resource_impact: ResourceImpact,

    /// Goal achievement impact
    pub goal_impact: f32,
}

/// Resource impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceImpact {
    pub cpu_usage_change: f32,
    pub memory_usage_change: f32,
    pub network_usage_change: f32,
    pub cost_impact_cents: f32,
}

impl DecisionTracker {
    /// Create new decision tracker
    pub async fn new(memory: Arc<CognitiveMemory>, config: DecisionTrackingConfig) -> Result<Self> {
        info!("🔍 Initializing Comprehensive Decision Tracking System");

        let (event_tx, _) = broadcast::channel(1000);

        let analytics = Arc::new(DecisionAnalytics::new(memory.clone(), config.clone()).await?);

        let monitor = Arc::new(DecisionMonitor::new(event_tx.clone(), config.clone()).await?);

        let debugger = Arc::new(DecisionDebugger::new(memory.clone(), config.clone()).await?);

        let flow_analyzer =
            Arc::new(DecisionFlowAnalyzer::new(memory.clone(), config.clone()).await?);

        Ok(Self {
            analytics,
            monitor,
            debugger,
            flow_analyzer,
            memory,
            event_tx,
            config,
            tracking_sessions: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(DecisionTrackingMetrics::default())),
        })
    }

    /// Start tracking a new decision
    pub async fn start_decision_tracking(
        &self,
        decision_id: DecisionId,
        context: String,
        session_id: Option<String>,
    ) -> Result<String> {
        let session_id = session_id.unwrap_or_else(|| {
            format!(
                "session_{}",
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
            )
        });

        let timestamp = SystemTime::now();

        // Create or update tracking session
        {
            let mut sessions = self.tracking_sessions.write().await;
            let session = sessions.entry(session_id.clone()).or_insert_with(|| TrackingSession {
                session_id: session_id.clone(),
                started_at: timestamp,
                context: context.clone(),
                decisions: Vec::new(),
                current_step: Some(DecisionStep::ContextAnalysis),
                step_timings: HashMap::new(),
                metadata: HashMap::new(),
            });

            session.decisions.push(decision_id.clone());
        }

        // Emit tracking event
        let _ = self.event_tx.send(DecisionTrackingEvent::DecisionStarted {
            session_id: session_id.clone(),
            decision_id,
            context,
            timestamp,
        });

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_decisions_tracked += 1;
        }

        Ok(session_id)
    }

    /// Track a decision step completion
    pub async fn track_decision_step(
        &self,
        session_id: &str,
        decision_id: DecisionId,
        step: DecisionStep,
        duration: Duration,
    ) -> Result<()> {
        // Update session
        {
            let mut sessions = self.tracking_sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                session.step_timings.insert(step.clone(), duration);
                session.current_step = self.get_next_step(&step);
            }
        }

        // Emit tracking event
        let _ = self.event_tx.send(DecisionTrackingEvent::DecisionStepCompleted {
            session_id: session_id.to_string(),
            decision_id,
            step,
            duration_ms: duration.as_millis() as u64,
        });

        Ok(())
    }

    /// Complete decision tracking
    pub async fn complete_decision_tracking(
        &self,
        session_id: &str,
        decision_id: DecisionId,
        decision: Decision,
        total_duration: Duration,
    ) -> Result<()> {
        let context_captured = self.capture_decision_context(&decision).await?;

        // Emit completion event
        let _ = self.event_tx.send(DecisionTrackingEvent::DecisionCompleted {
            session_id: session_id.to_string(),
            decision_id: decision_id.clone(),
            decision: decision.clone(),
            total_duration_ms: total_duration.as_millis() as u64,
            context_captured,
        });

        // Add to analytics
        self.analytics.record_decision(decision.clone()).await?;

        // Check for patterns
        self.check_decision_patterns(&decision).await?;

        // Update metrics
        self.update_tracking_metrics(&decision, total_duration).await;

        // Store in memory if configured
        if self.config.memory_integration.store_context {
            self.store_decision_in_memory(&decision).await?;
        }

        Ok(())
    }

    /// Record decision outcome for learning
    pub async fn record_decision_outcome(
        &self,
        decision_id: DecisionId,
        outcome: ActualOutcome,
    ) -> Result<()> {
        // Perform impact analysis
        let impact_analysis = self.analyze_decision_impact(&decision_id, &outcome).await?;

        // Emit outcome event
        let _ = self.event_tx.send(DecisionTrackingEvent::OutcomeRecorded {
            decision_id: decision_id.clone(),
            outcome: outcome.clone(),
            impact_analysis: impact_analysis.clone(),
        });

        // Update analytics
        self.analytics.record_outcome(decision_id, outcome, impact_analysis).await?;

        Ok(())
    }

    /// Get comprehensive decision analytics
    pub async fn get_decision_analytics(&self) -> Result<DecisionAnalyticsReport> {
        self.analytics.generate_comprehensive_report().await
    }

    /// Get real-time decision monitoring status
    pub async fn get_monitoring_status(&self) -> Result<DecisionMonitoringStatus> {
        self.monitor.get_current_status().await
    }

    /// Get decision debugging information
    pub async fn get_debug_info(&self, decision_id: DecisionId) -> Result<DecisionDebugInfo> {
        self.debugger.get_decision_debug_info(decision_id).await
    }

    /// Get decision flow analysis
    pub async fn get_flow_analysis(&self, session_id: &str) -> Result<DecisionFlowAnalysis> {
        self.flow_analyzer.analyze_session_flow(session_id).await
    }

    /// Start real-time monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        self.monitor.start().await
    }

    /// Stop real-time monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        self.monitor.stop().await
    }

    /// Helper: Get next decision step
    fn get_next_step(&self, current_step: &DecisionStep) -> Option<DecisionStep> {
        match current_step {
            DecisionStep::ContextAnalysis => Some(DecisionStep::OptionGeneration),
            DecisionStep::OptionGeneration => Some(DecisionStep::CriteriaDefinition),
            DecisionStep::CriteriaDefinition => Some(DecisionStep::OptionEvaluation),
            DecisionStep::OptionEvaluation => Some(DecisionStep::EmotionalInfluence),
            DecisionStep::EmotionalInfluence => Some(DecisionStep::ConsequencePrediction),
            DecisionStep::ConsequencePrediction => Some(DecisionStep::FinalSelection),
            DecisionStep::FinalSelection => Some(DecisionStep::SafetyValidation),
            DecisionStep::SafetyValidation => Some(DecisionStep::Execution),
            DecisionStep::Execution => None,
        }
    }

    /// Capture comprehensive decision context
    async fn capture_decision_context(&self, decision: &Decision) -> Result<bool> {
        // Capture system state, memory state, emotional state, etc.
        // This would integrate with various system components

        debug!("Capturing decision context for decision: {:?}", decision.id);

        // For now, return success
        Ok(true)
    }

    /// Check for decision patterns
    async fn check_decision_patterns(&self, decision: &Decision) -> Result<()> {
        // This would implement pattern detection algorithms
        // For now, basic implementation

        debug!("Checking patterns for decision: {:?}", decision.id);

        Ok(())
    }

    /// Update tracking metrics
    async fn update_tracking_metrics(&self, decision: &Decision, duration: Duration) {
        let mut metrics = self.metrics.write().await;

        let n = metrics.total_decisions_tracked as f64;
        let duration_ms = duration.as_millis() as f64;

        metrics.avg_decision_time_ms = (metrics.avg_decision_time_ms * (n - 1.0) + duration_ms) / n;

        metrics.avg_confidence =
            (metrics.avg_confidence * (n as f32 - 1.0) + decision.confidence) / n as f32;
    }

    /// Store decision in memory
    async fn store_decision_in_memory(&self, decision: &Decision) -> Result<()> {
        if decision.confidence >= self.config.memory_integration.importance_threshold {
            let content = format!(
                "Decision made: {} (confidence: {:.2})",
                decision.context, decision.confidence
            );

            let associations = decision.reasoning.iter().map(|step| step.content.clone()).collect();

            self.memory
                .store(
                    content,
                    associations,
                    MemoryMetadata {
                        source: "decision_tracker".to_string(),
                        tags: vec!["decision".to_string(), "tracking".to_string()],
                        importance: decision.confidence,
                        associations: vec![],
                        context: Some("decision tracking data".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "cognitive".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;
        }

        Ok(())
    }

    /// Analyze decision impact
    async fn analyze_decision_impact(
        &self,
        _decision_id: &DecisionId,
        outcome: &ActualOutcome,
    ) -> Result<ImpactAnalysis> {
        // Comprehensive impact analysis
        Ok(ImpactAnalysis {
            performance_impact: outcome.success_rate,
            cross_system_impact: HashMap::new(),
            long_term_effects: outcome.learning_points.clone(),
            side_effects: outcome.unexpected_consequences.clone(),
            resource_impact: ResourceImpact {
                cpu_usage_change: 0.0,
                memory_usage_change: 0.0,
                network_usage_change: 0.0,
                cost_impact_cents: 0.0,
            },
            goal_impact: outcome.success_rate,
        })
    }
}

/// Decision analytics engine
#[derive(Debug)]
pub struct DecisionAnalytics {
    memory: Arc<CognitiveMemory>,
    decision_history: Arc<RwLock<VecDeque<Decision>>>,
    outcome_history: Arc<RwLock<HashMap<DecisionId, (ActualOutcome, ImpactAnalysis)>>>,
    pattern_cache: Arc<RwLock<HashMap<PatternType, Vec<DecisionPattern>>>>,
    config: DecisionTrackingConfig,
}

impl DecisionAnalytics {
    async fn new(memory: Arc<CognitiveMemory>, config: DecisionTrackingConfig) -> Result<Self> {
        Ok(Self {
            memory,
            decision_history: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.max_history_size,
            ))),
            outcome_history: Arc::new(RwLock::new(HashMap::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }

    async fn record_decision(&self, decision: Decision) -> Result<()> {
        let mut history = self.decision_history.write().await;

        if history.len() >= self.config.max_history_size {
            history.pop_front();
        }

        history.push_back(decision);
        Ok(())
    }

    async fn record_outcome(
        &self,
        decision_id: DecisionId,
        outcome: ActualOutcome,
        impact: ImpactAnalysis,
    ) -> Result<()> {
        self.outcome_history.write().await.insert(decision_id, (outcome, impact));
        Ok(())
    }

    async fn generate_comprehensive_report(&self) -> Result<DecisionAnalyticsReport> {
        let history = self.decision_history.read().await;
        let outcomes = self.outcome_history.read().await;

        // Generate comprehensive analytics
        Ok(DecisionAnalyticsReport {
            total_decisions: history.len() as u64,
            avg_confidence: history.iter().map(|d| d.confidence).sum::<f32>()
                / history.len() as f32,
            avg_decision_time: Duration::from_millis(
                history.iter().map(|d| d.decision_time.as_millis()).sum::<u128>() as u64
                    / history.len() as u64,
            ),
            success_rate: self.calculate_success_rate(&outcomes).await,
            pattern_insights: self.generate_pattern_insights().await?,
            performance_trends: self.analyze_performance_trends(&history).await,
            recommendations: self.generate_recommendations(&history, &outcomes).await,
        })
    }

    async fn calculate_success_rate(
        &self,
        outcomes: &HashMap<DecisionId, (ActualOutcome, ImpactAnalysis)>,
    ) -> f32 {
        if outcomes.is_empty() {
            return 0.0;
        }

        let successful =
            outcomes.values().filter(|(outcome, _)| outcome.success_rate > 0.5).count();

        successful as f32 / outcomes.len() as f32
    }

    async fn generate_pattern_insights(&self) -> Result<Vec<PatternInsight>> {
        // Implement pattern analysis
        Ok(vec![])
    }

    async fn analyze_performance_trends(
        &self,
        _history: &VecDeque<Decision>,
    ) -> Vec<PerformanceTrend> {
        // Implement trend analysis
        vec![]
    }

    async fn generate_recommendations(
        &self,
        _history: &VecDeque<Decision>,
        _outcomes: &HashMap<DecisionId, (ActualOutcome, ImpactAnalysis)>,
    ) -> Vec<DecisionRecommendation> {
        // Implement recommendation generation
        vec![]
    }
}

/// Decision monitoring system
#[derive(Debug)]
pub struct DecisionMonitor {
    event_tx: broadcast::Sender<DecisionTrackingEvent>,
    #[allow(dead_code)]
    config: DecisionTrackingConfig,
    monitoring_active: Arc<RwLock<bool>>,
    active_alerts: Arc<RwLock<Vec<ActiveAlert>>>,
}

impl DecisionMonitor {
    async fn new(
        event_tx: broadcast::Sender<DecisionTrackingEvent>,
        config: DecisionTrackingConfig,
    ) -> Result<Self> {
        Ok(Self {
            event_tx,
            config,
            monitoring_active: Arc::new(RwLock::new(false)),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
        })
    }

    async fn start(&self) -> Result<()> {
        *self.monitoring_active.write().await = true;
        info!("Decision monitoring started");
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        *self.monitoring_active.write().await = false;
        info!("Decision monitoring stopped");
        Ok(())
    }

    async fn get_current_status(&self) -> Result<DecisionMonitoringStatus> {
        Ok(DecisionMonitoringStatus {
            monitoring_active: *self.monitoring_active.read().await,
            active_alerts: self.active_alerts.read().await.clone(),
            system_health: SystemHealth::Good, // Would be calculated
        })
    }
}

/// Decision debugging system
#[derive(Debug)]
pub struct DecisionDebugger {
    memory: Arc<CognitiveMemory>,
    #[allow(dead_code)]
    config: DecisionTrackingConfig,
    debug_traces: Arc<RwLock<HashMap<DecisionId, DebugTrace>>>,
}

impl DecisionDebugger {
    async fn new(memory: Arc<CognitiveMemory>, config: DecisionTrackingConfig) -> Result<Self> {
        Ok(Self { memory, config, debug_traces: Arc::new(RwLock::new(HashMap::new())) })
    }

    async fn get_decision_debug_info(&self, decision_id: DecisionId) -> Result<DecisionDebugInfo> {
        // Implementation for debug info retrieval
        Ok(DecisionDebugInfo {
            decision_id,
            trace_available: false,
            context_snapshot: None,
            step_breakdown: vec![],
            resource_usage: None,
        })
    }
}

/// Decision flow analyzer
#[derive(Debug)]
pub struct DecisionFlowAnalyzer {
    memory: Arc<CognitiveMemory>,
    #[allow(dead_code)]
    config: DecisionTrackingConfig,
    flow_maps: Arc<RwLock<HashMap<String, DecisionFlowMap>>>,
}

impl DecisionFlowAnalyzer {
    async fn new(memory: Arc<CognitiveMemory>, config: DecisionTrackingConfig) -> Result<Self> {
        Ok(Self { memory, config, flow_maps: Arc::new(RwLock::new(HashMap::new())) })
    }

    async fn analyze_session_flow(&self, session_id: &str) -> Result<DecisionFlowAnalysis> {
        // Implementation for flow analysis
        Ok(DecisionFlowAnalysis {
            session_id: session_id.to_string(),
            decision_count: 0,
            flow_efficiency: 0.0,
            bottlenecks: vec![],
            optimization_suggestions: vec![],
        })
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct DecisionAnalyticsReport {
    pub total_decisions: u64,
    pub avg_confidence: f32,
    pub avg_decision_time: Duration,
    pub success_rate: f32,
    pub pattern_insights: Vec<PatternInsight>,
    pub performance_trends: Vec<PerformanceTrend>,
    pub recommendations: Vec<DecisionRecommendation>,
}

#[derive(Debug, Clone)]
pub struct PatternInsight {
    pub pattern_type: PatternType,
    pub description: String,
    pub frequency: f32,
    pub impact: f32,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub metric: String,
    pub trend_direction: TrendDirection,
    pub change_rate: f32,
    pub significance: f32,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Declining,
    Stable,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct DecisionRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub priority: RecommendationPriority,
    pub expected_impact: f32,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    ProcessOptimization,
    CriteriaAdjustment,
    ContextEnhancement,
    ToolIntegration,
    MonitoringImprovement,
}

#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct DecisionMonitoringStatus {
    pub monitoring_active: bool,
    pub active_alerts: Vec<ActiveAlert>,
    pub system_health: SystemHealth,
}

#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: SystemTime,
    pub acknowledged: bool,
}

#[derive(Debug, Clone)]
pub enum SystemHealth {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

#[derive(Debug, Clone)]
pub struct DecisionDebugInfo {
    pub decision_id: DecisionId,
    pub trace_available: bool,
    pub context_snapshot: Option<ContextSnapshot>,
    pub step_breakdown: Vec<StepDebugInfo>,
    pub resource_usage: Option<ResourceUsageInfo>,
}

#[derive(Debug, Clone)]
pub struct ContextSnapshot {
    pub timestamp: SystemTime,
    pub system_state: HashMap<String, String>,
    pub memory_state: String,
    pub emotional_state: String,
}

#[derive(Debug, Clone)]
pub struct StepDebugInfo {
    pub step: DecisionStep,
    pub duration: Duration,
    pub inputs: HashMap<String, String>,
    pub outputs: HashMap<String, String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ResourceUsageInfo {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub network_io: f32,
}

#[derive(Debug, Clone)]
pub struct DecisionFlowAnalysis {
    pub session_id: String,
    pub decision_count: u32,
    pub flow_efficiency: f32,
    pub bottlenecks: Vec<FlowBottleneck>,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FlowBottleneck {
    pub step: DecisionStep,
    pub avg_duration: Duration,
    pub frequency: u32,
    pub impact_score: f32,
}

#[derive(Debug, Clone)]
pub struct DecisionPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub frequency: u32,
    pub confidence: f32,
    pub related_decisions: Vec<DecisionId>,
}

#[derive(Debug, Clone)]
pub struct DecisionFlowMap {
    pub session_id: String,
    pub decisions: Vec<DecisionId>,
    pub flow_graph: HashMap<DecisionId, Vec<DecisionId>>,
    pub timing_data: HashMap<DecisionId, Duration>,
}

#[derive(Debug, Clone)]
pub struct DebugTrace {
    pub decision_id: DecisionId,
    pub trace_data: Vec<TraceEntry>,
    pub context_snapshots: Vec<ContextSnapshot>,
}

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub timestamp: SystemTime,
    pub component: String,
    pub operation: String,
    pub data: HashMap<String, String>,
}
