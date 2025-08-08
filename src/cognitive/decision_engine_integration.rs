//! Decision Engine Integration with Comprehensive Tracking
//!
//! This module provides integration between the existing DecisionEngine
//! and the new comprehensive DecisionTracking system.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{debug, info};

use super::decision_engine::{
    ActualOutcome,
    Decision,
    DecisionCriterion,
    DecisionEngine,
    DecisionId,
    DecisionOption,
};
use super::decision_tracking::{
    DecisionAnalyticsReport,
    DecisionMonitoringStatus,
    DecisionStep,
    DecisionTracker,
    DecisionTrackingConfig,
};
use crate::memory::CognitiveMemory;

/// Enhanced decision engine with comprehensive tracking
pub struct TrackedDecisionEngine {
    /// Core decision engine
    decision_engine: Arc<DecisionEngine>,

    /// Decision tracker
    tracker: Arc<DecisionTracker>,

    /// Integration configuration
    config: TrackedDecisionConfig,
}

/// Configuration for tracked decision engine
#[derive(Debug, Clone)]
pub struct TrackedDecisionConfig {
    /// Enable real-time tracking
    pub enable_tracking: bool,

    /// Track all decision steps
    pub track_all_steps: bool,

    /// Auto-start monitoring
    pub auto_start_monitoring: bool,

    /// Session naming strategy
    pub session_naming: SessionNamingStrategy,
}

impl Default for TrackedDecisionConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            track_all_steps: true,
            auto_start_monitoring: true,
            session_naming: SessionNamingStrategy::ContextBased,
        }
    }
}

/// Session naming strategies
#[derive(Debug, Clone)]
pub enum SessionNamingStrategy {
    /// Based on decision context
    ContextBased,

    /// Timestamp-based
    TimestampBased,

    /// Sequential numbering
    Sequential,

    /// Custom naming function
    Custom(fn(&str) -> String),
}

impl TrackedDecisionEngine {
    /// Create new tracked decision engine
    pub async fn new(
        decision_engine: Arc<DecisionEngine>,
        memory: Arc<CognitiveMemory>,
        config: TrackedDecisionConfig,
    ) -> Result<Self> {
        info!("🔍 Initializing Tracked Decision Engine with comprehensive analytics");

        let trackingconfig = DecisionTrackingConfig {
            enable_monitoring: config.auto_start_monitoring,
            enable_flow_analysis: true,
            enable_profiling: true,
            ..Default::default()
        };

        let tracker = Arc::new(DecisionTracker::new(memory, trackingconfig).await?);

        if config.auto_start_monitoring {
            tracker.start_monitoring().await?;
        }

        Ok(Self { decision_engine, tracker, config })
    }

    /// Make a tracked decision with comprehensive analytics
    pub async fn make_tracked_decision(
        &self,
        context: String,
        options: Vec<DecisionOption>,
        criteria: Vec<DecisionCriterion>,
        session_id: Option<String>,
    ) -> Result<TrackedDecisionResult> {
        let overall_start = Instant::now();

        // Generate session ID if not provided
        let session_id = session_id.unwrap_or_else(|| self.generate_session_id(&context));

        // Create temporary decision ID for tracking
        let decision_id = DecisionId::new();

        // Start decision tracking
        let actual_session_id = if self.config.enable_tracking {
            self.tracker
                .start_decision_tracking(decision_id.clone(), context.clone(), Some(session_id))
                .await?
        } else {
            session_id
        };

        // Track decision steps if enabled
        if self.config.track_all_steps {
            // Context Analysis Step
            let step_start = Instant::now();
            // Simulate context analysis work
            let _ = self.analyze_decision_context(&context, &options).await;
            self.track_step(
                &actual_session_id,
                &decision_id,
                DecisionStep::ContextAnalysis,
                step_start.elapsed(),
            )
            .await?;

            // Option Generation Step
            let step_start = Instant::now();
            // Options are already provided, but we could enhance them here
            let enhanced_options = self.enhance_decision_options(options).await?;
            self.track_step(
                &actual_session_id,
                &decision_id,
                DecisionStep::OptionGeneration,
                step_start.elapsed(),
            )
            .await?;

            // Criteria Definition Step
            let step_start = Instant::now();
            let enhanced_criteria = self.enhance_decision_criteria(criteria).await?;
            self.track_step(
                &actual_session_id,
                &decision_id,
                DecisionStep::CriteriaDefinition,
                step_start.elapsed(),
            )
            .await?;

            // Option Evaluation Step
            let step_start = Instant::now();
            let decision = self
                .decision_engine
                .make_decision(context.clone(), enhanced_options, enhanced_criteria)
                .await?;
            self.track_step(
                &actual_session_id,
                &decision_id,
                DecisionStep::OptionEvaluation,
                step_start.elapsed(),
            )
            .await?;

            // Final Selection Step
            let step_start = Instant::now();
            // Decision already made, but we could add post-processing here
            self.track_step(
                &actual_session_id,
                &decision_id,
                DecisionStep::FinalSelection,
                step_start.elapsed(),
            )
            .await?;

            // Complete tracking
            let total_duration = overall_start.elapsed();
            self.tracker
                .complete_decision_tracking(
                    &actual_session_id,
                    decision_id,
                    decision.clone(),
                    total_duration,
                )
                .await?;

            Ok(TrackedDecisionResult {
                decision,
                session_id: actual_session_id,
                total_duration,
                tracking_enabled: true,
                analytics_available: true,
            })
        } else {
            // Simple decision without step tracking
            let decision = self.decision_engine.make_decision(context, options, criteria).await?;
            let total_duration = overall_start.elapsed();

            if self.config.enable_tracking {
                self.tracker
                    .complete_decision_tracking(
                        &actual_session_id,
                        decision_id,
                        decision.clone(),
                        total_duration,
                    )
                    .await?;
            }

            Ok(TrackedDecisionResult {
                decision,
                session_id: actual_session_id,
                total_duration,
                tracking_enabled: self.config.enable_tracking,
                analytics_available: self.config.enable_tracking,
            })
        }
    }

    /// Record decision outcome with impact analysis
    pub async fn record_tracked_outcome(
        &self,
        decision_id: DecisionId,
        outcome: ActualOutcome,
    ) -> Result<()> {
        if self.config.enable_tracking {
            self.tracker.record_decision_outcome(decision_id, outcome).await?;
        } else {
            // Fallback to decision engine
            self.decision_engine.record_outcome(decision_id, outcome).await?;
        }
        Ok(())
    }

    /// Get comprehensive decision analytics
    pub async fn get_decision_analytics(&self) -> Result<Option<DecisionAnalyticsReport>> {
        if self.config.enable_tracking {
            Ok(Some(self.tracker.get_decision_analytics().await?))
        } else {
            Ok(None)
        }
    }

    /// Get real-time monitoring status
    pub async fn get_monitoring_status(&self) -> Result<Option<DecisionMonitoringStatus>> {
        if self.config.enable_tracking {
            Ok(Some(self.tracker.get_monitoring_status().await?))
        } else {
            Ok(None)
        }
    }

    /// Get the underlying decision engine for advanced operations
    pub fn decision_engine(&self) -> &Arc<DecisionEngine> {
        &self.decision_engine
    }

    /// Get the decision tracker for advanced analytics
    pub fn tracker(&self) -> Option<&Arc<DecisionTracker>> {
        if self.config.enable_tracking { Some(&self.tracker) } else { None }
    }

    /// Enable or disable tracking
    pub async fn set_tracking_enabled(&mut self, enabled: bool) -> Result<()> {
        self.config.enable_tracking = enabled;

        if enabled && self.config.auto_start_monitoring {
            self.tracker.start_monitoring().await?;
        } else if !enabled {
            self.tracker.stop_monitoring().await?;
        }

        Ok(())
    }

    /// Helper: Generate session ID based on strategy
    fn generate_session_id(&self, context: &str) -> String {
        match &self.config.session_naming {
            SessionNamingStrategy::ContextBased => {
                let hash = context
                    .chars()
                    .take(20)
                    .collect::<String>()
                    .replace(' ', "_")
                    .replace('\n', "")
                    .to_lowercase();
                format!("decision_{}", hash)
            }
            SessionNamingStrategy::TimestampBased => {
                format!("decision_{}", chrono::Utc::now().timestamp_millis())
            }
            SessionNamingStrategy::Sequential => {
                // This would need a counter in real implementation
                format!("decision_seq_{}", chrono::Utc::now().timestamp_millis())
            }
            SessionNamingStrategy::Custom(func) => func(context),
        }
    }

    /// Helper: Track a decision step
    async fn track_step(
        &self,
        session_id: &str,
        decision_id: &DecisionId,
        step: DecisionStep,
        duration: Duration,
    ) -> Result<()> {
        if self.config.enable_tracking {
            self.tracker
                .track_decision_step(session_id, decision_id.clone(), step, duration)
                .await?;
        }
        Ok(())
    }

    /// Analyze decision context (placeholder for actual implementation)
    async fn analyze_decision_context(
        &self,
        _context: &str,
        options: &[DecisionOption],
    ) -> Result<ContextAnalysis> {
        debug!("Analyzing decision context: {} options", options.len());

        // This would perform actual context analysis
        Ok(ContextAnalysis {
            complexity_score: options.len() as f32 * 0.1,
            urgency_level: 0.5,   // Would be determined from context
            stakeholder_count: 1, // Would be extracted from context
        })
    }

    /// Enhance decision options (placeholder for actual implementation)
    async fn enhance_decision_options(
        &self,
        options: Vec<DecisionOption>,
    ) -> Result<Vec<DecisionOption>> {
        debug!("Enhancing {} decision options", options.len());

        // This would perform actual option enhancement
        // For now, just return the original options
        Ok(options)
    }

    /// Enhance decision criteria (placeholder for actual implementation)
    async fn enhance_decision_criteria(
        &self,
        criteria: Vec<DecisionCriterion>,
    ) -> Result<Vec<DecisionCriterion>> {
        debug!("Enhancing {} decision criteria", criteria.len());

        // This would perform actual criteria enhancement
        // For now, just return the original criteria
        Ok(criteria)
    }
}

/// Result of a tracked decision
#[derive(Debug, Clone)]
pub struct TrackedDecisionResult {
    /// The actual decision made
    pub decision: Decision,

    /// Session ID for this decision
    pub session_id: String,

    /// Total time taken for decision
    pub total_duration: Duration,

    /// Whether tracking was enabled
    pub tracking_enabled: bool,

    /// Whether analytics are available
    pub analytics_available: bool,
}

/// Context analysis result
#[derive(Debug, Clone)]
pub struct ContextAnalysis {
    pub complexity_score: f32,
    pub urgency_level: f32,
    pub stakeholder_count: u32,
}

/// Factory for creating tracked decision engines
pub struct TrackedDecisionFactory;

impl TrackedDecisionFactory {
    /// Create a new tracked decision engine with default configuration
    pub async fn create_default(
        decision_engine: Arc<DecisionEngine>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<TrackedDecisionEngine> {
        TrackedDecisionEngine::new(decision_engine, memory, TrackedDecisionConfig::default()).await
    }

    /// Create a new tracked decision engine with monitoring disabled
    pub async fn create_minimal(
        decision_engine: Arc<DecisionEngine>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<TrackedDecisionEngine> {
        let config = TrackedDecisionConfig {
            enable_tracking: true,
            track_all_steps: false,
            auto_start_monitoring: false,
            session_naming: SessionNamingStrategy::TimestampBased,
        };

        TrackedDecisionEngine::new(decision_engine, memory, config).await
    }

    /// Create a new tracked decision engine with full analytics
    pub async fn create_full_analytics(
        decision_engine: Arc<DecisionEngine>,
        memory: Arc<CognitiveMemory>,
    ) -> Result<TrackedDecisionEngine> {
        let config = TrackedDecisionConfig {
            enable_tracking: true,
            track_all_steps: true,
            auto_start_monitoring: true,
            session_naming: SessionNamingStrategy::ContextBased,
        };

        TrackedDecisionEngine::new(decision_engine, memory, config).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::cognitive::decision_engine::{CriterionType, DecisionConfig, OptimizationType};
    use crate::cognitive::{EmotionalCore, LokiCharacter, NeuroProcessor};
    use crate::memory::{CognitiveMemory, MemoryConfig};
    use crate::safety::ActionValidator;
    use crate::tools::IntelligentToolManager;

    #[tokio::test]
    async fn test_tracked_decision_creation() {
        // Create required components
        let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await.unwrap());

        // Create minimal components for DecisionEngine
        let cache = Arc::new(crate::memory::simd_cache::SimdSmartCache::new(
            crate::memory::SimdCacheConfig::default(),
        ));
        let neural_processor = Arc::new(NeuroProcessor::new(cache).await.unwrap());
        let emotional_core = Arc::new(
            EmotionalCore::new(memory.clone(), crate::cognitive::EmotionalConfig::default())
                .await
                .unwrap(),
        );
        let character = Arc::new(LokiCharacter::new_minimal().await.unwrap());
        let tool_manager = Arc::new(IntelligentToolManager::new_minimal().await.unwrap());
        let safety_validator = Arc::new(ActionValidator::new_minimal().await.unwrap());

        let decision_engine = Arc::new(
            DecisionEngine::new(
                neural_processor,
                emotional_core,
                memory.clone(),
                character,
                tool_manager,
                safety_validator,
                DecisionConfig::default(),
            )
            .await
            .unwrap(),
        );

        let tracked_engine =
            TrackedDecisionFactory::create_default(decision_engine, memory).await.unwrap();

        assert!(tracked_engine.config.enable_tracking);
    }

    #[tokio::test]
    async fn test_tracked_decision_execution() {
        // Setup similar to above test
        let memory = Arc::new(CognitiveMemory::new(MemoryConfig::default()).await.unwrap());

        let cache = Arc::new(crate::memory::simd_cache::SimdSmartCache::new(
            crate::memory::SimdCacheConfig::default(),
        ));
        let neural_processor = Arc::new(NeuroProcessor::new(cache).await.unwrap());
        let emotional_core = Arc::new(
            EmotionalCore::new(memory.clone(), crate::cognitive::EmotionalConfig::default())
                .await
                .unwrap(),
        );
        let character = Arc::new(LokiCharacter::new_minimal().await.unwrap());
        let tool_manager = Arc::new(IntelligentToolManager::new_minimal().await.unwrap());
        let safety_validator = Arc::new(ActionValidator::new_minimal().await.unwrap());

        let decision_engine = Arc::new(
            DecisionEngine::new(
                neural_processor,
                emotional_core,
                memory.clone(),
                character,
                tool_manager,
                safety_validator,
                DecisionConfig::default(),
            )
            .await
            .unwrap(),
        );

        let tracked_engine =
            TrackedDecisionFactory::create_minimal(decision_engine, memory).await.unwrap();

        // Create test decision
        let options = vec![DecisionOption {
            id: "option1".to_string(),
            description: "Test option 1".to_string(),
            scores: HashMap::from([("test".to_string(), 0.8)]),
            feasibility: 0.9,
            risk_level: 0.1,
            emotional_appeal: 0.5,
            expected_outcome: "Positive test outcome".to_string(),
            confidence: 0.8,
            resources_required: vec!["test_resource".to_string()],
            time_estimate: std::time::Duration::from_secs(300),
            success_probability: 0.85,
        }];

        let criteria = vec![DecisionCriterion {
            name: "test".to_string(),
            weight: 1.0,
            criterion_type: CriterionType::Quantitative,
            optimization: OptimizationType::Maximize,
        }];

        let result = tracked_engine
            .make_tracked_decision("Test decision context".to_string(), options, criteria, None)
            .await
            .unwrap();

        assert!(result.tracking_enabled);
        assert!(!result.session_id.is_empty());
        assert!(result.total_duration.as_millis() > 0);
    }
}
