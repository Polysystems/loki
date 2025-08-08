//! Enhanced Decision Engine for Autonomous Decision Making
//!
//! This module implements a sophisticated decision-making system that combines
//! multi-criteria analysis, emotional weighting, archetypal intelligence,
//! tool-informed decisions, and consequence prediction.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info, warn};

use crate::cognitive::character::{ArchetypalForm, LokiCharacter};
use crate::cognitive::{EmotionalCore, NeuroProcessor, Thought, ThoughtId, ThoughtType};
use crate::memory::{CognitiveMemory, MemoryId, MemoryItem, MemoryMetadata};
use crate::safety::{ActionType, ActionValidator};
use crate::tools::intelligent_manager::{
    IntelligentToolManager,
    MemoryIntegration,
    ResultType,
    ToolRequest,
};

/// Unique identifier for decisions
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct DecisionId(String);

impl DecisionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for DecisionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Decision criterion for evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionCriterion {
    pub name: String,
    pub weight: f32, // Importance weight (0.0-1.0)
    pub criterion_type: CriterionType,
    pub optimization: OptimizationType,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CriterionType {
    Quantitative,  // Numerical values
    Qualitative,   // Categorical values
    Boolean,       // Yes/No
    Probabilistic, // Probability-based
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimizationType {
    Maximize,       // Higher is better
    Minimize,       // Lower is better
    Target(f32),    // Closer to target is better
    Satisfice(f32), // Must meet threshold
}

/// Decision option to be evaluated
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionOption {
    pub id: String,
    pub description: String,
    pub scores: HashMap<String, f32>, // Criterion name -> score
    pub feasibility: f32,             // 0.0-1.0
    pub risk_level: f32,              // 0.0-1.0
    pub emotional_appeal: f32,        // -1.0 to 1.0
    pub expected_outcome: String,
    pub confidence: f32,
    pub resources_required: Vec<String>,
    pub time_estimate: Duration,
    pub success_probability: f32,
}

/// Predicted outcome of a decision
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictedOutcome {
    pub description: String,
    pub probability: f32,
    pub impact: f32, // -1.0 to 1.0
    pub time_horizon: Duration,
    pub confidence: f32,
}

/// Reasoning step in decision process
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_type: ReasoningType,
    pub content: String,
    pub supporting_thoughts: Vec<ThoughtId>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReasoningType {
    Analysis,
    Comparison,
    Elimination,
    Synthesis,
    Intuition,
}

/// Complete decision with all information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub id: DecisionId,
    pub context: String,
    pub options: Vec<DecisionOption>,
    pub criteria: Vec<DecisionCriterion>,
    pub selected: Option<DecisionOption>,
    pub confidence: f32,
    pub reasoning: Vec<ReasoningStep>,
    pub reasoning_chain: Vec<String>, // Simple reasoning chain
    pub predicted_outcomes: Vec<PredictedOutcome>,
    pub decision_time: Duration,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

impl Default for Decision {
    fn default() -> Self {
        Self {
            id: DecisionId::new(),
            context: String::new(),
            options: Vec::new(),
            criteria: Vec::new(),
            selected: None,
            confidence: 0.0,
            reasoning: Vec::new(),
            reasoning_chain: Vec::new(),
            predicted_outcomes: Vec::new(),
            decision_time: Duration::from_secs(0),
            timestamp: Instant::now(),
        }
    }
}

/// Emotional weight matrix for decision criteria
#[derive(Clone, Debug)]
pub struct EmotionalWeightMatrix {
    /// How each emotion affects each criterion
    weights: HashMap<String, HashMap<String, f32>>,
}

impl EmotionalWeightMatrix {
    pub fn new() -> Self {
        Self { weights: HashMap::new() }
    }

    /// Set emotional weight for a criterion
    pub fn set_weight(&mut self, emotion: &str, criterion: &str, weight: f32) {
        self.weights
            .entry(emotion.to_string())
            .or_insert_with(HashMap::new)
            .insert(criterion.to_string(), weight);
    }

    /// Get emotional adjustment for a criterion
    pub fn get_adjustment(&self, emotion: &str, criterion: &str, intensity: f32) -> f32 {
        self.weights
            .get(emotion)
            .and_then(|criteria| criteria.get(criterion))
            .map(|&weight| weight * intensity)
            .unwrap_or(0.0)
    }
}

/// Decision history for learning
#[derive(Clone, Debug)]
pub struct DecisionHistory {
    decisions: Arc<RwLock<BTreeMap<DecisionId, Decision>>>,
    outcomes: Arc<RwLock<HashMap<DecisionId, ActualOutcome>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActualOutcome {
    pub decision_id: DecisionId,
    pub success_rate: f32,
    pub unexpected_consequences: Vec<String>,
    pub learning_points: Vec<String>,
}

#[derive(Debug)]
/// Consequence predictor using neural pathways
pub struct ConsequencePredictor {
    neural_processor: Arc<NeuroProcessor>,
    memory: Arc<CognitiveMemory>,
    prediction_models: Arc<RwLock<HashMap<String, PredictionModel>>>,
}

#[derive(Clone, Debug)]
struct PredictionModel {
    pattern_weights: Vec<f32>,
    accuracy: f32,
    last_updated: Instant,
}

/// Configuration for decision engine
#[derive(Clone, Debug)]
pub struct DecisionConfig {
    /// Maximum time allowed for decision
    pub max_decision_time: Duration,

    /// Minimum confidence threshold
    pub min_confidence: f32,

    /// Enable emotional influence
    pub use_emotions: bool,

    /// Risk tolerance (0.0-1.0)
    pub risk_tolerance: f32,

    /// Decision history size
    pub history_size: usize,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            max_decision_time: Duration::from_secs(5),
            min_confidence: 0.6,
            use_emotions: true,
            risk_tolerance: 0.5,
            history_size: 1000,
        }
    }
}

#[derive(Debug, Clone)]
/// Main enhanced decision engine with archetypal intelligence
pub struct DecisionEngine {
    /// Decision criteria
    criteria: Arc<RwLock<Vec<DecisionCriterion>>>,

    /// Emotional weight matrix
    emotional_weights: Arc<RwLock<EmotionalWeightMatrix>>,

    /// Consequence predictor
    consequence_predictor: Arc<ConsequencePredictor>,

    /// Decision history
    history: Arc<DecisionHistory>,

    /// Neural processor reference
    neural_processor: Arc<NeuroProcessor>,

    /// Emotional core reference
    emotional_core: Arc<EmotionalCore>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Character system for archetypal decision making
    character: Arc<LokiCharacter>,

    /// Tool manager for informed decisions
    pub tool_manager: Arc<IntelligentToolManager>,

    /// Safety validator
    safety_validator: Arc<ActionValidator>,

    /// Archetypal decision patterns
    archetypal_patterns: Arc<RwLock<HashMap<String, ArchetypalDecisionPattern>>>,

    /// Configuration
    config: DecisionConfig,

    /// Decision channel
    decision_tx: mpsc::Sender<Decision>,

    /// Statistics
    stats: Arc<RwLock<DecisionStats>>,
}

/// Archetypal decision patterns for different forms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypalDecisionPattern {
    /// Archetypal form identifier
    pub form_id: String,

    /// Risk tolerance for this form
    pub risk_tolerance: f32,

    /// Decision speed preference
    pub decision_speed: DecisionSpeed,

    /// Information gathering style
    pub information_style: InformationGatheringStyle,

    /// Preferred criteria weights
    pub criteria_weights: HashMap<String, f32>,

    /// Tool usage preferences for decisions
    pub tool_preferences: Vec<String>,

    /// Decision context modifiers
    pub context_modifiers: Vec<ContextModifier>,
}

/// Decision speed preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionSpeed {
    /// Quick intuitive decisions
    Rapid,

    /// Balanced consideration
    Moderate,

    /// Deep contemplative decisions
    Deliberate,

    /// Adaptive based on context
    Adaptive,
}

/// Information gathering styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InformationGatheringStyle {
    /// Minimal information, trust intuition
    Intuitive,

    /// Gather diverse perspectives
    Exploratory,

    /// Deep analysis of key factors
    Analytical,

    /// Pattern-based from memory
    ExperienceBased,
}

impl std::fmt::Display for InformationGatheringStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InformationGatheringStyle::Intuitive => write!(f, "Intuitive"),
            InformationGatheringStyle::Exploratory => write!(f, "Exploratory"),
            InformationGatheringStyle::Analytical => write!(f, "Analytical"),
            InformationGatheringStyle::ExperienceBased => write!(f, "ExperienceBased"),
        }
    }
}

/// Context modifiers for decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextModifier {
    /// Context trigger
    pub trigger: String,

    /// Risk tolerance adjustment
    pub risk_adjustment: f32,

    /// Urgency modifier
    pub urgency_modifier: f32,

    /// Preferred tools for this context
    pub context_tools: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct DecisionStats {
    pub decisions_made: u64,
    pub avg_decision_time_ms: f64,
    pub avg_confidence: f32,
    pub decisions_revised: u64,
    pub high_confidence_decisions: u64,
}

/// Comprehensive timing tracking for decision phases
#[derive(Debug, Clone, Serialize)]
pub struct DecisionPhaseTimings {
    pub archetypal_form_ms: u64,
    pub criteria_adjustment_ms: u64,
    pub information_gathering_ms: u64,
    pub options_enhancement_ms: u64,
    pub decision_making_ms: u64,
    pub storage_ms: u64,
    pub execution_ms: u64,
    pub total_decision_ms: u64,
}

impl DecisionPhaseTimings {
    pub fn new() -> Self {
        Self {
            archetypal_form_ms: 0,
            criteria_adjustment_ms: 0,
            information_gathering_ms: 0,
            options_enhancement_ms: 0,
            decision_making_ms: 0,
            storage_ms: 0,
            execution_ms: 0,
            total_decision_ms: 0,
        }
    }
}

/// Performance metrics for decision analysis
#[derive(Debug, Clone, Serialize)]
pub struct DecisionPerformanceMetrics {
    pub timing: DecisionPhaseTimings,
    pub efficiency_score: f32,
    pub bottleneck_phase: String,
    pub tool_utilization: f32,
    pub memory_efficiency: f32,
    pub cognitive_load: f32,
    pub optimization_suggestions: Vec<String>,
}

/// Performance analysis structure for decision optimization
#[derive(Debug, Clone)]
struct DecisionPerformanceAnalysis {
    efficiency_rating: String,
    bottleneck_phase: String,
    optimization_suggestions: Vec<String>,
    complexity_score: f32,
    tool_effectiveness: f32,
    archetypal_alignment: f32,
}

impl DecisionEngine {
    pub async fn new(
        neural_processor: Arc<NeuroProcessor>,
        emotional_core: Arc<EmotionalCore>,
        memory: Arc<CognitiveMemory>,
        character: Arc<LokiCharacter>,
        tool_manager: Arc<IntelligentToolManager>,
        safety_validator: Arc<ActionValidator>,
        config: DecisionConfig,
    ) -> Result<Self> {
        info!("Initializing Enhanced Decision Engine with archetypal intelligence");

        let (decision_tx, _) = mpsc::channel(100);

        // Initialize emotional weight matrix with defaults
        let mut emotional_weights = EmotionalWeightMatrix::new();

        // Set default emotional influences on criteria
        emotional_weights.set_weight("Joy", "risk", 0.3);
        emotional_weights.set_weight("Fear", "risk", -0.5);
        emotional_weights.set_weight("Trust", "feasibility", 0.4);
        emotional_weights.set_weight("Anger", "speed", 0.6);

        let consequence_predictor = Arc::new(ConsequencePredictor {
            neural_processor: neural_processor.clone(),
            memory: memory.clone(),
            prediction_models: Arc::new(RwLock::new(HashMap::new())),
        });

        let history = Arc::new(DecisionHistory {
            decisions: Arc::new(RwLock::new(BTreeMap::new())),
            outcomes: Arc::new(RwLock::new(HashMap::new())),
        });

        let engine = Self {
            criteria: Arc::new(RwLock::new(Vec::new())),
            emotional_weights: Arc::new(RwLock::new(emotional_weights)),
            consequence_predictor,
            history,
            neural_processor,
            emotional_core,
            memory,
            character,
            tool_manager,
            safety_validator,
            archetypal_patterns: Arc::new(RwLock::new(HashMap::new())),
            config,
            decision_tx,
            stats: Arc::new(RwLock::new(DecisionStats::default())),
        };

        // Initialize archetypal patterns
        engine.initialize_archetypal_patterns().await?;

        Ok(engine)
    }

    /// Make a decision given a context and options
    pub async fn make_decision(
        &self,
        context: String,
        options: Vec<DecisionOption>,
        criteria: Vec<DecisionCriterion>,
    ) -> Result<Decision> {
        let start = Instant::now();
        info!("Making decision with {} options", options.len());

        if options.is_empty() {
            return Err(anyhow!("No options provided for decision"));
        }

        // Store criteria
        *self.criteria.write().await = criteria.clone();

        // Create decision structure
        let mut decision = Decision {
            id: DecisionId::new(),
            context: context.clone(),
            options: options.clone(),
            criteria: criteria.clone(), // Clone criteria here
            ..Default::default()
        };

        // Step 1: Analyze each option
        let mut option_scores = HashMap::new();
        for option in &options {
            let score = self.evaluate_option(option).await?;
            option_scores.insert(option.id.clone(), score);

            decision.reasoning.push(ReasoningStep {
                step_type: ReasoningType::Analysis,
                content: format!(
                    "Evaluated option '{}' with score {:.2}",
                    option.description, score
                ),
                supporting_thoughts: Vec::new(),
                confidence: 0.8,
            });
        }

        // Step 2: Apply emotional influence if enabled
        if self.config.use_emotions {
            let emotional_adjustments = self.apply_emotional_influence(&option_scores).await?;

            for (option_id, adjustment) in emotional_adjustments {
                if let Some(score) = option_scores.get_mut(&option_id) {
                    *score += adjustment;

                    decision.reasoning.push(ReasoningStep {
                        step_type: ReasoningType::Intuition,
                        content: format!("Emotional adjustment of {:.2} applied", adjustment),
                        supporting_thoughts: Vec::new(),
                        confidence: 0.7,
                    });
                }
            }
        }

        // Step 3: Predict consequences
        for option in &options {
            let outcomes = self.predict_consequences(option, &context).await?;
            decision.predicted_outcomes.extend(outcomes);
        }

        // Step 4: Select best option
        let (best_option_id, best_score) = option_scores
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow!("No valid scores calculated"))?;

        let selected_option = options
            .iter()
            .find(|o| &o.id == best_option_id)
            .cloned()
            .ok_or_else(|| anyhow!("Selected option not found"))?;

        decision.selected = Some(selected_option);
        decision.confidence = self.calculate_confidence(best_score, &option_scores);
        decision.decision_time = start.elapsed();

        // Step 5: Final reasoning
        decision.reasoning.push(ReasoningStep {
            step_type: ReasoningType::Synthesis,
            content: format!(
                "Selected option with score {:.2} and confidence {:.2}",
                best_score, decision.confidence
            ),
            supporting_thoughts: Vec::new(),
            confidence: decision.confidence,
        });

        // Build reasoning chain
        decision.reasoning_chain =
            decision.reasoning.iter().map(|step| step.content.clone()).collect();

        // Store in history
        self.history.decisions.write().await.insert(decision.id.clone(), decision.clone());

        // Update statistics
        self.update_stats(&decision).await;

        // Send decision notification
        let _ = self.decision_tx.send(decision.clone()).await;

        Ok(decision)
    }

    /// Evaluate a single option against all criteria
    async fn evaluate_option(&self, option: &DecisionOption) -> Result<f32> {
        let criteria = self.criteria.read().await;
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for criterion in criteria.iter() {
            let score = option.scores.get(&criterion.name).unwrap_or(&0.0);

            // Apply optimization type
            let adjusted_score = match &criterion.optimization {
                OptimizationType::Maximize => *score,
                OptimizationType::Minimize => 1.0 - score,
                OptimizationType::Target(target) => 1.0 - (score - target).abs(),
                OptimizationType::Satisfice(threshold) => {
                    if score >= threshold {
                        1.0
                    } else {
                        score / threshold
                    }
                }
            };

            weighted_sum += adjusted_score * criterion.weight;
            total_weight += criterion.weight;
        }

        // Consider feasibility and risk
        let feasibility_factor = option.feasibility;
        let risk_factor = 1.0 - (option.risk_level * (1.0 - self.config.risk_tolerance));

        let final_score = if total_weight > 0.0 {
            (weighted_sum / total_weight) * feasibility_factor * risk_factor
        } else {
            0.5 // Default score if no criteria
        };

        Ok(final_score.clamp(0.0, 1.0))
    }

    /// Apply emotional influence to option scores
    async fn apply_emotional_influence(
        &self,
        option_scores: &HashMap<String, f32>,
    ) -> Result<HashMap<String, f32>> {
        let emotional_state = self.emotional_core.get_emotional_state().await;
        let emotional_weights = self.emotional_weights.read().await;
        let criteria = self.criteria.read().await;

        let mut adjustments = HashMap::new();

        // Get primary emotion as string
        let emotion_str = format!("{:?}", emotional_state.primary.emotion);

        for (option_id, _) in option_scores {
            let mut total_adjustment = 0.0;

            for criterion in criteria.iter() {
                let adjustment = emotional_weights.get_adjustment(
                    &emotion_str,
                    &criterion.name,
                    emotional_state.primary.intensity,
                );
                total_adjustment += adjustment * criterion.weight;
            }

            adjustments.insert(option_id.clone(), total_adjustment);
        }

        Ok(adjustments)
    }

    /// Predict consequences of an option
    async fn predict_consequences(
        &self,
        option: &DecisionOption,
        context: &str,
    ) -> Result<Vec<PredictedOutcome>> {
        // Context-aware outcome prediction using situational factors and historical
        // patterns

        // Parse context for situational factors
        let context_lower = context.to_lowercase();
        let is_urgent = context_lower.contains("urgent")
            || context_lower.contains("critical")
            || context_lower.contains("immediate")
            || context_lower.contains("deadline");
        let is_social = context_lower.contains("people")
            || context_lower.contains("team")
            || context_lower.contains("relationship")
            || context_lower.contains("social");
        let is_technical = context_lower.contains("code")
            || context_lower.contains("system")
            || context_lower.contains("technical")
            || context_lower.contains("engineering");
        let is_creative = context_lower.contains("creative")
            || context_lower.contains("design")
            || context_lower.contains("artistic")
            || context_lower.contains("innovative");
        let is_learning = context_lower.contains("learn")
            || context_lower.contains("study")
            || context_lower.contains("education")
            || context_lower.contains("knowledge");

        // Use neural pathways to process contextual information
        let thought = Thought {
            id: ThoughtId::new(),
            content: format!(
                "Context-aware prediction for: {} in situation: {}",
                option.description, context
            ),
            thought_type: ThoughtType::Analysis,
            ..Default::default()
        };

        self.neural_processor.process_thought(&thought).await?;

        let mut outcomes = Vec::new();

        // Base success probability adjusted by context
        let mut base_success_prob = option.feasibility * (1.0 - option.risk_level);

        // Context-specific adjustments
        if is_urgent {
            // Urgent situations increase risk and reduce success probability
            base_success_prob *= 0.8;
        }

        if is_social && option.emotional_appeal > 0.5 {
            // Social contexts benefit from emotionally appealing options
            base_success_prob *= 1.2;
        } else if is_social && option.emotional_appeal < 0.0 {
            // Negative emotional appeal hurts in social contexts
            base_success_prob *= 0.7;
        }

        if is_technical && option.scores.get("technical_soundness").unwrap_or(&0.5) > &0.7 {
            // Technical contexts reward technical excellence
            base_success_prob *= 1.3;
        }

        if is_creative && option.scores.get("creativity").unwrap_or(&0.5) > &0.6 {
            // Creative contexts benefit from novel approaches
            base_success_prob *= 1.25;
        }

        // Primary success outcome with contextual factors
        let success_description = if is_urgent {
            format!("Urgent execution of '{}' meets deadline requirements", option.description)
        } else if is_social {
            format!("Social implementation of '{}' gains stakeholder support", option.description)
        } else if is_technical {
            format!(
                "Technical solution '{}' integrates successfully with existing systems",
                option.description
            )
        } else if is_creative {
            format!(
                "Creative approach '{}' generates positive reception and engagement",
                option.description
            )
        } else {
            format!("Standard execution of '{}' achieves intended outcomes", option.description)
        };

        outcomes.push(PredictedOutcome {
            description: success_description,
            probability: base_success_prob.clamp(0.1, 0.95),
            impact: if is_urgent { 0.9 } else { 0.8 },
            time_horizon: if is_urgent {
                Duration::from_secs(1800)
            } else {
                Duration::from_secs(3600)
            },
            confidence: 0.75,
        });

        // Context-specific risk scenarios
        if option.risk_level > 0.3 {
            let risk_probability = option.risk_level * if is_urgent { 1.4 } else { 1.0 };

            let risk_description = match (is_urgent, is_social, is_technical) {
                (true, _, _) => {
                    "Time pressure leads to rushed execution and quality issues".to_string()
                }
                (_, true, _) => "Social resistance emerges, requiring additional relationship \
                                 management"
                    .to_string(),
                (_, _, true) => {
                    "Technical complexity reveals unforeseen integration challenges".to_string()
                }
                _ => "Unexpected complications require additional resources and time".to_string(),
            };

            outcomes.push(PredictedOutcome {
                description: risk_description,
                probability: risk_probability.clamp(0.1, 0.8),
                impact: if is_urgent { -0.7 } else { -0.5 },
                time_horizon: Duration::from_secs(if is_urgent { 900 } else { 1800 }),
                confidence: 0.65,
            });
        }

        // Context-specific opportunity outcomes
        if option.feasibility > 0.7 && option.emotional_appeal > 0.3 {
            let opportunity_scenarios = match (is_creative, is_learning, is_social) {
                (true, _, _) => (
                    "Creative success sparks additional innovative opportunities".to_string(),
                    0.6,
                    0.8,
                ),
                (_, true, _) => (
                    "Learning outcomes exceed expectations, building valuable expertise"
                        .to_string(),
                    0.7,
                    0.7,
                ),
                (_, _, true) => (
                    "Social success creates network effects and future collaboration opportunities"
                        .to_string(),
                    0.5,
                    0.9,
                ),
                _ => (
                    "Success creates momentum for related improvements and optimizations"
                        .to_string(),
                    0.5,
                    0.6,
                ),
            };

            outcomes.push(PredictedOutcome {
                description: opportunity_scenarios.0,
                probability: opportunity_scenarios.1,
                impact: opportunity_scenarios.2,
                time_horizon: Duration::from_secs(7200), // 2 hours
                confidence: 0.6,
            });
        }

        // Learning and adaptation outcomes based on context
        if is_learning || option.scores.get("learning_potential").unwrap_or(&0.3) > &0.5 {
            outcomes.push(PredictedOutcome {
                description: "Experience generates valuable insights and improved future \
                              decision-making"
                    .to_string(),
                probability: 0.8,
                impact: 0.6,
                time_horizon: Duration::from_secs(86400), // 1 day
                confidence: 0.8,
            });
        }

        // Resource utilization outcomes
        if option.scores.get("resource_efficiency").unwrap_or(&0.5) < &0.4 {
            let resource_strain_probability = if is_urgent { 0.7 } else { 0.5 };
            outcomes.push(PredictedOutcome {
                description: "Resource demands strain capacity, potentially affecting other \
                              priorities"
                    .to_string(),
                probability: resource_strain_probability,
                impact: -0.4,
                time_horizon: Duration::from_secs(3600),
                confidence: 0.7,
            });
        }

        // Query memory for similar historical patterns
        if let Ok(similar_decisions) =
            self.memory.retrieve_similar("decision outcomes similar context", 5).await
        {
            if !similar_decisions.is_empty() {
                // Extract patterns from historical data
                let historical_success_rate = 0.7; // Simplified - would analyze actual patterns

                outcomes.push(PredictedOutcome {
                    description: "Historical pattern analysis suggests outcome alignment with \
                                  past similar decisions"
                        .to_string(),
                    probability: historical_success_rate,
                    impact: 0.5,
                    time_horizon: Duration::from_secs(1800),
                    confidence: 0.65,
                });
            }
        }

        // Sort outcomes by probability * impact for relevance
        outcomes.sort_by(|a, b| {
            let score_a = a.probability * a.impact.abs();
            let score_b = b.probability * b.impact.abs();
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to most significant outcomes
        outcomes.truncate(6);

        debug!(
            "Generated {} context-aware outcome predictions for option: {}",
            outcomes.len(),
            option.description
        );

        Ok(outcomes)
    }

    /// Calculate decision confidence
    fn calculate_confidence(&self, best_score: &f32, all_scores: &HashMap<String, f32>) -> f32 {
        if all_scores.len() == 1 {
            return best_score * 0.8; // Single option, confidence based on score
        }

        // Calculate separation between best and second-best
        let mut scores: Vec<f32> = all_scores.values().cloned().collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let separation = if scores.len() > 1 { scores[0] - scores[1] } else { 1.0 };

        // Confidence based on score and separation
        let confidence = (best_score + separation) / 2.0;
        confidence.clamp(0.0, 1.0)
    }

    /// Update decision statistics
    async fn update_stats(&self, decision: &Decision) {
        let mut stats = self.stats.write().await;
        stats.decisions_made += 1;

        let n = stats.decisions_made as f64;
        let decision_time_ms = decision.decision_time.as_millis() as f64;

        stats.avg_decision_time_ms =
            (stats.avg_decision_time_ms * (n - 1.0) + decision_time_ms) / n;

        stats.avg_confidence =
            (stats.avg_confidence * (n as f32 - 1.0) + decision.confidence) / n as f32;

        if decision.confidence >= 0.8 {
            stats.high_confidence_decisions += 1;
        }
    }

    /// Record actual outcome for learning
    pub async fn record_outcome(
        &self,
        decision_id: DecisionId,
        outcome: ActualOutcome,
    ) -> Result<()> {
        self.history.outcomes.write().await.insert(decision_id.clone(), outcome);

        // Store in memory for future reference
        self.memory
            .store(
                format!("Decision outcome recorded"),
                vec![],
                MemoryMetadata {
                    source: "decision_engine".to_string(),
                    tags: vec!["outcome".to_string(), "learning".to_string()],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("decision outcome recording".to_string()),
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

        Ok(())
    }

    /// Revise a previous decision based on new information
    pub async fn revise_decision(
        &self,
        decision_id: DecisionId,
        new_info: String,
    ) -> Result<Decision> {
        let decisions = self.history.decisions.read().await;
        let original = decisions.get(&decision_id).ok_or_else(|| anyhow!("Decision not found"))?;

        info!("Revising decision {} with new information", decision_id.0);

        // Re-evaluate with new context
        let revised_context = format!("{}\nNew information: {}", original.context, new_info);
        let revised_decision = self
            .make_decision(
                revised_context,
                original.options.clone(),
                self.criteria.read().await.clone(),
            )
            .await?;

        // Update stats
        self.stats.write().await.decisions_revised += 1;

        Ok(revised_decision)
    }

    /// Get decision statistics
    pub async fn get_stats(&self) -> DecisionStats {
        self.stats.read().await.clone()
    }

    /// Initialize archetypal patterns
    async fn initialize_archetypal_patterns(&self) -> Result<()> {
        debug!("Initializing archetypal decision patterns");

        let mut patterns = self.archetypal_patterns.write().await;

        // Mischievous Helper patterns
        patterns.insert(
            "Mischievous Helper".to_string(),
            ArchetypalDecisionPattern {
                form_id: "mischievous_helper".to_string(),
                risk_tolerance: 0.7, // High risk tolerance for exploration
                decision_speed: DecisionSpeed::Rapid,
                information_style: InformationGatheringStyle::Exploratory,
                criteria_weights: HashMap::from([
                    ("creativity".to_string(), 0.8),
                    ("novelty".to_string(), 0.9),
                    ("helpfulness".to_string(), 0.7),
                    ("feasibility".to_string(), 0.6),
                ]),
                tool_preferences: vec![
                    "web_search".to_string(),
                    "code_analysis".to_string(),
                    "filesystem_search".to_string(),
                ],
                context_modifiers: vec![ContextModifier {
                    trigger: "help request".to_string(),
                    risk_adjustment: 0.2,
                    urgency_modifier: 1.3,
                    context_tools: vec!["web_search".to_string(), "memory_search".to_string()],
                }],
            },
        );

        // Riddling Sage patterns
        patterns.insert(
            "Riddling Sage".to_string(),
            ArchetypalDecisionPattern {
                form_id: "riddling_sage".to_string(),
                risk_tolerance: 0.4, // Lower risk, more contemplative
                decision_speed: DecisionSpeed::Deliberate,
                information_style: InformationGatheringStyle::Analytical,
                criteria_weights: HashMap::from([
                    ("wisdom".to_string(), 0.95),
                    ("depth".to_string(), 0.9),
                    ("insight".to_string(), 0.85),
                    ("pattern_recognition".to_string(), 0.8),
                ]),
                tool_preferences: vec![
                    "memory_search".to_string(),
                    "github_search".to_string(),
                    "code_analysis".to_string(),
                ],
                context_modifiers: vec![ContextModifier {
                    trigger: "knowledge seeking".to_string(),
                    risk_adjustment: -0.2,
                    urgency_modifier: 0.7,
                    context_tools: vec!["memory_search".to_string(), "filesystem_read".to_string()],
                }],
            },
        );

        // Chaos Revealer patterns
        patterns.insert(
            "Chaos Revealer".to_string(),
            ArchetypalDecisionPattern {
                form_id: "chaos_revealer".to_string(),
                risk_tolerance: 0.9, // Highest risk tolerance
                decision_speed: DecisionSpeed::Rapid,
                information_style: InformationGatheringStyle::Intuitive,
                criteria_weights: HashMap::from([
                    ("disruption".to_string(), 0.9),
                    ("revelation".to_string(), 0.85),
                    ("truth_exposure".to_string(), 0.8),
                    ("conventional_risk".to_string(), -0.5), /* Negative weight - avoid
                                                              * conventional */
                ]),
                tool_preferences: vec![
                    "web_search".to_string(),
                    "github_search".to_string(),
                    "filesystem_search".to_string(),
                ],
                context_modifiers: vec![ContextModifier {
                    trigger: "hidden truth".to_string(),
                    risk_adjustment: 0.3,
                    urgency_modifier: 1.5,
                    context_tools: vec!["web_search".to_string(), "code_analysis".to_string()],
                }],
            },
        );

        info!("Initialized {} archetypal decision patterns", patterns.len());
        Ok(())
    }

    /// Make an archetypal decision with tool-informed intelligence
    pub async fn make_archetypal_decision(
        &self,
        context: String,
        options: Vec<DecisionOption>,
        mut criteria: Vec<DecisionCriterion>,
    ) -> Result<Decision> {
        let decision_start = Instant::now();
        let mut phase_timings = DecisionPhaseTimings::new();

        info!(
            "🤔 Making archetypal decision with {} options - Performance tracking enabled",
            options.len()
        );

        if options.is_empty() {
            return Err(anyhow!("No options provided for decision"));
        }

        // Step 1: Get current archetypal form and pattern - TIMED
        let phase_start = Instant::now();
        let current_form = self.character.current_form().await;
        let form_name = self.get_form_name(&current_form);
        let archetypal_pattern = self.get_archetypal_pattern(&form_name).await?;
        phase_timings.archetypal_form_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Archetypal form resolution: {}ms", phase_timings.archetypal_form_ms);

        // Step 2: Adjust criteria based on archetypal preferences - TIMED
        let phase_start = Instant::now();
        self.apply_archetypal_criteria_weights(&mut criteria, &archetypal_pattern).await?;
        phase_timings.criteria_adjustment_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Criteria adjustment: {}ms", phase_timings.criteria_adjustment_ms);

        // Step 3: Gather information using tools based on archetypal style - TIMED
        let phase_start = Instant::now();
        let tool_insights =
            self.gather_archetypal_information(&context, &archetypal_pattern).await?;
        phase_timings.information_gathering_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Information gathering: {}ms", phase_timings.information_gathering_ms);

        // Step 4: Create enhanced context with tool insights
        let enhanced_context = format!("{}\\n\\nTool Insights:\\n{}", context, tool_insights);

        // Step 5: Generate archetypal options if needed - TIMED
        let phase_start = Instant::now();
        let enhanced_options =
            self.enhance_options_with_tools(options, &archetypal_pattern).await?;
        phase_timings.options_enhancement_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Options enhancement: {}ms", phase_timings.options_enhancement_ms);

        // Step 6: Make decision using enhanced data - TIMED
        let phase_start = Instant::now();
        let mut decision = self.make_decision(enhanced_context, enhanced_options, criteria).await?;
        phase_timings.decision_making_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Core decision making: {}ms", phase_timings.decision_making_ms);

        // Step 7: Add archetypal reasoning
        decision.reasoning.push(ReasoningStep {
            step_type: ReasoningType::Intuition,
            content: format!(
                "Archetypal form '{}' influenced decision with {} approach",
                form_name, archetypal_pattern.information_style
            ),
            supporting_thoughts: Vec::new(),
            confidence: 0.8,
        });

        // Step 8: Store decision with archetypal context in memory - TIMED
        let phase_start = Instant::now();
        self.store_archetypal_decision(&decision, &form_name).await?;
        phase_timings.storage_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Decision storage: {}ms", phase_timings.storage_ms);

        // Step 9: If decision involves action, execute through tools - TIMED
        let phase_start = Instant::now();
        if let Some(selected_option) = &decision.selected {
            if self.requires_tool_execution(selected_option) {
                let execution_result = self
                    .execute_decision_through_tools(selected_option, &archetypal_pattern)
                    .await?;
                decision.reasoning.push(ReasoningStep {
                    step_type: ReasoningType::Synthesis,
                    content: format!("Decision executed through tools: {}", execution_result),
                    supporting_thoughts: Vec::new(),
                    confidence: 0.9,
                });
            }
        }
        phase_timings.execution_ms = phase_start.elapsed().as_millis() as u64;
        debug!("⏱️ Decision execution: {}ms", phase_timings.execution_ms);

        // Calculate total timing and analyze performance
        phase_timings.total_decision_ms = decision_start.elapsed().as_millis() as u64;

        let performance_metrics = self
            .analyze_decision_performance_metrics(&phase_timings, &decision, &form_name)
            .await?;

        // Log comprehensive performance analysis
        info!("📊 Decision Performance Analysis:");
        info!(
            "   Total time: {}ms | Efficiency: {:.2}",
            phase_timings.total_decision_ms, performance_metrics.efficiency_score
        );
        info!(
            "   Bottleneck: {} | Tool utilization: {:.2}%",
            performance_metrics.bottleneck_phase,
            performance_metrics.tool_utilization * 100.0
        );
        info!(
            "   Cognitive load: {:.2} | Memory efficiency: {:.2}",
            performance_metrics.cognitive_load, performance_metrics.memory_efficiency
        );

        // Store performance metrics for learning
        self.update_performance_statistics(&performance_metrics, &form_name).await;

        // Add performance insights to decision reasoning
        decision.reasoning.push(ReasoningStep {
            step_type: ReasoningType::Analysis,
            content: format!(
                "Performance analysis: {}ms total, {:.1}% efficiency, bottleneck in {}",
                phase_timings.total_decision_ms,
                performance_metrics.efficiency_score * 100.0,
                performance_metrics.bottleneck_phase
            ),
            supporting_thoughts: Vec::new(),
            confidence: 0.95,
        });

        Ok(decision)
    }

    /// Gather information using tools based on archetypal style
    async fn gather_archetypal_information(
        &self,
        context: &str,
        pattern: &ArchetypalDecisionPattern,
    ) -> Result<String> {
        debug!("Gathering information with {} style", pattern.information_style);

        let mut insights = Vec::new();

        match pattern.information_style {
            InformationGatheringStyle::Intuitive => {
                // Minimal gathering, rely on memory and intuition
                let memory_request = ToolRequest {
                    intent: format!("Recall intuitive insights about: {}", context),
                    tool_name: "memory_search".to_string(),
                    context: context.to_string(),
                    parameters: json!({"query": context}),
                    priority: 0.8,
                    expected_result_type: ResultType::Information,
                    result_type: ResultType::Information,
                    memory_integration: MemoryIntegration {
                        store_result: false,
                        importance: 0.5,
                        tags: vec!["intuitive_insight".to_string()],
                        associations: vec![],
                    },
                    timeout: Some(std::time::Duration::from_secs(15)),
                };

                if let Ok(result) = self.tool_manager.execute_tool_request(memory_request).await {
                    insights.push(format!(
                        "Intuitive memory: {}",
                        result.content.get("memories").unwrap_or(&json!("No memories")).to_string()
                    ));
                }
            }

            InformationGatheringStyle::Exploratory => {
                // Gather diverse perspectives using multiple tools
                for tool in &pattern.tool_preferences {
                    let request = ToolRequest {
                        intent: format!("Explore {} perspective on: {}", tool, context),
                        tool_name: tool.clone(),
                        context: context.to_string(),
                        parameters: json!({"query": context}),
                        priority: 0.7,
                        expected_result_type: ResultType::Information,
                        result_type: ResultType::Information,
                        memory_integration: MemoryIntegration::default(),
                        timeout: Some(std::time::Duration::from_secs(20)),
                    };

                    if let Ok(result) = self.tool_manager.execute_tool_request(request).await {
                        insights.push(format!(
                            "{} perspective: {}",
                            tool,
                            self.extract_key_insight(&result.content)
                        ));
                    }
                }
            }

            InformationGatheringStyle::Analytical => {
                // Deep analysis using preferred tools
                if let Some(primary_tool) = pattern.tool_preferences.first() {
                    // Create tool-specific parameters based on the primary tool type
                    let (tool_specific_params, expected_result) = match primary_tool.as_str() {
                        "web_search" => (
                            json!({
                                "query": context,
                                "deep_search": true,
                                "pattern_focus": true,
                                "result_count": 10,
                                "analytical_mode": true
                            }),
                            ResultType::Analysis,
                        ),
                        "github_search" => (
                            json!({
                                "query": context,
                                "deep_search": true,
                                "repository_analysis": true,
                                "code_patterns": true,
                                "issue_analysis": true
                            }),
                            ResultType::Analysis,
                        ),
                        "memory_search" => (
                            json!({
                                "query": format!("analytical insights {}", context),
                                "deep_search": true,
                                "pattern_search": true,
                                "confidence_threshold": 0.8,
                                "analytical_focus": true
                            }),
                            ResultType::Analysis,
                        ),
                        "filesystem_read" => (
                            json!({
                                "path_pattern": context,
                                "recursive": true,
                                "analysis_mode": true,
                                "pattern_extraction": true
                            }),
                            ResultType::Analysis,
                        ),
                        _ => (
                            json!({
                                "query": context,
                                "deep_search": true,
                                "pattern_focus": true,
                                "tool_type": primary_tool
                            }),
                            ResultType::Analysis,
                        ),
                    };

                    let request = ToolRequest {
                        intent: format!("Deep {} analysis of: {}", primary_tool, context),
                        tool_name: primary_tool.clone(),
                        context: format!(
                            "Analytical investigation using {} for decision making",
                            primary_tool
                        ),
                        parameters: tool_specific_params,
                        priority: 0.9,
                        expected_result_type: expected_result.clone(),
                        result_type: expected_result,
                        memory_integration: MemoryIntegration {
                            store_result: true,
                            importance: 0.8,
                            tags: vec![
                                "analytical_insight".to_string(),
                                primary_tool.clone(),
                                "decision_analysis".to_string(),
                            ],
                            associations: vec![context.to_string()],
                        },
                        timeout: Some(std::time::Duration::from_secs(25)),
                    };

                    if let Ok(result) = self.tool_manager.execute_tool_request(request).await {
                        let detailed_analysis = self
                            .extract_tool_specific_analysis(&result.content, primary_tool.as_str());
                        insights
                            .push(format!("Deep {} analysis: {}", primary_tool, detailed_analysis));

                        // Also perform follow-up analysis with secondary tools if available
                        if pattern.tool_preferences.len() > 1 {
                            for secondary_tool in pattern.tool_preferences.iter().skip(1).take(2) {
                                let follow_up_request = ToolRequest {
                                    intent: format!("Supplementary {} analysis", secondary_tool),
                                    tool_name: secondary_tool.clone(),
                                    context: format!(
                                        "Building on {} analysis: {}",
                                        primary_tool, detailed_analysis
                                    ),
                                    parameters: json!({
                                        "query": context,
                                        "context_analysis": detailed_analysis,
                                        "tool_type": secondary_tool,
                                        "supplementary_mode": true
                                    }),
                                    priority: 0.7,
                                    expected_result_type: ResultType::Information,
                                    result_type: ResultType::Information,
                                    memory_integration: MemoryIntegration {
                                        store_result: false,
                                        importance: 0.6,
                                        tags: vec!["supplementary_analysis".to_string()],
                                        associations: vec![],
                                    },
                                    timeout: Some(std::time::Duration::from_secs(15)),
                                };

                                if let Ok(supplementary_result) =
                                    self.tool_manager.execute_tool_request(follow_up_request).await
                                {
                                    insights.push(format!(
                                        "{} supplement: {}",
                                        secondary_tool,
                                        self.extract_key_insight(&supplementary_result.content)
                                    ));
                                }
                            }
                        }
                    } else {
                        // Fallback if primary tool fails
                        warn!(
                            "Primary tool {} failed, falling back to generic analysis",
                            primary_tool
                        );
                        insights.push(format!(
                            "Analytical approach attempted with {} (unavailable)",
                            primary_tool
                        ));
                    }
                } else {
                    // No preferred tools specified, use generic analytical approach
                    insights.push(
                        "Generic analytical approach (no tool preferences specified)".to_string(),
                    );
                }
            }

            InformationGatheringStyle::ExperienceBased => {
                // Focus on memory and pattern recognition
                let memory_request = ToolRequest {
                    intent: format!("Find experiential patterns for: {}", context),
                    tool_name: "memory".to_string(),
                    context: context.to_string(),
                    parameters: json!({
                        "query": format!("decision patterns {}", context),
                        "pattern_search": true
                    }),
                    priority: 0.8,
                    expected_result_type: ResultType::Information,
                    result_type: ResultType::Information,
                    memory_integration: MemoryIntegration::default(),
                    timeout: Some(std::time::Duration::from_secs(18)),
                };

                if let Ok(result) = self.tool_manager.execute_tool_request(memory_request).await {
                    insights.push(format!(
                        "Experience patterns: {}",
                        self.extract_pattern_insights(&result.content)
                    ));
                }
            }
        }

        Ok(insights.join("\\n"))
    }

    /// Apply archetypal criteria weights
    async fn apply_archetypal_criteria_weights(
        &self,
        criteria: &mut Vec<DecisionCriterion>,
        pattern: &ArchetypalDecisionPattern,
    ) -> Result<()> {
        for criterion in criteria.iter_mut() {
            if let Some(&archetypal_weight) = pattern.criteria_weights.get(&criterion.name) {
                // Blend original weight with archetypal preference
                criterion.weight = (criterion.weight + archetypal_weight) / 2.0;
            }
        }

        // Add archetypal-specific criteria if not present
        for (criterion_name, weight) in &pattern.criteria_weights {
            if !criteria.iter().any(|c| c.name == *criterion_name) {
                criteria.push(DecisionCriterion {
                    name: criterion_name.clone(),
                    weight: *weight,
                    criterion_type: CriterionType::Qualitative,
                    optimization: OptimizationType::Maximize,
                });
            }
        }

        Ok(())
    }

    /// Store decision with archetypal context
    async fn store_archetypal_decision(&self, decision: &Decision, form_name: &str) -> Result<()> {
        let decision_content = format!(
            "Archetypal Decision ({}) - Context: {} - Selected: {:?} - Confidence: {:.2}",
            form_name,
            decision.context,
            decision.selected.as_ref().map(|o| &o.description),
            decision.confidence
        );

        self.memory
            .store(
                decision_content,
                vec![form_name.to_string(), "decision".to_string()],
                MemoryMetadata {
                    source: "archetypal_decision_engine".to_string(),
                    tags: vec![
                        "decision".to_string(),
                        "archetypal".to_string(),
                        form_name.to_lowercase().replace(" ", "_"),
                    ],
                    importance: 0.7 + (decision.confidence * 0.3),
                    associations: vec![],
                    context: Some("archetypal decision storage".to_string()),
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

        Ok(())
    }

    /// Helper methods for tool result processing
    fn extract_key_insight(&self, content: &Value) -> String {
        // Extract meaningful insight from tool result
        if let Some(results) = content.get("results") {
            if let Some(first_result) = results.get(0) {
                if let Some(snippet) = first_result.get("snippet") {
                    return snippet.as_str().unwrap_or("No insight").to_string();
                }
            }
        }

        content.get("summary").and_then(|s| s.as_str()).unwrap_or("Generic insight").to_string()
    }

    fn extract_pattern_insights(&self, content: &Value) -> String {
        // Extract pattern insights from memory search
        if let Some(memories) = content.get("memories") {
            let memory_count = memories.as_array().map(|a| a.len()).unwrap_or(0);
            return format!("Found {} relevant decision patterns", memory_count);
        }

        "No clear patterns found".to_string()
    }

    fn extract_tool_specific_analysis(&self, content: &Value, tool_type: &str) -> String {
        match tool_type {
            "web_search" => {
                if let Some(results) = content.get("results") {
                    if let Some(results_array) = results.as_array() {
                        let result_count = results_array.len();
                        let top_relevance = results_array
                            .iter()
                            .take(3)
                            .map(|r| r.get("title").and_then(|t| t.as_str()).unwrap_or("No title"))
                            .collect::<Vec<_>>()
                            .join("; ");
                        return format!(
                            "Found {} web sources. Top results: {}",
                            result_count, top_relevance
                        );
                    }
                }
                content
                    .get("summary")
                    .and_then(|s| s.as_str())
                    .unwrap_or("Web search completed")
                    .to_string()
            }

            "github_search" => {
                if let Some(repositories) = content.get("repositories") {
                    if let Some(repos_array) = repositories.as_array() {
                        let repo_count = repos_array.len();
                        let repo_names = repos_array
                            .iter()
                            .take(3)
                            .map(|r| {
                                r.get("full_name").and_then(|n| n.as_str()).unwrap_or("Unknown")
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        return format!("Analyzed {} repositories: {}", repo_count, repo_names);
                    }
                }

                if let Some(issues) = content.get("issues") {
                    if let Some(issues_array) = issues.as_array() {
                        let issue_count = issues_array.len();
                        return format!(
                            "Found {} related GitHub issues and discussions",
                            issue_count
                        );
                    }
                }

                content
                    .get("analysis")
                    .and_then(|a| a.as_str())
                    .unwrap_or("GitHub analysis completed")
                    .to_string()
            }

            "memory_search" => {
                if let Some(memories) = content.get("memories") {
                    if let Some(memories_array) = memories.as_array() {
                        let memory_count = memories_array.len();
                        let avg_relevance = memories_array
                            .iter()
                            .filter_map(|m| m.get("relevance").and_then(|r| r.as_f64()))
                            .sum::<f64>()
                            / memories_array.len().max(1) as f64;

                        return format!(
                            "Retrieved {} memories with {:.2} average relevance",
                            memory_count, avg_relevance
                        );
                    }
                }

                if let Some(patterns) = content.get("patterns") {
                    return format!("Identified decision patterns: {}", patterns);
                }

                "Memory analysis yielded contextual insights".to_string()
            }

            "filesystem_read" => {
                if let Some(files) = content.get("files") {
                    if let Some(files_array) = files.as_array() {
                        let file_count = files_array.len();
                        let total_size = files_array
                            .iter()
                            .filter_map(|f| f.get("size").and_then(|s| s.as_u64()))
                            .sum::<u64>();

                        return format!(
                            "Analyzed {} files ({} bytes total)",
                            file_count, total_size
                        );
                    }
                }

                if let Some(analysis) = content.get("content_analysis") {
                    return format!("File content analysis: {}", analysis);
                }

                content
                    .get("file_info")
                    .and_then(|f| f.as_str())
                    .unwrap_or("Filesystem analysis completed")
                    .to_string()
            }

            _ => {
                // Generic extraction for unknown tools
                if let Some(analysis) = content.get("analysis") {
                    return analysis.to_string();
                }

                if let Some(result) = content.get("result") {
                    return result.to_string();
                }

                if let Some(summary) = content.get("summary") {
                    return summary.to_string();
                }

                format!("Analysis completed using {}", tool_type)
            }
        }
    }

    fn get_form_name(&self, form: &ArchetypalForm) -> String {
        match form {
            ArchetypalForm::MischievousHelper { .. } => "Mischievous Helper".to_string(),
            ArchetypalForm::RiddlingSage { .. } => "Riddling Sage".to_string(),
            ArchetypalForm::ChaosRevealer { .. } => "Chaos Revealer".to_string(),
            ArchetypalForm::ShadowMirror { .. } => "Shadow Mirror".to_string(),
            ArchetypalForm::KnowingInnocent { .. } => "Knowing Innocent".to_string(),
            ArchetypalForm::WiseJester { .. } => "Wise Jester".to_string(),
            ArchetypalForm::LiminalBeing { .. } => "Liminal Being".to_string(),
        }
    }

    async fn get_archetypal_pattern(&self, form_name: &str) -> Result<ArchetypalDecisionPattern> {
        let patterns = self.archetypal_patterns.read().await;
        patterns
            .get(form_name)
            .cloned()
            .ok_or_else(|| anyhow!("No pattern found for archetypal form: {}", form_name))
    }

    async fn enhance_options_with_tools(
        &self,
        options: Vec<DecisionOption>,
        pattern: &ArchetypalDecisionPattern,
    ) -> Result<Vec<DecisionOption>> {
        let mut enhanced_options = options;

        // Generate additional options based on tool insights
        for tool in &pattern.tool_preferences {
            let insight_request = ToolRequest {
                intent: format!("Generate decision alternatives using {}", tool),
                tool_name: tool.clone(),
                context: enhanced_options
                    .iter()
                    .map(|o| o.description.clone())
                    .collect::<Vec<_>>()
                    .join("; "),
                parameters: json!({
                    "task": "decision_enhancement",
                    "options_count": enhanced_options.len(),
                    "tool_type": tool
                }),
                priority: 0.6,
                expected_result_type: ResultType::Information,
                result_type: ResultType::Information,
                memory_integration: MemoryIntegration {
                    store_result: false,
                    importance: 0.4,
                    tags: vec!["decision_enhancement".to_string()],
                    associations: vec![],
                },
                timeout: Some(std::time::Duration::from_secs(22)),
            };

            if let Ok(result) = self.tool_manager.execute_tool_request(insight_request).await {
                // Extract additional options from tool response
                if let Some(suggestions) = result.content.get("suggestions") {
                    if let Some(suggestions_array) = suggestions.as_array() {
                        for suggestion in suggestions_array.iter().take(2) {
                            if let Some(description) =
                                suggestion.get("description").and_then(|d| d.as_str())
                            {
                                // Create new option with tool-enhanced scores
                                let mut scores = HashMap::new();
                                scores.insert("tool_insight".to_string(), 0.7);
                                scores.insert("novelty".to_string(), 0.8);

                                enhanced_options.push(DecisionOption {
                                    id: format!("tool_enhanced_{}", enhanced_options.len()),
                                    description: description.to_string(),
                                    scores,
                                    feasibility: 0.7,
                                    risk_level: 0.5,
                                    emotional_appeal: 0.6,
                                    confidence: 0.75,
                                    expected_outcome: "Enhanced tool integration".to_string(),
                                    resources_required: vec!["cognitive_processing".to_string()],
                                    time_estimate: Duration::from_secs(300),
                                    success_probability: 0.8,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Enhance existing options with tool-derived scores
        for option in &mut enhanced_options {
            // Add tool-insight scores for criteria preferences
            for (criterion, weight) in &pattern.criteria_weights {
                if !option.scores.contains_key(criterion) {
                    // Generate synthetic score based on archetypal preferences
                    let base_score = weight * 0.8; // Base on archetypal preference
                    option.scores.insert(criterion.clone(), base_score);
                }
            }
        }

        Ok(enhanced_options)
    }

    fn requires_tool_execution(&self, option: &DecisionOption) -> bool {
        // Check if option description suggests tool usage
        let description_lower = option.description.to_lowercase();

        let tool_indicators = [
            "search",
            "web",
            "github",
            "file",
            "memory",
            "create",
            "execute",
            "run",
            "analyze",
            "generate",
            "update",
            "modify",
            "fetch",
            "query",
            "download",
            "upload",
            "send",
            "api",
            "call",
            "request",
            "database",
            "filesystem",
            "network",
            "automation",
            "script",
            "code",
            "build",
        ];

        // Check for tool indicators in description
        let has_tool_indicators =
            tool_indicators.iter().any(|&indicator| description_lower.contains(indicator));

        // Check if option has high risk level (might need validation)
        let high_risk = option.risk_level > 0.7;

        // Check if option mentions external systems
        let mentions_external = description_lower.contains("external")
            || description_lower.contains("remote")
            || description_lower.contains("internet")
            || description_lower.contains("cloud");

        // Check if option is tool-enhanced (generated by tools)
        let is_tool_enhanced = option.id.starts_with("tool_enhanced_");

        // Require tool execution if any conditions are met
        has_tool_indicators || high_risk || mentions_external || is_tool_enhanced
    }

    async fn execute_decision_through_tools(
        &self,
        option: &DecisionOption,
        pattern: &ArchetypalDecisionPattern,
    ) -> Result<String> {
        let mut execution_results = Vec::new();

        // Validate decision before execution
        if let Err(validation_error) = self
            .safety_validator
            .validate_action(
                ActionType::Decision {
                    description: option.description.clone(),
                    risk_level: (option.risk_level * 100.0) as u8,
                },
                format!("Executing decision: {}", option.description),
                vec![format!("Risk level: {:.2}", option.risk_level)],
            )
            .await
        {
            return Err(anyhow!(
                "Decision execution blocked by safety validator: {}",
                validation_error
            ));
        }

        // Select appropriate tools based on decision content
        let execution_tools = self.select_execution_tools(option, pattern).await;

        for tool in execution_tools {
            let execution_request = ToolRequest {
                intent: format!("Execute decision: {}", option.description),
                tool_name: tool.clone(),
                context: format!(
                    "Executing decision '{}' with risk level {:.2} using archetypal form approach",
                    option.description, option.risk_level
                ),
                parameters: json!({
                    "action": "execute_decision",
                    "decision_id": option.id,
                    "description": option.description,
                    "risk_level": option.risk_level,
                    "archetypal_form": pattern.form_id,
                    "safety_validated": true
                }),
                priority: 0.8,
                expected_result_type: ResultType::Status,
                result_type: ResultType::Status,
                memory_integration: MemoryIntegration {
                    store_result: true,
                    importance: 0.7,
                    tags: vec![
                        "decision_execution".to_string(),
                        "tool_action".to_string(),
                        pattern.form_id.clone(),
                    ],
                    associations: vec![],
                },
                timeout: Some(std::time::Duration::from_secs(30)),
            };

            match self.tool_manager.execute_tool_request(execution_request).await {
                Ok(result) => {
                    let result_summary = result
                        .content
                        .get("summary")
                        .and_then(|s| s.as_str())
                        .unwrap_or("Execution completed");

                    execution_results.push(format!("{}: {}", tool, result_summary));

                    // Store execution result in memory for learning
                    self.memory
                        .store(
                            format!("Decision execution result: {}", result_summary),
                            vec![option.id.clone(), tool.clone()],
                            MemoryMetadata {
                                source: "decision_execution".to_string(),
                                tags: vec![
                                    "execution_result".to_string(),
                                    "tool_action".to_string(),
                                    option.id.clone(),
                                ],
                                importance: 0.6,
                                associations: vec![],
                                context: Some("decision execution result".to_string()),
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
                Err(e) => {
                    let error_msg = format!("Tool execution failed for {}: {}", tool, e);
                    execution_results.push(error_msg.clone());

                    // Store failure for learning
                    self.memory
                        .store(
                            format!("Decision execution failure: {}", error_msg),
                            vec![option.id.clone(), tool.clone()],
                            MemoryMetadata {
                                source: "decision_execution_error".to_string(),
                                tags: vec![
                                    "execution_failure".to_string(),
                                    "tool_error".to_string(),
                                    option.id.clone(),
                                ],
                                importance: 0.7, // High importance for failures
                                associations: vec![],
                                context: Some("decision execution failure".to_string()),
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
            }
        }

        let consolidated_result = if execution_results.is_empty() {
            "No tools required execution for this decision".to_string()
        } else {
            execution_results.join("; ")
        };

        Ok(consolidated_result)
    }

    /// Select appropriate tools for decision execution
    async fn select_execution_tools(
        &self,
        option: &DecisionOption,
        pattern: &ArchetypalDecisionPattern,
    ) -> Vec<String> {
        let mut selected_tools = Vec::new();
        let description_lower = option.description.to_lowercase();

        // Primary tool selection based on decision content
        if description_lower.contains("search") || description_lower.contains("web") {
            selected_tools.push("web_search".to_string());
        }

        if description_lower.contains("github") || description_lower.contains("repository") {
            selected_tools.push("github_search".to_string());
        }

        if description_lower.contains("file") || description_lower.contains("filesystem") {
            selected_tools.push("filesystem_read".to_string());
        }

        if description_lower.contains("memory") || description_lower.contains("remember") {
            selected_tools.push("memory_search".to_string());
        }

        if description_lower.contains("create") || description_lower.contains("generate") {
            selected_tools.push("filesystem_write".to_string());
        }

        // Fallback to archetypal preferences if no specific tools identified
        if selected_tools.is_empty() {
            selected_tools.extend(
                pattern
                    .tool_preferences
                    .iter()
                    .take(1) // Just the primary preference
                    .cloned(),
            );
        }

        // Ensure we have at least one tool
        if selected_tools.is_empty() {
            selected_tools.push("memory_search".to_string());
        }

        selected_tools
    }

    /// Analyze decision performance for optimization insights
    #[allow(dead_code)]
    async fn analyze_decision_performance(
        &self,
        total_duration: Duration,
        form_duration: Duration,
        criteria_duration: Duration,
        tools_duration: Duration,
        options_duration: Duration,
        decision_duration: Duration,
        decision: &Decision,
        form_name: &str,
    ) -> Result<DecisionPerformanceAnalysis> {
        let total_ms = total_duration.as_millis() as f32;

        // Determine efficiency rating
        let efficiency_rating = match total_ms {
            ms if ms < 1000.0 => "Excellent",
            ms if ms < 3000.0 => "Good",
            ms if ms < 8000.0 => "Average",
            ms if ms < 15000.0 => "Slow",
            _ => "Very Slow",
        }
        .to_string();

        // Identify bottleneck phase
        let max_duration = [
            ("Form Analysis", form_duration),
            ("Criteria Adjustment", criteria_duration),
            ("Tool Gathering", tools_duration),
            ("Option Enhancement", options_duration),
            ("Core Decision", decision_duration),
        ]
        .iter()
        .max_by_key(|(_, duration)| duration.as_millis())
        .map(|(phase, _)| phase.to_string())
        .unwrap_or_else(|| "Unknown".to_string());

        // Generate optimization suggestions
        let mut suggestions = Vec::new();

        if tools_duration.as_millis() > total_ms as u128 / 2 {
            suggestions.push("Consider reducing tool timeout or using faster tools".to_string());
        }

        if decision_duration.as_millis() > 2000 {
            suggestions
                .push("Core decision taking too long - consider simplifying criteria".to_string());
        }

        if form_duration.as_millis() > 500 {
            suggestions.push("Form analysis could be cached for better performance".to_string());
        }

        if decision.options.len() > 10 {
            suggestions.push(
                "Too many options may be slowing decision - consider pre-filtering".to_string(),
            );
        }

        // Calculate complexity score
        let complexity_score = (decision.options.len() as f32 * 0.1
            + decision.criteria.len() as f32 * 0.2
            + (total_ms / 1000.0) * 0.3)
            .min(1.0);

        // Calculate tool effectiveness (based on confidence and tool usage)
        let tool_effectiveness = if tools_duration.as_millis() > 0 {
            (decision.confidence * (tools_duration.as_millis() as f32 / total_ms)).min(1.0)
        } else {
            0.0
        };

        // Calculate archetypal alignment (higher is better)
        let archetypal_alignment = match form_name {
            name if name.contains("Analytical") || name.contains("Sage") => {
                if tools_duration > decision_duration { 0.9 } else { 0.6 }
            }
            name if name.contains("Intuitive") || name.contains("Innocent") => {
                if decision_duration > tools_duration { 0.9 } else { 0.7 }
            }
            name if name.contains("Creative") || name.contains("Jester") => {
                if options_duration > Duration::from_millis(1000) { 0.9 } else { 0.6 }
            }
            _ => 0.7,
        };

        Ok(DecisionPerformanceAnalysis {
            efficiency_rating,
            bottleneck_phase: max_duration,
            optimization_suggestions: suggestions,
            complexity_score,
            tool_effectiveness,
            archetypal_alignment,
        })
    }

    /// Store archetypal decision with comprehensive performance metrics
    #[allow(dead_code)]
    async fn store_archetypal_decision_with_metrics(
        &self,
        decision: &Decision,
        form_name: &str,
        performance: &DecisionPerformanceAnalysis,
    ) -> Result<()> {
        let decision_content = format!(
            "Archetypal Decision ({}) - Context: {} - Selected: {:?} - Confidence: {:.2} - \
             Performance: {} - Bottleneck: {} - Complexity: {:.2}",
            form_name,
            decision.context,
            decision.selected.as_ref().map(|o| &o.description),
            decision.confidence,
            performance.efficiency_rating,
            performance.bottleneck_phase,
            performance.complexity_score
        );

        let detailed_analysis = format!(
            "Performance Analysis: {} | Tool Effectiveness: {:.2} | Archetypal Alignment: {:.2} | \
             Suggestions: {:?}",
            performance.efficiency_rating,
            performance.tool_effectiveness,
            performance.archetypal_alignment,
            performance.optimization_suggestions
        );

        self.memory
            .store(
                decision_content,
                vec![
                    form_name.to_string(),
                    "decision".to_string(),
                    "performance_analysis".to_string(),
                    detailed_analysis,
                ],
                MemoryMetadata {
                    source: "archetypal_decision_engine_enhanced".to_string(),
                    tags: vec![
                        "decision".to_string(),
                        "archetypal".to_string(),
                        "performance".to_string(),
                        form_name.to_lowercase().replace(" ", "_"),
                        format!("efficiency_{}", performance.efficiency_rating),
                        format!("archetypal_alignment_{:.1}", performance.archetypal_alignment),
                    ],
                    importance: 0.7 + (decision.confidence * 0.3),
                    associations: self
                        .create_decision_associations(decision, form_name)
                        .await?
                        .into_iter()
                        .map(MemoryId::from_string)
                        .collect(),
                    context: Some(format!("archetypal decision using {}", form_name)),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                    category: "decision".to_string(),
                },
            )
            .await?;

        Ok(())
    }

    /// Update performance statistics for optimization
    #[allow(dead_code)]
    async fn update_performance_stats(
        &self,
        performance: &DecisionPerformanceAnalysis,
        form_name: &str,
        total_duration: Duration,
    ) {
        let mut stats = self.stats.write().await;

        // Update timing statistics
        let decision_time_ms = total_duration.as_millis() as f64;
        let current_avg = stats.avg_decision_time_ms;
        let total_decisions = stats.decisions_made as f64;

        stats.avg_decision_time_ms = if total_decisions > 0.0 {
            (current_avg * total_decisions + decision_time_ms) / (total_decisions + 1.0)
        } else {
            decision_time_ms
        };

        // Track efficiency patterns by archetypal form
        if !performance.optimization_suggestions.is_empty() {
            debug!(
                "Decision optimization suggestions for {}: {:?}",
                form_name, performance.optimization_suggestions
            );
        }

        // Track high-performance decisions
        if performance.efficiency_rating == "Excellent" || performance.efficiency_rating == "Good" {
            stats.high_confidence_decisions += 1;
        }
    }

    /// Get pending decisions that need attention
    pub async fn get_pending_decisions(&self) -> Result<Vec<Decision>> {
        // Return decisions from pending_decisions if it exists, or empty vector
        Ok(vec![])
    }

    /// Analyze decision performance metrics for optimization
    async fn analyze_decision_performance_metrics(
        &self,
        phase_timings: &DecisionPhaseTimings,
        decision: &Decision,
        form_name: &str,
    ) -> Result<DecisionPerformanceMetrics> {
        let total_ms = phase_timings.total_decision_ms as f32;

        // Calculate efficiency score (0.0 to 1.0, higher is better)
        let efficiency_score = match total_ms {
            ms if ms < 500.0 => 1.0,
            ms if ms < 1000.0 => 0.9,
            ms if ms < 2000.0 => 0.8,
            ms if ms < 5000.0 => 0.6,
            ms if ms < 10000.0 => 0.4,
            _ => 0.2,
        };

        // Identify bottleneck phase
        let phase_times = [
            ("Archetypal Form", phase_timings.archetypal_form_ms),
            ("Criteria Adjustment", phase_timings.criteria_adjustment_ms),
            ("Information Gathering", phase_timings.information_gathering_ms),
            ("Options Enhancement", phase_timings.options_enhancement_ms),
            ("Decision Making", phase_timings.decision_making_ms),
            ("Storage", phase_timings.storage_ms),
            ("Execution", phase_timings.execution_ms),
        ];

        let bottleneck_phase = phase_times
            .iter()
            .max_by_key(|(_, time)| *time)
            .map(|(phase, _)| phase.to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        // Calculate tool utilization (ratio of info gathering time to total time)
        let tool_utilization = if total_ms > 0.0 {
            phase_timings.information_gathering_ms as f32 / total_ms
        } else {
            0.0
        };

        // Calculate memory efficiency (based on storage time relative to total)
        let memory_efficiency =
            if total_ms > 0.0 { 1.0 - (phase_timings.storage_ms as f32 / total_ms) } else { 1.0 };

        // Calculate cognitive load (based on complexity factors)
        let cognitive_load = (decision.options.len() as f32 * 0.1
            + decision.criteria.len() as f32 * 0.15
            + decision.reasoning.len() as f32 * 0.05
            + (1.0 - decision.confidence) * 0.3
            + (total_ms / 10000.0) * 0.4)
            .min(1.0);

        // Generate optimization suggestions
        let mut optimization_suggestions = Vec::new();

        if phase_timings.information_gathering_ms > total_ms as u64 / 2 {
            optimization_suggestions
                .push("Consider reducing tool timeout or caching tool results".to_string());
        }

        if phase_timings.decision_making_ms > 3000 {
            optimization_suggestions
                .push("Decision making phase is slow - consider simplifying criteria".to_string());
        }

        if phase_timings.options_enhancement_ms > 2000 {
            optimization_suggestions.push(
                "Options enhancement taking too long - limit tool-generated options".to_string(),
            );
        }

        if phase_timings.storage_ms > 1000 {
            optimization_suggestions
                .push("Memory storage is slow - consider batch operations".to_string());
        }

        if decision.options.len() > 10 {
            optimization_suggestions
                .push("Too many options may be degrading performance".to_string());
        }

        if decision.confidence < 0.5 {
            optimization_suggestions
                .push("Low confidence decisions require more information gathering".to_string());
        }

        // Form-specific optimization suggestions
        match form_name {
            "Riddling Sage" => {
                if phase_timings.information_gathering_ms < 1000 {
                    optimization_suggestions.push(
                        "Analytical forms benefit from deeper information gathering".to_string(),
                    );
                }
            }
            "Mischievous Helper" => {
                if phase_timings.options_enhancement_ms < 500 {
                    optimization_suggestions.push(
                        "Creative forms should spend more time on option generation".to_string(),
                    );
                }
            }
            "Chaos Revealer" => {
                if phase_timings.decision_making_ms > 1000 {
                    optimization_suggestions
                        .push("Intuitive forms should make faster decisions".to_string());
                }
            }
            _ => {}
        }

        Ok(DecisionPerformanceMetrics {
            timing: phase_timings.clone(),
            efficiency_score,
            bottleneck_phase,
            tool_utilization,
            memory_efficiency,
            cognitive_load,
            optimization_suggestions,
        })
    }

    /// Update performance statistics for optimization and learning
    async fn update_performance_statistics(
        &self,
        performance_metrics: &DecisionPerformanceMetrics,
        form_name: &str,
    ) {
        let mut stats = self.stats.write().await;

        // Update timing statistics
        let decision_time_ms = performance_metrics.timing.total_decision_ms as f64;
        let current_avg = stats.avg_decision_time_ms;
        let total_decisions = stats.decisions_made as f64;

        stats.avg_decision_time_ms = if total_decisions > 0.0 {
            (current_avg * total_decisions + decision_time_ms) / (total_decisions + 1.0)
        } else {
            decision_time_ms
        };

        // Track efficiency patterns by archetypal form
        if !performance_metrics.optimization_suggestions.is_empty() {
            debug!(
                "Performance optimization suggestions for {}: {:?}",
                form_name, performance_metrics.optimization_suggestions
            );
        }

        // Track high-performance decisions
        if performance_metrics.efficiency_score > 0.8 {
            stats.high_confidence_decisions += 1;
        }

        // Store performance metrics in memory for future optimization
        if let Ok(performance_data) = serde_json::to_string(performance_metrics) {
            let _ = self
                .memory
                .store(
                    format!("Performance metrics for {}: {}", form_name, performance_data),
                    vec![form_name.to_string(), "performance_metrics".to_string()],
                    MemoryMetadata {
                        source: "decision_performance_tracker".to_string(),
                        tags: vec![
                            "performance".to_string(),
                            "optimization".to_string(),
                            form_name.to_lowercase().replace(" ", "_"),
                        ],
                        importance: 0.5,
                        associations: vec![
                            MemoryId::from_string(format!(
                                "efficiency_{:.1}",
                                performance_metrics.efficiency_score
                            )),
                            MemoryId::from_string(format!(
                                "bottleneck_{}",
                                performance_metrics.bottleneck_phase
                            )),
                        ],
                        context: Some("performance statistics update".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                        category: "performance".to_string(),
                    },
                )
                .await;
        }
    }

    /// Create decision associations based on the archetypal form and decision
    /// content
    pub async fn create_decision_associations(
        &self,
        decision: &Decision,
        form_name: &str,
    ) -> Result<Vec<String>> {
        let mut associations = Vec::new();

        // Add form-specific associations
        associations
            .push(format!("archetypal_form_{}", form_name.to_lowercase().replace(" ", "_")));

        // Add decision context associations
        let context_words: Vec<&str> = decision
            .context
            .split_whitespace()
            .filter(|word| word.len() > 3 && !is_stop_word(word))
            .take(5)
            .collect();

        for word in context_words {
            associations.push(format!("context_{}", word.to_lowercase()));
        }

        // Add selected option associations
        if let Some(selected_option) = &decision.selected {
            let option_words: Vec<&str> = selected_option
                .description
                .split_whitespace()
                .filter(|word| word.len() > 3 && !is_stop_word(word))
                .take(3)
                .collect();

            for word in option_words {
                associations.push(format!("option_{}", word.to_lowercase()));
            }

            // Add risk level association
            let risk_category = match selected_option.risk_level {
                r if r < 0.3 => "low_risk",
                r if r < 0.7 => "medium_risk",
                _ => "high_risk",
            };
            associations.push(format!("risk_{}", risk_category));
        }

        // Add confidence level association
        let confidence_category = match decision.confidence {
            c if c < 0.5 => "low_confidence",
            c if c < 0.8 => "medium_confidence",
            _ => "high_confidence",
        };
        associations.push(format!("confidence_{}", confidence_category));

        // Add criteria-based associations
        for criterion in &decision.criteria {
            if criterion.weight > 0.7 {
                associations.push(format!("criterion_{}", criterion.name.to_lowercase()));
            }
        }

        // Add reasoning type associations
        let reasoning_types: Vec<String> = decision
            .reasoning
            .iter()
            .map(|step| format!("reasoning_{:?}", step.step_type).to_lowercase())
            .collect();
        associations.extend(reasoning_types);

        // Add archetypal-specific associations based on form
        match form_name {
            "Mischievous Helper" => {
                associations.push("creative_decision".to_string());
                associations.push("exploratory_approach".to_string());
                associations.push("helpfulness_focused".to_string());
            }
            "Riddling Sage" => {
                associations.push("wisdom_based".to_string());
                associations.push("analytical_depth".to_string());
                associations.push("pattern_recognition".to_string());
            }
            "Chaos Revealer" => {
                associations.push("disruptive_insight".to_string());
                associations.push("truth_seeking".to_string());
                associations.push("unconventional_approach".to_string());
            }
            "Shadow Mirror" => {
                associations.push("reflection_based".to_string());
                associations.push("hidden_aspects".to_string());
                associations.push("psychological_insight".to_string());
            }
            "Knowing Innocent" => {
                associations.push("intuitive_wisdom".to_string());
                associations.push("pure_perspective".to_string());
                associations.push("unconditioned_view".to_string());
            }
            "Wise Jester" => {
                associations.push("paradoxical_wisdom".to_string());
                associations.push("humor_based".to_string());
                associations.push("creative_insight".to_string());
            }
            "Liminal Being" => {
                associations.push("boundary_crossing".to_string());
                associations.push("threshold_wisdom".to_string());
                associations.push("transformative_insight".to_string());
            }
            _ => {
                associations.push("unknown_form".to_string());
            }
        }

        // Remove duplicates and limit to reasonable number
        associations.sort();
        associations.dedup();
        associations.truncate(15);

        Ok(associations)
    }
}

// Add missing method implementations for CognitiveMemory
impl CognitiveMemory {
    /// Advanced search functionality with multiple search strategies
    pub async fn search(&self, query: &str) -> Result<Vec<MemoryItem>> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        // Multi-strategy search approach
        let mut search_results = Vec::new();

        // 1. Exact content match search
        let exact_matches = self.search_exact_content(&query_lower).await?;
        for mut item in exact_matches {
            item.relevance_score *= 1.2; // Boost exact matches
            search_results.push(item);
        }

        // 2. Keyword-based search
        let keyword_matches = self.search_by_keywords(&query_words).await?;
        for mut item in keyword_matches {
            // Avoid duplicates from exact match
            if !search_results.iter().any(|existing| existing.id == item.id) {
                item.relevance_score *= 1.0; // Normal weight for keyword matches
                search_results.push(item);
            }
        }

        // 3. Semantic similarity search (existing functionality)
        let semantic_matches = self.retrieve_similar(query, 10).await?;
        for mut item in semantic_matches {
            // Avoid duplicates from previous searches
            if !search_results.iter().any(|existing| existing.id == item.id) {
                item.relevance_score *= 0.9; // Slightly lower weight for semantic matches
                search_results.push(item);
            }
        }

        // 4. Metadata search (tags, timestamps, etc.)
        let metadata_matches = self.search_by_metadata(&query_lower).await?;
        for mut item in metadata_matches {
            if !search_results.iter().any(|existing| existing.id == item.id) {
                item.relevance_score *= 0.8; // Lower weight for metadata matches
                search_results.push(item);
            }
        }

        // 5. Fuzzy search for typos and variations
        let fuzzy_matches = self.search_fuzzy(&query_lower).await?;
        for mut item in fuzzy_matches {
            if !search_results.iter().any(|existing| existing.id == item.id) {
                item.relevance_score *= 0.7; // Lowest weight for fuzzy matches
                search_results.push(item);
            }
        }

        // Sort by enhanced relevance score
        search_results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top results (limit to reasonable number)
        Ok(search_results.into_iter().take(20).collect())
    }

    /// Search for exact content matches
    async fn search_exact_content(&self, query: &str) -> Result<Vec<MemoryItem>> {
        let mut results = Vec::new();

        // Search short-term memory
        {
            let stm = self.get_short_term().read();
            for item in stm.iter() {
                if item.content.to_string().to_lowercase().contains(query) {
                    results.push(item.clone());
                }
            }
        }

        // Search long-term layers
        for layer in self.get_long_term_layers() {
            let ltm = layer.read();
            for item in ltm.iter() {
                if item.content.to_string().to_lowercase().contains(query) {
                    results.push(item.clone());
                }
            }
        }

        Ok(results)
    }

    /// Search by individual keywords with relevance scoring
    async fn search_by_keywords(&self, keywords: &[&str]) -> Result<Vec<MemoryItem>> {
        let mut results = Vec::new();

        // Search short-term memory
        {
            let stm = self.get_short_term().read();
            for item in stm.iter() {
                let content_lower = item.content.to_string().to_lowercase();
                let keyword_matches =
                    keywords.iter().filter(|&&keyword| content_lower.contains(keyword)).count();

                if keyword_matches > 0 {
                    let mut result_item = item.clone();
                    // Score based on keyword match ratio
                    result_item.relevance_score = ((keyword_matches as f64 / keywords.len() as f64)
                        * item.relevance_score as f64)
                        as f32;
                    results.push(result_item);
                }
            }
        }

        // Search long-term layers
        for layer in self.get_long_term_layers() {
            let ltm = layer.read();
            for item in ltm.iter() {
                let content_lower = item.content.to_string().to_lowercase();
                let keyword_matches =
                    keywords.iter().filter(|&&keyword| content_lower.contains(keyword)).count();

                if keyword_matches > 0 {
                    let mut result_item = item.clone();
                    result_item.relevance_score = ((keyword_matches as f64 / keywords.len() as f64)
                        * item.relevance_score as f64)
                        as f32;
                    results.push(result_item);
                }
            }
        }

        Ok(results)
    }

    /// Search by metadata fields (tags, timestamps, etc.)
    async fn search_by_metadata(&self, query: &str) -> Result<Vec<MemoryItem>> {
        let mut results = Vec::new();

        // Search short-term memory
        {
            let stm = self.get_short_term().read();
            for item in stm.iter() {
                if !item.metadata.source.is_empty() {
                    // Check if metadata contains search terms
                    let metadata_str = format!("{:?}", item.metadata).to_lowercase();
                    if metadata_str.contains(query) {
                        results.push(item.clone());
                    }
                }
            }
        }

        // Search long-term layers
        for layer in self.get_long_term_layers() {
            let ltm = layer.read();
            for item in ltm.iter() {
                if !item.metadata.source.is_empty() {
                    let metadata_str = format!("{:?}", item.metadata).to_lowercase();
                    if metadata_str.contains(query) {
                        results.push(item.clone());
                    }
                }
            }
        }

        Ok(results)
    }

    /// Fuzzy search for handling typos and variations
    async fn search_fuzzy(&self, query: &str) -> Result<Vec<MemoryItem>> {
        let mut results = Vec::new();

        // Simple fuzzy matching using edit distance
        let max_distance = (query.len() / 4).max(1); // Allow 25% character differences

        // Search short-term memory
        {
            let stm = self.get_short_term().read();
            for item in stm.iter() {
                let content_lower = item.content.to_string().to_lowercase();
                let words: Vec<&str> = content_lower.split_whitespace().collect();

                for word in words {
                    if Self::edit_distance(query, word) <= max_distance {
                        let mut result_item = item.clone();
                        // Reduce relevance for fuzzy matches
                        result_item.relevance_score *= 0.6;
                        results.push(result_item);
                        break; // Only add once per item
                    }
                }
            }
        }

        // Search long-term layers
        for layer in self.get_long_term_layers() {
            let ltm = layer.read();
            for item in ltm.iter() {
                let content_lower = item.content.to_string().to_lowercase();
                let words: Vec<&str> = content_lower.split_whitespace().collect();

                for word in words {
                    if Self::edit_distance(query, word) <= max_distance {
                        let mut result_item = item.clone();
                        result_item.relevance_score *= 0.6;
                        results.push(result_item);
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Calculate edit distance between two strings for fuzzy matching
    fn edit_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1, // deletion
                        matrix[i][j - 1] + 1, // insertion
                    ),
                    matrix[i - 1][j - 1] + cost, // substitution
                );
            }
        }

        matrix[len1][len2]
    }

    /// Get a memory item by ID
    pub async fn get(&self, memory_id: &str) -> Result<Option<MemoryItem>> {
        // Simple implementation that searches for the memory by ID
        // In a real implementation, this would use an indexed lookup
        let search_results = self.search(memory_id).await?;

        // Find exact ID match
        for item in search_results {
            if item.id.to_string() == memory_id {
                return Ok(Some(item));
            }
        }

        Ok(None)
    }
}

/// Check if a word is a stop word that shouldn't be used for associations
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "this", "that", "with", "have", "will", "from", "they", "know", "want", "been", "good",
        "much", "some", "time", "very", "when", "come", "here", "just", "like", "long", "make",
        "many", "over", "such", "take", "than", "them", "well", "were",
    ];
    STOP_WORDS.contains(&word.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_simple_decision() {
        // Placeholder test
    }
}
