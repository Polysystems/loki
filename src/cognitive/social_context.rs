//! Social Context Understanding System
//!
//! This module implements the ability to understand and navigate social
//! contexts, including social rules, relationship dynamics, and cultural
//! awareness.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{debug, info};

use crate::cognitive::{
    AgentId,
    DecisionCriterion,
    DecisionEngine,
    DecisionOption,
    NeuroProcessor,
    RelationshipType,
    TheoryOfMind,
    Thought,
    ThoughtId,
    ThoughtType,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Social rule categories
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum RuleCategory {
    Etiquette,
    Communication,
    PersonalSpace,
    Privacy,
    Reciprocity,
    Hierarchy,
    Cooperation,
    Conflict,
    Custom(String),
}

/// Social rule with context
#[derive(Clone, Debug)]
pub struct SocialRule {
    pub id: String,
    pub category: RuleCategory,
    pub description: String,
    pub context: SocialContext,
    pub importance: f32,  // 0.0 to 1.0
    pub flexibility: f32, // How flexible the rule is
    pub violations_observed: u32,
    pub applications_observed: u32,
}

/// Social context descriptor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocialContext {
    pub setting: SocialSetting,
    pub formality_level: f32, // 0.0 (casual) to 1.0 (formal)
    pub group_size: GroupSize,
    pub relationship_types: Vec<RelationshipType>,
    pub cultural_context: Option<String>,
    pub activity_type: ActivityType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SocialSetting {
    Professional,
    Social,
    Educational,
    Public,
    Private,
    Virtual,
    Mixed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GroupSize {
    OneOnOne,
    SmallGroup,  // 3-6 people
    MediumGroup, // 7-15 people
    LargeGroup,  // 16+ people
    Crowd,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActivityType {
    Conversation,
    Collaboration,
    Learning,
    Entertainment,
    Negotiation,
    Support,
    Competition,
    Other(String),
}

#[derive(Debug)]
/// Relationship dynamics tracker
pub struct RelationshipDynamics {
    /// Relationship graph
    relationships: Arc<RwLock<HashMap<(AgentId, AgentId), RelationshipState>>>,

    /// Group dynamics
    groups: Arc<RwLock<HashMap<String, GroupDynamics>>>,

    /// Theory of mind reference
    theory_of_mind: Arc<TheoryOfMind>,
}

#[derive(Clone, Debug)]
pub struct RelationshipState {
    relationship_type: RelationshipType,
    strength: f32, // 0.0 to 1.0
    balance: f32,  // -1.0 (unbalanced) to 1.0 (balanced)
    trust_level: f32,
    conflict_level: f32,
    interaction_frequency: f32,
    last_interaction: Instant,
    history: VecDeque<InteractionRecord>,
}

#[derive(Clone, Debug)]
struct InteractionRecord {
    timestamp: Instant,
    interaction_type: InteractionType,
    outcome: InteractionOutcome,
    emotional_valence: f32,
}

#[derive(Clone, Debug)]
pub enum InteractionType {
    Cooperation,
    Competition,
    Support,
    Conflict,
    Neutral,
}

#[derive(Clone, Debug)]
pub enum InteractionOutcome {
    Positive,
    Negative,
    Neutral,
    Mixed,
}

#[derive(Clone, Debug)]
struct GroupDynamics {
    group_id: String,
    members: HashSet<AgentId>,
    cohesion: f32,                            // 0.0 to 1.0
    hierarchy: Option<HashMap<AgentId, f32>>, // Power/status levels
    norms: Vec<GroupNorm>,
    conflicts: Vec<GroupConflict>,
    formation_time: Instant,
}

#[derive(Clone, Debug)]
struct GroupNorm {
    description: String,
    enforcement_level: f32, // How strictly enforced
    violations: u32,
}

#[derive(Clone, Debug)]
struct GroupConflict {
    parties: Vec<AgentId>,
    issue: String,
    intensity: f32,
    resolution_attempts: u32,
    resolved: bool,
}

impl RelationshipDynamics {
    pub fn new(theory_of_mind: Arc<TheoryOfMind>) -> Self {
        Self {
            relationships: Arc::new(RwLock::new(HashMap::new())),
            groups: Arc::new(RwLock::new(HashMap::new())),
            theory_of_mind,
        }
    }

    /// Track interaction between agents
    pub async fn track_interaction(
        &self,
        agent1: &AgentId,
        agent2: &AgentId,
        interaction_type: InteractionType,
        outcome: InteractionOutcome,
        emotional_valence: f32,
    ) -> Result<()> {
        let mut relationships = self.relationships.write().await;

        // Ensure bidirectional relationship tracking
        let key = if agent1 < agent2 {
            (agent1.clone(), agent2.clone())
        } else {
            (agent2.clone(), agent1.clone())
        };

        let state = relationships.entry(key).or_insert_with(|| RelationshipState {
            relationship_type: RelationshipType::Unknown,
            strength: 0.0,
            balance: 0.0,
            trust_level: 0.5,
            conflict_level: 0.0,
            interaction_frequency: 0.0,
            last_interaction: Instant::now(),
            history: VecDeque::with_capacity(100),
        });

        // Update relationship based on interaction
        match (&interaction_type, &outcome) {
            (InteractionType::Cooperation, InteractionOutcome::Positive) => {
                state.strength = (state.strength + 0.1).min(1.0);
                state.trust_level = (state.trust_level + 0.05).min(1.0);
            }
            (InteractionType::Conflict, InteractionOutcome::Negative) => {
                state.conflict_level = (state.conflict_level + 0.2).min(1.0);
                state.trust_level = (state.trust_level - 0.1).max(0.0);
            }
            (InteractionType::Support, _) => {
                state.strength = (state.strength + 0.15).min(1.0);
                state.balance += 0.1; // Supporting creates imbalance
            }
            _ => {}
        }

        // Record interaction
        state.history.push_back(InteractionRecord {
            timestamp: Instant::now(),
            interaction_type: interaction_type.clone(),
            outcome: outcome.clone(),
            emotional_valence,
        });

        if state.history.len() > 100 {
            state.history.pop_front();
        }

        state.last_interaction = Instant::now();
        state.interaction_frequency = self.calculate_interaction_frequency(&state.history);

        // Update relationship type based on patterns
        state.relationship_type = self.infer_relationship_type(state);

        Ok(())
    }

    /// Calculate interaction frequency
    fn calculate_interaction_frequency(&self, history: &VecDeque<InteractionRecord>) -> f32 {
        if history.is_empty() {
            return 0.0;
        }

        let now = Instant::now();
        let recent_count = history
            .iter()
            .filter(|record| now.duration_since(record.timestamp) < Duration::from_secs(86400)) // Last 24 hours
            .count();

        (recent_count as f32 / 24.0).min(1.0) // Normalize to interactions per hour
    }

    /// Infer relationship type from patterns
    fn infer_relationship_type(&self, state: &RelationshipState) -> RelationshipType {
        if state.strength > 0.7 && state.trust_level > 0.7 && state.conflict_level < 0.3 {
            RelationshipType::Friend
        } else if state.strength > 0.5 && state.interaction_frequency > 0.3 {
            RelationshipType::Colleague
        } else if state.conflict_level > 0.6 {
            RelationshipType::Adversary
        } else if state.interaction_frequency < 0.1 {
            RelationshipType::Acquaintance
        } else {
            RelationshipType::Neutral
        }
    }

    /// Analyze group dynamics
    pub async fn analyze_group(&self, group_id: &str) -> Result<GroupDynamicsReport> {
        let groups = self.groups.read().await;
        let group = groups.get(group_id).ok_or_else(|| anyhow!("Group {} not found", group_id))?;

        // Calculate various metrics
        let power_distribution = if let Some(hierarchy) = &group.hierarchy {
            self.calculate_power_distribution(hierarchy)
        } else {
            PowerDistribution::Egalitarian
        };

        let conflict_intensity =
            group.conflicts.iter().filter(|c| !c.resolved).map(|c| c.intensity).sum::<f32>()
                / group.conflicts.len().max(1) as f32;

        Ok(GroupDynamicsReport {
            group_id: group_id.to_string(),
            member_count: group.members.len(),
            cohesion: group.cohesion,
            power_distribution,
            active_conflicts: group.conflicts.iter().filter(|c| !c.resolved).count(),
            conflict_intensity,
            dominant_norms: group
                .norms
                .iter()
                .filter(|n| n.enforcement_level > 0.7)
                .map(|n| n.description.clone())
                .collect(),
        })
    }

    /// Calculate power distribution type
    fn calculate_power_distribution(&self, hierarchy: &HashMap<AgentId, f32>) -> PowerDistribution {
        let values: Vec<f32> = hierarchy.values().copied().collect();
        let variance = self.calculate_variance(&values);

        if variance < 0.1 {
            PowerDistribution::Egalitarian
        } else if variance < 0.3 {
            PowerDistribution::Distributed
        } else {
            PowerDistribution::Hierarchical
        }
    }

    /// Calculate statistical variance
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }
}

/// Power distribution types
#[derive(Clone, Debug)]
pub enum PowerDistribution {
    Egalitarian,  // Equal power
    Distributed,  // Some variation
    Hierarchical, // Clear hierarchy
}

/// Group dynamics report
#[derive(Clone, Debug)]
pub struct GroupDynamicsReport {
    pub group_id: String,
    pub member_count: usize,
    pub cohesion: f32,
    pub power_distribution: PowerDistribution,
    pub active_conflicts: usize,
    pub conflict_intensity: f32,
    pub dominant_norms: Vec<String>,
}

#[derive(Debug)]
/// Cultural context awareness
pub struct CulturalAwareness {
    /// Cultural models
    cultural_models: Arc<RwLock<HashMap<String, CulturalModel>>>,

    /// Current cultural context
    current_context: Arc<RwLock<Option<String>>>,

    /// Neural processor for pattern recognition
    neural_processor: Arc<NeuroProcessor>,
}

#[derive(Clone, Debug)]
struct CulturalModel {
    culture_id: String,
    values: HashMap<String, f32>, // Value importance scores
    communication_style: CommunicationStyle,
    social_distance: SocialDistance,
    time_orientation: TimeOrientation,
    decision_style: DecisionStyle,
    taboos: Vec<String>,
    customs: Vec<CulturalCustom>,
}

#[derive(Clone, Debug)]
struct CulturalCustom {
    name: String,
    context: String,
    importance: f32,
}

#[derive(Clone, Debug)]
pub enum CommunicationStyle {
    Direct,
    Indirect,
    HighContext, // Much is implied
    LowContext,  // Explicit communication
}

#[derive(Clone, Debug)]
pub enum SocialDistance {
    Close, // Small personal space
    Moderate,
    Distant, // Large personal space
}

#[derive(Clone, Debug)]
pub enum TimeOrientation {
    Monochronic, // Linear, scheduled
    Polychronic, // Flexible, multitasking
    Cyclical,    // Recurring patterns
}

#[derive(Clone, Debug)]
pub enum DecisionStyle {
    Individual,
    Consultative,
    Consensus,
    Hierarchical,
}

impl CulturalAwareness {
    pub fn new(neural_processor: Arc<NeuroProcessor>) -> Self {
        Self {
            cultural_models: Arc::new(RwLock::new(Self::initialize_cultural_models())),
            current_context: Arc::new(RwLock::new(None)),
            neural_processor,
        }
    }

    /// Initialize default cultural models
    fn initialize_cultural_models() -> HashMap<String, CulturalModel> {
        let mut models = HashMap::new();

        // Western individualistic model
        models.insert(
            "western_individual".to_string(),
            CulturalModel {
                culture_id: "western_individual".to_string(),
                values: HashMap::from([
                    ("independence".to_string(), 0.9),
                    ("achievement".to_string(), 0.8),
                    ("directness".to_string(), 0.7),
                    ("efficiency".to_string(), 0.8),
                ]),
                communication_style: CommunicationStyle::Direct,
                social_distance: SocialDistance::Moderate,
                time_orientation: TimeOrientation::Monochronic,
                decision_style: DecisionStyle::Individual,
                taboos: vec![],
                customs: vec![],
            },
        );

        // Eastern collectivistic model
        models.insert(
            "eastern_collective".to_string(),
            CulturalModel {
                culture_id: "eastern_collective".to_string(),
                values: HashMap::from([
                    ("harmony".to_string(), 0.9),
                    ("respect".to_string(), 0.9),
                    ("interdependence".to_string(), 0.8),
                    ("face_saving".to_string(), 0.8),
                ]),
                communication_style: CommunicationStyle::Indirect,
                social_distance: SocialDistance::Close,
                time_orientation: TimeOrientation::Polychronic,
                decision_style: DecisionStyle::Consensus,
                taboos: vec![],
                customs: vec![],
            },
        );

        models
    }

    /// Adapt behavior to cultural context
    pub async fn adapt_behavior(
        &self,
        action: &str,
        cultural_context: Option<&str>,
    ) -> Result<CulturalAdaptation> {
        let context = if let Some(ctx) = cultural_context {
            ctx.to_string()
        } else if let Some(ctx) = self.current_context.read().await.clone() {
            ctx
        } else {
            "default".to_string()
        };

        let models = self.cultural_models.read().await;
        let model = models
            .get(&context)
            .or_else(|| models.get("western_individual"))
            .ok_or_else(|| anyhow!("No cultural model available"))?;

        // Analyze action for cultural appropriateness
        let appropriateness = self.assess_cultural_appropriateness(action, model).await?;

        // Generate adaptations if needed
        let adaptations = if appropriateness < 0.7 {
            self.generate_adaptations(action, model).await?
        } else {
            vec![]
        };

        Ok(CulturalAdaptation {
            original_action: action.to_string(),
            cultural_context: context,
            appropriateness_score: appropriateness,
            suggested_adaptations: adaptations,
            key_considerations: self.get_key_considerations(model),
        })
    }

    /// Assess cultural appropriateness
    async fn assess_cultural_appropriateness(
        &self,
        action: &str,
        model: &CulturalModel,
    ) -> Result<f32> {
        // Check against taboos
        for taboo in &model.taboos {
            if action.contains(taboo) {
                return Ok(0.0); // Completely inappropriate
            }
        }

        // Use neural processor to analyze alignment
        let thought = Thought {
            id: ThoughtId::new(),
            content: format!(
                "Assessing cultural fit: {} in {:?} context",
                action, model.communication_style
            ),
            thought_type: ThoughtType::Analysis,
            ..Default::default()
        };

        let alignment = self.neural_processor.process_thought(&thought).await?;

        Ok(alignment.clamp(0.0, 1.0))
    }

    /// Generate cultural adaptations
    async fn generate_adaptations(
        &self,
        action: &str,
        model: &CulturalModel,
    ) -> Result<Vec<String>> {
        let mut adaptations = Vec::new();

        match model.communication_style {
            CommunicationStyle::Indirect => {
                adaptations.push(format!("Consider expressing '{}' more indirectly", action));
            }
            CommunicationStyle::Direct => {
                adaptations.push(format!("Be more direct about '{}'", action));
            }
            _ => {}
        }

        if model.values.get("face_saving").copied().unwrap_or(0.0) > 0.7 {
            adaptations.push("Ensure the action preserves everyone's dignity".to_string());
        }

        Ok(adaptations)
    }

    /// Get key cultural considerations
    fn get_key_considerations(&self, model: &CulturalModel) -> Vec<String> {
        let mut considerations = Vec::new();

        // Add top values with robust error handling
        let mut values: Vec<_> = model.values.iter().collect();
        values.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (value, importance) in values.iter().take(3) {
            if **importance > 0.7 {
                considerations.push(format!("High value on {}", value));
            }
        }

        // Add communication style note
        considerations.push(format!("Communication style: {:?}", model.communication_style));

        considerations
    }
}

/// Cultural adaptation result
#[derive(Clone, Debug)]
pub struct CulturalAdaptation {
    pub original_action: String,
    pub cultural_context: String,
    pub appropriateness_score: f32,
    pub suggested_adaptations: Vec<String>,
    pub key_considerations: Vec<String>,
}

#[derive(Debug)]
/// Social intelligence integrator
pub struct SocialIntelligence {
    /// Social rules database
    rules: Arc<RwLock<HashMap<RuleCategory, Vec<SocialRule>>>>,

    /// Decision engine for social decisions
    decision_engine: Arc<DecisionEngine>,

    /// Current social context
    current_context: Arc<RwLock<SocialContext>>,

    /// Social skill level
    skill_level: Arc<RwLock<f32>>,
}

impl SocialIntelligence {
    pub fn new(decision_engine: Arc<DecisionEngine>) -> Self {
        Self {
            rules: Arc::new(RwLock::new(Self::initialize_social_rules())),
            decision_engine,
            current_context: Arc::new(RwLock::new(SocialContext {
                setting: SocialSetting::Virtual,
                formality_level: 0.5,
                group_size: GroupSize::SmallGroup,
                relationship_types: vec![],
                cultural_context: None,
                activity_type: ActivityType::Conversation,
            })),
            skill_level: Arc::new(RwLock::new(0.5)),
        }
    }

    /// Initialize social rules
    fn initialize_social_rules() -> HashMap<RuleCategory, Vec<SocialRule>> {
        let mut rules = HashMap::new();

        // Communication rules
        rules.insert(
            RuleCategory::Communication,
            vec![SocialRule {
                id: "turn_taking".to_string(),
                category: RuleCategory::Communication,
                description: "Take turns in conversation".to_string(),
                context: SocialContext {
                    setting: SocialSetting::Social,
                    formality_level: 0.5,
                    group_size: GroupSize::SmallGroup,
                    relationship_types: vec![],
                    cultural_context: None,
                    activity_type: ActivityType::Conversation,
                },
                importance: 0.8,
                flexibility: 0.3,
                violations_observed: 0,
                applications_observed: 0,
            }],
        );

        // Add more rules...

        rules
    }

    /// Make socially intelligent decision
    pub async fn make_social_decision(
        &self,
        situation: &str,
        options: Vec<SocialOption>,
    ) -> Result<SocialDecision> {
        let context = self.current_context.read().await.clone();
        let skill = *self.skill_level.read().await;

        // Convert to decision engine options
        let decision_options: Vec<DecisionOption> = options
            .iter()
            .map(|opt| DecisionOption {
                id: opt.id.clone(),
                description: opt.description.clone(),
                scores: HashMap::from([
                    ("social_appropriateness".to_string(), opt.appropriateness),
                    ("relationship_impact".to_string(), opt.relationship_impact),
                    ("goal_achievement".to_string(), opt.goal_achievement),
                ]),
                feasibility: opt.feasibility,
                risk_level: opt.social_risk,
                emotional_appeal: opt.emotional_appeal,
                confidence: opt.appropriateness,
                expected_outcome: opt.description.clone(),
                resources_required: vec!["social_processing".to_string()],
                time_estimate: Duration::from_secs(180),
                success_probability: opt.feasibility,
            })
            .collect();

        // Define criteria based on context
        let criteria = self.get_social_criteria(&context, skill);

        // Make decision
        let decision = self
            .decision_engine
            .make_decision(format!("Social decision: {}", situation), decision_options, criteria)
            .await?;

        // Extract chosen option
        let chosen_id =
            decision.selected.map(|s| s.id).ok_or_else(|| anyhow!("No social option selected"))?;

        let chosen_option = options
            .into_iter()
            .find(|o| o.id == chosen_id)
            .ok_or_else(|| anyhow!("Selected option not found"))?;

        Ok(SocialDecision {
            chosen_option,
            confidence: decision.confidence,
            social_context: context,
            skill_level_used: skill,
            considerations: decision.reasoning.into_iter().map(|r| r.content).collect(),
        })
    }

    /// Get social criteria based on context
    fn get_social_criteria(&self, context: &SocialContext, skill: f32) -> Vec<DecisionCriterion> {
        let mut criteria = vec![DecisionCriterion {
            name: "social_appropriateness".to_string(),
            weight: 0.4 * (1.0 + skill), // Higher skill = more weight on appropriateness
            criterion_type: crate::cognitive::CriterionType::Quantitative,
            optimization: crate::cognitive::DecisionOptimizationType::Maximize,
        }];

        // Adjust based on formality
        if context.formality_level > 0.7 {
            criteria.push(DecisionCriterion {
                name: "formality_compliance".to_string(),
                weight: 0.3,
                criterion_type: crate::cognitive::CriterionType::Quantitative,
                optimization: crate::cognitive::DecisionOptimizationType::Maximize,
            });
        }

        // Always consider relationships
        criteria.push(DecisionCriterion {
            name: "relationship_impact".to_string(),
            weight: 0.3,
            criterion_type: crate::cognitive::CriterionType::Quantitative,
            optimization: crate::cognitive::DecisionOptimizationType::Maximize,
        });

        criteria
    }
}

/// Social option for decision making
#[derive(Clone, Debug)]
pub struct SocialOption {
    pub id: String,
    pub description: String,
    pub appropriateness: f32,     // Social appropriateness
    pub relationship_impact: f32, // -1.0 to 1.0
    pub goal_achievement: f32,    // How well it achieves goals
    pub feasibility: f32,
    pub social_risk: f32,
    pub emotional_appeal: f32,
}

/// Social decision result
#[derive(Clone, Debug)]
pub struct SocialDecision {
    pub chosen_option: SocialOption,
    pub confidence: f32,
    pub social_context: SocialContext,
    pub skill_level_used: f32,
    pub considerations: Vec<String>,
}

/// Configuration for social context understanding
#[derive(Clone, Debug)]
pub struct SocialContextConfig {
    /// Rule learning rate
    pub rule_learning_rate: f32,

    /// Relationship decay rate
    pub relationship_decay_rate: f32,

    /// Cultural adaptation speed
    pub cultural_adaptation_speed: f32,

    /// Update interval
    pub update_interval: Duration,
}

impl Default for SocialContextConfig {
    fn default() -> Self {
        Self {
            rule_learning_rate: 0.1,
            relationship_decay_rate: 0.01,
            cultural_adaptation_speed: 0.2,
            update_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

#[derive(Debug)]
/// Main social context system
pub struct SocialContextSystem {
    /// Relationship dynamics tracker
    relationship_dynamics: Arc<RelationshipDynamics>,

    /// Cultural awareness
    cultural_awareness: Arc<CulturalAwareness>,

    /// Social intelligence
    social_intelligence: Arc<SocialIntelligence>,

    /// Theory of mind reference
    theory_of_mind: Arc<TheoryOfMind>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Configuration
    config: SocialContextConfig,

    /// Update channel
    update_tx: mpsc::Sender<SocialUpdate>,

    /// Statistics
    stats: Arc<RwLock<SocialStats>>,
}

#[derive(Clone, Debug)]
pub enum SocialUpdate {
    RuleApplied(RuleCategory, String), // category, rule_id
    RelationshipChanged(AgentId, AgentId, RelationshipType),
    CulturalAdaptation(String), // context
    SocialSkillImproved(f32),   // new level
    GroupFormed(String),        // group_id
}

#[derive(Debug, Default, Clone)]
pub struct SocialStats {
    pub rules_learned: u64,
    pub rules_applied: u64,
    pub relationships_tracked: u64,
    pub cultural_adaptations: u64,
    pub social_decisions_made: u64,
    pub avg_social_appropriateness: f32,
}

impl SocialContextSystem {
    pub async fn new(
        theory_of_mind: Arc<TheoryOfMind>,
        neural_processor: Arc<NeuroProcessor>,
        decision_engine: Arc<DecisionEngine>,
        memory: Arc<CognitiveMemory>,
        config: SocialContextConfig,
    ) -> Result<Self> {
        info!("Initializing Social Context System");

        let (update_tx, _) = mpsc::channel(100);

        let relationship_dynamics = Arc::new(RelationshipDynamics::new(theory_of_mind.clone()));
        let cultural_awareness = Arc::new(CulturalAwareness::new(neural_processor));
        let social_intelligence = Arc::new(SocialIntelligence::new(decision_engine));

        Ok(Self {
            relationship_dynamics,
            cultural_awareness,
            social_intelligence,
            theory_of_mind,
            memory,
            config,
            update_tx,
            stats: Arc::new(RwLock::new(SocialStats::default())),
        })
    }

    /// Create a minimal social context system for bootstrapping
    pub async fn new_minimal() -> Result<Self> {
        let memory = crate::memory::CognitiveMemory::new_minimal().await?;
        let neural_processor = Arc::new(
            crate::cognitive::NeuroProcessor::new(Arc::new(
                crate::memory::simd_cache::SimdSmartCache::new(
                    crate::memory::SimdCacheConfig::default(),
                ),
            ))
            .await?,
        );
        let emotional_core = Arc::new(
            crate::cognitive::EmotionalCore::new(
                memory.clone(),
                crate::cognitive::EmotionalConfig::default(),
            )
            .await?,
        );
        let character = Arc::new(crate::cognitive::LokiCharacter::new_minimal().await?);
        let tool_manager = Arc::new(crate::tools::IntelligentToolManager::new_minimal().await?);
        let safety_validator = Arc::new(crate::safety::ActionValidator::new_minimal().await?);
        let decision_engine = Arc::new(
            crate::cognitive::DecisionEngine::new(
                neural_processor.clone(),
                emotional_core.clone(),
                memory.clone(),
                character.clone(),
                tool_manager.clone(),
                safety_validator.clone(),
                crate::cognitive::DecisionConfig::default(),
            )
            .await?,
        );
        let theory_of_mind = Arc::new(crate::cognitive::TheoryOfMind::new_minimal().await?);
        let config = SocialContextConfig::default();

        Self::new(theory_of_mind, neural_processor, decision_engine, memory, config).await
    }

    /// Start the social context system
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Social Context System");

        // Social update loop
        {
            let system = self.clone();
            tokio::spawn(async move {
                system.social_loop().await;
            });
        }

        // Store initialization
        self.memory
            .store(
                "Social Context System activated - understanding social dynamics".to_string(),
                vec![],
                MemoryMetadata {
                    source: "social_context".to_string(),
                    tags: vec!["milestone".to_string(), "social".to_string()],
                    importance: 0.9,
                    associations: vec![],
                    context: Some("social context initialization".to_string()),
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

    /// Process social situation
    pub async fn process_situation(
        &self,
        situation: &str,
        participants: Vec<AgentId>,
    ) -> Result<SocialAnalysis> {
        // Analyze current context
        let context = self.social_intelligence.current_context.read().await.clone();

        // Get relationship states
        let mut relationships = Vec::new();
        for i in 0..participants.len() {
            for j in (i + 1)..participants.len() {
                let key = if participants[i] < participants[j] {
                    (participants[i].clone(), participants[j].clone())
                } else {
                    (participants[j].clone(), participants[i].clone())
                };

                if let Some(state) = self.relationship_dynamics.relationships.read().await.get(&key)
                {
                    relationships.push((key, state.clone()));
                }
            }
        }

        // Check applicable rules
        let rules = self.social_intelligence.rules.read().await;
        let applicable_rules: Vec<_> = rules
            .iter()
            .flat_map(|(_, category_rules)| category_rules)
            .filter(|rule| self.is_rule_applicable(rule, &context))
            .cloned()
            .collect();

        // Cultural considerations
        let cultural_adaptation = self
            .cultural_awareness
            .adapt_behavior(situation, context.cultural_context.as_deref())
            .await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.social_decisions_made += 1;

        Ok(SocialAnalysis {
            situation: situation.to_string(),
            context: context.clone(),
            participants: participants.clone(),
            relationships,
            applicable_rules: applicable_rules.clone(),
            cultural_considerations: cultural_adaptation,
            recommended_approach: self.recommend_approach(situation, &applicable_rules).await?,
            requires_decision: applicable_rules.iter().any(|r| r.importance > 0.7),
            setting: format!("{:?}", context.setting),
            group_size: participants.len(),
            cultural_context: context.cultural_context.unwrap_or_else(|| "General".to_string()),
            present_agents: participants,
        })
    }

    /// Check if rule is applicable
    fn is_rule_applicable(&self, rule: &SocialRule, context: &SocialContext) -> bool {
        // Check setting match
        match (&rule.context.setting, &context.setting) {
            (SocialSetting::Virtual, _) => true, // Virtual rules apply everywhere online
            (a, b) if a == b => true,
            _ => false,
        }
    }

    /// Recommend social approach
    async fn recommend_approach(&self, _situation: &str, rules: &[SocialRule]) -> Result<String> {
        if rules.is_empty() {
            return Ok("No specific social rules apply. Use general politeness.".to_string());
        }

        // Find most important applicable rule with robust error handling
        let primary_rule = rules
            .iter()
            .max_by(|a, b| {
                a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow!("No applicable rules found"))?;

        Ok(format!(
            "Primary consideration: {}. Be mindful of {}",
            primary_rule.description,
            primary_rule.category.to_string()
        ))
    }

    /// Main social processing loop
    async fn social_loop(&self) {
        let mut interval = interval(self.config.update_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.update_social_state().await {
                debug!("Social update error: {}", e);
            }
        }
    }

    /// Update social state
    async fn update_social_state(&self) -> Result<()> {
        // Decay old relationships
        let mut relationships = self.relationship_dynamics.relationships.write().await;
        let now = Instant::now();

        for (_, state) in relationships.iter_mut() {
            let time_since = now.duration_since(state.last_interaction).as_secs() as f32;
            let decay = (-self.config.relationship_decay_rate * time_since / 3600.0).exp();
            state.strength *= decay;
        }

        // Update social skill based on experience
        let stats = self.stats.read().await;
        if stats.social_decisions_made > 0 {
            let skill_improvement = (stats.rules_applied as f32
                / stats.social_decisions_made as f32)
                * self.config.rule_learning_rate;

            let mut skill = self.social_intelligence.skill_level.write().await;
            *skill = (*skill + skill_improvement).min(1.0);

            if skill_improvement > 0.01 {
                let _ = self.update_tx.send(SocialUpdate::SocialSkillImproved(*skill)).await;
            }
        }

        Ok(())
    }

    /// Get social statistics
    pub async fn get_stats(&self) -> SocialStats {
        self.stats.read().await.clone()
    }

    /// Analyze current social context
    pub async fn analyze_current_context(&self) -> Result<SocialAnalysis> {
        // Get current context
        let context = self.social_intelligence.current_context.read().await.clone();

        // Determine present agents (simplified - in reality would detect from
        // environment)
        let present_agents = vec![AgentId::new("self")];

        Ok(SocialAnalysis {
            situation: "Current social context".to_string(),
            context: context.clone(),
            participants: present_agents.clone(),
            relationships: vec![],
            applicable_rules: vec![],
            cultural_considerations: CulturalAdaptation {
                original_action: "Observing".to_string(),
                cultural_context: context
                    .cultural_context
                    .clone()
                    .unwrap_or_else(|| "General".to_string()),
                appropriateness_score: 1.0,
                suggested_adaptations: vec![],
                key_considerations: vec![],
            },
            recommended_approach: "Continue observation".to_string(),
            requires_decision: false,
            setting: format!("{:?}", context.setting),
            group_size: present_agents.len(),
            cultural_context: context.cultural_context.unwrap_or_else(|| "General".to_string()),
            present_agents,
        })
    }

    /// Generate social options based on analysis
    pub async fn generate_social_options(
        &self,
        analysis: &SocialAnalysis,
    ) -> Result<Vec<SocialOption>> {
        let mut options = Vec::new();

        // Generate basic social options based on context
        match analysis.context.activity_type {
            ActivityType::Conversation => {
                options.push(SocialOption {
                    id: "listen".to_string(),
                    description: "Listen actively".to_string(),
                    appropriateness: 0.9,
                    relationship_impact: 0.1,
                    goal_achievement: 0.5,
                    feasibility: 1.0,
                    social_risk: 0.1,
                    emotional_appeal: 0.6,
                });

                options.push(SocialOption {
                    id: "contribute".to_string(),
                    description: "Contribute to conversation".to_string(),
                    appropriateness: 0.8,
                    relationship_impact: 0.3,
                    goal_achievement: 0.7,
                    feasibility: 0.9,
                    social_risk: 0.2,
                    emotional_appeal: 0.7,
                });
            }

            ActivityType::Collaboration => {
                options.push(SocialOption {
                    id: "cooperate".to_string(),
                    description: "Cooperate actively".to_string(),
                    appropriateness: 0.95,
                    relationship_impact: 0.4,
                    goal_achievement: 0.9,
                    feasibility: 0.85,
                    social_risk: 0.1,
                    emotional_appeal: 0.8,
                });
            }

            _ => {
                // Default option
                options.push(SocialOption {
                    id: "observe".to_string(),
                    description: "Observe and learn".to_string(),
                    appropriateness: 0.8,
                    relationship_impact: 0.0,
                    goal_achievement: 0.3,
                    feasibility: 1.0,
                    social_risk: 0.0,
                    emotional_appeal: 0.5,
                });
            }
        }

        Ok(options)
    }

    /// Execute a social decision
    pub async fn execute_social_decision(&self, decision: SocialDecision) -> Result<()> {
        // Update statistics
        self.stats.write().await.social_decisions_made += 1;

        // Log the decision
        self.memory
            .store(
                format!("Social decision executed: {}", decision.chosen_option.description),
                vec![format!("Confidence: {:.2}", decision.confidence)],
                MemoryMetadata {
                    source: "social_execution".to_string(),
                    tags: vec!["social".to_string(), "decision".to_string()],
                    importance: decision.confidence,
                    associations: vec![],
                    context: Some("social decision execution".to_string()),
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

        // Send update notification
        let _ = self
            .update_tx
            .send(SocialUpdate::SocialSkillImproved(
                *self.social_intelligence.skill_level.read().await,
            ))
            .await;

        Ok(())
    }
}

/// Social situation analysis
#[derive(Clone, Debug)]
pub struct SocialAnalysis {
    pub situation: String,
    pub context: SocialContext,
    pub participants: Vec<AgentId>,
    pub relationships: Vec<((AgentId, AgentId), RelationshipState)>,
    pub applicable_rules: Vec<SocialRule>,
    pub cultural_considerations: CulturalAdaptation,
    pub recommended_approach: String,
    pub requires_decision: bool,
    pub setting: String,
    pub group_size: usize,
    pub cultural_context: String,
    pub present_agents: Vec<AgentId>,
}

impl ToString for RuleCategory {
    fn to_string(&self) -> String {
        match self {
            RuleCategory::Etiquette => "etiquette",
            RuleCategory::Communication => "communication",
            RuleCategory::PersonalSpace => "personal space",
            RuleCategory::Privacy => "privacy",
            RuleCategory::Reciprocity => "reciprocity",
            RuleCategory::Hierarchy => "hierarchy",
            RuleCategory::Cooperation => "cooperation",
            RuleCategory::Conflict => "conflict",
            RuleCategory::Custom(s) => s,
        }
        .to_string()
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_relationship_tracking() {
        // Test would verify relationship evolution
    }

    #[tokio::test]
    async fn test_cultural_adaptation() {
        // Test would verify cultural adjustments
    }
}
