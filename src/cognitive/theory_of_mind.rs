//! Theory of Mind System
//!
//! This module implements the ability to model and reason about other agents'
//! mental states, including their beliefs, desires, intentions, and knowledge.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{info, warn};

use crate::cognitive::{
    CoreEmotion,
    EmotionalBlend,
    EmotionalCore,
    GoalId,
    NeuroProcessor,
    Thought,
    ThoughtId,
    ThoughtType,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Unique identifier for agents
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }

    pub fn system() -> Self {
        Self("SYSTEM".to_string())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Mental model of another agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MentalModel {
    pub agent_id: AgentId,
    pub beliefs: Vec<Belief>,
    pub desires: Vec<Desire>,
    pub intentions: Vec<Intention>,
    pub emotional_state: EmotionalStateModel,
    pub knowledge_base: Vec<Knowledge>,
    pub personality_traits: PersonalityProfile,
    pub relationship_status: RelationshipStatus,
    #[serde(skip, default = "Instant::now")]
    pub last_updated: Instant,
    pub confidence: f32, // 0.0 to 1.0
}

/// Belief held by an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Belief {
    pub id: String,
    pub content: String,
    pub confidence: f32, // How strongly they believe it
    pub source: BeliefSource,
    pub evidence: Vec<String>, // Supporting observations
    #[serde(skip, default = "Instant::now")]
    pub formed_at: Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BeliefSource {
    Observation,
    Communication,
    Inference,
    Assumption,
}

/// Desire or goal of an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Desire {
    pub id: String,
    pub description: String,
    pub priority: f32, // 0.0 to 1.0
    pub desire_type: DesireType,
    pub target: Option<String>, // Object of desire
    pub strength: f32,          // How much they want it
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DesireType {
    Achievement, // Want to accomplish something
    Acquisition, // Want to obtain something
    Experience,  // Want to experience something
    Avoidance,   // Want to avoid something
    Social,      // Want social connection
    Knowledge,   // Want to know/understand
}

/// Intention or plan of an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Intention {
    pub id: String,
    pub description: String,
    pub goal: Option<GoalId>,  // Associated goal
    pub commitment_level: f32, // 0.0 to 1.0
    pub likelihood: f32,       // Probability they'll follow through
    pub timeline: IntentionTimeline,
    pub preconditions: Vec<String>,
    pub confidence: f32, // Confidence in this prediction
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IntentionTimeline {
    Immediate,  // Within minutes
    ShortTerm,  // Within hours
    MediumTerm, // Within days
    LongTerm,   // Within weeks/months
    Indefinite,
}

/// Model of agent's emotional state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalStateModel {
    pub primary_emotion: CoreEmotion,
    pub emotional_blend: EmotionalBlend,
    pub valence: f32,   // -1.0 to 1.0
    pub arousal: f32,   // 0.0 to 1.0
    pub stability: f32, // How stable their emotions are
    pub triggers: Vec<EmotionalTrigger>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalTrigger {
    pub trigger: String,
    pub emotion: CoreEmotion,
    pub intensity: f32,
    pub observed_count: u32,
}

/// Knowledge attributed to an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Knowledge {
    pub id: String,
    pub fact: String,
    pub domain: String,
    pub confidence: f32, // How certain we are they know this
    pub source: KnowledgeSource,
    pub relevance: f32, // How relevant to current context
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum KnowledgeSource {
    Demonstrated, // They've shown they know it
    Communicated, // They've said they know it
    Inferred,     // We think they know it
    Assumed,      // We assume they know it
}

/// Personality profile
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalityProfile {
    // Big Five personality traits
    pub openness: f32, // 0.0 to 1.0
    pub conscientiousness: f32,
    pub extraversion: f32,
    pub agreeableness: f32,
    pub neuroticism: f32,

    // Additional traits
    pub risk_tolerance: f32,
    pub curiosity: f32,
    pub empathy: f32,
    pub assertiveness: f32,
    pub humor: f32,
}

impl Default for PersonalityProfile {
    fn default() -> Self {
        Self {
            openness: 0.5,
            conscientiousness: 0.5,
            extraversion: 0.5,
            agreeableness: 0.5,
            neuroticism: 0.5,
            risk_tolerance: 0.5,
            curiosity: 0.5,
            empathy: 0.5,
            assertiveness: 0.5,
            humor: 0.5,
        }
    }
}

/// Relationship status with this agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelationshipStatus {
    pub relationship_type: RelationshipType,
    pub trust_level: f32, // 0.0 to 1.0
    pub familiarity: f32, // How well we know them
    pub affinity: f32,    // How much we like them
    pub interaction_count: u32,
    pub positive_interactions: u32,
    pub negative_interactions: u32,
    #[serde(skip, default = "Instant::now")]
    pub first_interaction: Instant,
    #[serde(skip, default = "Instant::now")]
    pub last_interaction: Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RelationshipType {
    Unknown,
    Acquaintance,
    Colleague,
    Friend,
    Mentor,
    Student,
    Adversary,
    Neutral,
}

#[derive(Debug)]
/// Belief tracker for managing beliefs about others
pub struct BeliefTracker {
    /// Beliefs organized by agent
    beliefs: Arc<RwLock<BTreeMap<AgentId, Vec<Belief>>>>,

    /// Belief revision history
    revisions: Arc<RwLock<Vec<BeliefRevision>>>,

    /// Confidence decay rate
    decay_rate: f32,
}

#[derive(Clone, Debug)]
struct BeliefRevision {
    agent_id: AgentId,
    old_belief: Belief,
    new_belief: Belief,
    reason: String,
    timestamp: Instant,
}

impl BeliefTracker {
    pub fn new(decay_rate: f32) -> Self {
        Self {
            beliefs: Arc::new(RwLock::new(BTreeMap::new())),
            revisions: Arc::new(RwLock::new(Vec::new())),
            decay_rate,
        }
    }

    /// Add or update a belief about an agent
    pub async fn update_belief(&self, agent_id: &AgentId, belief: Belief) -> Result<()> {
        let mut beliefs = self.beliefs.write().await;
        let agent_beliefs = beliefs.entry(agent_id.clone()).or_insert_with(Vec::new);

        // Check if belief already exists
        if let Some(existing) = agent_beliefs.iter_mut().find(|b| b.id == belief.id) {
            // Record revision
            let revision = BeliefRevision {
                agent_id: agent_id.clone(),
                old_belief: existing.clone(),
                new_belief: belief.clone(),
                reason: "Updated based on new evidence".to_string(),
                timestamp: Instant::now(),
            };

            self.revisions.write().await.push(revision);
            *existing = belief;
        } else {
            agent_beliefs.push(belief);
        }

        Ok(())
    }

    /// Apply confidence decay to old beliefs
    pub async fn decay_beliefs(&self) -> Result<()> {
        let mut beliefs = self.beliefs.write().await;

        for (_, agent_beliefs) in beliefs.iter_mut() {
            for belief in agent_beliefs.iter_mut() {
                let age = belief.formed_at.elapsed().as_secs() as f32 / 3600.0; // Hours
                let decay = (-self.decay_rate * age).exp();
                belief.confidence *= decay;
            }
        }

        // Remove beliefs with very low confidence
        for (_, agent_beliefs) in beliefs.iter_mut() {
            agent_beliefs.retain(|b| b.confidence > 0.1);
        }

        Ok(())
    }
}

#[derive(Debug)]
/// Intention predictor for anticipating agent actions
pub struct IntentionPredictor {
    /// Historical intentions and outcomes
    history: Arc<RwLock<HashMap<AgentId, Vec<IntentionRecord>>>>,

    /// Pattern matcher
    patterns: Arc<RwLock<HashMap<String, IntentionPattern>>>,

    /// Neural processor for pattern recognition
    neural_processor: Arc<NeuroProcessor>,
}

#[derive(Clone, Debug)]
struct IntentionRecord {
    intention: Intention,
    predicted_at: Instant,
    outcome: Option<IntentionOutcome>,
}

#[derive(Clone, Debug)]
enum IntentionOutcome {
    Fulfilled,
    Abandoned,
    Modified(String), // New intention
    Unknown,
}

#[derive(Clone, Debug)]
struct IntentionPattern {
    pattern_id: String,
    agent_type: Option<PersonalityProfile>,
    context: String,
    typical_sequence: Vec<String>,
    confidence: f32,
}

impl IntentionPredictor {
    pub fn new(neural_processor: Arc<NeuroProcessor>) -> Self {
        Self {
            history: Arc::new(RwLock::new(HashMap::new())),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            neural_processor,
        }
    }

    /// Predict likely intentions based on mental model
    pub async fn predict_intentions(&self, model: &MentalModel) -> Result<Vec<Intention>> {
        let mut predicted = Vec::new();

        // Based on desires, predict intentions
        for desire in &model.desires {
            if desire.priority > 0.5 {
                let intention = self.desire_to_intention(desire, model).await?;
                predicted.push(intention);
            }
        }

        // Based on personality, predict behavioral intentions
        if model.personality_traits.conscientiousness > 0.7 {
            predicted.push(Intention {
                id: uuid::Uuid::new_v4().to_string(),
                description: "Complete planned tasks".to_string(),
                goal: None,
                commitment_level: model.personality_traits.conscientiousness,
                likelihood: 0.8,
                timeline: IntentionTimeline::ShortTerm,
                preconditions: vec![],
                confidence: 0.7,
            });
        }

        // Sort by likelihood with robust error handling
        predicted.sort_by(|a, b| {
            b.likelihood.partial_cmp(&a.likelihood).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(predicted)
    }

    /// Convert desire to likely intention
    async fn desire_to_intention(&self, desire: &Desire, model: &MentalModel) -> Result<Intention> {
        let commitment = desire.strength * model.personality_traits.conscientiousness;
        let likelihood = commitment * (1.0 - model.personality_traits.neuroticism);

        let timeline = match desire.priority {
            p if p > 0.8 => IntentionTimeline::Immediate,
            p if p > 0.6 => IntentionTimeline::ShortTerm,
            p if p > 0.4 => IntentionTimeline::MediumTerm,
            _ => IntentionTimeline::LongTerm,
        };

        Ok(Intention {
            id: uuid::Uuid::new_v4().to_string(),
            description: format!("Pursue: {}", desire.description),
            goal: None,
            commitment_level: commitment,
            likelihood,
            timeline,
            preconditions: vec![],
            confidence: commitment * 0.8, // Confidence based on commitment
        })
    }
}

/// Configuration for Theory of Mind
#[derive(Clone, Debug)]
pub struct TheoryOfMindConfig {
    /// How often to update mental models
    pub update_interval: Duration,

    /// Belief confidence decay rate
    pub belief_decay_rate: f32,

    /// Maximum number of agents to track
    pub max_agents: usize,

    /// Enable perspective simulation
    pub enable_simulation: bool,

    /// Simulation depth (recursive modeling)
    pub simulation_depth: u32,
}

impl Default for TheoryOfMindConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(60),
            belief_decay_rate: 0.01,
            max_agents: 100,
            enable_simulation: true,
            simulation_depth: 2,
        }
    }
}

#[derive(Debug)]
/// Main Theory of Mind system
pub struct TheoryOfMind {
    /// Mental models of other agents
    mental_models: Arc<RwLock<BTreeMap<AgentId, MentalModel>>>,

    /// Belief tracker
    belief_tracker: Arc<BeliefTracker>,

    /// Intention predictor
    intention_predictor: Arc<IntentionPredictor>,

    /// Neural processor reference
    neural_processor: Arc<NeuroProcessor>,

    /// Emotional core reference
    emotional_core: Arc<EmotionalCore>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Configuration
    config: TheoryOfMindConfig,

    /// Update channel
    update_tx: mpsc::Sender<TheoryOfMindUpdate>,

    /// Statistics
    stats: Arc<RwLock<TheoryOfMindStats>>,
}

#[derive(Clone, Debug)]
pub enum TheoryOfMindUpdate {
    AgentModelUpdated(AgentId),
    BeliefRevised(AgentId, String), // agent, belief_id
    IntentionPredicted(AgentId, Intention),
    RelationshipChanged(AgentId, RelationshipType),
}

#[derive(Debug, Default, Clone)]
pub struct TheoryOfMindStats {
    pub total_agents_modeled: u64,
    pub beliefs_tracked: u64,
    pub intentions_predicted: u64,
    pub successful_predictions: u64,
    pub model_updates: u64,
    pub avg_model_confidence: f32,
}

impl TheoryOfMind {
    pub async fn new(
        neural_processor: Arc<NeuroProcessor>,
        emotional_core: Arc<EmotionalCore>,
        memory: Arc<CognitiveMemory>,
        config: TheoryOfMindConfig,
    ) -> Result<Self> {
        info!("Initializing Theory of Mind system");

        let (update_tx, _) = mpsc::channel(100);

        let belief_tracker = Arc::new(BeliefTracker::new(config.belief_decay_rate));
        let intention_predictor = Arc::new(IntentionPredictor::new(neural_processor.clone()));

        Ok(Self {
            mental_models: Arc::new(RwLock::new(BTreeMap::new())),
            belief_tracker,
            intention_predictor,
            neural_processor,
            emotional_core,
            memory,
            config,
            update_tx,
            stats: Arc::new(RwLock::new(TheoryOfMindStats::default())),
        })
    }

    /// Create a minimal theory of mind for bootstrapping
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
        let config = TheoryOfMindConfig::default();

        Self::new(neural_processor, emotional_core, memory, config).await
    }

    /// Start the Theory of Mind processing loop
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Theory of Mind system");

        // Model update loop
        {
            let tom = self.clone();
            tokio::spawn(async move {
                tom.update_loop().await;
            });
        }

        // Belief decay loop
        {
            let tom = self.clone();
            tokio::spawn(async move {
                tom.belief_decay_loop().await;
            });
        }

        // Store initialization in memory
        self.memory
            .store(
                "Theory of Mind system activated - can now model other agents".to_string(),
                vec![],
                MemoryMetadata {
                    source: "theory_of_mind".to_string(),
                    tags: vec!["milestone".to_string(), "social".to_string()],
                    importance: 0.9,
                    associations: vec![],
                    context: Some("theory of mind initialization".to_string()),
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

    /// Create or update mental model for an agent
    pub async fn model_agent(&self, agent_id: AgentId, observations: Vec<String>) -> Result<()> {
        let mut models = self.mental_models.write().await;

        let model = models.entry(agent_id.clone()).or_insert_with(|| MentalModel {
            agent_id: agent_id.clone(),
            beliefs: Vec::new(),
            desires: Vec::new(),
            intentions: Vec::new(),
            emotional_state: EmotionalStateModel {
                primary_emotion: CoreEmotion::Trust,
                emotional_blend: EmotionalBlend::default(),
                valence: 0.0,
                arousal: 0.5,
                stability: 0.5,
                triggers: Vec::new(),
            },
            knowledge_base: Vec::new(),
            personality_traits: PersonalityProfile::default(),
            relationship_status: RelationshipStatus {
                relationship_type: RelationshipType::Unknown,
                trust_level: 0.5,
                familiarity: 0.0,
                affinity: 0.5,
                interaction_count: 0,
                positive_interactions: 0,
                negative_interactions: 0,
                first_interaction: Instant::now(),
                last_interaction: Instant::now(),
            },
            last_updated: Instant::now(),
            confidence: 0.5,
        });

        // Process observations to update model
        for observation in observations {
            self.process_observation(&agent_id, &observation, model).await?;
        }

        model.last_updated = Instant::now();
        model.relationship_status.interaction_count += 1;
        model.relationship_status.last_interaction = Instant::now();

        // Update stats
        self.stats.write().await.model_updates += 1;

        // Send update
        let _ = self.update_tx.send(TheoryOfMindUpdate::AgentModelUpdated(agent_id.clone())).await;

        Ok(())
    }

    /// Process observation to update mental model
    async fn process_observation(
        &self,
        agent_id: &AgentId,
        observation: &str,
        model: &mut MentalModel,
    ) -> Result<()> {
        // Use neural processor to analyze observation
        let thought = Thought {
            id: ThoughtId::new(),
            content: format!("Observing {}: {}", agent_id, observation),
            thought_type: ThoughtType::Observation,
            ..Default::default()
        };

        let activation = self.neural_processor.process_thought(&thought).await?;

        // High activation suggests important observation
        if activation > 0.6 {
            // Extract beliefs from observation
            if observation.contains("believes") || observation.contains("thinks") {
                let belief = Belief {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: observation.to_string(),
                    confidence: activation,
                    source: BeliefSource::Observation,
                    evidence: vec![observation.to_string()],
                    formed_at: Instant::now(),
                };

                model.beliefs.push(belief.clone());
                self.belief_tracker.update_belief(agent_id, belief).await?;
            }

            // Extract desires from observation
            if observation.contains("wants") || observation.contains("desires") {
                let desire = Desire {
                    id: uuid::Uuid::new_v4().to_string(),
                    description: observation.to_string(),
                    priority: activation,
                    desire_type: DesireType::Achievement,
                    target: None,
                    strength: activation,
                };

                model.desires.push(desire);
            }

            // Update personality traits based on behavior
            if observation.contains("helpful") || observation.contains("kind") {
                model.personality_traits.agreeableness =
                    (model.personality_traits.agreeableness + 0.1).min(1.0);
            }
        }

        Ok(())
    }

    /// Get mental model for an agent
    pub async fn get_model(&self, agent_id: &AgentId) -> Option<MentalModel> {
        self.mental_models.read().await.get(agent_id).cloned()
    }

    /// Predict what an agent might do next
    pub async fn predict_behavior(&self, agent_id: &AgentId) -> Result<Vec<Intention>> {
        let models = self.mental_models.read().await;
        let model =
            models.get(agent_id).ok_or_else(|| anyhow!("No model for agent {}", agent_id))?;

        let intentions = self.intention_predictor.predict_intentions(model).await?;

        // Update stats
        self.stats.write().await.intentions_predicted += intentions.len() as u64;

        // Send updates
        for intention in &intentions {
            let _ = self
                .update_tx
                .send(TheoryOfMindUpdate::IntentionPredicted(agent_id.clone(), intention.clone()))
                .await;
        }

        Ok(intentions)
    }

    /// Simulate agent's perspective (recursive ToM)
    pub async fn simulate_perspective(
        &self,
        agent_id: &AgentId,
        situation: &str,
    ) -> Result<SimulatedPerspective> {
        if !self.config.enable_simulation {
            return Err(anyhow!("Perspective simulation disabled"));
        }

        let models = self.mental_models.read().await;
        let model =
            models.get(agent_id).ok_or_else(|| anyhow!("No model for agent {}", agent_id))?;

        // Simulate how they would perceive the situation
        let mut perspective = SimulatedPerspective {
            agent_id: agent_id.clone(),
            perceived_situation: situation.to_string(),
            likely_thoughts: Vec::new(),
            likely_emotions: model.emotional_state.clone(),
            likely_actions: Vec::new(),
            confidence: model.confidence,
        };

        // Generate likely thoughts based on beliefs and personality
        for belief in &model.beliefs {
            if belief.confidence > 0.5 {
                perspective
                    .likely_thoughts
                    .push(format!("Based on belief '{}', they might think...", belief.content));
            }
        }

        // Predict emotional response
        if situation.contains("threat") && model.personality_traits.neuroticism > 0.6 {
            perspective.likely_emotions.primary_emotion = CoreEmotion::Fear;
            perspective.likely_emotions.arousal = 0.8;
        }

        // Predict likely actions
        let intentions = self.intention_predictor.predict_intentions(model).await?;
        perspective.likely_actions = intentions.into_iter().map(|i| i.description).collect();

        Ok(perspective)
    }

    /// Update relationship status
    pub async fn update_relationship(
        &self,
        agent_id: &AgentId,
        interaction_quality: f32, // -1.0 to 1.0
    ) -> Result<()> {
        let mut models = self.mental_models.write().await;

        if let Some(model) = models.get_mut(agent_id) {
            let relationship = &mut model.relationship_status;

            // Update interaction counts
            if interaction_quality > 0.0 {
                relationship.positive_interactions += 1;
            } else if interaction_quality < 0.0 {
                relationship.negative_interactions += 1;
            }

            // Update trust and affinity
            relationship.trust_level =
                (relationship.trust_level + interaction_quality * 0.1).max(0.0).min(1.0);
            relationship.affinity =
                (relationship.affinity + interaction_quality * 0.05).max(-1.0).min(1.0);

            // Update familiarity
            relationship.familiarity = (relationship.familiarity + 0.05).min(1.0);

            // Determine relationship type based on metrics
            relationship.relationship_type = self.classify_relationship(relationship);

            // Send update
            let _ = self
                .update_tx
                .send(TheoryOfMindUpdate::RelationshipChanged(
                    agent_id.clone(),
                    relationship.relationship_type.clone(),
                ))
                .await;
        }

        Ok(())
    }

    /// Classify relationship based on metrics
    fn classify_relationship(&self, status: &RelationshipStatus) -> RelationshipType {
        match (status.trust_level, status.affinity, status.familiarity) {
            (t, a, f) if t > 0.8 && a > 0.8 && f > 0.7 => RelationshipType::Friend,
            (t, a, f) if t > 0.6 && a > 0.5 && f > 0.5 => RelationshipType::Colleague,
            (t, a, _) if t < 0.3 && a < 0.3 => RelationshipType::Adversary,
            (_, _, f) if f < 0.2 => RelationshipType::Unknown,
            _ => RelationshipType::Acquaintance,
        }
    }

    /// Model update loop
    async fn update_loop(&self) {
        let mut interval = interval(self.config.update_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.update_models().await {
                warn!("Model update error: {}", e);
            }
        }
    }

    /// Update all mental models
    async fn update_models(&self) -> Result<()> {
        let mut models = self.mental_models.write().await;
        let mut total_confidence = 0.0;
        let count = models.len() as f32;

        // Update confidence based on age
        for (_, model) in models.iter_mut() {
            let age = model.last_updated.elapsed().as_secs() as f32 / 3600.0; // Hours
            let decay = (-0.1 * age).exp(); // Exponential decay
            model.confidence *= decay;
            total_confidence += model.confidence;
        }

        // Update stats
        if count > 0.0 {
            self.stats.write().await.avg_model_confidence = total_confidence / count;
        }

        Ok(())
    }

    /// Belief decay loop
    async fn belief_decay_loop(&self) {
        let mut interval = interval(Duration::from_secs(3600)); // Every hour

        loop {
            interval.tick().await;

            if let Err(e) = self.belief_tracker.decay_beliefs().await {
                warn!("Belief decay error: {}", e);
            }
        }
    }

    /// Get statistics
    pub async fn get_stats(&self) -> TheoryOfMindStats {
        let models = self.mental_models.read().await;
        let mut stats = self.stats.read().await.clone();

        stats.total_agents_modeled = models.len() as u64;
        stats.beliefs_tracked = models.values().map(|m| m.beliefs.len() as u64).sum();

        stats
    }

    /// Get mental model for a specific agent
    pub async fn get_mental_model(&self, agent_id: &AgentId) -> Option<MentalModel> {
        self.mental_models.read().await.get(agent_id).cloned()
    }

    /// Predict intention based on mental model and current context
    pub async fn predict_intention(
        &self,
        _agent_id: &AgentId,
        model: &MentalModel,
        context: &str,
    ) -> Result<Intention> {
        // Use the model's desires and personality to predict most likely intention
        let mut best_intention = Intention {
            id: uuid::Uuid::new_v4().to_string(),
            description: format!("Likely action in context: {}", context),
            goal: None,
            commitment_level: 0.5,
            likelihood: 0.5,
            timeline: IntentionTimeline::ShortTerm,
            preconditions: vec![],
            confidence: 0.5, // Add confidence field
        };

        // If they have strong desires, predict intention based on those
        if let Some(strongest_desire) = model
            .desires
            .iter()
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
        {
            best_intention.description = format!("Act on desire: {}", strongest_desire.description);
            best_intention.commitment_level = strongest_desire.priority;
            best_intention.likelihood =
                strongest_desire.priority * model.personality_traits.conscientiousness;
            best_intention.confidence = model.confidence * 0.8;
        }

        // Consider personality traits
        if model.personality_traits.extraversion > 0.7 && context.contains("social") {
            best_intention.description = "Engage socially with others".to_string();
            best_intention.likelihood *= 1.2;
            best_intention.confidence *= 1.1;
        }

        // Normalize values
        best_intention.likelihood = best_intention.likelihood.clamp(0.0, 1.0);
        best_intention.confidence = best_intention.confidence.clamp(0.0, 1.0);

        Ok(best_intention)
    }

    /// Update a belief about an agent
    pub async fn update_belief(&self, agent_id: &AgentId, belief: Belief) -> Result<()> {
        // Update through belief tracker
        self.belief_tracker.update_belief(agent_id, belief.clone()).await?;

        // Also update in the mental model
        let mut models = self.mental_models.write().await;
        if let Some(model) = models.get_mut(agent_id) {
            // Check if belief already exists
            if let Some(existing) = model.beliefs.iter_mut().find(|b| b.id == belief.id) {
                *existing = belief.clone();
            } else {
                model.beliefs.push(belief.clone());
            }
            model.last_updated = Instant::now();

            // Update stats
            self.stats.write().await.beliefs_tracked += 1;

            // Send update notification
            let _ = self
                .update_tx
                .send(TheoryOfMindUpdate::BeliefRevised(agent_id.clone(), belief.id))
                .await;
        }

        Ok(())
    }
    
    /// Get list of known agents
    pub async fn get_known_agents(&self) -> Vec<AgentId> {
        let models = self.mental_models.read().await;
        models.keys().cloned().collect()
    }
}

/// Simulated perspective of another agent
#[derive(Clone, Debug)]
pub struct SimulatedPerspective {
    pub agent_id: AgentId,
    pub perceived_situation: String,
    pub likely_thoughts: Vec<String>,
    pub likely_emotions: EmotionalStateModel,
    pub likely_actions: Vec<String>,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{SimdCacheConfig, SimdSmartCache};

    #[tokio::test]
    async fn test_agent_modeling() {
        let cache = Arc::new(SimdSmartCache::new(SimdCacheConfig::default()));
        let neural = Arc::new(NeuroProcessor::new(cache).await.unwrap());
        let memory = Arc::new(CognitiveMemory::new_for_test().await.unwrap());
        let emotional =
            Arc::new(EmotionalCore::new(memory.clone(), Default::default()).await.unwrap());

        let tom = Arc::new(
            TheoryOfMind::new(neural, emotional, memory, TheoryOfMindConfig::default())
                .await
                .unwrap(),
        );

        // Model an agent
        let agent_id = AgentId::new("TestAgent");
        tom.model_agent(
            agent_id.clone(),
            vec![
                "TestAgent believes the weather is nice".to_string(),
                "TestAgent wants to go for a walk".to_string(),
            ],
        )
        .await
        .unwrap();

        // Get model
        let model = tom.get_model(&agent_id).await.unwrap();
        assert!(!model.beliefs.is_empty());
        assert!(!model.desires.is_empty());
    }
}
