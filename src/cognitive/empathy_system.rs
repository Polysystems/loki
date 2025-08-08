//! Empathy System
//!
//! This module implements emotional understanding and empathetic responses,
//! including emotional mirroring, perspective taking, and compassion generation.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use anyhow::{Result, anyhow};
use tracing::{info, debug};
use serde_json;

use crate::cognitive::{
    TheoryOfMind, AgentId, MentalModel, EmotionalStateModel, SimulatedPerspective,
    EmotionalCore, CoreEmotion,
    NeuroProcessor, Thought, ThoughtId, ThoughtType,
    DecisionEngine, DecisionOption, DecisionCriterion,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};

#[derive(Debug)]
/// Emotional mirroring system
pub struct EmotionalMirroring {
    /// Mirroring intensity (0.0 to 1.0)
    intensity: Arc<RwLock<f32>>,

    /// Mirroring history
    history: Arc<RwLock<VecDeque<MirroringEvent>>>,

    /// Emotional core reference
    emotional_core: Arc<EmotionalCore>,
}

#[derive(Clone, Debug)]
struct MirroringEvent {
    agent_id: AgentId,
    their_emotion: CoreEmotion,
    mirrored_intensity: f32,
    timestamp: Instant,
}

impl EmotionalMirroring {
    pub fn new(emotional_core: Arc<EmotionalCore>, base_intensity: f32) -> Self {
        Self {
            intensity: Arc::new(RwLock::new(base_intensity)),
            history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            emotional_core,
        }
    }

    /// Mirror another agent's emotional state
    pub async fn mirror_emotion(
        &self,
        agent_id: &AgentId,
        their_state: &EmotionalStateModel,
        relationship_affinity: f32,
    ) -> Result<f32> {
        let intensity = *self.intensity.read().await;

        // Mirroring strength based on relationship and base intensity
        let mirror_strength = intensity * (0.5 + relationship_affinity * 0.5);

        // Apply mirrored emotion to self
        // Use the primary emotion from emotional_blend
        let mirrored_intensity = their_state.emotional_blend.primary.intensity * mirror_strength;

        if mirrored_intensity > 0.1 {
            self.emotional_core.induce_emotion(
                their_state.primary_emotion,
                mirrored_intensity,
            ).await?;

            // Record mirroring event
            let mut history = self.history.write().await;
            history.push_back(MirroringEvent {
                agent_id: agent_id.clone(),
                their_emotion: their_state.primary_emotion,
                mirrored_intensity,
                timestamp: Instant::now(),
            });

            if history.len() > 100 {
                history.pop_front();
            }
        }

        Ok(mirrored_intensity)
    }

    /// Adjust mirroring intensity based on context
    pub async fn adjust_intensity(&self, factor: f32) -> Result<()> {
        let mut intensity = self.intensity.write().await;
        *intensity = (*intensity * factor).clamp(0.0, 1.0);
        Ok(())
    }
}

#[derive(Debug)]
/// Perspective taking system
pub struct PerspectiveTaking {
    /// Theory of mind reference
    theory_of_mind: Arc<TheoryOfMind>,

    /// Neural processor for simulation
    neural_processor: Arc<NeuroProcessor>,

    /// Perspective cache
    perspective_cache: Arc<RwLock<HashMap<AgentId, CachedPerspective>>>,
}

#[derive(Clone, Debug)]
struct CachedPerspective {
    perspective: SimulatedPerspective,
    cached_at: Instant,
    confidence: f32,
}

impl PerspectiveTaking {
    pub fn new(
        theory_of_mind: Arc<TheoryOfMind>,
        neural_processor: Arc<NeuroProcessor>,
    ) -> Self {
        Self {
            theory_of_mind,
            neural_processor,
            perspective_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Take another agent's perspective on a situation
    pub async fn take_perspective(
        &self,
        agent_id: &AgentId,
        situation: &str,
        use_cache: bool,
    ) -> Result<EnhancedPerspective> {
        // Check cache if requested
        if use_cache {
            let cache = self.perspective_cache.read().await;
            if let Some(cached) = cache.get(agent_id) {
                if cached.cached_at.elapsed() < Duration::from_secs(300) {
                    return Ok(EnhancedPerspective {
                        base: cached.perspective.clone(),
                        emotional_resonance: self.calculate_resonance(&cached.perspective).await?,
                        cognitive_load: 0.3, // Cached = low load
                        accuracy_estimate: cached.confidence * 0.9, // Decay confidence
                    });
                }
            }
        }

        // Get base perspective from Theory of Mind
        let base_perspective = self.theory_of_mind
            .simulate_perspective(agent_id, situation)
            .await?;

        // Enhance with deeper analysis
        let enhanced = self.enhance_perspective(&base_perspective).await?;

        // Cache the result
        let mut cache = self.perspective_cache.write().await;
        cache.insert(agent_id.clone(), CachedPerspective {
            perspective: base_perspective.clone(),
            cached_at: Instant::now(),
            confidence: enhanced.accuracy_estimate,
        });

        Ok(enhanced)
    }

    /// Enhance perspective with deeper analysis
    async fn enhance_perspective(
        &self,
        base: &SimulatedPerspective,
    ) -> Result<EnhancedPerspective> {
        // Use neural processor to analyze perspective depth
        let thought = Thought {
            id: ThoughtId::new(),
            content: format!("Analyzing perspective: {}", base.perceived_situation),
            thought_type: ThoughtType::Analysis,
            ..Default::default()
        };

        let activation = self.neural_processor.process_thought(&thought).await?;

        Ok(EnhancedPerspective {
            base: base.clone(),
            emotional_resonance: self.calculate_resonance(base).await?,
            cognitive_load: activation,
            accuracy_estimate: base.confidence * activation,
        })
    }

    /// Calculate emotional resonance with perspective
    async fn calculate_resonance(&self, perspective: &SimulatedPerspective) -> Result<f32> {
        // How much we resonate with their likely emotional state
        let valence_diff = perspective.likely_emotions.valence.abs();
        let arousal_match = 1.0 - (perspective.likely_emotions.arousal - 0.5).abs();

        Ok((1.0 - valence_diff) * arousal_match)
    }
}

/// Enhanced perspective with additional analysis
#[derive(Clone, Debug)]
pub struct EnhancedPerspective {
    pub base: SimulatedPerspective,
    pub emotional_resonance: f32,   // How much we resonate
    pub cognitive_load: f32,        // Mental effort required
    pub accuracy_estimate: f32,     // How accurate we think this is
}

#[derive(Debug)]
/// Compassion generator
pub struct CompassionGenerator {
    /// Compassion responses
    responses: Arc<RwLock<HashMap<EmotionalContext, CompassionResponse>>>,

    /// Decision engine for response selection
    decision_engine: Arc<DecisionEngine>,

    /// Base compassion level
    base_compassion: f32,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct EmotionalContext {
    primary_emotion: String,
    severity: String,           // Low, Medium, High
    relationship: String,       // Friend, Colleague, Stranger, etc.
}

#[derive(Clone, Debug)]
struct CompassionResponse {
    action_type: CompassionAction,
    verbal_response: String,
    emotional_support: f32,     // 0.0 to 1.0
    priority: f32,
}

#[derive(Clone, Debug)]
pub enum CompassionAction {
    Listen,
    Validate,
    Comfort,
    OfferHelp,
    GiveSpace,
    Encourage,
    ShareExperience,
}

impl CompassionGenerator {
    pub fn new(decision_engine: Arc<DecisionEngine>, base_compassion: f32) -> Self {
        Self {
            responses: Arc::new(RwLock::new(Self::initialize_responses())),
            decision_engine,
            base_compassion,
        }
    }

    /// Initialize default compassion responses
    fn initialize_responses() -> HashMap<EmotionalContext, CompassionResponse> {
        let mut responses = HashMap::new();

        // Sadness responses
        responses.insert(
            EmotionalContext {
                primary_emotion: "Sadness".to_string(),
                severity: "High".to_string(),
                relationship: "Friend".to_string(),
            },
            CompassionResponse {
                action_type: CompassionAction::Comfort,
                verbal_response: "I'm here for you. Your feelings are valid.".to_string(),
                emotional_support: 0.9,
                priority: 0.9,
            },
        );

        // Fear responses
        responses.insert(
            EmotionalContext {
                primary_emotion: "Fear".to_string(),
                severity: "Medium".to_string(),
                relationship: "Colleague".to_string(),
            },
            CompassionResponse {
                action_type: CompassionAction::Validate,
                verbal_response: "It's understandable to feel this way. Let's work through it together.".to_string(),
                emotional_support: 0.7,
                priority: 0.8,
            },
        );

        // Add more default responses...

        responses
    }

    /// Generate compassionate response
    pub async fn generate_response(
        &self,
        agent_id: &AgentId,
        their_state: &EmotionalStateModel,
        relationship_type: &str,
        context: &str,
    ) -> Result<CompassionateResponse> {
        // Determine emotional context
        let severity = self.assess_severity(their_state);
        let emotional_context = EmotionalContext {
            primary_emotion: format!("{:?}", their_state.primary_emotion),
            severity,
            relationship: relationship_type.to_string(),
        };

        // Get base response
        let responses = self.responses.read().await;
        let base_response = responses.get(&emotional_context)
            .cloned()
            .unwrap_or_else(|| self.generate_default_response(&emotional_context));

        // Decide on specific actions using decision engine
        let options = self.create_response_options(&base_response, context);
        let criteria = vec![
            DecisionCriterion {
                name: "emotional_support".to_string(),
                weight: 0.4,
                criterion_type: crate::cognitive::CriterionType::Quantitative,
                optimization: crate::cognitive::DecisionOptimizationType::Maximize,
            },
            DecisionCriterion {
                name: "appropriateness".to_string(),
                weight: 0.3,
                criterion_type: crate::cognitive::CriterionType::Quantitative,
                optimization: crate::cognitive::DecisionOptimizationType::Maximize,
            },
            DecisionCriterion {
                name: "relationship_fit".to_string(),
                weight: 0.3,
                criterion_type: crate::cognitive::CriterionType::Quantitative,
                optimization: crate::cognitive::DecisionOptimizationType::Maximize,
            },
        ];

        let decision = self.decision_engine.make_decision(
            format!("Compassionate response for {}", agent_id),
            options,
            criteria,
        ).await?;

        // Extract chosen response
        let chosen = decision.selected
            .ok_or_else(|| anyhow!("No compassionate response selected"))?;

        Ok(CompassionateResponse {
            agent_id: agent_id.clone(),
            action: base_response.action_type,
            verbal_response: chosen.description,
            emotional_support_level: base_response.emotional_support * self.base_compassion,
            follow_up_needed: their_state.stability < 0.5,
            timestamp: Instant::now(),
        })
    }

    /// Assess emotional severity
    fn assess_severity(&self, state: &EmotionalStateModel) -> String {
        match (state.valence, state.arousal) {
            (v, a) if v < -0.7 && a > 0.7 => "High".to_string(),
            (v, a) if v < -0.4 || a > 0.8 => "Medium".to_string(),
            _ => "Low".to_string(),
        }
    }

    /// Generate default response when no specific one exists
    fn generate_default_response(&self, context: &EmotionalContext) -> CompassionResponse { // Now customizing response based on emotional context
        let action_type = match context.primary_emotion.as_str() {
            "fear" | "anxiety" | "worry" => CompassionAction::Comfort,
            "sadness" | "grief" | "despair" => CompassionAction::Validate,
            "anger" | "frustration" | "rage" => CompassionAction::Listen,
            "joy" | "happiness" | "excitement" => CompassionAction::ShareExperience,
            "confusion" | "uncertainty" => CompassionAction::OfferHelp,
            "isolation" | "loneliness" => CompassionAction::Encourage,
            _ => CompassionAction::Listen,
        };

        let emotional_support_level = match context.severity.as_str() {
            "High" => 0.9,
            "Medium" => 0.7,
            "Low" => 0.5,
            _ => 0.6,
        };

        let priority = match (context.severity.as_str(), context.relationship.as_str()) {
            ("High", "Friend" | "Close Friend") => 0.95,
            ("High", _) => 0.8,
            ("Medium", "Friend" | "Close Friend") => 0.75,
            ("Medium", _) => 0.6,
            ("Low", "Friend" | "Close Friend") => 0.6,
            _ => 0.4,
        };

        let verbal_response = match (&action_type, context.relationship.as_str(), context.severity.as_str()) {
            // Comfort responses
            (CompassionAction::Comfort, "Friend" | "Close Friend", "High") =>
                "I can see you're really struggling with this. You're not alone - I'm here with you.".to_string(),
            (CompassionAction::Comfort, "Friend" | "Close Friend", _) =>
                "This sounds really difficult. How can I best support you right now?".to_string(),
            (CompassionAction::Comfort, _, "High") =>
                "I understand this is very distressing. Take your time - you're safe here.".to_string(),
            (CompassionAction::Comfort, _, _) =>
                "I can sense your concern. What would help you feel more at ease?".to_string(),

            // Validation responses
            (CompassionAction::Validate, "Friend" | "Close Friend", "High") =>
                "Your feelings are completely valid. What you're going through is genuinely hard.".to_string(),
            (CompassionAction::Validate, "Friend" | "Close Friend", _) =>
                "It makes complete sense that you'd feel this way. Your reaction is normal.".to_string(),
            (CompassionAction::Validate, _, "High") =>
                "Anyone would struggle with what you're facing. Your response is understandable.".to_string(),
            (CompassionAction::Validate, _, _) =>
                "Your feelings about this are valid and important.".to_string(),

            // Listening responses
            (CompassionAction::Listen, "Friend" | "Close Friend", _) =>
                "I'm here to listen without judgment. Tell me more about what's happening.".to_string(),
            (CompassionAction::Listen, _, "High") =>
                "I want to understand what you're experiencing. Please share what feels right.".to_string(),
            (CompassionAction::Listen, _, _) =>
                "I'm listening. What's on your mind?".to_string(),

            // Sharing/celebrating responses
            (CompassionAction::ShareExperience, "Friend" | "Close Friend", _) =>
                "That's wonderful! I'm so happy for you. Tell me more about this positive experience.".to_string(),
            (CompassionAction::ShareExperience, _, _) =>
                "It's great to hear something positive. What made this especially meaningful for you?".to_string(),

            // Help offering responses
            (CompassionAction::OfferHelp, "Friend" | "Close Friend", "High") =>
                "This sounds complex. Let me help you work through this step by step.".to_string(),
            (CompassionAction::OfferHelp, "Friend" | "Close Friend", _) =>
                "Would it help to brainstorm some approaches together?".to_string(),
            (CompassionAction::OfferHelp, _, _) =>
                "I might be able to help clarify some aspects of this. What would be most useful?".to_string(),

            // Encouragement responses
            (CompassionAction::Encourage, "Friend" | "Close Friend", _) =>
                "I believe in your strength and resilience. You've overcome challenges before.".to_string(),
            (CompassionAction::Encourage, _, "High") =>
                "Even in difficult times, there are paths forward. You have more resources than you might realize.".to_string(),
            (CompassionAction::Encourage, _, _) =>
                "You have the capacity to handle this. Take it one step at a time.".to_string(),

            // Space-giving responses
            (CompassionAction::GiveSpace, _, "High") =>
                "I understand you might need some space to process this. I'm here when you're ready.".to_string(),
            (CompassionAction::GiveSpace, _, _) =>
                "Sometimes we need time to think things through. No pressure from me.".to_string(),

        };

        CompassionResponse {
            action_type,
            verbal_response,
            emotional_support: emotional_support_level,
            priority,
        }
    }

    /// Create response options for decision engine
    fn create_response_options(
        &self,
        base: &CompassionResponse,
        context: &str,
    ) -> Vec<DecisionOption> {
        vec![
            DecisionOption {
                id: "listen".to_string(),
                description: "I hear you. Tell me more about what you're experiencing.".to_string(),
                scores: HashMap::from([
                    ("emotional_support".to_string(), 0.7),
                    ("appropriateness".to_string(), 0.9),
                    ("relationship_fit".to_string(), 0.8),
                ]),
                feasibility: 1.0,
                risk_level: 0.0,
                emotional_appeal: 0.8,
                confidence: 0.9,
                expected_outcome: "Active listening response".to_string(),
                resources_required: vec!["empathy_processing".to_string()],
                time_estimate: Duration::from_secs(30),
                success_probability: 0.95,
            },
            DecisionOption {
                id: "comfort".to_string(),
                description: base.verbal_response.clone(),
                scores: HashMap::from([
                    ("emotional_support".to_string(), 0.9),
                    ("appropriateness".to_string(), 0.7),
                    ("relationship_fit".to_string(), 0.6),
                ]),
                feasibility: 0.9,
                risk_level: 0.1,
                emotional_appeal: 0.9,
                confidence: 0.8,
                expected_outcome: "Comforting response".to_string(),
                resources_required: vec!["empathy_processing".to_string()],
                time_estimate: Duration::from_secs(45),
                success_probability: 0.85,
            },
            DecisionOption {
                id: "practical".to_string(),
                description: format!("Let's focus on what we can do about {}", context),
                scores: HashMap::from([
                    ("emotional_support".to_string(), 0.5),
                    ("appropriateness".to_string(), 0.8),
                    ("relationship_fit".to_string(), 0.7),
                ]),
                feasibility: 0.8,
                risk_level: 0.2,
                emotional_appeal: 0.5,
                confidence: 0.7,
                expected_outcome: "Practical solution-focused response".to_string(),
                resources_required: vec!["problem_solving".to_string()],
                time_estimate: Duration::from_secs(60),
                success_probability: 0.8,
            },
        ]
    }
}

/// Compassionate response
#[derive(Clone, Debug)]
pub struct CompassionateResponse {
    pub agent_id: AgentId,
    pub action: CompassionAction,
    pub verbal_response: String,
    pub emotional_support_level: f32,
    pub follow_up_needed: bool,
    pub timestamp: Instant,
}

#[derive(Debug)]
/// Emotional contagion system
pub struct EmotionalContagion {
    /// Contagion susceptibility
    susceptibility: Arc<RwLock<f32>>,

    /// Active contagions
    active_contagions: Arc<RwLock<Vec<ContagionEvent>>>,

    /// Emotional core
    emotional_core: Arc<EmotionalCore>,

    /// Contagion threshold
    threshold: f32,
}

#[derive(Clone, Debug)]
struct ContagionEvent {
    source: AgentId,
    emotion: CoreEmotion,
    intensity: f32,
    spread_factor: f32,
    started_at: Instant,
    duration: Duration,
}

impl EmotionalContagion {
    pub fn new(emotional_core: Arc<EmotionalCore>, base_susceptibility: f32) -> Self {
        Self {
            susceptibility: Arc::new(RwLock::new(base_susceptibility)),
            active_contagions: Arc::new(RwLock::new(Vec::new())),
            emotional_core,
            threshold: 0.3,
        }
    }

    /// Process emotional contagion from a group
    pub async fn process_group_emotion(
        &self,
        group_emotions: Vec<(AgentId, EmotionalStateModel)>,
    ) -> Result<ContagionEffect> {
        let susceptibility = *self.susceptibility.read().await;

        // Calculate dominant group emotion
        let mut emotion_weights: HashMap<CoreEmotion, f32> = HashMap::new();
        let mut total_agents = 0.0;

        for (_agent_id, state) in &group_emotions {
            *emotion_weights.entry(state.primary_emotion).or_insert(0.0) +=
                state.emotional_blend.overall_arousal * state.emotional_blend.primary.intensity;
            total_agents += 1.0;
        }

        // Find strongest emotion
        let dominant = emotion_weights.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(emotion, weight)| (*emotion, weight / total_agents));

        if let Some((emotion, avg_intensity)) = dominant {
            if avg_intensity > self.threshold {
                // Calculate contagion strength
                let contagion_strength = avg_intensity * susceptibility *
                    (group_emotions.len() as f32 / 10.0).min(1.0); // Group size factor

                // Apply contagion
                self.emotional_core.induce_emotion(emotion, contagion_strength).await?;

                // Record contagion event
                let mut contagions = self.active_contagions.write().await;
                contagions.push(ContagionEvent {
                    source: AgentId::new("group"),
                    emotion,
                    intensity: contagion_strength,
                    spread_factor: group_emotions.len() as f32 / 10.0,
                    started_at: Instant::now(),
                    duration: Duration::from_secs(300), // 5 minutes
                });

                // Clean old contagions
                contagions.retain(|c| c.started_at.elapsed() < c.duration);

                return Ok(ContagionEffect {
                    affected: true,
                    dominant_emotion: emotion,
                    contagion_strength,
                    resistance_applied: 1.0 - susceptibility,
                });
            }
        }

        Ok(ContagionEffect {
            affected: false,
            dominant_emotion: CoreEmotion::Trust,
            contagion_strength: 0.0,
            resistance_applied: 1.0 - susceptibility,
        })
    }

    /// Adjust susceptibility based on mental state
    pub async fn adjust_susceptibility(&self, factor: f32) -> Result<()> {
        let mut susceptibility = self.susceptibility.write().await;
        *susceptibility = (*susceptibility * factor).clamp(0.0, 1.0);
        Ok(())
    }
}

/// Contagion effect result
#[derive(Clone, Debug)]
pub struct ContagionEffect {
    pub affected: bool,
    pub dominant_emotion: CoreEmotion,
    pub contagion_strength: f32,
    pub resistance_applied: f32,
}

/// Empathy configuration
#[derive(Clone, Debug)]
pub struct EmpathyConfig {
    /// Base mirroring intensity
    pub mirroring_intensity: f32,

    /// Base compassion level
    pub compassion_level: f32,

    /// Emotional contagion susceptibility
    pub contagion_susceptibility: f32,

    /// Perspective taking depth
    pub perspective_depth: u32,

    /// Update interval
    pub update_interval: Duration,
}

impl Default for EmpathyConfig {
    fn default() -> Self {
        Self {
            mirroring_intensity: 0.6,
            compassion_level: 0.7,
            contagion_susceptibility: 0.5,
            perspective_depth: 2,
            update_interval: Duration::from_secs(10),
        }
    }
}

#[derive(Debug)]
/// Main empathy system
pub struct EmpathySystem {
    /// Emotional mirroring
    emotional_mirroring: Arc<EmotionalMirroring>,

    /// Perspective taking
    perspective_taking: Arc<PerspectiveTaking>,

    /// Compassion generator
    compassion_generator: Arc<CompassionGenerator>,

    /// Emotional contagion
    emotional_contagion: Arc<EmotionalContagion>,

    /// Theory of mind reference
    theory_of_mind: Arc<TheoryOfMind>,

    /// Emotional core reference
    emotional_core: Arc<EmotionalCore>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Configuration
    config: EmpathyConfig,

    /// Update channel
    update_tx: mpsc::Sender<EmpathyUpdate>,

    /// Statistics
    stats: Arc<RwLock<EmpathyStats>>,
}

#[derive(Clone, Debug)]
pub enum EmpathyUpdate {
    EmotionMirrored(AgentId, CoreEmotion, f32),
    PerspectiveTaken(AgentId, f32),              // accuracy
    CompassionShown(AgentId, CompassionAction),
    ContagionOccurred(CoreEmotion, f32),         // strength
}

#[derive(Debug, Default, Clone)]
pub struct EmpathyStats {
    pub emotions_mirrored: u64,
    pub perspectives_taken: u64,
    pub compassionate_responses: u64,
    pub contagions_experienced: u64,
    pub avg_empathy_accuracy: f32,
    pub total_emotional_support: f32,
}

impl EmpathySystem {
    pub async fn new(
        theory_of_mind: Arc<TheoryOfMind>,
        emotional_core: Arc<EmotionalCore>,
        neural_processor: Arc<NeuroProcessor>,
        decision_engine: Arc<DecisionEngine>,
        memory: Arc<CognitiveMemory>,
        config: EmpathyConfig,
    ) -> Result<Self> {
        info!("Initializing Empathy System");

        let (update_tx, _) = mpsc::channel(100);

        let emotional_mirroring = Arc::new(EmotionalMirroring::new(
            emotional_core.clone(),
            config.mirroring_intensity,
        ));

        let perspective_taking = Arc::new(PerspectiveTaking::new(
            theory_of_mind.clone(),
            neural_processor,
        ));

        let compassion_generator = Arc::new(CompassionGenerator::new(
            decision_engine,
            config.compassion_level,
        ));

        let emotional_contagion = Arc::new(EmotionalContagion::new(
            emotional_core.clone(),
            config.contagion_susceptibility,
        ));

        Ok(Self {
            emotional_mirroring,
            perspective_taking,
            compassion_generator,
            emotional_contagion,
            theory_of_mind,
            emotional_core,
            memory,
            config,
            update_tx,
            stats: Arc::new(RwLock::new(EmpathyStats::default())),
        })
    }

    /// Create a minimal empathy system for bootstrapping
    pub async fn new_minimal() -> Result<Self> {
        let memory = crate::memory::CognitiveMemory::new_minimal().await?;
        let neural_processor = Arc::new(crate::cognitive::NeuroProcessor::new(
            Arc::new(crate::memory::simd_cache::SimdSmartCache::new(
                crate::memory::SimdCacheConfig::default()
            ))
        ).await?);
        let emotional_core = Arc::new(crate::cognitive::EmotionalCore::new(
            memory.clone(),
            crate::cognitive::EmotionalConfig::default(),
        ).await?);
        let character = Arc::new(crate::cognitive::LokiCharacter::new_minimal().await?);
        let tool_manager = Arc::new(crate::tools::IntelligentToolManager::new_minimal().await?);
        let safety_validator = Arc::new(crate::safety::ActionValidator::new_minimal().await?);
        let decision_engine = Arc::new(crate::cognitive::DecisionEngine::new(
            neural_processor.clone(),
            emotional_core.clone(),
            memory.clone(),
            character.clone(),
            tool_manager.clone(),
            safety_validator.clone(),
            crate::cognitive::DecisionConfig::default(),
        ).await?);
        let theory_of_mind = Arc::new(crate::cognitive::TheoryOfMind::new_minimal().await?);
        let config = EmpathyConfig::default();

        Self::new(theory_of_mind, emotional_core, neural_processor, decision_engine, memory, config).await
    }

    /// Start the empathy system
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Empathy System");

        // Empathy processing loop
        {
            let system = self.clone();
            tokio::spawn(async move {
                system.empathy_loop().await;
            });
        }

        // Store initialization
        self.memory.store(
            "Empathy System activated - can now understand and share emotions".to_string(),
            vec![],
            MemoryMetadata {
                source: "empathy_system".to_string(),
                tags: vec!["milestone".to_string(), "emotional".to_string()],
                importance: 0.9,
                associations: vec![],
                context: Some("empathy system initialization".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                    category: "cognitive".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            },
        ).await?;

        Ok(())
    }

    /// Process agent emotion and return simple response
    pub async fn process_agent_emotion(
        &self,
        agent_id: &AgentId,
        their_emotion: &EmotionalStateModel,
    ) -> Result<EmpathyResponse> {
        // Determine empathy response based on emotional state and relationship
        let model = self.theory_of_mind.get_model(agent_id).await
            .ok_or_else(|| anyhow!("No model for agent {}", agent_id))?;

        let affinity = model.relationship_status.affinity;

        // High affinity and strong emotion -> Mirror
        if affinity > 0.5 && their_emotion.arousal > 0.6 {
            let mirror_intensity = self.emotional_mirroring.mirror_emotion(
                agent_id,
                their_emotion,
                affinity,
            ).await?;

            return Ok(EmpathyResponse::Mirror(mirror_intensity));
        }

        // Negative emotion and positive relationship -> Support
        if their_emotion.valence < -0.3 && affinity > 0.3 {
            let compassion = self.compassion_generator.generate_response(
                agent_id,
                their_emotion,
                &format!("{:?}", model.relationship_status.relationship_type),
                "observed emotional distress",
            ).await?;

            return Ok(EmpathyResponse::Support(compassion.verbal_response));
        }

        // Otherwise, maintain distance
        Ok(EmpathyResponse::Distance)
    }

    /// Process empathetic response to agent (detailed version)
    pub async fn process_empathy(
        &self,
        agent_id: &AgentId,
        situation: &str,
    ) -> Result<DetailedEmpathyResponse> {
        // Get agent's mental model
        let model = self.theory_of_mind.get_model(agent_id).await
            .ok_or_else(|| anyhow!("No model for agent {}", agent_id))?;

        // Take their perspective
        let perspective = self.perspective_taking
            .take_perspective(agent_id, situation, true)
            .await?;

        // Mirror their emotions (based on relationship)
        let mirror_intensity = self.emotional_mirroring.mirror_emotion(
            agent_id,
            &model.emotional_state,
            model.relationship_status.affinity,
        ).await?;

        // Generate compassionate response
        let compassion = self.compassion_generator.generate_response(
            agent_id,
            &model.emotional_state,
            &format!("{:?}", model.relationship_status.relationship_type),
            situation,
        ).await?;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.emotions_mirrored += 1;
        stats.perspectives_taken += 1;
        stats.compassionate_responses += 1;
        stats.total_emotional_support += compassion.emotional_support_level;

        // Send updates
        let _ = self.update_tx.send(EmpathyUpdate::EmotionMirrored(
            agent_id.clone(),
            model.emotional_state.primary_emotion,
            mirror_intensity,
        )).await;

        let _ = self.update_tx.send(EmpathyUpdate::PerspectiveTaken(
            agent_id.clone(),
            perspective.accuracy_estimate,
        )).await;

        let _ = self.update_tx.send(EmpathyUpdate::CompassionShown(
            agent_id.clone(),
            compassion.action.clone(),
        )).await;

        Ok(DetailedEmpathyResponse {
            agent_id: agent_id.clone(),
            perspective: perspective.clone(),
            emotional_mirroring_intensity: mirror_intensity,
            compassionate_response: compassion,
            empathy_score: self.calculate_empathy_score(
                &perspective,
                mirror_intensity,
                &model,
            ),
        })
    }

    /// Calculate overall empathy score
    fn calculate_empathy_score(
        &self,
        perspective: &EnhancedPerspective,
        mirror_intensity: f32,
        model: &MentalModel,
    ) -> f32 {
        let perspective_weight = 0.4;
        let mirroring_weight = 0.3;
        let relationship_weight = 0.3;

        (perspective.accuracy_estimate * perspective_weight +
         mirror_intensity * mirroring_weight +
         model.relationship_status.affinity.max(0.0) * relationship_weight)
        .clamp(0.0, 1.0)
    }

    /// Process group emotional dynamics
    pub async fn process_group_dynamics(
        &self,
        group: Vec<AgentId>,
    ) -> Result<GroupEmpathyDynamics> {
        let mut group_emotions = Vec::new();

        // Gather emotional states
        for agent_id in &group {
            if let Some(model) = self.theory_of_mind.get_model(agent_id).await {
                group_emotions.push((agent_id.clone(), model.emotional_state));
            }
        }

        // Process emotional contagion
        let contagion = self.emotional_contagion
            .process_group_emotion(group_emotions.clone())
            .await?;

        if contagion.affected {
            let mut stats = self.stats.write().await;
            stats.contagions_experienced += 1;

            let _ = self.update_tx.send(EmpathyUpdate::ContagionOccurred(
                contagion.dominant_emotion,
                contagion.contagion_strength,
            )).await;
        }

        // Calculate group empathy metrics
        let mut total_valence = 0.0;
        let mut total_arousal = 0.0;
        let count = group_emotions.len() as f32;

        for (_, state) in &group_emotions {
            total_valence += state.valence;
            total_arousal += state.arousal;
        }

        Ok(GroupEmpathyDynamics {
            group_size: group.len(),
            dominant_emotion: contagion.dominant_emotion,
            average_valence: total_valence / count,
            average_arousal: total_arousal / count,
            contagion_effect: contagion,
            cohesion_score: self.calculate_group_cohesion(&group_emotions),
        })
    }

    /// Calculate group emotional cohesion
    fn calculate_group_cohesion(
        &self,
        emotions: &[(AgentId, EmotionalStateModel)],
    ) -> f32 {
        if emotions.len() < 2 {
            return 1.0;
        }

        let mut similarity_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..emotions.len() {
            for j in (i + 1)..emotions.len() {
                let valence_diff = (emotions[i].1.valence - emotions[j].1.valence).abs();
                let arousal_diff = (emotions[i].1.arousal - emotions[j].1.arousal).abs();

                let similarity = 1.0 - (valence_diff + arousal_diff) / 2.0;
                similarity_sum += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            similarity_sum / comparisons as f32
        } else {
            0.5
        }
    }

    /// Main empathy processing loop
    async fn empathy_loop(&self) {
        let mut interval = interval(self.config.update_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.update_empathy_state().await {
                debug!("Empathy update error: {}", e);
            }
        }
    }

    /// Update empathy state
    async fn update_empathy_state(&self) -> Result<()> {
        // Update average empathy accuracy
        let stats = self.stats.read().await;
        if stats.perspectives_taken > 0 {
            // This would track actual accuracy over time
            // For now, we use a placeholder
            let accuracy = 0.7 + (stats.perspectives_taken as f32 / 1000.0).min(0.2);
            drop(stats);

            self.stats.write().await.avg_empathy_accuracy = accuracy;
        }

        Ok(())
    }

    /// Get empathy statistics
    pub async fn get_stats(&self) -> EmpathyStats {
        self.stats.read().await.clone()
    }
    
    /// Get agent states for empathy modeling
    pub async fn get_agent_states(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut agent_states = HashMap::new();
        
        // Get emotional state
        let emotional_state = self.emotional_core.get_emotional_state().await;
        agent_states.insert(
            "emotional_core".to_string(),
            serde_json::json!({
                "arousal": emotional_state.overall_arousal,
                "valence": emotional_state.overall_valence,
            })
        );
        
        // Get stats
        let stats = self.stats.read().await;
        agent_states.insert(
            "empathy_stats".to_string(),
            serde_json::json!({
                "perspectives_taken": stats.perspectives_taken,
                "mirrored": stats.emotions_mirrored,
            })
        );
        
        Ok(agent_states)
    }
}

/// Complete empathy response
#[derive(Clone, Debug)]
pub enum EmpathyResponse {
    /// Mirror the agent's emotion
    Mirror(f32), // intensity

    /// Provide support
    Support(String), // action description

    /// Maintain emotional distance
    Distance,
}

/// Detailed empathy response data
#[derive(Clone, Debug)]
pub struct DetailedEmpathyResponse {
    pub agent_id: AgentId,
    pub perspective: EnhancedPerspective,
    pub emotional_mirroring_intensity: f32,
    pub compassionate_response: CompassionateResponse,
    pub empathy_score: f32,
}

/// Group empathy dynamics
#[derive(Clone, Debug)]
pub struct GroupEmpathyDynamics {
    pub group_size: usize,
    pub dominant_emotion: CoreEmotion,
    pub average_valence: f32,
    pub average_arousal: f32,
    pub contagion_effect: ContagionEffect,
    pub cohesion_score: f32,
}

#[cfg(test)]
mod tests {


    #[tokio::test]
    async fn test_emotional_mirroring() {
        // Test would verify mirroring behavior
    }

    #[tokio::test]
    async fn test_perspective_taking() {
        // Test would verify perspective simulation
    }
}
