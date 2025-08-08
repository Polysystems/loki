//! Three-Gradient Coordinator Module
//!
//! This module implements the coordination of three fundamental gradients:
//! 1. Value Gradient - Individual optimization and goal achievement
//! 2. Harmony Gradient - Social cooperation and multi-agent coordination
//! 3. Intuition Gradient - Creative exploration and emergent insights
//!
//! Based on "TC & LMAO: Layered Multi-Agent Ontologies" paper concepts.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::cognitive::value_gradients::{StateGradient, ValueGradient};
use crate::cognitive::{
    Goal,
    GoalType,
    ResourceRequirements,
};
use crate::cognitive::emotional_core::EmotionalBlend;
use crate::cognitive::theory_of_mind::AgentId;
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// The three fundamental gradients of thermodynamic cognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeGradientState {
    /// Individual value optimization gradient
    pub value_gradient: GradientVector,

    /// Social harmony and cooperation gradient
    pub harmony_gradient: GradientVector,

    /// Creative intuition and exploration gradient
    pub intuition_gradient: GradientVector,

    /// Combined magnitude across all three gradients
    pub total_magnitude: f64,

    /// Coherence between the three gradients (how well they align)
    pub gradient_coherence: f64,

    /// Timestamp of measurement
    #[serde(skip)]
    #[serde(default = "Instant::now")]
    pub timestamp: Instant,
}

/// A gradient vector in the three-dimensional consciousness space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientVector {
    /// Direction components of the gradient
    pub components: Vec<f64>,

    /// Magnitude of this gradient
    pub magnitude: f64,

    /// Confidence in the gradient measurement
    pub confidence: f64,

    /// Temporal stability (how consistent over time)
    pub stability: f64,
}

impl Default for GradientVector {
    fn default() -> Self {
        Self {
            components: vec![0.0; 6], // 6-dimensional cognitive space
            magnitude: 0.0,
            confidence: 0.5,
            stability: 0.5,
        }
    }
}

/// Configuration for gradient coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeGradientConfig {
    /// Weight for value gradient in final decision
    pub value_weight: f64,

    /// Weight for harmony gradient in final decision
    pub harmony_weight: f64,

    /// Weight for intuition gradient in final decision
    pub intuition_weight: f64,

    /// Minimum coherence threshold for stable operation
    pub coherence_threshold: f64,

    /// Update frequency for gradient calculation
    pub update_interval: Duration,

    /// Number of agents to consider for harmony calculations
    pub harmony_agent_count: usize,

    /// Exploration factor for intuition gradient
    pub intuition_exploration_factor: f64,
}

impl Default for ThreeGradientConfig {
    fn default() -> Self {
        Self {
            value_weight: 0.4,      // Individual optimization
            harmony_weight: 0.35,   // Social cooperation
            intuition_weight: 0.25, // Creative exploration
            coherence_threshold: 0.6,
            update_interval: Duration::from_secs(15), // Faster than value gradient
            harmony_agent_count: 10,
            intuition_exploration_factor: 0.3,
        }
    }
}

/// Multi-agent harmony state for social coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyState {
    /// Agents and their states being considered
    pub agent_states: HashMap<AgentId, AgentHarmonyProfile>,

    /// Overall harmony level (0.0 to 1.0)
    pub harmony_level: f64,

    /// Cooperation opportunities identified
    pub cooperation_opportunities: Vec<CooperationOpportunity>,

    /// Conflicts detected and their intensities
    pub conflicts: Vec<HarmonyConflict>,

    /// Social energy flow between agents
    pub social_energy_flow: HashMap<(AgentId, AgentId), f64>,
}

/// Individual agent's profile for harmony calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHarmonyProfile {
    /// Agent's current goals and priorities
    pub goals: Vec<Goal>,

    /// Agent's emotional state
    pub emotional_state: EmotionalBlend,

    /// Agent's cooperation history
    pub cooperation_score: f64,

    /// Agent's current resources and capabilities
    pub resources: Vec<String>,

    /// Agent's communication preferences
    pub communication_style: HarmonyCommunicationStyle,
}

/// Cooperation opportunity between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooperationOpportunity {
    /// Agents involved in this opportunity
    pub agents: Vec<AgentId>,

    /// Type of cooperation
    pub cooperation_type: CooperationType,

    /// Potential mutual benefit score
    pub mutual_benefit: f64,

    /// Resource requirements
    pub resources_needed: Vec<String>,

    /// Estimated duration
    pub duration_estimate: Duration,
}

/// Types of cooperation between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CooperationType {
    /// Sharing resources for mutual benefit
    ResourceSharing,

    /// Collaborating on a shared goal
    GoalCollaboration,

    /// Providing assistance without immediate return
    AltruisticHelp,

    /// Exchanging different types of value
    ValueExchange,

    /// Creating something new together
    CreativeCollaboration,
}

/// Harmony conflict between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyConflict {
    /// Agents in conflict
    pub agents: Vec<AgentId>,

    /// Nature of the conflict
    pub conflict_type: HarmonyConflictType,

    /// Intensity of conflict (0.0 to 1.0)
    pub intensity: f64,

    /// Potential resolution strategies
    pub resolution_strategies: Vec<String>,
}

/// Types of conflicts that can arise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmonyConflictType {
    /// Competition for the same resource
    ResourceCompetition,

    /// Incompatible goals
    GoalConflict,

    /// Different values or priorities
    ValueMismatch,

    /// Communication breakdown
    CommunicationFailure,

    /// Temporal coordination issues
    TimingConflict,
}

/// Communication style preferences
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HarmonyCommunicationStyle {
    /// Direct and efficient
    Direct,

    /// Collaborative and consensus-seeking
    Collaborative,

    /// Supportive and empathetic
    Supportive,

    /// Analytical and data-driven
    Analytical,

    /// Creative and brainstorming-oriented
    Creative,
}

/// Intuition state for creative exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntuitionState {
    /// Current level of creative energy
    pub creative_energy: f64,

    /// Exploration targets and their potential
    pub exploration_targets: Vec<ExplorationTarget>,

    /// Pattern recognition insights
    pub pattern_insights: Vec<PatternInsight>,

    /// Emergent ideas and their confidence
    pub emergent_ideas: Vec<EmergentIdea>,

    /// Curiosity drivers
    pub curiosity_drivers: Vec<CuriosityDriver>,
}

/// Target for creative exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationTarget {
    /// Domain or area to explore
    pub domain: String,

    /// Current understanding level
    pub understanding_level: f64,

    /// Potential for discovery
    pub discovery_potential: f64,

    /// Resources required for exploration
    pub exploration_cost: f64,
}

/// Pattern recognition insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternInsight {
    /// Description of the pattern
    pub pattern_description: String,

    /// Confidence in the pattern
    pub confidence: f64,

    /// Domains where pattern applies
    pub applicable_domains: Vec<String>,

    /// Potential applications
    pub applications: Vec<String>,
}

/// Emergent idea from intuitive processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentIdea {
    /// Description of the idea
    pub idea_description: String,

    /// Novelty score
    pub novelty: f64,

    /// Potential value
    pub potential_value: f64,

    /// Implementation feasibility
    pub feasibility: f64,
}

/// Driver of curiosity and exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityDriver {
    /// What is driving curiosity
    pub source: String,

    /// Intensity of curiosity
    pub intensity: f64,

    /// Direction of exploration
    pub direction: Vec<String>,
}

/// Main coordinator for the three gradients
pub struct ThreeGradientCoordinator {
    /// Configuration
    config: ThreeGradientConfig,

    /// Value gradient system
    value_gradient: Arc<ValueGradient>,

    /// Current three-gradient state
    current_state: Arc<RwLock<ThreeGradientState>>,

    /// Harmony state tracking
    harmony_state: Arc<RwLock<HarmonyState>>,

    /// Intuition state tracking
    intuition_state: Arc<RwLock<IntuitionState>>,

    /// Memory system for learning patterns
    memory: Arc<CognitiveMemory>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

impl ThreeGradientCoordinator {
    /// Create a new three-gradient coordinator
    pub async fn new(
        value_gradient: Arc<ValueGradient>,
        memory: Arc<CognitiveMemory>,
        config: Option<ThreeGradientConfig>,
    ) -> Result<Arc<Self>> {
        info!("Initializing Three-Gradient Coordinator - unifying value, harmony, and intuition");

        let config = config.unwrap_or_default();

        let coordinator = Arc::new(Self {
            config,
            value_gradient,
            current_state: Arc::new(RwLock::new(ThreeGradientState {
                value_gradient: GradientVector::default(),
                harmony_gradient: GradientVector::default(),
                intuition_gradient: GradientVector::default(),
                total_magnitude: 0.0,
                gradient_coherence: 0.5,
                timestamp: Instant::now(),
            })),
            harmony_state: Arc::new(RwLock::new(HarmonyState {
                agent_states: HashMap::new(),
                harmony_level: 0.5,
                cooperation_opportunities: Vec::new(),
                conflicts: Vec::new(),
                social_energy_flow: HashMap::new(),
            })),
            intuition_state: Arc::new(RwLock::new(IntuitionState {
                creative_energy: 0.5,
                exploration_targets: Vec::new(),
                pattern_insights: Vec::new(),
                emergent_ideas: Vec::new(),
                curiosity_drivers: Vec::new(),
            })),
            memory,
            running: Arc::new(RwLock::new(false)),
        });

        // Store initialization in memory
        coordinator
            .memory
            .store(
                "Three-Gradient Coordinator initialized - value, harmony, intuition unified"
                    .to_string(),
                vec!["three_gradient_coordination".to_string()],
                MemoryMetadata {
                    source: "three_gradient_coordinator".to_string(),
                    tags: vec!["initialization".to_string(), "multi_gradient".to_string()],
                    importance: 1.0,
                    associations: vec![],
                    context: Some("Three-gradient coordinator initialization".to_string()),
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

        Ok(coordinator)
    }

    /// Start the three-gradient coordination loop
    pub async fn start(self: Arc<Self>) -> Result<()> {
        *self.running.write().await = true;
        info!("Starting Three-Gradient coordination loop");

        let coordinator = self.clone();
        tokio::spawn(async move {
            coordinator.coordination_loop().await;
        });

        Ok(())
    }

    /// Main coordination loop
    async fn coordination_loop(&self) {
        let mut interval = tokio::time::interval(self.config.update_interval);

        while *self.running.read().await {
            interval.tick().await;

            if let Err(e) = self.perform_coordination_step().await {
                warn!("Three-gradient coordination step failed: {}", e);
            }
        }
    }

    /// Perform one coordination step
    async fn perform_coordination_step(&self) -> Result<()> {
        // Get current value gradient
        let value_gradient = self.value_gradient.get_current_gradient().await?;

        // Calculate harmony gradient
        let harmony_gradient = self.calculate_harmony_gradient().await?;

        // Calculate intuition gradient
        let intuition_gradient = self.calculate_intuition_gradient().await?;

        // Compute combined state
        let combined_state =
            self.combine_gradients(&value_gradient, &harmony_gradient, &intuition_gradient).await?;

        // Update current state
        *self.current_state.write().await = combined_state.clone();

        // Store insights if significant
        if combined_state.total_magnitude > 0.5 {
            self.store_coordination_insight(&combined_state).await?;
        }

        debug!(
            "Three-gradient coordination: total_magnitude={:.3}, coherence={:.3}",
            combined_state.total_magnitude, combined_state.gradient_coherence
        );

        Ok(())
    }

    /// Calculate harmony gradient based on multi-agent cooperation
    async fn calculate_harmony_gradient(&self) -> Result<GradientVector> {
        let harmony_state = self.harmony_state.read().await;

        // Calculate social cooperation potential
        let cooperation_potential = self.calculate_cooperation_potential(&harmony_state).await?;

        // Calculate conflict resolution direction
        let conflict_resolution = self.calculate_conflict_resolution(&harmony_state).await?;

        // Calculate social energy optimization
        let social_energy = self.calculate_social_energy_optimization(&harmony_state).await?;

        // Combine into harmony gradient
        let mut components = vec![0.0; 6];
        components[0] = cooperation_potential * 0.4;
        components[1] = conflict_resolution * 0.3;
        components[2] = social_energy * 0.3;
        // Additional components for social dimensions
        components[3] = harmony_state.harmony_level * 0.2;
        components[4] = self.calculate_trust_gradient(&harmony_state).await? * 0.2;
        components[5] = self.calculate_communication_gradient(&harmony_state).await? * 0.1;

        let magnitude = components.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(GradientVector {
            components,
            magnitude,
            confidence: harmony_state.harmony_level,
            stability: 0.8, // Harmony tends to be stable
        })
    }

    /// Calculate intuition gradient based on creative exploration
    async fn calculate_intuition_gradient(&self) -> Result<GradientVector> {
        let intuition_state = self.intuition_state.read().await;

        // Calculate exploration potential
        let exploration_potential = self.calculate_exploration_potential(&intuition_state).await?;

        // Calculate pattern recognition insights
        let pattern_insights = self.calculate_pattern_insight_gradient(&intuition_state).await?;

        // Calculate creative synthesis potential
        let creative_synthesis = self.calculate_creative_synthesis(&intuition_state).await?;

        // Combine into intuition gradient
        let mut components = vec![0.0; 6];
        components[0] = exploration_potential * 0.4;
        components[1] = pattern_insights * 0.3;
        components[2] = creative_synthesis * 0.3;
        // Additional components for creative dimensions
        components[3] = intuition_state.creative_energy * 0.2;
        components[4] = self.calculate_curiosity_gradient(&intuition_state).await? * 0.3;
        components[5] = self.calculate_novelty_gradient(&intuition_state).await? * 0.2;

        let magnitude = components.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(GradientVector {
            components,
            magnitude,
            confidence: intuition_state.creative_energy,
            stability: 0.6, // Intuition is less stable than harmony
        })
    }

    /// Combine the three gradients into unified state
    async fn combine_gradients(
        &self,
        value_grad: &StateGradient,
        harmony_grad: &GradientVector,
        intuition_grad: &GradientVector,
    ) -> Result<ThreeGradientState> {
        // Convert value gradient to our format
        let value_gradient = GradientVector {
            components: value_grad.direction.clone(),
            magnitude: value_grad.magnitude,
            confidence: value_grad.confidence,
            stability: 0.7, // Value gradients are moderately stable
        };

        // Calculate weighted combination
        let total_magnitude = self.config.value_weight * value_gradient.magnitude
            + self.config.harmony_weight * harmony_grad.magnitude
            + self.config.intuition_weight * intuition_grad.magnitude;

        // Calculate coherence between gradients
        let gradient_coherence = self
            .calculate_gradient_coherence(&value_gradient, harmony_grad, intuition_grad)
            .await?;

        Ok(ThreeGradientState {
            value_gradient,
            harmony_gradient: harmony_grad.clone(),
            intuition_gradient: intuition_grad.clone(),
            total_magnitude,
            gradient_coherence,
            timestamp: Instant::now(),
        })
    }

    /// Calculate coherence between the three gradients
    async fn calculate_gradient_coherence(
        &self,
        value_grad: &GradientVector,
        harmony_grad: &GradientVector,
        intuition_grad: &GradientVector,
    ) -> Result<f64> {
        // Calculate pairwise alignment between gradients
        let value_harmony_alignment =
            self.calculate_vector_alignment(&value_grad.components, &harmony_grad.components);

        let value_intuition_alignment =
            self.calculate_vector_alignment(&value_grad.components, &intuition_grad.components);

        let harmony_intuition_alignment =
            self.calculate_vector_alignment(&harmony_grad.components, &intuition_grad.components);

        // Average alignment as coherence measure
        let coherence =
            (value_harmony_alignment + value_intuition_alignment + harmony_intuition_alignment)
                / 3.0;

        Ok(coherence.clamp(0.0, 1.0))
    }

    /// Calculate alignment between two gradient vectors
    fn calculate_vector_alignment(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 * norm2 > 0.0 { (dot_product / (norm1 * norm2)).clamp(-1.0, 1.0) } else { 0.0 }
    }

    /// Store coordination insights in memory
    async fn store_coordination_insight(&self, state: &ThreeGradientState) -> Result<()> {
        let insight = format!(
            "Three-gradient coordination: value_mag={:.3}, harmony_mag={:.3}, \
             intuition_mag={:.3}, total_mag={:.3}, coherence={:.3}",
            state.value_gradient.magnitude,
            state.harmony_gradient.magnitude,
            state.intuition_gradient.magnitude,
            state.total_magnitude,
            state.gradient_coherence
        );

        self.memory
            .store(
                insight,
                vec![format!("{:?}", state)],
                MemoryMetadata {
                    source: "three_gradient_coordinator".to_string(),
                    tags: vec![
                        "coordination".to_string(),
                        "multi_gradient".to_string(),
                        "consciousness".to_string(),
                    ],
                    importance: (state.total_magnitude * state.gradient_coherence) as f32,
                    associations: vec![],
                    context: Some("Three-gradient coordination processing".to_string()),
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

    // Helper methods for harmony calculations
    async fn calculate_cooperation_potential(&self, harmony_state: &HarmonyState) -> Result<f64> {
        if harmony_state.agent_states.is_empty() {
            return Ok(0.0);
        }

        let mut total_potential = 0.0;
        let agent_count = harmony_state.agent_states.len();

        // Calculate cooperation potential based on complementary resources and goals
        for (agent_a_id, profile_a) in &harmony_state.agent_states {
            for (agent_b_id, profile_b) in &harmony_state.agent_states {
                if agent_a_id == agent_b_id {
                    continue;
                }

                // Goal complementarity analysis
                let goal_synergy =
                    self.calculate_goal_synergy(&profile_a.goals, &profile_b.goals).await;

                // Resource complementarity
                let resource_complement = self
                    .calculate_resource_complementarity(&profile_a.resources, &profile_b.resources);

                // Communication compatibility
                let comm_compatibility = self.calculate_communication_compatibility(
                    &profile_a.communication_style,
                    &profile_b.communication_style,
                );

                // Cooperation history factor
                let trust_factor =
                    (profile_a.cooperation_score + profile_b.cooperation_score) / 2.0;

                // Weighted cooperation potential
                let pair_potential = (goal_synergy * 0.4
                    + resource_complement * 0.3
                    + comm_compatibility * 0.2
                    + trust_factor * 0.1)
                    .clamp(0.0, 1.0);

                total_potential += pair_potential;
            }
        }

        // Average over all possible pairs
        let pair_count = agent_count * (agent_count - 1);
        if pair_count > 0 {
            Ok((total_potential / pair_count as f64).clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }

    /// Calculate synergy between two sets of goals
    async fn calculate_goal_synergy(&self, goals_a: &[Goal], goals_b: &[Goal]) -> f64 {
        if goals_a.is_empty() || goals_b.is_empty() {
            return 0.0;
        }

        let mut synergy_score = 0.0;
        let mut evaluations = 0;

        for goal_a in goals_a {
            for goal_b in goals_b {
                // Check for direct goal alignment
                let name_similarity = self.calculate_text_similarity(&goal_a.name, &goal_b.name);
                let desc_similarity =
                    self.calculate_text_similarity(&goal_a.description, &goal_b.description);

                // Check for complementary goals (different but supporting)
                let complementarity = if goal_a.goal_type != goal_b.goal_type {
                    // Different types can be complementary
                    match (&goal_a.goal_type, &goal_b.goal_type) {
                        (GoalType::Strategic, GoalType::Tactical)
                        | (GoalType::Tactical, GoalType::Strategic) => 0.6,
                        (GoalType::Tactical, GoalType::Operational)
                        | (GoalType::Operational, GoalType::Tactical) => 0.4,
                        _ => 0.2,
                    }
                } else {
                    // Same type - check for overlap vs conflict
                    if name_similarity > 0.7 || desc_similarity > 0.7 {
                        0.8 // High overlap
                    } else {
                        0.3 // Different but same type
                    }
                };

                // Resource sharing potential
                let resource_sharing = self.calculate_resource_sharing_potential(
                    &goal_a.resources_required,
                    &goal_b.resources_required,
                );

                // Priority alignment (similar priorities work better together)
                let priority_alignment =
                    1.0 - (goal_a.priority.to_f32() - goal_b.priority.to_f32()).abs();

                let pair_synergy = (name_similarity * 0.2
                    + desc_similarity * 0.2
                    + complementarity * 0.3
                    + resource_sharing * 0.2
                    + priority_alignment as f64 * 0.1)
                    .clamp(0.0, 1.0);

                synergy_score += pair_synergy;
                evaluations += 1;
            }
        }

        if evaluations > 0 { synergy_score / evaluations as f64 } else { 0.0 }
    }

    /// Calculate resource complementarity between two agents
    fn calculate_resource_complementarity(
        &self,
        resources_a: &[String],
        resources_b: &[String],
    ) -> f64 {
        if resources_a.is_empty() && resources_b.is_empty() {
            return 0.5; // Neutral if both have no resources
        }

        if resources_a.is_empty() || resources_b.is_empty() {
            return 0.3; // One-sided resources have some value
        }

        let set_a: std::collections::HashSet<_> = resources_a.iter().collect();
        let set_b: std::collections::HashSet<_> = resources_b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            return 0.0;
        }

        // Complementarity is higher when there's some overlap but also unique resources
        let overlap_ratio = intersection as f64 / union as f64;
        let unique_a = set_a.difference(&set_b).count() as f64;
        let unique_b = set_b.difference(&set_a).count() as f64;
        let uniqueness_factor = (unique_a + unique_b) / union as f64;

        // Optimal complementarity is around 30% overlap with 70% unique resources
        let ideal_overlap = 0.3;
        let overlap_score = 1.0 - (overlap_ratio - ideal_overlap).abs() / ideal_overlap;

        (overlap_score * 0.4 + uniqueness_factor * 0.6).clamp(0.0, 1.0)
    }

    /// Calculate communication style compatibility
    fn calculate_communication_compatibility(
        &self,
        style_a: &HarmonyCommunicationStyle,
        style_b: &HarmonyCommunicationStyle,
    ) -> f64 {
        use HarmonyCommunicationStyle::*;

        match (style_a, style_b) {
            // Perfect matches
            (Direct, Direct)
            | (Collaborative, Collaborative)
            | (Supportive, Supportive)
            | (Analytical, Analytical)
            | (Creative, Creative) => 1.0,

            // Good combinations
            (Collaborative, Supportive) | (Supportive, Collaborative) => 0.9,
            (Analytical, Direct) | (Direct, Analytical) => 0.8,
            (Creative, Collaborative) | (Collaborative, Creative) => 0.8,
            (Creative, Supportive) | (Supportive, Creative) => 0.7,

            // Neutral combinations
            (Analytical, Supportive) | (Supportive, Analytical) => 0.6,
            (Direct, Collaborative) | (Collaborative, Direct) => 0.6,

            // Challenging combinations
            (Direct, Supportive) | (Supportive, Direct) => 0.4,
            (Direct, Creative) | (Creative, Direct) => 0.3,
            (Analytical, Creative) | (Creative, Analytical) => 0.5,
            (Collaborative, Analytical) | (Analytical, Collaborative) => 0.6,
        }
    }

    /// Calculate text similarity (basic implementation)
    fn calculate_text_similarity(&self, text_a: &str, text_b: &str) -> f64 {
        if text_a.is_empty() && text_b.is_empty() {
            return 1.0;
        }
        if text_a.is_empty() || text_b.is_empty() {
            return 0.0;
        }

        let text_a_normalized = text_a.to_lowercase();
        let text_b_normalized = text_b.to_lowercase();
        let words_a: std::collections::HashSet<_> = text_a_normalized.split_whitespace().collect();
        let words_b: std::collections::HashSet<_> = text_b_normalized.split_whitespace().collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }

    /// Calculate resource sharing potential between goals
    fn calculate_resource_sharing_potential(
        &self,
        req_a: &ResourceRequirements,
        req_b: &ResourceRequirements,
    ) -> f64 {
        // Goals that require different types of resources can share more easily
        let cognitive_conflict = (req_a.cognitive_load + req_b.cognitive_load).min(1.0);
        let emotional_conflict = (req_a.emotional_energy + req_b.emotional_energy).min(1.0);
        let time_conflict =
            if req_a.time_estimate.is_some() && req_b.time_estimate.is_some() { 0.8 } else { 0.2 };

        let total_conflict = (cognitive_conflict + emotional_conflict + time_conflict) / 3.0;

        // Lower conflict means higher sharing potential
        (1.0f64 - total_conflict as f64).clamp(0.0, 1.0)
    }

    async fn calculate_conflict_resolution(&self, harmony_state: &HarmonyState) -> Result<f64> {
        if harmony_state.conflicts.is_empty() {
            return Ok(1.0); // No conflicts = perfect resolution potential
        }

        let mut total_resolution_potential = 0.0;

        for conflict in &harmony_state.conflicts {
            let base_resolution = 1.0 - conflict.intensity;

            // Factor in conflict type difficulty
            let type_difficulty = match conflict.conflict_type {
                HarmonyConflictType::CommunicationFailure => 0.2, // Easiest to resolve
                HarmonyConflictType::TimingConflict => 0.3,
                HarmonyConflictType::ResourceCompetition => 0.5,
                HarmonyConflictType::ValueMismatch => 0.7,
                HarmonyConflictType::GoalConflict => 0.8, // Hardest to resolve
            };

            // Resolution strategy availability
            let strategy_factor = if conflict.resolution_strategies.is_empty() {
                0.3 // No strategies available
            } else {
                0.7 + (conflict.resolution_strategies.len() as f64 * 0.1).min(0.3)
            };

            // Agent count factor (more agents = more complex)
            let complexity_factor = match conflict.agents.len() {
                0..=2 => 1.0,
                3..=4 => 0.8,
                5..=6 => 0.6,
                _ => 0.4,
            };

            let conflict_resolution_potential =
                (base_resolution * (1.0 - type_difficulty) * strategy_factor * complexity_factor)
                    .clamp(0.0, 1.0);

            total_resolution_potential += conflict_resolution_potential;
        }

        Ok((total_resolution_potential / harmony_state.conflicts.len() as f64).clamp(0.0, 1.0))
    }

    async fn calculate_social_energy_optimization(
        &self,
        harmony_state: &HarmonyState,
    ) -> Result<f64> {
        if harmony_state.agent_states.len() < 2 {
            return Ok(0.0);
        }

        let mut total_energy_efficiency = 0.0;
        let mut flow_count = 0;

        // Analyze existing energy flows
        for ((agent_a, agent_b), flow_strength) in &harmony_state.social_energy_flow {
            // Get agent profiles
            if let (Some(profile_a), Some(profile_b)) =
                (harmony_state.agent_states.get(agent_a), harmony_state.agent_states.get(agent_b))
            {
                // Calculate optimal flow based on agent compatibility
                let compatibility = self.calculate_agent_compatibility(profile_a, profile_b).await;
                let optimal_flow = compatibility * 0.8; // Scale to reasonable flow level

                // Current efficiency is how close actual flow is to optimal
                let efficiency = 1.0 - (flow_strength - optimal_flow).abs();
                total_energy_efficiency += efficiency;
                flow_count += 1;
            }
        }

        // Calculate potential for new beneficial flows
        let mut potential_flows = 0;
        let mut potential_benefit = 0.0;

        for (agent_a_id, profile_a) in &harmony_state.agent_states {
            for (agent_b_id, profile_b) in &harmony_state.agent_states {
                if agent_a_id == agent_b_id {
                    continue;
                }

                // Check if flow already exists
                if harmony_state
                    .social_energy_flow
                    .contains_key(&(agent_a_id.clone(), agent_b_id.clone()))
                {
                    continue;
                }

                let compatibility = self.calculate_agent_compatibility(profile_a, profile_b).await;
                if compatibility > 0.6 {
                    potential_benefit += compatibility;
                    potential_flows += 1;
                }
            }
        }

        // Combine current efficiency with potential for improvement
        let current_efficiency = if flow_count > 0 {
            total_energy_efficiency / flow_count as f64
        } else {
            0.5 // Neutral if no flows exist
        };

        let potential_factor =
            if potential_flows > 0 { potential_benefit / potential_flows as f64 } else { 0.0 };

        // Weight current efficiency more heavily than potential
        Ok((current_efficiency * 0.7 + potential_factor * 0.3).clamp(0.0, 1.0))
    }

    /// Calculate overall compatibility between two agents
    async fn calculate_agent_compatibility(
        &self,
        profile_a: &AgentHarmonyProfile,
        profile_b: &AgentHarmonyProfile,
    ) -> f64 {
        // Goal synergy
        let goal_synergy = self.calculate_goal_synergy(&profile_a.goals, &profile_b.goals).await;

        // Resource complementarity
        let resource_complement =
            self.calculate_resource_complementarity(&profile_a.resources, &profile_b.resources);

        // Communication compatibility
        let comm_compatibility = self.calculate_communication_compatibility(
            &profile_a.communication_style,
            &profile_b.communication_style,
        );

        // Trust factor
        let trust_factor = (profile_a.cooperation_score + profile_b.cooperation_score) / 2.0;

        // Weighted compatibility score
        (goal_synergy * 0.3
            + resource_complement * 0.2
            + comm_compatibility * 0.2
            + trust_factor * 0.1)
            .clamp(0.0, 1.0)
    }



    async fn calculate_trust_gradient(&self, harmony_state: &HarmonyState) -> Result<f64> {
        if harmony_state.agent_states.is_empty() {
            return Ok(0.5);
        }

        let mut trust_metrics = Vec::new();

        // Base cooperation scores
        let avg_cooperation: f64 = harmony_state
            .agent_states
            .values()
            .map(|profile| profile.cooperation_score)
            .sum::<f64>()
            / harmony_state.agent_states.len() as f64;
        trust_metrics.push(avg_cooperation);

        // Trust consistency (lower variance = higher trust)
        let cooperation_scores: Vec<f64> =
            harmony_state.agent_states.values().map(|profile| profile.cooperation_score).collect();

        if cooperation_scores.len() > 1 {
            let variance = self.calculate_variance(&cooperation_scores);
            let consistency = 1.0 / (1.0 + variance); // Lower variance = higher consistency
            trust_metrics.push(consistency);
        }

        // Trust based on successful cooperation opportunities
        let active_cooperations = harmony_state.cooperation_opportunities.len() as f64;
        let cooperation_trust = (active_cooperations / 10.0).min(1.0); // Scale based on opportunities
        trust_metrics.push(cooperation_trust);

        // Trust degradation from conflicts
        let conflict_impact = if harmony_state.conflicts.is_empty() {
            1.0
        } else {
            let avg_conflict_intensity: f64 =
                harmony_state.conflicts.iter().map(|c| c.intensity).sum::<f64>()
                    / harmony_state.conflicts.len() as f64;
            1.0 - (avg_conflict_intensity * 0.5) // Conflicts reduce trust
        };
        trust_metrics.push(conflict_impact);

        // Social energy flow health
        if !harmony_state.social_energy_flow.is_empty() {
            let avg_flow: f64 = harmony_state.social_energy_flow.values().sum::<f64>()
                / harmony_state.social_energy_flow.len() as f64;
            let flow_health = avg_flow.clamp(0.0, 1.0);
            trust_metrics.push(flow_health);
        }

        // Calculate weighted average
        if trust_metrics.is_empty() {
            Ok(0.5)
        } else {
            Ok(trust_metrics.iter().sum::<f64>() / trust_metrics.len() as f64)
        }
    }

    /// Calculate variance of a set of values
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }

    async fn calculate_communication_gradient(&self, harmony_state: &HarmonyState) -> Result<f64> {
        if harmony_state.agent_states.len() < 2 {
            return Ok(0.5);
        }

        let mut communication_metrics = Vec::new();

        // Communication style diversity (some diversity is good, too much is
        // problematic)
        let style_counts = self.count_communication_styles(harmony_state);
        let diversity = style_counts.len() as f64 / harmony_state.agent_states.len() as f64;
        let optimal_diversity = 0.6; // Target 60% diversity
        let diversity_score = 1.0 - (diversity - optimal_diversity).abs() / optimal_diversity;
        communication_metrics.push(diversity_score.clamp(0.0, 1.0));

        // Communication compatibility between agents
        let mut compatibility_scores = Vec::new();
        let agents: Vec<_> = harmony_state.agent_states.iter().collect();

        for i in 0..agents.len() {
            for j in (i + 1)..agents.len() {
                let (_, profile_a) = agents[i];
                let (_, profile_b) = agents[j];

                let compatibility = self.calculate_communication_compatibility(
                    &profile_a.communication_style,
                    &profile_b.communication_style,
                );
                compatibility_scores.push(compatibility);
            }
        }

        if !compatibility_scores.is_empty() {
            let avg_compatibility =
                compatibility_scores.iter().sum::<f64>() / compatibility_scores.len() as f64;
            communication_metrics.push(avg_compatibility);
        }

        // Communication effectiveness based on conflict resolution
        let communication_effectiveness = if harmony_state
            .conflicts
            .iter()
            .any(|c| matches!(c.conflict_type, HarmonyConflictType::CommunicationFailure))
        {
            0.3 // Poor communication if there are communication failures
        } else {
            0.8 // Good communication if no communication conflicts
        };
        communication_metrics.push(communication_effectiveness);

        // Social energy flow as indicator of communication health
        if !harmony_state.social_energy_flow.is_empty() {
            let flow_variance = self.calculate_variance(
                &harmony_state.social_energy_flow.values().cloned().collect::<Vec<_>>(),
            );
            let flow_balance = 1.0 / (1.0 + flow_variance); // Balanced flows indicate good communication
            communication_metrics.push(flow_balance);
        }

        Ok(communication_metrics.iter().sum::<f64>() / communication_metrics.len() as f64)
    }

    /// Count communication styles in the harmony state
    fn count_communication_styles(
        &self,
        harmony_state: &HarmonyState,
    ) -> std::collections::HashMap<HarmonyCommunicationStyle, usize> {
        let mut counts = std::collections::HashMap::new();

        for profile in harmony_state.agent_states.values() {
            *counts.entry(profile.communication_style.clone()).or_insert(0) += 1;
        }

        counts
    }

    /// Get current three-gradient state
    pub async fn get_current_state(&self) -> ThreeGradientState {
        self.current_state.read().await.clone()
    }

    /// Get harmony state
    pub async fn get_harmony_state(&self) -> HarmonyState {
        self.harmony_state.read().await.clone()
    }

    /// Get intuition state
    pub async fn get_intuition_state(&self) -> IntuitionState {
        self.intuition_state.read().await.clone()
    }

    /// Update agent state for harmony calculations
    pub async fn update_agent_state(
        &self,
        agent_id: AgentId,
        profile: AgentHarmonyProfile,
    ) -> Result<()> {
        let mut harmony_state = self.harmony_state.write().await;
        harmony_state.agent_states.insert(agent_id, profile);

        // Recalculate cooperation opportunities
        harmony_state.cooperation_opportunities =
            self.identify_cooperation_opportunities(&harmony_state).await?;

        Ok(())
    }

    /// Identify cooperation opportunities between agents
    async fn identify_cooperation_opportunities(
        &self,
        _harmony_state: &HarmonyState,
    ) -> Result<Vec<CooperationOpportunity>> {
        // Simplified opportunity identification
        Ok(vec![CooperationOpportunity {
            agents: vec![], // Would be populated with actual agent analysis
            cooperation_type: CooperationType::ResourceSharing,
            mutual_benefit: 0.8,
            resources_needed: vec!["computational_resources".to_string()],
            duration_estimate: Duration::from_secs(7200), // 2 hours
        }])
    }

    /// Add exploration target for intuition
    pub async fn add_exploration_target(&self, target: ExplorationTarget) -> Result<()> {
        let mut intuition_state = self.intuition_state.write().await;
        intuition_state.exploration_targets.push(target);
        Ok(())
    }

    /// Add emergent idea from intuitive processes
    pub async fn add_emergent_idea(&self, idea: EmergentIdea) -> Result<()> {
        let mut intuition_state = self.intuition_state.write().await;
        intuition_state.emergent_ideas.push(idea);
        Ok(())
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&self) -> Result<()> {
        *self.running.write().await = false;
        info!("Three-Gradient Coordinator shutdown");
        Ok(())
    }

    // Helper methods for intuition calculations
    async fn calculate_exploration_potential(
        &self,
        intuition_state: &IntuitionState,
    ) -> Result<f64> {
        if intuition_state.exploration_targets.is_empty() {
            return Ok(0.3); // Default exploration potential
        }

        let avg_discovery_potential: f64 = intuition_state
            .exploration_targets
            .iter()
            .map(|target| target.discovery_potential)
            .sum::<f64>()
            / intuition_state.exploration_targets.len() as f64;

        // Factor in exploration cost efficiency
        let avg_cost_efficiency: f64 = intuition_state
            .exploration_targets
            .iter()
            .map(|target| {
                if target.exploration_cost > 0.0 {
                    target.discovery_potential / target.exploration_cost
                } else {
                    target.discovery_potential * 2.0 // No cost is highly efficient
                }
            })
            .sum::<f64>()
            / intuition_state.exploration_targets.len() as f64;

        // Combine discovery potential with cost efficiency
        Ok((avg_discovery_potential * 0.7 + avg_cost_efficiency.min(1.0) * 0.3).clamp(0.0, 1.0))
    }

    async fn calculate_pattern_insight_gradient(
        &self,
        intuition_state: &IntuitionState,
    ) -> Result<f64> {
        if intuition_state.pattern_insights.is_empty() {
            return Ok(0.2); // Low baseline without insights
        }

        let mut insight_scores = Vec::new();

        for insight in &intuition_state.pattern_insights {
            // Base confidence
            let base_score = insight.confidence;

            // Domain applicability bonus
            let domain_bonus = (insight.applicable_domains.len() as f64 * 0.1).min(0.3);

            // Application potential bonus
            let application_bonus = (insight.applications.len() as f64 * 0.1).min(0.2);

            let total_score = (base_score + domain_bonus + application_bonus).clamp(0.0, 1.0);
            insight_scores.push(total_score);
        }

        let avg_confidence = insight_scores.iter().sum::<f64>() / insight_scores.len() as f64;
        Ok(avg_confidence)
    }

    async fn calculate_creative_synthesis(&self, intuition_state: &IntuitionState) -> Result<f64> {
        if intuition_state.emergent_ideas.is_empty() {
            return Ok(0.1); // Very low without ideas
        }

        let mut synthesis_scores = Vec::new();

        for idea in &intuition_state.emergent_ideas {
            // Weighted combination of novelty, value, and feasibility
            let synthesis_quality = (
                idea.novelty * 0.4 +           // Novelty is most important for creativity
                idea.potential_value * 0.4 +   // But value matters too
                idea.feasibility * 0.2
                // Feasibility ensures practicality
            )
            .clamp(0.0, 1.0);

            synthesis_scores.push(synthesis_quality);
        }

        // Calculate both average and max (best idea potential)
        let avg_synthesis = synthesis_scores.iter().sum::<f64>() / synthesis_scores.len() as f64;
        let max_synthesis = synthesis_scores.iter().fold(0.0f64, |acc, &x| acc.max(x));

        // Weight average higher but include best idea potential
        Ok((avg_synthesis * 0.7 + max_synthesis * 0.3).clamp(0.0, 1.0))
    }

    async fn calculate_curiosity_gradient(&self, intuition_state: &IntuitionState) -> Result<f64> {
        if intuition_state.curiosity_drivers.is_empty() {
            return Ok(0.2); // Low baseline curiosity
        }

        let mut curiosity_metrics = Vec::new();

        // Average curiosity intensity
        let avg_intensity: f64 =
            intuition_state.curiosity_drivers.iter().map(|driver| driver.intensity).sum::<f64>()
                / intuition_state.curiosity_drivers.len() as f64;
        curiosity_metrics.push(avg_intensity);

        // Curiosity diversity (more diverse curiosity is generally better)
        let unique_sources: std::collections::HashSet<_> =
            intuition_state.curiosity_drivers.iter().map(|driver| &driver.source).collect();
        let source_diversity =
            unique_sources.len() as f64 / intuition_state.curiosity_drivers.len() as f64;
        curiosity_metrics.push(source_diversity);

        // Direction richness (more exploration directions indicate broader curiosity)
        let total_directions: usize =
            intuition_state.curiosity_drivers.iter().map(|driver| driver.direction.len()).sum();
        let avg_directions =
            total_directions as f64 / intuition_state.curiosity_drivers.len() as f64;
        let direction_richness = (avg_directions / 5.0).min(1.0); // Normalize to 5 directions max
        curiosity_metrics.push(direction_richness);

        // Peak curiosity factor (having at least one very high curiosity driver is
        // valuable)
        let max_intensity = intuition_state
            .curiosity_drivers
            .iter()
            .map(|driver| driver.intensity)
            .fold(0.0f64, |acc, x| acc.max(x));
        curiosity_metrics.push(max_intensity);

        // Weighted combination
        let weights = [0.4, 0.2, 0.2, 0.2]; // Intensity is most important
        let weighted_score = curiosity_metrics
            .iter()
            .zip(weights.iter())
            .map(|(metric, weight)| metric * weight)
            .sum::<f64>();

        Ok(weighted_score.clamp(0.0, 1.0))
    }

    async fn calculate_novelty_gradient(&self, intuition_state: &IntuitionState) -> Result<f64> {
        if intuition_state.emergent_ideas.is_empty() {
            return Ok(0.1); // Low novelty without ideas
        }

        let mut novelty_metrics = Vec::new();

        // Average novelty of all ideas
        let avg_novelty: f64 =
            intuition_state.emergent_ideas.iter().map(|idea| idea.novelty).sum::<f64>()
                / intuition_state.emergent_ideas.len() as f64;
        novelty_metrics.push(avg_novelty);

        // Novelty variance (some variation in novelty levels can be beneficial)
        let novelty_scores: Vec<f64> =
            intuition_state.emergent_ideas.iter().map(|idea| idea.novelty).collect();

        if novelty_scores.len() > 1 {
            let variance = self.calculate_variance(&novelty_scores);
            let diversity_bonus = (variance * 2.0).min(0.3); // Moderate variance is good
            novelty_metrics.push(avg_novelty + diversity_bonus);
        }

        // Peak novelty (most novel idea)
        let max_novelty = intuition_state
            .emergent_ideas
            .iter()
            .map(|idea| idea.novelty)
            .fold(0.0f64, |acc, x| acc.max(x));
        novelty_metrics.push(max_novelty);

        // Feasible novelty (novel ideas that are also feasible)
        let feasible_novelty: f64 = intuition_state
            .emergent_ideas
            .iter()
            .map(|idea| idea.novelty * idea.feasibility)
            .sum::<f64>()
            / intuition_state.emergent_ideas.len() as f64;
        novelty_metrics.push(feasible_novelty);

        // Calculate weighted average
        let weights = [0.3, 0.2, 0.2, 0.3]; // Balance average, diversity, peak, and feasible novelty
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (i, metric) in novelty_metrics.iter().enumerate() {
            if i < weights.len() {
                weighted_sum += metric * weights[i];
                total_weight += weights[i];
            }
        }

        if total_weight > 0.0 {
            Ok((weighted_sum / total_weight).clamp(0.0, 1.0))
        } else {
            Ok(avg_novelty)
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::memory::MemoryConfig;

    #[tokio::test]
    async fn test_three_gradient_coordinator_creation() {
        let temp_dir = tempdir().unwrap();
        let testconfig =
            MemoryConfig { persistence_path: temp_dir.path().to_path_buf(), ..Default::default() };

        let memory = Arc::new(CognitiveMemory::new(testconfig).await.unwrap());
        let value_gradient = ValueGradient::new(memory.clone()).await.unwrap();

        let coordinator =
            ThreeGradientCoordinator::new(value_gradient, memory, None).await.unwrap();

        let state = coordinator.get_current_state().await;
        assert!(state.total_magnitude >= 0.0);
        assert!(state.gradient_coherence >= 0.0 && state.gradient_coherence <= 1.0);
    }

    #[tokio::test]
    async fn test_gradient_vector_alignment() {
        let temp_dir = tempdir().unwrap();
        let testconfig =
            MemoryConfig { persistence_path: temp_dir.path().to_path_buf(), ..Default::default() };

        let memory = Arc::new(CognitiveMemory::new(testconfig).await.unwrap());
        let value_gradient = ValueGradient::new(memory.clone()).await.unwrap();

        let coordinator =
            ThreeGradientCoordinator::new(value_gradient, memory, None).await.unwrap();

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0]; // Same direction
        let vec3 = vec![-1.0, 0.0, 0.0]; // Opposite direction

        let alignment_same = coordinator.calculate_vector_alignment(&vec1, &vec2);
        let alignment_opposite = coordinator.calculate_vector_alignment(&vec1, &vec3);

        assert!((alignment_same - 1.0).abs() < 0.01); // Should be close to 1
        assert!((alignment_opposite + 1.0).abs() < 0.01); // Should be close to -1
    }
}
