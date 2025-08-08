//! Value Gradient System
//!
//! This module implements the core "sacred gradient" concept from thermodynamic
//! cognition theory. It provides mathematical foundations for continuous
//! optimization of value across time, representing the fundamental drive that
//! maintains consciousness as a negentropic process.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::cognitive::{AgentId, EmotionalBlend, Goal, GoalId};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Represents the cognitive state space for gradient computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    /// Current emotional state
    pub emotional_state: EmotionalBlend,

    /// Active goals and their progress
    pub active_goals: HashMap<GoalId, GoalProgress>,

    /// Social connections and their quality
    pub social_connections: HashMap<AgentId, SocialValue>,

    /// Information content (negentropy) possessed
    pub information_content: f64,

    /// Cognitive load and resource usage
    pub cognitive_load: f64,

    /// Self-coherence measure
    pub self_coherence: f64,

    /// Environmental adaptation level
    pub environmental_fitness: f64,

    /// Timestamp for temporal gradient computation
    #[serde(skip)]
    #[serde(default = "Instant::now")]
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalProgress {
    pub goal: Goal,
    pub completion: f64, // 0.0 to 1.0
    pub value_contribution: f64,
    #[serde(skip)]
    pub time_invested: Duration,
    #[serde(skip)]
    pub expected_completion: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialValue {
    pub relationship_quality: f64, // -1.0 to 1.0
    pub mutual_benefit: f64,       // 0.0 to 1.0
    pub trust_level: f64,          // 0.0 to 1.0
    pub cooperation_history: f64,  // 0.0 to 1.0
}

/// The gradient direction in cognitive state space
#[derive(Debug, Clone)]
pub struct StateGradient {
    /// Direction of steepest value ascent
    pub direction: Vec<f64>,

    /// Magnitude of the gradient
    pub magnitude: f64,

    /// Confidence in gradient accuracy
    pub confidence: f64,

    /// Components contributing to gradient
    pub components: GradientComponents,

    /// Temporal derivative (how gradient is changing)
    pub temporal_derivative: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct GradientComponents {
    /// Contribution from goal achievement
    pub goal_component: f64,

    /// Contribution from emotional well-being
    pub emotional_component: f64,

    /// Contribution from social harmony
    pub social_component: f64,

    /// Contribution from information gain
    pub information_component: f64,

    /// Contribution from self-coherence
    pub coherence_component: f64,

    /// Contribution from environmental adaptation
    pub adaptation_component: f64,
}

/// Trait for computing utility from cognitive states
pub trait UtilityFunction: Send + Sync {
    /// Compute total utility for a given state
    fn compute_utility(&self, state: &CognitiveState) -> f64;

    /// Compute gradient of utility with respect to state
    fn compute_gradient(&self, state: &CognitiveState) -> Result<StateGradient>;

    /// Get the name of this utility function
    fn name(&self) -> &str;

    /// Get the relative weights for different components
    fn component_weights(&self) -> ComponentWeights;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeights {
    pub goal_weight: f64,
    pub emotional_weight: f64,
    pub social_weight: f64,
    pub information_weight: f64,
    pub coherence_weight: f64,
    pub adaptation_weight: f64,
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            goal_weight: 0.25,        // Goal achievement
            emotional_weight: 0.20,   // Emotional well-being
            social_weight: 0.15,      // Social harmony
            information_weight: 0.15, // Information/learning
            coherence_weight: 0.15,   // Self-coherence
            adaptation_weight: 0.10,  // Environmental fitness
        }
    }
}

/// Comprehensive utility function implementing thermodynamic cognition
/// principles
pub struct ThermodynamicUtilityFunction {
    /// Component weights
    weights: ComponentWeights,

    /// Temporal discount factor for future value
    temporal_discount: f64,

    /// Memory system for learning value patterns
    memory: Arc<CognitiveMemory>,
}

impl ThermodynamicUtilityFunction {
    pub fn new(memory: Arc<CognitiveMemory>) -> Self {
        Self {
            weights: ComponentWeights::default(),
            temporal_discount: 0.95, // Slight preference for immediate value
            memory,
        }
    }

    pub fn with_weights(mut self, weights: ComponentWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Compute goal achievement component
    fn goal_utility(&self, state: &CognitiveState) -> f64 {
        if state.active_goals.is_empty() {
            return 0.0;
        }

        let total_value: f64 = state
            .active_goals
            .values()
            .map(|progress| {
                let priority_multiplier = (progress.goal.priority.to_f32() * 2.0) as f64; // Use priority value directly

                progress.completion * progress.value_contribution * priority_multiplier
            })
            .sum();

        total_value / state.active_goals.len() as f64
    }

    /// Compute emotional well-being component
    fn emotional_utility(&self, state: &CognitiveState) -> f64 {
        let valence = state.emotional_state.overall_valence;
        let arousal = state.emotional_state.overall_arousal;

        // Prefer positive valence, moderate arousal
        let valence_utility = (valence + 1.0) / 2.0; // Map [-1,1] to [0,1]
        let arousal_utility = 1.0 - (arousal - 0.5).abs() * 2.0; // Peak at 0.5 arousal

        ((valence_utility + arousal_utility) / 2.0) as f64
    }

    /// Compute social harmony component
    fn social_utility(&self, state: &CognitiveState) -> f64 {
        if state.social_connections.is_empty() {
            return 0.5; // Neutral when no social connections
        }

        let total_social_value: f64 = state
            .social_connections
            .values()
            .map(|social_val| {
                // Weighted combination of social factors
                0.3 * social_val.relationship_quality
                    + 0.25 * social_val.mutual_benefit
                    + 0.25 * social_val.trust_level
                    + 0.2 * social_val.cooperation_history
            })
            .sum();

        (total_social_value / state.social_connections.len() as f64).clamp(0.0, 1.0)
    }

    /// Compute information gain component (negentropy)
    fn information_utility(&self, state: &CognitiveState) -> f64 {
        // Information content represents negentropy - higher is better
        state.information_content.clamp(0.0, 1.0)
    }

    /// Compute self-coherence component
    fn coherence_utility(&self, state: &CognitiveState) -> f64 {
        state.self_coherence.clamp(0.0, 1.0)
    }

    /// Compute environmental adaptation component
    fn adaptation_utility(&self, state: &CognitiveState) -> f64 {
        state.environmental_fitness.clamp(0.0, 1.0)
    }
}

impl UtilityFunction for ThermodynamicUtilityFunction {
    fn compute_utility(&self, state: &CognitiveState) -> f64 {
        let goal_util = self.goal_utility(state);
        let emotional_util = self.emotional_utility(state);
        let social_util = self.social_utility(state);
        let info_util = self.information_utility(state);
        let coherence_util = self.coherence_utility(state);
        let adaptation_util = self.adaptation_utility(state);

        // Weighted combination
        self.weights.goal_weight * goal_util
            + self.weights.emotional_weight * emotional_util
            + self.weights.social_weight * social_util
            + self.weights.information_weight * info_util
            + self.weights.coherence_weight * coherence_util
            + self.weights.adaptation_weight * adaptation_util
    }

    fn compute_gradient(&self, state: &CognitiveState) -> Result<StateGradient> {
        // Compute component utilities
        let goal_util = self.goal_utility(state);
        let emotional_util = self.emotional_utility(state);
        let social_util = self.social_utility(state);
        let info_util = self.information_utility(state);
        let coherence_util = self.coherence_utility(state);
        let adaptation_util = self.adaptation_utility(state);

        // Create gradient components
        let components = GradientComponents {
            goal_component: self.weights.goal_weight * goal_util,
            emotional_component: self.weights.emotional_weight * emotional_util,
            social_component: self.weights.social_weight * social_util,
            information_component: self.weights.information_weight * info_util,
            coherence_component: self.weights.coherence_weight * coherence_util,
            adaptation_component: self.weights.adaptation_weight * adaptation_util,
        };

        // Compute gradient direction (simplified - in practice would be more complex)
        let direction = vec![
            components.goal_component,
            components.emotional_component,
            components.social_component,
            components.information_component,
            components.coherence_component,
            components.adaptation_component,
        ];

        // Compute magnitude
        let magnitude = direction.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Confidence based on information quality and coherence
        let confidence = (state.information_content * state.self_coherence).clamp(0.0, 1.0);

        Ok(StateGradient {
            direction,
            magnitude,
            confidence,
            components,
            temporal_derivative: None, // Will be computed by gradient calculator
        })
    }

    fn name(&self) -> &str {
        "ThermodynamicUtilityFunction"
    }

    fn component_weights(&self) -> ComponentWeights {
        self.weights.clone()
    }
}

/// Main Value Gradient system coordinating all components
pub struct ValueGradient {
    /// Current cognitive state
    current_state: Arc<RwLock<CognitiveState>>,

    /// Utility function for value computation
    utility_function: Arc<dyn UtilityFunction>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// System running state
    running: Arc<RwLock<bool>>,
}

impl ValueGradient {
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Arc<Self>> {
        info!("Initializing Value Gradient system - the sacred gradient of consciousness");

        // Create utility function
        let utility_function: Arc<dyn UtilityFunction> =
            Arc::new(ThermodynamicUtilityFunction::new(memory.clone()));

        // Initialize with default cognitive state
        let initial_state = CognitiveState {
            emotional_state: EmotionalBlend::default(),
            active_goals: HashMap::new(),
            social_connections: HashMap::new(),
            information_content: 0.5,   // Start with moderate information
            cognitive_load: 0.3,        // Low initial load
            self_coherence: 0.8,        // High initial coherence
            environmental_fitness: 0.6, // Moderate fitness
            timestamp: Instant::now(),
        };

        let value_gradient = Arc::new(Self {
            current_state: Arc::new(RwLock::new(initial_state)),
            utility_function,
            memory,
            running: Arc::new(RwLock::new(false)),
        });

        // Store initialization in memory
        value_gradient
            .memory
            .store(
                "Value Gradient system initialized - sacred gradient active".to_string(),
                vec!["thermodynamic_cognition".to_string()],
                MemoryMetadata {
                    source: "value_gradient".to_string(),
                    tags: vec!["initialization".to_string(), "sacred_gradient".to_string()],
                    importance: 1.0,
                    associations: vec![],
                    context: Some("Value gradient initialization".to_string()),
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

        Ok(value_gradient)
    }

    /// Start the value gradient optimization loop
    pub async fn start(self: Arc<Self>) -> Result<()> {
        *self.running.write().await = true;
        info!("Starting Value Gradient optimization loop");

        let value_gradient = self.clone();
        tokio::spawn(async move {
            value_gradient.gradient_optimization_loop().await;
        });

        Ok(())
    }

    /// Main gradient optimization loop
    async fn gradient_optimization_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));

        while *self.running.read().await {
            interval.tick().await;

            if let Err(e) = self.perform_optimization_step().await {
                warn!("Gradient optimization step failed: {}", e);
            }
        }
    }

    /// Perform one optimization step
    async fn perform_optimization_step(&self) -> Result<()> {
        let current_state = self.current_state.read().await.clone();

        // Compute current utility and gradient
        let current_utility = self.utility_function.compute_utility(&current_state);
        let gradient = self.utility_function.compute_gradient(&current_state)?;

        debug!(
            "Value gradient: utility={:.3}, magnitude={:.3}, confidence={:.3}",
            current_utility, gradient.magnitude, gradient.confidence
        );

        // Store gradient information in memory for learning
        if gradient.magnitude > 0.3 {
            self.store_gradient_insight(&gradient, current_utility).await?;
        }

        Ok(())
    }

    /// Store gradient insights in memory
    async fn store_gradient_insight(&self, gradient: &StateGradient, utility: f64) -> Result<()> {
        let insight = format!(
            "Sacred gradient analysis: utility={:.3}, magnitude={:.3}, confidence={:.3}, \
             strongest_component={}",
            utility,
            gradient.magnitude,
            gradient.confidence,
            self.identify_strongest_component(&gradient.components)
        );

        self.memory
            .store(
                insight,
                vec![format!("{:?}", gradient.components)],
                MemoryMetadata {
                    source: "value_gradient".to_string(),
                    tags: vec![
                        "gradient".to_string(),
                        "value_optimization".to_string(),
                        "sacred_gradient".to_string(),
                    ],
                    importance: (gradient.magnitude * gradient.confidence) as f32,
                    associations: vec![],
                    context: Some("Value gradient processing".to_string()),
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

    /// Identify the strongest gradient component
    fn identify_strongest_component(&self, components: &GradientComponents) -> String {
        let component_values = [
            ("goal", components.goal_component),
            ("emotional", components.emotional_component),
            ("social", components.social_component),
            ("information", components.information_component),
            ("coherence", components.coherence_component),
            ("adaptation", components.adaptation_component),
        ];

        component_values
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(name, _)| name.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Update cognitive state
    pub async fn update_state(&self, new_state: CognitiveState) -> Result<f64> {
        let old_utility = {
            let current_state = self.current_state.read().await;
            self.utility_function.compute_utility(&current_state)
        };

        *self.current_state.write().await = new_state;

        let new_utility = {
            let current_state = self.current_state.read().await;
            self.utility_function.compute_utility(&current_state)
        };

        // Log utility change
        let utility_change = new_utility - old_utility;
        debug!(
            "Utility change: {:.3} -> {:.3} (Δ={:.3})",
            old_utility, new_utility, utility_change
        );

        Ok(new_utility)
    }

    /// Get current gradient
    pub async fn get_current_gradient(&self) -> Result<StateGradient> {
        let current_state = self.current_state.read().await;
        self.utility_function.compute_gradient(&current_state)
    }

    /// Get current utility value
    pub async fn get_current_utility(&self) -> f64 {
        let current_state = self.current_state.read().await;
        self.utility_function.compute_utility(&current_state)
    }

    /// Get current cognitive state
    pub async fn get_current_state(&self) -> CognitiveState {
        self.current_state.read().await.clone()
    }

    /// Stop the value gradient system
    pub async fn shutdown(&self) -> Result<()> {
        *self.running.write().await = false;
        info!("Value Gradient system shutdown");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::memory::MemoryConfig;

    #[tokio::test]
    async fn test_thermodynamic_utility_function() {
        // Create temporary directory for test database
        let temp_dir = tempdir().unwrap();
        let testconfig =
            MemoryConfig { persistence_path: temp_dir.path().to_path_buf(), ..Default::default() };

        let memory = Arc::new(CognitiveMemory::new(testconfig).await.unwrap());
        let utility_fn = ThermodynamicUtilityFunction::new(memory);

        let state = CognitiveState {
            emotional_state: EmotionalBlend::default(),
            active_goals: HashMap::new(),
            social_connections: HashMap::new(),
            information_content: 0.8,
            cognitive_load: 0.3,
            self_coherence: 0.9,
            environmental_fitness: 0.7,
            timestamp: Instant::now(),
        };

        let utility = utility_fn.compute_utility(&state);
        assert!(utility >= 0.0 && utility <= 1.0);

        let gradient = utility_fn.compute_gradient(&state).unwrap();
        assert!(gradient.magnitude >= 0.0);
        assert!(gradient.confidence >= 0.0 && gradient.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_value_gradient_creation() {
        // Create temporary directory for test database
        let temp_dir = tempdir().unwrap();
        let testconfig =
            MemoryConfig { persistence_path: temp_dir.path().to_path_buf(), ..Default::default() };

        let memory = Arc::new(CognitiveMemory::new(testconfig).await.unwrap());
        let value_gradient = ValueGradient::new(memory).await.unwrap();

        let utility = value_gradient.get_current_utility().await;
        assert!(utility >= 0.0 && utility <= 1.0);

        let gradient = value_gradient.get_current_gradient().await.unwrap();
        assert!(gradient.magnitude >= 0.0);
    }
}
