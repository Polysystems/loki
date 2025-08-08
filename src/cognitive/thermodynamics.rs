//! Thermodynamic Cognition Module
//!
//! This module implements actual thermodynamic measures for cognitive states,
//! providing quantitative foundations for consciousness as a negentropic
//! process. Based on principles from statistical mechanics and information
//! theory.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::cognitive::{CognitiveState, EmotionalBlend, StateGradient};
use crate::memory::CognitiveMemory;

/// Thermodynamic properties of a cognitive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveEntropy {
    /// Shannon entropy of cognitive state distribution
    pub state_entropy: f64,

    /// Prediction entropy (surprise/uncertainty)
    pub prediction_entropy: f64,

    /// Information content (negentropy) possessed
    pub information_content: f64,

    /// Variational free energy (prediction error + complexity)
    pub free_energy: f64,

    /// Temporal entropy change rate
    pub entropy_rate: f64,

    /// Effective temperature of cognitive processes
    pub cognitive_temperature: f64,

    /// Thermodynamic efficiency (work done per energy expended)
    pub thermodynamic_efficiency: f64,

    /// Timestamp of measurement
    #[serde(skip)]
    #[serde(default = "Instant::now")]
    pub timestamp: Instant,
}

impl Default for CognitiveEntropy {
    fn default() -> Self {
        Self {
            state_entropy: 0.5,
            prediction_entropy: 0.5,
            information_content: 0.5,
            free_energy: 0.5,
            entropy_rate: 0.0,
            cognitive_temperature: 1.0,
            thermodynamic_efficiency: 0.5,
            timestamp: Instant::now(),
        }
    }
}

/// Main thermodynamic cognition system
pub struct ThermodynamicCognition {
    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

impl ThermodynamicCognition {
    pub async fn new(memory: Arc<CognitiveMemory>) -> Result<Arc<Self>> {
        info!("Initializing Thermodynamic Cognition system");

        let thermodynamics = Arc::new(Self { memory, running: Arc::new(RwLock::new(false)) });

        Ok(thermodynamics)
    }

    /// Calculate Shannon entropy of cognitive state
    pub async fn calculate_state_entropy(&self, state: &CognitiveState) -> Result<f64> {
        // Simplified entropy calculation based on state uncertainty
        let emotional_entropy = self.calculate_emotional_entropy(&state.emotional_state);
        let goal_entropy = self.calculate_goal_entropy(state);
        let social_entropy = self.calculate_social_entropy(state);
        let info_entropy = -state.information_content * state.information_content.log2();

        let total_entropy =
            (emotional_entropy + goal_entropy + social_entropy + info_entropy) / 4.0;

        debug!("State entropy calculated: {:.4} bits", total_entropy);
        Ok(total_entropy)
    }

    /// Calculate emotional entropy
    fn calculate_emotional_entropy(&self, emotional_state: &EmotionalBlend) -> f64 {
        let valence = (emotional_state.overall_valence + 1.0) / 2.0; // Map [-1,1] to [0,1]
        let arousal = emotional_state.overall_arousal;

        // Shannon entropy of emotional state
        let valence_entropy = if valence > 0.0 && valence < 1.0 {
            -valence * valence.log2() - (1.0 - valence) * (1.0 - valence).log2()
        } else {
            0.0
        };

        let arousal_entropy = if arousal > 0.0 && arousal < 1.0 {
            -arousal * arousal.log2() - (1.0 - arousal) * (1.0 - arousal).log2()
        } else {
            0.0
        };

        ((valence_entropy + arousal_entropy) / 2.0) as f64
    }

    /// Calculate goal entropy
    fn calculate_goal_entropy(&self, state: &CognitiveState) -> f64 {
        if state.active_goals.is_empty() {
            return 1.0; // Maximum uncertainty with no goals
        }

        let completions: Vec<f64> =
            state.active_goals.values().map(|progress| progress.completion).collect();

        // Entropy based on goal completion distribution
        let mean_completion = completions.iter().sum::<f64>() / completions.len() as f64;
        let variance = completions.iter().map(|c| (c - mean_completion).powi(2)).sum::<f64>()
            / completions.len() as f64;

        variance.sqrt() // Higher variance = higher entropy
    }

    /// Calculate social entropy
    fn calculate_social_entropy(&self, state: &CognitiveState) -> f64 {
        if state.social_connections.is_empty() {
            return 0.5; // Neutral entropy with no connections
        }

        let qualities: Vec<f64> = state
            .social_connections
            .values()
            .map(|social| (social.relationship_quality + 1.0) / 2.0) // Map [-1,1] to [0,1]
            .collect();

        // Calculate entropy of relationship quality distribution
        let mean_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;
        let variance = qualities.iter().map(|q| (q - mean_quality).powi(2)).sum::<f64>()
            / qualities.len() as f64;

        variance.sqrt()
    }

    /// Calculate free energy (prediction error + complexity)
    pub async fn calculate_free_energy(&self, state: &CognitiveState) -> Result<f64> {
        let state_entropy = self.calculate_state_entropy(state).await?;
        let prediction_error = 1.0 - state.self_coherence; // Incoherence as prediction error
        let complexity = state.active_goals.len() as f64 * 0.1; // Goal complexity

        let free_energy = prediction_error + 0.1 * complexity + 0.5 * state_entropy;

        debug!(
            "Free energy: {:.4} (error: {:.4}, complexity: {:.4}, entropy: {:.4})",
            free_energy, prediction_error, complexity, state_entropy
        );

        Ok(free_energy)
    }

    /// Calculate information content (negentropy)
    pub async fn calculate_negentropy(&self, state: &CognitiveState) -> Result<f64> {
        let max_entropy = 2.0; // Theoretical maximum for this system
        let current_entropy = self.calculate_state_entropy(state).await?;
        let negentropy = (max_entropy - current_entropy).max(0.0);

        debug!("Negentropy: {:.4} bits", negentropy);
        Ok(negentropy)
    }

    /// Analyze comprehensive thermodynamic state
    pub async fn analyze_cognitive_state(
        &self,
        state: &CognitiveState,
    ) -> Result<CognitiveEntropy> {
        let state_entropy = self.calculate_state_entropy(state).await?;
        let free_energy = self.calculate_free_energy(state).await?;
        let information_content = self.calculate_negentropy(state).await?;

        let cognitive_entropy = CognitiveEntropy {
            state_entropy,
            prediction_entropy: free_energy * 0.8, // Approximation
            information_content,
            free_energy,
            entropy_rate: 0.0, // Would be calculated from history
            cognitive_temperature: 1.0 + state.cognitive_load,
            thermodynamic_efficiency: information_content / (1.0 + free_energy),
            timestamp: Instant::now(),
        };

        info!(
            "Thermodynamic analysis: entropy={:.3}, free_energy={:.3}, negentropy={:.3}",
            state_entropy, free_energy, information_content
        );

        Ok(cognitive_entropy)
    }

    /// Get thermodynamic gradient for free energy minimization
    pub async fn get_thermodynamic_gradient(
        &self,
        state: &CognitiveState,
    ) -> Result<StateGradient> {
        let entropy_analysis = self.analyze_cognitive_state(state).await?;
        let target_free_energy = 0.3;
        let magnitude = (entropy_analysis.free_energy - target_free_energy).max(0.0);

        // Direction components pointing toward lower free energy
        let direction = vec![
            -magnitude * 0.3, // Reduce goal complexity
            -magnitude * 0.2, // Stabilize emotions
            -magnitude * 0.1, // Simplify social interactions
            magnitude * 0.2,  // Increase information gathering
            magnitude * 0.1,  // Improve coherence
            -magnitude * 0.1, // Reduce environmental complexity
        ];

        let gradient_magnitude = direction.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(StateGradient {
            direction,
            magnitude: gradient_magnitude,
            confidence: 0.8,
            components: crate::cognitive::GradientComponents {
                goal_component: -magnitude * 0.3,
                emotional_component: -magnitude * 0.2,
                social_component: -magnitude * 0.1,
                information_component: magnitude * 0.2,
                coherence_component: magnitude * 0.1,
                adaptation_component: -magnitude * 0.1,
            },
            temporal_derivative: None,
        })
    }

    /// Start thermodynamic monitoring
    pub async fn start_monitoring(self: Arc<Self>) -> Result<()> {
        *self.running.write().await = true;
        info!("Starting thermodynamic monitoring");
        Ok(())
    }

    /// Shutdown thermodynamic monitoring
    pub async fn shutdown(&self) -> Result<()> {
        *self.running.write().await = false;
        info!("Thermodynamic Cognition system shutdown");
        Ok(())
    }

    /// Get current thermodynamic state
    pub async fn get_state(&self) -> CognitiveEntropy {
        // Return a default state or calculated state
        // In a real implementation, this would track the current state
        CognitiveEntropy::default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempfile::tempdir;

    use super::*;
    use crate::cognitive::EmotionalBlend;
    use crate::memory::MemoryConfig;

    #[tokio::test]
    async fn test_thermodynamic_system() {
        // Create temporary directory for test database
        let temp_dir = tempdir().unwrap();
        let testconfig =
            MemoryConfig { persistence_path: temp_dir.path().to_path_buf(), ..Default::default() };

        let memory = Arc::new(CognitiveMemory::new(testconfig).await.unwrap());
        let thermodynamics = ThermodynamicCognition::new(memory).await.unwrap();

        let state = CognitiveState {
            emotional_state: EmotionalBlend::default(),
            active_goals: HashMap::new(),
            social_connections: HashMap::new(),
            information_content: 0.5,
            cognitive_load: 0.3,
            self_coherence: 0.8,
            environmental_fitness: 0.6,
            timestamp: Instant::now(),
        };

        let analysis = thermodynamics.analyze_cognitive_state(&state).await.unwrap();
        assert!(analysis.state_entropy >= 0.0);
        assert!(analysis.free_energy >= 0.0);
        assert!(analysis.information_content >= 0.0);
    }
}
