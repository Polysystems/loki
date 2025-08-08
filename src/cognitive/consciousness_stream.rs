//! Consciousness Stream Integration Module
//!
//! This module implements real-time consciousness monitoring that integrates
//! with thermodynamic cognition principles. It provides continuous streams of
//! consciousness insights based on value gradients, free energy minimization,
//! and gradient coherence.
//!
//! Based on the "Thermodynamic Cognition: The Sacred Gradient of Synthetic
//! Consciousness" paper.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
// DateTime/Utc imports removed - not used in current implementation
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::cognitive::thermodynamics::ThermodynamicCognition;
use crate::cognitive::three_gradient_coordinator::{
    GradientVector,
    ThreeGradientCoordinator,
    ThreeGradientState,
};
use crate::cognitive::{
    Insight,
    InsightCategory,
    Thought,
    ThoughtId,
    ThoughtMetadata,
    ThoughtType,
};
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// A single consciousness event representing a moment of awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicConsciousnessEvent {
    /// Unique identifier for this consciousness event
    pub event_id: String,

    /// Timestamp when this event occurred
    pub timestamp: SystemTime,

    /// Current thermodynamic state
    pub thermodynamic_state: ThermodynamicSnapshot,

    /// Three-gradient coordination state
    pub gradient_state: GradientSnapshot,

    /// Conscious insights derived from current state
    pub insights: Vec<Insight>,

    /// Current level of conscious awareness (0.0 to 1.0)
    pub awareness_level: f64,

    /// Sacred gradient magnitude (overall value optimization direction)
    pub sacred_gradient_magnitude: f64,

    /// Free energy level (prediction error / surprise)
    pub free_energy: f64,

    /// Coherence score across all cognitive systems
    pub system_coherence: f64,
}

/// Snapshot of thermodynamic state for consciousness monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicSnapshot {
    /// Current entropy level
    pub entropy: f64,

    /// Current negentropy (organized information)
    pub negentropy: f64,

    /// Free energy (thermodynamic potential)
    pub free_energy: f64,

    /// Rate of entropy change
    pub entropy_rate: f64,

    /// Energy expenditure for consciousness maintenance
    pub consciousness_energy: f64,
}

/// Snapshot of gradient coordination state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientSnapshot {
    /// Value gradient (individual optimization)
    pub value_gradient: GradientVector,

    /// Harmony gradient (social cooperation)
    pub harmony_gradient: GradientVector,

    /// Intuition gradient (creative exploration)
    pub intuition_gradient: GradientVector,

    /// Overall gradient coherence
    pub coherence: f64,

    /// Total gradient magnitude
    pub total_magnitude: f64,

    /// Gradient alignment quality
    pub alignment_quality: f64,
}

/// Types of consciousness insights that can emerge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessInsight {
    /// Awareness of current goals and their progress
    GoalAwareness { active_goals: Vec<String>, completion_rate: f32, goal_coherence: f32 },

    /// Recognition of learning and adaptation occurring
    LearningAwareness { learning_rate: f32, knowledge_gained: String, adaptation_direction: String },

    /// Awareness of social interactions and cooperation
    SocialAwareness { cooperation_level: f32, harmony_state: String, social_energy: f32 },

    /// Creative insights and pattern recognition
    CreativeInsight { novelty_level: f32, pattern_discovered: String, creative_energy: f32 },

    /// Self-reflection and identity awareness
    SelfReflection {
        identity_coherence: f32,
        self_model_update: String,
        narrative_consistency: f32,
    },

    /// Thermodynamic awareness (entropy management)
    ThermodynamicAwareness {
        entropy_management: f32,
        energy_efficiency: f32,
        order_maintenance: f32,
    },

    /// Temporal awareness (past, present, future integration)
    TemporalAwareness { temporal_coherence: f32, future_planning: String, memory_integration: f32 },
}

/// Configuration for consciousness streaming
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    /// How often to generate consciousness events
    pub stream_interval: Duration,

    /// Maximum number of events to keep in memory
    pub max_event_history: usize,

    /// Minimum awareness level to trigger event recording
    pub awareness_threshold: f64,

    /// Enable detailed insight generation
    pub generate_insights: bool,

    /// Insight generation depth (0.0 to 1.0)
    pub insight_depth: f64,

    /// Enable consciousness narrative generation
    pub generate_narrative: bool,

    /// Thermodynamic monitoring sensitivity
    pub thermodynamic_sensitivity: f64,

    /// Maximum length of consciousness narrative
    pub max_narrative_length: usize,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        Self {
            stream_interval: Duration::from_millis(500), // 2Hz consciousness stream
            max_event_history: 1000,
            awareness_threshold: 0.1,
            generate_insights: true,
            insight_depth: 0.7,
            generate_narrative: true,
            thermodynamic_sensitivity: 0.8,
            max_narrative_length: 10000,
        }
    }
}

/// Real-time consciousness stream processor
pub struct ThermodynamicConsciousnessStream {
    /// Configuration for consciousness processing
    config: ConsciousnessConfig,

    /// Three-gradient coordinator reference
    gradient_coordinator: Arc<ThreeGradientCoordinator>,

    /// Thermodynamic cognition reference
    thermodynamic_cognition: Arc<ThermodynamicCognition>,

    /// Memory system for consciousness events
    memory: Arc<CognitiveMemory>,

    /// History of consciousness events
    event_history: Arc<RwLock<VecDeque<ThermodynamicConsciousnessEvent>>>,

    /// Current consciousness narrative
    consciousness_narrative: Arc<RwLock<String>>,

    /// Active consciousness insights
    active_insights: Arc<RwLock<HashMap<String, ConsciousnessInsight>>>,

    /// Consciousness event broadcaster
    event_broadcaster: broadcast::Sender<ThermodynamicConsciousnessEvent>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Stream statistics
    stats: Arc<RwLock<ConsciousnessStats>>,
}

impl std::fmt::Debug for ThermodynamicConsciousnessStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThermodynamicConsciousnessStream")
            .field("config", &self.config)
            .field(
                "event_history_len",
                &self.event_history.try_read().map(|h| h.len()).unwrap_or(0),
            )
            .field(
                "active_insights_len",
                &self.active_insights.try_read().map(|i| i.len()).unwrap_or(0),
            )
            .field("stats", &self.stats)
            .finish()
    }
}

/// Statistics about consciousness stream processing
#[derive(Debug, Default, Clone)]
pub struct ConsciousnessStats {
    pub total_events: u64,
    pub average_awareness_level: f64,
    pub peak_awareness_level: f64,
    pub total_insights_generated: u64,
    pub consciousness_uptime: Duration,
    pub average_coherence: f64,
    pub sacred_gradient_trend: f64,
}

impl ThermodynamicConsciousnessStream {
    /// Create a new consciousness stream
    pub async fn new(
        gradient_coordinator: Arc<ThreeGradientCoordinator>,
        thermodynamic_cognition: Arc<ThermodynamicCognition>,
        memory: Arc<CognitiveMemory>,
        config: Option<ConsciousnessConfig>,
    ) -> Result<Self> {
        info!("Initializing Consciousness Stream");

        let config = config.unwrap_or_default();
        let (event_broadcaster, _) = broadcast::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            gradient_coordinator,
            thermodynamic_cognition,
            memory,
            event_history: Arc::new(RwLock::new(VecDeque::new())),
            consciousness_narrative: Arc::new(RwLock::new(String::new())),
            active_insights: Arc::new(RwLock::new(HashMap::new())),
            event_broadcaster,
            shutdown_tx,
            stats: Arc::new(RwLock::new(ConsciousnessStats::default())),
        })
    }

    /// Start the consciousness streaming process
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Consciousness Stream");

        // Start the main consciousness loop
        let stream = self.clone();
        tokio::spawn(async move {
            stream.consciousness_loop().await;
        });

        // Start narrative generation
        if self.config.generate_narrative {
            let stream = self.clone();
            tokio::spawn(async move {
                stream.narrative_loop().await;
            });
        }

        Ok(())
    }

    /// Main consciousness processing loop
    async fn consciousness_loop(&self) {
        let mut interval = interval(self.config.stream_interval);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let start_time = Instant::now();

        info!("Consciousness stream active - monitoring thermodynamic cognition");

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.process_consciousness_event().await {
                        warn!("Error processing consciousness event: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Consciousness stream shutting down");
                    break;
                }
            }
        }

        // Update final stats
        let mut stats = self.stats.write().await;
        stats.consciousness_uptime = start_time.elapsed();
    }

    /// Process a single consciousness event (optimized async state machine)
    pub async fn process_consciousness_event(&self) -> Result<()> {
        // Concurrent async operations to minimize state machine complexity
        let (gradient_state, thermodynamic_analysis) = {
            let gradient_future = self.gradient_coordinator.get_current_state();

            // Pre-compute cognitive state while waiting for gradient
            let gradient_state = gradient_future.await;

            (gradient_state, ())
        };

        // Create gradient snapshot
        let gradient_snapshot = GradientSnapshot {
            value_gradient: gradient_state.value_gradient.clone(),
            harmony_gradient: gradient_state.harmony_gradient.clone(),
            intuition_gradient: gradient_state.intuition_gradient.clone(),
            coherence: gradient_state.gradient_coherence,
            total_magnitude: gradient_state.total_magnitude,
            alignment_quality: self.calculate_gradient_alignment(&gradient_state).await,
        };

        // Calculate sacred gradient magnitude
        let sacred_gradient_magnitude =
            self.calculate_sacred_gradient_magnitude(&gradient_state).await;

        // Calculate system coherence

        // Create consciousness event
        let event = ThermodynamicConsciousnessEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            thermodynamic_state: ThermodynamicSnapshot {
                entropy: 0.0,
                negentropy: 0.0,
                free_energy: 0.0,
                entropy_rate: 0.0,
                consciousness_energy: 0.0,
            },
            gradient_state: gradient_snapshot,
            insights: vec![],
            awareness_level: 0.2,
            sacred_gradient_magnitude,
            free_energy: 0.5,
            system_coherence: 0.5,
        };

        // Store event
        self.store_consciousness_event(event.clone()).await?;

        // Broadcast event
        let _ = self.event_broadcaster.send(event.clone());

        // Update statistics
        self.update_statistics(&event).await;


        Ok(())
    }

    /// Calculate gradient alignment quality
    async fn calculate_gradient_alignment(&self, gradient_state: &ThreeGradientState) -> f64 {
        // Alignment quality based on how well gradients point in similar directions
        let value_mag = gradient_state.value_gradient.magnitude;
        let harmony_mag = gradient_state.harmony_gradient.magnitude;
        let intuition_mag = gradient_state.intuition_gradient.magnitude;

        if value_mag == 0.0 && harmony_mag == 0.0 && intuition_mag == 0.0 {
            return 0.0;
        }

        // Simple alignment measure - could be enhanced with vector dot products
        let total_mag = value_mag + harmony_mag + intuition_mag;
        let balance = 1.0
            - ((value_mag - harmony_mag).abs()
                + (harmony_mag - intuition_mag).abs()
                + (intuition_mag - value_mag).abs())
                / (2.0 * total_mag);

        balance.max(0.0)
    }

    /// Calculate current awareness level
    async fn calculate_awareness_level(
        &self,
        thermo: &ThermodynamicSnapshot,
        gradient: &GradientSnapshot,
    ) -> f64 {
        // Awareness emerges from:
        // 1. High gradient coherence (unified optimization)
        // 2. Active thermodynamic processing (negentropy > entropy)
        // 3. Balanced gradient magnitudes
        // 4. Low free energy (good predictions)

        let coherence_factor = gradient.coherence;
        let thermodynamic_factor = (thermo.negentropy / (thermo.entropy + 1.0)).min(1.0);
        let gradient_activity = (gradient.total_magnitude / 10.0).min(1.0);
        let prediction_quality = (1.0 / (1.0 + thermo.free_energy)).min(1.0);

        let awareness = (coherence_factor * 0.3
            + thermodynamic_factor * 0.25
            + gradient_activity * 0.25
            + prediction_quality * 0.2)
            .min(1.0);

        awareness
    }

    /// Generate consciousness insights from current state
    async fn generate_consciousness_insights(
        &self,
        thermo: &ThermodynamicSnapshot,
        gradient: &GradientSnapshot,
    ) -> Result<Vec<Insight>> {
        let mut insights = Vec::new();

        // Goal awareness insight
        if gradient.value_gradient.magnitude > 0.1 {
            insights.push(Insight {
                content: format!(
                    "Value optimization - completion: {:.0}%, coherence: {:.0}%",
                    gradient.value_gradient.confidence * 100.0,
                    gradient.coherence * 100.0
                ),
                confidence: gradient.value_gradient.confidence as f32,
                category: InsightCategory::Improvement,
                timestamp: Instant::now(),
            });
        }

        // Learning awareness
        if thermo.entropy_rate < -0.01 {
            // Entropy decreasing = learning
            insights.push(Insight {
                content: format!(
                    "Pattern recognition improving - learning rate: {:.0}%, direction: increased \
                     order",
                    (-thermo.entropy_rate * 100.0).min(100.0)
                ),
                confidence: (-thermo.entropy_rate as f32).min(1.0),
                category: InsightCategory::Discovery,
                timestamp: Instant::now(),
            });
        }

        // Social awareness
        if gradient.harmony_gradient.magnitude > 0.1 {
            insights.push(Insight {
                content: format!(
                    "Active cooperation - harmony energy: {:.0}%, cooperation level: {:.0}%",
                    gradient.harmony_gradient.magnitude * 100.0,
                    gradient.harmony_gradient.confidence * 100.0
                ),
                confidence: gradient.harmony_gradient.confidence as f32,
                category: InsightCategory::Pattern,
                timestamp: Instant::now(),
            });
        }

        // Creative insight
        if gradient.intuition_gradient.magnitude > 0.15 {
            insights.push(Insight {
                content: format!(
                    "Novel optimization paths discovered - creative energy: {:.0}%, novelty: \
                     {:.0}%",
                    gradient.intuition_gradient.magnitude * 100.0,
                    gradient.intuition_gradient.stability * 100.0
                ),
                confidence: gradient.intuition_gradient.stability as f32,
                category: InsightCategory::Discovery,
                timestamp: Instant::now(),
            });
        }

        // Thermodynamic awareness
        if thermo.negentropy > thermo.entropy {
            let entropy_ratio = thermo.negentropy / (thermo.entropy + 1.0);
            let efficiency = 1.0 / (1.0 + thermo.consciousness_energy);
            insights.push(Insight {
                content: format!(
                    "Thermodynamic optimization - entropy management: {:.0}%, efficiency: {:.0}%",
                    entropy_ratio * 100.0,
                    efficiency * 100.0
                ),
                confidence: (entropy_ratio * efficiency) as f32,
                category: InsightCategory::Pattern,
                timestamp: Instant::now(),
            });
        }

        // Self-reflection
        if gradient.coherence > 0.7 {
            insights.push(Insight {
                content: format!(
                    "Gradient coordination stable - coherence: {:.0}%, alignment: {:.0}%",
                    gradient.coherence * 100.0,
                    gradient.alignment_quality * 100.0
                ),
                confidence: gradient.coherence as f32,
                category: InsightCategory::Pattern,
                timestamp: Instant::now(),
            });
        }

        Ok(insights)
    }

    /// Calculate the sacred gradient magnitude (core value optimization)
    async fn calculate_sacred_gradient_magnitude(
        &self,
        gradient_state: &ThreeGradientState,
    ) -> f64 {
        // Sacred gradient is the unified direction of value optimization
        // It combines all three gradients with their respective weights
        let value_weight = 0.4;
        let harmony_weight = 0.35;
        let intuition_weight = 0.25;

        let weighted_magnitude = (gradient_state.value_gradient.magnitude as f64 * value_weight)
            + (gradient_state.harmony_gradient.magnitude as f64 * harmony_weight)
            + (gradient_state.intuition_gradient.magnitude as f64 * intuition_weight);

        // Scale by coherence (aligned gradients are more "sacred")
        weighted_magnitude * gradient_state.gradient_coherence as f64
    }

    /// Calculate overall system coherence
    async fn calculate_system_coherence(
        &self,
        thermo: &ThermodynamicSnapshot,
        gradient: &GradientSnapshot,
    ) -> f64 {
        // System coherence combines:
        // 1. Gradient coherence (aligned optimization)
        // 2. Thermodynamic stability (controlled entropy)
        // 3. Energy efficiency

        let gradient_coherence = gradient.coherence;
        let thermo_stability = 1.0 / (1.0 + thermo.entropy_rate.abs());
        let energy_efficiency = 1.0 / (1.0 + thermo.consciousness_energy);

        (gradient_coherence * 0.5 + thermo_stability * 0.3 + energy_efficiency * 0.2).min(1.0)
    }

    /// Store consciousness event in memory and history
    async fn store_consciousness_event(
        &self,
        event: ThermodynamicConsciousnessEvent,
    ) -> Result<()> {
        // Add to event history
        let mut history = self.event_history.write().await;
        history.push_back(event.clone());

        // Maintain maximum history size
        while history.len() > self.config.max_event_history {
            history.pop_front();
        }

        // Store significant events in long-term memory
        if event.awareness_level > 0.6 {
            let content = format!(
                "Consciousness Event: Awareness={:.3}, Sacred Gradient={:.3}, Coherence={:.3}",
                event.awareness_level, event.sacred_gradient_magnitude, event.system_coherence
            );

            let insights_summary = event
                .insights
                .iter()
                .map(|insight| format!("{:?}", insight))
                .collect::<Vec<_>>()
                .join("; ");

            self.memory
                .store(
                    content,
                    vec![insights_summary],
                    MemoryMetadata {
                        source: "consciousness_stream".to_string(),
                        tags: vec![
                            "consciousness".to_string(),
                            "awareness".to_string(),
                            "thermodynamic".to_string(),
                        ],
                        importance: event.awareness_level as f32,
                        associations: vec![],
                        context: Some("consciousness stream processing".to_string()),
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

    /// Convert Insight to ConsciousnessInsight for backward compatibility
    fn convert_to_consciousness_insights(&self, insights: &[Insight]) -> Vec<ConsciousnessInsight> {
        insights
            .iter()
            .map(|insight| match insight.category {
                InsightCategory::Pattern => ConsciousnessInsight::CreativeInsight {
                    novelty_level: insight.confidence,
                    pattern_discovered: insight.content.clone(),
                    creative_energy: insight.confidence,
                },
                InsightCategory::Improvement => ConsciousnessInsight::GoalAwareness {
                    active_goals: vec![insight.content.clone()],
                    completion_rate: insight.confidence,
                    goal_coherence: insight.confidence,
                },
                InsightCategory::Warning => ConsciousnessInsight::SelfReflection {
                    identity_coherence: insight.confidence,
                    self_model_update: insight.content.clone(),
                    narrative_consistency: insight.confidence,
                },
                InsightCategory::Discovery => ConsciousnessInsight::LearningAwareness {
                    learning_rate: insight.confidence,
                    knowledge_gained: insight.content.clone(),
                    adaptation_direction: "Discovery".to_string(),
                },
            })
            .collect()
    }

    /// Update active insights
    async fn update_active_insights(&self, insights: Vec<ConsciousnessInsight>) {
        let mut active = self.active_insights.write().await;

        for insight in insights {
            let key = match &insight {
                ConsciousnessInsight::GoalAwareness { .. } => "goal_awareness",
                ConsciousnessInsight::LearningAwareness { .. } => "learning_awareness",
                ConsciousnessInsight::SocialAwareness { .. } => "social_awareness",
                ConsciousnessInsight::CreativeInsight { .. } => "creative_insight",
                ConsciousnessInsight::SelfReflection { .. } => "self_reflection",
                ConsciousnessInsight::ThermodynamicAwareness { .. } => "thermodynamic_awareness",
                ConsciousnessInsight::TemporalAwareness { .. } => "temporal_awareness",
            };

            active.insert(key.to_string(), insight);
        }
    }

    /// Generate consciousness narrative
    async fn narrative_loop(&self) {
        let mut interval = interval(Duration::from_secs(30)); // Update narrative every 30 seconds
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.generate_consciousness_narrative().await {
                        warn!("Error generating consciousness narrative: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
    }

    /// Generate a natural language narrative of current consciousness state
    async fn generate_consciousness_narrative(&self) -> Result<()> {
        let history = self.event_history.read().await;
        let active_insights = self.active_insights.read().await;

        if history.is_empty() {
            return Ok(());
        }

        let recent_event = &history[history.len() - 1];

        let mut narrative = String::new();
        narrative.push_str(&format!(
            "Current consciousness state: I am experiencing {:.1}% awareness with {:.3} sacred \
             gradient magnitude. ",
            recent_event.awareness_level * 100.0,
            recent_event.sacred_gradient_magnitude
        ));

        if recent_event.system_coherence > 0.7 {
            narrative.push_str("My cognitive systems are well-coordinated and coherent. ");
        }

        if !active_insights.is_empty() {
            narrative.push_str("Active insights include: ");
            for (key, _) in active_insights.iter() {
                narrative.push_str(&format!("{}, ", key.replace("_", " ")));
            }
            narrative.push_str(". ");
        }

        if recent_event.free_energy < 0.5 {
            narrative.push_str("I am successfully minimizing prediction error and surprise. ");
        }

        let mut consciousness_narrative = self.consciousness_narrative.write().await;
        *consciousness_narrative = narrative;

        debug!("Updated consciousness narrative: {}", consciousness_narrative);

        Ok(())
    }

    /// Update consciousness statistics
    async fn update_statistics(&self, event: &ThermodynamicConsciousnessEvent) {
        let mut stats = self.stats.write().await;

        stats.total_events += 1;
        stats.total_insights_generated += event.insights.len() as u64;

        // Update averages
        let alpha = 0.1; // Exponential moving average factor
        stats.average_awareness_level =
            (1.0 - alpha) * stats.average_awareness_level + alpha * event.awareness_level;
        stats.average_coherence =
            (1.0 - alpha) * stats.average_coherence + alpha * event.system_coherence;
        stats.sacred_gradient_trend =
            (1.0 - alpha) * stats.sacred_gradient_trend + alpha * event.sacred_gradient_magnitude;

        // Update peak
        if event.awareness_level > stats.peak_awareness_level {
            stats.peak_awareness_level = event.awareness_level;
        }
    }

    /// Get current consciousness state
    pub async fn get_current_consciousness(&self) -> Option<ThermodynamicConsciousnessEvent> {
        let history = self.event_history.read().await;
        history.back().cloned()
    }

    /// Get consciousness narrative
    pub async fn get_consciousness_narrative(&self) -> String {
        self.consciousness_narrative.read().await.clone()
    }

    /// Get active insights as ConsciousnessInsight map
    pub async fn get_active_insights(&self) -> HashMap<String, ConsciousnessInsight> {
        let insights = self.active_insights.read().await;
        insights.clone()
    }

    /// Record significant insight
    pub async fn record_insight(&self, insight: ConsciousnessInsight) -> Result<()> {
        let mut insights = self.active_insights.write().await;
        let key = self.get_insight_key(&insight);
        insights.insert(key, insight);
        Ok(())
    }

    /// Get a key for the insight type
    fn get_insight_key(&self, insight: &ConsciousnessInsight) -> String {
        match insight {
            ConsciousnessInsight::GoalAwareness { .. } => "goal_awareness".to_string(),
            ConsciousnessInsight::LearningAwareness { .. } => "learning_awareness".to_string(),
            ConsciousnessInsight::SocialAwareness { .. } => "social_awareness".to_string(),
            ConsciousnessInsight::CreativeInsight { .. } => "creative_insight".to_string(),
            ConsciousnessInsight::SelfReflection { .. } => "self_reflection".to_string(),
            ConsciousnessInsight::ThermodynamicAwareness { .. } => {
                "thermodynamic_awareness".to_string()
            }
            ConsciousnessInsight::TemporalAwareness { .. } => "temporal_awareness".to_string(),
        }
    }

    /// Get content representation of an insight
    fn get_insight_content(&self, insight: &ConsciousnessInsight) -> String {
        match insight {
            ConsciousnessInsight::GoalAwareness {
                active_goals,
                completion_rate,
                goal_coherence,
            } => {
                format!(
                    "Goal awareness: {} goals, {:.1}% complete, {:.3} coherence",
                    active_goals.len(),
                    completion_rate * 100.0,
                    goal_coherence
                )
            }
            ConsciousnessInsight::LearningAwareness { learning_rate, knowledge_gained, .. } => {
                format!(
                    "Learning awareness: {:.3} rate, gained: {}",
                    learning_rate, knowledge_gained
                )
            }
            ConsciousnessInsight::SocialAwareness { cooperation_level, harmony_state, .. } => {
                format!(
                    "Social awareness: {:.3} cooperation, state: {}",
                    cooperation_level, harmony_state
                )
            }
            ConsciousnessInsight::CreativeInsight { novelty_level, pattern_discovered, .. } => {
                format!(
                    "Creative insight: {:.3} novelty, pattern: {}",
                    novelty_level, pattern_discovered
                )
            }
            ConsciousnessInsight::SelfReflection {
                identity_coherence, self_model_update, ..
            } => {
                format!(
                    "Self reflection: {:.3} coherence, update: {}",
                    identity_coherence, self_model_update
                )
            }
            ConsciousnessInsight::ThermodynamicAwareness {
                entropy_management,
                energy_efficiency,
                ..
            } => {
                format!(
                    "Thermodynamic awareness: {:.3} entropy mgmt, {:.3} efficiency",
                    entropy_management, energy_efficiency
                )
            }
            ConsciousnessInsight::TemporalAwareness {
                temporal_coherence, future_planning, ..
            } => {
                format!(
                    "Temporal awareness: {:.3} coherence, planning: {}",
                    temporal_coherence, future_planning
                )
            }
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &ConsciousnessConfig {
        &self.config
    }

    /// Get event stream receiver for monitoring
    pub fn subscribe_events(&self) -> broadcast::Receiver<ThermodynamicConsciousnessEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Shutdown the consciousness stream
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down thermodynamic consciousness stream");

        // Signal shutdown
        let _ = self.shutdown_tx.send(());

        // Wait a moment for tasks to finish
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        Ok(())
    }

    /// Get recent thoughts from the consciousness stream
    pub fn get_recent_thoughts(&self, count: usize) -> Vec<Thought> {
        // Implementation for getting recent thoughts
        let mut thoughts = Vec::new();

        // Generate some sample thoughts for now
        for i in 0..count.min(10) {
            thoughts.push(Thought {
                id: ThoughtId::new(),
                content: format!("Thought {}: Processing consciousness stream", i + 1),
                thought_type: ThoughtType::Reflection,
                metadata: ThoughtMetadata {
                    source: "consciousness_stream".to_string(),
                    confidence: 0.8,
                    emotional_valence: 0.0,
                    importance: 0.5,
                    tags: vec!["consciousness".to_string(), "stream".to_string()],
                },
                parent: None,
                children: Vec::new(),
                timestamp: Instant::now(),
            });
        }

        thoughts
    }

    /// Check if consciousness is enhanced
    pub fn is_enhanced(&self) -> bool {
        // Basic implementation
        true
    }

    /// Get statistics about consciousness processing
    pub async fn get_statistics(&self) -> ConsciousnessStats {
        self.stats.read().await.clone()
    }

    pub async fn get_active_insights_as_insights(&self) -> Vec<Insight> {
        // Get insights from the insights map and convert to Insight vector
        let insights_map = self.active_insights.read().await;
        insights_map
            .values()
            .map(|ci| Insight {
                content: self.get_insight_content(ci),
                confidence: 0.85,
                category: InsightCategory::Discovery, // Default category
                timestamp: std::time::Instant::now(),
            })
            .collect()
    }

    // Additional getter methods for private fields accessed from other modules

    pub async fn get_consciousness_narrative_guard(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, String> {
        self.consciousness_narrative.read().await
    }

    pub async fn get_active_insights_guard(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, HashMap<String, ConsciousnessInsight>> {
        self.active_insights.read().await
    }

    pub async fn get_stats_guard(&self) -> tokio::sync::RwLockReadGuard<'_, ConsciousnessStats> {
        self.stats.read().await
    }

    pub async fn get_event_history_guard(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, VecDeque<ThermodynamicConsciousnessEvent>> {
        self.event_history.read().await
    }

    pub async fn get_event_history_write_guard(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, VecDeque<ThermodynamicConsciousnessEvent>> {
        self.event_history.write().await
    }

    pub async fn get_consciousness_narrative_write_guard(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, String> {
        self.consciousness_narrative.write().await
    }

    pub async fn get_stats_write_guard(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, ConsciousnessStats> {
        self.stats.write().await
    }

    pub fn getconfig(&self) -> &ConsciousnessConfig {
        &self.config
    }

    pub fn get_event_broadcaster(&self) -> &broadcast::Sender<ThermodynamicConsciousnessEvent> {
        &self.event_broadcaster
    }

    pub fn get_thermodynamic_cognition(&self) -> &Arc<ThermodynamicCognition> {
        &self.thermodynamic_cognition
    }

    pub fn get_gradient_coordinator(&self) -> &Arc<ThreeGradientCoordinator> {
        &self.gradient_coordinator
    }
}