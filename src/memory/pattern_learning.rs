//! Pattern Learning System
//!
//! Implements advanced pattern learning from multi-agent interactions,
//! decision history, and consciousness evolution.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::time::interval;
use tracing::{debug, info, error};

use crate::cognitive::agents::{
    DistributedDecisionResult,
};
use crate::memory::{
    FractalMemoryActivator, LearnedPattern, PatternType,
    ConsciousnessSnapshot, EmergenceEventRecord,
};

/// Configuration for pattern learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternLearningConfig {
    /// Enable pattern learning
    pub enabled: bool,

    /// Learning rate for pattern adaptation
    pub learning_rate: f64,

    /// Minimum pattern confidence threshold
    pub min_confidence: f64,

    /// Maximum pattern age before decay
    pub max_pattern_age: Duration,

    /// Pattern validation threshold
    pub validation_threshold: f64,

    /// Cross-validation window size
    pub cross_validation_window: usize,

    /// Pattern clustering threshold
    pub clustering_threshold: f64,

    /// Enable pattern evolution
    pub enable_evolution: bool,

    /// Evolution rate
    pub evolution_rate: f64,
}

impl Default for PatternLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.1,
            min_confidence: 0.7,
            max_pattern_age: Duration::from_secs(3600), // 1 hour
            validation_threshold: 0.8,
            cross_validation_window: 10,
            clustering_threshold: 0.85,
            enable_evolution: true,
            evolution_rate: 0.05,
        }
    }
}

/// Pattern learning analytics
#[derive(Debug, Clone, Default)]
pub struct PatternLearningAnalytics {
    /// Total patterns learned
    pub total_patterns_learned: u64,

    /// Patterns validated
    pub patterns_validated: u64,

    /// Patterns evolved
    pub patterns_evolved: u64,

    /// Patterns discarded
    pub patterns_discarded: u64,

    /// Average pattern confidence
    pub average_confidence: f64,

    /// Learning efficiency rate
    pub learning_efficiency: f64,

    /// Pattern accuracy over time
    pub pattern_accuracy_history: VecDeque<f64>,

    /// Cross-validation success rate
    pub cross_validation_success_rate: f64,
}

/// Pattern validation result
#[derive(Debug, Clone)]
pub struct PatternValidationResult {
    /// Whether pattern is valid
    pub is_valid: bool,

    /// Validation confidence
    pub confidence: f64,

    /// Validation reasons
    pub reasons: Vec<String>,

    /// Suggested improvements
    pub improvements: Vec<String>,
}

/// Pattern cluster for similar patterns
#[derive(Debug, Clone)]
pub struct PatternCluster {
    /// Cluster ID
    pub id: String,

    /// Cluster center pattern
    pub center_pattern: LearnedPattern,

    /// Patterns in cluster
    pub patterns: Vec<LearnedPattern>,

    /// Cluster coherence score
    pub coherence: f64,

    /// Cluster creation time
    pub created_at: Instant,

    /// Last update time
    pub last_updated: Instant,
}

/// Pattern evolution record
#[derive(Debug, Clone)]
pub struct PatternEvolution {
    /// Original pattern ID
    pub original_id: String,

    /// Evolved pattern ID
    pub evolved_id: String,

    /// Evolution type
    pub evolution_type: EvolutionType,

    /// Evolution confidence
    pub confidence: f64,

    /// Evolution timestamp
    pub timestamp: Instant,

    /// Evolution description
    pub description: String,
}

/// Types of pattern evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionType {
    /// Pattern refinement
    Refinement,

    /// Pattern specialization
    Specialization,

    /// Pattern generalization
    Generalization,

    /// Pattern combination
    Combination,

    /// Pattern adaptation
    Adaptation,
}

/// Advanced pattern learning system
#[derive(Clone)]
pub struct PatternLearningSystem {
    /// Configuration
    config: PatternLearningConfig,

    /// Fractal memory activator
    fractal_activator: Arc<FractalMemoryActivator>,

    /// Learned patterns with metadata
    patterns: Arc<RwLock<HashMap<String, LearnedPattern>>>,

    /// Pattern clusters
    clusters: Arc<RwLock<HashMap<String, PatternCluster>>>,

    /// Pattern evolution history
    evolution_history: Arc<RwLock<Vec<PatternEvolution>>>,

    /// Pattern validation cache
    validation_cache: Arc<RwLock<HashMap<String, PatternValidationResult>>>,

    /// Learning analytics
    analytics: Arc<RwLock<PatternLearningAnalytics>>,

    /// Learning events channel
    learning_tx: mpsc::Sender<LearningEvent>,
    learning_rx: Arc<RwLock<mpsc::Receiver<LearningEvent>>>,

    /// Pattern events broadcast
    pattern_events_tx: broadcast::Sender<PatternEvent>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Learning events
#[derive(Debug, Clone)]
pub enum LearningEvent {
    /// New pattern discovered
    PatternDiscovered(LearnedPattern),

    /// Pattern validated
    PatternValidated(String, PatternValidationResult),

    /// Pattern evolved
    PatternEvolved(PatternEvolution),

    /// Pattern cluster updated
    ClusterUpdated(String),

    /// Learning analytics updated
    AnalyticsUpdated(PatternLearningAnalytics),
}

/// Pattern events
#[derive(Debug, Clone)]
pub enum PatternEvent {
    /// Pattern learned
    PatternLearned(String),

    /// Pattern validated
    PatternValidated(String),

    /// Pattern evolved
    PatternEvolved(String, String),

    /// Pattern cluster formed
    ClusterFormed(String),

    /// Pattern discarded
    PatternDiscarded(String),
}

impl PatternLearningSystem {
    /// Create new pattern learning system
    pub async fn new(
        config: PatternLearningConfig,
        fractal_activator: Arc<FractalMemoryActivator>,
    ) -> Result<Self> {
        info!("🧠 Initializing Pattern Learning System");

        let (learning_tx, learning_rx) = mpsc::channel(1000);
        let (pattern_events_tx, _) = broadcast::channel(100);

        Ok(Self {
            config,
            fractal_activator,
            patterns: Arc::new(RwLock::new(HashMap::new())),
            clusters: Arc::new(RwLock::new(HashMap::new())),
            evolution_history: Arc::new(RwLock::new(Vec::new())),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
            analytics: Arc::new(RwLock::new(PatternLearningAnalytics::default())),
            learning_tx,
            learning_rx: Arc::new(RwLock::new(learning_rx)),
            pattern_events_tx,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the pattern learning system
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Pattern learning disabled in configuration");
            return Ok(());
        }

        *self.running.write().await = true;
        info!("🚀 Pattern Learning System started");

        // Start pattern learning loop
        let self_clone = Arc::new(self.clone());
        let learning_handle = self_clone.spawn_learning_loop();
        tokio::spawn(learning_handle);

        // Start pattern validation loop
        let self_clone = Arc::new(self.clone());
        let validation_handle = self_clone.spawn_validation_loop();
        tokio::spawn(validation_handle);

        // Start pattern evolution loop
        if self.config.enable_evolution {
            let self_clone = Arc::new(self.clone());
            let evolution_handle = self_clone.spawn_evolution_loop();
            tokio::spawn(evolution_handle);
        }

        // Start pattern clustering loop
        let self_clone = Arc::new(self.clone());
        let clustering_handle = self_clone.spawn_clustering_loop();
        tokio::spawn(clustering_handle);

        Ok(())
    }

    /// Stop the pattern learning system
    pub async fn stop(&self) -> Result<()> {
        *self.running.write().await = false;
        info!("🛑 Pattern Learning System stopped");
        Ok(())
    }

    /// Learn from multi-agent decision
    pub async fn learn_from_decision(&self, decision: &DistributedDecisionResult) -> Result<()> {
        debug!("Learning from decision: {}", decision.decision_id);

        // Extract decision patterns
        let patterns = self.extract_decision_patterns(decision).await?;

        // Process each pattern
        for pattern in patterns {
            self.process_learned_pattern(pattern).await?;
        }

        Ok(())
    }

    /// Learn from consciousness evolution
    pub async fn learn_from_consciousness(&self, snapshot: &ConsciousnessSnapshot) -> Result<()> {
        debug!("Learning from consciousness evolution with {} agents", snapshot.agent_count);

        // Extract consciousness patterns
        let patterns = self.extract_consciousness_patterns(snapshot).await?;

        // Process each pattern
        for pattern in patterns {
            self.process_learned_pattern(pattern).await?;
        }

        Ok(())
    }

    /// Learn from emergence events
    pub async fn learn_from_emergence(&self, event: &EmergenceEventRecord) -> Result<()> {
        debug!("Learning from emergence event: {}", event.event_type);

        // Extract emergence patterns
        let patterns = self.extract_emergence_patterns(event).await?;

        // Process each pattern
        for pattern in patterns {
            self.process_learned_pattern(pattern).await?;
        }

        Ok(())
    }

    /// Get learning analytics
    pub async fn get_analytics(&self) -> PatternLearningAnalytics {
        self.analytics.read().await.clone()
    }

    /// Get learned patterns
    pub async fn get_patterns(&self) -> HashMap<String, LearnedPattern> {
        self.patterns.read().await.clone()
    }

    /// Get pattern clusters
    pub async fn get_clusters(&self) -> HashMap<String, PatternCluster> {
        self.clusters.read().await.clone()
    }

    /// Subscribe to pattern events
    pub fn subscribe_to_pattern_events(&self) -> broadcast::Receiver<PatternEvent> {
        self.pattern_events_tx.subscribe()
    }

    /// Extract decision patterns from decision result
    async fn extract_decision_patterns(&self, decision: &DistributedDecisionResult) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();

        // Success pattern
        if decision.consensus_reached {
            let success_pattern = LearnedPattern {
                id: format!("decision_success_{}", decision.decision_id),
                pattern: PatternType::DecisionPattern {
                    domain: "decision_making".to_string(),
                    success_rate: 1.0,
                    consensus_threshold: decision.consensus_score,
                    agent_specializations: decision.participating_agents
                        .iter()
                        .map(|id| id.to_string())
                        .collect(),
                },
                confidence: decision.quality_metrics.overall_quality,
                observation_count: 1,
                stability: 0.8,
                last_observed: chrono::Utc::now(),
                fractal_node_id: None,
                associated_memories: vec![],
            };
            patterns.push(success_pattern);
        }

        // Consensus pattern
        if decision.consensus_score > 0.8 {
            let consensus_pattern = LearnedPattern {
                id: format!("high_consensus_{}", decision.decision_id),
                pattern: PatternType::CollaborationPattern {
                    agent_pairs: decision.participating_agents
                        .iter()
                        .enumerate()
                        .flat_map(|(i, agent1)| {
                            decision.participating_agents
                                .iter()
                                .skip(i + 1)
                                .map(|agent2| (agent1.clone(), agent2.clone()))
                        })
                        .collect(),
                    interaction_strength: decision.consensus_score,
                    complementarity_score: decision.quality_metrics.expertise_coverage,
                    conflict_resolution_rate: 0.9,
                },
                confidence: decision.consensus_score,
                observation_count: 1,
                stability: 0.7,
                last_observed: chrono::Utc::now(),
                fractal_node_id: None,
                associated_memories: vec![],
            };
            patterns.push(consensus_pattern);
        }

        Ok(patterns)
    }

    /// Extract consciousness patterns from snapshot
    async fn extract_consciousness_patterns(&self, snapshot: &ConsciousnessSnapshot) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();

        // Coherence evolution pattern
        if snapshot.coherence_delta.abs() > 0.1 {
            let coherence_pattern = LearnedPattern {
                id: format!("coherence_evolution_{}", snapshot.timestamp.elapsed().as_secs()),
                pattern: PatternType::ConsciousnessPattern {
                    coherence_trend: snapshot.coherence_delta,
                    awareness_evolution: snapshot.collective_state.unified_state.awareness_level,
                    emergence_frequency: snapshot.emergence_indicators.len() as f64,
                    synchronization_quality: snapshot.collective_state.collective_coherence,
                },
                confidence: snapshot.collective_state.collective_coherence,
                observation_count: 1,
                stability: 0.6,
                last_observed: chrono::Utc::now(),
                fractal_node_id: None,
                associated_memories: vec![],
            };
            patterns.push(coherence_pattern);
        }

        // Emergence pattern
        if !snapshot.emergence_indicators.is_empty() {
            let emergence_pattern = LearnedPattern {
                id: format!("emergence_indicators_{}", snapshot.timestamp.elapsed().as_secs()),
                pattern: PatternType::EmergencePattern {
                    emergence_type: "consciousness_emergence".to_string(),
                    trigger_conditions: snapshot.emergence_indicators
                        .iter()
                        .map(|ind| format!("{:?}", ind.indicator_type))
                        .collect(),
                    propagation_speed: 0.8,
                    stability_duration: Duration::from_secs(300),
                },
                confidence: 0.8,
                observation_count: 1,
                stability: 0.7,
                last_observed: chrono::Utc::now(),
                fractal_node_id: None,
                associated_memories: vec![],
            };
            patterns.push(emergence_pattern);
        }

        Ok(patterns)
    }

    /// Extract emergence patterns from event
    async fn extract_emergence_patterns(&self, event: &EmergenceEventRecord) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();

        // Emergence type pattern
        let emergence_pattern = LearnedPattern {
            id: format!("emergence_{}_{}", event.event_type, event.timestamp.elapsed().as_secs()),
            pattern: PatternType::EmergencePattern {
                emergence_type: event.event_type.clone(),
                trigger_conditions: event.context.keys().cloned().collect(),
                propagation_speed: event.strength,
                stability_duration: Duration::from_secs(300),
            },
            confidence: event.strength,
            observation_count: 1,
            stability: 0.6,
            last_observed: chrono::Utc::now(),
            fractal_node_id: None,
            associated_memories: vec![],
        };
        patterns.push(emergence_pattern);

        // Multi-agent collaboration pattern if multiple agents involved
        if event.participating_agents.len() > 1 {
            let collaboration_pattern = LearnedPattern {
                id: format!("collaboration_{}_{}", event.event_type, event.timestamp.elapsed().as_secs()),
                pattern: PatternType::CollaborationPattern {
                    agent_pairs: event.participating_agents
                        .iter()
                        .enumerate()
                        .flat_map(|(i, agent1)| {
                            event.participating_agents
                                .iter()
                                .skip(i + 1)
                                .map(|agent2| (agent1.clone(), agent2.clone()))
                        })
                        .collect(),
                    interaction_strength: event.strength,
                    complementarity_score: 0.8,
                    conflict_resolution_rate: 0.9,
                },
                confidence: event.strength,
                observation_count: 1,
                stability: 0.7,
                last_observed: chrono::Utc::now(),
                fractal_node_id: None,
                associated_memories: vec![],
            };
            patterns.push(collaboration_pattern);
        }

        Ok(patterns)
    }

    /// Process a learned pattern
    async fn process_learned_pattern(&self, pattern: LearnedPattern) -> Result<()> {
        // Validate pattern
        let validation = self.validate_pattern(&pattern).await?;
        if !validation.is_valid {
            debug!("Pattern {} failed validation", pattern.id);
            return Ok(());
        }

        // Store pattern
        {
            let mut patterns = self.patterns.write().await;
            patterns.insert(pattern.id.clone(), pattern.clone());
        }

        // Update analytics
        self.update_analytics_for_new_pattern(&pattern).await?;

        // Send learning event
        self.learning_tx.send(LearningEvent::PatternDiscovered(pattern.clone())).await?;

        // Broadcast pattern event
        let _ = self.pattern_events_tx.send(PatternEvent::PatternLearned(pattern.id.clone()));

        info!("📚 Pattern learned: {}", pattern.id);

        Ok(())
    }

    /// Validate a pattern
    async fn validate_pattern(&self, pattern: &LearnedPattern) -> Result<PatternValidationResult> {
        // Check cache first
        if let Some(cached) = self.validation_cache.read().await.get(&pattern.id) {
            return Ok(cached.clone());
        }

        let mut reasons = Vec::new();
        let mut improvements = Vec::new();
        let mut confidence = pattern.confidence;

        // Basic validation checks
        if pattern.confidence < self.config.min_confidence {
            reasons.push("Confidence below threshold".to_string());
            confidence *= 0.8;
        }

        if pattern.observation_count < 2 {
            reasons.push("Insufficient observations".to_string());
            improvements.push("Collect more observations".to_string());
        }

        if pattern.stability < 0.5 {
            reasons.push("Low stability score".to_string());
            improvements.push("Improve pattern stability".to_string());
        }

        // Pattern-specific validation
        match &pattern.pattern {
            PatternType::DecisionPattern { success_rate, .. } => {
                if *success_rate < 0.7 {
                    reasons.push("Low success rate".to_string());
                    improvements.push("Improve decision quality".to_string());
                }
            }
            PatternType::ConsciousnessPattern { synchronization_quality, .. } => {
                if *synchronization_quality < 0.6 {
                    reasons.push("Poor synchronization".to_string());
                    improvements.push("Improve consciousness sync".to_string());
                }
            }
            _ => {}
        }

        let is_valid = confidence >= self.config.validation_threshold && reasons.is_empty();

        let result = PatternValidationResult {
            is_valid,
            confidence,
            reasons,
            improvements,
        };

        // Cache result
        self.validation_cache.write().await.insert(pattern.id.clone(), result.clone());

        Ok(result)
    }

    /// Update analytics for new pattern
    async fn update_analytics_for_new_pattern(&self, pattern: &LearnedPattern) -> Result<()> {
        let mut analytics = self.analytics.write().await;

        analytics.total_patterns_learned += 1;

        // Update average confidence
        let total_confidence = analytics.average_confidence * (analytics.total_patterns_learned - 1) as f64;
        analytics.average_confidence = (total_confidence + pattern.confidence) / analytics.total_patterns_learned as f64;

        // Update pattern accuracy history
        analytics.pattern_accuracy_history.push_back(pattern.confidence);
        if analytics.pattern_accuracy_history.len() > 100 {
            analytics.pattern_accuracy_history.pop_front();
        }

        // Update learning efficiency
        analytics.learning_efficiency = analytics.patterns_validated as f64 / analytics.total_patterns_learned as f64;

        Ok(())
    }

    /// Spawn learning loop
    async fn spawn_learning_loop(self: Arc<Self>) -> Result<()> {
        let mut rx = self.learning_rx.write().await;
        let mut learning_interval = interval(Duration::from_secs(30));

        while *self.running.read().await {
            tokio::select! {
                _ = learning_interval.tick() => {
                    if let Err(e) = self.perform_learning_analysis().await {
                        error!("Learning analysis error: {}", e);
                    }
                }

                event = rx.recv() => {
                    if let Some(event) = event {
                        if let Err(e) = self.handle_learning_event(event).await {
                            error!("Learning event handling error: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle learning events
    async fn handle_learning_event(&self, event: LearningEvent) -> Result<()> {
        match event {
            LearningEvent::PatternDiscovered(pattern) => {
                debug!("Processing discovered pattern: {}", pattern.id);
            }
            LearningEvent::PatternValidated(pattern_id, result) => {
                debug!("Pattern {} validated: {}", pattern_id, result.is_valid);
            }
            LearningEvent::PatternEvolved(evolution) => {
                debug!("Pattern evolved: {} -> {}", evolution.original_id, evolution.evolved_id);
            }
            LearningEvent::ClusterUpdated(cluster_id) => {
                debug!("Cluster updated: {}", cluster_id);
            }
            LearningEvent::AnalyticsUpdated(analytics) => {
                debug!("Analytics updated: {} patterns learned", analytics.total_patterns_learned);
            }
        }
        Ok(())
    }

    /// Perform learning analysis
    async fn perform_learning_analysis(&self) -> Result<()> {
        debug!("Performing learning analysis");

        // Analyze pattern trends
        self.analyze_pattern_trends().await?;

        // Update pattern clusters
        self.update_pattern_clusters().await?;

        // Validate existing patterns
        self.validate_existing_patterns().await?;

        Ok(())
    }

    /// Analyze pattern trends
    async fn analyze_pattern_trends(&self) -> Result<()> {
        let patterns = self.patterns.read().await;

        if patterns.len() < 5 {
            return Ok(());
        }

        // Analyze decision patterns
        let decision_patterns: Vec<_> = patterns.values()
            .filter(|p| matches!(p.pattern, PatternType::DecisionPattern { .. }))
            .collect();

        if decision_patterns.len() >= 3 {
            let avg_confidence = decision_patterns.iter()
                .map(|p| p.confidence)
                .sum::<f64>() / decision_patterns.len() as f64;

            debug!("Decision patterns trend: {} patterns, avg confidence: {:.2}",
                   decision_patterns.len(), avg_confidence);
        }

        // Analyze consciousness patterns
        let consciousness_patterns: Vec<_> = patterns.values()
            .filter(|p| matches!(p.pattern, PatternType::ConsciousnessPattern { .. }))
            .collect();

        if consciousness_patterns.len() >= 3 {
            let avg_confidence = consciousness_patterns.iter()
                .map(|p| p.confidence)
                .sum::<f64>() / consciousness_patterns.len() as f64;

            debug!("Consciousness patterns trend: {} patterns, avg confidence: {:.2}",
                   consciousness_patterns.len(), avg_confidence);
        }

        Ok(())
    }

    /// Update pattern clusters
    async fn update_pattern_clusters(&self) -> Result<()> {
        let patterns = self.patterns.read().await;
        let mut clusters = self.clusters.write().await;

        // Simple clustering based on pattern types
        let mut pattern_groups: HashMap<String, Vec<LearnedPattern>> = HashMap::new();

        for pattern in patterns.values() {
            let group_key = match &pattern.pattern {
                PatternType::DecisionPattern { .. } => "decision",
                PatternType::ConsciousnessPattern { .. } => "consciousness",
                PatternType::CollaborationPattern { .. } => "collaboration",
                PatternType::EmergencePattern { .. } => "emergence",
                PatternType::LearningPattern { .. } => "learning",
            };

            pattern_groups.entry(group_key.to_string())
                .or_insert_with(Vec::new)
                .push(pattern.clone());
        }

        // Create or update clusters
        for (group_key, group_patterns) in pattern_groups {
            if group_patterns.len() >= 3 {
                let cluster_id = format!("cluster_{}", group_key);

                // Calculate cluster center (pattern with highest confidence)
                let center_pattern = group_patterns.iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .unwrap()
                    .clone();

                // Calculate coherence (average confidence)
                let coherence = group_patterns.iter()
                    .map(|p| p.confidence)
                    .sum::<f64>() / group_patterns.len() as f64;

                let cluster = PatternCluster {
                    id: cluster_id.clone(),
                    center_pattern,
                    patterns: group_patterns,
                    coherence,
                    created_at: Instant::now(),
                    last_updated: Instant::now(),
                };

                clusters.insert(cluster_id.clone(), cluster);

                // Broadcast cluster event
                let _ = self.pattern_events_tx.send(PatternEvent::ClusterFormed(cluster_id));
            }
        }

        Ok(())
    }

    /// Validate existing patterns
    async fn validate_existing_patterns(&self) -> Result<()> {
        let patterns = self.patterns.read().await;
        let mut validated_count = 0;

        for pattern in patterns.values() {
            let validation = self.validate_pattern(pattern).await?;
            if validation.is_valid {
                validated_count += 1;
            }
        }

        // Update analytics
        {
            let mut analytics = self.analytics.write().await;
            analytics.patterns_validated = validated_count;
            analytics.cross_validation_success_rate = validated_count as f64 / patterns.len() as f64;
        }

        debug!("Validated {} out of {} patterns", validated_count, patterns.len());

        Ok(())
    }

    /// Spawn validation loop
    async fn spawn_validation_loop(self: Arc<Self>) -> Result<()> {
        let mut validation_interval = interval(Duration::from_secs(60));

        while *self.running.read().await {
            validation_interval.tick().await;

            if let Err(e) = self.validate_existing_patterns().await {
                error!("Pattern validation error: {}", e);
            }
        }

        Ok(())
    }

    /// Spawn evolution loop
    async fn spawn_evolution_loop(self: Arc<Self>) -> Result<()> {
        let mut evolution_interval = interval(Duration::from_secs(120));

        while *self.running.read().await {
            evolution_interval.tick().await;

            if let Err(e) = self.evolve_patterns().await {
                error!("Pattern evolution error: {}", e);
            }
        }

        Ok(())
    }

    /// Evolve patterns
    async fn evolve_patterns(&self) -> Result<()> {
        debug!("Evolving patterns");

        let patterns = self.patterns.read().await;
        let mut evolved_count = 0;

        // Simple evolution: refine patterns with low confidence
        for pattern in patterns.values() {
            if pattern.confidence < 0.8 && pattern.observation_count >= 5 {
                // Create evolved pattern with improved confidence
                let evolved_pattern = LearnedPattern {
                    id: format!("evolved_{}", pattern.id),
                    pattern: pattern.pattern.clone(),
                    confidence: (pattern.confidence + self.config.evolution_rate).min(1.0),
                    observation_count: pattern.observation_count,
                    stability: (pattern.stability + self.config.evolution_rate * 0.5).min(1.0),
                    last_observed: chrono::Utc::now(),
                    fractal_node_id: pattern.fractal_node_id.clone(),
                    associated_memories: pattern.associated_memories.clone(),
                };

                // Record evolution
                let evolution = PatternEvolution {
                    original_id: pattern.id.clone(),
                    evolved_id: evolved_pattern.id.clone(),
                    evolution_type: EvolutionType::Refinement,
                    confidence: evolved_pattern.confidence,
                    timestamp: Instant::now(),
                    description: "Refined pattern with improved confidence".to_string(),
                };

                // Store evolution
                self.evolution_history.write().await.push(evolution.clone());

                // Process evolved pattern
                drop(patterns); // Release read lock
                self.process_learned_pattern(evolved_pattern).await?;

                evolved_count += 1;

                // Broadcast evolution event
                let _ = self.pattern_events_tx.send(
                    PatternEvent::PatternEvolved(evolution.original_id, evolution.evolved_id)
                );

                break; // Only evolve one pattern per cycle
            }
        }

        if evolved_count > 0 {
            // Update analytics
            let mut analytics = self.analytics.write().await;
            analytics.patterns_evolved += evolved_count;

            info!("🧬 Evolved {} patterns", evolved_count);
        }

        Ok(())
    }

    /// Spawn clustering loop
    async fn spawn_clustering_loop(self: Arc<Self>) -> Result<()> {
        let mut clustering_interval = interval(Duration::from_secs(90));

        while *self.running.read().await {
            clustering_interval.tick().await;

            if let Err(e) = self.update_pattern_clusters().await {
                error!("Pattern clustering error: {}", e);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::FractalActivationConfig;
    use super::*;

    #[tokio::test]
    async fn test_pattern_learning_system_creation() {
        let config = PatternLearningConfig::default();
        let fractal_config = FractalActivationConfig::default();
        let memory = Arc::new(crate::memory::CognitiveMemory::new(Default::default()).await.unwrap());
        let fractal_activator = Arc::new(FractalMemoryActivator::new(fractal_config, memory).await.unwrap());

        let system = PatternLearningSystem::new(config, fractal_activator).await.unwrap();
        assert!(!*system.running.read().await);
    }

    #[tokio::test]
    async fn test_pattern_validation() {
        let config = PatternLearningConfig::default();
        let fractal_config = FractalActivationConfig::default();
        let memory = Arc::new(crate::memory::CognitiveMemory::new(Default::default()).await.unwrap());
        let fractal_activator = Arc::new(FractalMemoryActivator::new(fractal_config, memory).await.unwrap());
        let system = PatternLearningSystem::new(config, fractal_activator).await.unwrap();

        let pattern = LearnedPattern {
            id: "test_pattern".to_string(),
            pattern: PatternType::DecisionPattern {
                domain: "test".to_string(),
                success_rate: 0.9,
                consensus_threshold: 0.8,
                agent_specializations: vec!["test_agent".to_string()],
            },
            confidence: 0.8,
            observation_count: 5,
            stability: 0.7,
            last_observed: chrono::Utc::now(),
            fractal_node_id: None,
            associated_memories: vec![],
        };

        let validation = system.validate_pattern(&pattern).await.unwrap();
        assert!(validation.is_valid);
        assert!(validation.confidence >= 0.8);
    }
}
