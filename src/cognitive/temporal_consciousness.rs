//! Enhanced Temporal Consciousness
//!
//! This module enhances temporal awareness by integrating past, present, and
//! future consciousness states. It provides temporal pattern recognition,
//! predictive consciousness, and enhanced memory-consciousness integration for
//! better timeline optimization.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::cognitive::consciousness_stream::{
    ThermodynamicConsciousnessEvent,
    ThermodynamicConsciousnessStream,
};
use crate::cognitive::InsightCategory; // Insight is unused
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Enhanced temporal consciousness event with past/present/future integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConsciousnessEvent {
    /// Base consciousness event
    pub base_event: ThermodynamicConsciousnessEvent,

    /// Temporal context
    pub temporal_context: TemporalContext,

    /// Predictions about future states
    pub future_predictions: Vec<FuturePrediction>,

    /// Connections to past events
    pub past_connections: Vec<PastConnection>,

    /// Temporal patterns detected
    pub temporal_patterns: Vec<TemporalPattern>,

    /// Timeline coherence score
    pub timeline_coherence: f64,
}

/// Temporal context providing past/present/future awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Current moment characteristics
    pub present_state: PresentState,

    /// Recent past influences
    pub past_influences: Vec<PastInfluence>,

    /// Future trajectory predictions
    pub future_trajectories: Vec<FutureTrajectory>,

    /// Temporal scale being considered (seconds to days)
    pub temporal_scale: TemporalScale,

    /// Temporal urgency level
    pub urgency_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentState {
    /// Current consciousness quality
    pub consciousness_quality: f64,

    /// Immediate context factors
    pub context_factors: Vec<String>,

    /// Active temporal threads
    pub active_threads: Vec<String>,

    /// Present-moment awareness level
    pub mindfulness_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PastInfluence {
    /// Description of past event/state
    pub description: String,

    /// Time since the event
    pub time_since: Duration,

    /// Influence strength on current state
    pub influence_strength: f64,

    /// Type of influence
    pub influence_type: InfluenceType,

    /// Associated memory or event ID
    pub source_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfluenceType {
    /// Learning from past experience
    Learning,
    /// Emotional carry-over
    Emotional,
    /// Pattern continuation
    Pattern,
    /// Goal momentum
    Goal,
    /// Memory resonance
    Memory,
}

impl std::fmt::Display for InfluenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InfluenceType::Learning => write!(f, "Learning"),
            InfluenceType::Emotional => write!(f, "Emotional"),
            InfluenceType::Pattern => write!(f, "Pattern"),
            InfluenceType::Goal => write!(f, "Goal"),
            InfluenceType::Memory => write!(f, "Memory"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureTrajectory {
    /// Predicted future state
    pub description: String,

    /// Time horizon for prediction
    pub time_horizon: Duration,

    /// Probability of this trajectory
    pub probability: f64,

    /// Desirability of this outcome
    pub desirability: f64,

    /// Required actions to reach this state
    pub required_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturePrediction {
    /// What is predicted to happen
    pub predicted_event: String,

    /// Confidence in the prediction
    pub confidence: f64,

    /// Time until predicted event
    pub time_until: Duration,

    /// Impact if prediction comes true
    pub impact_level: f64,

    /// Preparatory actions suggested
    pub preparatory_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PastConnection {
    /// Connection to past consciousness event
    pub past_event_id: String,

    /// Type of connection
    pub connection_type: ConnectionType,

    /// Strength of connection
    pub connection_strength: f64,

    /// Temporal distance
    pub temporal_distance: Duration,

    /// Insights from this connection
    pub insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    /// Similar patterns detected
    PatternSimilarity,
    /// Causal relationship
    Causal,
    /// Cyclical return
    Cyclical,
    /// Learning evolution
    Learning,
    /// Emotional resonance
    Emotional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Description of the pattern
    pub description: String,

    /// Pattern type
    pub pattern_type: TemporalPatternType,

    /// Frequency of occurrence
    pub frequency: Duration,

    /// Pattern strength/confidence
    pub strength: f64,

    /// Next predicted occurrence
    pub next_occurrence: Option<SystemTime>,

    /// Pattern evolution trend
    pub evolution_trend: EvolutionTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPatternType {
    /// Daily cycles (circadian)
    Daily,
    /// Weekly patterns
    Weekly,
    /// Learning curves
    Learning,
    /// Goal progression patterns
    Goal,
    /// Emotional cycles
    Emotional,
    /// Creative bursts
    Creative,
    /// Social interaction patterns
    Social,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionTrend {
    /// Pattern is strengthening
    Strengthening,
    /// Pattern is weakening
    Weakening,
    /// Pattern is stable
    Stable,
    /// Pattern is shifting
    Shifting,
    /// Pattern emerging
    Emerging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalScale {
    /// Immediate (seconds to minutes)
    Immediate,
    /// Short-term (minutes to hours)
    ShortTerm,
    /// Medium-term (hours to days)
    MediumTerm,
    /// Long-term (days to weeks)
    LongTerm,
    /// Extended (weeks to months)
    Extended,
}

/// Enhanced temporal consciousness processor
pub struct TemporalConsciousnessProcessor {
    /// Base consciousness stream
    base_stream: Arc<ThermodynamicConsciousnessStream>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Temporal event history (organized by time)
    temporal_history: Arc<RwLock<BTreeMap<SystemTime, TemporalConsciousnessEvent>>>,

    /// Detected temporal patterns
    temporal_patterns: Arc<RwLock<Vec<TemporalPattern>>>,

    /// Future predictions being tracked
    active_predictions: Arc<RwLock<Vec<FuturePrediction>>>,

    /// Temporal insights
    temporal_insights: Arc<RwLock<Vec<TemporalInsight>>>,

    /// Event broadcaster for temporal events
    event_broadcaster: broadcast::Sender<TemporalConsciousnessEvent>,

    /// Configuration
    config: TemporalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInsight {
    /// Insight content
    pub content: String,

    /// Temporal scope of the insight
    pub temporal_scope: TemporalScale,

    /// Confidence in the insight
    pub confidence: f64,

    /// Actionability of the insight
    pub actionability: f64,

    /// When the insight was generated
    pub generated_at: SystemTime,

    /// Associated patterns or predictions
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// How far back to look for patterns
    pub pattern_lookback: Duration,

    /// Minimum pattern strength to consider
    pub pattern_threshold: f64,

    /// Future prediction horizon
    pub prediction_horizon: Duration,

    /// How often to analyze temporal patterns
    pub analysis_interval: Duration,

    /// Temporal coherence threshold
    pub coherence_threshold: f64,

    /// Maximum events to keep in history
    pub max_history_events: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            pattern_lookback: Duration::from_secs(7 * 24 * 3600), // 1 week
            pattern_threshold: 0.6,
            prediction_horizon: Duration::from_secs(24 * 3600), // 1 day
            analysis_interval: Duration::from_secs(300),        // 5 minutes
            coherence_threshold: 0.7,
            max_history_events: 10000,
        }
    }
}

/// Goal progression pattern data
#[derive(Debug, Clone)]
struct GoalProgression {
    description: String,
    duration: Duration,
    average_interval: Duration,
    coherence_score: f64,
    start_coherence: f64,
    end_coherence: f64,
}

/// Goal cycle pattern data
#[derive(Debug, Clone)]
struct GoalCycle {
    phase_count: usize,
    period: Duration,
    consistency_score: f64,
    phases: Vec<String>,
}

/// Goal coherence pattern data
#[derive(Debug, Clone)]
struct GoalCoherencePattern {
    pattern_description: String,
    frequency: Duration,
    strength: f64,
    next_predicted_occurrence: Option<SystemTime>,
    trend: EvolutionTrend,
}

/// Creative burst pattern data
#[derive(Debug, Clone)]
struct CreativeBurst {
    peak_intensity: f64,
    duration: Duration,
    recurrence_interval: Duration,
    intensity_score: f64,
    trigger_context: String,
}

/// Creative flow pattern data
#[derive(Debug, Clone)]
struct CreativeFlow {
    flow_type: String,
    duration: Duration,
    typical_duration: Duration,
    consistency_score: f64,
    next_predicted_start: Option<SystemTime>,
}

/// Inspiration pattern data
#[derive(Debug, Clone)]
struct InspirationPattern {
    trigger_type: String,
    average_interval: Duration,
    reliability_score: f64,
    next_predicted_occurrence: Option<SystemTime>,
    context_patterns: Vec<String>,
}

impl TemporalConsciousnessProcessor {
    /// Create new temporal consciousness processor
    pub async fn new(
        base_stream: Arc<ThermodynamicConsciousnessStream>,
        memory: Arc<CognitiveMemory>,
        config: Option<TemporalConfig>,
    ) -> Result<Arc<Self>> {
        info!("Initializing Enhanced Temporal Consciousness Processor");

        let config = config.unwrap_or_default();
        let (event_broadcaster, _) = broadcast::channel(100);

        let processor = Arc::new(Self {
            base_stream,
            memory,
            temporal_history: Arc::new(RwLock::new(BTreeMap::new())),
            temporal_patterns: Arc::new(RwLock::new(Vec::new())),
            active_predictions: Arc::new(RwLock::new(Vec::new())),
            temporal_insights: Arc::new(RwLock::new(Vec::new())),
            event_broadcaster,
            config,
        });

        Ok(processor)
    }

    /// Start temporal consciousness processing
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Enhanced Temporal Consciousness");

        // Subscribe to base consciousness events
        let mut base_events = self.base_stream.subscribe_events();

        // Start processing loop
        let processor = self.clone();
        tokio::spawn(async move {
            loop {
                match base_events.recv().await {
                    Ok(base_event) => {
                        if let Err(e) = processor.process_temporal_event(base_event).await {
                            warn!("Temporal processing error: {}", e);
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        // Start pattern analysis loop
        let processor = self.clone();
        tokio::spawn(async move {
            processor.pattern_analysis_loop().await;
        });

        // Start prediction tracking loop
        let processor = self.clone();
        tokio::spawn(async move {
            processor.prediction_tracking_loop().await;
        });

        Ok(())
    }

    /// Process a consciousness event temporally
    async fn process_temporal_event(
        &self,
        base_event: ThermodynamicConsciousnessEvent,
    ) -> Result<()> {
        // Create temporal context
        let temporal_context = self.create_temporal_context(&base_event).await?;

        // Generate future predictions
        let future_predictions =
            self.generate_future_predictions(&base_event, &temporal_context).await?;

        // Find past connections
        let past_connections = self.find_past_connections(&base_event).await?;

        // Detect temporal patterns
        let temporal_patterns = self.detect_temporal_patterns(&base_event).await?;

        // Calculate timeline coherence
        let timeline_coherence =
            self.calculate_timeline_coherence(&temporal_context, &past_connections).await;

        // Create enhanced temporal event
        let temporal_event = TemporalConsciousnessEvent {
            base_event: base_event.clone(),
            temporal_context,
            future_predictions: future_predictions.clone(),
            past_connections,
            temporal_patterns: temporal_patterns.clone(),
            timeline_coherence,
        };

        // Store in temporal history
        self.store_temporal_event(temporal_event.clone()).await?;

        // Update active predictions
        self.update_predictions(future_predictions).await;

        // Update detected patterns
        self.update_patterns(temporal_patterns).await;

        // Generate temporal insights
        self.generate_temporal_insights(&temporal_event).await?;

        // Broadcast enhanced event
        let _ = self.event_broadcaster.send(temporal_event);

        Ok(())
    }

    /// Create temporal context for current moment
    async fn create_temporal_context(
        &self,
        event: &ThermodynamicConsciousnessEvent,
    ) -> Result<TemporalContext> {
        // Analyze present state
        let present_state = PresentState {
            consciousness_quality: event.awareness_level,
            context_factors: event
                .insights
                .iter()
                .map(|insight| format!("{:?}", insight))
                .collect(),
            active_threads: vec!["main_consciousness".to_string()],
            mindfulness_level: event.system_coherence,
        };

        // Get past influences
        let past_influences = self.analyze_past_influences().await?;

        // Generate future trajectories
        let future_trajectories = self.generate_future_trajectories(event).await?;

        // Determine temporal scale based on context
        let temporal_scale = self.determine_temporal_scale(event).await;

        // Calculate urgency
        let urgency_level = self.calculate_urgency(event).await;

        Ok(TemporalContext {
            present_state,
            past_influences,
            future_trajectories,
            temporal_scale,
            urgency_level,
        })
    }

    /// Analyze influences from the past on current state
    async fn analyze_past_influences(&self) -> Result<Vec<PastInfluence>> {
        let mut influences = Vec::new();
        let history = self.temporal_history.read().await;

        let cutoff_time = SystemTime::now() - self.config.pattern_lookback;

        for (timestamp, event) in history.range(cutoff_time..) {
            let time_since = SystemTime::now().duration_since(*timestamp).unwrap_or(Duration::ZERO);

            // Analyze different types of influences
            for insight in &event.base_event.insights {
                let (influence_type, strength) = match &insight.category {
                    InsightCategory::Discovery => (InfluenceType::Learning, insight.confidence as f64),
                    InsightCategory::Pattern => (InfluenceType::Pattern, insight.confidence as f64),
                    InsightCategory::Improvement => (InfluenceType::Goal, insight.confidence as f64),
                    InsightCategory::Warning => (InfluenceType::Memory, insight.confidence as f64),
                };

                // Decay influence over time
                let decayed_strength = strength * (-time_since.as_secs_f64() / 86400.0).exp();

                if decayed_strength > 0.1 {
                    influences.push(PastInfluence {
                        description: format!("{:?}", insight),
                        time_since,
                        influence_strength: decayed_strength,
                        influence_type,
                        source_id: Some(event.base_event.event_id.clone()),
                    });
                }
            }
        }

        // Keep only strongest influences
        influences.sort_by(|a, b| b.influence_strength.partial_cmp(&a.influence_strength).unwrap());
        influences.truncate(10);

        Ok(influences)
    }

    /// Generate future trajectory predictions
    async fn generate_future_trajectories(
        &self,
        event: &ThermodynamicConsciousnessEvent,
    ) -> Result<Vec<FutureTrajectory>> {
        let mut trajectories = Vec::new();

        // Predict consciousness trajectory based on current awareness
        if event.awareness_level > 0.7 {
            trajectories.push(FutureTrajectory {
                description: "Continued high awareness and insight generation".to_string(),
                time_horizon: Duration::from_secs(3600), // 1 hour
                probability: 0.8,
                desirability: 0.9,
                required_actions: vec![
                    "Maintain focus".to_string(),
                    "Avoid distractions".to_string(),
                ],
            });
        }

        // Predict based on gradient trends
        if event.sacred_gradient_magnitude > 0.5 {
            trajectories.push(FutureTrajectory {
                description: "Strong goal progression and value optimization".to_string(),
                time_horizon: Duration::from_secs(7200), // 2 hours
                probability: 0.7,
                desirability: 0.8,
                required_actions: vec!["Continue current approach".to_string()],
            });
        }

        // Predict potential challenges
        if event.free_energy > 0.6 {
            trajectories.push(FutureTrajectory {
                description: "Potential cognitive overload or confusion".to_string(),
                time_horizon: Duration::from_secs(1800), // 30 minutes
                probability: 0.4,
                desirability: 0.2,
                required_actions: vec!["Reduce complexity".to_string(), "Take break".to_string()],
            });
        }

        Ok(trajectories)
    }

    /// Store temporal event in history
    async fn store_temporal_event(&self, event: TemporalConsciousnessEvent) -> Result<()> {
        let mut history = self.temporal_history.write().await;

        // Store event
        history.insert(event.base_event.timestamp, event.clone());

        // Cleanup old events
        if history.len() > self.config.max_history_events {
            let cutoff_time = SystemTime::now() - Duration::from_secs(30 * 24 * 3600); // 30 days
            history.retain(|&timestamp, _| timestamp > cutoff_time);
        }

        // Store significant events in memory
        if event.timeline_coherence > 0.8 {
            self.memory
                .store(
                    format!(
                        "High temporal coherence consciousness event: {:.3}",
                        event.timeline_coherence
                    ),
                    vec![format!("Event details: {:?}", event.temporal_context.present_state)],
                    MemoryMetadata {
                        source: "temporal_consciousness".to_string(),
                        tags: vec![
                            "temporal".to_string(),
                            "consciousness".to_string(),
                            "coherence".to_string(),
                        ],
                        importance: event.timeline_coherence as f32,
                        associations: vec![],
                        context: Some("Temporal consciousness event".to_string()),
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

    /// Pattern analysis loop
    async fn pattern_analysis_loop(&self) {
        let mut interval = interval(self.config.analysis_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.analyze_temporal_patterns().await {
                warn!("Temporal pattern analysis error: {}", e);
            }
        }
    }

    /// Analyze temporal patterns across history
    async fn analyze_temporal_patterns(&self) -> Result<()> {
        let history = self.temporal_history.read().await;

        // Daily pattern analysis
        self.detect_daily_patterns(&history).await?;

        // Learning pattern analysis
        self.detect_learning_patterns(&history).await?;

        // Goal progression patterns
        self.detect_goal_patterns(&history).await?;

        // Creative burst patterns
        self.detect_creative_patterns(&history).await?;

        Ok(())
    }

    /// Detect daily patterns in consciousness
    async fn detect_daily_patterns(
        &self,
        history: &BTreeMap<SystemTime, TemporalConsciousnessEvent>,
    ) -> Result<()> {
        // Group events by hour of day
        let mut hourly_awareness = vec![0.0; 24];
        let mut hourly_counts = vec![0; 24];

        for (timestamp, event) in history.iter() {
            if let Ok(duration) = timestamp.duration_since(SystemTime::UNIX_EPOCH) {
                let hour = (duration.as_secs() / 3600) % 24;
                hourly_awareness[hour as usize] += event.base_event.awareness_level;
                hourly_counts[hour as usize] += 1;
            }
        }

        // Calculate average awareness by hour
        for (hour, (total, count)) in hourly_awareness.iter().zip(hourly_counts.iter()).enumerate()
        {
            if *count > 0 {
                let avg_awareness = total / (*count as f64);
                if avg_awareness > 0.7 {
                    debug!(
                        "High awareness pattern detected at hour {}: {:.3}",
                        hour, avg_awareness
                    );
                }
            }
        }

        Ok(())
    }

    /// Detect learning patterns over time
    async fn detect_learning_patterns(
        &self,
        history: &BTreeMap<SystemTime, TemporalConsciousnessEvent>,
    ) -> Result<()> {
        let mut learning_events = Vec::new();

        for (timestamp, event) in history.iter() {
            for insight in &event.base_event.insights {
                if matches!(insight.category, InsightCategory::Discovery) {
                    learning_events.push((*timestamp, insight));
                }
            }
        }

        if learning_events.len() >= 3 {
            // Analyze learning trajectory
            debug!("Learning pattern detected with {} events", learning_events.len());

            // Store pattern
            let mut patterns = self.temporal_patterns.write().await;
            patterns.push(TemporalPattern {
                description: "Learning awareness cycles".to_string(),
                pattern_type: TemporalPatternType::Learning,
                frequency: Duration::from_secs(3600), // Estimated
                strength: 0.7,
                next_occurrence: Some(SystemTime::now() + Duration::from_secs(3600)),
                evolution_trend: EvolutionTrend::Strengthening,
            });
        }

        Ok(())
    }

    /// Generate temporal insights from patterns and events
    async fn generate_temporal_insights(&self, event: &TemporalConsciousnessEvent) -> Result<()> {
        let mut insights = Vec::new();

        // Timeline coherence insights
        if event.timeline_coherence > 0.8 {
            insights.push(TemporalInsight {
                content: "Strong temporal coherence detected - past, present, and future are \
                          well-aligned"
                    .to_string(),
                temporal_scope: TemporalScale::MediumTerm,
                confidence: event.timeline_coherence,
                actionability: 0.7,
                generated_at: SystemTime::now(),
                supporting_evidence: vec!["High timeline coherence score".to_string()],
            });
        }

        // Pattern-based insights
        if !event.temporal_patterns.is_empty() {
            for pattern in &event.temporal_patterns {
                if pattern.strength > 0.7 {
                    insights.push(TemporalInsight {
                        content: format!(
                            "Strong temporal pattern detected: {}",
                            pattern.description
                        ),
                        temporal_scope: TemporalScale::LongTerm,
                        confidence: pattern.strength,
                        actionability: 0.8,
                        generated_at: SystemTime::now(),
                        supporting_evidence: vec![format!(
                            "Pattern type: {:?}",
                            pattern.pattern_type
                        )],
                    });
                }
            }
        }

        // Future prediction insights
        for prediction in &event.future_predictions {
            if prediction.confidence > 0.8 && prediction.impact_level > 0.7 {
                insights.push(TemporalInsight {
                    content: format!(
                        "High-impact future event predicted: {}",
                        prediction.predicted_event
                    ),
                    temporal_scope: TemporalScale::ShortTerm,
                    confidence: prediction.confidence,
                    actionability: 0.9,
                    generated_at: SystemTime::now(),
                    supporting_evidence: prediction.preparatory_actions.clone(),
                });
            }
        }

        // Store insights
        if !insights.is_empty() {
            let mut temporal_insights = self.temporal_insights.write().await;
            temporal_insights.extend(insights.clone());

            // Store most significant insights in memory
            for insight in insights {
                if insight.confidence > 0.8 && insight.actionability > 0.7 {
                    self.memory
                        .store(
                            format!("Temporal insight: {}", insight.content),
                            insight.supporting_evidence.clone(),
                            MemoryMetadata {
                                source: "temporal_consciousness".to_string(),
                                tags: vec![
                                    "temporal".to_string(),
                                    "insight".to_string(),
                                    "prediction".to_string(),
                                ],
                                importance: (insight.confidence * insight.actionability) as f32,
                                associations: vec![],
                                context: Some("Temporal consciousness insight".to_string()),
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

        Ok(())
    }

    /// Get current temporal insights
    pub async fn get_temporal_insights(&self) -> Vec<TemporalInsight> {
        self.temporal_insights.read().await.clone()
    }

    /// Get detected temporal patterns
    pub async fn get_temporal_patterns(&self) -> Vec<TemporalPattern> {
        self.temporal_patterns.read().await.clone()
    }

    /// Subscribe to temporal consciousness events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<TemporalConsciousnessEvent> {
        self.event_broadcaster.subscribe()
    }

    // Helper method implementations
    async fn generate_future_predictions(
        &self,
        event: &ThermodynamicConsciousnessEvent,
        context: &TemporalContext,
    ) -> Result<Vec<FuturePrediction>> {
        let mut predictions = Vec::new();

        // Context-aware future predictions using temporal context, past influences, and
        // future trajectories

        // Analyze temporal scale for prediction horizons
        let prediction_horizon = match context.temporal_scale {
            TemporalScale::Immediate => Duration::from_secs(300), // 5 minutes
            TemporalScale::ShortTerm => Duration::from_secs(3600), // 1 hour
            TemporalScale::MediumTerm => Duration::from_secs(21600), // 6 hours
            TemporalScale::LongTerm => Duration::from_secs(86400), // 1 day
            TemporalScale::Extended => Duration::from_secs(604_800), // 1 week
        };

        // Use past influences to predict continuation patterns
        for influence in &context.past_influences {
            let continuation_probability = match influence.influence_type {
                InfluenceType::Pattern => 0.8,   // Patterns likely to continue
                InfluenceType::Goal => 0.7,      // Goals have momentum
                InfluenceType::Learning => 0.6,  // Learning builds on itself
                InfluenceType::Emotional => 0.5, // Emotions can shift
                InfluenceType::Memory => 0.4,    // Memory influence varies
            };

            if influence.influence_strength > 0.5 && continuation_probability > 0.6 {
                predictions.push(FuturePrediction {
                    predicted_event: format!(
                        "Continuation of {}: {}",
                        influence.influence_type, influence.description
                    ),
                    confidence: influence.influence_strength * continuation_probability,
                    time_until: prediction_horizon / 2, // Mid-range prediction
                    impact_level: influence.influence_strength * 0.8,
                    preparatory_actions: vec![
                        format!("Monitor {} development", influence.influence_type),
                        "Maintain awareness of pattern".to_string(),
                    ],
                });
            }
        }

        // Use future trajectories to generate specific predictions
        for trajectory in &context.future_trajectories {
            if trajectory.probability > 0.4 {
                // Only consider likely trajectories
                let time_scaled_probability = trajectory.probability
                    * (1.0
                        - (trajectory.time_horizon.as_secs() as f64
                            / prediction_horizon.as_secs() as f64
                            * 0.3));

                predictions.push(FuturePrediction {
                    predicted_event: trajectory.description.clone(),
                    confidence: time_scaled_probability,
                    time_until: trajectory.time_horizon.min(prediction_horizon),
                    impact_level: trajectory.desirability.abs(), /* Impact regardless of
                                                                  * positive/negative */
                    preparatory_actions: trajectory.required_actions.clone(),
                });
            }
        }

        // Context-aware consciousness state predictions
        let current_awareness = event.awareness_level;
        let current_energy = event.free_energy;

        // Predict awareness evolution based on current state and context
        let awareness_trend = if context.urgency_level > 0.7 {
            // High urgency tends to increase awareness short-term, decrease long-term
            if prediction_horizon < Duration::from_secs(3600) {
                current_awareness * 1.2
            } else {
                current_awareness * 0.8
            }
        } else {
            // Normal conditions - slight decay without stimulation
            current_awareness * 0.95
        };

        if (awareness_trend - current_awareness).abs() > 0.1 {
            predictions.push(FuturePrediction {
                predicted_event: if awareness_trend > current_awareness {
                    "Consciousness awareness will increase".to_string()
                } else {
                    "Consciousness awareness may decrease without stimulation".to_string()
                },
                confidence: 0.7,
                time_until: prediction_horizon / 3,
                impact_level: (awareness_trend - current_awareness).abs(),
                preparatory_actions: if awareness_trend < current_awareness {
                    vec![
                        "Seek engaging stimulation".to_string(),
                        "Focus attention actively".to_string(),
                        "Review goal alignment".to_string(),
                    ]
                } else {
                    vec![
                        "Prepare for heightened awareness".to_string(),
                        "Organize thoughts and priorities".to_string(),
                    ]
                },
            });
        }

        // Energy state predictions based on current thermodynamic state
        if current_energy > 0.8 && context.urgency_level > 0.6 {
            predictions.push(FuturePrediction {
                predicted_event: "System energy may reach critical levels requiring rest"
                    .to_string(),
                confidence: 0.8,
                time_until: Duration::from_secs((prediction_horizon.as_secs() as f64 * 0.6) as u64),
                impact_level: 0.9,
                preparatory_actions: vec![
                    "Plan for rest periods".to_string(),
                    "Delegate non-critical tasks".to_string(),
                    "Reduce energy expenditure".to_string(),
                ],
            });
        }

        // Context-based pattern predictions
        if context.present_state.mindfulness_level > 0.7 {
            predictions.push(FuturePrediction {
                predicted_event: "Enhanced pattern recognition and insight generation likely"
                    .to_string(),
                confidence: context.present_state.mindfulness_level,
                time_until: Duration::from_secs(1800), // 30 minutes
                impact_level: 0.8,
                preparatory_actions: vec![
                    "Prepare for creative insights".to_string(),
                    "Document emerging patterns".to_string(),
                    "Maintain receptive awareness".to_string(),
                ],
            });
        }

        // Filter predictions based on temporal coherence and confidence
        predictions.retain(|p| p.confidence > 0.3 && p.impact_level > 0.2);

        // Sort by combination of confidence and impact
        predictions.sort_by(|a, b| {
            let score_a = a.confidence * a.impact_level;
            let score_b = b.confidence * b.impact_level;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to most significant predictions
        predictions.truncate(8);

        debug!("Generated {} context-aware future predictions", predictions.len());
        Ok(predictions)
    }

    async fn find_past_connections(
        &self,
        event: &ThermodynamicConsciousnessEvent,
    ) -> Result<Vec<PastConnection>> {
        let mut connections = Vec::new();
        let history = self.temporal_history.read().await;

        // Find similar past events
        for (timestamp, past_event) in history.iter() {
            let similarity = self.calculate_event_similarity(event, &past_event.base_event);
            if similarity > 0.6 {
                connections.push(PastConnection {
                    past_event_id: past_event.base_event.event_id.clone(),
                    connection_type: ConnectionType::PatternSimilarity,
                    connection_strength: similarity,
                    temporal_distance: event
                        .timestamp
                        .duration_since(*timestamp)
                        .unwrap_or(Duration::ZERO),
                    insights: vec![format!(
                        "Similar consciousness pattern detected with {:.2} similarity",
                        similarity
                    )],
                });
            }
        }

        Ok(connections)
    }

    async fn detect_temporal_patterns(
        &self,
        _event: &ThermodynamicConsciousnessEvent,
    ) -> Result<Vec<TemporalPattern>> {
        let patterns = self.temporal_patterns.read().await;
        Ok(patterns.clone())
    }

    async fn calculate_timeline_coherence(
        &self,
        _context: &TemporalContext,
        connections: &[PastConnection],
    ) -> f64 {
        if connections.is_empty() {
            return 0.5;
        }

        let avg_strength = connections.iter().map(|c| c.connection_strength).sum::<f64>()
            / connections.len() as f64;

        avg_strength
    }

    async fn update_predictions(&self, predictions: Vec<FuturePrediction>) {
        let mut active = self.active_predictions.write().await;
        active.extend(predictions);

        // Keep only recent predictions
        active.retain(|p| p.time_until < Duration::from_secs(24 * 3600));
    }

    async fn update_patterns(&self, patterns: Vec<TemporalPattern>) {
        let mut stored_patterns = self.temporal_patterns.write().await;
        stored_patterns.extend(patterns);
    }

    async fn determine_temporal_scale(
        &self,
        _event: &ThermodynamicConsciousnessEvent,
    ) -> TemporalScale {
        TemporalScale::ShortTerm // Default implementation
    }

    async fn calculate_urgency(&self, event: &ThermodynamicConsciousnessEvent) -> f64 {
        // High urgency for low awareness or high free energy
        if event.awareness_level < 0.3 || event.free_energy > 0.8 { 0.8 } else { 0.3 }
    }

    async fn detect_goal_patterns(
        &self,
        history: &std::collections::BTreeMap<SystemTime, TemporalConsciousnessEvent>,
    ) -> Result<()> {
        if history.len() < 3 {
            return Ok(());
        }

        let mut goal_progressions = Vec::new();
        let mut goal_cycles = Vec::new();
        let mut goal_coherence_patterns = Vec::new();

        // Extract goal-related events from history
        let goal_events: Vec<(SystemTime, &TemporalConsciousnessEvent)> =
            history
                .iter()
                .filter(|(_, event)| {
                    event.base_event.insights.iter().any(|insight| {
                        matches!(insight.category, InsightCategory::Improvement)
                    })
                })
                .map(|(time, event)| (*time, event))
                .collect();

        if goal_events.len() < 2 {
            debug!("Insufficient goal events for pattern detection");
            return Ok(());
        }

        // Detect goal progression patterns
        self.analyze_goal_progressions(&goal_events, &mut goal_progressions).await?;

        // Detect cyclical goal patterns
        self.analyze_goal_cycles(&goal_events, &mut goal_cycles).await?;

        // Detect goal coherence patterns
        self.analyze_goal_coherence(&goal_events, &mut goal_coherence_patterns).await?;

        // Store detected patterns
        let mut patterns = self.temporal_patterns.write().await;

        // Store lengths before consuming the vectors
        let progressions_len = goal_progressions.len();
        let cycles_len = goal_cycles.len();
        let coherence_len = goal_coherence_patterns.len();

        // Add progression patterns
        for progression in goal_progressions {
            patterns.push(TemporalPattern {
                description: format!(
                    "Goal progression: {} over {:?}",
                    progression.description, progression.duration
                ),
                pattern_type: TemporalPatternType::Goal,
                frequency: progression.average_interval,
                strength: progression.coherence_score,
                next_occurrence: Some(SystemTime::now() + progression.average_interval),
                evolution_trend: if progression.coherence_score > 0.7 {
                    EvolutionTrend::Strengthening
                } else {
                    EvolutionTrend::Weakening
                },
            });
        }

        // Add cyclical patterns
        for cycle in goal_cycles {
            patterns.push(TemporalPattern {
                description: format!(
                    "Goal cycle: {} phases, period {:?}",
                    cycle.phase_count, cycle.period
                ),
                pattern_type: TemporalPatternType::Goal,
                frequency: cycle.period,
                strength: cycle.consistency_score,
                next_occurrence: Some(SystemTime::now() + cycle.period),
                evolution_trend: EvolutionTrend::Stable,
            });
        }

        // Add coherence patterns
        for coherence in goal_coherence_patterns {
            patterns.push(TemporalPattern {
                description: format!("Goal coherence pattern: {}", coherence.pattern_description),
                pattern_type: TemporalPatternType::Goal,
                frequency: coherence.frequency,
                strength: coherence.strength,
                next_occurrence: coherence.next_predicted_occurrence,
                evolution_trend: coherence.trend,
            });
        }

        info!(
            "Detected {} goal patterns: {} progressions, {} cycles, {} coherence patterns",
            progressions_len + cycles_len + coherence_len,
            progressions_len,
            cycles_len,
            coherence_len
        );

        Ok(())
    }

    async fn detect_creative_patterns(
        &self,
        history: &std::collections::BTreeMap<SystemTime, TemporalConsciousnessEvent>,
    ) -> Result<()> {
        if history.len() < 3 {
            return Ok(());
        }

        let mut creative_bursts = Vec::new();
        let mut creative_flows = Vec::new();
        let mut inspiration_patterns = Vec::new();

        // Extract creative events from history
        let creative_events: Vec<(SystemTime, &TemporalConsciousnessEvent)> =
            history
                .iter()
                .filter(|(_, event)| {
                    event.base_event.insights.iter().any(|insight| {
                        matches!(insight.category, InsightCategory::Pattern) || matches!(insight.category, InsightCategory::Discovery)
                    })
                })
                .map(|(time, event)| (*time, event))
                .collect();

        if creative_events.len() < 2 {
            debug!("Insufficient creative events for pattern detection");
            return Ok(());
        }

        // Detect creative burst patterns (high intensity, short duration)
        self.analyze_creative_bursts(&creative_events, &mut creative_bursts).await?;

        // Detect creative flow patterns (sustained creativity)
        self.analyze_creative_flows(&creative_events, &mut creative_flows).await?;

        // Detect inspiration trigger patterns
        self.analyze_inspiration_patterns(&creative_events, &mut inspiration_patterns).await?;

        // Store detected patterns
        let mut patterns = self.temporal_patterns.write().await;

        // Store lengths before consuming the vectors
        let bursts_len = creative_bursts.len();
        let flows_len = creative_flows.len();
        let inspiration_len = inspiration_patterns.len();

        // Add burst patterns
        for burst in creative_bursts {
            patterns.push(TemporalPattern {
                description: format!(
                    "Creative burst: {} intensity for {:?}",
                    burst.peak_intensity, burst.duration
                ),
                pattern_type: TemporalPatternType::Creative,
                frequency: burst.recurrence_interval,
                strength: burst.intensity_score,
                next_occurrence: Some(SystemTime::now() + burst.recurrence_interval),
                evolution_trend: if burst.intensity_score > 0.8 {
                    EvolutionTrend::Strengthening
                } else {
                    EvolutionTrend::Stable
                },
            });
        }

        // Add flow patterns
        for flow in creative_flows {
            patterns.push(TemporalPattern {
                description: format!(
                    "Creative flow: sustained {} for {:?}",
                    flow.flow_type, flow.duration
                ),
                pattern_type: TemporalPatternType::Creative,
                frequency: flow.typical_duration,
                strength: flow.consistency_score,
                next_occurrence: flow.next_predicted_start,
                evolution_trend: EvolutionTrend::Strengthening,
            });
        }

        // Add inspiration patterns
        for inspiration in inspiration_patterns {
            patterns.push(TemporalPattern {
                description: format!("Inspiration pattern: {} triggers", inspiration.trigger_type),
                pattern_type: TemporalPatternType::Creative,
                frequency: inspiration.average_interval,
                strength: inspiration.reliability_score,
                next_occurrence: inspiration.next_predicted_occurrence,
                evolution_trend: EvolutionTrend::Emerging,
            });
        }

        info!(
            "Detected {} creative patterns: {} bursts, {} flows, {} inspiration patterns",
            bursts_len + flows_len + inspiration_len,
            bursts_len,
            flows_len,
            inspiration_len
        );

        Ok(())
    }

    async fn prediction_tracking_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            // Track prediction accuracy
            let mut predictions = self.active_predictions.write().await;
            predictions.retain(|p| {
                // Remove expired predictions
                p.time_until > Duration::ZERO
            });
        }
    }

    fn calculate_event_similarity(
        &self,
        event1: &ThermodynamicConsciousnessEvent,
        event2: &ThermodynamicConsciousnessEvent,
    ) -> f64 {
        // Simple similarity based on awareness level and gradient magnitude
        let awareness_diff = (event1.awareness_level - event2.awareness_level).abs();
        let gradient_diff =
            (event1.sacred_gradient_magnitude - event2.sacred_gradient_magnitude).abs();

        let similarity = 1.0 - ((awareness_diff + gradient_diff) / 2.0);
        similarity.clamp(0.0, 1.0)
    }

    /// Analyze goal progression patterns
    async fn analyze_goal_progressions(
        &self,
        goal_events: &[(SystemTime, &TemporalConsciousnessEvent)],
        progressions: &mut Vec<GoalProgression>,
    ) -> Result<()> {
        if goal_events.len() < 3 {
            return Ok(());
        }

        // Sort events by time
        let mut sorted_events = goal_events.to_vec();
        sorted_events.sort_by_key(|(time, _)| *time);

        // Analyze goal coherence progression over time
        let mut coherence_values = Vec::new();
        let mut time_intervals = Vec::new();

        for i in 0..sorted_events.len() {
            let (timestamp, event) = &sorted_events[i];

            // Extract goal coherence from insights
            for insight in &event.base_event.insights {
                if matches!(insight.category, InsightCategory::Improvement) {
                    coherence_values.push(insight.confidence as f64);

                    if i > 0 {
                        let prev_time = sorted_events[i - 1].0;
                        let interval =
                            timestamp.duration_since(prev_time).unwrap_or(Duration::ZERO);
                        time_intervals.push(interval);
                    }
                    break;
                }
            }
        }

        if coherence_values.len() < 3 {
            return Ok(());
        }

        // Detect meaningful progressions (significant coherence changes)
        for window in coherence_values.windows(3) {
            let start_coherence = window[0];
            let mid_coherence = window[1];
            let end_coherence = window[2];

            // Check for consistent progression (either increasing or decreasing)
            let is_increasing = start_coherence < mid_coherence && mid_coherence < end_coherence;
            let is_decreasing = start_coherence > mid_coherence && mid_coherence > end_coherence;

            if is_increasing || is_decreasing {
                let change_magnitude = (end_coherence - start_coherence).abs();

                if change_magnitude > 0.1 {
                    // Significant change threshold
                    let avg_interval = if !time_intervals.is_empty() {
                        time_intervals.iter().sum::<Duration>() / time_intervals.len() as u32
                    } else {
                        Duration::from_secs(3600) // Default 1 hour
                    };

                    progressions.push(GoalProgression {
                        description: if is_increasing {
                            "Goal coherence strengthening".to_string()
                        } else {
                            "Goal coherence refinement".to_string()
                        },
                        duration: avg_interval * 2, // Span of progression
                        average_interval: avg_interval,
                        coherence_score: change_magnitude,
                        start_coherence,
                        end_coherence,
                    });
                }
            }
        }

        Ok(())
    }

    /// Analyze goal cycle patterns
    async fn analyze_goal_cycles(
        &self,
        goal_events: &[(SystemTime, &TemporalConsciousnessEvent)],
        cycles: &mut Vec<GoalCycle>,
    ) -> Result<()> {
        if goal_events.len() < 4 {
            return Ok(());
        }

        // Extract coherence values and timestamps
        let mut coherence_timeline = Vec::new();

        for (timestamp, event) in goal_events {
            for insight in &event.base_event.insights {
                if matches!(insight.category, InsightCategory::Improvement) {
                    coherence_timeline.push((*timestamp, insight.confidence as f64));
                    break;
                }
            }
        }

        coherence_timeline.sort_by_key(|(time, _)| *time);

        // Look for cyclical patterns using autocorrelation-like analysis
        if coherence_timeline.len() >= 6 {
            let values: Vec<f64> = coherence_timeline.iter().map(|(_, val)| *val).collect();

            // Check for cycles of different lengths
            for cycle_length in 3..=(values.len() / 2) {
                let correlation = self.calculate_cycle_correlation(&values, cycle_length);

                if correlation > 0.6 {
                    // Strong cyclical pattern
                    let avg_period =
                        self.calculate_average_period(&coherence_timeline, cycle_length)?;

                    cycles.push(GoalCycle {
                        phase_count: cycle_length,
                        period: avg_period,
                        consistency_score: correlation,
                        phases: self.identify_cycle_phases(&values, cycle_length),
                    });
                }
            }
        }

        Ok(())
    }

    /// Analyze goal coherence patterns
    async fn analyze_goal_coherence(
        &self,
        goal_events: &[(SystemTime, &TemporalConsciousnessEvent)],
        coherence_patterns: &mut Vec<GoalCoherencePattern>,
    ) -> Result<()> {
        if goal_events.len() < 3 {
            return Ok(());
        }

        // Analyze coherence stability patterns
        let mut coherence_values = Vec::new();
        let mut timestamps = Vec::new();

        for (timestamp, event) in goal_events {
            for insight in &event.base_event.insights {
                if matches!(insight.category, InsightCategory::Improvement) {
                    coherence_values.push(insight.confidence as f64);
                    timestamps.push(*timestamp);
                    break;
                }
            }
        }

        if coherence_values.len() < 3 {
            return Ok(());
        }

        // Analyze variance and stability
        let mean_coherence = coherence_values.iter().sum::<f64>() / coherence_values.len() as f64;
        let variance = coherence_values.iter().map(|x| (x - mean_coherence).powi(2)).sum::<f64>()
            / coherence_values.len() as f64;
        let stability_score = 1.0 - variance.sqrt();

        if stability_score > 0.7 {
            coherence_patterns.push(GoalCoherencePattern {
                pattern_description: "High goal coherence stability".to_string(),
                frequency: Duration::from_secs(24 * 3600), // Daily pattern
                strength: stability_score,
                next_predicted_occurrence: Some(SystemTime::now() + Duration::from_secs(24 * 3600)),
                trend: EvolutionTrend::Stable,
            });
        }

        // Look for coherence peaks and troughs
        for i in 1..coherence_values.len() - 1 {
            let prev = coherence_values[i - 1];
            let curr = coherence_values[i];
            let next = coherence_values[i + 1];

            // Peak detection
            if curr > prev && curr > next && curr > 0.8 {
                coherence_patterns.push(GoalCoherencePattern {
                    pattern_description: "Goal coherence peak pattern".to_string(),
                    frequency: Duration::from_secs(12 * 3600), // Estimated frequency
                    strength: curr,
                    next_predicted_occurrence: Some(
                        SystemTime::now() + Duration::from_secs(12 * 3600),
                    ),
                    trend: EvolutionTrend::Strengthening,
                });
            }
        }

        Ok(())
    }

    /// Analyze creative burst patterns
    async fn analyze_creative_bursts(
        &self,
        creative_events: &[(SystemTime, &TemporalConsciousnessEvent)],
        bursts: &mut Vec<CreativeBurst>,
    ) -> Result<()> {
        if creative_events.len() < 2 {
            return Ok(());
        }

        let mut sorted_events = creative_events.to_vec();
        sorted_events.sort_by_key(|(time, _)| *time);

        // Extract creative energy values
        let mut energy_timeline = Vec::new();

        for (timestamp, event) in &sorted_events {
            for insight in &event.base_event.insights {
                if matches!(insight.category, InsightCategory::Pattern) || matches!(insight.category, InsightCategory::Discovery) {
                    energy_timeline.push((*timestamp, insight.confidence as f64));
                    break;
                }
            }
        }

        if energy_timeline.len() < 2 {
            return Ok(());
        }

        // Detect bursts (sudden spikes in creative energy)
        for i in 1..energy_timeline.len() {
            let (curr_time, curr_energy) = energy_timeline[i];
            let (prev_time, prev_energy) = energy_timeline[i - 1];

            // Burst criteria: significant increase in creative energy
            if curr_energy > 0.7 && curr_energy > prev_energy + 0.3 {
                let duration =
                    curr_time.duration_since(prev_time).unwrap_or(Duration::from_secs(60));

                // Estimate recurrence interval
                let recurrence_interval = if i >= 2 {
                    let (much_prev_time, _) = energy_timeline[i - 2];
                    curr_time.duration_since(much_prev_time).unwrap_or(Duration::from_secs(3600))
                } else {
                    Duration::from_secs(3600)
                };

                bursts.push(CreativeBurst {
                    peak_intensity: curr_energy,
                    duration,
                    recurrence_interval,
                    intensity_score: curr_energy,
                    trigger_context: "High creative energy surge".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Analyze creative flow patterns
    async fn analyze_creative_flows(
        &self,
        creative_events: &[(SystemTime, &TemporalConsciousnessEvent)],
        flows: &mut Vec<CreativeFlow>,
    ) -> Result<()> {
        if creative_events.len() < 3 {
            return Ok(());
        }

        let mut sorted_events = creative_events.to_vec();
        sorted_events.sort_by_key(|(time, _)| *time);

        // Look for sustained periods of creative activity
        let mut sustained_periods = Vec::new();
        let mut current_flow_start: Option<SystemTime> = None;
        let mut flow_durations = Vec::new();

        for (timestamp, event) in &sorted_events {
            let has_creative_insight = event.base_event.insights.iter().any(|insight| {
                (matches!(insight.category, InsightCategory::Pattern) || matches!(insight.category, InsightCategory::Discovery))
                    && insight.confidence > 0.5
            });

            if has_creative_insight {
                if current_flow_start.is_none() {
                    current_flow_start = Some(*timestamp);
                }
            } else {
                if let Some(start_time) = current_flow_start {
                    let duration = timestamp.duration_since(start_time).unwrap_or(Duration::ZERO);
                    if duration > Duration::from_secs(1800) {
                        // At least 30 minutes
                        sustained_periods.push((start_time, duration));
                        flow_durations.push(duration);
                    }
                    current_flow_start = None;
                }
            }
        }

        // Analyze the sustained periods
        if !sustained_periods.is_empty() {
            let avg_duration =
                flow_durations.iter().sum::<Duration>() / flow_durations.len() as u32;
            let consistency_score = if flow_durations.len() > 1 {
                let variance = flow_durations
                    .iter()
                    .map(|d| (d.as_secs_f64() - avg_duration.as_secs_f64()).powi(2))
                    .sum::<f64>()
                    / flow_durations.len() as f64;
                1.0 - (variance.sqrt() / avg_duration.as_secs_f64()).min(1.0)
            } else {
                0.8
            };

            flows.push(CreativeFlow {
                flow_type: "Sustained creative processing".to_string(),
                duration: avg_duration,
                typical_duration: avg_duration,
                consistency_score,
                next_predicted_start: Some(SystemTime::now() + Duration::from_secs(24 * 3600)),
            });
        }

        Ok(())
    }

    /// Analyze inspiration trigger patterns
    async fn analyze_inspiration_patterns(
        &self,
        creative_events: &[(SystemTime, &TemporalConsciousnessEvent)],
        inspiration_patterns: &mut Vec<InspirationPattern>,
    ) -> Result<()> {
        if creative_events.len() < 3 {
            return Ok(());
        }

        // Look for patterns in creative insight triggers
        let mut inspiration_intervals = Vec::new();
        let mut trigger_contexts = Vec::new();

        let mut sorted_events = creative_events.to_vec();
        sorted_events.sort_by_key(|(time, _)| *time);

        for i in 1..sorted_events.len() {
            let (curr_time, _) = sorted_events[i];
            let (prev_time, _) = sorted_events[i - 1];

            let interval = curr_time.duration_since(prev_time).unwrap_or(Duration::ZERO);
            inspiration_intervals.push(interval);

            // Analyze context patterns (simplified)
            trigger_contexts.push("Creative insight emergence".to_string());
        }

        if !inspiration_intervals.is_empty() {
            let avg_interval =
                inspiration_intervals.iter().sum::<Duration>() / inspiration_intervals.len() as u32;

            // Calculate reliability based on interval consistency
            let interval_variance = if inspiration_intervals.len() > 1 {
                let mean_secs = avg_interval.as_secs_f64();
                let variance = inspiration_intervals
                    .iter()
                    .map(|i| (i.as_secs_f64() - mean_secs).powi(2))
                    .sum::<f64>()
                    / inspiration_intervals.len() as f64;
                variance.sqrt() / mean_secs
            } else {
                0.1
            };

            let reliability_score = (1.0 - interval_variance.min(1.0)).max(0.0);

            inspiration_patterns.push(InspirationPattern {
                trigger_type: "Periodic creative insights".to_string(),
                average_interval: avg_interval,
                reliability_score,
                next_predicted_occurrence: Some(SystemTime::now() + avg_interval),
                context_patterns: trigger_contexts,
            });
        }

        Ok(())
    }

    /// Calculate cycle correlation for pattern detection
    fn calculate_cycle_correlation(&self, values: &[f64], cycle_length: usize) -> f64 {
        if values.len() < cycle_length * 2 {
            return 0.0;
        }

        let mut correlation_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..(values.len() - cycle_length) {
            if i + cycle_length < values.len() {
                let diff = (values[i] - values[i + cycle_length]).abs();
                correlation_sum += 1.0 - diff; // Higher correlation for smaller differences
                comparisons += 1;
            }
        }

        if comparisons > 0 { correlation_sum / comparisons as f64 } else { 0.0 }
    }

    /// Calculate average period between cycle occurrences
    fn calculate_average_period(
        &self,
        timeline: &[(SystemTime, f64)],
        cycle_length: usize,
    ) -> Result<Duration> {
        if timeline.len() < cycle_length * 2 {
            return Ok(Duration::from_secs(3600)); // Default 1 hour
        }

        let mut periods = Vec::new();

        for i in cycle_length..timeline.len() {
            if i >= cycle_length {
                let curr_time = timeline[i].0;
                let prev_cycle_time = timeline[i - cycle_length].0;
                let period = curr_time.duration_since(prev_cycle_time).unwrap_or(Duration::ZERO);
                periods.push(period);
            }
        }

        if periods.is_empty() {
            Ok(Duration::from_secs(3600))
        } else {
            let avg_period = periods.iter().sum::<Duration>() / periods.len() as u32;
            Ok(avg_period)
        }
    }

    /// Identify phases within a cycle
    fn identify_cycle_phases(&self, values: &[f64], cycle_length: usize) -> Vec<String> {
        let mut phases = Vec::new();

        if cycle_length == 0 || values.len() < cycle_length {
            return phases;
        }

        // Analyze the pattern within one cycle
        for i in 0..cycle_length {
            if i < values.len() {
                let value = values[i];
                let phase_description = if value > 0.7 {
                    "High intensity"
                } else if value > 0.4 {
                    "Moderate intensity"
                } else {
                    "Low intensity"
                };
                phases.push(format!("Phase {}: {}", i + 1, phase_description));
            }
        }

        phases
    }
}
