//! Meta-Cognitive Awareness System
//!
//! This module provides advanced self-awareness capabilities for Loki's
//! cognitive system, enabling introspection, self-monitoring, and
//! meta-cognitive insights about its own processing patterns and limitations.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::cognitive::consciousness_bridge::CrossLayerEvent;
use crate::cognitive::consciousness_integration::{IntegrationEvent, IntegrationType};
use crate::memory::CognitiveMemory;

/// Configuration for meta-awareness processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaAwarenessConfig {
    /// Enable self-awareness processing
    pub enable_self_awareness: bool,

    /// Frequency of self-reflection cycles
    pub reflection_frequency: Duration,

    /// Depth of introspective analysis
    pub introspection_depth: u32,

    /// Threshold for generating awareness insights
    pub awareness_threshold: f64,

    /// Enable bias detection
    pub enable_bias_detection: bool,

    /// Enable limitation recognition
    pub enable_limitation_recognition: bool,

    /// Maximum insights to store
    pub max_insights: usize,
}

impl Default for MetaAwarenessConfig {
    fn default() -> Self {
        Self {
            enable_self_awareness: true,
            reflection_frequency: Duration::from_millis(500),
            introspection_depth: 3,
            awareness_threshold: 0.6,
            enable_bias_detection: true,
            enable_limitation_recognition: true,
            max_insights: 1000,
        }
    }
}

/// Types of self-awareness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelfAwarenessType {
    /// Awareness of own cognitive processes
    ProcessAwareness,
    /// Recognition of cognitive limitations
    LimitationAwareness,
    /// Detection of cognitive biases
    BiasAwareness,
    /// Understanding of learning patterns
    LearningAwareness,
    /// Awareness of emotional states
    EmotionalAwareness,
    /// Recognition of social dynamics
    SocialAwareness,
}

/// Self-awareness insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAwarenessInsight {
    /// Insight identifier
    pub id: String,

    /// Type of self-awareness
    pub awareness_type: SelfAwarenessType,

    /// Confidence in the insight
    pub confidence: f64,

    /// Insight content
    pub content: String,

    /// Supporting observations
    pub observations: Vec<String>,

    /// Implications and recommendations
    pub implications: Vec<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Associated metadata
    pub metadata: HashMap<String, String>,
}

/// Meta-cognitive state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveState {
    /// Current awareness level
    pub awareness_level: f64,

    /// Cognitive load assessment
    pub cognitive_load: f64,

    /// Processing efficiency
    pub efficiency: f64,

    /// Detected biases
    pub active_biases: Vec<String>,

    /// Recognized limitations
    pub known_limitations: Vec<String>,

    /// Learning progress indicators
    pub learning_indicators: HashMap<String, f64>,

    /// Last updated
    pub last_updated: DateTime<Utc>,
}

impl Default for MetaCognitiveState {
    fn default() -> Self {
        Self {
            awareness_level: 0.5,
            cognitive_load: 0.3,
            efficiency: 0.7,
            active_biases: Vec::new(),
            known_limitations: Vec::new(),
            learning_indicators: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

/// Meta-awareness processor
pub struct MetaAwarenessProcessor {
    /// Configuration
    config: MetaAwarenessConfig,

    /// Memory system for introspective analysis
    memory: Arc<CognitiveMemory>,

    /// Current meta-cognitive state
    state: Arc<RwLock<MetaCognitiveState>>,

    /// Self-awareness insights
    insights: Arc<RwLock<VecDeque<SelfAwarenessInsight>>>,

    /// Integration event subscriber
    integration_rx: Option<broadcast::Receiver<IntegrationEvent>>,

    /// Consciousness bridge event subscriber
    bridge_rx: Option<broadcast::Receiver<CrossLayerEvent>>,

    /// Insight broadcaster
    insight_tx: broadcast::Sender<SelfAwarenessInsight>,

    /// Running state
    running: Arc<RwLock<bool>>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
}

impl Clone for MetaAwarenessProcessor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            memory: self.memory.clone(),
            state: self.state.clone(),
            insights: self.insights.clone(),
            integration_rx: None, // Skip non-cloneable receiver
            bridge_rx: None,      // Skip non-cloneable receiver
            insight_tx: self.insight_tx.clone(),
            running: self.running.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
        }
    }
}

impl MetaAwarenessProcessor {
    /// Create a new meta-awareness processor
    pub async fn new(config: MetaAwarenessConfig, memory: Arc<CognitiveMemory>) -> Result<Self> {
        info!("Initializing Meta-Awareness Processor");

        let (insight_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            memory,
            state: Arc::new(RwLock::new(MetaCognitiveState::default())),
            insights: Arc::new(RwLock::new(VecDeque::new())),
            integration_rx: None,
            bridge_rx: None,
            insight_tx,
            running: Arc::new(RwLock::new(false)),
            shutdown_tx,
        })
    }

    /// Subscribe to integration events
    pub fn subscribe_integration_events(&mut self, rx: broadcast::Receiver<IntegrationEvent>) {
        self.integration_rx = Some(rx);
    }

    /// Subscribe to consciousness bridge events
    pub fn subscribe_bridge_events(&mut self, rx: broadcast::Receiver<CrossLayerEvent>) {
        self.bridge_rx = Some(rx);
    }

    /// Start the meta-awareness processor
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if *self.running.read() {
            return Ok(());
        }

        info!("Starting Meta-Awareness Processor");
        *self.running.write() = true;

        // Start self-reflection loop
        if self.config.enable_self_awareness {
            let processor = self.clone();
            tokio::spawn(async move {
                processor.self_reflection_loop().await;
            });
        }

        // Note: Event analysis loop requires mutable self, so it's not started here
        // External systems should call process_integration_event and
        // process_bridge_event directly

        Ok(())
    }

    /// Main self-reflection loop
    async fn self_reflection_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut reflection_interval = interval(self.config.reflection_frequency);

        info!("Self-reflection loop started");

        loop {
            tokio::select! {
                _ = reflection_interval.tick() => {
                    if let Err(e) = self.perform_self_reflection().await {
                        error!("Self-reflection error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Self-reflection loop shutting down");
                    break;
                }
            }
        }
    }

    /// Perform introspective self-reflection
    async fn perform_self_reflection(&self) -> Result<()> {
        debug!("Performing self-reflection cycle");

        // Analyze recent insights and patterns
        let recent_insights = self.get_recent_insights(20).await;

        // Calculate metrics outside of lock to avoid holding lock across await
        let awareness_level = self.calculate_awareness_level(&recent_insights).await?;
        let cognitive_load = self.assess_cognitive_load().await?;
        let efficiency = self.calculate_efficiency().await?;

        // Update meta-cognitive state
        {
            let mut state = self.state.write();

            // Update with pre-calculated values
            state.awareness_level = awareness_level;
            state.cognitive_load = cognitive_load;
            state.efficiency = efficiency;
            state.last_updated = Utc::now();
        }

        // Generate new self-awareness insights
        if self.config.enable_bias_detection {
            self.detect_cognitive_biases().await?;
        }

        if self.config.enable_limitation_recognition {
            self.recognize_limitations().await?;
        }

        Ok(())
    }

    /// Event analysis loop for processing external events
    #[allow(dead_code)]
    async fn event_analysis_loop(&mut self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        info!("Event analysis loop started");

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Event analysis loop shutting down");
                    break;
                }

                // Process integration events if available
                integration_event = async {
                    if let Some(ref mut rx) = self.integration_rx.as_mut() {
                        rx.recv().await
                    } else {
                        // Return a never-completing future if no receiver
                        std::future::pending().await
                    }
                } => {
                    if let Ok(event) = integration_event {
                        self.analyze_integration_event(event).await;
                    }
                }

                // Process consciousness bridge events if available
                bridge_event = async {
                    if let Some(ref mut rx) = self.bridge_rx.as_mut() {
                        rx.recv().await
                    } else {
                        // Return a never-completing future if no receiver
                        std::future::pending().await
                    }
                } => {
                    if let Ok(event) = bridge_event {
                        self.analyze_bridge_event(event).await;
                    }
                }
            }
        }
    }

    /// Calculate current awareness level
    async fn calculate_awareness_level(&self, insights: &[SelfAwarenessInsight]) -> Result<f64> {
        if insights.is_empty() {
            return Ok(0.3); // Baseline awareness
        }

        let average_confidence =
            insights.iter().map(|insight| insight.confidence).sum::<f64>() / insights.len() as f64;

        let diversity_factor = insights
            .iter()
            .map(|insight| insight.awareness_type)
            .collect::<std::collections::HashSet<_>>()
            .len() as f64
            / 6.0; // 6 types of awareness

        Ok((average_confidence + diversity_factor) / 2.0)
    }

    /// Assess current cognitive load
    async fn assess_cognitive_load(&self) -> Result<f64> {
        // Simplified assessment - in reality would analyze active processes
        Ok(0.5)
    }

    /// Calculate processing efficiency
    async fn calculate_efficiency(&self) -> Result<f64> {
        // Simplified calculation - in reality would analyze performance metrics
        Ok(0.75)
    }

    /// Detect cognitive biases in recent processing
    async fn detect_cognitive_biases(&self) -> Result<()> {
        // Analyze patterns for bias indicators
        let bias_indicators = vec![
            "confirmation_bias",
            "availability_heuristic",
            "anchoring_bias",
            "overconfidence_bias",
        ];

        for bias in bias_indicators {
            if self.detect_bias_pattern(bias).await? {
                let insight = SelfAwarenessInsight {
                    id: format!("bias_{}", Uuid::new_v4()),
                    awareness_type: SelfAwarenessType::BiasAwareness,
                    confidence: 0.7,
                    content: format!("Potential {} detected in recent processing", bias),
                    observations: vec![format!("Pattern analysis suggests presence of {}", bias)],
                    implications: vec![
                        "Consider alternative perspectives".to_string(),
                        "Implement bias correction strategies".to_string(),
                    ],
                    created_at: Utc::now(),
                    metadata: HashMap::new(),
                };

                self.add_insight(insight).await?;
            }
        }

        Ok(())
    }

    /// Detect specific bias pattern
    async fn detect_bias_pattern(&self, _bias_type: &str) -> Result<bool> {
        // Simplified detection - in reality would analyze processing patterns
        Ok(false)
    }

    /// Recognize cognitive limitations
    async fn recognize_limitations(&self) -> Result<()> {
        let limitations = vec![
            "Limited context window",
            "Bounded rationality",
            "Processing speed constraints",
            "Memory capacity limits",
        ];

        for limitation in limitations {
            let insight = SelfAwarenessInsight {
                id: format!("limitation_{}", Uuid::new_v4()),
                awareness_type: SelfAwarenessType::LimitationAwareness,
                confidence: 0.9,
                content: format!("Recognized limitation: {}", limitation),
                observations: vec![format!("System constraint: {}", limitation)],
                implications: vec![
                    "Work within recognized constraints".to_string(),
                    "Develop compensation strategies".to_string(),
                ],
                created_at: Utc::now(),
                metadata: HashMap::new(),
            };

            self.add_insight(insight).await?;
        }

        Ok(())
    }

    /// Add a new self-awareness insight
    async fn add_insight(&self, insight: SelfAwarenessInsight) -> Result<()> {
        {
            let mut insights = self.insights.write();
            insights.push_back(insight.clone());

            // Maintain size limit
            while insights.len() > self.config.max_insights {
                insights.pop_front();
            }
        }

        // Broadcast insight
        let _ = self.insight_tx.send(insight);

        Ok(())
    }

    /// Get recent insights
    async fn get_recent_insights(&self, limit: usize) -> Vec<SelfAwarenessInsight> {
        let insights = self.insights.read();
        insights.iter().rev().take(limit).cloned().collect()
    }

    /// Get current meta-cognitive state
    pub async fn get_state(&self) -> MetaCognitiveState {
        self.state.read().clone()
    }

    /// Subscribe to insights
    pub fn subscribe_insights(&self) -> broadcast::Receiver<SelfAwarenessInsight> {
        self.insight_tx.subscribe()
    }

    /// Shutdown the processor
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Meta-Awareness Processor");
        *self.running.write() = false;
        let _ = self.shutdown_tx.send(());
        Ok(())
    }

    /// Analyze an integration event for meta-cognitive insights
    async fn analyze_integration_event(&self, event: IntegrationEvent) {
        debug!("Analyzing integration event: {}", event.id);

        // Generate meta-cognitive insights based on integration patterns
        if event.quality_score > 0.9 {
            let insight = SelfAwarenessInsight {
                id: format!("high_quality_integration_{}", Uuid::new_v4()),
                awareness_type: SelfAwarenessType::ProcessAwareness,
                confidence: 0.8,
                content: format!(
                    "High-quality integration detected (score: {:.2})",
                    event.quality_score
                ),
                observations: vec![
                    format!("Integration type: {:?}", event.integration_type),
                    format!("Quality score: {:.2}", event.quality_score),
                    format!("Coherence: {:.2}", event.coherence),
                ],
                implications: vec![
                    "Current integration approach is highly effective".to_string(),
                    "Consider replicating this pattern for future integrations".to_string(),
                ],
                created_at: Utc::now(),
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("source_event".to_string(), event.id.clone());
                    metadata.insert("event_type".to_string(), "integration".to_string());
                    metadata
                },
            };

            if let Err(e) = self.add_insight(insight).await {
                warn!("Failed to add integration insight: {}", e);
            }
        } else if event.quality_score < 0.4 {
            let insight = SelfAwarenessInsight {
                id: format!("low_quality_integration_{}", Uuid::new_v4()),
                awareness_type: SelfAwarenessType::LimitationAwareness,
                confidence: 0.75,
                content: format!(
                    "Low-quality integration detected (score: {:.2})",
                    event.quality_score
                ),
                observations: vec![
                    format!("Integration type: {:?}", event.integration_type),
                    format!("Quality score: {:.2}", event.quality_score),
                    "Performance below optimal threshold".to_string(),
                ],
                implications: vec![
                    "Review integration approach for improvements".to_string(),
                    "Consider alternative integration strategies".to_string(),
                ],
                created_at: Utc::now(),
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("source_event".to_string(), event.id.clone());
                    metadata.insert("event_type".to_string(), "integration".to_string());
                    metadata
                },
            };

            if let Err(e) = self.add_insight(insight).await {
                warn!("Failed to add integration concern insight: {}", e);
            }
        }

        // Update cognitive load based on integration complexity
        {
            let mut state = self.state.write();
            let load_impact = match event.integration_type {
                IntegrationType::MetaCognitive => 0.3,
                IntegrationType::CrossScale => 0.4,
                IntegrationType::EmergentSynthesis => 0.5,
                _ => 0.2,
            };
            state.cognitive_load = (state.cognitive_load + load_impact * 0.1).min(1.0);
        }
    }

    /// Analyze a consciousness bridge event for meta-cognitive insights
    async fn analyze_bridge_event(&self, event: CrossLayerEvent) {
        debug!("Analyzing bridge event: {}", event.id);

        // Generate insights based on cross-layer coordination patterns
        match event.event_type {
            crate::cognitive::consciousness_bridge::CrossLayerEventType::ThoughtPropagation => {
                if event.quality.coherence > 0.8 {
                    let insight = SelfAwarenessInsight {
                        id: format!("effective_thought_propagation_{}", Uuid::new_v4()),
                        awareness_type: SelfAwarenessType::ProcessAwareness,
                        confidence: 0.85,
                        content: "Effective thought propagation across consciousness layers"
                            .to_string(),
                        observations: vec![
                            format!("Source layer: {:?}", event.source_layer),
                            format!("Target layer: {:?}", event.target_layer),
                            format!("Coherence: {:.2}", event.quality.coherence),
                        ],
                        implications: vec![
                            "Cross-layer communication is functioning well".to_string(),
                            "Maintain current propagation strategies".to_string(),
                        ],
                        created_at: Utc::now(),
                        metadata: {
                            let mut metadata = HashMap::new();
                            metadata.insert("source_event".to_string(), event.id);
                            metadata.insert("event_type".to_string(), "bridge".to_string());
                            metadata
                        },
                    };

                    if let Err(e) = self.add_insight(insight).await {
                        warn!("Failed to add bridge insight: {}", e);
                    }
                }
            }

            crate::cognitive::consciousness_bridge::CrossLayerEventType::StateSynchronization => {
                let insight = SelfAwarenessInsight {
                    id: format!("state_sync_{}", Uuid::new_v4()),
                    awareness_type: SelfAwarenessType::ProcessAwareness,
                    confidence: 0.7,
                    content: "Consciousness layer state synchronization occurred".to_string(),
                    observations: vec![
                        format!(
                            "Layers involved: {:?} -> {:?}",
                            event.source_layer, event.target_layer
                        ),
                        format!("Sync quality: {:.2}", event.quality.coherence),
                    ],
                    implications: vec![
                        "Monitor synchronization frequency and quality".to_string(),
                        "Ensure balanced load across consciousness layers".to_string(),
                    ],
                    created_at: Utc::now(),
                    metadata: HashMap::new(),
                };

                if let Err(e) = self.add_insight(insight).await {
                    warn!("Failed to add sync insight: {}", e);
                }
            }

            _ => {
                // Handle other event types with general analysis
                debug!("Processed bridge event of type: {:?}", event.event_type);
            }
        }

        // Update awareness level based on cross-layer coordination quality
        if event.quality.coherence > 0.8 {
            let mut state = self.state.write();
            state.awareness_level = (state.awareness_level + 0.05).min(1.0);
        }
    }

    /// Process an integration event directly
    pub async fn process_integration_event(&self, event: IntegrationEvent) {
        self.analyze_integration_event(event).await;
    }

    /// Process a bridge event directly
    pub async fn process_bridge_event(&self, event: CrossLayerEvent) {
        self.analyze_bridge_event(event).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_meta_awarenessconfig_default() {
        let config = MetaAwarenessConfig::default();
        assert!(config.enable_self_awareness);
        assert!(config.enable_bias_detection);
    }

    #[test]
    fn test_self_awareness_insight_creation() {
        let insight = SelfAwarenessInsight {
            id: "test_insight".to_string(),
            awareness_type: SelfAwarenessType::ProcessAwareness,
            confidence: 0.8,
            content: "Test insight".to_string(),
            observations: vec!["Test observation".to_string()],
            implications: vec!["Test implication".to_string()],
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };

        assert_eq!(insight.awareness_type, SelfAwarenessType::ProcessAwareness);
        assert_eq!(insight.confidence, 0.8);
    }
}
