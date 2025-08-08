//! Cognitive Data Stream Manager
//!
//! Manages real-time data flow from cognitive systems to UI panels

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::sync::{RwLock, broadcast, mpsc};
use tracing::info;

use crate::cognitive::consciousness_stream::ConsciousnessInsight;
use crate::tui::cognitive::integration::main::CognitiveModality;

/// Maximum history size for each data type
const MAX_HISTORY_SIZE: usize = 100;
const MAX_REASONING_STEPS: usize = 50;
const MAX_PROCESSING_STEPS: usize = 100;

/// Types of cognitive processing steps
#[derive(Debug, Clone)]
pub enum ProcessingStepType {
    /// Memory retrieval operation
    MemoryRetrieval,
    /// Pattern recognition
    PatternRecognition,
    /// Concept formation
    ConceptFormation,
    /// Reasoning chain construction
    ReasoningConstruction,
    /// Tool selection
    ToolSelection,
    /// Response generation
    ResponseGeneration,
    /// Self-reflection
    SelfReflection,
    /// Context analysis
    ContextAnalysis,
    /// Goal evaluation
    GoalEvaluation,
    /// Creative synthesis
    CreativeSynthesis,
}

/// Cognitive data update types
#[derive(Debug, Clone)]
pub enum CognitiveDataUpdate {
    /// New reasoning step
    ReasoningStep { chain_id: String, timestamp: Instant },

    /// Consciousness activity update
    CognitiveActivity {
        awareness_level: f64,
        coherence: f64,
        free_energy: f64,
        current_focus: String,
    },

    /// New insight generated
    Insight { insight: ConsciousnessInsight, relevance: f64, novelty: f64 },

    /// Cognitive processing status
    ProcessingStatus {
        is_processing: bool,
        processing_depth: f64,
        active_tasks: Vec<String>,
        resource_usage: f64,
    },

    /// Background thought
    BackgroundThought { content: String, category: String, importance: f64 },

    /// Modality activation
    ModalityActivation { modality: CognitiveModality, activation_level: f64, context: String },

    /// Learning event
    LearningEvent { topic: String, understanding_delta: f64, key_realization: String },

    /// Detailed processing step
    ProcessingStep {
        step_type: ProcessingStepType,
        description: String,
        input_summary: String,
        output_summary: String,
        duration_ms: u64,
        timestamp: Instant,
    },
}

/// Aggregated cognitive state for UI display
#[derive(Debug, Clone, Default)]
pub struct CognitiveDisplayState {
    /// Current reasoning chain (if active)
    pub active_reasoning: Option<ActiveReasoningDisplay>,

    /// Consciousness metrics
    pub consciousness_metrics: ConsciousnessMetrics,

    /// Recent insights
    pub recent_insights: VecDeque<InsightDisplay>,

    /// Active modalities with levels
    pub modality_activations: Vec<(CognitiveModality, f64)>,

    /// Processing status
    pub processing_status: ProcessingStatusDisplay,

    /// Background thoughts
    pub background_thoughts: VecDeque<BackgroundThoughtDisplay>,

    /// Learning progress
    pub learning_events: VecDeque<LearningDisplay>,

    /// Processing steps history
    pub processing_steps: VecDeque<ProcessingStepDisplay>,
}

/// Active reasoning display data
#[derive(Debug, Clone)]
pub struct ActiveReasoningDisplay {
    pub chain_id: String,
    pub current_step: String,
    pub step_number: usize,
    pub total_steps: usize,
    pub confidence: f64,
    pub recent_steps: VecDeque<ReasoningStepDisplay>,
}

/// Reasoning step for display
#[derive(Debug, Clone)]
pub struct ReasoningStepDisplay {
    pub content: String,
    pub step_type: String,
    pub confidence: f64,
    pub timestamp: Instant,
}

/// Consciousness metrics for display
#[derive(Debug, Clone, Default)]
pub struct ConsciousnessMetrics {
    pub awareness_level: f64,
    pub coherence: f64,
    pub free_energy: f64,
    pub current_focus: String,
    pub awareness_history: VecDeque<(Instant, f64)>,
    pub coherence_history: VecDeque<(Instant, f64)>,
}

/// Insight for display
#[derive(Debug, Clone)]
pub struct InsightDisplay {
    pub content: String,
    pub category: String,
    pub relevance: f64,
    pub novelty: f64,
    pub timestamp: Instant,
}

/// Processing status for display
#[derive(Debug, Clone, Default)]
pub struct ProcessingStatusDisplay {
    pub is_processing: bool,
    pub processing_depth: f64,
    pub active_tasks: Vec<String>,
    pub resource_usage: f64,
    pub resource_history: VecDeque<(Instant, f64)>,
}

/// Background thought for display
#[derive(Debug, Clone)]
pub struct BackgroundThoughtDisplay {
    pub content: String,
    pub category: String,
    pub importance: f64,
    pub timestamp: Instant,
}

/// Learning event for display
#[derive(Debug, Clone)]
pub struct LearningDisplay {
    pub topic: String,
    pub understanding_delta: f64,
    pub key_realization: String,
    pub timestamp: Instant,
}

/// Processing step for display
#[derive(Debug, Clone)]
pub struct ProcessingStepDisplay {
    pub step_type: ProcessingStepType,
    pub description: String,
    pub input_summary: String,
    pub output_summary: String,
    pub duration_ms: u64,
    pub timestamp: Instant,
}

/// Cognitive data stream manager
pub struct CognitiveDataStream {
    /// Current display state
    display_state: Arc<RwLock<CognitiveDisplayState>>,

    /// Update receiver
    update_rx: Arc<RwLock<mpsc::Receiver<CognitiveDataUpdate>>>,

    /// Update sender (for producers)
    update_tx: mpsc::Sender<CognitiveDataUpdate>,

    /// State change broadcaster
    state_change_tx: broadcast::Sender<()>,

    /// Processing task handle
    processing_handle: Option<tokio::task::JoinHandle<()>>,

    /// Active state
    is_active: Arc<RwLock<bool>>,
}

impl CognitiveDataStream {
    /// Create new cognitive data stream
    pub fn new() -> Self {
        let (update_tx, update_rx) = mpsc::channel(100);
        let (state_change_tx, _) = broadcast::channel(10);

        Self {
            display_state: Arc::new(RwLock::new(CognitiveDisplayState::default())),
            update_rx: Arc::new(RwLock::new(update_rx)),
            update_tx,
            state_change_tx,
            processing_handle: None,
            is_active: Arc::new(RwLock::new(false)),
        }
    }

    /// Get update sender for producers
    pub fn get_update_sender(&self) -> mpsc::Sender<CognitiveDataUpdate> {
        self.update_tx.clone()
    }

    /// Subscribe to state changes
    pub fn subscribe_state_changes(&self) -> broadcast::Receiver<()> {
        self.state_change_tx.subscribe()
    }

    /// Get current display state
    pub async fn get_display_state(&self) -> CognitiveDisplayState {
        self.display_state.read().await.clone()
    }

    /// Start processing updates
    pub async fn start(&mut self) -> Result<()> {
        if *self.is_active.read().await {
            return Ok(());
        }

        info!("Starting cognitive data stream processing");
        *self.is_active.write().await = true;

        let display_state = self.display_state.clone();
        let update_rx = self.update_rx.clone();
        let state_change_tx = self.state_change_tx.clone();
        let is_active = self.is_active.clone();

        self.processing_handle = Some(tokio::spawn(async move {
            process_updates(display_state, update_rx, state_change_tx, is_active).await;
        }));

        Ok(())
    }

    /// Stop processing
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping cognitive data stream");
        *self.is_active.write().await = false;

        if let Some(handle) = self.processing_handle.take() {
            handle.abort();
        }

        Ok(())
    }
}

/// Process cognitive data updates
async fn process_updates(
    display_state: Arc<RwLock<CognitiveDisplayState>>,
    update_rx: Arc<RwLock<mpsc::Receiver<CognitiveDataUpdate>>>,
    state_change_tx: broadcast::Sender<()>,
    is_active: Arc<RwLock<bool>>,
) {
    let mut rx = update_rx.write().await;

    while *is_active.read().await {
        match rx.recv().await {
            Some(update) => {
                let mut state = display_state.write().await;

                match update {
                    CognitiveDataUpdate::ReasoningStep { chain_id, timestamp } => {
                        update_reasoning_display(&mut state, chain_id, timestamp);
                    }

                    CognitiveDataUpdate::CognitiveActivity {
                        awareness_level,
                        coherence,
                        free_energy,
                        current_focus,
                    } => {
                        update_consciousness_metrics(
                            &mut state,
                            awareness_level,
                            coherence,
                            free_energy,
                            current_focus,
                        );
                    }

                    CognitiveDataUpdate::Insight { insight, relevance, novelty } => {
                        add_insight(&mut state, insight, relevance, novelty);
                    }

                    CognitiveDataUpdate::ProcessingStatus {
                        is_processing,
                        processing_depth,
                        active_tasks,
                        resource_usage,
                    } => {
                        update_processing_status(
                            &mut state,
                            is_processing,
                            processing_depth,
                            active_tasks,
                            resource_usage,
                        );
                    }

                    CognitiveDataUpdate::BackgroundThought { content, category, importance } => {
                        add_background_thought(&mut state, content, category, importance);
                    }

                    CognitiveDataUpdate::ModalityActivation {
                        modality,
                        activation_level,
                        context,
                    } => {
                        update_modality_activation(&mut state, modality, activation_level);
                    }

                    CognitiveDataUpdate::LearningEvent {
                        topic,
                        understanding_delta,
                        key_realization,
                    } => {
                        add_learning_event(&mut state, topic, understanding_delta, key_realization);
                    }

                    CognitiveDataUpdate::ProcessingStep {
                        step_type,
                        description,
                        input_summary,
                        output_summary,
                        duration_ms,
                        timestamp,
                    } => {
                        add_processing_step(
                            &mut state,
                            step_type,
                            description,
                            input_summary,
                            output_summary,
                            duration_ms,
                            timestamp,
                        );
                    }
                }

                // Notify UI of state change
                let _ = state_change_tx.send(());
            }
            None => {
                // Channel closed
                break;
            }
        }
    }
}

/// Update reasoning display
fn update_reasoning_display(
    state: &mut CognitiveDisplayState,
    chain_id: String,
    timestamp: Instant,
) {
    // Create or update active reasoning
    if state.active_reasoning.is_none()
        || state.active_reasoning.as_ref().unwrap().chain_id != chain_id
    {
        state.active_reasoning = Some(ActiveReasoningDisplay {
            chain_id,
            current_step: String::new(),
            step_number: 1,
            total_steps: 1,
            confidence: 0.0,
            recent_steps: VecDeque::new(),
        });
    }

    if let Some(active) = &mut state.active_reasoning {
        // Add step to history
        active.recent_steps.push_back(ReasoningStepDisplay {
            content: String::new(),
            step_type: format!("{:?}", String::new()),
            confidence: 0.0,
            timestamp,
        });

        // Keep only recent steps
        while active.recent_steps.len() > MAX_REASONING_STEPS {
            active.recent_steps.pop_front();
        }

        // Update current step
        active.current_step = String::new();
        active.step_number += 1;
        active.confidence = 0.0;
    }
}

/// Update consciousness metrics
fn update_consciousness_metrics(
    state: &mut CognitiveDisplayState,
    awareness_level: f64,
    coherence: f64,
    free_energy: f64,
    current_focus: String,
) {
    let now = Instant::now();
    let metrics = &mut state.consciousness_metrics;

    metrics.awareness_level = awareness_level;
    metrics.coherence = coherence;
    metrics.free_energy = free_energy;
    metrics.current_focus = current_focus;

    // Update history
    metrics.awareness_history.push_back((now, awareness_level));
    metrics.coherence_history.push_back((now, coherence));

    // Trim history
    while metrics.awareness_history.len() > MAX_HISTORY_SIZE {
        metrics.awareness_history.pop_front();
    }
    while metrics.coherence_history.len() > MAX_HISTORY_SIZE {
        metrics.coherence_history.pop_front();
    }
}

/// Add insight to display
fn add_insight(
    state: &mut CognitiveDisplayState,
    insight: ConsciousnessInsight,
    relevance: f64,
    novelty: f64,
) {
    let display_insight = match insight {
        ConsciousnessInsight::GoalAwareness { active_goals, .. } => InsightDisplay {
            content: format!("Goal awareness: {} active goals", active_goals.len()),
            category: "Goals".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
        ConsciousnessInsight::LearningAwareness { knowledge_gained, .. } => InsightDisplay {
            content: format!("Learning: {}", knowledge_gained),
            category: "Learning".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
        ConsciousnessInsight::SocialAwareness { harmony_state, .. } => InsightDisplay {
            content: format!("Social harmony: {}", harmony_state),
            category: "Social".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
        ConsciousnessInsight::CreativeInsight { pattern_discovered, .. } => InsightDisplay {
            content: format!("Creative pattern: {}", pattern_discovered),
            category: "Creative".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
        ConsciousnessInsight::SelfReflection { self_model_update, .. } => InsightDisplay {
            content: format!("Self-reflection: {}", self_model_update),
            category: "Self".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
        ConsciousnessInsight::ThermodynamicAwareness { entropy_management, .. } => InsightDisplay {
            content: format!("Entropy management: {:.1}%", entropy_management * 100.0),
            category: "Thermodynamic".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
        ConsciousnessInsight::TemporalAwareness { future_planning, .. } => InsightDisplay {
            content: format!("Temporal: {}", future_planning),
            category: "Temporal".to_string(),
            relevance,
            novelty,
            timestamp: Instant::now(),
        },
    };

    state.recent_insights.push_back(display_insight);

    // Keep only recent insights
    while state.recent_insights.len() > 20 {
        state.recent_insights.pop_front();
    }
}

/// Update processing status
fn update_processing_status(
    state: &mut CognitiveDisplayState,
    is_processing: bool,
    processing_depth: f64,
    active_tasks: Vec<String>,
    resource_usage: f64,
) {
    let now = Instant::now();
    let status = &mut state.processing_status;

    status.is_processing = is_processing;
    status.processing_depth = processing_depth;
    status.active_tasks = active_tasks;
    status.resource_usage = resource_usage;

    // Update history
    status.resource_history.push_back((now, resource_usage));

    // Trim history
    while status.resource_history.len() > MAX_HISTORY_SIZE {
        status.resource_history.pop_front();
    }
}

/// Add background thought
fn add_background_thought(
    state: &mut CognitiveDisplayState,
    content: String,
    category: String,
    importance: f64,
) {
    state.background_thoughts.push_back(BackgroundThoughtDisplay {
        content,
        category,
        importance,
        timestamp: Instant::now(),
    });

    // Keep only recent thoughts
    while state.background_thoughts.len() > 10 {
        state.background_thoughts.pop_front();
    }
}

/// Update modality activation
fn update_modality_activation(
    state: &mut CognitiveDisplayState,
    modality: CognitiveModality,
    activation_level: f64,
) {
    // Update or add modality
    if let Some(existing) = state.modality_activations.iter_mut().find(|(m, _)| m == &modality) {
        existing.1 = activation_level;
    } else {
        state.modality_activations.push((modality, activation_level));
    }

    // Sort by activation level
    state.modality_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
}

/// Add learning event
fn add_learning_event(
    state: &mut CognitiveDisplayState,
    topic: String,
    understanding_delta: f64,
    key_realization: String,
) {
    state.learning_events.push_back(LearningDisplay {
        topic,
        understanding_delta,
        key_realization,
        timestamp: Instant::now(),
    });

    // Keep only recent events
    while state.learning_events.len() > 10 {
        state.learning_events.pop_front();
    }
}

/// Add processing step
fn add_processing_step(
    state: &mut CognitiveDisplayState,
    step_type: ProcessingStepType,
    description: String,
    input_summary: String,
    output_summary: String,
    duration_ms: u64,
    timestamp: Instant,
) {
    state.processing_steps.push_back(ProcessingStepDisplay {
        step_type,
        description,
        input_summary,
        output_summary,
        duration_ms,
        timestamp,
    });

    // Keep only recent steps
    while state.processing_steps.len() > MAX_PROCESSING_STEPS {
        state.processing_steps.pop_front();
    }
}
