//! Streaming Consciousness Bridge
//!
//! This module provides a bridge between the streaming system and the consciousness
//! orchestrator, enabling real-time cognitive processing of streaming data.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info};

use crate::streaming::enhanced_context_processor::{EnhancedContext, EnhancedContextProcessor};
use crate::streaming::{StreamChunk, ProcessedInput};

/// Configuration for streaming consciousness bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConsciousnessBridgeConfig {
    /// Enable real-time consciousness integration
    pub enable_real_time_integration: bool,

    /// Consciousness activation threshold (0.0 to 1.0)
    pub consciousness_threshold: f32,

    /// Maximum processing latency in milliseconds
    pub max_processing_latency_ms: u64,

    /// Buffer size for consciousness events
    pub event_buffer_size: usize,

    /// Enable consciousness feedback to streaming
    pub enable_consciousness_feedback: bool,

    /// Consciousness update interval
    pub consciousness_update_interval: Duration,
}

impl Default for StreamingConsciousnessBridgeConfig {
    fn default() -> Self {
        Self {
            enable_real_time_integration: true,
            consciousness_threshold: 0.7,
            max_processing_latency_ms: 100,
            event_buffer_size: 1000,
            enable_consciousness_feedback: true,
            consciousness_update_interval: Duration::from_millis(500),
        }
    }
}

/// Bridge between streaming system and consciousness orchestrator
pub struct StreamingConsciousnessBridge {
    /// Configuration
    config: StreamingConsciousnessBridgeConfig,


    /// Enhanced context processor for cognitive integration
    context_processor: Arc<EnhancedContextProcessor>,

    /// Consciousness event broadcaster
    consciousness_events: broadcast::Sender<ConsciousnessEvent>,

    /// Processing metrics
    metrics: Arc<RwLock<BridgeMetrics>>,

    /// Active processing sessions
    active_sessions: Arc<RwLock<HashMap<String, ProcessingSession>>>,
}

/// Consciousness events from streaming integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessEvent {
    /// Stream chunk processed through consciousness
    StreamChunkProcessed {
        chunk_id: String,
        stream_id: String,
        awareness_level: f64,
        coherence_score: f64,
        insights: Vec<String>,
    },

    /// Consciousness state updated from streaming
    ConsciousnessStateUpdated {
        previous_awareness: f64,
        new_awareness: f64,
        trigger_source: String,
    },

    /// Pattern detected in streaming data
    PatternDetected {
        pattern_type: String,
        confidence: f32,
        stream_context: String,
    },

    /// Consciousness feedback to streaming
    ConsciousnessFeedback {
        feedback_type: FeedbackType,
        target_stream: String,
        adjustment_factor: f32,
    },
}

/// Types of consciousness feedback to streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Increase attention to this stream
    IncreaseAttention,
    /// Decrease attention to this stream
    DecreaseAttention,
    /// Adjust processing parameters
    AdjustProcessing { parameter: String, value: f32 },
    /// Request deeper analysis
    RequestDeepAnalysis,
}

/// Processing session for consciousness integration
#[derive(Debug, Clone)]
pub struct ProcessingSession {
    /// Session identifier
    pub session_id: String,

    /// Stream identifier
    pub stream_id: String,

    /// Session start time
    pub started_at: Instant,

    /// Chunks processed in this session
    pub chunks_processed: usize,

    /// Current awareness level
    pub awareness_level: f64,

    /// Session insights
    pub insights: Vec<String>,
}

/// Metrics for consciousness bridge
#[derive(Debug, Default, Clone)]
pub struct BridgeMetrics {
    /// Total chunks processed
    pub total_chunks_processed: u64,

    /// Consciousness integrations performed
    pub consciousness_integrations: u64,

    /// Average processing latency
    pub avg_processing_latency_ms: f32,

    /// Successful integrations
    pub successful_integrations: u64,

    /// Failed integrations
    pub failed_integrations: u64,

    /// Current awareness level
    pub current_awareness_level: f64,

    /// Consciousness state updates
    pub consciousness_state_updates: u64,
}

impl StreamingConsciousnessBridge {
    /// Create a new streaming consciousness bridge
    pub async fn new(
        config: StreamingConsciousnessBridgeConfig,
        context_processor: Arc<EnhancedContextProcessor>,
    ) -> Result<Self> {
        info!("🌊 Initializing Streaming Consciousness Bridge");
        let (consciousness_events, _) = broadcast::channel(config.event_buffer_size);

        let bridge = Self {
            config,
            context_processor,
            consciousness_events,
            metrics: Arc::new(RwLock::new(BridgeMetrics::default())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        };

        info!("✅ Streaming Consciousness Bridge initialized");
        Ok(bridge)
    }



    /// Update processing metrics
    async fn update_metrics(&self, processing_time: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_chunks_processed += 1;

        if success {
            metrics.successful_integrations += 1;
            metrics.consciousness_integrations += 1;
        } else {
            metrics.failed_integrations += 1;
        }

        let latency_ms = processing_time.as_millis() as f32;
        metrics.avg_processing_latency_ms =
            (metrics.avg_processing_latency_ms + latency_ms) / 2.0;

    }


    /// Get processing metrics
    pub async fn get_metrics(&self) -> BridgeMetrics {
        self.metrics.read().await.clone()
    }

    /// Subscribe to consciousness events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ConsciousnessEvent> {
        self.consciousness_events.subscribe()
    }

    /// Start a processing session
    pub async fn start_session(&self, stream_id: String) -> String {
        let session_id = format!("session_{}_{}", stream_id, Instant::now().elapsed().as_millis());
        let session = ProcessingSession {
            session_id: session_id.clone(),
            stream_id,
            started_at: Instant::now(),
            chunks_processed: 0,
            awareness_level: 0.0,
            insights: Vec::new(),
        };

        self.active_sessions.write().await.insert(session_id.clone(), session);
        info!("🎯 Started consciousness processing session: {}", session_id);
        session_id
    }

    /// End a processing session
    pub async fn end_session(&self, session_id: &str) -> Option<ProcessingSession> {
        let session = self.active_sessions.write().await.remove(session_id);
        if let Some(ref s) = session {
            info!("✅ Ended consciousness processing session: {} (processed {} chunks)",
                  session_id, s.chunks_processed);
        }
        session
    }
}

/// Result of consciousness integration
#[derive(Debug)]
pub struct ConsciousnessIntegrationResult {
    /// Whether integration was successful
    pub success: bool,

    /// Enhanced context used
    pub enhanced_context: Option<EnhancedContext>,

    /// Processing time
    pub processing_time: Duration,

    /// Reason for failure/skip
    pub reason: Option<String>,
}

impl ConsciousnessIntegrationResult {
    /// Create a successful result
    pub fn success(
        enhanced_context: EnhancedContext,
        processing_time: Duration,
    ) -> Self {
        Self {
            success: true,
            enhanced_context: Some(enhanced_context),
            processing_time,
            reason: None,
        }
    }

    /// Create a skipped result
    pub fn skipped(reason: String) -> Self {
        Self {
            success: false,
            enhanced_context: None,
            processing_time: Duration::from_millis(0),
            reason: Some(reason),
        }
    }

    /// Create a failed result
    pub fn failed(reason: String, processing_time: Duration) -> Self {
        Self {
            success: false,
            enhanced_context: None,
            processing_time,
            reason: Some(reason),
        }
    }
}
