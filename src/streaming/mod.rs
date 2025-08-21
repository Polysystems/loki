use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use dashmap::DashMap;
use tokio::sync::broadcast;
use tracing::{info, warn};

pub mod buffer;
pub mod enhanced_context_processor;
pub mod lockfree_enhanced_context_processor;
pub mod pipeline;
pub mod processor;
pub mod consciousness_bridge;

pub use buffer::{RingBuffer, StreamBuffer};
pub use pipeline::{PipelineConfig, ProcessedInput, ProcessingMode, StreamPipeline};
pub use processor::{ProcessorConfig, StreamProcessor};
pub use consciousness_bridge::StreamingConsciousnessBridge;

/// Stream type identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StreamId(pub String);

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stream data chunk
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamChunk {
    pub stream_id: StreamId,
    pub sequence: u64,
    #[serde(skip, default = "Instant::now")]  // Instant can't be serialized directly
    pub timestamp: Instant,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

/// Stream processor type enumeration
#[derive(Debug, Clone)]
pub enum StreamProcessorType {
    /// Cognitive processor
    Cognitive { purpose: String },
    /// Default processor
    Default,
}

/// Stream configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub name: String,
    pub buffer_size: usize,
    pub max_latency: Duration,
    pub processors: Vec<StreamProcessorType>,
}

/// Legacy stream configuration for compatibility
#[derive(Debug, Clone)]
pub struct LegacyStreamConfig {
    pub id: StreamId,
    pub model_id: String,
    pub device_affinity: Option<String>,
    pub buffer_size: usize,
    pub chunk_size: usize,
    pub max_latency_ms: u64,
}


#[derive(Clone, Debug)]
/// Stream manager for handling multiple concurrent streams (lock-free)
pub struct StreamManager {
    streams: Arc<DashMap<StreamId, StreamHandle>>,
    pipelines: Arc<DashMap<StreamId, Arc<StreamPipeline>>>,
    event_bus: broadcast::Sender<StreamEvent>,
    config: crate::config::Config,
    model_loader: Arc<crate::models::ModelLoader>,
    compute_manager: Arc<crate::compute::ComputeManager>,
}

#[derive(Debug)]
struct StreamHandle {
    config: LegacyStreamConfig,
    input_sender: Sender<StreamChunk>,
    output_receiver: Receiver<StreamChunk>,
    stats: StreamStats,
}

#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub chunks_processed: u64,
    pub bytes_processed: u64,
    pub errors: u64,
    pub avg_latency_ms: f64,
    pub last_update: Option<Instant>,
}

/// Stream events
#[derive(Debug, Clone)]
pub enum StreamEvent {
    StreamStarted(StreamId),
    StreamStopped(StreamId),
    ChunkProcessed(StreamId, Duration),
    Error(StreamId, String),
}

impl StreamManager {
    /// Create a new stream manager with proper configuration
    pub fn new(config: crate::config::Config) -> Result<Self> {
        let (event_sender, _) = broadcast::channel(1000);
        let model_loader = Arc::new(crate::models::ModelLoader::new(config.clone())?);
        let compute_manager = Arc::new(crate::compute::ComputeManager::new()?);

        Ok(Self {
            streams: Arc::new(DashMap::new()),
            pipelines: Arc::new(DashMap::new()),
            event_bus: event_sender,
            config,
            model_loader,
            compute_manager,
        })
    }

    /// Create a new stream with proper model loader integration
    pub async fn create_stream(&self, config: StreamConfig) -> Result<String> {
        let stream_id = StreamId(config.name.clone());

        // Create channels
        let (input_tx, input_rx) = bounded(config.buffer_size);
        let (output_tx, output_rx) = bounded(config.buffer_size);

        // Create pipeline with proper configuration
        let pipelineconfig = PipelineConfig {
            model_id: self.config.models.default_model.clone(),
            device_affinity: self.config.device.preferred_device.clone(),
            chunk_size: self.config.streaming.chunk_size.unwrap_or(1024),
            enable_context_memory: self.config.memory.enable_context_memory,
            context_window_size: self.config.memory.context_window_size,
            processing_mode: match self.config.streaming.processing_mode.as_deref() {
                Some("fast") => crate::streaming::pipeline::ProcessingMode::Fast,
                Some("quality") => crate::streaming::pipeline::ProcessingMode::HighQuality,
                Some("balanced") => crate::streaming::pipeline::ProcessingMode::Balanced,
                _ => crate::streaming::pipeline::ProcessingMode::Balanced,
            },
            quality_threshold: self.config.streaming.quality_threshold.unwrap_or(0.8),
            enable_consciousness_integration: true,
            consciousness_threshold: 0.7,
            enable_enhanced_context: true,
        };

        let pipeline = Arc::new(
            StreamPipeline::new(
                pipelineconfig,
                self.model_loader.clone(),
                self.compute_manager.clone(),
                None, // Memory integration to be added later
            )
            .await?,
        );

        // Start processing loop
        let pipeline_clone = pipeline.clone();
        let stream_id_clone = stream_id.clone();
        let event_bus = self.event_bus.clone();

        tokio::spawn(async move {
            stream_processing_loop(stream_id_clone, input_rx, output_tx, pipeline_clone, event_bus)
                .await;
        });

        // Create legacy config for handle
        let legacyconfig = LegacyStreamConfig {
            id: stream_id.clone(),
            model_id: self.config.models.default_model.clone(),
            device_affinity: self.config.device.preferred_device.clone(),
            buffer_size: config.buffer_size,
            chunk_size: self.config.streaming.chunk_size.unwrap_or(1024),
            max_latency_ms: config.max_latency.as_millis() as u64,
        };

        // Store stream handle
        let handle = StreamHandle {
            config: legacyconfig,
            input_sender: input_tx,
            output_receiver: output_rx,
            stats: StreamStats::default(),
        };

        self.streams.insert(stream_id.clone(), handle);
        self.pipelines.insert(stream_id.clone(), pipeline);

        // Emit event
        let _ = self.event_bus.send(StreamEvent::StreamStarted(stream_id.clone()));

        info!("Created stream: {:?}", stream_id);
        Ok(config.name)
    }

    /// Send data to a stream
    pub fn send(&self, stream_id: &StreamId, chunk: StreamChunk) -> Result<()> {
        let handle = self.streams
            .get(stream_id)
            .ok_or_else(|| anyhow::anyhow!("Stream not found: {:?}", stream_id))?;

        handle.input_sender.send(chunk)?;
        Ok(())
    }

    /// Receive processed data from a stream
    pub fn receive(&self, stream_id: &StreamId) -> Result<Option<StreamChunk>> {
        let handle = self.streams
            .get(stream_id)
            .ok_or_else(|| anyhow::anyhow!("Stream not found: {:?}", stream_id))?;

        Ok(handle.output_receiver.try_recv().ok())
    }

    /// Stop a stream
    pub async fn stop_stream(&self, stream_id: &StreamId) -> Result<()> {
        self.streams.remove(stream_id);
        self.pipelines.remove(stream_id);

        let _ = self.event_bus.send(StreamEvent::StreamStopped(stream_id.clone()));

        info!("Stopped stream: {:?}", stream_id);
        Ok(())
    }

    /// Get stream statistics
    pub fn stream_stats(&self, stream_id: &StreamId) -> Option<StreamStats> {
        self.streams.get(stream_id).map(|handle| handle.stats.clone())
    }

    /// Subscribe to stream events
    pub fn subscribe(&self) -> broadcast::Receiver<StreamEvent> {
        self.event_bus.subscribe()
    }

    /// List active streams
    pub fn list_streams(&self) -> Vec<(StreamId, LegacyStreamConfig)> {
        self.streams.iter().map(|entry| (entry.key().clone(), entry.value().config.clone())).collect()
    }

    /// Create a session-specific stream
    pub async fn create_session_stream(&self, session_id: &str) -> Result<String> {
        let streamconfig = StreamConfig {
            name: format!("session_{}", session_id),
            buffer_size: 1000,
            max_latency: Duration::from_millis(100),
            processors: vec![StreamProcessorType::Cognitive {
                purpose: "session_processing".to_string(),
            }],
        };

        self.create_stream(streamconfig).await
    }

    /// Clean up streams for a session
    pub async fn cleanup_session_streams(&self, session_id: &str) -> Result<()> {
        let stream_name = format!("session_{}", session_id);
        let stream_id = StreamId(stream_name);

        if self.streams.contains_key(&stream_id) {
            self.stop_stream(&stream_id).await?;
        }

        Ok(())
    }
}

impl Default for StreamManager {
    fn default() -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let config = crate::config::Config::default();
        
        Self {
            streams: Arc::new(DashMap::new()),
            pipelines: Arc::new(DashMap::new()),
            event_bus: event_sender,
            config: config.clone(),
            model_loader: Arc::new(crate::models::ModelLoader::new(config.clone()).unwrap_or_else(|_| {
                // Create a minimal model loader for fallback with default config
                crate::models::ModelLoader::new(crate::config::Config::default())
                    .expect("Default model loader should always work")
            })),
            compute_manager: Arc::new(crate::compute::ComputeManager::default()),
        }
    }
}

/// Stream processing loop
async fn stream_processing_loop(
    stream_id: StreamId,
    input: Receiver<StreamChunk>,
    output: Sender<StreamChunk>,
    pipeline: Arc<StreamPipeline>,
    event_bus: broadcast::Sender<StreamEvent>,
) {
    info!("Starting stream processing loop for {:?}", stream_id);

    while let Ok(chunk) = input.recv() {
        let start = Instant::now();

        match pipeline.process(chunk.clone()).await {
            Ok(processed) => {
                let duration = start.elapsed();

                // Send processed chunk
                if let Err(e) = output.send(processed) {
                    warn!("Failed to send processed chunk: {}", e);
                    break;
                }

                // Emit event
                let _ = event_bus.send(StreamEvent::ChunkProcessed(stream_id.clone(), duration));
            }
            Err(e) => {
                warn!("Error processing chunk for stream {:?}: {}", stream_id, e);
                let _ = event_bus.send(StreamEvent::Error(stream_id.clone(), e.to_string()));
            }
        }
    }

    info!("Stream processing loop ended for {:?}", stream_id);
}

/// Optimized model cluster with enum-based dispatch for better performance (lock-free)
pub struct ModelCluster {
    // Use enum types for zero-cost dispatch in hot paths
    models: Arc<DashMap<String, ModelInstanceType>>,
    // Keep trait objects for extensibility (non-hot path)
    trait_models: Arc<DashMap<String, Arc<dyn ModelInstance>>>,
    routing_table: Arc<DashMap<String, Vec<String>>>,
}

/// Optimized enum-based model instance for reduced dispatch overhead
#[derive(Clone)]
pub enum ModelInstanceType {
    Local {
        id: String,
        device_id: String,
        max_batch_size: usize,
        ready: bool,
    },
    Remote {
        id: String,
        device_id: String,
        max_batch_size: usize,
        ready: bool,
        endpoint: String,
    },
    Distributed {
        id: String,
        device_id: String,
        max_batch_size: usize,
        ready: bool,
        nodes: Vec<String>,
    },
}

impl ModelInstanceType {
    #[inline] // Hot path method for frequent ID lookups
    pub fn id(&self) -> &str {
        match self {
            Self::Local { id, .. } => id,
            Self::Remote { id, .. } => id,
            Self::Distributed { id, .. } => id,
        }
    }

    #[inline] // Hot path method for device routing
    pub fn device_id(&self) -> &str {
        match self {
            Self::Local { device_id, .. } => device_id,
            Self::Remote { device_id, .. } => device_id,
            Self::Distributed { device_id, .. } => device_id,
        }
    }

    #[inline] // Frequently called for batch planning
    pub fn max_batch_size(&self) -> usize {
        match self {
            Self::Local { max_batch_size, .. } => *max_batch_size,
            Self::Remote { max_batch_size, .. } => *max_batch_size,
            Self::Distributed { max_batch_size, .. } => *max_batch_size,
        }
    }

    #[inline] // Critical for scheduling decisions
    pub fn is_ready(&self) -> bool {
        match self {
            Self::Local { ready, .. } => *ready,
            Self::Remote { ready, .. } => *ready,
            Self::Distributed { ready, .. } => *ready,
        }
    }
}

/// Trait for model instances (kept for backward compatibility)
pub trait ModelInstance: Send + Sync {
    fn id(&self) -> &str;
    fn device_id(&self) -> &str;
    fn max_batch_size(&self) -> usize;
    fn is_ready(&self) -> bool;
}

impl ModelCluster {
    /// Create a new model cluster
    pub fn new() -> Self {
        Self {
            models: Arc::new(DashMap::new()),
            trait_models: Arc::new(DashMap::new()),
            routing_table: Arc::new(DashMap::new()),
        }
    }

    /// Add an optimized model instance to the cluster (preferred for performance)
    pub fn add_model_fast(&self, model: ModelInstanceType) {
        let model_id = model.id().to_string();
        let device_id = model.device_id().to_string();

        self.models.insert(model_id.clone(), model);

        // Update routing table
        self.routing_table
            .entry(device_id)
            .or_insert_with(Vec::new)
            .push(model_id);
    }

    /// Add a trait-based model instance (for compatibility)
    pub fn add_model(&self, model: Arc<dyn ModelInstance>) {
        let model_id = model.id().to_string();
        let device_id = model.device_id().to_string();

        self.trait_models.insert(model_id.clone(), model);

        // Update routing table
        self.routing_table
            .entry(device_id)
            .or_insert_with(Vec::new)
            .push(model_id);
    }

    /// Get models on a specific device
    pub fn models_on_device(&self, device_id: &str) -> Vec<String> {
        self.routing_table.get(device_id).map(|entry| entry.value().clone()).unwrap_or_default()
    }

    /// Get an optimized model instance (fast path)
    #[inline] // Hot path for model lookups
    pub fn get_model_fast(&self, model_id: &str) -> Option<ModelInstanceType> {
        self.models.get(model_id).map(|entry| entry.value().clone())
    }

    /// Get a trait-based model instance (compatibility path)
    pub fn get_model(&self, model_id: &str) -> Option<Arc<dyn ModelInstance>> {
        // Try fast path first
        if let Some(fast_model) = self.get_model_fast(model_id) {
            // Convert FastModelInstance to trait object for compatibility
            // This wraps the fast model in a trait-compatible adapter
            struct FastModelAdapter {
                model: ModelInstanceType,
            }
            
            impl ModelInstance for FastModelAdapter {
                fn id(&self) -> &str {
                    self.model.id()
                }
                
                fn device_id(&self) -> &str {
                    self.model.device_id()
                }
                
                fn max_batch_size(&self) -> usize {
                    self.model.max_batch_size()
                }
                
                fn is_ready(&self) -> bool {
                    self.model.is_ready()
                }
            }
            
            return Some(Arc::new(FastModelAdapter { model: fast_model }));
        }

        // Fallback to trait models
        self.trait_models.get(model_id).map(|entry| entry.value().clone())
    }

    /// List all models
    pub fn list_models(&self) -> Vec<String> {
        self.models.iter().map(|entry| entry.key().clone()).collect()
    }
}
