//! Unified Stream Manager - Multiplexed streaming for all sources
//! 
//! Consolidates streaming from models, agents, cognitive systems, and stories
//! into a unified, multiplexed stream with proper buffering and metadata.

use std::sync::Arc;
use std::collections::HashMap;
use std::pin::Pin;
use tokio::sync::{mpsc, RwLock, broadcast};
use tokio_stream::{Stream, StreamExt, StreamMap};
use futures::stream::{self, BoxStream};
use anyhow::{Result, Context as AnyhowContext};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn, error};

use crate::tui::run::AssistantResponseType;
use super::streaming::{StreamEvent, StreamHandler, StreamConfig};

/// Unified stream manager for all streaming sources
pub struct UnifiedStreamManager {
    /// Stream handlers by source
    handlers: Arc<RwLock<HashMap<StreamSource, Arc<StreamHandler>>>>,
    
    /// Active streams
    active_streams: Arc<RwLock<StreamMap<StreamId, Pin<Box<dyn Stream<Item = StreamPacket> + Send + Sync>>>>>,
    
    /// Stream metadata
    stream_metadata: Arc<RwLock<HashMap<StreamId, StreamMetadata>>>,
    
    /// Response channel for UI updates
    response_tx: mpsc::Sender<AssistantResponseType>,
    
    /// Event broadcaster for stream events
    event_tx: broadcast::Sender<UnifiedStreamEvent>,
    
    /// Configuration
    config: UnifiedStreamConfig,
    
    /// Statistics
    stats: Arc<RwLock<StreamStatistics>>,
}

/// Stream source types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamSource {
    Model(ModelSource),
    Agent(AgentSource),
    Cognitive(CognitiveSource),
    Story(StorySource),
    Todo,
    Orchestration,
    Tool,
    Custom(u32),
}

/// Model streaming sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelSource {
    Primary,
    Fallback,
    Parallel(u8),
}

/// Agent streaming sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentSource {
    Analytical,
    Creative,
    Strategic,
    Technical,
    Empathetic,
}

/// Cognitive streaming sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveSource {
    Thoughts,
    Reasoning,
    Decisions,
    Emotions,
    Goals,
}

/// Story streaming sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorySource {
    Narrative,
    PlotPoints,
    Arcs,
    Events,
}

/// Stream identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub uuid::Uuid);

/// Stream packet with metadata
#[derive(Debug, Clone)]
pub struct StreamPacket {
    pub id: StreamId,
    pub source: StreamSource,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
    pub content: StreamContent,
    pub metadata: HashMap<String, String>,
}

/// Stream content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamContent {
    Token(String),
    Chunk(String),
    Event(StreamEventData),
    Progress(f32),
    Status(StreamStatus),
    Error(String),
    Complete,
}

/// Stream event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEventData {
    pub event_type: String,
    pub data: serde_json::Value,
}

/// Stream status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StreamStatus {
    Starting,
    Active,
    Paused,
    Completing,
    Completed,
    Failed,
}

/// Stream metadata
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub id: StreamId,
    pub source: StreamSource,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub priority: u8,
    pub buffer_size: usize,
    pub tokens_streamed: usize,
    pub events_streamed: usize,
    pub context: HashMap<String, String>,
}

/// Unified stream events
#[derive(Debug, Clone)]
pub enum UnifiedStreamEvent {
    StreamStarted {
        id: StreamId,
        source: StreamSource,
    },
    StreamCompleted {
        id: StreamId,
        duration_ms: u64,
        tokens: usize,
    },
    StreamFailed {
        id: StreamId,
        error: String,
    },
    StreamMerged {
        streams: Vec<StreamId>,
        merged_id: StreamId,
    },
}

/// Configuration for unified streaming
#[derive(Debug, Clone)]
pub struct UnifiedStreamConfig {
    pub max_concurrent_streams: usize,
    pub buffer_size_per_stream: usize,
    pub merge_similar_streams: bool,
    pub priority_scheduling: bool,
    pub adaptive_buffering: bool,
    pub stream_timeout_seconds: u64,
    pub enable_compression: bool,
    pub enable_deduplication: bool,
}

impl Default for UnifiedStreamConfig {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 10,
            buffer_size_per_stream: 256,
            merge_similar_streams: true,
            priority_scheduling: true,
            adaptive_buffering: true,
            stream_timeout_seconds: 300,
            enable_compression: false,
            enable_deduplication: true,
        }
    }
}

/// Stream statistics
#[derive(Debug, Clone, Default)]
pub struct StreamStatistics {
    pub total_streams_created: usize,
    pub active_streams: usize,
    pub completed_streams: usize,
    pub failed_streams: usize,
    pub total_tokens_streamed: usize,
    pub total_events_streamed: usize,
    pub average_stream_duration_ms: f64,
    pub streams_by_source: HashMap<String, usize>,
}

impl UnifiedStreamManager {
    /// Create a new unified stream manager
    pub fn new(response_tx: mpsc::Sender<AssistantResponseType>) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(RwLock::new(StreamMap::new())),
            stream_metadata: Arc::new(RwLock::new(HashMap::new())),
            response_tx,
            event_tx,
            config: UnifiedStreamConfig::default(),
            stats: Arc::new(RwLock::new(StreamStatistics::default())),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(
        response_tx: mpsc::Sender<AssistantResponseType>,
        config: UnifiedStreamConfig,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(RwLock::new(StreamMap::new())),
            stream_metadata: Arc::new(RwLock::new(HashMap::new())),
            response_tx,
            event_tx,
            config,
            stats: Arc::new(RwLock::new(StreamStatistics::default())),
        }
    }
    
    /// Register a stream handler for a source
    pub async fn register_handler(&self, source: StreamSource, handler: Arc<StreamHandler>) {
        self.handlers.write().await.insert(source, handler);
        debug!("Registered stream handler for {:?}", source);
    }
    
    /// Create a new stream
    pub async fn create_stream<S>(
        &self,
        source: StreamSource,
        stream: S,
        context: HashMap<String, String>,
    ) -> Result<StreamId>
    where
        S: Stream<Item = Result<StreamEvent>> + Send + Sync + 'static,
    {
        let stream_id = StreamId(uuid::Uuid::new_v4());
        let started_at = Utc::now();
        
        // Create metadata
        let metadata = StreamMetadata {
            id: stream_id,
            source,
            started_at,
            ended_at: None,
            priority: self.get_source_priority(source),
            buffer_size: self.config.buffer_size_per_stream,
            tokens_streamed: 0,
            events_streamed: 0,
            context,
        };
        
        // Convert stream to packet stream
        let packet_stream = self.create_packet_stream(stream_id, source, stream);
        
        // Add to active streams
        self.active_streams.write().await.insert(stream_id, packet_stream);
        self.stream_metadata.write().await.insert(stream_id, metadata);
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_streams_created += 1;
        stats.active_streams += 1;
        *stats.streams_by_source.entry(format!("{:?}", source)).or_insert(0) += 1;
        
        // Broadcast event
        let _ = self.event_tx.send(UnifiedStreamEvent::StreamStarted {
            id: stream_id,
            source,
        });
        
        // Start processing if this is the first stream
        if stats.active_streams == 1 {
            self.start_processing().await;
        }
        
        info!("Created stream {:?} from source {:?}", stream_id, source);
        Ok(stream_id)
    }
    
    /// Create packet stream from raw stream
    fn create_packet_stream<S>(
        &self,
        stream_id: StreamId,
        source: StreamSource,
        stream: S,
    ) -> Pin<Box<dyn Stream<Item = StreamPacket> + Send + Sync + 'static>>
    where
        S: Stream<Item = Result<StreamEvent>> + Send + Sync + 'static,
    {
        let mut sequence = 0u64;
        
        Box::pin(stream.map(move |result| {
            sequence += 1;
            
            let content = match result {
                Ok(event) => match event {
                    StreamEvent::Token(token) => StreamContent::Token(token),
                    StreamEvent::Error(error) => StreamContent::Error(error),
                    StreamEvent::Done => StreamContent::Complete,
                    StreamEvent::Metadata(meta) => StreamContent::Event(StreamEventData {
                        event_type: "metadata".to_string(),
                        data: serde_json::to_value(meta).unwrap_or_default(),
                    }),
                    StreamEvent::Story(story_event) => StreamContent::Event(StreamEventData {
                        event_type: "story".to_string(),
                        data: serde_json::to_value(story_event).unwrap_or_default(),
                    }),
                },
                Err(e) => StreamContent::Error(e.to_string()),
            };
            
            StreamPacket {
                id: stream_id,
                source,
                timestamp: Utc::now(),
                sequence,
                content,
                metadata: HashMap::new(),
            }
        }))
    }
    
    /// Start processing active streams
    async fn start_processing(&self) {
        let manager = self.clone();
        
        tokio::spawn(async move {
            manager.process_streams().await;
        });
    }
    
    /// Process all active streams
    async fn process_streams(&self) {
        loop {
            let mut streams = self.active_streams.write().await;
            
            if streams.is_empty() {
                break;
            }
            
            // Process next packet from any stream
            if let Some((stream_id, packet)) = streams.next().await {
                drop(streams); // Release lock before processing
                
                if let Err(e) = self.handle_packet(packet).await {
                    error!("Error handling packet from stream {:?}: {}", stream_id, e);
                }
                
                // Check if stream is complete
                if self.is_stream_complete(stream_id).await {
                    self.complete_stream(stream_id).await;
                }
            } else {
                // All streams exhausted
                break;
            }
        }
        
        debug!("Stream processing completed");
    }
    
    /// Handle a stream packet
    async fn handle_packet(&self, packet: StreamPacket) -> Result<()> {
        // Update metadata
        if let Some(metadata) = self.stream_metadata.write().await.get_mut(&packet.id) {
            match &packet.content {
                StreamContent::Token(_) | StreamContent::Chunk(_) => {
                    metadata.tokens_streamed += 1;
                }
                StreamContent::Event(_) => {
                    metadata.events_streamed += 1;
                }
                _ => {}
            }
        }
        
        // Route packet based on source
        match packet.source {
            StreamSource::Model(_) => self.handle_model_packet(packet).await?,
            StreamSource::Agent(_) => self.handle_agent_packet(packet).await?,
            StreamSource::Cognitive(_) => self.handle_cognitive_packet(packet).await?,
            StreamSource::Story(_) => self.handle_story_packet(packet).await?,
            StreamSource::Todo => self.handle_todo_packet(packet).await?,
            StreamSource::Orchestration => self.handle_orchestration_packet(packet).await?,
            StreamSource::Tool => self.handle_tool_packet(packet).await?,
            StreamSource::Custom(_) => self.handle_custom_packet(packet).await?,
        }
        
        Ok(())
    }
    
    /// Handle model stream packet
    async fn handle_model_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Token(token) => {
                let message = AssistantResponseType::new_ai_message(
                    token,
                    Some(format!("model-{:?}", packet.source)),
                );
                self.response_tx.send(message).await?;
            }
            StreamContent::Chunk(chunk) => {
                let message = AssistantResponseType::new_ai_message(
                    chunk,
                    Some(format!("model-{:?}", packet.source)),
                );
                self.response_tx.send(message).await?;
            }
            StreamContent::Complete => {
                debug!("Model stream {:?} completed", packet.id);
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle agent stream packet
    async fn handle_agent_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Token(token) => {
                let message = AssistantResponseType::new_ai_message(
                    token,
                    Some(format!("agent-{:?}", packet.source)),
                );
                self.response_tx.send(message).await?;
            }
            StreamContent::Chunk(chunk) => {
                let message = AssistantResponseType::new_ai_message(
                    chunk,
                    Some(format!("agent-{:?}", packet.source)),
                );
                self.response_tx.send(message).await?;
            }
            StreamContent::Event(event) => {
                debug!("Agent event: {:?}", event);
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle cognitive stream packet
    async fn handle_cognitive_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Event(event) => {
                if event.event_type == "thought" {
                    let message = AssistantResponseType::new_ai_message(
                        format!("💭 {}", event.data),
                        Some("cognitive".to_string()),
                    );
                    self.response_tx.send(message).await?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle story stream packet
    async fn handle_story_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Chunk(narrative) => {
                let message = AssistantResponseType::new_ai_message(
                    format!("📖 {}", narrative),
                    Some("story".to_string()),
                );
                self.response_tx.send(message).await?;
            }
            StreamContent::Event(event) => {
                if event.event_type == "plot_point" {
                    debug!("New plot point: {:?}", event.data);
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle todo stream packet
    async fn handle_todo_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Progress(progress) => {
                let message = AssistantResponseType::new_ai_message(
                    format!("📊 Todo progress: {:.0}%", progress * 100.0),
                    Some("todo".to_string()),
                );
                self.response_tx.send(message).await?;
            }
            StreamContent::Status(status) => {
                debug!("Todo status: {:?}", status);
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle orchestration stream packet
    async fn handle_orchestration_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Event(event) => {
                if event.event_type == "model_selected" {
                    debug!("Model selected: {:?}", event.data);
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle tool stream packet
    async fn handle_tool_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Chunk(output) => {
                let message = AssistantResponseType::new_ai_message(
                    format!("🔧 {}", output),
                    Some("tool".to_string()),
                );
                self.response_tx.send(message).await?;
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Handle custom stream packet
    async fn handle_custom_packet(&self, packet: StreamPacket) -> Result<()> {
        match packet.content {
            StreamContent::Chunk(content) => {
                let message = AssistantResponseType::new_ai_message(
                    content,
                    Some("custom".to_string()),
                );
                self.response_tx.send(message).await?;
            }
            _ => {}
        }
        Ok(())
    }
    
    /// Check if stream is complete
    async fn is_stream_complete(&self, stream_id: StreamId) -> bool {
        // Check if stream has sent completion signal
        // This would need to track completion status
        false
    }
    
    /// Complete a stream
    async fn complete_stream(&self, stream_id: StreamId) {
        // Remove from active streams
        self.active_streams.write().await.remove(&stream_id);
        
        // Update metadata
        if let Some(metadata) = self.stream_metadata.write().await.get_mut(&stream_id) {
            metadata.ended_at = Some(Utc::now());
            
            let duration_ms = metadata.ended_at.unwrap()
                .signed_duration_since(metadata.started_at)
                .num_milliseconds() as u64;
            
            // Broadcast completion event
            let _ = self.event_tx.send(UnifiedStreamEvent::StreamCompleted {
                id: stream_id,
                duration_ms,
                tokens: metadata.tokens_streamed,
            });
        }
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.active_streams -= 1;
        stats.completed_streams += 1;
        
        info!("Stream {:?} completed", stream_id);
    }
    
    /// Merge multiple streams into one
    pub async fn merge_streams(
        &self,
        stream_ids: Vec<StreamId>,
        merge_strategy: MergeStrategy,
    ) -> Result<StreamId> {
        let merged_id = StreamId(uuid::Uuid::new_v4());
        
        // Implementation would merge streams based on strategy
        
        let _ = self.event_tx.send(UnifiedStreamEvent::StreamMerged {
            streams: stream_ids,
            merged_id,
        });
        
        Ok(merged_id)
    }
    
    /// Get source priority
    fn get_source_priority(&self, source: StreamSource) -> u8 {
        match source {
            StreamSource::Model(_) => 10,
            StreamSource::Agent(_) => 8,
            StreamSource::Cognitive(_) => 7,
            StreamSource::Story(_) => 5,
            StreamSource::Todo => 6,
            StreamSource::Orchestration => 9,
            StreamSource::Tool => 7,
            StreamSource::Custom(_) => 5,
        }
    }
    
    /// Get stream statistics
    pub async fn get_statistics(&self) -> StreamStatistics {
        self.stats.read().await.clone()
    }
    
    /// Subscribe to stream events
    pub fn subscribe_events(&self) -> broadcast::Receiver<UnifiedStreamEvent> {
        self.event_tx.subscribe()
    }
}

/// Stream merge strategies
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    RoundRobin,
    Priority,
    Interleave,
    Sequential,
}

impl Clone for UnifiedStreamManager {
    fn clone(&self) -> Self {
        Self {
            handlers: self.handlers.clone(),
            active_streams: self.active_streams.clone(),
            stream_metadata: self.stream_metadata.clone(),
            response_tx: self.response_tx.clone(),
            event_tx: self.event_tx.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
        }
    }
}