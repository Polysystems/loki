//! Message streaming handler

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{Stream, StreamExt};
use anyhow::{Result, Context};
use futures::stream;
use serde::{Serialize, Deserialize};

use crate::tui::run::AssistantResponseType;
use crate::story::StoryContext;
use crate::story::StoryEvent as StoryEventType;
// Define streaming types locally until we can import from models
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Error(String),
    Done,
    Metadata(std::collections::HashMap<String, String>),
    Story(StoryStreamEvent),
}

/// Story-specific stream events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StoryStreamEvent {
    PlotPoint(String),
    ArcTransition { from: String, to: String },
    CharacterAction { character: String, action: String },
    NarrativeUpdate(String),
    ThemeEvolution { theme: String, evolution: String },
}

/// Handles message streaming
pub struct StreamHandler {
    /// Response channel for streaming updates
    response_tx: mpsc::Sender<AssistantResponseType>,
    
    /// Buffer for accumulating streamed content
    buffer: Arc<RwLock<String>>,
    
    /// Stream configuration
    config: StreamConfig,
    
    /// Story context for narrative-aware streaming
    story_context: Arc<RwLock<Option<StoryContext>>>,
    
    /// Story event buffer
    story_events: Arc<RwLock<Vec<StoryStreamEvent>>>,
}

/// Stream configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Buffer size before flushing
    pub buffer_size: usize,
    
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    
    /// Enable token-by-token streaming
    pub token_streaming: bool,
    
    /// Enable markdown rendering during stream
    pub render_markdown: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 128,
            flush_interval_ms: 50,
            token_streaming: true,
            render_markdown: false,
        }
    }
}

impl StreamHandler {
    /// Create a new stream handler
    pub fn new(response_tx: mpsc::Sender<AssistantResponseType>) -> Self {
        Self {
            response_tx,
            buffer: Arc::new(RwLock::new(String::new())),
            config: StreamConfig::default(),
            story_context: Arc::new(RwLock::new(None)),
            story_events: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(response_tx: mpsc::Sender<AssistantResponseType>, config: StreamConfig) -> Self {
        Self {
            response_tx,
            buffer: Arc::new(RwLock::new(String::new())),
            config,
            story_context: Arc::new(RwLock::new(None)),
            story_events: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Set story context for narrative-aware streaming
    pub async fn set_story_context(&self, context: StoryContext) {
        *self.story_context.write().await = Some(context);
    }
    
    /// Handle a model stream
    pub async fn handle_stream<S>(&self, mut stream: S, model: String) -> Result<String>
    where
        S: Stream<Item = Result<StreamEvent>> + Unpin,
    {
        let mut accumulated = String::new();
        let mut token_count = 0;
        
        // Set up flush timer
        let flush_interval = tokio::time::interval(
            tokio::time::Duration::from_millis(self.config.flush_interval_ms)
        );
        tokio::pin!(flush_interval);
        
        loop {
            tokio::select! {
                // Handle stream events
                event = stream.next() => {
                    match event {
                        Some(Ok(stream_event)) => {
                            match stream_event {
                                StreamEvent::Token(token) => {
                                    accumulated.push_str(&token);
                                    token_count += 1;
                                    
                                    if self.config.token_streaming {
                                        self.append_to_buffer(&token).await?;
                                    }
                                    
                                    // Flush if buffer is full
                                    if token_count >= self.config.buffer_size {
                                        self.flush_buffer(&model).await?;
                                        token_count = 0;
                                    }
                                }
                                StreamEvent::Story(story_event) => {
                                    self.handle_story_event(story_event).await?;
                                }
                                StreamEvent::Error(error) => {
                                    tracing::error!("Stream error: {}", error);
                                    return Err(anyhow::anyhow!("Stream error: {}", error));
                                }
                                StreamEvent::Done => {
                                    // Final flush
                                    self.flush_buffer(&model).await?;
                                    self.flush_story_events().await?;
                                    break;
                                }
                                StreamEvent::Metadata(metadata) => {
                                    tracing::debug!("Stream metadata: {:?}", metadata);
                                }
                            }
                        }
                        Some(Err(e)) => {
                            return Err(e).context("Stream processing error");
                        }
                        None => {
                            // Stream ended
                            self.flush_buffer(&model).await?;
                            break;
                        }
                    }
                }
                
                // Handle periodic flush
                _ = flush_interval.tick() => {
                    if token_count > 0 {
                        self.flush_buffer(&model).await?;
                        token_count = 0;
                    }
                }
            }
        }
        
        Ok(accumulated)
    }
    
    /// Handle parallel streams from multiple models
    pub async fn handle_parallel_streams(
        &self,
        streams: Vec<(String, Box<dyn Stream<Item = Result<StreamEvent>> + Unpin + Send>)>
    ) -> Result<Vec<(String, String)>> {
        let mut handles = vec![];
        
        for (model, stream) in streams {
            let handler = self.clone();
            let handle = tokio::spawn(async move {
                handler.handle_stream(stream, model.clone()).await
                    .map(|content| (model, content))
            });
            handles.push(handle);
        }
        
        // Wait for all streams to complete
        let mut results = vec![];
        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => tracing::error!("Stream handling error: {}", e),
                Err(e) => tracing::error!("Task join error: {}", e),
            }
        }
        
        Ok(results)
    }
    
    /// Append to buffer
    async fn append_to_buffer(&self, content: &str) -> Result<()> {
        let mut buffer = self.buffer.write().await;
        buffer.push_str(content);
        Ok(())
    }
    
    /// Flush buffer to response channel
    async fn flush_buffer(&self, model: &str) -> Result<()> {
        let mut buffer = self.buffer.write().await;
        
        if buffer.is_empty() {
            return Ok(());
        }
        
        let content = std::mem::take(&mut *buffer);
        
        // Optionally render markdown
        let final_content = if self.config.render_markdown {
            render_markdown_preview(&content)
        } else {
            content
        };
        
        // Create a stream response using the actual AssistantResponseType structure
        let response = AssistantResponseType::Stream {
            id: uuid::Uuid::new_v4().to_string(),
            author: model.to_string(),
            partial_content: final_content,
            timestamp: chrono::Utc::now().to_rfc3339(),
            stream_state: crate::tui::run::StreamingState::Streaming { 
                progress: 0.5,
                estimated_total_tokens: None 
            },
            metadata: crate::tui::run::MessageMetadata::default(),
        };
        
        self.response_tx.send(response).await
            .context("Failed to send stream update")?;
        
        Ok(())
    }
    
    /// Handle story stream event
    async fn handle_story_event(&self, event: StoryStreamEvent) -> Result<()> {
        // Store event for later processing
        self.story_events.write().await.push(event.clone());
        
        // Generate narrative update based on event type
        let narrative_update = match event {
            StoryStreamEvent::PlotPoint(plot) => {
                format!("📖 Plot Development: {}", plot)
            }
            StoryStreamEvent::ArcTransition { from, to } => {
                format!("🎭 Story Arc Transition: {} → {}", from, to)
            }
            StoryStreamEvent::CharacterAction { character, action } => {
                format!("👤 {}: {}", character, action)
            }
            StoryStreamEvent::NarrativeUpdate(update) => {
                format!("📝 {}", update)
            }
            StoryStreamEvent::ThemeEvolution { theme, evolution } => {
                format!("🎨 Theme Evolution - {}: {}", theme, evolution)
            }
        };
        
        // Send narrative update as a special message
        let response = AssistantResponseType::new_ai_message(
            narrative_update,
            Some("story-stream".to_string()),
        );
        
        self.response_tx.send(response).await
            .context("Failed to send story update")?;
        
        Ok(())
    }
    
    /// Flush story events
    async fn flush_story_events(&self) -> Result<()> {
        let events = self.story_events.write().await.drain(..).collect::<Vec<_>>();
        
        if events.is_empty() {
            return Ok(());
        }
        
        // Create summary of story events
        let mut summary = String::from("📚 Story Summary:\n");
        for event in events {
            match event {
                StoryStreamEvent::PlotPoint(plot) => {
                    summary.push_str(&format!("  • {}\n", plot));
                }
                StoryStreamEvent::ArcTransition { from, to } => {
                    summary.push_str(&format!("  • Transitioned from {} to {}\n", from, to));
                }
                _ => {}
            }
        }
        
        if summary.len() > 20 {
            let response = AssistantResponseType::new_ai_message(
                summary,
                Some("story-summary".to_string()),
            );
            
            self.response_tx.send(response).await
                .context("Failed to send story summary")?;
        }
        
        Ok(())
    }
    
    /// Create a story-aware stream
    pub async fn create_story_stream(&self, content: &str) -> Result<impl Stream<Item = Result<StreamEvent>>> {
        let story_context = self.story_context.read().await.clone();
        
        let mut events = vec![];
        
        // Add narrative context if available
        if let Some(context) = story_context {
            events.push(Ok(StreamEvent::Story(StoryStreamEvent::NarrativeUpdate(
                format!("Continuing from: {}", context.current_plot)
            ))));
            
            // Add content tokens
            let tokens: Vec<_> = content
                .split_whitespace()
                .map(|s| Ok(StreamEvent::Token(format!("{} ", s))))
                .collect();
            events.extend(tokens);
            
            // Add story conclusion
            events.push(Ok(StreamEvent::Story(StoryStreamEvent::PlotPoint(
                "Response integrated into narrative".to_string()
            ))));
        } else {
            // Just stream tokens without story context
            let tokens: Vec<_> = content
                .split_whitespace()
                .map(|s| Ok(StreamEvent::Token(format!("{} ", s))))
                .collect();
            events.extend(tokens);
        }
        
        events.push(Ok(StreamEvent::Done));
        
        Ok(stream::iter(events))
    }
}

impl Clone for StreamHandler {
    fn clone(&self) -> Self {
        Self {
            response_tx: self.response_tx.clone(),
            buffer: Arc::new(RwLock::new(String::new())),
            config: self.config.clone(),
            story_context: Arc::new(RwLock::new(None)),
            story_events: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

/// Convert StreamPacket to StreamEvent
impl From<crate::tui::chat::processing::unified_streaming::StreamPacket> for StreamEvent {
    fn from(packet: crate::tui::chat::processing::unified_streaming::StreamPacket) -> Self {
        use crate::tui::chat::processing::unified_streaming::StreamContent;
        
        match packet.content {
            StreamContent::Token(text) => StreamEvent::Token(text),
            StreamContent::Chunk(text) => StreamEvent::Token(text),
            StreamContent::Error(err) => StreamEvent::Error(err),
            StreamContent::Complete => StreamEvent::Done,
            StreamContent::Status(status) => {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("status".to_string(), format!("{:?}", status));
                StreamEvent::Metadata(metadata)
            },
            StreamContent::Progress(progress) => {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("progress".to_string(), progress.to_string());
                StreamEvent::Metadata(metadata)
            },
            StreamContent::Event(event_data) => {
                // Convert event data to story stream event
                StreamEvent::Story(StoryStreamEvent::NarrativeUpdate(
                    event_data.data.as_str().unwrap_or("").to_string()
                ))
            }
        }
    }
}

/// Render markdown preview (simplified)
fn render_markdown_preview(content: &str) -> String {
    // This is a simplified version - in production, use a proper markdown renderer
    content
        .replace("**", "")
        .replace("*", "")
        .replace("#", "")
}

/// Create a test stream for development
pub fn create_test_stream(message: &str) -> impl Stream<Item = Result<StreamEvent>> {
    let tokens: Vec<String> = message
        .split_whitespace()
        .map(|s| format!("{} ", s))
        .collect();
    
    stream::iter(tokens)
        .map(|token| Ok(StreamEvent::Token(token)))
        .chain(stream::once(async { Ok(StreamEvent::Done) }))
}