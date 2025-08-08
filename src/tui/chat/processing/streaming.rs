//! Message streaming handler

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::{Stream, StreamExt};
use anyhow::{Result, Context};
use futures::stream;

use crate::tui::run::AssistantResponseType;
// Define streaming types locally until we can import from models
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Error(String),
    Done,
    Metadata(std::collections::HashMap<String, String>),
}

/// Handles message streaming
pub struct StreamHandler {
    /// Response channel for streaming updates
    response_tx: mpsc::Sender<AssistantResponseType>,
    
    /// Buffer for accumulating streamed content
    buffer: Arc<RwLock<String>>,
    
    /// Stream configuration
    config: StreamConfig,
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
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(response_tx: mpsc::Sender<AssistantResponseType>, config: StreamConfig) -> Self {
        Self {
            response_tx,
            buffer: Arc::new(RwLock::new(String::new())),
            config,
        }
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
                                StreamEvent::Error(error) => {
                                    tracing::error!("Stream error: {}", error);
                                    return Err(anyhow::anyhow!("Stream error: {}", error));
                                }
                                StreamEvent::Done => {
                                    // Final flush
                                    self.flush_buffer(&model).await?;
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
}

impl Clone for StreamHandler {
    fn clone(&self) -> Self {
        Self {
            response_tx: self.response_tx.clone(),
            buffer: Arc::new(RwLock::new(String::new())),
            config: self.config.clone(),
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