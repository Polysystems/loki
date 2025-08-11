//! Error types for the chat module
//! 
//! Provides comprehensive error handling with specific error types
//! for different failure scenarios in the chat system.

use std::time::Duration;
use thiserror::Error;

/// Main error type for chat operations
#[derive(Debug, Error)]
pub enum ChatError {
    /// Errors related to message processing
    #[error("Message processing error: {0}")]
    MessageProcessing(#[from] MessageError),
    
    /// Errors related to orchestration
    #[error("Orchestration error: {0}")]
    Orchestration(#[from] OrchestrationError),
    
    /// Errors related to agent operations
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),
    
    /// Errors related to tool execution
    #[error("Tool execution error: {0}")]
    Tool(#[from] ToolError),
    
    /// Errors related to NLP processing
    #[error("NLP processing error: {0}")]
    Nlp(#[from] NlpError),
    
    /// Errors related to editor operations
    #[error("Editor error: {0}")]
    Editor(#[from] EditorError),
    
    /// Errors related to session management
    #[error("Session error: {0}")]
    Session(#[from] SessionError),
    
    /// Errors related to rendering
    #[error("Rendering error: {0}")]
    Rendering(#[from] RenderingError),
    
    /// Network-related errors
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization/Deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Timeout errors
    #[error("Operation timed out after {0:?}")]
    Timeout(Duration),
    
    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// Unknown or unexpected error
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Errors specific to message processing
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Invalid message format: {0}")]
    InvalidFormat(String),
    
    #[error("Message too long: {length} bytes (max: {max})")]
    TooLong { length: usize, max: usize },
    
    #[error("Empty message")]
    Empty,
    
    #[error("Failed to parse message: {0}")]
    ParseError(String),
    
    #[error("Message validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Unsupported message type: {0}")]
    UnsupportedType(String),
}

/// Errors specific to orchestration
#[derive(Debug, Error)]
pub enum OrchestrationError {
    #[error("No models available")]
    NoModelsAvailable,
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Failed to route request: {0}")]
    RoutingFailed(String),
    
    #[error("Pipeline execution failed: {0}")]
    PipelineFailed(String),
    
    #[error("Ensemble aggregation failed: {0}")]
    EnsembleFailed(String),
    
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Model quota exceeded for: {0}")]
    QuotaExceeded(String),
}

/// Errors specific to agent operations
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Agent not found: {0}")]
    NotFound(String),
    
    #[error("Agent creation failed: {0}")]
    CreationFailed(String),
    
    #[error("Agent execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Invalid agent configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Agent communication failed: {0}")]
    CommunicationFailed(String),
    
    #[error("Agent timeout: {0}")]
    Timeout(String),
}

/// Errors specific to tool execution
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Invalid tool parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Tool permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Tool initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Tool timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Tool bridge not connected")]
    BridgeNotConnected,
}

/// Errors specific to NLP processing
#[derive(Debug, Error)]
pub enum NlpError {
    #[error("Failed to detect intent: {0}")]
    IntentDetectionFailed(String),
    
    #[error("Entity extraction failed: {0}")]
    EntityExtractionFailed(String),
    
    #[error("Sentiment analysis failed: {0}")]
    SentimentAnalysisFailed(String),
    
    #[error("Language not supported: {0}")]
    UnsupportedLanguage(String),
    
    #[error("NLP orchestrator not available")]
    OrchestratorUnavailable,
    
    #[error("Pattern compilation failed: {0}")]
    PatternCompilationFailed(String),
}

/// Errors specific to editor operations
#[derive(Debug, Error)]
pub enum EditorError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Failed to save file: {0}")]
    SaveFailed(String),
    
    #[error("Invalid cursor position: line {line}, column {column}")]
    InvalidCursorPosition { line: usize, column: usize },
    
    #[error("Syntax highlighting failed: {0}")]
    SyntaxHighlightingFailed(String),
    
    #[error("LSP connection failed: {0}")]
    LspConnectionFailed(String),
    
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),
    
    #[error("Undo/Redo operation failed: {0}")]
    UndoRedoFailed(String),
}

/// Errors specific to session management
#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    NotFound(String),
    
    #[error("Session expired: {0}")]
    Expired(String),
    
    #[error("Failed to create session: {0}")]
    CreationFailed(String),
    
    #[error("Failed to restore session: {0}")]
    RestoreFailed(String),
    
    #[error("Session limit exceeded: max {max}, current {current}")]
    LimitExceeded { max: usize, current: usize },
}

/// Errors specific to rendering
#[derive(Debug, Error)]
pub enum RenderingError {
    #[error("Render state not initialized")]
    NotInitialized,
    
    #[error("Failed to acquire render lock")]
    LockFailed,
    
    #[error("Invalid render area: {0}")]
    InvalidArea(String),
    
    #[error("Theme not found: {0}")]
    ThemeNotFound(String),
    
    #[error("Widget render failed: {0}")]
    WidgetFailed(String),
}

/// Network-related errors
#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Request timeout after {0:?}")]
    RequestTimeout(Duration),
    
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    
    #[error("Rate limited: retry after {0:?}")]
    RateLimited(Duration),
    
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
}

/// Result type alias for chat operations
pub type ChatResult<T> = Result<T, ChatError>;

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context_chat(self, msg: impl Into<String>) -> ChatResult<T>;
    
    /// Add lazy context to an error
    fn with_context_chat<F>(self, f: F) -> ChatResult<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<ChatError>,
{
    fn context_chat(self, msg: impl Into<String>) -> ChatResult<T> {
        self.map_err(|e| {
            let base_error = e.into();
            ChatError::Internal(format!("{}: {}", msg.into(), base_error))
        })
    }
    
    fn with_context_chat<F>(self, f: F) -> ChatResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let base_error = e.into();
            ChatError::Internal(format!("{}: {}", f(), base_error))
        })
    }
}

/// Helper function to create timeout errors
pub fn timeout_error(duration: Duration) -> ChatError {
    ChatError::Timeout(duration)
}

/// Helper function to create internal errors
pub fn internal_error(msg: impl Into<String>) -> ChatError {
    ChatError::Internal(msg.into())
}

/// Helper function to create unknown errors
pub fn unknown_error(msg: impl Into<String>) -> ChatError {
    ChatError::Unknown(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = ChatError::MessageProcessing(MessageError::TooLong { 
            length: 10000, 
            max: 4096 
        });
        assert_eq!(
            err.to_string(), 
            "Message processing error: Message too long: 10000 bytes (max: 4096)"
        );
    }
    
    #[test]
    fn test_error_context() {
        let result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found"
        ));
        
        let with_context = result.context_chat("Failed to load configuration");
        assert!(with_context.is_err());
        assert!(with_context.unwrap_err().to_string().contains("Failed to load configuration"));
    }
}