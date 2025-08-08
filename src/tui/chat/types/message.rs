//! Message type definitions for the chat system

use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Basic message representation for the chat system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: String,
    
    /// Message author
    pub author: String,
    
    /// Message content
    pub content: String,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Message role (user, assistant, system)
    pub role: MessageRole,
    
    /// Additional metadata
    pub metadata: Option<MessageMetadata>,
}

/// Message role enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Model used for generation
    pub model: Option<String>,
    
    /// Token count
    pub tokens: Option<usize>,
    
    /// Generation time
    pub generation_time_ms: Option<u64>,
}

impl Message {
    /// Create a new message
    pub fn new(author: String, content: String, role: MessageRole) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            author,
            content,
            timestamp: Utc::now(),
            role,
            metadata: None,
        }
    }
    
    /// Create a user message
    pub fn user(author: String, content: String) -> Self {
        Self::new(author, content, MessageRole::User)
    }
    
    /// Create an assistant message
    pub fn assistant(author: String, content: String) -> Self {
        Self::new(author, content, MessageRole::Assistant)
    }
    
    /// Create a system message
    pub fn system(content: String) -> Self {
        Self::new("System".to_string(), content, MessageRole::System)
    }
    
    /// Associated constructor for User messages (compatibility)
    pub fn User(author: String, content: String) -> Self {
        Self::user(author, content)
    }
    
    /// Associated constructor for Assistant messages (compatibility)
    pub fn Assistant(author: String, content: String) -> Self {
        Self::assistant(author, content)
    }
    
    /// Associated constructor for System messages (compatibility)
    pub fn System(content: String) -> Self {
        Self::system(content)
    }
    
    /// Associated constructor for Error messages (compatibility)
    pub fn Error(content: String) -> Self {
        Self::system(format!("Error: {}", content))
    }
}