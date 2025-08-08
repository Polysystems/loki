//! Chat export functionality
//! 
//! Provides various export formats for chat conversations

use std::path::Path;
use anyhow::{Result, Context};
use chrono::{DateTime, Local};
use serde::{Serialize, Deserialize};

use crate::tui::run::AssistantResponseType;

pub mod formats;
pub mod exporter;

pub use exporter::ChatExporter;
pub use formats::{ExportFormat, ExportOptions};

/// Represents an exported chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedChat {
    /// Export metadata
    pub metadata: ExportMetadata,
    
    /// Chat messages
    pub messages: Vec<ExportedMessage>,
    
    /// Chat statistics
    pub statistics: ChatStatistics,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Export timestamp
    pub exported_at: DateTime<Local>,
    
    /// Chat title/name
    pub chat_title: String,
    
    /// Export format
    pub format: String,
    
    /// Loki version
    pub loki_version: String,
    
    /// Total messages
    pub message_count: usize,
    
    /// Date range
    pub date_range: Option<(DateTime<Local>, DateTime<Local>)>,
}

/// Exported message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMessage {
    /// Message ID
    pub id: String,
    
    /// Author (user, assistant, system)
    pub author: String,
    
    /// Message content
    pub content: String,
    
    /// Timestamp
    pub timestamp: DateTime<Local>,
    
    /// Message type
    pub message_type: String,
    
    /// Additional metadata
    pub metadata: Option<MessageMetadata>,
}

/// Additional message metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    /// Model used (for assistant messages)
    pub model: Option<String>,
    
    /// Token count
    pub tokens: Option<usize>,
    
    /// Processing time
    pub processing_time_ms: Option<u64>,
    
    /// Error details (for error messages)
    pub error: Option<String>,
    
    /// Tool calls made
    pub tools_used: Option<Vec<String>>,
}

/// Chat statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStatistics {
    /// Total messages
    pub total_messages: usize,
    
    /// Messages by author
    pub messages_by_author: std::collections::HashMap<String, usize>,
    
    /// Average message length
    pub avg_message_length: usize,
    
    /// Total tokens (estimated)
    pub total_tokens: usize,
    
    /// Chat duration
    pub duration_minutes: Option<f64>,
    
    /// Most active hour
    pub most_active_hour: Option<u32>,
    
    /// Topics discussed
    pub topics: Vec<String>,
    
    /// Entities mentioned
    pub entities: Vec<String>,
}

/// Convert AssistantResponseType to ExportedMessage
impl From<&AssistantResponseType> for ExportedMessage {
    fn from(msg: &AssistantResponseType) -> Self {
        match msg {
            AssistantResponseType::Message { id, author, message, timestamp, metadata, .. } => {
                let parsed_timestamp = DateTime::parse_from_rfc3339(timestamp)
                    .unwrap_or_else(|_| Local::now().into())
                    .with_timezone(&Local);
                
                ExportedMessage {
                    id: id.clone(),
                    author: author.clone(),
                    content: message.clone(),
                    timestamp: parsed_timestamp,
                    message_type: "message".to_string(),
                    metadata: Some(MessageMetadata {
                        model: metadata.model_used.clone(),
                        tokens: metadata.tokens_used.map(|t| t as usize),
                        processing_time_ms: metadata.generation_time_ms,
                        error: None,
                        tools_used: None, // Tools tracking not available in current metadata
                    }),
                }
            }
            AssistantResponseType::Error { error_type, message, timestamp, .. } => {
                let parsed_timestamp = DateTime::parse_from_rfc3339(timestamp)
                    .unwrap_or_else(|_| Local::now().into())
                    .with_timezone(&Local);
                
                ExportedMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    author: "system".to_string(),
                    content: message.clone(),
                    timestamp: parsed_timestamp,
                    message_type: "error".to_string(),
                    metadata: Some(MessageMetadata {
                        model: None,
                        tokens: None,
                        processing_time_ms: None,
                        error: Some(error_type.clone()),
                        tools_used: None,
                    }),
                }
            }
            AssistantResponseType::UserMessage { content, timestamp, id, metadata } => {
                let parsed_timestamp = DateTime::parse_from_rfc3339(timestamp)
                    .unwrap_or_else(|_| Local::now().into())
                    .with_timezone(&Local);
                
                ExportedMessage {
                    id: id.clone(),
                    author: "user".to_string(),
                    content: content.clone(),
                    timestamp: parsed_timestamp,
                    message_type: "message".to_string(),
                    metadata: None,
                }
            }
            _ => {
                // For other types, create a basic message
                ExportedMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    author: "system".to_string(),
                    content: "System message".to_string(),
                    timestamp: Local::now(),
                    message_type: "system".to_string(),
                    metadata: None,
                }
            }
        }
    }
}