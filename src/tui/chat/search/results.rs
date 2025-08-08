//! Search result types

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Search result for chat messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSearchResult {
    /// Chat ID where message was found
    pub chat_id: usize,
    
    /// Message index in the chat
    pub message_index: usize,
    
    /// Snippet of the message with query highlighted
    pub snippet: String,
    
    /// Full message content
    pub full_content: String,
    
    /// Message author
    pub author: String,
    
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Relevance score
    pub score: f32,
    
    /// Context lines before
    pub context_before: Vec<String>,
    
    /// Context lines after
    pub context_after: Vec<String>,
}