//! Search filters for chat messages

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Filters for searching chat messages
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatSearchFilters {
    /// Text query
    pub query: Option<String>,
    
    /// Filter by author
    pub author: Option<String>,
    
    /// Filter by message type
    pub message_type: Option<MessageTypeFilter>,
    
    /// Filter by date range
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    
    /// Filter by chat ID
    pub chat_id: Option<usize>,
    
    /// Include system messages
    pub include_system: bool,
    
    /// Case sensitive search
    pub case_sensitive: bool,
    
    /// Use regex
    pub use_regex: bool,
}

/// Message type filter options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageTypeFilter {
    UserMessages,
    AssistantMessages,
    SystemMessages,
    ToolExecutions,
    Errors,
    All,
}