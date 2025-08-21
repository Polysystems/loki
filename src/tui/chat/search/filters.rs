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
    
    /// Minimum message length
    pub min_length: Option<usize>,
    
    /// Maximum message length
    pub max_length: Option<usize>,
    
    /// Filter for messages with code blocks
    pub has_code_blocks: Option<bool>,
    
    /// Filter for messages with links
    pub has_links: Option<bool>,
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