//! Smart context management with intelligent compression
//! 
//! Manages context window, token counting, and automatic compression

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::token_estimator::{TokenEstimator};

/// Smart context manager for efficient token usage
#[derive(Debug, Clone)]
pub struct SmartContextManager {
    /// Current token count in active context window
    pub current_tokens: usize,

    /// Maximum tokens for current model
    pub max_context_tokens: usize,

    /// Threshold at which to trigger context compaction (default: 85% of max)
    pub compaction_threshold: f32,

    /// Context chunks (older conversations that have been compressed)
    pub context_chunks: Vec<ContextChunk>,

    /// Active messages in current context window
    pub active_messages: Vec<usize>, // Message indices in the conversation

    /// Summary of compressed context
    pub compressed_context_summary: Option<String>,

    /// Background compaction status
    pub is_compacting: bool,

    /// Compaction progress (0.0 to 1.0)
    pub compaction_progress: f32,

    /// Token estimation model
    pub token_estimator: TokenEstimator,

    /// Auto-compaction enabled
    pub auto_compaction_enabled: bool,

    /// Last compaction time
    pub last_compaction: Option<DateTime<Utc>>,

    /// Compaction statistics
    pub compaction_stats: CompactionStats,
}

/// Compressed context chunk representing older conversation segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    /// Unique chunk identifier
    pub chunk_id: String,

    /// Summary of this chunk's conversation
    pub summary: String,

    /// Key topics and themes in this chunk
    pub key_topics: Vec<String>,

    /// Important entities mentioned (files, configs, errors, etc.)
    pub important_entities: Vec<String>,

    /// Time range this chunk covers
    pub time_range: (DateTime<Utc>, DateTime<Utc>),

    /// Original message count in this chunk
    pub original_message_count: usize,

    /// Original token count before compression
    pub original_tokens: usize,

    /// Compressed token count
    pub compressed_tokens: usize,

    /// Compression ratio achieved
    pub compression_ratio: f32,

    /// Importance score of this chunk
    pub importance_score: f32,
}

/// Statistics about context compaction
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Total number of compactions performed
    pub total_compactions: usize,

    /// Total tokens saved through compaction
    pub total_tokens_saved: usize,

    /// Average compression ratio achieved
    pub average_compression_ratio: f32,

    /// Total messages compressed
    pub total_messages_compressed: usize,

    /// Last compaction duration
    pub last_compaction_duration_ms: u64,
}

impl Default for SmartContextManager {
    fn default() -> Self {
        Self {
            current_tokens: 0,
            max_context_tokens: 8192, // Default for GPT-4
            compaction_threshold: 0.85,
            context_chunks: Vec::new(),
            active_messages: Vec::new(),
            compressed_context_summary: None,
            is_compacting: false,
            compaction_progress: 0.0,
            token_estimator: TokenEstimator::default(),
            auto_compaction_enabled: true,
            last_compaction: None,
            compaction_stats: CompactionStats::default(),
        }
    }
}

impl SmartContextManager {
    /// Create a new context manager with specified token limit
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_context_tokens: max_tokens,
            ..Default::default()
        }
    }
    
    /// Estimate tokens for a given text
    pub fn estimate_tokens(&self, text: &str) -> usize {
        self.token_estimator.estimate_tokens(text)
    }
    
    /// Check if compaction is needed
    pub fn needs_compaction(&self) -> bool {
        let threshold = (self.max_context_tokens as f32 * self.compaction_threshold) as usize;
        self.current_tokens > threshold
    }
    
    /// Update current token count
    pub fn update_token_count(&mut self, tokens: usize) {
        self.current_tokens = tokens;
    }
    
    /// Add a context chunk
    pub fn add_context_chunk(&mut self, summary: String, topics: Vec<String>, tokens: usize) {
        let chunk = ContextChunk {
            chunk_id: uuid::Uuid::new_v4().to_string(),
            summary,
            key_topics: topics,
            important_entities: Vec::new(),
            time_range: (Utc::now(), Utc::now()),
            original_message_count: 0,
            original_tokens: tokens,
            compressed_tokens: tokens / 10, // Estimate 10x compression
            compression_ratio: 10.0,
            importance_score: 0.5,
        };
        
        self.context_chunks.push(chunk);
    }
    
    /// Get context utilization percentage
    pub fn utilization_percentage(&self) -> f32 {
        (self.current_tokens as f32 / self.max_context_tokens as f32) * 100.0
    }
    
    /// Get suggestions for context optimization
    pub fn get_optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if self.utilization_percentage() > 90.0 {
            suggestions.push("Consider clearing old messages to free up context".to_string());
        }
        
        if self.context_chunks.len() > 5 {
            suggestions.push("Many compressed chunks - consider archiving older conversations".to_string());
        }
        
        if !self.auto_compaction_enabled && self.needs_compaction() {
            suggestions.push("Enable auto-compaction to optimize context usage".to_string());
        }
        
        suggestions
    }
}