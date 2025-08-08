//! Token estimation and counting utilities

use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

/// Model-specific token counting method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelTokenType {
    /// OpenAI GPT models (tiktoken-style)
    OpenAI,
    /// Anthropic Claude models
    Anthropic,
    /// Local/Ollama models (character-based estimation)
    Local,
    /// Custom token counting
    Custom(f32), // Custom tokens-per-character ratio
}

/// Token estimation and counting system
#[derive(Debug, Clone)]
pub struct TokenEstimator {
    /// Tokens per character ratio for current model
    pub tokens_per_char: f32,

    /// Model-specific token counting method
    pub model_type: ModelTokenType,

    /// Cache of recent estimations for performance
    pub estimation_cache: HashMap<u64, usize>, // hash -> token count

    /// Estimation statistics for accuracy tracking
    pub stats: EstimationStats,
}

/// Statistics for token estimation accuracy
#[derive(Debug, Clone)]
pub struct EstimationStats {
    /// Total estimations made
    pub total_estimations: usize,

    /// Average estimation accuracy when verified
    pub average_accuracy: f32,

    /// Estimation vs actual token counts (for calibration)
    pub calibration_data: VecDeque<(usize, usize)>, // (estimated, actual)
}

/// Context compaction statistics and metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Total compactions performed
    pub total_compactions: usize,

    /// Total tokens saved through compression
    pub tokens_saved: usize,

    /// Average compression ratio achieved
    pub average_compression_ratio: f32,

    /// Total processing time for compactions (in ms)
    pub total_processing_time_ms: u64,

    /// Successful vs failed compactions
    pub success_rate: f32,

    /// Last compaction metrics
    pub last_compaction_metrics: Option<CompactionMetrics>,
}

/// Metrics for individual compaction operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// Messages processed
    pub messages_processed: usize,

    /// Tokens before compression
    pub tokens_before: usize,

    /// Tokens after compression
    pub tokens_after: usize,

    /// Processing time taken (in ms)
    pub processing_time_ms: u64,

    /// Compression ratio achieved
    pub compression_ratio: f32,

    /// Quality score of compression (0.0 to 1.0)
    pub quality_score: f32,
}

use std::collections::HashMap;

impl Default for TokenEstimator {
    fn default() -> Self {
        Self {
            tokens_per_char: 0.25, // Default ratio
            model_type: ModelTokenType::OpenAI,
            estimation_cache: HashMap::new(),
            stats: EstimationStats::default(),
        }
    }
}

impl Default for EstimationStats {
    fn default() -> Self {
        Self {
            total_estimations: 0,
            average_accuracy: 1.0,
            calibration_data: VecDeque::with_capacity(100),
        }
    }
}

impl TokenEstimator {
    /// Create a new token estimator for a specific model type
    pub fn new(model_type: ModelTokenType) -> Self {
        let tokens_per_char = match &model_type {
            ModelTokenType::OpenAI => 0.25,
            ModelTokenType::Anthropic => 0.23,
            ModelTokenType::Local => 0.3,
            ModelTokenType::Custom(ratio) => *ratio,
        };
        
        Self {
            tokens_per_char,
            model_type,
            estimation_cache: HashMap::new(),
            stats: EstimationStats::default(),
        }
    }
    
    /// Estimate token count for a given text
    pub fn estimate_tokens(&self, text: &str) -> usize {
        // Check cache first
        let hash = self.hash_text(text);
        if let Some(&cached) = self.estimation_cache.get(&hash) {
            return cached;
        }
        
        // Basic estimation based on model type
        let tokens = match &self.model_type {
            ModelTokenType::OpenAI | ModelTokenType::Anthropic => {
                // More sophisticated estimation for API models
                let chars = text.chars().count();
                let words = text.split_whitespace().count();
                let avg = (chars as f32 * self.tokens_per_char + words as f32 * 1.3) / 2.0;
                avg as usize
            }
            ModelTokenType::Local | ModelTokenType::Custom(_) => {
                // Simple character-based estimation
                (text.chars().count() as f32 * self.tokens_per_char) as usize
            }
        };
        
        tokens
    }
    
    /// Update estimation accuracy with actual token count
    pub fn update_calibration(&mut self, estimated: usize, actual: usize) {
        self.stats.calibration_data.push_back((estimated, actual));
        
        // Keep only last 100 calibrations
        while self.stats.calibration_data.len() > 100 {
            self.stats.calibration_data.pop_front();
        }
        
        // Update accuracy
        if !self.stats.calibration_data.is_empty() {
            let total_error: f32 = self.stats.calibration_data.iter()
                .map(|(est, act)| {
                    let diff = (*est as f32 - *act as f32).abs();
                    diff / *act as f32
                })
                .sum();
            
            self.stats.average_accuracy = 1.0 - (total_error / self.stats.calibration_data.len() as f32);
        }
    }
    
    /// Hash text for caching
    fn hash_text(&self, text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
}