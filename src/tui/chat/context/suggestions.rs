//! Context suggestions and recommendations

use serde::{Serialize, Deserialize};

/// Context suggestion for smart completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSuggestion {
    pub id: String,
    pub suggestion_type: ContextType,
    pub content: String,
    pub confidence: f32,
    pub source: String,
}

/// Types of context that can be suggested
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContextType {
    Code,
    Documentation,
    Error,
    Command,
    File,
    Configuration,
    Memory,
    Tool,
}