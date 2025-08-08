//! Context management module
//! 
//! Smart context management, token estimation, and conversation indexing

pub mod smart_context;
pub mod token_estimator;
pub mod suggestions;
pub mod indexer;

// Re-export commonly used types
pub use smart_context::{SmartContextManager, ContextChunk};
pub use token_estimator::{TokenEstimator, EstimationStats, CompactionStats};
pub use suggestions::{ContextSuggestion, ContextType};
pub use indexer::ConversationIndexer;