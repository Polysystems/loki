//! Chat system integrations
//! 
//! Connects chat to various subsystems

pub mod cognitive;
pub mod story;
pub mod nlp;
pub mod tools;

// Re-export commonly used types
pub use cognitive::{CognitiveChatEnhancement, CognitiveIntegration};
pub use story::StoryChatIntegration;
pub use tools::ToolIntegration;
pub use nlp::NlpIntegration;
