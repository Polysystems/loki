//! Type definitions and conversions for the chat module

pub mod conversions;
pub mod message;

// Re-export commonly used types
pub use conversions::orchestration::sync_orchestration_config;
pub use message::{Message, MessageRole, MessageMetadata};