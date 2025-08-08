//! Consolidated cognitive system module

pub mod core;
pub mod commands;
pub mod persistence;
pub mod integration;

// Re-export commonly used items
pub use core::{
    controls::CognitiveControlState,
    mode_controls::CognitiveProcessingMode,
    tone_detector::CognitiveToneDetector,
};

pub use commands::{
    executor::CognitiveCommandExecutor,
    router::is_cognitive_command,
};

pub use persistence::{
    state::CognitiveStateRestorer,
    session::CognitiveSessionManager,
};