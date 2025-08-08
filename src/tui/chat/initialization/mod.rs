//! Initialization and setup module
//! 
//! Handles initial configuration, API setup, and system bootstrapping

pub mod setup;
pub mod sync_components;
pub mod model_registry;

// Re-export commonly used types
pub use setup::{ChatConfig, ChatComponents};
pub use sync_components::sync_components_to_modular;
pub use model_registry::{ModelRegistry, RegisteredModel};