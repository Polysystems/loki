//! Centralized keybinding management system
//! 
//! This module provides a unified system for managing keybindings across the entire application,
//! preventing conflicts and allowing user customization.

pub mod registry;
pub mod config;
pub mod actions;

pub use registry::{KeyBindingRegistry, KeyContext, InputResult};
pub use config::{KeyBindingConfig, KeyCombo};
pub use actions::{Action, ActionCategory};

use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;

/// Global keybinding registry instance
pub static KEYBINDING_REGISTRY: Lazy<Arc<RwLock<KeyBindingRegistry>>> = Lazy::new(|| {
    Arc::new(RwLock::new(KeyBindingRegistry::new()))
});

/// Initialize the keybinding system with default bindings
pub fn initialize() -> anyhow::Result<()> {
    let mut registry = KEYBINDING_REGISTRY.write().unwrap();
    registry.load_defaults()?;
    
    // Try to load user configuration if it exists
    if let Ok(config) = KeyBindingConfig::load_from_file() {
        registry.apply_config(config)?;
    }
    
    Ok(())
}

/// Get a reference to the global keybinding registry
pub fn get_registry() -> Arc<RwLock<KeyBindingRegistry>> {
    KEYBINDING_REGISTRY.clone()
}