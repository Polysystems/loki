//! Settings module for TUI
//! 
//! Provides centralized settings management and configuration

pub mod manager;

pub use manager::{
    SettingsManager,
    TuiSettings,
    ThemePreference,
    NotificationSettings,
    KeyboardShortcuts,
    initialize_settings_manager,
    get_settings_manager,
    get_or_create_settings_manager,
};