//! Chat settings management

use serde::{Serialize, Deserialize};

/// Chat configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSettings {
    pub store_history: bool,
    pub threads: usize,
    pub temperature: f32,
    pub max_tokens: u32,
    pub auto_save: bool,
    pub word_wrap: bool,
    pub dark_theme: bool,
    pub save_history: bool,
    pub enhanced_ui: bool,
    pub auto_project_detection: bool,
    pub show_token_count: bool,
    pub stream_responses: bool,
    pub enable_code_completion: bool,
    pub enable_smart_context: bool,
    pub multi_panel_enabled: bool,
    pub api_endpoint: Option<String>,
    pub default_model: Option<String>,
    pub selected_index: usize,
}

impl Default for ChatSettings {
    fn default() -> Self {
        Self {
            store_history: false,
            threads: 1,
            temperature: 0.7,
            max_tokens: 4096,
            auto_save: true,
            word_wrap: true,
            dark_theme: true,
            save_history: false,
            enhanced_ui: false,
            auto_project_detection: true,
            show_token_count: true,
            stream_responses: true,
            enable_code_completion: true,
            enable_smart_context: true,
            multi_panel_enabled: true,
            api_endpoint: None,
            default_model: None,
            selected_index: 0,
        }
    }
}

impl ChatSettings {
    /// Create settings with enhanced UI features
    pub fn with_enhanced_ui() -> Self {
        Self {
            enhanced_ui: true,
            multi_panel_enabled: true,
            show_token_count: true,
            enable_smart_context: true,
            ..Default::default()
        }
    }
    
    /// Create minimal settings for performance
    pub fn minimal() -> Self {
        Self {
            enhanced_ui: false,
            multi_panel_enabled: false,
            show_token_count: false,
            enable_code_completion: false,
            enable_smart_context: false,
            stream_responses: false,
            ..Default::default()
        }
    }
}