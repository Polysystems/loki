//! Modular Utilities Tab System
//!
//! This module provides a modular architecture for the utilities tab,
//! following the same pattern as the chat system for better maintainability
//! and separation of concerns.

pub mod bridges;
pub mod components;
pub mod config;
pub mod handlers;
pub mod integration;
pub mod metrics;
pub mod rendering;
pub mod state;
pub mod subtabs;
pub mod types;

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::sync::{Arc, RwLock as StdRwLock};
use anyhow::Result;
use tokio::sync::RwLock;
use ratatui::prelude::*;
use crossterm::event::KeyEvent;
use std::collections::HashMap;

// Re-export core types
pub use integration::UtilitiesSubtabManager;
pub use state::UtilitiesState;
pub use types::*;

use crate::tools::IntelligentToolManager;
use crate::mcp::McpManager;
use crate::plugins::PluginManager;
use crate::daemon::DaemonClient;
use self::config::ConfigManager;

/// Cached metrics structure for monitoring
#[derive(Debug, Clone, Default)]
pub struct CachedMetrics {
    pub last_update: Option<std::time::Instant>,
    pub tool_metrics: HashMap<String, f64>,
    pub mcp_metrics: HashMap<String, f64>,
    pub plugin_metrics: HashMap<String, f64>,
    pub daemon_metrics: HashMap<String, f64>,
    pub system_metrics: HashMap<String, f64>,
}

/// Main modular utilities structure
pub struct ModularUtilities {
    /// Subtab manager that coordinates all utility tabs
    pub subtab_manager: RefCell<UtilitiesSubtabManager>,
    
    /// Shared state for all utilities
    pub state: Arc<RwLock<UtilitiesState>>,
    
    /// Tool manager connection
    pub tool_manager: Option<Arc<IntelligentToolManager>>,
    
    /// MCP manager connection
    pub mcp_manager: Option<Arc<McpManager>>,
    
    /// Plugin manager connection
    pub plugin_manager: Option<Arc<PluginManager>>,
    
    /// Daemon client connection
    pub daemon_client: Option<Arc<DaemonClient>>,
    
    /// Currently active utility tab
    pub active_tab: usize,
    
    /// Cached metrics for monitoring (compatibility with old code)
    pub cached_metrics: Arc<StdRwLock<CachedMetrics>>,
    
    /// Configuration manager
    pub config_manager: Option<ConfigManager>,
}

impl ModularUtilities {
    /// Create a new modular utilities system
    pub fn new() -> Self {
        let state = Arc::new(RwLock::new(UtilitiesState::new()));
        
        let subtab_manager = UtilitiesSubtabManager::new(
            state.clone(),
            None, // tool_manager will be set later
            None, // mcp_manager will be set later
            None, // plugin_manager will be set later
            None, // daemon_client will be set later
        ).expect("Failed to create UtilitiesSubtabManager");
        
        // Try to create config manager
        let config_manager = match ConfigManager::new() {
            Ok(mgr) => Some(mgr),
            Err(e) => {
                tracing::warn!("Failed to initialize config manager: {}", e);
                None
            }
        };
        
        Self {
            subtab_manager: RefCell::new(subtab_manager),
            state,
            tool_manager: None,
            mcp_manager: None,
            plugin_manager: None,
            daemon_client: None,
            active_tab: 0,
            cached_metrics: Arc::new(StdRwLock::new(CachedMetrics::default())),
            config_manager,
        }
    }
    
    /// Connect to backend systems (compatibility with old interface)
    pub fn connect_systems(
        &mut self,
        mcp_client: Option<Arc<crate::mcp::McpClient>>,
        tool_manager: Option<Arc<IntelligentToolManager>>,
        _monitoring_system: Option<Arc<dyn std::any::Any + Send + Sync>>, // RealTimeMonitor type not available
        _real_time_aggregator: Option<Arc<crate::tui::real_time_integration::RealTimeMetricsAggregator>>,
        _health_monitor: Option<Arc<crate::monitoring::health::HealthMonitor>>,
        _safety_validator: Option<Arc<dyn std::any::Any + Send + Sync>>, // ActionValidator type not available
        _cognitive_system: Option<Arc<crate::cognitive::CognitiveSystem>>,
        _memory_system: Option<Arc<crate::memory::CognitiveMemory>>,
        plugin_manager: Option<Arc<PluginManager>>,
        daemon_client: Option<Arc<DaemonClient>>,
        _natural_language_orchestrator: Option<Arc<dyn std::any::Any + Send + Sync>>, // NaturalLanguageOrchestrator type not available
    ) {
        // Create MCP manager from client if provided
        let mcp_manager = mcp_client.map(|client| {
            Arc::new(crate::mcp::McpManager {
                client,
                discovery: crate::mcp::McpDiscovery::new(),
                marketplace: crate::mcp::McpMarketplace::new(),
                config: crate::mcp::McpConfig::default(),
            })
        });
        
        self.tool_manager = tool_manager.clone();
        self.mcp_manager = mcp_manager.clone();
        self.plugin_manager = plugin_manager.clone();
        self.daemon_client = daemon_client.clone();
        
        // Update subtab manager with connections
        self.subtab_manager.get_mut().update_connections(
            tool_manager,
            mcp_manager,
            plugin_manager,
            daemon_client,
        );
    }
    
    /// Initialize with backend connections
    pub async fn initialize(
        &mut self,
        tool_manager: Option<Arc<IntelligentToolManager>>,
        mcp_manager: Option<Arc<McpManager>>,
        plugin_manager: Option<Arc<PluginManager>>,
        daemon_client: Option<Arc<DaemonClient>>,
    ) -> Result<()> {
        self.tool_manager = tool_manager.clone();
        self.mcp_manager = mcp_manager.clone();
        self.plugin_manager = plugin_manager.clone();
        self.daemon_client = daemon_client.clone();
        
        // Update subtab manager with connections
        self.subtab_manager.get_mut().update_connections(
            tool_manager,
            mcp_manager,
            plugin_manager,
            daemon_client,
        );
        
        Ok(())
    }
    
    /// Render the utilities interface
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        self.subtab_manager.get_mut().render(f, area);
    }
    
    /// Handle key events
    pub async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool> {
        self.subtab_manager.get_mut().handle_key_event(event).await
    }
    
    /// Get current tab name
    pub fn current_tab_name(&self) -> String {
        self.subtab_manager.borrow().current_tab_name()
    }
    
    /// Switch to a specific tab
    pub fn switch_tab(&mut self, index: usize) {
        self.subtab_manager.get_mut().switch_tab(index);
    }
    
    /// Get number of tabs
    pub fn tab_count(&self) -> usize {
        self.subtab_manager.borrow().tab_count()
    }
    
    // Compatibility methods for old interface
    
    /// Initialize example data (no-op in new system)
    pub async fn initialize_example_data(&mut self) {
        // No-op - data is initialized on demand in the new system
    }
    
    /// Update cache (no-op in new system)
    pub async fn update_cache(&mut self) -> Result<()> {
        // No-op - caching is handled internally by subtabs
        Ok(())
    }
    
    /// Handle tools input
    pub async fn handle_tools_input(&mut self, key: crossterm::event::KeyCode) -> Result<()> {
        use crossterm::event::{KeyEvent, KeyModifiers};
        let event = KeyEvent::new(key, KeyModifiers::empty());
        self.subtab_manager.get_mut().handle_key_event(event).await?;
        Ok(())
    }
    
    /// Handle MCP input
    pub async fn handle_mcp_input(&mut self, key: crossterm::event::KeyCode) -> Result<()> {
        use crossterm::event::{KeyEvent, KeyModifiers};
        let event = KeyEvent::new(key, KeyModifiers::empty());
        self.subtab_manager.get_mut().handle_key_event(event).await?;
        Ok(())
    }
    
    /// Handle daemon input
    pub async fn handle_daemon_input(&mut self, key: crossterm::event::KeyCode) -> Result<()> {
        use crossterm::event::{KeyEvent, KeyModifiers};
        let event = KeyEvent::new(key, KeyModifiers::empty());
        self.subtab_manager.get_mut().handle_key_event(event).await?;
        Ok(())
    }
    
    /// Handle plugins input
    pub async fn handle_plugins_input(&mut self, key: crossterm::event::KeyCode) -> Result<()> {
        use crossterm::event::{KeyEvent, KeyModifiers};
        let event = KeyEvent::new(key, KeyModifiers::empty());
        self.subtab_manager.get_mut().handle_key_event(event).await?;
        Ok(())
    }
    
    /// Handle monitoring input
    pub async fn handle_monitoring_input(&mut self, key: crossterm::event::KeyCode) -> Result<()> {
        use crossterm::event::{KeyEvent, KeyModifiers};
        let event = KeyEvent::new(key, KeyModifiers::empty());
        self.subtab_manager.get_mut().handle_key_event(event).await?;
        Ok(())
    }
    
    // Compatibility methods for old code that still references these fields
    
    /// Check if orchestrator capabilities are available
    pub fn has_orchestrator_capabilities(&self) -> bool {
        // Always return false for now - orchestrator not yet integrated
        false
    }
}

// Compatibility structures for fields that are accessed by old code
// These will be removed once we complete the migration
impl ModularUtilities {
    /// Get a dummy value for marketplace_plugins (used by old rendering code)
    pub fn marketplace_plugins(&self) -> Vec<String> {
        Vec::new()
    }
    
    /// Get a dummy value for installed_plugins (used by old rendering code)
    pub fn installed_plugins(&self) -> Vec<String> {
        Vec::new()
    }
}