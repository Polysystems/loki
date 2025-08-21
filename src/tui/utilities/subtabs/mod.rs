//! Utilities subtab implementations

pub mod tools_tab;
pub mod mcp_tab;
pub mod plugins_tab;
pub mod daemon_tab;

use async_trait::async_trait;
use anyhow::Result;
use ratatui::prelude::*;
use crossterm::event::KeyEvent;
use std::any::Any;

use crate::tui::utilities::types::UtilitiesAction;

// Re-export tabs
pub use tools_tab::ToolsTab;
pub use mcp_tab::McpTab;
pub use plugins_tab::PluginsTab;
pub use daemon_tab::DaemonTab;

/// Trait for utilities subtab controllers
#[async_trait]
pub trait UtilitiesSubtabController: Send {
    /// Get the tab name
    fn name(&self) -> &str;
    
    /// Get mutable reference as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
    
    /// Render the tab content
    fn render(&mut self, f: &mut Frame, area: Rect);
    
    /// Handle key events
    async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool>;
    
    /// Handle actions from other tabs
    async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()>;
    
    /// Refresh tab data
    async fn refresh(&mut self) -> Result<()>;
    
    /// Check if the tab is in an editing mode
    fn is_editing(&self) -> bool {
        false
    }
}