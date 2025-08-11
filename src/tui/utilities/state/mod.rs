//! Utilities state management

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

use crate::tui::utilities::types::{UtilitiesCache, ToolEntry, McpServerStatus, PluginInfo, DaemonStatus};

/// Main utilities state
#[derive(Debug, Clone)]
pub struct UtilitiesState {
    /// Cached data for display
    pub cache: UtilitiesCache,
    
    /// Selected indices for each tab
    pub selected_tool: usize,
    pub selected_mcp_server: usize,
    pub selected_plugin: usize,
    pub selected_daemon: usize,
    
    /// Edit states
    pub editing_tool_config: Option<String>,
    pub editing_mcp_config: Option<String>,
    
    /// Search/filter states
    pub tool_search_query: String,
    pub plugin_search_query: String,
    pub mcp_search_query: String,
    
    /// View modes
    pub tool_view_mode: ToolViewMode,
    pub mcp_view_mode: McpViewMode,
    pub plugin_view_mode: PluginViewMode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolViewMode {
    List,
    Details,
    Configure,
}

#[derive(Debug, Clone, PartialEq)]
pub enum McpViewMode {
    LocalServers,
    Marketplace,
    Editor,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PluginViewMode {
    Installed,
    Marketplace,
    Details,
}

impl UtilitiesState {
    /// Create new utilities state
    pub fn new() -> Self {
        Self {
            cache: UtilitiesCache {
                tools: Vec::new(),
                mcp_servers: HashMap::new(),
                plugins: Vec::new(),
                daemons: HashMap::new(),
                last_update: Utc::now(),
            },
            selected_tool: 0,
            selected_mcp_server: 0,
            selected_plugin: 0,
            selected_daemon: 0,
            editing_tool_config: None,
            editing_mcp_config: None,
            tool_search_query: String::new(),
            plugin_search_query: String::new(),
            mcp_search_query: String::new(),
            tool_view_mode: ToolViewMode::List,
            mcp_view_mode: McpViewMode::LocalServers,
            plugin_view_mode: PluginViewMode::Installed,
        }
    }
    
    /// Update cache with new data
    pub fn update_cache(&mut self, cache: UtilitiesCache) {
        self.cache = cache;
    }
    
    /// Get filtered tools based on search query
    pub fn get_filtered_tools(&self) -> Vec<ToolEntry> {
        if self.tool_search_query.is_empty() {
            self.cache.tools.clone()
        } else {
            let query = self.tool_search_query.to_lowercase();
            self.cache.tools
                .iter()
                .filter(|tool| {
                    tool.name.to_lowercase().contains(&query)
                        || tool.description.to_lowercase().contains(&query)
                        || tool.category.to_lowercase().contains(&query)
                })
                .cloned()
                .collect()
        }
    }
    
    /// Get filtered plugins based on search query
    pub fn get_filtered_plugins(&self) -> Vec<PluginInfo> {
        if self.plugin_search_query.is_empty() {
            self.cache.plugins.clone()
        } else {
            let query = self.plugin_search_query.to_lowercase();
            self.cache.plugins
                .iter()
                .filter(|plugin| {
                    plugin.name.to_lowercase().contains(&query)
                        || plugin.description.to_lowercase().contains(&query)
                        || plugin.author.to_lowercase().contains(&query)
                })
                .cloned()
                .collect()
        }
    }
}