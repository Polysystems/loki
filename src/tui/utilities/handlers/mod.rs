//! Input and command handlers for utilities system
//!
//! This module provides specialized handlers for processing user input,
//! executing commands, and managing interactions within the utilities system.

use anyhow::Result;
use async_trait::async_trait;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tracing::{debug, info};

use crate::tui::utilities::types::{UtilitiesAction, NotificationType};

/// Handler for keyboard shortcuts and navigation
pub struct KeyboardHandler {
    /// Current focus context
    context: FocusContext,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FocusContext {
    TabNavigation,
    ToolList,
    ToolConfig,
    McpServerList,
    McpConfig,
    PluginList,
    PluginConfig,
    DaemonList,
    DaemonLogs,
    Search,
    CommandPalette,
}

impl KeyboardHandler {
    pub fn new() -> Self {
        Self {
            context: FocusContext::TabNavigation,
        }
    }
    
    /// Set the current focus context
    pub fn set_context(&mut self, context: FocusContext) {
        self.context = context;
    }
    
    /// Process a keyboard event and return an action if applicable
    pub fn handle_key(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match self.context {
            FocusContext::TabNavigation => self.handle_tab_navigation(event),
            FocusContext::ToolList => self.handle_tool_list(event),
            FocusContext::ToolConfig => self.handle_config_editor(event),
            FocusContext::McpServerList => self.handle_mcp_list(event),
            FocusContext::McpConfig => self.handle_config_editor(event),
            FocusContext::PluginList => self.handle_plugin_list(event),
            FocusContext::PluginConfig => self.handle_config_editor(event),
            FocusContext::DaemonList => self.handle_daemon_list(event),
            FocusContext::DaemonLogs => self.handle_log_viewer(event),
            FocusContext::Search => self.handle_search(event),
            FocusContext::CommandPalette => self.handle_command_palette(event),
        }
    }
    
    fn handle_tab_navigation(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Tab => {
                // Switch to next tab
                None // Handled by subtab manager
            }
            KeyCode::BackTab => {
                // Switch to previous tab
                None // Handled by subtab manager
            }
            _ => None,
        }
    }
    
    fn handle_tool_list(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Enter => {
                // Open tool configuration
                Some(UtilitiesAction::OpenToolConfig("selected".to_string()))
            }
            KeyCode::Char('r') => {
                Some(UtilitiesAction::RefreshTools)
            }
            KeyCode::Char('e') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(UtilitiesAction::ExecuteTool(
                    "selected".to_string(),
                    serde_json::Value::Null,
                ))
            }
            _ => None,
        }
    }
    
    fn handle_mcp_list(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Char('c') => {
                Some(UtilitiesAction::ConnectMcpServer("selected".to_string()))
            }
            KeyCode::Char('d') => {
                Some(UtilitiesAction::DisconnectMcpServer("selected".to_string()))
            }
            KeyCode::Char('r') => {
                Some(UtilitiesAction::RefreshMcpServers)
            }
            _ => None,
        }
    }
    
    fn handle_plugin_list(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Char('i') => {
                Some(UtilitiesAction::InstallPlugin("selected".to_string()))
            }
            KeyCode::Char('u') => {
                Some(UtilitiesAction::UninstallPlugin("selected".to_string()))
            }
            KeyCode::Char('e') => {
                Some(UtilitiesAction::EnablePlugin("selected".to_string()))
            }
            KeyCode::Char('d') => {
                Some(UtilitiesAction::DisablePlugin("selected".to_string()))
            }
            KeyCode::Char('r') => {
                Some(UtilitiesAction::RefreshPlugins)
            }
            _ => None,
        }
    }
    
    fn handle_daemon_list(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Char('s') => {
                Some(UtilitiesAction::StartDaemon("selected".to_string()))
            }
            KeyCode::Char('S') => {
                Some(UtilitiesAction::StopDaemon("selected".to_string()))
            }
            KeyCode::Char('r') => {
                Some(UtilitiesAction::RestartDaemon("selected".to_string()))
            }
            KeyCode::Char('l') => {
                Some(UtilitiesAction::ViewDaemonLogs("selected".to_string()))
            }
            _ => None,
        }
    }
    
    fn handle_config_editor(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Esc => {
                // Exit editor
                self.context = FocusContext::TabNavigation;
                None
            }
            KeyCode::Char('s') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                // Save configuration
                Some(UtilitiesAction::ShowNotification(
                    "Configuration saved".to_string(),
                    NotificationType::Success,
                ))
            }
            _ => None,
        }
    }
    
    fn handle_log_viewer(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.context = FocusContext::DaemonList;
                None
            }
            _ => None,
        }
    }
    
    fn handle_search(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Esc => {
                self.context = FocusContext::TabNavigation;
                None
            }
            KeyCode::Enter => {
                // Execute search
                None
            }
            _ => None,
        }
    }
    
    fn handle_command_palette(&mut self, event: KeyEvent) -> Option<UtilitiesAction> {
        match event.code {
            KeyCode::Esc => {
                self.context = FocusContext::TabNavigation;
                None
            }
            KeyCode::Enter => {
                // Execute command
                None
            }
            _ => None,
        }
    }
}

/// Handler for executing commands and actions
pub struct CommandHandler {
    /// Command history for replay
    history: Vec<UtilitiesAction>,
    
    /// Maximum history size
    max_history: usize,
}

impl CommandHandler {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 100,
        }
    }
    
    /// Execute a utilities action
    pub async fn execute(&mut self, action: UtilitiesAction) -> Result<()> {
        debug!("Executing action: {:?}", action);
        
        // Add to history
        self.add_to_history(action.clone());
        
        // Process the action
        match action {
            UtilitiesAction::RefreshAll => {
                info!("Refreshing all utilities data");
                // Trigger refresh for all subtabs
            }
            UtilitiesAction::ShowNotification(msg, notification_type) => {
                info!("Notification ({}): {}", 
                    match notification_type {
                        NotificationType::Info => "Info",
                        NotificationType::Success => "Success",
                        NotificationType::Warning => "Warning",
                        NotificationType::Error => "Error",
                    },
                    msg
                );
            }
            _ => {
                // Other actions are handled by specific subtabs
            }
        }
        
        Ok(())
    }
    
    /// Add action to history
    fn add_to_history(&mut self, action: UtilitiesAction) {
        self.history.push(action);
        
        // Trim history if it exceeds max size
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
    
    /// Get command history
    pub fn get_history(&self) -> &[UtilitiesAction] {
        &self.history
    }
    
    /// Clear command history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
    
    /// Replay last command
    pub fn replay_last(&self) -> Option<UtilitiesAction> {
        self.history.last().cloned()
    }
}

/// Handler for search functionality
pub struct SearchHandler {
    /// Current search query
    query: String,
    
    /// Search results cache
    results: Vec<SearchResult>,
    
    /// Selected result index
    selected: usize,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub category: String,
    pub name: String,
    pub description: String,
    pub action: UtilitiesAction,
}

impl SearchHandler {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            results: Vec::new(),
            selected: 0,
        }
    }
    
    /// Update search query
    pub fn update_query(&mut self, query: String) {
        self.query = query;
        self.selected = 0;
    }
    
    /// Perform search across all utilities
    pub async fn search(&mut self, state: &crate::tui::utilities::state::UtilitiesState) -> Result<()> {
        debug!("Searching for: {}", self.query);
        
        self.results.clear();
        
        if self.query.is_empty() {
            return Ok(());
        }
        
        let query_lower = self.query.to_lowercase();
        
        // Search tools
        for tool in &state.cache.tools {
            if tool.name.to_lowercase().contains(&query_lower) 
                || tool.description.to_lowercase().contains(&query_lower) 
                || tool.category.to_lowercase().contains(&query_lower) {
                self.results.push(SearchResult {
                    category: "Tool".to_string(),
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    action: UtilitiesAction::OpenToolConfig(tool.id.clone()),
                });
            }
        }
        
        // Search MCP servers
        for (_, server) in &state.cache.mcp_servers {
            if server.name.to_lowercase().contains(&query_lower) 
                || server.description.to_lowercase().contains(&query_lower) {
                self.results.push(SearchResult {
                    category: "MCP Server".to_string(),
                    name: server.name.clone(),
                    description: server.description.clone(),
                    action: UtilitiesAction::ConnectMcpServer(server.name.clone()),
                });
            }
        }
        
        // Search plugins
        for plugin in &state.cache.plugins {
            if plugin.name.to_lowercase().contains(&query_lower) 
                || plugin.description.to_lowercase().contains(&query_lower) 
                || plugin.author.to_lowercase().contains(&query_lower) {
                self.results.push(SearchResult {
                    category: "Plugin".to_string(),
                    name: plugin.name.clone(),
                    description: plugin.description.clone(),
                    action: if plugin.enabled {
                        UtilitiesAction::DisablePlugin(plugin.id.clone())
                    } else {
                        UtilitiesAction::EnablePlugin(plugin.id.clone())
                    },
                });
            }
        }
        
        // Search daemons
        for (_, daemon) in &state.cache.daemons {
            if daemon.name.to_lowercase().contains(&query_lower) {
                self.results.push(SearchResult {
                    category: "Daemon".to_string(),
                    name: daemon.name.clone(),
                    description: format!("Status: {:?}", daemon.status),
                    action: match daemon.status {
                        crate::tui::utilities::types::DaemonState::Running => 
                            UtilitiesAction::StopDaemon(daemon.name.clone()),
                        _ => UtilitiesAction::StartDaemon(daemon.name.clone()),
                    },
                });
            }
        }
        
        debug!("Found {} search results", self.results.len());
        Ok(())
    }
    
    /// Get current search results
    pub fn get_results(&self) -> &[SearchResult] {
        &self.results
    }
    
    /// Select next result
    pub fn next_result(&mut self) {
        if !self.results.is_empty() {
            self.selected = (self.selected + 1) % self.results.len();
        }
    }
    
    /// Select previous result
    pub fn prev_result(&mut self) {
        if !self.results.is_empty() {
            if self.selected == 0 {
                self.selected = self.results.len() - 1;
            } else {
                self.selected -= 1;
            }
        }
    }
    
    /// Get selected result action
    pub fn get_selected_action(&self) -> Option<UtilitiesAction> {
        self.results.get(self.selected).map(|r| r.action.clone())
    }
    
    /// Clear search
    pub fn clear(&mut self) {
        self.query.clear();
        self.results.clear();
        self.selected = 0;
    }
}

/// Central handler coordinator
pub struct HandlerCoordinator {
    pub keyboard: KeyboardHandler,
    pub command: CommandHandler,
    pub search: SearchHandler,
}

impl HandlerCoordinator {
    pub fn new() -> Self {
        Self {
            keyboard: KeyboardHandler::new(),
            command: CommandHandler::new(),
            search: SearchHandler::new(),
        }
    }
    
    /// Process a key event
    pub async fn handle_key_event(&mut self, event: KeyEvent) -> Result<Option<UtilitiesAction>> {
        // Check for global shortcuts first
        if event.modifiers.contains(KeyModifiers::CONTROL) {
            match event.code {
                KeyCode::Char('p') => {
                    // Open command palette
                    self.keyboard.set_context(FocusContext::CommandPalette);
                    return Ok(None);
                }
                KeyCode::Char('f') => {
                    // Open search
                    self.keyboard.set_context(FocusContext::Search);
                    return Ok(None);
                }
                KeyCode::Char('r') => {
                    // Refresh all
                    return Ok(Some(UtilitiesAction::RefreshAll));
                }
                _ => {}
            }
        }
        
        // Handle based on current context
        Ok(self.keyboard.handle_key(event))
    }
    
    /// Execute an action
    pub async fn execute_action(&mut self, action: UtilitiesAction) -> Result<()> {
        self.command.execute(action).await
    }
    
    /// Perform search
    pub async fn search(&mut self, query: String, state: &crate::tui::utilities::state::UtilitiesState) -> Result<()> {
        self.search.update_query(query);
        self.search.search(state).await
    }
}