//! Utilities Subtab Manager - Coordinates all utility tabs

use std::sync::Arc;
use anyhow::Result;
use ratatui::prelude::*;
use crossterm::event::KeyEvent;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::state::UtilitiesState;
use super::subtabs::{
    UtilitiesSubtabController,
    ToolsTab,
    McpTab,
    PluginsTab,
    DaemonTab,
};
use super::types::UtilitiesAction;
use super::config::ConfigManager;

use crate::tools::IntelligentToolManager;
use crate::tools::task_management::TaskManager;
use crate::mcp::McpManager;
use crate::plugins::PluginManager;
use crate::daemon::DaemonClient;

/// Manager that coordinates all utility subtabs
pub struct UtilitiesSubtabManager {
    /// All available tabs
    tabs: Vec<Box<dyn UtilitiesSubtabController + Send>>,
    
    /// Currently active tab index
    current_index: usize,
    
    /// Shared state
    state: Arc<RwLock<UtilitiesState>>,
    
    /// Backend connections
    tool_manager: Option<Arc<IntelligentToolManager>>,
    task_manager: Option<Arc<TaskManager>>,
    mcp_manager: Option<Arc<McpManager>>,
    plugin_manager: Option<Arc<PluginManager>>,
    daemon_client: Option<Arc<DaemonClient>>,
    
    
    /// Configuration manager
    config_manager: Arc<RwLock<ConfigManager>>,
}

impl UtilitiesSubtabManager {
    /// Create a new subtab manager with all tabs initialized
    pub fn new(
        state: Arc<RwLock<UtilitiesState>>,
        tool_manager: Option<Arc<IntelligentToolManager>>,
        mcp_manager: Option<Arc<McpManager>>,
        plugin_manager: Option<Arc<PluginManager>>,
        daemon_client: Option<Arc<DaemonClient>>,
    ) -> Result<Self> {
        // Create config manager
        let config_manager = Arc::new(RwLock::new(ConfigManager::new()?));
        
        let mut tabs: Vec<Box<dyn UtilitiesSubtabController + Send>> = Vec::new();
        
        // Create all tabs
        tabs.push(Box::new(ToolsTab::new(
            state.clone(),
            tool_manager.clone(),
            Some(config_manager.clone()),
        )));
        
        tabs.push(Box::new(McpTab::new(
            state.clone(),
            mcp_manager.clone(),
            Some(config_manager.clone()),
        )));
        
        tabs.push(Box::new(PluginsTab::new(
            state.clone(),
            plugin_manager.clone(),
            Some(config_manager.clone()),
        )));
        
        tabs.push(Box::new(DaemonTab::new(
            state.clone(),
            daemon_client.clone(),
        )));
        
        // Create TodoManager and TodosTab
        // Note: TodoManager requires PipelineOrchestrator and ModelCallTracker
        // which are not available here, so we'll create a basic instance
        // This will be properly initialized when the orchestration is set up
        
        info!("Initialized {} utility tabs", tabs.len());
        
        Ok(Self {
            tabs,
            current_index: 0,
            state,
            tool_manager,
            task_manager: None,
            mcp_manager,
            plugin_manager,
            daemon_client,
            config_manager,
        })
    }
    
    /// Update backend connections
    pub fn update_connections(
        &mut self,
        tool_manager: Option<Arc<IntelligentToolManager>>,
        mcp_manager: Option<Arc<McpManager>>,
        plugin_manager: Option<Arc<PluginManager>>,
        daemon_client: Option<Arc<DaemonClient>>,
    ) {
        self.tool_manager = tool_manager.clone();
        self.mcp_manager = mcp_manager.clone();
        self.plugin_manager = plugin_manager.clone();
        self.daemon_client = daemon_client.clone();
        
        // Update individual tabs using downcasting
        self.update_tab_connections();
    }
    
    /// Update connections for individual tabs using downcasting
    fn update_tab_connections(&mut self) {
        for (i, tab) in self.tabs.iter_mut().enumerate() {
            let tab_any = tab.as_any_mut();
            
            match i {
                0 => {
                    // Tools tab
                    if let Some(tools_tab) = tab_any.downcast_mut::<ToolsTab>() {
                        tools_tab.update_tool_manager(self.tool_manager.clone());
                        debug!("Updated tool manager for Tools tab");
                    }
                }
                1 => {
                    // MCP tab
                    if let Some(mcp_tab) = tab_any.downcast_mut::<McpTab>() {
                        mcp_tab.update_mcp_manager(self.mcp_manager.clone());
                        debug!("Updated MCP manager for MCP tab");
                    }
                }
                2 => {
                    // Plugins tab
                    if let Some(plugins_tab) = tab_any.downcast_mut::<PluginsTab>() {
                        plugins_tab.update_plugin_manager(self.plugin_manager.clone());
                        debug!("Updated plugin manager for Plugins tab");
                    }
                }
                3 => {
                    // Daemon tab
                    if let Some(daemon_tab) = tab_any.downcast_mut::<DaemonTab>() {
                        daemon_tab.update_daemon_client(self.daemon_client.clone());
                        debug!("Updated daemon client for Daemon tab");
                    }
                }
                _ => {
                    // Other tabs (like Todos) don't need backend updates
                }
            }
        }
    }
    
    
    /// Render the current tab
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        // Draw tab bar at the top
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Tab bar
                Constraint::Min(0),     // Content area
            ])
            .split(area);
        
        self.render_tab_bar(f, chunks[0]);
        
        // Render the active tab
        if let Some(tab) = self.tabs.get_mut(self.current_index) {
            tab.render(f, chunks[1]);
        }
    }
    
    /// Render the tab bar
    fn render_tab_bar(&self, f: &mut Frame, area: Rect) {
        use ratatui::widgets::{Block, Borders, Tabs};
        
        let tab_titles: Vec<Line> = self.tabs
            .iter()
            .enumerate()
            .map(|(i, tab)| {
                let title = tab.name();
                if i == self.current_index {
                    Line::from(vec![
                        Span::styled(title, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                    ])
                } else {
                    Line::from(vec![
                        Span::styled(title, Style::default().fg(Color::White))
                    ])
                }
            })
            .collect();
        
        let tabs = Tabs::new(tab_titles)
            .block(Block::default()
                .borders(Borders::ALL)
                .title("🛠️ Utilities")
                .title_bottom(" Ctrl+J/K or Tab: Navigate Tabs "))
            .select(self.current_index)
            .divider(" │ ");
        
        f.render_widget(tabs, area);
    }
    
    /// Handle key events
    pub async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool> {
        use crossterm::event::{KeyCode, KeyModifiers};
        
        // Check for tab switching
        match event.code {
            KeyCode::Tab => {
                self.next_tab();
                return Ok(true);
            }
            KeyCode::BackTab => {
                self.previous_tab();
                return Ok(true);
            }
            // Add Ctrl+J for next tab
            KeyCode::Char('j') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                self.next_tab();
                return Ok(true);
            }
            // Add Ctrl+K for previous tab
            KeyCode::Char('k') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                self.previous_tab();
                return Ok(true);
            }
            _ => {}
        }
        
        // Pass to active tab
        if let Some(tab) = self.tabs.get_mut(self.current_index) {
            tab.handle_key_event(event).await
        } else {
            Ok(false)
        }
    }
    
    /// Switch to the next tab
    pub fn next_tab(&mut self) {
        self.current_index = (self.current_index + 1) % self.tabs.len();
        debug!("Switched to tab: {}", self.current_tab_name());
    }
    
    /// Switch to the previous tab
    pub fn previous_tab(&mut self) {
        if self.current_index == 0 {
            self.current_index = self.tabs.len() - 1;
        } else {
            self.current_index -= 1;
        }
        debug!("Switched to tab: {}", self.current_tab_name());
    }
    
    /// Switch to a specific tab by index
    pub fn switch_tab(&mut self, index: usize) {
        if index < self.tabs.len() {
            self.current_index = index;
            debug!("Switched to tab: {}", self.current_tab_name());
        }
    }
    
    /// Get the current tab name
    pub fn current_tab_name(&self) -> String {
        self.tabs
            .get(self.current_index)
            .map(|tab| tab.name().to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }
    
    /// Get the number of tabs
    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }
    
    /// Handle cross-tab actions
    pub async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()> {
        // Route action to appropriate tab or handle globally
        match action {
            UtilitiesAction::RefreshAll => {
                for tab in &mut self.tabs {
                    tab.refresh().await?;
                }
            }
            _ => {
                // Pass to current tab
                if let Some(tab) = self.tabs.get_mut(self.current_index) {
                    tab.handle_action(action).await?;
                }
            }
        }
        Ok(())
    }
}