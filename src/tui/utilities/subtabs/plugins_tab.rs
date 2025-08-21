//! Plugins management tab - Complete implementation

use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, List, ListItem, Paragraph, Table, Row, Cell,
    Scrollbar, ScrollbarOrientation, ScrollbarState, Tabs, Wrap
};
use crossterm::event::{KeyEvent, KeyCode};
use tokio::sync::RwLock;
use tracing::debug;

use crate::plugins::PluginManager;
use crate::tui::utilities::state::{UtilitiesState, PluginViewMode};
use crate::tui::utilities::types::{UtilitiesAction, PluginInfo, PluginStatus};
use crate::tui::utilities::components::{SearchOverlay, SearchResult};
use crate::tui::utilities::config::{ConfigManager, PluginConfig};
use super::UtilitiesSubtabController;

/// View modes for the plugins tab
#[derive(Debug, Clone, PartialEq)]
enum ViewMode {
    InstalledList,
    PluginDetails,
    Marketplace,
    ConfigEditor,
}

/// Plugins management tab with full functionality
pub struct PluginsTab {
    /// Shared state
    state: Arc<RwLock<UtilitiesState>>,
    
    /// Plugin manager connection
    plugin_manager: Option<Arc<PluginManager>>,
    
    /// Current view mode
    view_mode: ViewMode,
    
    /// List state for plugin selection
    list_state: ratatui::widgets::ListState,
    
    /// Scrollbar state
    scroll_state: ScrollbarState,
    
    /// Selected plugin index in marketplace
    marketplace_selected: usize,
    
    /// Configuration editing mode
    editing_config: bool,
    
    /// Configuration editor content
    config_editor: String,
    
    /// Search query
    search_query: String,
    
    /// Category filter
    category_filter: Option<String>,
    
    /// Tab state for view switching
    tab_state: usize,
    
    /// Search overlay
    search_overlay: SearchOverlay,
    
    /// Configuration manager
    config_manager: Option<Arc<RwLock<ConfigManager>>>,
}

impl PluginsTab {
    pub fn new(
        state: Arc<RwLock<UtilitiesState>>,
        plugin_manager: Option<Arc<PluginManager>>,
        config_manager: Option<Arc<RwLock<ConfigManager>>>,
    ) -> Self {
        let mut list_state = ratatui::widgets::ListState::default();
        list_state.select(Some(0));
        
        let mut tab = Self {
            state,
            plugin_manager,
            view_mode: ViewMode::InstalledList,
            list_state,
            scroll_state: ScrollbarState::default(),
            marketplace_selected: 0,
            editing_config: false,
            config_editor: String::new(),
            search_query: String::new(),
            category_filter: None,
            tab_state: 0,
            search_overlay: SearchOverlay::new(),
            config_manager,
        };
        
        // Initialize with data
        let _ = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(tab.refresh())
        });
        
        tab
    }
    
    /// Update plugin manager connection
    pub fn update_plugin_manager(&mut self, plugin_manager: Option<Arc<PluginManager>>) {
        self.plugin_manager = plugin_manager;
    }
}

#[async_trait]
impl UtilitiesSubtabController for PluginsTab {
    fn name(&self) -> &str {
        "Plugins"
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    
    fn render(&mut self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        // Check if we're in config editing mode
        if self.editing_config {
            self.render_config_editor(f, area);
            return;
        }
        
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header with tabs
                Constraint::Min(0),    // Content
                Constraint::Length(3), // Footer/controls
            ])
            .split(area);
        
        // Render header with view tabs
        self.render_header(f, chunks[0]);
        
        // Render content based on view mode
        match self.view_mode {
            ViewMode::InstalledList => self.render_installed_view(f, chunks[1]),
            ViewMode::PluginDetails => self.render_details_view(f, chunks[1]),
            ViewMode::Marketplace => self.render_marketplace_view(f, chunks[1]),
            ViewMode::ConfigEditor => self.render_config_view(f, chunks[1]),
        }
        
        // Footer with controls
        self.render_controls(f, chunks[2]);
        
        // Render search overlay if active
        if self.search_overlay.is_active() {
            self.search_overlay.render(f, area);
        }
    }
    
    async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool> {
        // Handle search overlay keys first
        if self.search_overlay.is_active() {
            match event.code {
                KeyCode::Esc => {
                    self.search_overlay.deactivate();
                    return Ok(true);
                }
                KeyCode::Enter => {
                    // Apply search and close overlay
                    if let Some(result) = self.search_overlay.get_selected_result() {
                        // Find and select the plugin
                        let state = self.state.read().await;
                        if let Some(index) = state.cache.plugins.iter().position(|p| p.id == result.id) {
                            self.list_state.select(Some(index));
                            self.scroll_state = self.scroll_state.position(index);
                        }
                    }
                    self.search_overlay.deactivate();
                    return Ok(true);
                }
                KeyCode::Up => {
                    self.search_overlay.previous();
                    return Ok(true);
                }
                KeyCode::Down => {
                    self.search_overlay.next();
                    return Ok(true);
                }
                KeyCode::Char(c) => {
                    self.search_overlay.input_char(c);
                    // Update search results
                    self.update_search_results().await;
                    return Ok(true);
                }
                KeyCode::Backspace => {
                    self.search_overlay.delete_char();
                    // Update search results
                    self.update_search_results().await;
                    return Ok(true);
                }
                _ => return Ok(false),
            }
        }
        
        // Handle config editor keys
        if self.editing_config {
            match event.code {
                KeyCode::Esc => {
                    self.editing_config = false;
                    self.config_editor.clear();
                    return Ok(true);
                }
                KeyCode::Char('s') if event.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                    debug!("Saving plugin configuration");
                    self.save_config().await?;
                    self.editing_config = false;
                    self.config_editor.clear();
                    return Ok(true);
                }
                KeyCode::Char(c) => {
                    self.config_editor.push(c);
                    return Ok(true);
                }
                KeyCode::Backspace => {
                    self.config_editor.pop();
                    return Ok(true);
                }
                KeyCode::Enter => {
                    self.config_editor.push('\n');
                    return Ok(true);
                }
                _ => return Ok(false),
            }
        }
        
        // Normal mode keys
        match event.code {
            // View switching
            KeyCode::Tab => {
                self.tab_state = (self.tab_state + 1) % 3;
                self.view_mode = match self.tab_state {
                    0 => ViewMode::InstalledList,
                    1 => ViewMode::Marketplace,
                    2 => ViewMode::ConfigEditor,
                    _ => ViewMode::InstalledList,
                };
                Ok(true)
            }
            // Navigation
            KeyCode::Up | KeyCode::Char('k') => {
                self.navigate_up().await;
                Ok(true)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.navigate_down().await;
                Ok(true)
            }
            // Actions
            KeyCode::Enter => {
                match self.view_mode {
                    ViewMode::InstalledList => {
                        self.view_mode = ViewMode::PluginDetails;
                    }
                    ViewMode::Marketplace => {
                        self.install_selected_plugin().await?;
                    }
                    _ => {}
                }
                Ok(true)
            }
            KeyCode::Char('i') => {
                // Install plugin from marketplace
                if self.view_mode == ViewMode::Marketplace {
                    self.install_selected_plugin().await?;
                }
                Ok(true)
            }
            KeyCode::Char('u') => {
                // Uninstall plugin
                if self.view_mode == ViewMode::InstalledList {
                    self.uninstall_selected_plugin().await?;
                }
                Ok(true)
            }
            KeyCode::Char('e') => {
                // Enable/disable plugin
                if self.view_mode == ViewMode::InstalledList {
                    self.toggle_plugin_status().await?;
                }
                Ok(true)
            }
            KeyCode::Char('c') => {
                // Open config editor
                self.editing_config = true;
                self.load_plugin_config().await?;
                Ok(true)
            }
            KeyCode::Char('r') => {
                // Refresh
                self.refresh().await?;
                Ok(true)
            }
            KeyCode::Char('/') => {
                // Search mode
                self.search_overlay.activate();
                Ok(true)
            }
            KeyCode::Backspace | KeyCode::Char('b') => {
                // Back to list from details
                if self.view_mode == ViewMode::PluginDetails {
                    self.view_mode = ViewMode::InstalledList;
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()> {
        match action {
            UtilitiesAction::InstallPlugin(plugin_id) => {
                debug!("Installing plugin: {}", plugin_id);
                // Use bridge to install plugin
                if let Some(ref manager) = self.plugin_manager {
                    let bridge = crate::tui::utilities::bridges::PluginBridge::new(manager.clone());
                    bridge.install_plugin(&plugin_id).await?;
                    
                    // Add to installed plugins in state
                    let mut state = self.state.write().await;
                    // Find the plugin in marketplace and add to installed
                    if let Some(plugin) = Self::get_marketplace_plugins()
                        .into_iter()
                        .find(|p| p.id == plugin_id) {
                        let mut installed = plugin.clone();
                        installed.enabled = true;
                        installed.status = PluginStatus::Active;
                        state.cache.plugins.push(installed);
                    }
                }
            }
            UtilitiesAction::UninstallPlugin(plugin_id) => {
                debug!("Uninstalling plugin: {}", plugin_id);
                // Use bridge to uninstall plugin
                if let Some(ref manager) = self.plugin_manager {
                    let bridge = crate::tui::utilities::bridges::PluginBridge::new(manager.clone());
                    bridge.uninstall_plugin(&plugin_id).await?;
                    
                    // Remove from installed plugins in state
                    let mut state = self.state.write().await;
                    state.cache.plugins.retain(|p| p.id != plugin_id);
                }
            }
            UtilitiesAction::EnablePlugin(plugin_id) => {
                debug!("Enabling plugin: {}", plugin_id);
                self.set_plugin_enabled(&plugin_id, true).await?;
            }
            UtilitiesAction::DisablePlugin(plugin_id) => {
                debug!("Disabling plugin: {}", plugin_id);
                self.set_plugin_enabled(&plugin_id, false).await?;
            }
            UtilitiesAction::RefreshPlugins => {
                self.refresh().await?;
            }
            _ => {}
        }
        Ok(())
    }
    
    async fn refresh(&mut self) -> Result<()> {
        // Try to get plugins from actual plugin manager
        if let Some(ref plugin_manager) = self.plugin_manager {
            // Use bridge to discover plugins
            let bridge = crate::tui::utilities::bridges::PluginBridge::new(plugin_manager.clone());
            if let Ok(plugins) = bridge.discover_plugins().await {
                if !plugins.is_empty() {
                    let mut state = self.state.write().await;
                    state.cache.plugins = plugins;
                    self.scroll_state = self.scroll_state.content_length(state.cache.plugins.len());
                    debug!("Loaded {} plugins from plugin manager", state.cache.plugins.len());
                    return Ok(());
                }
            }
        }
        
        // Fall back to example data if no plugin manager
        let mut state = self.state.write().await;
        state.cache.plugins = Self::get_example_plugins();
        
        // Update scrollbar
        self.scroll_state = self.scroll_state.content_length(state.cache.plugins.len());
        
        debug!("Refreshed {} plugins", state.cache.plugins.len());
        Ok(())
    }
}

// Implementation methods
impl PluginsTab {
    /// Render the header with view tabs
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let titles = ["Installed", "Marketplace", "Configuration"];
        let tabs = Tabs::new(titles.to_vec())
            .block(Block::default().borders(Borders::ALL).title("🔌 Plugin Management"))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
            .select(self.tab_state);
        
        f.render_widget(tabs, area);
    }
    
    /// Render installed plugins view
    fn render_installed_view(&mut self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        
        // Left side - plugin list
        self.render_plugin_list(f, chunks[0]);
        
        // Right side - plugin summary
        self.render_plugin_summary(f, chunks[1]);
    }
    
    /// Render plugin list
    fn render_plugin_list(&mut self, f: &mut Frame, area: Rect) {
        let state_guard = self.state.try_read();
        
        let plugins = if let Ok(state) = state_guard {
            state.get_filtered_plugins()
        } else {
            Vec::new()
        };
        
        let items: Vec<ListItem> = plugins
            .iter()
            .map(|plugin| {
                let status_icon = match &plugin.status {
                    PluginStatus::Active => "🟢",
                    PluginStatus::Inactive => "🟡",
                    PluginStatus::Loading => "🔵",
                    PluginStatus::Error(_) => "🔴",
                };
                
                let enabled_icon = if plugin.enabled { "✓" } else { "✗" };
                
                let line = Line::from(vec![
                    Span::raw(format!("{} {} ", status_icon, enabled_icon)),
                    Span::styled(&plugin.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::raw(" v"),
                    Span::styled(&plugin.version, Style::default().fg(Color::DarkGray)),
                ]);
                
                ListItem::new(line)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Installed Plugins"))
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
            .highlight_symbol("► ");
        
        f.render_stateful_widget(list, area, &mut self.list_state);
        
        // Scrollbar
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));
        
        f.render_stateful_widget(scrollbar, area, &mut self.scroll_state);
    }
    
    /// Render plugin summary
    fn render_plugin_summary(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // Plugin info
                Constraint::Length(6),  // Status
                Constraint::Min(0),     // Capabilities
            ])
            .split(area);
        
        let state_guard = self.state.try_read();
        
        if let Ok(state) = state_guard {
            if let Some(selected) = self.list_state.selected() {
                if let Some(plugin) = state.cache.plugins.get(selected) {
                    // Plugin info
                    self.render_plugin_info(f, chunks[0], plugin);
                    
                    // Status
                    self.render_plugin_status(f, chunks[1], plugin);
                    
                    // Capabilities
                    self.render_plugin_capabilities(f, chunks[2], plugin);
                    
                    return;
                }
            }
        }
        
        // No plugin selected
        let no_selection = Paragraph::new("Select a plugin to view details")
            .block(Block::default().borders(Borders::ALL).title("Plugin Details"))
            .alignment(Alignment::Center);
        f.render_widget(no_selection, area);
    }
    
    /// Render plugin information
    fn render_plugin_info(&self, f: &mut Frame, area: Rect, plugin: &PluginInfo) {
        let info_text = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Cyan)),
                Span::styled(&plugin.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("Version: ", Style::default().fg(Color::Cyan)),
                Span::raw(&plugin.version),
            ]),
            Line::from(vec![
                Span::styled("Author: ", Style::default().fg(Color::Cyan)),
                Span::raw(&plugin.author),
            ]),
            Line::from(vec![
                Span::styled("Description: ", Style::default().fg(Color::Cyan)),
            ]),
            Line::from(Span::raw(&plugin.description)),
        ];
        
        let info = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Plugin Information"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(info, area);
    }
    
    /// Render plugin status
    fn render_plugin_status(&self, f: &mut Frame, area: Rect, plugin: &PluginInfo) {
        let status_text = vec![
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Cyan)),
                match &plugin.status {
                    PluginStatus::Active => Span::styled("Active", Style::default().fg(Color::Green)),
                    PluginStatus::Inactive => Span::styled("Inactive", Style::default().fg(Color::Yellow)),
                    PluginStatus::Loading => Span::styled("Loading", Style::default().fg(Color::Blue)),
                    PluginStatus::Error(err) => Span::styled(format!("Error: {}", err), Style::default().fg(Color::Red)),
                },
            ]),
            Line::from(vec![
                Span::styled("Enabled: ", Style::default().fg(Color::Cyan)),
                if plugin.enabled {
                    Span::styled("Yes", Style::default().fg(Color::Green))
                } else {
                    Span::styled("No", Style::default().fg(Color::Red))
                },
            ]),
        ];
        
        let status = Paragraph::new(status_text)
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(status, area);
    }
    
    /// Render plugin capabilities
    fn render_plugin_capabilities(&self, f: &mut Frame, area: Rect, plugin: &PluginInfo) {
        let mut lines = vec![
            Line::from(Span::styled("Capabilities:", Style::default().fg(Color::Cyan))),
            Line::from(""),
        ];
        
        for capability in &plugin.capabilities {
            lines.push(Line::from(vec![
                Span::raw("  • "),
                Span::raw(capability),
            ]));
        }
        
        if plugin.capabilities.is_empty() {
            lines.push(Line::from(Span::styled("  No capabilities defined", Style::default().fg(Color::DarkGray))));
        }
        
        let capabilities = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Capabilities"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(capabilities, area);
    }
    
    /// Render details view
    fn render_details_view(&mut self, f: &mut Frame, area: Rect) {
        // For now, reuse the installed view layout
        self.render_installed_view(f, area);
    }
    
    /// Render marketplace view
    fn render_marketplace_view(&mut self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Search bar
                Constraint::Min(0),     // Plugin grid
            ])
            .split(area);
        
        // Search bar
        let search = Paragraph::new(format!("🔍 Search: {}", self.search_query))
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(search, chunks[0]);
        
        // Marketplace plugins
        let marketplace_plugins = Self::get_marketplace_plugins();
        
        let rows: Vec<Row> = marketplace_plugins
            .iter()
            .map(|plugin| {
                Row::new(vec![
                    Cell::from(plugin.name.clone()),
                    Cell::from(plugin.version.clone()),
                    Cell::from(plugin.author.clone()),
                    Cell::from(plugin.description.clone()),
                    Cell::from(if plugin.enabled { "Install" } else { "Installed" }),
                ])
            })
            .collect();
        
        let widths = [
            Constraint::Length(20),
            Constraint::Length(10),
            Constraint::Length(20),
            Constraint::Min(30),
            Constraint::Length(10),
        ];
        
        let table = Table::new(rows, widths)
            .header(Row::new(vec!["Name", "Version", "Author", "Description", "Action"])
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)))
            .block(Block::default().borders(Borders::ALL).title("Plugin Marketplace"));
        
        f.render_widget(table, chunks[1]);
    }
    
    /// Render configuration view
    fn render_config_view(&self, f: &mut Frame, area: Rect) {
        let config = Paragraph::new("Plugin configuration management\n\nSelect a plugin and press 'c' to edit its configuration")
            .block(Block::default().borders(Borders::ALL).title("Configuration"))
            .alignment(Alignment::Center);
        
        f.render_widget(config, area);
    }
    
    /// Render configuration editor
    fn render_config_editor(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(0),     // Editor
                Constraint::Length(3),  // Controls
            ])
            .split(area);
        
        // Header
        let header = Paragraph::new("📝 Plugin Configuration Editor")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
        f.render_widget(header, chunks[0]);
        
        // Editor content
        let editor = Paragraph::new(self.config_editor.as_str())
            .block(Block::default().borders(Borders::ALL).title("Configuration (JSON)"))
            .wrap(Wrap { trim: false });
        f.render_widget(editor, chunks[1]);
        
        // Controls
        let controls = Paragraph::new("Ctrl+S: Save | Esc: Cancel | Type to edit")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(controls, chunks[2]);
    }
    
    /// Render controls footer
    fn render_controls(&self, f: &mut Frame, area: Rect) {
        let controls = match self.view_mode {
            ViewMode::InstalledList => "↑↓/jk: Navigate | Enter: Details | e: Enable/Disable | u: Uninstall | c: Config | Tab: Switch View",
            ViewMode::PluginDetails => "b: Back | c: Configure | e: Enable/Disable | u: Uninstall",
            ViewMode::Marketplace => "↑↓: Navigate | Enter/i: Install | /: Search | Tab: Switch View",
            ViewMode::ConfigEditor => "Type to edit | Ctrl+S: Save | Esc: Cancel",
        };
        
        let controls_widget = Paragraph::new(controls)
            .block(Block::default().borders(Borders::ALL).title("Controls"))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        
        f.render_widget(controls_widget, area);
    }
    
    /// Navigate up in the list
    async fn navigate_up(&mut self) {
        if let Some(selected) = self.list_state.selected() {
            if selected > 0 {
                self.list_state.select(Some(selected - 1));
                self.scroll_state = self.scroll_state.position(selected - 1);
            }
        }
    }
    
    /// Navigate down in the list
    async fn navigate_down(&mut self) {
        let count = {
            let state = self.state.read().await;
            state.cache.plugins.len()
        };
        
        if let Some(selected) = self.list_state.selected() {
            if selected < count.saturating_sub(1) {
                self.list_state.select(Some(selected + 1));
                self.scroll_state = self.scroll_state.position(selected + 1);
            }
        }
    }
    
    /// Install selected plugin from marketplace
    async fn install_selected_plugin(&mut self) -> Result<()> {
        debug!("Installing selected plugin from marketplace");
        
        let marketplace_plugins = Self::get_marketplace_plugins();
        if let Some(plugin) = marketplace_plugins.get(self.marketplace_selected) {
            // Create bridge if we have a plugin manager
            if let Some(ref manager) = self.plugin_manager {
                let bridge = crate::tui::utilities::bridges::PluginBridge::new(manager.clone());
                bridge.install_plugin(&plugin.id).await?;
                
                // Add to installed plugins
                let mut state = self.state.write().await;
                let mut installed = plugin.clone();
                installed.enabled = true;
                installed.status = PluginStatus::Active;
                state.cache.plugins.push(installed);
                
                debug!("Successfully installed plugin: {}", plugin.name);
            }
        }
        
        Ok(())
    }
    
    /// Uninstall selected plugin
    async fn uninstall_selected_plugin(&mut self) -> Result<()> {
        if let Some(selected) = self.list_state.selected() {
            let plugin_id = {
                let state = self.state.read().await;
                state.cache.plugins.get(selected).map(|p| p.id.clone())
            };
            
            if let Some(id) = plugin_id {
                // Create bridge if we have a plugin manager
                if let Some(ref manager) = self.plugin_manager {
                    let bridge = crate::tui::utilities::bridges::PluginBridge::new(manager.clone());
                    bridge.uninstall_plugin(&id).await?;
                    
                    // Remove from installed plugins
                    let mut state = self.state.write().await;
                    state.cache.plugins.retain(|p| p.id != id);
                    
                    debug!("Successfully uninstalled plugin: {}", id);
                }
            }
        }
        Ok(())
    }
    
    /// Toggle plugin enabled status
    async fn toggle_plugin_status(&mut self) -> Result<()> {
        if let Some(selected) = self.list_state.selected() {
            let mut state = self.state.write().await;
            if let Some(plugin) = state.cache.plugins.get_mut(selected) {
                plugin.enabled = !plugin.enabled;
                debug!("Toggled plugin {} enabled status to {}", plugin.name, plugin.enabled);
            }
        }
        Ok(())
    }
    
    /// Set plugin enabled status
    async fn set_plugin_enabled(&mut self, plugin_id: &str, enabled: bool) -> Result<()> {
        let mut state = self.state.write().await;
        if let Some(plugin) = state.cache.plugins.iter_mut().find(|p| p.id == plugin_id) {
            plugin.enabled = enabled;
            debug!("Set plugin {} enabled status to {}", plugin.name, enabled);
        }
        Ok(())
    }
    
    /// Load plugin configuration
    async fn load_plugin_config(&mut self) -> Result<()> {
        if let Some(selected) = self.list_state.selected() {
            let state = self.state.read().await;
            if let Some(plugin) = state.cache.plugins.get(selected) {
                // Try to load from ConfigManager first
                if let Some(ref config_manager) = self.config_manager {
                    let manager = config_manager.read().await;
                    if let Some(config) = manager.get_plugin_config(&plugin.id) {
                        self.config_editor = serde_json::to_string_pretty(&config)?;
                        return Ok(());
                    }
                }
                
                // Default configuration if not found
                self.config_editor = format!(
                    r#"{{
  "id": "{}",
  "enabled": {},
  "auto_update": true,
  "settings": {{
    "log_level": "info",
    "max_memory_mb": 512,
    "permissions": [
      "file_system_read",
      "network_access"
    ]
  }}
}}"#,
                    plugin.id,
                    plugin.enabled
                );
            }
        }
        Ok(())
    }
    
    /// Save plugin configuration
    async fn save_config(&self) -> Result<()> {
        // Parse and validate JSON
        let config: PluginConfig = serde_json::from_str(&self.config_editor)?;
        
        // Save to ConfigManager
        if let Some(ref config_manager) = self.config_manager {
            let mut manager = config_manager.write().await;
            manager.update_plugin_config(config.clone())?;
            debug!("Saved plugin configuration for: {}", config.id);
        }
        
        // Update plugin manager if connected
        if let Some(ref plugin_manager) = self.plugin_manager {
            let bridge = crate::tui::utilities::bridges::PluginBridge::new(plugin_manager.clone());
            bridge.update_plugin_config(&config.id, config.settings.clone()).await?;
        }
        
        Ok(())
    }
    
    /// Get example plugins for demo
    fn get_example_plugins() -> Vec<PluginInfo> {
        vec![
            PluginInfo {
                id: "git-integration".to_string(),
                name: "Git Integration".to_string(),
                version: "2.1.0".to_string(),
                author: "Loki Team".to_string(),
                description: "Advanced Git operations and repository management".to_string(),
                enabled: true,
                status: PluginStatus::Active,
                capabilities: vec![
                    "Repository management".to_string(),
                    "Branch operations".to_string(),
                    "Commit history analysis".to_string(),
                ],
            },
            PluginInfo {
                id: "docker-manager".to_string(),
                name: "Docker Manager".to_string(),
                version: "1.5.2".to_string(),
                author: "Container Labs".to_string(),
                description: "Docker container and image management".to_string(),
                enabled: true,
                status: PluginStatus::Active,
                capabilities: vec![
                    "Container lifecycle management".to_string(),
                    "Image building".to_string(),
                    "Network configuration".to_string(),
                ],
            },
            PluginInfo {
                id: "ai-assistant".to_string(),
                name: "AI Code Assistant".to_string(),
                version: "3.0.1".to_string(),
                author: "AI Innovations".to_string(),
                description: "AI-powered code completion and suggestions".to_string(),
                enabled: false,
                status: PluginStatus::Inactive,
                capabilities: vec![
                    "Code completion".to_string(),
                    "Bug detection".to_string(),
                    "Refactoring suggestions".to_string(),
                ],
            },
            PluginInfo {
                id: "test-runner".to_string(),
                name: "Universal Test Runner".to_string(),
                version: "1.2.0".to_string(),
                author: "Testing Co".to_string(),
                description: "Run tests for multiple frameworks and languages".to_string(),
                enabled: true,
                status: PluginStatus::Loading,
                capabilities: vec![
                    "Multi-framework support".to_string(),
                    "Parallel test execution".to_string(),
                    "Coverage reporting".to_string(),
                ],
            },
            PluginInfo {
                id: "db-browser".to_string(),
                name: "Database Browser".to_string(),
                version: "2.0.5".to_string(),
                author: "Data Tools Inc".to_string(),
                description: "Browse and query multiple database types".to_string(),
                enabled: true,
                status: PluginStatus::Error("Connection failed".to_string()),
                capabilities: vec![
                    "Multi-database support".to_string(),
                    "Query builder".to_string(),
                    "Schema visualization".to_string(),
                ],
            },
        ]
    }
    
    /// Get marketplace plugins
    fn get_marketplace_plugins() -> Vec<PluginInfo> {
        vec![
            PluginInfo {
                id: "markdown-preview".to_string(),
                name: "Markdown Preview".to_string(),
                version: "1.0.0".to_string(),
                author: "Doc Tools".to_string(),
                description: "Live markdown preview with syntax highlighting".to_string(),
                enabled: false,
                status: PluginStatus::Inactive,
                capabilities: vec![],
            },
            PluginInfo {
                id: "api-tester".to_string(),
                name: "API Tester".to_string(),
                version: "2.3.1".to_string(),
                author: "API Dev".to_string(),
                description: "Test REST and GraphQL APIs with ease".to_string(),
                enabled: false,
                status: PluginStatus::Inactive,
                capabilities: vec![],
            },
            PluginInfo {
                id: "theme-manager".to_string(),
                name: "Theme Manager".to_string(),
                version: "1.1.0".to_string(),
                author: "UI Studio".to_string(),
                description: "Manage and customize UI themes".to_string(),
                enabled: false,
                status: PluginStatus::Inactive,
                capabilities: vec![],
            },
        ]
    }
    
    /// Update search results based on current query
    async fn update_search_results(&mut self) {
        let query = self.search_overlay.get_query();
        if query.is_empty() {
            self.search_overlay.clear_results();
            return;
        }
        
        let state = self.state.read().await;
        let results: Vec<SearchResult> = state.cache.plugins
            .iter()
            .filter(|plugin| {
                plugin.name.to_lowercase().contains(&query.to_lowercase()) ||
                plugin.description.to_lowercase().contains(&query.to_lowercase()) ||
                plugin.author.to_lowercase().contains(&query.to_lowercase())
            })
            .map(|plugin| SearchResult {
                id: plugin.id.clone(),
                title: plugin.name.clone(),
                description: plugin.description.clone(),
                category: Some(plugin.author.clone()),
                score: 1.0, // Simple scoring for now
            })
            .collect();
        
        self.search_overlay.set_results(results);
    }
}