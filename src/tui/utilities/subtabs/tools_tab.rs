//! Tools management tab - Complete implementation

use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, List, ListItem, Paragraph, Gauge, Table, Row, Cell,
    Scrollbar, ScrollbarOrientation, ScrollbarState, Padding, Wrap
};
use crossterm::event::{KeyEvent, KeyCode};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::tools::IntelligentToolManager;
use crate::tui::utilities::state::UtilitiesState;
use crate::tui::utilities::types::{UtilitiesAction, ToolEntry, ToolStatus};
use crate::tui::utilities::components::{SearchOverlay, SearchResult};
use crate::tui::utilities::config::{ConfigManager, ToolConfig};
use super::UtilitiesSubtabController;

/// Tools management tab with full functionality
pub struct ToolsTab {
    /// Shared state
    state: Arc<RwLock<UtilitiesState>>,
    
    /// Tool manager connection
    tool_manager: Option<Arc<IntelligentToolManager>>,
    
    /// List state for tool selection
    list_state: ratatui::widgets::ListState,
    
    /// Scrollbar state
    scroll_state: ScrollbarState,
    
    /// Configuration editing mode
    editing_config: bool,
    
    /// Configuration editor content
    config_editor: String,
    
    /// Tool creation mode
    creating_tool: bool,
    
    /// New tool template
    new_tool_config: String,
    
    /// Search query
    search_query: String,
    
    /// Filter by category
    category_filter: Option<String>,
    
    /// Search overlay
    search_overlay: SearchOverlay,
    
    /// Configuration manager
    config_manager: Option<Arc<RwLock<ConfigManager>>>,
}

impl ToolsTab {
    pub fn new(
        state: Arc<RwLock<UtilitiesState>>,
        tool_manager: Option<Arc<IntelligentToolManager>>,
        config_manager: Option<Arc<RwLock<ConfigManager>>>,
    ) -> Self {
        let mut list_state = ratatui::widgets::ListState::default();
        list_state.select(Some(0));
        
        let mut tab = Self {
            state,
            tool_manager,
            list_state,
            scroll_state: ScrollbarState::default(),
            editing_config: false,
            config_editor: String::new(),
            creating_tool: false,
            new_tool_config: String::new(),
            search_query: String::new(),
            category_filter: None,
            search_overlay: SearchOverlay::new(),
            config_manager,
        };
        
        // Initialize with data
        let _ = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(tab.refresh())
        });
        
        tab
    }
    
    /// Update tool manager connection
    pub fn update_tool_manager(&mut self, tool_manager: Option<Arc<IntelligentToolManager>>) {
        self.tool_manager = tool_manager;
    }
}

#[async_trait]
impl UtilitiesSubtabController for ToolsTab {
    fn name(&self) -> &str {
        "Tools"
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
        
        // Check if we're in tool creation mode
        if self.creating_tool {
            self.render_tool_creator(f, area);
            return;
        }
        
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Content
                Constraint::Length(3), // Footer/controls
            ])
            .split(area);
        
        // Render header with status
        self.render_header(f, chunks[0]);
        
        // Content area - split horizontally
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(chunks[1]);
        
        // Left side - tools list
        self.render_tools_list(f, content_chunks[0]);
        
        // Right side - tool details/configuration
        self.render_tool_details(f, content_chunks[1]);
        
        // Footer with controls
        self.render_controls(f, chunks[2]);
        
        // Render search overlay if active
        if self.search_overlay.is_active() {
            self.search_overlay.render(f, area);
        }
    }
    
    async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool> {
        // Handle tool creation keys
        if self.creating_tool {
            match event.code {
                KeyCode::Esc => {
                    // Exit tool creation without saving
                    self.creating_tool = false;
                    self.new_tool_config.clear();
                    return Ok(true);
                }
                KeyCode::Char('s') if event.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                    // Save new tool
                    debug!("Creating new tool");
                    self.save_new_tool().await?;
                    self.creating_tool = false;
                    self.new_tool_config.clear();
                    return Ok(true);
                }
                KeyCode::Char(c) => {
                    self.new_tool_config.push(c);
                    return Ok(true);
                }
                KeyCode::Backspace => {
                    self.new_tool_config.pop();
                    return Ok(true);
                }
                KeyCode::Enter => {
                    self.new_tool_config.push('\n');
                    return Ok(true);
                }
                _ => return Ok(false),
            }
        }
        
        // Handle search overlay keys
        if self.search_overlay.is_active() {
            match event.code {
                KeyCode::Esc => {
                    self.search_overlay.deactivate();
                    return Ok(true);
                }
                KeyCode::Enter => {
                    // Apply search and close overlay
                    if let Some(result) = self.search_overlay.get_selected_result() {
                        // Find and select the tool
                        let state = self.state.read().await;
                        if let Some(index) = state.cache.tools.iter().position(|t| t.id == result.id) {
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
                    // Exit config editor without saving
                    self.editing_config = false;
                    self.config_editor.clear();
                    return Ok(true);
                }
                KeyCode::Char('s') if event.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                    // Save configuration
                    debug!("Saving tool configuration");
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
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(selected) = self.list_state.selected() {
                    if selected > 0 {
                        self.list_state.select(Some(selected - 1));
                        self.scroll_state = self.scroll_state.position(selected - 1);
                    }
                }
                Ok(true)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let tools_count = {
                    let state = self.state.read().await;
                    state.cache.tools.len()
                };
                if let Some(selected) = self.list_state.selected() {
                    if selected < tools_count.saturating_sub(1) {
                        self.list_state.select(Some(selected + 1));
                        self.scroll_state = self.scroll_state.position(selected + 1);
                    }
                }
                Ok(true)
            }
            KeyCode::Enter => {
                // Open configuration editor
                self.editing_config = true;
                if let Some(selected) = self.list_state.selected() {
                    let tool_id = {
                        let state = self.state.read().await;
                        if let Some(tool) = state.cache.tools.get(selected) {
                            debug!("Opening configuration for tool: {}", tool.name);
                            Some(tool.id.clone())
                        } else {
                            None
                        }
                    };
                    if let Some(id) = tool_id {
                        self.load_config(&id).await?;
                    }
                }
                Ok(true)
            }
            KeyCode::Char('r') => {
                // Refresh tools
                self.refresh().await?;
                Ok(true)
            }
            KeyCode::Char('/') => {
                // Start search
                self.search_overlay.activate();
                Ok(true)
            }
            KeyCode::Char('c') => {
                // Create new tool
                self.start_tool_creation();
                Ok(true)
            }
            KeyCode::Char('d') => {
                // Disable/enable tool
                if let Some(selected) = self.list_state.selected() {
                    self.toggle_tool_status(selected).await?;
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()> {
        match action {
            UtilitiesAction::RefreshTools => {
                self.refresh().await?;
            }
            UtilitiesAction::OpenToolConfig(tool_id) => {
                // Find and select the tool
                let state = self.state.read().await;
                if let Some(index) = state.cache.tools.iter().position(|t| t.id == tool_id) {
                    self.list_state.select(Some(index));
                    drop(state);
                    self.editing_config = true;
                    self.load_config(&tool_id).await?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    async fn refresh(&mut self) -> Result<()> {
        if self.tool_manager.is_some() {
            // Get tools from the registry
            let tool_registry = crate::tools::get_tool_registry();
            debug!("Refreshed {} tools from registry", tool_registry.len());
            
            // Convert to ToolEntry format
            let tool_entries: Vec<ToolEntry> = tool_registry.iter().map(|info| {
                ToolEntry {
                    id: info.id.clone(),
                    name: info.name.clone(),
                    category: info.category.clone(),
                    description: info.description.clone(),
                    status: if info.available { 
                        ToolStatus::Active 
                    } else { 
                        ToolStatus::Idle 
                    },
                    last_used: None,
                    usage_count: 0,
                }
            }).collect();
            
            // Update state cache
            let mut state = self.state.write().await;
            state.cache.tools = tool_entries;
            
            // Update scrollbar
            self.scroll_state = self.scroll_state.content_length(state.cache.tools.len());
        } else {
            // Use example data if no tool manager connected
            let mut state = self.state.write().await;
            state.cache.tools = Self::get_example_tools();
            self.scroll_state = self.scroll_state.content_length(state.cache.tools.len());
        }
        Ok(())
    }
}

// Implementation methods
impl ToolsTab {
    /// Render the header with connection status
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let status = if self.tool_manager.is_some() {
            "🟢 Connected to Tool Manager"
        } else {
            "⚪ Tool Manager Not Connected"
        };
        
        let state_guard = self.state.try_read();
        let tool_count = if let Ok(state) = state_guard {
            state.cache.tools.len()
        } else {
            0
        };
        
        let header = Paragraph::new(format!("🔧 Tool Management - {} tools available | {}", tool_count, status))
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        
        f.render_widget(header, area);
    }
    
    /// Render the tools list
    fn render_tools_list(&mut self, f: &mut Frame, area: Rect) {
        let state_guard = self.state.try_read();
        
        let tools = if let Ok(state) = state_guard {
            state.get_filtered_tools()
        } else {
            Vec::new()
        };
        
        // Create list items with status indicators
        let items: Vec<ListItem> = tools
            .iter()
            .map(|tool| {
                let status_icon = match tool.status {
                    ToolStatus::Active => "🟢",
                    ToolStatus::Idle => "🟡",
                    ToolStatus::Error => "🔴",
                    ToolStatus::Disabled => "⚫",
                    ToolStatus::Processing => "🔵",
                };
                
                let line = Line::from(vec![
                    Span::raw(format!("{} ", status_icon)),
                    Span::styled(&tool.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::raw(" - "),
                    Span::styled(&tool.category, Style::default().fg(Color::DarkGray)),
                ]);
                
                ListItem::new(line)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Available Tools"))
            .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
            .highlight_symbol("► ");
        
        f.render_stateful_widget(list, area, &mut self.list_state);
        
        // Render scrollbar
        let scrollbar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("↑"))
            .end_symbol(Some("↓"));
        
        f.render_stateful_widget(scrollbar, area, &mut self.scroll_state);
    }
    
    /// Render tool details panel
    fn render_tool_details(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10), // Tool info
                Constraint::Length(8),  // Status & metrics
                Constraint::Min(0),     // Configuration preview
            ])
            .split(area);
        
        let state_guard = self.state.try_read();
        
        if let Ok(state) = state_guard {
            if let Some(selected) = self.list_state.selected() {
                if let Some(tool) = state.cache.tools.get(selected) {
                    // Tool information
                    self.render_tool_info(f, chunks[0], tool);
                    
                    // Status and metrics
                    self.render_tool_metrics(f, chunks[1], tool);
                    
                    // Configuration preview
                    self.render_config_preview(f, chunks[2], tool);
                    
                    return;
                }
            }
        }
        
        // No tool selected
        let no_selection = Paragraph::new("Select a tool to view details")
            .block(Block::default().borders(Borders::ALL).title("Tool Details"))
            .alignment(Alignment::Center);
        f.render_widget(no_selection, area);
    }
    
    /// Render tool information
    fn render_tool_info(&self, f: &mut Frame, area: Rect, tool: &ToolEntry) {
        let info_text = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Cyan)),
                Span::styled(&tool.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("ID: ", Style::default().fg(Color::Cyan)),
                Span::raw(&tool.id),
            ]),
            Line::from(vec![
                Span::styled("Category: ", Style::default().fg(Color::Cyan)),
                Span::raw(&tool.category),
            ]),
            Line::from(vec![
                Span::styled("Description: ", Style::default().fg(Color::Cyan)),
            ]),
            Line::from(Span::raw(&tool.description)),
            Line::from(""),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Cyan)),
                match tool.status {
                    ToolStatus::Active => Span::styled("Active", Style::default().fg(Color::Green)),
                    ToolStatus::Idle => Span::styled("Idle", Style::default().fg(Color::Yellow)),
                    ToolStatus::Error => Span::styled("Error", Style::default().fg(Color::Red)),
                    ToolStatus::Disabled => Span::styled("Disabled", Style::default().fg(Color::DarkGray)),
                    ToolStatus::Processing => Span::styled("Processing", Style::default().fg(Color::Blue)),
                },
            ]),
        ];
        
        let info = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Tool Information"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(info, area);
    }
    
    /// Render tool metrics
    fn render_tool_metrics(&self, f: &mut Frame, area: Rect, tool: &ToolEntry) {
        let metrics_text = vec![
            Line::from(vec![
                Span::styled("Usage Count: ", Style::default().fg(Color::Cyan)),
                Span::raw(format!("{} calls", tool.usage_count)),
            ]),
            Line::from(vec![
                Span::styled("Last Used: ", Style::default().fg(Color::Cyan)),
                Span::raw(tool.last_used.clone().unwrap_or_else(|| "Never".to_string())),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Performance: ", Style::default().fg(Color::Cyan)),
            ]),
        ];
        
        let metrics = Paragraph::new(metrics_text)
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(metrics, area);
    }
    
    /// Render configuration preview
    fn render_config_preview(&self, f: &mut Frame, area: Rect, tool: &ToolEntry) {
        let config_text = vec![
            Line::from(Span::styled("Configuration Preview:", Style::default().fg(Color::Cyan))),
            Line::from(""),
            Line::from(Span::raw("{")),
            Line::from(Span::raw(format!("  \"id\": \"{}\",", tool.id))),
            Line::from(Span::raw("  \"enabled\": true,")),
            Line::from(Span::raw("  \"api_key\": \"***\",")),
            Line::from(Span::raw("  \"timeout\": 30,")),
            Line::from(Span::raw("  \"retry_count\": 3")),
            Line::from(Span::raw("}")),
            Line::from(""),
            Line::from(Span::styled("Press Enter to edit configuration", Style::default().fg(Color::DarkGray))),
        ];
        
        let config = Paragraph::new(config_text)
            .block(Block::default().borders(Borders::ALL).title("Configuration"))
            .wrap(Wrap { trim: true });
        
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
        let header = Paragraph::new("📝 Configuration Editor")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
        f.render_widget(header, chunks[0]);
        
        // Editor content
        let editor = Paragraph::new(self.config_editor.as_str())
            .block(Block::default().borders(Borders::ALL).title("JSON Configuration"))
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
        let controls = if self.editing_config {
            "Ctrl+S: Save | Esc: Cancel"
        } else {
            "↑↓/jk: Navigate | Enter: Configure | r: Refresh | c: Create | d: Toggle | /: Search | q: Back"
        };
        
        let controls_widget = Paragraph::new(controls)
            .block(Block::default().borders(Borders::ALL).title("Controls"))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        
        f.render_widget(controls_widget, area);
    }
    
    /// Load configuration for a tool
    async fn load_config(&mut self, tool_id: &str) -> Result<()> {
        // Try to load from ConfigManager first
        if let Some(ref config_manager) = self.config_manager {
            let manager = config_manager.read().await;
            if let Some(config) = manager.get_tool_config(tool_id) {
                self.config_editor = serde_json::to_string_pretty(&config)?;
                return Ok(());
            }
        }
        
        // Default configuration if not found
        self.config_editor = format!(
            r#"{{
  "id": "{}",
  "enabled": true,
  "settings": {{
    "api_key": "",
    "endpoint": "https://api.example.com",
    "timeout": 30,
    "retry_count": 3,
    "rate_limit": {{
      "requests_per_minute": 60,
      "burst_size": 10
    }}
  }}
}}"#,
            tool_id
        );
        Ok(())
    }
    
    /// Save configuration
    async fn save_config(&self) -> Result<()> {
        // Parse and validate JSON
        let config: ToolConfig = serde_json::from_str(&self.config_editor)?;
        
        // Save to ConfigManager
        if let Some(ref config_manager) = self.config_manager {
            let mut manager = config_manager.write().await;
            manager.update_tool_config(config.clone())?;
            debug!("Saved tool configuration for: {}", config.id);
        }
        
        // Update tool manager if connected
        if let Some(ref tool_manager) = self.tool_manager {
            // Update the tool configuration in the manager
            let bridge = crate::tui::utilities::bridges::ToolBridge::new(tool_manager.clone());
            bridge.update_tool_config(&config.id, config.settings.clone()).await?;
        }
        
        Ok(())
    }
    
    /// Toggle tool enabled/disabled status
    async fn toggle_tool_status(&mut self, index: usize) -> Result<()> {
        let mut state = self.state.write().await;
        if let Some(tool) = state.cache.tools.get_mut(index) {
            tool.status = match tool.status {
                ToolStatus::Active => ToolStatus::Disabled,
                ToolStatus::Disabled => ToolStatus::Active,
                _ => tool.status.clone(),
            };
            debug!("Toggled tool {} status to {:?}", tool.name, tool.status);
        }
        Ok(())
    }
    
    /// Get fallback tools when no manager is connected
    fn get_example_tools() -> Vec<ToolEntry> {
        vec![
            ToolEntry {
                id: "computer-use".to_string(),
                name: "Computer Use".to_string(),
                category: "Automation".to_string(),
                description: "Advanced screen control and UI automation capabilities".to_string(),
                status: ToolStatus::Active,
                last_used: Some("2 mins ago".to_string()),
                usage_count: 142,
            },
            ToolEntry {
                id: "creative-media".to_string(),
                name: "Creative Media Manager".to_string(),
                category: "AI Generation".to_string(),
                description: "AI-powered image, video, and audio generation tools".to_string(),
                status: ToolStatus::Active,
                last_used: Some("5 mins ago".to_string()),
                usage_count: 89,
            },
            ToolEntry {
                id: "blender-3d".to_string(),
                name: "Blender 3D Integration".to_string(),
                category: "3D Modeling".to_string(),
                description: "3D modeling, animation, and procedural generation".to_string(),
                status: ToolStatus::Idle,
                last_used: Some("1 hour ago".to_string()),
                usage_count: 23,
            },
            ToolEntry {
                id: "vision-ai".to_string(),
                name: "Vision AI".to_string(),
                category: "Analysis".to_string(),
                description: "Advanced computer vision and image analysis".to_string(),
                status: ToolStatus::Active,
                last_used: Some("10 mins ago".to_string()),
                usage_count: 67,
            },
            ToolEntry {
                id: "code-gen".to_string(),
                name: "Code Generator".to_string(),
                category: "Development".to_string(),
                description: "AI-powered code generation and refactoring".to_string(),
                status: ToolStatus::Processing,
                last_used: Some("Just now".to_string()),
                usage_count: 203,
            },
            ToolEntry {
                id: "web-scraper".to_string(),
                name: "Web Scraper".to_string(),
                category: "Data".to_string(),
                description: "Intelligent web scraping and data extraction".to_string(),
                status: ToolStatus::Idle,
                last_used: None,
                usage_count: 0,
            },
            ToolEntry {
                id: "file-manager".to_string(),
                name: "File Manager".to_string(),
                category: "System".to_string(),
                description: "Advanced file system operations and management".to_string(),
                status: ToolStatus::Error,
                last_used: Some("2 hours ago".to_string()),
                usage_count: 45,
            },
            ToolEntry {
                id: "db-client".to_string(),
                name: "Database Client".to_string(),
                category: "Data".to_string(),
                description: "Multi-database client with query builder".to_string(),
                status: ToolStatus::Disabled,
                last_used: None,
                usage_count: 0,
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
        let results: Vec<SearchResult> = state.cache.tools
            .iter()
            .filter(|tool| {
                tool.name.to_lowercase().contains(&query.to_lowercase()) ||
                tool.description.to_lowercase().contains(&query.to_lowercase()) ||
                tool.category.to_lowercase().contains(&query.to_lowercase())
            })
            .map(|tool| SearchResult {
                id: tool.id.clone(),
                title: tool.name.clone(),
                description: tool.description.clone(),
                category: Some(tool.category.clone()),
                score: 1.0, // Simple scoring for now
            })
            .collect();
        
        self.search_overlay.set_results(results);
    }
    
    /// Start tool creation mode
    fn start_tool_creation(&mut self) {
        self.creating_tool = true;
        self.new_tool_config = r#"{
  "id": "new-tool",
  "name": "New Tool",
  "category": "Custom",
  "description": "A new custom tool",
  "enabled": true,
  "settings": {
    "api_key": "",
    "endpoint": "https://api.example.com",
    "timeout": 30,
    "retry_count": 3
  }
}"#.to_string();
    }
    
    /// Save a newly created tool
    async fn save_new_tool(&mut self) -> Result<()> {
        // Parse and validate the new tool configuration
        let new_tool: serde_json::Value = serde_json::from_str(&self.new_tool_config)?;
        
        // Extract tool information
        let tool_id = new_tool.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Tool ID is required"))?;
        let tool_name = new_tool.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(tool_id);
        let tool_category = new_tool.get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("Custom");
        let tool_description = new_tool.get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("Custom tool");
        
        // Create ToolConfig for ConfigManager
        if let Some(ref config_manager) = self.config_manager {
            let tool_config = ToolConfig {
                id: tool_id.to_string(),
                enabled: new_tool.get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                settings: new_tool.get("settings")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
            };
            
            let mut manager = config_manager.write().await;
            manager.update_tool_config(tool_config)?;
            debug!("Created new tool: {}", tool_id);
        }
        
        // Add to state cache
        let new_entry = ToolEntry {
            id: tool_id.to_string(),
            name: tool_name.to_string(),
            category: tool_category.to_string(),
            description: tool_description.to_string(),
            status: ToolStatus::Active,
            last_used: None,
            usage_count: 0,
        };
        
        let mut state = self.state.write().await;
        state.cache.tools.push(new_entry);
        
        // If we have a tool manager, register the tool
        if let Some(ref tool_manager) = self.tool_manager {
            // Register tool with tool manager via bridge
            let bridge = crate::tui::utilities::bridges::ToolBridge::new(tool_manager.clone());
            bridge.update_tool_config(tool_id, new_tool.get("settings")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}))).await?;
        }
        
        info!("Successfully created tool: {}", tool_name);
        Ok(())
    }
    
    /// Render tool creation dialog
    fn render_tool_creator(&self, f: &mut Frame, area: Rect) {
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
        let header = Paragraph::new("🔧 Create New Tool")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD));
        f.render_widget(header, chunks[0]);
        
        // JSON editor
        let editor = Paragraph::new(self.new_tool_config.as_str())
            .block(Block::default().borders(Borders::ALL).title("Tool Configuration (JSON)"))
            .wrap(Wrap { trim: false });
        f.render_widget(editor, chunks[1]);
        
        // Controls
        let controls = Paragraph::new("Ctrl+S: Create Tool | Esc: Cancel | Type to edit")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(controls, chunks[2]);
    }
}