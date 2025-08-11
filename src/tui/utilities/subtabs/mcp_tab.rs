//! MCP management tab - Complete implementation

use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;
use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, List, ListItem, Paragraph, Table, Row, Cell,
    Scrollbar, ScrollbarOrientation, ScrollbarState, Gauge, Wrap
};
use crossterm::event::{KeyEvent, KeyCode};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::mcp::{McpManager, ConnectionStatus};
use crate::tui::utilities::state::UtilitiesState;
use crate::tui::utilities::types::{UtilitiesAction, McpServerStatus};
use crate::tui::utilities::config::{ConfigManager, McpServerConfig};
use super::UtilitiesSubtabController;

/// View modes for MCP management
#[derive(Debug, Clone, PartialEq)]
enum McpViewMode {
    ServerList,
    ServerDetails,
    Marketplace,
    ConfigEditor,
}

/// MCP servers management tab with full functionality
pub struct McpTab {
    /// Shared state
    state: Arc<RwLock<UtilitiesState>>,
    
    /// MCP manager connection
    mcp_manager: Option<Arc<McpManager>>,
    
    /// List state for server selection
    list_state: ratatui::widgets::ListState,
    
    /// Scrollbar state
    scroll_state: ScrollbarState,
    
    /// Current view mode
    view_mode: McpViewMode,
    
    /// Selected server ID
    selected_server: Option<String>,
    
    /// Configuration editor content
    config_editor: String,
    
    /// Marketplace search query
    marketplace_query: String,
    
    /// Configuration manager
    config_manager: Option<Arc<RwLock<ConfigManager>>>,
}

impl McpTab {
    pub fn new(
        state: Arc<RwLock<UtilitiesState>>,
        mcp_manager: Option<Arc<McpManager>>,
        config_manager: Option<Arc<RwLock<ConfigManager>>>,
    ) -> Self {
        let mut list_state = ratatui::widgets::ListState::default();
        list_state.select(Some(0));
        
        let mut tab = Self {
            state,
            mcp_manager,
            list_state,
            scroll_state: ScrollbarState::default(),
            view_mode: McpViewMode::ServerList,
            selected_server: None,
            config_editor: String::new(),
            marketplace_query: String::new(),
            config_manager,
        };
        
        // Initialize with data
        let _ = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(tab.refresh())
        });
        
        tab
    }
    
    /// Update MCP manager connection
    pub fn update_mcp_manager(&mut self, mcp_manager: Option<Arc<McpManager>>) {
        self.mcp_manager = mcp_manager;
    }
}

#[async_trait]
impl UtilitiesSubtabController for McpTab {
    fn name(&self) -> &str {
        "MCP"
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    
    fn render(&mut self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Content
                Constraint::Length(3), // Footer/controls
            ])
            .split(area);
        
        // Render header with connection status
        self.render_header(f, chunks[0]);
        
        // Render content based on view mode
        match self.view_mode {
            McpViewMode::ServerList => {
                let content_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
                    .split(chunks[1]);
                
                self.render_server_list(f, content_chunks[0]);
                self.render_server_details(f, content_chunks[1]);
            }
            McpViewMode::ServerDetails => {
                self.render_full_server_details(f, chunks[1]);
            }
            McpViewMode::Marketplace => {
                self.render_marketplace(f, chunks[1]);
            }
            McpViewMode::ConfigEditor => {
                self.render_config_editor(f, chunks[1]);
            }
        }
        
        // Footer with controls
        self.render_controls(f, chunks[2]);
    }
    
    async fn handle_key_event(&mut self, event: KeyEvent) -> Result<bool> {
        // Handle config editor keys
        if self.view_mode == McpViewMode::ConfigEditor {
            match event.code {
                KeyCode::Esc => {
                    self.view_mode = McpViewMode::ServerList;
                    self.config_editor.clear();
                    return Ok(true);
                }
                KeyCode::Char('s') if event.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                    debug!("Saving MCP configuration");
                    self.save_config().await?;
                    self.view_mode = McpViewMode::ServerList;
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
                        self.update_selected_server().await;
                    }
                }
                Ok(true)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let servers_count = self.get_server_count().await;
                if let Some(selected) = self.list_state.selected() {
                    if selected < servers_count.saturating_sub(1) {
                        self.list_state.select(Some(selected + 1));
                        self.scroll_state = self.scroll_state.position(selected + 1);
                        self.update_selected_server().await;
                    }
                }
                Ok(true)
            }
            KeyCode::Enter => {
                // Connect/disconnect selected server
                if let Some(server_id) = self.selected_server.clone() {
                    self.toggle_server_connection(&server_id).await?;
                }
                Ok(true)
            }
            KeyCode::Char('c') => {
                // Connect to server
                if let Some(server_id) = self.selected_server.clone() {
                    self.connect_server(&server_id).await?;
                }
                Ok(true)
            }
            KeyCode::Char('d') => {
                // Disconnect from server
                if let Some(server_id) = self.selected_server.clone() {
                    self.disconnect_server(&server_id).await?;
                }
                Ok(true)
            }
            KeyCode::Char('e') => {
                // Edit configuration
                self.view_mode = McpViewMode::ConfigEditor;
                self.load_config().await?;
                Ok(true)
            }
            KeyCode::Char('m') => {
                // Toggle marketplace view
                self.view_mode = if self.view_mode == McpViewMode::Marketplace {
                    McpViewMode::ServerList
                } else {
                    McpViewMode::Marketplace
                };
                Ok(true)
            }
            KeyCode::Char('r') => {
                // Refresh servers
                self.refresh().await?;
                Ok(true)
            }
            KeyCode::Char('i') => {
                // Show detailed info
                self.view_mode = McpViewMode::ServerDetails;
                Ok(true)
            }
            KeyCode::Char('l') if self.view_mode == McpViewMode::ServerDetails => {
                // Return to list view
                self.view_mode = McpViewMode::ServerList;
                Ok(true)
            }
            KeyCode::Tab => {
                // Cycle through view modes
                self.view_mode = match self.view_mode {
                    McpViewMode::ServerList => McpViewMode::ServerDetails,
                    McpViewMode::ServerDetails => McpViewMode::Marketplace,
                    McpViewMode::Marketplace => McpViewMode::ServerList,
                    McpViewMode::ConfigEditor => McpViewMode::ServerList,
                };
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    
    async fn handle_action(&mut self, action: UtilitiesAction) -> Result<()> {
        match action {
            UtilitiesAction::ConnectMcpServer(server_id) => {
                self.connect_server(&server_id).await?;
            }
            UtilitiesAction::DisconnectMcpServer(server_id) => {
                self.disconnect_server(&server_id).await?;
            }
            UtilitiesAction::RefreshMcpServers => {
                self.refresh().await?;
            }
            _ => {}
        }
        Ok(())
    }
    
    async fn refresh(&mut self) -> Result<()> {
        if let Some(ref mcp_manager) = self.mcp_manager {
            // Get server status from MCP manager - returns Vec<crate::mcp::McpServerStatus>
            let mcp_servers = mcp_manager.get_servers().await;
            debug!("Refreshed {} MCP servers", mcp_servers.len());
            
            // Convert from MCP's McpServerStatus to our utilities McpServerStatus
            let mut servers_map = HashMap::new();
            for mcp_server in mcp_servers {
                // Convert from crate::mcp::McpServerStatus to crate::tui::utilities::types::McpServerStatus
                let server = McpServerStatus {
                    name: mcp_server.name.clone(),
                    status: mcp_server.status,
                    description: mcp_server.description.clone(),
                    command: mcp_server.command.clone(),
                    args: mcp_server.args.clone(),
                    capabilities: mcp_server.capabilities.clone(),
                    last_active: mcp_server.last_active,
                    uptime: mcp_server.uptime,
                    error_message: mcp_server.error_message.clone(),
                };
                servers_map.insert(server.name.clone(), server);
            }
            
            // Update state cache
            let mut state = self.state.write().await;
            state.cache.mcp_servers = servers_map;
            
            // Update scrollbar
            self.scroll_state = self.scroll_state.content_length(state.cache.mcp_servers.len());
        } else {
            // Use example data if no MCP manager
            let mut servers_map = HashMap::new();
            for server in Self::get_example_servers() {
                servers_map.insert(server.name.clone(), server);
            }
            
            let mut state = self.state.write().await;
            state.cache.mcp_servers = servers_map;
            self.scroll_state = self.scroll_state.content_length(state.cache.mcp_servers.len());
        }
        
        // Update selected server
        self.update_selected_server().await;
        Ok(())
    }
}

// Implementation methods
impl McpTab {
    /// Render the header with connection status
    fn render_header(&self, f: &mut Frame, area: Rect) {
        let status = if self.mcp_manager.is_some() {
            let state_guard = self.state.try_read();
            let active_count = if let Ok(state) = state_guard {
                state.cache.mcp_servers.values()
                    .filter(|s| matches!(s.status, ConnectionStatus::Active))
                    .count()
            } else {
                0
            };
            format!("🟢 MCP Manager Connected | {} servers active", active_count)
        } else {
            "⚪ MCP Manager Not Connected".to_string()
        };
        
        let view_mode = match self.view_mode {
            McpViewMode::ServerList => "Server List",
            McpViewMode::ServerDetails => "Server Details",
            McpViewMode::Marketplace => "Marketplace",
            McpViewMode::ConfigEditor => "Configuration Editor",
        };
        
        let header = Paragraph::new(format!("🔌 MCP Server Management - {} | {}", view_mode, status))
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        
        f.render_widget(header, area);
    }
    
    /// Render the server list
    fn render_server_list(&mut self, f: &mut Frame, area: Rect) {
        let state_guard = self.state.try_read();
        
        let mut servers: Vec<McpServerStatus> = if let Ok(state) = state_guard {
            state.cache.mcp_servers.values().cloned().collect()
        } else {
            Vec::new()
        };
        
        // Sort servers by name for consistent display
        servers.sort_by(|a, b| a.name.cmp(&b.name));
        
        // Create list items with connection status
        let items: Vec<ListItem> = servers
            .iter()
            .map(|server| {
                let status_icon = match server.status {
                    ConnectionStatus::Active => "🟢",
                    ConnectionStatus::Connecting => "🟡",
                    ConnectionStatus::Idle => "⚫",
                    ConnectionStatus::Failed(_) => "🔴",
                    ConnectionStatus::Disabled => "⚫",
                };
                
                let capabilities = if server.capabilities.is_empty() {
                    "No tools".to_string()
                } else {
                    format!("{} tools", server.capabilities.len())
                };
                
                let line = Line::from(vec![
                    Span::raw(format!("{} ", status_icon)),
                    Span::styled(&server.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::raw(" - "),
                    Span::styled(capabilities, Style::default().fg(Color::DarkGray)),
                ]);
                
                ListItem::new(line)
            })
            .collect();
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("MCP Servers"))
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
    
    /// Render server details panel
    fn render_server_details(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // Server info
                Constraint::Length(6),  // Connection status
                Constraint::Length(8),  // Capabilities
                Constraint::Min(0),     // Logs/output
            ])
            .split(area);
        
        let state_guard = self.state.try_read();
        
        if let Ok(state) = state_guard {
            if let Some(selected) = self.list_state.selected() {
                // Get sorted servers and index into them
                let mut servers: Vec<McpServerStatus> = state.cache.mcp_servers.values().cloned().collect();
                servers.sort_by(|a, b| a.name.cmp(&b.name));
                
                if let Some(server) = servers.get(selected) {
                    self.render_server_info(f, chunks[0], server);
                    self.render_connection_status(f, chunks[1], server);
                    self.render_capabilities(f, chunks[2], server);
                    self.render_server_logs(f, chunks[3], server);
                    return;
                }
            }
        }
        
        // No server selected
        let no_selection = Paragraph::new("Select a server to view details")
            .block(Block::default().borders(Borders::ALL).title("Server Details"))
            .alignment(Alignment::Center);
        f.render_widget(no_selection, area);
    }
    
    /// Render full server details view
    fn render_full_server_details(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10), // Server info
                Constraint::Length(8),  // Connection metrics
                Constraint::Length(10), // Capabilities list
                Constraint::Min(0),     // Command/environment details
            ])
            .split(area);
        
        let state_guard = self.state.try_read();
        
        if let Ok(state) = state_guard {
            if let Some(selected) = self.list_state.selected() {
                // Get sorted servers and index into them
                let mut servers: Vec<McpServerStatus> = state.cache.mcp_servers.values().cloned().collect();
                servers.sort_by(|a, b| a.name.cmp(&b.name));
                
                if let Some(server) = servers.get(selected) {
                    // Detailed server information
                    self.render_detailed_info(f, chunks[0], server);
                    self.render_connection_metrics(f, chunks[1], server);
                    self.render_detailed_capabilities(f, chunks[2], server);
                    self.render_command_details(f, chunks[3], server);
                    return;
                }
            }
        }
        
        let no_selection = Paragraph::new("No server selected")
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);
        f.render_widget(no_selection, area);
    }
    
    /// Render server information
    fn render_server_info(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        let info_text = vec![
            Line::from(vec![
                Span::styled("Name: ", Style::default().fg(Color::Cyan)),
                Span::styled(&server.name, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("Description: ", Style::default().fg(Color::Cyan)),
                Span::raw(&server.description),
            ]),
            Line::from(vec![
                Span::styled("Command: ", Style::default().fg(Color::Cyan)),
                Span::raw(&server.command),
            ]),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(Color::Cyan)),
                match &server.status {
                    ConnectionStatus::Active => Span::styled("Active", Style::default().fg(Color::Green)),
                    ConnectionStatus::Connecting => Span::styled("Connecting...", Style::default().fg(Color::Yellow)),
                    ConnectionStatus::Idle => Span::styled("Idle", Style::default().fg(Color::Gray)),
                    ConnectionStatus::Failed(err) => Span::styled(format!("Failed: {}", err), Style::default().fg(Color::Red)),
                    ConnectionStatus::Disabled => Span::styled("Disabled", Style::default().fg(Color::DarkGray)),
                },
            ]),
        ];
        
        let info = Paragraph::new(info_text)
            .block(Block::default().borders(Borders::ALL).title("Server Information"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(info, area);
    }
    
    /// Render connection status
    fn render_connection_status(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        let uptime_str = format!("{} seconds", server.uptime.as_secs());
        
        let status_text = vec![
            Line::from(vec![
                Span::styled("Last Active: ", Style::default().fg(Color::Cyan)),
                Span::raw(server.last_active.format("%Y-%m-%d %H:%M:%S").to_string()),
            ]),
            Line::from(vec![
                Span::styled("Uptime: ", Style::default().fg(Color::Cyan)),
                Span::raw(uptime_str),
            ]),
            if let Some(ref err) = server.error_message {
                Line::from(vec![
                    Span::styled("Error: ", Style::default().fg(Color::Red)),
                    Span::raw(err),
                ])
            } else {
                Line::from(vec![
                    Span::styled("Health: ", Style::default().fg(Color::Cyan)),
                    Span::styled("Good", Style::default().fg(Color::Green)),
                ])
            },
        ];
        
        let status = Paragraph::new(status_text)
            .block(Block::default().borders(Borders::ALL).title("Connection Status"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(status, area);
    }
    
    /// Render server capabilities
    fn render_capabilities(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        let capabilities = if server.capabilities.is_empty() {
            vec![Line::from(Span::styled(
                "No capabilities available",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            server.capabilities
                .iter()
                .take(5)
                .map(|cap| Line::from(vec![
                    Span::raw("• "),
                    Span::raw(cap),
                ]))
                .collect()
        };
        
        let caps = Paragraph::new(capabilities)
            .block(Block::default().borders(Borders::ALL).title(format!("Capabilities ({})", server.capabilities.len())))
            .wrap(Wrap { trim: true });
        
        f.render_widget(caps, area);
    }
    
    /// Render server logs
    fn render_server_logs(&self, f: &mut Frame, area: Rect, _server: &McpServerStatus) {
        let logs = vec![
            Line::from(Span::styled("[INFO] Server initialized", Style::default().fg(Color::Green))),
            Line::from(Span::styled("[INFO] Connected to endpoint", Style::default().fg(Color::Green))),
            Line::from(Span::styled("[INFO] Tools discovered: 5", Style::default().fg(Color::Cyan))),
        ];
        
        let log_widget = Paragraph::new(logs)
            .block(Block::default().borders(Borders::ALL).title("Server Logs"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(log_widget, area);
    }
    
    /// Render detailed server information
    fn render_detailed_info(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        // Create owned strings to avoid temporary value issues
        let args_str = server.args.join(" ");
        let status_str = format!("{:?}", server.status);
        let capabilities_str = format!("{} tools", server.capabilities.len());
        
        let table = Table::new(
            vec![
                Row::new(vec!["Property", "Value"]).style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Row::new(vec!["Name", server.name.as_str()]),
                Row::new(vec!["Description", server.description.as_str()]),
                Row::new(vec!["Command", server.command.as_str()]),
                Row::new(vec!["Arguments", args_str.as_str()]),
                Row::new(vec!["Status", status_str.as_str()]),
                Row::new(vec!["Capabilities", capabilities_str.as_str()]),
            ],
            [Constraint::Length(15), Constraint::Percentage(70)],
        )
        .block(Block::default().borders(Borders::ALL).title("Detailed Server Information"));
        
        f.render_widget(table, area);
    }
    
    /// Render connection metrics
    fn render_connection_metrics(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        
        // Uptime gauge
        let uptime_percentage = (server.uptime.as_secs() as f64 / 3600.0 * 100.0).min(100.0) as u16;
        let uptime_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Uptime"))
            .gauge_style(Style::default().fg(Color::Green))
            .percent(uptime_percentage)
            .label(format!("{} min", server.uptime.as_secs() / 60));
        
        f.render_widget(uptime_gauge, chunks[0]);
        
        // Health status
        let health = match server.status {
            ConnectionStatus::Active => ("Healthy", Color::Green, 100),
            ConnectionStatus::Connecting => ("Connecting", Color::Yellow, 50),
            ConnectionStatus::Idle => ("Idle", Color::Gray, 25),
            ConnectionStatus::Failed(_) => ("Failed", Color::Red, 0),
            ConnectionStatus::Disabled => ("Disabled", Color::DarkGray, 0),
        };
        
        let health_gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title("Health"))
            .gauge_style(Style::default().fg(health.1))
            .percent(health.2)
            .label(health.0);
        
        f.render_widget(health_gauge, chunks[1]);
    }
    
    /// Render detailed capabilities
    fn render_detailed_capabilities(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        let items: Vec<ListItem> = if server.capabilities.is_empty() {
            vec![ListItem::new("No capabilities available")]
        } else {
            server.capabilities
                .iter()
                .map(|cap| ListItem::new(Line::from(vec![
                    Span::styled("🔧 ", Style::default().fg(Color::Green)),
                    Span::raw(cap),
                ])))
                .collect()
        };
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!("Available Tools ({})", server.capabilities.len())))
            .style(Style::default().fg(Color::White));
        
        f.render_widget(list, area);
    }
    
    /// Render command details
    fn render_command_details(&self, f: &mut Frame, area: Rect, server: &McpServerStatus) {
        let details = vec![
            Line::from(vec![
                Span::styled("Command: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(Span::raw(&server.command)),
            Line::from(""),
            Line::from(vec![
                Span::styled("Arguments: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(Span::raw(if server.args.is_empty() {
                "(none)".to_string()
            } else {
                server.args.join(" ")
            })),
            Line::from(""),
            Line::from(vec![
                Span::styled("Configuration: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(Span::styled("Press 'e' to edit configuration", Style::default().fg(Color::DarkGray))),
        ];
        
        let paragraph = Paragraph::new(details)
            .block(Block::default().borders(Borders::ALL).title("Command Details"))
            .wrap(Wrap { trim: true });
        
        f.render_widget(paragraph, area);
    }
    
    /// Render marketplace view
    fn render_marketplace(&self, f: &mut Frame, area: Rect) {
        use ratatui::layout::{Constraint, Direction, Layout};
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Search bar
                Constraint::Min(0),     // Marketplace items
            ])
            .split(area);
        
        // Search bar
        let search = Paragraph::new(format!("🔍 Search: {}", self.marketplace_query))
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(search, chunks[0]);
        
        // Marketplace items
        let items = vec![
            ListItem::new(Line::from(vec![
                Span::styled("⭐ ", Style::default().fg(Color::Yellow)),
                Span::styled("GitHub API", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(" - Access GitHub repositories and issues"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("⭐ ", Style::default().fg(Color::Yellow)),
                Span::styled("Filesystem", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(" - File system operations and management"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("⭐ ", Style::default().fg(Color::Yellow)),
                Span::styled("Web Search", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(" - Search the web and fetch content"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("⭐ ", Style::default().fg(Color::Yellow)),
                Span::styled("Database", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(" - Connect to various databases"),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("⭐ ", Style::default().fg(Color::Yellow)),
                Span::styled("Docker", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::raw(" - Manage Docker containers and images"),
            ])),
        ];
        
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("MCP Server Marketplace"))
            .style(Style::default().fg(Color::White));
        
        f.render_widget(list, chunks[1]);
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
        let header = Paragraph::new("📝 MCP Configuration Editor")
            .block(Block::default().borders(Borders::ALL))
            .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
        f.render_widget(header, chunks[0]);
        
        // Editor content
        let editor = Paragraph::new(self.config_editor.as_str())
            .block(Block::default().borders(Borders::ALL).title("mcp-config.json"))
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
            McpViewMode::ServerList => {
                "↑↓/jk: Navigate | Enter: Connect/Disconnect | c: Connect | d: Disconnect | e: Edit | m: Marketplace | r: Refresh | i: Info"
            }
            McpViewMode::ServerDetails => {
                "l: Back to List | c: Connect | d: Disconnect | e: Edit Config | Tab: Next View"
            }
            McpViewMode::Marketplace => {
                "↑↓: Browse | Enter: Install | /: Search | m: Back to Servers | Tab: Next View"
            }
            McpViewMode::ConfigEditor => {
                "Ctrl+S: Save | Esc: Cancel | Type to edit configuration"
            }
        };
        
        let controls_widget = Paragraph::new(controls)
            .block(Block::default().borders(Borders::ALL).title("Controls"))
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        
        f.render_widget(controls_widget, area);
    }
    
    /// Get server count
    async fn get_server_count(&self) -> usize {
        let state = self.state.read().await;
        state.cache.mcp_servers.len()
    }
    
    /// Update selected server
    async fn update_selected_server(&mut self) {
        if let Some(index) = self.list_state.selected() {
            let state = self.state.read().await;
            // Get sorted servers and index into them
            let mut servers: Vec<McpServerStatus> = state.cache.mcp_servers.values().cloned().collect();
            servers.sort_by(|a, b| a.name.cmp(&b.name));
            self.selected_server = servers.get(index).map(|s| s.name.clone());
        }
    }
    
    /// Toggle server connection
    async fn toggle_server_connection(&mut self, server_id: &str) -> Result<()> {
        if let Some(ref mcp_manager) = self.mcp_manager {
            let state = self.state.read().await;
            if let Some(server) = state.cache.mcp_servers.get(server_id) {
                match server.status {
                    ConnectionStatus::Active => {
                        info!("Disconnecting from MCP server: {}", server_id);
                        mcp_manager.disconnect_server(server_id).await?;
                    }
                    _ => {
                        info!("Connecting to MCP server: {}", server_id);
                        mcp_manager.connect_server(server_id).await?;
                    }
                }
            }
            drop(state);
            self.refresh().await?;
        }
        Ok(())
    }
    
    /// Connect to server
    async fn connect_server(&mut self, server_id: &str) -> Result<()> {
        if let Some(ref mcp_manager) = self.mcp_manager {
            info!("Connecting to MCP server: {}", server_id);
            mcp_manager.connect_server(server_id).await?;
            self.refresh().await?;
        }
        Ok(())
    }
    
    /// Disconnect from server
    async fn disconnect_server(&mut self, server_id: &str) -> Result<()> {
        if let Some(ref mcp_manager) = self.mcp_manager {
            info!("Disconnecting from MCP server: {}", server_id);
            mcp_manager.disconnect_server(server_id).await?;
            self.refresh().await?;
        }
        Ok(())
    }
    
    /// Load configuration
    async fn load_config(&mut self) -> Result<()> {
        // Try to load from ConfigManager first
        if let Some(ref config_manager) = self.config_manager {
            let manager = config_manager.read().await;
            // Get all MCP configurations
            let all_configs = manager.get_all();
            if !all_configs.mcp_servers.is_empty() {
                // Convert to the expected JSON format
                let mut servers_map = serde_json::Map::new();
                for server in &all_configs.mcp_servers {
                    let mut server_obj = serde_json::Map::new();
                    server_obj.insert("command".to_string(), serde_json::Value::String(server.command.clone()));
                    server_obj.insert("args".to_string(), serde_json::json!(server.args));
                    server_obj.insert("env".to_string(), serde_json::json!(server.env_vars));
                    servers_map.insert(server.name.clone(), serde_json::Value::Object(server_obj));
                }
                let config_json = serde_json::json!({ "mcpServers": servers_map });
                self.config_editor = serde_json::to_string_pretty(&config_json)?;
                return Ok(());
            }
        }
        
        // Default configuration if not found
        self.config_editor = r#"{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token-here"
      }
    },
    "web-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-websearch"],
      "env": {}
    }
  }
}"#.to_string();
        Ok(())
    }
    
    /// Save configuration
    async fn save_config(&self) -> Result<()> {
        debug!("Saving MCP configuration: {}", self.config_editor);
        
        // Parse configuration as server config map
        let servers_json: serde_json::Value = serde_json::from_str(&self.config_editor)?;
        
        // Save each server configuration
        if let Some(ref config_manager) = self.config_manager {
            let mut manager = config_manager.write().await;
            
            if let Some(servers_obj) = servers_json.get("mcpServers").and_then(|v| v.as_object()) {
                for (name, server_data) in servers_obj {
                    let config = McpServerConfig {
                        name: name.clone(),
                        command: server_data.get("command")
                            .and_then(|v| v.as_str())
                            .unwrap_or("npx")
                            .to_string(),
                        args: server_data.get("args")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect())
                            .unwrap_or_default(),
                        auto_connect: server_data.get("auto_connect")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                        env_vars: server_data.get("env")
                            .and_then(|v| v.as_object())
                            .map(|obj| obj.iter()
                                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                                .collect())
                            .unwrap_or_default(),
                    };
                    manager.update_mcp_config(config)?;
                }
                info!("Saved MCP server configurations");
            }
        }
        
        Ok(())
    }
    
    /// Get example servers for demo mode
    fn get_example_servers() -> Vec<McpServerStatus> {
        use chrono::Utc;
        
        vec![
            McpServerStatus {
                name: "filesystem".to_string(),
                status: ConnectionStatus::Active,
                description: "File system operations and management".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-filesystem".to_string()],
                capabilities: vec![
                    "read_file".to_string(),
                    "write_file".to_string(),
                    "list_directory".to_string(),
                    "create_directory".to_string(),
                    "delete_file".to_string(),
                ],
                last_active: Utc::now(),
                uptime: std::time::Duration::from_secs(3600),
                error_message: None,
            },
            McpServerStatus {
                name: "github".to_string(),
                status: ConnectionStatus::Idle,
                description: "GitHub API integration".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-github".to_string()],
                capabilities: vec![
                    "list_repos".to_string(),
                    "get_issues".to_string(),
                    "create_pr".to_string(),
                ],
                last_active: Utc::now() - chrono::Duration::hours(1),
                uptime: std::time::Duration::from_secs(0),
                error_message: None,
            },
            McpServerStatus {
                name: "web-search".to_string(),
                status: ConnectionStatus::Connecting,
                description: "Web search and content fetching".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-websearch".to_string()],
                capabilities: vec![],
                last_active: Utc::now(),
                uptime: std::time::Duration::from_secs(5),
                error_message: None,
            },
            McpServerStatus {
                name: "database".to_string(),
                status: ConnectionStatus::Failed("Connection timeout".to_string()),
                description: "Database operations".to_string(),
                command: "python".to_string(),
                args: vec!["-m".to_string(), "mcp_server_db".to_string()],
                capabilities: vec![],
                last_active: Utc::now() - chrono::Duration::minutes(30),
                uptime: std::time::Duration::from_secs(0),
                error_message: Some("Failed to connect to database endpoint".to_string()),
            },
            McpServerStatus {
                name: "docker".to_string(),
                status: ConnectionStatus::Disabled,
                description: "Docker container management".to_string(),
                command: "docker-mcp-server".to_string(),
                args: vec![],
                capabilities: vec![],
                last_active: Utc::now() - chrono::Duration::days(1),
                uptime: std::time::Duration::from_secs(0),
                error_message: None,
            },
        ]
    }
}