//! Shared types for the utilities module

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use crate::tools::metrics_collector::ToolStatus;  // Re-export for convenience
use crate::mcp::ConnectionStatus;

/// Tool entry for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolEntry {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub status: ToolStatus,
    pub last_used: Option<String>,  // Changed to String for display
    pub usage_count: u32,  // Changed to u32
}

/// MCP server status for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    pub name: String,
    pub status: ConnectionStatus,
    pub description: String,
    pub command: String,
    pub args: Vec<String>,
    pub capabilities: Vec<String>,
    pub last_active: DateTime<Utc>,
    pub uptime: std::time::Duration,
    pub error_message: Option<String>,
}

/// Plugin information for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub enabled: bool,
    pub status: PluginStatus,
    pub capabilities: Vec<String>,
}

/// Plugin status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PluginStatus {
    Active,
    Inactive,
    Loading,
    Error(String),
}

/// Daemon status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub name: String,
    pub description: String,
    pub pid: Option<u32>,
    pub status: DaemonState,
    pub uptime: Option<std::time::Duration>,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub last_restart: Option<DateTime<Utc>>,
}

/// Daemon state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DaemonState {
    Running,
    Stopped,
    Starting,
    Stopping,
    Error(String),
}

/// Utilities action for cross-tab communication
#[derive(Debug, Clone)]
pub enum UtilitiesAction {
    /// Tool-related actions
    ConfigureTool(String),
    ExecuteTool(String, serde_json::Value),
    RefreshTools,
    OpenToolConfig(String),
    
    /// MCP-related actions
    ConnectMcpServer(String),
    DisconnectMcpServer(String),
    RefreshMcpServers,
    
    /// Plugin-related actions
    InstallPlugin(String),
    UninstallPlugin(String),
    EnablePlugin(String),
    DisablePlugin(String),
    RefreshPlugins,
    
    /// Daemon-related actions
    StartDaemon(String),
    StopDaemon(String),
    RestartDaemon(String),
    ViewDaemonLogs(String),
    
    /// Todo-related actions
    TodoCreated { title: String, id: String },
    TodoCompleted { title: String, id: String },
    
    /// General actions
    RefreshAll,
    RefreshMonitoring,
    ShowNotification(String, NotificationType),
}

/// Notification type
#[derive(Debug, Clone)]
pub enum NotificationType {
    Info,
    Success,
    Warning,
    Error,
}

/// Utilities cache for UI display
#[derive(Debug, Clone)]
pub struct UtilitiesCache {
    pub tools: Vec<ToolEntry>,
    pub mcp_servers: HashMap<String, McpServerStatus>,
    pub plugins: Vec<PluginInfo>,
    pub daemons: HashMap<String, DaemonStatus>,
    pub last_update: DateTime<Utc>,
}