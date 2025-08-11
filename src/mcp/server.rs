//! MCP Server Management

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// MCP server instance (internal representation)
#[derive(Debug, Clone)]
pub struct McpServerInternal {
    /// Server ID/name
    pub id: String,
    
    /// Server description
    pub description: String,
    
    /// Command to execute
    pub command: String,
    
    /// Command arguments
    pub args: Vec<String>,
    
    /// Environment variables
    pub env: HashMap<String, String>,
    
    /// Available tools/capabilities
    pub capabilities: Vec<String>,
    
    /// Server configuration
    pub config: McpServerConfig,
}

/// MCP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Auto-start on connection
    pub auto_start: bool,
    
    /// Restart on failure
    pub auto_restart: bool,
    
    /// Maximum restart attempts
    pub max_restarts: u32,
    
    /// Health check interval in seconds
    pub health_check_interval: u64,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            auto_start: true,
            auto_restart: true,
            max_restarts: 3,
            health_check_interval: 30,
        }
    }
}

/// MCP server status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    /// Server name
    pub name: String,
    
    /// Connection status
    pub status: ConnectionStatus,
    
    /// Server description
    pub description: String,
    
    /// Command being run
    pub command: String,
    
    /// Command arguments
    pub args: Vec<String>,
    
    /// Available capabilities/tools
    pub capabilities: Vec<String>,
    
    /// Last time server was active
    pub last_active: DateTime<Utc>,
    
    /// Server uptime
    pub uptime: std::time::Duration,
    
    /// Error message if any
    pub error_message: Option<String>,
}

/// Connection status for MCP servers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    /// Server is connected and active
    Active,
    
    /// Server is connecting
    Connecting,
    
    /// Server is idle/disconnected
    Idle,
    
    /// Server connection failed
    Failed(String),
    
    /// Server is disabled
    Disabled,
}

impl ConnectionStatus {
    /// Get display color for the status
    pub fn color(&self) -> &str {
        match self {
            ConnectionStatus::Active => "green",
            ConnectionStatus::Connecting => "yellow",
            ConnectionStatus::Idle => "gray",
            ConnectionStatus::Failed(_) => "red",
            ConnectionStatus::Disabled => "dark_gray",
        }
    }
    
    /// Get status icon
    pub fn icon(&self) -> &str {
        match self {
            ConnectionStatus::Active => "🟢",
            ConnectionStatus::Connecting => "🟡",
            ConnectionStatus::Idle => "⚫",
            ConnectionStatus::Failed(_) => "🔴",
            ConnectionStatus::Disabled => "⚫",
        }
    }
}