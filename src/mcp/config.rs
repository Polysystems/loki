//! MCP Configuration Module

use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};

use super::client::McpClientConfig;

/// Main MCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Client configuration
    pub client_config: McpClientConfig,
    
    /// Enable marketplace features
    pub enable_marketplace: bool,
    
    /// Configuration file paths to check
    pub config_paths: Vec<PathBuf>,
    
    /// Auto-connect to servers on startup
    pub auto_connect: bool,
    
    /// Server discovery settings
    pub discovery: DiscoveryConfig,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            client_config: McpClientConfig::default(),
            enable_marketplace: true,
            config_paths: vec![
                dirs::home_dir().unwrap_or_default().join(".cursor/mcp.json"),
                dirs::home_dir().unwrap_or_default().join(".eigencode/mcp-servers/mcp-config-multi.json"),
                dirs::home_dir().unwrap_or_default().join("Library/Application Support/Claude/claude_desktop_config.json"),
            ],
            auto_connect: true,
            discovery: DiscoveryConfig::default(),
        }
    }
}

/// MCP server discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable automatic discovery
    pub enabled: bool,
    
    /// Discovery interval
    pub interval: Duration,
    
    /// Paths to search for MCP configurations
    pub search_paths: Vec<PathBuf>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            search_paths: vec![
                dirs::config_dir().unwrap_or_default(),
                dirs::home_dir().unwrap_or_default(),
            ],
        }
    }
}

/// MCP server configuration from JSON files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name/ID
    pub name: String,
    
    /// Command to execute
    pub command: String,
    
    /// Command arguments
    pub args: Vec<String>,
    
    /// Environment variables
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    
    /// Working directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    
    /// Auto-start this server
    #[serde(default = "default_true")]
    pub auto_start: bool,
}

fn default_true() -> bool {
    true
}

/// Root MCP configuration structure (as found in config files)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpFileConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: std::collections::HashMap<String, McpServerConfig>,
}