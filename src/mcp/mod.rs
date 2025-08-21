//! MCP (Model Context Protocol) Module
//!
//! This module provides comprehensive MCP integration for Loki, allowing
//! communication with MCP servers to extend capabilities with external tools
//! and services through the standardized MCP protocol.

pub mod client;
pub mod config;
pub mod connection;
pub mod discovery;
pub mod marketplace;
pub mod protocol;
pub mod server;

// Re-export main types for convenience
pub use client::{McpClient, McpClientConfig, McpToolCall, McpToolResponse, McpServer};
pub use config::McpConfig;
pub use connection::{McpConnection, ConnectionStatus};
pub use server::{McpServerStatus};
pub use protocol::{McpRequest, McpResponse, McpMessage};
pub use discovery::McpDiscovery;
pub use marketplace::{McpMarketplaceEntry, McpMarketplace};

use std::sync::Arc;
use anyhow::Result;
use tracing::{info, warn};

/// Main MCP Manager that coordinates all MCP functionality
pub struct McpManager {
    /// MCP client for server communication
    pub client: Arc<McpClient>,
    /// MCP discovery for finding available servers
    pub discovery: McpDiscovery,
    /// MCP marketplace for browsing available servers
    pub marketplace: McpMarketplace,
    /// Configuration
    pub config: McpConfig,
}

impl McpManager {
    /// Create a new MCP manager with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(McpConfig::default()).await
    }
    
    /// Create a new MCP manager with custom configuration
    pub async fn with_config(config: McpConfig) -> Result<Self> {
        let client = Arc::new(McpClient::new(config.client_config.clone()));
        let discovery = McpDiscovery::new();
        let marketplace = McpMarketplace::new();
        
        Ok(Self {
            client,
            discovery,
            marketplace,
            config,
        })
    }
    
    /// Initialize and discover available MCP servers
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing MCP Manager");
        
        // Discover local MCP servers
        let servers = self.discovery.discover_servers().await?;
        info!("Discovered {} MCP servers", servers.len());
        
        // Initialize client with discovered servers
        for server in servers {
            if let Err(e) = self.client.register_server(server).await {
                warn!("Failed to register MCP server: {}", e);
            }
        }
        
        // Load marketplace data if configured
        if self.config.enable_marketplace {
            if let Err(e) = self.marketplace.refresh().await {
                warn!("Failed to load MCP marketplace: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Get list of available MCP servers
    pub async fn get_servers(&self) -> Vec<McpServerStatus> {
        self.client.get_server_status().await
    }
    
    /// Connect to a specific MCP server
    pub async fn connect_server(&self, server_id: &str) -> Result<()> {
        self.client.connect(server_id).await
    }
    
    /// Disconnect from a specific MCP server
    pub async fn disconnect_server(&self, server_id: &str) -> Result<()> {
        self.client.disconnect(server_id).await
    }
    
    /// Execute a tool on an MCP server
    pub async fn execute_tool(&self, server_id: &str, tool_name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let tool_call = McpToolCall {
            name: tool_name.to_string(),
            arguments: args,
        };
        let response = self.client.call_tool(server_id, tool_call).await?;
        if response.success {
            Ok(response.content)
        } else {
            Err(anyhow::anyhow!("Tool execution failed: {:?}", response.error))
        }
    }
}

/// Create a pre-configured MCP client with standard configuration
///
/// This function creates an MCP client and automatically loads configuration
/// from standard locations in this order:
/// 1. ~/.eigencode/mcp-servers/mcp-config-multi.json (Loki standard)
/// 2. ~/.cursor/mcp.json (Cursor IDE)
/// 3. ~/Library/Application Support/Claude/claude_desktop_config.json (Claude Desktop)
///
/// # Returns
/// A configured MCP client ready for use, or an error if initialization fails
pub async fn create_standard_mcp_client() -> Result<McpClient> {
    let config = McpClientConfig::default();
    McpClient::new_with_standardconfig(config).await
}

/// Convenience alias for create_standard_mcp_client
pub async fn create_mcp_client() -> Result<McpClient> {
    create_standard_mcp_client().await
}