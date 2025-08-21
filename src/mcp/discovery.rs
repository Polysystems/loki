//! MCP Server Discovery

use std::path::PathBuf;
use anyhow::Result;
use tracing::{debug, info, warn};

use super::config::{McpFileConfig, McpServerConfig};
use super::client::McpServer;

/// MCP server discovery service
pub struct McpDiscovery {
    /// Paths to search for MCP configurations
    search_paths: Vec<PathBuf>,
}

impl McpDiscovery {
    /// Create a new discovery service
    pub fn new() -> Self {
        Self {
            search_paths: Self::default_search_paths(),
        }
    }
    
    /// Get default search paths for MCP configurations
    fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        if let Some(home) = dirs::home_dir() {
            // Cursor configuration
            paths.push(home.join(".cursor/mcp.json"));
            
            // Eigencode configuration
            paths.push(home.join(".eigencode/mcp-servers/mcp-config-multi.json"));
            
            // Claude Desktop configuration
            paths.push(home.join("Library/Application Support/Claude/claude_desktop_config.json"));
            
            // Generic MCP config locations
            paths.push(home.join(".mcp/config.json"));
            paths.push(home.join(".config/mcp/config.json"));
        }
        
        if let Some(config_dir) = dirs::config_dir() {
            paths.push(config_dir.join("loki/mcp.json"));
            paths.push(config_dir.join("mcp/config.json"));
        }
        
        paths
    }
    
    /// Discover available MCP servers from configuration files
    pub async fn discover_servers(&self) -> Result<Vec<McpServer>> {
        let mut servers = Vec::new();
        
        for path in &self.search_paths {
            if path.exists() {
                debug!("Checking MCP configuration at: {:?}", path);
                match self.load_config_file(path).await {
                    Ok(mut discovered) => {
                        info!("Found {} MCP servers in {:?}", discovered.len(), path);
                        servers.append(&mut discovered);
                    }
                    Err(e) => {
                        warn!("Failed to load MCP config from {:?}: {}", path, e);
                    }
                }
            }
        }
        
        // Remove duplicates based on server name
        servers.dedup_by(|a, b| a.name == b.name);
        
        info!("Discovered total of {} unique MCP servers", servers.len());
        Ok(servers)
    }
    
    /// Load MCP servers from a configuration file
    async fn load_config_file(&self, path: &PathBuf) -> Result<Vec<McpServer>> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: McpFileConfig = serde_json::from_str(&content)?;
        
        let mut servers = Vec::new();
        for (name, server_config) in config.mcp_servers {
            servers.push(self.convert_to_server(name, server_config));
        }
        
        Ok(servers)
    }
    
    /// Convert configuration to McpServer
    fn convert_to_server(&self, id: String, config: McpServerConfig) -> McpServer {
        McpServer {
            name: id.clone(),
            description: format!("MCP Server: {}", id),
            command: config.command,
            args: config.args,
            env: config.env,
            capabilities: Vec::new(), // Will be populated on connection
            enabled: config.auto_start,
        }
    }
    
    /// Add a custom search path
    pub fn add_search_path(&mut self, path: PathBuf) {
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }
}