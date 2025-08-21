//! Configuration management for utilities

use std::path::{Path, PathBuf};
use std::fs;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json;
use tracing::{debug, error, info};

/// Configuration directory path
const CONFIG_DIR: &str = ".loki/utilities";

/// Tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub id: String,
    pub enabled: bool,
    pub settings: serde_json::Value,
}

/// MCP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub auto_connect: bool,
    pub env_vars: std::collections::HashMap<String, String>,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub id: String,
    pub enabled: bool,
    pub auto_update: bool,
    pub settings: serde_json::Value,
}

/// Daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub name: String,
    pub auto_start: bool,
    pub restart_policy: RestartPolicy,
    pub log_level: String,
}

/// Restart policy for daemons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartPolicy {
    Never,
    OnFailure,
    Always,
}

/// Main utilities configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilitiesConfig {
    pub tools: Vec<ToolConfig>,
    pub mcp_servers: Vec<McpServerConfig>,
    pub plugins: Vec<PluginConfig>,
    pub daemons: Vec<DaemonConfig>,
}

impl Default for UtilitiesConfig {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            mcp_servers: Vec::new(),
            plugins: Vec::new(),
            daemons: Vec::new(),
        }
    }
}

/// Configuration manager
pub struct ConfigManager {
    config_dir: PathBuf,
    config: UtilitiesConfig,
}

impl ConfigManager {
    /// Create a new config manager
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        let config_dir = home.join(CONFIG_DIR);
        
        // Create config directory if it doesn't exist
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)?;
            info!("Created config directory: {:?}", config_dir);
        }
        
        let mut manager = Self {
            config_dir,
            config: UtilitiesConfig::default(),
        };
        
        // Load existing config if available
        if let Err(e) = manager.load() {
            debug!("No existing config found or failed to load: {}", e);
        }
        
        Ok(manager)
    }
    
    /// Get the config file path
    fn config_file_path(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }
    
    /// Load configuration from disk
    pub fn load(&mut self) -> Result<()> {
        let config_path = self.config_file_path();
        
        if !config_path.exists() {
            debug!("Config file does not exist: {:?}", config_path);
            return Ok(());
        }
        
        let contents = fs::read_to_string(&config_path)?;
        self.config = serde_json::from_str(&contents)?;
        
        info!("Loaded configuration from {:?}", config_path);
        Ok(())
    }
    
    /// Save configuration to disk
    pub fn save(&self) -> Result<()> {
        let config_path = self.config_file_path();
        let contents = serde_json::to_string_pretty(&self.config)?;
        
        fs::write(&config_path, contents)?;
        info!("Saved configuration to {:?}", config_path);
        
        Ok(())
    }
    
    /// Get tool configuration
    pub fn get_tool_config(&self, tool_id: &str) -> Option<&ToolConfig> {
        self.config.tools.iter().find(|t| t.id == tool_id)
    }
    
    /// Update tool configuration
    pub fn update_tool_config(&mut self, config: ToolConfig) -> Result<()> {
        // Remove old config if exists
        self.config.tools.retain(|t| t.id != config.id);
        // Add new config
        self.config.tools.push(config);
        // Save to disk
        self.save()
    }
    
    /// Get MCP server configuration
    pub fn get_mcp_config(&self, server_name: &str) -> Option<&McpServerConfig> {
        self.config.mcp_servers.iter().find(|s| s.name == server_name)
    }
    
    /// Update MCP server configuration
    pub fn update_mcp_config(&mut self, config: McpServerConfig) -> Result<()> {
        // Remove old config if exists
        self.config.mcp_servers.retain(|s| s.name != config.name);
        // Add new config
        self.config.mcp_servers.push(config);
        // Save to disk
        self.save()
    }
    
    /// Get plugin configuration
    pub fn get_plugin_config(&self, plugin_id: &str) -> Option<&PluginConfig> {
        self.config.plugins.iter().find(|p| p.id == plugin_id)
    }
    
    /// Update plugin configuration
    pub fn update_plugin_config(&mut self, config: PluginConfig) -> Result<()> {
        // Remove old config if exists
        self.config.plugins.retain(|p| p.id != config.id);
        // Add new config
        self.config.plugins.push(config);
        // Save to disk
        self.save()
    }
    
    /// Get daemon configuration
    pub fn get_daemon_config(&self, daemon_name: &str) -> Option<&DaemonConfig> {
        self.config.daemons.iter().find(|d| d.name == daemon_name)
    }
    
    /// Update daemon configuration
    pub fn update_daemon_config(&mut self, config: DaemonConfig) -> Result<()> {
        // Remove old config if exists
        self.config.daemons.retain(|d| d.name != config.name);
        // Add new config
        self.config.daemons.push(config);
        // Save to disk
        self.save()
    }
    
    /// Export configuration to a specific path
    pub fn export(&self, path: &Path) -> Result<()> {
        let contents = serde_json::to_string_pretty(&self.config)?;
        fs::write(path, contents)?;
        info!("Exported configuration to {:?}", path);
        Ok(())
    }
    
    /// Import configuration from a specific path
    pub fn import(&mut self, path: &Path) -> Result<()> {
        let contents = fs::read_to_string(path)?;
        self.config = serde_json::from_str(&contents)?;
        self.save()?;
        info!("Imported configuration from {:?}", path);
        Ok(())
    }
    
    /// Get all configurations
    pub fn get_all(&self) -> &UtilitiesConfig {
        &self.config
    }
    
    /// Clear all configurations
    pub fn clear(&mut self) -> Result<()> {
        self.config = UtilitiesConfig::default();
        self.save()
    }
}