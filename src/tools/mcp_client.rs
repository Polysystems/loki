//! MCP (Model Context Protocol) Client Integration for Loki
//!
//! This module provides integration with MCP servers to extend Loki's
//! capabilities with external tools and services through the standardized MCP
//! protocol.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// MCP Client for communicating with MCP servers
#[derive(Debug)]
pub struct McpClient {
    /// Available MCP servers
    servers: HashMap<String, McpServer>,
    /// Client configuration
    config: McpClientConfig,
    /// Active server connections
    active_connections: HashMap<String, McpConnection>,
}

/// Configuration for MCP client
#[derive(Debug, Clone)]
pub struct McpClientConfig {
    /// Timeout for MCP operations
    pub timeout: Duration,
    /// Maximum number of concurrent MCP calls
    pub max_concurrent: usize,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self { timeout: Duration::from_secs(30), max_concurrent: 10, verbose: false }
    }
}

/// MCP server configuration
#[derive(Debug, Clone)]
pub struct McpServer {
    /// Server name/identifier
    pub name: String,
    /// Server description
    pub description: String,
    /// Command to start the server
    pub command: String,
    /// Command arguments
    pub args: Vec<String>,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Server capabilities
    pub capabilities: Vec<String>,
    /// Whether the server is enabled
    pub enabled: bool,
}

/// MCP tool call request
#[derive(Debug, Clone, Serialize)]
pub struct McpToolCall {
    /// Tool name to call
    pub name: String,
    /// Arguments for the tool
    pub arguments: Value,
}

/// MCP tool call response
#[derive(Debug, Clone, Deserialize)]
pub struct McpToolResponse {
    /// Whether the call was successful
    pub success: bool,
    /// Response content
    pub content: Value,
    /// Error message if failed
    pub error: Option<String>,
}

/// MCP server capabilities
#[derive(Debug, Clone, Deserialize)]
pub struct McpCapabilities {
    /// Available tools
    pub tools: Vec<McpTool>,
    /// Server information
    pub server_info: McpServerInfo,
}

/// MCP tool definition
#[derive(Debug, Clone, Deserialize)]
pub struct McpTool {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema
    pub input_schema: Value,
}

/// MCP server information
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
}

/// Active MCP server connection
#[derive(Debug)]
pub struct McpConnection {
    /// Server name
    pub server_name: String,
    /// Process handle
    pub process: Arc<Mutex<Child>>,
    /// Stdin writer
    pub stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    /// Stdout reader
    pub stdout: Arc<Mutex<BufReader<tokio::process::ChildStdout>>>,
    /// Whether the connection is initialized
    pub initialized: bool,
    /// Connection start time
    pub connected_at: std::time::Instant,
}

/// MCP server status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    /// Server name
    pub name: String,
    /// Server description
    pub description: String,
    /// Connection status
    pub status: String,
    /// Whether the server is enabled
    pub enabled: bool,
    /// Server capabilities
    pub capabilities: Vec<String>,
}

impl McpClient {
    /// Create a new MCP client
    pub fn new(config: McpClientConfig) -> Self {
        Self { 
            servers: HashMap::new(), 
            config,
            active_connections: HashMap::new(),
        }
    }

    /// Add an MCP server
    pub fn add_server(&mut self, server: McpServer) {
        info!("Adding MCP server: {}", server.name);
        self.servers.insert(server.name.clone(), server);
    }

    /// Load MCP servers from configuration
    pub async fn load_fromconfig(&mut self, config_path: &str) -> Result<()> {
        info!("Loading MCP configuration from: {}", config_path);

        let config_content = tokio::fs::read_to_string(config_path).await?;
        let config: Value = serde_json::from_str(&config_content)?;

        if let Some(mcp_servers) = config.get("mcpServers").and_then(|v| v.as_object()) {
            for (name, serverconfig) in mcp_servers {
                let server = self.parse_serverconfig(name, serverconfig)?;
                self.add_server(server);
            }
        }

        Ok(())
    }

    /// Create a new MCP client and load from standard configuration locations
    pub async fn new_with_standardconfig(config: McpClientConfig) -> Result<Self> {
        let mut client = Self::new(config);

        // Try loading from different standard locations in order of preference
        let config_paths = [
            "/Users/thermo/.eigencode/mcp-servers/mcp-config-multi.json",
            "/Users/thermo/.cursor/mcp.json",
            &format!(
                "{}/Library/Application Support/Claude/claude_desktopconfig.json",
                std::env::var("HOME").unwrap_or_default()
            ),
        ];

        let mut config_loaded = false;
        for config_path in &config_paths {
            if std::path::Path::new(config_path).exists() {
                match client.load_fromconfig(config_path).await {
                    Ok(()) => {
                        info!("Successfully loaded MCP configuration from: {}", config_path);
                        config_loaded = true;
                        break;
                    }
                    Err(e) => {
                        warn!("Failed to load MCP config from {}: {}", config_path, e);
                        continue;
                    }
                }
            }
        }

        if !config_loaded {
            warn!("No MCP configuration found in standard locations. Client will have no servers.");
        }

        Ok(client)
    }

    /// Parse server configuration from JSON
    fn parse_serverconfig(&self, name: &str, config: &Value) -> Result<McpServer> {
        let command = config
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing command for server {}", name))?;

        let args: Vec<String> = config
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let env: HashMap<String, String> = config
            .get("env")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        Ok(McpServer {
            name: name.to_string(),
            description: format!("MCP server for {}", name),
            command: command.to_string(),
            args,
            env,
            capabilities: Vec::new(), // Will be populated when server starts
            enabled: true,
        })
    }

    /// List available MCP servers
    pub async fn list_servers(&self) -> Result<Vec<String>> {
        Ok(self.servers.keys().cloned().collect())
    }
    
    /// Get capabilities for an MCP server
    pub async fn get_capabilities(&self, server_name: &str) -> Result<McpCapabilities> {
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| anyhow!("MCP server '{}' not found", server_name))?;

        if !server.enabled {
            return Err(anyhow!("MCP server '{}' is disabled", server_name));
        }

        debug!("Getting capabilities for MCP server: {}", server_name);

        // Start the MCP server process
        let mut cmd = Command::new(&server.command);
        cmd.args(&server.args);
        cmd.envs(&server.env);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        // Get stdin and stdout handles
        let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("Failed to get stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to get stdout"))?;
        let mut reader = BufReader::new(stdout);

        // Send initialize request
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": true
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "Loki AI",
                    "version": "1.0.0"
                }
            }
        });

        let request_line = format!("{}\n", init_request);
        stdin.write_all(request_line.as_bytes()).await?;
        stdin.flush().await?;

        // Read response with timeout
        let mut response_line = String::new();
        let read_result = timeout(self.config.timeout, reader.read_line(&mut response_line)).await;

        match read_result {
            Ok(Ok(_)) => {
                let response: Value = serde_json::from_str(&response_line)?;

                if let Some(result) = response.get("result") {
                    let server_info = McpServerInfo {
                        name: result
                            .get("serverInfo")
                            .and_then(|v| v.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or(server_name)
                            .to_string(),
                        version: result
                            .get("serverInfo")
                            .and_then(|v| v.get("version"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    };

                    // Get tools list
                    let tools = self.get_tools_list(&mut stdin, &mut reader).await?;

                    // Clean up process
                    let _ = child.kill().await;

                    return Ok(McpCapabilities { tools, server_info });
                } else {
                    return Err(anyhow!("Invalid response from MCP server: {}", response));
                }
            }
            Ok(Err(e)) => return Err(anyhow!("Failed to read from MCP server: {}", e)),
            Err(_) => return Err(anyhow!("Timeout waiting for MCP server response")),
        }
    }

    /// Get tools list from MCP server
    async fn get_tools_list(
        &self,
        stdin: &mut tokio::process::ChildStdin,
        reader: &mut BufReader<tokio::process::ChildStdout>,
    ) -> Result<Vec<McpTool>> {
        // Send tools/list request
        let tools_request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        });

        let request_line = format!("{}\n", tools_request);
        stdin.write_all(request_line.as_bytes()).await?;
        stdin.flush().await?;

        // Read response
        let mut response_line = String::new();
        let read_result = timeout(self.config.timeout, reader.read_line(&mut response_line)).await;

        match read_result {
            Ok(Ok(_)) => {
                let response: Value = serde_json::from_str(&response_line)?;

                if let Some(result) = response.get("result") {
                    if let Some(tools_array) = result.get("tools").and_then(|v| v.as_array()) {
                        let mut tools = Vec::new();

                        for tool_value in tools_array {
                            let tool = McpTool {
                                name: tool_value
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown")
                                    .to_string(),
                                description: tool_value
                                    .get("description")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                input_schema: tool_value
                                    .get("inputSchema")
                                    .cloned()
                                    .unwrap_or(json!({})),
                            };
                            tools.push(tool);
                        }

                        return Ok(tools);
                    }
                }

                Err(anyhow!("Invalid tools response from MCP server"))
            }
            Ok(Err(e)) => Err(anyhow!("Failed to read tools from MCP server: {}", e)),
            Err(_) => Err(anyhow!("Timeout waiting for tools response")),
        }
    }

    /// Call an MCP tool
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_call: McpToolCall,
    ) -> Result<McpToolResponse> {
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| anyhow!("MCP server '{}' not found", server_name))?;

        if !server.enabled {
            return Err(anyhow!("MCP server '{}' is disabled", server_name));
        }

        debug!("Calling MCP tool '{}' on server '{}'", tool_call.name, server_name);

        // Start the MCP server process
        let mut cmd = Command::new(&server.command);
        cmd.args(&server.args);
        cmd.envs(&server.env);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        // Get stdin and stdout handles
        let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("Failed to get stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to get stdout"))?;
        let mut reader = BufReader::new(stdout);

        // Initialize the server first
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": true
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "Loki AI",
                    "version": "1.0.0"
                }
            }
        });

        let request_line = format!("{}\n", init_request);
        stdin.write_all(request_line.as_bytes()).await?;
        stdin.flush().await?;

        // Read initialization response
        let mut response_line = String::new();
        timeout(self.config.timeout, reader.read_line(&mut response_line)).await??;

        // Send tool call request
        let tool_request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }
        });

        let request_line = format!("{}\n", tool_request);
        stdin.write_all(request_line.as_bytes()).await?;
        stdin.flush().await?;

        // Read tool response
        response_line.clear();
        let read_result = timeout(self.config.timeout, reader.read_line(&mut response_line)).await;

        // Clean up process
        let _ = child.kill().await;

        match read_result {
            Ok(Ok(_)) => {
                let response: Value = serde_json::from_str(&response_line)?;

                if let Some(result) = response.get("result") {
                    Ok(McpToolResponse { success: true, content: result.clone(), error: None })
                } else if let Some(error) = response.get("error") {
                    Ok(McpToolResponse {
                        success: false,
                        content: json!(null),
                        error: Some(error.to_string()),
                    })
                } else {
                    Err(anyhow!("Invalid response from MCP server: {}", response))
                }
            }
            Ok(Err(e)) => Err(anyhow!("Failed to read from MCP server: {}", e)),
            Err(_) => Err(anyhow!("Timeout waiting for MCP tool response")),
        }
    }

    /// List all available MCP servers (synchronous)
    pub fn list_servers_sync(&self) -> Vec<&McpServer> {
        self.servers.values().collect()
    }

    /// Enable/disable an MCP server
    pub fn set_server_enabled(&mut self, server_name: &str, enabled: bool) -> Result<()> {
        let server = self
            .servers
            .get_mut(server_name)
            .ok_or_else(|| anyhow!("MCP server '{}' not found", server_name))?;

        server.enabled = enabled;
        info!("MCP server '{}' {}", server_name, if enabled { "enabled" } else { "disabled" });

        Ok(())
    }
    
    /// Connect to an MCP server and maintain persistent connection
    pub async fn connect_to_server(&mut self, server_name: &str) -> Result<()> {
        // Check if already connected
        if self.active_connections.contains_key(server_name) {
            info!("Already connected to MCP server '{}'", server_name);
            return Ok(());
        }
        
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| anyhow!("MCP server '{}' not found", server_name))?
            .clone();
            
        if !server.enabled {
            return Err(anyhow!("MCP server '{}' is disabled", server_name));
        }
        
        info!("Connecting to MCP server '{}'", server_name);
        
        // Start the MCP server process
        let mut cmd = Command::new(&server.command);
        cmd.args(&server.args);
        cmd.envs(&server.env);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        
        let mut child = cmd.spawn()?;
        
        // Get stdin and stdout handles
        let stdin = child.stdin.take()
            .ok_or_else(|| anyhow!("Failed to get stdin for MCP server"))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| anyhow!("Failed to get stdout for MCP server"))?;
        let reader = BufReader::new(stdout);
        
        // Create connection
        let mut connection = McpConnection {
            server_name: server_name.to_string(),
            process: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(reader)),
            initialized: false,
            connected_at: std::time::Instant::now(),
        };
        
        // Initialize the connection
        self.initialize_connection(&mut connection).await?;
        
        // Store active connection
        self.active_connections.insert(server_name.to_string(), connection);
        
        info!("Successfully connected to MCP server '{}'", server_name);
        
        Ok(())
    }
    
    /// Initialize an MCP connection
    async fn initialize_connection(&self, connection: &mut McpConnection) -> Result<()> {
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": true
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "Loki AI",
                    "version": "1.0.0"
                }
            }
        });
        
        let request_line = format!("{}\n", init_request);
        
        // Send initialization request
        {
            let mut stdin = connection.stdin.lock().await;
            stdin.write_all(request_line.as_bytes()).await?;
            stdin.flush().await?;
        }
        
        // Read initialization response
        let mut response_line = String::new();
        {
            let mut reader = connection.stdout.lock().await;
            timeout(self.config.timeout, reader.read_line(&mut response_line)).await??;
        }
        
        let response: Value = serde_json::from_str(&response_line)?;
        
        if response.get("result").is_some() {
            connection.initialized = true;
            Ok(())
        } else if let Some(error) = response.get("error") {
            Err(anyhow!("Failed to initialize MCP connection: {}", error))
        } else {
            Err(anyhow!("Invalid initialization response from MCP server"))
        }
    }
    
    /// Disconnect from an MCP server
    pub async fn disconnect_from_server(&mut self, server_name: &str) -> Result<()> {
        if let Some(connection) = self.active_connections.remove(server_name) {
            info!("Disconnecting from MCP server '{}'", server_name);
            
            // Kill the process
            let mut process = connection.process.lock().await;
            let _ = process.kill().await;
            
            info!("Disconnected from MCP server '{}'", server_name);
            Ok(())
        } else {
            Err(anyhow!("No active connection to MCP server '{}'", server_name))
        }
    }
    
    /// Check if connected to a server
    pub fn is_connected(&self, server_name: &str) -> bool {
        self.active_connections.contains_key(server_name)
    }
    
    /// Get connection uptime
    pub fn get_connection_uptime(&self, server_name: &str) -> Option<Duration> {
        self.active_connections.get(server_name)
            .map(|conn| conn.connected_at.elapsed())
    }
    
    /// Call tool using persistent connection
    pub async fn call_tool_persistent(
        &self,
        server_name: &str,
        tool_call: McpToolCall,
    ) -> Result<McpToolResponse> {
        let connection = self.active_connections.get(server_name)
            .ok_or_else(|| anyhow!("Not connected to MCP server '{}'", server_name))?;
            
        if !connection.initialized {
            return Err(anyhow!("MCP connection not initialized"));
        }
        
        debug!("Calling MCP tool '{}' on persistent connection '{}'", tool_call.name, server_name);
        
        // Create tool request
        let tool_request = json!({
            "jsonrpc": "2.0",
            "id": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            "method": "tools/call",
            "params": {
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }
        });
        
        let request_line = format!("{}\n", tool_request);
        
        // Send request
        {
            let mut stdin = connection.stdin.lock().await;
            stdin.write_all(request_line.as_bytes()).await?;
            stdin.flush().await?;
        }
        
        // Read response
        let mut response_line = String::new();
        let read_result = {
            let mut reader = connection.stdout.lock().await;
            timeout(self.config.timeout, reader.read_line(&mut response_line)).await
        };
        
        match read_result {
            Ok(Ok(_)) => {
                let response: Value = serde_json::from_str(&response_line)?;
                
                if let Some(result) = response.get("result") {
                    Ok(McpToolResponse {
                        success: true,
                        content: result.clone(),
                        error: None,
                    })
                } else if let Some(error) = response.get("error") {
                    Ok(McpToolResponse {
                        success: false,
                        content: json!(null),
                        error: Some(error.to_string()),
                    })
                } else {
                    Err(anyhow!("Invalid response from MCP server: {}", response))
                }
            }
            Ok(Err(e)) => Err(anyhow!("Failed to read from MCP server: {}", e)),
            Err(_) => Err(anyhow!("Timeout waiting for MCP tool response")),
        }
    }

    /// Web search using MCP Brave Search
    pub async fn web_search(&self, query: &str, num_results: Option<usize>) -> Result<Value> {
        let tool_call = McpToolCall {
            name: "brave_web_search".to_string(),
            arguments: json!({
                "query": query,
                "count": num_results.unwrap_or(10)
            }),
        };

        let response = self.call_tool("web-search", tool_call).await?;

        if response.success {
            Ok(response.content)
        } else {
            Err(anyhow!("Web search failed: {:?}", response.error))
        }
    }

    /// List directory contents using MCP filesystem
    pub async fn list_directory(&self, path: &str) -> Result<Value> {
        let tool_call = McpToolCall {
            name: "list_directory".to_string(),
            arguments: json!({
                "path": path
            }),
        };

        let response = self.call_tool("filesystem", tool_call).await?;

        if response.success {
            Ok(response.content)
        } else {
            Err(anyhow!("Directory listing failed: {:?}", response.error))
        }
    }

    /// Read file contents using MCP filesystem
    pub async fn read_file(&self, path: &str) -> Result<String> {
        let tool_call = McpToolCall {
            name: "read_file".to_string(),
            arguments: json!({
                "path": path
            }),
        };

        let response = self.call_tool("filesystem", tool_call).await?;

        if response.success {
            if let Some(content) = response.content.get("content").and_then(|v| v.as_str()) {
                Ok(content.to_string())
            } else {
                Err(anyhow!("Invalid file content response"))
            }
        } else {
            Err(anyhow!("File read failed: {:?}", response.error))
        }
    }

    /// Write file contents using MCP filesystem
    pub async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        let tool_call = McpToolCall {
            name: "write_file".to_string(),
            arguments: json!({
                "path": path,
                "content": content
            }),
        };

        let response = self.call_tool("filesystem", tool_call).await?;

        if response.success {
            Ok(())
        } else {
            Err(anyhow!("File write failed: {:?}", response.error))
        }
    }

    /// GitHub operations using MCP GitHub
    pub async fn github_operation(&self, operation: &str, args: Value) -> Result<Value> {
        let tool_call = McpToolCall { name: operation.to_string(), arguments: args };

        let response = self.call_tool("github", tool_call).await?;

        if response.success {
            Ok(response.content)
        } else {
            Err(anyhow!("GitHub operation failed: {:?}", response.error))
        }
    }

    /// Store data in MCP memory
    pub async fn store_memory(&self, key: &str, value: Value) -> Result<()> {
        let tool_call = McpToolCall {
            name: "store".to_string(),
            arguments: json!({
                "key": key,
                "value": value
            }),
        };

        let response = self.call_tool("memory", tool_call).await?;

        if response.success {
            Ok(())
        } else {
            Err(anyhow!("Memory store failed: {:?}", response.error))
        }
    }

    /// Retrieve data from MCP memory
    pub async fn retrieve_memory(&self, key: &str) -> Result<Option<Value>> {
        let tool_call = McpToolCall {
            name: "retrieve".to_string(),
            arguments: json!({
                "key": key
            }),
        };

        let response = self.call_tool("memory", tool_call).await?;

        if response.success {
            Ok(Some(response.content))
        } else if response.error.as_deref() == Some("Key not found") {
            Ok(None)
        } else {
            Err(anyhow!("Memory retrieve failed: {:?}", response.error))
        }
    }

    /// Fetch web content using MCP fetch
    pub async fn fetch_url(&self, url: &str) -> Result<String> {
        let tool_call = McpToolCall {
            name: "fetch".to_string(),
            arguments: json!({
                "url": url
            }),
        };

        let response = self.call_tool("fetch", tool_call).await?;

        if response.success {
            if let Some(content) = response.content.get("content").and_then(|v| v.as_str()) {
                Ok(content.to_string())
            } else {
                Err(anyhow!("Invalid fetch response"))
            }
        } else {
            Err(anyhow!("URL fetch failed: {:?}", response.error))
        }
    }

    /// Use Puppeteer for web automation
    pub async fn puppeteer_action(&self, action: &str, args: Value) -> Result<Value> {
        let tool_call = McpToolCall { name: action.to_string(), arguments: args };

        let response = self.call_tool("puppeteer", tool_call).await?;

        if response.success {
            Ok(response.content)
        } else {
            Err(anyhow!("Puppeteer action failed: {:?}", response.error))
        }
    }

    /// Check if a server is healthy and reachable
    pub async fn check_server_health(&self, server_name: &str) -> Result<bool> {
        let server = self.servers.get(server_name)
            .ok_or_else(|| anyhow!("Server '{}' not found", server_name))?;
        
        if !server.enabled {
            return Ok(false);
        }

        // Try to get server capabilities as a health check
        match self.get_capabilities(server_name).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get server statistics including uptime and last activity
    pub async fn get_server_statistics(&self, server_name: &str) -> Result<(Duration, chrono::DateTime<chrono::Utc>)> {
        // For now, return placeholder values
        // In a real implementation, this would track server start time and last request time
        let uptime = Duration::from_secs(3600); // 1 hour placeholder
        let last_active = chrono::Utc::now();
        Ok((uptime, last_active))
    }
}

#[cfg(test)]
mod tests {
    use tokio;

    use super::*;

    #[tokio::test]
    async fn test_mcp_client_creation() {
        let config = McpClientConfig::default();
        let client = McpClient::new(config);

        assert_eq!(client.servers.len(), 0);
    }

    #[tokio::test]
    async fn test_server_management() {
        let mut client = McpClient::new(McpClientConfig::default());

        let server = McpServer {
            name: "test-server".to_string(),
            description: "Test MCP server".to_string(),
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "@modelcontextprotocol/server-memory".to_string()],
            env: HashMap::new(),
            capabilities: Vec::new(),
            enabled: true,
        };

        client.add_server(server);
        assert_eq!(client.servers.len(), 1);

        // Test enable/disable
        client.set_server_enabled("test-server", false).unwrap();
        assert!(!client.servers.get("test-server").unwrap().enabled);
    }
}
