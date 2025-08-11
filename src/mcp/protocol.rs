//! MCP Protocol Implementation

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "jsonrpc")]
pub enum McpMessage {
    #[serde(rename = "2.0")]
    JsonRpc(JsonRpcMessage),
}

/// JSON-RPC 2.0 message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    Request(McpRequest),
    Response(McpResponse),
    Notification(McpNotification),
}

/// MCP request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    /// Request ID
    pub id: u64,
    
    /// Method name
    pub method: String,
    
    /// Method parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// MCP response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    /// Request ID this responds to
    pub id: u64,
    
    /// Result data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    
    /// Error information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

/// MCP notification message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpNotification {
    /// Method name
    pub method: String,
    
    /// Notification parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// MCP error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    /// Error code
    pub code: i32,
    
    /// Error message
    pub message: String,
    
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Standard MCP methods
pub mod methods {
    /// Initialize connection
    pub const INITIALIZE: &str = "initialize";
    
    /// List available tools
    pub const LIST_TOOLS: &str = "tools/list";
    
    /// Call a tool
    pub const CALL_TOOL: &str = "tools/call";
    
    /// List resources
    pub const LIST_RESOURCES: &str = "resources/list";
    
    /// Read resource
    pub const READ_RESOURCE: &str = "resources/read";
    
    /// List prompts
    pub const LIST_PROMPTS: &str = "prompts/list";
    
    /// Get prompt
    pub const GET_PROMPT: &str = "prompts/get";
}

/// MCP capability flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCapabilities {
    /// Supports tools
    #[serde(default)]
    pub tools: bool,
    
    /// Supports resources
    #[serde(default)]
    pub resources: bool,
    
    /// Supports prompts
    #[serde(default)]
    pub prompts: bool,
    
    /// Supports sampling
    #[serde(default)]
    pub sampling: bool,
}