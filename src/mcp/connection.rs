//! MCP Connection Management

use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::Mutex;
use anyhow::Result;

pub use super::server::ConnectionStatus;

/// Active MCP server connection
#[derive(Debug)]
pub struct McpConnection {
    /// Server ID
    pub server_id: String,
    
    /// Process handle
    pub process: Arc<Mutex<Child>>,
    
    /// Connection status
    pub status: ConnectionStatus,
    
    /// Message counter for request IDs
    pub message_id: Arc<Mutex<u64>>,
}

impl McpConnection {
    /// Create a new MCP connection
    pub fn new(server_id: String, process: Child) -> Self {
        Self {
            server_id,
            process: Arc::new(Mutex::new(process)),
            status: ConnectionStatus::Connecting,
            message_id: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Get next message ID
    pub async fn next_message_id(&self) -> u64 {
        let mut id = self.message_id.lock().await;
        *id += 1;
        *id
    }
    
    /// Check if connection is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, ConnectionStatus::Active)
    }
    
    /// Terminate the connection
    pub async fn terminate(&self) -> Result<()> {
        let mut process = self.process.lock().await;
        process.kill().await?;
        Ok(())
    }
}