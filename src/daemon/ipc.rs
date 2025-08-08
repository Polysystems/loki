use std::path::PathBuf;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;
use tokio_util::codec::{FramedRead, FramedWrite, LinesCodec};
use uuid::Uuid;

use super::ProcessStatus;

/// IPC message for daemon communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcMessage {
    pub id: String,
    pub command: DaemonCommand,
    pub response: Option<DaemonResponse>,
}

/// Commands that can be sent to the daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonCommand {
    /// Get daemon status
    Status,
    /// Stop the daemon
    Stop,
    /// Send a query to the cognitive system
    Query { query: String },
    /// List active streams
    ListStreams,
    /// Get system metrics
    GetMetrics,
}

/// Responses from the daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonResponse {
    /// Status response
    Status { status: ProcessStatus },
    /// Success response
    Success { message: String },
    /// Error response
    Error { message: String },
    /// Query result
    QueryResult { result: String },
    /// List of streams
    StreamList { streams: Vec<String> },
    /// System metrics
    Metrics { metrics: serde_json::Value },
}

/// Client for communicating with the daemon
#[derive(Debug, Clone)]
pub struct DaemonClient {
    socket_path: PathBuf,
}

impl DaemonClient {
    /// Create a new daemon client
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    /// Connect to the daemon and send a command
    pub async fn send_command(&self, command: DaemonCommand) -> Result<DaemonResponse> {
        // Connect to the daemon socket
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .context("Failed to connect to daemon socket")?;

        let (reader, writer) = stream.into_split();
        let mut framed_read = FramedRead::new(reader, LinesCodec::new());
        let mut framed_write = FramedWrite::new(writer, LinesCodec::new());

        // Create message
        let message = IpcMessage { id: Uuid::new_v4().to_string(), command, response: None };

        // Send command
        let message_json =
            serde_json::to_string(&message).context("Failed to serialize command")?;

        framed_write.send(message_json).await.context("Failed to send command")?;

        // Wait for response
        if let Some(response_line) = framed_read.next().await {
            let response_line = response_line.context("Failed to read response")?;

            let response_message: IpcMessage =
                serde_json::from_str(&response_line).context("Failed to parse response")?;

            response_message.response.context("No response in message")
        } else {
            anyhow::bail!("No response received from daemon")
        }
    }

    /// Check if the daemon is running and responsive
    pub async fn is_daemon_responsive(&self) -> bool {
        match self.send_command(DaemonCommand::Status).await {
            Ok(DaemonResponse::Status { .. }) => true,
            _ => false,
        }
    }

    /// Get daemon status
    pub async fn get_status(&self) -> Result<ProcessStatus> {
        match self.send_command(DaemonCommand::Status).await? {
            DaemonResponse::Status { status } => Ok(status),
            DaemonResponse::Error { message } => anyhow::bail!("Error: {}", message),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    /// Stop the daemon
    pub async fn stop_daemon(&self) -> Result<String> {
        match self.send_command(DaemonCommand::Stop).await? {
            DaemonResponse::Success { message } => Ok(message),
            DaemonResponse::Error { message } => anyhow::bail!("Error: {}", message),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    /// Send a query to the cognitive system
    pub async fn query(&self, query: String) -> Result<String> {
        match self.send_command(DaemonCommand::Query { query }).await? {
            DaemonResponse::QueryResult { result } => Ok(result),
            DaemonResponse::Error { message } => anyhow::bail!("Error: {}", message),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    /// List active streams
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        match self.send_command(DaemonCommand::ListStreams).await? {
            DaemonResponse::StreamList { streams } => Ok(streams),
            DaemonResponse::Error { message } => anyhow::bail!("Error: {}", message),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    /// Get system metrics
    pub async fn get_metrics(&self) -> Result<serde_json::Value> {
        match self.send_command(DaemonCommand::GetMetrics).await? {
            DaemonResponse::Metrics { metrics } => Ok(metrics),
            DaemonResponse::Error { message } => anyhow::bail!("Error: {}", message),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }
}

/// Batch operations for multiple commands
impl DaemonClient {
    /// Send multiple commands concurrently
    pub async fn send_commands_concurrent(
        &self,
        commands: Vec<DaemonCommand>,
    ) -> Result<Vec<DaemonResponse>> {
        let futures = commands.into_iter().map(|cmd| self.send_command(cmd));

        let results = futures::future::join_all(futures).await;

        let mut responses = Vec::new();
        for result in results {
            responses.push(result?);
        }

        Ok(responses)
    }

    /// Monitor daemon status continuously
    pub async fn monitor_status<F>(&self, mut callback: F, interval_secs: u64) -> Result<()>
    where
        F: FnMut(ProcessStatus) -> bool + Send,
    {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            match self.get_status().await {
                Ok(status) => {
                    // Call callback, if it returns false, stop monitoring
                    if !callback(status) {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Failed to get daemon status: {}", e);
                    // Continue monitoring even on errors
                }
            }
        }

        Ok(())
    }
}
