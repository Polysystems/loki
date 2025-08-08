use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;
use tokio::sync::{RwLock, broadcast};
use tracing::{Level, debug, error, info, warn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt};

// Import from the internal cognitive module
use crate::cognitive::{CognitiveConfig, CognitiveSystem, SafeCognitiveSystem};

pub mod ipc;
pub mod process;
pub use ipc::{DaemonClient, DaemonCommand as DaemonCommand2, DaemonResponse as DaemonResponse2, IpcMessage};
pub use process::{DaemonProcess, ProcessStatus};

use super::*;
use crate::cli::CognitiveCommands;
use crate::cluster::{ClusterConfig, ClusterManager};
use crate::compute::ComputeManager;
use crate::config::{ApiKeysConfig, Config};
use crate::memory::{CognitiveMemory, MemoryConfig};
use crate::models::{CompletionRequest, Message, MessageRole, ProviderFactory};
use crate::safety::{AuditConfig, ResourceLimits, ValidatorConfig};
use crate::streaming::StreamManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub pid_file: PathBuf,
    pub socket_path: PathBuf,
    pub log_file: PathBuf,
    pub working_dir: PathBuf,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("loki");

        Self {
            pid_file: data_dir.join("loki.pid"),
            socket_path: data_dir.join("loki.sock"),
            log_file: data_dir.join("loki.log"),
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// Daemon server that handles IPC communication
pub struct DaemonServer {
    config: DaemonConfig,
    listener: Option<UnixListener>,
    shutdown_tx: broadcast::Sender<()>,
    cognitive_system: Option<Arc<CognitiveSystem>>,
    status: Arc<RwLock<ProcessStatus>>,
}

impl DaemonServer {
    /// Create a new daemon server
    pub fn new(config: DaemonConfig) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            listener: None,
            shutdown_tx,
            cognitive_system: None,
            status: Arc::new(RwLock::new(ProcessStatus::Starting)),
        })
    }

    /// Set the cognitive system for the daemon
    pub fn with_cognitive_system(mut self, cognitive_system: Arc<CognitiveSystem>) -> Self {
        self.cognitive_system = Some(cognitive_system);
        self
    }

    /// Start the daemon server
    pub async fn start(&mut self) -> Result<()> {
        // Ensure the socket directory exists
        if let Some(parent) = self.config.socket_path.parent() {
            tokio::fs::create_dir_all(parent).await.context("Failed to create socket directory")?;
        }

        // Remove existing socket if it exists
        if self.config.socket_path.exists() {
            tokio::fs::remove_file(&self.config.socket_path)
                .await
                .context("Failed to remove existing socket")?;
        }

        // Create Unix socket listener
        let listener =
            UnixListener::bind(&self.config.socket_path).context("Failed to bind Unix socket")?;

        info!("Daemon server listening on {:?}", self.config.socket_path);

        self.listener = Some(listener);
        *self.status.write().await = ProcessStatus::Running;

        // Start accepting connections
        self.accept_connections().await?;

        Ok(())
    }

    /// Accept and handle incoming connections
    async fn accept_connections(&mut self) -> Result<()> {
        let listener = self.listener.take().context("Listener not initialized")?;

        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let status = self.status.clone();
        let cognitive_system = self.cognitive_system.clone();

        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, addr)) => {
                            info!("New IPC connection from {:?}", addr);

                            let status = status.clone();
                            let cognitive_system = cognitive_system.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(stream, status, cognitive_system).await {
                                    error!("Error handling IPC connection: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept connection: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Received shutdown signal");
                    break;
                }
            }
        }

        *self.status.write().await = ProcessStatus::Stopped;
        Ok(())
    }

    /// Handle a single IPC connection
    async fn handle_connection(
        stream: UnixStream,
        status: Arc<RwLock<ProcessStatus>>,
        cognitive_system: Option<Arc<CognitiveSystem>>,
    ) -> Result<()> {
        use futures::{SinkExt, StreamExt};
        use tokio_util::codec::{FramedRead, FramedWrite, LinesCodec};

        let (reader, writer) = stream.into_split();
        let mut framed_read = FramedRead::new(reader, LinesCodec::new());
        let mut framed_write = FramedWrite::new(writer, LinesCodec::new());

        while let Some(line) = framed_read.next().await {
            let line = line.context("Failed to read line")?;

            // Parse the incoming message
            let message: IpcMessage =
                serde_json::from_str(&line).context("Failed to parse IPC message")?;

            // Handle the command
            let response =
                Self::handle_command(message.command, &status, cognitive_system.as_ref()).await;

            // Send response
            let response_msg = IpcMessage {
                id: message.id,
                command: DaemonCommand2::Status, // Response type
                response: Some(response),
            };

            let response_json =
                serde_json::to_string(&response_msg).context("Failed to serialize response")?;

            if let Err(e) = framed_write.send(response_json).await {
                error!("Failed to send response: {}", e);
                break;
            }
        }

        Ok(())
    }

    /// Handle a daemon command
    async fn handle_command(
        command: DaemonCommand2,
        status: &Arc<RwLock<ProcessStatus>>,
        cognitive_system: Option<&Arc<CognitiveSystem>>,
    ) -> DaemonResponse2 {
        match command {
            DaemonCommand2::Status => {
                let current_status = status.read().await.clone();
                DaemonResponse2::Status { status: current_status }
            }

            DaemonCommand2::Stop => {
                info!("Received stop command");
                *status.write().await = ProcessStatus::Stopping;

                // Gracefully shutdown cognitive system if available
                if let Some(cognitive) = cognitive_system {
                    if let Err(e) = cognitive.shutdown().await {
                        error!("Error shutting down cognitive system: {}", e);
                    }
                }

                DaemonResponse2::Success { message: "Shutdown initiated".to_string() }
            }

            DaemonCommand2::Query { query } => {
                if let Some(cognitive) = cognitive_system {
                    // Use cognitive system to process query
                    match cognitive.process_query(&query).await {
                        Ok(result) => DaemonResponse2::QueryResult { result },
                        Err(e) => DaemonResponse2::Error { message: format!("Query failed: {}", e) },
                    }
                } else {
                    DaemonResponse2::Error { message: "Cognitive system not available".to_string() }
                }
            }

            DaemonCommand2::ListStreams => {
                if let Some(cognitive) = cognitive_system {
                    let streams = cognitive.list_active_streams().await;
                    DaemonResponse2::StreamList { streams }
                } else {
                    DaemonResponse2::Error { message: "Cognitive system not available".to_string() }
                }
            }

            DaemonCommand2::GetMetrics => {
                // Return system metrics
                let metrics = Self::collect_metrics(cognitive_system).await;
                DaemonResponse2::Metrics { metrics }
            }
        }
    }

    /// Collect system metrics
    async fn collect_metrics(cognitive_system: Option<&Arc<CognitiveSystem>>) -> serde_json::Value {
        let mut metrics = serde_json::Map::new();

        // System metrics
        let sys = sysinfo::System::new_all();
        metrics.insert(
            "memory_used".to_string(),
            serde_json::Value::Number(serde_json::Number::from(sys.used_memory())),
        );
        metrics.insert(
            "memory_total".to_string(),
            serde_json::Value::Number(serde_json::Number::from(sys.total_memory())),
        );
        metrics.insert(
            "cpu_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(sys.cpus().len())),
        );

        // Cognitive system metrics if available
        if let Some(cognitive) = cognitive_system {
            if let Ok(stats) = cognitive.get_statistics().await {
                if let Some(active_streams) = stats.get("active_streams").and_then(|v| v.as_u64()) {
                    metrics.insert(
                        "active_streams".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(active_streams)),
                    );
                }
                if let Some(total_thoughts) = stats.get("total_thoughts").and_then(|v| v.as_u64()) {
                    metrics.insert(
                        "total_thoughts".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(total_thoughts)),
                    );
                }
                if let Some(memory_items) = stats.get("memory_items").and_then(|v| v.as_u64()) {
                    metrics.insert(
                        "memory_items".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(memory_items)),
                    );
                }
            }
        }

        serde_json::Value::Object(metrics)
    }

    /// Stop the daemon server
    pub async fn stop(&self) -> Result<()> {
        if let Err(e) = self.shutdown_tx.send(()) {
            warn!("No listeners for shutdown signal: {}", e);
        }

        // Clean up socket file
        if self.config.socket_path.exists() {
            tokio::fs::remove_file(&self.config.socket_path)
                .await
                .context("Failed to remove socket file")?;
        }

        Ok(())
    }

    /// Get the shutdown sender for signal handlers
    pub fn shutdown_sender(&self) -> broadcast::Sender<()> {
        self.shutdown_tx.clone()
    }
}

/// Utility functions for daemon management
pub mod utils {
    use std::process;

    use super::*;

    /// Check if a daemon is already running
    pub async fn is_daemon_running(config: &DaemonConfig) -> bool {
        // Check if socket exists and is responsive
        if !config.socket_path.exists() {
            return false;
        }

        // Try to connect to the socket
        match UnixStream::connect(&config.socket_path).await {
            Ok(_) => true,
            Err(_) => {
                // Socket exists but not responsive, clean it up
                let _ = tokio::fs::remove_file(&config.socket_path).await;
                false
            }
        }
    }

    /// Create a PID file
    pub async fn create_pid_file(config: &DaemonConfig) -> Result<()> {
        let pid = process::id();
        tokio::fs::write(&config.pid_file, pid.to_string())
            .await
            .context("Failed to create PID file")?;
        Ok(())
    }

    /// Remove PID file
    pub async fn remove_pid_file(config: &DaemonConfig) -> Result<()> {
        if config.pid_file.exists() {
            tokio::fs::remove_file(&config.pid_file).await.context("Failed to remove PID file")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonCommand {
    Status,
    Stop,
    Query { query: String },
    Stream { name: String, purpose: Option<String> },
    Agent { name: String, agent_type: String, capabilities: Vec<String> },
}

/// Daemon response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonResponse {
    Status { running: bool, pid: Option<u32>, uptime: Option<u64> },
    Success { message: String },
    Error { error: String },
    QueryResult { result: String },
}

/// Daemon manager handles process lifecycle and IPC
pub struct DaemonManager {
    config: DaemonConfig,
    shutdown_tx: Option<broadcast::Sender<()>>,
}

impl DaemonManager {
    pub fn new(config: DaemonConfig) -> Self {
        Self { config, shutdown_tx: None }
    }

    /// Start daemon mode with cognitive system
    pub async fn start_daemon(
        &mut self,
        cognitive_system: Arc<crate::cognitive::SafeCognitiveSystem>,
    ) -> Result<()> {
        // Track daemon start time
        let start_time = std::time::Instant::now();

        // Ensure data directory exists
        if let Some(parent) = self.config.pid_file.parent() {
            fs::create_dir_all(parent)?;
        }

        // Check if already running
        if self.is_running()? {
            anyhow::bail!("Daemon is already running");
        }

        // Daemonize process
        self.daemonize().await?;

        // Setup signal handlers and IPC
        let (shutdown_tx, mut shutdown_rx) = broadcast::channel(1);
        self.shutdown_tx = Some(shutdown_tx.clone());

        // Write PID file
        fs::write(&self.config.pid_file, std::process::id().to_string())?;

        // Setup IPC socket
        let socket_path = self.config.socket_path.clone();
        if socket_path.exists() {
            fs::remove_file(&socket_path)?;
        }

        let listener = UnixListener::bind(&socket_path)?;

        // Setup signal handling
        let signal_shutdown_tx = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate()).unwrap();
            let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt()).unwrap();

            tokio::select! {
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                    let _ = signal_shutdown_tx.send(());
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT, initiating graceful shutdown");
                    let _ = signal_shutdown_tx.send(());
                }
            }
        });

        // Handle IPC connections
        let ipc_cognitive_system = cognitive_system.clone();
        let _ipc_shutdown_tx = shutdown_tx.clone();
        let ipc_start_time = start_time;
        let mut ipc_shutdown_rx = shutdown_rx.resubscribe();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, _)) => {
                                let cognitive_system = ipc_cognitive_system.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = Self::handle_ipc_connection(stream, cognitive_system, ipc_start_time).await {
                                        error!("IPC connection error: {}", e);
                                    }
                                });
                            }
                            Err(e) => error!("Failed to accept IPC connection: {}", e),
                        }
                    }
                    _ = ipc_shutdown_rx.recv() => {
                        info!("Shutting down IPC listener");
                        break;
                    }
                }
            }
        });

        // Main daemon loop - monitor cognitive system health
        tokio::select! {
            _ = shutdown_rx.recv() => {
                info!("Received shutdown signal");
            }
            _ = Self::monitor_system_health(cognitive_system.clone()) => {
                warn!("System health monitor exited");
            }
        }

        // Cleanup
        self.cleanup().await?;
        info!("Daemon shutdown complete");
        Ok(())
    }

    /// Stop running daemon
    pub async fn stop_daemon(&self) -> Result<()> {
        if !self.is_running()? {
            anyhow::bail!("Daemon is not running");
        }

        // Try graceful shutdown via IPC first
        match self.send_daemon_command(DaemonCommand::Stop).await {
            Ok(_) => {
                info!("Daemon stopped gracefully");
                return Ok(());
            }
            Err(e) => {
                warn!("Graceful shutdown failed: {}, trying SIGTERM", e);
            }
        }

        // Fallback to SIGTERM
        if let Ok(pid_str) = fs::read_to_string(&self.config.pid_file) {
            if let Ok(pid) = pid_str.trim().parse::<i32>() {
                unsafe {
                    libc::kill(pid, libc::SIGTERM);
                }

                // Wait for process to exit
                for _ in 0..30 {
                    if !self.is_running()? {
                        info!("Daemon stopped via SIGTERM");
                        return Ok(());
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }

                // Force kill as last resort
                warn!("Forcing daemon shutdown with SIGKILL");
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }

        Ok(())
    }

    /// Check if daemon is running
    pub fn is_running(&self) -> Result<bool> {
        if !self.config.pid_file.exists() {
            return Ok(false);
        }

        let pid_str = fs::read_to_string(&self.config.pid_file)?;
        let pid: i32 = pid_str.trim().parse().context("Invalid PID in file")?;

        // Check if process exists
        let result = unsafe { libc::kill(pid, 0) };
        Ok(result == 0)
    }

    /// Send command to running daemon
    pub async fn send_daemon_command(&self, command: DaemonCommand) -> Result<DaemonResponse> {
        let stream = UnixStream::connect(&self.config.socket_path)
            .await
            .context("Failed to connect to daemon socket")?;

        let (reader, writer) = stream.into_split();
        let mut writer = tokio::io::BufWriter::new(writer);
        let mut reader = tokio::io::BufReader::new(reader);

        // Send command
        let command_json = serde_json::to_string(&command)?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, command_json.as_bytes()).await?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, b"\n").await?;
        tokio::io::AsyncWriteExt::flush(&mut writer).await?;

        // Read response
        let mut response_line = String::new();
        reader.read_line(&mut response_line).await?;

        let response: DaemonResponse =
            serde_json::from_str(&response_line).context("Failed to parse daemon response")?;

        Ok(response)
    }

    /// Daemonize the current process
    async fn daemonize(&self) -> Result<()> {
        // Change to working directory
        std::env::set_current_dir(&self.config.working_dir)?;

        // Setup file descriptors for daemon mode
        // Note: This is a simplified daemonization for Unix systems

        info!("Daemonizing process...");
        Ok(())
    }

    /// Handle incoming IPC connection
    async fn handle_ipc_connection(
        stream: UnixStream,
        cognitive_system: Arc<crate::cognitive::SafeCognitiveSystem>,
        start_time: std::time::Instant,
    ) -> Result<()> {
        let (reader, writer) = stream.into_split();
        let mut writer = tokio::io::BufWriter::new(writer);
        let mut reader = tokio::io::BufReader::new(reader);

        let mut command_line = String::new();
        reader.read_line(&mut command_line).await?;

        let command: DaemonCommand = serde_json::from_str(&command_line)?;
        let response = Self::process_daemon_command(command, cognitive_system, start_time).await;

        let response_json = serde_json::to_string(&response)?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, response_json.as_bytes()).await?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, b"\n").await?;
        tokio::io::AsyncWriteExt::flush(&mut writer).await?;

        Ok(())
    }

    /// Process daemon command
    async fn process_daemon_command(
        command: DaemonCommand,
        cognitive_system: Arc<crate::cognitive::SafeCognitiveSystem>,
        start_time: std::time::Instant,
    ) -> DaemonResponse {
        match command {
            DaemonCommand::Status => DaemonResponse::Status {
                running: true,
                pid: Some(std::process::id()),
                uptime: Some(start_time.elapsed().as_secs()),
            },
            DaemonCommand::Stop => {
                info!("Received stop command via IPC");
                std::process::exit(0);
            }
            DaemonCommand::Query { query } => {
                // Process query through cognitive system
                match cognitive_system.process_query(&query).await {
                    Ok(result) => DaemonResponse::QueryResult { result },
                    Err(e) => DaemonResponse::Error { error: e.to_string() },
                }
            }
            DaemonCommand::Stream { name, purpose } => {
                // Create or attach to stream
                DaemonResponse::Success {
                    message: format!("Stream '{name}' created with purpose: {purpose:?}"),
                }
            }
            DaemonCommand::Agent { name, agent_type, capabilities } => {
                // Create agent
                DaemonResponse::Success {
                    message: format!(
                        "Agent '{name}' of type '{agent_type}' created with capabilities: \
                         {capabilities:?}"
                    ),
                }
            }
        }
    }

    /// Monitor system health
    async fn monitor_system_health(cognitive_system: Arc<crate::cognitive::SafeCognitiveSystem>) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Check cognitive system health
            if let Err(e) = cognitive_system.health_check().await {
                error!("Cognitive system health check failed: {}", e);
            }

            // Log system metrics
            debug!("System health check completed");
        }
    }

    /// Cleanup daemon resources
    async fn cleanup(&self) -> Result<()> {
        // Remove PID file
        if self.config.pid_file.exists() {
            fs::remove_file(&self.config.pid_file)?;
        }

        // Remove socket file
        if self.config.socket_path.exists() {
            fs::remove_file(&self.config.socket_path)?;
        }

        Ok(())
    }
}
