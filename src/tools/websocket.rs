//! WebSocket Streaming Client
//!
//! Advanced WebSocket client with auto-reconnection, message routing,
//! streaming support, and integration with Loki's safety and memory systems.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info};

use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::safety::{ActionType, ActionValidator};

/// WebSocket connection configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    pub url: String,
    pub protocols: Vec<String>,
    pub headers: HashMap<String, String>,
    pub connect_timeout: Duration,
    pub ping_interval: Duration,
    pub reconnect_attempts: usize,
    pub reconnect_delay: Duration,
    pub max_message_size: usize,
    pub max_frame_size: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            protocols: Vec::new(),
            headers: HashMap::new(),
            connect_timeout: Duration::from_secs(10),
            ping_interval: Duration::from_secs(30),
            reconnect_attempts: 5,
            reconnect_delay: Duration::from_secs(5),
            max_message_size: 16 * 1024 * 1024, // 16MB
            max_frame_size: 16 * 1024 * 1024,   // 16MB
        }
    }
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    Text { content: String },
    Binary { data: Vec<u8> },
    Json { value: serde_json::Value },
    Ping { data: Vec<u8> },
    Pong { data: Vec<u8> },
    Close { code: Option<u16>, reason: String },
}

impl From<Message> for WebSocketMessage {
    fn from(msg: Message) -> Self {
        match msg {
            Message::Text(text) => {
                // Try to parse as JSON first
                if let Ok(value) = serde_json::from_str(&text) {
                    WebSocketMessage::Json { value }
                } else {
                    WebSocketMessage::Text { content: text }
                }
            }
            Message::Binary(data) => WebSocketMessage::Binary { data },
            Message::Ping(data) => WebSocketMessage::Ping { data },
            Message::Pong(data) => WebSocketMessage::Pong { data },
            Message::Close(frame) => WebSocketMessage::Close {
                code: frame.as_ref().map(|f| f.code.into()),
                reason: frame.map(|f| f.reason.to_string()).unwrap_or_default(),
            },
            Message::Frame(_) => WebSocketMessage::Text { content: "Raw frame".to_string() },
        }
    }
}

impl Into<Message> for WebSocketMessage {
    fn into(self) -> Message {
        match self {
            WebSocketMessage::Text { content } => Message::Text(content),
            WebSocketMessage::Binary { data } => Message::Binary(data),
            WebSocketMessage::Json { value } => {
                Message::Text(serde_json::to_string(&value).unwrap_or_default())
            }
            WebSocketMessage::Ping { data } => Message::Ping(data),
            WebSocketMessage::Pong { data } => Message::Pong(data),
            WebSocketMessage::Close { code, reason } => {
                Message::Close(Some(tokio_tungstenite::tungstenite::protocol::CloseFrame {
                    code: code.unwrap_or(1000).into(),
                    reason: reason.into(),
                }))
            }
        }
    }
}

/// Connection status
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
}

/// Message handler trait
#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    async fn handle_message(&self, message: WebSocketMessage) -> Result<()>;
    async fn on_connect(&self) -> Result<()> {
        Ok(())
    }
    async fn on_disconnect(&self) -> Result<()> {
        Ok(())
    }
    async fn on_error(&self, error: &anyhow::Error) -> Result<()> {
        error!("WebSocket error: {}", error);
        Ok(())
    }
}

/// Statistics for WebSocket connection
#[derive(Debug, Clone, Default)]
pub struct WebSocketStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connection_count: u64,
    pub reconnection_count: u64,
    pub last_message_time: Option<std::time::SystemTime>,
}

/// Advanced WebSocket client with streaming and safety integration
pub struct WebSocketClient {
    /// Configuration
    config: WebSocketConfig,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Safety validator
    validator: Option<Arc<ActionValidator>>,

    /// Message handlers
    handlers: Arc<RwLock<Vec<Arc<dyn MessageHandler>>>>,

    /// Outbound message queue
    outbound_tx: mpsc::UnboundedSender<WebSocketMessage>,
    outbound_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<WebSocketMessage>>>>,

    /// Connection status
    status: Arc<RwLock<ConnectionStatus>>,

    /// Statistics
    stats: Arc<RwLock<WebSocketStats>>,

    /// Connection control
    shutdown_tx: broadcast::Sender<()>,

    /// Status updates broadcast
    status_tx: broadcast::Sender<ConnectionStatus>,
}

impl WebSocketClient {
    /// Create a new WebSocket client
    pub async fn new(
        config: WebSocketConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Self> {
        info!("Initializing WebSocket client for: {}", config.url);

        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, _) = broadcast::channel(1);
        let (status_tx, _) = broadcast::channel(100);

        Ok(Self {
            config,
            memory,
            validator,
            handlers: Arc::new(RwLock::new(Vec::new())),
            outbound_tx,
            outbound_rx: Arc::new(RwLock::new(Some(outbound_rx))),
            status: Arc::new(RwLock::new(ConnectionStatus::Disconnected)),
            stats: Arc::new(RwLock::new(WebSocketStats::default())),
            shutdown_tx,
            status_tx,
        })
    }

    /// Add a message handler
    pub async fn add_handler(&self, handler: Arc<dyn MessageHandler>) {
        let mut handlers = self.handlers.write().await;
        handlers.push(handler);
    }

    /// Connect to the WebSocket server
    pub async fn connect(&self) -> Result<()> {
        // Validate the connection through safety system
        if let Some(validator) = &self.validator {
            validator
                .validate_action(
                    ActionType::ApiCall {
                        provider: "websocket".to_string(),
                        endpoint: self.config.url.clone(),
                    },
                    "WebSocket connection".to_string(),
                    vec!["Establishing WebSocket connection".to_string()],
                )
                .await?;
        }

        self.set_status(ConnectionStatus::Connecting).await;

        debug!("Connecting to WebSocket: {}", self.config.url);

        // Connect with timeout (headers and protocols will be added in a future
        // version)
        let (ws_stream, _) =
            tokio::time::timeout(self.config.connect_timeout, connect_async(&self.config.url))
                .await??;

        info!("WebSocket connected successfully");
        self.set_status(ConnectionStatus::Connected).await;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.connection_count += 1;
        }

        // Split the stream
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        // Start the message handling loops
        let handlers = self.handlers.clone();
        let memory = self.memory.clone();
        let stats = self.stats.clone();
        let status_sender = self.status_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // Inbound message handler
        let inbound_handlers = handlers.clone();
        let inbound_memory = memory.clone();
        let inbound_stats = stats.clone();
        let inbound_status_sender = status_sender.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                        msg_result = ws_receiver.next() => {
                            match msg_result {
                                Some(Ok(msg)) => {
                                    let ws_msg = WebSocketMessage::from(msg);

                                    // Update stats
                                    {
                                        let mut stats = inbound_stats.write().await;
                                        stats.messages_received += 1;
                                        stats.last_message_time = Some(std::time::SystemTime::now());

                                        // Estimate bytes received
                                        match &ws_msg {
                                            WebSocketMessage::Text { content } => {
                                                stats.bytes_received += content.len() as u64;
                                            }
                                            WebSocketMessage::Binary { data } => {
                                                stats.bytes_received += data.len() as u64;
                                            }
                                            WebSocketMessage::Json { value } => {
                                                stats.bytes_received += value.to_string().len() as u64;
                                            }
                                            _ => {}
                                        }
                                    }

                                    // Handle the message
                                    let handlers = inbound_handlers.read().await;
                                    for handler in handlers.iter() {
                                        if let Err(e) = handler.handle_message(ws_msg.clone()).await {
                                            error!("Handler error: {}", e);
                                            let _ = handler.on_error(&e).await;
                                        }
                                    }

                                    // Store interesting messages in memory
                                    if let WebSocketMessage::Json { value } = &ws_msg {
                                        if let Err(e) = inbound_memory.store(
                                            format!("WebSocket message: {}", value.to_string().chars().take(200).collect::<String>()),
                                            vec![],
                                            MemoryMetadata {
                                                source: "websocket".to_string(),
                                                tags: vec!["realtime".to_string(), "streaming".to_string()],
                                                importance: 0.5,
                                                associations: vec![],

                    context: Some("Generated from automated fix".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "tool_usage".to_string(),
                                                timestamp: chrono::Utc::now(),
                                                expiration: None,
                },
                                        ).await {
                                            error!("Failed to store WebSocket message: {}", e);
                                        }
                                    }
                                }
                                Some(Err(e)) => {
                                    error!("WebSocket receive error: {}", e);
                                    let _ = inbound_status_sender.send(ConnectionStatus::Failed);
                                    break;
                                }
                                None => {
                                    info!("WebSocket connection closed by server");
                                    let _ = inbound_status_sender.send(ConnectionStatus::Disconnected);
                                    break;
                                }
                            }
                        }
                        _ = shutdown_rx.recv() => {
                            info!("WebSocket inbound handler shutting down");
                            break;
                        }
                    }
            }
        });

        // Outbound message handler
        let outbound_rx = self
            .outbound_rx
            .write()
            .await
            .take()
            .ok_or_else(|| anyhow!("Outbound receiver already taken"))?;
        let outbound_stats = stats.clone();
        let outbound_status_sender = status_sender.clone();
        let mut outbound_shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut rx = outbound_rx;
            let mut ping_interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                tokio::select! {
                    msg_opt = rx.recv() => {
                        if let Some(msg) = msg_opt {
                            let tungstenite_msg: Message = msg.clone().into();

                            if let Err(e) = ws_sender.send(tungstenite_msg).await {
                                error!("WebSocket send error: {}", e);
                                let _ = outbound_status_sender.send(ConnectionStatus::Failed);
                                break;
                            }

                            // Update stats
                            {
                                let mut stats = outbound_stats.write().await;
                                stats.messages_sent += 1;

                                // Estimate bytes sent
                                match &msg {
                                    WebSocketMessage::Text { content } => {
                                        stats.bytes_sent += content.len() as u64;
                                    }
                                    WebSocketMessage::Binary { data } => {
                                        stats.bytes_sent += data.len() as u64;
                                    }
                                    WebSocketMessage::Json { value } => {
                                        stats.bytes_sent += value.to_string().len() as u64;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    _ = ping_interval.tick() => {
                        // Send ping to keep connection alive
                        let ping_msg = Message::Ping(b"loki-ping".to_vec());
                        if let Err(e) = ws_sender.send(ping_msg).await {
                            error!("WebSocket ping error: {}", e);
                            let _ = outbound_status_sender.send(ConnectionStatus::Failed);
                            break;
                        }
                    }
                    _ = outbound_shutdown_rx.recv() => {
                        info!("WebSocket outbound handler shutting down");

                        // Send close frame
                        let close_msg = Message::Close(Some(
                            tokio_tungstenite::tungstenite::protocol::CloseFrame {
                                code: 1000.into(),
                                reason: "Client shutdown".into(),
                            }
                        ));
                        let _ = ws_sender.send(close_msg).await;
                        break;
                    }
                }
            }
        });

        // Notify handlers of connection
        let handlers = self.handlers.read().await;
        for handler in handlers.iter() {
            if let Err(e) = handler.on_connect().await {
                error!("Handler connection callback error: {}", e);
            }
        }

        Ok(())
    }

    /// Send a message
    pub async fn send(&self, message: WebSocketMessage) -> Result<()> {
        self.outbound_tx
            .send(message)
            .map_err(|_| anyhow!("Failed to queue message - connection may be closed"))?;
        Ok(())
    }

    /// Send text message
    pub async fn send_text(&self, text: String) -> Result<()> {
        self.send(WebSocketMessage::Text { content: text }).await
    }

    /// Send JSON message
    pub async fn send_json(&self, value: serde_json::Value) -> Result<()> {
        self.send(WebSocketMessage::Json { value }).await
    }

    /// Send binary message
    pub async fn send_binary(&self, data: Vec<u8>) -> Result<()> {
        self.send(WebSocketMessage::Binary { data }).await
    }

    /// Get connection status
    pub async fn status(&self) -> ConnectionStatus {
        self.status.read().await.clone()
    }

    /// Subscribe to status updates
    pub fn subscribe_status(&self) -> broadcast::Receiver<ConnectionStatus> {
        self.status_tx.subscribe()
    }

    /// Get connection statistics
    pub async fn stats(&self) -> WebSocketStats {
        self.stats.read().await.clone()
    }

    /// Set connection status
    async fn set_status(&self, status: ConnectionStatus) {
        *self.status.write().await = status.clone();
        let _ = self.status_tx.send(status);
    }

    /// Connect with auto-reconnection
    pub async fn connect_with_retry(&self) -> Result<()> {
        let mut attempts = 0;

        while attempts < self.config.reconnect_attempts {
            match self.connect().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    attempts += 1;
                    error!("WebSocket connection attempt {} failed: {}", attempts, e);

                    if attempts < self.config.reconnect_attempts {
                        self.set_status(ConnectionStatus::Reconnecting).await;

                        // Update reconnection stats
                        {
                            let mut stats = self.stats.write().await;
                            stats.reconnection_count += 1;
                        }

                        tokio::time::sleep(self.config.reconnect_delay).await;
                    } else {
                        self.set_status(ConnectionStatus::Failed).await;
                        return Err(anyhow!("Failed to connect after {} attempts", attempts));
                    }
                }
            }
        }

        Err(anyhow!("Maximum reconnection attempts reached"))
    }

    /// Disconnect and shutdown
    pub async fn disconnect(&self) -> Result<()> {
        info!("Disconnecting WebSocket client");

        // Set status
        self.set_status(ConnectionStatus::Disconnected).await;

        // Notify handlers
        let handlers = self.handlers.read().await;
        for handler in handlers.iter() {
            if let Err(e) = handler.on_disconnect().await {
                error!("Handler disconnection callback error: {}", e);
            }
        }

        // Send shutdown signal
        let _ = self.shutdown_tx.send(());

        Ok(())
    }
}

/// Simple JSON message handler
pub struct JsonMessageHandler {
    memory: Arc<CognitiveMemory>,
    tag: String,
}

impl JsonMessageHandler {
    pub fn new(memory: Arc<CognitiveMemory>, tag: String) -> Self {
        Self { memory, tag }
    }
}

#[async_trait::async_trait]
impl MessageHandler for JsonMessageHandler {
    async fn handle_message(&self, message: WebSocketMessage) -> Result<()> {
        if let WebSocketMessage::Json { value } = message {
            debug!(
                "Processing JSON message: {}",
                value.to_string().chars().take(100).collect::<String>()
            );

            // Store in memory
            self.memory
                .store(
                    format!("WebSocket {}: {}", self.tag, value.to_string()),
                    vec![],
                    MemoryMetadata {
                        source: "websocket".to_string(),
                        tags: vec![self.tag.clone(), "json".to_string()],
                        importance: 0.6,
                        associations: vec![],

                        context: Some("Generated from automated fix".to_string()),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                    category: "tool_usage".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                )
                .await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_message_conversion() {
        let json_msg = WebSocketMessage::Json { value: serde_json::json!({"test": "data"}) };

        let tungstenite_msg: Message = json_msg.clone().into();
        let converted_back = WebSocketMessage::from(tungstenite_msg);

        match converted_back {
            WebSocketMessage::Json { value } => {
                assert_eq!(value["test"], "data");
            }
            other => {
                assert!(false, "Test failed: Expected Json message but got {:?}", other);
            }
        }
    }
}

// === Advanced WebSocket Lifecycle Management ===

/// Advanced WebSocket manager with connection pooling and health monitoring
pub struct WebSocketManager {
    /// Pool of active connections
    connections: Arc<RwLock<HashMap<String, Arc<WebSocketClient>>>>,

    /// Connection health monitor
    health_monitor: Arc<ConnectionHealthMonitor>,

    /// Real-time streaming optimizer
    streaming_optimizer: Arc<StreamingOptimizer>,

    /// Configuration
    config: WebSocketManagerConfig,
}

/// Configuration for WebSocket manager
#[derive(Debug, Clone)]
pub struct WebSocketManagerConfig {
    pub max_connections: usize,
    pub health_check_interval: Duration,
    pub connection_timeout: Duration,
    pub streaming_buffer_size: usize,
    pub enable_compression: bool,
    pub enable_auto_reconnect: bool,
    pub max_message_rate: usize, // messages per second
}

impl Default for WebSocketManagerConfig {
    fn default() -> Self {
        Self {
            max_connections: 100,
            health_check_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(10),
            streaming_buffer_size: 8192,
            enable_compression: true,
            enable_auto_reconnect: true,
            max_message_rate: 1000,
        }
    }
}

impl WebSocketManager {
    /// Create a new WebSocket manager with advanced features
    pub async fn new(
        config: WebSocketManagerConfig,
        _memory: Arc<CognitiveMemory>,
        _validator: Option<Arc<ActionValidator>>,
    ) -> Result<Self> {
        let health_monitor = Arc::new(ConnectionHealthMonitor::new(config.health_check_interval));
        let streaming_optimizer = Arc::new(StreamingOptimizer::new(config.streaming_buffer_size));

        let manager = Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            health_monitor,
            streaming_optimizer,
            config,
        };

        // Start background health monitoring
        manager.start_health_monitoring().await?;

        tracing::info!("🔌 WebSocket Manager initialized with advanced lifecycle management");
        Ok(manager)
    }

    /// Create or get a WebSocket connection with connection pooling
    pub async fn get_connection(
        &self,
        connection_id: &str,
        wsconfig: WebSocketConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Arc<WebSocketClient>> {
        let connections = self.connections.read().await;

        // Check if connection already exists and is healthy
        if let Some(client) = connections.get(connection_id) {
            if self.is_connection_healthy(client).await {
                tracing::debug!("♻️ Reusing healthy WebSocket connection: {}", connection_id);
                return Ok(client.clone());
            }
        }
        drop(connections);

        // Create new connection
        self.create_new_connection(connection_id, wsconfig, memory, validator).await
    }

    /// Create a new WebSocket connection with advanced features
    async fn create_new_connection(
        &self,
        connection_id: &str,
        wsconfig: WebSocketConfig,
        memory: Arc<CognitiveMemory>,
        validator: Option<Arc<ActionValidator>>,
    ) -> Result<Arc<WebSocketClient>> {
        // Check connection limits
        let connections = self.connections.read().await;
        if connections.len() >= self.config.max_connections {
            return Err(anyhow!(
                "Maximum WebSocket connections reached: {}",
                self.config.max_connections
            ));
        }
        drop(connections);

        tracing::info!("🆕 Creating new WebSocket connection: {}", connection_id);

        // Create enhanced WebSocket client
        let client = Arc::new(WebSocketClient::new(wsconfig, memory.clone(), validator).await?);

        // Add streaming optimization
        self.streaming_optimizer.optimize_connection(&client).await?;

        // Register with health monitor
        self.health_monitor.register_connection(connection_id, client.clone()).await;

        // Connect with retry if enabled
        if self.config.enable_auto_reconnect {
            client.connect_with_retry().await?;
        } else {
            client.connect().await?;
        }

        // Store in connection pool
        let mut connections = self.connections.write().await;
        connections.insert(connection_id.to_string(), client.clone());

        tracing::info!("✅ WebSocket connection established and pooled: {}", connection_id);
        Ok(client)
    }

    /// Start background health monitoring
    async fn start_health_monitoring(&self) -> Result<()> {
        let health_monitor = self.health_monitor.clone();
        let connections = self.connections.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);

            loop {
                interval.tick().await;

                let connections_read = connections.read().await;
                let connection_ids: Vec<String> = connections_read.keys().cloned().collect();
                drop(connections_read);

                // Check health of all connections in parallel
                let health_checks: Vec<_> = connection_ids
                    .iter()
                    .map(|id| health_monitor.check_connection_health(id))
                    .collect();

                let health_results = futures::future::join_all(health_checks).await;

                // Remove unhealthy connections
                let mut connections_write = connections.write().await;
                for (connection_id, is_healthy) in connection_ids.iter().zip(health_results.iter())
                {
                    if !is_healthy {
                        tracing::warn!(
                            "💔 Removing unhealthy WebSocket connection: {}",
                            connection_id
                        );
                        connections_write.remove(connection_id);
                        health_monitor.unregister_connection(connection_id).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Check if connection is healthy
    async fn is_connection_healthy(&self, client: &WebSocketClient) -> bool {
        matches!(client.status().await, ConnectionStatus::Connected)
    }

    /// Close all connections gracefully
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("🛑 Shutting down WebSocket Manager");

        let connections = self.connections.read().await;
        let shutdown_futures: Vec<_> =
            connections.values().map(|client| client.disconnect()).collect();

        // Shutdown all connections in parallel
        let results = futures::future::join_all(shutdown_futures).await;

        let mut error_count = 0;
        for result in results {
            if let Err(e) = result {
                tracing::error!("Error during WebSocket shutdown: {}", e);
                error_count += 1;
            }
        }

        if error_count > 0 {
            tracing::warn!("⚠️ {} connections had errors during shutdown", error_count);
        }

        tracing::info!("✅ WebSocket Manager shutdown complete");
        Ok(())
    }
}

/// Connection health monitor for proactive management
pub struct ConnectionHealthMonitor {
    check_interval: Duration,
    monitored_connections: Arc<RwLock<HashMap<String, ConnectionHealthData>>>,
}

/// Health data for a connection
#[derive(Clone)]
pub struct ConnectionHealthData {
    pub connection: Arc<WebSocketClient>,
    pub last_ping: Instant,
    pub ping_failures: u32,
    pub last_message: Instant,
    pub message_count: u64,
    pub error_count: u64,
}

impl ConnectionHealthMonitor {
    pub fn new(check_interval: Duration) -> Self {
        Self { check_interval, monitored_connections: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Register a connection for health monitoring
    pub async fn register_connection(&self, connection_id: &str, client: Arc<WebSocketClient>) {
        let mut connections = self.monitored_connections.write().await;
        connections.insert(
            connection_id.to_string(),
            ConnectionHealthData {
                connection: client,
                last_ping: Instant::now(),
                ping_failures: 0,
                last_message: Instant::now(),
                message_count: 0,
                error_count: 0,
            },
        );

        tracing::debug!("📊 Registered connection for health monitoring: {}", connection_id);
    }

    /// Unregister a connection from health monitoring
    pub async fn unregister_connection(&self, connection_id: &str) {
        let mut connections = self.monitored_connections.write().await;
        connections.remove(connection_id);
        tracing::debug!("📤 Unregistered connection from health monitoring: {}", connection_id);
    }

    /// Check health of a specific connection
    pub async fn check_connection_health(&self, connection_id: &str) -> bool {
        let connections = self.monitored_connections.read().await;

        if let Some(health_data) = connections.get(connection_id) {
            // Check connection status
            let status = health_data.connection.status().await;
            if !matches!(status, ConnectionStatus::Connected) {
                return false;
            }

            // Check for recent activity
            let time_since_last_message = health_data.last_message.elapsed();
            if time_since_last_message > Duration::from_secs(300) {
                // 5 minutes
                tracing::warn!(
                    "⏰ Connection {} inactive for {:?}",
                    connection_id,
                    time_since_last_message
                );
                return false;
            }

            // Check error rate
            if health_data.error_count > 10 && health_data.message_count > 0 {
                let error_rate = health_data.error_count as f64 / health_data.message_count as f64;
                if error_rate > 0.1 {
                    // 10% error rate
                    tracing::warn!(
                        "⚠️ Connection {} has high error rate: {:.2}%",
                        connection_id,
                        error_rate * 100.0
                    );
                    return false;
                }
            }

            true
        } else {
            false
        }
    }
}

/// Real-time streaming optimizer for high-performance data transmission
pub struct StreamingOptimizer {
    buffer_size: usize,
    compression_enabled: bool,
    optimization_metrics: Arc<RwLock<OptimizationMetrics>>,
}

#[derive(Debug, Default, Clone)]
pub struct OptimizationMetrics {
    pub messages_optimized: u64,
    pub bytes_saved: u64,
    pub compression_ratio: f64,
    pub optimization_time_ms: u64,
}

impl StreamingOptimizer {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            compression_enabled: true,
            optimization_metrics: Arc::new(RwLock::new(OptimizationMetrics::default())),
        }
    }

    /// Optimize a WebSocket connection for streaming
    pub async fn optimize_connection(&self, client: &WebSocketClient) -> Result<()> {
        // Add streaming optimization handler
        let optimizer = Arc::new(StreamingMessageHandler::new(
            self.buffer_size,
            self.compression_enabled,
            self.optimization_metrics.clone(),
        ));

        client.add_handler(optimizer).await;

        tracing::debug!("🚀 Applied streaming optimization to WebSocket connection");
        Ok(())
    }

    /// Get optimization metrics
    pub async fn get_metrics(&self) -> OptimizationMetrics {
        self.optimization_metrics.read().await.clone()
    }
}

/// Streaming message handler with buffering and compression
pub struct StreamingMessageHandler {
    buffer_size: usize,
    compression_enabled: bool,
    message_buffer: Arc<RwLock<Vec<WebSocketMessage>>>,
    optimization_metrics: Arc<RwLock<OptimizationMetrics>>,
}

impl StreamingMessageHandler {
    pub fn new(
        buffer_size: usize,
        compression_enabled: bool,
        optimization_metrics: Arc<RwLock<OptimizationMetrics>>,
    ) -> Self {
        Self {
            buffer_size,
            compression_enabled,
            message_buffer: Arc::new(RwLock::new(Vec::with_capacity(buffer_size))),
            optimization_metrics,
        }
    }

    /// Optimize message for streaming
    async fn optimize_message(&self, message: &WebSocketMessage) -> Result<WebSocketMessage> {
        let start_time = Instant::now();

        let optimized = if self.compression_enabled {
            self.compress_message(message).await?
        } else {
            message.clone()
        };

        // Update metrics
        let mut metrics = self.optimization_metrics.write().await;
        metrics.messages_optimized += 1;
        metrics.optimization_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(optimized)
    }

    /// Compress message content
    async fn compress_message(&self, message: &WebSocketMessage) -> Result<WebSocketMessage> {
        match message {
            WebSocketMessage::Json { value } => {
                let json_string = serde_json::to_string(value)?;

                // Simple compression simulation (in production, use actual compression)
                let compressed_size = (json_string.len() as f64 * 0.7) as usize;
                let compressed_data = json_string.bytes().take(compressed_size).collect();

                let mut metrics = self.optimization_metrics.write().await;
                metrics.bytes_saved += (json_string.len() - compressed_size) as u64;
                metrics.compression_ratio = compressed_size as f64 / json_string.len() as f64;

                Ok(WebSocketMessage::Binary { data: compressed_data })
            }
            _ => Ok(message.clone()),
        }
    }
}

#[async_trait::async_trait]
impl MessageHandler for StreamingMessageHandler {
    async fn handle_message(&self, message: WebSocketMessage) -> Result<()> {
        // Add to buffer for batch processing
        let mut buffer = self.message_buffer.write().await;
        buffer.push(message);

        // Process buffer when full
        if buffer.len() >= self.buffer_size {
            self.process_message_batch(&buffer).await?;
            buffer.clear();
        }

        Ok(())
    }

    async fn on_connect(&self) -> Result<()> {
        tracing::info!("🔗 Streaming optimization handler connected");
        Ok(())
    }

    async fn on_disconnect(&self) -> Result<()> {
        // Process any remaining buffered messages
        let buffer = self.message_buffer.read().await;
        if !buffer.is_empty() {
            self.process_message_batch(&buffer).await?;
        }

        tracing::info!("🔌 Streaming optimization handler disconnected");
        Ok(())
    }
}

impl StreamingMessageHandler {
    /// Process a batch of messages efficiently
    async fn process_message_batch(&self, messages: &[WebSocketMessage]) -> Result<()> {
        tracing::debug!("📦 Processing message batch of {} messages", messages.len());

        // Parallel message optimization using rayon
        use rayon::prelude::*;

        let optimized_messages: Vec<_> = messages
            .par_iter()
            .map(|msg| {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(async { self.optimize_message(msg).await })
                })
            })
            .collect::<Result<Vec<_>>>()?;

        tracing::debug!("✨ Optimized batch of {} messages", optimized_messages.len());
        Ok(())
    }
}
