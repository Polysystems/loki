//! Distributed Safety System
//!
//! Coordinates safety measures across multiple Loki instances in a cluster,
//! providing cluster-wide safety monitoring, emergency procedures, and
//! distributed consensus for critical actions.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, broadcast};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::safety::{ActionType, AuditEvent};

/// Node identifier in the distributed safety cluster
pub type NodeId = Uuid;

/// Distributed safety cluster member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyNode {
    pub id: NodeId,
    pub address: SocketAddr,
    pub role: NodeRole,
    pub status: NodeStatus,
    pub last_heartbeat: u64,
    pub capabilities: NodeCapabilities,
    pub resource_usage: NodeResourceUsage,
    pub safety_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeRole {
    /// Primary safety coordinator
    Coordinator,
    /// Secondary coordinator (backup)
    SecondaryCoordinator,
    /// Regular cluster member
    Member,
    /// Observer (read-only access)
    Observer,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Active,
    Degraded,
    Offline,
    Emergency,
    Quarantined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub can_approve_actions: bool,
    pub can_emergency_stop: bool,
    pub can_coordinate_cluster: bool,
    pub max_action_risk_level: u8,
    pub supports_consensus: bool,
    pub audit_retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResourceUsage {
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub disk_percent: f32,
    pub network_latency_ms: u64,
    pub active_connections: u32,
    pub actions_per_minute: u32,
}

/// Distributed safety event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedSafetyEvent {
    /// Request cluster consensus for high-risk action
    ConsensusRequest {
        id: Uuid,
        requester: NodeId,
        action: ActionType,
        risk_level: u8,
        context: String,
        votes_needed: u8,
        timeout_seconds: u64,
    },

    /// Vote on a consensus request
    ConsensusVote { request_id: Uuid, voter: NodeId, approved: bool, reason: String },

    /// Emergency cluster-wide safety alert
    EmergencyAlert {
        id: Uuid,
        source: NodeId,
        alert_type: EmergencyType,
        severity: EmergencySeverity,
        message: String,
        affected_nodes: Vec<NodeId>,
        auto_actions: Vec<EmergencyAction>,
    },

    /// Node health status update
    HealthUpdate { node: SafetyNode, metrics: NodeHealthMetrics },

    /// Audit event synchronization
    AuditSync { events: Vec<AuditEvent>, source_node: NodeId },

    /// Resource limit breach notification
    ResourceBreach {
        node: NodeId,
        resource_type: String,
        current_value: f64,
        limit: f64,
        severity: EmergencySeverity,
    },

    /// Cluster configuration update
    ConfigUpdate { config: DistributedSafetyConfig, updated_by: NodeId },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyType {
    SecurityBreach,
    ResourceExhaustion,
    NodeCompromise,
    MassiveFailure,
    DataCorruption,
    UnauthorizedAccess,
    SystemOverload,
    CostOverrun,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum EmergencySeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyAction {
    QuarantineNode(NodeId),
    EmergencyStop,
    IsolateService(String),
    RateLimitService(String, u32),
    NotifyOperators,
    SaveState,
    FailoverToBackup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealthMetrics {
    pub uptime_seconds: u64,
    pub total_actions: u64,
    pub failed_actions: u64,
    pub success_rate: f64,
    pub avg_response_time_ms: f64,
    pub memory_leaks_detected: u32,
    pub security_violations: u32,
    pub last_audit_timestamp: u64,
}

/// Consensus request tracking
#[derive(Debug, Clone)]
pub struct ConsensusRequest {
    pub id: Uuid,
    pub requester: NodeId,
    pub action: ActionType,
    pub risk_level: u8,
    pub context: String,
    pub votes_needed: u8,
    pub votes_received: HashMap<NodeId, bool>,
    pub created_at: SystemTime,
    pub timeout: Duration,
    pub result: Option<bool>,
}

/// Distributed safety configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedSafetyConfig {
    /// Minimum number of nodes for cluster operation
    pub min_cluster_size: usize,

    /// Consensus thresholds by risk level
    pub consensus_thresholds: HashMap<u8, f64>,

    /// Heartbeat interval
    pub heartbeat_interval_secs: u64,

    /// Node timeout before marked offline
    pub node_timeout_secs: u64,

    /// Emergency response settings
    pub emergency_response: EmergencyResponseConfig,

    /// Audit synchronization settings
    pub audit_sync: AuditSyncConfig,

    /// Resource monitoring thresholds
    pub resource_thresholds: DistributedResourceThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyResponseConfig {
    pub auto_quarantine_on_breach: bool,
    pub emergency_stop_threshold: EmergencySeverity,
    pub max_emergency_actions_per_hour: u32,
    pub operator_notification_methods: Vec<String>,
    pub escalation_delay_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSyncConfig {
    pub enabled: bool,
    pub sync_interval_secs: u64,
    pub batch_size: usize,
    pub retention_days: u32,
    pub encrypt_in_transit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedResourceThresholds {
    pub cpu_warning: f32,
    pub cpu_critical: f32,
    pub memory_warning: f32,
    pub memory_critical: f32,
    pub disk_warning: f32,
    pub disk_critical: f32,
    pub network_latency_warning_ms: u64,
    pub network_latency_critical_ms: u64,
}

impl Default for DistributedSafetyConfig {
    fn default() -> Self {
        let mut consensus_thresholds = HashMap::new();
        consensus_thresholds.insert(1, 0.5); // Low risk: 50% approval
        consensus_thresholds.insert(2, 0.6); // Medium risk: 60% approval
        consensus_thresholds.insert(3, 0.7); // High risk: 70% approval
        consensus_thresholds.insert(4, 0.8); // Very high risk: 80% approval
        consensus_thresholds.insert(5, 0.9); // Critical risk: 90% approval

        Self {
            min_cluster_size: 3,
            consensus_thresholds,
            heartbeat_interval_secs: 30,
            node_timeout_secs: 120,
            emergency_response: EmergencyResponseConfig {
                auto_quarantine_on_breach: true,
                emergency_stop_threshold: EmergencySeverity::Critical,
                max_emergency_actions_per_hour: 10,
                operator_notification_methods: vec!["email".to_string(), "slack".to_string()],
                escalation_delay_seconds: 300,
            },
            audit_sync: AuditSyncConfig {
                enabled: true,
                sync_interval_secs: 60,
                batch_size: 100,
                retention_days: 90,
                encrypt_in_transit: true,
            },
            resource_thresholds: DistributedResourceThresholds {
                cpu_warning: 70.0,
                cpu_critical: 90.0,
                memory_warning: 80.0,
                memory_critical: 95.0,
                disk_warning: 85.0,
                disk_critical: 95.0,
                network_latency_warning_ms: 1000,
                network_latency_critical_ms: 5000,
            },
        }
    }
}

/// Distributed safety system
pub struct DistributedSafety {
    /// This node's information
    node_info: Arc<RwLock<SafetyNode>>,

    /// Cluster configuration
    config: Arc<RwLock<DistributedSafetyConfig>>,

    /// Other nodes in the cluster
    cluster_nodes: Arc<RwLock<HashMap<NodeId, SafetyNode>>>,

    /// Active consensus requests
    consensus_requests: Arc<RwLock<HashMap<Uuid, ConsensusRequest>>>,

    /// Emergency alerts
    emergency_alerts: Arc<RwLock<Vec<DistributedSafetyEvent>>>,

    /// Event broadcast channels
    event_sender: broadcast::Sender<DistributedSafetyEvent>,

    /// Network communication
    listener: Option<TcpListener>,

    /// Outbound connections to other nodes
    connections: Arc<RwLock<HashMap<NodeId, TcpStream>>>,

    /// Health metrics for this node
    health_metrics: Arc<RwLock<NodeHealthMetrics>>,
}

impl DistributedSafety {
    /// Create a new distributed safety system
    pub async fn new(
        node_id: NodeId,
        listen_addr: SocketAddr,
        role: NodeRole,
        config: Option<DistributedSafetyConfig>,
    ) -> Result<Self> {
        let listener = TcpListener::bind(listen_addr).await?;
        info!("Distributed safety node listening on {}", listen_addr);

        let capabilities = NodeCapabilities {
            can_approve_actions: matches!(
                role,
                NodeRole::Coordinator | NodeRole::SecondaryCoordinator | NodeRole::Member
            ),
            can_emergency_stop: matches!(
                role,
                NodeRole::Coordinator | NodeRole::SecondaryCoordinator
            ),
            can_coordinate_cluster: matches!(role, NodeRole::Coordinator),
            max_action_risk_level: match role {
                NodeRole::Coordinator => 5,
                NodeRole::SecondaryCoordinator => 4,
                NodeRole::Member => 3,
                NodeRole::Observer => 0,
            },
            supports_consensus: true,
            audit_retention_days: 90,
        };

        let node_info = SafetyNode {
            id: node_id,
            address: listen_addr,
            role,
            status: NodeStatus::Active,
            last_heartbeat: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            capabilities,
            resource_usage: NodeResourceUsage {
                cpu_percent: 0.0,
                memory_percent: 0.0,
                disk_percent: 0.0,
                network_latency_ms: 0,
                active_connections: 0,
                actions_per_minute: 0,
            },
            safety_score: 1.0,
        };

        let health_metrics = NodeHealthMetrics {
            uptime_seconds: 0,
            total_actions: 0,
            failed_actions: 0,
            success_rate: 1.0,
            avg_response_time_ms: 0.0,
            memory_leaks_detected: 0,
            security_violations: 0,
            last_audit_timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        };

        let (event_sender, _) = broadcast::channel(1000);

        Ok(Self {
            node_info: Arc::new(RwLock::new(node_info)),
            config: Arc::new(RwLock::new(config.unwrap_or_default())),
            cluster_nodes: Arc::new(RwLock::new(HashMap::new())),
            consensus_requests: Arc::new(RwLock::new(HashMap::new())),
            emergency_alerts: Arc::new(RwLock::new(Vec::new())),
            event_sender,
            listener: Some(listener),
            connections: Arc::new(RwLock::new(HashMap::new())),
            health_metrics: Arc::new(RwLock::new(health_metrics)),
        })
    }

    /// Start the distributed safety system
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting distributed safety system");

        // Start heartbeat loop
        self.start_heartbeat_loop().await?;

        // Start consensus cleanup loop
        self.start_consensus_cleanup().await?;

        // Start network listener
        if let Some(listener) = self.listener.take() {
            self.start_network_listener(listener).await?;
        }

        // Start health monitoring
        self.start_health_monitoring().await?;

        info!("Distributed safety system started");
        Ok(())
    }

    /// Start the heartbeat loop
    async fn start_heartbeat_loop(&self) -> Result<()> {
        let node_info = self.node_info.clone();
        let config = self.config.clone();
        let cluster_nodes = self.cluster_nodes.clone();
        let event_sender = self.event_sender.clone();
        let health_metrics = self.health_metrics.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Default heartbeat

            loop {
                interval.tick().await;

                let heartbeat_interval = {
                    let config = config.read().await;
                    Duration::from_secs(config.heartbeat_interval_secs)
                };

                // Update our heartbeat
                {
                    let mut node = node_info.write().await;
                    node.last_heartbeat =
                        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
                }

                // Check for offline nodes
                let timeout_threshold = {
                    let config = config.read().await;
                    config.node_timeout_secs
                };

                let current_time =
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

                {
                    let mut nodes = cluster_nodes.write().await;
                    let mut offline_nodes = Vec::new();

                    for (node_id, node) in nodes.iter_mut() {
                        if current_time - node.last_heartbeat > timeout_threshold {
                            if node.status != NodeStatus::Offline {
                                warn!("Node {} went offline", node_id);
                                node.status = NodeStatus::Offline;
                                offline_nodes.push(*node_id);
                            }
                        }
                    }

                    // Notify about offline nodes
                    for node_id in offline_nodes {
                        let alert = DistributedSafetyEvent::EmergencyAlert {
                            id: Uuid::new_v4(),
                            source: node_info.read().await.id,
                            alert_type: EmergencyType::MassiveFailure,
                            severity: EmergencySeverity::Medium,
                            message: format!("Node {} went offline", node_id),
                            affected_nodes: vec![node_id],
                            auto_actions: vec![],
                        };

                        let _ = event_sender.send(alert);
                    }
                }

                // Broadcast health update
                let node = node_info.read().await.clone();
                let metrics = health_metrics.read().await.clone();

                let health_update = DistributedSafetyEvent::HealthUpdate { node, metrics };

                let _ = event_sender.send(health_update);

                // Update interval if config changed
                interval = tokio::time::interval(heartbeat_interval);
            }
        });

        Ok(())
    }

    /// Start consensus request cleanup
    async fn start_consensus_cleanup(&self) -> Result<()> {
        let consensus_requests = self.consensus_requests.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                let mut requests = consensus_requests.write().await;
                let current_time = SystemTime::now();

                // Remove expired consensus requests
                requests.retain(|_, request| {
                    let elapsed =
                        current_time.duration_since(request.created_at).unwrap_or_default();

                    if elapsed > request.timeout {
                        debug!("Consensus request {} expired", request.id);
                        false
                    } else {
                        true
                    }
                });
            }
        });

        Ok(())
    }

    /// Start network listener for incoming connections
    async fn start_network_listener(&self, listener: TcpListener) -> Result<()> {
        let event_sender = self.event_sender.clone();
        let cluster_nodes = self.cluster_nodes.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        debug!("Accepted connection from {}", addr);

                        let event_sender = event_sender.clone();
                        let cluster_nodes = cluster_nodes.clone();

                        tokio::spawn(async move {
                            if let Err(e) =
                                Self::handle_connection(stream, event_sender, cluster_nodes).await
                            {
                                error!("Error handling connection from {}: {}", addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle incoming network connection
    async fn handle_connection(
        mut stream: TcpStream,
        event_sender: broadcast::Sender<DistributedSafetyEvent>,
        cluster_nodes: Arc<RwLock<HashMap<NodeId, SafetyNode>>>,
    ) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut buffer = vec![0; 4096];

        loop {
            match stream.read(&mut buffer).await {
                Ok(0) => break, // Connection closed
                Ok(n) => {
                    let data = &buffer[..n];

                    // Parse the received event
                    match serde_json::from_slice::<DistributedSafetyEvent>(data) {
                        Ok(event) => {
                            debug!("Received distributed safety event: {:?}", event);

                            // Update cluster nodes if it's a health update
                            if let DistributedSafetyEvent::HealthUpdate { node, .. } = &event {
                                let mut nodes = cluster_nodes.write().await;
                                nodes.insert(node.id, node.clone());
                            }

                            // Broadcast the event
                            let _ = event_sender.send(event);

                            // Send acknowledgment
                            if let Err(e) = stream.write_all(b"ACK").await {
                                error!("Failed to send ACK: {}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse safety event: {}", e);
                            break;
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to read from stream: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) -> Result<()> {
        let health_metrics = self.health_metrics.clone();
        let node_info = self.node_info.clone();
        let config = self.config.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            let start_time = SystemTime::now();

            loop {
                interval.tick().await;

                // Update health metrics
                {
                    let mut metrics = health_metrics.write().await;
                    metrics.uptime_seconds =
                        SystemTime::now().duration_since(start_time).unwrap_or_default().as_secs();

                    // Calculate success rate
                    if metrics.total_actions > 0 {
                        metrics.success_rate =
                            1.0 - (metrics.failed_actions as f64 / metrics.total_actions as f64);
                    }
                }

                // Check resource thresholds
                let node = node_info.read().await;
                let thresholds = config.read().await.resource_thresholds.clone();

                // CPU threshold check
                if node.resource_usage.cpu_percent >= thresholds.cpu_critical {
                    let alert = DistributedSafetyEvent::ResourceBreach {
                        node: node.id,
                        resource_type: "cpu".to_string(),
                        current_value: node.resource_usage.cpu_percent as f64,
                        limit: thresholds.cpu_critical as f64,
                        severity: EmergencySeverity::Critical,
                    };
                    let _ = event_sender.send(alert);
                } else if node.resource_usage.cpu_percent >= thresholds.cpu_warning {
                    let alert = DistributedSafetyEvent::ResourceBreach {
                        node: node.id,
                        resource_type: "cpu".to_string(),
                        current_value: node.resource_usage.cpu_percent as f64,
                        limit: thresholds.cpu_warning as f64,
                        severity: EmergencySeverity::Medium,
                    };
                    let _ = event_sender.send(alert);
                }

                // Memory threshold check
                if node.resource_usage.memory_percent >= thresholds.memory_critical {
                    let alert = DistributedSafetyEvent::ResourceBreach {
                        node: node.id,
                        resource_type: "memory".to_string(),
                        current_value: node.resource_usage.memory_percent as f64,
                        limit: thresholds.memory_critical as f64,
                        severity: EmergencySeverity::Critical,
                    };
                    let _ = event_sender.send(alert);
                }

                // Network latency check
                if node.resource_usage.network_latency_ms >= thresholds.network_latency_critical_ms
                {
                    let alert = DistributedSafetyEvent::ResourceBreach {
                        node: node.id,
                        resource_type: "network_latency".to_string(),
                        current_value: node.resource_usage.network_latency_ms as f64,
                        limit: thresholds.network_latency_critical_ms as f64,
                        severity: EmergencySeverity::High,
                    };
                    let _ = event_sender.send(alert);
                }
            }
        });

        Ok(())
    }

    /// Request cluster consensus for a high-risk action
    pub async fn request_consensus(
        &self,
        action: ActionType,
        risk_level: u8,
        context: String,
    ) -> Result<bool> {
        let config = self.config.read().await;
        let threshold = config.consensus_thresholds.get(&risk_level).copied().unwrap_or(0.5);

        let cluster_nodes = self.cluster_nodes.read().await;
        let active_nodes: Vec<_> = cluster_nodes
            .values()
            .filter(|node| {
                node.status == NodeStatus::Active && node.capabilities.can_approve_actions
            })
            .collect();

        if active_nodes.is_empty() {
            warn!("No active nodes available for consensus");
            return Ok(false);
        }

        let votes_needed = (active_nodes.len() as f64 * threshold).ceil() as u8;

        let request_id = Uuid::new_v4();
        let consensus_request = ConsensusRequest {
            id: request_id,
            requester: self.node_info.read().await.id,
            action: action.clone(),
            risk_level,
            context: context.clone(),
            votes_needed,
            votes_received: HashMap::new(),
            created_at: SystemTime::now(),
            timeout: Duration::from_secs(300), // 5 minutes
            result: None,
        };

        // Store the request
        {
            let mut requests = self.consensus_requests.write().await;
            requests.insert(request_id, consensus_request);
        }

        // Broadcast consensus request
        let event = DistributedSafetyEvent::ConsensusRequest {
            id: request_id,
            requester: self.node_info.read().await.id,
            action,
            risk_level,
            context,
            votes_needed,
            timeout_seconds: 300,
        };

        self.event_sender.send(event)?;

        // Wait for consensus result
        let result = timeout(Duration::from_secs(300), async {
            let mut receiver = self.event_sender.subscribe();

            loop {
                if let Ok(event) = receiver.recv().await {
                    if let DistributedSafetyEvent::ConsensusVote {
                        request_id: vote_request_id,
                        ..
                    } = event
                    {
                        if vote_request_id == request_id {
                            // Check if we have enough votes
                            let requests = self.consensus_requests.read().await;
                            if let Some(request) = requests.get(&request_id) {
                                let approvals =
                                    request.votes_received.values().filter(|&&v| v).count();
                                if approvals >= votes_needed as usize {
                                    return true;
                                } else if request.votes_received.len() >= active_nodes.len() {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        })
        .await;

        match result {
            Ok(consensus) => {
                info!("Consensus reached for action: {}", consensus);
                Ok(consensus)
            }
            Err(_) => {
                warn!("Consensus timeout for request {}", request_id);
                Ok(false)
            }
        }
    }

    /// Vote on a consensus request
    pub async fn vote_on_consensus(
        &self,
        request_id: Uuid,
        approved: bool,
        reason: String,
    ) -> Result<()> {
        let node_id = self.node_info.read().await.id;

        // Update the request with our vote
        {
            let mut requests = self.consensus_requests.write().await;
            if let Some(request) = requests.get_mut(&request_id) {
                request.votes_received.insert(node_id, approved);
            }
        }

        // Broadcast our vote
        let vote =
            DistributedSafetyEvent::ConsensusVote { request_id, voter: node_id, approved, reason };

        self.event_sender.send(vote)?;

        info!(
            "Voted {} on consensus request {}",
            if approved { "approve" } else { "deny" },
            request_id
        );
        Ok(())
    }

    /// Trigger emergency alert to cluster
    pub async fn trigger_emergency_alert(
        &self,
        alert_type: EmergencyType,
        severity: EmergencySeverity,
        message: String,
        affected_nodes: Vec<NodeId>,
        auto_actions: Vec<EmergencyAction>,
    ) -> Result<()> {
        let alert = DistributedSafetyEvent::EmergencyAlert {
            id: Uuid::new_v4(),
            source: self.node_info.read().await.id,
            alert_type,
            severity,
            message: message.clone(),
            affected_nodes,
            auto_actions,
        };

        // Store alert
        {
            let mut alerts = self.emergency_alerts.write().await;
            alerts.push(alert.clone());

            // Limit alert history
            if alerts.len() > 1000 {
                let drain_count = alerts.len() - 1000;
                alerts.drain(0..drain_count);
            }
        }

        // Broadcast alert
        self.event_sender.send(alert)?;

        error!("Emergency alert triggered: {}", message);
        Ok(())
    }

    /// Get cluster status
    pub async fn get_cluster_status(&self) -> ClusterStatus {
        let cluster_nodes = self.cluster_nodes.read().await;
        let config = self.config.read().await;

        let total_nodes = cluster_nodes.len() + 1; // +1 for this node
        let active_nodes =
            cluster_nodes.values().filter(|node| node.status == NodeStatus::Active).count() + 1; // +1 for this node

        let degraded_nodes =
            cluster_nodes.values().filter(|node| node.status == NodeStatus::Degraded).count();

        let offline_nodes =
            cluster_nodes.values().filter(|node| node.status == NodeStatus::Offline).count();

        let cluster_health = if active_nodes >= config.min_cluster_size {
            if degraded_nodes == 0 && offline_nodes == 0 {
                ClusterHealth::Healthy
            } else if degraded_nodes > 0 && offline_nodes == 0 {
                ClusterHealth::Degraded
            } else {
                ClusterHealth::Unhealthy
            }
        } else {
            ClusterHealth::Critical
        };

        ClusterStatus {
            total_nodes,
            active_nodes,
            degraded_nodes,
            offline_nodes,
            health: cluster_health,
            coordinator: cluster_nodes
                .values()
                .find(|node| matches!(node.role, NodeRole::Coordinator))
                .map(|node| node.id),
            consensus_requests_active: self.consensus_requests.read().await.len(),
            emergency_alerts_active: self
                .emergency_alerts
                .read()
                .await
                .iter()
                .filter(|alert| {
                    if let DistributedSafetyEvent::EmergencyAlert { severity, .. } = alert {
                        *severity >= EmergencySeverity::High
                    } else {
                        false
                    }
                })
                .count(),
        }
    }

    /// Subscribe to distributed safety events
    pub fn subscribe_events(&self) -> broadcast::Receiver<DistributedSafetyEvent> {
        self.event_sender.subscribe()
    }

    /// Update node resource usage
    pub async fn update_resource_usage(&self, usage: NodeResourceUsage) -> Result<()> {
        let mut node = self.node_info.write().await;
        node.resource_usage = usage;
        Ok(())
    }

    /// Record action completion for health metrics
    pub async fn record_action_completion(
        &self,
        success: bool,
        response_time_ms: u64,
    ) -> Result<()> {
        let mut metrics = self.health_metrics.write().await;

        metrics.total_actions += 1;
        if !success {
            metrics.failed_actions += 1;
        }

        // Update average response time
        metrics.avg_response_time_ms =
            (metrics.avg_response_time_ms * 0.9) + (response_time_ms as f64 * 0.1);

        Ok(())
    }
}

/// Cluster status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub degraded_nodes: usize,
    pub offline_nodes: usize,
    pub health: ClusterHealth,
    pub coordinator: Option<NodeId>,
    pub consensus_requests_active: usize,
    pub emergency_alerts_active: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}
