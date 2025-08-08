use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, timeout};
use tracing::{debug, info, warn};

use super::coordinator::NodeInfo;
use super::ClusterConfig;

/// Network discovery service for finding other Loki instances
pub struct NetworkDiscovery {
    config: DiscoveryConfig,
    local_node: Arc<RwLock<NodeInfo>>,
    discovered_nodes: Arc<RwLock<HashMap<String, DiscoveredNode>>>,
    discovery_methods: Vec<DiscoveryMethod>,
    event_sender: broadcast::Sender<DiscoveryEvent>,
    shutdown_sender: mpsc::Sender<()>,
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// UDP broadcast port for node discovery
    pub discovery_port: u16,
    /// TCP port for node communication
    pub communication_port: u16,
    /// Discovery interval in seconds
    pub discovery_interval_secs: u64,
    /// Node timeout in seconds
    pub node_timeout_secs: u64,
    /// Enable multicast discovery
    pub enable_multicast: bool,
    /// Multicast address for discovery
    pub multicast_address: String,
    /// Enable mDNS/Bonjour discovery
    pub enable_mdns: bool,
    /// Enable local network scanning
    pub enable_network_scan: bool,
    /// Network scan ranges (CIDR notation)
    pub scan_ranges: Vec<String>,
    /// Maximum concurrent discovery connections
    pub max_concurrent_discoveries: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            discovery_port: 7878,
            communication_port: 7879,
            discovery_interval_secs: 30,
            node_timeout_secs: 180,
            enable_multicast: true,
            multicast_address: "239.255.42.42".to_string(),
            enable_mdns: true,
            enable_network_scan: true,
            scan_ranges: vec![
                "192.168.0.0/16".to_string(),
                "10.0.0.0/8".to_string(),
                "172.16.0.0/12".to_string(),
            ],
            max_concurrent_discoveries: 50,
        }
    }
}

/// Information about a discovered node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredNode {
    pub node_info: NodeInfo,
    pub discovery_method: DiscoveryMethod,
    pub endpoint: SocketAddr,
    pub last_seen: SystemTime,
    pub response_time_ms: u64,
    pub capabilities: NodeCapabilities,
    pub trust_level: TrustLevel,
}

/// Node capabilities discovered during handshake
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub loki_version: String,
    pub supported_protocols: Vec<String>,
    pub available_models: Vec<String>,
    pub compute_capabilities: ComputeCapabilities,
    pub cluster_role: ClusterRole,
}

/// Compute capabilities of a discovered node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub gpu_count: u32,
    pub gpu_memory_gb: f64,
    pub inference_performance: f64, // tokens per second
    pub supports_batching: bool,
    pub supports_streaming: bool,
}

/// Role of the node in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterRole {
    Coordinator,    // Cluster management and coordination
    Worker,         // Model inference and processing
    Storage,        // Model storage and caching
    Gateway,        // External API gateway
    Hybrid(Vec<ClusterRole>), // Multiple roles
}

/// Trust level for discovered nodes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    Untrusted,      // Unknown or suspicious node
    Discovering,    // Currently verifying node
    Verified,       // Basic verification complete
    Authenticated,  // Cryptographically verified
    Trusted,        // Long-term trusted node
}

/// Discovery methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DiscoveryMethod {
    UdpBroadcast,
    Multicast,
    MDns,
    NetworkScan,
    Manual,
    Gossip,
}

/// Discovery events
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    NodeDiscovered(String, DiscoveredNode),
    NodeUpdated(String, DiscoveredNode),
    NodeLost(String),
    NodeVerified(String),
    DiscoveryStarted(DiscoveryMethod),
    DiscoveryCompleted(DiscoveryMethod, usize),
    DiscoveryError(DiscoveryMethod, String),
}

/// Discovery protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryMessage {
    Announce {
        node_id: String,
        node_info: NodeInfo,
        capabilities: NodeCapabilities,
        cluster_id: String,
    },
    Probe {
        requester_id: String,
        cluster_id: String,
    },
    Response {
        node_id: String,
        node_info: NodeInfo,
        capabilities: NodeCapabilities,
        cluster_id: String,
    },
    Heartbeat {
        node_id: String,
        timestamp: SystemTime,
        status: String,
    },
    Goodbye {
        node_id: String,
        reason: String,
    },
}

impl NetworkDiscovery {
    /// Create a new network discovery service
    pub async fn new(
        config: DiscoveryConfig,
        _cluster_config: ClusterConfig,
        local_node: NodeInfo,
    ) -> Result<Self> {
        let (event_sender, _) = broadcast::channel(1000);
        let (shutdown_sender, _) = mpsc::channel(10);

        let discovery_methods = vec![
            DiscoveryMethod::UdpBroadcast,
            DiscoveryMethod::Multicast,
            DiscoveryMethod::NetworkScan,
        ];

        Ok(Self {
            config,
            local_node: Arc::new(RwLock::new(local_node)),
            discovered_nodes: Arc::new(RwLock::new(HashMap::new())),
            discovery_methods,
            event_sender,
            shutdown_sender,
        })
    }

    /// Start the discovery service
    pub async fn start(&self) -> Result<()> {
        info!("🔍 Starting network discovery service on port {}", self.config.discovery_port);

        // Start UDP broadcast listener
        self.start_udp_listener().await?;

        // Start TCP communication listener
        self.start_tcp_listener().await?;

        // Start discovery methods
        for method in &self.discovery_methods {
            match method {
                DiscoveryMethod::UdpBroadcast => {
                    self.start_udp_discovery().await?;
                }
                DiscoveryMethod::Multicast => {
                    if self.config.enable_multicast {
                        self.start_multicast_discovery().await?;
                    }
                }
                DiscoveryMethod::NetworkScan => {
                    if self.config.enable_network_scan {
                        self.start_network_scan().await?;
                    }
                }
                _ => {}
            }
        }

        // Start periodic cleanup
        self.start_node_cleanup().await?;

        info!("✅ Network discovery service started successfully");
        Ok(())
    }

    /// Start UDP broadcast listener
    async fn start_udp_listener(&self) -> Result<()> {
        let socket = UdpSocket::bind(("0.0.0.0", self.config.discovery_port)).await?;
        socket.set_broadcast(true)?;
        
        let discovered_nodes = self.discovered_nodes.clone();
        let local_node = self.local_node.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut buffer = [0u8; 4096];
            
            loop {
                match socket.recv_from(&mut buffer).await {
                    Ok((size, addr)) => {
                        if let Ok(message) = serde_json::from_slice::<DiscoveryMessage>(&buffer[..size]) {
                            Self::handle_discovery_message(
                                message,
                                addr,
                                &discovered_nodes,
                                &local_node,
                                &event_sender,
                                DiscoveryMethod::UdpBroadcast,
                            ).await;
                        }
                    }
                    Err(e) => {
                        warn!("UDP listener error: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Start TCP communication listener
    async fn start_tcp_listener(&self) -> Result<()> {
        let listener = TcpListener::bind(("0.0.0.0", self.config.communication_port)).await?;
        let discovered_nodes = self.discovered_nodes.clone();
        let local_node = self.local_node.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let nodes = discovered_nodes.clone();
                        let local = local_node.clone();
                        
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_tcp_connection(stream, addr, nodes, local).await {
                                debug!("TCP connection error from {}: {}", addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        warn!("TCP listener error: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Start UDP broadcast discovery
    async fn start_udp_discovery(&self) -> Result<()> {
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        socket.set_broadcast(true)?;
        
        let local_node = self.local_node.clone();
        let config = self.config.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut discovery_interval = interval(Duration::from_secs(config.discovery_interval_secs));
            
            loop {
                discovery_interval.tick().await;
                
                let announce_message = DiscoveryMessage::Announce {
                    node_id: local_node.read().id.clone(),
                    node_info: local_node.read().clone(),
                    capabilities: Self::get_local_capabilities(),
                    cluster_id: "loki-cluster".to_string(),
                };

                if let Ok(serialized) = serde_json::to_vec(&announce_message) {
                    // Broadcast to common subnets
                    let broadcast_addresses = vec![
                        "255.255.255.255",
                        "192.168.255.255",
                        "10.255.255.255",
                        "172.31.255.255",
                    ];

                    for addr in broadcast_addresses {
                        let target = format!("{}:{}", addr, config.discovery_port);
                        if let Ok(target_addr) = target.parse::<SocketAddr>() {
                            if let Err(e) = socket.send_to(&serialized, target_addr).await {
                                debug!("Failed to send broadcast to {}: {}", target, e);
                            }
                        }
                    }
                }

                let _ = event_sender.send(DiscoveryEvent::DiscoveryCompleted(
                    DiscoveryMethod::UdpBroadcast, 
                    0
                ));
            }
        });

        Ok(())
    }

    /// Start multicast discovery
    async fn start_multicast_discovery(&self) -> Result<()> {
        let multicast_addr: Ipv4Addr = self.config.multicast_address.parse()?;
        let socket = UdpSocket::bind(("0.0.0.0", self.config.discovery_port)).await?;
        socket.join_multicast_v4(multicast_addr, Ipv4Addr::UNSPECIFIED)?;

        let discovered_nodes = self.discovered_nodes.clone();
        let local_node = self.local_node.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut buffer = [0u8; 4096];
            
            loop {
                match socket.recv_from(&mut buffer).await {
                    Ok((size, addr)) => {
                        if let Ok(message) = serde_json::from_slice::<DiscoveryMessage>(&buffer[..size]) {
                            Self::handle_discovery_message(
                                message,
                                addr,
                                &discovered_nodes,
                                &local_node,
                                &event_sender,
                                DiscoveryMethod::Multicast,
                            ).await;
                        }
                    }
                    Err(e) => {
                        warn!("Multicast listener error: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Start network scanning discovery
    async fn start_network_scan(&self) -> Result<()> {
        let config = self.config.clone();
        let discovered_nodes = self.discovered_nodes.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut scan_interval = interval(Duration::from_secs(config.discovery_interval_secs * 5));
            
            loop {
                scan_interval.tick().await;
                
                let _ = event_sender.send(DiscoveryEvent::DiscoveryStarted(DiscoveryMethod::NetworkScan));
                
                for range in &config.scan_ranges {
                    if let Ok(targets) = Self::generate_scan_targets(range) {
                        let discovered_count = Self::scan_network_range(
                            targets,
                            config.communication_port,
                            &discovered_nodes,
                            &event_sender,
                            config.max_concurrent_discoveries,
                        ).await;
                        
                        let _ = event_sender.send(DiscoveryEvent::DiscoveryCompleted(
                            DiscoveryMethod::NetworkScan,
                            discovered_count,
                        ));
                    }
                }
            }
        });

        Ok(())
    }

    /// Start periodic node cleanup
    async fn start_node_cleanup(&self) -> Result<()> {
        let discovered_nodes = self.discovered_nodes.clone();
        let config = self.config.clone();
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut cleanup_interval = interval(Duration::from_secs(config.node_timeout_secs / 2));
            
            loop {
                cleanup_interval.tick().await;
                
                let timeout_duration = Duration::from_secs(config.node_timeout_secs);
                let now = SystemTime::now();
                let mut nodes_to_remove = Vec::new();

                {
                    let nodes = discovered_nodes.read();
                    for (node_id, node) in nodes.iter() {
                        if let Ok(elapsed) = now.duration_since(node.last_seen) {
                            if elapsed > timeout_duration {
                                nodes_to_remove.push(node_id.clone());
                            }
                        }
                    }
                }

                for node_id in nodes_to_remove {
                    discovered_nodes.write().remove(&node_id);
                    let _ = event_sender.send(DiscoveryEvent::NodeLost(node_id));
                }
            }
        });

        Ok(())
    }

    /// Handle discovery messages
    async fn handle_discovery_message(
        message: DiscoveryMessage,
        addr: SocketAddr,
        discovered_nodes: &Arc<RwLock<HashMap<String, DiscoveredNode>>>,
        local_node: &Arc<RwLock<NodeInfo>>,
        event_sender: &broadcast::Sender<DiscoveryEvent>,
        method: DiscoveryMethod,
    ) {
        match message {
            DiscoveryMessage::Announce { node_id, node_info, capabilities, cluster_id: _ } => {
                // Don't process our own announcements
                if node_id == local_node.read().id {
                    return;
                }

                let discovered_node = DiscoveredNode {
                    node_info: node_info.clone(),
                    discovery_method: method,
                    endpoint: addr,
                    last_seen: SystemTime::now(),
                    response_time_ms: 0,
                    capabilities,
                    trust_level: TrustLevel::Discovering,
                };

                let is_new_node = {
                    let mut nodes = discovered_nodes.write();
                    let is_new = !nodes.contains_key(&node_id);
                    nodes.insert(node_id.clone(), discovered_node.clone());
                    is_new
                };

                if is_new_node {
                    let _ = event_sender.send(DiscoveryEvent::NodeDiscovered(node_id, discovered_node));
                } else {
                    let _ = event_sender.send(DiscoveryEvent::NodeUpdated(node_id, discovered_node));
                }
            }
            DiscoveryMessage::Probe { requester_id: _, cluster_id: _ } => {
                // Respond to probes with our information
                let _response = DiscoveryMessage::Response {
                    node_id: local_node.read().id.clone(),
                    node_info: local_node.read().clone(),
                    capabilities: Self::get_local_capabilities(),
                    cluster_id: "loki-cluster".to_string(),
                };

                // Send response back to requester
                // Implementation would send UDP response back to addr
            }
            _ => {
                debug!("Received discovery message: {:?}", message);
            }
        }
    }

    /// Handle TCP connections for detailed node communication
    async fn handle_tcp_connection(
        _stream: TcpStream,
        addr: SocketAddr,
        _discovered_nodes: Arc<RwLock<HashMap<String, DiscoveredNode>>>,
        _local_node: Arc<RwLock<NodeInfo>>,
    ) -> Result<()> {
        // Implementation for handling detailed TCP communication
        // This would include node verification, capability exchange, etc.
        debug!("Handling TCP connection from {}", addr);
        Ok(())
    }

    /// Generate network scan targets from CIDR range
    fn generate_scan_targets(cidr_range: &str) -> Result<Vec<IpAddr>> {
        // Parse CIDR and generate IP addresses to scan
        // For now, return a simple range for common networks
        let mut targets = Vec::new();
        
        if cidr_range.starts_with("192.168.") {
            for i in 1..255 {
                targets.push(IpAddr::V4(Ipv4Addr::new(192, 168, 1, i)));
            }
        }
        
        Ok(targets)
    }

    /// Scan a range of network addresses for Loki instances
    async fn scan_network_range(
        targets: Vec<IpAddr>,
        port: u16,
        _discovered_nodes: &Arc<RwLock<HashMap<String, DiscoveredNode>>>,
        _event_sender: &broadcast::Sender<DiscoveryEvent>,
        max_concurrent: usize,
    ) -> usize {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut discovered_count = 0;

        for target_ip in targets {
            let _permit = semaphore.acquire().await.unwrap();
            let target_addr = SocketAddr::new(target_ip, port);
            
            // Try to connect and probe for Loki instances
            if let Ok(_stream) = timeout(
                Duration::from_millis(500),
                TcpStream::connect(target_addr)
            ).await {
                // Connection successful, this might be a Loki instance
                // Would perform handshake and verification here
                debug!("Found potential Loki instance at {}", target_addr);
                discovered_count += 1;
            }
        }

        discovered_count
    }

    /// Get local node capabilities
    fn get_local_capabilities() -> NodeCapabilities {
        NodeCapabilities {
            loki_version: env!("CARGO_PKG_VERSION").to_string(),
            supported_protocols: vec!["loki-v1".to_string(), "http".to_string()],
            available_models: vec![], // Would be populated from actual model registry
            compute_capabilities: ComputeCapabilities {
                cpu_cores: num_cpus::get() as u32,
                memory_gb: 8.0, // Would detect actual memory
                gpu_count: 1,   // Would detect actual GPUs
                gpu_memory_gb: 8.0,
                inference_performance: 100.0,
                supports_batching: true,
                supports_streaming: true,
            },
            cluster_role: ClusterRole::Hybrid(vec![
                ClusterRole::Worker,
                ClusterRole::Coordinator,
            ]),
        }
    }

    /// Get discovered nodes
    pub fn get_discovered_nodes(&self) -> HashMap<String, DiscoveredNode> {
        self.discovered_nodes.read().clone()
    }

    /// Get discovery events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<DiscoveryEvent> {
        self.event_sender.subscribe()
    }

    /// Manually add a node (for static configuration)
    pub fn add_manual_node(&self, node_id: String, endpoint: SocketAddr, node_info: NodeInfo) {
        let discovered_node = DiscoveredNode {
            node_info,
            discovery_method: DiscoveryMethod::Manual,
            endpoint,
            last_seen: SystemTime::now(),
            response_time_ms: 0,
            capabilities: Self::get_local_capabilities(), // Would probe for actual capabilities
            trust_level: TrustLevel::Verified,
        };

        self.discovered_nodes.write().insert(node_id.clone(), discovered_node.clone());
        let _ = self.event_sender.send(DiscoveryEvent::NodeDiscovered(node_id, discovered_node));
    }

    /// Shutdown the discovery service
    pub async fn shutdown(&self) -> Result<()> {
        info!("🔍 Shutting down network discovery service");
        let _ = self.shutdown_sender.send(()).await;
        Ok(())
    }
}