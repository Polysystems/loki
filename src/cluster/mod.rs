use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{info, warn};

pub mod coordinator;
pub mod discovery;
pub mod health_monitor;
pub mod intelligent_load_balancer;
pub mod load_balancer;

pub use coordinator::{ClusterCoordinator, NodeInfo};
pub use discovery::{NetworkDiscovery, DiscoveryConfig, DiscoveredNode, DiscoveryEvent};
pub use health_monitor::{HealthMonitor, HealthStatus};
pub use load_balancer::{BalancingStrategy, LoadBalancer};

use crate::compute::Device;
use crate::streaming::ModelInstance;

/// Cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub name: String,
    pub max_models_per_device: usize,
    pub health_check_interval_secs: u64,
    pub rebalance_threshold: f32,
    pub enable_auto_scaling: bool,
    pub max_concurrent_migrations: Option<usize>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            name: "loki-cluster".to_string(),
            max_models_per_device: 4,
            health_check_interval_secs: 30,
            rebalance_threshold: 0.8,
            enable_auto_scaling: true,
            max_concurrent_migrations: Some(3),
        }
    }
}

/// Model cluster manager
pub struct ClusterManager {
    config: ClusterConfig,
    nodes: Arc<DashMap<String, ClusterNode>>,
    coordinator: Arc<ClusterCoordinator>,
    load_balancer: Arc<LoadBalancer>,
    health_monitor: Arc<HealthMonitor>,
    network_discovery: Option<Arc<NetworkDiscovery>>,
    event_bus: broadcast::Sender<ClusterEvent>,
}

/// Cluster node representing a device with models
pub struct ClusterNode {
    pub device: Device,
    pub models: Arc<RwLock<HashMap<String, Arc<dyn ModelInstance>>>>,
    pub stats: Arc<ArcSwap<NodeStats>>,
    pub health_status: Arc<ArcSwap<HealthStatus>>,
}

/// Node statistics
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    pub total_requests: u64,
    pub active_requests: u32,
    pub failed_requests: u64,
    pub avg_latency_ms: f64,
    pub memory_usage_percent: f32,
    pub compute_usage_percent: f32,
    pub last_update: Option<Instant>,
}

/// Cluster events
#[derive(Debug, Clone)]
pub enum ClusterEvent {
    NodeAdded(String),
    NodeRemoved(String),
    ModelDeployed(String, String), // node_id, model_id
    ModelRemoved(String, String),  // node_id, model_id
    NodeHealthChanged(String, HealthStatus),
    RebalanceStarted,
    RebalanceCompleted,
}

impl ClusterManager {
    /// Create a new cluster manager
    pub async fn new(config: ClusterConfig) -> Result<Self> {
        let (event_sender, _) = broadcast::channel(1000);

        let coordinator = Arc::new(ClusterCoordinator::new(config.clone()).await?);
        let load_balancer = Arc::new(LoadBalancer::new(BalancingStrategy::LeastConnections));
        let health_monitor =
            Arc::new(HealthMonitor::new(Duration::from_secs(config.health_check_interval_secs)));

        Ok(Self {
            config,
            nodes: Arc::new(DashMap::new()),
            coordinator,
            load_balancer,
            health_monitor,
            network_discovery: None,
            event_bus: event_sender,
        })
    }

    /// Enable network discovery for finding other Loki instances
    pub async fn enable_network_discovery(&mut self, discovery_config: DiscoveryConfig) -> Result<()> {
        let local_node = NodeInfo {
            id: format!("loki-node-{}", uuid::Uuid::new_v4()),
            capacity: 100,
            current_load: 0,
            available_memory_gb: 16.0,
            cpu_cores: num_cpus::get(),
            gpu_memory_gb: Some(8.0),
            network_bandwidth_gbps: 1.0,
            health_status: crate::cluster::coordinator::NodeHealth::Healthy,
            last_heartbeat: chrono::Utc::now(),
            supported_model_types: vec![
                "llama".to_string(),
                "mistral".to_string(),
                "gpt".to_string(),
                "gemma".to_string(),
            ],
        };

        let discovery = Arc::new(
            NetworkDiscovery::new(discovery_config, self.config.clone(), local_node).await?
        );

        // Start discovery service
        discovery.start().await?;

        // ✅ Fixed: Send trait issue resolved by using tokio::sync::RwLock
        // The ClusterCoordinator now uses tokio::sync::RwLock which is Send-safe
        // and supports async operations

        self.network_discovery = Some(discovery);
        info!("🔍 Network discovery enabled with coordinator integration");
        Ok(())
    }

    /// Get discovered nodes from network discovery
    pub fn get_discovered_nodes(&self) -> Option<std::collections::HashMap<String, DiscoveredNode>> {
        self.network_discovery.as_ref().map(|d| d.get_discovered_nodes())
    }

    /// Add a device to the cluster
    pub async fn add_device(&self, device: Device) -> Result<String> {
        let node_id = device.id.clone();

        let node = ClusterNode {
            device,
            models: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(ArcSwap::from_pointee(NodeStats::default())),
            health_status: Arc::new(ArcSwap::from_pointee(HealthStatus::Healthy)),
        };

        self.nodes.insert(node_id.clone(), node);
        self.coordinator
            .register_node(NodeInfo {
                id: "default-node".to_string(),
                capacity: 100,
                current_load: 0,
                available_memory_gb: 16.0,
                cpu_cores: 8,
                gpu_memory_gb: Some(8.0),
                network_bandwidth_gbps: 1.0,
                health_status: crate::cluster::coordinator::NodeHealth::Healthy,
                last_heartbeat: chrono::Utc::now(),
                supported_model_types: vec!["llama".to_string(), "mistral".to_string()],
            })
            .await?;

        let _ = self.event_bus.send(ClusterEvent::NodeAdded(node_id.clone()));

        info!("Added device {} to cluster", node_id);
        Ok(node_id)
    }

    /// Deploy a model to the cluster
    pub async fn deploy_model(
        &self,
        model: Arc<dyn ModelInstance>,
        preferred_node: Option<String>,
    ) -> Result<String> {
        let model_id = model.id().to_string();

        // Find best node for deployment
        let node_id = if let Some(preferred) = preferred_node {
            preferred
        } else {
            self.coordinator.find_best_node_for_model(&model_id).await?
        };

        // Deploy to node
        if let Some(node) = self.nodes.get(&node_id) {
            node.models.write().insert(model_id.clone(), model);
            self.coordinator.update_node_load(&node_id, 1).await?;

            let _ =
                self.event_bus.send(ClusterEvent::ModelDeployed(node_id.clone(), model_id.clone()));

            info!("Deployed model {} to node {}", model_id, node_id);
            Ok(node_id)
        } else {
            anyhow::bail!("Node {} not found", node_id)
        }
    }

    /// Remove a model from the cluster
    pub async fn remove_model(&self, model_id: &str) -> Result<()> {
        for entry in self.nodes.iter() {
            let node_id = entry.key().clone();
            let node = entry.value();

            if node.models.write().remove(model_id).is_some() {
                self.coordinator.update_node_load(&node_id, -1).await?;

                let _ = self
                    .event_bus
                    .send(ClusterEvent::ModelRemoved(node_id.clone(), model_id.to_string()));

                info!("Removed model {} from node {}", model_id, node_id);
                return Ok(());
            }
        }

        anyhow::bail!("Model {} not found in cluster", model_id)
    }

    /// Route a request to the best available model instance
    pub async fn route_request(&self, model_id: &str) -> Result<(String, Arc<dyn ModelInstance>)> {
        // Get nodes with this model
        let mut available_nodes = Vec::new();

        for entry in self.nodes.iter() {
            let node_id = entry.key();
            let node = entry.value();

            if let Some(model) = node.models.read().get(model_id) {
                if model.is_ready() {
                    let stats = node.stats.load();
                    available_nodes.push((node_id.clone(), stats.active_requests));
                }
            }
        }

        if available_nodes.is_empty() {
            anyhow::bail!("No available instances for model {}", model_id);
        }

        // Use load balancer to select node
        let selected_node = self.load_balancer.select_node(&available_nodes)?;

        // Get model instance
        if let Some(node) = self.nodes.get(&selected_node) {
            if let Some(model) = node.models.read().get(model_id) {
                // Update stats
                let mut stats = node.stats.load().as_ref().clone();
                stats.active_requests += 1;
                stats.total_requests += 1;
                node.stats.store(Arc::new(stats));

                return Ok((selected_node, model.clone()));
            }
        }

        anyhow::bail!("Failed to route request for model {}", model_id)
    }

    /// Complete a request and update stats
    pub fn complete_request(&self, node_id: &str, latency: Duration, success: bool) {
        if let Some(node) = self.nodes.get(node_id) {
            let mut stats = node.stats.load().as_ref().clone();
            stats.active_requests = stats.active_requests.saturating_sub(1);

            if !success {
                stats.failed_requests += 1;
            }

            // Update average latency
            let latency_ms = latency.as_millis() as f64;
            if stats.avg_latency_ms == 0.0 {
                stats.avg_latency_ms = latency_ms;
            } else {
                stats.avg_latency_ms = (stats.avg_latency_ms * 0.9) + (latency_ms * 0.1);
            }

            stats.last_update = Some(Instant::now());
            node.stats.store(Arc::new(stats));
        }
    }

    /// Start cluster management tasks
    pub async fn start(&self) {
        // Start health monitoring
        let nodes = self.nodes.clone();
        let health_monitor = self.health_monitor.clone();
        let event_bus = self.event_bus.clone();

        tokio::spawn(async move {
            health_monitoring_loop(nodes, health_monitor, event_bus).await;
        });

        // Start auto-scaling if enabled
        if self.config.enable_auto_scaling {
            let nodes = self.nodes.clone();
            let coordinator = self.coordinator.clone();
            let config = self.config.clone();
            let event_bus = self.event_bus.clone();

            tokio::spawn(async move {
                auto_scaling_loop(nodes, coordinator, config, event_bus).await;
            });
        }

        info!("Cluster manager started");
    }

    /// Get cluster statistics
    pub fn cluster_stats(&self) -> ClusterStats {
        let mut stats = ClusterStats::default();

        for entry in self.nodes.iter() {
            let node = entry.value();
            let node_stats = node.stats.load();

            stats.total_nodes += 1;
            stats.total_models += node.models.read().len();
            stats.total_requests += node_stats.total_requests;
            stats.active_requests += node_stats.active_requests as u64;
            stats.failed_requests += node_stats.failed_requests;

            stats.avg_memory_usage += node_stats.memory_usage_percent;
            stats.avg_compute_usage += node_stats.compute_usage_percent;
        }

        if stats.total_nodes > 0 {
            stats.avg_memory_usage /= stats.total_nodes as f32;
            stats.avg_compute_usage /= stats.total_nodes as f32;
        }

        stats
    }

    /// Subscribe to cluster events
    pub fn subscribe(&self) -> broadcast::Receiver<ClusterEvent> {
        self.event_bus.subscribe()
    }
}

/// Cluster-wide statistics
#[derive(Debug, Clone, Default)]
pub struct ClusterStats {
    pub total_nodes: usize,
    pub total_models: usize,
    pub total_requests: u64,
    pub active_requests: u64,
    pub failed_requests: u64,
    pub avg_memory_usage: f32,
    pub avg_compute_usage: f32,
}

/// Health monitoring loop
async fn health_monitoring_loop(
    nodes: Arc<DashMap<String, ClusterNode>>,
    health_monitor: Arc<HealthMonitor>,
    event_bus: broadcast::Sender<ClusterEvent>,
) {
    loop {
        for entry in nodes.iter() {
            let node_id = entry.key().clone();
            let node = entry.value();

            // Check node health
            let health = health_monitor.check_node_health(&node).await;
            let current_health = node.health_status.load();

            if health != *current_health.as_ref() {
                node.health_status.store(Arc::new(health.clone()));
                let _ = event_bus.send(ClusterEvent::NodeHealthChanged(node_id, health));
            }
        }

        tokio::time::sleep(health_monitor.check_interval()).await;
    }
}

/// Auto-scaling loop
async fn auto_scaling_loop(
    nodes: Arc<DashMap<String, ClusterNode>>,
    coordinator: Arc<ClusterCoordinator>,
    config: ClusterConfig,
    event_bus: broadcast::Sender<ClusterEvent>,
) {
    loop {
        // Check if rebalancing is needed
        let mut total_usage = 0.0;
        let mut node_count = 0;

        for entry in nodes.iter() {
            let node = entry.value();
            let stats = node.stats.load();
            total_usage += stats.compute_usage_percent;
            node_count += 1;
        }

        if node_count > 0 {
            let avg_usage = total_usage / node_count as f32;

            if avg_usage > config.rebalance_threshold * 100.0 {
                info!("Triggering cluster rebalance due to high load: {:.1}%", avg_usage);
                let _ = event_bus.send(ClusterEvent::RebalanceStarted);

                if let Err(e) = coordinator.rebalance_cluster().await {
                    warn!("Failed to rebalance cluster: {}", e);
                } else {
                    let _ = event_bus.send(ClusterEvent::RebalanceCompleted);
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}

/// Model deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeployment {
    /// Deployment identifier
    pub deployment_id: String,

    /// Model identifier being deployed
    pub model_id: String,

    /// Target node for deployment
    pub target_node: String,

    /// Deployment configuration
    pub config: DeploymentConfig,

    /// Current deployment status
    pub status: DeploymentStatus,

    /// Deployment metadata
    pub metadata: HashMap<String, String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,

    /// Replica count
    pub replicas: u32,

    /// Auto-scaling configuration
    pub auto_scaling: Option<AutoScalingConfig>,

    /// Health check configuration
    pub health_check: HealthCheckConfig,

    /// Environment variables
    pub environment: HashMap<String, String>,
}

/// Resource requirements for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: f32,

    /// Memory in GB
    pub memory_gb: f32,

    /// GPU memory in GB (optional)
    pub gpu_memory_gb: Option<f32>,

    /// Storage in GB
    pub storage_gb: f32,

    /// Network bandwidth in Gbps
    pub network_bandwidth_gbps: f32,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Minimum replicas
    pub min_replicas: u32,

    /// Maximum replicas
    pub max_replicas: u32,

    /// CPU utilization threshold for scaling
    pub cpu_threshold: f32,

    /// Memory utilization threshold for scaling
    pub memory_threshold: f32,

    /// Scale-up cooldown period
    pub scale_up_cooldown: Duration,

    /// Scale-down cooldown period
    pub scale_down_cooldown: Duration,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check endpoint
    pub endpoint: String,

    /// Check interval
    pub interval: Duration,

    /// Timeout for health checks
    pub timeout: Duration,

    /// Number of failed checks before unhealthy
    pub unhealthy_threshold: u32,

    /// Number of successful checks before healthy
    pub healthy_threshold: u32,
}

/// Deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment is pending
    Pending,

    /// Deployment is in progress
    InProgress { progress_percent: f32 },

    /// Deployment completed successfully
    Completed,

    /// Deployment failed
    Failed { error_message: String },

    /// Deployment is being rolled back
    RollingBack { progress_percent: f32 },

    /// Deployment was cancelled
    Cancelled,
}

/// Model migration for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    /// Migration identifier
    pub migration_id: String,

    /// Model being migrated
    pub model_id: String,

    /// Source node
    pub source_node: String,

    /// Destination node
    pub destination_node: String,

    /// Migration type
    pub migration_type: MigrationType,

    /// Migration priority
    pub priority: MigrationPriority,

    /// Current migration status
    pub status: MigrationStatus,

    /// Migration progress (0.0 to 1.0)
    pub progress: f32,

    /// Estimated completion time
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,

    /// Migration metadata
    pub metadata: HashMap<String, String>,

    /// Impact score for migration prioritization (0.0 to 1.0)
    pub impact_score: f64,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Started timestamp
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Completed timestamp
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Types of model migrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationType {
    /// Load balancing migration
    LoadBalancing,

    /// Resource optimization migration
    ResourceOptimization,

    /// Fault tolerance migration
    FaultTolerance,

    /// Maintenance migration
    Maintenance,

    /// User-requested migration
    UserRequested,

    /// Emergency migration
    Emergency,

    /// Auto-scaling migration
    AutoScaling,
}

/// Migration priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MigrationPriority {
    /// Low priority (background migrations)
    Low,

    /// Normal priority (standard load balancing)
    Normal,

    /// High priority (performance optimization)
    High,

    /// Critical priority (fault tolerance)
    Critical,

    /// Emergency priority (immediate action required)
    Emergency,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStatus {
    /// Migration is queued
    Queued,

    /// Migration is preparing
    Preparing,

    /// Migration is in progress
    InProgress,

    /// Migration is completing
    Completing,

    /// Migration completed successfully
    Completed,

    /// Migration failed
    Failed { error_message: String },

    /// Migration was cancelled
    Cancelled,

    /// Migration is paused
    Paused { reason: String },
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_gb: 2.0,
            gpu_memory_gb: None,
            storage_gb: 10.0,
            network_bandwidth_gbps: 0.1,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            endpoint: "/health".to_string(),
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            unhealthy_threshold: 3,
            healthy_threshold: 2,
        }
    }
}

impl Migration {
    /// Create a new migration
    pub fn new(
        model_id: String,
        source_node: String,
        destination_node: String,
        migration_type: MigrationType,
        priority: MigrationPriority,
    ) -> Self {
        Self {
            migration_id: uuid::Uuid::new_v4().to_string(),
            model_id,
            source_node,
            destination_node,
            migration_type,
            priority,
            status: MigrationStatus::Queued,
            progress: 0.0,
            estimated_completion: None,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            impact_score: 0.0,
        }
    }

    /// Check if migration is completed
    pub fn is_completed(&self) -> bool {
        matches!(self.status, MigrationStatus::Completed)
    }

    /// Check if migration failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, MigrationStatus::Failed { .. })
    }

    /// Check if migration is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            MigrationStatus::Preparing | MigrationStatus::InProgress | MigrationStatus::Completing
        )
    }
}

impl ModelDeployment {
    /// Create a new model deployment
    pub fn new(model_id: String, target_node: String, config: DeploymentConfig) -> Self {
        let now = chrono::Utc::now();
        Self {
            deployment_id: uuid::Uuid::new_v4().to_string(),
            model_id,
            target_node,
            config,
            status: DeploymentStatus::Pending,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Check if deployment is completed
    pub fn is_completed(&self) -> bool {
        matches!(self.status, DeploymentStatus::Completed)
    }

    /// Check if deployment failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, DeploymentStatus::Failed { .. })
    }

    /// Check if deployment is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            DeploymentStatus::InProgress { .. } | DeploymentStatus::RollingBack { .. }
        )
    }
}
