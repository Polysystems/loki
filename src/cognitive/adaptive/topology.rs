//! Adaptive Topology
//!
//! Implements the morphing cognitive topology that can reshape itself based on
//! task demands, performance feedback, and evolutionary pressures.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// Removed unused std::sync::Arc import
// RwLock imports removed - not used in current implementation
use chrono::{DateTime, Utc};
use super::{NodeId, EdgeId, CognitiveFunction};

/// Adaptive topology that can reshape itself
#[derive(Clone, Debug)]
pub struct AdaptiveTopology {
    /// Nodes in the cognitive network
    pub nodes: HashMap<NodeId, CognitiveFunction>,

    /// Edges connecting the nodes
    pub edges: HashMap<EdgeId, InformationChannel>,

    /// Topology metrics for performance monitoring
    pub metrics: TopologyMetrics,

    /// Reconfiguration rules (reserved for future rule engine)
    #[allow(dead_code)]
    adaptation_rules: Vec<TopologyRule>,

    /// Configuration history (reserved for future persistence)
    #[allow(dead_code)]
    configuration_history: Vec<TopologySnapshot>,

    /// Current topology state
    current_state: TopologyState,

    /// Current configuration version
    configuration_version: u64,
}

/// Information channel between cognitive nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InformationChannel {
    /// Source node
    pub from: NodeId,

    /// Destination node
    pub to: NodeId,

    /// Channel bandwidth (information units per second)
    pub bandwidth: f64,

    /// Channel latency (seconds)
    pub latency: f64,

    /// Channel reliability (0.0 to 1.0)
    pub reliability: f64,

    /// Current load (0.0 to 1.0)
    pub current_load: f64,

    /// Channel type
    pub channel_type: ChannelType,

    /// Quality of service parameters
    pub qos: QualityOfService,

    /// Channel statistics
    pub stats: ChannelStatistics,

    /// Channel capacity
    pub capacity: u32,

    /// Channel priority
    pub priority: super::Priority,

    /// Current throughput (information units per second)
    pub throughput: f64,

    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

impl InformationChannel {
    pub fn new(from: NodeId, to: NodeId, bandwidth: f64, latency: f64) -> Self {
        Self {
            from,
            to,
            bandwidth,
            latency,
            reliability: 1.0,
            current_load: 0.0,
            channel_type: ChannelType::Standard,
            qos: QualityOfService::default(),
            stats: ChannelStatistics::default(),
            capacity: 0,
            priority: super::Priority::default(),
            throughput: 0.0,
            last_activity: chrono::Utc::now(),
        }
    }
}

/// Types of information channels
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ChannelType {
    /// Standard information flow
    Standard,
    /// High-priority express channel
    Express,
    /// Feedback loop
    Feedback,
    /// Control signals
    Control,
    /// Broadcast to multiple nodes
    Broadcast,
    /// Bidirectional channel
    Bidirectional,
}

/// Quality of service parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityOfService {
    /// Priority level (0-10)
    pub priority: u8,

    /// Guaranteed bandwidth (fraction of total)
    pub guaranteed_bandwidth: f64,

    /// Maximum acceptable latency
    pub max_latency: f64,

    /// Required reliability
    pub min_reliability: f64,
}

impl Default for QualityOfService {
    fn default() -> Self {
        Self {
            priority: 5,
            guaranteed_bandwidth: 0.1,
            max_latency: 0.1,
            min_reliability: 0.95,
        }
    }
}

/// Channel usage statistics
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ChannelStatistics {
    /// Total messages passed
    pub total_messages: u64,

    /// Average message size
    pub avg_message_size: f64,

    /// Total bandwidth used
    pub total_bandwidth_used: f64,

    /// Average latency observed
    pub avg_latency: f64,

    /// Error count
    pub error_count: u64,

    /// Last usage timestamp
    pub last_used: Option<DateTime<Utc>>,
}

/// Metrics for topology performance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyMetrics {
    /// Number of nodes
    pub node_count: usize,

    /// Number of edges
    pub edge_count: usize,

    /// Network density
    pub network_density: f64,

    /// Average processing efficiency
    pub processing_efficiency: f64,

    /// Resource utilization efficiency
    pub resource_efficiency: f64,

    /// Specialization score
    pub specialization_score: f64,

    /// Fault tolerance score
    pub fault_tolerance_score: f64,
}

impl Default for TopologyMetrics {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            network_density: 0.0,
            processing_efficiency: 0.7,
            resource_efficiency: 0.8,
            specialization_score: 0.6,
            fault_tolerance_score: 0.7,
        }
    }
}

/// Rules for topology adaptation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyRule {
    /// Rule identifier
    pub id: String,

    /// Rule name
    pub name: String,

    /// Condition that triggers the rule
    pub trigger_condition: TriggerCondition,

    /// Action to take when triggered
    pub action: TopologyAction,

    /// Rule priority (higher numbers = higher priority)
    pub priority: u32,

    /// Rule activation count
    pub activation_count: u64,

    /// Success rate of this rule
    pub success_rate: f64,
}

/// Conditions that trigger topology adaptation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Performance below threshold
    PerformanceThreshold {
        metric: String,
        threshold: f64,
        comparison: Comparison,
    },

    /// Load above/below threshold
    LoadThreshold {
        node_id: Option<NodeId>,
        threshold: f64,
        comparison: Comparison,
    },

    /// Error rate threshold
    ErrorThreshold {
        threshold: f64,
        time_window: u64, // seconds
    },

    /// New task type detected
    NewTaskType {
        task_pattern: String,
    },

    /// Resource constraints
    ResourceConstraints {
        resource_type: String,
        available: f64,
        required: f64,
    },

    /// Time-based trigger
    TimeBased {
        interval: u64, // seconds
    },

    /// Complex condition (multiple sub-conditions)
    Complex {
        conditions: Vec<TriggerCondition>,
        logic: LogicOperator,
    },
}

/// Comparison operators
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Comparison {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

/// Logic operators for complex conditions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum LogicOperator {
    And,
    Or,
    Not,
    Xor,
}

/// Actions to take when adapting topology
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TopologyAction {
    /// Add a new node
    AddNode {
        node_function: CognitiveFunction,
        connections: Vec<(NodeId, ChannelType)>,
    },

    /// Remove a node
    RemoveNode {
        node_id: NodeId,
        migration_strategy: MigrationStrategy,
    },

    /// Modify node function
    ModifyNode {
        node_id: NodeId,
        new_function: CognitiveFunction,
    },

    /// Add edge
    AddEdge {
        channel: InformationChannel,
    },

    /// Remove edge
    RemoveEdge {
        edge_id: EdgeId,
    },

    /// Modify edge properties
    ModifyEdge {
        edge_id: EdgeId,
        new_properties: ChannelProperties,
    },

    /// Reorganize subnetwork
    Reorganize {
        affected_nodes: Vec<NodeId>,
        new_structure: NetworkStructure,
    },

    /// Split node into multiple nodes
    SplitNode {
        node_id: NodeId,
        split_strategy: SplitStrategy,
    },

    /// Merge multiple nodes
    MergeNodes {
        node_ids: Vec<NodeId>,
        merge_strategy: MergeStrategy,
    },
}

/// Strategies for migrating functionality when removing nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// Distribute to connected nodes
    Distribute,
    /// Move to specific node
    MoveTo(NodeId),
    /// Create new node for functionality
    CreateNew,
    /// Drop functionality (if non-critical)
    Drop,
}

/// Properties for modifying channels
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelProperties {
    pub bandwidth: Option<f64>,
    pub latency: Option<f64>,
    pub reliability: Option<f64>,
    pub channel_type: Option<ChannelType>,
    pub qos: Option<QualityOfService>,
}

/// Network structure templates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NetworkStructure {
    /// Linear chain
    Chain,
    /// Star topology
    Star,
    /// Mesh network
    Mesh,
    /// Hierarchical tree
    Tree,
    /// Small-world network
    SmallWorld,
    /// Scale-free network
    ScaleFree,
    /// Custom structure
    Custom(Vec<(NodeId, Vec<NodeId>)>),
}

/// Strategies for splitting nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Split by function type
    ByFunction,
    /// Split by load
    ByLoad,
    /// Split by specialization
    BySpecialization,
    /// Custom split logic
    Custom(String),
}

/// Strategies for merging nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Combine functions
    CombineFunctions,
    /// Select best function
    SelectBest,
    /// Create hybrid function
    CreateHybrid,
    /// Voting mechanism
    Voting,
}

/// Snapshot of topology state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologySnapshot {
    /// Snapshot identifier
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Node configuration
    pub nodes: HashMap<NodeId, CognitiveFunction>,

    /// Edge configuration
    pub edges: HashMap<EdgeId, InformationChannel>,

    /// Metrics at time of snapshot
    pub metrics: TopologyMetrics,

    /// Reason for snapshot
    pub reason: String,
}

/// Current state of the topology
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyState {
    /// Is topology currently reconfiguring?
    pub is_reconfiguring: bool,

    /// Current configuration version
    pub configuration_version: u64,

    /// Active adaptation rules
    pub active_rules: Vec<String>,

    /// Recent changes
    pub recent_changes: Vec<TopologyChange>,

    /// Performance trend
    pub performance_trend: PerformanceTrend,
}

impl Default for TopologyState {
    fn default() -> Self {
        Self {
            is_reconfiguring: false,
            configuration_version: 1,
            active_rules: Vec::new(),
            recent_changes: Vec::new(),
            performance_trend: PerformanceTrend::Stable,
        }
    }
}

/// Recent topology changes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyChange {
    /// Change identifier
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Type of change
    pub change_type: ChangeType,

    /// Affected nodes/edges
    pub affected_elements: Vec<String>,

    /// Reason for change
    pub reason: String,

    /// Performance impact
    pub performance_impact: f64,
}

/// Types of topology changes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChangeType {
    NodeAdded,
    NodeRemoved,
    NodeModified,
    EdgeAdded,
    EdgeRemoved,
    EdgeModified,
    Reorganization,
    Split,
    Merge,
}

/// Performance trend indicators
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

impl AdaptiveTopology {
    /// Create a new adaptive topology
    pub async fn new() -> Result<Self> {
        Ok(Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            metrics: TopologyMetrics::default(),
            adaptation_rules: Vec::new(),
            configuration_history: Vec::new(),
            current_state: TopologyState::default(),
            configuration_version: 1,
        })
    }

    /// Add a node to the topology
    pub async fn add_node(&mut self, node_id: NodeId, function: CognitiveFunction) -> Result<()> {
        self.nodes.insert(node_id.clone(), function);
        self.update_metrics().await?;

        // Record change
        self.record_change(ChangeType::NodeAdded, vec![node_id.to_string()], "Manual addition".to_string()).await?;

        Ok(())
    }

    /// Remove a node from the topology
    pub async fn remove_node(&mut self, node_id: &NodeId) -> Result<()> {
        // Remove associated edges first
        let edges_to_remove: Vec<EdgeId> = self.edges
            .iter()
            .filter(|(_, channel)| &channel.from == node_id || &channel.to == node_id)
            .map(|(edge_id, _)| edge_id.clone())
            .collect();

        for edge_id in edges_to_remove {
            self.edges.remove(&edge_id);
        }

        // Remove the node
        self.nodes.remove(node_id);
        self.update_metrics().await?;

        // Record change
        self.record_change(ChangeType::NodeRemoved, vec![node_id.to_string()], "Manual removal".to_string()).await?;

        Ok(())
    }

    /// Add an edge to the topology
    pub async fn add_edge(&mut self, edge_id: EdgeId, channel: InformationChannel) -> Result<()> {
        // Verify that both nodes exist
        if !self.nodes.contains_key(&channel.from) || !self.nodes.contains_key(&channel.to) {
            return Err(anyhow::anyhow!("Cannot add edge: one or both nodes do not exist"));
        }

        self.edges.insert(edge_id.clone(), channel);
        self.update_metrics().await?;

        // Record change
        self.record_change(ChangeType::EdgeAdded, vec![edge_id.to_string()], "Manual addition".to_string()).await?;

        Ok(())
    }

    /// Remove an edge from the topology
    pub async fn remove_edge(&mut self, edge_id: &EdgeId) -> Result<()> {
        self.edges.remove(edge_id);
        self.update_metrics().await?;

        // Record change
        self.record_change(ChangeType::EdgeRemoved, vec![edge_id.to_string()], "Manual removal".to_string()).await?;

        Ok(())
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &TopologyMetrics {
        &self.metrics
    }

    /// Update topology metrics
    async fn update_metrics(&mut self) -> Result<()> {
        self.metrics.node_count = self.nodes.len();
        self.metrics.edge_count = self.edges.len();

        // Calculate network density
        let max_edges = if self.nodes.len() > 1 {
            self.nodes.len() * (self.nodes.len() - 1)
        } else {
            1
        };
        self.metrics.network_density = self.edges.len() as f64 / max_edges as f64;

        // Update configuration version
        self.configuration_version += 1;

        Ok(())
    }

    /// Record a topology change
    async fn record_change(&mut self, change_type: ChangeType, affected_elements: Vec<String>, reason: String) -> Result<()> {
        let change = TopologyChange {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            change_type,
            affected_elements,
            reason,
            performance_impact: 0.0, // Would be calculated based on before/after metrics
        };

        self.current_state.recent_changes.push(change);

        // Keep only last 100 changes
        if self.current_state.recent_changes.len() > 100 {
            self.current_state.recent_changes.remove(0);
        }

        // Update configuration version
        self.current_state.configuration_version += 1;

        Ok(())
    }

    /// Get current topology state
    pub fn get_state(&self) -> &TopologyState {
        &self.current_state
    }

    /// Check if topology is currently stable
    pub fn is_stable(&self) -> bool {
        // Simple stability check based on metrics
        self.metrics.processing_efficiency > 0.5 && self.metrics.resource_efficiency > 0.5
    }

    /// Get network efficiency score
    pub fn get_efficiency_score(&self) -> f64 {
        (self.metrics.processing_efficiency + self.metrics.resource_efficiency) / 2.0
    }
}
