use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Represents the overall state of the distributed training cluster
#[derive(Debug, Clone, Default)]
pub struct ClusterState {
    /// List of active training jobs
    pub training_jobs: Vec<TrainingJob>,
    
    /// List of cluster nodes with their resource usage
    pub nodes: Vec<ClusterNode>,
    
    /// Overall cluster health status
    pub health_status: ClusterHealthStatus,
    
    /// Last update timestamp
    pub last_update: Option<Instant>,
}

/// Represents a training job in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    /// Unique identifier for the job
    pub id: String,
    
    /// Name of the training job
    pub name: String,
    
    /// Current status of the job
    pub status: JobStatus,
    
    /// Job type/category
    pub job_type: JobType,
    
    /// Node this job is running on
    pub node_id: Option<String>,
    
    /// Progress percentage (0-100)
    pub progress: f32,
    
    /// Estimated time remaining
    pub estimated_time_remaining: Option<Duration>,
    
    /// GPU memory usage in GB
    pub gpu_memory_usage: f32,
    
    /// CPU usage percentage
    pub cpu_usage: f32,
    
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Start timestamp
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Priority level
    pub priority: JobPriority,
}

/// Status of a training job
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    /// Job is queued and waiting for resources
    Queued,
    
    /// Job is pending and will start soon
    Pending,
    
    /// Job is currently running
    Running,
    
    /// Job is paused
    Paused,
    
    /// Job completed successfully
    Completed,
    
    /// Job failed with error
    Failed(String),
    
    /// Job was cancelled
    Cancelled,
}

/// Type of training job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobType {
    /// LLM fine-tuning
    LLMFineTuning,
    
    /// Cognitive model training
    CognitiveModelTraining,
    
    /// Reinforcement learning
    ReinforcementLearning,
    
    /// Custom training job
    Custom(String),
}

/// Priority levels for jobs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Represents a node in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Unique identifier for the node
    pub id: String,
    
    /// Node name
    pub name: String,
    
    /// Node status
    pub status: NodeStatus,
    
    /// Total GPU count
    pub total_gpus: u32,
    
    /// Available GPU count
    pub available_gpus: u32,
    
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    
    /// Memory usage in GB
    pub memory_usage_gb: f32,
    
    /// Total memory in GB
    pub total_memory_gb: f32,
    
    /// Network bandwidth usage in Gbps
    pub network_usage_gbps: f32,
    
    /// Running jobs on this node
    pub running_jobs: Vec<String>,
    
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

/// Status of a cluster node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy and accepting jobs
    Healthy,
    
    /// Node is under high load but still operational
    HighLoad,
    
    /// Node is not accepting new jobs
    Draining,
    
    /// Node is offline
    Offline,
    
    /// Node is in maintenance mode
    Maintenance,
}

/// Overall cluster health status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ClusterHealthStatus {
    /// All systems operational
    #[default]
    Healthy,
    
    /// Some issues but cluster is operational
    Degraded,
    
    /// Major issues affecting cluster operation
    Critical,
    
    /// Cluster is not operational
    Offline,
}

impl ClusterState {
    /// Create a new cluster state
    pub fn new() -> Self {
        Self {
            training_jobs: Vec::new(),
            nodes: Vec::new(),
            health_status: ClusterHealthStatus::Healthy,
            last_update: Some(Instant::now()),
        }
    }
    
    /// Update cluster state with new data
    pub fn update(&mut self, jobs: Vec<TrainingJob>, nodes: Vec<ClusterNode>) {
        self.training_jobs = jobs;
        self.nodes = nodes;
        self.last_update = Some(Instant::now());
        self.update_health_status();
    }
    
    /// Update overall health status based on node states
    fn update_health_status(&mut self) {
        let total_nodes = self.nodes.len();
        if total_nodes == 0 {
            self.health_status = ClusterHealthStatus::Offline;
            return;
        }
        
        let healthy_nodes = self.nodes.iter()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();
        
        let offline_nodes = self.nodes.iter()
            .filter(|n| n.status == NodeStatus::Offline)
            .count();
        
        if offline_nodes == total_nodes {
            self.health_status = ClusterHealthStatus::Offline;
        } else if healthy_nodes == total_nodes {
            self.health_status = ClusterHealthStatus::Healthy;
        } else if offline_nodes > total_nodes / 2 {
            self.health_status = ClusterHealthStatus::Critical;
        } else {
            self.health_status = ClusterHealthStatus::Degraded;
        }
    }
    
    /// Get active job count
    pub fn active_job_count(&self) -> usize {
        self.training_jobs.iter()
            .filter(|j| matches!(j.status, JobStatus::Running | JobStatus::Pending))
            .count()
    }
    
    /// Get total GPU utilization across cluster
    pub fn total_gpu_utilization(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        
        let total: f32 = self.nodes.iter()
            .map(|n| n.gpu_utilization)
            .sum();
        
        total / self.nodes.len() as f32
    }
    
    /// Get total available GPUs
    pub fn total_available_gpus(&self) -> u32 {
        self.nodes.iter()
            .map(|n| n.available_gpus)
            .sum()
    }
}

impl TrainingJob {
    /// Create a new training job
    pub fn new(name: String, job_type: JobType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            status: JobStatus::Queued,
            job_type,
            node_id: None,
            progress: 0.0,
            estimated_time_remaining: None,
            gpu_memory_usage: 0.0,
            cpu_usage: 0.0,
            created_at: chrono::Utc::now(),
            started_at: None,
            priority: JobPriority::Normal,
        }
    }
    
    /// Check if job is active
    pub fn is_active(&self) -> bool {
        matches!(self.status, JobStatus::Running | JobStatus::Pending)
    }
    
    /// Get status color for TUI display
    pub fn status_color(&self) -> ratatui::style::Color {
        use ratatui::style::Color;
        
        match &self.status {
            JobStatus::Running => Color::Green,
            JobStatus::Pending => Color::Yellow,
            JobStatus::Queued => Color::Blue,
            JobStatus::Paused => Color::Magenta,
            JobStatus::Completed => Color::Cyan,
            JobStatus::Failed(_) => Color::Red,
            JobStatus::Cancelled => Color::Gray,
        }
    }
    
    /// Get job type icon for TUI display
    pub fn job_type_icon(&self) -> &'static str {
        match &self.job_type {
            JobType::LLMFineTuning => "📊",
            JobType::CognitiveModelTraining => "🧠",
            JobType::ReinforcementLearning => "🎯",
            JobType::Custom(_) => "🔧",
        }
    }
}

impl ClusterNode {
    /// Create a new cluster node
    pub fn new(id: String, name: String, total_gpus: u32, total_memory_gb: f32) -> Self {
        Self {
            id,
            name,
            status: NodeStatus::Healthy,
            total_gpus,
            available_gpus: total_gpus,
            gpu_utilization: 0.0,
            cpu_utilization: 0.0,
            memory_usage_gb: 0.0,
            total_memory_gb,
            network_usage_gbps: 0.0,
            running_jobs: Vec::new(),
            last_heartbeat: chrono::Utc::now(),
        }
    }
    
    /// Get status color for TUI display
    pub fn status_color(&self) -> ratatui::style::Color {
        use ratatui::style::Color;
        
        match &self.status {
            NodeStatus::Healthy => Color::Green,
            NodeStatus::HighLoad => Color::Yellow,
            NodeStatus::Draining => Color::Magenta,
            NodeStatus::Offline => Color::Red,
            NodeStatus::Maintenance => Color::Blue,
        }
    }
    
    /// Get status icon for TUI display
    pub fn status_icon(&self) -> &'static str {
        match &self.status {
            NodeStatus::Healthy => "🟢",
            NodeStatus::HighLoad => "🟡",
            NodeStatus::Draining => "🟣",
            NodeStatus::Offline => "🔴",
            NodeStatus::Maintenance => "🔵",
        }
    }
    
    /// Calculate memory utilization percentage
    pub fn memory_utilization(&self) -> f32 {
        if self.total_memory_gb > 0.0 {
            (self.memory_usage_gb / self.total_memory_gb) * 100.0
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Queued => write!(f, "Queued"),
            JobStatus::Pending => write!(f, "Pending"),
            JobStatus::Running => write!(f, "Running"),
            JobStatus::Paused => write!(f, "Paused"),
            JobStatus::Completed => write!(f, "Completed"),
            JobStatus::Failed(err) => write!(f, "Failed: {}", err),
            JobStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeStatus::Healthy => write!(f, "Healthy"),
            NodeStatus::HighLoad => write!(f, "High Load"),
            NodeStatus::Draining => write!(f, "Draining"),
            NodeStatus::Offline => write!(f, "Offline"),
            NodeStatus::Maintenance => write!(f, "Maintenance"),
        }
    }
}