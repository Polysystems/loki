//! Agent coordination and task distribution
//! 
//! This module handles the coordination of multiple agents,
//! task distribution, and workload management.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tokio::sync::{RwLock, mpsc};
use tokio::time::{Duration, interval};
use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use tracing::{debug, warn, error};

use crate::cognitive::agents::{AgentConfiguration, AgentSpecialization, LoadBalancingStrategy};
use crate::compute::ComputeManager;
use crate::memory::CognitiveMemory;
use crate::ollama::OllamaManager;
use super::specialization::{SpecializationRegistry, TaskType};

/// Agent coordinator for managing multi-agent systems
pub struct AgentCoordinator {
    /// Agent pool
    agents: RwLock<HashMap<String, Agent>>,
    
    /// Task queue
    task_queue: RwLock<VecDeque<CoordinationTask>>,
    
    /// Active tasks
    active_tasks: RwLock<HashMap<String, ActiveTask>>,
    
    /// Coordination configuration
    config: CoordinationConfig,
    
    /// Specialization registry
    specializations: Arc<SpecializationRegistry>,
    
    /// Event channel for coordination events
    event_tx: mpsc::Sender<CoordinationEvent>,
    event_rx: RwLock<mpsc::Receiver<CoordinationEvent>>,
    
    /// Event bus for system-wide events
    event_bus: Option<Arc<EventBus>>,
}

/// Individual agent in the system
#[derive(Debug, Clone)]
pub struct Agent {
    /// Unique agent ID
    pub id: String,
    
    /// Agent name
    pub name: String,
    
    /// Specialization
    pub specialization: AgentSpecialization,
    
    /// Current state
    pub state: AgentState,
    
    /// Performance metrics
    pub metrics: AgentMetrics,
    
    /// Current workload (0.0 - 1.0)
    pub workload: f32,
    
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    
    /// Active task IDs
    pub active_tasks: Vec<String>,
}

impl Agent {
    /// Create an agent from configuration
    pub(crate) async fn from_config(
        config: AgentConfiguration,
        _ollama: Arc<OllamaManager>,
        _memory: Arc<CognitiveMemory>,
        _compute: Arc<ComputeManager>,
    ) -> Result<Self> {
        Ok(Self {
            id: config.id.clone(),
            name: config.name.clone(),
            specialization: config.specialization.clone(),
            state: AgentState::Idle,
            metrics: AgentMetrics::default(),
            workload: 0.0,
            max_concurrent_tasks: config.max_concurrent_tasks.unwrap_or(5),
            active_tasks: Vec::new(),
        })
    }
}

/// Agent state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Working,
    Overloaded,
    Unavailable,
    Failed,
}

/// Agent performance metrics
#[derive(Debug, Clone)]
pub struct AgentMetrics {
    /// Total tasks completed
    pub tasks_completed: u64,
    
    /// Total tasks failed
    pub tasks_failed: u64,
    
    /// Average task completion time
    pub avg_completion_time_ms: u64,
    
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
    
    /// Quality score (0.0 - 1.0)
    pub quality_score: f32,
    
    /// Last active timestamp
    pub last_active: std::time::Instant,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            tasks_completed: 0,
            tasks_failed: 0,
            avg_completion_time_ms: 0,
            success_rate: 1.0,
            quality_score: 1.0,
            last_active: std::time::Instant::now(),
        }
    }
}

/// Task for coordination
#[derive(Debug, Clone)]
pub struct CoordinationTask {
    /// Task ID
    pub id: String,
    
    /// Task type
    pub task_type: TaskType,
    
    /// Task description
    pub description: String,
    
    /// Required specializations (any of these)
    pub required_specializations: Vec<AgentSpecialization>,
    
    /// Priority (higher = more important)
    pub priority: f32,
    
    /// Deadline (optional)
    pub deadline: Option<std::time::Instant>,
    
    /// Context data
    pub context: HashMap<String, String>,
    
    /// Dependencies on other tasks
    pub dependencies: Vec<String>,
}

/// Active task being processed
#[derive(Debug, Clone)]
struct ActiveTask {
    /// The task
    task: CoordinationTask,
    
    /// Assigned agent ID
    agent_id: String,
    
    /// Start time
    started_at: std::time::Instant,
    
    /// Current status
    status: TaskStatus,
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq)]
enum TaskStatus {
    Assigned,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Coordination configuration
#[derive(Debug, Clone)]
pub struct CoordinationConfig {
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    
    /// Maximum queue size
    pub max_queue_size: usize,
    
    /// Task timeout
    pub task_timeout: Duration,
    
    /// Agent idle timeout before marking unavailable
    pub agent_idle_timeout: Duration,
    
    /// Workload threshold for overloaded state
    pub overload_threshold: f32,
    
    /// Enable automatic scaling
    pub auto_scaling: bool,
    
    /// Minimum agents per specialization
    pub min_agents_per_spec: usize,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            load_balancing: LoadBalancingStrategy::DynamicPriority,
            max_queue_size: 1000,
            task_timeout: Duration::from_secs(300),
            agent_idle_timeout: Duration::from_secs(600),
            overload_threshold: 0.8,
            auto_scaling: true,
            min_agents_per_spec: 1,
        }
    }
}

/// Coordination events
#[derive(Debug, Clone)]
pub enum CoordinationEvent {
    /// Task assigned to agent
    TaskAssigned { task_id: String, agent_id: String },
    
    /// Task completed
    TaskCompleted { task_id: String, agent_id: String, duration_ms: u64 },
    
    /// Task failed
    TaskFailed { task_id: String, agent_id: String, error: String },
    
    /// Agent state changed
    AgentStateChanged { agent_id: String, old_state: AgentState, new_state: AgentState },
    
    /// Queue size alert
    QueueAlert { size: usize, threshold: usize },
    
    /// Performance alert
    PerformanceAlert { agent_id: String, metric: String, value: f32 },
}

impl AgentCoordinator {
    /// Create a new agent coordinator
    pub fn new(config: CoordinationConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1000);
        
        Self {
            agents: RwLock::new(HashMap::new()),
            task_queue: RwLock::new(VecDeque::new()),
            active_tasks: RwLock::new(HashMap::new()),
            config,
            specializations: Arc::new(SpecializationRegistry::new()),
            event_tx,
            event_rx: RwLock::new(event_rx),
            event_bus: None,
        }
    }
    
    /// Set the event bus for broadcasting agent events
    pub fn set_event_bus(&mut self, event_bus: Arc<EventBus>) {
        self.event_bus = Some(event_bus);
    }
    
    /// Broadcast agent creation event
    async fn broadcast_agent_created(&self, agent_id: &str, specialization: &AgentSpecialization) {
        if let Some(ref event_bus) = self.event_bus {
            if let Err(e) = event_bus.publish(SystemEvent::AgentCreated {
                agent_id: agent_id.to_string(),
                specialization: specialization.clone(),
                config: serde_json::json!({}),
            }).await {
                warn!("Failed to broadcast agent creation event: {}", e);
            }
        }
    }
    
    /// Broadcast task assignment event
    async fn broadcast_task_assigned(&self, agent_id: &str, task: &CoordinationTask) {
        if let Some(ref event_bus) = self.event_bus {
            if let Err(e) = event_bus.publish(SystemEvent::AgentTaskAssigned {
                agent_id: agent_id.to_string(),
                task_id: task.id.clone(),
                task_description: task.description.clone(),
            }).await {
                warn!("Failed to broadcast task assignment event: {}", e);
            }
        }
    }
    
    /// Broadcast agent status change
    async fn broadcast_agent_status(&self, agent_id: &str, status: AgentState) {
        if let Some(ref event_bus) = self.event_bus {
            if let Err(e) = event_bus.publish(SystemEvent::AgentStatusChanged {
                agent_id: agent_id.to_string(),
                status: format!("{:?}", status),
            }).await {
                warn!("Failed to broadcast agent status change: {}", e);
            }
        }
    }
    
    /// Register an agent
    pub async fn register_agent(
        &self,
        id: String,
        name: String,
        specialization: AgentSpecialization,
        max_concurrent: usize,
    ) -> Result<()> {
        debug!("Registering agent: {} ({:?})", id, specialization);
        let mut agents = self.agents.write().await;
        
        let agent = Agent {
            id: id.clone(),
            name,
            specialization: specialization.clone(),
            state: AgentState::Idle,
            metrics: AgentMetrics::default(),
            workload: 0.0,
            max_concurrent_tasks: max_concurrent,
            active_tasks: Vec::new(),
        };
        
        agents.insert(id.clone(), agent);
        
        // Broadcast agent creation event
        self.broadcast_agent_created(&id, &specialization).await;
        
        Ok(())
    }
    
    /// Submit a task for processing
    pub async fn submit_task(&self, task: CoordinationTask) -> Result<()> {
        let mut queue = self.task_queue.write().await;
        
        if queue.len() >= self.config.max_queue_size {
            error!("Task queue is full: {} tasks (max: {})", queue.len(), self.config.max_queue_size);
            return Err(anyhow!("Task queue is full"));
        }
        
        // Check dependencies
        if !task.dependencies.is_empty() {
            let active = self.active_tasks.read().await;
            for dep in &task.dependencies {
                if !active.contains_key(dep) {
                    warn!("Task {} has missing dependency: {}", task.id, dep);
                    return Err(anyhow!("Dependency {} not found", dep));
                }
            }
        }
        
        queue.push_back(task);
        
        // Alert if queue is getting large
        if queue.len() > self.config.max_queue_size * 8 / 10 {
            warn!("Task queue reaching capacity: {} tasks (80% of max {})", queue.len(), self.config.max_queue_size);
            let _ = self.event_tx.send(CoordinationEvent::QueueAlert {
                size: queue.len(),
                threshold: self.config.max_queue_size,
            }).await;
        }
        
        Ok(())
    }
    
    /// Process the task queue
    pub async fn process_queue(&self) -> Result<()> {
        let mut queue = self.task_queue.write().await;
        let queue_size = queue.len();
        if queue_size > 0 {
            debug!("Processing task queue with {} tasks", queue_size);
        }
        let mut processed = Vec::new();
        
        // Try to assign each task
        for (idx, task) in queue.iter().enumerate() {
            if let Some(agent_id) = self.find_suitable_agent(task).await? {
                if self.assign_task_to_agent(task.clone(), agent_id.clone()).await.is_ok() {
                    processed.push(idx);
                    
                    let _ = self.event_tx.send(CoordinationEvent::TaskAssigned {
                        task_id: task.id.clone(),
                        agent_id,
                    }).await;
                }
            }
        }
        
        // Remove assigned tasks from queue
        for idx in processed.into_iter().rev() {
            queue.remove(idx);
        }
        
        Ok(())
    }
    
    /// Find suitable agent for a task
    async fn find_suitable_agent(&self, task: &CoordinationTask) -> Result<Option<String>> {
        let agents = self.agents.read().await;
        
        // Filter eligible agents
        let eligible: Vec<(&String, &Agent)> = agents.iter()
            .filter(|(_, agent)| {
                agent.state == AgentState::Idle || agent.state == AgentState::Working
            })
            .filter(|(_, agent)| {
                agent.active_tasks.len() < agent.max_concurrent_tasks
            })
            .filter(|(_, agent)| {
                task.required_specializations.is_empty() ||
                task.required_specializations.contains(&agent.specialization)
            })
            .collect();
        
        if eligible.is_empty() {
            return Ok(None);
        }
        
        // Apply load balancing strategy
        let selected = match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin (pick first available)
                eligible.first().map(|(id, _)| (*id).clone())
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Pick agent with lowest workload
                eligible.iter()
                    .min_by(|(_, a), (_, b)| {
                        a.workload.partial_cmp(&b.workload).unwrap()
                    })
                    .map(|(id, _)| (*id).clone())
            }
            LoadBalancingStrategy::CapabilityOptimal => {
                // Pick agent best suited for the task
                eligible.iter()
                    .max_by_key(|(_, agent)| {
                        let score = self.specializations
                            .calculate_performance_score(&agent.specialization, task.task_type);
                        (score * 1000.0) as u32
                    })
                    .map(|(id, _)| (*id).clone())
            }
            LoadBalancingStrategy::DynamicPriority => {
                // Consider workload, performance, and capability
                eligible.iter()
                    .max_by_key(|(_, agent)| {
                        let capability_score = self.specializations
                            .calculate_performance_score(&agent.specialization, task.task_type);
                        let workload_score = 1.0 - agent.workload;
                        let performance_score = agent.metrics.success_rate * agent.metrics.quality_score;
                        
                        let combined = (capability_score * 0.4 + 
                                       workload_score * 0.3 + 
                                       performance_score * 0.3) * 1000.0;
                        combined as u32
                    })
                    .map(|(id, _)| (*id).clone())
            }
            LoadBalancingStrategy::PerformanceBased => {
                agents.iter()
                    .max_by(|a, b| {
                        let a_score = a.1.metrics.success_rate * a.1.metrics.quality_score;
                        let b_score = b.1.metrics.success_rate * b.1.metrics.quality_score;
                        a_score.partial_cmp(&b_score).unwrap()
                    })
                    .map(|(id, _)| (*id).clone())
            }
            LoadBalancingStrategy::Random => {
                use rand::seq::IteratorRandom;
                let mut rng = rand::thread_rng();
                agents.iter()
                    .choose(&mut rng)
                    .map(|(id, _)| (*id).clone())
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                agents.iter()
                    .max_by(|a, b| {
                        let a_score = a.1.metrics.success_rate * a.1.metrics.quality_score;
                        let b_score = b.1.metrics.success_rate * b.1.metrics.quality_score;
                        a_score.partial_cmp(&b_score).unwrap()
                    })
                    .map(|(id, _)| (*id).clone())
            }
        };
        
        Ok(selected)
    }
    
    /// Assign task to agent
    async fn assign_task_to_agent(&self, task: CoordinationTask, agent_id: String) -> Result<()> {
        let mut agents = self.agents.write().await;
        let mut active_tasks = self.active_tasks.write().await;
        
        let agent = agents.get_mut(&agent_id)
            .ok_or_else(|| anyhow!("Agent not found"))?;
        
        // Update agent state
        agent.active_tasks.push(task.id.clone());
        agent.workload = agent.active_tasks.len() as f32 / agent.max_concurrent_tasks as f32;
        
        if agent.workload >= self.config.overload_threshold {
            let old_state = agent.state;
            agent.state = AgentState::Overloaded;
            
            let _ = self.event_tx.send(CoordinationEvent::AgentStateChanged {
                agent_id: agent_id.clone(),
                old_state,
                new_state: agent.state,
            }).await;
        } else if agent.state == AgentState::Idle {
            agent.state = AgentState::Working;
        }
        
        // Record active task  
        let task_clone = task.clone();
        active_tasks.insert(task.id.clone(), ActiveTask {
            task,
            agent_id: agent_id.clone(),
            started_at: std::time::Instant::now(),
            status: TaskStatus::Assigned,
        });
        
        // Broadcast task assignment event
        self.broadcast_task_assigned(&agent_id, &task_clone).await;
        
        Ok(())
    }
    
    /// Complete a task
    pub async fn complete_task(&self, task_id: &str, success: bool) -> Result<()> {
        let mut active_tasks = self.active_tasks.write().await;
        let mut agents = self.agents.write().await;
        
        let active_task = active_tasks.remove(task_id)
            .ok_or_else(|| anyhow!("Task not found"))?;
        
        let duration_ms = active_task.started_at.elapsed().as_millis() as u64;
        
        // Update agent
        if let Some(agent) = agents.get_mut(&active_task.agent_id) {
            agent.active_tasks.retain(|id| id != task_id);
            agent.workload = agent.active_tasks.len() as f32 / agent.max_concurrent_tasks as f32;
            
            // Update metrics
            if success {
                agent.metrics.tasks_completed += 1;
            } else {
                agent.metrics.tasks_failed += 1;
            }
            
            // Update average completion time
            let total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed;
            agent.metrics.avg_completion_time_ms = 
                (agent.metrics.avg_completion_time_ms * (total_tasks - 1) + duration_ms) / total_tasks;
            
            // Update success rate
            agent.metrics.success_rate = 
                agent.metrics.tasks_completed as f32 / total_tasks as f32;
            
            agent.metrics.last_active = std::time::Instant::now();
            
            // Update state
            if agent.active_tasks.is_empty() {
                agent.state = AgentState::Idle;
            } else if agent.workload < self.config.overload_threshold {
                agent.state = AgentState::Working;
            }
        }
        
        // Send event
        let event = if success {
            CoordinationEvent::TaskCompleted {
                task_id: task_id.to_string(),
                agent_id: active_task.agent_id,
                duration_ms,
            }
        } else {
            CoordinationEvent::TaskFailed {
                task_id: task_id.to_string(),
                agent_id: active_task.agent_id,
                error: "Task failed".to_string(),
            }
        };
        
        let _ = self.event_tx.send(event).await;
        
        Ok(())
    }
    
    /// Get agent statistics
    pub async fn get_agent_stats(&self) -> HashMap<String, AgentStats> {
        let agents = self.agents.read().await;
        
        agents.iter()
            .map(|(id, agent)| {
                let stats = AgentStats {
                    agent_id: id.clone(),
                    name: agent.name.clone(),
                    specialization: agent.specialization.clone(),
                    state: agent.state,
                    workload: agent.workload,
                    active_tasks: agent.active_tasks.len(),
                    total_completed: agent.metrics.tasks_completed,
                    success_rate: agent.metrics.success_rate,
                    avg_completion_time_ms: agent.metrics.avg_completion_time_ms,
                };
                (id.clone(), stats)
            })
            .collect()
    }
    
    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        let queue = self.task_queue.read().await;
        let active = self.active_tasks.read().await;
        
        let by_type = queue.iter()
            .fold(HashMap::new(), |mut acc, task| {
                *acc.entry(task.task_type).or_insert(0) += 1;
                acc
            });
        
        QueueStats {
            queued_tasks: queue.len(),
            active_tasks: active.len(),
            tasks_by_type: by_type,
            oldest_task_age_ms: queue.front()
                .and_then(|t| t.deadline)
                .map(|d| d.elapsed().as_millis() as u64),
        }
    }
    
    /// Start monitoring loop
    pub async fn start_monitoring(self: Arc<Self>) {
        let mut interval = interval(Duration::from_secs(60));
        
        tokio::spawn(async move {
            loop {
                interval.tick().await;
                
                // Check agent health
                let agents = self.agents.read().await;
                for (id, agent) in agents.iter() {
                    // Check if agent is idle too long
                    if agent.state == AgentState::Idle &&
                       agent.metrics.last_active.elapsed() > self.config.agent_idle_timeout {
                        warn!("Agent {} idle for too long, marking as unavailable", id);
                        let _ = self.event_tx.send(CoordinationEvent::AgentStateChanged {
                            agent_id: id.clone(),
                            old_state: agent.state,
                            new_state: AgentState::Unavailable,
                        }).await;
                    }
                    
                    // Check performance
                    if agent.metrics.success_rate < 0.5 {
                        error!("Agent {} has low success rate: {:.2}%", id, agent.metrics.success_rate * 100.0);
                        let _ = self.event_tx.send(CoordinationEvent::PerformanceAlert {
                            agent_id: id.clone(),
                            metric: "success_rate".to_string(),
                            value: agent.metrics.success_rate,
                        }).await;
                    }
                }
                drop(agents);
                
                // Process queue
                let _ = self.process_queue().await;
            }
        });
    }
    
    /// Get statistics for all agents
    pub async fn get_stats(&self) -> Vec<AgentStats> {
        let agents = self.agents.read().await;
        agents.values().map(|agent| AgentStats {
            agent_id: agent.id.clone(),
            name: agent.name.clone(),
            specialization: agent.specialization.clone(),
            state: agent.state,
            workload: agent.workload,
            active_tasks: agent.active_tasks.len(),
            total_completed: agent.metrics.tasks_completed,
            success_rate: agent.metrics.success_rate,
            avg_completion_time_ms: agent.metrics.avg_completion_time_ms,
        }).collect()
    }
}

/// Agent statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub agent_id: String,
    pub name: String,
    pub specialization: AgentSpecialization,
    pub state: AgentState,
    pub workload: f32,
    pub active_tasks: usize,
    pub total_completed: u64,
    pub success_rate: f32,
    pub avg_completion_time_ms: u64,
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub queued_tasks: usize,
    pub active_tasks: usize,
    pub tasks_by_type: HashMap<TaskType, usize>,
    pub oldest_task_age_ms: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_agent_registration() {
        let coordinator = AgentCoordinator::new(CoordinationConfig::default());
        
        coordinator.register_agent(
            "agent1".to_string(),
            "Analytical Agent 1".to_string(),
            AgentSpecialization::Analytical,
            3,
        ).await.unwrap();
        
        let stats = coordinator.get_agent_stats().await;
        assert_eq!(stats.len(), 1);
        assert!(stats.contains_key("agent1"));
    }
    
    #[tokio::test]
    async fn test_task_submission() {
        let coordinator = AgentCoordinator::new(CoordinationConfig::default());
        
        let task = CoordinationTask {
            id: "task1".to_string(),
            task_type: TaskType::Analysis,
            description: "Analyze data".to_string(),
            required_specializations: vec![AgentSpecialization::Analytical],
            priority: 0.8,
            deadline: None,
            context: HashMap::new(),
            dependencies: vec![],
        };
        
        coordinator.submit_task(task).await.unwrap();
        
        let stats = coordinator.get_queue_stats().await;
        assert_eq!(stats.queued_tasks, 1);
    }
}