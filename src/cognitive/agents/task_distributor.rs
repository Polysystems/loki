//! Task Distributor
//!
//! Implements intelligent task distribution among agents with various
//! load balancing strategies and performance optimization.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Result, anyhow};
use rand::prelude::*;
use rand_distr::{WeightedIndex, Distribution};
// thread_rng removed - using rand::thread_rng() instead
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

use crate::models::agent_specialization_router::AgentId;
use super::{Task, TaskPriority, AgentEntry};
#[cfg(test)]
use super::AgentCapability;

/// Task distributor for multi-agent systems
pub struct TaskDistributor {
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,

    /// Task queue
    task_queue: Arc<RwLock<VecDeque<Task>>>,

    /// Active allocations
    active_allocations: Arc<RwLock<HashMap<String, TaskAllocation>>>,

    /// Distribution metrics
    metrics: Arc<RwLock<DistributionMetrics>>,

    /// Performance history
    performance_history: Arc<RwLock<HashMap<AgentId, AgentPerformance>>>,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,

    /// Least loaded agent first
    LeastLoaded,

    /// Performance-based distribution
    PerformanceBased,

    /// Capability matching
    CapabilityOptimal,

    /// Dynamic priority-based
    DynamicPriority,

    /// Random distribution
    Random,

    /// Weighted round-robin
    WeightedRoundRobin,
}

/// Task allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAllocation {
    pub task_id: String,
    pub assigned_agents: Vec<AgentId>,
    pub allocation_strategy: LoadBalancingStrategy,
    pub estimated_completion: Duration,
    pub confidence: f32,
    pub timestamp: SystemTime,
}

/// Agent performance metrics
#[derive(Debug, Clone)]
pub struct AgentPerformance {
    pub agent_id: AgentId,
    pub tasks_completed: u64,
    pub success_rate: f32,
    pub average_completion_time: Duration,
    pub specialization_scores: HashMap<String, f32>,
    pub current_load: f32,
    pub reliability_score: f32,
}

/// Distribution metrics
#[derive(Debug, Clone, Default)]
pub struct DistributionMetrics {
    pub total_tasks_distributed: u64,
    pub successful_allocations: u64,
    pub failed_allocations: u64,
    pub average_allocation_time: Duration,
    pub load_balance_score: f32,
    pub strategy_effectiveness: HashMap<LoadBalancingStrategy, f32>,
}

impl TaskDistributor {
    /// Create a new task distributor
    pub async fn new(strategy: LoadBalancingStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            task_queue: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            active_allocations: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(DistributionMetrics::default())),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Allocate a task to agents
    pub async fn allocate_task(
        &self,
        task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<TaskAllocation> {
        let start_time = SystemTime::now();

        if available_agents.is_empty() {
            return Err(anyhow!("No agents available for task allocation"));
        }

        // Select agents based on strategy
        let selected_agents = match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.round_robin_selection(task, available_agents).await?
            }
            LoadBalancingStrategy::LeastLoaded => {
                self.least_loaded_selection(task, available_agents).await?
            }
            LoadBalancingStrategy::PerformanceBased => {
                self.performance_based_selection(task, available_agents).await?
            }
            LoadBalancingStrategy::CapabilityOptimal => {
                self.capability_optimal_selection(task, available_agents).await?
            }
            LoadBalancingStrategy::DynamicPriority => {
                self.dynamic_priority_selection(task, available_agents).await?
            }
            LoadBalancingStrategy::Random => {
                self.random_selection(task, available_agents).await?
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_selection(task, available_agents).await?
            }
        };

        // Create allocation
        let allocation = TaskAllocation {
            task_id: task.id.clone(),
            assigned_agents: selected_agents,
            allocation_strategy: self.strategy.clone(),
            estimated_completion: self.estimate_completion_time(task, available_agents).await?,
            confidence: self.calculate_allocation_confidence(task, available_agents).await?,
            timestamp: SystemTime::now(),
        };

        // Store allocation
        self.active_allocations.write().await.insert(task.id.clone(), allocation.clone());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_tasks_distributed += 1;
        metrics.successful_allocations += 1;
        let elapsed = start_time.elapsed().unwrap_or_default();
        metrics.average_allocation_time = Duration::from_millis(
            (metrics.average_allocation_time.as_millis() as u64 * (metrics.total_tasks_distributed - 1)
            + elapsed.as_millis() as u64) / metrics.total_tasks_distributed
        );

        info!("Task {} allocated to {} agents using {:?} strategy",
            task.id, allocation.assigned_agents.len(), self.strategy);

        Ok(allocation)
    }

    /// Round-robin selection
    async fn round_robin_selection(
        &self,
        _task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        // Simple round-robin: select the first available agent
        Ok(vec![available_agents[0].0.clone()])
    }

    /// Least loaded selection
    async fn least_loaded_selection(
        &self,
        _task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        // Select agent with lowest workload
        let least_loaded = available_agents.iter()
            .min_by(|a, b| {
                a.1.task_count.cmp(&b.1.task_count)
            })
            .ok_or_else(|| anyhow!("No agents available"))?;

        Ok(vec![least_loaded.0.clone()])
    }

    /// Performance-based selection
    async fn performance_based_selection(
        &self,
        task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        // Select agent with best performance score
        let best_performer = available_agents.iter()
            .max_by(|a, b| {
                a.1.performance_score.partial_cmp(&b.1.performance_score).unwrap()
            })
            .ok_or_else(|| anyhow!("No agents available"))?;

        // For critical tasks, assign multiple agents
        if task.priority == TaskPriority::Critical && available_agents.len() > 1 {
            let mut selected = vec![best_performer.0.clone()];

            // Add second-best performer
            if let Some(second_best) = available_agents.iter()
                .filter(|(id, _)| id != &best_performer.0)
                .max_by(|a, b| {
                    a.1.performance_score.partial_cmp(&b.1.performance_score).unwrap()
                }) {
                selected.push(second_best.0.clone());
            }

            Ok(selected)
        } else {
            Ok(vec![best_performer.0.clone()])
        }
    }

    /// Capability-optimal selection
    async fn capability_optimal_selection(
        &self,
        task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        // Score agents based on capability match
        let mut scored_agents: Vec<(AgentId, f32)> = available_agents.iter()
            .map(|(id, entry)| {
                let capability_score = task.requirements.iter()
                    .filter(|req| entry.capabilities.contains(req))
                    .count() as f32 / task.requirements.len().max(1) as f32;

                (id.clone(), capability_score * entry.performance_score)
            })
            .collect();

        // Sort by score
        scored_agents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select best match
        if let Some((best_id, _)) = scored_agents.first() {
            Ok(vec![best_id.clone()])
        } else {
            Err(anyhow!("No suitable agents found"))
        }
    }

    /// Dynamic priority selection
    async fn dynamic_priority_selection(
        &self,
        task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        // Combine multiple factors for dynamic selection
        let mut scored_agents: Vec<(AgentId, f32)> = available_agents.iter()
            .map(|(id, entry)| {
                let capability_score = task.requirements.iter()
                    .filter(|req| entry.capabilities.contains(req))
                    .count() as f32 / task.requirements.len().max(1) as f32;

                let load_score = 1.0 - (entry.task_count as f32 / 10.0).min(1.0);
                let performance_score = entry.performance_score;

                // Weight factors based on task priority
                let (cap_weight, load_weight, perf_weight) = match task.priority {
                    TaskPriority::Critical => (0.5, 0.1, 0.4),
                    TaskPriority::High => (0.4, 0.2, 0.4),
                    TaskPriority::Normal => (0.3, 0.4, 0.3),
                    TaskPriority::Low => (0.2, 0.5, 0.3),
                };

                let total_score = capability_score * cap_weight
                    + load_score * load_weight
                    + performance_score * perf_weight;

                (id.clone(), total_score)
            })
            .collect();

        scored_agents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if let Some((best_id, _)) = scored_agents.first() {
            Ok(vec![best_id.clone()])
        } else {
            Err(anyhow!("No suitable agents found"))
        }
    }

    /// Random selection
    async fn random_selection(
        &self,
        _task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..available_agents.len());
        Ok(vec![available_agents[index].0.clone()])
    }

    /// Weighted round-robin selection
    async fn weighted_round_robin_selection(
        &self,
        _task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Vec<AgentId>> {
        // Weight by performance score

        let weights: Vec<f32> = available_agents.iter()
            .map(|(_, entry)| entry.performance_score)
            .collect();

        let dist = WeightedIndex::new(&weights)
            .map_err(|e| anyhow!("Failed to create weighted distribution: {}", e))?;

        let mut rng = rand::thread_rng();
        let index = dist.sample(&mut rng);

        Ok(vec![available_agents[index].0.clone()])
    }

    /// Estimate task completion time
    async fn estimate_completion_time(
        &self,
        task: &Task,
        _available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<Duration> {
        // Simple estimation based on task type and priority
        let base_time = match task.priority {
            TaskPriority::Critical => Duration::from_secs(30),
            TaskPriority::High => Duration::from_secs(60),
            TaskPriority::Normal => Duration::from_secs(120),
            TaskPriority::Low => Duration::from_secs(300),
        };

        Ok(base_time)
    }

    /// Calculate allocation confidence
    async fn calculate_allocation_confidence(
        &self,
        task: &Task,
        available_agents: &[(AgentId, AgentEntry)],
    ) -> Result<f32> {
        // Calculate confidence based on capability match and agent performance
        let best_match = available_agents.iter()
            .map(|(_, entry)| {
                let capability_match = task.requirements.iter()
                    .filter(|req| entry.capabilities.contains(req))
                    .count() as f32 / task.requirements.len().max(1) as f32;

                capability_match * entry.performance_score
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        Ok(best_match)
    }

    /// Update agent performance
    pub async fn update_agent_performance(
        &self,
        agent_id: &AgentId,
        task_result: &super::TaskResult,
    ) -> Result<()> {
        let mut history = self.performance_history.write().await;

        let performance = history.entry(agent_id.clone()).or_insert(AgentPerformance {
            agent_id: agent_id.clone(),
            tasks_completed: 0,
            success_rate: 1.0,
            average_completion_time: Duration::from_secs(0),
            specialization_scores: HashMap::new(),
            current_load: 0.0,
            reliability_score: 1.0,
        });

        // Update metrics
        performance.tasks_completed += 1;

        if task_result.success {
            performance.success_rate = (performance.success_rate * (performance.tasks_completed - 1) as f32
                + 1.0) / performance.tasks_completed as f32;
        } else {
            performance.success_rate = (performance.success_rate * (performance.tasks_completed - 1) as f32)
                / performance.tasks_completed as f32;
        }

        // Update average completion time
        let total_ms = performance.average_completion_time.as_millis() as u64
            * (performance.tasks_completed - 1)
            + task_result.execution_time.as_millis() as u64;
        performance.average_completion_time = Duration::from_millis(
            total_ms / performance.tasks_completed
        );

        // Update reliability score
        performance.reliability_score = performance.success_rate * task_result.confidence;

        Ok(())
    }

    /// Get distribution metrics
    pub async fn get_metrics(&self) -> Result<DistributionMetrics> {
        Ok(self.metrics.read().await.clone())
    }

    /// Get load balance score
    pub async fn calculate_load_balance_score(
        &self,
        agents: &HashMap<AgentId, AgentEntry>,
    ) -> Result<f32> {
        if agents.is_empty() {
            return Ok(0.0);
        }

        let loads: Vec<f32> = agents.values()
            .map(|entry| entry.task_count as f32)
            .collect();

        let mean = loads.iter().sum::<f32>() / loads.len() as f32;
        let variance = loads.iter()
            .map(|&load| (load - mean).powi(2))
            .sum::<f32>() / loads.len() as f32;

        // Lower variance means better load balance
        let balance_score = 1.0 / (1.0 + variance.sqrt());

        Ok(balance_score)
    }
}

impl PartialEq for LoadBalancingStrategy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::RoundRobin, Self::RoundRobin) => true,
            (Self::LeastLoaded, Self::LeastLoaded) => true,
            (Self::PerformanceBased, Self::PerformanceBased) => true,
            (Self::CapabilityOptimal, Self::CapabilityOptimal) => true,
            (Self::DynamicPriority, Self::DynamicPriority) => true,
            (Self::Random, Self::Random) => true,
            (Self::WeightedRoundRobin, Self::WeightedRoundRobin) => true,
            _ => false,
        }
    }
}

impl Eq for LoadBalancingStrategy {}

impl std::hash::Hash for LoadBalancingStrategy {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::RoundRobin => 0.hash(state),
            Self::LeastLoaded => 1.hash(state),
            Self::PerformanceBased => 2.hash(state),
            Self::CapabilityOptimal => 3.hash(state),
            Self::DynamicPriority => 4.hash(state),
            Self::Random => 5.hash(state),
            Self::WeightedRoundRobin => 6.hash(state),
        }
    }
}