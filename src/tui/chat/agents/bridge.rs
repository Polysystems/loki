//! Bridge between Agent Configuration and Runtime Coordination
//! 
//! Provides seamless conversion and integration between the configuration
//! system (AgentConfig) and the runtime coordination system (CoordinationAgent).

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use chrono::Utc;

use super::creation::{AgentConfig, AgentSkill};
use super::coordination::{self, AgentCoordinator, Agent as CoordinationAgent, AgentState};
use super::registry::{AgentRegistry, RuntimeState, AgentStatus};
use crate::cognitive::agents::AgentSpecialization;

/// Bridge for converting and managing agents between configuration and runtime
pub struct AgentBridge {
    /// Agent registry for persistence
    registry: Arc<RwLock<AgentRegistry>>,
    
    /// Agent coordinator for runtime management
    coordinator: Arc<RwLock<AgentCoordinator>>,
}

impl AgentBridge {
    /// Create a new agent bridge
    pub fn new(
        registry: Arc<RwLock<AgentRegistry>>,
        coordinator: Arc<RwLock<AgentCoordinator>>,
    ) -> Self {
        Self {
            registry,
            coordinator,
        }
    }
    
    /// Convert AgentConfig to CoordinationAgent and register with coordinator
    pub async fn instantiate_agent(&self, config: AgentConfig) -> Result<String> {
        let agent_id = config.id.clone();
        
        // First, ensure the agent is registered in the registry
        {
            let registry = self.registry.read().await;
            if !registry.exists(&agent_id).await? {
                drop(registry); // Release read lock before acquiring write lock
                let mut registry = self.registry.write().await;
                registry.register(config.clone()).await?;
            }
        }
        
        // Create the coordination agent
        let coord_agent = self.config_to_coordination(config).await?;
        
        // Register with the coordinator
        let coordinator = self.coordinator.write().await;
        coordinator.register_agent(
            coord_agent.id.clone(),
            coord_agent.name.clone(),
            coord_agent.specialization.clone(),
            coord_agent.max_concurrent_tasks,
        ).await?;
        
        // Update registry runtime state
        self.registry.read().await.update_runtime_state(&agent_id, RuntimeState {
            is_active: true,
            status: AgentStatus::Ready,
            resources: super::registry::ResourceAllocation::default(),
            session_id: Some(uuid::Uuid::new_v4().to_string()),
        }).await?;
        
        // Update activation stats
        self.registry.read().await.update_usage_stats(&agent_id, |stats| {
            stats.activation_count += 1;
            stats.last_activated = Some(Utc::now());
        }).await?;
        
        tracing::info!("Agent {} instantiated and registered with coordinator", agent_id);
        Ok(agent_id)
    }
    
    /// Convert AgentConfig to CoordinationAgent
    async fn config_to_coordination(&self, config: AgentConfig) -> Result<CoordinationAgent> {
        // Get any existing stats from registry
        let existing_stats = if let Some(entry) = self.registry.read().await.get(&config.id).await? {
            Some(entry.metadata.usage_stats)
        } else {
            None
        };
        
        // Calculate success rate
        let success_rate = if let Some(stats) = &existing_stats {
            if stats.successful_tasks + stats.failed_tasks > 0 {
                stats.successful_tasks as f32 / (stats.successful_tasks + stats.failed_tasks) as f32
            } else {
                1.0
            }
        } else {
            1.0
        };
        
        Ok(CoordinationAgent {
            id: config.id.clone(),
            name: config.name.clone(),
            specialization: config.specialization.clone(),
            state: AgentState::Idle,
            metrics: coordination::AgentMetrics {
                tasks_completed: existing_stats.as_ref()
                    .map(|s| s.successful_tasks)
                    .unwrap_or(0),
                tasks_failed: existing_stats.as_ref()
                    .map(|s| s.failed_tasks)
                    .unwrap_or(0),
                avg_completion_time_ms: existing_stats.as_ref()
                    .map(|s| s.avg_response_time_ms)
                    .unwrap_or(0),
                success_rate,
                quality_score: 1.0,
                last_active: std::time::Instant::now(),
            },
            workload: 0.0,
            max_concurrent_tasks: self.calculate_max_tasks(&config),
            active_tasks: Vec::new(),
        })
    }
    
    /// Extract capabilities from skills
    fn extract_capabilities(&self, skills: &[AgentSkill]) -> Vec<String> {
        skills.iter()
            .map(|skill| skill.name.clone())
            .collect()
    }
    
    /// Calculate maximum concurrent tasks based on configuration
    fn calculate_max_tasks(&self, config: &AgentConfig) -> usize {
        // Base it on performance settings and model preferences
        let base_tasks = if config.performance_settings.parallel_processing {
            config.performance_settings.batch_size
        } else {
            1
        };
        
        // Adjust based on specialization
        match config.specialization {
            AgentSpecialization::Technical | AgentSpecialization::Analytical => base_tasks,
            AgentSpecialization::Creative => 1, // Creative tasks usually need focus
            AgentSpecialization::Strategic => base_tasks / 2, // Strategic needs more resources
            _ => base_tasks,
        }
    }
    
    /// Deactivate an agent and update its state
    pub async fn deactivate_agent(&self, agent_id: &str) -> Result<()> {
        // Remove agent from coordinator tracking
        // Note: AgentCoordinator doesn't have unregister_agent method
        // We'll just update the registry state
        
        // Update registry state
        self.registry.read().await.update_runtime_state(agent_id, RuntimeState {
            is_active: false,
            status: AgentStatus::Inactive,
            resources: super::registry::ResourceAllocation::default(),
            session_id: None,
        }).await?;
        
        tracing::info!("Agent {} deactivated", agent_id);
        Ok(())
    }
    
    /// Reactivate a previously deactivated agent
    pub async fn reactivate_agent(&self, agent_id: &str) -> Result<()> {
        // Get configuration from registry
        let entry = self.registry.read().await.get(agent_id).await?
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found", agent_id))?;
        
        // Instantiate the agent again
        self.instantiate_agent(entry.config).await?;
        
        Ok(())
    }
    
    /// Sync agent state from coordinator to registry
    pub async fn sync_agent_state(&self, agent_id: &str) -> Result<()> {
        let coordinator = self.coordinator.read().await;
        
        // Get all agent stats from coordinator
        let all_stats = coordinator.get_agent_stats().await;
        let stats = all_stats.get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found in coordinator", agent_id))?;
        
        // Map coordinator state to registry status
        let status = match stats.state {
            AgentState::Idle => AgentStatus::Ready,
            AgentState::Working => AgentStatus::Busy,
            AgentState::Failed => AgentStatus::Error("Agent in failed state".to_string()),
            _ => AgentStatus::Ready,
        };
        
        // Update registry
        self.registry.read().await.update_runtime_state(agent_id, RuntimeState {
            is_active: true,
            status,
            resources: super::registry::ResourceAllocation {
                memory_mb: 256,
                cpu_percent: (stats.workload * 100.0) as u8,
                token_budget: 10000,
                rate_limit: 60,
            },
            session_id: Some(agent_id.to_string()),
        }).await?;
        
        // Update usage stats based on available AgentStats fields
        self.registry.read().await.update_usage_stats(agent_id, |usage_stats| {
            usage_stats.successful_tasks = stats.total_completed;
            usage_stats.failed_tasks = 0; // Not available in AgentStats
            usage_stats.total_runtime_secs = 0; // Not available in AgentStats
            usage_stats.avg_response_time_ms = stats.avg_completion_time_ms;
        }).await?;
        
        Ok(())
    }
    
    /// Batch instantiate multiple agents
    pub async fn instantiate_agents(&self, configs: Vec<AgentConfig>) -> Result<Vec<String>> {
        let mut agent_ids = Vec::new();
        
        for config in configs {
            match self.instantiate_agent(config).await {
                Ok(id) => agent_ids.push(id),
                Err(e) => {
                    tracing::error!("Failed to instantiate agent: {}", e);
                }
            }
        }
        
        Ok(agent_ids)
    }
    
    /// Get all active agents
    pub async fn get_active_agents(&self) -> Result<Vec<String>> {
        let agents = self.registry.read().await.list().await?;
        
        Ok(agents.into_iter()
            .filter(|(_, entry)| entry.runtime_state.is_active)
            .map(|(id, _)| id)
            .collect())
    }
    
    /// Migrate an agent to a new specialization
    pub async fn migrate_specialization(
        &self,
        agent_id: &str,
        new_spec: AgentSpecialization,
    ) -> Result<()> {
        // Get current configuration
        let entry = self.registry.read().await.get(agent_id).await?
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found", agent_id))?;
        
        let mut config = entry.config;
        let was_active = entry.runtime_state.is_active;
        
        // Deactivate if active
        if was_active {
            self.deactivate_agent(agent_id).await?;
        }
        
        // Update specialization
        config.specialization = new_spec.clone();
        
        // Update in registry
        self.registry.write().await.update(agent_id, config.clone()).await?;
        
        // Reactivate if was active
        if was_active {
            self.instantiate_agent(config).await?;
        }
        
        tracing::info!("Agent {} migrated to specialization {:?}", agent_id, new_spec);
        Ok(())
    }
}