//! Agent management and coordination

use std::sync::Arc;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use crate::cognitive::agents::{AgentSpecialization, LoadBalancingStrategy};
use super::coordination::{AgentCoordinator, CoordinationConfig};
use super::collaboration::{CollaborationCoordinator, CollaborationConfig};

/// Collaboration modes for multi-agent systems
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CollaborationMode {
    Independent,
    Coordinated,
    Hierarchical,
    Democratic,
}

/// Manager for agent system configuration and state
#[derive(Clone)]
pub struct AgentManager {
    pub selected_index: usize,
    pub agent_system_enabled: bool,
    pub active_specializations: Vec<AgentSpecialization>,
    pub consensus_threshold: f32,
    pub collaboration_mode: CollaborationMode,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub min_agents_for_consensus: usize,
    
    /// Agent coordinator for managing the agent pool
    pub coordinator: Option<Arc<RwLock<AgentCoordinator>>>,
    
    /// Collaboration coordinator for multi-agent tasks
    pub collaboration_coordinator: Option<Arc<RwLock<CollaborationCoordinator>>>,
}

impl Default for AgentManager {
    fn default() -> Self {
        Self {
            selected_index: 0,
            agent_system_enabled: false,
            active_specializations: vec![
                AgentSpecialization::Analytical,
                AgentSpecialization::Creative,
                AgentSpecialization::Strategic,
            ],
            consensus_threshold: 0.7,
            collaboration_mode: CollaborationMode::Coordinated,
            load_balancing_strategy: LoadBalancingStrategy::DynamicPriority,
            min_agents_for_consensus: 3,
            coordinator: None,
            collaboration_coordinator: None,
        }
    }
}

impl AgentManager {
    /// Create manager with agent system enabled
    pub fn enabled() -> Self {
        Self {
            agent_system_enabled: true,
            ..Default::default()
        }
    }
    
    /// Create a new instance with default settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Enable agent system with specific mode
    pub fn enable_with_mode(&mut self, mode: CollaborationMode) {
        self.agent_system_enabled = true;
        self.collaboration_mode = mode;
        
        // Adjust settings based on mode
        match mode {
            CollaborationMode::Independent => {
                self.min_agents_for_consensus = 1;
            }
            CollaborationMode::Democratic => {
                self.min_agents_for_consensus = (self.active_specializations.len() / 2) + 1;
            }
            CollaborationMode::Hierarchical => {
                self.min_agents_for_consensus = 2;
            }
            _ => {}
        }
    }
    
    /// Add an agent specialization
    pub fn add_specialization(&mut self, spec: AgentSpecialization) {
        if !self.active_specializations.contains(&spec) {
            self.active_specializations.push(spec);
        }
    }
    
    /// Remove an agent specialization
    pub fn remove_specialization(&mut self, spec: &AgentSpecialization) {
        self.active_specializations.retain(|s| s != spec);
    }
    
    /// Check if a specialization is active
    pub fn has_specialization(&self, spec: &AgentSpecialization) -> bool {
        self.active_specializations.contains(spec)
    }
    
    /// Get the number of active agents
    pub fn active_agent_count(&self) -> usize {
        if self.agent_system_enabled {
            self.active_specializations.len()
        } else {
            0
        }
    }
    
    /// Check if consensus is possible with current settings
    pub fn can_reach_consensus(&self) -> bool {
        self.active_agent_count() >= self.min_agents_for_consensus
    }
    
    /// Get a human-readable status string
    pub fn status_string(&self) -> String {
        if !self.agent_system_enabled {
            return "Disabled".to_string();
        }
        
        format!(
            "{} agents in {:?} mode",
            self.active_agent_count(),
            self.collaboration_mode
        )
    }
    
    /// Initialize the agent coordinator and create the agent pool
    pub async fn initialize_agent_pool(&mut self) -> anyhow::Result<()> {
        // Create coordinator config based on current settings
        let mut config = CoordinationConfig::default();
        config.load_balancing = self.load_balancing_strategy.clone();
        
        // Create the coordinator
        let coordinator = Arc::new(RwLock::new(AgentCoordinator::new(config)));
        
        // Register agents based on active specializations
        for (idx, spec) in self.active_specializations.iter().enumerate() {
            let agent_id = format!("agent_{}", idx);
            let agent_name = format!("{:?} Agent {}", spec, idx + 1);
            
            coordinator.write().await.register_agent(
                agent_id,
                agent_name,
                spec.clone(),
                3, // max concurrent tasks per agent
            ).await?;
        }
        
        self.coordinator = Some(coordinator.clone());
        
        // Also initialize collaboration coordinator
        let collab_config = CollaborationConfig {
            mode: self.collaboration_mode,
            consensus_threshold: self.consensus_threshold,
            max_agents: self.active_specializations.len(),
            load_balancing: self.load_balancing_strategy.clone(),
            ..Default::default()
        };
        
        let collab_coordinator = Arc::new(RwLock::new(
            CollaborationCoordinator::new(collab_config)
        ));
        
        // Register agents with collaboration coordinator
        for (idx, spec) in self.active_specializations.iter().enumerate() {
            let agent_id = format!("agent_{}", idx);
            collab_coordinator.write().await.register_agent(
                agent_id,
                spec.clone(),
            ).await?;
        }
        
        self.collaboration_coordinator = Some(collab_coordinator);
        
        tracing::info!("Agent pool and collaboration system initialized with {} agents", 
                       self.active_specializations.len());
        
        Ok(())
    }
    
    /// Connect to an existing agent coordinator
    pub fn connect_coordinator(&mut self, coordinator: Arc<RwLock<AgentCoordinator>>) {
        self.coordinator = Some(coordinator);
        tracing::info!("Connected to external agent coordinator");
    }
    
    /// Get the agent coordinator if available
    pub fn get_coordinator(&self) -> Option<Arc<RwLock<AgentCoordinator>>> {
        self.coordinator.clone()
    }
    
    /// Submit a task to the agent pool
    pub async fn submit_task(&self, task: super::coordination::CoordinationTask) -> anyhow::Result<()> {
        if let Some(coordinator) = &self.coordinator {
            coordinator.read().await.submit_task(task).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Agent coordinator not initialized"))
        }
    }
    
    /// Process the agent task queue
    pub async fn process_queue(&self) -> anyhow::Result<()> {
        if let Some(coordinator) = &self.coordinator {
            coordinator.read().await.process_queue().await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Agent coordinator not initialized"))
        }
    }
    
    /// Get agent statistics
    pub async fn get_agent_stats(&self) -> Vec<super::coordination::AgentStats> {
        if let Some(coordinator) = &self.coordinator {
            coordinator.read().await.get_stats().await
        } else {
            Vec::new()
        }
    }
    
    /// Execute a collaborative task using multiple agents
    pub async fn execute_collaborative_task(
        &self, 
        task: super::collaboration::CollaborativeTask
    ) -> anyhow::Result<super::collaboration::CollaborationResult> {
        if let Some(collab_coordinator) = &self.collaboration_coordinator {
            collab_coordinator.read().await.execute_task(task).await
        } else {
            Err(anyhow::anyhow!("Collaboration coordinator not initialized"))
        }
    }
    
    /// Update collaboration mode
    pub async fn set_collaboration_mode(&mut self, mode: CollaborationMode) -> anyhow::Result<()> {
        self.collaboration_mode = mode;
        
        // Update the collaboration coordinator if it exists
        if let Some(collab_coordinator) = &self.collaboration_coordinator {
            let config = CollaborationConfig {
                mode,
                consensus_threshold: self.consensus_threshold,
                max_agents: self.active_specializations.len(),
                load_balancing: self.load_balancing_strategy.clone(),
                ..Default::default()
            };
            
            // Re-create the coordinator with new config
            let new_coordinator = Arc::new(RwLock::new(
                CollaborationCoordinator::new(config)
            ));
            
            // Re-register all agents
            for (idx, spec) in self.active_specializations.iter().enumerate() {
                let agent_id = format!("agent_{}", idx);
                new_coordinator.write().await.register_agent(
                    agent_id,
                    spec.clone(),
                ).await?;
            }
            
            self.collaboration_coordinator = Some(new_coordinator);
            tracing::info!("Collaboration mode updated to {:?}", mode);
        }
        
        Ok(())
    }
    
    /// Get the collaboration coordinator if available
    pub fn get_collaboration_coordinator(&self) -> Option<Arc<RwLock<CollaborationCoordinator>>> {
        self.collaboration_coordinator.clone()
    }
    
    /// Enable the agent system
    pub fn enable_agent_system(&mut self) {
        self.agent_system_enabled = true;
        tracing::info!("Agent system enabled");
    }
}