//! Agent management and coordination

use serde::{Serialize, Deserialize};
use crate::cognitive::agents::{AgentSpecialization, LoadBalancingStrategy};

/// Collaboration modes for multi-agent systems
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CollaborationMode {
    Independent,
    Coordinated,
    Hierarchical,
    Democratic,
}

/// Manager for agent system configuration and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentManager {
    pub selected_index: usize,
    pub agent_system_enabled: bool,
    pub active_specializations: Vec<AgentSpecialization>,
    pub consensus_threshold: f32,
    pub collaboration_mode: CollaborationMode,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub min_agents_for_consensus: usize,
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
    
    /// Create a placeholder instance
    pub fn placeholder() -> Self {
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
}