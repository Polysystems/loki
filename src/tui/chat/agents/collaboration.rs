//! Multi-agent collaboration modes and coordination
//! 
//! This module implements different collaboration strategies for
//! coordinating multiple AI agents working together.

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tokio::sync::{RwLock, mpsc};

use crate::cognitive::agents::{AgentSpecialization, LoadBalancingStrategy};
use super::manager::CollaborationMode;

/// Configuration for multi-agent collaboration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationConfig {
    /// Current collaboration mode
    pub mode: CollaborationMode,
    
    /// Consensus threshold for democratic mode
    pub consensus_threshold: f32,
    
    /// Maximum agents to involve
    pub max_agents: usize,
    
    /// Timeout for agent responses
    pub agent_timeout_ms: u64,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    
    /// Whether to wait for all agents
    pub wait_for_all: bool,
}

impl Default for CollaborationConfig {
    fn default() -> Self {
        Self {
            mode: CollaborationMode::Coordinated,
            consensus_threshold: 0.7,
            max_agents: 5,
            agent_timeout_ms: 30000,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            wait_for_all: false,
        }
    }
}

/// Coordinator for multi-agent collaboration
pub struct CollaborationCoordinator {
    config: RwLock<CollaborationConfig>,
    active_agents: RwLock<HashMap<String, AgentInfo>>,
    task_queue: RwLock<Vec<CollaborativeTask>>,
}

/// Information about an active agent
#[derive(Debug, Clone)]
struct AgentInfo {
    /// Agent ID
    id: String,
    
    /// Agent specialization
    specialization: AgentSpecialization,
    
    /// Current workload
    workload: f32,
    
    /// Performance score
    performance_score: f32,
    
    /// Available for tasks
    available: bool,
}

/// A task for collaborative execution
#[derive(Debug, Clone)]
pub struct CollaborativeTask {
    /// Task ID
    pub id: String,
    
    /// Task description
    pub description: String,
    
    /// Required specializations
    pub required_specializations: Vec<AgentSpecialization>,
    
    /// Priority
    pub priority: f32,
    
    /// Context
    pub context: HashMap<String, String>,
}

/// Result from collaborative execution
#[derive(Debug, Clone)]
pub struct CollaborationResult {
    /// Task ID
    pub task_id: String,
    
    /// Combined result
    pub result: String,
    
    /// Individual agent contributions
    pub contributions: Vec<AgentContribution>,
    
    /// Consensus achieved
    pub consensus_score: f32,
    
    /// Execution time
    pub execution_time_ms: u64,
}

/// Individual agent contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContribution {
    /// Agent ID
    pub agent_id: String,
    
    /// Agent's response
    pub response: String,
    
    /// Confidence score
    pub confidence: f32,
    
    /// Reasoning provided
    pub reasoning: Option<String>,
}

impl CollaborationCoordinator {
    /// Create a new collaboration coordinator
    pub fn new(config: CollaborationConfig) -> Self {
        Self {
            config: RwLock::new(config),
            active_agents: RwLock::new(HashMap::new()),
            task_queue: RwLock::new(Vec::new()),
        }
    }
    
    /// Register an agent
    pub async fn register_agent(
        &self,
        id: String,
        specialization: AgentSpecialization,
    ) -> Result<()> {
        let mut agents = self.active_agents.write().await;
        agents.insert(id.clone(), AgentInfo {
            id,
            specialization,
            workload: 0.0,
            performance_score: 1.0,
            available: true,
        });
        Ok(())
    }
    
    /// Execute a collaborative task
    pub async fn execute_task(
        &self,
        task: CollaborativeTask,
    ) -> Result<CollaborationResult> {
        let config = self.config.read().await;
        let start_time = std::time::Instant::now();
        
        match config.mode {
            CollaborationMode::Independent => {
                self.execute_independent(task, &config).await
            }
            CollaborationMode::Coordinated => {
                self.execute_coordinated(task, &config).await
            }
            CollaborationMode::Hierarchical => {
                self.execute_hierarchical(task, &config).await
            }
            CollaborationMode::Democratic => {
                self.execute_democratic(task, &config).await
            }
        }
        .map(|mut result| {
            result.execution_time_ms = start_time.elapsed().as_millis() as u64;
            result
        })
    }
    
    /// Independent execution - agents work in parallel without coordination
    async fn execute_independent(
        &self,
        task: CollaborativeTask,
        config: &CollaborationConfig,
    ) -> Result<CollaborationResult> {
        let agents = self.select_agents(&task, config).await?;
        let (tx, mut rx) = mpsc::channel(agents.len());
        
        // Launch all agents in parallel
        for agent in agents {
            let task_clone = task.clone();
            let tx = tx.clone();
            
            tokio::spawn(async move {
                let response = Self::simulate_agent_response(&agent, &task_clone).await;
                let _ = tx.send((agent.id.clone(), response)).await;
            });
        }
        
        // Collect responses
        let mut contributions = Vec::new();
        let timeout = tokio::time::Duration::from_millis(config.agent_timeout_ms);
        
        loop {
            tokio::select! {
                Some((agent_id, response)) = rx.recv() => {
                    contributions.push(AgentContribution {
                        agent_id,
                        response: response.content,
                        confidence: response.confidence,
                        reasoning: response.reasoning,
                    });
                    
                    if !config.wait_for_all && contributions.len() >= 1 {
                        break;
                    }
                }
                _ = tokio::time::sleep(timeout) => {
                    if contributions.is_empty() {
                        return Err(anyhow!("No agents responded within timeout"));
                    }
                    break;
                }
            }
        }
        
        // Combine results (simple concatenation for independent mode)
        let combined = contributions.iter()
            .map(|c| &c.response)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n");
        
        Ok(CollaborationResult {
            task_id: task.id,
            result: combined,
            contributions,
            consensus_score: 0.0, // No consensus in independent mode
            execution_time_ms: 0, // Set by caller
        })
    }
    
    /// Coordinated execution - agents work together with shared context
    async fn execute_coordinated(
        &self,
        task: CollaborativeTask,
        config: &CollaborationConfig,
    ) -> Result<CollaborationResult> {
        let agents = self.select_agents(&task, config).await?;
        let mut contributions = Vec::new();
        let mut shared_context = task.context.clone();
        
        // Agents work in sequence, building on each other's work
        for agent in agents {
            let response = Self::simulate_agent_response_with_context(
                &agent,
                &task,
                &shared_context,
            ).await;
            
            // Add agent's contribution to shared context
            shared_context.insert(
                format!("{}_response", agent.id),
                response.content.clone(),
            );
            
            contributions.push(AgentContribution {
                agent_id: agent.id.clone(),
                response: response.content,
                confidence: response.confidence,
                reasoning: response.reasoning,
            });
        }
        
        // Final result is the last agent's response (built on all previous)
        let result = contributions.last()
            .map(|c| c.response.clone())
            .unwrap_or_default();
        
        Ok(CollaborationResult {
            task_id: task.id,
            result,
            contributions: contributions.clone(),
            consensus_score: Self::calculate_consensus(&contributions),
            execution_time_ms: 0,
        })
    }
    
    /// Hierarchical execution - leader agent coordinates others
    async fn execute_hierarchical(
        &self,
        task: CollaborativeTask,
        config: &CollaborationConfig,
    ) -> Result<CollaborationResult> {
        let agents = self.select_agents(&task, config).await?;
        
        // Select leader (highest performance score)
        let leader = agents.iter()
            .max_by(|a, b| a.performance_score.partial_cmp(&b.performance_score).unwrap())
            .ok_or_else(|| anyhow!("No agents available"))?;
        
        let mut contributions = Vec::new();
        
        // Leader decomposes task
        let subtasks = Self::decompose_task(&task, agents.len() - 1);
        
        // Assign subtasks to other agents
        for (i, agent) in agents.iter().enumerate() {
            if agent.id == leader.id {
                continue;
            }
            
            let subtask = &subtasks[i % subtasks.len()];
            let response = Self::simulate_agent_response_with_context(
                agent,
                &task,
                &HashMap::from([("subtask".to_string(), subtask.clone())]),
            ).await;
            
            contributions.push(AgentContribution {
                agent_id: agent.id.clone(),
                response: response.content,
                confidence: response.confidence,
                reasoning: response.reasoning,
            });
        }
        
        // Leader synthesizes results
        let synthesis_context = HashMap::from([
            ("role".to_string(), "synthesize".to_string()),
            ("contributions".to_string(), serde_json::to_string(&contributions).unwrap()),
        ]);
        
        let leader_response = Self::simulate_agent_response_with_context(
            leader,
            &task,
            &synthesis_context,
        ).await;
        
        contributions.push(AgentContribution {
            agent_id: leader.id.clone(),
            response: leader_response.content.clone(),
            confidence: leader_response.confidence,
            reasoning: Some("Leader synthesis".to_string()),
        });
        
        Ok(CollaborationResult {
            task_id: task.id,
            result: leader_response.content,
            contributions,
            consensus_score: leader_response.confidence,
            execution_time_ms: 0,
        })
    }
    
    /// Democratic execution - agents vote on best approach
    async fn execute_democratic(
        &self,
        task: CollaborativeTask,
        config: &CollaborationConfig,
    ) -> Result<CollaborationResult> {
        let agents = self.select_agents(&task, config).await?;
        let mut contributions = Vec::new();
        let mut proposals: Vec<(String, String, f32)> = Vec::new();
        
        // Each agent proposes a solution
        for agent in &agents {
            let response = Self::simulate_agent_response(agent, &task).await;
            proposals.push((
                agent.id.clone(),
                response.content.clone(),
                response.confidence,
            ));
            
            contributions.push(AgentContribution {
                agent_id: agent.id.clone(),
                response: response.content,
                confidence: response.confidence,
                reasoning: response.reasoning,
            });
        }
        
        // Agents vote on proposals
        let mut votes: HashMap<usize, f32> = HashMap::new();
        
        for (voter_idx, voter) in agents.iter().enumerate() {
            for (proposal_idx, (proposer_id, proposal, _)) in proposals.iter().enumerate() {
                if &voter.id != proposer_id {
                    // Simulate voting (in practice, would ask agent to evaluate)
                    let vote_weight = Self::simulate_vote(voter, proposal);
                    *votes.entry(proposal_idx).or_insert(0.0) += vote_weight;
                }
            }
        }
        
        // Find winning proposal
        let winner_idx = votes.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| *idx)
            .unwrap_or(0);
        
        let (_, winning_proposal, _) = &proposals[winner_idx];
        let consensus_score = votes.get(&winner_idx).unwrap_or(&0.0) / (agents.len() - 1) as f32;
        
        Ok(CollaborationResult {
            task_id: task.id,
            result: winning_proposal.clone(),
            contributions,
            consensus_score,
            execution_time_ms: 0,
        })
    }
    
    /// Select agents for a task
    async fn select_agents(
        &self,
        task: &CollaborativeTask,
        config: &CollaborationConfig,
    ) -> Result<Vec<AgentInfo>> {
        let agents = self.active_agents.read().await;
        
        // Filter by required specializations and availability
        let mut eligible: Vec<AgentInfo> = agents.values()
            .filter(|agent| {
                agent.available &&
                task.required_specializations.is_empty() ||
                task.required_specializations.contains(&agent.specialization)
            })
            .cloned()
            .collect();
        
        if eligible.is_empty() {
            return Err(anyhow!("No eligible agents available"));
        }
        
        // Apply load balancing
        match config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin selection
                eligible.truncate(config.max_agents);
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Sort by workload
                eligible.sort_by(|a, b| a.workload.partial_cmp(&b.workload).unwrap());
                eligible.truncate(config.max_agents);
            }
            LoadBalancingStrategy::CapabilityOptimal => {
                // Prefer agents with matching specializations
                eligible.sort_by_key(|a| {
                    !task.required_specializations.contains(&a.specialization)
                });
                eligible.truncate(config.max_agents);
            }
            LoadBalancingStrategy::DynamicPriority => {
                // Consider both workload and performance
                eligible.sort_by(|a, b| {
                    let a_score = a.performance_score / (1.0 + a.workload);
                    let b_score = b.performance_score / (1.0 + b.workload);
                    b_score.partial_cmp(&a_score).unwrap()
                });
                eligible.truncate(config.max_agents);
            }
            LoadBalancingStrategy::PerformanceBased => {
                // Sort by performance score
                eligible.sort_by(|a, b| b.performance_score.partial_cmp(&a.performance_score).unwrap());
                eligible.truncate(config.max_agents);
            }
            LoadBalancingStrategy::Random => {
                // Random selection
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                eligible.shuffle(&mut rng);
                eligible.truncate(config.max_agents);
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                // Weight by performance score
                eligible.sort_by(|a, b| b.performance_score.partial_cmp(&a.performance_score).unwrap());
                eligible.truncate(config.max_agents);
            }
        }
        
        Ok(eligible)
    }
    
    /// Simulate agent response (in practice would call actual agent)
    async fn simulate_agent_response(agent: &AgentInfo, task: &CollaborativeTask) -> AgentResponse {
        AgentResponse {
            content: format!(
                "Response from {} agent for task: {}",
                agent.specialization.to_string(),
                task.description
            ),
            confidence: 0.85,
            reasoning: Some(format!("Based on {} analysis", agent.specialization.to_string())),
        }
    }
    
    /// Simulate agent response with context
    async fn simulate_agent_response_with_context(
        agent: &AgentInfo,
        task: &CollaborativeTask,
        context: &HashMap<String, String>,
    ) -> AgentResponse {
        let context_summary = context.keys().cloned().collect::<Vec<_>>().join(", ");
        AgentResponse {
            content: format!(
                "Response from {} agent for task: {} (with context: {})",
                agent.specialization.to_string(),
                task.description,
                context_summary
            ),
            confidence: 0.9,
            reasoning: Some(format!("Considering context: {}", context_summary)),
        }
    }
    
    /// Decompose task into subtasks
    fn decompose_task(task: &CollaborativeTask, num_subtasks: usize) -> Vec<String> {
        (0..num_subtasks)
            .map(|i| format!("Subtask {} of: {}", i + 1, task.description))
            .collect()
    }
    
    /// Simulate voting
    fn simulate_vote(voter: &AgentInfo, proposal: &str) -> f32 {
        // In practice, would ask agent to evaluate proposal
        // For now, simulate based on agent specialization
        if proposal.contains(&voter.specialization.to_string()) {
            0.9
        } else {
            0.5
        }
    }
    
    /// Calculate consensus score
    fn calculate_consensus(contributions: &[AgentContribution]) -> f32 {
        if contributions.is_empty() {
            return 0.0;
        }
        
        let avg_confidence: f32 = contributions.iter()
            .map(|c| c.confidence)
            .sum::<f32>() / contributions.len() as f32;
        
        // In practice, would also consider semantic similarity of responses
        avg_confidence
    }
}

/// Internal agent response structure
struct AgentResponse {
    content: String,
    confidence: f32,
    reasoning: Option<String>,
}

// Display implementation for AgentSpecialization is already in specialized_agent.rs

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_independent_collaboration() {
        let coordinator = CollaborationCoordinator::new(CollaborationConfig {
            mode: CollaborationMode::Independent,
            max_agents: 3,
            ..Default::default()
        });
        
        // Register agents
        coordinator.register_agent("agent1".to_string(), AgentSpecialization::Analytical).await.unwrap();
        coordinator.register_agent("agent2".to_string(), AgentSpecialization::Creative).await.unwrap();
        coordinator.register_agent("agent3".to_string(), AgentSpecialization::Technical).await.unwrap();
        
        // Create task
        let task = CollaborativeTask {
            id: "task1".to_string(),
            description: "Analyze and improve code".to_string(),
            required_specializations: vec![],
            priority: 0.8,
            context: HashMap::new(),
        };
        
        // Execute
        let result = coordinator.execute_task(task).await.unwrap();
        assert!(!result.contributions.is_empty());
        assert!(!result.result.is_empty());
    }
}