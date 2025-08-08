//! Agent Task Mapper
//! 
//! Maps tasks to appropriate agents based on task requirements, agent capabilities,
//! and current agent availability.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{ info};

use crate::models::multi_agent_orchestrator::{
    MultiAgentOrchestrator, AgentType as OrchestratorAgentType, AgentInstance
};
use crate::tui::task_decomposer::{Subtask, AgentType as TaskAgentType};
use crate::tui::nlp::core::orchestrator::ExtractedTaskType;
use crate::tui::ui::chat::agent_stream_manager::AgentStreamManager;

/// Agent task mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMapperConfig {
    /// Enable automatic agent selection
    pub auto_select_agents: bool,
    
    /// Maximum agents per task
    pub max_agents_per_task: usize,
    
    /// Prefer specialized agents over general purpose
    pub prefer_specialists: bool,
    
    /// Agent selection strategy
    pub selection_strategy: SelectionStrategy,
}

impl Default for AgentMapperConfig {
    fn default() -> Self {
        Self {
            auto_select_agents: true,
            max_agents_per_task: 4,
            prefer_specialists: true,
            selection_strategy: SelectionStrategy::BestMatch,
        }
    }
}

/// Agent selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Select best matching agent
    BestMatch,
    
    /// Load balance across agents
    LoadBalance,
    
    /// Minimize cost
    CostOptimized,
    
    /// Maximize quality
    QualityFirst,
}

/// Agent task mapper
pub struct AgentTaskMapper {
    /// Configuration
    config: AgentMapperConfig,
    
    /// Multi-agent orchestrator reference
    orchestrator: Arc<MultiAgentOrchestrator>,
    
    /// Agent stream manager reference
    stream_manager: Arc<AgentStreamManager>,
    
    /// Task-to-agent assignments
    assignments: Arc<RwLock<HashMap<String, AgentAssignment>>>,
    
    /// Agent capability matrix
    capability_matrix: Arc<RwLock<CapabilityMatrix>>,
}

/// Agent assignment for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAssignment {
    pub task_id: String,
    pub agent_id: String,
    pub agent_type: String,
    pub confidence_score: f64,
    pub estimated_completion_time: std::time::Duration,
    pub assignment_reason: String,
}

/// Agent capability matrix for matching
#[derive(Debug, Clone)]
struct CapabilityMatrix {
    /// Agent type to capabilities mapping
    capabilities: HashMap<String, Vec<String>>,
    
    /// Agent performance scores
    performance_scores: HashMap<String, f64>,
}

impl AgentTaskMapper {
    pub fn new(
        orchestrator: Arc<MultiAgentOrchestrator>,
        stream_manager: Arc<AgentStreamManager>,
        config: AgentMapperConfig,
    ) -> Self {
        Self {
            config,
            orchestrator,
            stream_manager,
            assignments: Arc::new(RwLock::new(HashMap::new())),
            capability_matrix: Arc::new(RwLock::new(CapabilityMatrix::default())),
        }
    }
    
    /// Map a subtask to an appropriate agent
    pub async fn map_task_to_agent(&self, subtask: &Subtask) -> Result<AgentAssignment> {
        info!("Mapping task to agent: {}", subtask.description);
        
        // Get available agents
        let available_agents = self.get_available_agents().await?;
        
        if available_agents.is_empty() {
            return Err(anyhow!("No agents available for task assignment"));
        }
        
        // Score agents based on task requirements
        let mut agent_scores = Vec::new();
        
        for agent in &available_agents {
            let score = self.score_agent_for_task(agent, subtask).await?;
            agent_scores.push((agent, score));
        }
        
        // Sort by total score (highest first)
        agent_scores.sort_by(|a, b| {
            let a_total = a.1.capability_match + a.1.performance_score + 
                         a.1.availability_score + a.1.cost_score;
            let b_total = b.1.capability_match + b.1.performance_score + 
                         b.1.availability_score + b.1.cost_score;
            b_total.partial_cmp(&a_total).unwrap()
        });
        
        // Select best agent based on strategy
        let selected_agent = match self.config.selection_strategy {
            SelectionStrategy::BestMatch => {
                agent_scores.first()
                    .ok_or_else(|| anyhow!("No suitable agent found"))?
                    .0
            }
            SelectionStrategy::LoadBalance => {
                // Find least loaded agent with acceptable score
                self.select_least_loaded_agent(&agent_scores).await?
            }
            SelectionStrategy::CostOptimized => {
                // Select cheapest agent with acceptable score
                self.select_cost_optimal_agent(&agent_scores).await?
            }
            SelectionStrategy::QualityFirst => {
                // Select highest quality agent regardless of cost
                agent_scores.first()
                    .ok_or_else(|| anyhow!("No suitable agent found"))?
                    .0
            }
        };
        
        // Create assignment
        let assignment = AgentAssignment {
            task_id: subtask.id.clone(),
            agent_id: selected_agent.id.clone(),
            agent_type: format!("{:?}", selected_agent.agent_type),
            confidence_score: agent_scores.iter()
                .find(|(agent, _)| agent.id == selected_agent.id)
                .map(|(_, score)| score.confidence)
                .unwrap_or(0.0),
            estimated_completion_time: subtask.estimated_effort,
            assignment_reason: self.generate_assignment_reason(selected_agent, subtask),
        };
        
        // Store assignment
        self.assignments.write().await.insert(subtask.id.clone(), assignment.clone());
        
        Ok(assignment)
    }
    
    /// Create agent streams for assigned tasks with full context
    pub async fn create_agent_streams_with_context(
        &self,
        assignments: Vec<AgentAssignment>,
        parent_task_id: &str,
        parent_task_description: &str,
        subtasks: &[crate::tui::task_decomposer::Subtask],
    ) -> Result<Vec<String>> {
        let mut stream_ids = Vec::new();
        
        for assignment in assignments {
            // Find the corresponding subtask
            let subtask = subtasks.iter()
                .find(|s| s.id == assignment.task_id);
                
            if let Some(subtask) = subtask {
                let stream_id = self.stream_manager.create_agent_stream_with_context(
                    assignment.agent_type.clone(),
                    format!("Agent-{}", &assignment.agent_id[..8]),
                    subtask.description.clone(),
                    Some(parent_task_id.to_string()),
                    Some(parent_task_description.to_string()),
                    Some(subtask.id.clone()),
                    Some(format!("{:?}", subtask.task_type)),
                    Some(subtask.estimated_effort),
                    subtask.dependencies.clone(),
                    subtask.parallel_group,
                ).await?;
                
                info!(
                    "Created agent stream {} for subtask {} with agent {} in parallel group {:?}",
                    &stream_id, assignment.task_id, assignment.agent_id, subtask.parallel_group
                );
                
                stream_ids.push(stream_id);
            }
        }
        
        Ok(stream_ids)
    }

    /// Create agent streams for assigned tasks
    pub async fn create_agent_streams(
        &self,
        assignments: Vec<AgentAssignment>,
        parent_task_description: &str,
    ) -> Result<Vec<String>> {
        let mut stream_ids = Vec::new();
        
        for assignment in assignments {
            let stream_id = self.stream_manager.create_agent_stream(
                assignment.agent_type.clone(),
                format!("Agent-{}", &assignment.agent_id[..8]),
                format!("Task: {} (Parent: {})", assignment.task_id, parent_task_description),
            ).await?;
            
            info!(
                "Created agent stream {} for task {} with agent {}",
                &stream_id, assignment.task_id, assignment.agent_id
            );
            
            stream_ids.push(stream_id);
        }
        
        Ok(stream_ids)
    }
    
    /// Get available agents from orchestrator
    async fn get_available_agents(&self) -> Result<Vec<AgentInstance>> {
        // In a real implementation, this would query the orchestrator
        // For now, create some mock agents
        Ok(vec![
            AgentInstance {
                id: "agent_code_1".to_string(),
                name: "CodeGen-Alpha".to_string(),
                agent_type: OrchestratorAgentType::CodeGeneration,
                models: vec!["gpt-4".to_string()],
                capabilities: vec!["coding".to_string(), "refactoring".to_string()],
                status: crate::models::multi_agent_orchestrator::AgentStatus::Active,
                performance_metrics: Default::default(),
                cost_tracker: Default::default(),
                last_used: None,
                error_count: 0,
                success_rate: 0.95,
            },
            AgentInstance {
                id: "agent_doc_1".to_string(),
                name: "DocWriter-Beta".to_string(),
                agent_type: OrchestratorAgentType::GeneralPurpose,
                models: vec!["claude-3".to_string()],
                capabilities: vec!["documentation".to_string(), "writing".to_string()],
                status: crate::models::multi_agent_orchestrator::AgentStatus::Active,
                performance_metrics: Default::default(),
                cost_tracker: Default::default(),
                last_used: None,
                error_count: 0,
                success_rate: 0.92,
            },
            AgentInstance {
                id: "agent_test_1".to_string(),
                name: "TestRunner-Gamma".to_string(),
                agent_type: OrchestratorAgentType::GeneralPurpose,
                models: vec!["gpt-3.5-turbo".to_string()],
                capabilities: vec!["testing".to_string(), "validation".to_string()],
                status: crate::models::multi_agent_orchestrator::AgentStatus::Active,
                performance_metrics: Default::default(),
                cost_tracker: Default::default(),
                last_used: None,
                error_count: 0,
                success_rate: 0.88,
            },
        ])
    }
    
    /// Score an agent for a specific task
    async fn score_agent_for_task(
        &self,
        agent: &AgentInstance,
        subtask: &Subtask,
    ) -> Result<AgentScore> {
        // Calculate success rate from performance metrics and error count
        let total_requests = 100.0; // Assume base of 100 requests
        let success_rate = if agent.error_count > 0 {
            (total_requests - agent.error_count as f64) / total_requests
        } else {
            agent.performance_metrics.quality_score as f64
        };
        
        // Calculate availability based on last used time and error rate
        let availability_score = self.calculate_availability_score(agent).await;
        
        // Calculate cost score based on agent's cost per request and budget
        let cost_score = self.calculate_cost_score(agent, subtask).await;
        
        let mut score = AgentScore {
            agent_id: agent.id.clone(),
            capability_match: 0.0,
            performance_score: success_rate,
            availability_score,
            cost_score,
            confidence: 0.0,
        };
        
        // Calculate capability match
        let required_capabilities = &subtask.required_capabilities;
        let agent_capabilities = &agent.capabilities;
        
        let matching_capabilities = required_capabilities.iter()
            .filter(|req| agent_capabilities.iter().any(|cap| cap.contains(req.as_str())))
            .count();
            
        score.capability_match = if required_capabilities.is_empty() {
            0.5 // Default score if no specific requirements
        } else {
            matching_capabilities as f64 / required_capabilities.len() as f64
        };
        
        // Check agent type match
        let type_match = match (&subtask.preferred_agent_type, &agent.agent_type) {
            (TaskAgentType::CodeGeneration, OrchestratorAgentType::CodeGeneration) => 1.0,
            (TaskAgentType::Documentation, _) if agent.capabilities.contains(&"documentation".to_string()) => 0.9,
            (TaskAgentType::Testing, _) if agent.capabilities.contains(&"testing".to_string()) => 0.9,
            (TaskAgentType::Research, _) if agent.capabilities.contains(&"research".to_string()) => 0.9,
            (TaskAgentType::GeneralPurpose, OrchestratorAgentType::GeneralPurpose) => 0.8,
            _ => 0.3, // Low score for mismatched types
        };
        
        // Calculate overall confidence
        score.confidence = (score.capability_match * 0.4) +
                          (type_match * 0.3) +
                          (score.performance_score * 0.2) +
                          (score.availability_score * 0.1);
                          
        Ok(score)
    }
    
    /// Select least loaded agent from scored agents
    async fn select_least_loaded_agent<'a>(
        &self,
        agent_scores: &'a [(&'a AgentInstance, AgentScore)],
    ) -> Result<&'a AgentInstance> {
        // Filter agents with acceptable scores
        let acceptable_agents: Vec<_> = agent_scores.iter()
            .filter(|(_, score)| score.confidence > 0.6)
            .collect();
            
        if acceptable_agents.is_empty() {
            return Err(anyhow!("No agents meet minimum confidence threshold"));
        }
        
        // Sort by availability score (highest availability = least loaded)
        let mut sorted_agents = acceptable_agents;
        sorted_agents.sort_by(|a, b| {
            b.1.availability_score.partial_cmp(&a.1.availability_score).unwrap()
        });
        
        // Return the least loaded agent (highest availability)
        Ok(sorted_agents[0].0)
    }
    
    /// Select cost-optimal agent
    async fn select_cost_optimal_agent<'a>(
        &self,
        agent_scores: &'a [(&'a AgentInstance, AgentScore)],
    ) -> Result<&'a AgentInstance> {
        // Filter agents with acceptable scores
        let acceptable_agents: Vec<_> = agent_scores.iter()
            .filter(|(_, score)| score.confidence > 0.5)
            .collect();
            
        if acceptable_agents.is_empty() {
            return Err(anyhow!("No agents meet minimum confidence threshold"));
        }
        
        // Sort by cost score (highest = cheapest)
        let mut by_cost = acceptable_agents;
        by_cost.sort_by(|a, b| b.1.cost_score.partial_cmp(&a.1.cost_score).unwrap());
        
        Ok(by_cost[0].0)
    }
    
    /// Generate human-readable assignment reason
    fn generate_assignment_reason(&self, agent: &AgentInstance, subtask: &Subtask) -> String {
        format!(
            "Agent {} selected for its {} capabilities and {:.0}% success rate",
            agent.name,
            agent.capabilities.join(", "),
            agent.success_rate * 100.0
        )
    }
    
    /// Get all current assignments
    pub async fn get_assignments(&self) -> HashMap<String, AgentAssignment> {
        self.assignments.read().await.clone()
    }
    
    /// Clear completed assignments
    pub async fn clear_completed_assignments(&self, task_ids: Vec<String>) -> Result<()> {
        let mut assignments = self.assignments.write().await;
        for task_id in task_ids {
            assignments.remove(&task_id);
        }
        Ok(())
    }
    
    /// Calculate agent availability score
    async fn calculate_availability_score(&self, agent: &AgentInstance) -> f64 {
        // Check if agent was recently used
        let recently_used_penalty = if let Some(last_used) = agent.last_used {
            let elapsed = last_used.elapsed();
            if elapsed < std::time::Duration::from_secs(5) {
                0.3 // Heavy penalty for very recent use
            } else if elapsed < std::time::Duration::from_secs(30) {
                0.1 // Light penalty for recent use
            } else {
                0.0 // No penalty
            }
        } else {
            0.0 // Never used, fully available
        };
        
        // Factor in error rate
        let error_penalty = (agent.error_count as f64 / 100.0).min(0.5);
        
        // Factor in uptime
        let uptime_bonus = (agent.performance_metrics.uptime_percentage / 100.0) as f64;
        
        // Calculate final availability (0.0 to 1.0)
        ((1.0 - recently_used_penalty - error_penalty) * uptime_bonus).max(0.0).min(1.0)
    }
    
    /// Calculate cost score for agent
    async fn calculate_cost_score(&self, agent: &AgentInstance, subtask: &Subtask) -> f64 {
        // Get cost per request from agent metrics
        let cost_per_request = agent.performance_metrics.cost_per_request;
        
        // Estimate requests needed based on task complexity
        let estimated_requests = match subtask.task_type {
            ExtractedTaskType::CodingTask => 3.0,
            ExtractedTaskType::ResearchTask => 5.0,
            ExtractedTaskType::DocumentationTask => 2.0,
            ExtractedTaskType::TestingTask => 4.0,
            ExtractedTaskType::DesignTask => 3.0,
            ExtractedTaskType::GeneralTask => 2.0,
        };
        
        let estimated_cost = cost_per_request * estimated_requests;
        
        // Convert to score (lower cost = higher score)
        // Assume $1 is our baseline for a single task
        if estimated_cost <= 0.0 {
            1.0 // Free is best
        } else if estimated_cost < 0.5 {
            0.9 // Very cheap
        } else if estimated_cost < 1.0 {
            0.7 // Reasonable
        } else if estimated_cost < 2.0 {
            0.5 // Expensive
        } else {
            0.3 // Very expensive
        }
    }
}

/// Agent scoring result
#[derive(Debug, Clone)]
struct AgentScore {
    agent_id: String,
    capability_match: f64,
    performance_score: f64,
    availability_score: f64,
    cost_score: f64,
    confidence: f64,
}

impl Default for CapabilityMatrix {
    fn default() -> Self {
        let mut capabilities = HashMap::new();
        
        // Define default capabilities for agent types
        capabilities.insert(
            "CodeGeneration".to_string(),
            vec![
                "coding".to_string(),
                "implementation".to_string(),
                "refactoring".to_string(),
                "debugging".to_string(),
            ],
        );
        
        capabilities.insert(
            "Documentation".to_string(),
            vec![
                "writing".to_string(),
                "documentation".to_string(),
                "technical_writing".to_string(),
                "editing".to_string(),
            ],
        );
        
        capabilities.insert(
            "Testing".to_string(),
            vec![
                "testing".to_string(),
                "test_planning".to_string(),
                "test_implementation".to_string(),
                "test_execution".to_string(),
                "validation".to_string(),
            ],
        );
        
        capabilities.insert(
            "Research".to_string(),
            vec![
                "research".to_string(),
                "analysis".to_string(),
                "web_search".to_string(),
                "information_gathering".to_string(),
            ],
        );
        
        capabilities.insert(
            "Design".to_string(),
            vec![
                "design".to_string(),
                "architecture".to_string(),
                "planning".to_string(),
                "system_design".to_string(),
            ],
        );
        
        Self {
            capabilities,
            performance_scores: HashMap::new(),
        }
    }
}