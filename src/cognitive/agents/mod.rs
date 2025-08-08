//! Multi-Agent Cognitive System
//!
//! This module implements a distributed multi-agent architecture where
//! specialized agents collaborate to achieve complex cognitive tasks.

pub mod orchestrator;
pub mod specialized_agent;
pub mod coordination_protocol;
pub mod consensus;
pub mod task_distributor;
pub mod consciousness_sync;
pub mod distributed_decision;

// Core agent types
pub use specialized_agent::{
    SpecializedAgent, SpecializedAgent as Agent, AgentSpecialization,
    AgentCapability, AgentConfiguration, AgentConfig, TaskResult, DynamicRole,
};
pub use orchestrator::{MultiAgentOrchestrator, OrchestratorConfig, Task, TaskType, TaskPriority, ConsensusRequest};
pub use coordination_protocol::{CoordinationProtocol, CoordinationMessage, MessageType, MessagePriority};
pub use consensus::{ConsensusMechanism, ConsensusResult, VotingStrategy, ConsensusOption, ConsensusVote};
pub use task_distributor::{TaskDistributor, TaskAllocation, LoadBalancingStrategy};
pub use consciousness_sync::{ConsciousnessSync, ConsciousnessSyncConfig, CollectiveConsciousness, EmergenceIndicator};
pub use distributed_decision::{
    DistributedDecisionMaker, DistributedDecisionConfig, DecisionProposal, DecisionContext,
    DecisionOption, DistributedVote, DistributedDecisionResult, DecisionDomain, DecisionUrgency,
    DecisionQualityMetrics,
};

// Note: AgentConfig is now a separate struct for runtime configuration
// AgentConfiguration is for static agent setup

// Agent entry for task distribution
#[derive(Debug, Clone)]
pub struct AgentEntry {
    pub agent_id: String,
    pub capabilities: Vec<AgentCapability>,
    pub current_load: f32,
    pub status: AgentStatus,
    pub task_count: usize,
    pub performance_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Active,
    Busy,
    Idle,
    Offline,
}
