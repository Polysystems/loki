//! Distributed Decision Making System
//!
//! Implements consensus-based decision making across multiple agents
//! leveraging consciousness synchronization for collective intelligence.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc, broadcast};
use tokio::time::interval;
use tracing::{debug, info, warn, error};
// uuid import removed as unused

use crate::cognitive::{GoalId};
use crate::cognitive::consciousness::ConsciousnessState;
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::models::agent_specialization_router::AgentId;
use super::{
    AgentSpecialization,
    ConsciousnessSync, CollectiveConsciousness,
};

/// Configuration for distributed decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedDecisionConfig {
    /// Decision timeout
    pub decision_timeout: Duration,

    /// Minimum agents required for decision
    pub min_agents_for_decision: usize,

    /// Consensus threshold (0.0 - 1.0)
    pub consensus_threshold: f64,

    /// Maximum decision rounds
    pub max_decision_rounds: u32,

    /// Enable consciousness-informed decisions
    pub consciousness_informed: bool,

    /// Voting strategy
    pub voting_strategy: DistributedVotingStrategy,

    /// Decision quality threshold
    pub quality_threshold: f64,
}

impl Default for DistributedDecisionConfig {
    fn default() -> Self {
        Self {
            decision_timeout: Duration::from_secs(30),
            min_agents_for_decision: 3,
            consensus_threshold: 0.75,
            max_decision_rounds: 3,
            consciousness_informed: true,
            voting_strategy: DistributedVotingStrategy::ConsciousnessWeighted,
            quality_threshold: 0.7,
        }
    }
}

/// Voting strategies for distributed decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedVotingStrategy {
    /// Simple majority voting
    SimpleMajority,

    /// Weighted by agent specialization
    SpecializationWeighted,

    /// Weighted by consciousness coherence
    ConsciousnessWeighted,

    /// Weighted by agent performance
    PerformanceWeighted,

    /// Byzantine fault tolerant
    ByzantineFaultTolerant,

    /// Consensus with deliberation rounds
    DeliberativeConsensus,
}

/// Decision proposal for distributed consideration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionProposal {
    /// Unique proposal ID
    pub id: String,

    /// Proposing agent
    pub proposer: AgentId,

    /// Decision context
    pub context: DecisionContext,

    /// Available options
    pub options: Vec<DecisionOption>,

    /// Required agent capabilities
    pub required_capabilities: Vec<super::AgentCapability>,

    /// Decision urgency
    pub urgency: DecisionUrgency,

    /// Quality requirements
    pub quality_requirements: QualityRequirements,

    /// Deadline for decision
    pub deadline: Option<SystemTime>,

    /// Proposal timestamp
    pub created_at: SystemTime,
}

/// Context for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    /// Decision domain
    pub domain: DecisionDomain,

    /// Problem description
    pub problem: String,

    /// Relevant goals
    pub goals: Vec<GoalId>,

    /// Constraints
    pub constraints: Vec<String>,

    /// Available resources
    pub resources: HashMap<String, f64>,

    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,

    /// Success criteria
    pub success_criteria: Vec<String>,
}

/// Domain of decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionDomain {
    Strategic,
    Tactical,
    Operational,
    Creative,
    Analytical,
    Social,
    Ethical,
    Technical,
    Resource,
    Emergency,
}

/// Decision option with evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    /// Option ID
    pub id: String,

    /// Option description
    pub description: String,

    /// Estimated outcomes
    pub outcomes: Vec<Outcome>,

    /// Resource requirements
    pub resource_cost: HashMap<String, f64>,

    /// Risk assessment
    pub risk_level: f64,

    /// Expected utility
    pub expected_utility: f64,

    /// Implementation complexity
    pub complexity: f64,

    /// Time to implement
    pub time_to_implement: Duration,
}

/// Outcome prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// Outcome description
    pub description: String,

    /// Probability of occurrence
    pub probability: f64,

    /// Impact score
    pub impact: f64,

    /// Confidence in prediction
    pub confidence: f64,
}

/// Risk factor assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Risk description
    pub description: String,

    /// Risk probability
    pub probability: f64,

    /// Risk impact
    pub impact: f64,

    /// Mitigation strategies
    pub mitigation: Vec<String>,
}

/// Quality requirements for decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum confidence level
    pub min_confidence: f64,

    /// Require unanimous consensus
    pub require_unanimous: bool,

    /// Require specific expertise
    pub required_expertise: Vec<AgentSpecialization>,

    /// Quality metrics
    pub quality_metrics: Vec<String>,
}

/// Decision urgency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DecisionUrgency {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Agent's vote on a decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedVote {
    /// Voting agent
    pub agent_id: AgentId,

    /// Chosen option
    pub option_id: String,

    /// Vote confidence
    pub confidence: f64,

    /// Reasoning behind vote
    pub reasoning: String,

    /// Alternative preferences
    pub alternatives: Vec<AlternativePreference>,

    /// Consciousness state at time of vote
    pub consciousness_snapshot: Option<ConsciousnessState>,

    /// Vote timestamp
    pub timestamp: SystemTime,
}

/// Alternative option preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativePreference {
    /// Option ID
    pub option_id: String,

    /// Preference score
    pub score: f64,

    /// Reasoning
    pub reasoning: String,
}

/// Result of distributed decision process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedDecisionResult {
    /// Decision ID
    pub decision_id: String,

    /// Chosen option
    pub chosen_option: Option<DecisionOption>,

    /// Consensus reached
    pub consensus_reached: bool,

    /// Final consensus score
    pub consensus_score: f64,

    /// Participating agents
    pub participating_agents: Vec<AgentId>,

    /// All votes received
    pub votes: Vec<DistributedVote>,

    /// Decision rounds conducted
    pub rounds_conducted: u32,

    /// Decision quality metrics
    pub quality_metrics: DecisionQualityMetrics,

    /// Collective consciousness at decision time
    pub collective_consciousness: Option<CollectiveConsciousness>,

    /// Decision timestamp
    pub decided_at: SystemTime,

    /// Total decision time
    pub decision_duration: Duration,
}

/// Quality metrics for decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionQualityMetrics {
    /// Overall quality score
    pub overall_quality: f64,

    /// Confidence in decision
    pub confidence: f64,

    /// Consensus strength
    pub consensus_strength: f64,

    /// Expertise coverage
    pub expertise_coverage: f64,

    /// Risk assessment quality
    pub risk_assessment_quality: f64,

    /// Deliberation depth
    pub deliberation_depth: f64,
}

/// Distributed decision making system
pub struct DistributedDecisionMaker {
    /// Configuration
    config: DistributedDecisionConfig,

    /// Active decision processes
    active_decisions: Arc<RwLock<HashMap<String, DecisionProcess>>>,

    /// Decision history
    decision_history: Arc<RwLock<VecDeque<DistributedDecisionResult>>>,

    /// Consciousness sync system
    consciousness_sync: Arc<ConsciousnessSync>,

    /// Shared memory
    shared_memory: Arc<CognitiveMemory>,

    /// Decision event channel
    decision_tx: mpsc::Sender<DecisionEvent>,
    decision_rx: Arc<RwLock<mpsc::Receiver<DecisionEvent>>>,

    /// Result broadcast channel
    result_tx: broadcast::Sender<DistributedDecisionResult>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Internal decision process state
#[derive(Debug)]
struct DecisionProcess {
    proposal: DecisionProposal,
    votes: Vec<DistributedVote>,
    current_round: u32,
    started_at: SystemTime,
    participating_agents: Vec<AgentId>,
    deliberation_messages: Vec<DeliberationMessage>,
}

/// Decision events
#[derive(Debug, Clone)]
pub enum DecisionEvent {
    ProposalSubmitted(DecisionProposal),
    VoteReceived(String, DistributedVote), // decision_id, vote
    DeliberationMessage(String, DeliberationMessage), // decision_id, message
    DecisionTimeout(String), // decision_id
    ConsensusReached(String, DistributedDecisionResult), // decision_id, result
}

/// Deliberation message between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliberationMessage {
    /// Sender agent
    pub sender: AgentId,

    /// Message content
    pub content: String,

    /// Referenced option
    pub option_reference: Option<String>,

    /// Message type
    pub message_type: DeliberationMessageType,

    /// Timestamp
    pub timestamp: SystemTime,
}

/// Types of deliberation messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliberationMessageType {
    Question,
    Clarification,
    Objection,
    Support,
    Alternative,
    Information,
    Concern,
}

impl DistributedDecisionMaker {
    /// Create new distributed decision maker
    pub async fn new(
        config: DistributedDecisionConfig,
        consciousness_sync: Arc<ConsciousnessSync>,
        shared_memory: Arc<CognitiveMemory>,
    ) -> Result<Self> {
        info!("🎯 Initializing Distributed Decision Making System");

        let (decision_tx, decision_rx) = mpsc::channel(1000);
        let (result_tx, _) = broadcast::channel(100);

        Ok(Self {
            config,
            active_decisions: Arc::new(RwLock::new(HashMap::new())),
            decision_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            consciousness_sync,
            shared_memory,
            decision_tx,
            decision_rx: Arc::new(RwLock::new(decision_rx)),
            result_tx,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the decision making system
    pub async fn start(self: Arc<Self>) -> Result<()> {
        *self.running.write().await = true;
        info!("✨ Distributed Decision Making System started");

        // Start decision processing loop
        let process_self = self.clone();
        tokio::spawn(async move {
            if let Err(e) = process_self.spawn_decision_processor().await {
                tracing::error!("Decision processor failed: {}", e);
            }
        });

        // Start timeout monitoring
        let timeout_self = self.clone();
        tokio::spawn(async move {
            if let Err(e) = timeout_self.spawn_timeout_monitor().await {
                tracing::error!("Timeout monitor failed: {}", e);
            }
        });

        Ok(())
    }

    /// Stop the decision making system
    pub async fn stop(&self) -> Result<()> {
        *self.running.write().await = false;
        info!("🛑 Distributed Decision Making System stopped");
        Ok(())
    }

    /// Submit a decision proposal
    pub async fn submit_proposal(&self, proposal: DecisionProposal) -> Result<()> {
        info!("📋 Submitting decision proposal: {}", proposal.id);

        // Validate proposal
        self.validate_proposal(&proposal).await?;

        // Create decision process
        let process = DecisionProcess {
            proposal: proposal.clone(),
            votes: Vec::new(),
            current_round: 1,
            started_at: SystemTime::now(),
            participating_agents: Vec::new(),
            deliberation_messages: Vec::new(),
        };

        // Store active decision
        self.active_decisions.write().await.insert(proposal.id.clone(), process);

        // Send event
        self.decision_tx.send(DecisionEvent::ProposalSubmitted(proposal)).await?;

        Ok(())
    }

    /// Submit a vote for a decision
    pub async fn submit_vote(&self, decision_id: String, vote: DistributedVote) -> Result<()> {
        info!("🗳️ Submitting vote from {} for decision {}", vote.agent_id, decision_id);

        // Validate vote
        self.validate_vote(&decision_id, &vote).await?;

        // Send event
        self.decision_tx.send(DecisionEvent::VoteReceived(decision_id, vote)).await?;

        Ok(())
    }

    /// Submit deliberation message
    pub async fn submit_deliberation(
        &self,
        decision_id: String,
        message: DeliberationMessage,
    ) -> Result<()> {
        debug!("💬 Deliberation message from {} for decision {}", message.sender, decision_id);

        // Send event
        self.decision_tx.send(DecisionEvent::DeliberationMessage(decision_id, message)).await?;

        Ok(())
    }

    /// Get active decisions
    pub async fn get_active_decisions(&self) -> Vec<DecisionProposal> {
        self.active_decisions.read().await
            .values()
            .map(|process| process.proposal.clone())
            .collect()
    }

    /// Get decision result
    pub async fn get_decision_result(&self, decision_id: &str) -> Option<DistributedDecisionResult> {
        self.decision_history.read().await
            .iter()
            .find(|result| result.decision_id == decision_id)
            .cloned()
    }

    /// Subscribe to decision results
    pub fn subscribe_to_results(&self) -> broadcast::Receiver<DistributedDecisionResult> {
        self.result_tx.subscribe()
    }

    /// Main decision processing loop
    async fn spawn_decision_processor(self: Arc<Self>) -> Result<()> {
        let mut rx = self.decision_rx.write().await;

        while *self.running.read().await {
            if let Some(event) = rx.recv().await {
                if let Err(e) = self.handle_decision_event(event).await {
                    error!("Decision processing error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle decision events
    async fn handle_decision_event(&self, event: DecisionEvent) -> Result<()> {
        match event {
            DecisionEvent::ProposalSubmitted(proposal) => {
                self.process_new_proposal(proposal).await?;
            }
            DecisionEvent::VoteReceived(decision_id, vote) => {
                self.process_vote(decision_id, vote).await?;
            }
            DecisionEvent::DeliberationMessage(decision_id, message) => {
                self.process_deliberation(decision_id, message).await?;
            }
            DecisionEvent::DecisionTimeout(decision_id) => {
                self.handle_decision_timeout(decision_id).await?;
            }
            DecisionEvent::ConsensusReached(decision_id, result) => {
                self.finalize_decision(decision_id, result).await?;
            }
        }
        Ok(())
    }

    /// Process new proposal
    async fn process_new_proposal(&self, proposal: DecisionProposal) -> Result<()> {
        info!("🚀 Processing new proposal: {}", proposal.id);

        // Determine participating agents based on requirements
        let participating_agents = self.select_participating_agents(&proposal).await?;

        // Update process with participating agents
        if let Some(process) = self.active_decisions.write().await.get_mut(&proposal.id) {
            process.participating_agents = participating_agents;
        }

        // Broadcast proposal to participating agents
        self.broadcast_proposal_to_agents(&proposal).await?;

        Ok(())
    }

    /// Process received vote
    async fn process_vote(&self, decision_id: String, vote: DistributedVote) -> Result<()> {
        let mut active_decisions = self.active_decisions.write().await;

        if let Some(process) = active_decisions.get_mut(&decision_id) {
            // Add vote
            process.votes.push(vote);

            // Check if we have enough votes
            if process.votes.len() >= self.config.min_agents_for_decision {
                // Calculate consensus
                let consensus_result = self.calculate_consensus(&process).await?;

                if consensus_result.consensus_reached {
                    // Consensus reached!
                    let result = self.create_decision_result(&process, consensus_result).await?;

                    // Send consensus event
                    self.decision_tx.send(DecisionEvent::ConsensusReached(decision_id.clone(), result)).await?;
                } else if process.current_round < self.config.max_decision_rounds {
                    // Start new deliberation round
                    process.current_round += 1;
                    self.start_deliberation_round(&process).await?;
                } else {
                    // Max rounds reached without consensus
                    let result = self.create_no_consensus_result(&process).await?;
                    self.decision_tx.send(DecisionEvent::ConsensusReached(decision_id.clone(), result)).await?;
                }
            }
        }

        Ok(())
    }

    /// Calculate consensus based on voting strategy
    async fn calculate_consensus(&self, process: &DecisionProcess) -> Result<ConsensusCalculation> {
        let votes = &process.votes;

        if votes.is_empty() {
            return Ok(ConsensusCalculation {
                consensus_reached: false,
                consensus_score: 0.0,
                chosen_option: None,
                quality_score: 0.0,
            });
        }

        let result = match self.config.voting_strategy {
            DistributedVotingStrategy::SimpleMajority => {
                self.calculate_simple_majority(votes).await?
            }
            DistributedVotingStrategy::ConsciousnessWeighted => {
                self.calculate_consciousness_weighted(votes).await?
            }
            DistributedVotingStrategy::SpecializationWeighted => {
                self.calculate_specialization_weighted(votes).await?
            }
            DistributedVotingStrategy::ByzantineFaultTolerant => {
                self.calculate_byzantine_consensus(votes).await?
            }
            DistributedVotingStrategy::DeliberativeConsensus => {
                self.calculate_deliberative_consensus(process).await?
            }
            _ => self.calculate_simple_majority(votes).await?,
        };

        Ok(result)
    }

    /// Calculate simple majority consensus
    async fn calculate_simple_majority(&self, votes: &[DistributedVote]) -> Result<ConsensusCalculation> {
        let mut option_counts = HashMap::new();
        let mut total_confidence = 0.0;

        for vote in votes {
            *option_counts.entry(vote.option_id.clone()).or_insert(0) += 1;
            total_confidence += vote.confidence;
        }

        let majority_threshold = (votes.len() as f64 * self.config.consensus_threshold).ceil() as usize;
        let avg_confidence = total_confidence / votes.len() as f64;

        let (chosen_option, max_count) = option_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(option, count)| (option.clone(), *count))
            .unwrap_or_default();

        Ok(ConsensusCalculation {
            consensus_reached: max_count >= majority_threshold,
            consensus_score: max_count as f64 / votes.len() as f64,
            chosen_option: if max_count >= majority_threshold { Some(chosen_option) } else { None },
            quality_score: avg_confidence,
        })
    }

    /// Calculate consciousness-weighted consensus
    async fn calculate_consciousness_weighted(&self, votes: &[DistributedVote]) -> Result<ConsensusCalculation> {
        let mut option_weights = HashMap::new();
        let mut total_weight = 0.0;

        for vote in votes {
            let weight = if let Some(ref consciousness) = vote.consciousness_snapshot {
                consciousness.coherence_score * consciousness.awareness_level * vote.confidence
            } else {
                vote.confidence
            };

            *option_weights.entry(vote.option_id.clone()).or_insert(0.0) += weight;
            total_weight += weight;
        }

        let consensus_threshold = total_weight * self.config.consensus_threshold;

        let (chosen_option, max_weight) = option_weights
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(option, weight)| (option.clone(), *weight))
            .unwrap_or_default();

        Ok(ConsensusCalculation {
            consensus_reached: max_weight >= consensus_threshold,
            consensus_score: max_weight / total_weight,
            chosen_option: if max_weight >= consensus_threshold { Some(chosen_option) } else { None },
            quality_score: max_weight / total_weight,
        })
    }

    /// Calculate specialization-weighted consensus
    async fn calculate_specialization_weighted(&self, votes: &[DistributedVote]) -> Result<ConsensusCalculation> {
        // For now, treat all specializations equally
        // This would be enhanced with actual specialization weights
        self.calculate_simple_majority(votes).await
    }

    /// Calculate Byzantine fault tolerant consensus
    async fn calculate_byzantine_consensus(&self, votes: &[DistributedVote]) -> Result<ConsensusCalculation> {
        // Simplified Byzantine consensus - requires 2/3 majority
        let byzantine_threshold = (votes.len() as f64 * 0.67).ceil() as usize;

        let mut option_counts = HashMap::new();
        for vote in votes {
            *option_counts.entry(vote.option_id.clone()).or_insert(0) += 1;
        }

        let (chosen_option, max_count) = option_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(option, count)| (option.clone(), *count))
            .unwrap_or_default();

        Ok(ConsensusCalculation {
            consensus_reached: max_count >= byzantine_threshold,
            consensus_score: max_count as f64 / votes.len() as f64,
            chosen_option: if max_count >= byzantine_threshold { Some(chosen_option) } else { None },
            quality_score: 0.8, // Fixed quality for Byzantine consensus
        })
    }

    /// Calculate deliberative consensus
    async fn calculate_deliberative_consensus(&self, process: &DecisionProcess) -> Result<ConsensusCalculation> {
        // Enhanced consensus considering deliberation quality
        let base_consensus = self.calculate_simple_majority(&process.votes).await?;

        // Boost quality based on deliberation
        let deliberation_boost = (process.deliberation_messages.len() as f64 * 0.1).min(0.3);
        let enhanced_quality = (base_consensus.quality_score + deliberation_boost).min(1.0);

        Ok(ConsensusCalculation {
            consensus_reached: base_consensus.consensus_reached,
            consensus_score: base_consensus.consensus_score,
            chosen_option: base_consensus.chosen_option,
            quality_score: enhanced_quality,
        })
    }

    /// Validate proposal
    async fn validate_proposal(&self, proposal: &DecisionProposal) -> Result<()> {
        if proposal.options.is_empty() {
            return Err(anyhow::anyhow!("Proposal must have at least one option"));
        }

        if proposal.options.len() > 10 {
            return Err(anyhow::anyhow!("Too many options (max 10)"));
        }

        Ok(())
    }

    /// Validate vote
    async fn validate_vote(&self, decision_id: &str, vote: &DistributedVote) -> Result<()> {
        let active_decisions = self.active_decisions.read().await;

        if let Some(process) = active_decisions.get(decision_id) {
            // Check if option exists
            if !process.proposal.options.iter().any(|opt| opt.id == vote.option_id) {
                return Err(anyhow::anyhow!("Invalid option ID: {}", vote.option_id));
            }

            // Check if agent already voted in this round
            if process.votes.iter().any(|v| v.agent_id == vote.agent_id) {
                return Err(anyhow::anyhow!("Agent {} already voted", vote.agent_id));
            }
        } else {
            return Err(anyhow::anyhow!("Decision {} not found", decision_id));
        }

        Ok(())
    }

    /// Select participating agents
    async fn select_participating_agents(&self, _proposal: &DecisionProposal) -> Result<Vec<AgentId>> {
        // For now, return all available agents
        // This would be enhanced with capability matching
        let collective = self.consciousness_sync.get_collective_state().await;
        Ok(collective.agent_states.keys().cloned().collect())
    }

    /// Broadcast proposal to agents
    async fn broadcast_proposal_to_agents(&self, proposal: &DecisionProposal) -> Result<()> {
        info!("📢 Broadcasting proposal {} to agents", proposal.id);
        // Implementation would send coordination messages
        Ok(())
    }

    /// Start deliberation round
    async fn start_deliberation_round(&self, process: &DecisionProcess) -> Result<()> {
        info!("🗣️ Starting deliberation round {} for decision {}",
            process.current_round, process.proposal.id);
        // Implementation would facilitate agent deliberation
        Ok(())
    }

    /// Create decision result
    async fn create_decision_result(
        &self,
        process: &DecisionProcess,
        consensus: ConsensusCalculation,
    ) -> Result<DistributedDecisionResult> {
        let chosen_option = if let Some(ref option_id) = consensus.chosen_option {
            process.proposal.options.iter().find(|opt| opt.id == *option_id).cloned()
        } else {
            None
        };

        let quality_metrics = DecisionQualityMetrics {
            overall_quality: consensus.quality_score,
            confidence: consensus.quality_score,
            consensus_strength: consensus.consensus_score,
            expertise_coverage: 0.8, // Would be calculated based on participating agents
            risk_assessment_quality: 0.7,
            deliberation_depth: process.deliberation_messages.len() as f64 * 0.1,
        };

        Ok(DistributedDecisionResult {
            decision_id: process.proposal.id.clone(),
            chosen_option,
            consensus_reached: consensus.consensus_reached,
            consensus_score: consensus.consensus_score,
            participating_agents: process.participating_agents.clone(),
            votes: process.votes.clone(),
            rounds_conducted: process.current_round,
            quality_metrics,
            collective_consciousness: Some(self.consciousness_sync.get_collective_state().await),
            decided_at: SystemTime::now(),
            decision_duration: SystemTime::now().duration_since(process.started_at).unwrap_or_default(),
        })
    }

    /// Create no consensus result
    async fn create_no_consensus_result(&self, process: &DecisionProcess) -> Result<DistributedDecisionResult> {
        let quality_metrics = DecisionQualityMetrics {
            overall_quality: 0.3,
            confidence: 0.3,
            consensus_strength: 0.0,
            expertise_coverage: 0.5,
            risk_assessment_quality: 0.5,
            deliberation_depth: process.deliberation_messages.len() as f64 * 0.1,
        };

        Ok(DistributedDecisionResult {
            decision_id: process.proposal.id.clone(),
            chosen_option: None,
            consensus_reached: false,
            consensus_score: 0.0,
            participating_agents: process.participating_agents.clone(),
            votes: process.votes.clone(),
            rounds_conducted: process.current_round,
            quality_metrics,
            collective_consciousness: Some(self.consciousness_sync.get_collective_state().await),
            decided_at: SystemTime::now(),
            decision_duration: SystemTime::now().duration_since(process.started_at).unwrap_or_default(),
        })
    }

    /// Finalize decision
    async fn finalize_decision(&self, decision_id: String, result: DistributedDecisionResult) -> Result<()> {
        info!("✅ Finalizing decision {}: consensus={}", decision_id, result.consensus_reached);

        // Remove from active decisions
        self.active_decisions.write().await.remove(&decision_id);

        // Add to history
        let mut history = self.decision_history.write().await;
        if history.len() >= 1000 {
            history.pop_front();
        }
        history.push_back(result.clone());

        // Store in shared memory
        self.shared_memory.store(
            format!("Decision Result: {}", decision_id),
            vec![],
            MemoryMetadata {
                source: "distributed_decision".to_string(),
                tags: vec!["decision".to_string(), "consensus".to_string()],
                importance: 0.9,
                associations: vec![],
                context: Some(format!("Decision: {}", decision_id)),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                timestamp: chrono::Utc::now(),
                expiration: None,
                category: "multi_agent".to_string(),
            }
        ).await?;

        // Broadcast result
        let _ = self.result_tx.send(result);

        Ok(())
    }

    /// Spawn timeout monitor
    async fn spawn_timeout_monitor(self: Arc<Self>) -> Result<()> {
        let mut interval = interval(Duration::from_secs(5));

        while *self.running.read().await {
            interval.tick().await;

            let mut timed_out = Vec::new();

            {
                let active_decisions = self.active_decisions.read().await;
                for (decision_id, process) in active_decisions.iter() {
                    if let Some(deadline) = process.proposal.deadline {
                        if SystemTime::now() > deadline {
                            timed_out.push(decision_id.clone());
                        }
                    } else if SystemTime::now().duration_since(process.started_at).unwrap_or_default() > self.config.decision_timeout {
                        timed_out.push(decision_id.clone());
                    }
                }
            }

            for decision_id in timed_out {
                let _ = self.decision_tx.send(DecisionEvent::DecisionTimeout(decision_id)).await;
            }
        }

        Ok(())
    }

    /// Process deliberation message
    async fn process_deliberation(&self, decision_id: String, message: DeliberationMessage) -> Result<()> {
        debug!("Processing deliberation message for decision: {}", decision_id);

        // Get active decision process
        let mut processes = self.active_decisions.write().await;
        if let Some(process) = processes.get_mut(&decision_id) {
            // Add message to deliberation history
            process.deliberation_messages.push(message.clone());

            // Process message based on type
            match message.message_type {
                DeliberationMessageType::Question => {
                    // Handle question message
                    debug!("Processing question from {}: {}", message.sender, message.content);
                }
                DeliberationMessageType::Clarification => {
                    // Handle clarification message
                    debug!("Processing clarification from {}: {}", message.sender, message.content);
                }
                DeliberationMessageType::Objection => {
                    // Handle objection message
                    debug!("Processing objection from {}: {}", message.sender, message.content);
                }
                DeliberationMessageType::Support => {
                    // Handle support message
                    debug!("Processing support from {}: {}", message.sender, message.content);
                }
                DeliberationMessageType::Alternative => {
                    // Handle alternative message
                    debug!("Processing alternative from {}: {}", message.sender, message.content);
                }
                DeliberationMessageType::Information => {
                    // Handle information message
                    debug!("Processing information from {}: {}", message.sender, message.content);
                }
                DeliberationMessageType::Concern => {
                    // Handle concern message
                    debug!("Processing concern from {}: {}", message.sender, message.content);
                }
            }

            // Note: Process activity is tracked via started_at field
        }

        Ok(())
    }

    /// Handle decision timeout
    async fn handle_decision_timeout(&self, decision_id: String) -> Result<()> {
        warn!("⏰ Decision {} timed out", decision_id);

        if let Some(process) = self.active_decisions.read().await.get(&decision_id) {
            // Create timeout result
            let result = self.create_no_consensus_result(process).await?;
            self.decision_tx.send(DecisionEvent::ConsensusReached(decision_id, result)).await?;
        }

        Ok(())
    }
}

/// Internal consensus calculation result
#[derive(Debug)]
struct ConsensusCalculation {
    consensus_reached: bool,
    consensus_score: f64,
    chosen_option: Option<String>,
    quality_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_distributed_decision_creation() {
        let config = DistributedDecisionConfig::default();
        let memory = Arc::new(CognitiveMemory::new(Default::default()).await.unwrap());
        let consciousness_sync = Arc::new(
            ConsciousnessSync::new(Default::default(), memory.clone()).await.unwrap()
        );

        let decision_maker = DistributedDecisionMaker::new(config, consciousness_sync, memory).await.unwrap();
        assert!(!*decision_maker.running.read().await);
    }

    #[tokio::test]
    async fn test_simple_majority_consensus() {
        let config = DistributedDecisionConfig::default();
        let memory = Arc::new(CognitiveMemory::new(Default::default()).await.unwrap());
        let consciousness_sync = Arc::new(
            ConsciousnessSync::new(Default::default(), memory.clone()).await.unwrap()
        );

        let decision_maker = Arc::new(
            DistributedDecisionMaker::new(config, consciousness_sync, memory).await.unwrap()
        );

        let votes = vec![
            DistributedVote {
                agent_id: AgentId::new_v4(),
                option_id: "option_a".to_string(),
                confidence: 0.8,
                reasoning: "Good choice".to_string(),
                alternatives: vec![],
                consciousness_snapshot: None,
                timestamp: SystemTime::now(),
            },
            DistributedVote {
                agent_id: AgentId::new_v4(),
                option_id: "option_a".to_string(),
                confidence: 0.9,
                reasoning: "Agree".to_string(),
                alternatives: vec![],
                consciousness_snapshot: None,
                timestamp: SystemTime::now(),
            },
            DistributedVote {
                agent_id: AgentId::new_v4(),
                option_id: "option_b".to_string(),
                confidence: 0.7,
                reasoning: "Different approach".to_string(),
                alternatives: vec![],
                consciousness_snapshot: None,
                timestamp: SystemTime::now(),
            },
        ];

        let result = decision_maker.calculate_simple_majority(&votes).await.unwrap();
        assert!(result.consensus_reached);
        assert_eq!(result.chosen_option, Some("option_a".to_string()));
        assert!(result.consensus_score > 0.6);
    }
}
