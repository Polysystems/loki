//! Consensus Mechanism
//!
//! Implements various consensus algorithms for multi-agent decision making,
//! including weighted voting, Byzantine fault tolerance, and emergent consensus.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};
use crate::models::agent_specialization_router::AgentId;

/// Consensus mechanism for multi-agent decisions
pub struct ConsensusMechanism {
    /// Voting strategy
    strategy: VotingStrategy,

    /// Consensus threshold
    threshold: f32,

    /// Active voting rounds
    active_rounds: Arc<RwLock<HashMap<String, VotingRound>>>,

    /// Consensus history
    history: Arc<RwLock<Vec<ConsensusResult>>>,

    /// Metrics
    metrics: Arc<RwLock<ConsensusMetrics>>,
}

/// Voting strategies
#[derive(Debug, Clone)]
pub enum VotingStrategy {
    /// Simple majority (>50%)
    SimpleMajority,

    /// Weighted majority based on agent performance
    WeightedMajority,

    /// Supermajority (>66%)
    Supermajority,

    /// Unanimous decision
    Unanimous,

    /// Byzantine fault tolerant (>2/3)
    Byzantine,

    /// Ranked choice voting
    RankedChoice,

    /// Quadratic voting
    Quadratic,
}

/// Voting round
#[derive(Debug, Clone)]
pub struct VotingRound {
    pub id: String,
    pub topic: String,
    pub options: Vec<ConsensusOption>,
    pub votes: HashMap<AgentId, Vote>,
    pub started_at: Instant,
    pub deadline: Instant,
    pub strategy: VotingStrategy,
    pub threshold: f32,
}

/// Consensus option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusOption {
    pub id: String,
    pub description: String,
    pub data: serde_json::Value,
}

/// Individual vote
#[derive(Debug, Clone)]
pub struct Vote {
    pub agent_id: AgentId,
    pub option_id: String,
    pub weight: f32,
    pub confidence: f32,
    pub reasoning: String,
    pub timestamp: Instant,
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub round_id: String,
    pub topic: String,
    pub consensus_reached: bool,
    pub winning_option: Option<String>,
    pub vote_distribution: HashMap<String, f32>,
    pub participation_rate: f32,
    pub confidence_score: f32,
    pub duration: Duration,
    pub dissenting_agents: Vec<AgentId>,
}

/// Consensus metrics
#[derive(Debug, Clone, Default)]
pub struct ConsensusMetrics {
    pub total_rounds: u64,
    pub successful_rounds: u64,
    pub failed_rounds: u64,
    pub average_participation: f32,
    pub average_confidence: f32,
    pub average_duration: Duration,
    pub strategy_success_rates: HashMap<VotingStrategy, f32>,
}

impl ConsensusMechanism {
    /// Create a new consensus mechanism
    pub async fn new(strategy: VotingStrategy, threshold: f32) -> Result<Self> {
        Ok(Self {
            strategy,
            threshold: threshold.clamp(0.0, 1.0),
            active_rounds: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            metrics: Arc::new(RwLock::new(ConsensusMetrics::default())),
        })
    }

    /// Start a new voting round
    pub async fn start_round(
        &self,
        topic: String,
        options: Vec<ConsensusOption>,
        deadline: Duration,
    ) -> Result<String> {
        let round_id = uuid::Uuid::new_v4().to_string();
        let now = Instant::now();

        let round = VotingRound {
            id: round_id.clone(),
            topic: topic.clone(),
            options,
            votes: HashMap::new(),
            started_at: now,
            deadline: now + deadline,
            strategy: self.strategy.clone(),
            threshold: self.threshold,
        };

        self.active_rounds.write().await.insert(round_id.clone(), round);

        info!("Started consensus round {} on topic: {}", round_id, topic);

        Ok(round_id)
    }

    /// Submit a vote
    pub async fn submit_vote(
        &self,
        round_id: &str,
        agent_id: AgentId,
        option_id: String,
        confidence: f32,
        reasoning: String,
    ) -> Result<()> {
        let mut rounds = self.active_rounds.write().await;

        let round = rounds.get_mut(round_id)
            .ok_or_else(|| anyhow!("Voting round not found"))?;

        if Instant::now() > round.deadline {
            return Err(anyhow!("Voting deadline has passed"));
        }

        // Calculate vote weight based on strategy
        let weight = match &round.strategy {
            VotingStrategy::SimpleMajority => 1.0,
            VotingStrategy::WeightedMajority => confidence, // Weight by confidence
            VotingStrategy::Quadratic => confidence.sqrt(), // Quadratic voting
            _ => 1.0,
        };

        let vote = Vote {
            agent_id: agent_id.clone(),
            option_id,
            weight,
            confidence,
            reasoning,
            timestamp: Instant::now(),
        };

        round.votes.insert(agent_id, vote);

        debug!("Vote submitted for round {}", round_id);

        Ok(())
    }

    /// Process votes and determine consensus
    pub async fn process_votes(&self, votes: Vec<ConsensusVote>) -> Result<ConsensusResult> {
        // Create temporary round for processing
        let round_id = uuid::Uuid::new_v4().to_string();
        let mut vote_map = HashMap::new();

        for vote in votes {
            vote_map.insert(
                vote.agent_id.clone(),
                Vote {
                    agent_id: vote.agent_id,
                    option_id: vote.option_id,
                    weight: vote.confidence,
                    confidence: vote.confidence,
                    reasoning: vote.reasoning,
                    timestamp: Instant::now(),
                }
            );
        }

        let result = self.calculate_consensus(&round_id, "Ad-hoc consensus", &vote_map)?;

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_rounds += 1;
        if result.consensus_reached {
            metrics.successful_rounds += 1;
        } else {
            metrics.failed_rounds += 1;
        }

        Ok(result)
    }

    /// Close a voting round and calculate results
    pub async fn close_round(&self, round_id: &str) -> Result<ConsensusResult> {
        let mut rounds = self.active_rounds.write().await;

        let round = rounds.remove(round_id)
            .ok_or_else(|| anyhow!("Voting round not found"))?;

        let result = self.calculate_consensus(&round.id, &round.topic, &round.votes)?;

        // Store in history
        self.history.write().await.push(result.clone());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_rounds += 1;
        if result.consensus_reached {
            metrics.successful_rounds += 1;
        } else {
            metrics.failed_rounds += 1;
        }
        metrics.average_participation =
            (metrics.average_participation * (metrics.total_rounds - 1) as f32 + result.participation_rate)
            / metrics.total_rounds as f32;
        metrics.average_confidence =
            (metrics.average_confidence * (metrics.total_rounds - 1) as f32 + result.confidence_score)
            / metrics.total_rounds as f32;

        info!("Consensus round {} closed. Result: {:?}", round_id, result.consensus_reached);

        Ok(result)
    }

    /// Calculate consensus from votes
    fn calculate_consensus(
        &self,
        round_id: &str,
        topic: &str,
        votes: &HashMap<AgentId, Vote>,
    ) -> Result<ConsensusResult> {
        let start_time = Instant::now();

        // Count votes by option
        let mut vote_counts: HashMap<String, f32> = HashMap::new();
        let mut total_weight = 0.0;
        let mut total_confidence = 0.0;

        for vote in votes.values() {
            *vote_counts.entry(vote.option_id.clone()).or_insert(0.0) += vote.weight;
            total_weight += vote.weight;
            total_confidence += vote.confidence;
        }

        // Normalize vote distribution
        let mut vote_distribution: HashMap<String, f32> = HashMap::new();
        for (option, count) in vote_counts {
            vote_distribution.insert(option, count / total_weight.max(1.0));
        }

        // Determine winning option based on strategy
        let (consensus_reached, winning_option) = match &self.strategy {
            VotingStrategy::SimpleMajority => {
                let max_vote = vote_distribution.values().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
                (max_vote > 0.5, vote_distribution.iter().find(|(_, &v)| v == max_vote).map(|(k, _)| k.clone()))
            }
            VotingStrategy::Supermajority => {
                let max_vote = vote_distribution.values().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
                (max_vote > 0.66, vote_distribution.iter().find(|(_, &v)| v == max_vote).map(|(k, _)| k.clone()))
            }
            VotingStrategy::Byzantine => {
                let max_vote = vote_distribution.values().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
                (max_vote > 0.67, vote_distribution.iter().find(|(_, &v)| v == max_vote).map(|(k, _)| k.clone()))
            }
            VotingStrategy::Unanimous => {
                let max_vote = vote_distribution.values().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
                (max_vote >= 0.99, vote_distribution.iter().find(|(_, &v)| v == max_vote).map(|(k, _)| k.clone()))
            }
            _ => {
                // Default to threshold-based
                let max_vote = vote_distribution.values().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0);
                (max_vote >= self.threshold, vote_distribution.iter().find(|(_, &v)| v == max_vote).map(|(k, _)| k.clone()))
            }
        };

        // Find dissenting agents
        let dissenting_agents = if let Some(ref winning) = winning_option {
            votes.iter()
                .filter(|(_, vote)| &vote.option_id != winning)
                .map(|(id, _)| id.clone())
                .collect()
        } else {
            Vec::new()
        };

        let participation_rate = votes.len() as f32 / 10.0; // Assume max 10 agents for now
        let confidence_score = if votes.is_empty() { 0.0 } else { total_confidence / votes.len() as f32 };

        Ok(ConsensusResult {
            round_id: round_id.to_string(),
            topic: topic.to_string(),
            consensus_reached,
            winning_option,
            vote_distribution,
            participation_rate: participation_rate.min(1.0),
            confidence_score,
            duration: start_time.elapsed(),
            dissenting_agents,
        })
    }

    /// Get consensus history
    pub async fn get_history(&self, limit: usize) -> Result<Vec<ConsensusResult>> {
        let history = self.history.read().await;
        let start = history.len().saturating_sub(limit);
        Ok(history[start..].to_vec())
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> Result<ConsensusMetrics> {
        Ok(self.metrics.read().await.clone())
    }

    /// Update voting strategy
    pub async fn update_strategy(&mut self, strategy: VotingStrategy) {
        self.strategy = strategy;
        info!("Updated voting strategy to {:?}", self.strategy);
    }
}

/// Vote structure for external use
#[derive(Debug, Clone)]
pub struct ConsensusVote {
    pub agent_id: AgentId,
    pub option_id: String,
    pub confidence: f32,
    pub reasoning: String,
}

impl PartialEq for VotingStrategy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::SimpleMajority, Self::SimpleMajority) => true,
            (Self::WeightedMajority, Self::WeightedMajority) => true,
            (Self::Supermajority, Self::Supermajority) => true,
            (Self::Unanimous, Self::Unanimous) => true,
            (Self::Byzantine, Self::Byzantine) => true,
            (Self::RankedChoice, Self::RankedChoice) => true,
            (Self::Quadratic, Self::Quadratic) => true,
            _ => false,
        }
    }
}

impl Eq for VotingStrategy {}

impl std::hash::Hash for VotingStrategy {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::SimpleMajority => 0.hash(state),
            Self::WeightedMajority => 1.hash(state),
            Self::Supermajority => 2.hash(state),
            Self::Unanimous => 3.hash(state),
            Self::Byzantine => 4.hash(state),
            Self::RankedChoice => 5.hash(state),
            Self::Quadratic => 6.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_mechanism() {
        let consensus = ConsensusMechanism::new(VotingStrategy::SimpleMajority, 0.5)
            .await
            .unwrap();

        // Test voting round
        let options = vec![
            ConsensusOption {
                id: "option1".to_string(),
                description: "Option 1".to_string(),
                data: serde_json::json!({}),
            },
            ConsensusOption {
                id: "option2".to_string(),
                description: "Option 2".to_string(),
                data: serde_json::json!({}),
            },
        ];

        let round_id = consensus.start_round(
            "Test decision".to_string(),
            options,
            Duration::from_secs(60),
        ).await.unwrap();

        // Submit votes
        consensus.submit_vote(
            &round_id,
            AgentId::new_v4(),
            "option1".to_string(),
            0.8,
            "Reasoning 1".to_string(),
        ).await.unwrap();

        consensus.submit_vote(
            &round_id,
            AgentId::new_v4(),
            "option1".to_string(),
            0.9,
            "Reasoning 2".to_string(),
        ).await.unwrap();

        consensus.submit_vote(
            &round_id,
            AgentId::new_v4(),
            "option2".to_string(),
            0.7,
            "Reasoning 3".to_string(),
        ).await.unwrap();

        // Close round and check result
        let result = consensus.close_round(&round_id).await.unwrap();
        assert!(result.consensus_reached);
        assert_eq!(result.winning_option, Some("option1".to_string()));
    }
}
