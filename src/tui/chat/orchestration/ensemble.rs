//! Ensemble voting and result aggregation
//! 
//! This module implements ensemble methods for combining outputs
//! from multiple models to improve quality and reliability.

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

/// Configuration for ensemble voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Minimum number of models required for ensemble
    pub min_models: usize,
    
    /// Maximum number of models to use
    pub max_models: usize,
    
    /// Voting strategy
    pub voting_strategy: VotingStrategy,
    
    /// Confidence threshold for consensus
    pub consensus_threshold: f32,
    
    /// Timeout for waiting on all models
    pub ensemble_timeout_ms: u64,
    
    /// Whether to wait for all models or return early
    pub wait_for_all: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            min_models: 2,
            max_models: 5,
            voting_strategy: VotingStrategy::WeightedConsensus,
            consensus_threshold: 0.7,
            ensemble_timeout_ms: 30000,
            wait_for_all: false,
        }
    }
}

/// Voting strategies for ensemble
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Simple majority voting
    SimpleMajority,
    
    /// Weighted voting based on model confidence
    WeightedConsensus,
    
    /// Quality-weighted voting
    QualityWeighted,
    
    /// Ranked choice voting
    RankedChoice,
    
    /// Statistical aggregation
    Statistical,
}

/// Ensemble coordinator for managing multi-model responses
pub struct EnsembleCoordinator {
    config: EnsembleConfig,
    model_weights: RwLock<HashMap<String, f32>>,
}

impl EnsembleCoordinator {
    /// Create a new ensemble coordinator
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            model_weights: RwLock::new(HashMap::new()),
        }
    }
    
    /// Set weight for a specific model
    pub async fn set_model_weight(&self, model: String, weight: f32) {
        let mut weights = self.model_weights.write().await;
        weights.insert(model, weight);
    }
    
    /// Aggregate responses from multiple models
    pub async fn aggregate_responses(
        &self,
        responses: Vec<ModelResponse>,
    ) -> Result<EnsembleResult> {
        if responses.len() < self.config.min_models {
            return Err(anyhow!(
                "Insufficient models for ensemble: {} < {}",
                responses.len(),
                self.config.min_models
            ));
        }
        
        match self.config.voting_strategy {
            VotingStrategy::SimpleMajority => self.simple_majority(responses).await,
            VotingStrategy::WeightedConsensus => self.weighted_consensus(responses).await,
            VotingStrategy::QualityWeighted => self.quality_weighted(responses).await,
            VotingStrategy::RankedChoice => self.ranked_choice(responses).await,
            VotingStrategy::Statistical => self.statistical_aggregation(responses).await,
        }
    }
    
    /// Simple majority voting
    async fn simple_majority(&self, responses: Vec<ModelResponse>) -> Result<EnsembleResult> {
        // Group similar responses
        let mut response_groups: HashMap<String, Vec<ModelResponse>> = HashMap::new();
        
        for response in &responses {
            let key = self.normalize_response(&response.content);
            response_groups.entry(key).or_insert_with(Vec::new).push(response.clone());
        }
        
        // Find the most common response
        let (content, group) = response_groups
            .into_iter()
            .max_by_key(|(_, v)| v.len())
            .ok_or_else(|| anyhow!("No responses to aggregate"))?;
        
        let votes = group.len();
        let total = responses.len();
        let consensus_score = votes as f32 / total as f32;
        
        Ok(EnsembleResult {
            content,
            consensus_score,
            participating_models: group.into_iter().map(|r| r.model).collect(),
            aggregation_method: "simple_majority".to_string(),
            confidence: consensus_score,
            metadata: HashMap::new(),
        })
    }
    
    /// Weighted consensus based on model confidence
    async fn weighted_consensus(&self, responses: Vec<ModelResponse>) -> Result<EnsembleResult> {
        let weights = self.model_weights.read().await;
        
        // Calculate weighted scores for each unique response
        let mut weighted_responses: HashMap<String, f32> = HashMap::new();
        let mut response_models: HashMap<String, Vec<String>> = HashMap::new();
        
        for response in &responses {
            let key = self.normalize_response(&response.content);
            let weight = weights.get(&response.model).unwrap_or(&1.0);
            let score = response.confidence * weight;
            
            *weighted_responses.entry(key.clone()).or_insert(0.0) += score;
            response_models.entry(key).or_insert_with(Vec::new).push(response.model.clone());
        }
        
        // Find response with highest weighted score
        let (content, score) = weighted_responses
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .ok_or_else(|| anyhow!("No responses to aggregate"))?;
        
        let models = response_models.remove(&content).unwrap_or_default();
        let total_weight: f32 = responses.iter()
            .map(|r| weights.get(&r.model).unwrap_or(&1.0) * r.confidence)
            .sum();
        
        let consensus_score = score / total_weight;
        
        Ok(EnsembleResult {
            content,
            consensus_score,
            participating_models: models,
            aggregation_method: "weighted_consensus".to_string(),
            confidence: consensus_score.min(1.0),
            metadata: HashMap::new(),
        })
    }
    
    /// Quality-weighted voting
    async fn quality_weighted(&self, responses: Vec<ModelResponse>) -> Result<EnsembleResult> {
        // Group responses and weight by quality metrics
        let mut quality_scores: HashMap<String, (f32, Vec<String>)> = HashMap::new();
        
        for response in responses {
            let key = self.normalize_response(&response.content);
            let quality = self.calculate_response_quality(&response);
            
            let entry = quality_scores.entry(key).or_insert((0.0, Vec::new()));
            entry.0 += quality;
            entry.1.push(response.model);
        }
        
        // Select response with highest quality score
        let (content, (score, models)) = quality_scores
            .into_iter()
            .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap())
            .ok_or_else(|| anyhow!("No responses to aggregate"))?;
        
        Ok(EnsembleResult {
            content,
            consensus_score: score / models.len() as f32,
            participating_models: models.clone(),
            aggregation_method: "quality_weighted".to_string(),
            confidence: (score / models.len() as f32).min(1.0),
            metadata: HashMap::new(),
        })
    }
    
    /// Ranked choice voting
    async fn ranked_choice(&self, responses: Vec<ModelResponse>) -> Result<EnsembleResult> {
        // Create preference rankings for each model
        let mut rankings: Vec<(String, Vec<String>)> = Vec::new();
        
        for response in &responses {
            let mut ranked_contents: Vec<String> = responses.iter()
                .map(|r| self.normalize_response(&r.content))
                .collect();
            
            // Sort by similarity to this response
            ranked_contents.sort_by_key(|content| {
                self.calculate_similarity(&response.content, content)
            });
            
            rankings.push((response.model.clone(), ranked_contents));
        }
        
        // Perform instant runoff voting
        let winner = self.instant_runoff(rankings)?;
        
        let participating_models: Vec<String> = responses.iter()
            .filter(|r| self.normalize_response(&r.content) == winner)
            .map(|r| r.model.clone())
            .collect();
        
        Ok(EnsembleResult {
            content: winner,
            consensus_score: participating_models.len() as f32 / responses.len() as f32,
            participating_models,
            aggregation_method: "ranked_choice".to_string(),
            confidence: 0.8, // Fixed confidence for ranked choice
            metadata: HashMap::new(),
        })
    }
    
    /// Statistical aggregation
    async fn statistical_aggregation(&self, responses: Vec<ModelResponse>) -> Result<EnsembleResult> {
        // For text responses, use semantic clustering
        // For numeric responses, use statistical measures
        
        // Group similar responses using semantic similarity
        let clusters = self.cluster_responses(&responses)?;
        
        // Find the largest cluster (mode)
        let (representative, cluster) = clusters
            .into_iter()
            .max_by_key(|(_, c)| c.len())
            .ok_or_else(|| anyhow!("No responses to aggregate"))?;
        
        // Calculate statistics
        let cluster_size = cluster.len();
        let total_size = responses.len();
        let consensus_score = cluster_size as f32 / total_size as f32;
        
        // Calculate confidence based on cluster cohesion
        let confidence = self.calculate_cluster_confidence(&cluster);
        
        Ok(EnsembleResult {
            content: representative,
            consensus_score,
            participating_models: cluster.into_iter().map(|r| r.model).collect(),
            aggregation_method: "statistical".to_string(),
            confidence,
            metadata: HashMap::from([
                ("cluster_size".to_string(), cluster_size.to_string()),
                ("total_responses".to_string(), total_size.to_string()),
            ]),
        })
    }
    
    /// Normalize response for comparison
    fn normalize_response(&self, content: &str) -> String {
        // Simple normalization - in practice would use more sophisticated methods
        content.trim().to_lowercase()
    }
    
    /// Calculate response quality
    fn calculate_response_quality(&self, response: &ModelResponse) -> f32 {
        // Factors: confidence, length, coherence, etc.
        let length_factor = (response.content.len() as f32 / 1000.0).min(1.0);
        let confidence_factor = response.confidence;
        
        // In practice, would include more sophisticated quality metrics
        (length_factor + confidence_factor) / 2.0
    }
    
    /// Calculate similarity between two responses
    fn calculate_similarity(&self, a: &str, b: &str) -> i32 {
        // Simple Levenshtein distance - in practice would use semantic similarity
        strsim::levenshtein(a, b) as i32
    }
    
    /// Perform instant runoff voting
    fn instant_runoff(&self, mut rankings: Vec<(String, Vec<String>)>) -> Result<String> {
        let total_voters = rankings.len();
        let majority = total_voters / 2 + 1;
        
        loop {
            // Count first-place votes
            let mut vote_counts: HashMap<String, usize> = HashMap::new();
            for (_, ranking) in &rankings {
                if let Some(first_choice) = ranking.first() {
                    *vote_counts.entry(first_choice.clone()).or_insert(0) += 1;
                }
            }
            
            // Check for majority winner
            if let Some((winner, count)) = vote_counts.iter().max_by_key(|&(_, c)| c) {
                if *count >= majority {
                    return Ok(winner.clone());
                }
            }
            
            // Eliminate candidate with fewest votes
            if let Some((loser, _)) = vote_counts.iter().min_by_key(|&(_, c)| c) {
                let loser = loser.clone();
                // Remove loser from all rankings
                for (_, ranking) in &mut rankings {
                    ranking.retain(|c| c != &loser);
                }
            } else {
                return Err(anyhow!("No votes to count"));
            }
            
            // Check if we've eliminated all candidates
            if rankings.iter().all(|(_, r)| r.is_empty()) {
                return Err(anyhow!("All candidates eliminated"));
            }
        }
    }
    
    /// Cluster similar responses
    fn cluster_responses(&self, responses: &[ModelResponse]) -> Result<Vec<(String, Vec<ModelResponse>)>> {
        // Simple clustering based on exact matches
        // In practice would use semantic embeddings and proper clustering
        let mut clusters: HashMap<String, Vec<ModelResponse>> = HashMap::new();
        
        for response in responses {
            let key = self.normalize_response(&response.content);
            clusters.entry(key).or_insert_with(Vec::new).push(response.clone());
        }
        
        Ok(clusters.into_iter().collect())
    }
    
    /// Calculate confidence for a cluster of responses
    fn calculate_cluster_confidence(&self, cluster: &[ModelResponse]) -> f32 {
        if cluster.is_empty() {
            return 0.0;
        }
        
        // Average confidence of cluster members
        let avg_confidence: f32 = cluster.iter().map(|r| r.confidence).sum::<f32>() / cluster.len() as f32;
        
        // Bonus for cluster size
        let size_bonus = (cluster.len() as f32 / 10.0).min(0.2);
        
        (avg_confidence + size_bonus).min(1.0)
    }
}

/// Response from a single model
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// Model that generated the response
    pub model: String,
    
    /// Response content
    pub content: String,
    
    /// Model's confidence in the response
    pub confidence: f32,
    
    /// Response metadata
    pub metadata: HashMap<String, String>,
}

/// Result of ensemble aggregation
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Aggregated content
    pub content: String,
    
    /// Consensus score (0.0 - 1.0)
    pub consensus_score: f32,
    
    /// Models that contributed to this result
    pub participating_models: Vec<String>,
    
    /// Method used for aggregation
    pub aggregation_method: String,
    
    /// Overall confidence in the result
    pub confidence: f32,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simple_majority() {
        let coordinator = EnsembleCoordinator::new(EnsembleConfig {
            voting_strategy: VotingStrategy::SimpleMajority,
            min_models: 2,
            ..Default::default()
        });
        
        let responses = vec![
            ModelResponse {
                model: "gpt-4".to_string(),
                content: "The answer is 42".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            },
            ModelResponse {
                model: "claude-3".to_string(),
                content: "The answer is 42".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            },
            ModelResponse {
                model: "llama-3".to_string(),
                content: "The answer is 43".to_string(),
                confidence: 0.8,
                metadata: HashMap::new(),
            },
        ];
        
        let result = coordinator.aggregate_responses(responses).await.unwrap();
        assert_eq!(result.content, "the answer is 42");
        assert_eq!(result.consensus_score, 2.0 / 3.0);
    }
    
    #[tokio::test]
    async fn test_weighted_consensus() {
        let coordinator = EnsembleCoordinator::new(EnsembleConfig {
            voting_strategy: VotingStrategy::WeightedConsensus,
            min_models: 2,
            ..Default::default()
        });
        
        // Set model weights
        coordinator.set_model_weight("gpt-4".to_string(), 2.0).await;
        coordinator.set_model_weight("claude-3".to_string(), 1.5).await;
        
        let responses = vec![
            ModelResponse {
                model: "gpt-4".to_string(),
                content: "High quality response".to_string(),
                confidence: 0.95,
                metadata: HashMap::new(),
            },
            ModelResponse {
                model: "claude-3".to_string(),
                content: "Different response".to_string(),
                confidence: 0.9,
                metadata: HashMap::new(),
            },
            ModelResponse {
                model: "llama-3".to_string(),
                content: "Different response".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            },
        ];
        
        let result = coordinator.aggregate_responses(responses).await.unwrap();
        // GPT-4's response should win due to higher weight and confidence
        assert!(result.participating_models.contains(&"gpt-4".to_string()));
    }
}