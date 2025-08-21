//! Routing strategies for model orchestration
//! 
//! This module implements various routing strategies for directing
//! requests to appropriate models based on different criteria.

use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

use super::manager::{RoutingStrategy, ModelCapability};
use super::models::ModelMetadata;

// Alias for compatibility
type LocalCapability = ModelCapability;

/// Configuration for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfiguration {
    /// Strategy to use for routing
    pub strategy: RoutingStrategy,
    
    /// Model capabilities mapping
    pub model_capabilities: HashMap<String, Vec<ModelCapability>>,
    
    /// Cost thresholds for routing decisions
    pub cost_threshold_cents: f32,
    
    /// Quality threshold (0.0 - 1.0)
    pub quality_threshold: f32,
    
    /// Preference for local models (0.0 = API only, 1.0 = local only)
    pub local_preference: f32,
    
    /// Maximum parallel models for ensemble
    pub max_parallel_models: usize,
    
    /// Timeout for routing decisions
    pub routing_timeout_ms: u64,
}

impl Default for RoutingConfiguration {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::Capability,
            model_capabilities: HashMap::new(),
            cost_threshold_cents: 10.0,
            quality_threshold: 0.7,
            local_preference: 0.5,
            max_parallel_models: 3,
            routing_timeout_ms: 1000,
        }
    }
}

/// Router that implements different routing strategies
pub struct ModelRouter {
    config: RoutingConfiguration,
    model_registry: HashMap<String, ModelMetadata>,
}

impl ModelRouter {
    /// Create a new model router
    pub fn new(config: RoutingConfiguration) -> Self {
        Self {
            config,
            model_registry: HashMap::new(),
        }
    }
    
    /// Register a model with its metadata
    pub fn register_model(&mut self, name: String, metadata: ModelMetadata) {
        self.model_registry.insert(name, metadata);
    }
    
    /// Route a request to appropriate model(s)
    pub async fn route_request(
        &self,
        request: &RoutingRequest,
    ) -> Result<RoutingDecision> {
        match self.config.strategy {
            RoutingStrategy::Capability | RoutingStrategy::CapabilityBased => self.route_by_capability(request).await,
            RoutingStrategy::Cost | RoutingStrategy::CostOptimized => self.route_by_cost(request).await,
            RoutingStrategy::Speed => self.route_by_speed(request).await,
            RoutingStrategy::Quality => self.route_by_quality(request).await,
            RoutingStrategy::QualityFirst => self.route_by_quality(request).await,
            RoutingStrategy::Availability => self.route_by_availability(request).await,
            RoutingStrategy::Hybrid => self.route_hybrid(request).await,
            RoutingStrategy::RoundRobin => self.route_round_robin(request).await,
            RoutingStrategy::LeastLatency => self.route_by_speed(request).await,
            RoutingStrategy::ContextAware => self.route_by_capability(request).await,
            RoutingStrategy::Adaptive => self.route_hybrid(request).await,
            RoutingStrategy::Custom(_) => self.route_hybrid(request).await,
        }
    }
    
    /// Route based on model capabilities
    async fn route_by_capability(&self, request: &RoutingRequest) -> Result<RoutingDecision> {
        let mut candidates = Vec::new();
        
        // Find models that match required capabilities
        for (model_name, capabilities) in &self.config.model_capabilities {
            if request.required_capabilities.iter().all(|req| {
                capabilities.iter().any(|cap| self.capability_matches(req, cap))
            }) {
                candidates.push(model_name.clone());
            }
        }
        
        if candidates.is_empty() {
            return Err(anyhow!("No models found with required capabilities"));
        }
        
        // Select best candidate based on additional criteria
        let selected = self.select_best_candidate(&candidates, request)?;
        
        Ok(RoutingDecision {
            primary_model: selected.clone(),
            fallback_models: candidates.into_iter()
                .filter(|m| m != &selected)
                .take(2)
                .collect(),
            reasoning: "Selected based on capability matching".to_string(),
            confidence: 0.85,
        })
    }
    
    /// Route based on cost optimization
    async fn route_by_cost(&self, request: &RoutingRequest) -> Result<RoutingDecision> {
        let mut cost_sorted: Vec<(String, f64)> = self.model_registry
            .iter()
            .map(|(name, metadata)| {
                (name.clone(), metadata.cost_per_1k_tokens)
            })
            .collect();
        
        cost_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find cheapest model under threshold
        let selected = cost_sorted.iter()
            .find(|(_, cost)| *cost <= self.config.cost_threshold_cents as f64)
            .map(|(name, _)| name.clone())
            .ok_or_else(|| anyhow!("No models found within cost threshold"))?;
        
        Ok(RoutingDecision {
            primary_model: selected.clone(),
            fallback_models: cost_sorted.into_iter()
                .map(|(name, _)| name)
                .filter(|n| n != &selected)
                .take(2)
                .collect(),
            reasoning: "Selected based on cost optimization".to_string(),
            confidence: 0.9,
        })
    }
    
    /// Route based on speed/latency
    async fn route_by_speed(&self, request: &RoutingRequest) -> Result<RoutingDecision> {
        let mut speed_sorted: Vec<(String, u64)> = self.model_registry
            .iter()
            .map(|(name, metadata)| {
                (name.clone(), metadata.avg_latency_ms)
            })
            .collect();
        
        speed_sorted.sort_by_key(|k| k.1);
        
        let selected = speed_sorted.first()
            .map(|(name, _)| name.clone())
            .ok_or_else(|| anyhow!("No models found with latency data"))?;
        
        Ok(RoutingDecision {
            primary_model: selected,
            fallback_models: speed_sorted.into_iter()
                .map(|(name, _)| name)
                .skip(1)
                .take(2)
                .collect(),
            reasoning: "Selected based on lowest latency".to_string(),
            confidence: 0.8,
        })
    }
    
    /// Route based on quality metrics
    async fn route_by_quality(&self, request: &RoutingRequest) -> Result<RoutingDecision> {
        let mut quality_sorted: Vec<(String, f64)> = self.model_registry
            .iter()
            .map(|(name, metadata)| {
                (name.clone(), metadata.quality_score)
            })
            .filter(|(_, quality)| *quality >= self.config.quality_threshold.into())
            .collect();
        
        quality_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected = quality_sorted.first()
            .map(|(name, _)| name.clone())
            .ok_or_else(|| anyhow!("No models found meeting quality threshold"))?;
        
        Ok(RoutingDecision {
            primary_model: selected,
            fallback_models: quality_sorted.into_iter()
                .map(|(name, _)| name)
                .skip(1)
                .take(2)
                .collect(),
            reasoning: "Selected based on highest quality score".to_string(),
            confidence: 0.95,
        })
    }
    
    /// Route based on availability
    async fn route_by_availability(&self, request: &RoutingRequest) -> Result<RoutingDecision> {
        // In real implementation, would check actual availability
        // For now, prefer local models based on preference
        let mut available: Vec<String> = self.model_registry
            .iter()
            .filter_map(|(name, metadata)| {
                let is_local = metadata.provider == "ollama";
                let availability_score = if is_local {
                    1.0 * self.config.local_preference
                } else {
                    1.0 * (1.0 - self.config.local_preference)
                };
                
                if availability_score > 0.5 {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        
        if available.is_empty() {
            return Err(anyhow!("No models currently available"));
        }
        
        let selected = available.remove(0);
        
        Ok(RoutingDecision {
            primary_model: selected,
            fallback_models: available.into_iter().take(2).collect(),
            reasoning: "Selected based on availability and local preference".to_string(),
            confidence: 0.7,
        })
    }
    
    /// Hybrid routing combining multiple strategies
    async fn route_hybrid(&self, request: &RoutingRequest) -> Result<RoutingDecision> {
        // Score each model across multiple dimensions
        let mut scores: HashMap<String, f32> = HashMap::new();
        
        for (model_name, metadata) in &self.model_registry {
            let mut score: f32 = 0.0;
            let mut factors = 0;
            
            // Capability score
            if let Some(caps) = self.config.model_capabilities.get(model_name) {
                let cap_score = request.required_capabilities.iter()
                    .filter(|req| caps.iter().any(|cap| self.capability_matches(req, cap)))
                    .count() as f32 / request.required_capabilities.len().max(1) as f32;
                score += cap_score * 0.3;
                factors += 1;
            }
            
            // Cost score
            let cost = metadata.cost_per_1k_tokens;
            if cost > 0.0 {
                let cost_score = ((self.config.cost_threshold_cents as f64 - cost).max(0.0) 
                    / self.config.cost_threshold_cents as f64) as f32;
                score += cost_score * 0.2;
                factors += 1;
            }
            
            // Quality score
            let quality = metadata.quality_score;
            if quality > 0.0 {
                score += (quality * 0.3) as f32;
                factors += 1;
            }
            
            // Speed score
            let latency = metadata.avg_latency_ms;
            if latency > 0 {
                let speed_score = ((1000.0 - latency as f64).max(0.0) / 1000.0) as f32;
                score += speed_score * 0.2;
                factors += 1;
            }
            
            if factors > 0 {
                scores.insert(model_name.clone(), score / factors as f32);
            }
        }
        
        if scores.is_empty() {
            return Err(anyhow!("No models could be scored"));
        }
        
        // Sort by score
        let mut sorted: Vec<_> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected = sorted[0].0.clone();
        
        Ok(RoutingDecision {
            primary_model: selected.clone(),
            fallback_models: sorted.into_iter()
                .map(|(name, _)| name)
                .filter(|n| n != &selected)
                .take(2)
                .collect(),
            reasoning: "Selected using hybrid scoring across capability, cost, quality, and speed".to_string(),
            confidence: 0.85,
        })
    }
    
    /// Route using round-robin strategy
    async fn route_round_robin(&self, _request: &RoutingRequest) -> Result<RoutingDecision> {
        // Get available models from registry
        let models: Vec<String> = self.model_registry.keys().cloned().collect();
        
        if models.is_empty() {
            return Err(anyhow::anyhow!("No models available"));
        }
        
        // Simple round-robin: pick the next model in the list
        // In a real implementation, you'd track the last used index
        let selected = models[0].clone();
        
        Ok(RoutingDecision {
            primary_model: selected,
            fallback_models: models.into_iter().skip(1).take(2).collect(),
            reasoning: "Selected using round-robin strategy".to_string(),
            confidence: 0.75,
        })
    }
    
    /// Check if a capability matches
    fn capability_matches(&self, required: &LocalCapability, available: &ModelCapability) -> bool {
        // Convert between local and model capability types
        match (required, available) {
            (LocalCapability::CodeGeneration, ModelCapability::CodeGeneration) => true,
            (LocalCapability::Analysis, ModelCapability::Analysis) => true,
            (LocalCapability::Creative, ModelCapability::Creative) => true,
            (LocalCapability::Conversation, ModelCapability::Conversation) => true,
            (LocalCapability::Embedding, ModelCapability::Embedding) => true,
            _ => false,
        }
    }
    
    /// Select best candidate from list
    fn select_best_candidate(&self, candidates: &[String], request: &RoutingRequest) -> Result<String> {
        // For now, prefer local models based on preference
        let local_candidates: Vec<_> = candidates.iter()
            .filter(|name| {
                self.model_registry.get(*name)
                    .map(|m| m.provider == "ollama")
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        
        let api_candidates: Vec<_> = candidates.iter()
            .filter(|name| {
                self.model_registry.get(*name)
                    .map(|m| m.provider != "ollama")
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        
        if self.config.local_preference > 0.5 && !local_candidates.is_empty() {
            Ok(local_candidates[0].clone())
        } else if !api_candidates.is_empty() {
            Ok(api_candidates[0].clone())
        } else if !candidates.is_empty() {
            Ok(candidates[0].clone())
        } else {
            Err(anyhow!("No candidates available"))
        }
    }
}

/// Request for routing decision
#[derive(Debug, Clone)]
pub struct RoutingRequest {
    /// Required capabilities for the task
    pub required_capabilities: Vec<LocalCapability>,
    
    /// Estimated tokens for the request
    pub estimated_tokens: usize,
    
    /// Priority of the request
    pub priority: f32,
    
    /// Maximum acceptable latency
    pub max_latency_ms: Option<u64>,
    
    /// Maximum acceptable cost
    pub max_cost_cents: Option<f32>,
    
    /// Context about the request
    pub context: String,
}

/// Decision made by the router
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Primary model to use
    pub primary_model: String,
    
    /// Fallback models in order of preference
    pub fallback_models: Vec<String>,
    
    /// Reasoning for the decision
    pub reasoning: String,
    
    /// Confidence in the decision (0.0 - 1.0)
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_routing_configuration_default() {
        let config = RoutingConfiguration::default();
        assert_eq!(config.strategy, RoutingStrategy::Capability);
        assert_eq!(config.cost_threshold_cents, 10.0);
        assert_eq!(config.quality_threshold, 0.7);
    }
    
    #[tokio::test]
    async fn test_capability_routing() {
        let mut config = RoutingConfiguration::default();
        config.model_capabilities.insert("gpt-4".to_string(), vec![
            ModelCapability::CodeGeneration,
            ModelCapability::Analysis,
        ]);
        config.model_capabilities.insert("claude-3".to_string(), vec![
            ModelCapability::Analysis,
            ModelCapability::Creative,
        ]);
        
        let router = ModelRouter::new(config);
        
        let request = RoutingRequest {
            required_capabilities: vec![LocalCapability::Analysis],
            estimated_tokens: 1000,
            priority: 0.5,
            max_latency_ms: None,
            max_cost_cents: None,
            context: "Test request".to_string(),
        };
        
        let decision = router.route_request(&request).await.unwrap();
        assert!(!decision.primary_model.is_empty());
        assert!(decision.confidence > 0.0);
    }
}