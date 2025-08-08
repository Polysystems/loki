//! Orchestration-specific command handling

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result};

use super::super::orchestration::{RoutingStrategy, OrchestrationSetup, ModelCapability};
use super::super::orchestration::OrchestrationManager;

/// Handles orchestration-specific commands
pub struct OrchestrationCommandHandler {
    /// Orchestration manager
    manager: Arc<RwLock<OrchestrationManager>>,
}

impl OrchestrationCommandHandler {
    /// Create a new orchestration command handler
    pub fn new(manager: Arc<RwLock<OrchestrationManager>>) -> Self {
        Self { manager }
    }
    
    /// Configure model routing
    pub async fn configure_routing(&self, models: Vec<String>, strategy: RoutingStrategy) -> Result<String> {
        let mut manager = self.manager.write().await;
        
        // Update model configuration
        manager.enabled_models = models.clone();
        manager.preferred_strategy = strategy.clone();
        
        // Update orchestration setup based on configuration
        if models.len() == 1 {
            manager.current_setup = OrchestrationSetup::SingleModel;
        } else if strategy == RoutingStrategy::RoundRobin || strategy == RoutingStrategy::LeastLatency {
            manager.current_setup = OrchestrationSetup::MultiModelRouting;
        } else {
            manager.current_setup = OrchestrationSetup::EnsembleVoting;
        }
        
        Ok(format!(
            "Configured routing with {} models using {:?} strategy",
            models.len(),
            strategy
        ))
    }
    
    /// Set parallel agent count
    pub async fn set_parallel_agents(&self, count: usize) -> Result<String> {
        let mut manager = self.manager.write().await;
        
        if count == 0 || count > 10 {
            return Ok("Parallel agent count must be between 1 and 10".to_string());
        }
        
        manager.parallel_models = count;
        
        Ok(format!("Set parallel agents to {}", count))
    }
    
    /// Configure model capabilities
    pub async fn configure_capabilities(&self, model: String, capabilities: Vec<ModelCapability>) -> Result<String> {
        let mut manager = self.manager.write().await;
        
        // Update model capabilities in the registry
        manager.model_capabilities.insert(model.clone(), capabilities.clone());
        
        Ok(format!(
            "Configured {} capabilities for model {}",
            capabilities.len(),
            model
        ))
    }
    
    /// Enable/disable fallback
    pub async fn set_fallback(&self, enabled: bool) -> Result<String> {
        let mut manager = self.manager.write().await;
        manager.allow_fallback = enabled;
        
        Ok(format!(
            "Fallback {}",
            if enabled { "enabled" } else { "disabled" }
        ))
    }
    
    /// Set context window size
    pub async fn set_context_window(&self, size: usize) -> Result<String> {
        let mut manager = self.manager.write().await;
        manager.context_window = size;
        
        Ok(format!("Context window set to {} tokens", size))
    }
    
    /// Configure temperature
    pub async fn set_temperature(&self, temp: f32) -> Result<String> {
        let mut manager = self.manager.write().await;
        
        if temp < 0.0 || temp > 2.0 {
            return Ok("Temperature must be between 0.0 and 2.0".to_string());
        }
        
        manager.temperature = temp;
        
        Ok(format!("Temperature set to {}", temp))
    }
    
    /// Enable/disable streaming
    pub async fn set_streaming(&self, enabled: bool) -> Result<String> {
        let mut manager = self.manager.write().await;
        manager.stream_responses = enabled;
        
        Ok(format!(
            "Response streaming {}",
            if enabled { "enabled" } else { "disabled" }
        ))
    }
    
    /// Configure retry settings
    pub async fn configure_retries(&self, max_retries: u32, timeout: u64) -> Result<String> {
        let mut manager = self.manager.write().await;
        manager.max_retries = max_retries;
        manager.timeout_seconds = timeout;
        
        Ok(format!(
            "Configured {} max retries with {}s timeout",
            max_retries, timeout
        ))
    }
    
    /// Get orchestration status
    pub async fn get_status(&self) -> Result<String> {
        let manager = self.manager.read().await;
        
        let status = format!(
            r#"Orchestration Status:
Enabled: {}
Strategy: {:?}
Active Models: {}
Parallel Agents: {}
Fallback: {}
Context Window: {} tokens
Temperature: {}
Streaming: {}
Max Retries: {}
Timeout: {}s"#,
            manager.orchestration_enabled,
            manager.preferred_strategy,
            manager.enabled_models.join(", "),
            manager.parallel_models,
            if manager.allow_fallback { "enabled" } else { "disabled" },
            manager.context_window,
            manager.temperature,
            if manager.stream_responses { "enabled" } else { "disabled" },
            manager.max_retries,
            manager.timeout_seconds
        );
        
        Ok(status)
    }
    
    /// Validate configuration
    pub async fn validate_config(&self) -> Result<Vec<String>> {
        let manager = self.manager.read().await;
        let mut warnings = Vec::new();
        
        // Check model availability
        if manager.enabled_models.is_empty() {
            warnings.push("No models configured".to_string());
        }
        
        // Check parallel agent configuration
        if manager.parallel_models > manager.enabled_models.len() {
            warnings.push(format!(
                "Parallel agents ({}) exceeds available models ({})",
                manager.parallel_models,
                manager.enabled_models.len()
            ));
        }
        
        // Check context window
        if manager.context_window < 1000 {
            warnings.push("Context window is very small (< 1000 tokens)".to_string());
        }
        
        // Check temperature
        if manager.temperature > 1.5 {
            warnings.push("Temperature is very high (> 1.5)".to_string());
        }
        
        Ok(warnings)
    }
}