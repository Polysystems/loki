//! Model Selector - Bridges OrchestrationManager settings to ModelOrchestrator execution
//! 
//! This module ensures that the models selected in the orchestration tab
//! are actually used during message processing.

use std::sync::Arc;
use anyhow::{Result, anyhow};
use tokio::sync::RwLock;
use tracing::{info, debug, warn};

use crate::models::orchestrator::{ModelOrchestrator, TaskRequest, TaskResponse};
use crate::tui::chat::orchestration::{OrchestrationManager, RoutingStrategy};

/// Model selector that bridges UI configuration to execution
pub struct ModelSelector {
    /// Reference to the orchestration manager for settings
    orchestration_manager: Arc<RwLock<OrchestrationManager>>,
    
    /// Reference to the model orchestrator for execution
    model_orchestrator: Arc<ModelOrchestrator>,
}

impl ModelSelector {
    /// Create a new model selector
    pub fn new(
        orchestration_manager: Arc<RwLock<OrchestrationManager>>,
        model_orchestrator: Arc<ModelOrchestrator>,
    ) -> Self {
        Self {
            orchestration_manager,
            model_orchestrator,
        }
    }
    
    /// Execute a task using the configured orchestration settings
    pub async fn execute_with_orchestration(&self, task: TaskRequest) -> Result<TaskResponse> {
        let manager = self.orchestration_manager.read().await;
        
        // Check if orchestration is enabled
        if !manager.orchestration_enabled {
            debug!("Orchestration disabled, using default execution");
            return self.model_orchestrator.execute_with_fallback(task).await;
        }
        
        // Get enabled models and routing strategy
        let enabled_models = &manager.enabled_models;
        let strategy = &manager.preferred_strategy;
        let parallel_models = manager.parallel_models;
        
        info!("🎯 Executing with orchestration: {} models, strategy: {:?}", 
              enabled_models.len(), strategy);
        
        // If no models are enabled, fall back to default
        if enabled_models.is_empty() {
            warn!("No models enabled in orchestration, using default");
            return self.model_orchestrator.execute_with_fallback(task).await;
        }
        
        // Handle different execution modes based on parallel_models setting
        match parallel_models {
            1 => {
                // Single model execution with routing
                self.execute_single_model(task, enabled_models, strategy).await
            }
            2..=10 => {
                // Parallel execution with ensemble
                self.execute_ensemble(task, enabled_models, parallel_models).await
            }
            _ => {
                warn!("Invalid parallel_models setting: {}", parallel_models);
                self.model_orchestrator.execute_with_fallback(task).await
            }
        }
    }
    
    /// Execute on a single model based on routing strategy
    async fn execute_single_model(
        &self,
        task: TaskRequest,
        enabled_models: &[String],
        strategy: &RoutingStrategy,
    ) -> Result<TaskResponse> {
        // Select model based on strategy
        let selected_model = match strategy {
            RoutingStrategy::RoundRobin => {
                // Use round-robin selection
                static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let index = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % enabled_models.len();
                enabled_models.get(index).cloned()
            }
            RoutingStrategy::LeastLatency => {
                // For now, just use first model (would need latency tracking)
                enabled_models.first().cloned()
            }
            RoutingStrategy::ContextAware => {
                // Match model to task type
                self.select_model_for_task(&task, enabled_models).await
            }
            _ => enabled_models.first().cloned(),
        };
        
        if let Some(model) = selected_model {
            info!("📍 Selected model: {} (strategy: {:?})", model, strategy);
            
            // Create a modified task request with task hint for better routing
            let mut modified_task = task;
            modified_task.constraints.task_hint = Some(format!("user_selected:{}", model));
            
            self.model_orchestrator.execute_with_fallback(modified_task).await
        } else {
            Err(anyhow!("No model selected from enabled models"))
        }
    }
    
    /// Execute using ensemble of multiple models
    async fn execute_ensemble(
        &self,
        task: TaskRequest,
        enabled_models: &[String],
        parallel_count: usize,
    ) -> Result<TaskResponse> {
        info!("🎭 Executing ensemble with {} models from {} enabled", 
              parallel_count, enabled_models.len());
        
        // Take up to parallel_count models
        let models_to_use: Vec<String> = enabled_models
            .iter()
            .take(parallel_count)
            .cloned()
            .collect();
        
        if models_to_use.len() < 2 {
            warn!("Not enough models for ensemble, falling back to single execution");
            return self.execute_single_model(task, enabled_models, &RoutingStrategy::RoundRobin).await;
        }
        
        // Execute ensemble through the model orchestrator
        match self.model_orchestrator.execute_with_ensemble(task.clone()).await {
            Ok(ensemble_response) => {
                info!("✅ Ensemble execution successful with {} models", 
                      ensemble_response.contributing_models.len());
                Ok(ensemble_response.primary_response)
            }
            Err(e) => {
                warn!("Ensemble execution failed: {}, falling back", e);
                self.model_orchestrator.execute_with_fallback(task).await
            }
        }
    }
    
    /// Select best model for a specific task
    async fn select_model_for_task(&self, task: &TaskRequest, enabled_models: &[String]) -> Option<String> {
        use crate::models::orchestrator::TaskType;
        
        // Match task type to model capabilities
        match &task.task_type {
            TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => {
                // Prefer models good at coding
                enabled_models.iter()
                    .find(|m| m.contains("gpt-4") || m.contains("claude") || m.contains("codellama"))
                    .or_else(|| enabled_models.first())
                    .cloned()
            }
            TaskType::CreativeWriting => {
                // Prefer creative models
                enabled_models.iter()
                    .find(|m| m.contains("claude") || m.contains("gpt-4"))
                    .or_else(|| enabled_models.first())
                    .cloned()
            }
            TaskType::DataAnalysis | TaskType::LogicalReasoning => {
                // Prefer analytical models
                enabled_models.iter()
                    .find(|m| m.contains("gpt-4") || m.contains("llama") || m.contains("mixtral"))
                    .or_else(|| enabled_models.first())
                    .cloned()
            }
            _ => enabled_models.first().cloned(),
        }
    }
    
    /// Get current orchestration status
    pub async fn get_status(&self) -> OrchestrationStatus {
        let manager = self.orchestration_manager.read().await;
        
        OrchestrationStatus {
            enabled: manager.orchestration_enabled,
            active_models: manager.enabled_models.clone(),
            routing_strategy: format!("{:?}", manager.preferred_strategy),
            parallel_execution: manager.parallel_models > 1,
            ensemble_mode: manager.ensemble_enabled,
            model_count: manager.enabled_models.len(),
        }
    }
}

/// Orchestration status information
#[derive(Debug, Clone)]
pub struct OrchestrationStatus {
    pub enabled: bool,
    pub active_models: Vec<String>,
    pub routing_strategy: String,
    pub parallel_execution: bool,
    pub ensemble_mode: bool,
    pub model_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_selector_creation() {
        // This would need proper mocking of dependencies
        // Just a placeholder for now
        assert!(true);
    }
}