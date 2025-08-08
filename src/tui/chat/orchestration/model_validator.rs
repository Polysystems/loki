//! Model validation for orchestration
//! 
//! Ensures models are available before routing requests

use std::sync::Arc;
use anyhow::{Result, bail};
use tokio::sync::RwLock;

use crate::tui::chat::initialization::ModelRegistry;

/// Model validator for orchestration
pub struct ModelValidator {
    /// Model registry reference
    registry: Arc<ModelRegistry>,
}

impl ModelValidator {
    /// Create a new model validator
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self { registry }
    }
    
    /// Validate that a model is available for use
    pub async fn validate_model(&self, model_id: &str) -> Result<()> {
        if !self.registry.is_model_available(model_id).await {
            // Try to re-verify in case status changed
            self.registry.verify_model(model_id).await?;
            
            // Check again
            if !self.registry.is_model_available(model_id).await {
                if let Some(model) = self.registry.get_model(model_id).await {
                    let error_msg = model.error.unwrap_or_else(|| "Unknown error".to_string());
                    bail!("Model '{}' is not available: {}", model_id, error_msg);
                } else {
                    bail!("Model '{}' is not registered", model_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate multiple models
    pub async fn validate_models(&self, model_ids: &[String]) -> Result<Vec<String>> {
        let mut available_models = Vec::new();
        let mut errors = Vec::new();
        
        for model_id in model_ids {
            match self.validate_model(model_id).await {
                Ok(_) => available_models.push(model_id.clone()),
                Err(e) => errors.push(format!("{}: {}", model_id, e)),
            }
        }
        
        if available_models.is_empty() && !errors.is_empty() {
            bail!("No models available:\n{}", errors.join("\n"));
        }
        
        if !errors.is_empty() {
            tracing::warn!("Some models unavailable:\n{}", errors.join("\n"));
        }
        
        Ok(available_models)
    }
    
    /// Find best model for a capability
    pub async fn find_model_for_capability(&self, capability: &str) -> Result<String> {
        match self.registry.find_model_for_capability(capability).await {
            Some(model_id) => {
                // Validate before returning
                self.validate_model(&model_id).await?;
                Ok(model_id)
            }
            None => {
                // Fall back to default model
                match self.registry.get_default_model().await {
                    Some(model_id) => {
                        self.validate_model(&model_id).await?;
                        tracing::warn!(
                            "No model found for capability '{}', using default '{}'",
                            capability,
                            model_id
                        );
                        Ok(model_id)
                    }
                    None => bail!("No models available for capability '{}'", capability),
                }
            }
        }
    }
    
    /// Get all available models with their metadata
    pub async fn get_available_models(&self) -> Vec<(String, Vec<String>)> {
        self.registry.list_available_models().await
            .into_iter()
            .map(|model| (model.id, model.capabilities))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_validation() {
        let registry = Arc::new(ModelRegistry::new());
        let validator = ModelValidator::new(registry);
        
        // This should fail since no models are registered
        assert!(validator.validate_model("gpt-4").await.is_err());
    }
}