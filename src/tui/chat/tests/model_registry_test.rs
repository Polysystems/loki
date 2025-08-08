//! Tests for model registry functionality

#[cfg(test)]
mod tests {
    use super::super::initialization::{ModelRegistry, RegisteredModel};
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_model_registry_initialization() {
        // Create a new registry
        let registry = ModelRegistry::new();
        
        // Register a test model
        let test_model = RegisteredModel {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            provider: "test".to_string(),
            capabilities: vec!["chat".to_string(), "test".to_string()],
            available: true,
            last_verified: Some(chrono::Utc::now()),
            error: None,
        };
        
        registry.register_model(test_model.clone()).await.unwrap();
        
        // Check if model is available
        assert!(registry.is_model_available("test-model").await);
        
        // Get model metadata
        let retrieved = registry.get_model("test-model").await.unwrap();
        assert_eq!(retrieved.id, "test-model");
        assert_eq!(retrieved.provider, "test");
        
        // List available models
        let available = registry.list_available_models().await;
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].id, "test-model");
        
        // Find model by capability
        let chat_model = registry.find_model_for_capability("chat").await;
        assert_eq!(chat_model, Some("test-model".to_string()));
        
        // Test default model selection
        let default = registry.get_default_model().await;
        assert_eq!(default, Some("test-model".to_string()));
    }
    
    #[tokio::test]
    async fn test_model_unavailable_handling() {
        let registry = ModelRegistry::new();
        
        // Register an unavailable model
        let unavailable_model = RegisteredModel {
            id: "unavailable".to_string(),
            name: "Unavailable Model".to_string(),
            provider: "test".to_string(),
            capabilities: vec!["chat".to_string()],
            available: false,
            last_verified: Some(chrono::Utc::now()),
            error: Some("API key missing".to_string()),
        };
        
        registry.register_model(unavailable_model).await.unwrap();
        
        // Check availability
        assert!(!registry.is_model_available("unavailable").await);
        
        // Should not appear in available models
        let available = registry.list_available_models().await;
        assert_eq!(available.len(), 0);
        
        // But should appear in all models
        let all_models = registry.list_all_models().await;
        assert_eq!(all_models.len(), 1);
        assert!(!all_models[0].available);
    }
    
    #[tokio::test]
    async fn test_model_capability_matching() {
        let registry = ModelRegistry::new();
        
        // Register models with different capabilities
        let code_model = RegisteredModel {
            id: "code-model".to_string(),
            name: "Code Model".to_string(),
            provider: "openai".to_string(),
            capabilities: vec!["code".to_string(), "chat".to_string()],
            available: true,
            last_verified: Some(chrono::Utc::now()),
            error: None,
        };
        
        let analysis_model = RegisteredModel {
            id: "analysis-model".to_string(),
            name: "Analysis Model".to_string(),
            provider: "anthropic".to_string(),
            capabilities: vec!["analysis".to_string(), "chat".to_string()],
            available: true,
            last_verified: Some(chrono::Utc::now()),
            error: None,
        };
        
        registry.register_model(code_model).await.unwrap();
        registry.register_model(analysis_model).await.unwrap();
        
        // Find models by capability
        let code_capable = registry.find_model_for_capability("code").await;
        assert_eq!(code_capable, Some("code-model".to_string()));
        
        let analysis_capable = registry.find_model_for_capability("analysis").await;
        assert_eq!(analysis_capable, Some("analysis-model".to_string()));
        
        // Both should be capable of chat
        let chat_capable = registry.find_model_for_capability("chat").await;
        assert!(chat_capable.is_some());
    }
    
    #[tokio::test]
    async fn test_model_validator() {
        use super::super::orchestration::ModelValidator;
        
        let registry = Arc::new(ModelRegistry::new());
        let validator = ModelValidator::new(registry.clone());
        
        // Register a valid model
        let valid_model = RegisteredModel {
            id: "valid-model".to_string(),
            name: "Valid Model".to_string(),
            provider: "test".to_string(),
            capabilities: vec!["chat".to_string()],
            available: true,
            last_verified: Some(chrono::Utc::now()),
            error: None,
        };
        
        registry.register_model(valid_model).await.unwrap();
        
        // Validation should succeed
        assert!(validator.validate_model("valid-model").await.is_ok());
        
        // Validation should fail for non-existent model
        assert!(validator.validate_model("non-existent").await.is_err());
        
        // Test capability-based selection
        let chat_model = validator.find_model_for_capability("chat").await;
        assert!(chat_model.is_ok());
        assert_eq!(chat_model.unwrap(), "valid-model");
    }
}