//! Model management and registry
//! 
//! This module handles model registration, capabilities tracking,
//! and model lifecycle management.

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

use super::manager::ModelCapability;
use crate::config::ApiKeysConfig;

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub capabilities: Vec<ModelCapability>,
    pub context_window: usize,
    pub cost_per_token: f32,
    pub average_latency_ms: u64,
    pub status: ModelStatus,
    pub cost_per_1k_tokens: f64,
    pub quality_score: f64,
    pub avg_latency_ms: u64,
}

/// Model status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Busy,
    Unavailable,
    RateLimited,
    Error,
    Unknown,
    NoCredentials,
}

/// Model registry for tracking available models
pub struct ModelRegistry {
    /// Registered models
    models: RwLock<HashMap<String, RegisteredModel>>,
    
    /// API configuration
    api_config: Option<ApiKeysConfig>,
    
    /// Model status cache
    status_cache: RwLock<HashMap<String, ModelStatusCache>>,
}

/// A registered model with metadata and status
#[derive(Debug, Clone)]
pub struct RegisteredModel {
    /// Model identifier
    pub id: String,
    
    /// Display name
    pub name: String,
    
    /// Provider (openai, anthropic, ollama, etc.)
    pub provider: String,
    
    /// Model metadata
    pub metadata: ModelMetadata,
    
    /// Current status
    pub status: ModelStatus,
    
    /// Enabled for use
    pub enabled: bool,
    
    /// Custom configuration
    pub config: ModelConfig,
}

/// Model-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Maximum tokens for this model
    pub max_tokens: usize,
    
    /// Temperature override
    pub temperature: Option<f32>,
    
    /// Top-p override
    pub top_p: Option<f32>,
    
    /// Custom prompt prefix
    pub prompt_prefix: Option<String>,
    
    /// Custom prompt suffix
    pub prompt_suffix: Option<String>,
    
    /// Rate limiting
    pub rate_limit: Option<RateLimit>,
    
    /// Custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: None,
            top_p: None,
            prompt_prefix: None,
            prompt_suffix: None,
            rate_limit: None,
            custom_params: HashMap::new(),
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Requests per minute
    pub requests_per_minute: u32,
    
    /// Tokens per minute
    pub tokens_per_minute: u32,
    
    /// Concurrent requests
    pub max_concurrent: u32,
}

/// Cached model status
#[derive(Debug, Clone)]
struct ModelStatusCache {
    /// Cached status
    status: ModelStatus,
    
    /// Last update time
    last_updated: std::time::Instant,
    
    /// Cache duration
    ttl_seconds: u64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(api_config: Option<ApiKeysConfig>) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            api_config,
            status_cache: RwLock::new(HashMap::new()),
        }
    }
    
    /// Initialize with default models
    pub async fn initialize_defaults(&self) -> Result<()> {
        // Register OpenAI models if API key available
        if let Some(ref config) = self.api_config {
            if config.ai_models.openai.is_some() {
                self.register_openai_models().await?;
            }
            
            if config.ai_models.anthropic.is_some() {
                self.register_anthropic_models().await?;
            }
        }
        
        // Always try to register local models
        self.register_local_models().await?;
        
        Ok(())
    }
    
    /// Register OpenAI models
    async fn register_openai_models(&self) -> Result<()> {
        let models = vec![
            ("gpt-4-turbo", "GPT-4 Turbo", vec![ModelCapability::CodeGeneration, ModelCapability::Analysis]),
            ("gpt-4", "GPT-4", vec![ModelCapability::CodeGeneration, ModelCapability::Analysis]),
            ("gpt-3.5-turbo", "GPT-3.5 Turbo", vec![ModelCapability::Conversation]),
        ];
        
        for (id, name, capabilities) in models {
            let metadata = ModelMetadata {
                id: id.to_string(),
                name: name.to_string(),
                provider: "openai".to_string(),
                capabilities,
                context_window: 128000,
                cost_per_token: match id {
                    "gpt-4-turbo" => 0.001,
                    "gpt-4" => 0.003,
                    "gpt-3.5-turbo" => 0.00015,
                    _ => 0.001,
                },
                average_latency_ms: match id {
                    "gpt-4-turbo" => 2000,
                    "gpt-4" => 3000,
                    "gpt-3.5-turbo" => 500,
                    _ => 1000,
                },
                status: ModelStatus::Available,
                cost_per_1k_tokens: match id {
                    "gpt-4-turbo" => 1.0,
                    "gpt-4" => 3.0,
                    "gpt-3.5-turbo" => 0.15,
                    _ => 1.0,
                },
                quality_score: match id {
                    "gpt-4-turbo" => 0.95,
                    "gpt-4" => 0.9,
                    "gpt-3.5-turbo" => 0.7,
                    _ => 0.5,
                },
                avg_latency_ms: match id {
                    "gpt-4-turbo" => 2000,
                    "gpt-4" => 3000,
                    "gpt-3.5-turbo" => 500,
                    _ => 1000,
                },
            };
            
            self.register_model(RegisteredModel {
                id: id.to_string(),
                name: name.to_string(),
                provider: "openai".to_string(),
                metadata,
                status: ModelStatus::Available,
                enabled: true,
                config: ModelConfig::default(),
            }).await?;
        }
        
        Ok(())
    }
    
    /// Register Anthropic models
    async fn register_anthropic_models(&self) -> Result<()> {
        let models = vec![
            ("claude-3-opus", "Claude 3 Opus", vec![ModelCapability::Analysis, ModelCapability::Creative]),
            ("claude-3-sonnet", "Claude 3 Sonnet", vec![ModelCapability::Conversation, ModelCapability::Analysis]),
        ];
        
        for (id, name, capabilities) in models {
            let metadata = ModelMetadata {
                id: id.to_string(),
                name: name.to_string(),
                provider: "anthropic".to_string(),
                capabilities,
                context_window: 200000,
                cost_per_token: match id {
                    "claude-3-opus" => 0.0015,
                    "claude-3-sonnet" => 0.0003,
                    _ => 0.001,
                },
                average_latency_ms: 800,
                status: ModelStatus::Available,
                cost_per_1k_tokens: match id {
                    "claude-3-opus" => 1.5,
                    "claude-3-sonnet" => 0.3,
                    _ => 1.0,
                },
                quality_score: match id {
                    "claude-3-opus" => 0.95,
                    "claude-3-sonnet" => 0.85,
                    _ => 0.5,
                },
                avg_latency_ms: 800,
            };
            
            self.register_model(RegisteredModel {
                id: id.to_string(),
                name: name.to_string(),
                provider: "anthropic".to_string(),
                metadata,
                status: ModelStatus::Available,
                enabled: true,
                config: ModelConfig::default(),
            }).await?;
        }
        
        Ok(())
    }
    
    /// Register local Ollama models
    async fn register_local_models(&self) -> Result<()> {
        // Try to detect available Ollama models
        // In practice, would query Ollama API
        let models = vec![
            ("llama3:latest", "Llama 3", vec![ModelCapability::Conversation]),
            ("codellama:latest", "Code Llama", vec![ModelCapability::CodeGeneration]),
            ("mistral:latest", "Mistral", vec![ModelCapability::Conversation, ModelCapability::Analysis]),
        ];
        
        for (id, name, capabilities) in models {
            let metadata = ModelMetadata {
                id: id.to_string(),
                name: name.to_string(),
                provider: "ollama".to_string(),
                capabilities,
                context_window: 8192,
                cost_per_token: 0.0, // Local models are free
                average_latency_ms: 200, // Fast local inference
                status: ModelStatus::Available,
                cost_per_1k_tokens: 0.0, // Local models are free
                quality_score: 0.7,
                avg_latency_ms: 200, // Fast local inference
            };
            
            self.register_model(RegisteredModel {
                id: id.to_string(),
                name: name.to_string(),
                provider: "ollama".to_string(),
                metadata,
                status: ModelStatus::Unknown, // Will check availability later
                enabled: false, // Disabled until verified
                config: ModelConfig::default(),
            }).await?;
        }
        
        Ok(())
    }
    
    /// Register a model
    pub async fn register_model(&self, model: RegisteredModel) -> Result<()> {
        let mut models = self.models.write().await;
        models.insert(model.id.clone(), model);
        Ok(())
    }
    
    /// Get a model by ID
    pub async fn get_model(&self, id: &str) -> Option<RegisteredModel> {
        let models = self.models.read().await;
        models.get(id).cloned()
    }
    
    /// Get all registered models
    pub async fn get_all_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }
    
    /// Get enabled models
    pub async fn get_enabled_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values()
            .filter(|m| m.enabled)
            .cloned()
            .collect()
    }
    
    /// Get models by capability
    pub async fn get_models_by_capability(&self, capability: ModelCapability) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values()
            .filter(|m| m.metadata.capabilities.contains(&capability))
            .cloned()
            .collect()
    }
    
    /// Enable/disable a model
    pub async fn set_model_enabled(&self, id: &str, enabled: bool) -> Result<()> {
        let mut models = self.models.write().await;
        if let Some(model) = models.get_mut(id) {
            model.enabled = enabled;
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", id))
        }
    }
    
    /// Update model configuration
    pub async fn update_model_config(&self, id: &str, config: ModelConfig) -> Result<()> {
        let mut models = self.models.write().await;
        if let Some(model) = models.get_mut(id) {
            model.config = config;
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", id))
        }
    }
    
    /// Check model availability
    pub async fn check_model_availability(&self, id: &str) -> Result<ModelStatus> {
        // Check cache first
        {
            let cache = self.status_cache.read().await;
            if let Some(cached) = cache.get(id) {
                if cached.last_updated.elapsed().as_secs() < cached.ttl_seconds {
                    return Ok(cached.status.clone());
                }
            }
        }
        
        // Check actual availability
        let status = self.check_model_status(id).await?;
        
        // Update cache
        {
            let mut cache = self.status_cache.write().await;
            cache.insert(id.to_string(), ModelStatusCache {
                status: status.clone(),
                last_updated: std::time::Instant::now(),
                ttl_seconds: 300, // 5 minutes
            });
        }
        
        // Update model status
        {
            let mut models = self.models.write().await;
            if let Some(model) = models.get_mut(id) {
                model.status = status.clone();
            }
        }
        
        Ok(status)
    }
    
    /// Check actual model status
    async fn check_model_status(&self, id: &str) -> Result<ModelStatus> {
        let models = self.models.read().await;
        let model = models.get(id).ok_or_else(|| anyhow!("Model not found: {}", id))?;
        
        match model.provider.as_str() {
            "openai" => {
                // Check OpenAI API availability
                if self.api_config.as_ref()
                    .and_then(|c| c.ai_models.openai.as_ref())
                    .is_some()
                {
                    Ok(ModelStatus::Available)
                } else {
                    Ok(ModelStatus::NoCredentials)
                }
            }
            "anthropic" => {
                // Check Anthropic API availability
                if self.api_config.as_ref()
                    .and_then(|c| c.ai_models.anthropic.as_ref())
                    .is_some()
                {
                    Ok(ModelStatus::Available)
                } else {
                    Ok(ModelStatus::NoCredentials)
                }
            }
            "ollama" => {
                // Would check if Ollama is running and model is pulled
                // For now, return Unknown
                Ok(ModelStatus::Unknown)
            }
            _ => Ok(ModelStatus::Unknown),
        }
    }
    
    /// Get model statistics
    pub async fn get_model_stats(&self) -> ModelStats {
        let models = self.models.read().await;
        
        let total = models.len();
        let enabled = models.values().filter(|m| m.enabled).count();
        let available = models.values()
            .filter(|m| matches!(m.status, ModelStatus::Available))
            .count();
        
        let by_provider = models.values()
            .fold(HashMap::new(), |mut acc, model| {
                *acc.entry(model.provider.clone()).or_insert(0) += 1;
                acc
            });
        
        ModelStats {
            total_models: total,
            enabled_models: enabled,
            available_models: available,
            models_by_provider: by_provider,
        }
    }
}

/// Model statistics
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Total registered models
    pub total_models: usize,
    
    /// Enabled models
    pub enabled_models: usize,
    
    /// Available models
    pub available_models: usize,
    
    /// Models by provider
    pub models_by_provider: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_registry() {
        let registry = ModelRegistry::new(None);
        
        // Register a test model
        let model = RegisteredModel {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            provider: "test".to_string(),
            metadata: ModelMetadata {
                id: "".to_string(),
                name: "".to_string(),
                provider: "test".to_string(),
                capabilities: vec![ModelCapability::Conversation],
                context_window: 4096,
                cost_per_token: 0.0,
                average_latency_ms: 0,
                status: ModelStatus::Available,
                cost_per_1k_tokens: 1.0,
                quality_score: 0.8,
                avg_latency_ms: 500,
            },
            status: ModelStatus::Available,
            enabled: true,
            config: ModelConfig::default(),
        };
        
        registry.register_model(model).await.unwrap();
        
        // Test retrieval
        let retrieved = registry.get_model("test-model").await.unwrap();
        assert_eq!(retrieved.name, "Test Model");
        
        // Test stats
        let stats = registry.get_model_stats().await;
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.enabled_models, 1);
    }
}