//! Model registry management for chat initialization
//! 
//! Ensures models are properly registered and available before use

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

use crate::models::orchestrator::ModelOrchestrator;
use crate::tui::chat::orchestration::models::ModelMetadata;

/// Model registry for tracking available models
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    /// Registered models
    models: Arc<RwLock<HashMap<String, RegisteredModel>>>,
    
    /// Model orchestrator reference
    orchestrator: Option<Arc<ModelOrchestrator>>,
}

/// Registered model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    /// Model identifier
    pub id: String,
    
    /// Display name
    pub name: String,
    
    /// Provider (openai, anthropic, etc.)
    pub provider: String,
    
    /// Model capabilities
    pub capabilities: Vec<String>,
    
    /// Is the model currently available
    pub available: bool,
    
    /// Last verification timestamp
    pub last_verified: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Error message if unavailable
    pub error: Option<String>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            orchestrator: None,
        }
    }
    
    /// Create with model orchestrator
    pub fn with_orchestrator(orchestrator: Arc<ModelOrchestrator>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            orchestrator: Some(orchestrator),
        }
    }
    
    /// Initialize and verify all available models
    pub async fn initialize(&self) -> Result<()> {
        tracing::info!("🔍 Initializing model registry");
        
        // Discover models from environment and configuration
        self.discover_models().await?;
        
        // Verify each model's availability
        self.verify_all_models().await?;
        
        // Log summary
        let models = self.models.read().await;
        let available_count = models.values().filter(|m| m.available).count();
        let total_count = models.len();
        
        tracing::info!(
            "✅ Model registry initialized: {}/{} models available",
            available_count, total_count
        );
        
        if available_count == 0 {
            bail!("No models are available. Please check API keys and configuration.");
        }
        
        Ok(())
    }
    
    /// Discover models from environment and configuration
    async fn discover_models(&self) -> Result<()> {
        let mut models = self.models.write().await;
        
        // Check for OpenAI models
        if std::env::var("OPENAI_API_KEY").is_ok() {
            models.insert("gpt-4".to_string(), RegisteredModel {
                id: "gpt-4".to_string(),
                name: "GPT-4".to_string(),
                provider: "openai".to_string(),
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
                available: false, // Will be verified
                last_verified: None,
                error: None,
            });
            
            models.insert("gpt-3.5-turbo".to_string(), RegisteredModel {
                id: "gpt-3.5-turbo".to_string(),
                name: "GPT-3.5 Turbo".to_string(),
                provider: "openai".to_string(),
                capabilities: vec!["chat".to_string(), "code".to_string()],
                available: false,
                last_verified: None,
                error: None,
            });
        }
        
        // Check for Anthropic models
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            models.insert("claude-3-opus".to_string(), RegisteredModel {
                id: "claude-3-opus".to_string(),
                name: "Claude 3 Opus".to_string(),
                provider: "anthropic".to_string(),
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
                available: false,
                last_verified: None,
                error: None,
            });
            
            models.insert("claude-3-sonnet".to_string(), RegisteredModel {
                id: "claude-3-sonnet".to_string(),
                name: "Claude 3 Sonnet".to_string(),
                provider: "anthropic".to_string(),
                capabilities: vec!["chat".to_string(), "code".to_string()],
                available: false,
                last_verified: None,
                error: None,
            });
        }
        
        // Check for Google models
        if std::env::var("GEMINI_API_KEY").is_ok() {
            models.insert("gemini-pro".to_string(), RegisteredModel {
                id: "gemini-pro".to_string(),
                name: "Gemini Pro".to_string(),
                provider: "google".to_string(),
                capabilities: vec!["chat".to_string(), "multimodal".to_string()],
                available: false,
                last_verified: None,
                error: None,
            });
        }
        
        // Check for local models (Ollama)
        if self.check_ollama_available().await {
            // Try to list Ollama models
            if let Ok(local_models) = self.list_ollama_models().await {
                for model_name in local_models {
                    models.insert(model_name.clone(), RegisteredModel {
                        id: model_name.clone(),
                        name: format!("Ollama: {}", model_name),
                        provider: "ollama".to_string(),
                        capabilities: vec!["chat".to_string(), "local".to_string()],
                        available: false,
                        last_verified: None,
                        error: None,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Verify all discovered models
    async fn verify_all_models(&self) -> Result<()> {
        let model_ids: Vec<String> = {
            let models = self.models.read().await;
            models.keys().cloned().collect()
        };
        
        for model_id in model_ids {
            if let Err(e) = self.verify_model(&model_id).await {
                tracing::warn!("Failed to verify model {}: {}", model_id, e);
            }
        }
        
        Ok(())
    }
    
    /// Verify a specific model is available
    pub async fn verify_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        
        if let Some(model) = models.get_mut(model_id) {
            tracing::debug!("Verifying model: {}", model_id);
            
            // If we have an orchestrator, use it to verify
            if let Some(orchestrator) = &self.orchestrator {
                match self.verify_with_orchestrator(orchestrator, model_id).await {
                    Ok(_) => {
                        model.available = true;
                        model.last_verified = Some(chrono::Utc::now());
                        model.error = None;
                        tracing::info!("✓ Model {} is available", model_id);
                    }
                    Err(e) => {
                        model.available = false;
                        model.error = Some(e.to_string());
                        tracing::warn!("✗ Model {} is unavailable: {}", model_id, e);
                    }
                }
            } else {
                // Basic verification based on API key presence
                let available = match model.provider.as_str() {
                    "openai" => std::env::var("OPENAI_API_KEY").is_ok(),
                    "anthropic" => std::env::var("ANTHROPIC_API_KEY").is_ok(),
                    "google" => std::env::var("GEMINI_API_KEY").is_ok(),
                    "ollama" => self.check_ollama_model(model_id).await.unwrap_or(false),
                    _ => false,
                };
                
                model.available = available;
                model.last_verified = Some(chrono::Utc::now());
                
                if !available {
                    model.error = Some("API key not configured or service unavailable".to_string());
                }
            }
        } else {
            bail!("Model {} not found in registry", model_id);
        }
        
        Ok(())
    }
    
    /// Verify model with orchestrator
    async fn verify_with_orchestrator(
        &self,
        orchestrator: &Arc<ModelOrchestrator>,
        model_id: &str,
    ) -> Result<()> {
        // Try a simple completion to verify the model works
        use crate::models::{TaskRequest, TaskType, TaskConstraints};
        let test_request = TaskRequest {
            task_type: TaskType::GeneralChat,
            content: "Test".to_string(),
            constraints: TaskConstraints {
                max_tokens: Some(10),
                context_size: None,
                max_time: None,
                max_latency_ms: None,
                max_cost_cents: None,
                quality_threshold: None,
                priority: "normal".to_string(),
                prefer_local: false,
                require_streaming: false,
                required_capabilities: vec![],
                creativity_level: Some(0.1),
                formality_level: None,
                target_audience: None,
            },
            context_integration: false,
            memory_integration: false,
            cognitive_enhancement: false,
        };
        
        match orchestrator.execute_with_fallback(test_request).await {
            Ok(_) => Ok(()),
            Err(e) => Err(e).context(format!("Model {} verification failed", model_id)),
        }
    }
    
    /// Check if a model is available and verified
    pub async fn is_model_available(&self, model_id: &str) -> bool {
        let models = self.models.read().await;
        models.get(model_id).map(|m| m.available).unwrap_or(false)
    }
    
    /// Get model metadata
    pub async fn get_model(&self, model_id: &str) -> Option<RegisteredModel> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }
    
    /// List all available models
    pub async fn list_available_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values()
            .filter(|m| m.available)
            .cloned()
            .collect()
    }
    
    /// List all registered models (including unavailable)
    pub async fn list_all_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }
    
    /// Find the best available model for a given capability
    pub async fn find_model_for_capability(&self, capability: &str) -> Option<String> {
        let models = self.models.read().await;
        
        // First try to find a model that explicitly has this capability
        let capable_model = models.values()
            .filter(|m| m.available && m.capabilities.contains(&capability.to_string()))
            .min_by_key(|m| {
                // Prefer certain providers for certain capabilities
                match (capability, m.provider.as_str()) {
                    ("code", "openai") => 0,
                    ("analysis", "anthropic") => 0,
                    ("multimodal", "google") => 0,
                    ("local", "ollama") => 0,
                    _ => 1,
                }
            });
        
        capable_model.map(|m| m.id.clone())
    }
    
    /// Get a default model (first available)
    pub async fn get_default_model(&self) -> Option<String> {
        let models = self.models.read().await;
        
        // Prefer models in this order
        let preferred_order = vec!["gpt-4", "claude-3-opus", "gemini-pro", "gpt-3.5-turbo"];
        
        for model_id in preferred_order {
            if let Some(model) = models.get(model_id) {
                if model.available {
                    return Some(model.id.clone());
                }
            }
        }
        
        // Fall back to any available model
        models.values()
            .find(|m| m.available)
            .map(|m| m.id.clone())
    }
    
    /// Check if Ollama is available
    async fn check_ollama_available(&self) -> bool {
        // Check if ollama service is running
        match std::process::Command::new("ollama")
            .arg("list")
            .output()
        {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }
    
    /// List Ollama models
    async fn list_ollama_models(&self) -> Result<Vec<String>> {
        let output = std::process::Command::new("ollama")
            .arg("list")
            .output()
            .context("Failed to run ollama list")?;
        
        if !output.status.success() {
            bail!("ollama list failed");
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let models: Vec<String> = stdout
            .lines()
            .skip(1) // Skip header
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() {
                    Some(parts[0].to_string())
                } else {
                    None
                }
            })
            .collect();
        
        Ok(models)
    }
    
    /// Check if a specific Ollama model is available
    async fn check_ollama_model(&self, model_name: &str) -> Result<bool> {
        let models = self.list_ollama_models().await?;
        Ok(models.contains(&model_name.to_string()))
    }
    
    /// Register a custom model
    pub async fn register_model(&self, model: RegisteredModel) -> Result<()> {
        let mut models = self.models.write().await;
        models.insert(model.id.clone(), model);
        Ok(())
    }
    
    /// Remove a model from the registry
    pub async fn unregister_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        models.remove(model_id);
        Ok(())
    }
}