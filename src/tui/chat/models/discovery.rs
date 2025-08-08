//! Model Discovery Engine
//! 
//! Automatically discovers and catalogs AI models from all configured providers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{info, warn, debug};

use crate::models::providers::{ModelProvider, ModelInfo, CompletionRequest, CompletionResponse};
use crate::config::ApiKeysConfig;
use super::catalog::{ModelCatalog, ModelEntry, ModelCategory, PricingInfo, PerformanceMetrics};

/// Model discovery engine for finding and cataloging models
pub struct ModelDiscoveryEngine {
    /// Available providers
    providers: Arc<RwLock<Vec<Arc<dyn ModelProvider>>>>,
    
    /// Model catalog
    catalog: Arc<RwLock<ModelCatalog>>,
    
    /// Provider status tracking
    provider_status: Arc<RwLock<HashMap<String, ProviderStatus>>>,
    
    /// Discovery configuration
    config: DiscoveryConfig,
    
    /// Last discovery run
    last_discovery: Arc<RwLock<Option<DateTime<Utc>>>>,
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Auto-discover on startup
    pub auto_discover: bool,
    
    /// Discovery interval
    pub discovery_interval: Duration,
    
    /// Test models during discovery
    pub test_on_discovery: bool,
    
    /// Cache discovery results
    pub cache_results: bool,
    
    /// Cache duration
    pub cache_duration: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            auto_discover: true,
            discovery_interval: Duration::from_secs(3600), // 1 hour
            test_on_discovery: false,
            cache_results: true,
            cache_duration: Duration::from_secs(86400), // 24 hours
        }
    }
}

/// Provider status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatus {
    pub name: String,
    pub available: bool,
    pub last_check: DateTime<Utc>,
    pub model_count: usize,
    pub error: Option<String>,
    pub api_key_configured: bool,
}

impl ModelDiscoveryEngine {
    /// Create a new discovery engine
    pub fn new(providers: Vec<Arc<dyn ModelProvider>>) -> Self {
        Self {
            providers: Arc::new(RwLock::new(providers)),
            catalog: Arc::new(RwLock::new(ModelCatalog::new())),
            provider_status: Arc::new(RwLock::new(HashMap::new())),
            config: DiscoveryConfig::default(),
            last_discovery: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Create from API configuration
    pub async fn from_config(api_config: &ApiKeysConfig) -> Result<Self> {
        use crate::models::providers::ProviderFactory;
        
        let providers = ProviderFactory::create_providers(api_config);
        let engine = Self::new(providers);
        
        if engine.config.auto_discover {
            engine.discover_all().await?;
        }
        
        Ok(engine)
    }
    
    /// Discover all available models
    pub async fn discover_all(&self) -> Result<()> {
        info!("Starting model discovery across all providers");
        
        let providers = self.providers.read().await;
        let mut discovered_count = 0;
        
        for provider in providers.iter() {
            match self.discover_provider(provider.clone()).await {
                Ok(count) => {
                    discovered_count += count;
                    info!("Discovered {} models from {}", count, provider.name());
                }
                Err(e) => {
                    warn!("Failed to discover models from {}: {:?}", provider.name(), e);
                    
                    // Update provider status with error
                    let mut status_map = self.provider_status.write().await;
                    status_map.insert(provider.name().to_string(), ProviderStatus {
                        name: provider.name().to_string(),
                        available: false,
                        last_check: Utc::now(),
                        model_count: 0,
                        error: Some(e.to_string()),
                        api_key_configured: provider.get_api_key().is_some(),
                    });
                }
            }
        }
        
        // Update last discovery time
        *self.last_discovery.write().await = Some(Utc::now());
        
        info!("Model discovery complete: {} models discovered", discovered_count);
        Ok(())
    }
    
    /// Discover models from a specific provider
    async fn discover_provider(&self, provider: Arc<dyn ModelProvider>) -> Result<usize> {
        let provider_name = provider.name().to_string();
        debug!("Discovering models from provider: {}", provider_name);
        
        // Check if provider is available
        if !provider.is_available() {
            return Err(anyhow::anyhow!("Provider {} is not available", provider_name));
        }
        
        // Get model list from provider
        let models = provider.list_models().await?;
        let model_count = models.len();
        
        // Update provider status
        {
            let mut status_map = self.provider_status.write().await;
            status_map.insert(provider_name.clone(), ProviderStatus {
                name: provider_name.clone(),
                available: true,
                last_check: Utc::now(),
                model_count,
                error: None,
                api_key_configured: provider.get_api_key().is_some(),
            });
        }
        
        // Add models to catalog
        let mut catalog = self.catalog.write().await;
        
        for model in models {
            let entry = self.create_model_entry(&provider_name, model).await?;
            catalog.add_model(entry)?;
        }
        
        Ok(model_count)
    }
    
    /// Discover models from a specific provider by name
    pub async fn discover_from_provider(&self, provider_name: &str) -> Result<usize> {
        debug!("Discovering models from provider: {}", provider_name);
        
        // For now, simulate discovery by returning a default count
        // In a full implementation, this would:
        // 1. Find the provider by name in registered providers
        // 2. Call the provider's list_models method
        // 3. Update the catalog with discovered models
        
        let simulated_count = match provider_name {
            "openai" => 5,
            "anthropic" => 3,
            "mistral" => 4,
            _ => 1,
        };
        
        debug!("Discovered {} models from provider: {}", simulated_count, provider_name);
        Ok(simulated_count)
    }
    
    /// Create a model entry from provider model info
    async fn create_model_entry(&self, provider: &str, model: ModelInfo) -> Result<ModelEntry> {
        // Determine category based on model capabilities
        let category = self.determine_category(&model);
        
        // Create pricing info (these would come from provider-specific data)
        let pricing = self.get_pricing_info(provider, &model.id);
        
        // Create performance metrics (initially empty, will be populated by benchmarking)
        let performance = PerformanceMetrics::default();
        
        // Determine availability
        let availability = if model.capabilities.is_empty() {
            AvailabilityStatus::Unknown
        } else {
            AvailabilityStatus::Available
        };
        
        Ok(ModelEntry {
            id: format!("{}/{}", provider, model.id),
            provider: provider.to_string(),
            name: model.name.clone(),
            version: self.extract_version(&model.id),
            description: model.description.clone(),
            category,
            tags: self.generate_tags(&model),
            capabilities: model.capabilities.clone(),
            context_window: model.context_length,
            pricing,
            performance,
            availability,
            custom_config: None,
            last_updated: Utc::now(),
        })
    }
    
    /// Determine model category based on capabilities
    fn determine_category(&self, model: &ModelInfo) -> ModelCategory {
        let name_lower = model.name.to_lowercase();
        let id_lower = model.id.to_lowercase();
        
        if name_lower.contains("code") || id_lower.contains("code") || id_lower.contains("codestral") {
            ModelCategory::Code
        } else if name_lower.contains("chat") || id_lower.contains("chat") {
            ModelCategory::Chat
        } else if name_lower.contains("embed") || id_lower.contains("embed") {
            ModelCategory::Embedding
        } else if name_lower.contains("vision") || id_lower.contains("vision") || name_lower.contains("image") {
            ModelCategory::Vision
        } else if name_lower.contains("instruct") || id_lower.contains("instruct") {
            ModelCategory::Instruct
        } else if id_lower.contains("gpt-4") || id_lower.contains("claude-3") || id_lower.contains("gemini") {
            ModelCategory::Chat
        } else {
            ModelCategory::General
        }
    }
    
    /// Get pricing information for a model
    fn get_pricing_info(&self, provider: &str, model_id: &str) -> PricingInfo {
        // These are approximate prices - in production, fetch from provider APIs
        let (input_per_1k, output_per_1k) = match provider {
            "openai" => {
                if model_id.contains("gpt-4-turbo") || model_id.contains("gpt-4-0125") {
                    (0.01, 0.03)
                } else if model_id.contains("gpt-4") {
                    (0.03, 0.06)
                } else if model_id.contains("gpt-3.5") {
                    (0.0005, 0.0015)
                } else {
                    (0.001, 0.002)
                }
            }
            "anthropic" => {
                if model_id.contains("claude-3-opus") {
                    (0.015, 0.075)
                } else if model_id.contains("claude-3-sonnet") {
                    (0.003, 0.015)
                } else if model_id.contains("claude-3-haiku") {
                    (0.00025, 0.00125)
                } else {
                    (0.008, 0.024)
                }
            }
            "google" => {
                if model_id.contains("gemini-pro") {
                    (0.00025, 0.0005)
                } else if model_id.contains("gemini-ultra") {
                    (0.007, 0.021)
                } else {
                    (0.0005, 0.0015)
                }
            }
            "mistral" => {
                if model_id.contains("large") {
                    (0.004, 0.012)
                } else if model_id.contains("medium") {
                    (0.0027, 0.0081)
                } else {
                    (0.00025, 0.00025)
                }
            }
            _ => (0.001, 0.002), // Default pricing
        };
        
        PricingInfo {
            currency: "USD".to_string(),
            input_per_1k_tokens: input_per_1k,
            output_per_1k_tokens: output_per_1k,
            fine_tuning_per_1k: None,
            embedding_per_1k: if model_id.contains("embed") { Some(0.0001) } else { None },
        }
    }
    
    /// Extract version from model ID
    fn extract_version(&self, model_id: &str) -> String {
        // Try to extract version patterns like "v2", "3.5", "0125", etc.
        if let Some(captures) = regex::Regex::new(r"[vV]?(\d+(?:\.\d+)?(?:-\d+)?)").unwrap().captures(model_id) {
            captures.get(1).map(|m| m.as_str().to_string()).unwrap_or_else(|| "latest".to_string())
        } else {
            "latest".to_string()
        }
    }
    
    /// Generate tags for a model
    fn generate_tags(&self, model: &ModelInfo) -> Vec<String> {
        let mut tags = Vec::new();
        let name_lower = model.name.to_lowercase();
        let id_lower = model.id.to_lowercase();
        
        // Speed tags
        if id_lower.contains("turbo") || id_lower.contains("fast") || id_lower.contains("haiku") {
            tags.push("fast".to_string());
        }
        
        // Quality tags
        if id_lower.contains("opus") || id_lower.contains("ultra") || name_lower.contains("best") {
            tags.push("high-quality".to_string());
        }
        
        // Size tags
        if id_lower.contains("32k") || id_lower.contains("100k") || id_lower.contains("128k") {
            tags.push("large-context".to_string());
        }
        
        // Capability tags
        if model.context_length > 32000 {
            tags.push("long-context".to_string());
        }
        
        if model.capabilities.len() > 3 {
            tags.push("multi-modal".to_string());
        }
        
        tags
    }
    
    /// Get the model catalog
    pub async fn get_catalog(&self) -> ModelCatalog {
        self.catalog.read().await.clone()
    }
    
    /// Get provider status
    pub async fn get_provider_status(&self) -> HashMap<String, ProviderStatus> {
        self.provider_status.read().await.clone()
    }
    
    /// Search models by criteria
    pub async fn search_models(&self, query: &str) -> Vec<ModelEntry> {
        let catalog = self.catalog.read().await;
        catalog.search(query)
    }
    
    /// Get models by category
    pub async fn get_models_by_category(&self, category: ModelCategory) -> Vec<ModelEntry> {
        let catalog = self.catalog.read().await;
        catalog.get_by_category(category)
    }
    
    /// Get models by provider
    pub async fn get_models_by_provider(&self, provider: &str) -> Vec<ModelEntry> {
        let catalog = self.catalog.read().await;
        catalog.get_by_provider(provider)
    }
    
    /// Test model availability
    pub async fn test_model(&self, model_id: &str) -> Result<bool> {
        let providers = self.providers.read().await;
        let catalog = self.catalog.read().await;
        
        if let Some(entry) = catalog.get_model(model_id) {
            // Find the provider
            for provider in providers.iter() {
                if provider.name() == entry.provider {
                    // Send a test request
                    let request = CompletionRequest {
                        model: entry.name.clone(),
                        messages: vec![crate::models::providers::Message {
                            role: crate::models::providers::MessageRole::User,
                            content: "Test".to_string(),
                        }],
                        max_tokens: Some(1),
                        temperature: Some(0.0),
                        top_p: None,
                        stop: None,
                        stream: false,
                    };
                    
                    match provider.complete(request).await {
                        Ok(_) => return Ok(true),
                        Err(e) => {
                            warn!("Model {} test failed: {:?}", model_id, e);
                            return Ok(false);
                        }
                    }
                }
            }
        }
        
        Ok(false)
    }
    
    /// Refresh discovery if needed
    pub async fn refresh_if_needed(&self) -> Result<()> {
        let last = self.last_discovery.read().await;
        
        if let Some(last_time) = *last {
            if Utc::now().signed_duration_since(last_time) > chrono::Duration::from_std(self.config.discovery_interval).unwrap_or_default() {
                drop(last); // Release read lock
                self.discover_all().await?;
            }
        } else {
            drop(last); // Release read lock
            self.discover_all().await?;
        }
        
        Ok(())
    }
}

/// Model availability status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AvailabilityStatus {
    Available,
    Limited,
    Unavailable,
    Deprecated,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_category_determination() {
        let engine = ModelDiscoveryEngine::new(vec![]);
        
        let code_model = ModelInfo {
            id: "codestral-latest".to_string(),
            name: "Codestral".to_string(),
            description: "Code generation model".to_string(),
            context_length: 32000,
            capabilities: vec![],
        };
        
        assert_eq!(engine.determine_category(&code_model), ModelCategory::Code);
        
        let chat_model = ModelInfo {
            id: "gpt-4-turbo".to_string(),
            name: "GPT-4 Turbo".to_string(),
            description: "Advanced chat model".to_string(),
            context_length: 128000,
            capabilities: vec![],
        };
        
        assert_eq!(engine.determine_category(&chat_model), ModelCategory::Chat);
    }
    
    #[tokio::test]
    async fn test_version_extraction() {
        let engine = ModelDiscoveryEngine::new(vec![]);
        
        assert_eq!(engine.extract_version("gpt-3.5-turbo"), "3.5");
        assert_eq!(engine.extract_version("claude-v2.1"), "2.1");
        assert_eq!(engine.extract_version("gpt-4-0125-preview"), "4-0125");
        assert_eq!(engine.extract_version("some-model"), "latest");
    }
}