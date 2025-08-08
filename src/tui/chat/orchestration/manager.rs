//! Orchestration manager for multi-model coordination

use serde::{Serialize, Deserialize};

// Define routing strategy locally until models module is available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RoutingStrategy {
    RoundRobin,
    LeastLatency,
    ContextAware,
    CapabilityBased,
    CostOptimized,
    Custom(String),
    Capability,
    Cost,
    Speed,
    Quality,
    Availability,
    Hybrid,
}

// Define model capability locally
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelCapability {
    Chat,
    Code,
    Analysis,
    Multimodal,
    LongContext,
    CodeGeneration,
    Conversation,
    Creative,
    Embedding,
}

/// Provider type for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelProviderType {
    OpenAI,
    Anthropic,
    Gemini,
    Mistral,
    Local,
}

/// Model information for orchestration
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub provider: ModelProviderType,
    pub available: bool,
    pub capabilities: Vec<ModelCapability>,
}

/// Orchestration configuration view for rendering
#[derive(Debug, Clone)]
pub struct OrchestrationConfigView {
    pub context_window: usize,
    pub cost_limit: Option<f32>,
    pub strategy: RoutingStrategy,
    pub quality_threshold: f32,
    pub fallback_enabled: bool,
}

/// Orchestration setup modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrchestrationSetup {
    SingleModel,
    MultiModelRouting,
    EnsembleVoting,
    SpecializedAgents,
}

/// Manager for orchestration configuration and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationManager {
    pub selected_index: usize,
    pub orchestration_enabled: bool,
    pub preferred_strategy: RoutingStrategy,
    pub local_models_preference: f32, // 0.0 = prefer API, 1.0 = prefer local
    pub quality_threshold: f32,
    pub cost_threshold_cents: f32,
    pub parallel_models: usize,
    pub ensemble_enabled: bool,
    pub current_setup: OrchestrationSetup,
    
    // Additional fields for new modular system
    pub enabled_models: Vec<String>,
    pub allow_fallback: bool,
    pub context_window: usize,
    pub temperature: f32,
    pub stream_responses: bool,
    pub max_retries: u32,
    pub timeout_seconds: u64,
    pub model_capabilities: std::collections::HashMap<String, Vec<ModelCapability>>,
}

impl Default for OrchestrationManager {
    fn default() -> Self {
        Self {
            selected_index: 0,
            orchestration_enabled: false,
            preferred_strategy: RoutingStrategy::CapabilityBased,
            local_models_preference: 0.7, // Prefer local models
            quality_threshold: 0.8,
            cost_threshold_cents: 10.0,
            parallel_models: 3,
            ensemble_enabled: false,
            current_setup: OrchestrationSetup::SingleModel,
            
            // New fields
            enabled_models: Vec::new(),
            allow_fallback: true,
            context_window: 8192,
            temperature: 0.7,
            stream_responses: true,
            max_retries: 3,
            timeout_seconds: 30,
            model_capabilities: std::collections::HashMap::new(),
        }
    }
}

impl OrchestrationManager {
    /// Create manager with orchestration enabled
    pub fn enabled() -> Self {
        Self {
            orchestration_enabled: true,
            current_setup: OrchestrationSetup::MultiModelRouting,
            ..Default::default()
        }
    }
    
    /// Enable orchestration with specific setup
    pub fn enable_with_setup(&mut self, setup: OrchestrationSetup) {
        self.orchestration_enabled = true;
        self.current_setup = setup;
        
        // Adjust settings based on setup
        match setup {
            OrchestrationSetup::SingleModel => {
                self.ensemble_enabled = false;
                self.parallel_models = 1;
            }
            OrchestrationSetup::MultiModelRouting => {
                self.ensemble_enabled = false;
                self.parallel_models = 3;
            }
            OrchestrationSetup::EnsembleVoting => {
                self.ensemble_enabled = true;
                self.parallel_models = 5;
            }
            OrchestrationSetup::SpecializedAgents => {
                self.ensemble_enabled = false;
                self.parallel_models = 4;
            }
        }
    }
    
    /// Set routing strategy
    pub fn set_strategy(&mut self, strategy: RoutingStrategy) {
        self.preferred_strategy = strategy;
    }
    
    /// Set cost threshold in dollars
    pub fn set_cost_threshold_dollars(&mut self, dollars: f32) {
        self.cost_threshold_cents = dollars * 100.0;
    }
    
    /// Get cost threshold in dollars
    pub fn get_cost_threshold_dollars(&self) -> f32 {
        self.cost_threshold_cents / 100.0
    }
    
    /// Check if using local models preference
    pub fn prefers_local_models(&self) -> bool {
        self.local_models_preference > 0.5
    }
    
    /// Get a human-readable status string
    pub fn status_string(&self) -> String {
        if !self.orchestration_enabled {
            return "Disabled".to_string();
        }
        
        match self.current_setup {
            OrchestrationSetup::SingleModel => "Single Model".to_string(),
            OrchestrationSetup::MultiModelRouting => {
                format!("Multi-Model ({:?})", self.preferred_strategy)
            }
            OrchestrationSetup::EnsembleVoting => {
                format!("Ensemble ({} models)", self.parallel_models)
            }
            OrchestrationSetup::SpecializedAgents => "Specialized Agents".to_string(),
        }
    }
    
    /// Get enabled models with their information
    pub async fn get_enabled_models(&self) -> Vec<ModelInfo> {
        self.enabled_models.iter().map(|name| {
            ModelInfo {
                name: name.clone(),
                provider: ModelProviderType::OpenAI, // Default, could be determined from model name
                available: true,
                capabilities: self.model_capabilities.get(name)
                    .map(|caps| caps.clone())
                    .unwrap_or_default(),
            }
        }).collect()
    }
    
    /// Get the current orchestration configuration
    pub async fn get_config(&self) -> OrchestrationConfigView {
        OrchestrationConfigView {
            context_window: self.context_window,
            cost_limit: Some(self.cost_threshold_cents / 100.0),
            strategy: self.preferred_strategy.clone(),
            quality_threshold: self.quality_threshold,
            fallback_enabled: self.allow_fallback,
        }
    }
    
    /// Get the primary model if configured
    pub fn get_primary_model(&self) -> Option<ModelInfo> {
        self.enabled_models.first().map(|name| {
            ModelInfo {
                name: name.clone(),
                provider: ModelProviderType::OpenAI, // Default
                available: true,
                capabilities: self.model_capabilities.get(name)
                    .map(|caps| caps.clone())
                    .unwrap_or_default(),
            }
        })
    }
    
    /// Get available models
    pub async fn get_available_models(&self) -> Vec<String> {
        let mut models = Vec::new();
        
        // Load local models
        if let Ok(mut wizard) = crate::config::SetupWizard::new() {
            if wizard.scan_local_models(false).await.is_ok() {
                for model in &wizard.config.modelconfig.local_models {
                    models.push(format!("local:{}", model));
                }
            }
        }
        
        // Load API models based on configured keys
        if let Ok(api_config) = crate::config::ApiKeysConfig::from_env() {
            let ai_models = api_config.ai_models;
            
            // OpenAI models
            if ai_models.openai.is_some() {
                models.extend([
                    "openai:gpt-4".to_string(),
                    "openai:gpt-4-turbo".to_string(),
                    "openai:gpt-3.5-turbo".to_string(),
                ]);
            }
            
            // Anthropic models
            if ai_models.anthropic.is_some() {
                models.extend([
                    "anthropic:claude-3-opus".to_string(),
                    "anthropic:claude-3-sonnet".to_string(),
                    "anthropic:claude-3-haiku".to_string(),
                ]);
            }
            
            // Google models
            if ai_models.gemini.is_some() {
                models.extend([
                    "google:gemini-pro".to_string(),
                    "google:gemini-ultra".to_string(),
                ]);
            }
            
            // Mistral models
            if ai_models.mistral.is_some() {
                models.extend([
                    "mistral:mistral-large".to_string(),
                    "mistral:mistral-medium".to_string(),
                    "mistral:mistral-small".to_string(),
                ]);
            }
            
            // DeepSeek models
            if ai_models.deepseek.is_some() {
                models.extend([
                    "deepseek:deepseek-coder".to_string(),
                    "deepseek:deepseek-chat".to_string(),
                ]);
            }
            
            // Codestral models
            if ai_models.codestral.is_some() {
                models.push("codestral:latest".to_string());
            }
        }
        
        // If no models found, return some defaults
        if models.is_empty() {
            models = vec![
                "gpt-4".to_string(),
                "gpt-3.5-turbo".to_string(),
                "claude-3-opus".to_string(),
                "gemini-pro".to_string(),
            ];
        }
        
        models
    }
}