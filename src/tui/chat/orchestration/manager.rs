//! Orchestration manager for multi-model coordination

use serde::{Serialize, Deserialize};
use std::sync::Arc;
use crate::models::ModelOrchestrator;
use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use tracing::{debug, warn, error};

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
    QualityFirst,
    Availability,
    Hybrid,
    Adaptive,
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
#[derive(Clone, Serialize, Deserialize)]
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
    
    #[serde(skip)]
    pub event_bus: Option<Arc<EventBus>>,
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
            event_bus: None,
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
    
    /// Connect to ModelOrchestrator and sync available models
    pub async fn connect_to_orchestrator(&mut self, orchestrator: Arc<ModelOrchestrator>) -> anyhow::Result<()> {
        debug!("Connecting OrchestrationManager to ModelOrchestrator");
        
        // Get available models from orchestrator
        let status = orchestrator.get_status().await;
        debug!("Received orchestrator status with {} providers", status.api_providers.len());
        
        // Update enabled models based on what's available
        self.enabled_models.clear();
        
        // Add models from API providers
        for (provider_name, provider_status) in status.api_providers {
            if provider_status.is_available {
                debug!("Adding available model: {}", provider_name);
                self.enabled_models.push(provider_name.clone());
                
                // Infer capabilities based on model name
                let mut capabilities = Vec::new();
                let model_lower = provider_name.to_lowercase();
                
                if model_lower.contains("gpt") || model_lower.contains("claude") {
                    capabilities.push(ModelCapability::Chat);
                    capabilities.push(ModelCapability::Conversation);
                }
                if model_lower.contains("code") || model_lower.contains("codex") {
                    capabilities.push(ModelCapability::Code);
                    capabilities.push(ModelCapability::CodeGeneration);
                }
                if model_lower.contains("gpt-4") || model_lower.contains("claude-3") {
                    capabilities.push(ModelCapability::Analysis);
                    capabilities.push(ModelCapability::Creative);
                }
                if model_lower.contains("vision") || model_lower.contains("multimodal") {
                    capabilities.push(ModelCapability::Multimodal);
                }
                if model_lower.contains("32k") || model_lower.contains("100k") || model_lower.contains("200k") {
                    capabilities.push(ModelCapability::LongContext);
                }
                
                self.model_capabilities.insert(provider_name, capabilities);
            }
        }
        
        debug!("Total enabled models: {}", self.enabled_models.len());
        
        // Update orchestration setup based on available models
        if self.enabled_models.len() > 1 {
            self.orchestration_enabled = true;
            if self.enabled_models.len() >= 3 && self.ensemble_enabled {
                self.current_setup = OrchestrationSetup::EnsembleVoting;
                debug!("Orchestration setup: EnsembleVoting with {} models", self.enabled_models.len());
            } else {
                self.current_setup = OrchestrationSetup::MultiModelRouting;
                debug!("Orchestration setup: MultiModelRouting with {} models", self.enabled_models.len());
            }
        } else if self.enabled_models.len() == 1 {
            self.current_setup = OrchestrationSetup::SingleModel;
            debug!("Orchestration setup: SingleModel");
        } else {
            warn!("No models available for orchestration");
        }
        
        Ok(())
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
    
    /// Set the event bus for broadcasting orchestration events
    pub fn set_event_bus(&mut self, event_bus: Arc<EventBus>) {
        self.event_bus = Some(event_bus);
    }
    
    /// Broadcast a model selection event
    async fn broadcast_model_selected(&self, model_id: &str) {
        if let Some(ref event_bus) = self.event_bus {
            if let Err(e) = event_bus.publish(SystemEvent::ModelSelected {
                model_id: model_id.to_string(),
                source: TabId::Chat,
            }).await {
                warn!("Failed to broadcast model selection event: {}", e);
            }
        }
    }
    
    /// Broadcast an orchestration event
    async fn broadcast_orchestration_event(&self, request_id: String, config: serde_json::Value) {
        if let Some(ref event_bus) = self.event_bus {
            if let Err(e) = event_bus.publish(SystemEvent::OrchestrationRequested {
                request_id,
                config,
            }).await {
                warn!("Failed to broadcast orchestration event: {}", e);
            }
        }
    }
    
    /// Select the best model for a given task based on routing strategy
    pub async fn select_model_for_task(&self, task_type: &str) -> Option<String> {
        if self.enabled_models.is_empty() {
            warn!("No models available for task: {}", task_type);
            return None;
        }
        
        let selected = match self.preferred_strategy {
            RoutingStrategy::CapabilityBased | RoutingStrategy::Capability => {
                // Select based on task type and model capabilities
                self.select_by_capability(task_type)
            }
            RoutingStrategy::CostOptimized | RoutingStrategy::Cost => {
                // Prefer cheaper models first
                self.select_by_cost()
            }
            RoutingStrategy::Speed => {
                // Select fastest model (usually smaller models)
                self.select_by_speed()
            }
            RoutingStrategy::QualityFirst | RoutingStrategy::Quality => {
                // Select highest quality model (usually larger models)
                self.select_by_quality()
            }
            RoutingStrategy::RoundRobin => {
                // Simple round-robin selection
                self.select_round_robin()
            }
            RoutingStrategy::LeastLatency => {
                // Select model with lowest current latency
                self.select_by_latency()
            }
            RoutingStrategy::Availability => {
                // Select first available model
                self.enabled_models.first().cloned()
            }
            _ => {
                // Default to first available model
                self.enabled_models.first().cloned()
            }
        };
        
        // Broadcast model selection event
        if let Some(ref model) = selected {
            debug!("Selected model '{}' for task type '{}'", model, task_type);
            self.broadcast_model_selected(model).await;
        } else {
            error!("Failed to select any model for task type '{}' with strategy {:?}", task_type, self.preferred_strategy);
        }
        
        selected
    }
    
    /// Select model based on capabilities
    fn select_by_capability(&self, task_type: &str) -> Option<String> {
        let task_lower = task_type.to_lowercase();
        
        // Map task types to required capabilities
        let required_capabilities = if task_lower.contains("code") || task_lower.contains("programming") {
            vec![ModelCapability::Code, ModelCapability::CodeGeneration]
        } else if task_lower.contains("chat") || task_lower.contains("conversation") {
            vec![ModelCapability::Chat, ModelCapability::Conversation]
        } else if task_lower.contains("analysis") || task_lower.contains("reasoning") {
            vec![ModelCapability::Analysis]
        } else if task_lower.contains("creative") || task_lower.contains("story") {
            vec![ModelCapability::Creative]
        } else if task_lower.contains("image") || task_lower.contains("visual") {
            vec![ModelCapability::Multimodal]
        } else if task_lower.contains("long") || task_lower.contains("document") {
            vec![ModelCapability::LongContext]
        } else {
            vec![ModelCapability::Chat] // Default to chat
        };
        
        // Find models with matching capabilities
        for model in &self.enabled_models {
            if let Some(capabilities) = self.model_capabilities.get(model) {
                if required_capabilities.iter().any(|req| capabilities.contains(req)) {
                    return Some(model.clone());
                }
            }
        }
        
        // Fallback to first available model
        self.enabled_models.first().cloned()
    }
    
    /// Select model based on cost optimization
    fn select_by_cost(&self) -> Option<String> {
        // Prefer models in this order: local > mistral > gemini > openai > anthropic
        for model in &self.enabled_models {
            let model_lower = model.to_lowercase();
            if model_lower.contains("local") || model_lower.contains("ollama") {
                return Some(model.clone());
            }
        }
        for model in &self.enabled_models {
            if model.to_lowercase().contains("mistral") || model.to_lowercase().contains("deepseek") {
                return Some(model.clone());
            }
        }
        self.enabled_models.first().cloned()
    }
    
    /// Select model based on speed
    fn select_by_speed(&self) -> Option<String> {
        // Prefer smaller, faster models
        for model in &self.enabled_models {
            let model_lower = model.to_lowercase();
            if model_lower.contains("haiku") || model_lower.contains("small") || model_lower.contains("3.5") {
                return Some(model.clone());
            }
        }
        self.enabled_models.first().cloned()
    }
    
    /// Select model based on quality
    fn select_by_quality(&self) -> Option<String> {
        // Prefer larger, more capable models
        for model in &self.enabled_models {
            let model_lower = model.to_lowercase();
            if model_lower.contains("opus") || model_lower.contains("gpt-4") || model_lower.contains("ultra") {
                return Some(model.clone());
            }
        }
        self.enabled_models.first().cloned()
    }
    
    /// Simple round-robin selection
    fn select_round_robin(&self) -> Option<String> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        
        if self.enabled_models.is_empty() {
            return None;
        }
        
        let index = COUNTER.fetch_add(1, Ordering::Relaxed) % self.enabled_models.len();
        self.enabled_models.get(index).cloned()
    }
    
    /// Select model with lowest latency (placeholder - would need actual metrics)
    fn select_by_latency(&self) -> Option<String> {
        // For now, just return first available model
        // In production, this would check actual latency metrics
        self.enabled_models.first().cloned()
    }
    
    /// Route a message to appropriate models based on orchestration setup
    pub async fn route_message(&self, message: &str, task_type: &str) -> Vec<String> {
        match self.current_setup {
            OrchestrationSetup::SingleModel => {
                // Route to single model
                if let Some(model) = self.select_model_for_task(task_type).await {
                    vec![model]
                } else {
                    vec![]
                }
            }
            OrchestrationSetup::MultiModelRouting => {
                // Route to single best model based on strategy
                if let Some(model) = self.select_model_for_task(task_type).await {
                    vec![model]
                } else {
                    vec![]
                }
            }
            OrchestrationSetup::EnsembleVoting => {
                // Route to multiple models for ensemble voting
                let num_models = self.parallel_models.min(self.enabled_models.len());
                self.enabled_models.iter().take(num_models).cloned().collect()
            }
            OrchestrationSetup::SpecializedAgents => {
                // Route to specialized models based on task
                // For now, select models based on capabilities
                let mut selected = Vec::new();
                
                // Add a code model if task involves coding
                if task_type.to_lowercase().contains("code") {
                    for model in &self.enabled_models {
                        if let Some(caps) = self.model_capabilities.get(model) {
                            if caps.contains(&ModelCapability::Code) {
                                selected.push(model.clone());
                                break;
                            }
                        }
                    }
                }
                
                // Add an analysis model if task involves analysis
                if task_type.to_lowercase().contains("analysis") {
                    for model in &self.enabled_models {
                        if let Some(caps) = self.model_capabilities.get(model) {
                            if caps.contains(&ModelCapability::Analysis) && !selected.contains(model) {
                                selected.push(model.clone());
                                break;
                            }
                        }
                    }
                }
                
                // Ensure at least one model is selected
                if selected.is_empty() {
                    if let Some(model) = self.enabled_models.first() {
                        selected.push(model.clone());
                    }
                }
                
                selected
            }
        }
    }
}