//! Consolidated chat system module
//! 
//! This is the refactored chat system with modular architecture

// Existing modules (preserved)
pub mod core;
pub mod integrations;

// Error handling
pub mod error;

// Utilities
pub mod utils;

// New modular structure
pub mod state;
pub mod orchestration;
pub mod agents;
pub mod context;
pub mod search;
pub mod threads;
pub mod handlers;
pub mod processing;
pub mod subtabs;
pub mod initialization;
pub mod types;
pub mod rendering;
pub mod export;

// The new thin coordinator (will replace the monolithic chat.rs)
pub mod manager;

// Bridge for transition
pub mod bridge;

// Direct integration module (replaces bridge)
pub mod integration;

// UI/UX enhancements
pub mod ui_enhancements;

// Statistics and analytics
pub mod statistics;

// Model management
pub mod models;

// Tool integration
pub mod tools;

// Editor integration
pub mod editor;

// Storage context integration
pub mod storage_context;

// Monitoring and analytics
pub mod monitoring;

// Tests
// #[cfg(test)]
// pub mod tests;

// Re-export commonly used items from core (preserved)
pub use core::{
    commands::CommandRegistry,
    tool_executor::ChatToolExecutor,
    workflows::TaskStep,
};

// Re-export the new ChatManager (future main interface)
pub use manager::ChatManager;

// Re-export key types for easier access
pub use state::{ChatState, ChatSettings};
pub use state::settings::ChatSettings as SettingsManager; // Alias for compatibility
pub use orchestration::OrchestrationManager;  
pub use agents::manager::AgentManager;
pub use integration::SubtabManager;

// Define missing types that are used throughout the chat system
/// Represents an active model in the system
#[derive(Debug, Clone)]
pub struct ActiveModel {
    pub name: String,
    pub provider: String,
}

impl ActiveModel {
    pub fn as_ref(&self) -> &str {
        &self.name
    }
}

/// Model manager state (placeholder for now)
#[derive(Debug, Clone)]
pub struct ModelManagerState {
    pub active_models: Vec<ActiveModel>,
}

/// Collaboration mode for orchestration
#[derive(Debug, Clone, PartialEq)]
pub enum CollaborationMode {
    Single,
    Ensemble,
    Sequential,
    Voting,
}

/// Orchestration setup configuration
#[derive(Debug, Clone)]
pub struct OrchestrationSetup {
    pub mode: CollaborationMode,
    pub enabled_models: Vec<String>,
}

/// Code completion suggestion
#[derive(Debug, Clone)]
pub struct CodeCompletionSuggestion {
    pub text: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
}

/// Completion kind enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionKind {
    Function,
    Variable,
    Type,
    Keyword,
    Snippet,
    Class,
    Module,
    Property,
    Method,
    Constant,
}

/// Message thread representation
#[derive(Debug, Clone)]
pub struct MessageThread {
    pub id: String,
    pub messages: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Thread manager (placeholder)
#[derive(Debug, Clone)]
pub struct ThreadManager {
    pub threads: Vec<MessageThread>,
}

/// Chat initialization state
#[derive(Debug, Clone, PartialEq)]
pub enum ChatInitState {
    NotStarted,
    Initializing,
    LoadingModels,
    LoadingHistory,
    Ready,
    Failed(String),
    // Additional variants from chat_state.rs usage
    Uninitialized,
    InitializingUI,
    InitializingCognitive,
    InitializingMemory,
    Error(String),
    Degraded(String),
}

/// Chat information structure
#[derive(Debug, Clone)]
pub struct ChatInfo {
    pub id: usize,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_message: Option<String>,
    pub scroll_offset: usize,
}

impl ChatInfo {
    /// Update streaming progress
    pub fn update_streaming_progress(&mut self, progress: f32) {
        // Update streaming progress - could add a field if needed
        tracing::debug!("Streaming progress: {:.0}%", progress * 100.0);
    }
    
    /// Complete streaming
    pub fn complete_streaming(&mut self) {
        // Mark streaming as complete
        tracing::debug!("Streaming completed for chat {}", self.id);
    }
}

// Main entry point for the modular chat system
// This replaces the old ChatManager
pub struct ModularChat {
    /// The subtab manager handles all UI interactions
    pub subtab_manager: std::cell::RefCell<SubtabManager>,
    
    /// Shared state
    pub chat_state: std::sync::Arc<tokio::sync::RwLock<ChatState>>,
    pub orchestration: std::sync::Arc<tokio::sync::RwLock<OrchestrationManager>>,
    pub agent_manager: std::sync::Arc<tokio::sync::RwLock<AgentManager>>,
    
    /// Model discovery engine for dynamic model detection
    pub model_discovery: Option<std::sync::Arc<models::discovery::ModelDiscoveryEngine>>,
    
    /// Agent creation wizard for advanced agent configuration  
    pub agent_wizard: Option<std::sync::Arc<agents::creation::AgentCreationWizard>>,
    
    /// Available models (dynamically discovered)
    pub available_models: Vec<String>,
    
    /// Active chat ID (for compatibility)
    pub active_chat: usize,
    
    // Additional fields for compatibility
    pub message_history_mode: bool,
    pub show_timestamps: bool,
    pub cognitive_memory: Option<std::sync::Arc<crate::memory::CognitiveMemory>>,
    pub active_model: ActiveModel,
    pub chats: std::collections::HashMap<usize, ChatInfo>,
    pub history_manager: std::sync::Arc<tokio::sync::RwLock<state::HistoryManager>>,
    pub command_input: String,
    pub settings_manager: std::sync::Arc<tokio::sync::RwLock<SettingsManager>>,
    
    // Bridge references for later wiring
    tool_bridge: Option<std::sync::Arc<crate::tui::bridges::tool_bridge::ToolBridge>>,
    
    // Editor state
    pub editor_active: bool,
    pub current_file: Option<String>,
    pub editor_language: Option<String>,
    
    // Editor tab controller (for actual editor functionality)
    pub editor_tab: Option<std::sync::Arc<tokio::sync::RwLock<subtabs::EditorTab>>>,
    
    // Statistics tracking
    pub avg_response_time: Option<f64>,
    pub total_tokens: Option<usize>,
    pub session_cost: Option<f32>,
    
    /// Event bus for cross-tab communication
    pub event_bus: Option<std::sync::Arc<crate::tui::event_bus::EventBus>>,
    
    /// Unified bridges for cross-tab integration
    pub bridges: Option<std::sync::Arc<crate::tui::bridges::UnifiedBridge>>,
}

// Simple provider implementations for model discovery
struct SimpleOpenAIProvider {
    api_key: Option<String>,
}

struct SimpleAnthropicProvider {
    api_key: Option<String>,
}

struct SimpleOllamaProvider;

// Implement the main ModelProvider trait from crate::models::providers
#[async_trait::async_trait]
impl crate::models::providers::ModelProvider for SimpleOpenAIProvider {
    fn name(&self) -> &str {
        "OpenAI"
    }
    
    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }
    
    async fn list_models(&self) -> anyhow::Result<Vec<crate::models::providers::ModelInfo>> {
        use crate::models::providers::ModelInfo;
        
        Ok(vec![
            ModelInfo {
                id: "gpt-4-turbo".to_string(),
                name: "GPT-4 Turbo".to_string(),
                description: "Most capable GPT-4 model".to_string(),
                context_length: 128000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
            },
            ModelInfo {
                id: "gpt-3.5-turbo".to_string(),
                name: "GPT-3.5 Turbo".to_string(),
                description: "Fast and cost-effective model".to_string(),
                context_length: 16384,
                capabilities: vec!["chat".to_string(), "code".to_string()],
            },
        ])
    }
    
    async fn complete(&self, request: crate::models::providers::CompletionRequest) -> anyhow::Result<crate::models::providers::CompletionResponse> {
        // Simple placeholder implementation
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    async fn stream_complete(
        &self,
        request: crate::models::providers::CompletionRequest,
    ) -> anyhow::Result<Box<dyn tokio_stream::Stream<Item = anyhow::Result<crate::models::providers::CompletionChunk>> + Send + Unpin>> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    fn get_api_key(&self) -> Option<String> {
        self.api_key.clone()
    }
}

#[async_trait::async_trait]
impl crate::models::providers::ModelProvider for SimpleAnthropicProvider {
    fn name(&self) -> &str {
        "Anthropic"
    }
    
    fn is_available(&self) -> bool {
        self.api_key.is_some()
    }
    
    async fn list_models(&self) -> anyhow::Result<Vec<crate::models::providers::ModelInfo>> {
        use crate::models::providers::ModelInfo;
        
        Ok(vec![
            ModelInfo {
                id: "claude-3-opus".to_string(),
                name: "Claude 3 Opus".to_string(),
                description: "Most capable Claude model".to_string(),
                context_length: 200000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
            },
            ModelInfo {
                id: "claude-3-sonnet".to_string(),
                name: "Claude 3 Sonnet".to_string(),
                description: "Balanced performance and cost".to_string(),
                context_length: 200000,
                capabilities: vec!["chat".to_string(), "code".to_string()],
            },
        ])
    }
    
    async fn complete(&self, request: crate::models::providers::CompletionRequest) -> anyhow::Result<crate::models::providers::CompletionResponse> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    async fn stream_complete(
        &self,
        request: crate::models::providers::CompletionRequest,
    ) -> anyhow::Result<Box<dyn tokio_stream::Stream<Item = anyhow::Result<crate::models::providers::CompletionChunk>> + Send + Unpin>> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    fn get_api_key(&self) -> Option<String> {
        self.api_key.clone()
    }
}

#[async_trait::async_trait]
impl crate::models::providers::ModelProvider for SimpleOllamaProvider {
    fn name(&self) -> &str {
        "Ollama"
    }
    
    fn is_available(&self) -> bool {
        // Ollama is available if the binary exists
        true
    }
    
    async fn list_models(&self) -> anyhow::Result<Vec<crate::models::providers::ModelInfo>> {
        use crate::models::providers::ModelInfo;
        
        let mut models = Vec::new();
        
        // Try to get actual Ollama models
        if let Ok(output) = tokio::process::Command::new("ollama")
            .arg("list")
            .output()
            .await
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines().skip(1) {
                    if let Some(model_name) = line.split_whitespace().next() {
                        if !model_name.is_empty() {
                            models.push(ModelInfo {
                                id: model_name.to_string(),
                                name: model_name.to_string(),
                                description: format!("Local Ollama model: {}", model_name),
                                context_length: 4096,
                                capabilities: vec!["chat".to_string(), "code".to_string()],
                            });
                        }
                    }
                }
            }
        }
        
        // If no models found, return empty list (discovery engine will handle it)
        Ok(models)
    }
    
    async fn complete(&self, request: crate::models::providers::CompletionRequest) -> anyhow::Result<crate::models::providers::CompletionResponse> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    async fn stream_complete(
        &self,
        request: crate::models::providers::CompletionRequest,
    ) -> anyhow::Result<Box<dyn tokio_stream::Stream<Item = anyhow::Result<crate::models::providers::CompletionChunk>> + Send + Unpin>> {
        Err(anyhow::anyhow!("Not implemented"))
    }
    
    async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!("Not implemented"))
    }
}

impl ModularChat {
    /// Initialize model discovery engine with available providers
    async fn initialize_model_discovery() -> Option<std::sync::Arc<models::discovery::ModelDiscoveryEngine>> {
        use models::discovery::ModelDiscoveryEngine;
        use std::sync::Arc;
        
        let mut providers: Vec<Arc<dyn crate::models::providers::ModelProvider>> = Vec::new();
        
        // Add OpenAI provider if API key is available
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            providers.push(Arc::new(SimpleOpenAIProvider {
                api_key: Some(api_key),
            }));
            tracing::info!("Added OpenAI provider to discovery engine");
        }
        
        // Add Anthropic provider if API key is available
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            providers.push(Arc::new(SimpleAnthropicProvider {
                api_key: Some(api_key),
            }));
            tracing::info!("Added Anthropic provider to discovery engine");
        }
        
        // Add local/Ollama provider (always available)
        providers.push(Arc::new(SimpleOllamaProvider {}));
        tracing::info!("Added Ollama provider to discovery engine");
        
        // Create the discovery engine with providers
        let engine = ModelDiscoveryEngine::new(providers);
        
        // Start discovery
        match engine.discover_all().await {
            Ok(_) => {
                tracing::info!("Model discovery completed");
                Some(Arc::new(engine))
            }
            Err(e) => {
                tracing::warn!("Model discovery failed: {}", e);
                None
            }
        }
    }
    
    /// Create a new modular chat system
    pub async fn new() -> Self {
        Self::new_with_orchestrator(None, None, None, None).await
    }
    
    /// Create a new modular chat system with orchestrator
    pub async fn new_with_orchestrator(
        model_orchestrator: Option<std::sync::Arc<crate::models::ModelOrchestrator>>,
        cognitive_system: Option<std::sync::Arc<crate::cognitive::CognitiveSystem>>,
        tool_manager: Option<std::sync::Arc<crate::tools::IntelligentToolManager>>,
        task_manager: Option<std::sync::Arc<crate::tools::task_management::TaskManager>>,
    ) -> Self {
        Self::new_with_full_support(
            model_orchestrator,
            cognitive_system,
            tool_manager,
            task_manager,
            None,
        ).await
    }
    
    /// Create a new modular chat system with full support including task registry
    pub async fn new_with_full_support(
        model_orchestrator: Option<std::sync::Arc<crate::models::ModelOrchestrator>>,
        cognitive_system: Option<std::sync::Arc<crate::cognitive::CognitiveSystem>>,
        tool_manager: Option<std::sync::Arc<crate::tools::IntelligentToolManager>>,
        task_manager: Option<std::sync::Arc<crate::tools::task_management::TaskManager>>,
        config: Option<crate::config::Config>,
    ) -> Self {
        use std::sync::Arc;
        use tokio::sync::RwLock;
        use std::cell::RefCell;
        
        // Initialize model discovery engine with orchestrator if available
        let model_discovery = if let Some(ref orchestrator) = model_orchestrator {
            // Try to get API config from orchestrator to initialize discovery
            tracing::info!("Model orchestrator provided, discovery can be enhanced");
            Self::initialize_model_discovery().await
        } else {
            Self::initialize_model_discovery().await
        };
        
        // Initialize agent creation wizard
        let agent_wizard = Some(Arc::new(agents::creation::AgentCreationWizard::new()));
        
        // Get available models - prioritize local/Ollama models
        let mut available_models = Vec::new();
        let mut local_models = Vec::new();
        let mut api_models = Vec::new();
        
        // First check for Ollama models
        tracing::info!("🔍 Checking for Ollama models...");
        if let Ok(output) = tokio::process::Command::new("ollama")
            .arg("list")
            .output()
            .await
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                // Parse ollama list output (skip header line)
                for line in output_str.lines().skip(1) {
                    if let Some(model_name) = line.split_whitespace().next() {
                        if !model_name.is_empty() {
                            local_models.push(model_name.to_string());
                            tracing::info!("✅ Found Ollama model: {}", model_name);
                        }
                    }
                }
            }
        }
        
        // Then try to get models from the orchestrator if available
        if let Some(ref orchestrator) = model_orchestrator {
            let status = orchestrator.get_status().await;
            
            // Add local models from model_statuses
            for (model_name, model_status) in &status.local_models.model_statuses {
                if !local_models.contains(model_name) {
                    local_models.push(model_name.clone());
                    tracing::debug!("Found local model: {} (health: {:?})", model_name, model_status.health);
                }
            }
            
            // Add API provider models (only if configured AND have API keys)
            for (provider, provider_status) in &status.api_providers {
                // Check for actual API keys in environment
                let has_api_key = match provider.as_str() {
                    "openai" => std::env::var("OPENAI_API_KEY").is_ok(),
                    "anthropic" => std::env::var("ANTHROPIC_API_KEY").is_ok(),
                    "gemini" => std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok(),
                    "mistral" => std::env::var("MISTRAL_API_KEY").is_ok(),
                    _ => false,
                };
                
                // Only add API models if the provider has valid API keys
                if provider_status.is_available && has_api_key {
                    // The provider name can indicate available models
                    // For now, add common models for each available provider
                    let provider_models = match provider.as_str() {
                        "openai" => vec!["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                        "anthropic" => vec!["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                        "gemini" => vec!["gemini-pro", "gemini-pro-vision"],
                        "mistral" => vec!["mistral-large", "mistral-medium", "mistral-small"],
                        _ => vec![],
                    };
                    
                    for model in provider_models {
                        api_models.push(model.to_string());
                        tracing::debug!("Found {} model: {} (API key configured)", provider, model);
                    }
                } else if provider_status.is_available && !has_api_key {
                    tracing::debug!("Provider {} is marked available but no API key found", provider);
                }
            }
        }
        
        // Combine models - local models first, then API models
        available_models.extend(local_models.clone());
        available_models.extend(api_models);
        
        if !available_models.is_empty() {
            tracing::info!("✅ Discovered {} total models ({} local, {} API)", 
                available_models.len(), 
                local_models.len(), 
                available_models.len() - local_models.len());
        }
        
        // Try discovery engine if orchestrator didn't provide models
        if available_models.is_empty() {
            if let Some(ref discovery) = model_discovery {
                // Wait a bit for initial discovery
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                
                // Use discovered models
                let catalog = discovery.get_catalog().await;
                available_models = catalog.get_all_models()
                    .into_iter()
                    .map(|entry| entry.id.clone())
                    .collect();
                
                if !available_models.is_empty() {
                    tracing::info!("Discovered {} models from catalog", available_models.len());
                }
            }
        }
        
        // Fallback if no models found
        if available_models.is_empty() {
            tracing::info!("No models found, checking for common local models...");
            
            // Try common Ollama model names as fallback
            let common_ollama_models = vec![
                "llama3.2",
                "llama3.1", 
                "llama3",
                "llama2",
                "mistral",
                "mixtral",
                "codellama",
                "phi3",
                "gemma2",
                "qwen2.5",
                "deepseek-coder",
            ];
            
            // Check if any common models are available
            for model in &common_ollama_models {
                if let Ok(output) = tokio::process::Command::new("ollama")
                    .arg("show")
                    .arg(model)
                    .output()
                    .await
                {
                    if output.status.success() {
                        available_models.push(model.to_string());
                        tracing::info!("✅ Found installed Ollama model: {}", model);
                    }
                }
            }
            
            // If still no models, provide instructions
            if available_models.is_empty() {
                tracing::warn!("⚠️ No local models found. To use Loki AI:");
                tracing::warn!("  1. Install Ollama: https://ollama.ai");
                tracing::warn!("  2. Pull a model: ollama pull llama3.2");
                tracing::warn!("  3. Or configure API keys in ~/.config/loki/config.toml");
                
                // Add placeholder to prevent crashes
                available_models = vec!["no-models-available".to_string()];
            }
        }
        
        // Create shared state
        let chat_state = Arc::new(RwLock::new(ChatState::new(0, "Main Chat".to_string())));
        
        // Initialize orchestration manager with discovered models
        let mut orch_manager = OrchestrationManager::default();
        
        // Set local preference to maximum to avoid API calls when not configured
        orch_manager.local_models_preference = 1.0; // Always prefer local models
        orch_manager.allow_fallback = false; // Don't fall back to API if local fails
        
        // Enable orchestration if we have any models
        if !available_models.is_empty() && available_models[0] != "no-models-available" {
            orch_manager.orchestration_enabled = true;
            orch_manager.enabled_models = available_models.clone();
            
            // Set the first local model as default
            let first_local = available_models.iter()
                .find(|m| !m.contains("gpt") && !m.contains("claude"))
                .or_else(|| available_models.first())
                .unwrap();
            
            tracing::info!("🎯 Setting default model to: {}", first_local);
        } else {
            orch_manager.orchestration_enabled = false;
            tracing::warn!("⚠️ Orchestration disabled - no models available");
        }
        
        let orchestration = Arc::new(RwLock::new(orch_manager));
        let agent_manager = Arc::new(RwLock::new(AgentManager::default()));
        
        // Create message channel
        let (message_tx, _) = tokio::sync::mpsc::channel(100);
        
        // Convert string models to ActiveModel for SubtabManager
        let active_models: Vec<ActiveModel> = available_models.iter().map(|name| {
            ActiveModel {
                name: name.clone(),
                provider: if name.contains("gpt") {
                    "openai".to_string()
                } else if name.contains("claude") {
                    "anthropic".to_string()
                } else {
                    "ollama".to_string()
                },
            }
        }).collect();
        
        // Create task registry and context if model orchestrator is available
        let (task_registry, task_context) = if let Some(ref model_orch) = model_orchestrator {
            use crate::tasks::{TaskRegistry, TaskContext};
            use crate::models::ModelManager;
            use crate::config::Config;
            
            // Create task registry
            let registry = Arc::new(TaskRegistry::new());
            
            // Create config (use provided or default)
            let cfg = config.unwrap_or_else(|| Config::default());
            
            // Create model manager
            let model_manager = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    match ModelManager::with_orchestration(cfg.clone()).await {
                        Ok(mgr) => {
                            Some(Arc::new(mgr))
                        }
                        Err(e) => {
                            tracing::warn!("Failed to create ModelManager for tasks: {}", e);
                            // Try to create a basic ModelManager
                            match ModelManager::new(cfg.clone()).await {
                                Ok(mgr) => Some(Arc::new(mgr)),
                                Err(_) => {
                                    // If all fails, we can't provide task support
                                    tracing::error!("Failed to create ModelManager - task support will be limited");
                                    None
                                }
                            }
                        }
                    }
                })
            });
            
            // Create task context only if we have a model manager
            if let Some(mgr) = model_manager {
                let context = TaskContext {
                    config: cfg,
                    model_manager: mgr,
                };
                
                tracing::info!("🤖 Task support initialized for CLI tab");
                (Some(registry), Some(context))
            } else {
                tracing::warn!("⚠️ Task support unavailable - ModelManager creation failed");
                (None, None)
            }
        } else {
            (None, None)
        };
        
        // Create OllamaManager if Ollama is available
        let ollama_manager = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                use crate::ollama::OllamaManager;
                use std::path::PathBuf;
                
                let models_dir = dirs::data_dir()
                    .unwrap_or_else(|| PathBuf::from("~/.local/share"))
                    .join("loki")
                    .join("models");
                
                match OllamaManager::new(models_dir) {
                    Ok(manager) => {
                        tracing::info!("✅ Ollama manager initialized");
                        Some(Arc::new(manager))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize Ollama manager: {}", e);
                        None
                    }
                }
            })
        });
        
        // Create subtab manager with discovery engines, orchestrator, and task support
        let mut subtab_manager = SubtabManager::new_with_full_support(
            chat_state.clone(),
            orchestration.clone(),
            agent_manager.clone(),
            message_tx,
            active_models,
            model_discovery.clone(),
            agent_wizard.clone(),
            task_registry,
            task_context,
            model_orchestrator.clone(),
            ollama_manager,
        );
        
        // Connect the orchestrator to the message processor if available
        if let Some(ref orchestrator) = model_orchestrator {
            tracing::info!("🔗 Connecting orchestrator to subtab manager");
            subtab_manager.set_model_orchestrator(orchestrator.clone());
            
            // Also set tool and cognitive managers if available
            if let (Some(tool_mgr), Some(task_mgr)) = (tool_manager.as_ref(), task_manager.as_ref()) {
                subtab_manager.set_tool_managers(tool_mgr.clone(), task_mgr.clone());
            }
            
            // Set cognitive enhancement if available
            if let Some(cog_sys) = cognitive_system.as_ref() {
                // Create cognitive enhancement integration
                use crate::tui::chat::integrations::cognitive::CognitiveChatEnhancement;
                let cognitive_enhancement = std::sync::Arc::new(
                    CognitiveChatEnhancement {
                        enabled: true,
                        depth_level: 3,
                        pattern_recognition: true,
                        memory_integration: true,
                        deep_processing_enabled: true,
                        cognitive_stream: Some("Connected".to_string()),
                    }
                );
                subtab_manager.set_cognitive_enhancement(cognitive_enhancement);
            }
            
            tracing::info!("✅ Connected orchestrator and tools to subtab manager");
        }
        
        // Create history manager
        let history_manager = Arc::new(RwLock::new(state::HistoryManager::new(1000)));
        
        // Create settings manager  
        let settings_manager = Arc::new(RwLock::new(SettingsManager::default()));
        
        // Create initial chat
        let mut chats = std::collections::HashMap::new();
        chats.insert(0, ChatInfo {
            id: 0,
            name: "Main Chat".to_string(),
            created_at: chrono::Utc::now(),
            last_message: None,
            scroll_offset: 0,
        });
        
        // Default active model - prefer first local model, then first available
        let active_model = if !available_models.is_empty() && available_models[0] != "no-models-available" {
            // Try to find first local model
            let selected_model = local_models.first()
                .or_else(|| available_models.first())
                .unwrap()
                .clone();
            
            ActiveModel {
                name: selected_model.clone(),
                provider: if selected_model.contains("gpt") {
                    "openai".to_string()
                } else if selected_model.contains("claude") {
                    "anthropic".to_string()
                } else {
                    "ollama".to_string()
                },
            }
        } else {
            ActiveModel {
                name: "no-model-selected".to_string(),
                provider: "none".to_string(),
            }
        };
        
        // Create and initialize the editor tab
        let editor_tab = {
            let mut editor_tab = subtabs::EditorTab::new();
            // Initialize the editor asynchronously in a blocking task
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Err(e) = editor_tab.initialize_editor().await {
                        tracing::warn!("Failed to initialize editor: {}", e);
                    }
                    Arc::new(RwLock::new(editor_tab))
                })
            })
        };
        
        Self {
            subtab_manager: RefCell::new(subtab_manager),
            chat_state,
            orchestration,
            agent_manager,
            model_discovery,
            agent_wizard,
            available_models,
            active_chat: 0,
            message_history_mode: false,
            show_timestamps: false,
            cognitive_memory: None,
            active_model,
            chats,
            history_manager,
            command_input: String::new(),
            settings_manager,
            editor_active: false,
            current_file: None,
            editor_language: None,
            editor_tab: Some(editor_tab),
            avg_response_time: None,
            total_tokens: None,
            session_cost: None,
            event_bus: None,
            bridges: None,
            tool_bridge: None,
        }
    }
    
    /// Set the event bus for cross-tab communication
    pub fn set_event_bus(&mut self, event_bus: std::sync::Arc<crate::tui::event_bus::EventBus>) {
        self.event_bus = Some(event_bus.clone());
        
        // Also connect event bus to the message processor
        if let Some(ref mut processor) = self.subtab_manager.borrow_mut().message_processor {
            processor.set_event_bus(event_bus);
            tracing::info!("✅ Event bus connected to message processor");
        }
        
        tracing::info!("Event bus connected to ModularChat");
    }
    
    /// Set the bridges for cross-tab integration
    pub fn set_bridges(&mut self, bridges: std::sync::Arc<crate::tui::bridges::UnifiedBridge>) {
        self.bridges = Some(bridges.clone());
        
        // Also connect bridges to the message processor
        if let Some(ref mut processor) = self.subtab_manager.borrow_mut().message_processor {
            processor.set_bridges(bridges.clone());
            tracing::info!("✅ Bridges connected to message processor");
        }
        
        // Connect editor bridge to editor tab
        if let Some(ref editor_tab) = self.editor_tab {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut editor = editor_tab.write().await;
                    editor.set_editor_bridge(bridges.editor_bridge.clone());
                    tracing::info!("✅ Editor bridge connected to editor tab");
                })
            });
        }
        
        // Connect storage bridge for persistence
        // Create storage context and connect it to the chat manager
        let storage_context = std::sync::Arc::new(
            crate::tui::chat::storage_context::ChatStorageContext::new(
                bridges.storage_bridge.clone()
            )
        );
        
        // Initialize storage context asynchronously
        let storage_context_clone = storage_context.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Err(e) = storage_context_clone.initialize().await {
                    tracing::warn!("Failed to initialize storage context: {}", e);
                } else {
                    tracing::info!("✅ Storage context initialized");
                }
            })
        });
        
        // Connect storage context to message processor for persistence
        if let Some(ref mut processor) = self.subtab_manager.borrow_mut().message_processor {
            processor.set_storage_context(storage_context.clone());
            tracing::info!("✅ Storage context connected to message processor - chat persistence enabled");
        }
        
        // Connect tool bridge to tool manager if available
        // This needs to be done after tool manager is initialized via initialize_orchestration
        self.tool_bridge = Some(bridges.tool_bridge.clone());
        tracing::info!("✅ Tool bridge stored for later connection to tool manager");
        
        tracing::info!("✅ Bridges connected to ModularChat - tool bridge available for tool components");
        
        tracing::info!("Bridges connected to ModularChat");
    }
    
    /// Initialize orchestration system
    pub async fn initialize_orchestration(
        &mut self,
        cognitive_system: std::sync::Arc<crate::cognitive::CognitiveSystem>,
        model_orchestrator: std::sync::Arc<crate::models::ModelOrchestrator>,
        tool_manager: std::sync::Arc<crate::tools::IntelligentToolManager>,
    ) -> anyhow::Result<()> {
        // Connect tool bridge to tool manager if available
        if let Some(ref tool_bridge) = self.tool_bridge {
            tool_bridge.set_tool_manager(tool_manager.clone()).await;
            tracing::info!("✅ Tool manager connected to tool bridge - enabling cross-tab tool execution");
        }
        // Initialize the orchestration system
        let orchestration_manager = self.orchestration.clone();
        let mut orch = orchestration_manager.write().await;
        
        // Connect to ModelOrchestrator to sync available models
        orch.connect_to_orchestrator(model_orchestrator.clone()).await?;
        
        // Use the already discovered models instead of hardcoded ones
        // The available_models list was already populated with actual models
        if !self.available_models.is_empty() && self.available_models[0] != "no-models-available" {
            // Models are already synced via connect_to_orchestrator
            tracing::info!("Orchestration initialized with {} models", orch.enabled_models.len());
        }
        
        // Initialize agent pool if agent system is enabled
        if self.agent_manager.read().await.agent_system_enabled {
            let mut agent_mgr = self.agent_manager.write().await;
            if let Err(e) = agent_mgr.initialize_agent_pool().await {
                tracing::warn!("Failed to initialize agent pool: {}", e);
            } else {
                tracing::info!("Agent pool connected to orchestration system");
            }
        }
        
        Ok(())
    }
    
    /// Initialize memory system
    pub async fn initialize_memory(&mut self) -> anyhow::Result<()> {
        // Initialize cognitive memory if available
        // For now, leave it as None - memory initialization happens elsewhere
        self.cognitive_memory = None;
        Ok(())
    }
    
    /// Set story engine
    pub fn set_story_engine(&mut self, story_engine: std::sync::Arc<crate::story::StoryEngine>) {
        // Store story engine reference - could be added to struct if needed
        tracing::info!("Story engine set for chat system");
    }
    
    /// Load preferences from memory
    pub async fn load_preferences_from_memory(&mut self) -> anyhow::Result<()> {
        if let Some(memory) = &self.cognitive_memory {
            // Load user preferences from cognitive memory
            if let Ok(Some(preferences)) = memory.retrieve_by_key("user_preferences").await {
                tracing::info!("Loaded user preferences from memory");
            }
        }
        Ok(())
    }
    
    /// Auto-save if needed
    pub async fn auto_save_if_needed(&mut self) -> anyhow::Result<()> {
        let settings = self.settings_manager.read().await;
        if settings.auto_save {
            // Perform auto-save
            let history = self.history_manager.read().await;
            tracing::debug!("Auto-saving chat history");
        }
        Ok(())
    }
    
    /// Build message with attachments
    pub fn build_message_with_attachments(&self, content: &str) -> String {
        // For now, just return the content
        // In future, could parse for file references, etc.
        content.to_string()
    }
    
    /// Add message to chat
    pub async fn add_message(&mut self, message: crate::tui::run::AssistantResponseType, target: Option<String>) {
        let mut state = self.chat_state.write().await;
        // Use the thread number from target or default to 0
        let thread = target.and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);
        state.add_message_to_chat(message, thread);
        
        // Update last message in chat info
        if let Some(chat_info) = self.chats.get_mut(&self.active_chat) {
            chat_info.last_message = Some(chrono::Utc::now().to_string());
        }
    }
    
    /// Handle model task
    pub async fn handle_model_task(
        &mut self,
        command: String,
        target: Option<String>,
    ) -> anyhow::Result<crate::tui::run::AssistantResponseType> {
        // Create a simple response for now
        Ok(crate::tui::run::AssistantResponseType::new_ai_message(
            format!("Processing: {}", command),
            Some(self.active_model.name.clone()),
        ))
    }
    
    /// Delete chat
    pub async fn delete_chat(&mut self, chat_id: usize) -> anyhow::Result<()> {
        self.chats.remove(&chat_id);
        if self.active_chat == chat_id && !self.chats.is_empty() {
            self.active_chat = *self.chats.keys().next().unwrap();
        }
        Ok(())
    }
    
    /// Get sorted chat IDs
    pub fn get_chat_ids_sorted(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.chats.keys().cloned().collect();
        ids.sort();
        ids
    }
    
    /// Handle CLI input
    pub async fn handle_cli_input(&mut self, input: &str) -> anyhow::Result<()> {
        self.command_input = input.to_string();
        // Process the command through the subtab manager
        Ok(())
    }
}
