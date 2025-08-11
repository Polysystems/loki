//! Chat system initialization

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use anyhow::{Result, bail, Context};
use uuid::Uuid;

use crate::cognitive::{CognitiveSystem, CognitiveOrchestrator};
use crate::models::orchestrator::ModelOrchestrator;
use crate::tools::intelligent_manager::IntelligentToolManager;
use crate::tools::task_management::TaskManager;
use crate::tui::cognitive_stream_integration::ChatCognitiveStream;
use crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator;
use crate::tui::run::AssistantResponseType;

use crate::tui::chat::state::ChatState;
use crate::tui::chat::orchestration::OrchestrationManager;
use crate::tui::chat::manager::ChatManager;
use crate::tui::chat::core::tool_executor::ChatToolExecutor;
use crate::tui::chat::integrations::{
    cognitive::{self, CognitiveIntegration},
    tools::ToolIntegration,
    nlp::NlpIntegration,
};
use super::model_registry::ModelRegistry;

/// Chat system initialization configuration
pub struct ChatConfig {
    /// Enable orchestration by default
    pub enable_orchestration: bool,
    
    /// Default model
    pub default_model: String,
    
    /// Message channel buffer size
    pub channel_buffer_size: usize,
    
    /// Enable streaming responses
    pub enable_streaming: bool,
    
    /// Context window size
    pub context_window: usize,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            enable_orchestration: true,
            default_model: String::new(), // Empty default, will be set based on available models
            channel_buffer_size: 100,
            enable_streaming: true,
            context_window: 8192,
        }
    }
}

/// Initialize the chat system
pub async fn initialize_chat_system(
    config: ChatConfig,
    cognitive_system: Arc<CognitiveSystem>,
    consciousness: Arc<CognitiveOrchestrator>,
    model_orchestrator: Arc<ModelOrchestrator>,
    tool_manager: Arc<IntelligentToolManager>,
    task_manager: Arc<TaskManager>,
) -> Result<ChatComponents> {
    tracing::info!("🚀 Initializing chat system");
    
    // Create channels
    let (message_tx, message_rx) = mpsc::channel::<(String, usize)>(config.channel_buffer_size);
    let (response_tx, response_rx) = mpsc::channel::<AssistantResponseType>(config.channel_buffer_size);
    
    // Create separate response channel for the return value
    let (return_response_tx, return_response_rx) = mpsc::channel::<AssistantResponseType>(config.channel_buffer_size);
    
    // Clone the response sender for background processors
    let bg_response_tx = return_response_tx.clone();
    
    // Initialize state
    let state = Arc::new(RwLock::new(ChatState::new(0, "Main Chat".to_string())));
    
    // Initialize model registry
    let model_registry = ModelRegistry::with_orchestrator(model_orchestrator.clone());
    model_registry.initialize().await
        .context("Failed to initialize model registry")?;
    
    // Verify default model exists
    let default_model = if model_registry.is_model_available(&config.default_model).await {
        config.default_model.clone()
    } else {
        // Fall back to first available model
        match model_registry.get_default_model().await {
            Some(model) => {
                tracing::warn!(
                    "Default model '{}' not available, using '{}' instead",
                    config.default_model,
                    model
                );
                model
            }
            None => {
                bail!("No models available. Please check API keys and configuration.");
            }
        }
    };
    
    // Initialize orchestration manager
    let orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
    {
        let mut orch = orchestration.write().await;
        orch.orchestration_enabled = config.enable_orchestration;
        orch.stream_responses = config.enable_streaming;
        orch.context_window = config.context_window;
        
        // Set verified default model
        orch.enabled_models.push(default_model.clone());
        
        // Add all available models to orchestration
        let available_models = model_registry.list_available_models().await;
        for model in available_models {
            if !orch.enabled_models.contains(&model.id) {
                orch.enabled_models.push(model.id);
            }
        }
        
        tracing::info!(
            "📦 Orchestration initialized with {} models",
            orch.enabled_models.len()
        );
    }
    
    // Initialize integrations
    let cognitive_integration = Arc::new(
        CognitiveIntegration::with_cognitive(
            consciousness.clone(),
            cognitive_system.clone(),
        )
    );
    
    let tool_integration = Arc::new(
        ToolIntegration::new(tool_manager.clone(), task_manager.clone())
    );
    
    // Initialize ChatToolExecutor for command processing
    let mcp_client = get_mcp_client();
    let tool_executor = Arc::new(ChatToolExecutor::new(
        Some(tool_manager.clone()),
        mcp_client.clone(),
        Some(task_manager.clone()),
        Some(model_orchestrator.clone()),
    ));
    tracing::info!("🔧 ChatToolExecutor initialized with MCP support: {}", mcp_client.is_some());
    
    // Create NLP integration - with orchestrator if all dependencies are available
    let nlp_integration = if let (Some(mcp_client), Some(safety_validator)) = 
        (get_mcp_client(), get_safety_validator()) {
        
        // Try to create with basic integration
        // Note: MultiAgentOrchestrator requires ApiKeysConfig which we don't have here
        // For now, create basic NLP integration without multi-agent support
        let integration = NlpIntegration::new();
        tracing::info!("✅ NLP integration created");
        Arc::new(integration)
    } else {
        tracing::info!("💡 Using basic NLP integration (orchestrator dependencies not available)");
        Arc::new(NlpIntegration::new())
    };
    
    // Create chat manager with proper initialization
    let mut chat_manager = ChatManager::new(
        state.clone(),
        orchestration.clone(),
        cognitive_integration.clone(),
        tool_integration.clone(),
        nlp_integration.clone(),
        message_tx.clone(),
        response_rx,
    ).await?;
    
    // Wire the tool executor to the message processor
    chat_manager.set_tool_executor(tool_executor.clone());
    
    // Start background processors
    start_background_processors(
        cognitive_integration.clone(),
        bg_response_tx,
    ).await?;
    
    tracing::info!("✅ Chat system initialized successfully");
    
    // Return components
    Ok(ChatComponents {
        state: state.clone(),
        orchestration: orchestration.clone(),
        chat_manager,
        message_tx,
        response_rx: return_response_rx,
        cognitive_integration,
        tool_integration,
        nlp_integration,
        nlp_orchestrator: None, // Set when full orchestrator is available
        tool_executor,
        model_registry: Arc::new(model_registry),
    })
}

/// Start background processors
async fn start_background_processors(
    cognitive_integration: Arc<CognitiveIntegration>,
    response_tx: mpsc::Sender<AssistantResponseType>,
) -> Result<()> {
    // Subscribe to cognitive events
    let mut event_rx = cognitive_integration.subscribe();
    let tx_clone = response_tx.clone();
    
    // Start cognitive event processor
    tokio::spawn(async move {
        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    match event {
                        cognitive::CognitiveEvent::NewInsight(insight) => {
                            let message = AssistantResponseType::Message {
                                id: uuid::Uuid::new_v4().to_string(),
                                author: "cognitive".to_string(),
                                message: format!(
                                    "💭 {} insight: {}",
                                    match insight.insight_type {
                                        cognitive::InsightType::Pattern => "Pattern",
                                        cognitive::InsightType::Anomaly => "Anomaly",
                                        cognitive::InsightType::Suggestion => "Suggestion",
                                        cognitive::InsightType::Observation => "Observation",
                                        cognitive::InsightType::Reflection => "Reflection",
                                    },
                                    insight.content
                                ),
                                timestamp: chrono::Utc::now().to_rfc3339(),
                                is_editing: false,
                                edit_history: Vec::new(),
                                streaming_state: crate::tui::run::StreamingState::Complete,
                                metadata: crate::tui::run::MessageMetadata::default(),
                            };
                            
                            if let Err(e) = tx_clone.send(message).await {
                                tracing::error!("Failed to send cognitive insight: {}", e);
                            }
                        }
                        cognitive::CognitiveEvent::FocusChanged(focus) => {
                            tracing::info!("Cognitive focus changed: {}", focus);
                        }
                        cognitive::CognitiveEvent::ActivityChanged(level) => {
                            tracing::debug!("Cognitive activity level: {:.2}", level);
                        }
                        cognitive::CognitiveEvent::MoodChanged(mood) => {
                            tracing::info!("Cognitive mood changed: {}", mood);
                        }
                        cognitive::CognitiveEvent::Error(error) => {
                            tracing::error!("Cognitive error: {}", error);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Error receiving cognitive event: {}", e);
                    break;
                }
            }
        }
    });
    
    // Start periodic activity reporter
    let cognitive_clone = cognitive_integration.clone();
    let tx_clone2 = response_tx.clone();
    
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            // Get consciousness activity summary
            match cognitive_clone.get_activity_summary().await {
                Ok(summary) => {
                    let message = AssistantResponseType::Message {
                        id: uuid::Uuid::new_v4().to_string(),
                        author: "system".to_string(),
                        message: summary,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        is_editing: false,
                        edit_history: Vec::new(),
                        streaming_state: crate::tui::run::StreamingState::Complete,
                        metadata: crate::tui::run::MessageMetadata::default(),
                    };
                    
                    if let Err(e) = tx_clone2.send(message).await {
                        tracing::error!("Failed to send consciousness activity: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to get consciousness activity: {}", e);
                }
            }
        }
    });
    
    Ok(())
}

/// Components created during initialization
pub struct ChatComponents {
    /// Chat state
    pub state: Arc<RwLock<ChatState>>,
    
    /// Orchestration manager
    pub orchestration: Arc<RwLock<OrchestrationManager>>,
    
    /// Chat manager
    pub chat_manager: ChatManager,
    
    /// Message sender channel
    pub message_tx: mpsc::Sender<(String, usize)>,
    
    /// Response receiver channel
    pub response_rx: mpsc::Receiver<AssistantResponseType>,
    
    /// Cognitive integration
    pub cognitive_integration: Arc<CognitiveIntegration>,
    
    /// Tool integration
    pub tool_integration: Arc<ToolIntegration>,
    
    /// NLP integration
    pub nlp_integration: Arc<NlpIntegration>,
    
    /// NLP orchestrator (optional)
    pub nlp_orchestrator: Option<Arc<NaturalLanguageOrchestrator>>,
    
    /// Tool executor for command processing
    pub tool_executor: Arc<ChatToolExecutor>,
    
    /// Model registry
    pub model_registry: Arc<ModelRegistry>,
}

/// Initialize a minimal chat system for testing
pub async fn initialize_minimal_chat() -> Result<ChatComponents> {
    tracing::info!("🚀 Initializing minimal chat system");
    
    let config = ChatConfig::default();
    
    // Create channels
    let (message_tx, message_rx) = mpsc::channel::<(String, usize)>(config.channel_buffer_size);
    let (response_tx, response_rx) = mpsc::channel::<AssistantResponseType>(config.channel_buffer_size);
    
    // Initialize state
    let state = Arc::new(RwLock::new(ChatState::new(0, "Test Chat".to_string())));
    
    // Initialize model registry (minimal)
    let model_registry = Arc::new(ModelRegistry::new());
    
    // Initialize orchestration manager
    let orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
    
    // Initialize tool and task managers
    let tool_manager = Arc::new(crate::tools::intelligent_manager::IntelligentToolManager::new());
    let task_manager = Arc::new(crate::tools::task_management::TaskManager::new());
    
    // Create minimal integrations
    let cognitive_integration = Arc::new(CognitiveIntegration::new());
    let tool_integration = Arc::new(ToolIntegration::new(
        tool_manager.clone(),
        task_manager.clone(),
    ));
    let nlp_integration = Arc::new(NlpIntegration::new());
    
    // Create chat manager
    let chat_manager = ChatManager::new(
        state.clone(),
        orchestration.clone(),
        cognitive_integration.clone(),
        tool_integration.clone(),
        nlp_integration.clone(),
        message_tx.clone(),
        response_rx,
    ).await?;
    
    // Create separate response channel for the return value
    let (return_response_tx, return_response_rx) = mpsc::channel::<AssistantResponseType>(config.channel_buffer_size);
    
    tracing::info!("✅ Minimal chat system initialized successfully");
    
    // Create a minimal tool executor
    let tool_executor = Arc::new(ChatToolExecutor::new(None, None, None, None));
    
    Ok(ChatComponents {
        state,
        orchestration,
        chat_manager,
        message_tx,
        response_rx: return_response_rx,
        cognitive_integration,
        tool_integration,
        nlp_integration,
        nlp_orchestrator: None,
        tool_executor,
        model_registry,
    })
}

// Helper functions for optional dependencies
fn get_mcp_client() -> Option<Arc<crate::mcp::McpClient>> {
    use crate::mcp::{McpClient, McpClientConfig, McpServer};
    use std::collections::HashMap;
    
    // Create MCP client with default config
    let mut client = McpClient::new(McpClientConfig::default());
    
    // Add filesystem MCP server if available
    let filesystem_server = McpServer {
        name: "filesystem".to_string(),
        description: "File system operations".to_string(),
        command: "npx".to_string(),
        args: vec!["-y".to_string(), "@modelcontextprotocol/server-filesystem".to_string()],
        env: HashMap::new(),
        capabilities: vec!["read_file".to_string(), "write_file".to_string(), "list_directory".to_string()],
        enabled: true,
    };
    client.add_server(filesystem_server);
    
    // Add web search MCP server if available
    let search_server = McpServer {
        name: "web-search".to_string(), 
        description: "Web search capabilities".to_string(),
        command: "npx".to_string(),
        args: vec!["-y".to_string(), "@modelcontextprotocol/server-brave-search".to_string()],
        env: HashMap::new(),
        capabilities: vec!["search".to_string()],
        enabled: true,
    };
    client.add_server(search_server);
    
    tracing::info!("🔌 MCP client initialized with {} servers", 2);
    Some(Arc::new(client))
}

fn get_safety_validator() -> Option<Arc<crate::safety::ActionValidator>> {
    use crate::safety::{ActionValidator, ValidatorConfig};
    
    // Create safety validator with default config
    let rt = tokio::runtime::Runtime::new().ok()?;
    let validator = rt.block_on(async {
        ActionValidator::new(ValidatorConfig::default()).await
    }).ok()?;
    
    tracing::info!("🛡️ Safety validator initialized");
    Some(Arc::new(validator))
}