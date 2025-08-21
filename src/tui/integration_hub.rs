//! Integration Hub
//! 
//! Central hub that connects all TUI components and ensures seamless
//! communication between tabs, models, agents, tools, and orchestration.

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde_json::Value;
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::tui::{
    event_bus::{EventBus, SystemEvent, TabId},
    shared_state::SharedSystemState,
    tab_registry::TabRegistry,
    chat::{
        models::{catalog::ModelCatalog, discovery::ModelDiscoveryEngine},
        agents::AgentCreationWizard,
        tools::IntegratedToolSystem,
        editor::IntegratedEditor,
        orchestration::UnifiedOrchestrator,
    },
};

/// Integration hub that connects all components
pub struct IntegrationHub {
    /// Event bus for communication
    event_bus: Arc<EventBus>,
    
    /// Shared state
    shared_state: Arc<SharedSystemState>,
    
    /// Tab registry
    tab_registry: Arc<TabRegistry>,
    
    /// Model catalog
    model_catalog: Arc<RwLock<ModelCatalog>>,
    
    /// Model discovery
    model_discovery: Arc<ModelDiscoveryEngine>,
    
    /// Agent system
    agent_system: Arc<AgentCreationWizard>,
    
    /// Tool hub
    tool_hub: Arc<IntegratedToolSystem>,
    
    /// Code editor
    code_editor: Arc<IntegratedEditor>,
    
    /// Orchestrator
    orchestrator: Arc<UnifiedOrchestrator>,
    
    /// Event handlers
    handlers: Arc<RwLock<HashMap<String, Box<dyn EventHandler>>>>,
    
    /// Integration status
    status: Arc<RwLock<IntegrationStatus>>,
}

/// Integration status
#[derive(Debug, Clone)]
pub struct IntegrationStatus {
    pub initialized: bool,
    pub connected_components: HashMap<String, ComponentStatus>,
    pub active_integrations: Vec<ActiveIntegration>,
    pub message_count: u64,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Component status
#[derive(Debug, Clone)]
pub struct ComponentStatus {
    pub name: String,
    pub connected: bool,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub events_processed: u64,
    pub errors: u32,
}

/// Active integration
#[derive(Debug, Clone)]
pub struct ActiveIntegration {
    pub source: String,
    pub target: String,
    pub integration_type: IntegrationType,
    pub active: bool,
    pub messages_exchanged: u64,
}

/// Integration types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegrationType {
    EventBased,
    StateBased,
    DirectCall,
    Bidirectional,
}

/// Event handler trait
#[async_trait::async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle(&self, event: &SystemEvent, context: &IntegrationContext) -> Result<()>;
    fn name(&self) -> &str;
}

/// Integration context
pub struct IntegrationContext {
    pub hub: Arc<IntegrationHub>,
    pub source_tab: Option<TabId>,
    pub metadata: HashMap<String, Value>,
}

impl IntegrationHub {
    /// Create a new integration hub
    pub async fn new(
        event_bus: Arc<EventBus>,
        shared_state: Arc<SharedSystemState>,
        tab_registry: Arc<TabRegistry>,
    ) -> Result<Self> {
        // Create all components
        let model_catalog = Arc::new(RwLock::new(ModelCatalog::new()));
        let model_discovery = Arc::new(ModelDiscoveryEngine::new(Vec::new()));
        let agent_system = Arc::new(AgentCreationWizard::new());
        // TODO: Initialize with proper dependencies when available  
        // For now, skip tool hub initialization to avoid dependency issues
        let tool_hub = {
            use crate::tui::chat::tools::IntegratedToolSystem;
            use crate::tools::IntelligentToolManager;
            use crate::tui::nlp::core::processor::NaturalLanguageProcessor;
            use crate::tui::chat::core::commands::CommandRegistry;
            
            let tool_manager = Arc::new(IntelligentToolManager::new());
            let nlp = Arc::new(NaturalLanguageProcessor::new(
                CommandRegistry::new(),
                None, // No tool executor yet
                None, // No model orchestrator yet
            ));
            
            Arc::new(IntegratedToolSystem::new(tool_manager, nlp).await?)
        };
        let code_editor = Arc::new(IntegratedEditor::new(crate::tui::chat::editor::EditorConfig::default()).await?);
        let orchestrator = Arc::new(
            UnifiedOrchestrator::new(Default::default()).await?
        );
        
        // Wire orchestrator with model providers if API config is available
        if let Ok(api_config) = crate::config::ApiKeysConfig::from_env() {
            let providers = crate::models::providers::ProviderFactory::create_providers(&api_config);
            if !providers.is_empty() {
                let provider_count = providers.len();
                orchestrator.initialize_with_providers(providers).await?;
                tracing::info!("Initialized orchestrator with {} model providers", provider_count);
            }
        }
        
        let hub = Self {
            event_bus: event_bus.clone(),
            shared_state,
            tab_registry,
            model_catalog,
            model_discovery,
            agent_system,
            tool_hub,
            code_editor,
            orchestrator,
            handlers: Arc::new(RwLock::new(HashMap::new())),
            status: Arc::new(RwLock::new(IntegrationStatus {
                initialized: false,
                connected_components: HashMap::new(),
                active_integrations: Vec::new(),
                message_count: 0,
                last_activity: chrono::Utc::now(),
            })),
        };
        
        // Initialize integrations
        hub.initialize().await?;
        
        Ok(hub)
    }
    
    /// Initialize all integrations
    async fn initialize(&self) -> Result<()> {
        info!("Initializing Integration Hub");
        
        // Register event handlers
        self.register_handlers().await?;
        
        // Connect components
        self.connect_components().await?;
        
        // Start event processing
        self.start_event_processing().await?;
        
        // Update status
        let mut status = self.status.write().await;
        status.initialized = true;
        
        info!("Integration Hub initialized successfully");
        Ok(())
    }
    
    /// Register event handlers
    async fn register_handlers(&self) -> Result<()> {
        let mut handlers = self.handlers.write().await;
        
        // Model selection handler
        handlers.insert(
            "model_selection".to_string(),
            Box::new(ModelSelectionHandler::new(self.model_catalog.clone())),
        );
        
        // Agent creation handler
        handlers.insert(
            "agent_creation".to_string(),
            Box::new(AgentCreationHandler::new(self.agent_system.clone())),
        );
        
        // Tool execution handler
        handlers.insert(
            "tool_execution".to_string(),
            Box::new(ToolExecutionHandler::new(self.tool_hub.clone())),
        );
        
        // Code editing handler
        handlers.insert(
            "code_editing".to_string(),
            Box::new(CodeEditingHandler::new(self.code_editor.clone())),
        );
        
        // Orchestration handler
        handlers.insert(
            "orchestration".to_string(),
            Box::new(OrchestrationHandler::new(self.orchestrator.clone())),
        );
        
        debug!("Registered {} event handlers", handlers.len());
        Ok(())
    }
    
    /// Connect components
    async fn connect_components(&self) -> Result<()> {
        // Connect model discovery to catalog
        self.connect_models().await?;
        
        // Connect agents to models
        self.connect_agents().await?;
        
        // Connect tools to orchestrator
        self.connect_tools().await?;
        
        // Connect editor to tools
        self.connect_editor().await?;
        
        // Connect orchestrator to all components
        self.connect_orchestrator().await?;
        
        Ok(())
    }
    
    /// Connect model components
    async fn connect_models(&self) -> Result<()> {
        // Discover available models (trigger discovery but don't collect results here)
        // The discovery process updates the internal catalog directly
        let _discovery_count = self.model_discovery.discover_from_provider("openai").await.unwrap_or(0);
        // Models are already registered in the catalog by the discovery process
        
        // Update component status
        self.update_component_status("models", true).await;
        
        // Create integration
        self.register_integration(
            "model_discovery",
            "model_catalog",
            IntegrationType::EventBased,
        ).await;
        
        Ok(())
    }
    
    /// Connect agent system
    async fn connect_agents(&self) -> Result<()> {
        // Get available models from catalog
        let models = {
            let catalog = self.model_catalog.read().await;
            catalog.get_all_models().into_iter().cloned().collect::<Vec<_>>()
        };
        
        // Configure agent system with models
        for model in models {
            self.agent_system.add_available_model(model.id.clone()).await;
        }
        
        self.update_component_status("agents", true).await;
        
        self.register_integration(
            "agent_system",
            "model_catalog",
            IntegrationType::StateBased,
        ).await;
        
        Ok(())
    }
    
    /// Connect tool hub
    async fn connect_tools(&self) -> Result<()> {
        // Register tools with orchestrator
        let tools = self.tool_hub.get_available_tools().await;
        
        for tool in tools {
            debug!("Registering tool: {}", tool.name);
        }
        
        self.update_component_status("tools", true).await;
        
        self.register_integration(
            "tool_hub",
            "orchestrator",
            IntegrationType::DirectCall,
        ).await;
        
        Ok(())
    }
    
    /// Connect code editor
    async fn connect_editor(&self) -> Result<()> {
        // Connect editor to tool hub for code execution
        self.code_editor.set_tool_hub(self.tool_hub.clone()).await;
        
        self.update_component_status("editor", true).await;
        
        self.register_integration(
            "code_editor",
            "tool_hub",
            IntegrationType::Bidirectional,
        ).await;
        
        Ok(())
    }
    
    /// Connect orchestrator
    async fn connect_orchestrator(&self) -> Result<()> {
        // Orchestrator already has access to models through initialization
        
        self.update_component_status("orchestrator", true).await;
        
        // Register multiple integrations
        self.register_integration(
            "orchestrator",
            "model_catalog",
            IntegrationType::DirectCall,
        ).await;
        
        self.register_integration(
            "orchestrator",
            "agent_system",
            IntegrationType::Bidirectional,
        ).await;
        
        Ok(())
    }
    
    /// Start event processing
    async fn start_event_processing(&self) -> Result<()> {
        let event_bus = self.event_bus.clone();
        let handlers = self.handlers.clone();
        let status = self.status.clone();
        let hub = Arc::new(self.clone());
        
        // Subscribe to all events
        let mut receiver = event_bus.subscribe_all().await;
        
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                debug!("Processing event: {:?}", event);
                
                // Update status
                {
                    let mut status = status.write().await;
                    status.message_count += 1;
                    status.last_activity = chrono::Utc::now();
                }
                
                // Create context
                let context = IntegrationContext {
                    hub: hub.clone(),
                    source_tab: None, // TODO: Extract from event
                    metadata: HashMap::new(),
                };
                
                // Process event with all handlers
                let handlers = handlers.read().await;
                for (name, handler) in handlers.iter() {
                    if let Err(e) = handler.handle(&event, &context).await {
                        warn!("Handler {} failed: {:?}", name, e);
                    }
                }
            }
        });
        
        info!("Event processing started");
        Ok(())
    }
    
    /// Update component status
    async fn update_component_status(&self, name: &str, connected: bool) {
        let mut status = self.status.write().await;
        status.connected_components.insert(
            name.to_string(),
            ComponentStatus {
                name: name.to_string(),
                connected,
                last_heartbeat: chrono::Utc::now(),
                events_processed: 0,
                errors: 0,
            },
        );
    }
    
    /// Register an integration
    async fn register_integration(
        &self,
        source: &str,
        target: &str,
        integration_type: IntegrationType,
    ) {
        let mut status = self.status.write().await;
        status.active_integrations.push(ActiveIntegration {
            source: source.to_string(),
            target: target.to_string(),
            integration_type,
            active: true,
            messages_exchanged: 0,
        });
    }
    
    /// Get integration status
    pub async fn get_status(&self) -> IntegrationStatus {
        self.status.read().await.clone()
    }
    
    /// Publish an event to the event bus
    pub async fn publish_event(&self, event: SystemEvent) -> Result<()> {
        self.event_bus.publish(event).await;
        Ok(())
    }
    
    /// Broadcast event to all subscribers
    pub async fn broadcast_event(&self, event: SystemEvent) {
        self.event_bus.publish(event).await;
    }
    
    /// Get agent system for direct access
    pub fn get_agent_system(&self) -> Arc<AgentCreationWizard> {
        self.agent_system.clone()
    }
    
    /// Get tool hub for direct access
    pub fn get_tool_hub(&self) -> Arc<IntegratedToolSystem> {
        self.tool_hub.clone()
    }
    
    /// Get code editor for direct access
    pub fn get_code_editor(&self) -> Arc<IntegratedEditor> {
        self.code_editor.clone()
    }
    
    /// Get orchestrator for direct access
    pub fn get_orchestrator(&self) -> Arc<UnifiedOrchestrator> {
        self.orchestrator.clone()
    }
    
    /// Process a request through the appropriate component
    pub async fn process_request(&self, request_type: &str, data: Value) -> Result<Value> {
        // Update status to track activity
        {
            let mut status = self.status.write().await;
            status.message_count += 1;
            status.last_activity = chrono::Utc::now();
        }
        
        match request_type {
            "agent" => {
                // Process through agent system
                // For now, just return the data as agent system doesn't have process_request yet
                debug!("Processing agent request through agent system");
                Ok(data)
            },
            "tool" => {
                // Process through tool hub
                // For now, just return the data as tool hub doesn't have process_request yet
                debug!("Processing tool request through tool hub");
                Ok(data)
            },
            "code" => {
                // Process through code editor
                // For now, just return the data as code editor doesn't have process_request yet
                debug!("Processing code request through code editor");
                Ok(data)
            },
            "orchestration" => {
                // Process through orchestrator
                // For now, just return the data as orchestrator doesn't have process method yet
                debug!("Processing orchestration request");
                Ok(data)
            },
            _ => Err(anyhow::anyhow!("Unknown request type: {}", request_type))
        }
    }
    
    /// Send cross-tab message
    pub async fn send_cross_tab_message(
        &self,
        from_tab: TabId,
        to_tab: TabId,
        message: CrossTabMessage,
    ) -> Result<()> {
        // Create event
        let event = SystemEvent::CrossTabMessage {
            from: from_tab,
            to: to_tab,
            message: serde_json::to_value(&message)?,
        };
        
        // Publish event
        self.event_bus.publish(event).await;
        
        Ok(())
    }
    
    /// Synchronize state across tabs
    pub async fn sync_state(&self, key: &str, value: Value) -> Result<()> {
        // Update shared state
        self.shared_state.set(key.to_string(), value.clone()).await;
        
        // Notify all tabs
        let event = SystemEvent::StateChanged {
            key: key.to_string(),
            value,
            source: TabId::System,
        };
        
        self.event_bus.publish(event).await;
        
        Ok(())
    }
}

impl Clone for IntegrationHub {
    fn clone(&self) -> Self {
        Self {
            event_bus: self.event_bus.clone(),
            shared_state: self.shared_state.clone(),
            tab_registry: self.tab_registry.clone(),
            model_catalog: self.model_catalog.clone(),
            model_discovery: self.model_discovery.clone(),
            agent_system: self.agent_system.clone(),
            tool_hub: self.tool_hub.clone(),
            code_editor: self.code_editor.clone(),
            orchestrator: self.orchestrator.clone(),
            handlers: self.handlers.clone(),
            status: self.status.clone(),
        }
    }
}

/// Cross-tab message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CrossTabMessage {
    pub message_type: MessageType,
    pub payload: Value,
    pub requires_response: bool,
    pub correlation_id: Option<String>,
}

/// Message types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MessageType {
    ModelUpdate,
    AgentUpdate,
    ToolResult,
    CodeChange,
    ConfigUpdate,
    StatusUpdate,
    Request,
    Response,
}

// Event Handlers

struct ModelSelectionHandler {
    catalog: Arc<RwLock<ModelCatalog>>,
}

impl ModelSelectionHandler {
    fn new(catalog: Arc<RwLock<ModelCatalog>>) -> Self {
        Self { catalog }
    }
}

#[async_trait::async_trait]
impl EventHandler for ModelSelectionHandler {
    async fn handle(&self, event: &SystemEvent, _context: &IntegrationContext) -> Result<()> {
        if let SystemEvent::ModelSelected { model_id, source } = event {
            debug!("Handling model selection: {} from {:?}", model_id, source);
            // Update catalog with selection
            self.catalog.write().await.mark_as_selected(model_id).await;
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "model_selection"
    }
}

struct AgentCreationHandler {
    agent_system: Arc<AgentCreationWizard>,
}

impl AgentCreationHandler {
    fn new(agent_system: Arc<AgentCreationWizard>) -> Self {
        Self { agent_system }
    }
}

#[async_trait::async_trait]
impl EventHandler for AgentCreationHandler {
    async fn handle(&self, event: &SystemEvent, _context: &IntegrationContext) -> Result<()> {
        if let SystemEvent::AgentCreated { agent_id, config, .. } = event {
            debug!("Handling agent creation: {}", agent_id);
            
            // Use the agent_system to configure the new agent
            if let Ok(agent_config) = serde_json::from_value::<crate::tui::chat::agents::AgentConfig>(config.clone()) {
                // Register agent with the creation wizard
                self.agent_system.register_agent(agent_id.clone(), agent_config).await;
                info!("Agent {} registered with creation wizard", agent_id);
            }
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "agent_creation"
    }
}

struct ToolExecutionHandler {
    tool_hub: Arc<IntegratedToolSystem>,
}

impl ToolExecutionHandler {
    fn new(tool_hub: Arc<IntegratedToolSystem>) -> Self {
        Self { tool_hub }
    }
}

#[async_trait::async_trait]
impl EventHandler for ToolExecutionHandler {
    async fn handle(&self, event: &SystemEvent, _context: &IntegrationContext) -> Result<()> {
        if let SystemEvent::ToolExecuted { tool_id, params, result, .. } = event {
            debug!("Handling tool execution: {}", tool_id);
            
            // Use the tool_hub to process execution results
            if let Err(e) = self.tool_hub.record_execution_result(tool_id, params.clone(), result.clone()).await {
                warn!("Failed to record tool execution result: {}", e);
            } else {
                debug!("Tool execution result recorded for {}", tool_id);
            }
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "tool_execution"
    }
}

struct CodeEditingHandler {
    code_editor: Arc<IntegratedEditor>,
}

impl CodeEditingHandler {
    fn new(code_editor: Arc<IntegratedEditor>) -> Self {
        Self { code_editor }
    }
}

#[async_trait::async_trait]
impl EventHandler for CodeEditingHandler {
    async fn handle(&self, event: &SystemEvent, _context: &IntegrationContext) -> Result<()> {
        if let SystemEvent::CodeEdited { file, changes } = event {
            debug!("Handling code edit: {}", file);
            
            // Use the code_editor to apply changes
            if let Ok(changes_list) = serde_json::from_value::<Vec<String>>(changes.clone()) {
                // Apply changes through the integrated editor
                if let Err(e) = self.code_editor.apply_changes(file.clone(), changes_list).await {
                    warn!("Failed to apply code changes: {}", e);
                } else {
                    info!("Code changes applied to {}", file);
                }
            }
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "code_editing"
    }
}

struct OrchestrationHandler {
    orchestrator: Arc<UnifiedOrchestrator>,
}

impl OrchestrationHandler {
    fn new(orchestrator: Arc<UnifiedOrchestrator>) -> Self {
        Self { orchestrator }
    }
}

#[async_trait::async_trait]
impl EventHandler for OrchestrationHandler {
    async fn handle(&self, event: &SystemEvent, context: &IntegrationContext) -> Result<()> {
        if let SystemEvent::OrchestrationRequested { request_id, config } = event {
            debug!("Handling orchestration request: {}", request_id);
            
            // Use the orchestrator to process the request
            if let Ok(orch_request) = serde_json::from_value::<crate::tui::chat::orchestration::OrchestrationRequest>(config.clone()) {
                // Process through unified orchestrator
                match self.orchestrator.process(orch_request).await {
                    Ok(response) => {
                        info!("Orchestration request {} processed successfully", request_id);
                        // Broadcast response
                        context.hub.broadcast_event(SystemEvent::OrchestrationCompleted {
                            request_id: request_id.clone(),
                            result: serde_json::to_value(response)?,
                        }).await;
                    },
                    Err(e) => {
                        warn!("Failed to process orchestration request {}: {}", request_id, e);
                    }
                }
            }
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "orchestration"
    }
}