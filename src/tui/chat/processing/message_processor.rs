//! Main message processing logic
//! 
//! This module handles the core message processing pipeline, including:
//! - Tool detection and execution
//! - Cognitive enhancement
//! - Natural language orchestration
//! - Model orchestration

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use anyhow::{Result};
use crate::tui::chat::error::ChatError;
use crate::tui::chat::utils::{with_timeout, with_timeout_retry, TimeoutConfig, DEFAULT_TIMEOUT};
use crate::tui::chat::utils::telemetry;
use uuid::Uuid;
use serde_json::json;
use regex::Regex;
// use chrono::Utc;

use crate::tui::run::AssistantResponseType;
use crate::tui::chat::state::ChatState;
use crate::tui::chat::orchestration::{
    OrchestrationManager, OrchestrationConnector,
    TodoManager, CreateTodoRequest, ModelCallTracker,
    ModelSelector,
    todo_manager::{TodoStoryContext, TodoPriority, TodoStatus, PriorityLevel},
};
use crate::tui::chat::agents::AgentManager;
use crate::tui::bridges::{
    orchestration_bridge::OrchestrationBridge,
    agent_bridge::AgentBridge,
    story_bridge::StoryBridge,
};
use crate::story::StoryContext as BridgeStoryContext;
use super::unified_streaming::{UnifiedStreamManager, StreamSource};
use crate::models::{ModelOrchestrator};
use crate::tui::chat::core::{ChatToolExecutor, CommandRegistry};
use crate::tui::chat::integrations::cognitive::CognitiveChatEnhancement;
use crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator;
use crate::tools::intelligent_manager::IntelligentToolManager;
use crate::tools::task_management::{TaskManager, TaskPriority};
use crate::tui::nlp::core::orchestrator::ExtractedTask;
use crate::tui::event_bus::{EventBus, SystemEvent, TabId};

/// Parsed task from model response
#[derive(Debug, Clone)]
struct ParsedTask {
    task_type: TaskType,
    description: String,
    path: Option<String>,
    content: Option<String>,
    language: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum TaskType {
    CreateDirectory,
    CreateFile,
    RunCommand,
    InstallDependency,
}

/// Main message processor for the chat system
#[derive(Clone)]
pub struct MessageProcessor {
    /// Chat state
    chat_state: Arc<RwLock<ChatState>>,
    
    /// Orchestration manager
    orchestration_manager: Arc<RwLock<OrchestrationManager>>,
    
    /// Agent manager
    agent_manager: Arc<RwLock<AgentManager>>,
    
    /// Model orchestrator
    model_orchestrator: Option<Arc<ModelOrchestrator>>,
    
    /// Orchestration connector
    orchestration_connector: Option<OrchestrationConnector>,
    
    /// Tool executor
    tool_executor: Option<Arc<ChatToolExecutor>>,
    
    /// Cognitive enhancement
    cognitive_enhancement: Option<Arc<CognitiveChatEnhancement>>,
    
    /// Natural language orchestrator
    nlp_orchestrator: Option<Arc<NaturalLanguageOrchestrator>>,
    
    /// Intelligent tool manager
    intelligent_tool_manager: Option<Arc<IntelligentToolManager>>,
    
    /// Task manager
    task_manager: Option<Arc<TaskManager>>,
    
    /// Tool integration for direct tool execution
    tools: Option<Arc<crate::tui::chat::integrations::ToolIntegration>>,
    
    /// Event bus for cross-tab communication
    event_bus: Option<Arc<EventBus>>,
    
    /// Bridges for cross-tab integration
    bridges: Option<Arc<crate::tui::bridges::UnifiedBridge>>,
    
    /// Todo manager for task management
    todo_manager: Option<Arc<TodoManager>>,
    
    /// Model call tracker for monitoring
    call_tracker: Option<Arc<ModelCallTracker>>,
    
    /// Orchestration bridge for settings sync
    orchestration_bridge: Option<Arc<OrchestrationBridge>>,
    
    /// Agent bridge for agent configuration
    agent_bridge: Option<Arc<AgentBridge>>,
    
    /// Story bridge for narrative integration
    story_bridge: Option<Arc<StoryBridge>>,
    
    /// Unified stream manager for multiplexed streaming
    stream_manager: Option<Arc<UnifiedStreamManager>>,
    
    /// Storage context for message persistence
    storage_context: Option<Arc<crate::tui::chat::storage_context::ChatStorageContext>>,
    
    /// Model selector for orchestrated execution
    model_selector: Option<Arc<ModelSelector>>,
    
    /// Response channel
    response_tx: mpsc::Sender<AssistantResponseType>,
}

impl MessageProcessor {
    /// Create a new message processor
    pub fn new(
        chat_state: Arc<RwLock<ChatState>>,
        orchestration_manager: Arc<RwLock<OrchestrationManager>>,
        agent_manager: Arc<RwLock<AgentManager>>,
        response_tx: mpsc::Sender<AssistantResponseType>,
    ) -> Self {
        Self {
            chat_state,
            orchestration_manager,
            agent_manager,
            model_orchestrator: None,
            orchestration_connector: None,
            tool_executor: None,
            cognitive_enhancement: None,
            nlp_orchestrator: None,
            intelligent_tool_manager: None,
            task_manager: None,
            tools: None,
            event_bus: None,
            bridges: None,
            todo_manager: None,
            call_tracker: None,
            orchestration_bridge: None,
            agent_bridge: None,
            story_bridge: None,
            stream_manager: None,
            storage_context: None,
            model_selector: None,
            response_tx,
        }
    }
    
    /// Set the event bus for cross-tab communication
    pub fn set_event_bus(&mut self, event_bus: Arc<EventBus>) {
        self.event_bus = Some(event_bus);
        tracing::info!("Event bus connected to message processor");
    }
    
    /// Set the bridges for cross-tab integration
    pub fn set_bridges(&mut self, bridges: Arc<crate::tui::bridges::UnifiedBridge>) {
        self.bridges = Some(bridges);
        tracing::info!("Bridges connected to message processor");
    }
    
    /// Set the model orchestrator
    pub fn set_model_orchestrator(&mut self, orchestrator: Arc<ModelOrchestrator>) {
        // Create orchestration connector
        let mut connector = OrchestrationConnector::new();
        connector.set_orchestrator(orchestrator.clone());
        
        // Create model selector with orchestration manager
        let selector = Arc::new(ModelSelector::new(
            self.orchestration_manager.clone(),
            orchestrator.clone(),
        ));
        self.model_selector = Some(selector);
        tracing::info!("Model selector created to bridge orchestration settings");
        
        self.model_orchestrator = Some(orchestrator);
        self.orchestration_connector = Some(connector);
    }
    
    /// Set the tool executor
    pub fn set_tool_executor(&mut self, executor: Arc<ChatToolExecutor>) {
        self.tool_executor = Some(executor);
    }
    
    /// Set the cognitive enhancement
    pub fn set_cognitive_enhancement(&mut self, enhancement: Arc<CognitiveChatEnhancement>) {
        self.cognitive_enhancement = Some(enhancement);
    }
    
    /// Set the NLP orchestrator
    pub fn set_nlp_orchestrator(&mut self, orchestrator: Arc<NaturalLanguageOrchestrator>) {
        self.nlp_orchestrator = Some(orchestrator);
    }
    
    /// Set tool and task managers
    pub fn set_tool_managers(
        &mut self,
        intelligent_tool_manager: Arc<IntelligentToolManager>,
        task_manager: Arc<TaskManager>,
    ) {
        self.intelligent_tool_manager = Some(intelligent_tool_manager);
        self.task_manager = Some(task_manager);
    }
    
    /// Set the tool integration
    pub fn set_tool_integration(&mut self, tools: Arc<crate::tui::chat::integrations::ToolIntegration>) {
        self.tools = Some(tools);
        tracing::info!("Tool integration connected to message processor");
    }
    
    /// Set the todo manager
    pub fn set_todo_manager(&mut self, todo_manager: Arc<TodoManager>) {
        self.todo_manager = Some(todo_manager);
        tracing::info!("Todo manager connected to message processor");
    }
    
    /// Set the model call tracker
    pub fn set_call_tracker(&mut self, tracker: Arc<ModelCallTracker>) {
        self.call_tracker = Some(tracker);
        tracing::info!("Model call tracker connected to message processor");
    }
    
    /// Set orchestration and agent bridges
    pub fn set_orchestration_bridges(
        &mut self,
        orchestration_bridge: Arc<OrchestrationBridge>,
        agent_bridge: Arc<AgentBridge>,
    ) {
        self.orchestration_bridge = Some(orchestration_bridge);
        self.agent_bridge = Some(agent_bridge);
        tracing::info!("Orchestration bridges connected to message processor");
    }
    
    /// Set story bridge and stream manager
    pub fn set_story_integration(
        &mut self,
        story_bridge: Arc<StoryBridge>,
        stream_manager: Arc<UnifiedStreamManager>,
    ) {
        self.story_bridge = Some(story_bridge);
        self.stream_manager = Some(stream_manager);
        tracing::info!("Story integration connected to message processor");
    }
    
    /// Set storage context for message persistence
    pub fn set_storage_context(&mut self, storage_context: Arc<crate::tui::chat::storage_context::ChatStorageContext>) {
        self.storage_context = Some(storage_context);
        tracing::info!("Storage context connected to message processor - enabling message persistence");
    }
    
    /// Process a user message through the full pipeline
    pub async fn process_message(&mut self, content: &str, chat_id: usize) -> Result<()> {
        tracing::info!("📨 MessageProcessor received message: {}", content);
        tracing::info!("🔍 Has model orchestrator: {}", self.model_orchestrator.is_some());
        
        // Track operation start time for telemetry
        let operation_start = std::time::Instant::now();
        let telemetry_collector = telemetry::telemetry();
        
        // Add user message to chat history
        {
            let mut state = self.chat_state.write().await;
            let user_message = AssistantResponseType::new_user_message(content.to_string());
            state.add_message_to_chat(user_message.clone(), chat_id);
            
            // Persist message to storage if available
            if let Some(ref storage_context) = self.storage_context {
                // Convert user message to appropriate format for storage
                let role = "user".to_string();
                if let Err(e) = storage_context.add_message(
                    role,
                    content.to_string(),
                    None, // token_count
                ).await {
                    tracing::warn!("Failed to persist user message to storage: {}", e);
                } else {
                    tracing::debug!("User message persisted to storage");
                }
            }
        }
        
        // Publish message received event
        if let Some(ref event_bus) = self.event_bus {
            let _ = event_bus.publish(SystemEvent::MessageReceived {
                message_id: uuid::Uuid::new_v4().to_string(),
                content: content.to_string(),
                source: TabId::Chat,
            }).await;
        }
        
        // 1. Extract story context if available
        let story_context = if let Some(ref story_bridge) = self.story_bridge {
            story_bridge.extract_story_context(content, &chat_id.to_string()).await.ok().flatten()
        } else {
            None
        };
        
        // 2. Check for todo queries first (e.g., "what's on my plate?")
        if let Some(query_response) = self.handle_todo_query(content).await {
            let response = AssistantResponseType::new_ai_message(query_response, None);
            self.response_tx.send(response).await?;
            return Ok(());
        }
        
        // 3. Check for todo completion (e.g., "I finished the auth refactoring")
        if let Some(completion_response) = self.detect_todo_completion(content).await {
            let response = AssistantResponseType::new_ai_message(completion_response, None);
            self.response_tx.send(response).await?;
            return Ok(());
        }
        
        // 4. Check for todo/task creation requests with story context
        if let Some(mut todo_request) = self.extract_todo_request(content).await {
            if let Some(ref todo_manager) = self.todo_manager {
                // Add story context to todo request if available
                if let Some(ref story_ctx) = story_context {
                    // Now we can properly set the story_context field
                    todo_request.story_context = Some(TodoStoryContext {
                        story_id: story_ctx.story_id.to_string(),
                        plot_point_id: story_ctx.recent_plot_points.last().map(|p| p.id.to_string()),
                        narrative: story_ctx.current_plot.clone(),
                        story_arc: None,
                        related_events: Vec::new(),
                    });
                }
                
                match todo_manager.create_todo(todo_request).await {
                    Ok(todo) => {
                        let response = AssistantResponseType::new_ai_message(
                            format!("✅ Created todo: {}\n🆔 ID: {}\n📊 Priority: {:?}\n🧠 Complexity: {:.0}%",
                                todo.title, 
                                &todo.id[..8],
                                todo.priority.level,
                                todo.cognitive_metadata.complexity_score * 100.0
                            ),
                            Some("todo-manager".to_string()),
                        );
                        self.response_tx.send(response).await?;
                        
                        // Map todo to story if story bridge is available
                        if let Some(ref story_bridge) = self.story_bridge {
                            let _ = story_bridge.map_todo_to_story(todo.id.clone(), content.to_string()).await;
                        }
                        
                        // Continue processing for additional context
                        tracing::info!("Todo created, continuing for additional processing");
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create todo: {}", e);
                    }
                }
            }
        }
        
        // 2. Check for direct tool/agent execution requests (higher priority)
        // These should be executed immediately, not sent to editor
        if self.looks_like_direct_execution(content) {
            tracing::info!("🔧 Detected direct tool/agent execution request: {}", content);
            
            // Try to execute the tool request directly
            match self.execute_tool_request(content, chat_id).await {
                Ok(Some(result)) => {
                    // Send the tool execution result
                    self.response_tx.send(result).await?;
                    return Ok(());
                }
                Ok(None) => {
                    // Tool request recognized but couldn't be executed directly
                    // Continue to model for interpretation
                    tracing::info!("Tool request needs model interpretation");
                }
                Err(e) => {
                    // Log error but continue with model processing
                    tracing::warn!("Tool execution failed: {}, falling back to model", e);
                    // Track error in telemetry
                    let telemetry = telemetry::telemetry();
                    let error = ChatError::Tool(crate::tui::chat::error::ToolError::ExecutionFailed(e.to_string()));
                    tokio::spawn(async move {
                        telemetry.record_error(&error, "message_processor::tool_execution").await;
                    });
                }
            }
        }
        
        // 2. Check for code generation requests and route to editor
        // Only if it's specifically about creating/writing new code, not executing tools
        if self.looks_like_code_generation(content) && !self.looks_like_direct_execution(content) {
            tracing::info!("📝 Detected code generation request: {}", content);
            
            // Route to editor through bridge
            if let Some(ref bridges) = self.bridges {
                let language = self.detect_language_from_request(content);
                match bridges.editor_bridge.request_code_generation(
                    chat_id.to_string(),
                    content.to_string(),
                    language,
                    None,
                ).await {
                    Ok(session_id) => {
                        // Notify user that editor is opening
                        let notification = AssistantResponseType::new_ai_message(
                            format!("🚀 Opening editor for code generation (session: {})\n\nI'll help you implement: {}", 
                                &session_id[..8], content),
                            Some("editor-bridge".to_string()),
                        );
                        self.response_tx.send(notification).await?;
                        
                        // Continue to model for actual code generation
                        tracing::info!("Editor opened, continuing to generate code...");
                    }
                    Err(e) => {
                        tracing::warn!("Failed to open editor: {}", e);
                    }
                }
            }
        }
        
        // 2. Try cognitive bridge for enhanced reasoning (only for complex requests)
        if let Some(ref bridges) = self.bridges {
            // Only use cognitive bridge for complex reasoning tasks
            let needs_reasoning = content.contains("reason") || content.contains("think") || 
                                 content.contains("analyze") || content.contains("explain") ||
                                 content.contains("why") || content.contains("how");
            
            if needs_reasoning && bridges.cognitive_bridge.is_enhancement_enabled().await {
                tracing::info!("🧠 Using cognitive bridge for complex reasoning request");
                
                // Get relevant context from memory bridge
                let context_items = bridges.memory_bridge.retrieve_context(content, 5).await
                    .unwrap_or_default();
                
                let context = json!({
                    "query": content,
                    "context_items": context_items.len(),
                    "chat_id": chat_id,
                });
                
                match bridges.cognitive_bridge.request_reasoning(content, context).await {
                    Ok(reasoning) => {
                        if let Some(chain) = reasoning.chain {
                            // Only return early if we got substantial reasoning
                            if !chain.steps.is_empty() || chain.confidence > 0.7 {
                                let mut response = if !chain.steps.is_empty() {
                                    format!("💭 Reasoning: {}\n\n", chain.steps.last().map(|s| s.conclusion.as_str()).unwrap_or("completed"))
                                } else {
                                    format!("💭 Reasoning completed (confidence: {:.0}%)\n\n", chain.confidence * 100.0)
                                };
                                
                                // Add suggestions but continue to model for full response
                                if !reasoning.suggestions.is_empty() {
                                    response.push_str("📝 Suggestions:\n");
                                    for suggestion in &reasoning.suggestions {
                                        response.push_str(&format!("• {}\n", suggestion));
                                    }
                                    response.push_str("\n");
                                }
                                
                                // Only send reasoning if it's meaningful content
                                if !response.trim().is_empty() && response.len() > 50 {
                                    let reasoning_msg = AssistantResponseType::new_ai_message(
                                        response,
                                        Some("loki-cognitive-bridge".to_string()),
                                    );
                                    self.response_tx.send(reasoning_msg).await?;
                                }
                                
                                // Continue to model orchestration for actual response
                                tracing::info!("Continuing to model orchestration after cognitive bridge");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Cognitive bridge reasoning failed: {}", e);
                        // Track error in telemetry
                        let telemetry = telemetry::telemetry();
                        let error = ChatError::Internal(format!("Cognitive bridge error: {}", e));
                        tokio::spawn(async move {
                            telemetry.record_error(&error, "message_processor::cognitive_bridge").await;
                        });
                    }
                }
            } else {
                tracing::debug!("Skipping cognitive bridge - not a reasoning request or bridge disabled");
            }
        }
        
        // 3. Process through Cognitive Enhancement if available (fallback)
        // Note: Only use this if explicitly requested or if no model orchestrator is available
        if self.model_orchestrator.is_none() {
            if let Some(ref cognitive_enhancement) = self.cognitive_enhancement {
                tracing::info!("🧠 Using cognitive enhancement as fallback (no model orchestrator)");
                
                // Add timeout for cognitive enhancement
                let response = match with_timeout(
                    DEFAULT_TIMEOUT,
                    async { Ok::<_, ChatError>(cognitive_enhancement.process_message(content).await) }
                ).await {
                    Ok(resp) => resp,
                    Err(e) => {
                        tracing::warn!("Cognitive enhancement timed out or failed: {}", e);
                        // Track error in telemetry
                        let telemetry = telemetry::telemetry();
                        tokio::spawn(async move {
                            telemetry.record_error(&e, "message_processor::cognitive_enhancement").await;
                        });
                        return Ok(());
                    }
                };
                
                if !response.content.is_empty() {
                    let ai_message = AssistantResponseType::new_ai_message(
                        response.content,
                        Some("loki-cognitive".to_string()),
                    );
                    self.response_tx.send(ai_message).await?;
                    return Ok(());
                }
            }
        }
        
        // 3. Process through Natural Language Orchestrator if available
        if let Some(ref nlp_orchestrator) = self.nlp_orchestrator {
            tracing::info!("📝 Processing through NLP orchestrator");
            
            match nlp_orchestrator.process_input(&chat_id.to_string(), content).await {
                Ok(response) => {
                    if !response.primary_response.is_empty() {
                        let ai_message = AssistantResponseType::new_ai_message(
                            response.primary_response,
                            Some("loki-nlp".to_string()),
                        );
                        self.response_tx.send(ai_message).await?;
                        
                        // Handle extracted tasks
                        if !response.extracted_tasks.is_empty() {
                            self.handle_extracted_tasks(response.extracted_tasks).await?;
                        }
                        
                        return Ok(());
                    }
                }
                Err(e) => {
                    tracing::warn!("NLP orchestration failed: {}", e);
                    // Track error in telemetry
                    let telemetry = telemetry::telemetry();
                    let error = ChatError::Nlp(crate::tui::chat::error::NlpError::OrchestratorUnavailable);
                    tokio::spawn(async move {
                        telemetry.record_error(&error, "message_processor::nlp_orchestration").await;
                    });
                    // Fall through to basic model
                }
            }
        }
        
        // 4. Check orchestration configuration for multi-model execution
        if let Some(ref orchestration_bridge) = self.orchestration_bridge {
            if orchestration_bridge.is_enabled().await {
                let config = orchestration_bridge.get_configuration().await;
                let parallel_count = config.parallel_models;
                
                if parallel_count > 1 {
                    tracing::info!("🎯 Using orchestration with {} parallel models", parallel_count);
                    
                    // Start tracking for orchestrated execution
                    if let Some(ref tracker) = self.call_tracker {
                        let session_id = tracker.start_tracking(
                            "orchestrated-chat",
                            content,
                            content.len(),
                            0.7,
                            Some(500),
                        ).await.ok();
                        
                        // Execute with orchestration
                        if let Some(ref orchestrator) = self.model_orchestrator {
                            // Create orchestrated request
                            let task_request = self.create_orchestrated_request(content, &config).await?;
                            
                            match orchestrator.execute_with_fallback(task_request).await {
                                Ok(response) => {
                                    // Complete tracking
                                    if let Some(sid) = session_id {
                                        let _ = tracker.complete_tracking(&sid, response.content.len()).await;
                                    }
                                    
                                    let ai_message = AssistantResponseType::new_ai_message(
                                        response.content,
                                        Some(format!("orchestrated-{}", response.model_used.model_id())),
                                    );
                                    self.response_tx.send(ai_message).await?;
                                    return Ok(());
                                }
                                Err(e) => {
                                    tracing::warn!("Orchestrated execution failed: {}", e);
                                    // Fall through to single model
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 5. Check agent configuration for collaborative execution
        if let Some(ref agent_bridge) = self.agent_bridge {
            if agent_bridge.is_enabled().await {
                let mode = agent_bridge.get_collaboration_mode().await;
                let active_configs = agent_bridge.get_active_configs().await;
                
                if !active_configs.is_empty() {
                    tracing::info!("🤝 Using agent collaboration mode: {:?}", mode);
                    
                    // Start agent sessions for each specialization
                    for config in active_configs {
                        let session_id = agent_bridge.start_session(
                            config.specialization.clone(),
                            format!("chat-{}", chat_id),
                        ).await.ok();
                        
                        tracing::debug!("Started agent session: {:?}", session_id);
                    }
                    
                    // Continue with model execution, agents will enhance response
                }
            }
        }
        
        // 6. Create story-driven stream if story bridge is available
        if let Some(ref story_bridge) = self.story_bridge {
            if let Some(ref stream_manager) = self.stream_manager {
                // Check if this is a story-driven request
                if story_bridge.is_story_driven_request(content).await {
                    tracing::info!("📖 Processing story-driven request");
                    
                    // Create story stream
                    if let Ok(story_stream) = story_bridge.create_story_stream(content).await {
                        // Wrap StreamPacket stream into Result<StreamEvent> stream
                        use futures::stream::StreamExt;
                        let wrapped_stream = story_stream.map(|packet| {
                            Ok(crate::tui::chat::processing::streaming::StreamEvent::from(packet))
                        });
                        
                        // Register with stream manager
                        let _ = stream_manager.create_stream(
                            StreamSource::Story(crate::tui::chat::processing::unified_streaming::StorySource::Narrative),
                            wrapped_stream,
                            std::collections::HashMap::from([
                                ("chat_id".to_string(), chat_id.to_string()),
                                ("content".to_string(), content.to_string()),
                            ]),
                        ).await;
                        
                        tracing::info!("Story stream created and registered");
                    }
                }
            }
        }
        
        // 7. Fall back to model orchestration
        if let Some(ref orchestrator) = self.model_orchestrator {
            tracing::info!("🤖 Processing through model orchestrator");
            let status = orchestrator.get_status().await;
            tracing::info!("📊 Orchestrator status - API providers: {:?}, Local models: {:?}", 
                status.api_providers, status.local_models);
            
            let task_request = self.create_task_request(content).await?;
            tracing::info!("📝 Task request: prefer_local={}, constraints={:?}", 
                task_request.constraints.prefer_local, task_request.constraints);
            
            // Use model selector if available for orchestrated execution
            let response = if let Some(ref selector) = self.model_selector {
                tracing::info!("🎯 Using model selector for orchestrated execution");
                
                // Get orchestration status
                let orch_status = selector.get_status().await;
                tracing::info!("📊 Orchestration: enabled={}, models={}, strategy={}, parallel={}", 
                    orch_status.enabled, orch_status.model_count, orch_status.routing_strategy, orch_status.parallel_execution);
                
                // Execute with orchestration
                match selector.execute_with_orchestration(task_request.clone()).await {
                    Ok(resp) => Ok(resp),
                    Err(e) => {
                        tracing::warn!("Model selector failed: {}, falling back to direct execution", e);
                        // Fall back to direct orchestrator execution
                        orchestrator.execute_with_fallback(task_request.clone()).await
                            .map_err(|e| ChatError::Internal(format!("Orchestrator error: {}", e)))
                    }
                }
            } else {
                // Direct orchestrator execution
                tracing::info!("Using direct orchestrator execution (no model selector)");
                
                // Use retry logic for model orchestrator calls
                let retry_config = TimeoutConfig::default();
                let orchestrator_clone = orchestrator.clone();
                let task_request_clone = task_request.clone();
                
                with_timeout_retry(&retry_config, || {
                    let orchestrator = orchestrator_clone.clone();
                    let request = task_request_clone.clone();
                    async move {
                        orchestrator.execute_with_fallback(request).await
                            .map_err(|e| ChatError::Internal(format!("Orchestrator error: {}", e)))
                    }
                }).await
            };
            
            match response {
                Ok(response) => {
                    tracing::info!("✅ Got response from model: {} (length: {})", 
                        response.model_used.model_id(), response.content.len());
                    
                    // Ensure the response is complete and not cut off
                    let complete_content = if response.content.len() < 20 && 
                        !response.content.contains(".") && 
                        !response.content.contains("!") && 
                        !response.content.contains("?") {
                        // Response seems incomplete, add a notice
                        format!("{} [Response may be incomplete - retrying...]", response.content)
                    } else {
                        response.content
                    };
                    
                    // Parse the response for tasks and tool calls
                    let tasks_to_execute = self.parse_tasks_from_response(&complete_content, content).await;
                    
                    // If we detected tasks in the user's request and the model provided implementation steps
                    if self.looks_like_task_request(content) && !tasks_to_execute.is_empty() {
                        tracing::info!("🎯 Detected {} tasks to execute from model response", tasks_to_execute.len());
                        
                        // Spawn code agents for complex tasks
                        let needs_agent = tasks_to_execute.iter().any(|t| 
                            t.task_type == TaskType::CreateFile && t.content.as_ref().map_or(false, |c| c.len() > 100)
                        );
                        
                        if needs_agent {
                            // Create and spawn a code agent
                            let (agent_tx, agent_rx) = mpsc::channel(100);
                            let agent = self.spawn_code_agent(content, agent_tx).await;
                            
                            // Execute tasks through agent
                            if let Some(agent) = agent {
                                for task in &tasks_to_execute {
                                    if task.task_type == TaskType::CreateFile {
                                        if let (Some(path), Some(content)) = (&task.path, &task.content) {
                                            // Agent will handle file creation with proper structure
                                            tracing::info!("🤖 Agent handling file creation: {}", path);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Execute the tasks directly or through agents
                        for task in tasks_to_execute {
                            tracing::info!("🔧 Executing task: {}", task.description);
                            if let Err(e) = self.execute_parsed_task(&task).await {
                                tracing::error!("Failed to execute task: {}", e);
                            }
                        }
                    }
                    
                    // Check if the response contains code and transfer to editor
                    if self.looks_like_code_generation(content) && complete_content.contains("```") {
                        if let Some(ref bridges) = self.bridges {
                            let (code, language) = self.extract_code_block(&complete_content);
                            if !code.is_empty() {
                                // Transfer code to editor
                                let _ = bridges.editor_bridge.transfer_code_to_editor(
                                    chat_id.to_string(),
                                    code.clone(),
                                    Some(language),
                                    None,
                                ).await;
                                
                                tracing::info!("📝 Code transferred to editor");
                            }
                        }
                    }
                    
                    let ai_message = AssistantResponseType::new_ai_message(
                        complete_content,
                        Some(response.model_used.model_id()),
                    );
                    match self.response_tx.send(ai_message).await {
                        Ok(_) => tracing::info!("✅ Response sent to UI"),
                        Err(e) => tracing::error!("❌ Failed to send response to UI: {}", e),
                    }
                }
                Err(e) => {
                    tracing::error!("❌ Model orchestration failed: {}", e);
                    let error_message = AssistantResponseType::Error {
                        id: uuid::Uuid::new_v4().to_string(),
                        error_type: "ModelError".to_string(),
                        message: format!("Failed to process message: {}", e),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        metadata: Default::default(),
                    };
                    self.response_tx.send(error_message).await?;
                }
            }
        } else {
            // No orchestrator available
            let error_message = AssistantResponseType::new_ai_message(
                "⚠️ No model orchestrator available. Please configure API keys.".to_string(),
                Some("system".to_string()),
            );
            self.response_tx.send(error_message).await?;
        }
        
        // Record successful operation in telemetry
        let operation_duration = operation_start.elapsed();
        let telemetry_clone = telemetry_collector.clone();
        tokio::spawn(async move {
            telemetry_clone.record_operation(operation_duration, true).await;
        });
        
        Ok(())
    }
    
    /// Check if input looks like direct tool/agent execution (not code generation)
    fn looks_like_direct_execution(&self, content: &str) -> bool {
        let lower = content.to_lowercase();
        
        // Check for explicit agent/tool invocation patterns
        if lower.starts_with("use ") || lower.starts_with("call ") || 
           lower.starts_with("invoke ") || lower.starts_with("execute ") ||
           lower.starts_with("run ") || lower.starts_with("agent:") ||
           lower.starts_with("tool:") || lower.starts_with("/") {
            return true;
        }
        
        // Check for specific tool action patterns
        let tool_patterns = [
            "search for", "find files", "grep for", "look for",
            "list files", "show files", "check status",
            "run tests", "run build", "compile", "execute script",
            "git status", "git diff", "git log", "commit",
            "analyze code", "review code", "check syntax",
            "open terminal", "run command", "shell command",
        ];
        
        for pattern in &tool_patterns {
            if lower.contains(pattern) {
                return true;
            }
        }
        
        // Check if it's a shell command pattern (contains common shell commands)
        let shell_commands = ["ls", "cd", "pwd", "cat", "grep", "find", "npm", "cargo", "python", "node"];
        for cmd in &shell_commands {
            if lower.split_whitespace().any(|word| word == *cmd) {
                return true;
            }
        }
        
        false
    }
    
    /// Check if input looks like a tool request
    fn looks_like_tool_request(&self, content: &str) -> bool {
        let lower = content.to_lowercase();
        
        // Check for direct commands
        if lower.starts_with('/') || lower.starts_with("run ") || lower.starts_with("execute ") {
            return true;
        }
        
        // Check for code execution patterns
        if self.looks_like_code_execution(content) {
            return true;
        }
        
        // Check for code generation patterns
        if self.looks_like_code_generation(content) {
            return true;
        }
        
        // Check for action words that indicate tool usage
        let action_words = [
            "search", "find", "look for", "locate",
            "github", "git", "repository", "repo", "pr", "pull request", "issue",
            "file", "open", "read", "write", "create", "delete", "edit",
            "analyze", "review", "examine", "inspect",
            "code", "implement", "generate", "refactor",
            "commit", "push", "pull", "merge",
            "browse", "web", "fetch", "download",
            "run", "execute", "launch", "start",
            "test", "build", "compile", "deploy",
            "install", "update", "upgrade",
        ];
        
        action_words.iter().any(|word| lower.contains(word))
    }
    
    /// Check if input looks like code to execute
    fn looks_like_code_execution(&self, content: &str) -> bool {
        let lower = content.to_lowercase();
        
        // Check for explicit code execution requests
        if lower.contains("run this") || lower.contains("execute this") || 
           lower.contains("run the following") || lower.contains("execute the following") ||
           lower.contains("run code") || lower.contains("execute code") {
            return true;
        }
        
        // Check for code blocks with execution hints
        if content.contains("```") && (lower.contains("run") || lower.contains("execute")) {
            return true;
        }
        
        // Check for shebang lines (indicates executable script)
        if content.contains("#!/") {
            return true;
        }
        
        false
    }
    
    /// Check if input looks like a code generation request
    fn looks_like_code_generation(&self, content: &str) -> bool {
        let lower = content.to_lowercase();
        
        // Don't treat as code generation if it's a direct execution request
        if self.looks_like_direct_execution(content) {
            return false;
        }
        
        // Explicit code generation patterns (for creating new code, not executing)
        let generation_patterns = [
            "implement a function", "create a function", "write a function",
            "implement a class", "create a class", "write a class",
            "write code for", "generate code for", "create code for",
            "build a component", "develop a feature", "design a system",
            "write a program", "create a program", "implement a solution",
            "code for me", "script for me", "function to do",
            "help me implement", "help me write", "help me create",
            "show me how to code", "example code for", "template for",
            "can you write", "can you implement", "please write code",
        ];
        
        // Check for any generation pattern
        for pattern in generation_patterns.iter() {
            if lower.contains(pattern) {
                return true;
            }
        }
        
        // Check for language-specific requests
        let language_patterns = [
            "in rust", "in python", "in javascript", "in typescript",
            "in go", "in java", "in c++", "in c#", "using rust",
            "using python", "with javascript", "rust code", "python code",
        ];
        
        // If mentions code/implement/write with a language
        if (lower.contains("code") || lower.contains("implement") || 
            lower.contains("write") || lower.contains("create")) &&
           language_patterns.iter().any(|lang| lower.contains(lang)) {
            return true;
        }
        
        false
    }
    
    /// Execute a tool request from the user
    async fn execute_tool_request(
        &self,
        content: &str,
        chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        // Check for system commands (starting with /)
        if content.starts_with('/') {
            return self.execute_system_command(content, chat_id).await;
        }
        
        // Check for code execution requests
        if self.looks_like_code_execution(content) {
            return self.execute_code_from_content(content, chat_id).await;
        }
        
        // Check for explicit run/execute commands
        if content.starts_with("run ") || content.starts_with("execute ") {
            let command = content.strip_prefix("run ").or_else(|| content.strip_prefix("execute "))
                .unwrap_or(content);
            return self.execute_shell_command(command, chat_id).await;
        }
        
        // Try intelligent tool detection and execution
        self.try_direct_tool_execution(content, chat_id).await
    }
    
    /// Execute system commands (starting with /)
    async fn execute_system_command(
        &self,
        command: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        let parts: Vec<&str> = command[1..].split_whitespace().collect();
        if parts.is_empty() {
            return Ok(None);
        }
        
        match parts[0] {
            "help" => {
                let help_text = r#"📚 **Available Commands:**
                
/help - Show this help message
/status - Show system status
/models - List available models
/tools - List available tools
/run <command> - Execute a shell command
/code <language> - Execute code in specified language
/python <code> - Execute Python code
/javascript <code> - Execute JavaScript code
/rust <code> - Execute Rust code
/git <args> - Execute git commands
/search <query> - Search for information
/file <path> - Read or analyze a file
/web <url> - Fetch web content
/task <description> - Create a task
/clear - Clear chat history
/export <format> - Export chat (json/markdown)
/save <filename> - Save chat to file
/load <filename> - Load chat from file
                
You can also use natural language to request actions or paste code blocks to execute!"#;
                
                Ok(Some(AssistantResponseType::new_ai_message(
                    help_text.to_string(),
                    Some("system".to_string()),
                )))
            }
            "status" => {
                let status = self.get_system_status().await?;
                Ok(Some(AssistantResponseType::new_ai_message(
                    status,
                    Some("system".to_string()),
                )))
            }
            "clear" => {
                // Clear chat history
                let mut state = self.chat_state.write().await;
                state.messages.clear();
                Ok(Some(AssistantResponseType::new_ai_message(
                    "✨ Chat history cleared".to_string(),
                    Some("system".to_string()),
                )))
            }
            "code" => {
                // Execute code with specified language
                let code = parts[2..].join(" ");
                let language = parts.get(1).copied().unwrap_or("auto");
                self.execute_code_snippet(code, language, _chat_id).await
            }
            "python" | "py" => {
                // Execute Python code
                let code = parts[1..].join(" ");
                self.execute_code_snippet(code, "python", _chat_id).await
            }
            "javascript" | "js" => {
                // Execute JavaScript code
                let code = parts[1..].join(" ");
                self.execute_code_snippet(code, "javascript", _chat_id).await
            }
            "rust" | "rs" => {
                // Execute Rust code
                let code = parts[1..].join(" ");
                self.execute_code_snippet(code, "rust", _chat_id).await
            }
            "export" => {
                // Export chat history
                let format = parts.get(1).copied().unwrap_or("markdown");
                self.export_chat(format, _chat_id).await
            }
            "save" => {
                // Save chat to file
                let filename = parts.get(1).map(|s| s.to_string())
                    .unwrap_or_else(|| format!("chat_{}.json", chrono::Utc::now().format("%Y%m%d_%H%M%S")));
                self.save_chat(&filename, _chat_id).await
            }
            "load" => {
                // Load chat from file
                if let Some(filename) = parts.get(1) {
                    self.load_chat(filename, _chat_id).await
                } else {
                    Ok(Some(AssistantResponseType::new_ai_message(
                        "❌ Please specify a filename to load".to_string(),
                        Some("system".to_string()),
                    )))
                }
            }
            _ => {
                // Try to execute as a tool command
                self.try_direct_tool_execution(command, _chat_id).await
            }
        }
    }
    
    /// Execute shell commands safely
    async fn execute_shell_command(
        &self,
        command: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        // Safety check - don't execute dangerous commands
        let dangerous_patterns = ["rm -rf", "format", "del /f", "sudo rm", "dd if="];
        for pattern in dangerous_patterns.iter() {
            if command.contains(pattern) {
                return Ok(Some(AssistantResponseType::new_ai_message(
                    format!("⚠️ Command blocked for safety: {}", command),
                    Some("system".to_string()),
                )));
            }
        }
        
        // Execute the command
        tracing::info!("Executing shell command: {}", command);
        
        match tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                let mut result = String::new();
                result.push_str(&format!("💻 Executed: `{}`\n\n", command));
                
                if !stdout.is_empty() {
                    result.push_str("**Output:**\n```\n");
                    result.push_str(&stdout);
                    result.push_str("```\n");
                }
                
                if !stderr.is_empty() {
                    result.push_str("\n**Errors:**\n```\n");
                    result.push_str(&stderr);
                    result.push_str("```\n");
                }
                
                if output.status.success() {
                    result.push_str("\n✅ Command completed successfully");
                } else {
                    result.push_str(&format!("\n❌ Command failed with exit code: {:?}", output.status.code()));
                }
                
                Ok(Some(AssistantResponseType::new_ai_message(
                    result,
                    Some("shell-executor".to_string()),
                )))
            }
            Err(e) => {
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("❌ Failed to execute command: {}", e),
                    Some("shell-executor".to_string()),
                )))
            }
        }
    }
    
    /// Get system status
    async fn get_system_status(&self) -> Result<String> {
        let mut status = String::from("📊 **System Status**\n\n");
        
        // Model orchestrator status
        if let Some(ref orchestrator) = self.model_orchestrator {
            let orch_status = orchestrator.get_status().await;
            status.push_str(&format!("**Models:**\n"));
            status.push_str(&format!("  • API Providers: {}\n", orch_status.api_providers.len()));
            status.push_str(&format!("  • Local Models: {}\n", orch_status.local_models.model_statuses.len()));
        } else {
            status.push_str("**Models:** Not configured\n");
        }
        
        // Tool status
        if self.tool_executor.is_some() {
            status.push_str("\n**Tools:** Available ✅\n");
        } else {
            status.push_str("\n**Tools:** Not configured\n");
        }
        
        // Cognitive enhancement status
        if self.cognitive_enhancement.is_some() {
            status.push_str("**Cognitive Enhancement:** Enabled ✅\n");
        } else {
            status.push_str("**Cognitive Enhancement:** Disabled\n");
        }
        
        // Bridge status
        if self.bridges.is_some() {
            status.push_str("**Cross-tab Bridges:** Connected ✅\n");
        } else {
            status.push_str("**Cross-tab Bridges:** Not connected\n");
        }
        
        Ok(status)
    }
    
    /// Try to execute tools directly
    async fn try_direct_tool_execution(
        &self,
        content: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        // First try to use the tool bridge if available (for cross-tab tools)
        if let Some(ref bridges) = self.bridges {
            // Check if the Utilities tab has configured tools we can use
            let available_tools = bridges.tool_bridge.get_available_tools().await;
            
            // Try to match content to available tools
            for (tool_id, config) in available_tools {
                if content.contains(&config.name) || content.contains(&tool_id) {
                    tracing::info!("🔧 Executing tool via bridge: {}", tool_id);
                    
                    // Execute through bridge
                    let params = serde_json::json!({
                        "input": content,
                        "chat_id": _chat_id,
                    });
                    
                    match bridges.tool_bridge.execute_from_chat(tool_id, params).await {
                        Ok(result) => {
                            let content_str = result.summary;
                            let tool_message = AssistantResponseType::new_ai_message(
                                content_str,
                                Some("loki-tools-bridge".to_string()),
                            );
                            return Ok(Some(tool_message));
                        }
                        Err(e) => {
                            tracing::warn!("Bridge tool execution failed: {}", e);
                        }
                    }
                }
            }
        }
        
        // Fall back to direct tool executor if available
        if let Some(ref tool_executor) = self.tool_executor {
            // Try to parse command and execute
            let command_registry = CommandRegistry::new();
            if let Ok(parsed_command) = command_registry.parse(content) {
                match tool_executor.execute(parsed_command).await {
                    Ok(result) => {
                        let content_str = result.content.as_str()
                            .unwrap_or("Tool executed successfully")
                            .to_string();
                        let tool_message = AssistantResponseType::new_ai_message(
                            content_str,
                            Some("loki-tools".to_string()),
                        );
                        return Ok(Some(tool_message));
                    }
                    Err(e) => {
                        tracing::warn!("Tool execution failed: {}", e);
                        return Ok(None);
                    }
                }
            } else {
                // Command parsing failed, not a tool request
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    /// Get recent conversation context
    async fn get_recent_context(&self, chat_id: usize) -> Result<Vec<String>> {
        let state = self.chat_state.read().await;
        let messages = state.messages
            .iter()
            .rev()
            .take(10)
            .map(|msg| format!("{}: {}", msg.get_author(), msg.get_content()))
            .collect();
        Ok(messages)
    }
    
    /// Create a task request from user input
    async fn create_task_request(&self, content: &str) -> Result<crate::models::orchestrator::TaskRequest> {
        use crate::models::orchestrator::{TaskRequest, TaskType, TaskConstraints};
        
        // Detect task type from content with priority-aware mapping
        let content_lower = content.to_lowercase();
        
        // Check for urgency indicators that affect priority
        let urgency_level = self.detect_urgency_level(&content_lower);
        
        let task_type = if content_lower.contains("code") || content_lower.contains("implement") {
            TaskType::CodeGeneration { language: "rust".to_string() }
        } else if content_lower.contains("review") || content_lower.contains("analyze") {
            TaskType::CodeReview { language: "rust".to_string() }
        } else if content_lower.contains("reason") || content_lower.contains("logic") {
            TaskType::LogicalReasoning
        } else if content_lower.contains("data") || content_lower.contains("statistics") {
            TaskType::DataAnalysis
        } else if content_lower.contains("creative") || content_lower.contains("story") {
            TaskType::CreativeWriting
        } else {
            TaskType::GeneralChat
        };
        
        // Get orchestration settings
        let orchestration = self.orchestration_manager.read().await;
        
        let constraints = TaskConstraints {
            max_tokens: Some(2000),
            context_size: Some(8000),
            max_time: Some(std::time::Duration::from_secs(30)),
            max_latency_ms: Some(5000),
            max_cost_cents: Some(orchestration.cost_threshold_cents as f32 / 100.0),
            quality_threshold: Some(orchestration.quality_threshold),
            priority: urgency_level,
            prefer_local: true, // Always prefer local models to avoid API calls
            require_streaming: true, // Enable streaming for agent streams and real-time responses
            required_capabilities: Vec::new(),
            task_hint: None,
            creativity_level: None,
            formality_level: None,
            target_audience: None,
        };
        
        let mut request = TaskRequest {
            task_type,
            content: content.to_string(),
            constraints,
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };
        
        // Apply orchestration settings through connector
        if let Some(ref connector) = self.orchestration_connector {
            request = connector.apply_to_task_request(request);
        }
        
        Ok(request)
    }
    
    /// Detect urgency level from content
    fn detect_urgency_level(&self, content_lower: &str) -> String {
        if content_lower.contains("urgent") || content_lower.contains("asap") || 
           content_lower.contains("immediately") || content_lower.contains("critical") {
            "high".to_string()
        } else if content_lower.contains("important") || content_lower.contains("soon") ||
                  content_lower.contains("priority") {
            "medium".to_string()
        } else if content_lower.contains("whenever") || content_lower.contains("eventually") ||
                  content_lower.contains("low priority") {
            "low".to_string()
        } else {
            "normal".to_string()
        }
    }
    
    /// Parse tasks from model response
    async fn parse_tasks_from_response(&self, response: &str, original_request: &str) -> Vec<ParsedTask> {
        let mut tasks = Vec::new();
        
        // Check if the user requested file/directory creation
        if original_request.to_lowercase().contains("create") || 
           original_request.to_lowercase().contains("setup") ||
           original_request.to_lowercase().contains("implement") {
            
            // Look for directory creation patterns
            if response.contains("mkdir") || response.contains("create directory") || 
               response.contains("create a directory") || response.contains("Create directory") {
                // Extract directory paths
                let dir_patterns = [
                    r"mkdir -p ([/\w\-\.]+)",
                    r"mkdir ([/\w\-\.]+)",
                    r"create directory: ([/\w\-\.]+)",
                    r"directory: ([/\w\-\.]+)",
                    r"`([/\w\-\.]+)` directory",
                ];
                
                for pattern in dir_patterns {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        for cap in re.captures_iter(response) {
                            if let Some(path) = cap.get(1) {
                                tasks.push(ParsedTask {
                                    task_type: TaskType::CreateDirectory,
                                    description: format!("Create directory: {}", path.as_str()),
                                    path: Some(path.as_str().to_string()),
                                    content: None,
                                    language: None,
                                });
                            }
                        }
                    }
                }
            }
            
            // Look for file creation patterns with code blocks
            if response.contains("```") {
                // Extract code blocks with file paths
                let code_block_re = match regex::Regex::new(r"(?s)(?:Create|create|File:|file:)\s*`?([/\w\-\.]+)`?.*?```(\w+)?\n(.*?)```") {
                    Ok(re) => re,
                    Err(e) => {
                        tracing::error!("Failed to compile regex: {}", e);
                        return Vec::new();
                    }
                };
                for cap in code_block_re.captures_iter(response) {
                    if let (Some(file_path), Some(code)) = (cap.get(1), cap.get(3)) {
                        let language = cap.get(2).map(|m| m.as_str().to_string());
                        tasks.push(ParsedTask {
                            task_type: TaskType::CreateFile,
                            description: format!("Create file: {}", file_path.as_str()),
                            path: Some(file_path.as_str().to_string()),
                            content: Some(code.as_str().to_string()),
                            language,
                        });
                    }
                }
            }
            
            // Look for package.json, Cargo.toml, etc. patterns
            if response.contains("package.json") || response.contains("Cargo.toml") || 
               response.contains("requirements.txt") || response.contains("pyproject.toml") {
                // These are configuration files that should be created
                let config_patterns = [
                    (r"package\.json.*?```json\n(.*?)```", "package.json", "json"),
                    (r"Cargo\.toml.*?```toml\n(.*?)```", "Cargo.toml", "toml"),
                    (r"requirements\.txt.*?```\n(.*?)```", "requirements.txt", "txt"),
                    (r"pyproject\.toml.*?```toml\n(.*?)```", "pyproject.toml", "toml"),
                ];
                
                for (pattern, filename, lang) in config_patterns {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if let Some(cap) = re.captures(response) {
                            if let Some(content) = cap.get(1) {
                                tasks.push(ParsedTask {
                                    task_type: TaskType::CreateFile,
                                    description: format!("Create {}", filename),
                                    path: Some(filename.to_string()),
                                    content: Some(content.as_str().to_string()),
                                    language: Some(lang.to_string()),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        tasks
    }
    
    /// Execute a parsed task
    async fn execute_parsed_task(&self, task: &ParsedTask) -> Result<()> {
        tracing::info!("Executing task: {:?}", task);
        
        match task.task_type {
            TaskType::CreateDirectory => {
                if let Some(path) = &task.path {
                    // Use the tool integration to create directory
                    if let Some(ref tools) = self.tools {
                        let command = format!("create_directory {}", path);
                        match tools.execute_tool(&command).await {
                            Ok(result) => {
                                tracing::info!("✅ Directory created: {}", path);
                                let notification = AssistantResponseType::new_ai_message(
                                    format!("✅ Created directory: {}", path),
                                    Some("tool-executor".to_string()),
                                );
                                if let Err(e) = self.response_tx.send(notification).await {
                                    tracing::warn!("Failed to send notification: {}", e);
                                }
                            }
                            Err(e) => {
                                tracing::error!("Failed to create directory {}: {}", path, e);
                                // Try direct filesystem operation as fallback
                                if let Err(e) = tokio::fs::create_dir_all(path).await {
                                    tracing::error!("Direct filesystem operation also failed: {}", e);
                                } else {
                                    tracing::info!("✅ Directory created via direct filesystem: {}", path);
                                    let notification = AssistantResponseType::new_ai_message(
                                        format!("✅ Created directory: {}", path),
                                        Some("filesystem".to_string()),
                                    );
                                    if let Err(e) = self.response_tx.send(notification).await {
                                    tracing::warn!("Failed to send notification: {}", e);
                                }
                                }
                            }
                        }
                    } else {
                        // Fallback to direct filesystem operation
                        tokio::fs::create_dir_all(path).await?;
                        tracing::info!("✅ Directory created: {}", path);
                        let notification = AssistantResponseType::new_ai_message(
                            format!("✅ Created directory: {}", path),
                            Some("filesystem".to_string()),
                        );
                        if let Err(e) = self.response_tx.send(notification).await {
                            tracing::warn!("Failed to send notification: {}", e);
                        }
                    }
                }
            }
            TaskType::CreateFile => {
                if let (Some(path), Some(content)) = (&task.path, &task.content) {
                    // Ensure parent directory exists
                    if let Some(parent) = std::path::Path::new(path).parent() {
                        if !parent.as_os_str().is_empty() {
                            tokio::fs::create_dir_all(parent).await?;
                        }
                    }
                    
                    // Write the file
                    tokio::fs::write(path, content).await?;
                    tracing::info!("✅ File created: {}", path);
                    
                    let notification = AssistantResponseType::new_ai_message(
                        format!("✅ Created file: {} ({} bytes)", path, content.len()),
                        Some("filesystem".to_string()),
                    );
                    let _ = self.response_tx.send(notification).await;
                }
            }
            _ => {
                tracing::warn!("Unsupported task type: {:?}", task.task_type);
            }
        }
        
        Ok(())
    }
    
    /// Analyze requirements to determine which types of agents are needed
    fn analyze_requirements(&self, content: &str) -> Vec<crate::tui::chat::agents::code_agent::AgentRequirement> {
        use crate::tui::chat::agents::code_agent::AgentRequirement;
        
        let mut requirements = Vec::new();
        let lower = content.to_lowercase();
        
        // Check for frontend requirements
        if lower.contains("frontend") || lower.contains("react") || lower.contains("typescript") ||
           lower.contains("ui") || lower.contains("interface") || lower.contains("vue") ||
           lower.contains("angular") || lower.contains("svelte") {
            requirements.push(AgentRequirement::Frontend);
        }
        
        // Check for backend requirements
        if lower.contains("backend") || lower.contains("rust") || lower.contains("server") ||
           lower.contains("api") || lower.contains("endpoint") || lower.contains("microservice") {
            requirements.push(AgentRequirement::Backend);
        }
        
        // Check for data processing/Python requirements
        if lower.contains("python") || lower.contains("streaming") || lower.contains("data process") ||
           lower.contains("ml") || lower.contains("machine learning") || lower.contains("pandas") {
            requirements.push(AgentRequirement::DataProcessing);
        }
        
        // Check for testing requirements
        if lower.contains("test") || lower.contains("jest") || lower.contains("pytest") ||
           lower.contains("unit test") || lower.contains("integration test") {
            requirements.push(AgentRequirement::Testing);
        }
        
        // Check for DevOps requirements
        if lower.contains("docker") || lower.contains("kubernetes") || lower.contains("ci/cd") ||
           lower.contains("deploy") || lower.contains("terraform") {
            requirements.push(AgentRequirement::DevOps);
        }
        
        // Check for database requirements
        if lower.contains("database") || lower.contains("sql") || lower.contains("postgres") ||
           lower.contains("mongodb") || lower.contains("redis") {
            requirements.push(AgentRequirement::Database);
        }
        
        // Check for mobile requirements
        if lower.contains("mobile") || lower.contains("react native") || lower.contains("flutter") ||
           lower.contains("ios") || lower.contains("android") {
            requirements.push(AgentRequirement::Mobile);
        }
        
        // If no specific requirements found, add a general agent
        if requirements.is_empty() {
            requirements.push(AgentRequirement::General);
        }
        
        // Remove duplicates
        requirements.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        requirements.dedup();
        
        requirements
    }
    
    /// Extract todo request from message content with enhanced patterns
    async fn extract_todo_request(&self, content: &str) -> Option<CreateTodoRequest> {
        let lower = content.to_lowercase();
        
        // Enhanced todo/task detection patterns
        let is_todo = lower.contains("todo") || lower.contains("task") || 
                     lower.contains("remind me") || lower.contains("need to") ||
                     lower.starts_with("todo:") || lower.starts_with("task:") ||
                     lower.contains("add to list") || lower.contains("create task") ||
                     lower.contains("i need to") || lower.contains("we should") ||
                     lower.contains("don't forget") || lower.contains("don't let me forget") ||
                     lower.contains("later we") || lower.contains("after this") ||
                     lower.contains("make sure to") || lower.contains("remember to");
        
        if !is_todo {
            return None;
        }
        
        // Extract title from content
        let title = if let Some(idx) = lower.find("todo:") {
            content[idx + 5..].trim().to_string()
        } else if let Some(idx) = lower.find("task:") {
            content[idx + 5..].trim().to_string()
        } else if lower.starts_with("remind me to") {
            content[12..].trim().to_string()
        } else if lower.starts_with("need to") {
            content[7..].trim().to_string()
        } else {
            // Use the whole content as title
            content.to_string()
        };
        
        if title.is_empty() {
            return None;
        }
        
        Some(CreateTodoRequest {
            title,
            description: None,
            creator: "user".to_string(),
            assignee: None,
            due_date: None,
            tags: self.extract_tags(content),
            parent_id: None,
            dependency_ids: Vec::new(),
            priority_hint: self.detect_priority_hint(content),
            story_context: None,
            priority: None,
            energy_required: None,
            focus_required: None,
            context: None,
        })
    }
    
    /// Extract tags from content
    fn extract_tags(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();
        
        // Extract hashtags
        let re = regex::Regex::new(r"#(\w+)").unwrap();
        for cap in re.captures_iter(content) {
            if let Some(tag) = cap.get(1) {
                tags.push(tag.as_str().to_string());
            }
        }
        
        // Add contextual tags
        let lower = content.to_lowercase();
        if lower.contains("urgent") || lower.contains("asap") {
            tags.push("urgent".to_string());
        }
        if lower.contains("bug") || lower.contains("fix") {
            tags.push("bug".to_string());
        }
        if lower.contains("feature") || lower.contains("implement") {
            tags.push("feature".to_string());
        }
        
        tags
    }
    
    /// Detect priority hint from content
    fn detect_priority_hint(&self, content: &str) -> Option<f32> {
        let lower = content.to_lowercase();
        
        if lower.contains("critical") || lower.contains("urgent") || lower.contains("asap") {
            Some(0.9)
        } else if lower.contains("high priority") || lower.contains("important") {
            Some(0.7)
        } else if lower.contains("low priority") || lower.contains("when you can") {
            Some(0.3)
        } else {
            None
        }
    }
    
    /// Handle todo queries (e.g., "what do I need to do?", "show my todos")
    async fn handle_todo_query(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        
        // Check if this is a todo query
        let is_query = lower.contains("what do i need") || lower.contains("what's on my") ||
                      lower.contains("show my todo") || lower.contains("list todo") ||
                      lower.contains("show todo") || lower.contains("my tasks") ||
                      lower.contains("what tasks") || lower.contains("pending todo") ||
                      lower.contains("active todo") || lower.contains("what's on my plate");
        
        if !is_query {
            return None;
        }
        
        // Get todos from todo manager if available
        if let Some(ref todo_manager) = self.todo_manager {
            let todos = todo_manager.get_todos().await;
            
            if todos.is_empty() {
                return Some("You don't have any active todos at the moment. 🎉".to_string());
            }
            
            let mut response = format!("You have {} active todo{}:\n\n", 
                                     todos.len(), 
                                     if todos.len() == 1 { "" } else { "s" });
            
            for (i, todo) in todos.iter().enumerate() {
                let priority_emoji = match &todo.priority.level {
                    PriorityLevel::Critical => "🔴",
                    PriorityLevel::High => "🟠", 
                    PriorityLevel::Medium => "🟡",
                    PriorityLevel::Low => "🟢",
                    PriorityLevel::Minimal => "⚪",
                };
                
                let status_emoji = match &todo.status {
                    TodoStatus::InProgress => "⚡",
                    TodoStatus::Blocked => "🚫",
                    TodoStatus::Review => "👁️",
                    _ => "📝",
                };
                
                response.push_str(&format!("{}. {} {} {} - {}\n",
                    i + 1,
                    priority_emoji,
                    status_emoji,
                    todo.title,
                    &todo.id[..8]
                ));
                
                if let Some(desc) = &todo.description {
                    response.push_str(&format!("   {}\n", desc));
                }
            }
            
            response.push_str("\n💡 Tip: Say 'I finished <task>' to mark a todo as complete.");
            
            return Some(response);
        }
        
        None
    }
    
    /// Detect todo completion in messages
    async fn detect_todo_completion(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        
        // Check for completion patterns
        let is_completion = lower.contains("finished") || lower.contains("completed") ||
                           lower.contains("done with") || lower.contains("i've done") ||
                           lower.contains("that's done") || lower.contains("task complete") ||
                           lower.contains("todo complete") || lower.contains("i just finished");
        
        if !is_completion {
            return None;
        }
        
        // Try to find what was completed
        if let Some(ref todo_manager) = self.todo_manager {
            let todos = todo_manager.get_todos().await;
            let has_todos = !todos.is_empty();
            
            // Look for matching todos by keywords in the content
            for todo in todos {
                let todo_lower = todo.title.to_lowercase();
                
                // Check if the content mentions this todo
                let words: Vec<&str> = todo_lower.split_whitespace().collect();
                let mut match_count = 0;
                
                for word in &words {
                    if lower.contains(word) && word.len() > 3 { // Skip short words
                        match_count += 1;
                    }
                }
                
                // If at least 50% of the todo's words are mentioned, consider it a match
                if match_count >= words.len() / 2 && match_count > 0 {
                    // Mark as complete
                    if let Ok(_) = todo_manager.update_status(&todo.id, TodoStatus::Completed).await {
                        return Some(format!(
                            "✅ Great! I've marked '{}' as complete.\n\n{}",
                            todo.title,
                            self.suggest_next_todo(&todo.id).await.unwrap_or_default()
                        ));
                    }
                }
            }
            
            // If no specific match found but user says something is done, ask for clarification
            if has_todos {
                return Some("Which todo did you complete? You can tell me the task name or ID.".to_string());
            }
        }
        
        None
    }
    
    /// Suggest the next todo after completing one
    async fn suggest_next_todo(&self, completed_id: &str) -> Option<String> {
        if let Some(ref todo_manager) = self.todo_manager {
            let todos = todo_manager.get_todos().await;
            
            // Filter out the completed one and get highest priority
            let mut remaining: Vec<_> = todos.iter()
                .filter(|t| t.id != completed_id && t.status != TodoStatus::Completed)
                .collect();
            
            if remaining.is_empty() {
                return Some("🎉 All todos complete! You're all caught up.".to_string());
            }
            
            // Sort by priority (Critical > High > Medium > Low > Optional)
            remaining.sort_by(|a, b| {
                let a_priority = match a.priority.level {
                    PriorityLevel::Critical => 5,
                    PriorityLevel::High => 4,
                    PriorityLevel::Medium => 3,
                    PriorityLevel::Low => 2,
                    PriorityLevel::Minimal => 1,
                };
                let b_priority = match b.priority.level {
                    PriorityLevel::Critical => 5,
                    PriorityLevel::High => 4,
                    PriorityLevel::Medium => 3,
                    PriorityLevel::Low => 2,
                    PriorityLevel::Minimal => 1,
                };
                b_priority.cmp(&a_priority)
            });
            
            if let Some(next) = remaining.first() {
                return Some(format!("Your next priority is: {} '{}'", 
                    match next.priority.level {
                        PriorityLevel::Critical => "🔴 Critical",
                        PriorityLevel::High => "🟠 High",
                        PriorityLevel::Medium => "🟡 Medium",
                        PriorityLevel::Low => "🟢 Low",
                        PriorityLevel::Minimal => "⚪ Minimal",
                    },
                    next.title
                ));
            }
        }
        
        None
    }
    
    /// Create orchestrated request with configuration
    async fn create_orchestrated_request(
        &self,
        content: &str,
        config: &crate::tui::bridges::orchestration_bridge::OrchestrationConfig,
    ) -> Result<crate::models::orchestrator::TaskRequest> {
        use crate::models::orchestrator::{TaskRequest, TaskType, TaskConstraints};
        
        let mut request = self.create_task_request(content).await?;
        
        // Apply orchestration configuration
        request.constraints.max_tokens = Some((config.timeout_seconds * 10) as u32);
        request.constraints.priority = if config.cost_optimization { "low".to_string() } else { "high".to_string() };
        request.constraints.quality_threshold = Some(config.quality_threshold);
        
        // Set parallel execution hint
        if config.parallel_models > 1 {
            request.constraints.required_capabilities.push("parallel".to_string());
        }
        
        Ok(request)
    }
    
    /// Spawn multiple specialized agents based on requirements
    async fn spawn_agents_for_requirements(
        &self,
        requirements: &str,
        update_tx: mpsc::Sender<crate::tui::chat::agents::code_agent::AgentUpdate>,
    ) -> Vec<Arc<crate::tui::chat::agents::code_agent::CodeAgent>> {
        use crate::tui::chat::agents::code_agent::{CodeAgentFactory, CodeAgent};
        
        let agent_requirements = self.analyze_requirements(requirements);
        let mut agents = Vec::new();
        
        tracing::info!("🎯 Detected {} agent requirements from user request", agent_requirements.len());
        
        for req in agent_requirements {
            // Create specialized agent for this requirement
            let agent = CodeAgentFactory::create_specialized_agent(req.clone(), update_tx.clone());
            
            // Set model orchestrator if available
            let agent = if let Some(ref orchestrator) = self.model_orchestrator {
                let mut agent = agent;
                agent.set_model_orchestrator(orchestrator.clone());
                agent
            } else {
                agent
            };
            
            // Set editor bridge if available
            let agent = if let Some(ref bridges) = self.bridges {
                let mut agent = agent;
                agent.set_editor_bridge(bridges.editor_bridge.clone());
                agent
            } else {
                agent
            };
            
            let agent = Arc::new(agent);
            
            // Notify about agent creation
            let notification = AssistantResponseType::new_ai_message(
                format!("🤖 {} spawned", agent.name()),
                Some("agent-manager".to_string()),
            );
            let _ = self.response_tx.send(notification).await;
            
            tracing::info!("Spawned specialized agent: {}", agent.name());
            agents.push(agent);
        }
        
        agents
    }
    
    /// Spawn a code agent for complex tasks (legacy - kept for compatibility)
    async fn spawn_code_agent(
        &self,
        requirements: &str,
        update_tx: mpsc::Sender<crate::tui::chat::agents::code_agent::AgentUpdate>,
    ) -> Option<Arc<crate::tui::chat::agents::code_agent::CodeAgent>> {
        // For backward compatibility, just spawn the first agent from multi-agent method
        let agents = self.spawn_agents_for_requirements(requirements, update_tx).await;
        agents.into_iter().next()
    }
    
    /// Check if input looks like a task request
    fn looks_like_task_request(&self, content: &str) -> bool {
        let lower = content.to_lowercase();
        
        // Task-oriented keywords
        let task_patterns = [
            "set up", "setup", "create a", "implement", "build",
            "initialize", "generate", "make a", "help me create",
            "help me implement", "help me build", "can you create",
            "i need", "i want", "please create", "please implement",
            "let's create", "let's build", "start a new", "new project",
            "directory", "folder", "file", "scaffold", "bootstrap",
        ];
        
        task_patterns.iter().any(|pattern| lower.contains(pattern))
    }
    
    /// Handle extracted tasks from NLP
    async fn handle_extracted_tasks(&self, tasks: Vec<ExtractedTask>) -> Result<()> {
        if let Some(ref task_manager) = self.task_manager {
            for task in tasks {
                // Map NLP task priority to tool task priority
                let priority = self.map_task_priority(&task.priority);
                
                // Calculate due date based on estimated effort
                let due_date = task.estimated_effort.map(|effort| {
                    chrono::Utc::now() + chrono::Duration::from_std(effort).unwrap_or(chrono::Duration::days(1))
                });
                
                // Build additional context from task metadata
                let additional_context = self.build_task_context(&task);
                
                match task_manager.create_task(
                    &task.description,
                    Some(&additional_context),
                    priority,
                    due_date,
                    crate::tools::task_management::TaskPlatform::Internal,
                ).await {
                    Ok(created_task) => {
                        tracing::info!(
                            "✅ Created task {}: {} (Priority: {:?}, Confidence: {:.2})", 
                            created_task.id, 
                            task.description,
                            priority.clone(),
                            task.confidence
                        );
                        
                        // Send notification about task creation
                        let notification = AssistantResponseType::new_ai_message(
                            format!("📋 Task created: {} (ID: {})", task.description, &created_task.id[..8]),
                            Some("task-manager".to_string()),
                        );
                        if let Err(e) = self.response_tx.send(notification).await {
                            tracing::warn!("Failed to send notification: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create task '{}': {}", task.description, e);
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Map NLP task priority to tool task priority
    fn map_task_priority(&self, nlp_priority: &TaskPriority) -> TaskPriority {
        // Since both use the same TaskPriority enum now, just clone it
        nlp_priority.clone()
    }
    
    /// Send response with optional storage persistence
    async fn send_response_with_persistence(
        &self,
        response: AssistantResponseType,
        _chat_id: usize,
    ) -> Result<()> {
        // Send the response through the channel
        self.response_tx.send(response.clone()).await?;
        
        // Persist to storage if available
        if let Some(ref storage_context) = self.storage_context {
            // Extract message content from AssistantResponseType
            if let AssistantResponseType::Message { message, metadata, .. } = &response {
                let role = "assistant".to_string();
                let content = message.clone();
                let token_count = metadata.tokens_used.map(|t| t as i32);
                
                if let Err(e) = storage_context.add_message(
                    role,
                    content,
                    token_count,
                ).await {
                    tracing::warn!("Failed to persist assistant message to storage: {}", e);
                } else {
                    tracing::debug!("Assistant message persisted to storage");
                }
            }
        }
        
        Ok(())
    }
    
    /// Build additional context for task creation
    fn build_task_context(&self, task: &ExtractedTask) -> String {
        let mut context_parts = Vec::new();
        
        // Add task type information
        context_parts.push(format!("Type: {:?}", task.task_type));
        
        // Add confidence level
        if task.confidence < 0.7 {
            context_parts.push(format!("⚠️ Low confidence: {:.0}%", task.confidence * 100.0));
        } else if task.confidence > 0.9 {
            context_parts.push(format!("✅ High confidence: {:.0}%", task.confidence * 100.0));
        }
        
        // Add estimated effort
        if let Some(effort) = &task.estimated_effort {
            let minutes = effort.as_secs() / 60;
            if minutes < 60 {
                context_parts.push(format!("Estimated time: {} minutes", minutes));
            } else {
                context_parts.push(format!("Estimated time: {:.1} hours", minutes as f64 / 60.0));
            }
        }
        
        // Add dependencies
        if !task.dependencies.is_empty() {
            context_parts.push(format!("Dependencies: {}", task.dependencies.join(", ")));
        }
        
        context_parts.join("\n")
    }
    
    /// Execute code from content that might contain code blocks
    async fn execute_code_from_content(
        &self,
        content: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        // Extract code blocks from content
        let (code, language) = self.extract_code_block(content);
        
        if code.is_empty() {
            // No code block found, try to interpret the entire content as code
            return self.execute_code_snippet(content.to_string(), "auto", _chat_id).await;
        }
        
        self.execute_code_snippet(code, &language, _chat_id).await
    }
    
    /// Extract code block and language from markdown-style content
    fn extract_code_block(&self, content: &str) -> (String, String) {
        // Look for ```language...``` blocks
        if let Some(start) = content.find("```") {
            let after_start = &content[start + 3..];
            
            // Extract language identifier
            let language = if let Some(newline_pos) = after_start.find('\n') {
                after_start[..newline_pos].trim().to_string()
            } else {
                "auto".to_string()
            };
            
            // Find the end of the code block
            if let Some(end) = after_start.find("```") {
                let code_start = after_start.find('\n').map(|p| p + 1).unwrap_or(0);
                let code = after_start[code_start..end].trim().to_string();
                return (code, if language.is_empty() { "auto".to_string() } else { language });
            }
        }
        
        // No code block found
        (String::new(), "auto".to_string())
    }
    
    /// Execute a code snippet in the specified language
    async fn execute_code_snippet(
        &self,
        code: String,
        language: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        // Use the integrated editor to execute code
        if let Some(ref tool_executor) = self.tool_executor {
            // Create a code execution command
            let mut args = std::collections::HashMap::new();
            args.insert("language".to_string(), serde_json::json!(language));
            args.insert("code".to_string(), serde_json::json!(code));
            
            let command = crate::tui::chat::core::ParsedCommand {
                command: "run".to_string(),
                args,
                raw_input: format!("run {} {}", language, code),
                is_help: false,
            };
            
            match tool_executor.execute(command).await {
                Ok(result) => {
                    let output = if result.success {
                        format!("✅ **Code Execution Result:**\n\n{}", 
                            serde_json::to_string_pretty(&result.content).unwrap_or_else(|_| result.content.to_string()))
                    } else {
                        format!("❌ **Code Execution Failed:**\n\n{}",
                            serde_json::to_string_pretty(&result.content).unwrap_or_else(|_| result.content.to_string()))
                    };
                    
                    Ok(Some(AssistantResponseType::new_ai_message(
                        output,
                        Some("code-executor".to_string()),
                    )))
                }
                Err(e) => {
                    Ok(Some(AssistantResponseType::new_ai_message(
                        format!("❌ Failed to execute code: {}", e),
                        Some("code-executor".to_string()),
                    )))
                }
            }
        } else {
            // Fallback to using the editor module directly
            let editor_config = crate::tui::chat::editor::EditorConfig::default();
            let editor = crate::tui::chat::editor::CodeEditor::new(editor_config);
            
            // Set the code content
            editor.set_content(code.clone()).await?;
            
            // Execute the code
            match editor.execute().await {
                Ok(result) => {
                    let output = if result.success {
                        format!("✅ **Execution successful:**\n\nOutput:\n```\n{}\n```\n\nExecution time: {}ms",
                            result.output,
                            result.execution_time)
                    } else {
                        format!("❌ **Execution failed:**\n\nError:\n```\n{}\n```\n\nExecution time: {}ms",
                            result.error.unwrap_or_else(|| "Unknown error".to_string()),
                            result.execution_time)
                    };
                    
                    Ok(Some(AssistantResponseType::new_ai_message(
                        output,
                        Some("code-executor".to_string()),
                    )))
                }
                Err(e) => {
                    Ok(Some(AssistantResponseType::new_ai_message(
                        format!("❌ Failed to execute code: {}", e),
                        Some("code-executor".to_string()),
                    )))
                }
            }
        }
    }
    
    /// Export chat history in specified format
    async fn export_chat(
        &self,
        format: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        use crate::tui::chat::state::persistence::{StatePersistence, ExportFormat};
        
        let state = self.chat_state.read().await;
        
        // Determine export format
        let export_format = match format.to_lowercase().as_str() {
            "json" => ExportFormat::Json,
            "markdown" | "md" => ExportFormat::Markdown,
            _ => {
                return Ok(Some(AssistantResponseType::new_ai_message(
                    format!("❌ Unknown format '{}'. Use 'json' or 'markdown'", format),
                    Some("system".to_string()),
                )))
            }
        };
        
        // Create persistence handler
        let persistence = StatePersistence::with_default_dir()?;
        
        // Export the chat
        match persistence.export_chat(&*state, export_format).await {
            Ok(exported) => {
                // For now, return the exported content
                // In a real implementation, this might save to a file or clipboard
                let preview = if exported.len() > 500 {
                    format!("{}...\n\n(Content truncated. Total length: {} characters)", 
                        &exported[..500], exported.len())
                } else {
                    exported.clone()
                };
                
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("✅ **Chat exported as {}:**\n\n```\n{}\n```\n\nUse /save to write to a file", 
                        format, preview),
                    Some("system".to_string()),
                )))
            }
            Err(e) => {
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("❌ Failed to export chat: {}", e),
                    Some("system".to_string()),
                )))
            }
        }
    }
    
    /// Save chat to a file
    async fn save_chat(
        &self,
        filename: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        use crate::tui::chat::state::persistence::ChatPersistence;
        
        let state = self.chat_state.read().await;
        
        // Create persistence handler
        let persistence = ChatPersistence::with_default_dir()?;
        
        // Save the chat
        match persistence.save_chat_with_filename(&*state, filename).await {
            Ok(_) => {
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("✅ Chat saved to: {}", filename),
                    Some("system".to_string()),
                )))
            }
            Err(e) => {
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("❌ Failed to save chat: {}", e),
                    Some("system".to_string()),
                )))
            }
        }
    }
    
    /// Detect programming language from request
    fn detect_language_from_request(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        
        // Language mapping
        let languages = [
            ("rust", vec!["rust", "rs", "cargo"]),
            ("python", vec!["python", "py", "pip", "django", "flask"]),
            ("javascript", vec!["javascript", "js", "node", "npm", "react", "vue"]),
            ("typescript", vec!["typescript", "ts", "angular"]),
            ("go", vec!["go", "golang"]),
            ("java", vec!["java", "spring", "maven"]),
            ("cpp", vec!["c++", "cpp", "c plus plus"]),
            ("csharp", vec!["c#", "csharp", "dotnet", ".net"]),
            ("ruby", vec!["ruby", "rails"]),
            ("php", vec!["php", "laravel"]),
            ("swift", vec!["swift", "ios"]),
            ("kotlin", vec!["kotlin", "android"]),
        ];
        
        for (lang, patterns) in languages.iter() {
            for pattern in patterns {
                if lower.contains(pattern) {
                    return Some(lang.to_string());
                }
            }
        }
        
        None
    }
    
    /// Load chat from a file
    async fn load_chat(
        &self,
        filename: &str,
        _chat_id: usize,
    ) -> Result<Option<AssistantResponseType>> {
        use crate::tui::chat::state::persistence::ChatPersistence;
        
        // Create persistence handler
        let persistence = ChatPersistence::with_default_dir()?;
        
        // Load the chat
        match persistence.load_chat_by_filename(filename).await {
            Ok(loaded_state) => {
                // Replace current chat state with loaded one
                let mut state = self.chat_state.write().await;
                state.messages = loaded_state.messages;
                state.title = loaded_state.title;
                state.id = loaded_state.id;
                
                let message_count = state.messages.len();
                
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("✅ Chat loaded from: {}\n{} messages restored", filename, message_count),
                    Some("system".to_string()),
                )))
            }
            Err(e) => {
                Ok(Some(AssistantResponseType::new_ai_message(
                    format!("❌ Failed to load chat: {}", e),
                    Some("system".to_string()),
                )))
            }
        }
    }
}