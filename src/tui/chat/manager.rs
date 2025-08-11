//! ChatManager - Thin coordinator for the chat system
//! 
//! This will become the main entry point after refactoring,
//! coordinating all subsystems without containing business logic

use std::sync::Arc;
use anyhow::{Result, Context};
use tokio::sync::{RwLock, mpsc};

// Subsystem imports
use crate::tui::chat::{
    state::{ChatState, SessionManager},
    orchestration::OrchestrationManager,
    agents::AgentManager,
    context::SmartContextManager,
    subtabs::{SubtabController},
    integrations::{CognitiveIntegration, ToolIntegration, NlpIntegration},
    handlers::{InputProcessor, CommandProcessor, NaturalLanguageHandler},
    processing::MessageProcessor,
    storage_context::ChatStorageContext,
};
use crate::tui::run::AssistantResponseType;

/// The refactored ChatManager - a thin coordinator
/// 
/// Target: <500 lines (down from 14,151)
pub struct ChatManager {
    /// State management
    pub state: Arc<RwLock<ChatState>>,
    pub session_manager: Arc<RwLock<SessionManager>>,
    
    /// Orchestration subsystem
    pub orchestration: Arc<RwLock<OrchestrationManager>>,
    
    /// Agent management
    pub agents: Arc<RwLock<AgentManager>>,
    
    /// Context management
    pub context: Arc<RwLock<SmartContextManager>>,
    
    /// Storage context for persistence
    pub storage_context: Option<Arc<ChatStorageContext>>,
    
    /// Subtab controllers
    pub subtabs: Vec<Box<dyn SubtabController>>,
    
    /// Current active subtab
    pub active_subtab: usize,
    
    /// Input/Output channels
    pub message_tx: mpsc::Sender<(String, usize)>,
    pub response_rx: mpsc::Receiver<AssistantResponseType>,
    response_tx: mpsc::Sender<AssistantResponseType>,
    
    /// Processors
    input_processor: Arc<InputProcessor>,
    command_processor: Arc<CommandProcessor>,
    message_processor: Arc<RwLock<MessageProcessor>>,
    
    /// Integrations
    cognitive: Arc<CognitiveIntegration>,
    tools: Arc<ToolIntegration>,
    nlp: Arc<NlpIntegration>,
}

impl ChatManager {
    /// Create a new ChatManager with all dependencies
    pub async fn new(
        state: Arc<RwLock<ChatState>>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        cognitive: Arc<CognitiveIntegration>,
        tools: Arc<ToolIntegration>,
        nlp: Arc<NlpIntegration>,
        message_tx: mpsc::Sender<(String, usize)>,
        response_rx: mpsc::Receiver<AssistantResponseType>,
    ) -> Result<Self> {
        // Create response channel for internal use
        let (response_tx, _internal_rx) = mpsc::channel(100);
        
        // Create session manager
        let session_manager = Arc::new(RwLock::new(
            SessionManager::new()
        ));
        
        // Create agent manager and initialize pool
        let agents = Arc::new(RwLock::new(
            AgentManager::enabled()
        ));
        
        // Initialize the agent pool
        {
            let mut agent_manager = agents.write().await;
            if let Err(e) = agent_manager.initialize_agent_pool().await {
                tracing::warn!("Failed to initialize agent pool: {}", e);
            } else {
                tracing::info!("Agent pool initialized successfully");
            }
        }
        
        // Create context manager with 8000 token limit
        let context = Arc::new(RwLock::new(
            SmartContextManager::new(8000)
        ));
        
        // Create processors
        let input_processor = Arc::new(
            InputProcessor::new(message_tx.clone())
        );
        
        let command_processor = Arc::new(
            CommandProcessor::new(state.clone(), orchestration.clone(), response_tx.clone())
        );
        
        // Natural language handler is used by subtabs, not directly by ChatManager
        // It will be created when subtabs are implemented
        
        let message_processor = Arc::new(RwLock::new(
            MessageProcessor::new(
                state.clone(),
                orchestration.clone(),
                agents.clone(),
                response_tx.clone(),
            )
        ));
        
        // Create subtabs
        let subtabs: Vec<Box<dyn SubtabController>> = vec![
            Box::new(crate::tui::chat::subtabs::ChatTab::new(
                state.clone(),
                message_tx.clone(),
            )),
            Box::new(crate::tui::chat::subtabs::AgentsTab::new()),
            Box::new(crate::tui::chat::subtabs::OrchestrationTab::new(
                orchestration.clone(),
            )),
            // Models tab needs proper orchestrator - will be set up by modular system
            // For now, create with a default orchestrator
            Box::new({
                let default_orchestrator = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        crate::models::ModelOrchestrator::new(&crate::config::ApiKeysConfig::default()).await
                    })
                });
                match default_orchestrator {
                    Ok(orchestrator) => crate::tui::chat::subtabs::ModelsTab::new(std::sync::Arc::new(orchestrator)),
                    Err(e) => {
                        tracing::error!("Failed to create ModelOrchestrator for models tab: {}", e);
                        panic!("Cannot initialize models tab without ModelOrchestrator");
                    }
                }
            }),
            Box::new(crate::tui::chat::subtabs::HistoryTab::new()),
            Box::new(crate::tui::chat::subtabs::SettingsTab::new()),
        ];
        
        // Initialize render state manager for synchronous rendering
        // This needs to be done after all components are created
        let state_clone = state.clone();
        let orch_clone = orchestration.clone();
        let agents_clone = agents.clone();
        let tools_clone = tools.clone();
        tokio::spawn(async move {
            if let Err(e) = crate::tui::chat::rendering::initialize_render_state(
                Some(state_clone.clone()),
                Some(orch_clone.clone()),
                Some(agents_clone.clone()),
                Some(tools_clone.clone()),
            ).await {
                tracing::error!("Failed to initialize render state: {}", e);
            }
            
            // Start background sync task
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
            loop {
                interval.tick().await;
                
                // Get the render state manager and sync
                let manager = crate::tui::chat::rendering::get_render_state_manager();
                let sync_result = {
                    if let Ok(mgr) = manager.try_lock() {
                        mgr.sync_state().await
                    } else {
                        Ok(())
                    }
                };
                if let Err(e) = sync_result {
                    tracing::debug!("Failed to sync render state: {}", e);
                }
            }
        });
        
        Ok(Self {
            state,
            session_manager,
            orchestration,
            agents,
            context,
            storage_context: None,  // Will be set with set_storage_context()
            subtabs,
            active_subtab: 0,
            message_tx,
            response_rx,
            response_tx,
            input_processor,
            command_processor,
            message_processor,
            cognitive,
            tools,
            nlp,
        })
    }
    
    /// Create a properly initialized instance with minimal defaults
    /// This is used when full initialization is not yet available
    pub fn with_defaults() -> Self {
        // Create proper channels
        let (message_tx, _) = mpsc::channel(100);
        let (response_tx, response_rx) = mpsc::channel(100);
        
        // Initialize with proper defaults instead of placeholders
        let state = Arc::new(RwLock::new(ChatState::new(0, "Main Chat".to_string())));
        let orchestration = Arc::new(RwLock::new(OrchestrationManager::default()));
        let agents = Arc::new(RwLock::new(AgentManager::new()));
        let session_manager = Arc::new(RwLock::new(SessionManager::new()));
        
        // Create tools with defaults
        let tool_manager = Arc::new(crate::tools::intelligent_manager::IntelligentToolManager::new());
        let task_manager = Arc::new(crate::tools::task_management::TaskManager::new());
        
        Self {
            state: state.clone(),
            session_manager,
            orchestration: orchestration.clone(),
            agents: agents.clone(),
            context: Arc::new(RwLock::new(SmartContextManager::new(8192))),
            storage_context: None,
            subtabs: vec![],
            active_subtab: 0,
            input_processor: Arc::new(InputProcessor::new(message_tx.clone())),
            command_processor: Arc::new(CommandProcessor::new(
                state.clone(),
                orchestration.clone(),
                response_tx.clone(),
            )),
            message_processor: Arc::new(RwLock::new(MessageProcessor::new(
                state.clone(),
                orchestration.clone(),
                agents.clone(),
                response_tx.clone(),
            ))),
            message_tx,
            response_rx,
            response_tx,
            cognitive: Arc::new(CognitiveIntegration::new()),
            tools: Arc::new(ToolIntegration::new(
                tool_manager,
                task_manager,
            )),
            nlp: Arc::new(NlpIntegration::new()),
        }
    }
    
    /// Render the chat interface
    pub fn render(&mut self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        // Delegate to active subtab
        if let Some(subtab) = self.subtabs.get_mut(self.active_subtab) {
            subtab.render(f, area);
        }
    }
    
    /// Handle input
    pub async fn handle_input(&mut self, key: crossterm::event::KeyEvent) -> Result<()> {
        use crossterm::event::{KeyCode, KeyModifiers};
        
        // Handle global shortcuts
        match (key.code, key.modifiers) {
            // Tab switching
            (KeyCode::Tab, KeyModifiers::NONE) => {
                self.switch_subtab((self.active_subtab + 1) % self.subtabs.len());
                return Ok(());
            }
            (KeyCode::BackTab, KeyModifiers::SHIFT) => {
                self.switch_subtab(
                    if self.active_subtab == 0 {
                        self.subtabs.len() - 1
                    } else {
                        self.active_subtab - 1
                    }
                );
                return Ok(());
            }
            // Direct tab access
            (KeyCode::Char('1'), KeyModifiers::CONTROL) => {
                self.switch_subtab(0); // Chat
                return Ok(());
            }
            (KeyCode::Char('2'), KeyModifiers::CONTROL) => {
                self.switch_subtab(1); // Agents
                return Ok(());
            }
            (KeyCode::Char('3'), KeyModifiers::CONTROL) => {
                self.switch_subtab(2); // Orchestration
                return Ok(());
            }
            (KeyCode::Char('4'), KeyModifiers::CONTROL) => {
                self.switch_subtab(3); // Tools
                return Ok(());
            }
            (KeyCode::Char('5'), KeyModifiers::CONTROL) => {
                self.switch_subtab(4); // Memory
                return Ok(());
            }
            (KeyCode::Char('6'), KeyModifiers::CONTROL) => {
                self.switch_subtab(5); // Settings
                return Ok(());
            }
            _ => {}
        }
        
        // Delegate to active subtab
        if let Some(subtab) = self.subtabs.get_mut(self.active_subtab) {
            subtab.handle_input(key)?;
        }
        Ok(())
    }
    
    /// Update state
    pub async fn update(&mut self) -> Result<()> {
        // Process any pending responses
        while let Ok(response) = self.response_rx.try_recv() {
            // Add response to chat state
            let mut state = self.state.write().await;
            state.messages.push(response);
        }
        
        // Update active subtab
        if let Some(subtab) = self.subtabs.get_mut(self.active_subtab) {
            subtab.update()?;
        }
        
        Ok(())
    }
    
    /// Switch to a different subtab
    pub fn switch_subtab(&mut self, index: usize) {
        if index < self.subtabs.len() {
            self.active_subtab = index;
        }
    }
}

/// Helper methods for ChatManager
impl ChatManager {
    /// Set the tool executor for command processing
    pub async fn set_tool_executor(&mut self, executor: Arc<crate::tui::chat::core::tool_executor::ChatToolExecutor>) {
        // Get mutable access to the message processor
        // Set tool executor on the message processor (now wrapped in RwLock)
        let mut processor = self.message_processor.write().await;
        processor.set_tool_executor(executor);
        tracing::info!("✅ Tool executor connected to message processor");
    }
    
    /// Get the name of the current subtab
    pub fn current_subtab_name(&self) -> &str {
        match self.active_subtab {
            0 => "Chat",
            1 => "Agents",
            2 => "Orchestration",
            3 => "Tools",
            4 => "Memory",
            5 => "Settings",
            _ => "Unknown",
        }
    }
    
    /// Send a message to be processed
    pub async fn send_message(&self, message: String) -> Result<()> {
        let chat_index = self.state.read().await.id.parse::<usize>().unwrap_or(0);
        self.message_tx.send((message, chat_index)).await
            .context("Failed to send message")?;
        Ok(())
    }
    
    /// Get current chat messages
    pub async fn get_messages(&self) -> Vec<AssistantResponseType> {
        self.state.read().await.messages.clone()
    }
    
    /// Clear chat messages
    pub async fn clear_messages(&self) -> Result<()> {
        self.state.write().await.messages.clear();
        Ok(())
    }
    
    /// Export chat to markdown
    pub async fn export_to_markdown(&self) -> Result<String> {
        let state = self.state.read().await;
        let mut markdown = format!("# Chat Export - {}\n\n", state.title);
        
        for msg in &state.messages {
            match msg {
                AssistantResponseType::Message { author, message, timestamp, .. } => {
                    markdown.push_str(&format!("**{}** (_{}_):\n{}\n\n", author, timestamp, message));
                }
                AssistantResponseType::UserMessage { content, timestamp, .. } => {
                    markdown.push_str(&format!("**User** (_{}_):\n{}\n\n", timestamp, content));
                }
                AssistantResponseType::SystemMessage { content, timestamp, .. } => {
                    markdown.push_str(&format!("_System ({}): {}_\n\n", timestamp, content));
                }
                _ => {}
            }
        }
        
        Ok(markdown)
    }
    
    /// Set the storage context for persistence
    pub async fn set_storage_context(&mut self, storage_context: Arc<ChatStorageContext>) -> Result<()> {
        // Initialize the storage context
        storage_context.initialize().await?;
        
        // Store the context
        self.storage_context = Some(storage_context.clone());
        
        // Auto-start a conversation if none exists
        if storage_context.get_current_conversation_id().await.is_none() {
            let _ = storage_context.start_conversation(
                "New Session".to_string(),
                "default".to_string(),
            ).await;
        }
        
        Ok(())
    }
    
    /// Process message with storage persistence
    pub async fn process_message_with_storage(&self, content: String, role: String) -> Result<()> {
        // Store message in persistent storage if available
        if let Some(storage) = &self.storage_context {
            // Estimate token count (rough approximation)
            let token_count = (content.len() / 4) as i32;
            
            storage.add_message(
                role.clone(),
                content.clone(),
                Some(token_count),
            ).await?;
        }
        
        // Process the message using the MessageProcessor
        let mut processor = self.message_processor.write().await;
        processor.process_message(&content, 0).await?;
        
        Ok(())
    }
    
    /// Get API key from storage
    pub async fn get_api_key(&self, provider: &str) -> Result<Option<String>> {
        if let Some(storage) = &self.storage_context {
            storage.get_api_key(provider).await
        } else {
            Ok(None)
        }
    }
    
    /// Search chat history
    pub async fn search_history(&self, query: &str) -> Result<Vec<crate::storage::chat_history::SearchResult>> {
        if let Some(storage) = &self.storage_context {
            storage.search_history(query, 50).await
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Switch to a different conversation
    pub async fn switch_conversation(&self, conversation_id: String) -> Result<()> {
        if let Some(storage) = &self.storage_context {
            // Switch to the new conversation
            storage.switch_conversation(conversation_id.clone()).await?;
            
            // Clear current messages
            self.clear_messages().await?;
            
            // Load messages from the switched conversation
            let messages = storage.get_conversation_messages(&conversation_id).await?;
            
            // Add loaded messages to the current state
            let mut state = self.state.write().await;
            for msg in messages {
                // Convert ChatMessage to AssistantResponseType
                state.messages.push(AssistantResponseType::ChatMessage {
                    id: msg.id,
                    role: msg.role,
                    content: msg.content,
                    timestamp: msg.timestamp.to_rfc3339(),
                    metadata: crate::tui::run::MessageMetadata {
                        model_used: None,
                        tokens_used: msg.token_count.map(|tc| tc as u32),
                        generation_time_ms: None,
                        confidence_score: None,
                        temperature: None,
                        is_favorited: false,
                        tags: Vec::new(),
                        user_rating: None,
                        created_at: msg.timestamp,
                        last_edited: None,
                        edit_count: 0,
                    },
                });
            }
            
            tracing::info!("Loaded {} messages for conversation {}", 
                state.messages.len(), conversation_id);
        }
        Ok(())
    }
    
    /// Update subtabs with dependencies (call after creation if needed)
    pub fn update_subtab_dependencies(
        &mut self,
        state: Arc<RwLock<ChatState>>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        agents: Arc<RwLock<AgentManager>>,
        session_manager: Arc<RwLock<SessionManager>>,
    ) {
        // This method can be used to update subtab dependencies after creation
        // Currently, the subtabs are created with placeholder data and this can
        // be used to update them with real dependencies if needed
    }
    
    /// Get the current input text from the edit buffer (async version)
    pub async fn get_input_text_async(&self) -> String {
        let state = self.state.read().await;
        state.edit_buffer.clone()
    }
    
    /// Get the current cursor position in the edit buffer (async version)
    pub async fn get_cursor_position_async(&self) -> usize {
        let state = self.state.read().await;
        // Return the end of the edit buffer as cursor position
        // This can be enhanced to track actual cursor position
        state.edit_buffer.len()
    }
    
    /// Get the current input text from the edit buffer (sync version for rendering)
    pub fn get_input_text(&self) -> String {
        // Use try_read for non-blocking access in render context
        self.state.try_read()
            .map(|state| state.edit_buffer.clone())
            .unwrap_or_default()
    }
    
    /// Get the current cursor position in the edit buffer (sync version for rendering)
    pub fn get_cursor_position(&self) -> usize {
        // Use try_read for non-blocking access in render context
        self.state.try_read()
            .map(|state| state.edit_buffer.len())
            .unwrap_or(0)
    }
}