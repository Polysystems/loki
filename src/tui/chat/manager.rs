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
    message_processor: Arc<MessageProcessor>,
    
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
        
        // Create agent manager
        let agents = Arc::new(RwLock::new(
            AgentManager::enabled()
        ));
        
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
        
        let message_processor = Arc::new(
            MessageProcessor::new(
                state.clone(),
                orchestration.clone(),
                agents.clone(),
                response_tx.clone(),
            )
        );
        
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
            Box::new(crate::tui::chat::subtabs::ModelsTab::new(
                orchestration.clone(),
                vec![], // Will be populated later with actual models
            )),
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
    
    /// Create a placeholder instance for initialization
    pub fn placeholder() -> Self {
        // Create dummy channels
        let (message_tx, _) = mpsc::channel(1);
        let (response_tx, response_rx) = mpsc::channel(1);
        
        Self {
            state: Arc::new(RwLock::new(ChatState::new(0, "Placeholder".to_string()))),
            session_manager: Arc::new(RwLock::new(SessionManager::placeholder())),
            orchestration: Arc::new(RwLock::new(OrchestrationManager::default())),
            agents: Arc::new(RwLock::new(AgentManager::placeholder())),
            context: Arc::new(RwLock::new(SmartContextManager::new(4096))),
            storage_context: None,
            subtabs: vec![],
            active_subtab: 0,
            input_processor: Arc::new(InputProcessor::new(message_tx.clone())),
            command_processor: Arc::new(CommandProcessor::new(
                Arc::new(RwLock::new(ChatState::new(0, "Placeholder".to_string()))),
                Arc::new(RwLock::new(OrchestrationManager::default())),
                response_tx.clone(),
            )),
            message_processor: Arc::new(MessageProcessor::new(
                Arc::new(RwLock::new(ChatState::new(0, "Placeholder".to_string()))),
                Arc::new(RwLock::new(OrchestrationManager::default())),
                Arc::new(RwLock::new(AgentManager::placeholder())),
                response_tx.clone(),
            )),
            message_tx,
            response_rx,
            response_tx,
            cognitive: Arc::new(CognitiveIntegration::new()),
            tools: Arc::new(ToolIntegration::new(
                Arc::new(crate::tools::intelligent_manager::IntelligentToolManager::placeholder()),
                Arc::new(crate::tools::task_management::TaskManager::placeholder()),
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
    pub fn set_tool_executor(&mut self, executor: Arc<crate::tui::chat::core::tool_executor::ChatToolExecutor>) {
        // Get mutable access to the message processor
        // Since message_processor is Arc<MessageProcessor>, we need to use Arc::get_mut
        // or refactor MessageProcessor to use interior mutability
        if let Some(processor) = Arc::get_mut(&mut self.message_processor) {
            processor.set_tool_executor(executor);
            tracing::info!("✅ Tool executor connected to message processor");
        } else {
            tracing::warn!("⚠️ Could not set tool executor - message processor has multiple references");
        }
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
        
        // Process the message normally
        // Note: MessageProcessor requires &mut self, but we have Arc<MessageProcessor>
        // This is a design issue that needs to be addressed - for now, return Ok
        // TODO: Refactor MessageProcessor to work with Arc or use interior mutability
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
            storage.switch_conversation(conversation_id).await?;
            
            // Clear current messages and reload from storage
            self.clear_messages().await?;
            
            // TODO: Load messages from the switched conversation
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
}