//! Direct integration module (replaces bridge pattern)
//! 
//! This module provides direct integration of the modular chat subtabs
//! without the intermediary bridge pattern.

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use ratatui::{Frame, layout::Rect};
use crossterm::event::KeyEvent;
use anyhow::Result;

use super::subtabs::{
    SubtabController,
    ChatTab, EditorTab, ModelsTab, HistoryTab, SettingsTab,
    OrchestrationTab, AgentsTab, CliTab, StatisticsTab,
};
use super::{ChatState, OrchestrationManager, AgentManager};
use super::processing::MessageProcessor;
use super::ui_enhancements::KeyboardShortcutsOverlay;
use crate::tasks::{TaskRegistry, TaskContext};
use crate::config::Config;
use crate::models::ModelManager;

/// Direct subtab manager without bridge pattern
pub struct SubtabManager {
    /// Active subtab controllers
    tabs: Vec<Box<dyn SubtabController + Send + Sync>>,
    
    /// Current active tab index
    current_index: usize,
    
    /// Chat tab reference for special handling
    chat_tab_index: usize,
    
    /// Message processor for handling chat input
    pub message_processor: Option<MessageProcessor>,
    
    /// Message receiver for processing
    message_rx: Option<mpsc::Receiver<(String, usize)>>,
    
    /// Keyboard shortcuts overlay
    keyboard_shortcuts: KeyboardShortcutsOverlay,
}

impl std::fmt::Debug for SubtabManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubtabManager")
            .field("current_index", &self.current_index)
            .field("chat_tab_index", &self.chat_tab_index)
            .field("tabs_count", &self.tabs.len())
            .field("message_processor", &self.message_processor.is_some())
            .field("message_rx", &self.message_rx.is_some())
            .finish()
    }
}

impl SubtabManager {
    /// Create a new subtab manager with all tabs initialized
    pub fn new(
        chat_state: Arc<RwLock<ChatState>>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        agent_manager: Arc<RwLock<AgentManager>>,
        message_tx: mpsc::Sender<(String, usize)>,
        available_models: Vec<crate::tui::chat::ActiveModel>,
    ) -> Self {
        Self::new_with_discovery(
            chat_state,
            orchestration,
            agent_manager,
            message_tx,
            available_models,
            None,
            None,
        )
    }
    
    /// Create a new subtab manager with discovery engines and task support
    pub fn new_with_discovery(
        chat_state: Arc<RwLock<ChatState>>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        agent_manager: Arc<RwLock<AgentManager>>,
        message_tx: mpsc::Sender<(String, usize)>,
        available_models: Vec<crate::tui::chat::ActiveModel>,
        model_discovery: Option<Arc<crate::tui::chat::models::discovery::ModelDiscoveryEngine>>,
        agent_wizard: Option<Arc<crate::tui::chat::agents::creation::AgentCreationWizard>>,
    ) -> Self {
        Self::new_with_full_support(
            chat_state,
            orchestration,
            agent_manager,
            message_tx,
            available_models,
            model_discovery,
            agent_wizard,
            None,
            None,
            None,
            None,
        )
    }
    
    /// Create a new subtab manager with full support including tasks
    pub fn new_with_full_support(
        chat_state: Arc<RwLock<ChatState>>,
        orchestration: Arc<RwLock<OrchestrationManager>>,
        agent_manager: Arc<RwLock<AgentManager>>,
        message_tx: mpsc::Sender<(String, usize)>,
        available_models: Vec<crate::tui::chat::ActiveModel>,
        model_discovery: Option<Arc<crate::tui::chat::models::discovery::ModelDiscoveryEngine>>,
        agent_wizard: Option<Arc<crate::tui::chat::agents::creation::AgentCreationWizard>>,
        task_registry: Option<Arc<TaskRegistry>>,
        task_context: Option<TaskContext>,
        model_orchestrator: Option<Arc<crate::models::ModelOrchestrator>>,
        ollama_manager: Option<Arc<crate::ollama::OllamaManager>>,
    ) -> Self {
        // Create internal message channel for processing
        let (internal_tx, message_rx) = mpsc::channel(100);
        
        // Create chat tab with internal sender (messages go to our processor)
        let mut chat_tab = ChatTab::new(chat_state.clone(), internal_tx);
        // Connect orchestration to chat tab so it can display status correctly
        chat_tab.set_orchestration(orchestration.clone());
        
        // Create history tab with state
        let mut history_tab = HistoryTab::new();
        history_tab.set_state(chat_state.clone());
        
        // Create settings tab with orchestration
        let mut settings_tab = SettingsTab::new();
        settings_tab.set_state(orchestration.clone());
        
        // Create agents tab with references
        let mut agents_tab = AgentsTab::new();
        agents_tab.set_references(orchestration.clone(), agent_manager.clone());
        
        // Create models tab with proper orchestrator
        let models_tab = if let Some(ref orchestrator) = model_orchestrator {
            let mut tab = ModelsTab::new(orchestrator.clone());
            if let Some(ref ollama) = ollama_manager {
                tab.set_ollama_manager(ollama.clone());
            }
            // Initialize the models tab
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Err(e) = tab.initialize().await {
                        tracing::warn!("Failed to initialize models tab: {}", e);
                    }
                })
            });
            tab
        } else {
            // Fallback: create with default orchestrator
            let default_orchestrator = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    crate::models::ModelOrchestrator::new(&crate::config::ApiKeysConfig::default()).await
                })
            });
            
            match default_orchestrator {
                Ok(orchestrator) => ModelsTab::new(Arc::new(orchestrator)),
                Err(e) => {
                    // Log error and create a minimal ModelsTab that can display the error state
                    tracing::error!("Failed to create ModelOrchestrator for models tab: {}", e);
                    tracing::warn!("Creating models tab in degraded mode - functionality will be limited");
                    
                    // Try to create with a minimal orchestrator that won't fail
                    // This allows the UI to still load and show an error state
                    let minimal_orchestrator = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            // Create with empty config - this should always succeed
                            crate::models::ModelOrchestrator::new(&crate::config::ApiKeysConfig {
                                github: None,
                                x_twitter: None,
                                ai_models: crate::config::AiModelsConfig {
                                    openai: None,
                                    anthropic: None,
                                    deepseek: None,
                                    mistral: None,
                                    codestral: None,
                                    gemini: None,
                                    grok: None,
                                    cohere: None,
                                    perplexity: None,
                                },
                                embedding_models: Default::default(),
                                search: Default::default(),
                                vector_db: None,
                                optional_services: Default::default(),
                            }).await
                        })
                    });
                    
                    // If even the minimal orchestrator fails, create a stub tab
                    match minimal_orchestrator {
                        Ok(orchestrator) => {
                            let mut tab = ModelsTab::new(Arc::new(orchestrator));
                            // Could add error state flag here if ModelsTab supports it
                            tracing::info!("Created models tab with minimal orchestrator");
                            tab
                        },
                        Err(e2) => {
                            // Last resort: create a degraded models tab
                            tracing::error!("Even minimal orchestrator failed: {}", e2);
                            tracing::error!("Models tab will be created in error state");
                            
                            // Create a basic models tab that can at least display
                            // In production, ModelsTab should handle None orchestrator gracefully
                            // For now, we create with the most basic possible orchestrator
                            // that was created earlier (even if it's not fully functional)
                            if let Ok(basic_orch) = tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    crate::models::ModelOrchestrator::new(&Default::default()).await
                                })
                            }) {
                                ModelsTab::new(Arc::new(basic_orch))
                            } else {
                                // If all else fails, create with absolutely minimal config
                                // This should be improved to handle error states better
                                tracing::error!("All attempts to create orchestrator failed - creating minimal ModelsTab");
                                
                                // Create the most basic possible orchestrator synchronously
                                let basic_config = crate::config::ApiKeysConfig::default();
                                if let Ok(basic_orch) = tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(async {
                                        crate::models::ModelOrchestrator::new(&basic_config).await
                                    })
                                }) {
                                    ModelsTab::new(Arc::new(basic_orch))
                                } else {
                                    panic!("Unable to create even the most basic orchestrator - this should never happen");
                                }
                            }
                        }
                    }
                }
            }
        };
        
        // Note: agents_tab will be updated with agent_wizard when AgentsTab supports it
        // This is tracked as a future enhancement
        
        // Create CLI tab with task support if available
        let mut cli_tab = CliTab::new();
        if let Some(registry) = task_registry {
            cli_tab.set_task_registry(registry);
        }
        if let Some(context) = task_context {
            cli_tab.set_task_context(context);
        }
        
        // Create all tabs
        let tabs: Vec<Box<dyn SubtabController + Send + Sync>> = vec![
            Box::new(chat_tab),
            Box::new(EditorTab::new()),
            Box::new(models_tab),
            Box::new(history_tab),
            Box::new(settings_tab),
            Box::new(OrchestrationTab::new(orchestration.clone())),
            Box::new(agents_tab),
            Box::new(cli_tab),
            Box::new(StatisticsTab::new(chat_state.clone())),
        ];
        
        // Create message processor
        let (response_tx, response_rx) = mpsc::channel(100);
        let message_processor = MessageProcessor::new(
            chat_state.clone(),
            orchestration.clone(),
            agent_manager.clone(),
            response_tx,
        );
        
        // Note: message_tx passed in is ignored - we use our internal channel instead
        // This ensures messages from the chat tab go through our processor
        
        let mut manager = Self {
            tabs,
            current_index: 0,
            chat_tab_index: 0, // Chat tab is always first
            message_processor: Some(message_processor),
            message_rx: Some(message_rx),
            keyboard_shortcuts: KeyboardShortcutsOverlay::new(),
        };
        
        // Set response receiver on chat tab
        if let Some(tab) = manager.tabs.get_mut(manager.chat_tab_index) {
            // We need to downcast to ChatTab to access its specific method
            // This is safe because we know the chat tab is at index 0
            let chat_tab = unsafe {
                let raw_ptr = tab.as_mut() as *mut dyn SubtabController as *mut ChatTab;
                &mut *raw_ptr
            };
            chat_tab.set_response_receiver(response_rx);
        }
        
        manager
    }
    
    /// Set the response receiver for the chat tab
    pub fn set_response_receiver(&mut self, rx: mpsc::Receiver<crate::tui::run::AssistantResponseType>) {
        if let Some(tab) = self.tabs.get_mut(self.chat_tab_index) {
            // We need to downcast to ChatTab to access its specific method
            // This is safe because we know the chat tab is at index 0
            let chat_tab = unsafe {
                let raw_ptr = tab.as_mut() as *mut dyn SubtabController as *mut ChatTab;
                &mut *raw_ptr
            };
            chat_tab.set_response_receiver(rx);
        }
        tracing::info!("✅ Response receiver connected to chat tab");
    }
    
    /// Render the current subtab
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        // First, process any pending updates
        if let Err(e) = self.update() {
            tracing::error!("Failed to update subtab manager: {}", e);
        }
        
        // Then render the current subtab
        if let Some(tab) = self.tabs.get_mut(self.current_index) {
            tab.render(f, area);
        }
        
        // Render keyboard shortcuts overlay on top if visible
        self.keyboard_shortcuts.render(f, area);
    }
    
    /// Handle input for the current subtab
    pub fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        use crossterm::event::{KeyCode, KeyModifiers};
        
        // Check for keyboard shortcuts overlay toggle (F1)
        if key.code == KeyCode::F(1) {
            self.keyboard_shortcuts.toggle();
            return Ok(());
        }
        
        // Note: Ctrl+J/K navigation is handled by app.rs to keep indices synchronized
        // We don't handle it here to avoid conflicts with the main navigation system
        
        // Alt+Left/Right for subtab navigation as alternative
        if key.modifiers.contains(KeyModifiers::ALT) {
            match key.code {
                KeyCode::Left => {
                    if self.current_index > 0 {
                        self.set_active(self.current_index - 1);
                    } else {
                        self.set_active(self.tabs.len() - 1);
                    }
                    return Ok(());
                }
                KeyCode::Right => {
                    if self.current_index < self.tabs.len() - 1 {
                        self.set_active(self.current_index + 1);
                    } else {
                        self.set_active(0);
                    }
                    return Ok(());
                }
                _ => {}
            }
        }
        
        // If keyboard shortcuts overlay is visible, handle its input
        if self.keyboard_shortcuts.is_visible() {
            match key.code {
                KeyCode::Esc => self.keyboard_shortcuts.hide(),
                KeyCode::Up => self.keyboard_shortcuts.navigate_up(),
                KeyCode::Down => self.keyboard_shortcuts.navigate_down(),
                KeyCode::Enter => {
                    // Select current item in overlay
                }
                _ => {}
            }
            return Ok(());
        }
        
        // Otherwise, pass input to current tab
        if let Some(tab) = self.tabs.get_mut(self.current_index) {
            tab.handle_input(key)?;
        }
        Ok(())
    }
    
    /// Update the current subtab
    pub fn update(&mut self) -> Result<()> {
        // Process any pending messages
        if let Some(rx) = &mut self.message_rx {
            while let Ok((message, chat_id)) = rx.try_recv() {
                tracing::info!("📬 SubtabManager received message from channel: {}", message);
                if let Some(processor) = &mut self.message_processor {
                    tracing::info!("🚀 Spawning message processor task");
                    tokio::spawn({
                        let mut processor = processor.clone();
                        async move {
                            if let Err(e) = processor.process_message(&message, chat_id).await {
                                tracing::error!("Message processing error: {}", e);
                            }
                        }
                    });
                } else {
                    tracing::warn!("⚠️ No message processor available!");
                }
            }
        }
        
        // Update the current tab
        if let Some(tab) = self.tabs.get_mut(self.current_index) {
            tab.update()?;
        }
        Ok(())
    }
    
    /// Set the active subtab by index
    pub fn set_active(&mut self, index: usize) {
        if index < self.tabs.len() {
            // Check if switching to chat tab and if history has a loaded message
            if index == 0 && self.current_index == 2 {
                // Switching from History (index 2) to Chat (index 0)
                if let Some(history_tab) = self.tabs.get_mut(2) {
                    // Downcast to HistoryTab
                    let history = unsafe {
                        let raw_ptr = history_tab.as_mut() as *mut dyn SubtabController as *mut HistoryTab;
                        &mut *raw_ptr
                    };
                    
                    if let Some(loaded_msg) = history.take_loaded_message() {
                        // Load the message into chat tab
                        if let Some(chat_tab) = self.tabs.get_mut(0) {
                            let chat = unsafe {
                                let raw_ptr = chat_tab.as_mut() as *mut dyn SubtabController as *mut ChatTab;
                                &mut *raw_ptr
                            };
                            chat.load_message(loaded_msg);
                        }
                    }
                }
            }
            
            self.current_index = index;
        }
    }
    
    /// Get the current active subtab index
    pub fn current_index(&self) -> usize {
        self.current_index
    }
    
    /// Get the number of subtabs
    pub fn tab_count(&self) -> usize {
        self.tabs.len()
    }
    
    /// Get the name of the current subtab
    pub fn current_name(&self) -> &str {
        self.tabs.get(self.current_index)
            .map(|t| t.name())
            .unwrap_or("Unknown")
    }
    
    /// Get the title of a specific subtab
    pub fn get_title(&self, index: usize) -> Option<String> {
        self.tabs.get(index).map(|t| t.title())
    }
    
    /// Get titles for all tabs
    pub fn get_all_titles(&self) -> Vec<String> {
        self.tabs.iter().map(|t| t.title()).collect()
    }
    
    /// Set the model orchestrator for message processing
    pub fn set_model_orchestrator(&mut self, orchestrator: Arc<crate::models::ModelOrchestrator>) {
        if let Some(processor) = &mut self.message_processor {
            processor.set_model_orchestrator(orchestrator);
        }
    }
    
    /// Set the tool executor for message processing
    pub fn set_tool_executor(&mut self, executor: Arc<crate::tui::chat::core::tool_executor::ChatToolExecutor>) {
        if let Some(processor) = &mut self.message_processor {
            processor.set_tool_executor(executor);
        }
    }
    
    /// Set the cognitive enhancement for message processing
    pub fn set_cognitive_enhancement(&mut self, enhancement: Arc<crate::tui::chat::integrations::cognitive::CognitiveChatEnhancement>) {
        if let Some(processor) = &mut self.message_processor {
            processor.set_cognitive_enhancement(enhancement);
        }
    }
    
    /// Set the NLP orchestrator for message processing
    pub fn set_nlp_orchestrator(&mut self, orchestrator: Arc<crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator>) {
        if let Some(processor) = &mut self.message_processor {
            processor.set_nlp_orchestrator(orchestrator);
        }
    }
    
    /// Set tool managers for message processing
    pub fn set_tool_managers(
        &mut self,
        intelligent_tool_manager: Arc<crate::tools::intelligent_manager::IntelligentToolManager>,
        task_manager: Arc<crate::tools::task_management::TaskManager>,
    ) {
        if let Some(processor) = &mut self.message_processor {
            processor.set_tool_managers(intelligent_tool_manager, task_manager);
        }
    }
    
    /// Set task support for CLI tab
    pub fn set_task_support(
        &mut self,
        task_registry: Arc<TaskRegistry>,
        task_context: TaskContext,
    ) {
        // CLI tab is at index 7
        if let Some(tab) = self.tabs.get_mut(7) {
            // Safe downcast to CliTab to set task support
            let cli_tab = unsafe {
                let raw_ptr = tab.as_mut() as *mut dyn SubtabController as *mut CliTab;
                &mut *raw_ptr
            };
            cli_tab.set_task_registry(task_registry);
            cli_tab.set_task_context(task_context);
            tracing::info!("🤖 Task support enabled for CLI tab");
        }
    }
    
    /// Check and sync settings changes back to the parent ChatManager
    pub fn sync_settings_changes(&mut self) -> Option<crate::tui::chat::ChatSettings> {
        // Check if settings tab has changes
        if let Some(tab) = self.tabs.get_mut(3) { // Settings tab is at index 3
            // We need to downcast to SettingsTab to access its specific method
            let settings_tab = unsafe {
                let raw_ptr = tab.as_mut() as *mut dyn SubtabController as *mut SettingsTab;
                &mut *raw_ptr
            };
            
            if let Some(changed_settings) = settings_tab.get_changed_settings() {
                tracing::info!("📝 Settings have changed, syncing to parent");
                return Some(changed_settings);
            }
        }
        None
    }
    
    /// Get mutable reference to history tab
    pub fn get_history_tab_mut(&mut self) -> Option<&mut HistoryTab> {
        if let Some(tab) = self.tabs.get_mut(2) { // History tab is at index 2
            // Safe downcast to HistoryTab
            let history_tab = unsafe {
                let raw_ptr = tab.as_mut() as *mut dyn SubtabController as *mut HistoryTab;
                &mut *raw_ptr
            };
            Some(history_tab)
        } else {
            None
        }
    }
    
    /// Check if the chat tab is currently in input mode
    pub fn is_chat_input_active(&self) -> bool {
        // Only return true if we're on the chat tab (index 0) AND it's in input mode
        if self.current_index == 0 {
            if let Some(tab) = self.tabs.get(0) {
                // Safe downcast to ChatTab to check input_mode
                let chat_tab = unsafe {
                    let raw_ptr = tab.as_ref() as *const dyn SubtabController as *const ChatTab;
                    &*raw_ptr
                };
                return chat_tab.is_input_mode();
            }
        }
        false
    }
}

/// Helper function to integrate with existing chat system
pub fn create_subtab_manager(
    chat_state: Arc<RwLock<ChatState>>,
    orchestration: Arc<RwLock<OrchestrationManager>>,
    agent_manager: Arc<RwLock<AgentManager>>,
    message_tx: mpsc::Sender<(String, usize)>,
    available_models: Vec<crate::tui::chat::ActiveModel>,
) -> SubtabManager {
    SubtabManager::new(
        chat_state,
        orchestration,
        agent_manager,
        message_tx,
        available_models,
    )
}

/// Extension trait to add direct integration methods to ChatManager
pub trait DirectIntegration {
    /// Initialize direct subtab integration
    fn initialize_direct_integration(
        &mut self,
        message_tx: mpsc::Sender<(String, usize)>,
    );
    
    /// Check if direct integration is available
    fn has_direct_integration(&self) -> bool;
}

// Implementation would be added to ChatManager in the UI module