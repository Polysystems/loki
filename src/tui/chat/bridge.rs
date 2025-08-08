//! Bridge module to connect old chat.rs with new modular structure
//! 
//! This provides a transition layer during refactoring

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use ratatui::{Frame, layout::Rect};
use crossterm::event::KeyEvent;
use anyhow::Result;

use crate::tui::App;
use super::subtabs::{
    SubtabController,
    ChatTab, EditorTab, ModelsTab, HistoryTab, SettingsTab,
    OrchestrationTab, AgentsTab, CliTab,
};
use super::agents::manager::AgentManager;

/// Bridge structure to manage new subtabs during transition
pub struct ChatSubtabBridge {
    /// Chat tab (stored separately for direct access)
    chat_tab: ChatTab,
    
    /// Other subtab controllers
    other_subtabs: Vec<Box<dyn SubtabController + Send + Sync>>,
    
    /// Current active subtab index
    current_index: usize,
}

impl ChatSubtabBridge {
    /// Create a new bridge with all subtabs initialized
    pub fn new(
        chat_state: Arc<RwLock<super::state::ChatState>>,
        orchestration: Arc<RwLock<super::orchestration::OrchestrationManager>>,
        message_tx: mpsc::Sender<(String, usize)>,
        available_models: Vec<crate::tui::chat::ActiveModel>,
    ) -> Self {
        let chat_tab = ChatTab::new(chat_state.clone(), message_tx);
        
        // Create history tab with state
        let mut history_tab = HistoryTab::new();
        history_tab.set_state(chat_state.clone());
        
        // Create settings tab with orchestration
        let mut settings_tab = SettingsTab::new();
        settings_tab.set_state(orchestration.clone());
        
        // Create agents tab with references
        let mut agents_tab = AgentsTab::new();
        
        // Create and initialize the editor tab
        let mut editor_tab = EditorTab::new();
        // Initialize the editor asynchronously in a blocking task
        let editor_tab_initialized = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Err(e) = editor_tab.initialize_editor().await {
                    tracing::warn!("Failed to initialize editor: {}", e);
                }
                editor_tab
            })
        });
        
        let other_subtabs: Vec<Box<dyn SubtabController + Send + Sync>> = vec![
            Box::new(editor_tab_initialized),
            Box::new(ModelsTab::new(orchestration.clone(), available_models)),
            Box::new(history_tab),
            Box::new(settings_tab),
            Box::new(OrchestrationTab::new(orchestration.clone())),
            Box::new(agents_tab),
            Box::new(CliTab::new()),
        ];
        
        Self {
            chat_tab,
            other_subtabs,
            current_index: 0,
        }
    }
    
    /// Set the response receiver for the chat tab
    pub fn set_response_receiver(&mut self, rx: mpsc::Receiver<super::super::run::AssistantResponseType>) {
        self.chat_tab.set_response_receiver(rx);
        tracing::info!("✅ Response receiver connected to chat tab");
    }
    
    /// Render the current subtab
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        match self.current_index {
            0 => self.chat_tab.render(f, area),
            n => {
                if let Some(subtab) = self.other_subtabs.get_mut(n - 1) {
                    subtab.render(f, area);
                }
            }
        }
    }
    
    /// Handle input for the current subtab
    pub fn handle_input(&mut self, key: KeyEvent) -> Result<()> {
        match self.current_index {
            0 => self.chat_tab.handle_input(key),
            n => {
                if let Some(subtab) = self.other_subtabs.get_mut(n - 1) {
                    subtab.handle_input(key)?;
                }
                Ok(())
            }
        }
    }
    
    /// Update the current subtab
    pub fn update(&mut self) -> Result<()> {
        match self.current_index {
            0 => self.chat_tab.update(),
            n => {
                if let Some(subtab) = self.other_subtabs.get_mut(n - 1) {
                    subtab.update()?;
                }
                Ok(())
            }
        }
    }
    
    /// Set the active subtab by index
    pub fn set_active(&mut self, index: usize) {
        if index <= self.other_subtabs.len() {
            self.current_index = index;
        }
    }
    
    /// Get the name of the current subtab
    pub fn current_name(&self) -> &str {
        match self.current_index {
            0 => self.chat_tab.name(),
            n => self.other_subtabs
                .get(n - 1)
                .map(|s| s.name())
                .unwrap_or("Unknown"),
        }
    }
    
    /// Get the current active subtab index
    pub fn current_index(&self) -> usize {
        self.current_index
    }
    
    /// Get the number of subtabs
    pub fn subtab_count(&self) -> usize {
        self.other_subtabs.len() + 1
    }
    
    /// Get the title of a specific subtab
    pub fn get_title(&self, index: usize) -> Option<String> {
        match index {
            0 => Some(self.chat_tab.title()),
            n => self.other_subtabs.get(n - 1).map(|tab| tab.title()),
        }
    }
}

/// Helper function to integrate with existing draw_tab_chat
pub fn draw_chat_with_new_subtabs(
    f: &mut Frame,
    app: &mut App,
    bridge: &mut ChatSubtabBridge,
    content_area: Rect,
) {
    // Update bridge to match current tab selection
    bridge.set_active(app.state.chat_tabs.current_index);
    
    // Render using new subtab system
    bridge.render(f, content_area);
}

/// Helper to handle input through the bridge
pub fn handle_chat_input_with_bridge(
    app: &mut App,
    bridge: &mut ChatSubtabBridge,
    key: KeyEvent,
) -> Result<()> {
    // Update bridge to match current tab
    bridge.set_active(app.state.chat_tabs.current_index);
    
    // Handle input through new system
    bridge.handle_input(key)
}