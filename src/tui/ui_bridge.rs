//! UI Bridge for Tab Communication
//! 
//! Provides the bridge between the TUI backend systems (event bus, state, orchestration)
//! and the frontend rendering with ratatui.

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use anyhow::Result;
use tracing::{debug, info};

use crate::tui::{
    event_bus::{EventBus, SystemEvent, TabId},
    shared_state::SharedSystemState,
    tab_registry::TabRegistry,
    integration_hub::IntegrationHub,
    state_sync::StateSyncManager,
};

/// UI Bridge that connects backend to frontend
pub struct UIBridge {
    /// Integration hub
    integration_hub: Arc<IntegrationHub>,
    
    /// State sync manager
    state_sync: Arc<StateSyncManager>,
    
    /// Current tab
    current_tab: Arc<RwLock<TabId>>,
    
    /// Tab states
    tab_states: Arc<RwLock<TabStates>>,
    
    /// UI event channel
    ui_tx: mpsc::UnboundedSender<UIEvent>,
    ui_rx: Arc<RwLock<mpsc::UnboundedReceiver<UIEvent>>>,
    
    /// Render cache
    render_cache: Arc<RwLock<RenderCache>>,
}

/// Tab states for rendering
#[derive(Default)]
pub struct TabStates {
    pub home: HomeTabState,
    pub chat: ChatTabState,
    pub utilities: UtilitiesTabState,
    pub memory: MemoryTabState,
    pub cognitive: CognitiveTabState,
    pub settings: SettingsTabState,
}

/// Home tab state
#[derive(Default)]
pub struct HomeTabState {
    pub system_status: String,
    pub active_models: Vec<String>,
    pub active_agents: Vec<String>,
    pub recent_events: Vec<String>,
}

/// Chat tab state
#[derive(Default)]
pub struct ChatTabState {
    pub messages: Vec<ChatMessage>,
    pub input_buffer: String,
    pub selected_model: Option<String>,
    pub active_agents: Vec<String>,
    pub orchestration_mode: String,
}

/// Chat message
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub model_used: Option<String>,
}

/// Utilities tab state
#[derive(Default)]
pub struct UtilitiesTabState {
    pub available_tools: Vec<String>,
    pub active_tools: Vec<String>,
    pub tool_results: Vec<String>,
    pub code_editor_open: bool,
}

/// Memory tab state
#[derive(Default)]
pub struct MemoryTabState {
    pub memory_usage: f64,
    pub stored_items: usize,
    pub knowledge_graph_nodes: usize,
    pub recent_queries: Vec<String>,
}

/// Cognitive tab state
#[derive(Default)]
pub struct CognitiveTabState {
    pub reasoning_chains: Vec<String>,
    pub active_goals: Vec<String>,
    pub insights: Vec<String>,
    pub orchestration_status: String,
}

/// Settings tab state
#[derive(Default)]
pub struct SettingsTabState {
    pub settings: Vec<(String, String)>,
    pub selected_index: usize,
    pub editing: bool,
}

/// UI Events
#[derive(Debug, Clone)]
pub enum UIEvent {
    TabSwitch(TabId),
    KeyPress(KeyEvent),
    Refresh,
    StateUpdate { key: String, value: serde_json::Value },
    ModelSelected(String),
    AgentActivated(String),
    ToolExecuted(String),
    ChatMessage(String),
}

/// Render cache for performance
#[derive(Default)]
struct RenderCache {
    last_render: Option<std::time::Instant>,
    cached_widgets: std::collections::HashMap<String, String>,
    dirty: bool,
}

impl UIBridge {
    /// Create a new UI bridge
    pub async fn new(
        event_bus: Arc<EventBus>,
        shared_state: Arc<SharedSystemState>,
        tab_registry: Arc<TabRegistry>,
    ) -> Result<Self> {
        let integration_hub = Arc::new(
            IntegrationHub::new(event_bus.clone(), shared_state.clone(), tab_registry).await?
        );
        
        let state_sync = Arc::new(
            StateSyncManager::new(shared_state, event_bus)
        );
        
        let (ui_tx, ui_rx) = mpsc::unbounded_channel();
        
        let bridge = Self {
            integration_hub,
            state_sync,
            current_tab: Arc::new(RwLock::new(TabId::Home)),
            tab_states: Arc::new(RwLock::new(TabStates::default())),
            ui_tx,
            ui_rx: Arc::new(RwLock::new(ui_rx)),
            render_cache: Arc::new(RwLock::new(RenderCache::default())),
        };
        
        // Start state synchronization
        bridge.state_sync.start().await?;
        
        // Setup watchers
        bridge.setup_watchers().await?;
        
        info!("UI Bridge initialized");
        Ok(bridge)
    }
    
    /// Setup state watchers
    async fn setup_watchers(&self) -> Result<()> {
        // Clone for model selection watcher
        let tab_states_model = self.tab_states.clone();
        let ui_tx = self.ui_tx.clone();
        
        // Watch model selection
        self.state_sync.watch(
            "models.selected",
            TabId::System,
            move |change| {
                if let Ok(model) = serde_json::from_value::<String>(change.new_value.clone()) {
                    let tab_states = tab_states_model.clone();
                    let ui_tx = ui_tx.clone();
                    tokio::spawn(async move {
                        let mut states = tab_states.write().await;
                        states.chat.selected_model = Some(model.clone());
                        ui_tx.send(UIEvent::ModelSelected(model)).ok();
                    });
                }
            },
        ).await;
        
        // Clone for agent activation watcher
        let tab_states_agent = self.tab_states.clone();
        
        // Watch agent activation
        self.state_sync.watch(
            "agents.active",
            TabId::System,
            move |change| {
                if let Ok(agents) = serde_json::from_value::<Vec<String>>(change.new_value) {
                    let tab_states = tab_states_agent.clone();
                    tokio::spawn(async move {
                        let mut states = tab_states.write().await;
                        states.chat.active_agents = agents.clone();
                        states.home.active_agents = agents;
                    });
                }
            },
        ).await;
        
        Ok(())
    }
    
    /// Handle key event
    pub async fn handle_key(&self, key: KeyEvent) -> Result<()> {
        let current_tab = self.current_tab.read().await.clone();
        
        match key.code {
            // Tab navigation
            KeyCode::Tab => {
                self.next_tab().await?;
            }
            KeyCode::BackTab => {
                self.previous_tab().await?;
            }
            
            // Tab-specific handling
            _ => {
                match current_tab {
                    TabId::Chat => self.handle_chat_key(key).await?,
                    TabId::Settings => self.handle_settings_key(key).await?,
                    _ => {}
                }
            }
        }
        
        // Mark cache as dirty
        let mut cache = self.render_cache.write().await;
        cache.dirty = true;
        
        Ok(())
    }
    
    /// Handle chat tab key events
    async fn handle_chat_key(&self, key: KeyEvent) -> Result<()> {
        let mut tab_states = self.tab_states.write().await;
        
        match key.code {
            KeyCode::Enter => {
                // Send message
                let message = tab_states.chat.input_buffer.clone();
                if !message.is_empty() {
                    self.send_chat_message(message).await?;
                    tab_states.chat.input_buffer.clear();
                }
            }
            KeyCode::Char(c) => {
                tab_states.chat.input_buffer.push(c);
            }
            KeyCode::Backspace => {
                tab_states.chat.input_buffer.pop();
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle settings tab key events
    async fn handle_settings_key(&self, key: KeyEvent) -> Result<()> {
        let mut tab_states = self.tab_states.write().await;
        
        match key.code {
            KeyCode::Up => {
                if tab_states.settings.selected_index > 0 {
                    tab_states.settings.selected_index -= 1;
                }
            }
            KeyCode::Down => {
                if tab_states.settings.selected_index < tab_states.settings.settings.len() - 1 {
                    tab_states.settings.selected_index += 1;
                }
            }
            KeyCode::Enter => {
                tab_states.settings.editing = !tab_states.settings.editing;
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Send chat message
    async fn send_chat_message(&self, message: String) -> Result<()> {
        // Add to chat history
        let mut tab_states = self.tab_states.write().await;
        tab_states.chat.messages.push(ChatMessage {
            role: "user".to_string(),
            content: message.clone(),
            timestamp: chrono::Utc::now(),
            model_used: None,
        });
        
        // Send through integration hub
        self.integration_hub.send_cross_tab_message(
            TabId::Chat,
            TabId::System,
            crate::tui::integration_hub::CrossTabMessage {
                message_type: crate::tui::integration_hub::MessageType::Request,
                payload: serde_json::json!({
                    "type": "chat",
                    "content": message,
                }),
                requires_response: true,
                correlation_id: Some(uuid::Uuid::new_v4().to_string()),
            },
        ).await?;
        
        Ok(())
    }
    
    /// Switch to next tab
    async fn next_tab(&self) -> Result<()> {
        let mut current = self.current_tab.write().await;
        let old_tab = current.clone();
        
        *current = match current.clone() {
            TabId::Home => TabId::Chat,
            TabId::Chat => TabId::Utilities,
            TabId::Utilities => TabId::Memory,
            TabId::Memory => TabId::Cognitive,
            TabId::Cognitive => TabId::Settings,
            TabId::Settings => TabId::Home,
            TabId::System => TabId::Home,
            TabId::Custom(_) => TabId::Home,
        };
        
        // Send tab switch event
        let event = SystemEvent::TabSwitched {
            from: old_tab,
            to: current.clone(),
        };
        
        self.integration_hub.publish_event(event).await?;
        
        Ok(())
    }
    
    /// Switch to previous tab
    async fn previous_tab(&self) -> Result<()> {
        let mut current = self.current_tab.write().await;
        let old_tab = current.clone();
        
        *current = match current.clone() {
            TabId::Home => TabId::Settings,
            TabId::Chat => TabId::Home,
            TabId::Utilities => TabId::Chat,
            TabId::Memory => TabId::Utilities,
            TabId::Cognitive => TabId::Memory,
            TabId::Settings => TabId::Cognitive,
            TabId::System => TabId::Settings,
            TabId::Custom(_) => TabId::Home,
        };
        
        // Send tab switch event
        let event = SystemEvent::TabSwitched {
            from: old_tab,
            to: current.clone(),
        };
        
        self.integration_hub.publish_event(event).await?;
        
        Ok(())
    }
    
    /// Render the current tab
    pub async fn render(&self, frame: &mut Frame<'_>) {
        let current_tab = self.current_tab.read().await.clone();
        let tab_states = self.tab_states.read().await;
        
        // Create layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Tab bar
                Constraint::Min(0),     // Content
                Constraint::Length(3),  // Status bar
            ])
            .split(frame.area());
        
        // Render tab bar
        self.render_tab_bar(frame, chunks[0], current_tab.clone()).await;
        
        // Render current tab content
        match current_tab {
            TabId::Home => self.render_home_tab(frame, chunks[1], &tab_states.home).await,
            TabId::Chat => self.render_chat_tab(frame, chunks[1], &tab_states.chat).await,
            TabId::Utilities => self.render_utilities_tab(frame, chunks[1], &tab_states.utilities).await,
            TabId::Memory => self.render_memory_tab(frame, chunks[1], &tab_states.memory).await,
            TabId::Cognitive => self.render_cognitive_tab(frame, chunks[1], &tab_states.cognitive).await,
            TabId::Settings => self.render_settings_tab(frame, chunks[1], &tab_states.settings).await,
            TabId::System => {} // System tab is not rendered
            TabId::Custom(_) => {} // Custom tabs handled separately
        }
        
        // Render status bar
        self.render_status_bar(frame, chunks[2]).await;
    }
    
    /// Render tab bar
    async fn render_tab_bar(&self, frame: &mut Frame<'_>, area: Rect, current: TabId) {
        let tabs = vec!["Home", "Chat", "Utilities", "Memory", "Cognitive", "Settings"];
        let selected = match current {
            TabId::Home => 0,
            TabId::Chat => 1,
            TabId::Utilities => 2,
            TabId::Memory => 3,
            TabId::Cognitive => 4,
            TabId::Settings => 5,
            TabId::System => 0,
            TabId::Custom(_) => 0,
        };
        
        let tabs_widget = ratatui::widgets::Tabs::new(tabs)
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" Loki AI TUI "))
            .select(selected)
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        
        frame.render_widget(tabs_widget, area);
    }
    
    /// Render home tab
    async fn render_home_tab(&self, frame: &mut Frame<'_>, area: Rect, state: &HomeTabState) {
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title(" Home ");
        
        let inner = block.inner(area);
        frame.render_widget(block, area);
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Status
                Constraint::Percentage(30),  // Models
                Constraint::Percentage(30),  // Agents
                Constraint::Percentage(40),  // Events
            ])
            .split(inner);
        
        // Status
        let status = ratatui::widgets::Paragraph::new(state.system_status.as_str())
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" System Status "));
        frame.render_widget(status, chunks[0]);
        
        // Active Models
        let models: Vec<ratatui::widgets::ListItem> = state.active_models.iter()
            .map(|m| ratatui::widgets::ListItem::new(m.as_str()))
            .collect();
        let models_list = ratatui::widgets::List::new(models)
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" Active Models "));
        frame.render_widget(models_list, chunks[1]);
        
        // Active Agents
        let agents: Vec<ratatui::widgets::ListItem> = state.active_agents.iter()
            .map(|a| ratatui::widgets::ListItem::new(a.as_str()))
            .collect();
        let agents_list = ratatui::widgets::List::new(agents)
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" Active Agents "));
        frame.render_widget(agents_list, chunks[2]);
        
        // Recent Events
        let events: Vec<ratatui::widgets::ListItem> = state.recent_events.iter()
            .map(|e| ratatui::widgets::ListItem::new(e.as_str()))
            .collect();
        let events_list = ratatui::widgets::List::new(events)
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" Recent Events "));
        frame.render_widget(events_list, chunks[3]);
    }
    
    /// Render chat tab
    async fn render_chat_tab(&self, frame: &mut Frame<'_>, area: Rect, state: &ChatTabState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),     // Messages
                Constraint::Length(3),  // Input
            ])
            .split(area);
        
        // Messages
        let messages: Vec<ratatui::widgets::ListItem> = state.messages.iter()
            .map(|m| {
                let content = format!("[{}] {}: {}", 
                    m.timestamp.format("%H:%M:%S"),
                    m.role,
                    m.content
                );
                ratatui::widgets::ListItem::new(content)
            })
            .collect();
        
        let messages_list = ratatui::widgets::List::new(messages)
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(format!(" Chat - {} ", state.orchestration_mode)));
        frame.render_widget(messages_list, chunks[0]);
        
        // Input
        let input = ratatui::widgets::Paragraph::new(state.input_buffer.as_str())
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" Input (Enter to send) "));
        frame.render_widget(input, chunks[1]);
    }
    
    /// Render utilities tab
    async fn render_utilities_tab(&self, frame: &mut Frame<'_>, area: Rect, state: &UtilitiesTabState) {
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title(" Utilities ");
        frame.render_widget(block, area);
    }
    
    /// Render memory tab
    async fn render_memory_tab(&self, frame: &mut Frame<'_>, area: Rect, state: &MemoryTabState) {
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title(" Memory ");
        frame.render_widget(block, area);
    }
    
    /// Render cognitive tab
    async fn render_cognitive_tab(&self, frame: &mut Frame<'_>, area: Rect, state: &CognitiveTabState) {
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title(" Cognitive ");
        frame.render_widget(block, area);
    }
    
    /// Render settings tab
    async fn render_settings_tab(&self, frame: &mut Frame<'_>, area: Rect, state: &SettingsTabState) {
        let settings: Vec<ratatui::widgets::ListItem> = state.settings.iter()
            .enumerate()
            .map(|(i, (key, value))| {
                let content = format!("{}: {}", key, value);
                let style = if i == state.selected_index {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ratatui::widgets::ListItem::new(content).style(style)
            })
            .collect();
        
        let list = ratatui::widgets::List::new(settings)
            .block(ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::ALL)
                .title(" Settings "));
        frame.render_widget(list, area);
    }
    
    /// Render status bar
    async fn render_status_bar(&self, frame: &mut Frame<'_>, area: Rect) {
        let status = self.integration_hub.get_status().await;
        let status_text = format!(
            " Connected: {} | Components: {} | Messages: {} ",
            status.initialized,
            status.connected_components.len(),
            status.message_count
        );
        
        let status_bar = ratatui::widgets::Paragraph::new(status_text)
            .style(Style::default().bg(Color::Blue).fg(Color::White));
        
        frame.render_widget(status_bar, area);
    }
}