//! Render state caching layer for async state access
//!
//! This module provides a caching layer that bridges async state to synchronous
//! rendering functions in the TUI. It periodically syncs with async sources
//! and provides fast synchronous access for rendering.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::models::providers::ModelProvider;
use crate::tui::chat::orchestration::OrchestrationConfig;
use crate::tui::chat::state::ChatState;
use crate::tui::ui_bridge::ChatMessage;
use crate::tui::chat::agents::AgentConfig;
use crate::tui::chat::tools::discovery::ToolCapability;
use crate::tui::run::AssistantResponseType;

/// Cached state for rendering
#[derive(Debug, Clone, Default)]
pub struct RenderState {
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    
    /// Cached orchestration state
    pub orchestration: CachedOrchestration,
    
    /// Cached chat messages
    pub messages: CachedMessages,
    
    /// Cached agent state
    pub agents: CachedAgents,
    
    /// Cached tool state
    pub tools: CachedTools,
    
    /// Cache validity duration in milliseconds
    pub cache_ttl_ms: u64,
}

/// Cached orchestration information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CachedOrchestration {
    /// Enabled models
    pub enabled_models: Vec<ModelInfo>,
    
    /// Active model
    pub active_model: Option<String>,
    
    /// Context window size
    pub context_window: usize,
    
    /// Cost limit
    pub cost_limit: f64,
    
    /// Strategy
    pub strategy: String,
    
    /// Quality threshold
    pub quality_threshold: f64,
    
    /// Fallback enabled
    pub fallback_enabled: bool,
    
    /// Model capabilities
    pub capabilities: HashMap<String, Vec<String>>,
}

/// Model information for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub status: String,
    pub capabilities: Vec<String>,
}

/// Cached chat messages
#[derive(Debug, Clone, Default)]
pub struct CachedMessages {
    /// Recent messages (limited for performance)
    pub recent: Vec<CachedMessage>,
    
    /// Total message count
    pub total_count: usize,
    
    /// Active chat ID
    pub active_chat: usize,
    
    /// Show timestamps
    pub show_timestamps: bool,
}

/// Cached message for rendering
#[derive(Debug, Clone)]
pub struct CachedMessage {
    pub content: String,
    pub role: String,
    pub timestamp: DateTime<Utc>,
    pub model: Option<String>,
    pub is_streaming: bool,
}

/// Cached agent information
#[derive(Debug, Clone, Default)]
pub struct CachedAgents {
    /// Available agents
    pub available: Vec<AgentInfo>,
    
    /// Active agents
    pub active: Vec<String>,
    
    /// Collaboration mode
    pub collaboration_mode: bool,
    
    /// Max active agents
    pub max_active: usize,
    
    /// Auto-selection enabled
    pub auto_selection: bool,
}

/// Agent information for rendering
#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub description: String,
    pub capabilities: Vec<String>,
    pub status: String,
}

/// Cached tool information
#[derive(Debug, Clone, Default)]
pub struct CachedTools {
    /// Available tools
    pub available: Vec<ToolInfo>,
    
    /// Active tool executions
    pub active_executions: Vec<String>,
    
    /// Recent tool results
    pub recent_results: Vec<ToolResult>,
}

/// Tool information for rendering
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInfo {
    pub name: String,
    pub category: String,
    pub available: bool,
}

/// Tool execution result
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool: String,
    pub success: bool,
    pub summary: String,
    pub timestamp: DateTime<Utc>,
}

/// Render state manager that handles caching and synchronization
pub struct RenderStateManager {
    /// Current cached state
    state: Arc<Mutex<RenderState>>,
    
    /// References to async sources
    chat_state: Option<Arc<RwLock<ChatState>>>,
    orchestration: Option<Arc<RwLock<crate::tui::chat::orchestration::OrchestrationManager>>>,
    agent_manager: Option<Arc<RwLock<crate::tui::chat::agents::manager::AgentManager>>>,
    tool_integration: Option<Arc<crate::tui::chat::integrations::ToolIntegration>>,
}

impl RenderStateManager {
    /// Create new render state manager
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(RenderState {
                cache_ttl_ms: 100, // 100ms default TTL
                ..Default::default()
            })),
            chat_state: None,
            orchestration: None,
            agent_manager: None,
            tool_integration: None,
        }
    }
    
    /// Set chat state source
    pub fn set_chat_state(&mut self, state: Arc<RwLock<ChatState>>) {
        self.chat_state = Some(state);
    }
    
    /// Set orchestration source
    pub fn set_orchestration(&mut self, orchestration: Arc<RwLock<crate::tui::chat::orchestration::OrchestrationManager>>) {
        self.orchestration = Some(orchestration);
    }
    
    /// Set agent manager source
    pub fn set_agent_manager(&mut self, manager: Arc<RwLock<crate::tui::chat::agents::manager::AgentManager>>) {
        self.agent_manager = Some(manager);
    }
    
    /// Set tool integration source
    pub fn set_tool_integration(&mut self, tools: Arc<crate::tui::chat::integrations::ToolIntegration>) {
        self.tool_integration = Some(tools);
    }
    
    /// Get current cached state (synchronous)
    pub fn get_state(&self) -> RenderState {
        // Use block_on to get the lock synchronously
        let state = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.state.lock().await.clone()
            })
        });
        state
    }
    
    /// Check if cache is still valid
    pub fn is_cache_valid(&self) -> bool {
        let state = self.get_state();
        let age_ms = (Utc::now() - state.last_updated).num_milliseconds() as u64;
        age_ms < state.cache_ttl_ms
    }
    
    /// Synchronize state from async sources
    pub async fn sync_state(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        
        // Sync orchestration state
        if let Some(orchestration) = &self.orchestration {
            let orch = orchestration.read().await;
            
            // Get enabled models
            let enabled_models: Vec<ModelInfo> = orch.get_enabled_models().await
                .into_iter()
                .map(|model| ModelInfo {
                    name: model.name.clone(),
                    provider: format!("{:?}", model.provider),
                    status: if model.available { "Available" } else { "Unavailable" }.to_string(),
                    capabilities: model.capabilities.iter().map(|c| format!("{:?}", c)).collect(),
                })
                .collect();
            
            // Get orchestration config
            let config = orch.get_config().await;
            
            // Build capabilities map before moving enabled_models
            let capabilities = {
                let mut caps: HashMap<String, Vec<String>> = HashMap::new();
                // Group capabilities by type
                for model in &enabled_models {
                    for cap in &model.capabilities {
                        let cap_str = format!("{:?}", cap);
                        caps.entry(cap_str.clone())
                            .or_insert_with(Vec::new)
                            .push(model.name.clone());
                    }
                }
                caps
            };
            
            state.orchestration = CachedOrchestration {
                enabled_models,
                active_model: orch.get_primary_model().map(|m| m.name.clone()),
                context_window: config.context_window,
                cost_limit: config.cost_limit.unwrap_or(0.0) as f64,
                strategy: format!("{:?}", config.strategy),
                quality_threshold: config.quality_threshold as f64,
                fallback_enabled: config.fallback_enabled,
                capabilities,
            };
        }
        
        // Sync chat messages
        if let Some(chat_state) = &self.chat_state {
            let chat = chat_state.read().await;
            
            // Get recent messages (limit to last 50 for performance)
            let messages = &chat.messages;
            let recent: Vec<CachedMessage> = messages
                .iter()
                .rev()
                .take(50)
                .filter_map(|msg| {
                    match msg {
                        AssistantResponseType::Message { 
                            author, 
                            message, 
                            timestamp, 
                            streaming_state, 
                            .. 
                        } => Some(CachedMessage {
                            content: message.clone(),
                            role: author.clone(),
                            timestamp: timestamp.parse::<DateTime<Utc>>().unwrap_or_else(|_| Utc::now()),
                            model: None, // Model info is not in the Message variant
                            is_streaming: matches!(streaming_state, crate::tui::run::StreamingState::Streaming { .. }),
                        }),
                        AssistantResponseType::Code { 
                            author, 
                            code, 
                            timestamp, 
                            language,
                            .. 
                        } => Some(CachedMessage {
                            content: format!("```{}\n{}\n```", language, code),
                            role: author.clone(),
                            timestamp: timestamp.parse::<DateTime<Utc>>().unwrap_or_else(|_| Utc::now()),
                            model: None,
                            is_streaming: false,
                        }),
                        AssistantResponseType::Error { 
                            message,
                            timestamp,
                            .. 
                        } => Some(CachedMessage {
                            content: format!("Error: {}", message),
                            role: "system".to_string(),
                            timestamp: timestamp.parse::<DateTime<Utc>>().unwrap_or_else(|_| Utc::now()),
                            model: None,
                            is_streaming: false,
                        }),
                        _ => None,
                    }
                })
                .rev()
                .collect();
            
            state.messages = CachedMessages {
                recent,
                total_count: messages.len(),
                active_chat: chat.active_chat,
                show_timestamps: chat.show_timestamps,
            };
        }
        
        // Sync agent state
        if let Some(agent_manager) = &self.agent_manager {
            let agents = agent_manager.read().await;
            
            // Map specializations to available agents
            let available: Vec<AgentInfo> = agents.active_specializations
                .iter()
                .map(|spec| {
                    let name = format!("{:?}", spec);
                    let (description, capabilities) = match spec {
                        crate::cognitive::agents::AgentSpecialization::Analytical => (
                            "Analytical reasoning and data analysis".to_string(),
                            vec!["analysis".to_string(), "data".to_string(), "logic".to_string()]
                        ),
                        crate::cognitive::agents::AgentSpecialization::Creative => (
                            "Creative problem solving and generation".to_string(),
                            vec!["creativity".to_string(), "generation".to_string(), "ideation".to_string()]
                        ),
                        crate::cognitive::agents::AgentSpecialization::Strategic => (
                            "Strategic planning and decision making".to_string(),
                            vec!["strategy".to_string(), "planning".to_string(), "decisions".to_string()]
                        ),
                        crate::cognitive::agents::AgentSpecialization::Technical => (
                            "Technical implementation and debugging".to_string(),
                            vec!["coding".to_string(), "debugging".to_string(), "technical".to_string()]
                        ),
                        crate::cognitive::agents::AgentSpecialization::Social => (
                            "Communication and content creation".to_string(),
                            vec!["writing".to_string(), "communication".to_string(), "content".to_string()]
                        ),
                        _ => (
                            "General purpose agent".to_string(),
                            vec!["general".to_string()]
                        ),
                    };
                    
                    AgentInfo {
                        name,
                        description,
                        capabilities,
                        status: if agents.agent_system_enabled {
                            "Active".to_string()
                        } else {
                            "Ready".to_string()
                        },
                    }
                })
                .collect();
            
            // Get active agent names
            let active = if agents.agent_system_enabled {
                agents.active_specializations
                    .iter()
                    .map(|spec| format!("{:?}", spec))
                    .collect()
            } else {
                Vec::new()
            };
            
            state.agents = CachedAgents {
                available,
                active,
                collaboration_mode: agents.collaboration_mode != crate::tui::chat::agents::manager::CollaborationMode::Independent,
                max_active: agents.active_specializations.len(),
                auto_selection: agents.agent_system_enabled,
            };
        }
        
        // Sync tool state
        if let Some(tool_integration) = &self.tool_integration {
            // Get available tools
            let tool_list = tool_integration.list_available_tools().await.unwrap_or_default();
            let available = tool_list
                .into_iter()
                .map(|tool_name| {
                    // Categorize tools based on name patterns
                    let category = if tool_name.contains("file") || tool_name.contains("read") || tool_name.contains("write") {
                        "filesystem"
                    } else if tool_name.contains("search") || tool_name.contains("web") {
                        "web"
                    } else if tool_name.contains("git") || tool_name.contains("code") {
                        "development"
                    } else if tool_name.contains("shell") || tool_name.contains("exec") {
                        "system"
                    } else {
                        "general"
                    };
                    
                    ToolInfo {
                        name: tool_name,
                        category: category.to_string(),
                        available: true, // Assume available if listed
                    }
                })
                .collect();
            
            // Get active contexts to determine active executions
            let active_contexts = tool_integration.get_active_contexts().await;
            let active_executions = active_contexts
                .into_iter()
                .map(|(_, context)| context.tool_name.clone())
                .collect();
            
            // Get recent execution history
            let history = tool_integration.get_execution_history(10).await;
            let recent_results = history
                .into_iter()
                .map(|record| ToolResult {
                    tool: record.tool_name,
                    success: record.success,
                    summary: record.output.chars().take(100).collect(),
                    timestamp: record.timestamp,
                })
                .collect();
            
            state.tools = CachedTools {
                available,
                active_executions,
                recent_results,
            };
        }
        
        // Update timestamp
        state.last_updated = Utc::now();
        
        Ok(())
    }
    
    /// Force synchronous update (for use in render functions)
    pub fn sync_update(&self) -> Result<()> {
        // Check if cache is still valid
        if self.is_cache_valid() {
            return Ok(());
        }
        
        // Use block_in_place to sync state synchronously
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.sync_state().await
            })
        })
    }
}

/// Global render state instance
static RENDER_STATE: once_cell::sync::Lazy<Arc<Mutex<RenderStateManager>>> = 
    once_cell::sync::Lazy::new(|| {
        Arc::new(Mutex::new(RenderStateManager::new()))
    });

/// Get the global render state manager
pub fn get_render_state_manager() -> Arc<Mutex<RenderStateManager>> {
    RENDER_STATE.clone()
}

/// Helper function to get current state synchronously
pub fn get_current_render_state() -> RenderState {
    let manager = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            RENDER_STATE.lock().await
        })
    });
    
    // Try to sync if needed (non-blocking)
    let _ = manager.sync_update();
    
    manager.get_state()
}

/// Initialize render state with async sources
pub async fn initialize_render_state(
    chat_state: Option<Arc<RwLock<ChatState>>>,
    orchestration: Option<Arc<RwLock<crate::tui::chat::orchestration::OrchestrationManager>>>,
    agent_manager: Option<Arc<RwLock<crate::tui::chat::agents::manager::AgentManager>>>,
    tool_integration: Option<Arc<crate::tui::chat::integrations::ToolIntegration>>,
) -> Result<()> {
    let mut manager = RENDER_STATE.lock().await;
    
    if let Some(state) = chat_state {
        manager.set_chat_state(state);
    }
    
    if let Some(orch) = orchestration {
        manager.set_orchestration(orch);
    }
    
    if let Some(agents) = agent_manager {
        manager.set_agent_manager(agents);
    }
    
    if let Some(tools) = tool_integration {
        manager.set_tool_integration(tools);
    }
    
    // Do initial sync
    manager.sync_state().await?;
    
    Ok(())
}