//! Shared State Management for TUI
//! 
//! This module provides centralized state management that is shared across all tabs
//! and components, ensuring consistency and enabling real-time synchronization.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde_json::Value;
use tracing::{info, warn};
use anyhow::Result;

use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use crate::tui::chat::state::ChatState;
use crate::tools::ToolStatus;
use crate::tui::bridges::UnifiedBridge;

/// Shared system state that all tabs can access
pub struct SharedSystemState {
    /// Event bus for cross-tab communication
    pub event_bus: Arc<EventBus>,
    
    /// Unified bridge system for cross-tab integration
    pub bridges: Option<Arc<UnifiedBridge>>,
    
    /// Tab registry for tab discovery and capabilities
    pub tab_registry: Arc<RwLock<TabRegistry>>,
    
    /// Tab-specific states
    pub chat_state: Arc<RwLock<ChatState>>,
    pub utilities_state: Arc<RwLock<UtilitiesState>>,
    pub memory_state: Arc<RwLock<MemoryState>>,
    pub cognitive_state: Arc<RwLock<CognitiveState>>,
    pub home_state: Arc<RwLock<HomeState>>,
    pub settings_state: Arc<RwLock<SettingsState>>,
    
    /// Cross-cutting concerns
    pub context_manager: Arc<RwLock<ContextManager>>,
    pub command_router: Arc<CommandRouter>,
    
    /// State synchronization channels
    state_update_tx: mpsc::UnboundedSender<StateUpdate>,
    state_update_rx: Arc<RwLock<mpsc::UnboundedReceiver<StateUpdate>>>,
    
    /// State snapshots for time-travel debugging
    snapshots: Arc<RwLock<VecDeque<StateSnapshot>>>,
    max_snapshots: usize,
}

/// State update notification
#[derive(Debug, Clone)]
pub struct StateUpdate {
    pub source: TabId,
    pub update_type: StateUpdateType,
    pub timestamp: Instant,
}

/// Types of state updates
#[derive(Debug, Clone)]
pub enum StateUpdateType {
    ChatMessage(String),
    ModelChanged(String),
    ToolExecuted(String),
    MemoryUpdated(String),
    CognitiveInsight(String),
    SettingChanged(String, Value),
}

/// State snapshot for debugging and rollback
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub timestamp: Instant,
    pub chat_state: ChatStateSnapshot,
    pub utilities_state: UtilitiesStateSnapshot,
    pub memory_state: MemoryStateSnapshot,
    pub cognitive_state: CognitiveStateSnapshot,
}

/// Simplified chat state snapshot
#[derive(Debug, Clone)]
pub struct ChatStateSnapshot {
    pub message_count: usize,
    pub active_model: Option<String>,
    pub orchestration_enabled: bool,
}

/// Simplified utilities state snapshot
#[derive(Debug, Clone)]
pub struct UtilitiesStateSnapshot {
    pub tools_configured: usize,
    pub mcp_servers_active: usize,
    pub plugins_loaded: usize,
}

/// Simplified memory state snapshot
#[derive(Debug, Clone)]
pub struct MemoryStateSnapshot {
    pub knowledge_nodes: usize,
    pub vector_count: usize,
    pub stories_active: usize,
}

/// Simplified cognitive state snapshot
#[derive(Debug, Clone)]
pub struct CognitiveStateSnapshot {
    pub reasoning_chains_active: usize,
    pub insights_generated: usize,
    pub goals_active: usize,
}

/// Utilities tab state
#[derive(Debug)]
pub struct UtilitiesState {
    /// Configured tools
    pub tools: HashMap<String, ToolConfiguration>,
    
    /// MCP server configurations
    pub mcp_servers: HashMap<String, McpServerConfig>,
    
    /// Plugin states
    pub plugins: HashMap<String, PluginState>,
    
    /// Daemon processes
    pub daemons: HashMap<String, DaemonState>,
    
    /// Tool execution history
    pub tool_history: VecDeque<ToolExecution>,
}

/// Tool configuration
#[derive(Debug, Clone)]
pub struct ToolConfiguration {
    pub tool_id: String,
    pub enabled: bool,
    pub config: Value,
    pub permissions: ToolPermissions,
    pub rate_limit: Option<RateLimit>,
}

/// Tool permissions
#[derive(Debug, Clone)]
pub struct ToolPermissions {
    pub read_files: bool,
    pub write_files: bool,
    pub network_access: bool,
    pub system_commands: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

/// Tool execution record
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_id: String,
    pub timestamp: Instant,
    pub duration: Duration,
    pub success: bool,
    pub source: TabId,
}

/// MCP server configuration
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub server_id: String,
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub enabled: bool,
    pub auto_start: bool,
}

/// Plugin state
#[derive(Debug, Clone)]
pub struct PluginState {
    pub plugin_id: String,
    pub name: String,
    pub version: String,
    pub enabled: bool,
    pub loaded: bool,
    pub config: Value,
}

/// Daemon state
#[derive(Debug, Clone)]
pub struct DaemonState {
    pub daemon_id: String,
    pub name: String,
    pub running: bool,
    pub pid: Option<u32>,
    pub uptime: Duration,
    pub restart_count: u32,
}

/// Memory tab state
#[derive(Debug)]
pub struct MemoryState {
    /// Knowledge graph
    pub knowledge_graph: Arc<RwLock<KnowledgeGraphState>>,
    
    /// Vector database state
    pub vector_db: VectorDbState,
    
    /// Story management
    pub stories: HashMap<String, StoryState>,
    
    /// Database connections
    pub databases: HashMap<String, DatabaseState>,
    
    /// Memory statistics
    pub stats: MemoryStats,
}

/// Knowledge graph state
#[derive(Debug)]
pub struct KnowledgeGraphState {
    pub node_count: usize,
    pub edge_count: usize,
    pub last_updated: Instant,
    pub active_queries: Vec<String>,
}

/// Vector database state
#[derive(Debug, Clone)]
pub struct VectorDbState {
    pub vector_count: usize,
    pub dimensions: usize,
    pub index_type: String,
    pub last_rebuild: Option<Instant>,
}

/// Story state
#[derive(Debug, Clone)]
pub struct StoryState {
    pub story_id: String,
    pub title: String,
    pub context_items: usize,
    pub active: bool,
    pub last_accessed: Instant,
}

/// Database connection state
#[derive(Debug, Clone)]
pub struct DatabaseState {
    pub backend: String,
    pub connected: bool,
    pub connection_string: String,
    pub last_query: Option<Instant>,
}

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_items: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub average_retrieval_ms: f64,
}

/// Cognitive tab state
#[derive(Debug)]
pub struct CognitiveState {
    /// Active reasoning chains
    pub reasoning_chains: Vec<ReasoningChainState>,
    
    /// Generated insights
    pub insights: VecDeque<InsightState>,
    
    /// Goal tracking
    pub goals: HashMap<String, GoalState>,
    
    /// Attention mechanism state
    pub attention: AttentionState,
    
    /// Cognitive metrics
    pub metrics: CognitiveMetrics,
}

/// Reasoning chain state
#[derive(Debug, Clone)]
pub struct ReasoningChainState {
    pub chain_id: String,
    pub steps_completed: usize,
    pub total_steps: usize,
    pub current_step: String,
    pub confidence: f32,
}

/// Insight state
#[derive(Debug, Clone)]
pub struct InsightState {
    pub insight_id: String,
    pub category: String,
    pub description: String,
    pub confidence: f32,
    pub timestamp: Instant,
    pub applied: bool,
}

/// Goal state
#[derive(Debug, Clone)]
pub struct GoalState {
    pub goal_id: String,
    pub description: String,
    pub priority: u8,
    pub progress: f32,
    pub status: GoalStatus,
    pub actions_taken: usize,
}

/// Goal status
#[derive(Debug, Clone)]
pub enum GoalStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Abandoned,
}

/// Attention mechanism state
#[derive(Debug, Clone)]
pub struct AttentionState {
    pub focus_areas: Vec<String>,
    pub attention_weights: HashMap<String, f32>,
    pub context_window_size: usize,
}

/// Cognitive metrics
#[derive(Debug, Clone, Default)]
pub struct CognitiveMetrics {
    pub reasoning_operations: usize,
    pub insights_generated: usize,
    pub goals_achieved: usize,
    pub average_reasoning_time_ms: f64,
}

/// Home tab state
#[derive(Debug)]
pub struct HomeState {
    /// System metrics
    pub metrics: SystemMetrics,
    
    /// Active alerts
    pub alerts: VecDeque<Alert>,
    
    /// Health status
    pub health: HealthStatus,
    
    /// Keybinding configuration
    pub keybindings: HashMap<String, String>,
}

/// System metrics
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub disk_usage: f32,
    pub network_throughput: f32,
    pub active_connections: usize,
    pub uptime: Duration,
}

/// Alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub source: TabId,
    pub timestamp: Instant,
    pub acknowledged: bool,
}

/// Alert severity
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub overall: HealthLevel,
    pub components: HashMap<String, HealthLevel>,
    pub last_check: Instant,
}

/// Health level
#[derive(Debug, Clone)]
pub enum HealthLevel {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Settings state
#[derive(Debug)]
pub struct SettingsState {
    /// Application settings
    pub app_settings: HashMap<String, Value>,
    
    /// User preferences
    pub preferences: UserPreferences,
    
    /// API configurations
    pub api_configs: HashMap<String, ApiConfig>,
    
    /// Theme settings
    pub theme: ThemeSettings,
}

/// User preferences
#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub auto_save: bool,
    pub notifications_enabled: bool,
    pub telemetry_enabled: bool,
    pub default_model: Option<String>,
    pub default_temperature: f32,
}

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub provider: String,
    pub api_key: String,
    pub endpoint: Option<String>,
    pub rate_limit: Option<RateLimit>,
}

/// Theme settings
#[derive(Debug, Clone)]
pub struct ThemeSettings {
    pub theme_name: String,
    pub dark_mode: bool,
    pub accent_color: String,
    pub font_size: u8,
}

/// Tab registry for tracking available tabs and their capabilities
#[derive(Debug)]
pub struct TabRegistry {
    tabs: HashMap<TabId, TabInfo>,
}

/// Tab information and capabilities
#[derive(Debug, Clone)]
pub struct TabInfo {
    pub id: TabId,
    pub name: String,
    pub capabilities: Vec<TabCapability>,
    pub shortcuts: Vec<String>,
    pub active: bool,
}

/// Tab capabilities
#[derive(Debug, Clone)]
pub enum TabCapability {
    ExecuteTools,
    ManageModels,
    StoreMemory,
    RetrieveContext,
    ProcessReasoning,
    GenerateInsights,
    ConfigureSettings,
    MonitorMetrics,
}

/// Context manager for cross-tab context sharing
#[derive(Debug)]
pub struct ContextManager {
    /// Active contexts by tab
    contexts: HashMap<TabId, TabContext>,
    
    /// Global context available to all tabs
    global_context: GlobalContext,
    
    /// Context history
    context_history: VecDeque<ContextSnapshot>,
}

/// Tab-specific context
#[derive(Debug, Clone)]
pub struct TabContext {
    pub tab_id: TabId,
    pub active_items: Vec<String>,
    pub focus: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Global context
#[derive(Debug, Clone)]
pub struct GlobalContext {
    pub current_conversation: Option<String>,
    pub active_model: Option<String>,
    pub active_tools: Vec<String>,
    pub user_preferences: HashMap<String, Value>,
}

/// Context snapshot
#[derive(Debug, Clone)]
pub struct ContextSnapshot {
    pub timestamp: Instant,
    pub contexts: HashMap<TabId, TabContext>,
    pub global: GlobalContext,
}

/// Command router for routing commands to appropriate handlers
pub struct CommandRouter {
    routes: Arc<RwLock<HashMap<String, CommandHandler>>>,
}

/// Command handler
pub type CommandHandler = Arc<dyn Fn(Command) -> Result<CommandResult> + Send + Sync>;

/// Command
#[derive(Debug, Clone)]
pub struct Command {
    pub source: TabId,
    pub command: String,
    pub args: Vec<String>,
    pub context: Option<Value>,
}

/// Command result
#[derive(Debug, Clone)]
pub struct CommandResult {
    pub success: bool,
    pub message: String,
    pub data: Option<Value>,
}

impl SharedSystemState {
    /// Get a value from global context
    pub async fn get(&self, key: &str) -> Option<Value> {
        let context = self.context_manager.read().await;
        context.global_context.user_preferences.get(key).cloned()
    }
    
    /// Set a value in global context
    pub async fn set(&self, key: String, value: Value) -> Result<()> {
        let mut context = self.context_manager.write().await;
        context.global_context.user_preferences.insert(key, value);
        Ok(())
    }

    /// Create new shared system state
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        
        Self {
            event_bus: event_bus.clone(),
            bridges: None, // Will be initialized separately to avoid circular dependency
            tab_registry: Arc::new(RwLock::new(TabRegistry::new())),
            chat_state: Arc::new(RwLock::new(ChatState::default())),
            utilities_state: Arc::new(RwLock::new(UtilitiesState::new())),
            memory_state: Arc::new(RwLock::new(MemoryState::new())),
            cognitive_state: Arc::new(RwLock::new(CognitiveState::new())),
            home_state: Arc::new(RwLock::new(HomeState::new())),
            settings_state: Arc::new(RwLock::new(SettingsState::new())),
            context_manager: Arc::new(RwLock::new(ContextManager::new())),
            command_router: Arc::new(CommandRouter::new()),
            state_update_tx: tx,
            state_update_rx: Arc::new(RwLock::new(rx)),
            snapshots: Arc::new(RwLock::new(VecDeque::new())),
            max_snapshots: 100,
        }
    }
    
    /// Initialize the bridge system (call after creation to avoid circular dependency)
    pub fn set_bridges(&mut self, bridges: Arc<UnifiedBridge>) {
        self.bridges = Some(bridges);
    }
    
    /// Get the bridge system
    pub fn get_bridges(&self) -> Option<Arc<UnifiedBridge>> {
        self.bridges.clone()
    }
    
    /// Notify state update
    pub async fn notify_update(&self, source: TabId, update_type: StateUpdateType) {
        let update = StateUpdate {
            source: source.clone(),
            update_type: update_type.clone(),
            timestamp: Instant::now(),
        };
        
        if let Err(e) = self.state_update_tx.send(update) {
            warn!("Failed to send state update: {:?}", e);
        }
        
        // Publish event to event bus
        let event = match update_type {
            StateUpdateType::ChatMessage(msg) => SystemEvent::MessageReceived {
                message_id: uuid::Uuid::new_v4().to_string(),
                content: msg,
                source,
            },
            StateUpdateType::ModelChanged(model) => SystemEvent::ModelSelected {
                model_id: model,
                source,
            },
            StateUpdateType::ToolExecuted(tool) => SystemEvent::ToolStatusChanged {
                tool_id: tool,
                status: ToolStatus::Success,
            },
            StateUpdateType::MemoryUpdated(key) => SystemEvent::MemoryStored {
                key,
                value_type: "update".to_string(),
                source,
            },
            StateUpdateType::CognitiveInsight(insight) => SystemEvent::InsightGenerated {
                insight_id: uuid::Uuid::new_v4().to_string(),
                category: "general".to_string(),
                confidence: 0.8,
            },
            StateUpdateType::SettingChanged(setting, value) => SystemEvent::ConfigurationChanged {
                setting,
                old_value: Value::Null,
                new_value: value,
            },
        };
        
        let _ = self.event_bus.publish(event).await;
    }
    
    /// Create a state snapshot
    pub async fn create_snapshot(&self) {
        let snapshot = StateSnapshot {
            timestamp: Instant::now(),
            chat_state: self.create_chat_snapshot().await,
            utilities_state: self.create_utilities_snapshot().await,
            memory_state: self.create_memory_snapshot().await,
            cognitive_state: self.create_cognitive_snapshot().await,
        };
        
        let mut snapshots = self.snapshots.write().await;
        if snapshots.len() >= self.max_snapshots {
            snapshots.pop_front();
        }
        snapshots.push_back(snapshot);
        
        info!("State snapshot created");
    }
    
    /// Create chat state snapshot
    async fn create_chat_snapshot(&self) -> ChatStateSnapshot {
        let state = self.chat_state.read().await;
        ChatStateSnapshot {
            message_count: state.messages.len(),
            active_model: state.current_model.clone(),
            orchestration_enabled: state.orchestration_enabled,
        }
    }
    
    /// Create utilities state snapshot
    async fn create_utilities_snapshot(&self) -> UtilitiesStateSnapshot {
        let state = self.utilities_state.read().await;
        UtilitiesStateSnapshot {
            tools_configured: state.tools.len(),
            mcp_servers_active: state.mcp_servers.values().filter(|s| s.enabled).count(),
            plugins_loaded: state.plugins.values().filter(|p| p.loaded).count(),
        }
    }
    
    /// Create memory state snapshot
    async fn create_memory_snapshot(&self) -> MemoryStateSnapshot {
        let state = self.memory_state.read().await;
        let kg = state.knowledge_graph.read().await;
        MemoryStateSnapshot {
            knowledge_nodes: kg.node_count,
            vector_count: state.vector_db.vector_count,
            stories_active: state.stories.values().filter(|s| s.active).count(),
        }
    }
    
    /// Create cognitive state snapshot
    async fn create_cognitive_snapshot(&self) -> CognitiveStateSnapshot {
        let state = self.cognitive_state.read().await;
        CognitiveStateSnapshot {
            reasoning_chains_active: state.reasoning_chains.len(),
            insights_generated: state.insights.len(),
            goals_active: state.goals.values().filter(|g| matches!(g.status, GoalStatus::InProgress)).count(),
        }
    }
    
    /// Get snapshots
    pub async fn get_snapshots(&self) -> Vec<StateSnapshot> {
        self.snapshots.read().await.iter().cloned().collect()
    }
    
    /// Rollback to snapshot
    pub async fn rollback_to_snapshot(&self, index: usize) -> Result<()> {
        let snapshots = self.snapshots.read().await;
        if let Some(_snapshot) = snapshots.get(index) {
            // Implement rollback logic here
            // This would involve restoring each state component
            warn!("Rollback not yet fully implemented");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Snapshot index {} not found", index))
        }
    }
}

// Default implementations for state components
impl Default for ChatState {
    fn default() -> Self {
        ChatState::new(0, "default".to_string())
    }
}

impl UtilitiesState {
    fn new() -> Self {
        Self {
            tools: HashMap::new(),
            mcp_servers: HashMap::new(),
            plugins: HashMap::new(),
            daemons: HashMap::new(),
            tool_history: VecDeque::with_capacity(100),
        }
    }
}

impl MemoryState {
    fn new() -> Self {
        Self {
            knowledge_graph: Arc::new(RwLock::new(KnowledgeGraphState {
                node_count: 0,
                edge_count: 0,
                last_updated: Instant::now(),
                active_queries: Vec::new(),
            })),
            vector_db: VectorDbState {
                vector_count: 0,
                dimensions: 768,
                index_type: "HNSW".to_string(),
                last_rebuild: None,
            },
            stories: HashMap::new(),
            databases: HashMap::new(),
            stats: MemoryStats::default(),
        }
    }
}

impl CognitiveState {
    fn new() -> Self {
        Self {
            reasoning_chains: Vec::new(),
            insights: VecDeque::with_capacity(100),
            goals: HashMap::new(),
            attention: AttentionState {
                focus_areas: Vec::new(),
                attention_weights: HashMap::new(),
                context_window_size: 2048,
            },
            metrics: CognitiveMetrics::default(),
        }
    }
}

impl HomeState {
    fn new() -> Self {
        Self {
            metrics: SystemMetrics::default(),
            alerts: VecDeque::with_capacity(50),
            health: HealthStatus {
                overall: HealthLevel::Healthy,
                components: HashMap::new(),
                last_check: Instant::now(),
            },
            keybindings: HashMap::new(),
        }
    }
}

impl SettingsState {
    fn new() -> Self {
        Self {
            app_settings: HashMap::new(),
            preferences: UserPreferences {
                auto_save: true,
                notifications_enabled: true,
                telemetry_enabled: false,
                default_model: None,
                default_temperature: 0.7,
            },
            api_configs: HashMap::new(),
            theme: ThemeSettings {
                theme_name: "dark".to_string(),
                dark_mode: true,
                accent_color: "#00ff00".to_string(),
                font_size: 12,
            },
        }
    }
}

impl TabRegistry {
    fn new() -> Self {
        let mut tabs = HashMap::new();
        
        // Register all tabs with their capabilities
        tabs.insert(TabId::Home, TabInfo {
            id: TabId::Home,
            name: "Home".to_string(),
            capabilities: vec![TabCapability::MonitorMetrics],
            shortcuts: vec!["1".to_string(), "h".to_string()],
            active: true,
        });
        
        tabs.insert(TabId::Chat, TabInfo {
            id: TabId::Chat,
            name: "Chat".to_string(),
            capabilities: vec![
                TabCapability::ExecuteTools,
                TabCapability::ManageModels,
                TabCapability::ProcessReasoning,
            ],
            shortcuts: vec!["2".to_string(), "c".to_string()],
            active: true,
        });
        
        tabs.insert(TabId::Utilities, TabInfo {
            id: TabId::Utilities,
            name: "Utilities".to_string(),
            capabilities: vec![TabCapability::ExecuteTools, TabCapability::ConfigureSettings],
            shortcuts: vec!["3".to_string(), "u".to_string()],
            active: true,
        });
        
        tabs.insert(TabId::Memory, TabInfo {
            id: TabId::Memory,
            name: "Memory".to_string(),
            capabilities: vec![TabCapability::StoreMemory, TabCapability::RetrieveContext],
            shortcuts: vec!["4".to_string(), "m".to_string()],
            active: true,
        });
        
        tabs.insert(TabId::Cognitive, TabInfo {
            id: TabId::Cognitive,
            name: "Cognitive".to_string(),
            capabilities: vec![TabCapability::ProcessReasoning, TabCapability::GenerateInsights],
            shortcuts: vec!["5".to_string(), "g".to_string()],
            active: true,
        });
        
        tabs.insert(TabId::Settings, TabInfo {
            id: TabId::Settings,
            name: "Settings".to_string(),
            capabilities: vec![TabCapability::ConfigureSettings],
            shortcuts: vec!["6".to_string(), "s".to_string()],
            active: true,
        });
        
        Self { tabs }
    }
}

impl ContextManager {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            global_context: GlobalContext {
                current_conversation: None,
                active_model: None,
                active_tools: Vec::new(),
                user_preferences: HashMap::new(),
            },
            context_history: VecDeque::with_capacity(50),
        }
    }
}

impl CommandRouter {
    fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a command handler
    pub async fn register(&self, command: String, handler: CommandHandler) {
        let mut routes = self.routes.write().await;
        let command_name = command.clone();
        routes.insert(command, handler);
        info!("Command handler registered for: {}", command_name);
    }
    
    /// Route a command to its handler
    pub async fn route(&self, command: Command) -> Result<CommandResult> {
        let command_name = command.command.clone();
        let routes = self.routes.read().await;
        if let Some(handler) = routes.get(&command_name) {
            handler(command)
        } else {
            Ok(CommandResult {
                success: false,
                message: format!("No handler for command: {}", command_name),
                data: None,
            })
        }
    }
}