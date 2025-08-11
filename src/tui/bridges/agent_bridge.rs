//! Agent Bridge - Shares agent configurations across tabs
//! 
//! This bridge enables agent configurations from the agents tab to be
//! accessible in the chat tab for task streaming and multi-agent collaboration.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::tui::event_bus::{EventBus, SystemEvent, TabId, Subscriber, create_handler};
use crate::tui::chat::agents::manager::{AgentManager, CollaborationMode};
use crate::tui::chat::agents::collaboration::{
    CollaborationCoordinator, CollaborationConfig, CollaborativeTask, CoordinationStrategy,
};
use crate::cognitive::agents::AgentSpecialization;

/// Agent bridge for configuration sharing
pub struct AgentBridge {
    /// Event bus for cross-tab communication
    event_bus: Arc<EventBus>,
    
    /// Shared agent manager
    agent_manager: Arc<RwLock<AgentManager>>,
    
    /// Collaboration coordinator
    collaboration_coordinator: Arc<CollaborationCoordinator>,
    
    /// Agent configurations by specialization
    agent_configs: Arc<RwLock<HashMap<AgentSpecialization, AgentConfig>>>,
    
    /// Active agent sessions
    active_sessions: Arc<RwLock<HashMap<String, AgentSession>>>,
    
    /// Bridge state
    state: Arc<RwLock<BridgeState>>,
}

/// Individual agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub specialization: AgentSpecialization,
    pub enabled: bool,
    pub priority: u8,
    pub max_concurrent_tasks: usize,
    pub timeout_seconds: u64,
    pub capabilities: Vec<String>,
    pub custom_prompts: HashMap<String, String>,
    pub model_preferences: Vec<String>,
}

/// Active agent session
#[derive(Debug, Clone)]
pub struct AgentSession {
    pub session_id: String,
    pub agent_specialization: AgentSpecialization,
    pub task_id: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub status: SessionStatus,
    pub progress: f32,
    pub messages: Vec<String>,
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionStatus {
    Initializing,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Bridge state
#[derive(Debug, Clone)]
struct BridgeState {
    initialized: bool,
    last_sync: chrono::DateTime<chrono::Utc>,
    active_tabs: Vec<TabId>,
    pending_updates: Vec<AgentUpdate>,
}

/// Agent update notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUpdate {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub update_type: UpdateType,
    pub specialization: Option<AgentSpecialization>,
    pub data: serde_json::Value,
}

/// Types of agent updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    ConfigurationChanged,
    AgentEnabled,
    AgentDisabled,
    CollaborationModeChanged,
    SpecializationAdded,
    SpecializationRemoved,
    SessionStarted,
    SessionCompleted,
    ProgressUpdate,
}

impl AgentBridge {
    /// Create a new agent bridge
    pub fn new(
        event_bus: Arc<EventBus>,
        agent_manager: Arc<RwLock<AgentManager>>,
    ) -> Self {
        // Create collaboration coordinator with default config
        let collab_config = CollaborationConfig::default();
        let collaboration_coordinator = Arc::new(CollaborationCoordinator::new(collab_config));
        
        Self {
            event_bus,
            agent_manager,
            collaboration_coordinator,
            agent_configs: Arc::new(RwLock::new(Self::default_agent_configs())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(BridgeState {
                initialized: false,
                last_sync: chrono::Utc::now(),
                active_tabs: vec![TabId::Chat, TabId::Settings],
                pending_updates: Vec::new(),
            })),
        }
    }
    
    /// Get default agent configurations
    fn default_agent_configs() -> HashMap<AgentSpecialization, AgentConfig> {
        let mut configs = HashMap::new();
        
        // Analytical agent
        configs.insert(AgentSpecialization::Analytical, AgentConfig {
            specialization: AgentSpecialization::Analytical,
            enabled: true,
            priority: 10,
            max_concurrent_tasks: 3,
            timeout_seconds: 60,
            capabilities: vec![
                "data_analysis".to_string(),
                "pattern_recognition".to_string(),
                "statistical_modeling".to_string(),
            ],
            custom_prompts: HashMap::new(),
            model_preferences: vec!["gpt-4".to_string(), "claude-3-opus".to_string()],
        });
        
        // Creative agent
        configs.insert(AgentSpecialization::Creative, AgentConfig {
            specialization: AgentSpecialization::Creative,
            enabled: true,
            priority: 8,
            max_concurrent_tasks: 2,
            timeout_seconds: 90,
            capabilities: vec![
                "content_generation".to_string(),
                "brainstorming".to_string(),
                "narrative_design".to_string(),
            ],
            custom_prompts: HashMap::new(),
            model_preferences: vec!["claude-3-opus".to_string(), "gpt-4".to_string()],
        });
        
        // Strategic agent
        configs.insert(AgentSpecialization::Strategic, AgentConfig {
            specialization: AgentSpecialization::Strategic,
            enabled: true,
            priority: 9,
            max_concurrent_tasks: 2,
            timeout_seconds: 120,
            capabilities: vec![
                "planning".to_string(),
                "decision_making".to_string(),
                "risk_assessment".to_string(),
            ],
            custom_prompts: HashMap::new(),
            model_preferences: vec!["gpt-4".to_string(), "gemini-pro".to_string()],
        });
        
        // Technical agent
        configs.insert(AgentSpecialization::Technical, AgentConfig {
            specialization: AgentSpecialization::Technical,
            enabled: true,
            priority: 10,
            max_concurrent_tasks: 4,
            timeout_seconds: 60,
            capabilities: vec![
                "code_generation".to_string(),
                "debugging".to_string(),
                "architecture_design".to_string(),
            ],
            custom_prompts: HashMap::new(),
            model_preferences: vec!["codellama".to_string(), "gpt-4".to_string()],
        });
        
        // Empathetic agent
        configs.insert(AgentSpecialization::Empathetic, AgentConfig {
            specialization: AgentSpecialization::Empathetic,
            enabled: false,
            priority: 7,
            max_concurrent_tasks: 2,
            timeout_seconds: 60,
            capabilities: vec![
                "emotional_intelligence".to_string(),
                "user_support".to_string(),
                "conflict_resolution".to_string(),
            ],
            custom_prompts: HashMap::new(),
            model_preferences: vec!["claude-3-sonnet".to_string()],
        });
        
        configs
    }
    
    /// Initialize the bridge
    pub async fn initialize(&self) -> Result<()> {
        // Subscribe to agent events
        self.subscribe_to_events().await?;
        
        // Load initial configurations
        self.sync_configurations().await?;
        
        // Start monitoring task
        self.start_monitoring().await;
        
        let mut state = self.state.write().await;
        state.initialized = true;
        
        tracing::info!("Agent bridge initialized");
        Ok(())
    }
    
    /// Subscribe to relevant events
    async fn subscribe_to_events(&self) -> Result<()> {
        let bus = self.event_bus.clone();
        let bridge = self.clone();
        
        // Subscribe to agent configuration events
        // We'll use subscribe_global to get all events and filter in the handler
        let subscriber = Subscriber {
            id: uuid::Uuid::new_v4().to_string(),
            tab_id: TabId::Settings,
            handler: create_handler(move |event| {
                let bridge = bridge.clone();
                tokio::spawn(async move {
                    match event {
                        SystemEvent::AgentConfig { config, .. } => {
                            let _ = bridge.handle_agent_config_update(config).await;
                        }
                        SystemEvent::AgentCreated { agent_id, .. } => {
                            let _ = bridge.handle_agent_created(agent_id).await;
                        }
                        SystemEvent::AgentStatusChanged { agent_id, status, .. } => {
                            let _ = bridge.handle_agent_status_change(agent_id, status).await;
                        }
                        _ => {}
                    }
                });
                Ok(())
            }),
            filter: None,
        };
        
        // Subscribe globally to receive all events
        bus.subscribe_global(subscriber).await?;
        
        Ok(())
    }
    
    /// Synchronize configurations with agent manager
    async fn sync_configurations(&self) -> Result<()> {
        let manager = self.agent_manager.read().await;
        let mut configs = self.agent_configs.write().await;
        
        // Update enabled status for each specialization
        for (spec, config) in configs.iter_mut() {
            config.enabled = manager.has_specialization(spec);
        }
        
        // Update active specializations
        for spec in &manager.active_specializations {
            if !configs.contains_key(spec) {
                // Add new specialization with default config
                configs.insert(spec.clone(), AgentConfig {
                    specialization: spec.clone(),
                    enabled: true,
                    priority: 5,
                    max_concurrent_tasks: 2,
                    timeout_seconds: 60,
                    capabilities: Vec::new(),
                    custom_prompts: HashMap::new(),
                    model_preferences: Vec::new(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Start monitoring task
    async fn start_monitoring(&self) {
        let bridge = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Process pending updates
                if let Err(e) = bridge.process_pending_updates_internal().await {
                    tracing::error!("Error processing agent updates: {}", e);
                }
                
                // Clean up completed sessions
                if let Err(e) = bridge.cleanup_sessions().await {
                    tracing::error!("Error cleaning up sessions: {}", e);
                }
            }
        });
    }
    
    /// Process pending updates (internal)
    async fn process_pending_updates_internal(&self) -> Result<()> {
        let mut state = self.state.write().await;
        
        for update in state.pending_updates.drain(..) {
            self.broadcast_update(update).await?;
        }
        
        state.last_sync = chrono::Utc::now();
        Ok(())
    }
    
    /// Clean up completed sessions
    async fn cleanup_sessions(&self) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        let now = chrono::Utc::now();
        
        sessions.retain(|_, session| {
            // Keep sessions that are still active or recently completed
            match session.status {
                SessionStatus::Running | SessionStatus::Initializing => true,
                SessionStatus::Completed | SessionStatus::Failed | SessionStatus::Cancelled => {
                    (now - session.started_at).num_seconds() < 300 // Keep for 5 minutes
                }
                SessionStatus::Paused => true,
            }
        });
        
        Ok(())
    }
    
    /// Handle agent configuration update
    async fn handle_agent_config_update(&self, config: serde_json::Value) -> Result<()> {
        if let Ok(agent_config) = serde_json::from_value::<AgentConfig>(config.clone()) {
            let mut configs = self.agent_configs.write().await;
            configs.insert(agent_config.specialization.clone(), agent_config.clone());
            
            // Create update notification
            let update = AgentUpdate {
                timestamp: chrono::Utc::now(),
                update_type: UpdateType::ConfigurationChanged,
                specialization: Some(agent_config.specialization),
                data: config,
            };
            
            self.state.write().await.pending_updates.push(update);
        }
        
        Ok(())
    }
    
    /// Handle agent created event
    async fn handle_agent_created(&self, agent_id: String) -> Result<()> {
        tracing::debug!("Agent created: {}", agent_id);
        
        // Sync configurations
        self.sync_configurations().await?;
        
        Ok(())
    }
    
    /// Handle agent status change
    async fn handle_agent_status_change(&self, agent_id: String, status: String) -> Result<()> {
        if let Some(session) = self.active_sessions.write().await.get_mut(&agent_id) {
            session.status = self.parse_status(&status);
            
            // Broadcast progress update
            let update = AgentUpdate {
                timestamp: chrono::Utc::now(),
                update_type: UpdateType::ProgressUpdate,
                specialization: Some(session.agent_specialization.clone()),
                data: serde_json::json!({
                    "session_id": session.session_id,
                    "status": status,
                    "progress": session.progress,
                }),
            };
            
            self.broadcast_update(update).await?;
        }
        
        Ok(())
    }
    
    /// Broadcast update to subscribers
    async fn broadcast_update(&self, update: AgentUpdate) -> Result<()> {
        let event = SystemEvent::CustomEvent {
            source: TabId::Settings,
            name: "agent_update".to_string(),
            data: serde_json::to_value(update)?,
            target: None,
        };
        
        self.event_bus.publish(event).await?;
        Ok(())
    }
    
    /// Start an agent session
    pub async fn start_session(
        &self,
        specialization: AgentSpecialization,
        task_id: String,
    ) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        let session = AgentSession {
            session_id: session_id.clone(),
            agent_specialization: specialization.clone(),
            task_id,
            started_at: chrono::Utc::now(),
            status: SessionStatus::Initializing,
            progress: 0.0,
            messages: Vec::new(),
        };
        
        self.active_sessions.write().await.insert(session_id.clone(), session);
        
        // Notify about session start
        let update = AgentUpdate {
            timestamp: chrono::Utc::now(),
            update_type: UpdateType::SessionStarted,
            specialization: Some(specialization),
            data: serde_json::json!({
                "session_id": session_id,
            }),
        };
        
        self.broadcast_update(update).await?;
        
        Ok(session_id)
    }
    
    /// Start agent lifecycle
    pub async fn start_agent_lifecycle(
        &self,
        specialization: AgentSpecialization,
    ) -> Result<()> {
        // Get agent config
        let configs = self.agent_configs.read().await;
        let config = configs.get(&specialization)
            .ok_or_else(|| anyhow::anyhow!("Agent config not found for {:?}", specialization))?;
        
        if !config.enabled {
            return Err(anyhow::anyhow!("Agent {:?} is not enabled", specialization));
        }
        
        // Update agent manager
        let mut manager = self.agent_manager.write().await;
        manager.enable_agent_system();
        
        // Broadcast lifecycle start
        let update = AgentUpdate {
            timestamp: chrono::Utc::now(),
            update_type: UpdateType::AgentEnabled,
            specialization: Some(specialization),
            data: serde_json::json!({
                "lifecycle": "started",
                "config": config,
            }),
        };
        
        self.broadcast_update(update).await?;
        Ok(())
    }
    
    /// Stop agent lifecycle
    pub async fn stop_agent_lifecycle(
        &self,
        specialization: AgentSpecialization,
    ) -> Result<()> {
        // Cancel any active sessions for this agent
        let mut sessions = self.active_sessions.write().await;
        let cancelled_sessions: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| s.agent_specialization == specialization && s.status == SessionStatus::Running)
            .map(|(id, _)| id.clone())
            .collect();
        
        for session_id in cancelled_sessions {
            if let Some(session) = sessions.get_mut(&session_id) {
                session.status = SessionStatus::Cancelled;
            }
        }
        
        // Broadcast lifecycle stop
        let update = AgentUpdate {
            timestamp: chrono::Utc::now(),
            update_type: UpdateType::AgentDisabled,
            specialization: Some(specialization),
            data: serde_json::json!({
                "lifecycle": "stopped",
                "cancelled_sessions": sessions.len(),
            }),
        };
        
        self.broadcast_update(update).await?;
        Ok(())
    }
    
    /// Negotiate task assignment between agents
    pub async fn negotiate_task_assignment(
        &self,
        task_description: String,
        required_capabilities: Vec<String>,
    ) -> Result<AgentSpecialization> {
        let configs = self.agent_configs.read().await;
        
        // Score each agent based on capabilities match
        let mut best_agent = None;
        let mut best_score = 0.0;
        
        for (spec, config) in configs.iter() {
            if !config.enabled {
                continue;
            }
            
            // Calculate capability match score
            let matching_caps = required_capabilities
                .iter()
                .filter(|cap| config.capabilities.contains(cap))
                .count();
            
            let score = (matching_caps as f32 / required_capabilities.len() as f32) 
                * (config.priority as f32 / 10.0);
            
            if score > best_score {
                best_score = score;
                best_agent = Some(spec.clone());
            }
        }
        
        best_agent.ok_or_else(|| anyhow::anyhow!("No suitable agent found for task"))
    }
    
    /// Coordinate multiple agents for a complex task
    pub async fn coordinate_agents(
        &self,
        task_id: String,
        agents: Vec<AgentSpecialization>,
        coordination_mode: CollaborationMode,
    ) -> Result<()> {
        // Update agent manager with collaboration mode
        let mut manager = self.agent_manager.write().await;
        manager.enable_with_mode(coordination_mode);
        
        // Register agents with collaboration coordinator
        for spec in agents.iter() {
            self.collaboration_coordinator
                .register_agent(
                    format!("agent_{:?}", spec),
                    spec.clone(),
                )
                .await?;
        }
        
        // Create collaborative task
        let task = CollaborativeTask {
            id: task_id.clone(),
            description: format!("Coordinated task for {} agents", agents.len()),
            required_specializations: agents.clone(),
            priority: 0.5,
            context: HashMap::new(),
        };
        
        // Determine coordination strategy based on collaboration mode
        let strategy = match coordination_mode {
            CollaborationMode::Independent => CoordinationStrategy::Parallel,
            CollaborationMode::Coordinated => CoordinationStrategy::Sequential,
            CollaborationMode::Hierarchical => CoordinationStrategy::Pipeline,
            CollaborationMode::Democratic => CoordinationStrategy::Consensus,
        };
        
        // Execute task through coordinator
        let agent_ids: Vec<String> = agents.iter()
            .map(|spec| format!("agent_{:?}", spec))
            .collect();
        
        let result = self.collaboration_coordinator
            .coordinate_task_execution(task_id.clone(), agent_ids, strategy)
            .await?;
        
        // Start sessions for tracking
        let mut session_ids = Vec::new();
        for spec in agents.iter() {
            match self.start_session(spec.clone(), task_id.clone()).await {
                Ok(session_id) => session_ids.push(session_id),
                Err(e) => {
                    tracing::warn!("Failed to start session for {:?}: {}", spec, e);
                }
            }
        }
        
        // Broadcast coordination event with result
        let update = AgentUpdate {
            timestamp: chrono::Utc::now(),
            update_type: UpdateType::CollaborationModeChanged,
            specialization: None,
            data: serde_json::json!({
                "task_id": task_id,
                "agents": agents,
                "mode": coordination_mode,
                "sessions": session_ids,
                "result": result.result,
                "consensus_score": result.consensus_score,
            }),
        };
        
        self.broadcast_update(update).await?;
        Ok(())
    }
    
    /// Execute collaborative task through the coordinator
    pub async fn execute_collaborative_task(
        &self,
        task: CollaborativeTask,
        strategy: CoordinationStrategy,
    ) -> Result<String> {
        // Register required agents if not already registered
        for spec in &task.required_specializations {
            let agent_id = format!("agent_{:?}", spec);
            // Try to register, ignore if already exists
            let _ = self.collaboration_coordinator
                .register_agent(agent_id.clone(), spec.clone())
                .await;
        }
        
        // Get agent IDs
        let agent_ids: Vec<String> = task.required_specializations
            .iter()
            .map(|spec| format!("agent_{:?}", spec))
            .collect();
        
        // Execute through coordinator
        let result = self.collaboration_coordinator
            .coordinate_task_execution(task.id.clone(), agent_ids, strategy)
            .await?;
        
        Ok(result.result)
    }
    
    /// Update session progress
    pub async fn update_session_progress(
        &self,
        session_id: &str,
        progress: f32,
        message: Option<String>,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.progress = progress;
            if let Some(msg) = message {
                session.messages.push(msg);
            }
            
            if progress >= 1.0 {
                session.status = SessionStatus::Completed;
            }
        }
        
        Ok(())
    }
    
    /// Get agent configuration for a specialization
    pub async fn get_agent_config(&self, specialization: &AgentSpecialization) -> Option<AgentConfig> {
        self.agent_configs.read().await.get(specialization).cloned()
    }
    
    /// Get all active agent configurations
    pub async fn get_active_configs(&self) -> Vec<AgentConfig> {
        self.agent_configs.read().await
            .values()
            .filter(|c| c.enabled)
            .cloned()
            .collect()
    }
    
    /// Get active sessions
    pub async fn get_active_sessions(&self) -> Vec<AgentSession> {
        self.active_sessions.read().await
            .values()
            .filter(|s| matches!(s.status, SessionStatus::Running | SessionStatus::Initializing))
            .cloned()
            .collect()
    }
    
    /// Check if agent system is enabled
    pub async fn is_enabled(&self) -> bool {
        self.agent_manager.read().await.agent_system_enabled
    }
    
    /// Get collaboration mode
    pub async fn get_collaboration_mode(&self) -> CollaborationMode {
        self.agent_manager.read().await.collaboration_mode
    }
    
    /// Update agent manager settings from chat
    pub async fn update_from_chat(&self, updates: AgentSystemUpdate) -> Result<()> {
        let mut manager = self.agent_manager.write().await;
        
        if let Some(enabled) = updates.enabled {
            manager.agent_system_enabled = enabled;
        }
        
        if let Some(mode) = updates.collaboration_mode {
            manager.collaboration_mode = mode;
        }
        
        if let Some(specs) = updates.specializations {
            manager.active_specializations = specs;
        }
        
        if let Some(threshold) = updates.consensus_threshold {
            manager.consensus_threshold = threshold;
        }
        
        Ok(())
    }
    
    /// Parse status string to enum
    fn parse_status(&self, status: &str) -> SessionStatus {
        match status.to_lowercase().as_str() {
            "initializing" => SessionStatus::Initializing,
            "running" => SessionStatus::Running,
            "paused" => SessionStatus::Paused,
            "completed" => SessionStatus::Completed,
            "failed" => SessionStatus::Failed,
            "cancelled" => SessionStatus::Cancelled,
            _ => SessionStatus::Running,
        }
    }
}

/// Updates from chat tab
#[derive(Debug, Clone)]
pub struct AgentSystemUpdate {
    pub enabled: Option<bool>,
    pub collaboration_mode: Option<CollaborationMode>,
    pub specializations: Option<Vec<AgentSpecialization>>,
    pub consensus_threshold: Option<f32>,
}

impl Clone for AgentBridge {
    fn clone(&self) -> Self {
        Self {
            event_bus: self.event_bus.clone(),
            agent_manager: self.agent_manager.clone(),
            collaboration_coordinator: self.collaboration_coordinator.clone(),
            agent_configs: self.agent_configs.clone(),
            active_sessions: self.active_sessions.clone(),
            state: self.state.clone(),
        }
    }
}

impl AgentBridge {
    /// Register a tab as active for agent bridge synchronization
    pub async fn register_tab(&self, tab_id: TabId) {
        let tab_id_for_log = tab_id.clone();
        let mut state = self.state.write().await;
        if !state.active_tabs.contains(&tab_id) {
            state.active_tabs.push(tab_id);
            tracing::info!("Registered tab {:?} for agent bridge sync", tab_id_for_log);
        }
    }
    
    /// Unregister a tab from agent bridge synchronization
    pub async fn unregister_tab(&self, tab_id: TabId) {
        let mut state = self.state.write().await;
        state.active_tabs.retain(|t| t != &tab_id);
        tracing::info!("Unregistered tab {:?} from agent bridge sync", tab_id);
    }
    
    /// Get list of active tabs
    pub async fn get_active_tabs(&self) -> Vec<TabId> {
        self.state.read().await.active_tabs.clone()
    }
    
    /// Check if a tab is registered for sync
    pub async fn is_tab_active(&self, tab_id: TabId) -> bool {
        self.state.read().await.active_tabs.contains(&tab_id)
    }
    
    /// Process pending updates and notify active tabs
    pub async fn process_pending_updates(&self) -> Result<()> {
        let mut state = self.state.write().await;
        let updates = state.pending_updates.drain(..).collect::<Vec<_>>();
        let active_tabs = state.active_tabs.clone();
        drop(state); // Release lock before processing
        
        for update in updates {
            // Notify all active tabs about the update
            for tab_id in &active_tabs {
                self.notify_tab_of_update(tab_id.clone(), &update).await;
            }
        }
        
        Ok(())
    }
    
    /// Notify a specific tab about an agent update
    async fn notify_tab_of_update(&self, tab_id: TabId, update: &AgentUpdate) {
        tracing::debug!("Notifying tab {:?} about agent update: {:?}", tab_id, update.update_type);
        
        // Create appropriate event based on update type
        let event = match &update.update_type {
            UpdateType::ConfigurationChanged => SystemEvent::ConfigurationChanged {
                setting: format!("agent.{:?}", update.specialization.as_ref().unwrap_or(&AgentSpecialization::General)),
                old_value: serde_json::Value::Null,
                new_value: update.data.clone(),
            },
            UpdateType::SessionStarted | UpdateType::SessionCompleted => SystemEvent::AgentStatusChanged {
                agent_id: update.specialization.as_ref().map(|s| format!("{:?}", s)).unwrap_or_default(),
                status: format!("{:?}", update.update_type),
            },
            _ => SystemEvent::CrossTabMessage {
                from: TabId::Settings,
                to: tab_id,
                message: serde_json::json!({
                    "type": format!("agent_{:?}", update.update_type),
                    "data": update.data
                }),
            },
        };
        
        self.event_bus.publish(event).await;
    }
}