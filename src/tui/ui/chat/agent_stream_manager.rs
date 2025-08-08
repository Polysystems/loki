//! Agent Stream Manager for TUI Chat Interface
//!
//! Manages multiple agent message streams, routing, and visualization
//! to enable parallel agent execution without cluttering the main chat.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;


/// Maximum messages to keep per agent stream
const MAX_MESSAGES_PER_STREAM: usize = 100;

/// Agent stream manager
#[derive(Clone)]
pub struct AgentStreamManager {
    /// Active agent streams
    agent_streams: Arc<RwLock<HashMap<String, AgentStream>>>,
    
    /// Panel allocation for agents
    agent_panels: Arc<RwLock<HashMap<String, String>>>, // agent_id -> panel_id
    
    /// Maximum concurrent agent panels
    max_agent_panels: usize,
    
    /// Stream routing rules
    routing_rules: Arc<RwLock<StreamRoutingRules>>,
    
    /// Message channel for updates
    update_tx: mpsc::Sender<AgentStreamUpdate>,
    update_rx: Arc<RwLock<mpsc::Receiver<AgentStreamUpdate>>>,
}

/// Individual agent stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStream {
    pub agent_id: String,
    pub agent_type: String,
    pub agent_name: String,
    pub task_description: String,
    pub messages: VecDeque<AgentMessage>,
    pub status: AgentStatus,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub progress: f32,
    pub metadata: HashMap<String, serde_json::Value>,
    // Task context fields
    pub parent_task_id: Option<String>,
    pub parent_task_description: Option<String>,
    pub subtask_id: Option<String>,
    pub subtask_type: Option<String>,
    pub estimated_effort: Option<std::time::Duration>,
    pub dependencies: Vec<String>,
    pub parallel_group: Option<usize>,
}

/// Agent message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content: String,
    pub message_type: AgentMessageType,
    pub priority: MessagePriority,
    pub metadata: Option<serde_json::Value>,
}

/// Agent message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessageType {
    Thought,           // Agent's reasoning
    Action,            // Action being taken
    Observation,       // Result of action
    Progress,          // Progress update
    Error,             // Error occurred
    Result,            // Final result
    Debug,             // Debug information
    ToolInvocation,    // Tool being called
    ToolResult,        // Tool output
}

/// Message priority for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Critical,  // Always show in main chat
    High,      // Important updates
    Normal,    // Regular agent activity
    Low,       // Detailed logs
    Debug,     // Debug only
}

/// Agent status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Initializing,
    Running,
    Thinking,
    ExecutingTool,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Stream routing rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRoutingRules {
    /// Route critical messages to main chat
    pub route_critical_to_main: bool,
    
    /// Route errors to main chat
    pub route_errors_to_main: bool,
    
    /// Route final results to main chat
    pub route_results_to_main: bool,
    
    /// Minimum priority for main chat
    pub main_chat_min_priority: MessagePriority,
    
    /// Enable agent panel auto-creation
    pub auto_create_panels: bool,
    
    /// Collapse completed agent panels
    pub auto_collapse_completed: bool,
}

impl Default for StreamRoutingRules {
    fn default() -> Self {
        Self {
            route_critical_to_main: true,
            route_errors_to_main: true,
            route_results_to_main: true,
            main_chat_min_priority: MessagePriority::High,
            auto_create_panels: true,
            auto_collapse_completed: false,
        }
    }
}

/// Updates from agent streams
#[derive(Debug, Clone)]
pub enum AgentStreamUpdate {
    NewMessage {
        agent_id: String,
        message: AgentMessage,
    },
    StatusChange {
        agent_id: String,
        old_status: AgentStatus,
        new_status: AgentStatus,
    },
    ProgressUpdate {
        agent_id: String,
        progress: f32,
    },
    StreamCreated {
        agent_id: String,
        stream: AgentStream,
    },
    StreamClosed {
        agent_id: String,
        reason: String,
    },
}

/// Stream target for routing
#[derive(Debug, Clone, PartialEq)]
pub enum StreamTarget {
    MainChat,
    AgentPanel(String),
    Background,
    Both { main: bool, panel: bool },
}

impl AgentStreamManager {
    /// Create a new agent stream manager
    pub fn new(max_agent_panels: usize) -> Self {
        let (update_tx, update_rx) = mpsc::channel(100);
        
        Self {
            agent_streams: Arc::new(RwLock::new(HashMap::new())),
            agent_panels: Arc::new(RwLock::new(HashMap::new())),
            max_agent_panels,
            routing_rules: Arc::new(RwLock::new(StreamRoutingRules::default())),
            update_tx,
            update_rx: Arc::new(RwLock::new(update_rx)),
        }
    }
    
    /// Create a new agent stream with task context
    pub async fn create_agent_stream_with_context(
        &self,
        agent_type: String,
        agent_name: String,
        task_description: String,
        parent_task_id: Option<String>,
        parent_task_description: Option<String>,
        subtask_id: Option<String>,
        subtask_type: Option<String>,
        estimated_effort: Option<std::time::Duration>,
        dependencies: Vec<String>,
        parallel_group: Option<usize>,
    ) -> Result<String> {
        let agent_id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        
        let stream = AgentStream {
            agent_id: agent_id.clone(),
            agent_type,
            agent_name,
            task_description,
            messages: VecDeque::with_capacity(MAX_MESSAGES_PER_STREAM),
            status: AgentStatus::Initializing,
            started_at: now,
            last_activity: now,
            progress: 0.0,
            metadata: HashMap::new(),
            parent_task_id,
            parent_task_description,
            subtask_id,
            subtask_type,
            estimated_effort,
            dependencies,
            parallel_group,
        };
        
        // Add to streams
        self.agent_streams.write().await.insert(agent_id.clone(), stream.clone());
        
        // Send update
        let _ = self.update_tx.send(AgentStreamUpdate::StreamCreated {
            agent_id: agent_id.clone(),
            stream,
        }).await;
        
        Ok(agent_id)
    }

    /// Create a new agent stream
    pub async fn create_agent_stream(
        &self,
        agent_type: String,
        agent_name: String,
        task_description: String,
    ) -> Result<String> {
        let agent_id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        
        let stream = AgentStream {
            agent_id: agent_id.clone(),
            agent_type,
            agent_name,
            task_description,
            messages: VecDeque::with_capacity(MAX_MESSAGES_PER_STREAM),
            status: AgentStatus::Initializing,
            started_at: now,
            last_activity: now,
            progress: 0.0,
            metadata: HashMap::new(),
            parent_task_id: None,
            parent_task_description: None,
            subtask_id: None,
            subtask_type: None,
            estimated_effort: None,
            dependencies: Vec::new(),
            parallel_group: None,
        };
        
        // Add to streams
        self.agent_streams.write().await.insert(agent_id.clone(), stream.clone());
        
        // Send update
        let _ = self.update_tx.send(AgentStreamUpdate::StreamCreated {
            agent_id: agent_id.clone(),
            stream,
        }).await;
        
        Ok(agent_id)
    }
    
    /// Add a message to an agent stream
    pub async fn add_message(
        &self,
        agent_id: &str,
        content: String,
        message_type: AgentMessageType,
        priority: MessagePriority,
        metadata: Option<serde_json::Value>,
    ) -> Result<StreamTarget> {
        let message = AgentMessage {
            id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            content,
            message_type: message_type.clone(),
            priority,
            metadata,
        };
        
        // Add to stream
        let mut streams = self.agent_streams.write().await;
        if let Some(stream) = streams.get_mut(agent_id) {
            // Update activity time
            stream.last_activity = chrono::Utc::now();
            
            // Add message (with size limit)
            stream.messages.push_back(message.clone());
            if stream.messages.len() > MAX_MESSAGES_PER_STREAM {
                stream.messages.pop_front();
            }
        }
        drop(streams);
        
        // Determine routing
        let target = self.route_message(agent_id, &message).await?;
        
        // Send update
        let _ = self.update_tx.send(AgentStreamUpdate::NewMessage {
            agent_id: agent_id.to_string(),
            message,
        }).await;
        
        Ok(target)
    }
    
    /// Route a message based on rules
    async fn route_message(
        &self,
        agent_id: &str,
        message: &AgentMessage,
    ) -> Result<StreamTarget> {
        let rules = self.routing_rules.read().await;
        
        let mut route_to_main = false;
        let route_to_panel = true;
        
        // Check priority routing
        if message.priority >= rules.main_chat_min_priority {
            route_to_main = true;
        }
        
        // Check specific routing rules
        match &message.message_type {
            AgentMessageType::Error if rules.route_errors_to_main => {
                route_to_main = true;
            }
            AgentMessageType::Result if rules.route_results_to_main => {
                route_to_main = true;
            }
            _ => {}
        }
        
        // Critical always goes to main
        if message.priority == MessagePriority::Critical {
            route_to_main = true;
        }
        
        // Debug messages only go to panel
        if message.priority == MessagePriority::Debug {
            route_to_main = false;
        }
        
        Ok(match (route_to_main, route_to_panel) {
            (true, true) => StreamTarget::Both { main: true, panel: true },
            (true, false) => StreamTarget::MainChat,
            (false, true) => StreamTarget::AgentPanel(agent_id.to_string()),
            (false, false) => StreamTarget::Background,
        })
    }
    
    /// Update agent status
    pub async fn update_status(
        &self,
        agent_id: &str,
        new_status: AgentStatus,
    ) -> Result<()> {
        let mut streams = self.agent_streams.write().await;
        if let Some(stream) = streams.get_mut(agent_id) {
            let old_status = stream.status;
            stream.status = new_status;
            stream.last_activity = chrono::Utc::now();
            
            drop(streams);
            
            // Send update
            let _ = self.update_tx.send(AgentStreamUpdate::StatusChange {
                agent_id: agent_id.to_string(),
                old_status,
                new_status,
            }).await;
            
            // Handle completion
            if matches!(new_status, AgentStatus::Completed | AgentStatus::Failed | AgentStatus::Cancelled) {
                let rules = self.routing_rules.read().await;
                if rules.auto_collapse_completed {
                    // Mark panel for collapse
                    self.agent_panels.write().await.remove(agent_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Update agent progress
    pub async fn update_progress(
        &self,
        agent_id: &str,
        progress: f32,
    ) -> Result<()> {
        let mut streams = self.agent_streams.write().await;
        if let Some(stream) = streams.get_mut(agent_id) {
            stream.progress = progress.clamp(0.0, 1.0);
            stream.last_activity = chrono::Utc::now();
            
            drop(streams);
            
            // Send update
            let _ = self.update_tx.send(AgentStreamUpdate::ProgressUpdate {
                agent_id: agent_id.to_string(),
                progress,
            }).await;
        }
        
        Ok(())
    }
    
    /// Get all active agent streams
    pub async fn get_active_streams(&self) -> Vec<AgentStream> {
        self.agent_streams.read().await
            .values()
            .filter(|stream| !matches!(
                stream.status,
                AgentStatus::Completed | AgentStatus::Failed | AgentStatus::Cancelled
            ))
            .cloned()
            .collect()
    }
    
    /// Get a specific agent stream
    pub async fn get_stream(&self, agent_id: &str) -> Option<AgentStream> {
        self.agent_streams.read().await.get(agent_id).cloned()
    }
    
    /// Allocate a panel for an agent
    pub async fn allocate_panel(&self, agent_id: &str) -> Option<String> {
        let mut panels = self.agent_panels.write().await;
        
        // Check if already allocated
        if let Some(panel_id) = panels.get(agent_id) {
            return Some(panel_id.clone());
        }
        
        // Check panel limit
        if panels.len() >= self.max_agent_panels {
            return None;
        }
        
        // Allocate new panel
        let panel_id = format!("agent_panel_{}", agent_id);
        panels.insert(agent_id.to_string(), panel_id.clone());
        
        Some(panel_id)
    }
    
    /// Get update receiver
    pub fn get_update_receiver(&self) -> Arc<RwLock<mpsc::Receiver<AgentStreamUpdate>>> {
        self.update_rx.clone()
    }
    
    /// Close an agent stream
    pub async fn close_stream(&self, agent_id: &str, reason: String) -> Result<()> {
        // Update status first
        self.update_status(agent_id, AgentStatus::Completed).await?;
        
        // Remove from active streams after a delay
        // (keep for history/review)
        let _ = self.update_tx.send(AgentStreamUpdate::StreamClosed {
            agent_id: agent_id.to_string(),
            reason,
        }).await;
        
        Ok(())
    }
    
    /// Get routing rules
    pub async fn get_routing_rules(&self) -> StreamRoutingRules {
        self.routing_rules.read().await.clone()
    }
    
    /// Update routing rules
    pub async fn update_routing_rules<F>(&self, updater: F)
    where
        F: FnOnce(&mut StreamRoutingRules),
    {
        let mut rules = self.routing_rules.write().await;
        updater(&mut *rules);
    }
}

impl AgentStream {
    /// Get recent messages
    pub fn get_recent_messages(&self, count: usize) -> Vec<&AgentMessage> {
        self.messages.iter().rev().take(count).collect()
    }
    
    /// Get messages by type
    pub fn get_messages_by_type(&self, message_type: &AgentMessageType) -> Vec<&AgentMessage> {
        self.messages.iter()
            .filter(|msg| std::mem::discriminant(&msg.message_type) == std::mem::discriminant(message_type))
            .collect()
    }
    
    /// Get high priority messages
    pub fn get_important_messages(&self) -> Vec<&AgentMessage> {
        self.messages.iter()
            .filter(|msg| msg.priority >= MessagePriority::High)
            .collect()
    }
    
    /// Get execution duration
    pub fn get_duration(&self) -> chrono::Duration {
        self.last_activity - self.started_at
    }
}

impl std::fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentStatus::Initializing => write!(f, "🔄 Initializing"),
            AgentStatus::Running => write!(f, "▶️ Running"),
            AgentStatus::Thinking => write!(f, "🤔 Thinking"),
            AgentStatus::ExecutingTool => write!(f, "🔧 Executing Tool"),
            AgentStatus::Paused => write!(f, "⏸️ Paused"),
            AgentStatus::Completed => write!(f, "✅ Completed"),
            AgentStatus::Failed => write!(f, "❌ Failed"),
            AgentStatus::Cancelled => write!(f, "🚫 Cancelled"),
        }
    }
}

impl std::fmt::Display for AgentMessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentMessageType::Thought => write!(f, "💭"),
            AgentMessageType::Action => write!(f, "⚡"),
            AgentMessageType::Observation => write!(f, "👁️"),
            AgentMessageType::Progress => write!(f, "📊"),
            AgentMessageType::Error => write!(f, "❌"),
            AgentMessageType::Result => write!(f, "✅"),
            AgentMessageType::Debug => write!(f, "🐛"),
            AgentMessageType::ToolInvocation => write!(f, "🔧"),
            AgentMessageType::ToolResult => write!(f, "📤"),
        }
    }
}