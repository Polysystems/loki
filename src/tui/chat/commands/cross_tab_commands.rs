//! Cross-Tab Command Execution
//! 
//! Enables command execution across different tabs with context sharing,
//! synchronization, and result propagation.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, broadcast};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use crate::tui::bridges::EventBridge;

/// Cross-tab command executor
pub struct CrossTabCommandExecutor {
    /// Event bridge for cross-tab communication
    event_bridge: Arc<EventBridge>,
    
    /// Command registry
    command_registry: Arc<RwLock<CommandRegistry>>,
    
    /// Active command executions
    active_executions: Arc<RwLock<HashMap<String, CommandExecution>>>,
    
    /// Command history
    command_history: Arc<RwLock<Vec<CommandHistoryEntry>>>,
    
    /// Command synchronizer
    synchronizer: Arc<CommandSynchronizer>,
    
    /// Event channel
    event_tx: broadcast::Sender<CommandEvent>,
    
    /// Configuration
    config: CommandExecutorConfig,
}

/// Command registry for available commands
#[derive(Debug, Clone)]
pub struct CommandRegistry {
    /// Registered commands by name
    commands: HashMap<String, RegisteredCommand>,
    
    /// Command aliases
    aliases: HashMap<String, String>,
    
    /// Tab-specific commands
    tab_commands: HashMap<TabId, Vec<String>>,
}

/// Registered command definition
#[derive(Debug, Clone)]
pub struct RegisteredCommand {
    pub name: String,
    pub description: String,
    pub parameters: Vec<CommandParameter>,
    pub source_tabs: Vec<TabId>,
    pub target_tabs: Vec<TabId>,
    pub requires_context: bool,
    pub supports_async: bool,
    pub category: CommandCategory,
}

/// Command parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandParameter {
    pub name: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default_value: Option<Value>,
    pub description: String,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Number,
    Boolean,
    Object,
    Array,
    TabId,
    ToolId,
    AgentId,
}

/// Command categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommandCategory {
    Navigation,
    Execution,
    Configuration,
    Communication,
    Analysis,
    Generation,
    Management,
    Custom(String),
}

/// Active command execution
#[derive(Debug, Clone)]
pub struct CommandExecution {
    pub id: String,
    pub command: String,
    pub parameters: HashMap<String, Value>,
    pub source_tab: TabId,
    pub target_tab: TabId,
    pub context: ExecutionContext,
    pub status: ExecutionStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub result: Option<CommandResult>,
}

/// Execution context for commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub user_context: Option<String>,
    pub tab_context: HashMap<TabId, Value>,
    pub shared_memory: HashMap<String, Value>,
    pub parent_execution_id: Option<String>,
    pub priority: ExecutionPriority,
}

/// Execution status
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// Execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExecutionPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Command result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub success: bool,
    pub output: Value,
    pub messages: Vec<String>,
    pub side_effects: Vec<SideEffect>,
    pub follow_up_commands: Vec<String>,
}

/// Side effects from command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffect {
    pub effect_type: String,
    pub tab: TabId,
    pub description: String,
    pub data: Option<Value>,
}

/// Command history entry
#[derive(Debug, Clone)]
pub struct CommandHistoryEntry {
    pub execution_id: String,
    pub command: String,
    pub timestamp: DateTime<Utc>,
    pub source_tab: TabId,
    pub target_tab: TabId,
    pub success: bool,
    pub duration_ms: u64,
}

/// Command synchronizer for coordinating executions
#[derive(Debug)]
pub struct CommandSynchronizer {
    /// Synchronization locks by command type
    locks: Arc<RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    
    /// Dependency graph
    dependencies: Arc<RwLock<HashMap<String, Vec<String>>>>,
    
    /// Conflict resolution strategy
    conflict_strategy: ConflictResolutionStrategy,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictResolutionStrategy {
    /// Queue conflicting commands
    Queue,
    
    /// Cancel older command
    CancelOlder,
    
    /// Merge parameters
    Merge,
    
    /// Reject new command
    Reject,
}

/// Command events
#[derive(Debug, Clone)]
pub enum CommandEvent {
    CommandQueued {
        execution_id: String,
        command: String,
    },
    CommandStarted {
        execution_id: String,
        tab: TabId,
    },
    CommandCompleted {
        execution_id: String,
        result: CommandResult,
    },
    CommandFailed {
        execution_id: String,
        error: String,
    },
    CommandCancelled {
        execution_id: String,
        reason: String,
    },
}

/// Configuration for command executor
#[derive(Debug, Clone)]
pub struct CommandExecutorConfig {
    pub max_concurrent_commands: usize,
    pub command_timeout_seconds: u64,
    pub enable_command_chaining: bool,
    pub enable_context_sharing: bool,
    pub history_limit: usize,
    pub retry_failed_commands: bool,
    pub max_retries: usize,
}

impl Default for CommandExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_commands: 10,
            command_timeout_seconds: 60,
            enable_command_chaining: true,
            enable_context_sharing: true,
            history_limit: 100,
            retry_failed_commands: true,
            max_retries: 3,
        }
    }
}

impl CrossTabCommandExecutor {
    /// Create a new cross-tab command executor
    pub fn new(event_bridge: Arc<EventBridge>) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        
        Self {
            event_bridge,
            command_registry: Arc::new(RwLock::new(CommandRegistry::new())),
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            command_history: Arc::new(RwLock::new(Vec::new())),
            synchronizer: Arc::new(CommandSynchronizer::new()),
            event_tx,
            config: CommandExecutorConfig::default(),
        }
    }
    
    /// Register a command
    pub async fn register_command(&self, command: RegisteredCommand) -> Result<()> {
        let mut registry = self.command_registry.write().await;
        registry.commands.insert(command.name.clone(), command.clone());
        
        // Update tab-specific commands
        for tab in &command.source_tabs {
            registry.tab_commands.entry(tab.clone())
                .or_insert_with(Vec::new)
                .push(command.name.clone());
        }
        
        info!("Registered command: {}", command.name);
        Ok(())
    }
    
    /// Execute a command across tabs
    pub async fn execute_command(
        &self,
        command_name: String,
        parameters: HashMap<String, Value>,
        source_tab: TabId,
        target_tab: TabId,
        context: Option<ExecutionContext>,
    ) -> Result<String> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        
        // Validate command exists
        let registry = self.command_registry.read().await;
        let command = registry.commands.get(&command_name)
            .ok_or_else(|| anyhow::anyhow!("Command not found: {}", command_name))?;
        
        // Validate parameters
        self.validate_parameters(command, &parameters)?;
        
        // Create execution context
        let exec_context = context.unwrap_or_else(|| ExecutionContext {
            user_context: None,
            tab_context: HashMap::new(),
            shared_memory: HashMap::new(),
            parent_execution_id: None,
            priority: ExecutionPriority::Normal,
        });
        
        // Create execution record
        let execution = CommandExecution {
            id: execution_id.clone(),
            command: command_name.clone(),
            parameters: parameters.clone(),
            source_tab: source_tab.clone(),
            target_tab: target_tab.clone(),
            context: exec_context,
            status: ExecutionStatus::Pending,
            started_at: Utc::now(),
            completed_at: None,
            result: None,
        };
        
        // Store execution
        self.active_executions.write().await.insert(execution_id.clone(), execution.clone());
        
        // Send queued event
        let _ = self.event_tx.send(CommandEvent::CommandQueued {
            execution_id: execution_id.clone(),
            command: command_name.clone(),
        });
        
        // Acquire synchronization lock if needed
        if let Some(lock) = self.synchronizer.get_lock(&command_name).await {
            let _guard = lock.lock().await;
            
            // Execute command
            self.execute_internal(execution).await?;
        } else {
            // Execute without lock
            self.execute_internal(execution).await?;
        }
        
        Ok(execution_id)
    }
    
    /// Internal execution logic
    async fn execute_internal(&self, mut execution: CommandExecution) -> Result<()> {
        // Update status
        execution.status = ExecutionStatus::Running;
        self.active_executions.write().await.insert(execution.id.clone(), execution.clone());
        
        // Send started event
        let _ = self.event_tx.send(CommandEvent::CommandStarted {
            execution_id: execution.id.clone(),
            tab: execution.target_tab.clone(),
        });
        
        // Publish cross-tab message
        self.event_bridge.publish(SystemEvent::CrossTabMessage {
            from: execution.source_tab.clone(),
            to: execution.target_tab.clone(),
            message: serde_json::json!({
                "type": "command_execution",
                "execution_id": execution.id,
                "command": execution.command,
                "parameters": execution.parameters,
                "context": execution.context,
            }),
        }).await?;
        
        // Simulate command execution (in real implementation, this would be handled by target tab)
        let result = self.simulate_command_execution(&execution).await?;
        
        // Update execution record
        execution.status = ExecutionStatus::Completed;
        execution.completed_at = Some(Utc::now());
        execution.result = Some(result.clone());
        
        self.active_executions.write().await.insert(execution.id.clone(), execution.clone());
        
        // Add to history
        self.add_to_history(&execution).await;
        
        // Send completed event
        let _ = self.event_tx.send(CommandEvent::CommandCompleted {
            execution_id: execution.id.clone(),
            result: result.clone(),
        });
        
        // Execute follow-up commands if enabled
        if self.config.enable_command_chaining {
            for follow_up in &result.follow_up_commands {
                let _ = self.execute_command(
                    follow_up.clone(),
                    HashMap::new(),
                    execution.target_tab.clone(),
                    execution.source_tab.clone(),
                    Some(ExecutionContext {
                        parent_execution_id: Some(execution.id.clone()),
                        ..execution.context.clone()
                    }),
                ).await;
            }
        }
        
        Ok(())
    }
    
    /// Validate command parameters
    fn validate_parameters(
        &self,
        command: &RegisteredCommand,
        parameters: &HashMap<String, Value>,
    ) -> Result<()> {
        for param in &command.parameters {
            if param.required && !parameters.contains_key(&param.name) {
                return Err(anyhow::anyhow!(
                    "Required parameter missing: {}",
                    param.name
                ));
            }
        }
        Ok(())
    }
    
    /// Execute command and return result
    async fn simulate_command_execution(&self, execution: &CommandExecution) -> Result<CommandResult> {
        // Match command name to actual implementation
        match execution.command.as_str() {
            "switch_tab" => self.execute_switch_tab(execution).await,
            "execute_tool" => self.execute_tool(execution).await,
            "sync_state" => self.execute_sync_state(execution).await,
            "broadcast_message" => self.execute_broadcast_message(execution).await,
            "query_data" => self.execute_query_data(execution).await,
            _ => {
                // For unknown commands, try to route through event bridge
                self.execute_generic_command(execution).await
            }
        }
    }
    
    /// Execute switch tab command
    async fn execute_switch_tab(&self, execution: &CommandExecution) -> Result<CommandResult> {
        let tab_id = execution.parameters.get("tab")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tab parameter"))?;
        
        // Send switch tab event
        self.event_bridge.publish(SystemEvent::TabSwitch {
            from: execution.source_tab.clone(),
            to: TabId::from_str(tab_id)?,
        }).await?;
        
        Ok(CommandResult {
            success: true,
            output: serde_json::json!({
                "switched_to": tab_id,
                "previous_tab": execution.source_tab,
            }),
            messages: vec![format!("Switched from {:?} to {}", execution.source_tab, tab_id)],
            side_effects: vec![
                SideEffect {
                    effect_type: "tab_switch".to_string(),
                    tab: TabId::from_str(tab_id)?,
                    description: "Tab focus changed".to_string(),
                    data: None,
                }
            ],
            follow_up_commands: vec![],
        })
    }
    
    /// Execute tool command
    async fn execute_tool(&self, execution: &CommandExecution) -> Result<CommandResult> {
        let tool_id = execution.parameters.get("tool_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tool_id parameter"))?;
        
        let params = execution.parameters.get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        
        // Send tool execution request
        self.event_bridge.publish(SystemEvent::ToolExecution {
            tab: execution.target_tab.clone(),
            tool_id: tool_id.to_string(),
            parameters: params.clone(),
        }).await?;
        
        Ok(CommandResult {
            success: true,
            output: serde_json::json!({
                "tool": tool_id,
                "parameters": params,
                "executed_in": execution.target_tab,
            }),
            messages: vec![format!("Executed tool {} in {:?}", tool_id, execution.target_tab)],
            side_effects: vec![
                SideEffect {
                    effect_type: "tool_execution".to_string(),
                    tab: execution.target_tab.clone(),
                    description: format!("Tool {} executed", tool_id),
                    data: Some(params),
                }
            ],
            follow_up_commands: vec![],
        })
    }
    
    /// Execute state synchronization
    async fn execute_sync_state(&self, execution: &CommandExecution) -> Result<CommandResult> {
        let state_key = execution.parameters.get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing state key"))?;
        
        let state_value = execution.parameters.get("value")
            .cloned()
            .unwrap_or_else(|| serde_json::json!(null));
        
        // Broadcast state update
        self.event_bridge.publish(SystemEvent::StateSync {
            source: execution.source_tab.clone(),
            key: state_key.to_string(),
            value: state_value.clone(),
        }).await?;
        
        Ok(CommandResult {
            success: true,
            output: serde_json::json!({
                "synced_key": state_key,
                "value": state_value,
                "broadcasted_from": execution.source_tab,
            }),
            messages: vec![format!("Synchronized state: {}", state_key)],
            side_effects: vec![
                SideEffect {
                    effect_type: "state_sync".to_string(),
                    tab: execution.target_tab.clone(),
                    description: format!("State {} synchronized", state_key),
                    data: Some(state_value),
                }
            ],
            follow_up_commands: vec![],
        })
    }
    
    /// Execute broadcast message
    async fn execute_broadcast_message(&self, execution: &CommandExecution) -> Result<CommandResult> {
        let message = execution.parameters.get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing message parameter"))?;
        
        let priority = execution.parameters.get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("normal");
        
        // Broadcast message to all tabs
        self.event_bridge.publish(SystemEvent::Broadcast {
            source: execution.source_tab.clone(),
            message: message.to_string(),
            priority: priority.to_string(),
        }).await?;
        
        Ok(CommandResult {
            success: true,
            output: serde_json::json!({
                "message": message,
                "priority": priority,
                "broadcasted_from": execution.source_tab,
            }),
            messages: vec![format!("Broadcasted: {}", message)],
            side_effects: vec![
                SideEffect {
                    effect_type: "broadcast".to_string(),
                    tab: TabId::System,
                    description: "Message broadcasted to all tabs".to_string(),
                    data: Some(serde_json::json!({ "message": message })),
                }
            ],
            follow_up_commands: vec![],
        })
    }
    
    /// Execute data query
    async fn execute_query_data(&self, execution: &CommandExecution) -> Result<CommandResult> {
        let query_type = execution.parameters.get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing query type"))?;
        
        let filter = execution.parameters.get("filter")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        
        // Execute query based on type
        let result_data = match query_type {
            "active_tabs" => {
                serde_json::json!({
                    "tabs": ["Chat", "Home", "Utilities", "System"],
                    "active": execution.source_tab,
                })
            }
            "system_status" => {
                serde_json::json!({
                    "status": "operational",
                    "uptime": 3600,
                    "memory_usage": 45.2,
                })
            }
            _ => serde_json::json!({ "error": "Unknown query type" })
        };
        
        Ok(CommandResult {
            success: true,
            output: result_data.clone(),
            messages: vec![format!("Query {} completed", query_type)],
            side_effects: vec![],
            follow_up_commands: vec![],
        })
    }
    
    /// Execute generic command through event bridge
    async fn execute_generic_command(&self, execution: &CommandExecution) -> Result<CommandResult> {
        // Send generic command event
        self.event_bridge.publish(SystemEvent::Command {
            source: execution.source_tab.clone(),
            target: execution.target_tab.clone(),
            command: execution.command.clone(),
            parameters: execution.parameters.clone(),
        }).await?;
        
        Ok(CommandResult {
            success: true,
            output: serde_json::json!({
                "command": execution.command,
                "routed_to": execution.target_tab,
            }),
            messages: vec![format!("Routed command {} to {:?}", execution.command, execution.target_tab)],
            side_effects: vec![],
            follow_up_commands: vec![],
        })
    }
    
    /// Add execution to history
    async fn add_to_history(&self, execution: &CommandExecution) {
        let duration_ms = execution.completed_at
            .map(|end| (end - execution.started_at).num_milliseconds() as u64)
            .unwrap_or(0);
        
        let entry = CommandHistoryEntry {
            execution_id: execution.id.clone(),
            command: execution.command.clone(),
            timestamp: execution.started_at,
            source_tab: execution.source_tab.clone(),
            target_tab: execution.target_tab.clone(),
            success: execution.result.as_ref().map(|r| r.success).unwrap_or(false),
            duration_ms,
        };
        
        let mut history = self.command_history.write().await;
        history.push(entry);
        
        // Limit history size
        if history.len() > self.config.history_limit {
            history.drain(0..history.len() - self.config.history_limit);
        }
    }
    
    /// Get command history
    pub async fn get_history(&self) -> Vec<CommandHistoryEntry> {
        self.command_history.read().await.clone()
    }
    
    /// Get active executions
    pub async fn get_active_executions(&self) -> HashMap<String, CommandExecution> {
        self.active_executions.read().await.clone()
    }
    
    /// Cancel a command execution
    pub async fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        let mut executions = self.active_executions.write().await;
        
        if let Some(mut execution) = executions.get_mut(execution_id) {
            execution.status = ExecutionStatus::Cancelled;
            
            let _ = self.event_tx.send(CommandEvent::CommandCancelled {
                execution_id: execution_id.to_string(),
                reason: "User requested cancellation".to_string(),
            });
            
            Ok(())
        } else {
            Err(anyhow::anyhow!("Execution not found"))
        }
    }
    
    /// Subscribe to command events
    pub fn subscribe(&self) -> broadcast::Receiver<CommandEvent> {
        self.event_tx.subscribe()
    }
}

impl CommandRegistry {
    /// Create a new command registry
    fn new() -> Self {
        let mut registry = Self {
            commands: HashMap::new(),
            aliases: HashMap::new(),
            tab_commands: HashMap::new(),
        };
        
        // Register default commands
        registry.register_default_commands();
        
        registry
    }
    
    /// Register default cross-tab commands
    fn register_default_commands(&mut self) {
        // Navigation commands
        self.commands.insert("switch_tab".to_string(), RegisteredCommand {
            name: "switch_tab".to_string(),
            description: "Switch to another tab".to_string(),
            parameters: vec![
                CommandParameter {
                    name: "tab".to_string(),
                    param_type: ParameterType::TabId,
                    required: true,
                    default_value: None,
                    description: "Target tab ID".to_string(),
                },
            ],
            source_tabs: vec![TabId::Chat, TabId::Home, TabId::Utilities],
            target_tabs: vec![TabId::System],
            requires_context: false,
            supports_async: false,
            category: CommandCategory::Navigation,
        });
        
        // Execution commands
        self.commands.insert("execute_tool".to_string(), RegisteredCommand {
            name: "execute_tool".to_string(),
            description: "Execute a tool in another tab".to_string(),
            parameters: vec![
                CommandParameter {
                    name: "tool_id".to_string(),
                    param_type: ParameterType::ToolId,
                    required: true,
                    default_value: None,
                    description: "Tool identifier".to_string(),
                },
                CommandParameter {
                    name: "params".to_string(),
                    param_type: ParameterType::Object,
                    required: false,
                    default_value: Some(serde_json::json!({})),
                    description: "Tool parameters".to_string(),
                },
            ],
            source_tabs: vec![TabId::Chat],
            target_tabs: vec![TabId::Utilities],
            requires_context: true,
            supports_async: true,
            category: CommandCategory::Execution,
        });
    }
}

impl CommandSynchronizer {
    /// Create a new command synchronizer
    fn new() -> Self {
        Self {
            locks: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            conflict_strategy: ConflictResolutionStrategy::Queue,
        }
    }
    
    /// Get lock for a command
    async fn get_lock(&self, command: &str) -> Option<Arc<tokio::sync::Mutex<()>>> {
        let locks = self.locks.read().await;
        locks.get(command).cloned()
    }
}