//! Tool system integration
//! 
//! Connects chat to IntelligentToolManager and TaskManager

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::tools::intelligent_manager::IntelligentToolManager;
use crate::tools::task_management::{TaskManager, Task, TaskStatus, TaskPlatform, TaskPriority, TaskCognitiveMetadata};
use crate::tui::bridges::ToolBridge;
// Tool types will be used when available
// use crate::tools::registry::ToolRegistry;
// use crate::tools::types::{ToolCapability, ToolInput, ToolOutput};

/// Tool system integration for chat
pub struct ToolIntegration {
    tool_manager: Arc<IntelligentToolManager>,
    task_manager: Arc<TaskManager>,
    
    /// Bridge for cross-tab tool coordination
    tool_bridge: Option<Arc<ToolBridge>>,
    
    /// Tool execution history
    execution_history: RwLock<Vec<ToolExecutionRecord>>,
    
    /// Active tool contexts
    active_contexts: RwLock<HashMap<String, ToolContext>>,
}

/// Record of a tool execution
#[derive(Debug, Clone)]
pub struct ToolExecutionRecord {
    /// Tool name
    pub tool_name: String,
    
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Input provided
    pub input: String,
    
    /// Output received
    pub output: String,
    
    /// Execution success
    pub success: bool,
    
    /// Execution duration
    pub duration_ms: u64,
}

/// Context for tool execution
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Context ID
    pub id: String,
    
    /// Tool name
    pub tool_name: String,
    
    /// Associated task ID (if any)
    pub task_id: Option<String>,
    
    /// Tool parameters
    pub parameters: HashMap<String, Value>,
    
    /// Execution state
    pub state: ToolExecutionState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolExecutionState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ToolIntegration {
    /// Create new tool integration
    pub fn new(
        tool_manager: Arc<IntelligentToolManager>,
        task_manager: Arc<TaskManager>,
    ) -> Self {
        Self {
            tool_manager,
            task_manager,
            tool_bridge: None,
            execution_history: RwLock::new(Vec::new()),
            active_contexts: RwLock::new(HashMap::new()),
        }
    }
    
    /// Set the tool bridge for cross-tab integration
    pub fn set_tool_bridge(&mut self, bridge: Arc<ToolBridge>) {
        self.tool_bridge = Some(bridge);
        tracing::info!("Tool bridge connected to chat tool integration");
    }
    
    /// Create a placeholder instance for initialization
    pub fn placeholder() -> Self {
        // Create dummy managers for placeholder
        // These will need to be replaced with actual instances when available
        use crate::tools::intelligent_manager::IntelligentToolManager;
        use crate::tools::task_management::TaskManager;
        
        // Create minimal config for placeholder
        let tool_config = crate::tools::intelligent_manager::ToolManagerConfig::default();
        let task_config = crate::tools::task_management::TaskConfig::default();
        
        // Note: These placeholders may panic if used without proper initialization
        // They should be replaced with actual instances as soon as possible
        let tool_manager = Arc::new(IntelligentToolManager::placeholder());
        let task_manager = Arc::new(TaskManager::placeholder());
        
        Self {
            tool_manager,
            task_manager,
            tool_bridge: None,
            execution_history: RwLock::new(Vec::new()),
            active_contexts: RwLock::new(HashMap::new()),
        }
    }
    
    /// Execute a tool command
    pub async fn execute_tool(&self, command: &str) -> Result<String> {
        let start_time = std::time::Instant::now();
        let timestamp = chrono::Utc::now();
        
        // Parse tool command
        let (tool_name, args) = self.parse_tool_command(command)?;
        
        // Create execution context
        let context_id = uuid::Uuid::new_v4().to_string();
        let context = ToolContext {
            id: context_id.clone(),
            tool_name: tool_name.clone(),
            task_id: None,
            parameters: self.parse_tool_args(&args)?,
            state: ToolExecutionState::Pending,
        };
        
        // Store context
        {
            let mut contexts = self.active_contexts.write().await;
            contexts.insert(context_id.clone(), context.clone());
        }
        
        // Update state to running
        self.update_context_state(&context_id, ToolExecutionState::Running).await?;
        
        // Execute tool through intelligent manager
        let args_json = serde_json::json!({ "args": args });
        let result = match self.tool_manager.execute_tool(&tool_name, args_json).await {
            Ok(output) => {
                self.update_context_state(&context_id, ToolExecutionState::Completed).await?;
                
                let duration_ms = start_time.elapsed().as_millis() as u64;
                
                // Record execution
                let record = ToolExecutionRecord {
                    tool_name: tool_name.clone(),
                    timestamp,
                    input: command.to_string(),
                    output: output.summary.clone(),
                    success: true,
                    duration_ms,
                };
                
                self.add_to_history(record).await;
                
                Ok(format!(
                    "✅ Tool '{}' executed successfully in {}ms\n\nOutput:\n{}",
                    tool_name, duration_ms, output.summary
                ))
            }
            Err(e) => {
                self.update_context_state(&context_id, ToolExecutionState::Failed).await?;
                
                let duration_ms = start_time.elapsed().as_millis() as u64;
                
                // Record failed execution
                let record = ToolExecutionRecord {
                    tool_name: tool_name.clone(),
                    timestamp,
                    input: command.to_string(),
                    output: e.to_string(),
                    success: false,
                    duration_ms,
                };
                
                self.add_to_history(record).await;
                
                Err(e).context(format!("Failed to execute tool '{}'", tool_name))
            }
        };
        
        // Clean up context
        {
            let mut contexts = self.active_contexts.write().await;
            contexts.remove(&context_id);
        }
        
        result
    }
    
    /// Execute a tool with task context
    pub async fn execute_tool_for_task(
        &self,
        tool_name: &str,
        args: &str,
        task_id: &str,
    ) -> Result<String> {
        // Create task in task manager
        let task = Task {
            id: task_id.to_string(),
            external_id: None,
            platform: crate::tools::task_management::TaskPlatform::Internal,
            title: format!("Execute {} tool", tool_name),
            description: Some(format!("Execute {} with args: {}", tool_name, args)),
            status: TaskStatus::InProgress,
            priority: crate::tools::task_management::TaskPriority::High,
            assignee: Some("loki".to_string()),
            reporter: Some("chat-user".to_string()),
            labels: vec![],
            due_date: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            estimate: None,
            time_spent: None,
            progress: 0.0,
            parent_task: None,
            subtasks: vec![],
            dependencies: vec![],
            cognitive_metadata: crate::tools::task_management::TaskCognitiveMetadata {
                cognitive_priority: 0.8,
                complexity_score: 0.5,
                energy_requirement: 0.5,
                focus_requirement: 0.5,
                context_switching_cost: 0.3,
                optimal_time_blocks: vec!["any".to_string()],
                prerequisite_knowledge: vec![],
                related_memories: vec![],
                burnout_risk: 0.2,
                motivation_factors: vec![],
            },
            name: format!("Execute {} tool", tool_name),
            metadata: serde_json::json!({
                "tool_name": tool_name,
                "args": args,
            }),
        };
        
        self.task_manager.create_task(
            &task.title,
            task.description.as_deref(),
            task.priority,
            task.due_date,
            task.platform,
        ).await?;
        
        // Execute tool
        let result = self.execute_tool(&format!("{} {}", tool_name, args)).await;
        
        // Update task status
        match &result {
            Ok(_) => {
                self.task_manager.update_task_status(task_id, TaskStatus::Completed).await?;
            }
            Err(_) => {
                self.task_manager.update_task_status(task_id, TaskStatus::Failed).await?;
            }
        }
        
        result
    }
    
    /// List available tools
    pub async fn list_available_tools(&self) -> Result<Vec<String>> {
        let tools = self.tool_manager.list_tools();
        Ok(tools)
    }
    
    /// Get tool capabilities
    pub async fn get_tool_capabilities(&self, tool_name: &str) -> Result<Vec<String>> {
        // Try to get tool info from the manager
        let tool = self.tool_manager.get_tool(tool_name)
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", tool_name))?;
        
        // Define capabilities based on tool category and name
        let capabilities = match tool.category.as_str() {
            "Development" => vec!["compile", "test", "debug", "format", "lint"],
            "File Management" => vec!["read", "write", "create", "delete", "list", "search"],
            "Deployment" => vec!["deploy", "rollback", "status", "logs", "scale"],
            "Security" => vec!["scan", "validate", "encrypt", "decrypt", "audit"],
            "Communication" => vec!["send", "receive", "broadcast", "subscribe", "notify"],
            "Database" => vec!["query", "insert", "update", "delete", "migrate", "backup"],
            "Cloud" => vec!["provision", "configure", "monitor", "scale", "terminate"],
            "Monitoring" => vec!["metrics", "logs", "alerts", "dashboards", "traces"],
            "Network" => vec!["connect", "disconnect", "ping", "trace", "scan"],
            _ => vec!["execute", "status", "configure"],
        };
        
        // Filter capabilities based on the specific tool
        let filtered_capabilities = match tool.name.as_str() {
            "GitHub Integration" => vec!["clone", "push", "pull", "commit", "pr", "issues"],
            "Web Search" => vec!["search", "fetch", "index", "cache"],
            "Shell Command" => vec!["execute", "pipe", "redirect", "background"],
            "Python Interpreter" => vec!["execute", "import", "debug", "profile"],
            "Docker" => vec!["build", "run", "push", "pull", "compose"],
            "Kubernetes" => vec!["deploy", "scale", "rollout", "expose", "config"],
            _ => capabilities,
        };
        
        Ok(filtered_capabilities.into_iter().map(|s| s.to_string()).collect())
    }
    
    /// Get execution history
    pub async fn get_execution_history(&self, limit: usize) -> Vec<ToolExecutionRecord> {
        let history = self.execution_history.read().await;
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Parse tool command
    fn parse_tool_command(&self, command: &str) -> Result<(String, String)> {
        let parts: Vec<&str> = command.splitn(2, ' ').collect();
        
        if parts.is_empty() {
            return Err(anyhow::anyhow!("Empty tool command"));
        }
        
        let tool_name = parts[0].to_string();
        let args = parts.get(1).map(|s| s.to_string()).unwrap_or_default();
        
        Ok((tool_name, args))
    }
    
    /// Parse tool arguments
    fn parse_tool_args(&self, args: &str) -> Result<HashMap<String, Value>> {
        let mut params = HashMap::new();
        
        // Simple key=value parsing
        for part in args.split_whitespace() {
            if let Some((key, value)) = part.split_once('=') {
                params.insert(
                    key.to_string(),
                    Value::String(value.to_string()),
                );
            } else {
                // Positional argument
                params.insert(
                    format!("arg{}", params.len()),
                    Value::String(part.to_string()),
                );
            }
        }
        
        Ok(params)
    }
    
    /// Update context state
    async fn update_context_state(
        &self,
        context_id: &str,
        state: ToolExecutionState,
    ) -> Result<()> {
        let mut contexts = self.active_contexts.write().await;
        if let Some(context) = contexts.get_mut(context_id) {
            context.state = state;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }
    
    /// Add execution record to history
    async fn add_to_history(&self, record: ToolExecutionRecord) {
        let mut history = self.execution_history.write().await;
        history.push(record);
        
        // Keep only last 1000 records
        if history.len() > 1000 {
            let excess = history.len() - 1000;
            history.drain(0..excess);
        }
    }
    
    /// Get active tool contexts
    pub async fn get_active_contexts(&self) -> HashMap<String, ToolContext> {
        self.active_contexts.read().await.clone()
    }
    
    /// Cancel tool execution
    pub async fn cancel_tool_execution(&self, context_id: &str) -> Result<()> {
        self.update_context_state(context_id, ToolExecutionState::Cancelled).await?;
        
        // In a real implementation, would also signal the tool to stop
        tracing::info!("Cancelled tool execution for context: {}", context_id);
        
        Ok(())
    }
}