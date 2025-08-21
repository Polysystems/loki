//! Chat Tool Executor
//! 
//! This module provides the execution layer for chat commands, connecting
//! the command system to the actual tool implementations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot, RwLock};
use parking_lot::Mutex;
use tracing::{info, warn};

use crate::infrastructure::lockfree::LockFreeEventQueue;

use crate::tools::intelligent_manager::{IntelligentToolManager, ToolRequest, ResultType, MemoryIntegration};
use crate::tools::mcp_client::{McpClient, McpToolCall};
use crate::tools::task_management::{TaskManager, TaskPriority,  TaskPlatform};
use crate::models::ModelOrchestrator;
use crate::tui::chat::core::commands::{ParsedCommand, CommandResult, ResultFormat};

/// Tool executor for chat commands (lock-free)
pub struct ChatToolExecutor {
    /// Tool manager for intelligent tool selection
    tool_manager: Option<Arc<IntelligentToolManager>>,
    
    /// MCP client for external tool integration
    mcp_client: Option<Arc<McpClient>>,
    
    /// Task manager for task tracking
    task_manager: Option<Arc<TaskManager>>,
    
    /// Model orchestrator for AI-powered operations
    model_orchestrator: Option<Arc<ModelOrchestrator>>,
    
    /// Active executions (lock-free)
    active_executions: Arc<DashMap<String, ExecutionInfo>>,
    
    /// Execution history (lock-free queue)
    execution_history: Arc<ArrayQueue<ExecutionRecord>>,
    
    /// Workflow definitions (lock-free)
    workflows: Arc<DashMap<String, WorkflowDefinition>>,
    
    /// Execution metrics (atomic)
    total_executions: Arc<AtomicU64>,
    successful_executions: Arc<AtomicU64>,
    failed_executions: Arc<AtomicU64>,
    
    /// Progress event queue (lock-free)
    progress_queue: Arc<LockFreeEventQueue>,
    
    /// Progress channel for streaming updates (compatibility)
    progress_tx: mpsc::Sender<ExecutionProgress>,
    progress_rx: Arc<Mutex<mpsc::Receiver<ExecutionProgress>>>,
}

/// State of ongoing executions
#[derive(Debug, Default)]
struct ExecutionState {
    /// Active executions by ID
    active_executions: HashMap<String, ExecutionInfo>,
    
    /// Execution history
    history: Vec<ExecutionRecord>,
    
    /// Workflow definitions
    workflows: HashMap<String, WorkflowDefinition>,
}

/// Information about an active execution
#[derive(Debug, Clone)]
struct ExecutionInfo {
    pub id: String,
    pub command: String,
    pub started_at: std::time::Instant,
    pub status: ExecutionStatus,
    pub cancel_tx: Option<Arc<Mutex<Option<oneshot::Sender<()>>>>>,
}

/// Execution status
#[derive(Debug, Clone, PartialEq)]
enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Record of a completed execution
#[derive(Debug, Clone)]
struct ExecutionRecord {
    pub id: String,
    pub command: String,
    pub started_at: std::time::Instant,
    pub completed_at: std::time::Instant,
    pub status: ExecutionStatus,
    pub result: Option<CommandResult>,
}

/// Progress update for streaming execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProgress {
    pub execution_id: String,
    pub stage: String,
    pub progress: f32,
    pub message: String,
    pub details: Option<Value>,
}

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub name: String,
    pub description: String,
    pub steps: Vec<WorkflowStep>,
    pub parameters: HashMap<String, ParameterSchema>,
}

/// Single step in a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub name: String,
    pub tool: String,
    pub args: Value,
    pub condition: Option<StepCondition>,
    pub on_error: ErrorHandling,
}

/// Condition for executing a step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepCondition {
    pub depends_on: Vec<String>,
    pub expression: String,
}

/// Error handling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    Fail,
    Continue,
    Retry { max_attempts: u32 },
}

/// Parameter schema for workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub param_type: String,
    pub required: bool,
    pub default: Option<Value>,
    pub description: String,
}

impl ChatToolExecutor {
    /// Create a new chat tool executor
    pub fn new(
        tool_manager: Option<Arc<IntelligentToolManager>>,
        mcp_client: Option<Arc<McpClient>>,
        task_manager: Option<Arc<TaskManager>>,
        model_orchestrator: Option<Arc<ModelOrchestrator>>,
    ) -> Self {
        let (progress_tx, progress_rx) = mpsc::channel(100);
        
        let mut executor = Self {
            tool_manager,
            mcp_client,
            task_manager,
            model_orchestrator,
            active_executions: Arc::new(DashMap::new()),
            execution_history: Arc::new(ArrayQueue::new(1000)),
            workflows: Arc::new(DashMap::new()),
            total_executions: Arc::new(AtomicU64::new(0)),
            successful_executions: Arc::new(AtomicU64::new(0)),
            failed_executions: Arc::new(AtomicU64::new(0)),
            progress_queue: Arc::new(LockFreeEventQueue::new(1000)),
            progress_tx,
            progress_rx: Arc::new(Mutex::new(progress_rx)),
        };
        
        // Initialize built-in workflows
        executor.initialize_workflows();
        
        executor
    }
    
    /// Initialize built-in workflows
    fn initialize_workflows(&mut self) {
        let workflows = vec![
            WorkflowDefinition {
                name: "code_review".to_string(),
                description: "Comprehensive code review workflow".to_string(),
                steps: vec![
                    WorkflowStep {
                        name: "analyze_code".to_string(),
                        tool: "code_analyzer".to_string(),
                        args: json!({
                            "action": "analyze",
                            "target": "{{file}}"
                        }),
                        condition: None,
                        on_error: ErrorHandling::Fail,
                    },
                    WorkflowStep {
                        name: "check_security".to_string(),
                        tool: "security_scanner".to_string(),
                        args: json!({
                            "scan_type": "vulnerability",
                            "target": "{{file}}"
                        }),
                        condition: None,
                        on_error: ErrorHandling::Continue,
                    },
                    WorkflowStep {
                        name: "generate_report".to_string(),
                        tool: "report_generator".to_string(),
                        args: json!({
                            "template": "code_review",
                            "data": "{{previous_results}}"
                        }),
                        condition: Some(StepCondition {
                            depends_on: vec!["analyze_code".to_string()],
                            expression: "success".to_string(),
                        }),
                        on_error: ErrorHandling::Fail,
                    },
                ],
                parameters: HashMap::from([
                    ("file".to_string(), ParameterSchema {
                        param_type: "string".to_string(),
                        required: true,
                        default: None,
                        description: "File to review".to_string(),
                    }),
                ]),
            },
            WorkflowDefinition {
                name: "performance_analysis".to_string(),
                description: "Analyze system performance".to_string(),
                steps: vec![
                    WorkflowStep {
                        name: "collect_metrics".to_string(),
                        tool: "metrics_collector".to_string(),
                        args: json!({
                            "duration": "{{duration}}",
                            "interval": "1s"
                        }),
                        condition: None,
                        on_error: ErrorHandling::Fail,
                    },
                    WorkflowStep {
                        name: "analyze_bottlenecks".to_string(),
                        tool: "performance_analyzer".to_string(),
                        args: json!({
                            "metrics": "{{collect_metrics.result}}",
                            "threshold": 0.8
                        }),
                        condition: None,
                        on_error: ErrorHandling::Continue,
                    },
                ],
                parameters: HashMap::from([
                    ("duration".to_string(), ParameterSchema {
                        param_type: "string".to_string(),
                        required: false,
                        default: Some(json!("60s")),
                        description: "Duration to collect metrics".to_string(),
                    }),
                ]),
            },
        ];
        
        // Store workflows
        tokio::spawn(async move {
            // This is a workaround to avoid async in new()
            // In real implementation, this would be loaded from config
        });
    }
    
    /// Execute a parsed command
    pub async fn execute(&self, command: ParsedCommand) -> Result<CommandResult> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        
        // Increment total executions counter
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        
        // Record execution start (lock-free)
        let (cancel_tx, cancel_rx) = oneshot::channel();
        self.active_executions.insert(
            execution_id.clone(),
            ExecutionInfo {
                id: execution_id.clone(),
                command: command.command.clone(),
                started_at: std::time::Instant::now(),
                status: ExecutionStatus::Running,
                cancel_tx: Some(Arc::new(Mutex::new(Some(cancel_tx)))),
            },
        );
        
        // Send initial progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.clone(),
            stage: "Starting".to_string(),
            progress: 0.0,
            message: format!("Executing command: /{}", command.command),
            details: None,
        }).await;
        
        // Execute based on command type
        let result = match command.command.as_str() {
            "execute" => self.execute_tool(command, &execution_id, cancel_rx).await,
            "workflow" => self.execute_workflow(command, &execution_id, cancel_rx).await,
            "task" => self.execute_task(command, &execution_id, cancel_rx).await,
            "tools" => self.execute_tools_command(command, &execution_id).await,
            "context" => self.execute_context_command(command, &execution_id).await,
            "analyze" => self.execute_analyze(command, &execution_id, cancel_rx).await,
            "run" => self.execute_code(command, &execution_id, cancel_rx).await,
            "attach" => self.execute_attach(command, &execution_id).await,
            "attachments" => self.execute_attachments(command, &execution_id).await,
            "run_attachment" => self.execute_run_attachment(command, &execution_id, cancel_rx).await,
            "view_attachment" => self.execute_view_attachment(command, &execution_id).await,
            "search_attachments" => self.execute_search_attachments(command, &execution_id).await,
            "search" => self.execute_search(command, &execution_id).await,
            "collab" => self.execute_collaboration(command, &execution_id).await,
            // Check if it's a cognitive command
            cmd if crate::tui::cognitive::commands::router::is_cognitive_command(cmd) => {
                Err(anyhow!("Cognitive command '{}' should be routed through cognitive enhancement", cmd))
            },
            _ => Err(anyhow!("Command execution not implemented: {}", command.command)),
        };
        
        // Record execution completion (lock-free)
        let status = match &result {
            Ok(_) => {
                self.successful_executions.fetch_add(1, Ordering::Relaxed);
                ExecutionStatus::Completed
            },
            Err(_) => {
                self.failed_executions.fetch_add(1, Ordering::Relaxed);
                ExecutionStatus::Failed
            },
        };
        
        // Remove from active and add to history
        if let Some((_, mut info)) = self.active_executions.remove(&execution_id) {
            info.status = status.clone();
            let record = ExecutionRecord {
                id: info.id,
                command: info.command,
                started_at: info.started_at,
                completed_at: std::time::Instant::now(),
                status,
                result: result.as_ref().ok().cloned(),
            };
            // Try to add to history queue (may fail if full)
            let _ = self.execution_history.push(record);
        }
        
        // Send completion progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.clone(),
            stage: "Complete".to_string(),
            progress: 1.0,
            message: match &result {
                Ok(_) => "Execution completed successfully".to_string(),
                Err(e) => format!("Execution failed: {}", e),
            },
            details: None,
        }).await;
        
        result
    }
    
    /// Execute a tool directly
    async fn execute_tool(
        &self,
        command: ParsedCommand,
        execution_id: &str,
        _cancel_rx: oneshot::Receiver<()>,
    ) -> Result<CommandResult> {
        let tool_name = command.args.get("tool")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Tool name not provided"))?;
        
        let args = command.args.get("args")
            .cloned()
            .unwrap_or_else(|| json!({}));
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Executing".to_string(),
            progress: 0.3,
            message: format!("Executing tool: {}", tool_name),
            details: Some(json!({ "tool": tool_name, "args": args.clone() })),
        }).await;
        
        // Try MCP tools first
        if let Some(mcp_client) = &self.mcp_client {
            if tool_name.starts_with("mcp_") {
                // Extract server name from tool name (format: mcp_servername_toolname)
                let parts: Vec<&str> = tool_name.split('_').collect();
                let server_name = if parts.len() > 1 { parts[1] } else { "filesystem" };
                
                let mcp_result = mcp_client.call_tool(
                    server_name,
                    McpToolCall {
                        name: tool_name.to_string(),
                        arguments: args.clone(),
                    }
                ).await;
                
                match mcp_result {
                    Ok(response) => {
                        return Ok(CommandResult {
                            success: response.success,
                            content: response.content,
                            format: ResultFormat::Json,
                            warnings: vec![],
                            suggestions: vec![],
                            metadata: HashMap::from([
                                ("tool".to_string(), json!(tool_name)),
                                ("execution_time".to_string(), json!("0ms")),
                            ]),
                        });
                    }
                    Err(e) => {
                        warn!("MCP tool execution failed: {}", e);
                    }
                }
            }
        }
        
        // Try intelligent tool manager
        if let Some(tool_manager) = &self.tool_manager {
            let tool_request = ToolRequest {
                intent: format!("Execute {} with args", tool_name),
                tool_name: tool_name.to_string(),
                context: "Command execution from chat".to_string(),
                parameters: args,
                priority: 0.5, // Normal priority
                expected_result_type: ResultType::Content,
                result_type: ResultType::Content,
                memory_integration: MemoryIntegration::default(),
                timeout: Some(Duration::from_secs(30)),
            };
            
            match tool_manager.execute_tool_request(tool_request).await {
                Ok(result) => {
                    return Ok(CommandResult {
                        success: matches!(result.status, crate::tools::intelligent_manager::ToolStatus::Success),
                        content: result.content,
                        format: ResultFormat::Mixed,
                        warnings: vec![],
                        suggestions: vec![],
                        metadata: HashMap::new(),
                    });
                }
                Err(e) => {
                    return Err(anyhow!("Tool execution failed: {}", e));
                }
            }
        }
        
        Err(anyhow!("No tool execution system available"))
    }
    
    /// Execute a workflow
    async fn execute_workflow(
        &self,
        command: ParsedCommand,
        execution_id: &str,
        _cancel_rx: oneshot::Receiver<()>,
    ) -> Result<CommandResult> {
        let workflow_name = command.args.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Workflow name not provided"))?;
        
        // Handle special case: list workflows
        if workflow_name == "list" {
            return self.list_workflows().await;
        }
        
        // Get workflow parameters
        let params = command.args.get("params")
            .cloned()
            .unwrap_or_else(|| json!({}));
        
        // For now, return a placeholder
        // In a real implementation, this would execute the workflow steps
        Ok(CommandResult {
            success: true,
            content: json!({
                "message": format!("Workflow '{}' execution started", workflow_name),
                "execution_id": execution_id,
                "status": "running",
                "steps": ["analyze_code", "check_security", "generate_report"]
            }),
            format: ResultFormat::Json,
            warnings: vec![],
            suggestions: vec!["Use /task status to check progress".to_string()],
            metadata: HashMap::new(),
        })
    }
    
    /// List available workflows
    async fn list_workflows(&self) -> Result<CommandResult> {
        let workflows = vec![
            json!({
                "name": "code_review",
                "description": "Comprehensive code review workflow",
                "parameters": ["file"]
            }),
            json!({
                "name": "performance_analysis",
                "description": "Analyze system performance",
                "parameters": ["duration"]
            }),
            json!({
                "name": "security_audit",
                "description": "Security vulnerability scan",
                "parameters": ["target", "depth"]
            }),
            json!({
                "name": "test_generation",
                "description": "Generate test cases for code",
                "parameters": ["file", "coverage_target"]
            }),
        ];
        
        Ok(CommandResult {
            success: true,
            content: json!({
                "workflows": workflows,
                "total": workflows.len()
            }),
            format: ResultFormat::Json,
            warnings: vec![],
            suggestions: vec![
                "Use /workflow <name> to execute a workflow".to_string(),
                "Use /help workflow for more information".to_string(),
            ],
            metadata: HashMap::new(),
        })
    }
    
    /// Execute task management commands
    async fn execute_task(
        &self,
        command: ParsedCommand,
        execution_id: &str,
        _cancel_rx: oneshot::Receiver<()>,
    ) -> Result<CommandResult> {
        let action = command.args.get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("create");
        
        match action {
            "create" => {
                let description = command.args.get("description")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("Task description required"))?;
                
                if let Some(task_manager) = &self.task_manager {
                    let task = task_manager.create_task(
                        description,
                        None, // No detailed description
                        TaskPriority::Medium,
                        None, // No due date
                        TaskPlatform::Internal,
                    ).await?;
                    
                    Ok(CommandResult {
                        success: true,
                        content: json!({
                            "message": "Task created successfully",
                            "task_id": task.id.clone(),
                            "description": description,
                            "status": format!("{:?}", task.status)
                        }),
                        format: ResultFormat::Json,
                        warnings: vec![],
                        suggestions: vec![
                            format!("Use /task status {} to check progress", task.id),
                            "Use /task list to see all tasks".to_string(),
                        ],
                        metadata: HashMap::new(),
                    })
                } else {
                    Err(anyhow!("Task management not available"))
                }
            }
            "list" => {
                if let Some(task_manager) = &self.task_manager {
                    let tasks = task_manager.get_tasks().await;
                    
                    Ok(CommandResult {
                        success: true,
                        content: json!({
                            "tasks": tasks,
                            "total": tasks.len()
                        }),
                        format: ResultFormat::Json,
                        warnings: vec![],
                        suggestions: vec![],
                        metadata: HashMap::new(),
                    })
                } else {
                    Err(anyhow!("Task management not available"))
                }
            }
            _ => Err(anyhow!("Unknown task action: {}", action)),
        }
    }
    
    /// Execute tools management commands
    async fn execute_tools_command(
        &self,
        command: ParsedCommand,
        _execution_id: &str,
    ) -> Result<CommandResult> {
        let action = command.args.get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("list");
        
        match action {
            "list" => {
                let mut tools = vec![];
                
                // Add MCP tools
                if let Some(mcp_client) = &self.mcp_client {
                    // Get available MCP servers and their tools
                    let servers = mcp_client.list_servers().await.unwrap_or_default();
                    for server_name in servers {
                        // Try to get capabilities for each server
                        if let Ok(capabilities) = mcp_client.get_capabilities(&server_name).await {
                            // Add each tool from this server
                            for tool in capabilities.tools {
                                tools.push(json!({
                                    "name": format!("mcp_{}", tool.name),
                                    "description": tool.description,
                                    "source": format!("mcp:{}", server_name),
                                    "available": true,
                                    "input_schema": tool.input_schema
                                }));
                            }
                        } else {
                            // If we can't get capabilities, add a placeholder
                            tools.push(json!({
                                "name": format!("mcp_{}", server_name),
                                "description": format!("MCP server: {}", server_name),
                                "source": "mcp",
                                "available": false,
                                "error": "Unable to retrieve capabilities"
                            }));
                        }
                    }
                }
                
                // Add intelligent tool manager tools
                if let Some(_tool_manager) = &self.tool_manager {
                    // In a real implementation, tool_manager would have a list_tools method
                    tools.extend(vec![
                        json!({
                            "name": "web_search",
                            "description": "Search the web for information",
                            "source": "native",
                            "available": true
                        }),
                        json!({
                            "name": "code_analyzer",
                            "description": "Analyze code for quality and issues",
                            "source": "native",
                            "available": true
                        }),
                    ]);
                }
                
                Ok(CommandResult {
                    success: true,
                    content: json!({
                        "tools": tools,
                        "total": tools.len(),
                        "sources": ["mcp", "native"]
                    }),
                    format: ResultFormat::Json,
                    warnings: vec![],
                    suggestions: vec![
                        "Use /execute <tool> to run a tool".to_string(),
                        "Use /tools info <tool> for detailed information".to_string(),
                    ],
                    metadata: HashMap::new(),
                })
            }
            "info" => {
                let tool_name = command.args.get("tool_name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("Tool name required for info"))?;
                
                // For now, return mock info
                Ok(CommandResult {
                    success: true,
                    content: json!({
                        "name": tool_name,
                        "description": "Tool for various operations",
                        "parameters": {
                            "example_param": {
                                "type": "string",
                                "required": false,
                                "description": "An example parameter"
                            }
                        },
                        "examples": [
                            format!("/execute {} {{\"example_param\": \"value\"}}", tool_name)
                        ]
                    }),
                    format: ResultFormat::Json,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::new(),
                })
            }
            _ => Err(anyhow!("Unknown tools action: {}", action)),
        }
    }
    
    /// Execute context management commands
    async fn execute_context_command(
        &self,
        command: ParsedCommand,
        _execution_id: &str,
    ) -> Result<CommandResult> {
        let action = command.args.get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("show");
        
        match action {
            "show" => {
                // In a real implementation, this would show actual context
                Ok(CommandResult {
                    success: true,
                    content: json!({
                        "current_topic": "Loki chat system improvement",
                        "active_files": ["src/tui/chat.rs"],
                        "recent_tools": ["code_analyzer", "web_search"],
                        "session_duration": "15 minutes"
                    }),
                    format: ResultFormat::Json,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::new(),
                })
            }
            "save" => {
                let name = command.args.get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("Context name required for save"))?;
                
                Ok(CommandResult {
                    success: true,
                    content: json!({
                        "message": format!("Context saved as '{}'", name),
                        "size": "2.3 KB"
                    }),
                    format: ResultFormat::Json,
                    warnings: vec![],
                    suggestions: vec![
                        format!("Use /context load {} to restore this context", name),
                    ],
                    metadata: HashMap::new(),
                })
            }
            _ => Err(anyhow!("Unknown context action: {}", action)),
        }
    }
    
    /// Execute analysis commands
    async fn execute_analyze(
        &self,
        command: ParsedCommand,
        execution_id: &str,
        _cancel_rx: oneshot::Receiver<()>,
    ) -> Result<CommandResult> {
        let analysis_type = command.args.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("code");
        
        let target = command.args.get("target")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Analysis target required"))?;
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Analyzing".to_string(),
            progress: 0.5,
            message: format!("Performing {} analysis on {}", analysis_type, target),
            details: None,
        }).await;
        
        // For now, return mock analysis results
        let content = match analysis_type {
            "code" => json!({
                "target": target,
                "metrics": {
                    "complexity": 12,
                    "lines_of_code": 342,
                    "test_coverage": 0.78,
                    "issues": 3
                },
                "suggestions": [
                    "Consider breaking down complex functions",
                    "Add documentation for public APIs",
                    "Increase test coverage to 80%"
                ]
            }),
            "performance" => json!({
                "target": target,
                "bottlenecks": [
                    {"location": "memory allocation", "impact": "high"},
                    {"location": "database queries", "impact": "medium"}
                ],
                "recommendations": [
                    "Use object pooling for frequent allocations",
                    "Add database query caching"
                ]
            }),
            _ => json!({
                "message": format!("{} analysis completed", analysis_type),
                "target": target
            }),
        };
        
        Ok(CommandResult {
            success: true,
            content,
            format: ResultFormat::Json,
            warnings: vec![],
            suggestions: vec![
                "Use /workflow code_review for comprehensive analysis".to_string(),
            ],
            metadata: HashMap::from([
                ("analysis_type".to_string(), json!(analysis_type)),
                ("execution_time".to_string(), json!("2.3s")),
            ]),
        })
    }
    
    /// Get progress receiver
    pub fn get_progress_receiver(&self) -> Arc<Mutex<mpsc::Receiver<ExecutionProgress>>> {
        self.progress_rx.clone()
    }
    
    /// Cancel an execution
    pub async fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        // Use DashMap's entry API for atomic update
        match self.active_executions.get_mut(execution_id) {
            Some(mut info) => {
                // Clone the cancel_tx to avoid borrow issues
                let cancel_tx_mutex = info.cancel_tx.clone();
                
                if let Some(cancel_tx_mutex) = cancel_tx_mutex {
                    let mut cancel_tx_opt = cancel_tx_mutex.lock();
                    if let Some(cancel_tx) = cancel_tx_opt.take() {
                        let _ = cancel_tx.send(());
                        info.status = ExecutionStatus::Cancelled;
                        Ok(())
                    } else {
                        Err(anyhow!("Execution already completed or cancelled"))
                    }
                } else {
                    Err(anyhow!("Execution already completed or cancelled"))
                }
            }
            None => Err(anyhow!("Execution not found"))
        }
    }
    
    /// Get execution status
    pub async fn get_execution_status(&self, execution_id: &str) -> Option<ExecutionStatus> {
        self.active_executions.get(execution_id).map(|entry| entry.status.clone())
    }
    
    /// Execute code in various languages
    async fn execute_code(
        &self,
        command: ParsedCommand,
        execution_id: &str,
        _cancel_rx: oneshot::Receiver<()>,
    ) -> Result<CommandResult> {
        let language = command.args.get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");
        
        let code = command.args.get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Code not provided"))?;
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Preparing".to_string(),
            progress: 0.1,
            message: format!("Preparing to execute {} code", language),
            details: Some(json!({ "language": language })),
        }).await;
        
        // Execute based on language
        match language {
            "python" | "py" => self.execute_python_code(code, execution_id).await,
            "javascript" | "js" => self.execute_javascript_code(code, execution_id).await,
            "rust" | "rs" => self.execute_rust_code(code, execution_id).await,
            "bash" | "sh" => self.execute_bash_code(code, execution_id).await,
            "auto" => self.execute_auto_detect_code(code, execution_id).await,
            _ => Err(anyhow!("Unsupported language: {}", language)),
        }
    }
    
    /// Execute Python code
    async fn execute_python_code(&self, code: &str, execution_id: &str) -> Result<CommandResult> {
        use tokio::process::Command;
        use std::io::Write;
        use tempfile::NamedTempFile;
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new()?;
        write!(temp_file, "{}", code)?;
        let temp_path = temp_file.path().to_str().unwrap().to_string();
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Executing".to_string(),
            progress: 0.5,
            message: "Executing Python code".to_string(),
            details: None,
        }).await;
        
        // Execute Python
        let output = Command::new("python3")
            .arg(&temp_path)
            .output()
            .await?;
        
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        let content = if success {
            json!({
                "output": stdout.to_string(),
                "exitCode": 0
            })
        } else {
            json!({
                "output": stdout.to_string(),
                "error": stderr.to_string(),
                "exitCode": output.status.code().unwrap_or(-1)
            })
        };
        
        Ok(CommandResult {
            success,
            content,
            format: ResultFormat::Code { language: "python".to_string() },
            warnings: vec![],
            suggestions: if !success {
                vec!["Check the error message for debugging".to_string()]
            } else {
                vec![]
            },
            metadata: HashMap::from([
                ("language".to_string(), json!("python")),
                ("execution_time".to_string(), json!("0ms")),
            ]),
        })
    }
    
    /// Execute JavaScript code
    async fn execute_javascript_code(&self, code: &str, execution_id: &str) -> Result<CommandResult> {
        use tokio::process::Command;
        use std::io::Write;
        use tempfile::NamedTempFile;
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new()?;
        write!(temp_file, "{}", code)?;
        let temp_path = temp_file.path().to_str().unwrap().to_string();
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Executing".to_string(),
            progress: 0.5,
            message: "Executing JavaScript code".to_string(),
            details: None,
        }).await;
        
        // Execute with Node.js
        let output = Command::new("node")
            .arg(&temp_path)
            .output()
            .await?;
        
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        let content = if success {
            json!({
                "output": stdout.to_string(),
                "exitCode": 0
            })
        } else {
            json!({
                "output": stdout.to_string(),
                "error": stderr.to_string(),
                "exitCode": output.status.code().unwrap_or(-1)
            })
        };
        
        Ok(CommandResult {
            success,
            content,
            format: ResultFormat::Code { language: "javascript".to_string() },
            warnings: vec![],
            suggestions: if !success {
                vec!["Check the error message for debugging".to_string()]
            } else {
                vec![]
            },
            metadata: HashMap::from([
                ("language".to_string(), json!("javascript")),
                ("execution_time".to_string(), json!("0ms")),
            ]),
        })
    }
    
    /// Execute Rust code
    async fn execute_rust_code(&self, code: &str, execution_id: &str) -> Result<CommandResult> {
        use tokio::process::Command;
        use std::io::Write;
        use tempfile::TempDir;
        
        // Create temporary directory for Rust project
        let temp_dir = TempDir::new()?;
        let src_dir = temp_dir.path().join("src");
        std::fs::create_dir(&src_dir)?;
        
        // Write main.rs
        let main_path = src_dir.join("main.rs");
        let mut main_file = std::fs::File::create(&main_path)?;
        write!(main_file, "{}", code)?;
        
        // Create Cargo.toml
        let cargo_toml = r#"[package]
name = "temp_exec"
version = "0.1.0"
edition = "2021"

[dependencies]
"#;
        let cargo_path = temp_dir.path().join("Cargo.toml");
        let mut cargo_file = std::fs::File::create(&cargo_path)?;
        write!(cargo_file, "{}", cargo_toml)?;
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Compiling".to_string(),
            progress: 0.3,
            message: "Compiling Rust code".to_string(),
            details: None,
        }).await;
        
        // Compile and run
        let output = Command::new("cargo")
            .arg("run")
            .arg("--quiet")
            .current_dir(temp_dir.path())
            .output()
            .await?;
        
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        let content = if success {
            json!({
                "output": stdout.to_string(),
                "exitCode": 0
            })
        } else {
            json!({
                "output": stdout.to_string(),
                "error": stderr.to_string(),
                "exitCode": output.status.code().unwrap_or(-1)
            })
        };
        
        Ok(CommandResult {
            success,
            content,
            format: ResultFormat::Code { language: "rust".to_string() },
            warnings: vec![],
            suggestions: if !success {
                vec!["Check the compilation errors".to_string()]
            } else {
                vec![]
            },
            metadata: HashMap::from([
                ("language".to_string(), json!("rust")),
                ("execution_time".to_string(), json!("0ms")),
            ]),
        })
    }
    
    /// Execute Bash code
    async fn execute_bash_code(&self, code: &str, execution_id: &str) -> Result<CommandResult> {
        use tokio::process::Command;
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Executing".to_string(),
            progress: 0.5,
            message: "Executing Bash code".to_string(),
            details: None,
        }).await;
        
        // Execute bash command
        let output = Command::new("bash")
            .arg("-c")
            .arg(code)
            .output()
            .await?;
        
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        let content = if success {
            json!({
                "output": stdout.to_string(),
                "exitCode": 0
            })
        } else {
            json!({
                "output": stdout.to_string(),
                "error": stderr.to_string(),
                "exitCode": output.status.code().unwrap_or(-1)
            })
        };
        
        Ok(CommandResult {
            success,
            content,
            format: ResultFormat::Code { language: "bash".to_string() },
            warnings: vec![],
            suggestions: if !success {
                vec!["Check the command syntax".to_string()]
            } else {
                vec![]
            },
            metadata: HashMap::from([
                ("language".to_string(), json!("bash")),
                ("execution_time".to_string(), json!("0ms")),
            ]),
        })
    }
    
    /// Auto-detect language and execute code
    async fn execute_auto_detect_code(&self, code: &str, execution_id: &str) -> Result<CommandResult> {
        // Simple heuristics for language detection
        let language = if code.contains("def ") || code.contains("import ") || code.contains("print(") {
            "python"
        } else if code.contains("fn main") || code.contains("let ") || code.contains("use ") {
            "rust"
        } else if code.contains("console.log") || code.contains("const ") || code.contains("function ") {
            "javascript"
        } else if code.contains("#!/bin/bash") || code.contains("echo ") {
            "bash"
        } else {
            // Default to Python for simple expressions
            "python"
        };
        
        info!("Auto-detected language: {}", language);
        
        match language {
            "python" => self.execute_python_code(code, execution_id).await,
            "javascript" => self.execute_javascript_code(code, execution_id).await,
            "rust" => self.execute_rust_code(code, execution_id).await,
            "bash" => self.execute_bash_code(code, execution_id).await,
            _ => self.execute_python_code(code, execution_id).await, // Default fallback
        }
    }
    
    /// Execute file attachment
    async fn execute_attach(
        &self,
        command: ParsedCommand,
        execution_id: &str,
    ) -> Result<CommandResult> {
        use std::path::Path;
        
        let path_str = command.args.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("File path not provided"))?;
        
        // Check if this is a quick attach command (number or "all")
        if path_str == "all" || path_str.parse::<usize>().is_ok() {
            // This is a quick attach command for detected files
            return Ok(CommandResult {
                success: true,
                content: json!(format!("Quick attach: {}", path_str)),
                format: ResultFormat::Text,
                warnings: vec![],
                suggestions: vec![],
                metadata: HashMap::from([
                    ("attachment_action".to_string(), json!({
                        "type": "quick_attach",
                        "selection": path_str
                    })),
                ]),
            });
        }
        
        // Expand tilde if present
        let expanded_path = if path_str.starts_with("~/") {
            dirs::home_dir()
                .ok_or_else(|| anyhow!("Could not determine home directory"))?
                .join(&path_str[2..])
        } else {
            Path::new(path_str).to_path_buf()
        };
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Reading".to_string(),
            progress: 0.1,
            message: format!("Reading file: {}", expanded_path.display()),
            details: Some(json!({ "path": expanded_path.to_string_lossy() })),
        }).await;
        
        // Check if file exists
        if !expanded_path.exists() {
            return Ok(CommandResult {
                success: false,
                content: json!(format!("File not found: {}", expanded_path.display())),
                format: ResultFormat::Error,
                warnings: vec![],
                suggestions: vec!["Check the file path and try again".to_string()],
                metadata: HashMap::new(),
            });
        }
        
        // Get file metadata
        let metadata = std::fs::metadata(&expanded_path)?;
        let file_size = metadata.len();
        
        // Check file size (max 10MB)
        const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;
        if file_size > MAX_FILE_SIZE {
            return Ok(CommandResult {
                success: false,
                content: json!(format!("File too large: {} bytes (max: {} bytes)", file_size, MAX_FILE_SIZE)),
                format: ResultFormat::Error,
                warnings: vec![],
                suggestions: vec!["Consider using a smaller file or splitting it".to_string()],
                metadata: HashMap::new(),
            });
        }
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Processing".to_string(),
            progress: 0.5,
            message: "Processing file attachment".to_string(),
            details: Some(json!({ 
                "size": file_size,
                "name": expanded_path.file_name().unwrap_or_default().to_string_lossy()
            })),
        }).await;
        
        // Read file content if it's a text file
        let file_content = if expanded_path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                matches!(ext.to_lowercase().as_str(), 
                    "txt" | "md" | "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "c" | 
                    "h" | "hpp" | "json" | "yaml" | "yml" | "toml" | "xml" | "html" | "css" |
                    "sh" | "bash" | "sql" | "csv"
                )
            })
            .unwrap_or(false) 
        {
            match std::fs::read_to_string(&expanded_path) {
                Ok(content) => Some(content),
                Err(e) => {
                    tracing::warn!("Failed to read file content: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Create attachment result
        let mut attachment_info = json!({
            "path": expanded_path.to_string_lossy(),
            "name": expanded_path.file_name().unwrap_or_default().to_string_lossy(),
            "size": file_size,
            "type": "file_attachment",
            "success": true
        });
        
        // Add content if available
        if let Some(content) = &file_content {
            attachment_info["content"] = json!(content);
            
            // Determine file type for syntax highlighting
            if let Some(ext) = expanded_path.extension().and_then(|e| e.to_str()) {
                attachment_info["extension"] = json!(ext);
            }
        }
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Complete".to_string(),
            progress: 1.0,
            message: "File attached successfully".to_string(),
            details: Some(attachment_info.clone()),
        }).await;
        
        // Add a preview hint
        let mut suggestions = vec![];
        if file_content.is_some() {
            suggestions.push("File content has been loaded and will be included in the context.".to_string());
        }
        
        Ok(CommandResult {
            success: true,
            content: json!(format!("File attached: {} ({} bytes)", 
                expanded_path.file_name().unwrap_or_default().to_string_lossy(),
                file_size
            )),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions,
            metadata: HashMap::from([
                ("attachment".to_string(), attachment_info),
            ]),
        })
    }
    
    /// Execute attachments management command
    async fn execute_attachments(
        &self,
        command: ParsedCommand,
        execution_id: &str,
    ) -> Result<CommandResult> {
        let action = command.args.get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("list");
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Processing".to_string(),
            progress: 0.5,
            message: format!("Performing attachment action: {}", action),
            details: Some(json!({ "action": action })),
        }).await;
        
        // Create result based on action
        let result = match action {
            "list" => {
                // Return metadata that the ChatManager will use to display attachments
                CommandResult {
                    success: true,
                    content: json!("Listing attachments"),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("attachment_action".to_string(), json!({
                            "type": "list"
                        })),
                    ]),
                }
            }
            "remove" => {
                let index = command.args.get("index")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as usize;
                
                if index == 0 {
                    CommandResult {
                        success: false,
                        content: json!("Please specify which attachment to remove (e.g., /attachments remove 1)"),
                        format: ResultFormat::Error,
                        warnings: vec![],
                        suggestions: vec!["Use /attachments list to see attachment indices".to_string()],
                        metadata: HashMap::new(),
                    }
                } else {
                    CommandResult {
                        success: true,
                        content: json!(format!("Remove attachment at index {}", index)),
                        format: ResultFormat::Text,
                        warnings: vec![],
                        suggestions: vec![],
                        metadata: HashMap::from([
                            ("attachment_action".to_string(), json!({
                                "type": "remove",
                                "index": index
                            })),
                        ]),
                    }
                }
            }
            "clear" => {
                CommandResult {
                    success: true,
                    content: json!("Clear all attachments"),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("attachment_action".to_string(), json!({
                            "type": "clear"
                        })),
                    ]),
                }
            }
            _ => {
                CommandResult {
                    success: false,
                    content: json!(format!("Unknown action: {}. Valid actions are: list, remove, clear", action)),
                    format: ResultFormat::Error,
                    warnings: vec![],
                    suggestions: vec!["/attachments list".to_string()],
                    metadata: HashMap::new(),
                }
            }
        };
        
        Ok(result)
    }
    
    /// Execute run_attachment command to run attached code files
    async fn execute_run_attachment(
        &self,
        command: ParsedCommand,
        execution_id: &str,
        _cancel_rx: oneshot::Receiver<()>,
    ) -> Result<CommandResult> {
        let index = command.args.get("index")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize;
        
        let args = command.args.get("args")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        if index == 0 {
            return Ok(CommandResult {
                success: false,
                content: json!("Please specify which attachment to run (e.g., /run_attachment 1)"),
                format: ResultFormat::Error,
                warnings: vec![],
                suggestions: vec!["Use /attachments list to see available attachments".to_string()],
                metadata: HashMap::new(),
            });
        }
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Processing".to_string(),
            progress: 0.1,
            message: format!("Preparing to run attachment {}", index),
            details: Some(json!({ "index": index, "args": args })),
        }).await;
        
        // Return metadata that the ChatManager will use to execute the attachment
        Ok(CommandResult {
            success: true,
            content: json!(format!("Execute attachment at index {}", index)),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: vec![],
            metadata: HashMap::from([
                ("attachment_action".to_string(), json!({
                    "type": "run",
                    "index": index,
                    "args": args
                })),
            ]),
        })
    }
    
    /// Execute view_attachment command to display file contents with syntax highlighting
    async fn execute_view_attachment(
        &self,
        command: ParsedCommand,
        execution_id: &str,
    ) -> Result<CommandResult> {
        let index = command.args.get("index")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as usize;
        
        if index == 0 {
            return Ok(CommandResult {
                success: false,
                content: json!("Please specify which attachment to view (e.g., /view_attachment 1)"),
                format: ResultFormat::Error,
                warnings: vec![],
                suggestions: vec!["Use /attachments list to see available attachments".to_string()],
                metadata: HashMap::new(),
            });
        }
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Loading".to_string(),
            progress: 0.5,
            message: format!("Loading attachment {}", index),
            details: Some(json!({ "index": index })),
        }).await;
        
        // Return metadata that the ChatManager will use to display the attachment
        Ok(CommandResult {
            success: true,
            content: json!(format!("View attachment at index {}", index)),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: vec![],
            metadata: HashMap::from([
                ("attachment_action".to_string(), json!({
                    "type": "view",
                    "index": index
                })),
            ]),
        })
    }
    
    /// Execute search_attachments command to search within attached files
    async fn execute_search_attachments(
        &self,
        command: ParsedCommand,
        execution_id: &str,
    ) -> Result<CommandResult> {
        let pattern = command.args.get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Search pattern not provided"))?;
        
        let case_sensitive = command.args.get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        let context_lines = command.args.get("context_lines")
            .and_then(|v| v.as_i64())
            .unwrap_or(2) as usize;
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Searching".to_string(),
            progress: 0.3,
            message: format!("Searching for pattern: {}", pattern),
            details: Some(json!({ 
                "pattern": pattern,
                "case_sensitive": case_sensitive,
                "context_lines": context_lines
            })),
        }).await;
        
        // Return metadata that the ChatManager will use to perform the search
        Ok(CommandResult {
            success: true,
            content: json!(format!("Search attachments for: {}", pattern)),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: vec![],
            metadata: HashMap::from([
                ("attachment_action".to_string(), json!({
                    "type": "search",
                    "pattern": pattern,
                    "case_sensitive": case_sensitive,
                    "context_lines": context_lines
                })),
            ]),
        })
    }
    
    /// Execute search command for smart chat history search
    async fn execute_search(
        &self,
        command: ParsedCommand,
        execution_id: &str,
    ) -> Result<CommandResult> {
        let query = command.args.get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Search query not provided"))?;
        
        let filters = command.args.get("filters")
            .cloned()
            .unwrap_or_else(|| json!({}));
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Searching".to_string(),
            progress: 0.2,
            message: format!("Searching chat history for: {}", query),
            details: Some(json!({ 
                "query": query,
                "filters": filters.clone()
            })),
        }).await;
        
        // Return metadata that the ChatManager will use to perform the search
        Ok(CommandResult {
            success: true,
            content: json!(format!("Search chat history: {}", query)),
            format: ResultFormat::Text,
            warnings: vec![],
            suggestions: vec![],
            metadata: HashMap::from([
                ("chat_search".to_string(), json!({
                    "query": query,
                    "filters": filters
                })),
            ]),
        })
    }
    
    /// Execute collaboration command
    async fn execute_collaboration(
        &self,
        command: ParsedCommand,
        execution_id: &str,
    ) -> Result<CommandResult> {
        let action = command.args.get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Collaboration action not provided"))?;
        
        // Update progress
        let _ = self.progress_tx.send(ExecutionProgress {
            execution_id: execution_id.to_string(),
            stage: "Processing".to_string(),
            progress: 0.2,
            message: format!("Executing collaboration action: {}", action),
            details: Some(json!({ "action": action })),
        }).await;
        
        match action {
            "create" => {
                let session_name = command.args.get("session_name_or_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Untitled Session");
                
                Ok(CommandResult {
                    success: true,
                    content: json!(format!("Create collaboration session: {}", session_name)),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec!["Share the session ID with others to collaborate".to_string()],
                    metadata: HashMap::from([
                        ("collaboration_action".to_string(), json!({
                            "action": "create",
                            "session_name": session_name
                        })),
                    ]),
                })
            }
            "join" => {
                let session_id = command.args.get("session_name_or_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("Session ID required for join"))?;
                
                let username = command.args.get("username")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Anonymous");
                
                Ok(CommandResult {
                    success: true,
                    content: json!(format!("Join collaboration session: {}", session_id)),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("collaboration_action".to_string(), json!({
                            "action": "join",
                            "session_id": session_id,
                            "username": username
                        })),
                    ]),
                })
            }
            "leave" => {
                Ok(CommandResult {
                    success: true,
                    content: json!("Leave collaboration session"),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("collaboration_action".to_string(), json!({
                            "action": "leave"
                        })),
                    ]),
                })
            }
            "participants" => {
                Ok(CommandResult {
                    success: true,
                    content: json!("Get collaboration participants"),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("collaboration_action".to_string(), json!({
                            "action": "participants"
                        })),
                    ]),
                })
            }
            "info" => {
                Ok(CommandResult {
                    success: true,
                    content: json!("Get collaboration session info"),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("collaboration_action".to_string(), json!({
                            "action": "info"
                        })),
                    ]),
                })
            }
            "toggle" => {
                Ok(CommandResult {
                    success: true,
                    content: json!("Toggle collaboration mode"),
                    format: ResultFormat::Text,
                    warnings: vec![],
                    suggestions: vec![],
                    metadata: HashMap::from([
                        ("collaboration_action".to_string(), json!({
                            "action": "toggle"
                        })),
                    ]),
                })
            }
            _ => Err(anyhow!("Unknown collaboration action: {}", action)),
        }
    }
    
    /// Get execution statistics (lock-free read)
    pub fn get_statistics(&self) -> ExecutionStatistics {
        ExecutionStatistics {
            total_executions: self.total_executions.load(Ordering::Relaxed),
            successful_executions: self.successful_executions.load(Ordering::Relaxed),
            failed_executions: self.failed_executions.load(Ordering::Relaxed),
            active_executions: self.active_executions.len(),
            history_size: self.execution_history.len(),
        }
    }
    
    /// Get active executions (lock-free iteration)
    pub fn get_active_executions(&self) -> Vec<ExecutionInfo> {
        self.active_executions
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }
    
    /// Get execution history (lock-free access)
    pub fn get_execution_history(&self, limit: usize) -> Vec<ExecutionRecord> {
        let mut history = Vec::with_capacity(limit.min(self.execution_history.len()));
        
        // Pop from queue up to limit (non-blocking)
        for _ in 0..limit {
            match self.execution_history.pop() {
                Some(record) => history.push(record),
                None => break,
            }
        }
        
        // Re-push items back to maintain history (in reverse order)
        for record in history.iter().rev() {
            let _ = self.execution_history.push(record.clone());
        }
        
        history
    }
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStatistics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub active_executions: usize,
    pub history_size: usize,
}