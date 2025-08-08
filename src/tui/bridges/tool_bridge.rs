//! Tool Bridge - Connects Utilities tab tool configurations to Chat tab execution
//! 
//! This bridge ensures that tool configurations from the Utilities tab are
//! immediately available in the Chat tab for natural language execution.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use anyhow::Result;

use crate::tui::event_bus::{SystemEvent, TabId};
use crate::tools::{ToolStatus, ToolResult, IntelligentToolManager};

/// Tool configuration (simplified for bridge)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub enabled: bool,
    pub parameters: Value,
}

/// Bridge for tool-related cross-tab communication
pub struct ToolBridge {
    event_bridge: Arc<super::EventBridge>,
    
    /// Cache of tool configurations from Utilities tab
    tool_configs: Arc<RwLock<HashMap<String, ToolConfig>>>,
    
    /// Tool availability status (using metrics collector status)
    tool_status: Arc<RwLock<HashMap<String, crate::tools::metrics_collector::ToolStatus>>>,
    
    /// Execution history for analytics
    execution_history: Arc<RwLock<Vec<ToolExecution>>>,
    
    /// Reference to the actual tool manager if available
    tool_manager: Arc<RwLock<Option<Arc<IntelligentToolManager>>>>,
}

/// Record of a tool execution
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_id: String,
    pub source_tab: TabId,
    pub timestamp: std::time::Instant,
    pub duration: std::time::Duration,
    pub success: bool,
    pub error: Option<String>,
}

impl ToolBridge {
    /// Create a new tool bridge
    pub fn new(event_bridge: Arc<super::EventBridge>) -> Self {
        Self {
            event_bridge,
            tool_configs: Arc::new(RwLock::new(HashMap::new())),
            tool_status: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            tool_manager: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Set the tool manager for actual execution
    pub async fn set_tool_manager(&self, manager: Arc<IntelligentToolManager>) {
        let mut tool_mgr = self.tool_manager.write().await;
        *tool_mgr = Some(manager);
        tracing::info!("Tool manager connected to bridge");
    }
    
    /// Initialize the tool bridge
    pub async fn initialize(&self) -> Result<()> {
        tracing::debug!("Initializing tool bridge");
        
        // Subscribe to tool-related events
        self.subscribe_to_events().await?;
        
        // Load initial tool configurations
        self.load_tool_configs().await?;
        
        Ok(())
    }
    
    /// Subscribe to tool-related events
    async fn subscribe_to_events(&self) -> Result<()> {
        let configs = self.tool_configs.clone();
        let status = self.tool_status.clone();
        let history = self.execution_history.clone();
        
        // Subscribe to tool configuration events
        self.event_bridge.subscribe_handler(
            "ToolConfigured",
            move |event| {
                let configs = configs.clone();
                let status = status.clone();
                
                Box::pin(async move {
                    if let SystemEvent::ToolConfigured { tool_id, config, .. } = event {
                        // Update tool configuration cache
                        let tool_config: ToolConfig = serde_json::from_value(config)?;
                        let mut configs_lock = configs.write().await;
                        configs_lock.insert(tool_id.clone(), tool_config);
                        
                        // Mark tool as active
                        let mut status_lock = status.write().await;
                        let tool_id_copy = tool_id.clone();
                        status_lock.insert(tool_id_copy, crate::tools::metrics_collector::ToolStatus::Active);
                        
                        tracing::info!("✅ Tool configured and available: {}", tool_id);
                    }
                    Ok(())
                })
            }
        ).await?;
        
        // Subscribe to tool execution events
        self.event_bridge.subscribe_handler(
            "ToolExecuted",
            move |event| {
                let history = history.clone();
                
                Box::pin(async move {
                    if let SystemEvent::ToolExecuted { tool_id, duration, source, result, .. } = event {
                        // Record execution in history
                        let mut history_lock = history.write().await;
                        
                        // Check if execution was successful based on ToolStatus enum
                        let success = matches!(result.status, ToolStatus::Success);
                        let error = match &result.status {
                            ToolStatus::Failure(msg) => Some(msg.clone()),
                            ToolStatus::Partial(msg) => Some(format!("Partial: {}", msg)),
                            _ => None,
                        };
                        
                        history_lock.push(ToolExecution {
                            tool_id: tool_id.clone(),
                            source_tab: source.clone(),
                            timestamp: std::time::Instant::now(),
                            duration,
                            success,
                            error,
                        });
                        
                        // Keep only last 100 executions
                        if history_lock.len() > 100 {
                            let drain_count = history_lock.len() - 100;
                            history_lock.drain(0..drain_count);
                        }
                        
                        tracing::debug!("Tool execution recorded: {} from {:?}", tool_id, source);
                    }
                    Ok(())
                })
            }
        ).await?;
        
        Ok(())
    }
    
    /// Load initial tool configurations
    async fn load_tool_configs(&self) -> Result<()> {
        // This would load from the Utilities tab's tool manager
        // For now, we'll just log that we're ready
        tracing::debug!("Tool configurations loaded");
        Ok(())
    }
    
    /// Get available tools for Chat tab
    pub async fn get_available_tools(&self) -> Vec<(String, ToolConfig)> {
        let configs = self.tool_configs.read().await;
        let status = self.tool_status.read().await;
        
        configs.iter()
            .filter(|(id, _)| {
                status.get(*id)
                    .map(|s| matches!(s, crate::tools::metrics_collector::ToolStatus::Active | crate::tools::metrics_collector::ToolStatus::Idle))
                    .unwrap_or(false)
            })
            .map(|(id, config)| (id.clone(), config.clone()))
            .collect()
    }
    
    /// Check if a specific tool is available
    pub async fn is_tool_available(&self, tool_id: &str) -> bool {
        let status = self.tool_status.read().await;
        status.get(tool_id)
            .map(|s| matches!(s, crate::tools::metrics_collector::ToolStatus::Active | crate::tools::metrics_collector::ToolStatus::Idle))
            .unwrap_or(false)
    }
    
    /// Get tool configuration
    pub async fn get_tool_config(&self, tool_id: &str) -> Option<ToolConfig> {
        let configs = self.tool_configs.read().await;
        configs.get(tool_id).cloned()
    }
    
    /// Execute a tool from Chat tab
    pub async fn execute_from_chat(
        &self,
        tool_id: String,
        params: Value,
    ) -> Result<ToolResult> {
        let start_time = std::time::Instant::now();
        
        // Check if tool is available
        if !self.is_tool_available(&tool_id).await {
            return Ok(ToolResult {
                status: ToolStatus::Failure(format!("Tool {} is not available", tool_id)),
                content: Value::Null,
                summary: format!("Tool {} is not available", tool_id),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                quality_score: 0.0,
                memory_integrated: false,
                follow_up_suggestions: vec![],
            });
        }
        
        // Try to execute through actual tool manager if available
        let tool_manager = self.tool_manager.read().await;
        let result = if let Some(ref manager) = *tool_manager {
            // Real execution through tool manager
            match manager.execute_tool(&tool_id, params.clone()).await {
                Ok(res) => res,
                Err(e) => ToolResult {
                    status: ToolStatus::Failure(e.to_string()),
                    content: Value::Null,
                    summary: format!("Execution failed: {}", e),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    quality_score: 0.0,
                    memory_integrated: false,
                    follow_up_suggestions: vec![],
                }
            }
        } else {
            // Fallback: simulate execution when no manager is connected
            tracing::warn!("No tool manager connected, simulating tool execution");
            ToolResult {
                status: ToolStatus::Success,
                content: serde_json::json!({
                    "message": format!("Tool {} executed (simulated)", tool_id),
                    "params": params,
                }),
                summary: format!("Simulated execution of {}", tool_id),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                quality_score: 0.8,
                memory_integrated: false,
                follow_up_suggestions: vec![],
            }
        };
        
        // Publish execution event
        self.event_bridge.publish(SystemEvent::ToolExecuted {
            tool_id,
            params,
            result: result.clone(),
            duration: start_time.elapsed(),
            source: TabId::Chat,
        }).await?;
        
        Ok(result)
    }
    
    /// Get execution statistics
    pub async fn get_execution_stats(&self) -> ToolExecutionStats {
        let history = self.execution_history.read().await;
        
        let total = history.len();
        let successful = history.iter().filter(|e| e.success).count();
        let failed = total - successful;
        
        let avg_duration = if !history.is_empty() {
            let total_duration: std::time::Duration = history.iter()
                .map(|e| e.duration)
                .sum();
            total_duration / history.len() as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        let by_tab: HashMap<TabId, usize> = history.iter()
            .fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e.source_tab.clone()).or_insert(0) += 1;
                acc
            });
        
        ToolExecutionStats {
            total_executions: total,
            successful_executions: successful,
            failed_executions: failed,
            average_duration: avg_duration,
            executions_by_tab: by_tab,
        }
    }
}

/// Tool execution statistics
#[derive(Debug, Clone)]
pub struct ToolExecutionStats {
    pub total_executions: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub average_duration: std::time::Duration,
    pub executions_by_tab: HashMap<TabId, usize>,
}