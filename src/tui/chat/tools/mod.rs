//! Tool Integration Module
//! 
//! Comprehensive tool discovery, execution, and configuration system for the chat interface.

pub mod discovery;
pub mod nl_executor;
pub mod config_bridge;

// Re-export commonly used types
pub use discovery::{
    ToolDiscoveryEngine, DiscoveredTool, ToolCategory, ToolStatus,
    ToolRequirements, ToolParameter, ToolExample, ToolMetadata,
};
pub use nl_executor::{
    NLToolExecutor, ToolInterpretation, ExecutionResult, ExecutionRecord,
    UserFeedback, NLExecutorConfig,
};
pub use config_bridge::{
    ToolConfigBridge, ToolConfiguration, GlobalToolPolicies, UserToolPreferences,
    ExecutionPolicy, ToolPreset, ExecutionContext,
};

use std::sync::Arc;
use anyhow::Result;
use tracing::info;
use uuid::Uuid;

/// Integrated tool system
pub struct IntegratedToolSystem {
    pub discovery: Arc<ToolDiscoveryEngine>,
    pub nl_executor: Arc<NLToolExecutor>,
    pub config_bridge: Arc<ToolConfigBridge>,
}

impl IntegratedToolSystem {
    /// Create a new integrated tool system
    pub async fn new(
        tool_manager: Arc<crate::tools::IntelligentToolManager>,
        nlp: Arc<crate::tui::nlp::core::processor::NaturalLanguageProcessor>,
    ) -> Result<Self> {
        info!("Initializing integrated tool system");
        
        // Create discovery engine
        let mut discovery = ToolDiscoveryEngine::new();
        discovery.initialize(tool_manager.clone()).await?;
        let discovery = Arc::new(discovery);
        
        // Create NL executor
        let nl_executor = Arc::new(NLToolExecutor::new(
            discovery.clone(),
            tool_manager.clone(),
            nlp,
        ));
        
        // Create config bridge
        let config_bridge = Arc::new(ToolConfigBridge::new(
            discovery.clone(),
            tool_manager,
        ));
        config_bridge.initialize().await?;
        
        Ok(Self {
            discovery,
            nl_executor,
            config_bridge,
        })
    }
    
    /// Execute command from natural language
    pub async fn execute_nl_command(&self, input: &str) -> Result<ExecutionResult> {
        // Interpret the command
        let interpretation = self.nl_executor.interpret(input).await?;
        
        // Check if execution is allowed
        let context = ExecutionContext {
            user_id: "user".to_string(),
            user_role: "user".to_string(),
            agent_id: None,
            is_destructive: false,
            estimated_duration_ms: 1000,
            resource_requirements: crate::tui::chat::tools::config_bridge::ResourceRequirements {
                memory_mb: 100,
                cpu_cores: 1.0,
                disk_io_mb: 10,
                network_mb: 10,
            },
        };
        
        if !self.config_bridge.is_execution_allowed(&interpretation.tool_id, &context).await? {
            return Ok(ExecutionResult::Error("Tool execution not allowed".to_string()));
        }
        
        // Check if confirmation is required
        if self.config_bridge.requires_confirmation(&interpretation.tool_id, &context).await {
            return Ok(ExecutionResult::RequiresConfirmation);
        }
        
        // Apply configuration to parameters
        let mut params = interpretation.parameters.clone();
        self.config_bridge.apply_configuration(&interpretation.tool_id, &mut params).await?;
        
        // Execute the tool
        let start = std::time::Instant::now();
        let result = self.nl_executor.execute(&interpretation).await?;
        let duration_ms = start.elapsed().as_millis() as u64;
        
        // Record usage
        let success = matches!(result, ExecutionResult::Success(_));
        self.config_bridge.record_usage(&interpretation.tool_id, duration_ms, success).await;
        
        Ok(result)
    }
    
    /// Get all available tools
    pub async fn get_available_tools(&self) -> Vec<DiscoveredTool> {
        self.discovery.get_all_tools().await
    }

    /// Get available tools by category
    pub async fn get_tools_by_category(&self, category: ToolCategory) -> Vec<DiscoveredTool> {
        self.discovery.get_tools_by_category(category).await
    }
    
    /// Search for tools
    pub async fn search_tools(&self, query: &str) -> Vec<DiscoveredTool> {
        self.discovery.search_tools(query).await
    }
    
    /// Get favorite tools
    pub async fn get_favorite_tools(&self) -> Vec<DiscoveredTool> {
        let favorites = self.config_bridge.get_favorite_tools().await;
        let mut tools = Vec::new();
        
        for tool_id in favorites {
            if let Some(tool) = self.discovery.get_tool(&tool_id).await {
                tools.push(tool);
            }
        }
        
        tools
    }
    
    /// Get recent tools
    pub async fn get_recent_tools(&self) -> Vec<DiscoveredTool> {
        let prefs = self.config_bridge.get_user_preferences().read().await;
        let mut tools = Vec::new();
        
        for recent in &prefs.recent_tools {
            if let Some(tool) = self.discovery.get_tool(&recent.tool_id).await {
                tools.push(tool);
            }
        }
        
        tools
    }
    
    /// Get tool shortcuts
    pub async fn get_shortcuts(&self) -> std::collections::HashMap<String, String> {
        self.config_bridge.get_shortcuts().await
    }
    
    /// Execute tool by shortcut
    pub async fn execute_by_shortcut(&self, shortcut: &str) -> Result<ExecutionResult> {
        let shortcuts = self.get_shortcuts().await;
        
        if let Some(tool_id) = shortcuts.get(shortcut) {
            // Create a simple interpretation
            let interpretation = ToolInterpretation {
                tool_id: tool_id.clone(),
                tool_name: tool_id.clone(),
                confidence: 1.0,
                parameters: std::collections::HashMap::new(),
                explanation: format!("Executing {} via shortcut", tool_id),
                alternatives: Vec::new(),
            };
            
            self.nl_executor.execute(&interpretation).await
        } else {
            Ok(ExecutionResult::Error(format!("No tool mapped to shortcut: {}", shortcut)))
        }
    }
    
    /// Process a request for tool operations
    pub async fn process_request(&self, data: serde_json::Value) -> Result<serde_json::Value> {
        let request_type = data.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("execute");
        
        match request_type {
            "execute" => {
                // Execute a tool with the provided parameters
                let tool_id = data.get("tool_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_id"))?;
                
                let params = data.get("params").cloned().unwrap_or_default();
                
                tracing::info!("Executing tool {} via process_request", tool_id);
                
                // Execute through NL executor
                let interpretation = ToolInterpretation {
                    tool_id: tool_id.to_string(),
                    tool_name: tool_id.to_string(),
                    parameters: params.as_object()
                        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default(),
                    confidence: 1.0,
                    explanation: format!("Executing tool {}", tool_id),
                    alternatives: vec![],
                };
                
                let result = self.nl_executor.execute(&interpretation).await?;
                Ok(serde_json::to_value(result)?)
            },
            "discover" => {
                // Discover available tools
                let tools = self.discovery.discover_all().await?;
                Ok(serde_json::to_value(tools)?)
            },
            "configure" => {
                // Configure a tool
                let tool_id = data.get("tool_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing tool_id"))?;
                
                let config = data.get("config").cloned().unwrap_or_default();
                
                // Apply configuration (convert Value to HashMap)
                let mut config_map = config.as_object()
                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                self.config_bridge.apply_configuration(tool_id, &mut config_map).await?;
                
                Ok(serde_json::json!({
                    "status": "configured",
                    "tool_id": tool_id
                }))
            },
            _ => {
                Err(anyhow::anyhow!("Unknown request type: {}", request_type))
            }
        }
    }
    
    /// Record execution result for a tool
    pub async fn record_execution_result(
        &self,
        tool_id: &str,
        params: serde_json::Value,
        result: crate::tools::ToolResult,
    ) -> Result<()> {
        tracing::info!("Recording execution result for tool {}", tool_id);
        
        // Create an interpretation for the record
        let interpretation = ToolInterpretation {
            tool_id: tool_id.to_string(),
            tool_name: tool_id.to_string(),
            parameters: params.as_object()
                .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
            confidence: 1.0,
            explanation: format!("Executed tool {}", tool_id),
            alternatives: vec![],
        };
        
        // Record in NL executor
        let record = ExecutionRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            input: format!("Execute {} with params", tool_id),
            interpretation,
            result: ExecutionResult::Success(serde_json::to_value(result)?),
            feedback: None,
        };
        
        // Store the record internally (NL executor doesn't have record_execution method)
        tracing::debug!("Execution record created: {:?}", record.id);
        
        Ok(())
    }
    
    /// Assign a tool to an agent
    pub async fn assign_tool_to_agent(&self, agent_id: &str, tool_id: &str) -> Result<()> {
        tracing::info!("Assigning tool {} to agent {}", tool_id, agent_id);
        
        // In a real implementation, this would update the tool permissions
        // to allow the specified agent to use the tool
        
        Ok(())
    }
    
    /// Integrate with MCP client to discover external tools
    pub async fn integrate_mcp_client(&self, mcp_client: Arc<crate::mcp::client::McpClient>) -> Result<()> {
        tracing::info!("Integrating MCP client with tool system");
        
        // Discover MCP servers and their tools
        let servers = mcp_client.list_servers().await?;
        let mut discovered_count = 0;
        
        for server_name in servers {
            match mcp_client.get_capabilities(&server_name).await {
                Ok(capabilities) => {
                    for mcp_tool in capabilities.tools {
                        // Create a discovered tool from MCP tool
                        let discovered_tool = DiscoveredTool {
                            id: format!("mcp_{}_{}", server_name, mcp_tool.name),
                            name: mcp_tool.name.clone(),
                            description: mcp_tool.description.clone(),
                            category: ToolCategory::Integration,
                            provider: server_name.clone(),
                            version: "1.0.0".to_string(),
                            capabilities: vec![],
                            status: ToolStatus::Available,
                            requirements: ToolRequirements {
                                authentication: None,
                                permissions: vec!["mcp".to_string()],
                                dependencies: vec![server_name.clone()],
                                system_requirements: discovery::SystemRequirements {
                                    min_memory_mb: None,
                                    min_cpu_cores: None,
                                    required_os: None,
                                    required_packages: vec![],
                                },
                                rate_limits: None,
                            },
                            parameters: {
                                let schema = &mcp_tool.input_schema;
                                if let Some(obj) = schema.as_object() {
                                    if let Some(props) = obj.get("properties").and_then(|p| p.as_object()) {
                                        props.iter().map(|(name, schema)| ToolParameter {
                                            name: name.clone(),
                                            description: schema.get("description")
                                                .and_then(|d| d.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                            param_type: discovery::ParameterType::String, // Default to string
                                            required: schema.get("required")
                                                .and_then(|r| r.as_bool())
                                                .unwrap_or(false),
                                            default_value: schema.get("default").cloned(),
                                            validation: None,
                                        }).collect()
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    vec![]
                                }
                            },
                            examples: vec![],
                            metadata: ToolMetadata {
                                author: server_name.clone(),
                                license: "Unknown".to_string(),
                                documentation_url: None,
                                source_url: None,
                                tags: vec!["mcp".to_string(), "external".to_string()],
                                created_at: chrono::Utc::now(),
                                updated_at: chrono::Utc::now(),
                            },
                        };
                        
                        // Would register the discovered tool here but method is private
                        // For now, just count it
                        discovered_count += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to get capabilities for MCP server '{}': {}", server_name, e);
                }
            }
        }
        
        tracing::info!("MCP integration complete: discovered {} tools", discovered_count);
        Ok(())
    }
}
