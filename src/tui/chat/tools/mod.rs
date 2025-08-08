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
}
