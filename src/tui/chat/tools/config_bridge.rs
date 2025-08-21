//! Tool Configuration Bridge
//! 
//! Bridges the gap between chat UI configuration and tool system settings,
//! managing tool preferences, permissions, and execution policies.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use tracing::{info, debug, warn};

use super::discovery::{ToolDiscoveryEngine, ToolCategory, ToolStatus};
use crate::tools::IntelligentToolManager;
use crate::tui::chat::agents::bindings::ToolBinding;

/// Tool configuration bridge
pub struct ToolConfigBridge {
    /// Tool discovery engine
    discovery: Arc<ToolDiscoveryEngine>,
    
    /// Tool manager
    tool_manager: Arc<IntelligentToolManager>,
    
    /// Tool configurations
    configurations: Arc<RwLock<HashMap<String, ToolConfiguration>>>,
    
    /// Global policies
    global_policies: Arc<RwLock<GlobalToolPolicies>>,
    
    /// User preferences
    user_preferences: Arc<RwLock<UserToolPreferences>>,
    
    /// Execution policies
    execution_policies: Arc<RwLock<Vec<ExecutionPolicy>>>,
}

/// Tool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfiguration {
    pub tool_id: String,
    pub enabled: bool,
    pub auto_execute: bool,
    pub require_confirmation: bool,
    pub timeout_seconds: Option<u32>,
    pub retry_attempts: u32,
    pub rate_limit: Option<RateLimit>,
    pub parameter_defaults: HashMap<String, serde_json::Value>,
    pub parameter_overrides: HashMap<String, serde_json::Value>,
    pub allowed_operations: HashSet<String>,
    pub blocked_operations: HashSet<String>,
    pub custom_settings: HashMap<String, serde_json::Value>,
}

impl Default for ToolConfiguration {
    fn default() -> Self {
        Self {
            tool_id: String::new(),
            enabled: true,
            auto_execute: false,
            require_confirmation: true,
            timeout_seconds: Some(30),
            retry_attempts: 3,
            rate_limit: None,
            parameter_defaults: HashMap::new(),
            parameter_overrides: HashMap::new(),
            allowed_operations: HashSet::new(),
            blocked_operations: HashSet::new(),
            custom_settings: HashMap::new(),
        }
    }
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_second: Option<f32>,
    pub requests_per_minute: Option<u32>,
    pub requests_per_hour: Option<u32>,
    pub burst_size: Option<u32>,
}

/// Global tool policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalToolPolicies {
    pub default_timeout_seconds: u32,
    pub max_parallel_executions: usize,
    pub require_authentication: bool,
    pub audit_all_executions: bool,
    pub sandbox_mode: bool,
    pub allowed_categories: HashSet<ToolCategory>,
    pub blocked_categories: HashSet<ToolCategory>,
    pub resource_limits: ResourceLimits,
}

impl Default for GlobalToolPolicies {
    fn default() -> Self {
        Self {
            default_timeout_seconds: 30,
            max_parallel_executions: 5,
            require_authentication: false,
            audit_all_executions: true,
            sandbox_mode: false,
            allowed_categories: ToolCategory::all().into_iter().collect(),
            blocked_categories: HashSet::new(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u32,
    pub max_cpu_percent: u8,
    pub max_disk_io_mb_per_sec: u32,
    pub max_network_bandwidth_mbps: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_percent: 50,
            max_disk_io_mb_per_sec: 100,
            max_network_bandwidth_mbps: 100,
        }
    }
}

/// User tool preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserToolPreferences {
    pub favorite_tools: HashSet<String>,
    pub recent_tools: Vec<RecentTool>,
    pub tool_shortcuts: HashMap<String, String>,
    pub confirmation_preferences: HashMap<String, ConfirmationPreference>,
    pub notification_settings: NotificationSettings,
}

impl Default for UserToolPreferences {
    fn default() -> Self {
        Self {
            favorite_tools: HashSet::new(),
            recent_tools: Vec::new(),
            tool_shortcuts: HashMap::new(),
            confirmation_preferences: HashMap::new(),
            notification_settings: NotificationSettings::default(),
        }
    }
}

/// Recent tool usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentTool {
    pub tool_id: String,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub usage_count: u32,
    pub average_duration_ms: u64,
}

/// Confirmation preference
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConfirmationPreference {
    Always,
    Never,
    FirstTime,
    Destructive,
}

/// Notification settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub on_start: bool,
    pub on_complete: bool,
    pub on_error: bool,
    pub on_warning: bool,
    pub sound_enabled: bool,
    pub desktop_notifications: bool,
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            on_start: false,
            on_complete: true,
            on_error: true,
            on_warning: true,
            sound_enabled: false,
            desktop_notifications: false,
        }
    }
}

/// Execution policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPolicy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub conditions: Vec<PolicyCondition>,
    pub actions: Vec<PolicyAction>,
    pub priority: u8,
    pub enabled: bool,
}

/// Policy condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    ToolCategory(ToolCategory),
    ToolId(String),
    TimeWindow { start: String, end: String },
    UserRole(String),
    ResourceUsage { metric: String, threshold: f64 },
    Custom(String),
}

/// Policy action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    Block,
    RequireApproval,
    Log,
    Notify(String),
    RateLimit(RateLimit),
    Redirect(String),
}

impl ToolConfigBridge {
    /// Create a new configuration bridge
    pub fn new(
        discovery: Arc<ToolDiscoveryEngine>,
        tool_manager: Arc<IntelligentToolManager>,
    ) -> Self {
        Self {
            discovery,
            tool_manager,
            configurations: Arc::new(RwLock::new(HashMap::new())),
            global_policies: Arc::new(RwLock::new(GlobalToolPolicies::default())),
            user_preferences: Arc::new(RwLock::new(UserToolPreferences::default())),
            execution_policies: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Initialize tool configurations
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing tool configuration bridge");
        
        // Load saved configurations
        self.load_configurations().await?;
        
        // Auto-configure discovered tools
        let tools = self.discovery.get_all_tools().await;
        for tool in tools {
            if !self.has_configuration(&tool.id).await {
                self.create_default_configuration(&tool.id).await?;
            }
        }
        
        // Initialize default policies
        self.initialize_default_policies().await?;
        
        Ok(())
    }
    
    /// Check if configuration exists
    async fn has_configuration(&self, tool_id: &str) -> bool {
        self.configurations.read().await.contains_key(tool_id)
    }
    
    /// Create default configuration for a tool
    async fn create_default_configuration(&self, tool_id: &str) -> Result<()> {
        let mut config = ToolConfiguration::default();
        config.tool_id = tool_id.to_string();
        
        // Set defaults based on tool category
        if let Some(tool) = self.discovery.get_tool(tool_id).await {
            match tool.category {
                ToolCategory::FileSystem | ToolCategory::Database => {
                    config.require_confirmation = true;
                    config.auto_execute = false;
                }
                ToolCategory::Research | ToolCategory::DataAnalysis => {
                    config.require_confirmation = false;
                    config.auto_execute = true;
                }
                _ => {}
            }
        }
        
        self.configurations.write().await.insert(tool_id.to_string(), config);
        Ok(())
    }
    
    /// Initialize default execution policies
    async fn initialize_default_policies(&self) -> Result<()> {
        let mut policies = self.execution_policies.write().await;
        
        // Sandbox policy
        policies.push(ExecutionPolicy {
            id: "sandbox".to_string(),
            name: "Sandbox Mode".to_string(),
            description: "Restrict dangerous operations in sandbox mode".to_string(),
            conditions: vec![],
            actions: vec![
                PolicyAction::Block,
                PolicyAction::Log,
            ],
            priority: 100,
            enabled: false,
        });
        
        // Rate limiting policy
        policies.push(ExecutionPolicy {
            id: "rate_limit".to_string(),
            name: "Global Rate Limiting".to_string(),
            description: "Apply rate limits to all tools".to_string(),
            conditions: vec![],
            actions: vec![
                PolicyAction::RateLimit(RateLimit {
                    requests_per_second: Some(10.0),
                    requests_per_minute: Some(100),
                    requests_per_hour: Some(1000),
                    burst_size: Some(20),
                }),
            ],
            priority: 50,
            enabled: true,
        });
        
        Ok(())
    }
    
    /// Get tool configuration
    pub async fn get_configuration(&self, tool_id: &str) -> Option<ToolConfiguration> {
        self.configurations.read().await.get(tool_id).cloned()
    }
    
    /// Update tool configuration
    pub async fn update_configuration(
        &self,
        tool_id: &str,
        config: ToolConfiguration,
    ) -> Result<()> {
        self.configurations.write().await.insert(tool_id.to_string(), config);
        info!("Updated configuration for tool: {}", tool_id);
        Ok(())
    }
    
    /// Check if tool execution is allowed
    pub async fn is_execution_allowed(
        &self,
        tool_id: &str,
        context: &ExecutionContext,
    ) -> Result<bool> {
        // Check if tool is enabled
        if let Some(config) = self.get_configuration(tool_id).await {
            if !config.enabled {
                return Ok(false);
            }
        }
        
        // Check global policies
        let policies = self.global_policies.read().await;
        if let Some(tool) = self.discovery.get_tool(tool_id).await {
            if policies.blocked_categories.contains(&tool.category) {
                return Ok(false);
            }
            if !policies.allowed_categories.contains(&tool.category) {
                return Ok(false);
            }
        }
        
        // Check execution policies
        let exec_policies = self.execution_policies.read().await;
        for policy in exec_policies.iter() {
            if !policy.enabled {
                continue;
            }
            
            if self.matches_conditions(&policy.conditions, tool_id, context) {
                for action in &policy.actions {
                    if let PolicyAction::Block = action {
                        return Ok(false);
                    }
                }
            }
        }
        
        Ok(true)
    }
    
    /// Check if confirmation is required
    pub async fn requires_confirmation(
        &self,
        tool_id: &str,
        context: &ExecutionContext,
    ) -> bool {
        // Check tool configuration
        if let Some(config) = self.get_configuration(tool_id).await {
            if config.require_confirmation {
                return true;
            }
        }
        
        // Check user preferences
        let prefs = self.user_preferences.read().await;
        if let Some(pref) = prefs.confirmation_preferences.get(tool_id) {
            match pref {
                ConfirmationPreference::Always => return true,
                ConfirmationPreference::Never => return false,
                ConfirmationPreference::FirstTime => {
                    // Check if used before
                    if !prefs.recent_tools.iter().any(|t| t.tool_id == tool_id) {
                        return true;
                    }
                }
                ConfirmationPreference::Destructive => {
                    // Check if operation is destructive
                    if context.is_destructive {
                        return true;
                    }
                }
            }
        }
        
        // Check policies
        let exec_policies = self.execution_policies.read().await;
        for policy in exec_policies.iter() {
            if self.matches_conditions(&policy.conditions, tool_id, context) {
                for action in &policy.actions {
                    if let PolicyAction::RequireApproval = action {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Apply configuration to execution
    pub async fn apply_configuration(
        &self,
        tool_id: &str,
        params: &mut HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        if let Some(config) = self.get_configuration(tool_id).await {
            // Apply parameter defaults
            for (key, value) in &config.parameter_defaults {
                params.entry(key.clone()).or_insert(value.clone());
            }
            
            // Apply parameter overrides
            for (key, value) in &config.parameter_overrides {
                params.insert(key.clone(), value.clone());
            }
        }
        
        Ok(())
    }
    
    /// Record tool usage
    pub async fn record_usage(
        &self,
        tool_id: &str,
        duration_ms: u64,
        success: bool,
    ) {
        let mut prefs = self.user_preferences.write().await;
        
        // Update recent tools
        if let Some(recent) = prefs.recent_tools.iter_mut().find(|t| t.tool_id == tool_id) {
            recent.last_used = chrono::Utc::now();
            recent.usage_count += 1;
            recent.average_duration_ms = 
                (recent.average_duration_ms * (recent.usage_count - 1) as u64 + duration_ms) 
                / recent.usage_count as u64;
        } else {
            prefs.recent_tools.push(RecentTool {
                tool_id: tool_id.to_string(),
                last_used: chrono::Utc::now(),
                usage_count: 1,
                average_duration_ms: duration_ms,
            });
        }
        
        // Keep only last 20 recent tools
        if prefs.recent_tools.len() > 20 {
            prefs.recent_tools.sort_by(|a, b| b.last_used.cmp(&a.last_used));
            prefs.recent_tools.truncate(20);
        }
    }
    
    /// Get favorite tools
    pub async fn get_favorite_tools(&self) -> HashSet<String> {
        self.user_preferences.read().await.favorite_tools.clone()
    }
    
    /// Add favorite tool
    pub async fn add_favorite(&self, tool_id: &str) {
        self.user_preferences.write().await.favorite_tools.insert(tool_id.to_string());
    }
    
    /// Remove favorite tool
    pub async fn remove_favorite(&self, tool_id: &str) {
        self.user_preferences.write().await.favorite_tools.remove(tool_id);
    }
    
    /// Get tool shortcuts
    pub async fn get_shortcuts(&self) -> HashMap<String, String> {
        self.user_preferences.read().await.tool_shortcuts.clone()
    }
    
    /// Set tool shortcut
    pub async fn set_shortcut(&self, shortcut: &str, tool_id: &str) {
        self.user_preferences.write().await
            .tool_shortcuts.insert(shortcut.to_string(), tool_id.to_string());
    }
    
    /// Get user preferences (read access)
    pub fn get_user_preferences(&self) -> &Arc<RwLock<UserToolPreferences>> {
        &self.user_preferences
    }
    
    /// Check if conditions match
    fn matches_conditions(
        &self,
        conditions: &[PolicyCondition],
        tool_id: &str,
        context: &ExecutionContext,
    ) -> bool {
        for condition in conditions {
            match condition {
                PolicyCondition::ToolId(id) => {
                    if id != tool_id {
                        return false;
                    }
                }
                PolicyCondition::UserRole(role) => {
                    if context.user_role != *role {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }
    
    /// Load configurations from storage
    async fn load_configurations(&self) -> Result<()> {
        use std::path::PathBuf;
        use tokio::fs;
        
        // Get config directory
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory. Please ensure HOME environment variable is set"))?
            .join("loki")
            .join("chat")
            .join("tools");
        
        // Load tool configurations if file exists
        let tool_config_path = config_dir.join("tool_configs.json");
        if tool_config_path.exists() {
            let content = fs::read_to_string(&tool_config_path).await
                .context("Failed to read tool configurations")?;
            let configs: HashMap<String, ToolConfiguration> = serde_json::from_str(&content)
                .context("Failed to parse tool configurations")?;
            
            let mut tool_configs = self.configurations.write().await;
            tool_configs.extend(configs);
            
            tracing::info!("Loaded {} tool configurations from storage", tool_configs.len());
        }
        
        // Load global policies if file exists
        let policy_path = config_dir.join("global_policies.json");
        if policy_path.exists() {
            let content = fs::read_to_string(&policy_path).await
                .context("Failed to read global policies")?;
            let policies: GlobalToolPolicies = serde_json::from_str(&content)
                .context("Failed to parse global policies")?;
            
            let mut global_policies = self.global_policies.write().await;
            *global_policies = policies;
            
            tracing::info!("Loaded global policies from storage");
        }
        
        Ok(())
    }
    
    /// Save configurations to storage
    pub async fn save_configurations(&self) -> Result<()> {
        use std::path::PathBuf;
        use tokio::fs;
        
        // Get config directory and create if needed
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory. Please ensure HOME environment variable is set"))?
            .join("loki")
            .join("chat")
            .join("tools");
        
        fs::create_dir_all(&config_dir).await
            .context("Failed to create config directory")?;
        
        // Save tool configurations
        let tool_configs = self.configurations.read().await;
        if !tool_configs.is_empty() {
            let tool_config_path = config_dir.join("tool_configs.json");
            let content = serde_json::to_string_pretty(&*tool_configs)
                .context("Failed to serialize tool configurations")?;
            fs::write(&tool_config_path, content).await
                .context("Failed to write tool configurations")?;
            
            tracing::info!("Saved {} tool configurations to storage", tool_configs.len());
        }
        
        // Save global policies
        let global_policies = self.global_policies.read().await;
        let policy_path = config_dir.join("global_policies.json");
        let content = serde_json::to_string_pretty(&*global_policies)
            .context("Failed to serialize global policies")?;
        fs::write(&policy_path, content).await
            .context("Failed to write global policies")?;
        
        tracing::info!("Saved global policies to storage");
        
        Ok(())
    }
}

/// Execution context for policy evaluation
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub user_id: String,
    pub user_role: String,
    pub agent_id: Option<String>,
    pub is_destructive: bool,
    pub estimated_duration_ms: u64,
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub memory_mb: u32,
    pub cpu_cores: f32,
    pub disk_io_mb: u32,
    pub network_mb: u32,
}

/// Tool preset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPreset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tool_configs: HashMap<String, ToolConfiguration>,
    pub global_policies: GlobalToolPolicies,
    pub icon: String,
}

impl ToolConfigBridge {
    /// Apply a preset configuration
    pub async fn apply_preset(&self, preset: &ToolPreset) -> Result<()> {
        // Apply tool configurations
        let mut configs = self.configurations.write().await;
        for (tool_id, config) in &preset.tool_configs {
            configs.insert(tool_id.clone(), config.clone());
        }
        
        // Apply global policies
        *self.global_policies.write().await = preset.global_policies.clone();
        
        info!("Applied preset: {}", preset.name);
        Ok(())
    }
    
    /// Create preset from current configuration
    pub async fn create_preset(&self, name: String, description: String) -> ToolPreset {
        ToolPreset {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            tool_configs: self.configurations.read().await.clone(),
            global_policies: self.global_policies.read().await.clone(),
            icon: "🛠️".to_string(),
        }
    }
}
