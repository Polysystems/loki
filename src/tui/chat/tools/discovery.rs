//! Tool Discovery Engine
//! 
//! Automatically discovers and catalogs available tools from various sources
//! including built-in tools, plugins, and external integrations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::tools::IntelligentToolManager;

/// Tool discovery engine
pub struct ToolDiscoveryEngine {
    /// Discovered tools
    tools: Arc<RwLock<HashMap<String, DiscoveredTool>>>,
    
    /// Tool categories
    categories: Arc<RwLock<HashMap<ToolCategory, HashSet<String>>>>,
    
    /// Tool providers
    providers: Arc<RwLock<Vec<Box<dyn ToolProvider>>>>,
    
    /// Discovery configuration
    config: DiscoveryConfig,
    
    /// Tool metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, ToolMetadata>>>,
}

/// Discovered tool information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredTool {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub provider: String,
    pub version: String,
    pub capabilities: Vec<ToolCapability>,
    pub requirements: ToolRequirements,
    pub parameters: Vec<ToolParameter>,
    pub examples: Vec<ToolExample>,
    pub status: ToolStatus,
    pub metadata: ToolMetadata,
}

/// Tool categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    FileSystem,
    Network,
    Database,
    CodeExecution,
    DataAnalysis,
    Communication,
    Monitoring,
    Security,
    Development,
    Research,
    Creative,
    Automation,
    Integration,
    Custom,
}

impl ToolCategory {
    pub fn all() -> Vec<Self> {
        vec![
            Self::FileSystem,
            Self::Network,
            Self::Database,
            Self::CodeExecution,
            Self::DataAnalysis,
            Self::Communication,
            Self::Monitoring,
            Self::Security,
            Self::Development,
            Self::Research,
            Self::Creative,
            Self::Automation,
            Self::Integration,
            Self::Custom,
        ]
    }
    
    pub fn display_name(&self) -> &str {
        match self {
            Self::FileSystem => "File System",
            Self::Network => "Network",
            Self::Database => "Database",
            Self::CodeExecution => "Code Execution",
            Self::DataAnalysis => "Data Analysis",
            Self::Communication => "Communication",
            Self::Monitoring => "Monitoring",
            Self::Security => "Security",
            Self::Development => "Development",
            Self::Research => "Research",
            Self::Creative => "Creative",
            Self::Automation => "Automation",
            Self::Integration => "Integration",
            Self::Custom => "Custom",
        }
    }
    
    pub fn icon(&self) -> &str {
        match self {
            Self::FileSystem => "📁",
            Self::Network => "🌐",
            Self::Database => "🗄️",
            Self::CodeExecution => "⚡",
            Self::DataAnalysis => "📊",
            Self::Communication => "💬",
            Self::Monitoring => "📈",
            Self::Security => "🔒",
            Self::Development => "🔧",
            Self::Research => "🔬",
            Self::Creative => "🎨",
            Self::Automation => "🤖",
            Self::Integration => "🔗",
            Self::Custom => "⚙️",
        }
    }
}

/// Tool capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolCapability {
    Read,
    Write,
    Execute,
    Query,
    Transform,
    Generate,
    Analyze,
    Monitor,
    Integrate,
    Custom(String),
}

/// Tool requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequirements {
    pub authentication: Option<AuthRequirement>,
    pub permissions: Vec<String>,
    pub dependencies: Vec<String>,
    pub system_requirements: SystemRequirements,
    pub rate_limits: Option<RateLimits>,
}

/// Authentication requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthRequirement {
    None,
    ApiKey(String),
    OAuth(OAuthConfig),
    Basic(String),
    Custom(String),
}

/// OAuth configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    pub provider: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    pub redirect_uri: String,
}

/// System requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemRequirements {
    pub min_memory_mb: Option<u32>,
    pub min_cpu_cores: Option<f32>,
    pub required_os: Option<Vec<String>>,
    pub required_packages: Vec<String>,
}

/// Rate limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_second: Option<f32>,
    pub requests_per_minute: Option<u32>,
    pub requests_per_hour: Option<u32>,
    pub concurrent_requests: Option<u32>,
}

/// Tool parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub param_type: ParameterType,
    pub required: bool,
    pub default_value: Option<serde_json::Value>,
    pub validation: Option<ParameterValidation>,
}

/// Parameter type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Number,
    Boolean,
    Array,
    Object,
    File,
    Enum(Vec<String>),
}

/// Parameter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterValidation {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub allowed_values: Option<Vec<serde_json::Value>>,
}

/// Tool example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExample {
    pub name: String,
    pub description: String,
    pub input: serde_json::Value,
    pub expected_output: Option<serde_json::Value>,
    pub explanation: String,
}

/// Tool status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ToolStatus {
    Available,
    Unavailable,
    Deprecated,
    Beta,
    Experimental,
    Maintenance,
}

/// Tool metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub author: String,
    pub license: String,
    pub documentation_url: Option<String>,
    pub source_url: Option<String>,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub auto_discover: bool,
    pub scan_interval: std::time::Duration,
    pub scan_directories: Vec<String>,
    pub include_experimental: bool,
    pub include_deprecated: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            auto_discover: true,
            scan_interval: std::time::Duration::from_secs(3600),
            scan_directories: vec![
                "./tools".to_string(),
                "./plugins".to_string(),
            ],
            include_experimental: false,
            include_deprecated: false,
        }
    }
}

/// Tool provider trait
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;
    
    /// Discover tools
    async fn discover(&self) -> Result<Vec<DiscoveredTool>>;
    
    /// Check if provider is available
    fn is_available(&self) -> bool;
}

impl ToolDiscoveryEngine {
    /// Create a new discovery engine
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            categories: Arc::new(RwLock::new(HashMap::new())),
            providers: Arc::new(RwLock::new(Vec::new())),
            config: DiscoveryConfig::default(),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Initialize with tool manager
    pub async fn initialize(&mut self, tool_manager: Arc<IntelligentToolManager>) -> Result<()> {
        // Register built-in provider
        self.register_provider(Box::new(BuiltinToolProvider::new(tool_manager))).await?;
        
        // Run initial discovery
        if self.config.auto_discover {
            self.discover_all().await?;
        }
        
        Ok(())
    }
    
    /// Register a tool provider
    pub async fn register_provider(&self, provider: Box<dyn ToolProvider>) -> Result<()> {
        self.providers.write().await.push(provider);
        Ok(())
    }
    
    /// Discover all tools
    pub async fn discover_all(&self) -> Result<usize> {
        info!("Starting tool discovery");
        
        let providers = self.providers.read().await;
        let mut total_discovered = 0;
        
        for provider in providers.iter() {
            if !provider.is_available() {
                warn!("Provider {} is not available", provider.name());
                continue;
            }
            
            match provider.discover().await {
                Ok(tools) => {
                    let count = tools.len();
                    for tool in tools {
                        self.register_tool(tool).await?;
                    }
                    total_discovered += count;
                    info!("Discovered {} tools from {}", count, provider.name());
                }
                Err(e) => {
                    warn!("Failed to discover tools from {}: {:?}", provider.name(), e);
                }
            }
        }
        
        info!("Tool discovery complete: {} tools discovered", total_discovered);
        Ok(total_discovered)
    }
    
    /// Register a discovered tool
    async fn register_tool(&self, tool: DiscoveredTool) -> Result<()> {
        let tool_id = tool.id.clone();
        let category = tool.category;
        
        // Add to category index
        self.categories
            .write()
            .await
            .entry(category)
            .or_insert_with(HashSet::new)
            .insert(tool_id.clone());
        
        // Add to tools
        self.tools.write().await.insert(tool_id.clone(), tool);
        
        Ok(())
    }
    
    /// Get all tools
    pub async fn get_all_tools(&self) -> Vec<DiscoveredTool> {
        self.tools.read().await.values().cloned().collect()
    }
    
    /// Get tools by category
    pub async fn get_tools_by_category(&self, category: ToolCategory) -> Vec<DiscoveredTool> {
        let tools = self.tools.read().await;
        let categories = self.categories.read().await;
        
        if let Some(tool_ids) = categories.get(&category) {
            tool_ids
                .iter()
                .filter_map(|id| tools.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Search tools
    pub async fn search_tools(&self, query: &str) -> Vec<DiscoveredTool> {
        let query_lower = query.to_lowercase();
        let tools = self.tools.read().await;
        
        tools
            .values()
            .filter(|tool| {
                tool.name.to_lowercase().contains(&query_lower) ||
                tool.description.to_lowercase().contains(&query_lower) ||
                tool.metadata.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect()
    }
    
    /// Get tool by ID
    pub async fn get_tool(&self, tool_id: &str) -> Option<DiscoveredTool> {
        self.tools.read().await.get(tool_id).cloned()
    }
    
    /// Check tool availability
    pub async fn check_availability(&self, tool_id: &str) -> Result<bool> {
        if let Some(tool) = self.get_tool(tool_id).await {
            Ok(tool.status == ToolStatus::Available)
        } else {
            Ok(false)
        }
    }
    
    /// Get tool requirements
    pub async fn get_requirements(&self, tool_id: &str) -> Option<ToolRequirements> {
        self.get_tool(tool_id).await.map(|tool| tool.requirements)
    }
    
    /// Validate tool parameters
    pub fn validate_parameters(
        &self,
        tool: &DiscoveredTool,
        params: &serde_json::Value,
    ) -> Result<()> {
        let params_obj = params.as_object()
            .ok_or_else(|| anyhow::anyhow!("Tool parameters must be a JSON object, received: {:?}", params))?;
        
        // Check required parameters
        for param in &tool.parameters {
            if param.required && !params_obj.contains_key(&param.name) {
                return Err(anyhow::anyhow!("Missing required parameter '{}' for tool '{}'", param.name, tool.name));
            }
            
            // Validate parameter type and constraints
            if let Some(value) = params_obj.get(&param.name) {
                self.validate_parameter_value(param, value)?;
            }
        }
        
        Ok(())
    }
    
    /// Validate individual parameter value
    fn validate_parameter_value(
        &self,
        param: &ToolParameter,
        value: &serde_json::Value,
    ) -> Result<()> {
        // Type validation
        match &param.param_type {
            ParameterType::String => {
                if !value.is_string() {
                    return Err(anyhow::anyhow!("Parameter '{}' must be a string, received: {:?}", param.name, value));
                }
            }
            ParameterType::Number => {
                if !value.is_number() {
                    return Err(anyhow::anyhow!("Parameter '{}' must be a number, received: {:?}", param.name, value));
                }
            }
            ParameterType::Boolean => {
                if !value.is_boolean() {
                    return Err(anyhow::anyhow!("Parameter '{}' must be a boolean, received: {:?}", param.name, value));
                }
            }
            ParameterType::Array => {
                if !value.is_array() {
                    return Err(anyhow::anyhow!("Parameter '{}' must be an array, received: {:?}", param.name, value));
                }
            }
            ParameterType::Object => {
                if !value.is_object() {
                    return Err(anyhow::anyhow!("Parameter '{}' must be an object, received: {:?}", param.name, value));
                }
            }
            ParameterType::Enum(values) => {
                if let Some(str_value) = value.as_str() {
                    if !values.contains(&str_value.to_string()) {
                        return Err(anyhow::anyhow!(
                            "Parameter {} must be one of: {:?}",
                            param.name,
                            values
                        ));
                    }
                }
            }
            _ => {}
        }
        
        // Additional validation
        if let Some(validation) = &param.validation {
            if let Some(str_value) = value.as_str() {
                if let Some(min_len) = validation.min_length {
                    if str_value.len() < min_len {
                        return Err(anyhow::anyhow!(
                            "Parameter {} must be at least {} characters",
                            param.name,
                            min_len
                        ));
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Built-in tool provider
struct BuiltinToolProvider {
    tool_manager: Arc<IntelligentToolManager>,
}

impl BuiltinToolProvider {
    fn new(tool_manager: Arc<IntelligentToolManager>) -> Self {
        Self { tool_manager }
    }
}

#[async_trait::async_trait]
impl ToolProvider for BuiltinToolProvider {
    fn name(&self) -> &str {
        "builtin"
    }
    
    async fn discover(&self) -> Result<Vec<DiscoveredTool>> {
        let mut tools = Vec::new();
        
        // File system tool
        tools.push(DiscoveredTool {
            id: "file_system".to_string(),
            name: "File System".to_string(),
            description: "Read, write, and manipulate files and directories".to_string(),
            category: ToolCategory::FileSystem,
            provider: "builtin".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec![
                ToolCapability::Read,
                ToolCapability::Write,
                ToolCapability::Execute,
            ],
            requirements: ToolRequirements {
                authentication: None,
                permissions: vec!["file_access".to_string()],
                dependencies: vec![],
                system_requirements: SystemRequirements {
                    min_memory_mb: None,
                    min_cpu_cores: None,
                    required_os: None,
                    required_packages: vec![],
                },
                rate_limits: None,
            },
            parameters: vec![
                ToolParameter {
                    name: "path".to_string(),
                    description: "File or directory path".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default_value: None,
                    validation: None,
                },
                ToolParameter {
                    name: "operation".to_string(),
                    description: "Operation to perform".to_string(),
                    param_type: ParameterType::Enum(vec![
                        "read".to_string(),
                        "write".to_string(),
                        "delete".to_string(),
                        "create".to_string(),
                    ]),
                    required: true,
                    default_value: None,
                    validation: None,
                },
            ],
            examples: vec![],
            status: ToolStatus::Available,
            metadata: ToolMetadata {
                author: "Loki AI".to_string(),
                license: "MIT".to_string(),
                documentation_url: None,
                source_url: None,
                tags: vec!["file".to_string(), "io".to_string()],
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            },
        });
        
        // Web search tool
        tools.push(DiscoveredTool {
            id: "web_search".to_string(),
            name: "Web Search".to_string(),
            description: "Search the web for information".to_string(),
            category: ToolCategory::Research,
            provider: "builtin".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec![
                ToolCapability::Query,
                ToolCapability::Analyze,
            ],
            requirements: ToolRequirements {
                authentication: Some(AuthRequirement::ApiKey("SEARCH_API_KEY".to_string())),
                permissions: vec!["internet_access".to_string()],
                dependencies: vec![],
                system_requirements: SystemRequirements {
                    min_memory_mb: None,
                    min_cpu_cores: None,
                    required_os: None,
                    required_packages: vec![],
                },
                rate_limits: Some(RateLimits {
                    requests_per_second: Some(1.0),
                    requests_per_minute: Some(60),
                    requests_per_hour: None,
                    concurrent_requests: Some(5),
                }),
            },
            parameters: vec![
                ToolParameter {
                    name: "query".to_string(),
                    description: "Search query".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default_value: None,
                    validation: Some(ParameterValidation {
                        min_length: Some(1),
                        max_length: Some(500),
                        pattern: None,
                        min_value: None,
                        max_value: None,
                        allowed_values: None,
                    }),
                },
            ],
            examples: vec![],
            status: ToolStatus::Available,
            metadata: ToolMetadata {
                author: "Loki AI".to_string(),
                license: "MIT".to_string(),
                documentation_url: None,
                source_url: None,
                tags: vec!["search".to_string(), "web".to_string(), "research".to_string()],
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            },
        });
        
        Ok(tools)
    }
    
    fn is_available(&self) -> bool {
        true
    }
}
