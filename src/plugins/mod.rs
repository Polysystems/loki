use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::error;

pub mod loader;
pub mod manager;
pub mod api;
pub mod sandbox;
pub mod registry;
pub mod marketplace;
pub mod wasm_engine;

pub use loader::PluginLoader;
pub use manager::PluginManager;
pub use api::{PluginApi, PluginContext};
pub use sandbox::PluginSandbox;
pub use registry::PluginRegistry;
pub use marketplace::{PluginMarketplace, MarketplaceApi, SearchFilters, PluginCategory, SortBy};
pub use wasm_engine::{WasmEngine, WasmPluginInstance, WasmPluginConfig};

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Unique plugin identifier
    pub id: String,

    /// Plugin name
    pub name: String,

    /// Plugin version
    pub version: String,

    /// Plugin author
    pub author: String,

    /// Plugin description
    pub description: String,

    /// Required Loki version
    pub loki_version: String,

    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,

    /// Plugin capabilities/permissions
    pub capabilities: Vec<PluginCapability>,

    /// Plugin homepage
    pub homepage: Option<String>,

    /// Plugin repository
    pub repository: Option<String>,

    /// Plugin license
    pub license: Option<String>,
}

/// Plugin dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    pub id: String,
    pub version: String,
    pub optional: bool,
}

/// Plugin capability/permission
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PluginCapability {
    /// Read access to memory
    MemoryRead,

    /// Write access to memory
    MemoryWrite,

    /// Access to consciousness stream
    ConsciousnessAccess,

    /// Ability to propose code changes
    CodeModification,

    /// Access to social media
    SocialMedia,

    /// Network access
    NetworkAccess,

    /// File system read
    FileSystemRead,

    /// File system write
    FileSystemWrite,

    /// Access to cognitive subsystems
    CognitiveAccess,

    /// Custom capability
    Custom(String),
}

/// Plugin lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is loaded but not initialized
    Loaded,

    /// Plugin is initializing
    Initializing,

    /// Plugin is active and running
    Active,

    /// Plugin is suspended
    Suspended,

    /// Plugin is shutting down
    Stopping,

    /// Plugin has stopped
    Stopped,

    /// Plugin encountered an error
    Error,
}

/// Plugin trait that all plugins must implement
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize the plugin
    async fn initialize(&mut self, context: PluginContext) -> Result<()>;

    /// Start the plugin
    async fn start(&mut self) -> Result<()>;

    /// Stop the plugin
    async fn stop(&mut self) -> Result<()>;

    /// Handle an event
    async fn handle_event(&mut self, event: PluginEvent) -> Result<()>;

    /// Get plugin state
    fn state(&self) -> PluginState;

    /// Health check
    async fn health_check(&self) -> Result<HealthStatus>;
}

/// Plugin events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginEvent {
    /// System event
    System(SystemEvent),

    /// Cognitive event
    Cognitive(CognitiveEvent),

    /// Memory event
    Memory(MemoryEvent),

    /// Social event
    Social(SocialEvent),

    /// Custom event
    Custom(String, serde_json::Value),
}

/// System events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEvent {
    Startup,
    Shutdown,
    ConfigChanged(String),
    ResourceWarning(ResourceType, f32),
}

/// Cognitive events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveEvent {
    ThoughtGenerated(String),
    DecisionMade(String),
    GoalAchieved(String),
    EmotionChanged(String, f32),
}

/// Memory events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryEvent {
    MemoryStored(String),
    MemoryRetrieved(String),
    MemoryConsolidated,
    MemoryDecayed,
}

/// Social events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SocialEvent {
    MessageReceived(String, String),
    PostCreated(String),
    FollowerGained(String),
    MentionReceived(String, String),
}

/// Resource type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    Memory,
    Cpu,
    Disk,
    Network,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub message: Option<String>,
    pub metrics: HashMap<String, f64>,
}

/// Plugin error types
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("Plugin not found: {0}")]
    NotFound(String),

    #[error("Plugin already loaded: {0}")]
    AlreadyLoaded(String),

    #[error("Plugin initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Plugin dependency missing: {0}")]
    DependencyMissing(String),

    #[error("Plugin capability denied: {0:?}")]
    CapabilityDenied(PluginCapability),

    #[error("Plugin version incompatible: {0}")]
    VersionIncompatible(String),

    #[error("Plugin sandbox violation: {0}")]
    SandboxViolation(String),

    #[error("Plugin communication error: {0}")]
    CommunicationError(String),
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin directory
    pub plugin_dir: PathBuf,

    /// Enable plugin sandboxing
    pub enable_sandbox: bool,

    /// Maximum memory per plugin (MB)
    pub max_memory_mb: usize,

    /// Maximum CPU percentage per plugin
    pub max_cpu_percent: f32,

    /// Plugin timeout (seconds)
    pub timeout_seconds: u64,

    /// Auto-load plugins on startup
    pub auto_load: bool,

    /// Plugin registry URL
    pub registry_url: Option<String>,

    /// Allowed capabilities by default
    pub default_capabilities: Vec<PluginCapability>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            plugin_dir: PathBuf::from("plugins"),
            enable_sandbox: true,
            max_memory_mb: 512,
            max_cpu_percent: 25.0,
            timeout_seconds: 30,
            auto_load: true,
            registry_url: Some("https://plugins.loki.ai".to_string()),
            default_capabilities: vec![
                PluginCapability::MemoryRead,
                PluginCapability::NetworkAccess,
            ],
        }
    }
}

impl std::fmt::Display for PluginCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PluginCapability::MemoryRead => "MemoryRead",
            PluginCapability::MemoryWrite => "MemoryWrite",
            PluginCapability::ConsciousnessAccess => "ConsciousnessAccess",
            PluginCapability::CodeModification => "CodeModification",
            PluginCapability::SocialMedia => "SocialMedia",
            PluginCapability::NetworkAccess => "NetworkAccess",
            PluginCapability::FileSystemRead => "FileSystemRead",
            PluginCapability::FileSystemWrite => "FileSystemWrite",
            PluginCapability::CognitiveAccess => "CognitiveAccess",
            PluginCapability::Custom(s) => s,
        };
        write!(f, "{}", s)
    }
}

impl PartialEq<str> for PluginCapability {
    fn eq(&self, other: &str) -> bool {
        match self {
            PluginCapability::MemoryRead => other == "MemoryRead",
            PluginCapability::MemoryWrite => other == "MemoryWrite",
            PluginCapability::ConsciousnessAccess => other == "ConsciousnessAccess",
            PluginCapability::CodeModification => other == "CodeModification",
            PluginCapability::SocialMedia => other == "SocialMedia",
            PluginCapability::NetworkAccess => other == "NetworkAccess",
            PluginCapability::FileSystemRead => other == "FileSystemRead",
            PluginCapability::FileSystemWrite => other == "FileSystemWrite",
            PluginCapability::CognitiveAccess => other == "CognitiveAccess",
            PluginCapability::Custom(s) => s == other,
        }
    }
}

impl PartialEq<&str> for PluginCapability {
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

/// Plugin statistics
#[derive(Debug, Clone, Default)]
pub struct PluginStats {
    pub total_loaded: usize,
    pub active_plugins: usize,
    pub events_processed: u64,
    pub errors_encountered: u64,
    pub total_cpu_time_ms: u64,
    pub total_memory_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_metadata() {
        let metadata = PluginMetadata {
            id: "test-plugin".to_string(),
            name: "Test Plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "A test plugin".to_string(),
            loki_version: "0.2.0".to_string(),
            dependencies: vec![],
            capabilities: vec![PluginCapability::MemoryRead],
            homepage: None,
            repository: None,
            license: Some("MIT".to_string()),
        };

        assert_eq!(metadata.id, "test-plugin");
        assert_eq!(metadata.capabilities.len(), 1);
    }

    #[test]
    fn test_pluginconfig_default() {
        let config = PluginConfig::default();
        assert!(config.enable_sandbox);
        assert_eq!(config.max_memory_mb, 512);
        assert_eq!(config.default_capabilities.len(), 2);
    }
}
