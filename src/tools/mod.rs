pub mod arxiv;
pub mod code_analysis;
pub mod doc_crawler;
pub mod file_system;
pub mod github;
pub mod graphql;
pub mod mcp_client;
pub mod web_search;
pub mod websocket;
pub mod computer_use;
pub mod vision_system;

// Enhanced tool capabilities
pub mod calendar;
pub mod discord;
pub mod email;
pub mod slack;
pub mod task_management;

// Creative media capabilities
pub mod blender_integration;
pub mod creative_generators;
pub mod creative_media;

// Intelligent tool management
pub mod emergent_types;
pub mod intelligent_manager;
pub mod metrics_collector;

// Re-export ToolConfig for configuration
pub use intelligent_manager::ToolConfig;

/// High-Performance Parallel Tool Execution System
pub mod parallel_execution;

/// Vector Database Memory Integration
pub mod vector_memory;

/// Database Cognitive Tool
pub mod database_cognitive;

/// Python Executor Tool
pub mod python_executor;

/// Universal API Interface Tool
pub mod api_connector;

/// Autonomous Browser Tool
pub mod autonomous_browser;

pub use blender_integration::{
    BlenderConfig,
    BlenderEvent,
    BlenderIntegration,
    BlenderProject,
    BlenderStats,
    ContentType,
    ParametricModel,
    PrimitiveShape,
    Transform,
};
pub use calendar::{
    CalendarConfig,
    CalendarEvent,
    CalendarManager,
    CalendarManagerEvent,
    CalendarStats,
    ScheduleAnalysis,
};
pub use code_analysis::{AnalysisResult, CodeAnalyzer};
pub use creative_generators::{ImageGenerator, VideoGenerator, VoiceGenerator};
// Creative media exports
pub use creative_media::{
    AudioGenre,
    CreativeEvent,
    CreativeMediaConfig,
    CreativeMediaManager,
    GeneratedMedia,
    ImageStyle,
    MediaMetadata,
    MediaStats,
    MediaType,
    VideoStyle,
    VoiceEmotion,
    VoiceStyle,
};
pub use discord::{
    CommunityContext,
    DiscordBot,
    DiscordConfig,
    DiscordEvent,
    DiscordMessage,
    DiscordStats,
};
pub use email::{
    Contact,
    EmailConfig,
    EmailEvent,
    EmailMessage,
    EmailProcessor,
    EmailStats,
    EmailThread,
};
pub use file_system::{
    FileOperation,
    FileOperationType,
    FileSystemConfig,
    FileSystemResult,
    FileSystemTool,
    SafetyConfig,
};
pub use github::{GitHubClient, GitHubConfig};
// Intelligent tool management exports
pub use intelligent_manager::{
    ArchetypalToolPattern,
    ContextualToolSelection,
    IntelligentToolManager,
    MemoryIntegration,
    ResultType,
    ToolIntegrationEvent,
    ToolManagerConfig,
    ToolRequest,
    ToolResult,
    ToolSelection,
    ToolStatus,
    ToolUsagePattern,
};
// Metrics collector exports
pub use metrics_collector::{
    ToolMetrics,
    ToolMetricsCollector,
    ActiveToolSession,
    SessionStatus,
    SystemMetrics,
    TuiToolMetrics,
};
pub use mcp_client::{
    McpCapabilities,
    McpClient,
    McpClientConfig,
    McpServer,
    McpServerInfo,
    McpTool,
    McpToolCall,
    McpToolResponse,
};
// Enhanced tools exports
pub use slack::{SlackClient, SlackConfig, SlackEvent, SlackMessage, SlackStats};
pub use task_management::{Task, TaskConfig, TaskEvent, TaskManager, TaskStats, WorkloadAnalysis};

/// Create a pre-configured MCP client with standard configuration
///
/// This function creates an MCP client and automatically loads configuration
/// from standard locations in this order:
/// 1. ~/.eigencode/mcp-servers/mcp-config-multi.json (Loki standard)
/// 2. ~/.cursor/mcp.json (Cursor IDE)
/// 3. ~/Library/Application Support/Claude/claude_desktopconfig.json (Claude
///    Desktop)
///
/// # Returns
/// A configured MCP client ready for use, or an error if initialization fails
///
/// # Example
/// ```rust
/// let mcp_client = create_standard_mcp_client().await?;
/// let dirs = mcp_client.list_directory("/Users/thermo/Documents/GitHub/").await?;
/// ```
pub async fn create_standard_mcp_client() -> anyhow::Result<McpClient> {
    let config = McpClientConfig::default();
    McpClient::new_with_standardconfig(config).await
}

/// Convenience alias for create_standard_mcp_client
pub async fn create_mcp_client() -> anyhow::Result<McpClient> {
    create_standard_mcp_client().await
}

pub use emergent_types::*;
pub use vector_memory::{
    VectorMemoryTool,
    VectorMemoryConfig,
    VectorProvider,
    VectorSearchResult,
    BatchOperationResult,
    VectorMemoryMetrics,
};
pub use database_cognitive::{
    DatabaseCognitiveTool,
    DatabaseCognitiveConfig,
    DatabaseQueryResult,
    QueryContext,
    QueryOperationType,
    QueryOptimization,
    QueryMetrics,
};
pub use python_executor::{
    PythonExecutorTool,
    PythonExecutorConfig,
    PythonExecutionContext,
    PythonExecutionResult,
    PythonExecutionMode,
    SecurityLevel,
    PythonExecutionMetrics,
};
/// Tool Registry - Lists all available tools in the system
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub icon: String,
    pub available: bool,
}

/// Get all available tools in the system
pub fn get_tool_registry() -> Vec<ToolInfo> {
    vec![
        // Cognitive Tools
        ToolInfo {
            id: "consciousness_engine".to_string(),
            name: "Consciousness Engine".to_string(),
            category: "🧠 Cognitive".to_string(),
            description: "Core AI decision making and self-awareness system".to_string(),
            icon: "🧠".to_string(),
            available: true,
        },
        ToolInfo {
            id: "memory_manager".to_string(),
            name: "Memory Manager".to_string(),
            category: "🧠 Cognitive".to_string(),
            description: "Hierarchical knowledge storage with vector embeddings".to_string(),
            icon: "💭".to_string(),
            available: true,
        },
        ToolInfo {
            id: "vector_memory".to_string(),
            name: "Vector Memory".to_string(),
            category: "🧠 Cognitive".to_string(),
            description: "High-performance vector database for semantic search".to_string(),
            icon: "🔮".to_string(),
            available: true,
        },
        
        // Development Tools
        ToolInfo {
            id: "github".to_string(),
            name: "GitHub Integration".to_string(),
            category: "💻 Development".to_string(),
            description: "Repository management, issue tracking, and PR automation".to_string(),
            icon: "🐙".to_string(),
            available: true,
        },
        ToolInfo {
            id: "code_analysis".to_string(),
            name: "Code Analyzer".to_string(),
            category: "💻 Development".to_string(),
            description: "Static code analysis, complexity metrics, and quality checks".to_string(),
            icon: "🔍".to_string(),
            available: true,
        },
        ToolInfo {
            id: "python_executor".to_string(),
            name: "Python Executor".to_string(),
            category: "💻 Development".to_string(),
            description: "Secure Python code execution with sandboxing".to_string(),
            icon: "🐍".to_string(),
            available: true,
        },
        
        // Creative Tools
        ToolInfo {
            id: "computer_use".to_string(),
            name: "Computer Use".to_string(),
            category: "🎨 Creative".to_string(),
            description: "Screen automation and AI-driven UI interaction".to_string(),
            icon: "🖥️".to_string(),
            available: true,
        },
        ToolInfo {
            id: "creative_media".to_string(),
            name: "Creative Media Manager".to_string(),
            category: "🎨 Creative".to_string(),
            description: "AI-powered image, video, and voice generation".to_string(),
            icon: "🎨".to_string(),
            available: true,
        },
        ToolInfo {
            id: "blender_integration".to_string(),
            name: "Blender Integration".to_string(),
            category: "🎨 Creative".to_string(),
            description: "3D modeling, animation, and procedural content generation".to_string(),
            icon: "🏗️".to_string(),
            available: true,
        },
        ToolInfo {
            id: "vision_system".to_string(),
            name: "Vision System".to_string(),
            category: "🎨 Creative".to_string(),
            description: "Advanced computer vision and image analysis".to_string(),
            icon: "👁️".to_string(),
            available: true,
        },
        
        // Communication Tools
        ToolInfo {
            id: "slack".to_string(),
            name: "Slack Integration".to_string(),
            category: "💬 Communication".to_string(),
            description: "Team messaging automation and bot capabilities".to_string(),
            icon: "💬".to_string(),
            available: true,
        },
        ToolInfo {
            id: "discord".to_string(),
            name: "Discord Bot".to_string(),
            category: "💬 Communication".to_string(),
            description: "Community management and Discord automation".to_string(),
            icon: "🎮".to_string(),
            available: true,
        },
        ToolInfo {
            id: "email".to_string(),
            name: "Email Processor".to_string(),
            category: "💬 Communication".to_string(),
            description: "Email management, threading, and automated responses".to_string(),
            icon: "📧".to_string(),
            available: true,
        },
        
        // Research Tools
        ToolInfo {
            id: "web_search".to_string(),
            name: "Web Search".to_string(),
            category: "🔬 Research".to_string(),
            description: "Real-time web search with multiple search engines".to_string(),
            icon: "🔍".to_string(),
            available: true,
        },
        ToolInfo {
            id: "arxiv".to_string(),
            name: "ArXiv Research".to_string(),
            category: "🔬 Research".to_string(),
            description: "Academic paper search and analysis from ArXiv".to_string(),
            icon: "📚".to_string(),
            available: true,
        },
        ToolInfo {
            id: "doc_crawler".to_string(),
            name: "Documentation Crawler".to_string(),
            category: "🔬 Research".to_string(),
            description: "Intelligent documentation parsing and extraction".to_string(),
            icon: "📖".to_string(),
            available: true,
        },
        
        // System Tools
        ToolInfo {
            id: "file_system".to_string(),
            name: "File System".to_string(),
            category: "⚙️ System".to_string(),
            description: "Safe file operations with sandboxing and validation".to_string(),
            icon: "📁".to_string(),
            available: true,
        },
        ToolInfo {
            id: "database_cognitive".to_string(),
            name: "Database Cognitive".to_string(),
            category: "⚙️ System".to_string(),
            description: "Intelligent database queries with natural language".to_string(),
            icon: "🗄️".to_string(),
            available: true,
        },
        ToolInfo {
            id: "autonomous_browser".to_string(),
            name: "Autonomous Browser".to_string(),
            category: "⚙️ System".to_string(),
            description: "Headless browser automation for web scraping".to_string(),
            icon: "🌐".to_string(),
            available: true,
        },
        
        // Integration Tools
        ToolInfo {
            id: "api_connector".to_string(),
            name: "API Connector".to_string(),
            category: "🔌 Integration".to_string(),
            description: "Universal REST/GraphQL API integration framework".to_string(),
            icon: "🔌".to_string(),
            available: true,
        },
        ToolInfo {
            id: "graphql".to_string(),
            name: "GraphQL Client".to_string(),
            category: "🔌 Integration".to_string(),
            description: "GraphQL query execution and schema introspection".to_string(),
            icon: "📊".to_string(),
            available: true,
        },
        ToolInfo {
            id: "websocket".to_string(),
            name: "WebSocket Manager".to_string(),
            category: "🔌 Integration".to_string(),
            description: "Real-time WebSocket connections and event handling".to_string(),
            icon: "🔗".to_string(),
            available: true,
        },
        ToolInfo {
            id: "mcp_client".to_string(),
            name: "MCP Client".to_string(),
            category: "🔌 Integration".to_string(),
            description: "Model Context Protocol server management".to_string(),
            icon: "🤖".to_string(),
            available: true,
        },
        
        // Productivity Tools
        ToolInfo {
            id: "calendar".to_string(),
            name: "Calendar Manager".to_string(),
            category: "📅 Productivity".to_string(),
            description: "Schedule management and calendar integration".to_string(),
            icon: "📅".to_string(),
            available: true,
        },
        ToolInfo {
            id: "task_management".to_string(),
            name: "Task Manager".to_string(),
            category: "📅 Productivity".to_string(),
            description: "Project planning and task tracking system".to_string(),
            icon: "✅".to_string(),
            available: true,
        },
    ]
}

/// Get all available tools (alias for get_tool_registry for compatibility)
pub fn get_available_tools() -> Vec<ToolInfo> {
    get_tool_registry()
}

pub use api_connector::{
    ApiConnectorTool,
    ApiConnectorConfig,
    ApiEndpointConfig,
    ApiRequestContext,
    ApiResponse,
    ApiType,
    AuthConfig,
    AuthType,
    HttpMethod,
    ApiMetrics,
};
pub use autonomous_browser::{
    AutonomousBrowserTool,
    AutonomousBrowserConfig,
    BrowserTask,
    BrowserTaskType,
    BrowserTaskResult,
    BrowserSession,
    BrowserMetrics,
    BrowserType,
};
