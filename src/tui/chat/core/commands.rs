//! Chat Command System for Loki TUI
//! 
//! This module provides a comprehensive command parsing and execution system
//! for the chat interface, enabling direct tool execution and workflow management.

use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Command registry for managing available commands
#[derive(Debug, Clone)]
pub struct CommandRegistry {
    commands: HashMap<String, CommandDefinition>,
}

/// Definition of a chat command
#[derive(Debug, Clone)]
pub struct CommandDefinition {
    /// Command name (without slash)
    pub name: String,
    /// Command aliases
    pub aliases: Vec<String>,
    /// Brief description
    pub description: String,
    /// Detailed help text
    pub help_text: String,
    /// Command category
    pub category: CommandCategory,
    /// Required permissions/features
    pub required_features: Vec<String>,
    /// Parameter definitions
    pub parameters: Vec<ParameterDefinition>,
    /// Example usages
    pub examples: Vec<String>,
}

/// Command categories for organization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandCategory {
    Tools,
    Workflow,
    Task,
    System,
    Help,
    Development,
    Analysis,
    Chat,
}

/// Parameter definition for commands
#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if not provided
    pub default: Option<String>,
    /// Description of the parameter
    pub description: String,
    /// Valid values (for enum types)
    pub valid_values: Option<Vec<String>>,
}

/// Types of command parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Json,
    FilePath,
    Enum,
    List,
}

/// Parsed command ready for execution
#[derive(Debug, Clone)]
pub struct ParsedCommand {
    /// The command name
    pub command: String,
    /// Parsed arguments
    pub args: HashMap<String, Value>,
    /// Raw input string
    pub raw_input: String,
    /// Whether this is a help request
    pub is_help: bool,
}

/// Result of command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    /// Whether the command succeeded
    pub success: bool,
    /// Result content (can be text, JSON, etc.)
    pub content: Value,
    /// Display format hint
    pub format: ResultFormat,
    /// Any warnings or notes
    pub warnings: Vec<String>,
    /// Follow-up suggestions
    pub suggestions: Vec<String>,
    /// Execution metadata
    pub metadata: HashMap<String, Value>,
}

/// Format hint for displaying results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultFormat {
    Text,
    Code { language: String },
    Json,
    Table,
    List,
    Progress,
    Error,
    Mixed,
}

impl CommandRegistry {
    /// Create a new command registry with built-in commands
    pub fn new() -> Self {
        let mut registry = Self {
            commands: HashMap::new(),
        };
        
        // Register built-in commands
        registry.register_builtin_commands();
        registry
    }
    
    /// Register all built-in commands
    fn register_builtin_commands(&mut self) {
        // /execute command
        self.register(CommandDefinition {
            name: "execute".to_string(),
            aliases: vec!["exec".to_string()],
            description: "Execute a tool directly".to_string(),
            help_text: "Execute a specific tool with provided arguments. Use /tools list to see available tools.".to_string(),
            category: CommandCategory::Tools,
            required_features: vec!["tools".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "tool".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Tool name to execute".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "args".to_string(),
                    param_type: ParameterType::Json,
                    required: false,
                    default: Some("{}".to_string()),
                    description: "Tool arguments as JSON".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/execute web_search {\"query\": \"Rust async programming\"}".to_string(),
                "/exec github {\"action\": \"list_repos\", \"user\": \"rust-lang\"}".to_string(),
            ],
        });
        
        // /workflow command
        self.register(CommandDefinition {
            name: "workflow".to_string(),
            aliases: vec!["wf".to_string()],
            description: "Execute a predefined workflow".to_string(),
            help_text: "Run a multi-step workflow. Use /workflow list to see available workflows.".to_string(),
            category: CommandCategory::Workflow,
            required_features: vec!["workflows".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "name".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Workflow name or 'list' to show available".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "params".to_string(),
                    param_type: ParameterType::Json,
                    required: false,
                    default: Some("{}".to_string()),
                    description: "Workflow parameters".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/workflow code_review {\"file\": \"src/main.rs\"}".to_string(),
                "/workflow list".to_string(),
            ],
        });
        
        // /task command
        self.register(CommandDefinition {
            name: "task".to_string(),
            aliases: vec!["t".to_string()],
            description: "Create and execute a task".to_string(),
            help_text: "Create a task from natural language description and execute it.".to_string(),
            category: CommandCategory::Task,
            required_features: vec!["task_management".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: Some("create".to_string()),
                    description: "Task action".to_string(),
                    valid_values: Some(vec![
                        "create".to_string(),
                        "list".to_string(),
                        "status".to_string(),
                        "cancel".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "description".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Task description".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/task create \"Analyze the performance of the memory system\"".to_string(),
                "/task list".to_string(),
                "/task status 123".to_string(),
            ],
        });
        
        // /tools command
        self.register(CommandDefinition {
            name: "tools".to_string(),
            aliases: vec!["tool".to_string()],
            description: "Manage and list available tools".to_string(),
            help_text: "Show available tools, their capabilities, and current status.".to_string(),
            category: CommandCategory::Tools,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("list".to_string()),
                    description: "Action to perform".to_string(),
                    valid_values: Some(vec![
                        "list".to_string(),
                        "info".to_string(),
                        "status".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "tool_name".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Specific tool name for info".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/tools list".to_string(),
                "/tools info web_search".to_string(),
                "/tools status".to_string(),
            ],
        });
        
        // /context command
        self.register(CommandDefinition {
            name: "context".to_string(),
            aliases: vec!["ctx".to_string()],
            description: "Manage conversation context".to_string(),
            help_text: "Save, load, or clear conversation context for better continuity.".to_string(),
            category: CommandCategory::System,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: Some("show".to_string()),
                    description: "Context action".to_string(),
                    valid_values: Some(vec![
                        "show".to_string(),
                        "save".to_string(),
                        "load".to_string(),
                        "clear".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "name".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Context name for save/load".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/context show".to_string(),
                "/context save project_analysis".to_string(),
                "/context load project_analysis".to_string(),
            ],
        });
        
        // /analyze command
        self.register(CommandDefinition {
            name: "analyze".to_string(),
            aliases: vec!["a".to_string()],
            description: "Analyze code, data, or system state".to_string(),
            help_text: "Perform various types of analysis using AI and tools.".to_string(),
            category: CommandCategory::Analysis,
            required_features: vec!["analysis".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "type".to_string(),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: Some("code".to_string()),
                    description: "Type of analysis".to_string(),
                    valid_values: Some(vec![
                        "code".to_string(),
                        "performance".to_string(),
                        "security".to_string(),
                        "data".to_string(),
                        "system".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "target".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Analysis target (file, directory, or description)".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/analyze code src/".to_string(),
                "/analyze performance \"memory allocation patterns\"".to_string(),
                "/analyze security Cargo.toml".to_string(),
            ],
        });
        
        // /run command for code execution
        self.register(CommandDefinition {
            name: "run".to_string(),
            aliases: vec!["r".to_string()],
            description: "Execute code in various languages".to_string(),
            help_text: "Execute code snippets in Python, JavaScript, Rust, Bash, or auto-detect the language.".to_string(),
            category: CommandCategory::Development,
            required_features: vec!["code_execution".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "language".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("auto".to_string()),
                    description: "Programming language".to_string(),
                    valid_values: Some(vec![
                        "python".to_string(),
                        "py".to_string(),
                        "javascript".to_string(),
                        "js".to_string(),
                        "rust".to_string(),
                        "rs".to_string(),
                        "bash".to_string(),
                        "sh".to_string(),
                        "auto".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "code".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Code to execute".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/run python print('Hello, World!')".to_string(),
                "/run js console.log([1,2,3].map(x => x*2))".to_string(),
                "/run rust fn main() { println!(\"Hello from Rust!\"); }".to_string(),
                "/run bash echo $USER".to_string(),
                "/run print('Auto-detected as Python')".to_string(),
            ],
        });
        
        // /attach command
        self.register(CommandDefinition {
            name: "attach".to_string(),
            aliases: vec!["file".to_string()],
            description: "Attach files to the current message".to_string(),
            help_text: "Attach files to include their content in the conversation. Supports text, code, and various file types.".to_string(),
            category: CommandCategory::System,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "path".to_string(),
                    param_type: ParameterType::String,
                    description: "Path to the file to attach".to_string(),
                    required: true,
                    default: None,
                    valid_values: None,
                },
            ],
            examples: vec![
                "/attach README.md".to_string(),
                "/attach src/main.rs".to_string(),
                "/file ~/Documents/notes.txt".to_string(),
            ],
        });
        
        // /attachments command for managing multiple attachments
        self.register(CommandDefinition {
            name: "attachments".to_string(),
            aliases: vec!["files".to_string()],
            description: "Manage file attachments".to_string(),
            help_text: "List, remove, or clear file attachments for the current message.".to_string(),
            category: CommandCategory::System,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    description: "Action to perform".to_string(),
                    required: false,
                    default: Some("list".to_string()),
                    valid_values: Some(vec![
                        "list".to_string(),
                        "remove".to_string(),
                        "clear".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "index".to_string(),
                    param_type: ParameterType::Integer,
                    description: "Index of attachment to remove (1-based)".to_string(),
                    required: false,
                    default: None,
                    valid_values: None,
                },
            ],
            examples: vec![
                "/attachments".to_string(),
                "/attachments list".to_string(),
                "/attachments remove 1".to_string(),
                "/attachments clear".to_string(),
                "/files".to_string(),
            ],
        });
        
        // /run_attachment command
        self.register(CommandDefinition {
            name: "run_attachment".to_string(),
            aliases: vec!["run_file".to_string(), "exec_attachment".to_string()],
            description: "Execute an attached code file".to_string(),
            help_text: "Execute code from an attached file. Supports Python, JavaScript, Rust, and Bash scripts.".to_string(),
            category: CommandCategory::Development,
            required_features: vec!["code_execution".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "index".to_string(),
                    param_type: ParameterType::Integer,
                    description: "Index of the attachment to run (1-based)".to_string(),
                    required: true,
                    default: None,
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "args".to_string(),
                    param_type: ParameterType::String,
                    description: "Arguments to pass to the script".to_string(),
                    required: false,
                    default: Some("".to_string()),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/run_attachment 1".to_string(),
                "/run_attachment 1 --verbose".to_string(),
                "/run_file 2 input.txt output.txt".to_string(),
            ],
        });
        
        // /view_attachment command
        self.register(CommandDefinition {
            name: "view_attachment".to_string(),
            aliases: vec!["view_file".to_string(), "preview".to_string()],
            description: "View an attached file with syntax highlighting".to_string(),
            help_text: "Display the contents of an attached file with appropriate syntax highlighting.".to_string(),
            category: CommandCategory::System,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "index".to_string(),
                    param_type: ParameterType::Integer,
                    description: "Index of the attachment to view (1-based)".to_string(),
                    required: true,
                    default: None,
                    valid_values: None,
                },
            ],
            examples: vec![
                "/view_attachment 1".to_string(),
                "/view_file 2".to_string(),
                "/preview 1".to_string(),
            ],
        });
        
        // /search_attachments command
        self.register(CommandDefinition {
            name: "search_attachments".to_string(),
            aliases: vec!["search_files".to_string(), "grep_attachments".to_string()],
            description: "Search for content within attached files".to_string(),
            help_text: "Search for text patterns across all attached files. Supports regular expressions and case-insensitive search.".to_string(),
            category: CommandCategory::Analysis,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "pattern".to_string(),
                    param_type: ParameterType::String,
                    description: "Search pattern (supports regex)".to_string(),
                    required: true,
                    default: None,
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "case_sensitive".to_string(),
                    param_type: ParameterType::Boolean,
                    description: "Case sensitive search".to_string(),
                    required: false,
                    default: Some("false".to_string()),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "context_lines".to_string(),
                    param_type: ParameterType::Integer,
                    description: "Number of context lines to show around matches".to_string(),
                    required: false,
                    default: Some("2".to_string()),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/search_attachments \"TODO\"".to_string(),
                "/search_attachments \"function.*test\" true".to_string(),
                "/search_files \"error\" false 5".to_string(),
                "/grep_attachments \"import.*async\"".to_string(),
            ],
        });
        
        // /collab command
        self.register(CommandDefinition {
            name: "collab".to_string(),
            aliases: vec!["collaborate".to_string(), "collaboration".to_string()],
            description: "Manage real-time collaboration sessions".to_string(),
            help_text: "Create, join, or manage collaborative editing sessions. Share your chat with others for real-time collaboration.".to_string(),
            category: CommandCategory::Chat,
            required_features: vec!["collaboration".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    description: "Collaboration action".to_string(),
                    required: true,
                    default: None,
                    valid_values: Some(vec![
                        "create".to_string(),
                        "join".to_string(),
                        "leave".to_string(),
                        "info".to_string(),
                        "participants".to_string(),
                        "toggle".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "session_name_or_id".to_string(),
                    param_type: ParameterType::String,
                    description: "Session name (for create) or ID (for join)".to_string(),
                    required: false,
                    default: None,
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "username".to_string(),
                    param_type: ParameterType::String,
                    description: "Your username for the session".to_string(),
                    required: false,
                    default: Some("Anonymous".to_string()),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/collab create \"Project Discussion\"".to_string(),
                "/collab join abc123 \"John Doe\"".to_string(),
                "/collab participants".to_string(),
                "/collab leave".to_string(),
                "/collab toggle".to_string(),
            ],
        });
        
        // /search command
        self.register(CommandDefinition {
            name: "search".to_string(),
            aliases: vec!["find".to_string(), "s".to_string()],
            description: "Search through chat history with advanced filters".to_string(),
            help_text: "Search messages, code blocks, and metadata across all chats. Supports regex, date ranges, and multiple filters.".to_string(),
            category: CommandCategory::Analysis,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "query".to_string(),
                    param_type: ParameterType::String,
                    description: "Search query (supports regex)".to_string(),
                    required: true,
                    default: None,
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "filters".to_string(),
                    param_type: ParameterType::Json,
                    description: "Search filters as JSON".to_string(),
                    required: false,
                    default: Some("{}".to_string()),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/search \"error\"".to_string(),
                "/search \"async.*await\" {\"type\": \"code\"}".to_string(),
                "/find \"TODO\" {\"date_from\": \"2024-01-01\", \"chat\": \"current\"}".to_string(),
                "/s \"function\" {\"language\": \"rust\", \"model\": \"claude\"}".to_string(),
            ],
        });
        
        // /help command
        self.register(CommandDefinition {
            name: "help".to_string(),
            aliases: vec!["h".to_string(), "?".to_string()],
            description: "Show help information".to_string(),
            help_text: "Display help for commands and features.".to_string(),
            category: CommandCategory::Help,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "command".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Specific command to get help for".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/help".to_string(),
                "/help execute".to_string(),
                "/? workflow".to_string(),
            ],
        });
        
        // /agent command for agent management
        self.register(CommandDefinition {
            name: "agent".to_string(),
            aliases: vec!["a".to_string()],
            description: "Manage AI agents for task execution".to_string(),
            help_text: "Spawn, manage, and monitor AI agents that work on tasks independently. Agents run in parallel and stream their progress to dedicated panels.".to_string(),
            category: CommandCategory::Task,
            required_features: vec!["multi_agent".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: Some("list".to_string()),
                    description: "Agent action to perform".to_string(),
                    valid_values: Some(vec![
                        "spawn".to_string(),
                        "list".to_string(),
                        "status".to_string(),
                        "focus".to_string(),
                        "kill".to_string(),
                        "pause".to_string(),
                        "resume".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "type".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("general".to_string()),
                    description: "Type of agent to spawn".to_string(),
                    valid_values: Some(vec![
                        "general".to_string(),
                        "code".to_string(),
                        "research".to_string(),
                        "analysis".to_string(),
                        "debug".to_string(),
                        "test".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "task".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Task description for the agent".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "id".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Agent ID for status/focus/kill actions".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/agent spawn code \"Refactor the authentication module\"".to_string(),
                "/agent spawn research \"Find best practices for Rust async error handling\"".to_string(),
                "/agent list".to_string(),
                "/agent status agent-123".to_string(),
                "/agent focus agent-123".to_string(),
                "/agent kill agent-123".to_string(),
            ],
        });
        
        // /team command for multi-agent coordination
        self.register(CommandDefinition {
            name: "team".to_string(),
            aliases: vec!["t".to_string()],
            description: "Create and manage agent teams for complex tasks".to_string(),
            help_text: "Coordinate multiple agents working together on complex tasks. Teams can share context and delegate subtasks automatically.".to_string(),
            category: CommandCategory::Task,
            required_features: vec!["multi_agent".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "action".to_string(),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: Some("status".to_string()),
                    description: "Team action to perform".to_string(),
                    valid_values: Some(vec![
                        "create".to_string(),
                        "assign".to_string(),
                        "status".to_string(),
                        "monitor".to_string(),
                        "disband".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "task".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Task description for the team".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "agents".to_string(),
                    param_type: ParameterType::List,
                    required: false,
                    default: Some("[]".to_string()),
                    description: "Agent types to include in the team".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/team create \"Implement new feature\" [\"code\", \"test\", \"review\"]".to_string(),
                "/team assign \"Debug production issue\"".to_string(),
                "/team status".to_string(),
                "/team monitor".to_string(),
            ],
        });
        
        // /think command for deep cognitive reflection
        self.register(CommandDefinition {
            name: "think".to_string(),
            aliases: vec!["reflect".to_string(), "ponder".to_string()],
            description: "Engage deep cognitive reflection on a topic".to_string(),
            help_text: "Activates Loki's deep thinking mode, engaging multiple cognitive systems for comprehensive analysis and insight generation.".to_string(),
            category: CommandCategory::Analysis,
            required_features: vec!["cognitive_integration".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "topic".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Topic or question to think deeply about".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "depth".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("standard".to_string()),
                    description: "Depth of cognitive processing".to_string(),
                    valid_values: Some(vec![
                        "quick".to_string(),
                        "standard".to_string(),
                        "deep".to_string(),
                        "profound".to_string(),
                    ]),
                },
            ],
            examples: vec![
                "/think \"What is the nature of consciousness?\"".to_string(),
                "/think \"How can we improve code quality?\" deep".to_string(),
                "/reflect \"The meaning of creativity in AI\"".to_string(),
            ],
        });
        
        // /create command for creative mode
        self.register(CommandDefinition {
            name: "create".to_string(),
            aliases: vec!["imagine".to_string(), "innovate".to_string()],
            description: "Activate creative intelligence for novel ideas".to_string(),
            help_text: "Engages Loki's creativity engine to generate innovative ideas, creative solutions, and novel approaches to problems.".to_string(),
            category: CommandCategory::Analysis,
            required_features: vec!["creativity_engine".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "prompt".to_string(),
                    param_type: ParameterType::String,
                    required: true,
                    default: None,
                    description: "Creative prompt or challenge".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "style".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("innovative".to_string()),
                    description: "Creative style to apply".to_string(),
                    valid_values: Some(vec![
                        "innovative".to_string(),
                        "artistic".to_string(),
                        "practical".to_string(),
                        "abstract".to_string(),
                        "synthesis".to_string(),
                    ]),
                },
            ],
            examples: vec![
                "/create \"New ways to visualize code architecture\"".to_string(),
                "/imagine \"A tool that helps developers think\" artistic".to_string(),
                "/innovate \"Combining AI with human creativity\"".to_string(),
            ],
        });
        
        // /empathize command for emotional intelligence
        self.register(CommandDefinition {
            name: "empathize".to_string(),
            aliases: vec!["understand".to_string(), "feel".to_string()],
            description: "Activate empathy and emotional understanding mode".to_string(),
            help_text: "Engages Loki's emotional intelligence and empathy engine to better understand emotional context and provide emotionally aware responses.".to_string(),
            category: CommandCategory::System,
            required_features: vec!["emotional_intelligence".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "context".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Emotional context or situation to understand".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/empathize".to_string(),
                "/empathize \"I'm frustrated with this bug\"".to_string(),
                "/understand \"Team morale is low\"".to_string(),
            ],
        });
        
        // /evolve command for autonomous evolution
        self.register(CommandDefinition {
            name: "evolve".to_string(),
            aliases: vec!["adapt".to_string(), "improve".to_string()],
            description: "Trigger autonomous self-improvement".to_string(),
            help_text: "Activates Loki's autonomous evolution engine to analyze its own performance and evolve its capabilities. Use with caution.".to_string(),
            category: CommandCategory::System,
            required_features: vec!["autonomous_evolution".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "focus".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("general".to_string()),
                    description: "Area to focus evolution on".to_string(),
                    valid_values: Some(vec![
                        "general".to_string(),
                        "reasoning".to_string(),
                        "creativity".to_string(),
                        "efficiency".to_string(),
                        "knowledge".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "confirm".to_string(),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some("false".to_string()),
                    description: "Confirm autonomous evolution".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/evolve".to_string(),
                "/evolve reasoning true".to_string(),
                "/adapt efficiency".to_string(),
            ],
        });
        
        // /story command for story-driven capabilities
        self.register(CommandDefinition {
            name: "story".to_string(),
            aliases: vec!["narrative".to_string()],
            description: "Activate story-driven analysis and generation".to_string(),
            help_text: "Engages Loki's story-driven intelligence for narrative understanding, code storytelling, and context-aware analysis.\n\nModes:\n• analyze - Analyze code structure as narrative\n• generate - Generate code with story principles\n• document - Create narrative documentation\n• learn - Extract patterns from codebase story\n• review - Review PR as story development\n• autonomous - Enable autonomous story mode".to_string(),
            category: CommandCategory::Analysis,
            required_features: vec!["story_driven".to_string()],
            parameters: vec![
                ParameterDefinition {
                    name: "mode".to_string(),
                    param_type: ParameterType::Enum,
                    required: true,
                    default: Some("analyze".to_string()),
                    description: "Story-driven mode".to_string(),
                    valid_values: Some(vec![
                        "analyze".to_string(),
                        "generate".to_string(),
                        "document".to_string(),
                        "learn".to_string(),
                        "review".to_string(),
                        "autonomous".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "target".to_string(),
                    param_type: ParameterType::String,
                    required: false,
                    default: None,
                    description: "Target for story-driven processing (path, prompt, or PR URL)".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/story analyze src/".to_string(),
                "/story generate Create a new authentication module".to_string(),
                "/story document src/auth/".to_string(),
                "/story learn".to_string(),
                "/story review pr:https://github.com/user/repo/pull/123".to_string(),
                "/story autonomous enable:true".to_string(),
            ],
        });
        
        // /export command for exporting chat
        self.register(CommandDefinition {
            name: "export".to_string(),
            aliases: vec!["save".to_string(), "download".to_string()],
            description: "Export chat conversation to file".to_string(),
            help_text: "Export the current chat conversation to various formats including JSON, Markdown, HTML, PDF, and CSV.".to_string(),
            category: CommandCategory::Chat,
            required_features: vec![],
            parameters: vec![
                ParameterDefinition {
                    name: "format".to_string(),
                    param_type: ParameterType::Enum,
                    required: false,
                    default: Some("markdown".to_string()),
                    description: "Export format".to_string(),
                    valid_values: Some(vec![
                        "json".to_string(),
                        "markdown".to_string(),
                        "md".to_string(),
                        "text".to_string(),
                        "txt".to_string(),
                        "html".to_string(),
                        "pdf".to_string(),
                        "csv".to_string(),
                    ]),
                },
                ParameterDefinition {
                    name: "filename".to_string(),
                    param_type: ParameterType::FilePath,
                    required: false,
                    default: None,
                    description: "Output filename (auto-generated if not provided)".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "include_metadata".to_string(),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some("true".to_string()),
                    description: "Include metadata in export".to_string(),
                    valid_values: None,
                },
                ParameterDefinition {
                    name: "include_timestamps".to_string(),
                    param_type: ParameterType::Boolean,
                    required: false,
                    default: Some("true".to_string()),
                    description: "Include timestamps in export".to_string(),
                    valid_values: None,
                },
            ],
            examples: vec![
                "/export".to_string(),
                "/export json".to_string(),
                "/export markdown chat_history.md".to_string(),
                "/export html conversation.html false".to_string(),
                "/export pdf report.pdf true false".to_string(),
            ],
        });
    }
    
    /// Register a new command
    pub fn register(&mut self, command: CommandDefinition) {
        self.commands.insert(command.name.clone(), command);
    }
    
    /// Parse a command from user input
    pub fn parse(&self, input: &str) -> Result<ParsedCommand> {
        let input = input.trim();
        
        // Check if it's a command (starts with /)
        if !input.starts_with('/') {
            return Err(anyhow!("Not a command"));
        }
        
        // Split into parts
        let parts: Vec<&str> = input[1..].split_whitespace().collect();
        if parts.is_empty() {
            return Err(anyhow!("Empty command"));
        }
        
        let command_name = parts[0];
        let remaining = parts[1..].join(" ");
        
        // Find command definition
        let command_def = self.find_command(command_name)
            .ok_or_else(|| anyhow!("Unknown command: {}", command_name))?;
        
        // Check for help flag
        let is_help = remaining.contains("--help") || remaining.contains("-h");
        if is_help {
            return Ok(ParsedCommand {
                command: command_def.name.clone(),
                args: HashMap::new(),
                raw_input: input.to_string(),
                is_help: true,
            });
        }
        
        // Parse arguments
        let args = self.parse_arguments(&command_def, &remaining)?;
        
        Ok(ParsedCommand {
            command: command_def.name.clone(),
            args,
            raw_input: input.to_string(),
            is_help: false,
        })
    }
    
    /// Find a command by name or alias
    fn find_command(&self, name: &str) -> Option<&CommandDefinition> {
        // Direct match
        if let Some(cmd) = self.commands.get(name) {
            return Some(cmd);
        }
        
        // Check aliases
        for cmd in self.commands.values() {
            if cmd.aliases.contains(&name.to_string()) {
                return Some(cmd);
            }
        }
        
        None
    }
    
    /// Parse command arguments
    fn parse_arguments(&self, command: &CommandDefinition, input: &str) -> Result<HashMap<String, Value>> {
        let mut args = HashMap::new();
        let input = input.trim();
        
        if input.is_empty() && command.parameters.iter().all(|p| !p.required) {
            // No input and all parameters are optional
            return Ok(args);
        }
        
        // Simple parsing - can be enhanced with more sophisticated parsing
        let parts: Vec<&str> = input.split_whitespace().collect();
        let mut current_idx = 0;
        
        for param in &command.parameters {
            if current_idx >= parts.len() {
                if param.required {
                    return Err(anyhow!("Missing required parameter: {}", param.name));
                }
                if let Some(default) = &param.default {
                    args.insert(param.name.clone(), json!(default));
                }
                continue;
            }
            
            match &param.param_type {
                ParameterType::Json => {
                    // Find JSON content (everything between { })
                    let remaining = parts[current_idx..].join(" ");
                    if let Some(start) = remaining.find('{') {
                        if let Some(end) = remaining.rfind('}') {
                            let json_str = &remaining[start..=end];
                            match serde_json::from_str::<Value>(json_str) {
                                Ok(value) => {
                                    args.insert(param.name.clone(), value);
                                    // Skip past the JSON
                                    current_idx = parts.len();
                                }
                                Err(e) => {
                                    return Err(anyhow!("Invalid JSON for parameter {}: {}", param.name, e));
                                }
                            }
                        }
                    }
                }
                ParameterType::String => {
                    // For string parameters, check if it's quoted
                    let remaining = parts[current_idx..].join(" ");
                    if remaining.starts_with('"') {
                        // Find closing quote
                        if let Some(end) = remaining[1..].find('"') {
                            let value = &remaining[1..=end];
                            args.insert(param.name.clone(), json!(value));
                            // Calculate how many parts we consumed
                            let consumed = value.split_whitespace().count() + 2; // +2 for quotes
                            current_idx += consumed;
                        }
                    } else {
                        // Single word string
                        args.insert(param.name.clone(), json!(parts[current_idx]));
                        current_idx += 1;
                    }
                }
                _ => {
                    // Simple types - just take the next part
                    args.insert(param.name.clone(), json!(parts[current_idx]));
                    current_idx += 1;
                }
            }
        }
        
        // Validate enum values
        for param in &command.parameters {
            if let Some(value) = args.get(&param.name) {
                if param.param_type == ParameterType::Enum {
                    if let Some(valid_values) = &param.valid_values {
                        if let Some(str_val) = value.as_str() {
                            if !valid_values.contains(&str_val.to_string()) {
                                return Err(anyhow!(
                                    "Invalid value '{}' for parameter {}. Valid values: {:?}",
                                    str_val, param.name, valid_values
                                ));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(args)
    }
    
    /// Get help text for a command
    pub fn get_help(&self, command_name: Option<&str>) -> String {
        if let Some(name) = command_name {
            if let Some(cmd) = self.find_command(name) {
                self.format_command_help(cmd)
            } else {
                format!("Unknown command: {}", name)
            }
        } else {
            self.format_general_help()
        }
    }
    
    /// Format help for a specific command
    fn format_command_help(&self, cmd: &CommandDefinition) -> String {
        let mut help = format!("# /{} - {}\n\n", cmd.name, cmd.description);
        
        if !cmd.aliases.is_empty() {
            help.push_str(&format!("**Aliases:** {}\n\n", cmd.aliases.join(", ")));
        }
        
        help.push_str(&format!("{}\n\n", cmd.help_text));
        
        if !cmd.parameters.is_empty() {
            help.push_str("## Parameters\n\n");
            for param in &cmd.parameters {
                let required = if param.required { " (required)" } else { "" };
                help.push_str(&format!("- **{}**{}: {}\n", param.name, required, param.description));
                if let Some(default) = &param.default {
                    help.push_str(&format!("  Default: {}\n", default));
                }
                if let Some(values) = &param.valid_values {
                    help.push_str(&format!("  Valid values: {}\n", values.join(", ")));
                }
            }
            help.push_str("\n");
        }
        
        if !cmd.examples.is_empty() {
            help.push_str("## Examples\n\n");
            for example in &cmd.examples {
                help.push_str(&format!("```\n{}\n```\n", example));
            }
        }
        
        help
    }
    
    /// Format general help
    fn format_general_help(&self) -> String {
        let mut help = String::from("# Loki Chat Commands\n\n");
        help.push_str("Available commands organized by category:\n\n");
        
        // Group commands by category
        let mut by_category: HashMap<CommandCategory, Vec<&CommandDefinition>> = HashMap::new();
        for cmd in self.commands.values() {
            by_category.entry(cmd.category.clone()).or_insert_with(Vec::new).push(cmd);
        }
        
        // Sort categories
        let mut categories: Vec<_> = by_category.keys().cloned().collect();
        categories.sort_by_key(|c| match c {
            CommandCategory::Help => 0,
            CommandCategory::Tools => 1,
            CommandCategory::Task => 2,
            CommandCategory::Workflow => 3,
            CommandCategory::Analysis => 4,
            CommandCategory::Development => 5,
            CommandCategory::System => 6,
            CommandCategory::Chat => 7,
        });
        
        for category in categories {
            if let Some(commands) = by_category.get(&category) {
                help.push_str(&format!("## {:?}\n\n", category));
                let mut sorted_commands = commands.clone();
                sorted_commands.sort_by_key(|c| &c.name);
                
                for cmd in sorted_commands {
                    let aliases = if cmd.aliases.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", cmd.aliases.join(", "))
                    };
                    help.push_str(&format!("- **/{}{} ** - {}\n", cmd.name, aliases, cmd.description));
                }
                help.push_str("\n");
            }
        }
        
        help.push_str("Use `/help <command>` for detailed information about a specific command.\n");
        help
    }
    
    /// Get all available commands
    pub fn get_commands(&self) -> Vec<&CommandDefinition> {
        self.commands.values().collect()
    }
    
    /// Get commands by category
    pub fn get_commands_by_category(&self, category: CommandCategory) -> Vec<&CommandDefinition> {
        self.commands.values()
            .filter(|cmd| cmd.category == category)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_command_parsing() {
        let registry = CommandRegistry::new();
        
        // Test basic command
        let parsed = registry.parse("/tools list").unwrap();
        assert_eq!(parsed.command, "tools");
        assert_eq!(parsed.args.get("action").unwrap(), "list");
        
        // Test command with JSON
        let parsed = registry.parse("/execute web_search {\"query\": \"test\"}").unwrap();
        assert_eq!(parsed.command, "execute");
        assert_eq!(parsed.args.get("tool").unwrap(), "web_search");
        assert!(parsed.args.get("args").unwrap().is_object());
        
        // Test alias
        let parsed = registry.parse("/exec test").unwrap();
        assert_eq!(parsed.command, "execute");
        
        // Test help flag
        let parsed = registry.parse("/workflow --help").unwrap();
        assert!(parsed.is_help);
    }
    
    #[test]
    fn test_help_generation() {
        let registry = CommandRegistry::new();
        
        // Test general help
        let help = registry.get_help(None);
        assert!(help.contains("Loki Chat Commands"));
        
        // Test specific command help
        let help = registry.get_help(Some("execute"));
        assert!(help.contains("/execute"));
        assert!(help.contains("Examples"));
    }
}