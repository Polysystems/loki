//! Enhanced Natural Language Processor for Loki Chat
//! 
//! This module provides sophisticated natural language understanding
//! to enable seamless task execution through conversational interactions.

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{debug};
use regex::Regex;

use crate::tui::chat::core::commands::CommandRegistry;
use crate::tui::chat::core::tool_executor::ChatToolExecutor;
use crate::models::ModelOrchestrator;

/// Natural language processor for understanding and executing user intents
pub struct NaturalLanguageProcessor {
    /// Command registry for fallback to direct commands
    command_registry: CommandRegistry,
    
    /// Tool executor for running identified tasks
    tool_executor: Option<Arc<ChatToolExecutor>>,
    
    /// Model orchestrator for AI-powered understanding
    model_orchestrator: Option<Arc<ModelOrchestrator>>,
    
    /// Intent patterns for quick matching
    intent_patterns: HashMap<String, IntentPattern>,
    
    /// Context tracking for multi-turn conversations
    conversation_context: ConversationContext,
}

/// Pattern for matching user intents
#[derive(Debug, Clone)]
struct IntentPattern {
    /// Intent type this pattern matches
    intent: EnhancedIntentType,
    
    /// Regex patterns to match
    patterns: Vec<Regex>,
    
    /// Keywords that strengthen this intent
    keywords: Vec<String>,
    
    /// Parameter extraction rules
    parameter_rules: Vec<ParameterRule>,
    
    /// Confidence boost if matched
    confidence_boost: f32,
}

/// Rule for extracting parameters from natural language
#[derive(Debug, Clone)]
struct ParameterRule {
    /// Parameter name
    name: String,
    
    /// Extraction pattern
    pattern: Regex,
    
    /// Type of parameter
    param_type: ParameterType,
    
    /// Whether this parameter is required
    required: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum ParameterType {
    FilePath,
    CodeSnippet,
    SearchQuery,
    Description,
    Number,
    Boolean,
}

/// Enhanced intent types that map to actual tool executions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnhancedIntentType {
    // File operations
    ReadFile { path: Option<String> },
    WriteFile { path: Option<String>, content: Option<String> },
    CreateFile { path: Option<String>, content: Option<String> },
    EditFile { path: Option<String>, changes: Option<String> },
    ListFiles { directory: Option<String>, pattern: Option<String> },
    SearchFiles { query: String, path: Option<String> },
    
    // Directory operations
    CreateDirectory { path: Option<String> },
    DeleteDirectory { path: Option<String> },
    MoveFile { from: Option<String>, to: Option<String> },
    CopyFile { from: Option<String>, to: Option<String> },
    DeleteFile { path: Option<String> },
    
    // Code operations
    AnalyzeCode { file: Option<String>, aspect: Option<String> },
    RefactorCode { file: Option<String>, description: Option<String> },
    GenerateCode { description: String, language: Option<String> },
    ExplainCode { file: Option<String>, snippet: Option<String> },
    FindBugs { file: Option<String> },
    OptimizeCode { file: Option<String>, focus: Option<String> },
    
    // Search and information
    WebSearch { query: String },
    SearchDocumentation { query: String, source: Option<String> },
    FindInCodebase { query: String, file_type: Option<String> },
    
    // Task and project management
    CreateTask { description: String, priority: Option<String> },
    ListTasks { filter: Option<String> },
    UpdateTask { id: Option<String>, updates: Option<String> },
    
    // System operations
    RunCommand { command: String, args: Option<Vec<String>> },
    CheckStatus { component: Option<String> },
    InstallDependency { package: String },
    
    // AI and model operations
    ChangeModel { model: Option<String> },
    ConfigureAI { settings: Option<HashMap<String, Value>> },
    
    // General conversation
    Question { topic: String },
    Clarification { about: String },
    Confirmation { action: String },
}

/// Conversation context for tracking multi-turn interactions
#[derive(Debug, Clone)]
pub struct ConversationContext {
    /// Recent files mentioned or worked on
    pub recent_files: Vec<String>,
    
    /// Current working directory/project
    pub current_project: Option<String>,
    
    /// Recent code snippets discussed
    pub code_context: Vec<String>,
    
    /// Active task or goal
    pub current_goal: Option<String>,
    
    /// Previous intents for context
    pub intent_history: Vec<EnhancedIntentType>,
    
    /// Entities mentioned (functions, classes, variables)
    pub entities: HashMap<String, EntityInfo>,
}

#[derive(Debug, Clone)]
pub struct EntityInfo {
    pub entity_type: String,
    pub location: Option<String>,
    pub last_mentioned: std::time::Instant,
}

/// Result of natural language processing
#[derive(Debug, Clone)]
pub struct NLPResult {
    /// Detected intent
    pub intent: EnhancedIntentType,
    
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    
    /// Extracted parameters
    pub parameters: HashMap<String, Value>,
    
    /// Suggested tool execution
    pub tool_suggestion: Option<ToolSuggestion>,
    
    /// Natural language response prefix
    pub response_prefix: Option<String>,
    
    /// Whether to execute immediately or confirm first
    pub requires_confirmation: bool,
}

#[derive(Debug, Clone)]
pub struct ToolSuggestion {
    /// Tool to execute
    pub tool: String,
    
    /// Arguments for the tool
    pub args: Value,
    
    /// Human-readable description
    pub description: String,
}

impl NaturalLanguageProcessor {
    /// Create a new natural language processor
    pub fn new(
        command_registry: CommandRegistry,
        tool_executor: Option<Arc<ChatToolExecutor>>,
        model_orchestrator: Option<Arc<ModelOrchestrator>>,
    ) -> Self {
        let mut processor = Self {
            command_registry,
            tool_executor,
            model_orchestrator,
            intent_patterns: HashMap::new(),
            conversation_context: ConversationContext {
                recent_files: Vec::new(),
                current_project: None,
                code_context: Vec::new(),
                current_goal: None,
                intent_history: Vec::new(),
                entities: HashMap::new(),
            },
        };
        
        processor.initialize_patterns();
        processor
    }
    
    /// Initialize intent patterns
    fn initialize_patterns(&mut self) {
        // File reading patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::ReadFile { path: None },
            patterns: vec![
                Regex::new(r"(?i)(show|read|display|open|look at|check out|view)\s+(?:the\s+)?(?:file\s+)?(.+?)(?:\s+file)?$").unwrap(),
                Regex::new(r"(?i)what(?:'s|\s+is)\s+in\s+(.+?)(?:\s+file)?$").unwrap(),
                Regex::new(r"(?i)(?:can\s+you\s+)?(?:please\s+)?show\s+me\s+(.+?)$").unwrap(),
            ],
            keywords: vec!["show".to_string(), "read".to_string(), "file".to_string(), "open".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "path".to_string(),
                    pattern: Regex::new(r"(?:(?:src|lib|tests?|docs?)/)?[\w\-_/]+\.(?:rs|toml|md|json|yaml|yml|txt|js|ts|py)").unwrap(),
                    param_type: ParameterType::FilePath,
                    required: true,
                },
            ],
            confidence_boost: 0.2,
        });
        
        // Code analysis patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::AnalyzeCode { file: None, aspect: None },
            patterns: vec![
                Regex::new(r"(?i)analyze\s+(?:the\s+)?(.+?)(?:\s+for\s+(.+?))?$").unwrap(),
                Regex::new(r"(?i)(?:can\s+you\s+)?check\s+(.+?)\s+for\s+(?:any\s+)?(.+?)$").unwrap(),
                Regex::new(r"(?i)review\s+(?:the\s+)?(.+?)(?:\s+code)?$").unwrap(),
            ],
            keywords: vec!["analyze".to_string(), "review".to_string(), "check".to_string(), "inspect".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "file".to_string(),
                    pattern: Regex::new(r"[\w\-_/]+\.(?:rs|js|ts|py|go|java|cpp|c)").unwrap(),
                    param_type: ParameterType::FilePath,
                    required: false,
                },
                ParameterRule {
                    name: "aspect".to_string(),
                    pattern: Regex::new(r"(?i)(performance|security|bugs?|issues?|problems?|complexity|quality)").unwrap(),
                    param_type: ParameterType::Description,
                    required: false,
                },
            ],
            confidence_boost: 0.3,
        });
        
        // Search patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::SearchFiles { query: String::new(), path: None },
            patterns: vec![
                Regex::new(r"(?i)(?:search|find|look)\s+for\s+(.+?)(?:\s+in\s+(.+?))?$").unwrap(),
                Regex::new(r"(?i)where\s+is\s+(.+?)(?:\s+defined)?$").unwrap(),
                Regex::new(r"(?i)find\s+(?:all\s+)?(.+?)\s+(?:in\s+)?(?:the\s+)?(?:code|files|project)?$").unwrap(),
            ],
            keywords: vec!["search".to_string(), "find".to_string(), "where".to_string(), "locate".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "query".to_string(),
                    pattern: Regex::new(r".+").unwrap(),
                    param_type: ParameterType::SearchQuery,
                    required: true,
                },
            ],
            confidence_boost: 0.2,
        });
        
        // Code generation patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::GenerateCode { description: String::new(), language: None },
            patterns: vec![
                Regex::new(r"(?i)(?:create|generate|write|make)\s+(?:a\s+)?(.+?)(?:\s+in\s+(\w+))?$").unwrap(),
                Regex::new(r"(?i)(?:can\s+you\s+)?(?:please\s+)?(?:help\s+me\s+)?(?:write|create)\s+(.+?)$").unwrap(),
                Regex::new(r"(?i)I\s+need\s+(?:a\s+)?(.+?)(?:\s+function|\s+class|\s+component)?").unwrap(),
            ],
            keywords: vec!["create".to_string(), "generate".to_string(), "write".to_string(), "make".to_string(), "need".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "description".to_string(),
                    pattern: Regex::new(r".+").unwrap(),
                    param_type: ParameterType::Description,
                    required: true,
                },
            ],
            confidence_boost: 0.3,
        });
        
        // Task creation patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::CreateTask { description: String::new(), priority: None },
            patterns: vec![
                Regex::new(r"(?i)(?:create|add|make)\s+(?:a\s+)?task\s+(?:to\s+)?(.+?)$").unwrap(),
                Regex::new(r"(?i)remind\s+me\s+to\s+(.+?)$").unwrap(),
                Regex::new(r"(?i)(?:I\s+)?need\s+to\s+(.+?)$").unwrap(),
            ],
            keywords: vec!["task".to_string(), "todo".to_string(), "remind".to_string(), "need to".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "description".to_string(),
                    pattern: Regex::new(r".+").unwrap(),
                    param_type: ParameterType::Description,
                    required: true,
                },
            ],
            confidence_boost: 0.2,
        });
        
        // File editing patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::EditFile { path: None, changes: None },
            patterns: vec![
                Regex::new(r"(?i)(?:edit|modify|change|update)\s+(.+?)\s+(?:to\s+)?(.+?)$").unwrap(),
                Regex::new(r"(?i)(?:can\s+you\s+)?(?:please\s+)?(?:fix|correct)\s+(.+?)$").unwrap(),
                Regex::new(r"(?i)replace\s+(.+?)\s+with\s+(.+?)$").unwrap(),
            ],
            keywords: vec!["edit".to_string(), "modify".to_string(), "change".to_string(), "fix".to_string(), "update".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "path".to_string(),
                    pattern: Regex::new(r"[\w\-_/]+\.\w+").unwrap(),
                    param_type: ParameterType::FilePath,
                    required: false,
                },
            ],
            confidence_boost: 0.3,
        });
        
        // Directory creation patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::CreateDirectory { path: None },
            patterns: vec![
                Regex::new(r"(?i)(?:create|make)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory|dir)\s+(?:named\s+|called\s+)?(.+?)(?:\s+in\s+(.+?))?$").unwrap(),
                Regex::new(r"(?i)(?:create|make)\s+(?:a\s+)?(?:new\s+)?folder\s+in\s+(.+?)$").unwrap(),
                Regex::new(r"(?i)mkdir\s+(.+?)$").unwrap(),
                Regex::new(r"(?i)(?:can\s+you\s+)?(?:please\s+)?(?:create|make)\s+(?:a\s+)?(?:new\s+)?folder\s+(?:on\s+|in\s+)?(?:my\s+)?(?:desktop|documents|downloads|home)\s+(?:for\s+|called\s+)?(.+?)$").unwrap(),
            ],
            keywords: vec!["create".to_string(), "make".to_string(), "folder".to_string(), "directory".to_string(), "mkdir".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "path".to_string(),
                    pattern: Regex::new(r".+").unwrap(),
                    param_type: ParameterType::FilePath,
                    required: true,
                },
            ],
            confidence_boost: 0.4,
        });
        
        // List files/directories patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::ListFiles { directory: None, pattern: None },
            patterns: vec![
                Regex::new(r"(?i)(?:list|show)\s+(?:all\s+)?(?:files|folders|directories)\s+(?:in\s+)?(.+?)$").unwrap(),
                Regex::new(r"(?i)(?:what(?:'s|\\s+is))\s+in\s+(?:the\s+)?(.+?)\s+(?:folder|directory)?$").unwrap(),
                Regex::new(r"(?i)ls\s+(.+?)$").unwrap(),
                Regex::new(r"(?i)(?:show\s+me\s+)?(?:what(?:'s|\\s+is))\s+in\s+(?:my\s+)?(?:desktop|documents|downloads|home)$").unwrap(),
            ],
            keywords: vec!["list".to_string(), "ls".to_string(), "show".to_string(), "what's in".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "directory".to_string(),
                    pattern: Regex::new(r".+").unwrap(),
                    param_type: ParameterType::FilePath,
                    required: false,
                },
            ],
            confidence_boost: 0.3,
        });
        
        // Delete file/directory patterns
        self.add_pattern(IntentPattern {
            intent: EnhancedIntentType::DeleteFile { path: None },
            patterns: vec![
                Regex::new(r"(?i)(?:delete|remove|rm)\s+(?:the\s+)?(?:file\s+)?(.+?)$").unwrap(),
                Regex::new(r"(?i)(?:can\s+you\s+)?(?:please\s+)?(?:delete|remove)\s+(.+?)$").unwrap(),
            ],
            keywords: vec!["delete".to_string(), "remove".to_string(), "rm".to_string()],
            parameter_rules: vec![
                ParameterRule {
                    name: "path".to_string(),
                    pattern: Regex::new(r".+").unwrap(),
                    param_type: ParameterType::FilePath,
                    required: true,
                },
            ],
            confidence_boost: 0.3,
        });
    }
    
    /// Add an intent pattern
    fn add_pattern(&mut self, pattern: IntentPattern) {
        let key = format!("{:?}", pattern.intent);
        self.intent_patterns.insert(key, pattern);
    }
    
    /// Process natural language input
    pub async fn process(&self, input: &str) -> Result<NLPResult> {
        debug!("Processing natural language input: {}", input);
        
        // TODO: Update context with the input (requires thread-safe context)
        // self.update_context_from_input(input);
        
        // Try pattern-based matching first
        if let Some(result) = self.match_patterns(input) {
            if result.confidence > 0.7 {
                return Ok(result);
            }
        }
        
        // Use AI model for more complex understanding if available
        if let Some(orchestrator) = &self.model_orchestrator {
            if let Ok(ai_result) = self.ai_powered_understanding(input, orchestrator).await {
                if ai_result.confidence > 0.8 {
                    return Ok(ai_result);
                }
            }
        }
        
        // Fallback to keyword-based detection
        self.keyword_based_detection(input)
    }
    
    /// Match input against patterns
    fn match_patterns(&self, input: &str) -> Option<NLPResult> {
        let mut best_match: Option<(f32, NLPResult)> = None;
        
        for pattern in self.intent_patterns.values() {
            let mut confidence = 0.0;
            let mut parameters = HashMap::new();
            
            // Check regex patterns
            for regex in &pattern.patterns {
                if let Some(captures) = regex.captures(input) {
                    confidence += 0.5;
                    
                    // Extract parameters from captures
                    for (i, cap) in captures.iter().enumerate().skip(1) {
                        if let Some(matched) = cap {
                            if i <= pattern.parameter_rules.len() {
                                let rule = &pattern.parameter_rules[i - 1];
                                parameters.insert(rule.name.clone(), json!(matched.as_str()));
                            }
                        }
                    }
                    break;
                }
            }
            
            // Check keywords
            let input_lower = input.to_lowercase();
            for keyword in &pattern.keywords {
                if input_lower.contains(keyword) {
                    confidence += 0.1;
                }
            }
            
            // Apply confidence boost
            if confidence > 0.0 {
                confidence += pattern.confidence_boost;
                confidence = confidence.min(1.0);
                
                let result = self.create_nlp_result(pattern.intent.clone(), confidence, parameters);
                
                if best_match.is_none() || confidence > best_match.as_ref().unwrap().0 {
                    best_match = Some((confidence, result));
                }
            }
        }
        
        best_match.map(|(_, result)| result)
    }
    
    /// Use AI model for understanding
    async fn ai_powered_understanding(
        &self,
        input: &str,
        orchestrator: &Arc<ModelOrchestrator>,
    ) -> Result<NLPResult> {
        // Create a prompt for intent detection
        let prompt = format!(
            "Analyze this user request and identify the intent and parameters:\n\
             User: {}\n\n\
             Identify:\n\
             1. Primary intent (file operation, code analysis, search, etc.)\n\
             2. Required parameters (file paths, search queries, etc.)\n\
             3. Confidence level (0-1)\n\n\
             Context:\n\
             - Recent files: {:?}\n\
             - Current project: {:?}\n\
             \n\
             Respond in JSON format.",
            input,
            self.conversation_context.recent_files.last(),
            self.conversation_context.current_project
        );
        
        // This would use the model orchestrator to get AI understanding
        // For now, return an error to fallback to other methods
        Err(anyhow!("AI understanding not yet implemented"))
    }
    
    /// Keyword-based detection fallback
    fn keyword_based_detection(&self, input: &str) -> Result<NLPResult> {
        let input_lower = input.to_lowercase();
        
        // File operations
        if input_lower.contains("show") || input_lower.contains("read") || input_lower.contains("open") {
            if let Some(file) = self.extract_file_path(input) {
                return Ok(self.create_nlp_result(
                    EnhancedIntentType::ReadFile { path: Some(file.clone()) },
                    0.6,
                    HashMap::from([("path".to_string(), json!(file))]),
                ));
            }
        }
        
        // Search operations
        if input_lower.contains("search") || input_lower.contains("find") || input_lower.contains("where") {
            let query = self.extract_search_query(input);
            return Ok(self.create_nlp_result(
                EnhancedIntentType::SearchFiles { query: query.clone(), path: None },
                0.6,
                HashMap::from([("query".to_string(), json!(query))]),
            ));
        }
        
        // Code analysis
        if input_lower.contains("analyze") || input_lower.contains("review") || input_lower.contains("check") {
            let file = self.extract_file_path(input);
            return Ok(self.create_nlp_result(
                EnhancedIntentType::AnalyzeCode { file: file.clone(), aspect: None },
                0.5,
                HashMap::from([("file".to_string(), json!(file))]),
            ));
        }
        
        // Directory/folder creation
        if (input_lower.contains("create") || input_lower.contains("make")) && 
           (input_lower.contains("folder") || input_lower.contains("directory")) {
            // Try to extract the path/name
            let path = self.extract_folder_name(input);
            return Ok(self.create_nlp_result(
                EnhancedIntentType::CreateDirectory { path: path.clone() },
                0.6,
                HashMap::from([("path".to_string(), json!(path))]),
            ));
        }
        
        // List files
        if input_lower.contains("list") || input_lower.contains("ls") || 
           (input_lower.contains("what") && input_lower.contains("in")) {
            let directory = self.extract_directory_path(input);
            return Ok(self.create_nlp_result(
                EnhancedIntentType::ListFiles { directory: directory.clone(), pattern: None },
                0.6,
                HashMap::from([("directory".to_string(), json!(directory))]),
            ));
        }
        
        // Default to a question
        Ok(self.create_nlp_result(
            EnhancedIntentType::Question { topic: input.to_string() },
            0.3,
            HashMap::new(),
        ))
    }
    
    /// Create NLP result with tool suggestion
    fn create_nlp_result(
        &self,
        intent: EnhancedIntentType,
        confidence: f32,
        parameters: HashMap<String, Value>,
    ) -> NLPResult {
        let (tool_suggestion, response_prefix, requires_confirmation) = match &intent {
            EnhancedIntentType::ReadFile { path: Some(p) } => {
                (
                    Some(ToolSuggestion {
                        tool: "read_file".to_string(),
                        args: json!({ "path": p }),
                        description: format!("Reading file: {}", p),
                    }),
                    Some(format!("Let me read {} for you.", p)),
                    false,
                )
            }
            EnhancedIntentType::SearchFiles { query, path } => {
                (
                    Some(ToolSuggestion {
                        tool: "search_files".to_string(),
                        args: json!({ "query": query, "path": path }),
                        description: format!("Searching for: {}", query),
                    }),
                    Some(format!("Searching for '{}' in the codebase...", query)),
                    false,
                )
            }
            EnhancedIntentType::AnalyzeCode { file, aspect } => {
                (
                    Some(ToolSuggestion {
                        tool: "analyze_code".to_string(),
                        args: json!({ "file": file, "aspect": aspect }),
                        description: format!("Analyzing code{}", 
                            file.as_ref().map(|f| format!(" in {}", f)).unwrap_or_default()
                        ),
                    }),
                    Some("Let me analyze the code for you.".to_string()),
                    false,
                )
            }
            EnhancedIntentType::EditFile { path, changes } => {
                (
                    Some(ToolSuggestion {
                        tool: "edit_file".to_string(),
                        args: json!({ "path": path, "changes": changes }),
                        description: "Editing file".to_string(),
                    }),
                    Some("I'll help you edit the file.".to_string()),
                    true, // Require confirmation for edits
                )
            }
            EnhancedIntentType::CreateTask { description, priority } => {
                (
                    Some(ToolSuggestion {
                        tool: "create_task".to_string(),
                        args: json!({ "description": description, "priority": priority }),
                        description: format!("Creating task: {}", description),
                    }),
                    Some("I'll create this task for you.".to_string()),
                    false,
                )
            }
            EnhancedIntentType::CreateDirectory { path: Some(p) } => {
                (
                    Some(ToolSuggestion {
                        tool: "mcp_filesystem_create_directory".to_string(),
                        args: json!({ "path": p }),
                        description: format!("Creating directory: {}", p),
                    }),
                    Some(format!("Creating folder: {}", p)),
                    false,
                )
            }
            EnhancedIntentType::ListFiles { directory, pattern } => {
                (
                    Some(ToolSuggestion {
                        tool: "mcp_filesystem_list_directory".to_string(),
                        args: json!({ "path": directory.as_ref().unwrap_or(&".".to_string()) }),
                        description: format!("Listing files in: {}", directory.as_ref().unwrap_or(&"current directory".to_string())),
                    }),
                    Some(format!("Listing files in {}", directory.as_ref().unwrap_or(&"current directory".to_string()))),
                    false,
                )
            }
            EnhancedIntentType::DeleteFile { path: Some(p) } => {
                (
                    Some(ToolSuggestion {
                        tool: "mcp_filesystem_delete".to_string(),
                        args: json!({ "path": p }),
                        description: format!("Deleting: {}", p),
                    }),
                    Some(format!("Deleting {}", p)),
                    true, // Require confirmation for deletions
                )
            }
            EnhancedIntentType::MoveFile { from: Some(f), to: Some(t) } => {
                (
                    Some(ToolSuggestion {
                        tool: "mcp_filesystem_move".to_string(),
                        args: json!({ "from": f, "to": t }),
                        description: format!("Moving {} to {}", f, t),
                    }),
                    Some(format!("Moving {} to {}", f, t)),
                    true, // Require confirmation for moves
                )
            }
            EnhancedIntentType::CopyFile { from: Some(f), to: Some(t) } => {
                (
                    Some(ToolSuggestion {
                        tool: "mcp_filesystem_copy".to_string(),
                        args: json!({ "from": f, "to": t }),
                        description: format!("Copying {} to {}", f, t),
                    }),
                    Some(format!("Copying {} to {}", f, t)),
                    false,
                )
            }
            _ => (None, None, false),
        };
        
        NLPResult {
            intent,
            confidence,
            parameters,
            tool_suggestion,
            response_prefix,
            requires_confirmation,
        }
    }
    
    /// Update context from input
    fn update_context_from_input(&mut self, input: &str) {
        // Extract and remember file paths
        if let Some(file) = self.extract_file_path(input) {
            self.conversation_context.recent_files.push(file);
            if self.conversation_context.recent_files.len() > 10 {
                self.conversation_context.recent_files.remove(0);
            }
        }
        
        // Extract entities (functions, classes, etc.)
        let entity_pattern = Regex::new(r"\b(fn|function|class|struct|trait|interface)\s+(\w+)").unwrap();
        for cap in entity_pattern.captures_iter(input) {
            if let (Some(entity_type), Some(name)) = (cap.get(1), cap.get(2)) {
                self.conversation_context.entities.insert(
                    name.as_str().to_string(),
                    EntityInfo {
                        entity_type: entity_type.as_str().to_string(),
                        location: None,
                        last_mentioned: std::time::Instant::now(),
                    },
                );
            }
        }
    }
    
    /// Extract file path from input
    fn extract_file_path(&self, input: &str) -> Option<String> {
        // Look for common file patterns
        let file_pattern = Regex::new(r"(?:(?:src|lib|tests?|docs?|examples?)/)?[\w\-_/]+\.(?:rs|toml|md|json|yaml|yml|txt|js|ts|py|go|java|cpp|c|h|hpp)").unwrap();
        
        file_pattern.find(input).map(|m| m.as_str().to_string())
    }
    
    /// Extract search query
    fn extract_search_query(&self, input: &str) -> String {
        // Remove common search keywords
        let cleaned = input
            .to_lowercase()
            .replace("search for", "")
            .replace("find", "")
            .replace("look for", "")
            .replace("where is", "")
            .trim()
            .to_string();
        
        cleaned
    }
    
    /// Extract folder name from input
    fn extract_folder_name(&self, input: &str) -> Option<String> {
        // Try to extract folder name after keywords
        let patterns = [
            r#"(?i)folder\s+(?:named\s+|called\s+)?["']?([^"']+)["']?"#,
            r#"(?i)directory\s+(?:named\s+|called\s+)?["']?([^"']+)["']?"#,
            r#"(?i)(?:create|make)\s+(?:a\s+)?(?:new\s+)?folder\s+in\s+(?:my\s+)?desktop\s+(?:for\s+|called\s+)?([^"']+)"#,
            r#"(?i)(?:create|make)\s+(?:a\s+)?(?:new\s+)?["']?([^"']+)["']?\s+folder"#,
        ];
        
        for pattern in patterns {
            if let Ok(re) = Regex::new(pattern) {
                if let Some(caps) = re.captures(input) {
                    if let Some(name) = caps.get(1) {
                        let folder_name = name.as_str().trim();
                        // Handle special paths
                        if input.to_lowercase().contains("desktop") {
                            return Some(format!("~/Desktop/{}", folder_name));
                        } else if input.to_lowercase().contains("documents") {
                            return Some(format!("~/Documents/{}", folder_name));
                        } else if input.to_lowercase().contains("downloads") {
                            return Some(format!("~/Downloads/{}", folder_name));
                        }
                        return Some(folder_name.to_string());
                    }
                }
            }
        }
        
        // Fallback: extract any quoted text or the last word
        if let Some(caps) = Regex::new(r#""([^"]+)"|'([^']+)'"#).unwrap().captures(input) {
            return Some(caps.get(1).or(caps.get(2)).unwrap().as_str().to_string());
        }
        
        None
    }
    
    /// Extract directory path from input
    fn extract_directory_path(&self, input: &str) -> Option<String> {
        // Look for common directory references
        if input.to_lowercase().contains("desktop") {
            return Some("~/Desktop".to_string());
        } else if input.to_lowercase().contains("documents") {
            return Some("~/Documents".to_string());
        } else if input.to_lowercase().contains("downloads") {
            return Some("~/Downloads".to_string());
        } else if input.to_lowercase().contains("home") {
            return Some("~".to_string());
        }
        
        // Try to extract path-like strings
        if let Some(caps) = Regex::new(r"(?:in\s+)?([~./][^\s]+)").unwrap().captures(input) {
            return Some(caps.get(1).unwrap().as_str().to_string());
        }
        
        None
    }
    
    /// Get conversation context
    pub fn get_context(&self) -> &ConversationContext {
        &self.conversation_context
    }
    
    /// Clear conversation context
    pub fn clear_context(&mut self) {
        self.conversation_context = ConversationContext {
            recent_files: Vec::new(),
            current_project: None,
            code_context: Vec::new(),
            current_goal: None,
            intent_history: Vec::new(),
            entities: HashMap::new(),
        };
    }
}