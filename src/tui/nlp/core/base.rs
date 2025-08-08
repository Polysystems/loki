use std::collections::HashMap;

use anyhow::Result;

#[derive(Clone)]
/// Natural language processor for understanding user commands
pub struct NaturalLanguageProcessor {
    commands: HashMap<String, CommandPattern>,
    // matcher: SkimMatcherV2,
    context_memory: Vec<String>,
    last_action: Option<CommandAction>,
    session_entities: HashMap<String, String>, // Track entities mentioned in session
}

/// Command pattern with variations and action
#[derive(Debug, Clone)]
pub struct CommandPattern {
    pub primary: String,
    pub variations: Vec<String>,
    pub action: CommandAction,
    pub help: String,
}

/// Actions that can be performed
#[derive(Debug, Clone)]
pub enum CommandAction {
    // Compute actions
    ShowDevices,
    MonitorGpu,
    SetDevice(String),

    // Stream actions
    CreateStream { name: String },
    ListStreams,
    StreamStatus(String),
    StopStream(String),

    // Cluster actions
    ShowCluster,
    DeployModel { model: String, node: Option<String> },
    RemoveModel(String),
    BalanceCluster,

    // Model actions
    ListModels,
    LoadModel(String),
    ModelInfo(String),

    // Model orchestration actions
    ShowTemplates,
    LaunchTemplate(String),
    ShowSessions,
    SwitchSession(String),
    StopSession(String),
    SetupApiKeys,
    InstallModel(String),
    ShowModelCosts,
    SaveFavorite(String),
    LoadFavorite(String),
    SetModelView(ModelViewType),

    // Task actions
    RunTask { task: String, target: String },

    // File system actions
    CreateFolder { path: String },
    NavigateTo { path: String },
    ListFiles { path: Option<String> },
    DeleteFile { path: String },
    ShowCurrentDirectory,

    // System actions
    ShowMetrics,
    ShowLogs,
    SetView(ViewType),
    Help,
    Quit,

    // Analytics actions
    ShowAnalytics,
    ShowPerformance,
    ShowCostAnalytics,
    ShowModelComparison,

    // Collaborative actions
    CreateSession { name: String },
    JoinSession { id: String },
    ListSessions,
    ShowParticipants,

    // Agent specialization actions
    ShowAgentSpecialization,
    ShowAgentRouting,
    ShowAgentPerformance,
    ShowLoadBalancing,

    // Plugin ecosystem actions
    ShowPlugins,
    InstallPlugin { name: String },
    EnablePlugin { name: String },
    DisablePlugin { name: String },
    ShowPluginMarketplace,

    // Evolution actions
    ShowEvolution,
    ShowAdaptation,
    ShowLearning,

    // Consciousness actions
    ShowConsciousness,
    ShowCognitiveState,
    ShowMemoryState,

    // Training actions
    ShowTraining,
    StartTraining { model: String },
    StopTraining { model: String },

    // Anomaly detection actions
    ShowAnomalies,
    ShowHealthMetrics,
    ShowSystemStatus,

    // Quantum optimization actions
    ShowQuantumOptimization,
    ShowQuantumAlgorithms,
    ShowQuantumPerformance,

    // Natural language actions
    ShowNaturalLanguage,
    ShowConversations,
    ShowLanguageModels,
    ShowVoiceInterface,
}

#[derive(Debug, Clone)]
pub enum ViewType {
    Dashboard,
    Compute,
    Streams,
    Cluster,
    Models,
    Logs,
}

#[derive(Debug, Clone)]
pub enum ModelViewType {
    Templates,
    Sessions,
    Registry,
    Analytics,
}

impl NaturalLanguageProcessor {
    pub fn new() -> Self {
        let mut processor = Self {
            commands: HashMap::new(),
            // matcher: SkimMatcherV2::default(),
            context_memory: Vec::with_capacity(10),
            last_action: None,
            session_entities: HashMap::new(),
        };

        processor.init_commands();
        processor
    }

    fn init_commands(&mut self) {
        // Compute commands
        self.add_command(CommandPattern {
            primary: "show devices".to_string(),
            variations: vec![
                "list devices".to_string(),
                "show gpus".to_string(),
                "what devices are available".to_string(),
                "show compute resources".to_string(),
            ],
            action: CommandAction::ShowDevices,
            help: "Show available compute devices (GPUs/CPUs)".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "monitor gpu".to_string(),
            variations: vec![
                "watch gpu".to_string(),
                "show gpu usage".to_string(),
                "gpu stats".to_string(),
                "monitor resources".to_string(),
            ],
            action: CommandAction::MonitorGpu,
            help: "Monitor GPU usage in real-time".to_string(),
        });

        // Stream commands
        self.add_command(CommandPattern {
            primary: "create stream".to_string(),
            variations: vec![
                "new stream".to_string(),
                "start stream".to_string(),
                "add stream".to_string(),
            ],
            action: CommandAction::CreateStream { name: String::new() },
            help: "Create a new streaming pipeline".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "list streams".to_string(),
            variations: vec![
                "show streams".to_string(),
                "active streams".to_string(),
                "what streams are running".to_string(),
            ],
            action: CommandAction::ListStreams,
            help: "List all active streams".to_string(),
        });

        // Cluster commands
        self.add_command(CommandPattern {
            primary: "show cluster".to_string(),
            variations: vec![
                "cluster status".to_string(),
                "show nodes".to_string(),
                "cluster info".to_string(),
            ],
            action: CommandAction::ShowCluster,
            help: "Show cluster status and topology".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "deploy model".to_string(),
            variations: vec![
                "add model".to_string(),
                "load model".to_string(),
                "start model".to_string(),
            ],
            action: CommandAction::DeployModel { model: String::new(), node: None },
            help: "Deploy a model to the cluster".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "balance cluster".to_string(),
            variations: vec![
                "rebalance".to_string(),
                "optimize cluster".to_string(),
                "redistribute models".to_string(),
            ],
            action: CommandAction::BalanceCluster,
            help: "Rebalance models across cluster nodes".to_string(),
        });

        // Model commands
        self.add_command(CommandPattern {
            primary: "list models".to_string(),
            variations: vec![
                "show models".to_string(),
                "available models".to_string(),
                "what models do I have".to_string(),
            ],
            action: CommandAction::ListModels,
            help: "List available models".to_string(),
        });

        // Model orchestration commands
        self.add_command(CommandPattern {
            primary: "show templates".to_string(),
            variations: vec![
                "templates".to_string(),
                "setup templates".to_string(),
                "model setups".to_string(),
                "show setups".to_string(),
            ],
            action: CommandAction::ShowTemplates,
            help: "Show available setup templates".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup lightning fast".to_string(),
            variations: vec![
                "quick setup".to_string(),
                "fast setup".to_string(),
                "setup free".to_string(),
                "lightning".to_string(),
            ],
            action: CommandAction::LaunchTemplate("lightning_fast".to_string()),
            help: "Quick setup with local models only".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup balanced pro".to_string(),
            variations: vec![
                "balanced".to_string(),
                "setup balanced".to_string(),
                "pro setup".to_string(),
            ],
            action: CommandAction::LaunchTemplate("balanced_pro".to_string()),
            help: "Balanced setup with local and API models".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup premium quality".to_string(),
            variations: vec![
                "premium".to_string(),
                "quality".to_string(),
                "setup premium".to_string(),
                "best models".to_string(),
            ],
            action: CommandAction::LaunchTemplate("premium_quality".to_string()),
            help: "Premium setup with best available models".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup research beast".to_string(),
            variations: vec![
                "research".to_string(),
                "beast".to_string(),
                "ensemble".to_string(),
                "research mode".to_string(),
            ],
            action: CommandAction::LaunchTemplate("research_beast".to_string()),
            help: "Research setup with 5-model ensemble".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup code master".to_string(),
            variations: vec![
                "code completion".to_string(),
                "coding".to_string(),
                "setup coding".to_string(),
            ],
            action: CommandAction::LaunchTemplate("code_master".to_string()),
            help: "Code completion optimized setup".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup writing pro".to_string(),
            variations: vec![
                "writing".to_string(),
                "content creation".to_string(),
                "setup writing".to_string(),
            ],
            action: CommandAction::LaunchTemplate("writing_pro".to_string()),
            help: "Professional writing assistant setup".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "show sessions".to_string(),
            variations: vec![
                "sessions".to_string(),
                "active sessions".to_string(),
                "my sessions".to_string(),
                "running models".to_string(),
            ],
            action: CommandAction::ShowSessions,
            help: "Show active model sessions".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "setup api keys".to_string(),
            variations: vec![
                "api keys".to_string(),
                "configure api".to_string(),
                "setup keys".to_string(),
                "add api keys".to_string(),
            ],
            action: CommandAction::SetupApiKeys,
            help: "Configure API keys for external models".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "install model".to_string(),
            variations: vec![
                "download model".to_string(),
                "get model".to_string(),
                "add model".to_string(),
            ],
            action: CommandAction::InstallModel(String::new()),
            help: "Install a new local model".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "show costs".to_string(),
            variations: vec![
                "costs".to_string(),
                "cost analytics".to_string(),
                "usage costs".to_string(),
                "spending".to_string(),
            ],
            action: CommandAction::ShowModelCosts,
            help: "Show cost analytics and usage".to_string(),
        });

        // View commands
        self.add_command(CommandPattern {
            primary: "show dashboard".to_string(),
            variations: vec!["dashboard".to_string(), "home".to_string(), "overview".to_string()],
            action: CommandAction::SetView(ViewType::Dashboard),
            help: "Show main dashboard".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "show logs".to_string(),
            variations: vec![
                "logs".to_string(),
                "view logs".to_string(),
                "system logs".to_string(),
            ],
            action: CommandAction::ShowLogs,
            help: "Show system logs".to_string(),
        });

        // File system commands
        self.add_command(CommandPattern {
            primary: "create folder".to_string(),
            variations: vec![
                "make folder".to_string(),
                "new folder".to_string(),
                "create directory".to_string(),
                "make directory".to_string(),
                "mkdir".to_string(),
                "make a new folder".to_string(),
                "create a folder".to_string(),
            ],
            action: CommandAction::CreateFolder { path: String::new() },
            help: "Create a new folder/directory".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "navigate to".to_string(),
            variations: vec![
                "go to".to_string(),
                "cd to".to_string(),
                "change to".to_string(),
                "move to".to_string(),
                "open".to_string(),
            ],
            action: CommandAction::NavigateTo { path: String::new() },
            help: "Navigate to a directory".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "list files".to_string(),
            variations: vec![
                "show files".to_string(),
                "ls".to_string(),
                "dir".to_string(),
                "what files are here".to_string(),
                "show directory contents".to_string(),
            ],
            action: CommandAction::ListFiles { path: None },
            help: "List files in current or specified directory".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "delete file".to_string(),
            variations: vec![
                "remove file".to_string(),
                "rm".to_string(),
                "delete".to_string(),
                "remove".to_string(),
            ],
            action: CommandAction::DeleteFile { path: String::new() },
            help: "Delete a file or directory".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "current directory".to_string(),
            variations: vec![
                "where am i".to_string(),
                "pwd".to_string(),
                "current folder".to_string(),
                "show current directory".to_string(),
            ],
            action: CommandAction::ShowCurrentDirectory,
            help: "Show current working directory".to_string(),
        });

        // System commands
        self.add_command(CommandPattern {
            primary: "help".to_string(),
            variations: vec![
                "?".to_string(),
                "commands".to_string(),
                "what can you do".to_string(),
            ],
            action: CommandAction::Help,
            help: "Show available commands".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "quit".to_string(),
            variations: vec!["exit".to_string(), "bye".to_string(), "close".to_string()],
            action: CommandAction::Quit,
            help: "Exit the application".to_string(),
        });

        // Analytics commands
        self.add_command(CommandPattern {
            primary: "show analytics".to_string(),
            variations: vec![
                "analytics".to_string(),
                "show performance".to_string(),
                "performance metrics".to_string(),
                "show stats".to_string(),
            ],
            action: CommandAction::ShowAnalytics,
            help: "Show analytics and performance metrics".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "show cost analytics".to_string(),
            variations: vec![
                "cost analytics".to_string(),
                "cost metrics".to_string(),
                "show costs".to_string(),
                "cost dashboard".to_string(),
            ],
            action: CommandAction::ShowCostAnalytics,
            help: "Show cost analytics and usage metrics".to_string(),
        });

        // Collaborative commands
        self.add_command(CommandPattern {
            primary: "create session".to_string(),
            variations: vec![
                "new session".to_string(),
                "start session".to_string(),
                "create collaborative session".to_string(),
            ],
            action: CommandAction::CreateSession { name: String::new() },
            help: "Create a new collaborative session".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "list sessions".to_string(),
            variations: vec![
                "show sessions".to_string(),
                "active sessions".to_string(),
                "session list".to_string(),
            ],
            action: CommandAction::ListSessions,
            help: "List all active collaborative sessions".to_string(),
        });

        // Agent specialization commands
        self.add_command(CommandPattern {
            primary: "show agent specialization".to_string(),
            variations: vec![
                "agent specialization".to_string(),
                "show specialization".to_string(),
                "specialization dashboard".to_string(),
            ],
            action: CommandAction::ShowAgentSpecialization,
            help: "Show agent specialization and routing".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "show agent performance".to_string(),
            variations: vec![
                "agent performance".to_string(),
                "agent metrics".to_string(),
                "agent stats".to_string(),
            ],
            action: CommandAction::ShowAgentPerformance,
            help: "Show agent performance metrics".to_string(),
        });

        // Plugin ecosystem commands
        self.add_command(CommandPattern {
            primary: "show plugins".to_string(),
            variations: vec![
                "plugins".to_string(),
                "plugin list".to_string(),
                "installed plugins".to_string(),
            ],
            action: CommandAction::ShowPlugins,
            help: "Show installed plugins and plugin ecosystem".to_string(),
        });

        self.add_command(CommandPattern {
            primary: "install plugin".to_string(),
            variations: vec![
                "add plugin".to_string(),
                "enable plugin".to_string(),
                "load plugin".to_string(),
            ],
            action: CommandAction::InstallPlugin { name: String::new() },
            help: "Install a new plugin".to_string(),
        });

        // Evolution commands
        self.add_command(CommandPattern {
            primary: "show evolution".to_string(),
            variations: vec![
                "evolution".to_string(),
                "autonomous evolution".to_string(),
                "show adaptation".to_string(),
            ],
            action: CommandAction::ShowEvolution,
            help: "Show autonomous evolution and adaptation".to_string(),
        });

        // Consciousness commands
        self.add_command(CommandPattern {
            primary: "show consciousness".to_string(),
            variations: vec![
                "consciousness".to_string(),
                "distributed consciousness".to_string(),
                "cognitive state".to_string(),
            ],
            action: CommandAction::ShowConsciousness,
            help: "Show distributed consciousness system".to_string(),
        });

        // Training commands
        self.add_command(CommandPattern {
            primary: "show training".to_string(),
            variations: vec![
                "training".to_string(),
                "distributed training".to_string(),
                "model training".to_string(),
            ],
            action: CommandAction::ShowTraining,
            help: "Show distributed training system".to_string(),
        });

        // Anomaly detection commands
        self.add_command(CommandPattern {
            primary: "show anomalies".to_string(),
            variations: vec![
                "anomalies".to_string(),
                "anomaly detection".to_string(),
                "show health".to_string(),
                "health metrics".to_string(),
            ],
            action: CommandAction::ShowAnomalies,
            help: "Show anomaly detection and health metrics".to_string(),
        });

        // Quantum optimization commands
        self.add_command(CommandPattern {
            primary: "show quantum".to_string(),
            variations: vec![
                "quantum".to_string(),
                "quantum optimization".to_string(),
                "quantum algorithms".to_string(),
            ],
            action: CommandAction::ShowQuantumOptimization,
            help: "Show quantum optimization system".to_string(),
        });

        // Natural language commands
        self.add_command(CommandPattern {
            primary: "show natural language".to_string(),
            variations: vec![
                "natural language".to_string(),
                "nl interface".to_string(),
                "language interface".to_string(),
                "conversations".to_string(),
            ],
            action: CommandAction::ShowNaturalLanguage,
            help: "Show natural language interface".to_string(),
        });
    }

    fn add_command(&mut self, pattern: CommandPattern) {
        self.commands.insert(pattern.primary.clone(), pattern);
    }

    /// Parse natural language input into a command action with context
    /// awareness
    pub fn parse(&mut self, input: &str) -> Result<CommandAction> {
        let input_lower = input.trim().to_lowercase();

        // Update context memory
        self.update_context_memory(input);

        // Extract and store entities from input
        self.extract_and_store_entities(&input_lower);

        // Context-aware command resolution using memory
        if let Some(action) = self.resolve_with_context(&input_lower) {
            self.last_action = Some(action.clone());
            return Ok(action);
        }

        // Direct command matching
        if let Some(action) = self.match_direct_command(&input_lower) {
            self.last_action = Some(action.clone());
            return Ok(action);
        }

        // Fuzzy matching
        if let Some(action) = self.fuzzy_match_command(&input_lower) {
            self.last_action = Some(action.clone());
            return Ok(action);
        }

        // Try to extract intent and parameters
        if let Some(action) = self.extract_intent(&input_lower) {
            self.last_action = Some(action.clone());
            return Ok(action);
        }

        Err(anyhow::anyhow!(
            "I didn't understand '{}'. Try 'help' to see available commands. Context: {}",
            input,
            self.get_context_hint()
        ))
    }

    fn match_direct_command(&self, input: &str) -> Option<CommandAction> {
        for (_, pattern) in &self.commands {
            if input == pattern.primary {
                return Some(pattern.action.clone());
            }

            for variation in &pattern.variations {
                if input == variation {
                    return Some(pattern.action.clone());
                }
            }
        }
        None
    }

    fn fuzzy_match_command(&self, input: &str) -> Option<CommandAction> {
        let best_match = None;
        let best_score = 0;

        // for (_, pattern) in &self.commands {
        //     // Check primary command
        //     if let Some(score) = self.matcher.fuzzy_match(&pattern.primary, input) {
        //         if score > best_score {
        //             best_score = score;
        //             best_match = Some(pattern.action.clone());
        //         }
        //     }
        // 
        //     // Check variations
        //     for variation in &pattern.variations {
        //         if let Some(score) = self.matcher.fuzzy_match(variation, input) {
        //             if score > best_score {
        //                 best_score = score;
        //                 best_match = Some(pattern.action.clone());
        //             }
        //         }
        //     }
        // }

        // Only return if we have a good match
        if best_score > 50 { best_match } else { None }
    }

    fn extract_intent(&self, input: &str) -> Option<CommandAction> {
        let words: Vec<&str> = input.split_whitespace().collect();

        // Use advanced intent analysis with semantic understanding
        if let Some(advanced_action) = self.analyze_advanced_intent(input, &words) {
            return Some(advanced_action);
        }

        // File system operations with path extraction
        if (input.contains("create") || input.contains("make") || input.contains("new"))
            && (input.contains("folder") || input.contains("directory"))
        {
            if let Some(path) = self.extract_file_path(&words, input) {
                return Some(CommandAction::CreateFolder { path });
            }
            return Some(CommandAction::CreateFolder { path: String::new() });
        }

        if (input.contains("navigate") || input.contains("go to") || input.contains("cd"))
            || (input.contains("open") && !input.contains("file"))
        {
            if let Some(path) = self.extract_file_path(&words, input) {
                return Some(CommandAction::NavigateTo { path });
            }
        }

        if (input.contains("list") || input.contains("show")) && input.contains("files") {
            let path = self.extract_file_path(&words, input);
            return Some(CommandAction::ListFiles { path });
        }

        // Stream creation with name
        if (input.contains("create") || input.contains("start") || input.contains("new"))
            && input.contains("stream")
        {
            if let Some(name) = self.extract_stream_name(&words) {
                return Some(CommandAction::CreateStream { name });
            }
        }

        // Model deployment
        if (input.contains("deploy") || input.contains("load") || input.contains("add"))
            && input.contains("model")
        {
            if let Some(model) = self.extract_model_name(&words) {
                return Some(CommandAction::DeployModel { model, node: None });
            }
        }

        // Stop stream
        if input.contains("stop") && input.contains("stream") {
            if let Some(name) = self.extract_stream_name(&words) {
                return Some(CommandAction::StopStream(name));
            }
        }

        // Run task
        if input.contains("run") || input.contains("execute") {
            if let Some((task, target)) = self.extract_task_info(&words) {
                return Some(CommandAction::RunTask { task, target });
            }
        }

        None
    }

    fn extract_stream_name(&self, words: &[&str]) -> Option<String> {
        // Look for quoted name
        let input = words.join(" ");
        if let Some(start) = input.find('"') {
            if let Some(end) = input[start + 1..].find('"') {
                return Some(input[start + 1..start + 1 + end].to_string());
            }
        }

        // Look for "called" or "named"
        for (i, word) in words.iter().enumerate() {
            if (*word == "called" || *word == "named") && i + 1 < words.len() {
                return Some(words[i + 1..].join("-"));
            }
        }

        None
    }

    fn extract_model_name(&self, words: &[&str]) -> Option<String> {
        // Common model names
        let models = vec!["phi-3.5-mini", "codellama", "mistral", "deepseek"];

        for word in words {
            for model in &models {
                if word.contains(model) {
                    return Some(model.to_string());
                }
            }
        }

        None
    }

    fn extract_task_info(&self, words: &[&str]) -> Option<(String, String)> {
        let tasks = vec!["complete", "review", "document", "refactor", "test"];

        for task in tasks {
            if words.iter().any(|w| w.contains(task)) {
                // Extract target (file or directory)
                for (i, word) in words.iter().enumerate() {
                    if word.contains(".") || word.contains("/") {
                        return Some((task.to_string(), word.to_string()));
                    }
                    if *word == "on" && i + 1 < words.len() {
                        return Some((task.to_string(), words[i + 1].to_string()));
                    }
                }
                return Some((task.to_string(), ".".to_string()));
            }
        }

        None
    }

    fn extract_file_path(&self, words: &[&str], input: &str) -> Option<String> {
        // Look for quoted paths (highest priority)
        if let Some(start) = input.find('"') {
            if let Some(end) = input[start + 1..].find('"') {
                return Some(input[start + 1..start + 1 + end].to_string());
            }
        }
        if let Some(start) = input.find('\'') {
            if let Some(end) = input[start + 1..].find('\'') {
                return Some(input[start + 1..start + 1 + end].to_string());
            }
        }

        // Look for complex folder naming patterns like "called 'my project'"
        if let Some(called_pos) = input.find("called") {
            let after_called = &input[called_pos + 6..].trim();
            if let Some(start) = after_called.find('\'') {
                if let Some(end) = after_called[start + 1..].find('\'') {
                    return Some(after_called[start + 1..start + 1 + end].to_string());
                }
            }
            // Handle unquoted names after "called"
            let name_words: Vec<&str> = after_called.split_whitespace().collect();
            if !name_words.is_empty() {
                let folder_name = name_words[0..name_words.len().min(3)].join("-");
                return Some(folder_name);
            }
        }

        // Look for "named" patterns
        if let Some(named_pos) = input.find("named") {
            let after_named = &input[named_pos + 5..].trim();
            let name_words: Vec<&str> = after_named.split_whitespace().collect();
            if !name_words.is_empty() {
                let folder_name = name_words[0..name_words.len().min(3)].join("-");
                return Some(folder_name);
            }
        }

        // Enhanced path indicators with more context
        let path_indicators = vec!["in", "to", "at", "on", "inside", "within", "under"];
        for (i, word) in words.iter().enumerate() {
            if path_indicators.contains(word) && i + 1 < words.len() {
                // Handle "in the" patterns
                let start_idx =
                    if i + 1 < words.len() && words[i + 1] == "the" { i + 2 } else { i + 1 };

                if start_idx < words.len() {
                    let mut potential_path = words[start_idx..].join(" ");

                    // Stop at certain words that indicate end of path
                    let stop_words = vec!["directory", "folder", "to", "begin", "for"];
                    for stop_word in stop_words {
                        if let Some(pos) = potential_path.find(stop_word) {
                            potential_path = potential_path[..pos].trim().to_string();
                            break;
                        }
                    }

                    // Check if it's an actual path
                    if potential_path.contains("/")
                        || potential_path.contains("\\")
                        || potential_path.starts_with("~")
                        || potential_path.starts_with(".")
                    {
                        return Some(potential_path);
                    }

                    // Handle common directory names with more variations
                    let common_dirs = [
                        "desktop",
                        "documents",
                        "downloads",
                        "home",
                        "tmp",
                        "temp",
                        "pictures",
                        "music",
                        "videos",
                        "applications",
                        "library",
                        "development",
                        "projects",
                        "workspace",
                        "code",
                    ];

                    if common_dirs.contains(&potential_path.to_lowercase().as_str()) {
                        return Some(self.resolve_common_path(&potential_path));
                    }

                    // Return as is if it looks like a reasonable path name
                    if !potential_path.is_empty() && potential_path.len() < 50 {
                        return Some(potential_path);
                    }
                }
            }
        }

        // Look for direct path-like words (expanded patterns)
        for word in words {
            if word.contains("/")
                || word.contains("\\")
                || word.starts_with("~")
                || word.starts_with(".")
            {
                return Some(word.to_string());
            }
        }

        // Look for filesystem-like patterns manually
        // Check for Unix absolute paths
        for word in words {
            if word.starts_with('/') && word.len() > 1 {
                return Some(word.to_string());
            }
        }

        // Check for patterns in the full input text
        let input_chars: Vec<char> = input.chars().collect();
        let mut start_pos = None;

        for (i, &ch) in input_chars.iter().enumerate() {
            if ch == '/' || ch == '~' {
                start_pos = Some(i);
                break;
            }
        }

        if let Some(start) = start_pos {
            let mut end_pos = start + 1;
            for (i, &ch) in input_chars[start + 1..].iter().enumerate() {
                if ch.is_whitespace() || ch == '"' || ch == '\'' {
                    break;
                }
                end_pos = start + 1 + i + 1;
            }
            if end_pos > start + 1 {
                let path: String = input_chars[start..end_pos].iter().collect();
                return Some(path);
            }
        }

        None
    }

    /// Advanced intent analysis with semantic understanding and multi-pattern
    /// matching
    fn analyze_advanced_intent(&self, input: &str, words: &[&str]) -> Option<CommandAction> {
        let input_lower = input.to_lowercase();

        // Multi-step command analysis
        if let Some(action) = self.parse_multi_step_command(&input_lower, words) {
            return Some(action);
        }

        // Conditional and complex commands
        if let Some(action) = self.parse_conditional_command(&input_lower, words) {
            return Some(action);
        }

        // Context-aware commands
        if let Some(action) = self.parse_contextual_command(&input_lower, words) {
            return Some(action);
        }

        // Semantic role labeling for commands
        if let Some(action) = self.semantic_role_analysis(&input_lower, words) {
            return Some(action);
        }

        // Intent classification with confidence scoring
        if let Some(action) = self.classify_intent_with_confidence(&input_lower, words) {
            return Some(action);
        }

        None
    }

    /// Parse multi-step commands like "create folder test and navigate to it"
    fn parse_multi_step_command(&self, input: &str, _words: &[&str]) -> Option<CommandAction> {
        // Look for coordinating conjunctions
        let coordinators = ["and", "then", "after", "followed by", "next"];

        for coordinator in coordinators {
            if input.contains(coordinator) {
                let parts: Vec<&str> = input.split(coordinator).collect();
                if parts.len() >= 2 {
                    let first_part = parts[0].trim();
                    // For now, execute the first part - in future could return a MultiStep action
                    if let Some(action) = self.extract_intent_from_part(first_part) {
                        return Some(action);
                    }
                }
            }
        }

        None
    }

    /// Parse conditional commands like "if folder exists, delete it, otherwise
    /// create it"
    fn parse_conditional_command(&self, input: &str, _words: &[&str]) -> Option<CommandAction> {
        // Look for conditional patterns
        if input.contains("if") && (input.contains("exists") || input.contains("found")) {
            // Extract the main action from the conditional
            let _condition_keywords = ["if", "when", "unless"];
            let action_keywords = ["then", "else", "otherwise"];

            for action_word in action_keywords {
                if let Some(action_pos) = input.find(action_word) {
                    let action_part = &input[action_pos + action_word.len()..].trim();
                    if let Some(action) = self.extract_intent_from_part(action_part) {
                        return Some(action);
                    }
                }
            }
        }

        None
    }

    /// Parse context-aware commands that reference previous actions or state
    fn parse_contextual_command(&self, input: &str, words: &[&str]) -> Option<CommandAction> {
        // Commands that reference previous state
        let _context_patterns = [
            ("it", "that", "this", "the same"),
            ("previous", "last", "recent", "prior"),
            ("current", "working", "active", "present"),
        ];

        // Look for contextual references
        if words.iter().any(|&word| word == "it" || word == "that" || word == "this") {
            // Infer action based on context - for now use common defaults
            if input.contains("delete") || input.contains("remove") {
                return Some(CommandAction::DeleteFile { path: ".".to_string() });
            }
            if input.contains("open") || input.contains("navigate") {
                return Some(CommandAction::NavigateTo { path: ".".to_string() });
            }
        }

        None
    }

    /// Semantic role labeling to identify actions, objects, and modifiers
    fn semantic_role_analysis(&self, input: &str, words: &[&str]) -> Option<CommandAction> {
        // Define semantic roles
        let action_verbs = [
            ("create", "make", "build", "generate"),
            ("delete", "remove", "erase", "destroy"),
            ("copy", "duplicate", "clone", "replicate"),
            ("move", "transfer", "relocate", "shift"),
            ("list", "show", "display", "enumerate"),
            ("find", "search", "locate", "discover"),
        ];

        let objects = [
            ("folder", "directory", "dir", "path"),
            ("file", "document", "doc", "item"),
            ("project", "workspace", "repository", "repo"),
        ];

        // Find primary action
        let mut primary_action = None;
        for (main_verb, syn1, syn2, syn3) in action_verbs.iter() {
            if input.contains(main_verb)
                || input.contains(syn1)
                || input.contains(syn2)
                || input.contains(syn3)
            {
                primary_action = Some(*main_verb);
                break;
            }
        }

        // Find primary object
        let mut primary_object = None;
        for (main_obj, syn1, syn2, syn3) in objects.iter() {
            if input.contains(main_obj)
                || input.contains(syn1)
                || input.contains(syn2)
                || input.contains(syn3)
            {
                primary_object = Some(*main_obj);
                break;
            }
        }

        // Combine action and object to determine command
        match (primary_action, primary_object) {
            (Some("create"), Some("folder" | "directory")) => {
                let path = self
                    .extract_file_path(words, input)
                    .unwrap_or_else(|| "new_folder".to_string());
                Some(CommandAction::CreateFolder { path })
            }
            (Some("delete"), Some("folder" | "directory")) => {
                let path = self.extract_file_path(words, input).unwrap_or_else(|| ".".to_string());
                Some(CommandAction::DeleteFile { path })
            }
            (Some("list"), Some("file")) => {
                let path = self.extract_file_path(words, input);
                Some(CommandAction::ListFiles { path })
            }
            _ => None,
        }
    }

    /// Intent classification with confidence scoring
    fn classify_intent_with_confidence(
        &self,
        input: &str,
        words: &[&str],
    ) -> Option<CommandAction> {
        let mut intent_scores = std::collections::HashMap::new();

        // Score different intents based on keyword presence and patterns

        // File system operations
        let fs_keywords =
            ["file", "folder", "directory", "path", "create", "delete", "copy", "move"];
        let fs_score = fs_keywords
            .iter()
            .map(|&keyword| if input.contains(keyword) { 1.0 } else { 0.0 })
            .sum::<f32>()
            / fs_keywords.len() as f32;
        intent_scores.insert("file_system", fs_score);

        // Stream operations
        let stream_keywords = ["stream", "pipeline", "flow", "process", "buffer"];
        let stream_score = stream_keywords
            .iter()
            .map(|&keyword| if input.contains(keyword) { 1.0 } else { 0.0 })
            .sum::<f32>()
            / stream_keywords.len() as f32;
        intent_scores.insert("stream", stream_score);

        // Model operations
        let model_keywords = ["model", "ai", "llm", "generate", "complete", "chat"];
        let model_score = model_keywords
            .iter()
            .map(|&keyword| if input.contains(keyword) { 1.0 } else { 0.0 })
            .sum::<f32>()
            / model_keywords.len() as f32;
        intent_scores.insert("model", model_score);

        // System operations
        let system_keywords = ["system", "status", "metrics", "performance", "memory", "cpu"];
        let system_score = system_keywords
            .iter()
            .map(|&keyword| if input.contains(keyword) { 1.0 } else { 0.0 })
            .sum::<f32>()
            / system_keywords.len() as f32;
        intent_scores.insert("system", system_score);

        // Find highest scoring intent (with minimum threshold)
        let threshold = 0.15;
        if let Some((best_intent, &score)) = intent_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if score > threshold {
                return self.intent_to_action(best_intent, input, words);
            }
        }

        None
    }

    /// Convert intent classification to specific command action
    fn intent_to_action(&self, intent: &str, input: &str, words: &[&str]) -> Option<CommandAction> {
        match intent {
            "file_system" => {
                if input.contains("create")
                    && (input.contains("folder") || input.contains("directory"))
                {
                    let path = self
                        .extract_file_path(words, input)
                        .unwrap_or_else(|| "new_folder".to_string());
                    Some(CommandAction::CreateFolder { path })
                } else if input.contains("list") || input.contains("show") {
                    let path = self.extract_file_path(words, input);
                    Some(CommandAction::ListFiles { path })
                } else {
                    Some(CommandAction::ShowCurrentDirectory)
                }
            }
            "stream" => {
                if input.contains("create") || input.contains("new") {
                    let name =
                        self.extract_stream_name(words).unwrap_or_else(|| "new_stream".to_string());
                    Some(CommandAction::CreateStream { name })
                } else {
                    Some(CommandAction::ListStreams)
                }
            }
            "model" => Some(CommandAction::ListModels),
            "system" => Some(CommandAction::ShowMetrics),
            _ => None,
        }
    }

    /// Extract intent from a part of a multi-step command
    fn extract_intent_from_part(&self, part: &str) -> Option<CommandAction> {
        let words: Vec<&str> = part.split_whitespace().collect();

        // Simplified intent extraction for command parts
        if (part.contains("create") || part.contains("make"))
            && (part.contains("folder") || part.contains("directory"))
        {
            let path =
                self.extract_file_path(&words, part).unwrap_or_else(|| "new_folder".to_string());
            return Some(CommandAction::CreateFolder { path });
        }

        if part.contains("navigate") || part.contains("go to") || part.contains("cd") {
            let path = self.extract_file_path(&words, part).unwrap_or_else(|| ".".to_string());
            return Some(CommandAction::NavigateTo { path });
        }

        if part.contains("list") && part.contains("files") {
            let path = self.extract_file_path(&words, part);
            return Some(CommandAction::ListFiles { path });
        }

        None
    }

    /// Update context memory with recent input
    fn update_context_memory(&mut self, input: &str) {
        self.context_memory.push(input.to_string());
        if self.context_memory.len() > 10 {
            self.context_memory.remove(0);
        }
    }

    /// Extract and store entities (file names, paths, etc.) from input
    fn extract_and_store_entities(&mut self, input: &str) {
        // Extract file/folder names
        let words: Vec<&str> = input.split_whitespace().collect();

        // Look for potential file/folder names
        for (i, word) in words.iter().enumerate() {
            if (*word == "folder" || *word == "file" || *word == "directory") && i > 0 {
                if let Some(prev_word) = words.get(i - 1) {
                    if !["a", "the", "this", "that", "new", "create", "make"].contains(prev_word) {
                        self.session_entities
                            .insert("last_file".to_string(), prev_word.to_string());
                    }
                }
            }

            // Look for quoted names
            if word.starts_with('"') || word.starts_with('\'') {
                let clean_name = word.trim_matches(|c| c == '"' || c == '\'');
                self.session_entities
                    .insert("last_quoted_name".to_string(), clean_name.to_string());
            }
        }

        // Extract paths
        if let Some(path) = self.extract_file_path(&words, input) {
            self.session_entities.insert("last_path".to_string(), path);
        }
    }

    /// Resolve commands using context and memory
    fn resolve_with_context(&self, input: &str) -> Option<CommandAction> {
        // Handle pronoun references
        if input.contains("it") || input.contains("that") || input.contains("this") {
            return self.resolve_pronoun_reference(input);
        }

        // Handle relative commands
        if input.contains("same") || input.contains("again") || input.contains("repeat") {
            return self.last_action.clone();
        }

        // Handle continuation commands
        if input.starts_with("and ") || input.starts_with("then ") || input.starts_with("also ") {
            let continuation_part = if input.starts_with("and ") {
                &input[4..]
            } else if input.starts_with("then ") {
                &input[5..]
            } else {
                &input[5..] // "also "
            };

            return self.extract_intent_from_part(continuation_part);
        }

        // Handle contextual file references
        if input.contains("the file") || input.contains("the folder") {
            return self.resolve_contextual_file_reference(input);
        }

        None
    }

    /// Resolve pronoun references like "delete it", "open that"
    fn resolve_pronoun_reference(&self, input: &str) -> Option<CommandAction> {
        // Get the most recent file/path from context
        let target_path = self
            .session_entities
            .get("last_path")
            .or_else(|| self.session_entities.get("last_quoted_name"))
            .or_else(|| self.session_entities.get("last_file"))
            .cloned()
            .unwrap_or_else(|| ".".to_string());

        if input.contains("delete") || input.contains("remove") {
            return Some(CommandAction::DeleteFile { path: target_path });
        }

        if input.contains("open") || input.contains("navigate") || input.contains("go") {
            return Some(CommandAction::NavigateTo { path: target_path });
        }

        if input.contains("list") || input.contains("show") {
            return Some(CommandAction::ListFiles { path: Some(target_path) });
        }

        if input.contains("copy") {
            // Need to infer target - use current directory as default
            return Some(CommandAction::NavigateTo { path: target_path });
        }

        None
    }

    /// Resolve contextual file references
    fn resolve_contextual_file_reference(&self, input: &str) -> Option<CommandAction> {
        if let Some(file_name) = self.session_entities.get("last_file") {
            let path = file_name.clone();

            if input.contains("open") || input.contains("edit") {
                return Some(CommandAction::NavigateTo { path });
            }

            if input.contains("delete") || input.contains("remove") {
                return Some(CommandAction::DeleteFile { path });
            }
        }

        None
    }

    /// Get context hint for error messages
    fn get_context_hint(&self) -> String {
        if let Some(last_cmd) = self.context_memory.last() {
            format!("Last command: '{}'", last_cmd)
        } else if !self.session_entities.is_empty() {
            let entities: Vec<String> =
                self.session_entities.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            format!("Session context: {}", entities.join(", "))
        } else {
            "No context available".to_string()
        }
    }

    /// Clear session context (useful for new conversations)
    pub fn clear_context(&mut self) {
        self.context_memory.clear();
        self.session_entities.clear();
        self.last_action = None;
    }

    /// Get current session entities for debugging
    pub fn get_session_entities(&self) -> &HashMap<String, String> {
        &self.session_entities
    }

    fn resolve_common_path(&self, path_name: &str) -> String {
        match path_name.to_lowercase().as_str() {
            "desktop" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Desktop", home)
                } else {
                    "/tmp/Desktop".to_string()
                }
            }
            "documents" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Documents", home)
                } else {
                    "/tmp/Documents".to_string()
                }
            }
            "downloads" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Downloads", home)
                } else {
                    "/tmp/Downloads".to_string()
                }
            }
            "pictures" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Pictures", home)
                } else {
                    "/tmp/Pictures".to_string()
                }
            }
            "music" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Music", home)
                } else {
                    "/tmp/Music".to_string()
                }
            }
            "videos" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Videos", home)
                } else {
                    "/tmp/Videos".to_string()
                }
            }
            "applications" => {
                if cfg!(target_os = "macos") {
                    "/Applications".to_string()
                } else {
                    "/usr/bin".to_string()
                }
            }
            "library" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Library", home)
                } else {
                    "/usr/lib".to_string()
                }
            }
            "development" | "projects" | "workspace" | "code" => {
                if let Some(home) = std::env::var("HOME").ok() {
                    format!("{}/Development", home)
                } else {
                    "/tmp/Development".to_string()
                }
            }
            "home" => std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()),
            "tmp" | "temp" => "/tmp".to_string(),
            _ => path_name.to_string(),
        }
    }

    /// Get suggestions based on partial input
    pub fn get_suggestions(&self, partial: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        let partial_lower = partial.to_lowercase();

        for (_, pattern) in &self.commands {
            if pattern.primary.starts_with(&partial_lower) {
                suggestions.push(pattern.primary.clone());
            }
        }

        suggestions.sort();
        suggestions.truncate(5);
        suggestions
    }

    /// Get help text for all commands
    pub fn get_help(&self) -> Vec<(String, String)> {
        let mut help_items: Vec<_> =
            self.commands.values().map(|p| (p.primary.clone(), p.help.clone())).collect();

        help_items.sort_by(|a, b| a.0.cmp(&b.0));
        help_items
    }
}
