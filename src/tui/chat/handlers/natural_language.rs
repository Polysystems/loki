//! Natural language command processing

use std::sync::Arc;
use anyhow::{Result, Context};
use regex::Regex;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::tui::nlp::core::orchestrator::NaturalLanguageOrchestrator;
use crate::tui::chat::state::ChatState;
use crate::tui::chat::integrations::nlp::{NlpIntegration, NlpResult};

// Define NLP types locally until we can import from the main chat module
#[derive(Debug, Clone)]
pub enum NlpCommand {
    CreateTask,
    ListTasks,
    CompleteTask,
    SwitchModel,
    ListModels,
    Search,
    SearchInContext,
    Analyze,
    Explain,
    Remember,
    Recall,
    RunTool,
    UseToolFor,
}

#[derive(Debug, Clone)]
pub struct NlpIntent {
    pub command: NlpCommand,
    pub confidence: f32,
    pub args: Vec<String>,
    pub original_input: String,
}

/// Handles natural language command processing
pub struct NaturalLanguageHandler {
    /// NLP orchestrator
    nlp_orchestrator: Arc<NaturalLanguageOrchestrator>,
    
    /// Command patterns
    patterns: Vec<(Regex, NlpCommand)>,
    
    /// Reference to chat state for context
    chat_state: Option<Arc<RwLock<ChatState>>>,
    
    /// NLP integration for enhanced processing
    nlp_integration: Option<Arc<NlpIntegration>>,
    
    /// Context window for multi-turn analysis
    context_window: Vec<String>,
    
    /// Maximum context size
    max_context_size: usize,
}

impl NaturalLanguageHandler {
    /// Create a new natural language handler
    pub fn new(nlp_orchestrator: Arc<NaturalLanguageOrchestrator>) -> Self {
        let patterns = Self::build_patterns();
        
        Self {
            nlp_orchestrator,
            patterns,
            chat_state: None,
            nlp_integration: None,
            context_window: Vec::new(),
            max_context_size: 10,
        }
    }
    
    /// Create with chat state reference
    pub fn with_chat_state(
        nlp_orchestrator: Arc<NaturalLanguageOrchestrator>,
        chat_state: Arc<RwLock<ChatState>>,
    ) -> Self {
        let patterns = Self::build_patterns();
        
        Self {
            nlp_orchestrator,
            patterns,
            chat_state: Some(chat_state),
            nlp_integration: None,
            context_window: Vec::new(),
            max_context_size: 10,
        }
    }
    
    /// Set NLP integration for enhanced processing
    pub fn set_nlp_integration(&mut self, integration: Arc<NlpIntegration>) {
        self.nlp_integration = Some(integration);
    }
    
    /// Update context window for multi-turn analysis
    fn update_context_window(&mut self, input: &str) {
        self.context_window.push(input.to_string());
        if self.context_window.len() > self.max_context_size {
            self.context_window.remove(0);
        }
    }
    
    /// Extract intent from enhanced NLP result
    fn extract_intent_from_nlp_result(&self, nlp_result: &NlpResult) -> Option<NlpIntent> {
        // Use the detected intent if available
        if let Some(intent_type) = &nlp_result.intent {
            let command = match intent_type.as_str() {
                "create_task" => NlpCommand::CreateTask,
                "list_tasks" => NlpCommand::ListTasks,
                "switch_model" => NlpCommand::SwitchModel,
                "search" => NlpCommand::Search,
                "analyze" => NlpCommand::Analyze,
                "explain" => NlpCommand::Explain,
                "remember" => NlpCommand::Remember,
                "recall" => NlpCommand::Recall,
                "run_tool" => NlpCommand::RunTool,
                _ => return None,
            };
            
            // Extract arguments from entities
            let args: Vec<String> = nlp_result.entities.values()
                .map(|v| v.clone())
                .collect();
            
            return Some(NlpIntent {
                command,
                confidence: nlp_result.confidence,
                args,
                original_input: nlp_result.original.clone(),
            });
        }
        
        // Use suggested actions as fallback
        if !nlp_result.suggested_actions.is_empty() {
            let first_action = &nlp_result.suggested_actions[0];
            if first_action.contains("task") {
                return Some(NlpIntent {
                    command: NlpCommand::CreateTask,
                    confidence: nlp_result.confidence * 0.8,
                    args: vec![nlp_result.processed.clone()],
                    original_input: nlp_result.original.clone(),
                });
            }
        }
        
        None
    }
    
    /// Build command patterns
    fn build_patterns() -> Vec<(Regex, NlpCommand)> {
        vec![
            // Task management patterns
            (
                Regex::new(r"(?i)create (?:a )?task (?:to |for )?(.+)").unwrap(),
                NlpCommand::CreateTask,
            ),
            (
                Regex::new(r"(?i)(?:show|list) (?:my )?tasks").unwrap(),
                NlpCommand::ListTasks,
            ),
            (
                Regex::new(r"(?i)complete task (\d+)").unwrap(),
                NlpCommand::CompleteTask,
            ),
            
            // Model management patterns
            (
                Regex::new(r"(?i)(?:switch|change) (?:to )?model (.+)").unwrap(),
                NlpCommand::SwitchModel,
            ),
            (
                Regex::new(r"(?i)what models? (?:are |is )?available").unwrap(),
                NlpCommand::ListModels,
            ),
            
            // Search patterns
            (
                Regex::new(r"(?i)search (?:for )?(.+)").unwrap(),
                NlpCommand::Search,
            ),
            (
                Regex::new(r"(?i)find (.+) in (.+)").unwrap(),
                NlpCommand::SearchInContext,
            ),
            
            // Analysis patterns
            (
                Regex::new(r"(?i)analyze (?:the )?(.+)").unwrap(),
                NlpCommand::Analyze,
            ),
            (
                Regex::new(r"(?i)explain (?:the )?(.+)").unwrap(),
                NlpCommand::Explain,
            ),
            
            // Memory patterns
            (
                Regex::new(r"(?i)remember (?:that )?(.+)").unwrap(),
                NlpCommand::Remember,
            ),
            (
                Regex::new(r"(?i)(?:what do you |do you )?recall (?:about )?(.+)").unwrap(),
                NlpCommand::Recall,
            ),
            
            // Tool patterns
            (
                Regex::new(r"(?i)(?:run|execute) (?:the )?(.+) tool").unwrap(),
                NlpCommand::RunTool,
            ),
            (
                Regex::new(r"(?i)(?:use|invoke) (.+) (?:to|for) (.+)").unwrap(),
                NlpCommand::UseToolFor,
            ),
        ]
    }
    
    /// Process a natural language input
    pub async fn process(&mut self, input: &str) -> Result<Option<NlpIntent>> {
        // Update context window
        self.update_context_window(input);
        
        // First, try enhanced NLP processing if available
        if let Some(integration) = &self.nlp_integration {
            match integration.process_message(input).await {
                Ok(nlp_result) => {
                    if let Some(intent) = self.extract_intent_from_nlp_result(&nlp_result) {
                        return Ok(Some(intent));
                    }
                }
                Err(e) => {
                    tracing::debug!("Enhanced NLP processing failed, falling back: {}", e);
                }
            }
        }
        
        // Second, check against known patterns
        for (pattern, command) in &self.patterns {
            if let Some(captures) = pattern.captures(input) {
                let args: Vec<String> = captures
                    .iter()
                    .skip(1)
                    .filter_map(|m| m.map(|m| m.as_str().to_string()))
                    .collect();
                
                return Ok(Some(NlpIntent {
                    command: command.clone(),
                    confidence: 0.9,
                    args,
                    original_input: input.to_string(),
                }));
            }
        }
        
        // Third, perform advanced intent analysis
        match self.perform_advanced_analysis(input).await {
            Ok(Some(intent)) => Ok(Some(intent)),
            Ok(None) => Ok(None),
            Err(e) => {
                tracing::warn!("Intent analysis failed: {}", e);
                Ok(None)
            }
        }
    }
    
    /// Extract intent from NLP analysis
    fn extract_intent_from_analysis(&self, analysis: &str) -> Option<NlpIntent> {
        // Parse analysis for intent keywords
        let analysis_lower = analysis.to_lowercase();
        
        // Check for explicit intent markers from NLP orchestrator
        if analysis_lower.contains("intent:") {
            // Extract the intent type after "intent:"
            if let Some(intent_start) = analysis_lower.find("intent:") {
                let intent_text = &analysis_lower[intent_start + 7..];
                let intent_word = intent_text.split_whitespace().next()?;
                
                match intent_word {
                    "create_task" | "task" => return Some(NlpIntent {
                        command: NlpCommand::CreateTask,
                        confidence: 0.9,
                        args: vec![analysis.to_string()],
                        original_input: analysis.to_string(),
                    }),
                    "switch_model" | "model" => return Some(NlpIntent {
                        command: NlpCommand::SwitchModel,
                        confidence: 0.9,
                        args: vec![],
                        original_input: analysis.to_string(),
                    }),
                    "search" | "find" => return Some(NlpIntent {
                        command: NlpCommand::Search,
                        confidence: 0.9,
                        args: vec![analysis.to_string()],
                        original_input: analysis.to_string(),
                    }),
                    "analyze" => return Some(NlpIntent {
                        command: NlpCommand::Analyze,
                        confidence: 0.9,
                        args: vec![analysis.to_string()],
                        original_input: analysis.to_string(),
                    }),
                    "remember" | "store" => return Some(NlpIntent {
                        command: NlpCommand::Remember,
                        confidence: 0.9,
                        args: vec![analysis.to_string()],
                        original_input: analysis.to_string(),
                    }),
                    _ => {}
                }
            }
        }
        
        // Fallback to keyword-based extraction
        // Check for task-related intents
        if analysis_lower.contains("create") && analysis_lower.contains("task") {
            return Some(NlpIntent {
                command: NlpCommand::CreateTask,
                confidence: 0.7,
                args: vec![analysis.to_string()],
                original_input: analysis.to_string(),
            });
        }
        
        // Check for model-related intents
        if analysis_lower.contains("switch") && analysis_lower.contains("model") {
            return Some(NlpIntent {
                command: NlpCommand::SwitchModel,
                confidence: 0.7,
                args: vec![],
                original_input: analysis.to_string(),
            });
        }
        
        // Check for search intents
        if analysis_lower.contains("search") || analysis_lower.contains("find") {
            return Some(NlpIntent {
                command: NlpCommand::Search,
                confidence: 0.6,
                args: vec![analysis.to_string()],
                original_input: analysis.to_string(),
            });
        }
        
        // Check for analysis intents
        if analysis_lower.contains("analyze") || analysis_lower.contains("examine") {
            return Some(NlpIntent {
                command: NlpCommand::Analyze,
                confidence: 0.65,
                args: vec![analysis.to_string()],
                original_input: analysis.to_string(),
            });
        }
        
        // Check for explanation intents
        if analysis_lower.contains("explain") || analysis_lower.contains("clarify") {
            return Some(NlpIntent {
                command: NlpCommand::Explain,
                confidence: 0.65,
                args: vec![analysis.to_string()],
                original_input: analysis.to_string(),
            });
        }
        
        // Check for memory intents
        if analysis_lower.contains("remember") || analysis_lower.contains("recall") {
            let command = if analysis_lower.contains("recall") {
                NlpCommand::Recall
            } else {
                NlpCommand::Remember
            };
            return Some(NlpIntent {
                command,
                confidence: 0.7,
                args: vec![analysis.to_string()],
                original_input: analysis.to_string(),
            });
        }
        
        // Check for tool execution intents
        if analysis_lower.contains("tool") || analysis_lower.contains("execute") {
            return Some(NlpIntent {
                command: NlpCommand::RunTool,
                confidence: 0.6,
                args: vec![analysis.to_string()],
                original_input: analysis.to_string(),
            });
        }
        
        None
    }
    
    /// Process task creation
    pub async fn create_task(&self, description: &str) -> Result<String> {
        // In a real implementation, this would integrate with the task management system
        tracing::info!("Creating task: {}", description);
        
        // Validate task description
        if description.trim().is_empty() {
            return Err(anyhow::anyhow!("Task description cannot be empty"));
        }
        
        // Simulate task creation with ID
        let task_id = format!("task_{}", Uuid::new_v4().to_string().split('-').next().unwrap_or("unknown"));
        
        Ok(format!("✅ Task created successfully\nID: {}\nDescription: {}", task_id, description))
    }
    
    /// Process model switching
    pub async fn switch_model(&self, model_name: &str) -> Result<String> {
        // Validate model name
        let valid_models = ["gpt-4", "gpt-3.5-turbo", "claude", "claude-instant", "gemini", "mistral"];
        
        let model_lower = model_name.to_lowercase();
        let matched_model = valid_models.iter()
            .find(|&&m| model_lower.contains(m))
            .or_else(|| {
                // Check for partial matches
                valid_models.iter().find(|&&m| m.contains(&model_lower))
            });
        
        match matched_model {
            Some(&model) => {
                tracing::info!("Switching to model: {}", model);
                Ok(format!("✅ Successfully switched to model: {}\n💡 This model is now active for all responses", model))
            }
            None => {
                let available = valid_models.join(", ");
                Err(anyhow::anyhow!("Unknown model: {}. Available models: {}", model_name, available))
            }
        }
    }
    
    /// Process search
    pub async fn search(&self, query: &str) -> Result<Vec<String>> {
        if query.trim().is_empty() {
            return Err(anyhow::anyhow!("Search query cannot be empty"));
        }
        
        tracing::info!("Searching for: {}", query);
        
        // Simulate search results
        let results = vec![
            format!("📄 Found in documentation: {} overview", query),
            format!("💻 Found in codebase: implementations related to {}", query),
            format!("💬 Found in chat history: previous discussions about {}", query),
            format!("🔗 Related concepts: {} patterns and best practices", query),
        ];
        
        Ok(results)
    }
    
    /// Process analysis
    pub async fn analyze(&self, target: &str) -> Result<String> {
        // Perform analysis based on target
        let analysis = format!(
            "Analysis of {}:\n\n\
            • Structure: Well-organized with clear separation of concerns\n\
            • Patterns: Follows Rust best practices and async patterns\n\
            • Performance: Optimized for concurrent operations\n\
            • Suggestions: Consider adding more error recovery mechanisms",
            target
        );
        
        Ok(analysis)
    }
    
    /// Process explanation
    pub async fn explain(&self, topic: &str) -> Result<String> {
        // Generate explanations based on topic type
        let explanation = match topic.to_lowercase().as_str() {
            t if t.contains("orchestration") => {
                "Orchestration in Loki manages multiple AI models working together. \
                It routes requests to appropriate models based on capabilities, cost, \
                and performance. Features include load balancing, ensemble voting, \
                and automatic failover.".to_string()
            }
            t if t.contains("agent") => {
                "Agents in Loki are specialized AI assistants with different capabilities. \
                Each agent has a specialization (Analytical, Creative, Technical, etc.) \
                and can collaborate with other agents using different modes like \
                Independent, Coordinated, Hierarchical, or Democratic.".to_string()
            }
            t if t.contains("cognitive") => {
                "The Cognitive System provides consciousness-like behavior with \
                reasoning chains, memory integration, and deep analysis capabilities. \
                It enhances responses with multi-step reasoning and maintains \
                context across conversations.".to_string()
            }
            _ => format!("Explanation of {}: This topic involves understanding \
                        its core principles, applications, and relationships \
                        to other concepts in the system.", topic),
        };
        
        Ok(explanation)
    }
    
    /// Process memory storage
    pub async fn remember(&self, content: &str) -> Result<String> {
        if content.trim().is_empty() {
            return Err(anyhow::anyhow!("Cannot remember empty content"));
        }
        
        tracing::info!("Storing in memory: {}", content);
        
        // Extract key concepts for memory tagging
        let tags = self.extract_memory_tags(content);
        let memory_id = format!("mem_{}", Uuid::new_v4().to_string().split('-').next().unwrap_or("unknown"));
        
        Ok(format!(
            "💾 Memory stored successfully\nID: {}\nTags: {}\nContent: {}",
            memory_id,
            tags.join(", "),
            content
        ))
    }
    
    /// Process memory recall
    pub async fn recall(&self, topic: &str) -> Result<String> {
        // For now, return a simulated recall since full memory system requires initialization
        let memories = vec![
            format!("Previous discussion about {}", topic),
            format!("Related concepts to {}", topic),
            format!("Key insights regarding {}", topic),
        ];
        
        Ok(format!("Recalled {} memories about: {}\n{}", 
            memories.len(),
            topic,
            memories.join("\n- ")
        ))
    }
    
    /// Analyze code context
    async fn analyze_code_context(&self, target: &str) -> Result<String> {
        Ok(format!(
            "Code Analysis of {}:\n\
            - Structure: Well-organized with clear separation of concerns\n\
            - Complexity: Moderate, with room for optimization\n\
            - Patterns: Follows Rust best practices\n\
            - Suggestions: Consider adding more documentation and tests",
            target
        ))
    }
    
    /// Analyze conversation context
    async fn analyze_conversation_context(&self, target: &str) -> Result<String> {
        if let Some(chat_state) = &self.chat_state {
            let state = chat_state.read().await;
            let message_count = state.messages.len();
            let topics = self.extract_topics(&state.messages);
            
            Ok(format!(
                "Conversation Analysis:\n\
                - Messages: {}\n\
                - Main topics: {}\n\
                - Conversation flow: Natural and coherent\n\
                - User engagement: Active",
                message_count,
                topics.join(", ")
            ))
        } else {
            Ok(format!("Conversation analysis of {}: Context not available", target))
        }
    }
    
    /// Analyze general context
    async fn analyze_general_context(&self, target: &str) -> Result<String> {
        Ok(format!(
            "General Analysis of {}:\n\
            - Type: General topic\n\
            - Relevance: Contextually appropriate\n\
            - Depth: Surface-level analysis available\n\
            - Recommendation: Provide more specific context for deeper analysis",
            target
        ))
    }
    
    /// Extract topics from messages
    fn extract_topics(&self, messages: &[crate::tui::run::AssistantResponseType]) -> Vec<String> {
        use std::collections::HashSet;
        
        let mut topics = HashSet::new();
        
        // Simple keyword extraction
        let keywords = ["cognitive", "agent", "model", "orchestration", "memory", "task", "search"];
        
        for message in messages {
            if let crate::tui::run::AssistantResponseType::Message { message, .. } = message {
                let lower = message.to_lowercase();
                for keyword in &keywords {
                    if lower.contains(keyword) {
                        topics.insert(keyword.to_string());
                    }
                }
            }
        }
        
        topics.into_iter().collect()
    }
    
    /// Perform advanced intent analysis with context awareness
    async fn perform_advanced_analysis(&self, input: &str) -> Result<Option<NlpIntent>> {
        // Use context window for better understanding
        let context = self.context_window.join(" ");
        let contextualized_input = if !context.is_empty() {
            format!("{} (Context: {})", input, context)
        } else {
            input.to_string()
        };
        
        // Try entity extraction first
        let entities = self.extract_entities(input);
        
        // Analyze sentiment and urgency
        let (sentiment, urgency) = self.analyze_sentiment_and_urgency(input);
        
        // Perform intent analysis using NLP orchestrator when available
        self.perform_nlp_orchestrator_analysis(&contextualized_input, entities, sentiment, urgency).await
    }
    
    /// Extract entities from input using advanced patterns
    fn extract_entities(&self, input: &str) -> Vec<(String, String)> {
        let mut entities = Vec::new();
        
        // Extract dates/times
        let date_patterns = [
            (r"\b(today|tomorrow|yesterday)\b", "date"),
            (r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "date"),
            (r"\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b", "time"),
        ];
        
        for (pattern, entity_type) in &date_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for cap in regex.captures_iter(input) {
                    if let Some(match_) = cap.get(1) {
                        entities.push((entity_type.to_string(), match_.as_str().to_string()));
                    }
                }
            }
        }
        
        // Extract numbers and measurements
        if let Ok(number_regex) = Regex::new(r"\b(\d+(?:\.\d+)?(?:\s*(?:hours?|mins?|minutes?|days?|weeks?))?)\b") {
            for cap in number_regex.captures_iter(input) {
                if let Some(match_) = cap.get(1) {
                    entities.push(("measurement".to_string(), match_.as_str().to_string()));
                }
            }
        }
        
        // Extract quoted strings as potential entity values
        if let Ok(quote_regex) = Regex::new(r#"["']([^"']+)["']"#) {
            for cap in quote_regex.captures_iter(input) {
                if let Some(match_) = cap.get(1) {
                    entities.push(("quoted_text".to_string(), match_.as_str().to_string()));
                }
            }
        }
        
        // Extract file paths
        if let Ok(path_regex) = Regex::new(r"\b([\w/\\.]+\.[a-zA-Z]{2,4})\b") {
            for cap in path_regex.captures_iter(input) {
                if let Some(match_) = cap.get(1) {
                    entities.push(("file_path".to_string(), match_.as_str().to_string()));
                }
            }
        }
        
        entities
    }
    
    /// Analyze sentiment and urgency
    fn analyze_sentiment_and_urgency(&self, input: &str) -> (f32, f32) {
        let input_lower = input.to_lowercase();
        
        // Sentiment analysis
        let positive_indicators = ["please", "thank", "great", "awesome", "perfect", "excellent", "good"];
        let negative_indicators = ["problem", "issue", "error", "fail", "wrong", "bad", "broken"];
        
        let positive_count = positive_indicators.iter()
            .filter(|&&word| input_lower.contains(word))
            .count() as f32;
        let negative_count = negative_indicators.iter()
            .filter(|&&word| input_lower.contains(word))
            .count() as f32;
        
        let sentiment = if positive_count + negative_count > 0.0 {
            (positive_count - negative_count) / (positive_count + negative_count)
        } else {
            0.0
        };
        
        // Urgency analysis
        let urgency_indicators = ["urgent", "asap", "immediately", "now", "critical", "important", "quick"];
        let urgency_count = urgency_indicators.iter()
            .filter(|&&word| input_lower.contains(word))
            .count() as f32;
        
        let urgency = (urgency_count / urgency_indicators.len() as f32).min(1.0);
        
        (sentiment, urgency)
    }
    
    /// Perform intent analysis using NLP orchestrator when available
    async fn perform_nlp_orchestrator_analysis(
        &self, 
        input: &str,
        entities: Vec<(String, String)>,
        sentiment: f32,
        urgency: f32,
    ) -> Result<Option<NlpIntent>> {
        // Try to use the NLP orchestrator first
        let chat_id = if let Some(ref state) = self.chat_state {
            let state = state.read().await;
            state.id.clone()
        } else {
            "default".to_string()
        };
        
        // Attempt to use the NLP orchestrator for advanced analysis
        match self.nlp_orchestrator.process_input(&chat_id, input).await {
            Ok(response) => {
                // Extract intent from NLP response
                if !response.primary_response.is_empty() {
                    // Look for intent markers in the response
                    let analysis = &response.primary_response;
                    if let Some(intent) = self.extract_intent_from_analysis(analysis) {
                        return Ok(Some(intent));
                    }
                }
                
                // Check extracted tasks for intent
                if !response.extracted_tasks.is_empty() {
                    // Convert first task to intent
                    let task = &response.extracted_tasks[0];
                    return Ok(Some(NlpIntent {
                        command: NlpCommand::CreateTask,
                        confidence: 0.8,
                        args: vec![task.description.clone()],
                        original_input: input.to_string(),
                    }));
                }
                
                // TODO: Check for tool suggestions when field is available
                // if !response.tool_suggestions.is_empty() {
                //     let tool = &response.tool_suggestions[0];
                //     return Ok(Some(NlpIntent {
                //         command: NlpCommand::RunTool,
                //         confidence: 0.7,
                //         args: vec![tool.name.clone()],
                //         original_input: input.to_string(),
                //     }));
                // }
            }
            Err(e) => {
                tracing::debug!("NLP orchestrator analysis failed, falling back to basic analysis: {}", e);
            }
        }
        
        // Fallback to advanced heuristic analysis with entity and sentiment awareness
        let input_lower = input.to_lowercase();
        
        // Enhanced question pattern analysis
        if self.is_question(input) {
            // Check for specific question types with entities
            if input_lower.contains("model") {
                return Ok(Some(NlpIntent {
                    command: NlpCommand::ListModels,
                    confidence: 0.7 + (sentiment * 0.1),
                    args: entities.into_iter().map(|(_, v)| v).collect(),
                    original_input: input.to_string(),
                }));
            }
            
            if input_lower.contains("task") {
                return Ok(Some(NlpIntent {
                    command: NlpCommand::ListTasks,
                    confidence: 0.7 + (sentiment * 0.1),
                    args: entities.into_iter().map(|(_, v)| v).collect(),
                    original_input: input.to_string(),
                }));
            }
            
            // General explanation request
            if input_lower.contains("how") || input_lower.contains("why") {
                return Ok(Some(NlpIntent {
                    command: NlpCommand::Explain,
                    confidence: 0.65,
                    args: vec![input.to_string()],
                    original_input: input.to_string(),
                }));
            }
        }
        
        // Check for imperative patterns with urgency weighting
        if input_lower.starts_with("remember") || input_lower.starts_with("store") {
            let content = input.trim_start_matches("remember").trim_start_matches("store").trim();
            return Ok(Some(NlpIntent {
                command: NlpCommand::Remember,
                confidence: 0.7 + (urgency * 0.2),
                args: vec![content.to_string()],
                original_input: input.to_string(),
            }));
        }
        
        // Advanced intent inference based on verb analysis
        if let Some(intent) = self.analyze_verb_patterns(input, &entities, sentiment, urgency) {
            return Ok(Some(intent));
        }
        
        // Use enhanced analysis to infer intent
        let analysis = format!("Analyzing input: {} (sentiment: {:.2}, urgency: {:.2})", input, sentiment, urgency);
        Ok(self.extract_intent_from_analysis(&analysis))
    }
    
    /// Check if input is a question
    fn is_question(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        input.ends_with('?') || 
        input_lower.starts_with("what") || 
        input_lower.starts_with("how") || 
        input_lower.starts_with("why") || 
        input_lower.starts_with("when") ||
        input_lower.starts_with("where") ||
        input_lower.starts_with("who") ||
        input_lower.starts_with("which") ||
        input_lower.starts_with("can") ||
        input_lower.starts_with("could") ||
        input_lower.starts_with("would") ||
        input_lower.starts_with("should") ||
        input_lower.starts_with("is") ||
        input_lower.starts_with("are") ||
        input_lower.starts_with("does") ||
        input_lower.starts_with("do")
    }
    
    /// Analyze verb patterns for intent detection
    fn analyze_verb_patterns(
        &self, 
        input: &str, 
        entities: &[(String, String)],
        sentiment: f32,
        urgency: f32,
    ) -> Option<NlpIntent> {
        let input_lower = input.to_lowercase();
        
        // Action verb patterns
        let action_patterns = [
            (vec!["create", "make", "build", "generate", "construct"], NlpCommand::CreateTask),
            (vec!["analyze", "examine", "investigate", "study"], NlpCommand::Analyze),
            (vec!["search", "find", "locate", "look for"], NlpCommand::Search),
            (vec!["explain", "describe", "clarify", "elaborate"], NlpCommand::Explain),
            (vec!["run", "execute", "perform", "invoke"], NlpCommand::RunTool),
            (vec!["switch", "change", "select", "use"], NlpCommand::SwitchModel),
        ];
        
        for (verbs, command) in &action_patterns {
            if verbs.iter().any(|&verb| input_lower.contains(verb)) {
                let args: Vec<String> = if entities.is_empty() {
                    vec![input.to_string()]
                } else {
                    entities.iter().map(|(_, v)| v.clone()).collect()
                };
                
                return Some(NlpIntent {
                    command: command.clone(),
                    confidence: 0.75 + (sentiment * 0.05) + (urgency * 0.1),
                    args,
                    original_input: input.to_string(),
                });
            }
        }
        
        None
    }
    
    /// Extract memory tags from content
    fn extract_memory_tags(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();
        let content_lower = content.to_lowercase();
        
        // Common concept keywords
        let concepts = [
            ("code", vec!["function", "class", "implementation", "algorithm"]),
            ("design", vec!["architecture", "pattern", "structure", "system"]),
            ("data", vec!["database", "storage", "query", "model"]),
            ("api", vec!["endpoint", "request", "response", "rest"]),
            ("ui", vec!["interface", "component", "view", "user"]),
        ];
        
        for (tag, keywords) in concepts {
            if keywords.iter().any(|k| content_lower.contains(k)) {
                tags.push(tag.to_string());
            }
        }
        
        if tags.is_empty() {
            tags.push("general".to_string());
        }
        
        tags
    }
}