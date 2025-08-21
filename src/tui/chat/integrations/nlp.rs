//! NLP orchestrator integration
//! 
//! Connects chat to the NaturalLanguageOrchestrator

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::{Result, Context};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use regex::Regex;

use crate::tui::nlp::core::orchestrator::{NaturalLanguageOrchestrator, OrchestratorConfig};
use crate::cognitive::CognitiveSystem;
use crate::memory::CognitiveMemory;
use crate::models::orchestrator::ModelOrchestrator;
use crate::models::multi_agent_orchestrator::MultiAgentOrchestrator;
use crate::tools::IntelligentToolManager;
use crate::tools::task_management::TaskManager;
use crate::mcp::McpClient;
use crate::safety::ActionValidator;

/// NLP processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlpResult {
    /// Original message
    pub original: String,
    
    /// Processed/enhanced message
    pub processed: String,
    
    /// Detected intent
    pub intent: Option<String>,
    
    /// Extracted entities
    pub entities: HashMap<String, String>,
    
    /// Sentiment score (-1.0 to 1.0)
    pub sentiment: f32,
    
    /// Suggested actions
    pub suggested_actions: Vec<String>,
    
    /// Confidence score
    pub confidence: f32,
}

/// NLP integration for chat
pub struct NlpIntegration {
    /// The NLP orchestrator (optional, as it requires many dependencies)
    orchestrator: Option<Arc<NaturalLanguageOrchestrator>>,
    
    /// Processing history
    history: Arc<RwLock<Vec<NlpResult>>>,
    
    /// Basic NLP capabilities (when orchestrator not available)
    basic_nlp: BasicNlpProcessor,
}

/// Basic NLP processor for when orchestrator is not available
struct BasicNlpProcessor {
    /// Common patterns for intent detection
    intent_patterns: HashMap<String, Vec<String>>,
    
    /// Entity patterns
    entity_patterns: HashMap<String, regex::Regex>,
}

impl NlpIntegration {
    /// Create new NLP integration without orchestrator
    pub fn new() -> Self {
        let mut intent_patterns = HashMap::new();
        intent_patterns.insert("question".to_string(), vec![
            "what".to_string(), "how".to_string(), "why".to_string(), 
            "when".to_string(), "where".to_string(), "who".to_string(),
            "?".to_string(),
        ]);
        intent_patterns.insert("command".to_string(), vec![
            "create".to_string(), "make".to_string(), "build".to_string(),
            "run".to_string(), "execute".to_string(), "start".to_string(),
            "stop".to_string(), "delete".to_string(), "remove".to_string(),
            "compile".to_string(), "test".to_string(), "deploy".to_string(),
            "install".to_string(), "update".to_string(), "fix".to_string(),
        ]);
        intent_patterns.insert("request".to_string(), vec![
            "please".to_string(), "could".to_string(), "would".to_string(),
            "can you".to_string(), "help".to_string(), "assist".to_string(),
        ]);
        intent_patterns.insert("analysis".to_string(), vec![
            "analyze".to_string(), "explain".to_string(), "describe".to_string(),
            "understand".to_string(), "review".to_string(), "check".to_string(),
            "inspect".to_string(), "examine".to_string(), "evaluate".to_string(),
        ]);
        intent_patterns.insert("search".to_string(), vec![
            "find".to_string(), "search".to_string(), "look for".to_string(),
            "locate".to_string(), "grep".to_string(), "where is".to_string(),
        ]);
        
        let mut entity_patterns = HashMap::new();
        // Use lazy_static or once_cell for these in production, but for now handle errors
        if let Ok(email_regex) = regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b") {
            entity_patterns.insert("email".to_string(), email_regex);
        } else {
            tracing::warn!("Failed to compile email regex pattern");
        }
        
        if let Ok(url_regex) = regex::Regex::new(r"https?://[^\s]+") {
            entity_patterns.insert("url".to_string(), url_regex);
        } else {
            tracing::warn!("Failed to compile URL regex pattern");
        }
        
        if let Ok(number_regex) = regex::Regex::new(r"\b\d+(\.\d+)?\b") {
            entity_patterns.insert("number".to_string(), number_regex);
        } else {
            tracing::warn!("Failed to compile number regex pattern");
        }
        
        Self {
            orchestrator: None,
            history: Arc::new(RwLock::new(Vec::new())),
            basic_nlp: BasicNlpProcessor {
                intent_patterns,
                entity_patterns,
            },
        }
    }
    
    /// Create with full NLP orchestrator
    pub async fn with_orchestrator(
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        model_orchestrator: Arc<ModelOrchestrator>,
        multi_agent_orchestrator: Arc<MultiAgentOrchestrator>,
        tool_manager: Arc<IntelligentToolManager>,
        task_manager: Arc<TaskManager>,
        mcp_client: Arc<McpClient>,
        safety_validator: Arc<ActionValidator>,
    ) -> Result<Self> {
        let config = OrchestratorConfig::default();
        
        let orchestrator = NaturalLanguageOrchestrator::new(
            cognitive_system,
            memory,
            model_orchestrator,
            multi_agent_orchestrator,
            tool_manager,
            task_manager,
            mcp_client,
            safety_validator,
            config,
        ).await?;
        
        let mut integration = Self::new();
        integration.orchestrator = Some(Arc::new(orchestrator));
        
        Ok(integration)
    }
    
    /// Process a message through NLP
    pub async fn process_message(&self, message: &str) -> Result<NlpResult> {
        // Use orchestrator if available
        if let Some(orchestrator) = &self.orchestrator {
            // Full NLP processing through orchestrator
            match orchestrator.process_message(message, None).await {
                Ok(response) => {
                    let result = NlpResult {
                        original: message.to_string(),
                        processed: response.clone(),
                        intent: Some("orchestrated".to_string()),
                        entities: HashMap::new(),
                        sentiment: 0.0,
                        suggested_actions: vec![],
                        confidence: 0.9,
                    };
                    
                    self.add_to_history(result.clone()).await;
                    Ok(result)
                }
                Err(e) => {
                    tracing::warn!("Orchestrator processing failed, using basic NLP: {}", e);
                    self.basic_process(message).await
                }
            }
        } else {
            // Basic NLP processing
            self.basic_process(message).await
        }
    }
    
    /// Basic NLP processing without orchestrator
    async fn basic_process(&self, message: &str) -> Result<NlpResult> {
        let message_lower = message.to_lowercase();
        
        // Detect intent
        let mut detected_intent = None;
        let mut max_score = 0;
        
        for (intent, patterns) in &self.basic_nlp.intent_patterns {
            let score = patterns.iter()
                .filter(|p| message_lower.contains(p.as_str()))
                .count();
            
            if score > max_score {
                max_score = score;
                detected_intent = Some(intent.clone());
            }
        }
        
        // Extract entities
        let mut entities = HashMap::new();
        for (entity_type, pattern) in &self.basic_nlp.entity_patterns {
            if let Some(m) = pattern.find(message) {
                entities.insert(entity_type.clone(), m.as_str().to_string());
            }
        }
        
        // Basic sentiment analysis
        let sentiment = self.analyze_sentiment(&message_lower);
        
        // Suggest actions based on intent
        let suggested_actions = match detected_intent.as_deref() {
            Some("question") => vec![
                "Search for answer".to_string(),
                "Consult knowledge base".to_string(),
                "Provide detailed explanation".to_string(),
            ],
            Some("command") => vec![
                "Execute command".to_string(),
                "Validate parameters".to_string(),
                "Check permissions".to_string(),
            ],
            Some("request") => vec![
                "Process request".to_string(),
                "Gather requirements".to_string(),
                "Confirm understanding".to_string(),
            ],
            Some("analysis") => vec![
                "Perform analysis".to_string(),
                "Generate report".to_string(),
                "Visualize data".to_string(),
            ],
            Some("search") => vec![
                "Search codebase".to_string(),
                "Query database".to_string(),
                "Check documentation".to_string(),
            ],
            _ => vec!["Clarify intent".to_string()],
        };
        
        // Create result
        let result = NlpResult {
            original: message.to_string(),
            processed: self.enhance_message(message, &detected_intent),
            intent: detected_intent,
            entities,
            sentiment,
            suggested_actions,
            confidence: 0.6 + (max_score as f32 * 0.1),
        };
        
        self.add_to_history(result.clone()).await;
        Ok(result)
    }
    
    /// Basic sentiment analysis
    fn analyze_sentiment(&self, text: &str) -> f32 {
        let positive_words = ["good", "great", "excellent", "happy", "love", "wonderful", "amazing"];
        let negative_words = ["bad", "terrible", "hate", "awful", "horrible", "poor", "worst"];
        
        let positive_count = positive_words.iter()
            .filter(|&&word| text.contains(word))
            .count() as f32;
        
        let negative_count = negative_words.iter()
            .filter(|&&word| text.contains(word))
            .count() as f32;
        
        let total = positive_count + negative_count;
        if total == 0.0 {
            0.0
        } else {
            (positive_count - negative_count) / total
        }
    }
    
    /// Enhance message based on intent
    fn enhance_message(&self, message: &str, intent: &Option<String>) -> String {
        match intent.as_deref() {
            Some("question") => format!("Query: {}", message),
            Some("command") => format!("Execute: {}", message),
            Some("request") => format!("Request: {}", message),
            _ => message.to_string(),
        }
    }
    
    /// Add result to history
    async fn add_to_history(&self, result: NlpResult) {
        let mut history = self.history.write().await;
        history.push(result);
        
        // Keep only last 100 results
        if history.len() > 100 {
            let excess = history.len() - 100;
            history.drain(0..excess);
        }
    }
    
    /// Get processing history
    pub async fn get_history(&self, limit: usize) -> Vec<NlpResult> {
        let history = self.history.read().await;
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Check if orchestrator is available
    pub fn has_orchestrator(&self) -> bool {
        self.orchestrator.is_some()
    }
}