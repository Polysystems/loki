//! Natural Language Tool Executor
//! 
//! Translates natural language commands into tool executions with intelligent
//! parameter extraction and intent recognition.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use anyhow::Result;
use tracing::{info, debug, warn};

use super::discovery::{ToolDiscoveryEngine, DiscoveredTool, ToolCategory};
use crate::tools::{IntelligentToolManager, intelligent_manager::ExecutionContext};
use crate::tui::nlp::core::processor::{NaturalLanguageProcessor, NLPResult};

/// Natural language tool executor
pub struct NLToolExecutor {
    /// Tool discovery engine
    discovery: Arc<ToolDiscoveryEngine>,
    
    /// Tool manager
    tool_manager: Arc<IntelligentToolManager>,
    
    /// NLP processor
    nlp: Arc<NaturalLanguageProcessor>,
    
    /// Intent patterns
    intent_patterns: Arc<RwLock<HashMap<String, IntentPattern>>>,
    
    /// Execution history
    history: Arc<RwLock<Vec<ExecutionRecord>>>,
    
    /// Configuration
    config: NLExecutorConfig,
}

/// NL executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLExecutorConfig {
    pub confidence_threshold: f32,
    pub max_suggestions: usize,
    pub auto_execute: bool,
    pub require_confirmation: bool,
    pub learn_from_corrections: bool,
    pub cache_interpretations: bool,
}

impl Default for NLExecutorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            max_suggestions: 3,
            auto_execute: false,
            require_confirmation: true,
            learn_from_corrections: true,
            cache_interpretations: true,
        }
    }
}

/// Intent pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPattern {
    pub id: String,
    pub patterns: Vec<String>,
    pub tool_id: String,
    pub parameter_extractors: Vec<ParameterExtractor>,
    pub examples: Vec<String>,
    pub confidence_boost: f32,
}

/// Parameter extractor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterExtractor {
    pub parameter_name: String,
    pub extraction_type: ExtractionType,
    pub patterns: Vec<String>,
    pub default_value: Option<Value>,
    pub required: bool,
}

/// Extraction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionType {
    Regex(String),
    Keyword(Vec<String>),
    Entity(EntityType),
    Position(usize),
    Custom(String),
}

/// Entity types for extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    FilePath,
    Url,
    Email,
    Number,
    Date,
    Time,
    Person,
    Location,
    Organization,
}

/// Tool execution interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInterpretation {
    pub tool_id: String,
    pub tool_name: String,
    pub confidence: f32,
    pub parameters: HashMap<String, Value>,
    pub explanation: String,
    pub alternatives: Vec<AlternativeInterpretation>,
}

/// Alternative interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeInterpretation {
    pub tool_id: String,
    pub tool_name: String,
    pub confidence: f32,
    pub explanation: String,
}

/// Execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub input: String,
    pub interpretation: ToolInterpretation,
    pub result: ExecutionResult,
    pub feedback: Option<UserFeedback>,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionResult {
    Success(Value),
    Error(String),
    Cancelled,
    RequiresConfirmation,
    Pending,
}

/// User feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub correct: bool,
    pub corrected_tool: Option<String>,
    pub corrected_parameters: Option<HashMap<String, Value>>,
    pub notes: Option<String>,
}

impl NLToolExecutor {
    /// Create a new NL tool executor
    pub fn new(
        discovery: Arc<ToolDiscoveryEngine>,
        tool_manager: Arc<IntelligentToolManager>,
        nlp: Arc<NaturalLanguageProcessor>,
    ) -> Self {
        let executor = Self {
            discovery,
            tool_manager,
            nlp,
            intent_patterns: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            config: NLExecutorConfig::default(),
        };
        
        // Initialize default patterns in the background
        let patterns = executor.intent_patterns.clone();
        tokio::spawn(async move {
            // TODO: Initialize patterns here without needing the full executor
            debug!("NL patterns initialization deferred");
        });
        
        executor
    }
    
    /// Initialize intent patterns
    async fn initialize_patterns(&self) -> Result<()> {
        let mut patterns = self.intent_patterns.write().await;
        
        // File operations
        patterns.insert("file_read".to_string(), IntentPattern {
            id: "file_read".to_string(),
            patterns: vec![
                "read file".to_string(),
                "show file".to_string(),
                "open file".to_string(),
                "display file".to_string(),
                "cat".to_string(),
            ],
            tool_id: "file_system".to_string(),
            parameter_extractors: vec![
                ParameterExtractor {
                    parameter_name: "path".to_string(),
                    extraction_type: ExtractionType::Entity(EntityType::FilePath),
                    patterns: vec![],
                    default_value: None,
                    required: true,
                },
                ParameterExtractor {
                    parameter_name: "operation".to_string(),
                    extraction_type: ExtractionType::Keyword(vec!["read".to_string()]),
                    patterns: vec![],
                    default_value: Some(Value::String("read".to_string())),
                    required: true,
                },
            ],
            examples: vec![
                "read file config.json".to_string(),
                "show me the contents of README.md".to_string(),
            ],
            confidence_boost: 0.1,
        });
        
        // Web search
        patterns.insert("web_search".to_string(), IntentPattern {
            id: "web_search".to_string(),
            patterns: vec![
                "search for".to_string(),
                "look up".to_string(),
                "find information about".to_string(),
                "google".to_string(),
                "search web".to_string(),
            ],
            tool_id: "web_search".to_string(),
            parameter_extractors: vec![
                ParameterExtractor {
                    parameter_name: "query".to_string(),
                    extraction_type: ExtractionType::Position(1),
                    patterns: vec![],
                    default_value: None,
                    required: true,
                },
            ],
            examples: vec![
                "search for rust programming tutorials".to_string(),
                "look up machine learning algorithms".to_string(),
            ],
            confidence_boost: 0.15,
        });
        
        // Code execution
        patterns.insert("code_run".to_string(), IntentPattern {
            id: "code_run".to_string(),
            patterns: vec![
                "run code".to_string(),
                "execute".to_string(),
                "eval".to_string(),
                "compile and run".to_string(),
            ],
            tool_id: "code_executor".to_string(),
            parameter_extractors: vec![
                ParameterExtractor {
                    parameter_name: "language".to_string(),
                    extraction_type: ExtractionType::Keyword(vec![
                        "python".to_string(),
                        "rust".to_string(),
                        "javascript".to_string(),
                        "typescript".to_string(),
                    ]),
                    patterns: vec![],
                    default_value: Some(Value::String("python".to_string())),
                    required: false,
                },
            ],
            examples: vec![
                "run this python code".to_string(),
                "execute the rust program".to_string(),
            ],
            confidence_boost: 0.1,
        });
        
        Ok(())
    }
    
    /// Interpret natural language command
    pub async fn interpret(&self, input: &str) -> Result<ToolInterpretation> {
        info!("Interpreting command: {}", input);
        
        // Process with NLP
        let nlp_result = self.nlp.process(input).await?;
        
        // Find matching tools
        let mut candidates = Vec::new();
        
        // Check intent patterns
        let patterns = self.intent_patterns.read().await;
        for (pattern_id, pattern) in patterns.iter() {
            let confidence = self.calculate_pattern_confidence(input, pattern, &nlp_result);
            if confidence > self.config.confidence_threshold {
                candidates.push((pattern.tool_id.clone(), confidence + pattern.confidence_boost));
            }
        }
        
        // If no pattern matches, try fuzzy matching
        if candidates.is_empty() {
            let tools = self.discovery.get_all_tools().await;
            for tool in tools {
                let confidence = self.calculate_tool_confidence(input, &tool, &nlp_result);
                if confidence > self.config.confidence_threshold {
                    candidates.push((tool.id.clone(), confidence));
                }
            }
        }
        
        // Sort by confidence
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        if candidates.is_empty() {
            return Err(anyhow::anyhow!("No matching tool found for: {}", input));
        }
        
        // Get the best match
        let (tool_id, confidence) = &candidates[0];
        let tool = self.discovery.get_tool(tool_id).await
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", tool_id))?;
        
        // Extract parameters
        let parameters = self.extract_parameters(input, &tool, &nlp_result).await?;
        
        // Create alternatives
        let alternatives: Vec<AlternativeInterpretation> = candidates
            .iter()
            .skip(1)
            .take(self.config.max_suggestions - 1)
            .map(|(id, conf)| AlternativeInterpretation {
                tool_id: id.clone(),
                tool_name: id.clone(), // TODO: Get actual tool name
                confidence: *conf,
                explanation: format!("Alternative tool with {:.0}% confidence", conf * 100.0),
            })
            .collect();
        
        Ok(ToolInterpretation {
            tool_id: tool_id.clone(),
            tool_name: tool.name.clone(),
            confidence: *confidence,
            parameters,
            explanation: self.generate_explanation(&tool, input),
            alternatives,
        })
    }
    
    /// Execute interpreted command
    pub async fn execute(
        &self,
        interpretation: &ToolInterpretation,
    ) -> Result<ExecutionResult> {
        // Check if confirmation is required
        if self.config.require_confirmation && !self.config.auto_execute {
            return Ok(ExecutionResult::RequiresConfirmation);
        }
        
        // Get the tool
        let tool = self.discovery.get_tool(&interpretation.tool_id).await
            .ok_or_else(|| anyhow::anyhow!("Tool not found"))?;
        
        // Validate parameters
        let params_value = serde_json::to_value(&interpretation.parameters)?;
        self.discovery.validate_parameters(&tool, &params_value)?;
        
        // Execute through tool manager
        match self.tool_manager.execute_tool(&tool.name, params_value).await {
            Ok(result) => Ok(ExecutionResult::Success(serde_json::to_value(result)?)),
            Err(e) => Ok(ExecutionResult::Error(e.to_string())),
        }
    }
    
    /// Execute from natural language
    pub async fn execute_from_nl(&self, input: &str) -> Result<ExecutionResult> {
        let interpretation = self.interpret(input).await?;
        
        // Record in history
        let record = ExecutionRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            input: input.to_string(),
            interpretation: interpretation.clone(),
            result: ExecutionResult::Pending, // Will be updated with actual result
            feedback: None,
        };
        
        let result = self.execute(&interpretation).await?;
        
        // Update record with actual result
        let mut history = self.history.write().await;
        let mut final_record = record;
        final_record.result = result.clone();
        history.push(final_record);
        
        Ok(result)
    }
    
    /// Learn from user feedback
    pub async fn learn_from_feedback(
        &self,
        execution_id: &str,
        feedback: UserFeedback,
    ) -> Result<()> {
        let mut history = self.history.write().await;
        
        if let Some(record) = history.iter_mut().find(|r| r.id == execution_id) {
            record.feedback = Some(feedback.clone());
            
            if self.config.learn_from_corrections && !feedback.correct {
                // Learn from the correction
                if let Some(corrected_tool) = feedback.corrected_tool {
                    // TODO: Update pattern confidence or create new pattern
                    info!("Learning from correction: {} -> {}", 
                          record.interpretation.tool_id, corrected_tool);
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate pattern confidence
    fn calculate_pattern_confidence(
        &self,
        input: &str,
        pattern: &IntentPattern,
        _nlp_result: &NLPResult,
    ) -> f32 {
        let input_lower = input.to_lowercase();
        
        for p in &pattern.patterns {
            if input_lower.contains(&p.to_lowercase()) {
                return 0.9; // High confidence for direct pattern match
            }
        }
        
        // Calculate similarity
        let mut max_similarity = 0.0;
        for p in &pattern.patterns {
            let similarity = self.calculate_similarity(&input_lower, &p.to_lowercase());
            if similarity > max_similarity {
                max_similarity = similarity;
            }
        }
        
        max_similarity
    }
    
    /// Calculate tool confidence
    fn calculate_tool_confidence(
        &self,
        input: &str,
        tool: &DiscoveredTool,
        _nlp_result: &NLPResult,
    ) -> f32 {
        let input_lower = input.to_lowercase();
        let name_lower = tool.name.to_lowercase();
        
        // Check for tool name in input
        if input_lower.contains(&name_lower) {
            return 0.8;
        }
        
        // Check for category keywords
        let category_name = tool.category.display_name().to_lowercase();
        if input_lower.contains(&category_name) {
            return 0.6;
        }
        
        // Check description similarity
        self.calculate_similarity(&input_lower, &tool.description.to_lowercase()) * 0.7
    }
    
    /// Calculate string similarity (simple implementation)
    fn calculate_similarity(&self, s1: &str, s2: &str) -> f32 {
        let words1: Vec<&str> = s1.split_whitespace().collect();
        let words2: Vec<&str> = s2.split_whitespace().collect();
        
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }
        
        let mut matches = 0;
        for w1 in &words1 {
            if words2.contains(w1) {
                matches += 1;
            }
        }
        
        matches as f32 / words1.len().max(words2.len()) as f32
    }
    
    /// Extract parameters from input
    async fn extract_parameters(
        &self,
        input: &str,
        tool: &DiscoveredTool,
        nlp_result: &NLPResult,
    ) -> Result<HashMap<String, Value>> {
        let mut parameters = HashMap::new();
        
        // Try to extract from patterns first
        let patterns = self.intent_patterns.read().await;
        if let Some(pattern) = patterns.values().find(|p| p.tool_id == tool.id) {
            for extractor in &pattern.parameter_extractors {
                if let Some(value) = self.extract_parameter(input, extractor, nlp_result) {
                    parameters.insert(extractor.parameter_name.clone(), value);
                } else if let Some(default) = &extractor.default_value {
                    parameters.insert(extractor.parameter_name.clone(), default.clone());
                } else if extractor.required {
                    return Err(anyhow::anyhow!(
                        "Required parameter {} not found",
                        extractor.parameter_name
                    ));
                }
            }
        }
        
        // Fill in any missing required parameters with defaults
        for param in &tool.parameters {
            if param.required && !parameters.contains_key(&param.name) {
                if let Some(default) = &param.default_value {
                    parameters.insert(param.name.clone(), default.clone());
                }
            }
        }
        
        Ok(parameters)
    }
    
    /// Extract single parameter
    fn extract_parameter(
        &self,
        input: &str,
        extractor: &ParameterExtractor,
        nlp_result: &NLPResult,
    ) -> Option<Value> {
        match &extractor.extraction_type {
            ExtractionType::Regex(pattern) => {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(captures) = re.captures(input) {
                        if let Some(m) = captures.get(1) {
                            return Some(Value::String(m.as_str().to_string()));
                        }
                    }
                }
            }
            ExtractionType::Keyword(keywords) => {
                for keyword in keywords {
                    if input.to_lowercase().contains(&keyword.to_lowercase()) {
                        return Some(Value::String(keyword.clone()));
                    }
                }
            }
            ExtractionType::Entity(entity_type) => {
                // Extract based on entity type
                match entity_type {
                    EntityType::FilePath => {
                        // Simple file path extraction
                        for word in input.split_whitespace() {
                            if word.contains('/') || word.contains('\\') || word.ends_with(".rs") 
                                || word.ends_with(".txt") || word.ends_with(".json") {
                                return Some(Value::String(word.to_string()));
                            }
                        }
                    }
                    EntityType::Number => {
                        // TODO: Extract numbers from NLP result or input
                        for word in input.split_whitespace() {
                            if let Ok(num) = word.parse::<f64>() {
                                return Some(Value::Number(serde_json::Number::from_f64(num)?));
                            }
                        }
                    }
                    _ => {}
                }
            }
            ExtractionType::Position(pos) => {
                let words: Vec<&str> = input.split_whitespace().collect();
                if words.len() > *pos {
                    let remaining = words[*pos..].join(" ");
                    return Some(Value::String(remaining));
                }
            }
            _ => {}
        }
        
        None
    }
    
    /// Generate explanation for interpretation
    fn generate_explanation(&self, tool: &DiscoveredTool, input: &str) -> String {
        format!(
            "I'll use the {} tool to help with: '{}'",
            tool.name,
            if input.len() > 50 {
                format!("{}...", &input[..50])
            } else {
                input.to_string()
            }
        )
    }
    
    /// Get execution history
    pub async fn get_history(&self) -> Vec<ExecutionRecord> {
        self.history.read().await.clone()
    }
    
    /// Clear execution history
    pub async fn clear_history(&self) {
        self.history.write().await.clear();
    }
}
