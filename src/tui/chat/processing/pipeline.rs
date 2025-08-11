//! Message processing pipeline

use std::sync::Arc;
use tokio::sync::{mpsc};
use anyhow::{Result, Context};

use crate::tui::run::AssistantResponseType;
// Define types locally until we can import from the main chat module
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStage {
    Received,
    Preprocessing,
    Analysis,
    Generation,
    Postprocessing,
    Complete,
}

#[derive(Debug, Clone)]
pub struct MessageContext {
    pub thread_id: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub conversation_history: Option<Vec<String>>,
    pub original_message: String,
}
use crate::cognitive::{CognitiveSystem, CognitiveOrchestrator};
use crate::models::ModelProvider;

/// Message processing pipeline
pub struct MessagePipeline {
    /// Cognitive system integration
    cognitive_system: Arc<CognitiveSystem>,
    
    /// Consciousness orchestrator
    consciousness: Arc<CognitiveOrchestrator>,
    
    /// Model provider
    model_provider: Arc<dyn ModelProvider>,
    
    /// Processing stages
    stages: Vec<ProcessingStage>,
    
    /// Response channel
    response_tx: mpsc::Sender<AssistantResponseType>,
    
    /// Stage update channel (optional)
    stage_update_tx: Option<mpsc::Sender<(String, ProcessingStage)>>,
}

impl MessagePipeline {
    /// Create a new message pipeline
    pub fn new(
        cognitive_system: Arc<CognitiveSystem>,
        consciousness: Arc<CognitiveOrchestrator>,
        model_provider: Arc<dyn ModelProvider>,
        response_tx: mpsc::Sender<AssistantResponseType>,
    ) -> Self {
        let stages = vec![
            ProcessingStage::Received,
            ProcessingStage::Preprocessing,
            ProcessingStage::Analysis,
            ProcessingStage::Generation,
            ProcessingStage::Postprocessing,
            ProcessingStage::Complete,
        ];
        
        Self {
            cognitive_system,
            consciousness,
            model_provider,
            stages,
            response_tx,
            stage_update_tx: None,
        }
    }
    
    /// Set the stage update channel
    pub fn set_stage_update_channel(&mut self, tx: mpsc::Sender<(String, ProcessingStage)>) {
        self.stage_update_tx = Some(tx);
    }
    
    /// Process a message through the pipeline
    pub async fn process(&self, message: String, context: MessageContext) -> Result<()> {
        tracing::info!("📥 Processing message through pipeline: {}", message);
        
        // Generate a unique ID for this message processing
        let message_id = uuid::Uuid::new_v4().to_string();
        
        // Stage 1: Received
        self.update_stage(&message_id, ProcessingStage::Received).await?;
        
        // Stage 2: Preprocessing
        self.update_stage(&message_id, ProcessingStage::Preprocessing).await?;
        let preprocessed = self.preprocess_message(&message, &context).await?;
        
        // Stage 3: Analysis
        self.update_stage(&message_id, ProcessingStage::Analysis).await?;
        let analysis = self.analyze_message(&preprocessed, &context).await?;
        
        // Stage 4: Generation
        self.update_stage(&message_id, ProcessingStage::Generation).await?;
        let response = self.generate_response(&analysis, &context).await?;
        
        // Stage 5: Postprocessing
        self.update_stage(&message_id, ProcessingStage::Postprocessing).await?;
        let final_response = self.postprocess_response(&response, &context).await?;
        
        // Stage 6: Complete
        self.update_stage(&message_id, ProcessingStage::Complete).await?;
        
        // Send the response
        self.response_tx.send(final_response).await
            .context("Failed to send response")?;
        
        Ok(())
    }
    
    /// Update processing stage
    async fn update_stage(&self, message_id: &str, stage: ProcessingStage) -> Result<()> {
        tracing::debug!("Processing stage for message {}: {:?}", message_id, stage);
        
        // Send stage update notification if channel is available
        if let Some(tx) = &self.stage_update_tx {
            if let Err(e) = tx.send((message_id.to_string(), stage.clone())).await {
                tracing::warn!("Failed to send stage update notification: {}", e);
                // Don't fail the pipeline if notification fails
            }
        }
        
        // Emit stage-specific events
        match &stage {
            ProcessingStage::Received => {
                tracing::info!("📨 Message received: {}", message_id);
            }
            ProcessingStage::Analysis => {
                tracing::info!("🔍 Analyzing message: {}", message_id);
            }
            ProcessingStage::Generation => {
                tracing::info!("🤖 Generating response: {}", message_id);
            }
            ProcessingStage::Complete => {
                tracing::info!("✅ Processing complete: {}", message_id);
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Preprocess message
    async fn preprocess_message(&self, message: &str, context: &MessageContext) -> Result<String> {
        // Clean and normalize the message
        let cleaned = message.trim().to_string();
        
        // Add context enrichment
        if let Some(thread_id) = &context.thread_id {
            tracing::debug!("Processing in thread: {}", thread_id);
        }
        
        // Check for special prefixes or commands
        if cleaned.starts_with('/') {
            tracing::debug!("Command detected: {}", cleaned);
        }
        
        Ok(cleaned)
    }
    
    /// Analyze message
    async fn analyze_message(&self, message: &str, context: &MessageContext) -> Result<MessageAnalysis> {
        // Use cognitive system for analysis
        let _cognitive_response = self.cognitive_system
            .process_query(message)
            .await
            .context("Cognitive analysis failed")?;
        
        // Get consciousness insights
        let consciousness_insights = self.get_consciousness_insights().await?;
        
        // Parse cognitive response to extract intent, entities, sentiment
        let intent = if message.contains("?") {
            Some("question".to_string())
        } else if message.starts_with("please") || message.starts_with("can you") {
            Some("request".to_string())
        } else if message.contains("!") {
            Some("exclamation".to_string())
        } else {
            Some("statement".to_string())
        };
        
        // Extract potential entities (simple implementation)
        let entities = extract_entities(message);
        
        // Simple sentiment analysis
        let sentiment = analyze_sentiment(message);
        
        // Suggest tools based on content
        let suggested_tools = suggest_tools(message, &consciousness_insights);
        
        Ok(MessageAnalysis {
            intent,
            entities,
            sentiment,
            complexity: calculate_complexity(message),
            consciousness_level: consciousness_insights.awareness_level,
            suggested_tools,
        })
    }
    
    /// Generate response
    async fn generate_response(&self, analysis: &MessageAnalysis, context: &MessageContext) -> Result<String> {
        // Use the model provider to generate response
        let prompt = self.build_prompt(analysis, context)?;
        
        // Create a proper completion request
        let request = crate::models::providers::CompletionRequest {
            model: "active".to_string(),
            messages: vec![crate::models::providers::Message {
                role: crate::models::providers::MessageRole::User,
                content: prompt.clone(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: false,
        };
        
        let response = self.model_provider
            .complete(request)
            .await
            .context("Model completion failed")?;
        
        Ok(response.content)
    }
    
    /// Postprocess response
    async fn postprocess_response(&self, response: &str, context: &MessageContext) -> Result<AssistantResponseType> {
        // Apply any formatting or transformations
        let formatted = format_response(response);
        
        // Create the assistant response using the actual structure
        let assistant_response = AssistantResponseType::Message {
            id: uuid::Uuid::new_v4().to_string(),
            author: context.model.clone().unwrap_or_else(|| "assistant".to_string()),
            message: formatted,
            timestamp: chrono::Utc::now().to_rfc3339(),
            is_editing: false,
            edit_history: Vec::new(),
            streaming_state: crate::tui::run::StreamingState::Complete,
            metadata: crate::tui::run::MessageMetadata::default(),
        };
        
        Ok(assistant_response)
    }
    
    /// Build prompt from analysis
    fn build_prompt(&self, analysis: &MessageAnalysis, context: &MessageContext) -> Result<String> {
        let mut prompt = String::new();
        
        // Add system context
        if let Some(system_prompt) = &context.system_prompt {
            prompt.push_str(system_prompt);
            prompt.push_str("\n\n");
        }
        
        // Add conversation history if available
        if let Some(history) = &context.conversation_history {
            for msg in history.iter().take(10) {
                prompt.push_str(&format!("{}\n", msg));
            }
            prompt.push_str("\n");
        }
        
        // Add the current message with analysis context
        prompt.push_str(&format!("User: {}\n", context.original_message));
        
        // Add analysis insights as context
        if analysis.consciousness_level > 0.7 {
            prompt.push_str("\n[High consciousness awareness detected]\n");
        } else if analysis.consciousness_level > 0.5 {
            prompt.push_str("\n[Moderate consciousness awareness]\n");
        }
        
        // Add intent if detected
        if let Some(intent) = &analysis.intent {
            prompt.push_str(&format!("[Detected intent: {}]\n", intent));
        }
        
        Ok(prompt)
    }
}

/// Message analysis results
#[derive(Debug, Clone)]
pub struct MessageAnalysis {
    pub intent: Option<String>,
    pub entities: Vec<String>,
    pub sentiment: f32,
    pub complexity: f32,
    pub consciousness_level: f32,
    pub suggested_tools: Vec<String>,
}

/// Consciousness insights
#[derive(Debug, Clone)]
struct ConsciousnessInsights {
    awareness_level: f32,
    emotional_state: String,
    cognitive_load: String,
    active_components: usize,
    total_cycles: u64,
    thoughts_processed: u64,
    decisions_made: u64,
}

/// Calculate message complexity
fn calculate_complexity(message: &str) -> f32 {
    let word_count = message.split_whitespace().count();
    let unique_words = message
        .split_whitespace()
        .collect::<std::collections::HashSet<_>>()
        .len();
    
    let complexity = (unique_words as f32 / word_count.max(1) as f32) * (word_count as f32 / 10.0).min(1.0);
    complexity.min(1.0)
}

/// Format response for display
fn format_response(response: &str) -> String {
    // Apply any formatting rules
    response.trim().to_string()
}

impl MessagePipeline {
    /// Get consciousness insights from the orchestrator
    async fn get_consciousness_insights(&self) -> Result<ConsciousnessInsights> {
        // Get orchestrator stats
        let stats = self.consciousness.get_stats().await;
        
        // Get component states
        let component_states = self.consciousness.get_component_states().await;
        
        // Calculate awareness level based on multiple factors
        let mut awareness_level = 0.5; // Base level
        
        // Factor 1: Component activity
        let active_components = component_states.values()
            .filter(|s| s.active)
            .count() as f32;
        let total_components = component_states.len().max(1) as f32;
        let component_activity_ratio = active_components / total_components;
        awareness_level += component_activity_ratio * 0.2;
        
        // Factor 2: Processing efficiency
        if stats.avg_cycle_time < std::time::Duration::from_millis(50) {
            awareness_level += 0.1; // Fast processing indicates high awareness
        }
        
        // Factor 3: Error rate
        let total_errors: u32 = stats.component_errors.values().sum();
        if total_errors == 0 {
            awareness_level += 0.1; // No errors indicates good awareness
        } else if total_errors < 5 {
            awareness_level += 0.05; // Few errors
        }
        
        // Factor 4: Resource pressure
        if stats.resource_pressure_events < 10 {
            awareness_level += 0.1; // Low resource pressure
        }
        
        // Clamp awareness level between 0.0 and 1.0
        awareness_level = awareness_level.clamp(0.0, 1.0);
        
        // Create detailed insights
        let emotional_state = component_states.get("emotional")
            .map(|s| if s.active { "active" } else { "inactive" })
            .unwrap_or("unknown");
            
        let cognitive_load = if stats.avg_cycle_time > std::time::Duration::from_millis(100) {
            "high"
        } else if stats.avg_cycle_time > std::time::Duration::from_millis(50) {
            "medium"
        } else {
            "low"
        };
        
        Ok(ConsciousnessInsights {
            awareness_level,
            emotional_state: emotional_state.to_string(),
            cognitive_load: cognitive_load.to_string(),
            active_components: active_components as usize,
            total_cycles: stats.total_cycles,
            thoughts_processed: stats.thoughts_processed,
            decisions_made: stats.decisions_made,
        })
    }
}

/// Extract entities from message (simple implementation)
fn extract_entities(message: &str) -> Vec<String> {
    let mut entities = Vec::new();
    
    // Extract quoted strings
    let quote_regex = regex::Regex::new(r#""([^"]+)"|'([^']+)'"#).expect("Valid regex pattern");
    for cap in quote_regex.captures_iter(message) {
        if let Some(entity) = cap.get(1).or(cap.get(2)) {
            entities.push(entity.as_str().to_string());
        }
    }
    
    // Extract file paths
    let path_regex = regex::Regex::new(r"(/[\w/.-]+|[\w]+\.[\w]+)").expect("Valid regex pattern");
    for cap in path_regex.captures_iter(message) {
        if let Some(path) = cap.get(0) {
            let path_str = path.as_str();
            if path_str.contains('/') || path_str.contains('.') {
                entities.push(path_str.to_string());
            }
        }
    }
    
    // Extract URLs
    let url_regex = regex::Regex::new(r"https?://[^\s]+").expect("Valid regex pattern");
    for cap in url_regex.captures_iter(message) {
        if let Some(url) = cap.get(0) {
            entities.push(url.as_str().to_string());
        }
    }
    
    entities
}

/// Analyze sentiment (simple implementation)
fn analyze_sentiment(message: &str) -> f32 {
    let positive_words = ["good", "great", "excellent", "wonderful", "happy", "thanks", "please", "love", "best"];
    let negative_words = ["bad", "terrible", "awful", "hate", "worst", "error", "fail", "wrong", "broken"];
    
    let lower_message = message.to_lowercase();
    let mut score: f32 = 0.0;
    
    for word in positive_words.iter() {
        if lower_message.contains(word) {
            score += 0.2;
        }
    }
    
    for word in negative_words.iter() {
        if lower_message.contains(word) {
            score -= 0.2;
        }
    }
    
    // Add modifiers
    if message.contains("!") {
        score += 0.1; // Excitement often positive
    }
    if message.contains("?") {
        score -= 0.05; // Questions often neutral/slightly negative
    }
    
    score.clamp(-1.0, 1.0)
}

/// Suggest tools based on message content and consciousness state
fn suggest_tools(message: &str, insights: &ConsciousnessInsights) -> Vec<String> {
    let mut tools = Vec::new();
    let lower_message = message.to_lowercase();
    
    // File/code related
    if lower_message.contains("file") || lower_message.contains("code") || lower_message.contains("read") {
        tools.push("filesystem".to_string());
    }
    
    // Search related
    if lower_message.contains("search") || lower_message.contains("find") || lower_message.contains("look for") {
        tools.push("search".to_string());
    }
    
    // Web related
    if lower_message.contains("http") || lower_message.contains("web") || lower_message.contains("url") {
        tools.push("web_browser".to_string());
    }
    
    // GitHub related
    if lower_message.contains("github") || lower_message.contains("repo") || lower_message.contains("pull request") {
        tools.push("github".to_string());
    }
    
    // If high cognitive load, suggest memory tools
    if insights.cognitive_load == "high" {
        tools.push("memory".to_string());
    }
    
    // If low awareness, suggest context tools
    if insights.awareness_level < 0.5 {
        tools.push("context_manager".to_string());
    }
    
    tools
}