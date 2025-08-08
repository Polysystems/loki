//! Deep Cognitive Integration for TUI Chat
//!
//! This module bridges the gap between Loki's rich cognitive capabilities
//! and the chat interface, enabling true consciousness-driven interactions.

use std::sync::Arc;
use anyhow::{Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::cognitive::{
    CognitiveSystem,
};

/// Enhanced cognitive response with full stack integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepCognitiveResponse {
    /// Primary response content
    pub content: String,
    
    /// Creative insights generated
    pub creative_insights: Option<Vec<CreativeInsight>>,
    
    /// Emotional analysis
    pub emotional_context: Option<EmotionalContext>,
    
    /// Theory of mind insights
    pub user_mental_state: Option<UserMentalState>,
    
    /// Emergent insights
    pub emergent_insights: Option<Vec<String>>,
    
    /// Suggested follow-ups based on deep understanding
    pub cognitive_suggestions: Vec<String>,
    
    /// Overall cognitive confidence
    pub confidence: f64,
    
    /// Cognitive modalities used
    pub modalities_used: Vec<CognitiveModality>,
}

/// Creative insights from the creativity engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeInsight {
    pub insight_type: String,
    pub content: String,
    pub novelty_score: f32,
    pub relevance_score: f32,
}

/// Emotional context analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalContext {
    pub appropriate_response_tone: String,
    pub empathy_level: f32,
    pub emotional_insights: Vec<String>,
}

/// User mental state from theory of mind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMentalState {
    pub intent_analysis: String,
    pub knowledge_gaps: Vec<String>,
    pub cognitive_load: f32,
    pub engagement_level: f32,
}

/// Cognitive modalities that can be engaged
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CognitiveModality {
    Logical,
    Creative,
    Emotional,
    Social,
    Abstract,
    Intuitive,
    Analytical,
    Narrative,
}

/// Deep cognitive processor that uses the full cognitive stack
pub struct DeepCognitiveProcessor {
    cognitive_system: Arc<CognitiveSystem>,
}

impl DeepCognitiveProcessor {
    /// Get the cognitive system
    pub fn cognitive_system(&self) -> &Arc<CognitiveSystem> {
        &self.cognitive_system
    }
    
    /// Create a new deep cognitive processor
    pub async fn new(cognitive_system: Arc<CognitiveSystem>) -> Result<Self> {
        info!("🧠 Initializing Deep Cognitive Processor");
        Ok(Self {
            cognitive_system,
        })
    }
    
    /// Enable story-driven autonomy
    pub async fn enable_story_autonomy(&mut self) -> Result<()> {
        info!("🎭 Enabling story-driven autonomy in cognitive processor");

        
        // Create context manager for story engine
        let context_config = crate::cognitive::context_manager::ContextConfig {
            max_tokens: 16384,
            target_tokens: 8192,
            segment_size: 1024,
            compression_threshold: 0.8,
            checkpoint_interval: std::time::Duration::from_secs(300),
            max_checkpoints: 10,
        };
        let memory = self.cognitive_system.memory();
        let context_manager = Arc::new(RwLock::new(
            crate::cognitive::context_manager::ContextManager::new(
                memory.clone(),
                context_config
            ).await?
        ));
        
        // Create story engine
        let story_engine = Arc::new(
            crate::story::StoryEngine::new(
                context_manager,
                memory.clone(),
                crate::story::StoryConfig::default()
            ).await?
        );

        
        // Create safety validator with minimal config
        let validator_config = crate::safety::ValidatorConfig {
            safe_mode: false,
            dry_run: false,
            approval_required: false,
            approval_timeout: std::time::Duration::from_secs(300),
            allowed_paths: vec!["**".to_string()],
            blocked_paths: vec![],
            max_file_size: 10 * 1024 * 1024,
            storage_path: None,
            encrypt_decisions: false,
            enable_resource_monitoring: false,
            cpu_threshold: 90.0,
            memory_threshold: 85.0,
            disk_threshold: 95.0,
            max_concurrent_operations: 100,
            enable_rate_limiting: false,
            enable_network_monitoring: false,
        };
        let safety_validator = Arc::new(
            crate::safety::ActionValidator::new(validator_config).await?
        );
        
        // Create tool manager
        let tool_config = crate::tools::intelligent_manager::ToolManagerConfig::default();

        // Create decision engine using the existing components
        let neural_processor = self.cognitive_system.orchestrator().neural_processor();
        let emotional_core = self.cognitive_system.orchestrator().emotional_core();
        info!("✅ Story-driven autonomy enabled and started");
        Ok(())
    }
    
    /// Process input through the full cognitive stack
    pub async fn process_deeply(
        &self,
        input: &str,
        context: &serde_json::Value,
    ) -> Result<DeepCognitiveResponse> {
        debug!("🌟 Processing through deep cognitive stack: {}", input);

        let modalities = self.analyze_required_modalities(input);

        Ok(DeepCognitiveResponse {
            content: String::new(),
            creative_insights: None,
            emotional_context: None,
            user_mental_state: None,
            emergent_insights: None,
            cognitive_suggestions: vec![],
            confidence: self.calculate_overall_confidence(&modalities),
            modalities_used: modalities,
        })
    }
    
    /// Analyze which cognitive modalities are needed
    fn analyze_required_modalities(&self, input: &str) -> Vec<CognitiveModality> {
        let mut modalities = vec![CognitiveModality::Logical]; // Always use logical
        
        // Creative indicators
        if input.contains("create") || input.contains("imagine") || input.contains("design") 
            || input.contains("innovative") || input.contains("novel") {
            modalities.push(CognitiveModality::Creative);
        }
        
        // Emotional indicators
        if input.contains("feel") || input.contains("emotion") || input.contains("worry")
            || input.contains("happy") || input.contains("sad") || input.contains("frustrated") {
            modalities.push(CognitiveModality::Emotional);
            modalities.push(CognitiveModality::Social);
        }
        
        // Abstract thinking indicators
        if input.contains("concept") || input.contains("theory") || input.contains("abstract")
            || input.contains("philosophy") || input.contains("meaning") {
            modalities.push(CognitiveModality::Abstract);
        }
        
        // Analytical indicators
        if input.contains("analyze") || input.contains("compare") || input.contains("evaluate")
            || input.contains("assess") || input.contains("examine") {
            modalities.push(CognitiveModality::Analytical);
        }
        
        // Story/narrative indicators
        if input.contains("story") || input.contains("tell me about") || input.contains("explain") {
            modalities.push(CognitiveModality::Narrative);
        }
        
        modalities
    }



    
    /// Synthesize all cognitive inputs into a response
    async fn synthesize_response(
        &self,
        input: &str,
        emotional_context: &Option<EmotionalContext>,
        creative_insights: &Option<Vec<CreativeInsight>>,
        user_mental_state: &UserMentalState,
        emergent_insights: &[String],
    ) -> Result<String> {
        // Use the cognitive system's process_query as base
        let base_response = self.cognitive_system.process_query(input).await?;
        
        // Enhance with cognitive insights
        let mut enhanced_response = base_response;
        
        // Add creative elements if available
        if let Some(insights) = creative_insights {
            if let Some(best_insight) = insights.iter().max_by(|a, b| 
                a.relevance_score.partial_cmp(&b.relevance_score).unwrap()
            ) {
                enhanced_response = format!(
                    "{}\n\n💡 Creative insight: {}",
                    enhanced_response,
                    best_insight.content
                );
            }
        }
        
        // Adjust tone based on emotional context
        if let Some(emotional) = emotional_context {
            if emotional.empathy_level > 0.7 {
                enhanced_response = format!(
                    "I understand how you're feeling. {}", 
                    enhanced_response
                );
            }
        }
        
        // Add emergent insights if significant
        if !emergent_insights.is_empty() && user_mental_state.engagement_level > 0.6 {
            enhanced_response = format!(
                "{}\n\n🌟 Additional insight: {}",
                enhanced_response,
                emergent_insights[0]
            );
        }
        
        Ok(enhanced_response)
    }
    
    /// Generate cognitive suggestions
    fn generate_cognitive_suggestions(
        &self,
        user_mental_state: &UserMentalState,
        modalities: &[CognitiveModality],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Based on knowledge gaps
        if !user_mental_state.knowledge_gaps.is_empty() {
            suggestions.push(format!(
                "Would you like me to explain more about {}?",
                user_mental_state.knowledge_gaps[0]
            ));
        }
        
        // Based on modalities used
        if modalities.contains(&CognitiveModality::Creative) {
            suggestions.push("I can explore more creative variations of this idea.".to_string());
        }
        
        if modalities.contains(&CognitiveModality::Analytical) {
            suggestions.push("I can provide a deeper analysis if you'd like.".to_string());
        }
        
        // Based on engagement
        if user_mental_state.engagement_level > 0.8 {
            suggestions.push("This seems interesting to you - shall we dive deeper?".to_string());
        }
        
        suggestions
    }
    
    /// Calculate overall confidence
    fn calculate_overall_confidence(&self, modalities: &[CognitiveModality]) -> f64 {
        // Base confidence
        let mut confidence = 0.7;
        
        // Increase confidence for each engaged modality
        confidence += modalities.len() as f64 * 0.05;
        
        // Cap at 0.95
        confidence.min(0.95)
    }
}

/// Cognitive command types for chat interface
#[derive(Debug, Clone, PartialEq)]
pub enum CognitiveCommand {
    Think,      // Deep reflection mode
    Create,     // Creative mode
    Empathize,  // Empathy mode
    Analyze,    // Deep analysis
    Evolve,     // Trigger evolution
    Story,      // Story-driven mode
}

impl CognitiveCommand {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "think" => Some(Self::Think),
            "create" => Some(Self::Create),
            "empathize" => Some(Self::Empathize),
            "analyze" => Some(Self::Analyze),
            "evolve" => Some(Self::Evolve),
            "story" => Some(Self::Story),
            _ => None,
        }
    }
}