//! Consolidated Natural Language Orchestrator
//! 
//! This module combines functionality from:
//! - natural_language_orchestrator.rs (base)
//! - enhanced_natural_language_orchestrator.rs (cognitive commands)
//! - natural_language_orchestrator_ext.rs (agent streaming)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info, warn};
use crate::cognitive::CognitiveSystem;
use crate::memory::CognitiveMemory;
use crate::models::orchestrator::ModelOrchestrator;
use crate::models::multi_agent_orchestrator::MultiAgentOrchestrator;
use crate::safety::ActionValidator;
use crate::tools::{IntelligentToolManager};
use crate::tools::task_management::{TaskManager, TaskPriority};
use crate::tools::mcp_client::McpClient;

// Feature-gated imports
#[cfg(feature = "deep-cognition")]
use crate::tui::cognitive::integration::main::{
    DeepCognitiveProcessor, DeepCognitiveResponse, CognitiveCommand
};

#[cfg(feature = "story-enhancement")]
use crate::tui::chat::integrations::story::{
    StoryChatEnhancement, StoryChatMode, StoryChatIntegration,
};

#[cfg(feature = "agent-streams")]
use crate::tui::ui::chat::agent_stream_manager::{
    AgentStreamManager, AgentMessageType, MessagePriority, AgentStatus
};

use crate::tui::cognitive::core::tone_detector::{CognitiveToneDetector, ToneDetectionResult};
use crate::tui::task_decomposer::{TaskDecomposer, DecomposedTask};
use crate::tui::agent_task_mapper::{AgentTaskMapper};
use crate::tui::task_progress_aggregator::{TaskProgressAggregator};
use crate::tui::nlp::analysis::enhanced_analyzer::{EnhancedNLPAnalyzer};

/// Configuration for the orchestrator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrchestratorConfig {
    pub session_timeout: Duration,
    pub max_context_tokens: usize,
    pub enable_learning: bool,
    
    #[cfg(feature = "deep-cognition")]
    pub deep_cognition_enabled: bool,
    
    #[cfg(feature = "story-enhancement")]
    pub story_enhancement_enabled: bool,
    
    #[cfg(feature = "agent-streams")]
    pub agent_streams_enabled: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::from_secs(3600),
            max_context_tokens: 4096,
            enable_learning: true,
            
            #[cfg(feature = "deep-cognition")]
            deep_cognition_enabled: true,
            
            #[cfg(feature = "story-enhancement")]
            story_enhancement_enabled: true,
            
            #[cfg(feature = "agent-streams")]
            agent_streams_enabled: true,
        }
    }
}

/// Consolidated Natural Language Orchestrator
pub struct NaturalLanguageOrchestrator {
    /// Core cognitive system for intelligent processing
    cognitive_system: Arc<CognitiveSystem>,

    /// Memory system for context and learning
    memory: Arc<CognitiveMemory>,

    /// Model orchestrator for routing tasks to appropriate models
    model_orchestrator: Arc<ModelOrchestrator>,

    /// Multi-agent orchestrator for complex task coordination
    multi_agent_orchestrator: Arc<MultiAgentOrchestrator>,

    /// Intelligent tool manager for external capabilities
    tool_manager: Arc<IntelligentToolManager>,

    /// Task management system for TODO creation and tracking
    task_manager: Arc<TaskManager>,

    /// MCP client for external tool integration
    mcp_client: Arc<McpClient>,

    /// Safety validator for all actions
    safety_validator: Arc<ActionValidator>,

    /// Active conversation sessions
    sessions: Arc<RwLock<HashMap<String, ConversationSession>>>,

    /// Broadcast channel for system events
    event_sender: broadcast::Sender<OrchestratorEvent>,

    /// Configuration
    config: OrchestratorConfig,
    
    // Feature-gated components
    #[cfg(feature = "deep-cognition")]
    deep_cognitive_processor: Option<Arc<DeepCognitiveProcessor>>,
    
    #[cfg(feature = "story-enhancement")]
    story_enhancement: Arc<RwLock<Option<Arc<StoryChatEnhancement>>>>,
    
    #[cfg(feature = "agent-streams")]
    agent_stream_manager: Option<Arc<AgentStreamManager>>,
    
    /// Cognitive tone detector for natural language
    tone_detector: CognitiveToneDetector,
    
    /// Task decomposer for complex tasks
    task_decomposer: TaskDecomposer,
    
    /// Agent task mapper for delegation
    agent_task_mapper: Option<Arc<AgentTaskMapper>>,
    
    /// Task progress aggregator
    task_progress_aggregator: Option<Arc<TaskProgressAggregator>>,
    
    /// Enhanced NLP analyzer for sophisticated language understanding
    enhanced_nlp_analyzer: Option<Arc<EnhancedNLPAnalyzer>>,
}

impl NaturalLanguageOrchestrator {
    /// Create a new orchestrator with all capabilities
    pub async fn new(
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        model_orchestrator: Arc<ModelOrchestrator>,
        multi_agent_orchestrator: Arc<MultiAgentOrchestrator>,
        tool_manager: Arc<IntelligentToolManager>,
        task_manager: Arc<TaskManager>,
        mcp_client: Arc<McpClient>,
        safety_validator: Arc<ActionValidator>,
        config: OrchestratorConfig,
    ) -> Result<Self> {
        info!("🧠 Initializing Consolidated Natural Language Orchestrator");
        
        let (event_sender, _) = broadcast::channel(1024);
        
        // Initialize deep cognitive processor if feature enabled
        #[cfg(feature = "deep-cognition")]
        let deep_cognitive_processor = if config.deep_cognition_enabled {
            match DeepCognitiveProcessor::new(cognitive_system.clone()).await {
                Ok(processor) => Some(Arc::new(processor)),
                Err(e) => {
                    warn!("Failed to initialize deep cognitive processor: {}, continuing without it", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Initialize agent stream manager if feature enabled
        #[cfg(feature = "agent-streams")]
        let agent_stream_manager = if config.agent_streams_enabled {
            Some(Arc::new(AgentStreamManager::new(4))) // max 4 agent panels
        } else {
            None
        };
        
        Ok(Self {
            cognitive_system,
            memory,
            model_orchestrator,
            multi_agent_orchestrator,
            tool_manager,
            task_manager,
            mcp_client,
            safety_validator,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            config,
            
            #[cfg(feature = "deep-cognition")]
            deep_cognitive_processor,
            
            #[cfg(feature = "story-enhancement")]
            story_enhancement: Arc::new(RwLock::new(None)),
            
            #[cfg(feature = "agent-streams")]
            agent_stream_manager,
            
            tone_detector: CognitiveToneDetector::new(),
            task_decomposer: TaskDecomposer::new(),
            agent_task_mapper: None, // Will be set after creation
            task_progress_aggregator: None, // Will be set after creation
            enhanced_nlp_analyzer: None, // Will be initialized separately
        })
    }
    
    /// Set the agent task mapper
    pub fn set_agent_task_mapper(&mut self, mapper: Arc<AgentTaskMapper>) {
        self.agent_task_mapper = Some(mapper);
    }
    
    /// Set the task progress aggregator
    pub fn set_task_progress_aggregator(&mut self, aggregator: Arc<TaskProgressAggregator>) {
        self.task_progress_aggregator = Some(aggregator);
    }
    
    /// Initialize enhanced NLP analyzer
    pub async fn initialize_nlp_analyzer(&mut self) -> Result<()> {
        info!("🔤 Initializing enhanced NLP analyzer");
        
        let analyzer = EnhancedNLPAnalyzer::new(
            self.cognitive_system.clone(),
            self.memory.clone(),
        ).await?;
        
        self.enhanced_nlp_analyzer = Some(Arc::new(analyzer));
        info!("✅ Enhanced NLP analyzer initialized");
        Ok(())
    }
    
    /// Initialize story enhancement if available
    #[cfg(feature = "story-enhancement")]
    pub async fn initialize_story_enhancement(&self) -> Result<()> {
        if self.config.story_enhancement_enabled {
            info!("📖 Initializing story enhancement");
            
            let story_enhancement = StoryChatEnhancement::new(
                self.cognitive_system.clone(),
                self.memory.clone(),
                self.tool_manager.clone(),
            ).await?;
            
            *self.story_enhancement.write().await = Some(Arc::new(story_enhancement));
            
            info!("✅ Story enhancement initialized");
        }
        Ok(())
    }

    /// Main entry point for processing user input
    pub async fn process_input(
        &self,
        session_id: &str,
        input: &str,
    ) -> Result<OrchestrationResponse> {
        info!("🎯 Processing input in session {}: {}", session_id, input);
        
        // Phase 1: Try enhanced commands if available
        if let Some(response) = self.try_enhanced_commands(session_id, input).await? {
            return Ok(response);
        }
        
        // Phase 2: Core processing
        self.process_core_workflow(session_id, input).await
    }
    
    /// Try to handle enhanced commands (cognitive, story, agent)
    async fn try_enhanced_commands(
        &self,
        session_id: &str,
        input: &str,
    ) -> Result<Option<OrchestrationResponse>> {
        if !input.starts_with('/') {
            return Ok(None);
        }
        
        let parts: Vec<&str> = input[1..].split_whitespace().collect();
        if parts.is_empty() {
            return Ok(None);
        }
        
        let command = parts[0];
        let args = parts[1..].join(" ");
        
        match command {
            // Cognitive commands
            #[cfg(feature = "deep-cognition")]
            "think" if self.config.deep_cognition_enabled => {
                info!("🤔 Activating deep thinking mode");
                Ok(Some(self.deep_think(&args).await?))
            }
            
            #[cfg(feature = "deep-cognition")]
            "create" if self.config.deep_cognition_enabled => {
                info!("🎨 Activating creative mode");
                Ok(Some(self.creative_process(&args).await?))
            }
            
            #[cfg(feature = "deep-cognition")]
            "empathize" if self.config.deep_cognition_enabled => {
                info!("💝 Activating empathy mode");
                Ok(Some(self.empathize_with(&args).await?))
            }
            
            #[cfg(feature = "deep-cognition")]
            "analyze" if self.config.deep_cognition_enabled => {
                info!("🔍 Activating deep analysis mode");
                Ok(Some(self.deep_analyze(&args).await?))
            }
            
            #[cfg(feature = "deep-cognition")]
            "evolve" if self.config.deep_cognition_enabled => {
                info!("🧬 Triggering autonomous evolution");
                Ok(Some(self.trigger_evolution(&args).await?))
            }
            
            // Story commands
            #[cfg(feature = "story-enhancement")]
            "story" if self.config.story_enhancement_enabled => {
                info!("📖 Activating story-driven mode");
                Ok(Some(self.story_driven_process(&args).await?))
            }
            
            // Agent commands
            #[cfg(feature = "agent-streams")]
            "agent" | "spawn" if self.config.agent_streams_enabled => {
                info!("🤖 Processing agent command");
                Ok(Some(self.handle_agent_command(session_id, command, &args).await?))
            }
            
            _ => Ok(None),
        }
    }
    
    /// Core workflow processing (from base orchestrator)
    async fn process_core_workflow(
        &self,
        session_id: &str,
        input: &str,
    ) -> Result<OrchestrationResponse> {
        // Get or create session
        let session = self.get_or_create_session(session_id).await?;
        
        // Phase 1: Cognitive Understanding
        let understanding = self.analyze_input(input, &session).await?;
        debug!("📊 Understanding: {:?}", understanding);
        
        // Phase 1.5: Detect Cognitive Tone
        let tone_result = self.tone_detector.detect_tones(input);
        if tone_result.confidence > 0.4 {
            info!("🎨 Detected cognitive tone: {:?} (confidence: {:.2})", 
                tone_result.primary_tone, tone_result.confidence);
            
            // Apply cognitive tone to enhance processing
            if let Some(enhanced_response) = self.apply_cognitive_tone(
                session_id, 
                input, 
                &tone_result,
                &understanding
            ).await? {
                return Ok(enhanced_response);
            }
        }
        
        // Phase 2: Intent Recognition
        let intent = self.recognize_intent(input, &understanding).await?;
        debug!("🎯 Intent: {:?}", intent);
        
        // Phase 3: Task Extraction
        let tasks = self.extract_tasks(input, &intent, &understanding).await?;
        
        // Phase 3.5: Decompose complex tasks
        let mut decomposed_tasks = Vec::new();
        for task in &tasks {
            if self.is_complex_task(&task) {
                match self.task_decomposer.decompose_task(&task) {
                    Ok(decomposed) => {
                        info!("🔀 Decomposed complex task into {} subtasks", decomposed.subtasks.len());
                        decomposed_tasks.push(decomposed);
                    }
                    Err(e) => {
                        warn!("Failed to decompose task: {}", e);
                    }
                }
            }
        }
        
        // Phase 3.6: Check if this requires tool execution
        if let Some(tool_response) = self.try_execute_tools(input, &intent, &tasks).await? {
            return Ok(tool_response);
        }
        
        // Phase 3.7: If we have decomposed tasks, prepare for delegation
        if !decomposed_tasks.is_empty() {
            return self.handle_decomposed_tasks(session_id, decomposed_tasks, &tasks).await;
        }
        
        // Phase 4: Safety Validation
        if let Err(e) = self.validate_safety(input, &intent).await {
            warn!("⚠️ Safety validation failed: {}", e);
            return Ok(OrchestrationResponse {
                primary_response: format!("I cannot process this request due to safety concerns: {}", e),
                intent: Some(intent),
                extracted_tasks: vec![],
                suggestions: vec!["Please rephrase your request.".to_string()],
                session_context: json!(session.context),
                follow_up_needed: false,
                confidence: 0.0,
            });
        }
        
        // Phase 5: Response Generation
        let response = self.generate_response(
            session_id,
            input,
            &intent,
            &understanding,
            &tasks,
        ).await?;
        
        // Phase 6: Session Update
        self.update_session(session_id, input, &response).await?;

        
        Ok(response)
    }
    
    // Cognitive command implementations
    #[cfg(feature = "deep-cognition")]
    async fn deep_think(&self, topic: &str) -> Result<OrchestrationResponse> {
        if let Some(processor) = &self.deep_cognitive_processor {
            let context = json!({
                "mode": "deep_thinking",
                "emphasis": "reasoning",
            });
            
            let deep_response = processor.process_deeply(topic, &context).await?;
            
            let content = format!(
                "🤔 Deep Reflection on: {}\n\n{}",
                topic,
                deep_response.content
            );
            
            Ok(OrchestrationResponse {
                primary_response: content,
                intent: None,
                extracted_tasks: vec![],
                suggestions: deep_response.cognitive_suggestions,
                session_context: context,
                follow_up_needed: true,
                reasoning_chain: deep_response.reasoning_chain,
                confidence: deep_response.confidence,
            })
        } else {
            Err(anyhow!("Deep cognition not enabled"))
        }
    }
    
    #[cfg(feature = "deep-cognition")]
    async fn creative_process(&self, prompt: &str) -> Result<OrchestrationResponse> {
        if let Some(processor) = &self.deep_cognitive_processor {
            let context = json!({
                "mode": "creative",
                "emphasis": "novelty",
            });
            
            let deep_response = processor.process_deeply(prompt, &context).await?;
            
            let mut content = format!("🎨 Creative Exploration: {}\n\n", prompt);
            
            if let Some(insights) = &deep_response.creative_insights {
                content.push_str("💡 Creative Insights:\n");
                for insight in insights {
                    content.push_str(&format!(
                        "• {} (novelty: {:.0}%, relevance: {:.0}%)\n",
                        insight.content,
                        insight.novelty_score * 100.0,
                        insight.relevance_score * 100.0
                    ));
                }
                content.push_str("\n");
            }
            
            content.push_str(&deep_response.content);
            
            Ok(OrchestrationResponse {
                primary_response: content,
                intent: None,
                extracted_tasks: vec![],
                suggestions: vec![
                    "Would you like me to explore variations of these ideas?".to_string(),
                    "I can combine these concepts in novel ways.".to_string(),
                ],
                session_context: context,
                follow_up_needed: true,
                reasoning_chain: deep_response.reasoning_chain,
                confidence: deep_response.confidence,
            })
        } else {
            Err(anyhow!("Deep cognition not enabled"))
        }
    }
    
    #[cfg(feature = "deep-cognition")]
    async fn empathize_with(&self, context_str: &str) -> Result<OrchestrationResponse> {
        if let Some(processor) = &self.deep_cognitive_processor {
            let context = json!({
                "mode": "empathy",
                "emphasis": "emotional_understanding",
            });
            
            let deep_response = processor.process_deeply(context_str, &context).await?;
            
            let mut content = String::new();
            
            if let Some(emotional) = &deep_response.emotional_context {
                content.push_str(&format!(
                    "I sense that you're feeling {}. ",
                    format!("{:?}", emotional.user_emotion).to_lowercase()
                ));
                
                if emotional.empathy_level > 0.7 {
                    content.push_str("I truly understand how that feels. ");
                }
            }
            
            content.push_str(&deep_response.content);
            
            Ok(OrchestrationResponse {
                primary_response: content,
                intent: None,
                extracted_tasks: vec![],
                suggestions: vec![
                    "Would you like to talk more about this?".to_string(),
                    "I'm here to listen and help however I can.".to_string(),
                ],
                session_context: context,
                follow_up_needed: false,
                reasoning_chain: deep_response.reasoning_chain,
                confidence: deep_response.confidence,
            })
        } else {
            Err(anyhow!("Deep cognition not enabled"))
        }
    }
    
    #[cfg(feature = "deep-cognition")]
    async fn deep_analyze(&self, target: &str) -> Result<OrchestrationResponse> {
        if let Some(processor) = &self.deep_cognitive_processor {
            let context = json!({
                "mode": "analytical",
                "emphasis": "comprehensive_analysis",
            });
            
            let deep_response = processor.process_deeply(target, &context).await?;
            
            let content = format!(
                "🔍 Deep Analysis: {}\n\n{}",
                target,
                deep_response.content
            );
            
            Ok(OrchestrationResponse {
                primary_response: content,
                intent: None,
                extracted_tasks: vec![],
                suggestions: deep_response.cognitive_suggestions,
                session_context: context,
                follow_up_needed: true,
                reasoning_chain: deep_response.reasoning_chain,
                confidence: deep_response.confidence,
            })
        } else {
            Err(anyhow!("Deep cognition not enabled"))
        }
    }
    
    #[cfg(feature = "deep-cognition")]
    async fn trigger_evolution(&self, focus: &str) -> Result<OrchestrationResponse> {
        info!("🧬 Evolution request with focus: {}", focus);
        
        let content = format!(
            "🧬 Autonomous Evolution Analysis\n\n\
            Focus Area: {}\n\n\
            ⚠️ Evolution capabilities are currently in safe mode.\n\
            To enable full autonomous evolution:\n\
            1. Ensure safety validators are active\n\
            2. Set evolution config parameters\n\
            3. Use /evolve <focus> true to confirm\n\n\
            Available evolution focuses:\n\
            • reasoning - Enhance reasoning capabilities\n\
            • creativity - Evolve creative processes\n\
            • efficiency - Optimize performance\n\
            • knowledge - Expand knowledge integration",
            if focus.is_empty() { "general" } else { focus }
        );
        
        Ok(OrchestrationResponse {
            primary_response: content,
            intent: None,
            extracted_tasks: vec![],
            suggestions: vec![
                "Review current capabilities with /status".to_string(),
                "Check evolution history with /history evolution".to_string(),
            ],
            session_context: json!({"evolution_requested": true}),
            follow_up_needed: false,
            reasoning_chain: None,
            confidence: 0.95,
        })
    }
    
    // Story command implementation
    #[cfg(feature = "story-enhancement")]
    async fn story_driven_process(&self, args: &str) -> Result<OrchestrationResponse> {
        if let Some(story) = self.story_enhancement.read().await.as_ref() {
            let suggestions = story.get_story_suggestions().await;
            let content = format!(
                "📖 Story-Driven Mode\n\n{}",
                StoryChatIntegration::format_story_help()
            );
            
            Ok(OrchestrationResponse {
                primary_response: content,
                intent: None,
                extracted_tasks: vec![],
                suggestions,
                session_context: json!({"mode": "story"}),
                follow_up_needed: true,
                reasoning_chain: None,
                confidence: 0.9,
            })
        } else {
            Ok(OrchestrationResponse {
                primary_response: "📖 Story-driven features not yet initialized. Initialize with cognitive system first.".to_string(),
                intent: None,
                extracted_tasks: vec![],
                suggestions: vec![],
                session_context: json!({"mode": "story"}),
                follow_up_needed: false,
                reasoning_chain: None,
                confidence: 0.85,
            })
        }
    }
    
    // Agent command implementation
    #[cfg(feature = "agent-streams")]
    async fn handle_agent_command(
        &self,
        session_id: &str,
        command: &str,
        args: &str,
    ) -> Result<OrchestrationResponse> {
        #[cfg(feature = "agent-streams")]
        if let Some(manager) = &self.agent_stream_manager {
            match command {
                "spawn" => {
                    let agent_id = Uuid::new_v4().to_string();
                    manager.create_agent_stream(
                        "general".to_string(), // agent type
                        format!("Agent-{}", &agent_id[..8]), // agent name
                        args.to_string(),
                    ).await?;
                    
                    Ok(OrchestrationResponse {
                        primary_response: format!("🤖 Spawning agent {} for task: {}", agent_id, args),
                        intent: None,
                        extracted_tasks: vec![],
                        suggestions: vec!["Use /agent status to check agent status".to_string()],
                        session_context: json!({"spawned_agent": agent_id}),
                        follow_up_needed: false,
                        reasoning_chain: None,
                        confidence: 0.95,
                    })
                }
                "agent" => {
                    let agents = manager.get_active_streams().await;
                    let content = format!(
                        "🤖 Active Agents: {}\n\nUse /spawn <task> to create a new agent",
                        agents.len()
                    );
                    
                    Ok(OrchestrationResponse {
                        primary_response: content,
                        intent: None,
                        extracted_tasks: vec![],
                        suggestions: vec![],
                        session_context: json!({"agent_count": agents.len()}),
                        follow_up_needed: false,
                        reasoning_chain: None,
                        confidence: 1.0,
                    })
                }
                _ => Err(anyhow!("Unknown agent command"))
            }
        } else {
            Err(anyhow!("Agent streams not enabled"))
        }
    }
    
    // Helper methods from base orchestrator
    async fn get_or_create_session(&self, session_id: &str) -> Result<ConversationSession> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get(session_id) {
            Ok(session.clone())
        } else {
            let new_session = ConversationSession {
                id: session_id.to_string(),
                context: ConversationContext::default(),
                conversation_history: Vec::new(),
                created_at: chrono::Utc::now(),
                last_active: chrono::Utc::now(),
                
                #[cfg(feature = "deep-cognition")]
                cognitive_modes: CognitiveModes::default(),
                
                #[cfg(feature = "agent-streams")]
                active_agents: Vec::new(),
            };
            
            sessions.insert(session_id.to_string(), new_session.clone());
            Ok(new_session)
        }
    }
    
    async fn analyze_input(
        &self,
        input: &str,
        session: &ConversationSession,
    ) -> Result<CognitiveUnderstanding> {
        // Use enhanced NLP analyzer if available
        if let Some(nlp_analyzer) = &self.enhanced_nlp_analyzer {
            info!("🔤 Using enhanced NLP analyzer for deep understanding");
            let nlp_result = nlp_analyzer.analyze(input, &session.id).await?;
            
            // Extract concepts from semantic representation
            let concepts: Vec<String> = nlp_result.semantics.predicates.iter()
                .map(|p| p.lemma.clone())
                .collect();
            
            // Extract entities
            let entities: Vec<String> = nlp_result.entities.iter()
                .map(|e| e.text.clone())
                .collect();
            
            // Calculate complexity based on parse tree depth and dependencies
            let complexity = (nlp_result.parsed.dependencies.len() as f32 / 10.0).min(1.0);
            
            // Get emotional tone from sentiment
            let emotional_tone = (nlp_result.sentiment.overall_sentiment.polarity + 1.0) / 2.0;
            
            // Calculate urgency based on detected intents
            let urgency = if nlp_result.intents.iter().any(|i| i.intent.contains("urgent") || i.intent.contains("asap")) {
                0.9
            } else {
                0.5
            };
            
            return Ok(CognitiveUnderstanding {
                concepts,
                entities,
                complexity,
                emotional_tone,
                urgency,
                confidence: nlp_result.context_relevance as f64,
            });
        }
        
        // Fallback to basic cognitive system analysis
        let response = self.cognitive_system.process_query(input).await?;
        
        Ok(CognitiveUnderstanding {
            concepts: vec![],  // Would extract from response
            entities: vec![],  // Would extract from response
            complexity: 0.5,   // Would calculate
            emotional_tone: 0.5,
            urgency: 0.5,
            confidence: 0.8,
        })
    }
    
    async fn recognize_intent(
        &self,
        input: &str,
        understanding: &CognitiveUnderstanding,
    ) -> Result<Intent> {
        let input_lower = input.to_lowercase();
        
        // Check for greetings first
        let greeting_words = ["hey", "hi", "hello", "howdy", "greetings", "sup", "yo", "morning", "evening", "afternoon"];
        if greeting_words.iter().any(|&word| input_lower.starts_with(word) || input_lower == word) {
            return Ok(Intent {
                primary_intent: IntentType::GeneralConversation,
                confidence: 0.95,
                parameters: [("type".to_string(), json!("greeting"))].into(),
            });
        }
        
        // Check for farewells
        let farewell_words = ["bye", "goodbye", "see you", "farewell", "later", "ciao"];
        if farewell_words.iter().any(|&word| input_lower.contains(word)) {
            return Ok(Intent {
                primary_intent: IntentType::GeneralConversation,
                confidence: 0.95,
                parameters: [("type".to_string(), json!("farewell"))].into(),
            });
        }
        
        // Check for thanks
        if input_lower.contains("thank") || input_lower.contains("thanks") {
            return Ok(Intent {
                primary_intent: IntentType::GeneralConversation,
                confidence: 0.9,
                parameters: [("type".to_string(), json!("gratitude"))].into(),
            });
        }
        
        // Check for questions
        let intent_type = if input.contains("?") || input_lower.starts_with("what") || input_lower.starts_with("how") 
            || input_lower.starts_with("why") || input_lower.starts_with("when") || input_lower.starts_with("where")
            || input_lower.starts_with("who") || input_lower.starts_with("can") || input_lower.starts_with("could") {
            IntentType::AskQuestion
        } else if input.contains("create") || input.contains("make") || input.contains("build") 
            || input.contains("generate") || input.contains("design") || input.contains("set up")
            || input.contains("setup") || input.contains("start") || input.contains("new project") {
            IntentType::CreateContent
        } else if input.contains("fix") || input.contains("debug") || input.contains("solve") 
            || input.contains("repair") || input.contains("troubleshoot") {
            IntentType::FixIssue
        } else {
            IntentType::GeneralConversation
        };
        
        Ok(Intent {
            primary_intent: intent_type,
            confidence: understanding.confidence,
            parameters: HashMap::new(),
        })
    }
    
    async fn extract_tasks(
        &self,
        input: &str,
        intent: &Intent,
        understanding: &CognitiveUnderstanding,
    ) -> Result<Vec<ExtractedTask>> {
        let mut tasks = Vec::new();
        let input_lower = input.to_lowercase();
        
        // Keywords that indicate task creation
        let task_indicators = [
            "need to", "want to", "should", "must", "have to",
            "can you", "could you", "please", "would you",
            "implement", "create", "build", "develop", "design",
            "fix", "solve", "analyze", "research", "document",
            "test", "review", "refactor", "optimize", "setup"
        ];
        
        // Check if input contains task indicators
        let contains_task = task_indicators.iter().any(|&indicator| input_lower.contains(indicator));
        
        if !contains_task && intent.primary_intent != IntentType::CreateContent {
            return Ok(tasks);
        }
        
        // Extract main task based on intent
        match &intent.primary_intent {
            IntentType::CreateContent => {
                tasks.push(self.extract_creation_task(input, understanding)?);
            }
            IntentType::FixIssue => {
                tasks.push(self.extract_problem_solving_task(input, understanding)?);
            }
            IntentType::ExecuteCommand => {
                tasks.push(self.extract_execution_task(input, understanding)?);
            }
            IntentType::SearchInformation => {
                tasks.push(self.extract_research_task(input, understanding)?);
            }
            _ => {
                // Try to extract generic task
                if contains_task {
                    tasks.push(self.extract_generic_task(input, understanding)?);
                }
            }
        }
        
        // Extract subtasks if the main task is complex
        for task in tasks.clone() {
            if self.is_complex_task(&task) {
                let subtasks = self.extract_subtasks(input, &task)?;
                tasks.extend(subtasks);
            }
        }
        
        Ok(tasks)
    }
    
    /// Extract a creation/development task
    fn extract_creation_task(&self, input: &str, understanding: &CognitiveUnderstanding) -> Result<ExtractedTask> {
        let input_lower = input.to_lowercase();
        
        // Determine what's being created
        let task_type = if input_lower.contains("implement") || input_lower.contains("function") || input_lower.contains("code") {
            ExtractedTaskType::CodingTask
        } else if input_lower.contains("document") || input_lower.contains("readme") {
            ExtractedTaskType::DocumentationTask
        } else if input_lower.contains("design") || input_lower.contains("architecture") {
            ExtractedTaskType::DesignTask
        } else {
            ExtractedTaskType::GeneralTask
        };
        
        // Estimate complexity based on keywords
        let complexity_keywords = ["complex", "comprehensive", "full", "complete", "entire", "advanced"];
        let is_complex = complexity_keywords.iter().any(|&kw| input_lower.contains(kw));
        
        Ok(ExtractedTask {
            description: self.clean_task_description(input),
            priority: if is_complex { TaskPriority::High } else { TaskPriority::Medium },
            task_type,
            confidence: understanding.confidence * 0.9,
            estimated_effort: if is_complex { 
                Some(Duration::from_secs(3600)) // 1 hour for complex tasks
            } else { 
                Some(Duration::from_secs(1800)) // 30 mins for normal tasks
            },
            dependencies: vec![],
        })
    }
    
    /// Extract a problem-solving task
    fn extract_problem_solving_task(&self, input: &str, understanding: &CognitiveUnderstanding) -> Result<ExtractedTask> {
        let input_lower = input.to_lowercase();
        
        let priority = if input_lower.contains("urgent") || input_lower.contains("critical") || input_lower.contains("asap") {
            TaskPriority::Critical
        } else if input_lower.contains("bug") || input_lower.contains("error") || input_lower.contains("broken") {
            TaskPriority::High
        } else {
            TaskPriority::Medium
        };
        
        Ok(ExtractedTask {
            description: self.clean_task_description(input),
            priority,
            task_type: ExtractedTaskType::CodingTask,
            confidence: understanding.confidence * 0.85,
            estimated_effort: Some(Duration::from_secs(2400)), // 40 mins
            dependencies: vec![],
        })
    }
    
    /// Extract an execution/command task
    fn extract_execution_task(&self, input: &str, understanding: &CognitiveUnderstanding) -> Result<ExtractedTask> {
        Ok(ExtractedTask {
            description: self.clean_task_description(input),
            priority: TaskPriority::Medium,
            task_type: ExtractedTaskType::GeneralTask,
            confidence: understanding.confidence * 0.95,
            estimated_effort: Some(Duration::from_secs(600)), // 10 mins
            dependencies: vec![],
        })
    }
    
    /// Extract a research task
    fn extract_research_task(&self, input: &str, understanding: &CognitiveUnderstanding) -> Result<ExtractedTask> {
        Ok(ExtractedTask {
            description: self.clean_task_description(input),
            priority: TaskPriority::Medium,
            task_type: ExtractedTaskType::ResearchTask,
            confidence: understanding.confidence * 0.8,
            estimated_effort: Some(Duration::from_secs(1200)), // 20 mins
            dependencies: vec![],
        })
    }
    
    /// Extract a generic task
    fn extract_generic_task(&self, input: &str, understanding: &CognitiveUnderstanding) -> Result<ExtractedTask> {
        let input_lower = input.to_lowercase();
        
        // Try to determine task type from keywords
        let task_type = if input_lower.contains("test") {
            ExtractedTaskType::TestingTask
        } else if input_lower.contains("document") || input_lower.contains("write") {
            ExtractedTaskType::DocumentationTask
        } else if input_lower.contains("design") {
            ExtractedTaskType::DesignTask
        } else {
            ExtractedTaskType::GeneralTask
        };
        
        Ok(ExtractedTask {
            description: self.clean_task_description(input),
            priority: TaskPriority::Medium,
            task_type,
            confidence: understanding.confidence * 0.7,
            estimated_effort: Some(Duration::from_secs(1800)), // 30 mins default
            dependencies: vec![],
        })
    }
    
    /// Check if a task is complex enough to need subtasks
    fn is_complex_task(&self, task: &ExtractedTask) -> bool {
        // Complex if estimated effort > 45 minutes or contains certain keywords
        if let Some(effort) = &task.estimated_effort {
            if effort.as_secs() > 2700 {
                return true;
            }
        }
        
        let complex_indicators = ["and", "then", "also", "with", "including"];
        complex_indicators.iter().any(|&indicator| task.description.to_lowercase().contains(indicator))
    }
    
    /// Extract subtasks from a complex task
    fn extract_subtasks(&self, input: &str, parent_task: &ExtractedTask) -> Result<Vec<ExtractedTask>> {
        let mut subtasks = Vec::new();
        
        // Split by conjunctions and task indicators
        let parts: Vec<&str> = input.split(|c| c == ',' || c == ';').collect();
        if parts.len() <= 1 {
            // Try splitting by "and", "then", etc.
            let conjunctions = [" and ", " then ", " also ", " with "];
            for conj in &conjunctions {
                if input.contains(conj) {
                    let parts: Vec<&str> = input.split(conj).collect();
                    for (i, part) in parts.iter().enumerate().skip(1) {
                        if part.len() > 10 { // Minimum length for a meaningful subtask
                            subtasks.push(ExtractedTask {
                                description: format!("{} (Part {})", self.clean_task_description(part), i + 1),
                                priority: parent_task.priority.clone(),
                                task_type: parent_task.task_type.clone(),
                                confidence: parent_task.confidence * 0.8,
                                estimated_effort: Some(Duration::from_secs(900)), // 15 mins per subtask
                                dependencies: vec![parent_task.description.clone()],
                            });
                        }
                    }
                    break;
                }
            }
        }
        
        Ok(subtasks)
    }
    
    /// Clean and format task description
    fn clean_task_description(&self, input: &str) -> String {
        let mut description = input.trim().to_string();
        
        // Remove common prefixes
        let prefixes = ["please ", "can you ", "could you ", "would you ", "i need to ", "i want to ", "let's "];
        for prefix in &prefixes {
            if description.to_lowercase().starts_with(prefix) {
                description = description[prefix.len()..].to_string();
                break;
            }
        }
        
        // Capitalize first letter
        if let Some(first_char) = description.chars().next() {
            description = first_char.to_uppercase().collect::<String>() + &description[1..];
        }
        
        // Ensure it ends with proper punctuation
        if !description.ends_with('.') && !description.ends_with('!') && !description.ends_with('?') {
            description.push('.');
        }
        
        description
    }
    
    /// Handle decomposed tasks - create in task manager and prepare for delegation
    async fn handle_decomposed_tasks(
        &self,
        session_id: &str,
        decomposed_tasks: Vec<DecomposedTask>,
        original_tasks: &[ExtractedTask],
    ) -> Result<OrchestrationResponse> {
        let mut created_tasks = Vec::new();
        let mut task_summaries = Vec::new();
        
        // Create tasks in task manager
        for decomposed in &decomposed_tasks {
            // Create parent task
            match self.task_manager.create_task(
                &decomposed.parent_task.description,
                Some("Complex task requiring decomposition"),
                decomposed.parent_task.priority.clone(),
                None,
                crate::tools::task_management::TaskPlatform::Internal,
            ).await {
                Ok(parent_task) => {
                    task_summaries.push(format!(
                        "📋 Created parent task: {} (ID: {})",
                        parent_task.title,
                        &parent_task.id[..8]
                    ));
                    
                    // Create subtasks
                    for subtask in &decomposed.subtasks {
                        match self.task_manager.create_task(
                            &subtask.description,
                            Some(&format!("Subtask of: {}", parent_task.title)),
                            subtask.priority.clone(),
                            None,
                            crate::tools::task_management::TaskPlatform::Internal,
                        ).await {
                            Ok(created_subtask) => {
                                created_tasks.push((created_subtask.id.clone(), subtask.clone()));
                                task_summaries.push(format!(
                                    "  ↳ Created subtask: {} (Agent: {:?})",
                                    created_subtask.title,
                                    subtask.preferred_agent_type
                                ));
                            }
                            Err(e) => {
                                warn!("Failed to create subtask: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to create parent task: {}", e);
                }
            }
        }
        
        // Try to automatically map tasks to agents if mapper is available
        let mut agent_assignments = Vec::new();
        if let Some(mapper) = &self.agent_task_mapper {
            info!("🤖 Mapping tasks to agents...");
            
            for (task_id, subtask) in &created_tasks {
                match mapper.map_task_to_agent(&subtask).await {
                    Ok(assignment) => {
                        task_summaries.push(format!(
                            "    🤖 Assigned to: {} ({})",
                            assignment.agent_id,
                            assignment.assignment_reason
                        ));
                        agent_assignments.push(assignment);
                    }
                    Err(e) => {
                        warn!("Failed to map task {} to agent: {}", task_id, e);
                    }
                }
            }
            
            // Create agent streams if we have assignments
            #[cfg(feature = "agent-streams")]
            if !agent_assignments.is_empty() && self.agent_stream_manager.is_some() {
                let stream_manager = self.agent_stream_manager.as_ref().unwrap();
                
                // Use first decomposed task as context
                if let Some(first_decomposed) = decomposed_tasks.first() {
                    match mapper.create_agent_streams_with_context(
                        agent_assignments.clone(),
                        &format!("task_{}", uuid::Uuid::new_v4()),
                        &first_decomposed.parent_task.description,
                        &first_decomposed.subtasks,
                    ).await {
                    Ok(stream_ids) => {
                        task_summaries.push(format!(
                            "\n🚀 Created {} agent streams for parallel execution",
                            stream_ids.len()
                        ));
                        
                        // Register task hierarchy for progress tracking
                        if let Some(aggregator) = &self.task_progress_aggregator {
                            for decomposed in &decomposed_tasks {
                                let parent_task = &decomposed.parent_task;
                                let subtask_ids: Vec<String> = created_tasks.iter()
                                    .filter(|(_, subtask)| 
                                        decomposed.subtasks.iter().any(|s| s.description == subtask.description)
                                    )
                                    .map(|(id, _)| id.clone())
                                    .collect();
                                    
                                let task_to_agent: HashMap<String, String> = agent_assignments.iter()
                                    .filter(|a| subtask_ids.contains(&a.task_id))
                                    .map(|a| (a.task_id.clone(), a.agent_id.clone()))
                                    .collect();
                                    
                                if let Err(e) = aggregator.register_task_hierarchy(
                                    parent_task.description.clone(), // Using description as ID for now
                                    subtask_ids,
                                    task_to_agent,
                                ).await {
                                    warn!("Failed to register task hierarchy: {}", e);
                                } else {
                                    // Start monitoring
                                    let _ = aggregator.start_monitoring().await;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to create agent streams: {}", e);
                    }
                }
                }
            }
        }
        
        // Prepare delegation info
        let delegation_summary = if !agent_assignments.is_empty() {
            format!(
                "\n\n✅ Automatically assigned {} subtasks to specialized agents.\n\
                Task execution has begun in parallel.",
                agent_assignments.len()
            )
        } else if !created_tasks.is_empty() {
            format!(
                "\n\n🤖 Ready to delegate {} subtasks to specialized agents.\n\
                Use /agent spawn to start task execution.",
                created_tasks.len()
            )
        } else {
            String::new()
        };
        
        Ok(OrchestrationResponse {
            primary_response: format!(
                "I've analyzed your request and broken it down into manageable tasks:\n\n{}{}",
                task_summaries.join("\n"),
                delegation_summary
            ),
            intent: None,
            extracted_tasks: original_tasks.to_vec(),
            suggestions: vec![
                "Review the task breakdown".to_string(),
                "Use /agent spawn to start parallel execution".to_string(),
                "Check task progress with /tasks status".to_string(),
            ],
            session_context: json!({
                "decomposed_tasks": decomposed_tasks.len(),
                "created_subtasks": created_tasks.len(),
                "ready_for_delegation": true
            }),
            follow_up_needed: false,
            confidence: 0.9,
        })
    }
    
    async fn validate_safety(
        &self,
        input: &str,
        intent: &Intent,
    ) -> Result<()> {
        // Use safety validator
        // For now, simple check
        if input.to_lowercase().contains("harmful") {
            Err(anyhow!("Request contains potentially harmful content"))
        } else {
            Ok(())
        }
    }
    
    /// Apply cognitive tone to enhance processing
    async fn apply_cognitive_tone(
        &self,
        session_id: &str,
        input: &str,
        tone_result: &ToneDetectionResult,
        understanding: &CognitiveUnderstanding,
    ) -> Result<Option<OrchestrationResponse>> {
        // Only apply if we have cognitive features enabled
        #[cfg(not(feature = "deep-cognition"))]
        {
            return Ok(None);
        }
        
        #[cfg(feature = "deep-cognition")]
        {
            if !self.config.deep_cognition_enabled {
                return Ok(None);
            }
            
            // Map cognitive tone to command
            let cognitive_command = tone_result.primary_tone.as_command();
            
            // Build enhanced input with cognitive context
            let enhanced_input = format!(
                "{} (Detected intent: {}, Intensity: {:.2})",
                input,
                tone_result.primary_tone.description(),
                tone_result.intensity
            );
            
            // Apply the appropriate cognitive processing based on tone
            let response = match tone_result.primary_tone {
                CognitiveTone::Analytical => {
                    info!("🤔 Applying analytical cognitive tone");
                    self.deep_think(&enhanced_input).await?
                }
                CognitiveTone::Creative => {
                    info!("🎨 Applying creative cognitive tone");
                    self.creative_process(&enhanced_input).await?
                }
                CognitiveTone::Empathetic => {
                    info!("💝 Applying empathetic cognitive tone");
                    self.empathize_with(&enhanced_input).await?
                }
                CognitiveTone::Narrative => {
                    info!("📖 Applying narrative cognitive tone");
                    #[cfg(feature = "story-enhancement")]
                    if self.config.story_enhancement_enabled {
                        return Ok(Some(self.story_driven_process(&enhanced_input).await?));
                    }
                    self.deep_think(&enhanced_input).await?
                }
                CognitiveTone::Reflective => {
                    info!("🧘 Applying reflective cognitive tone");
                    self.deep_think(&format!("Reflect deeply on: {}", enhanced_input)).await?
                }
                CognitiveTone::ProblemSolving => {
                    info!("🔧 Applying problem-solving cognitive tone");
                    self.deep_analyze(&enhanced_input).await?
                }
                CognitiveTone::Exploratory => {
                    info!("🔍 Applying exploratory cognitive tone");
                    self.deep_analyze(&format!("Explore: {}", enhanced_input)).await?
                }
                CognitiveTone::Evolutionary => {
                    info!("🧬 Applying evolutionary cognitive tone");
                    self.trigger_evolution(&enhanced_input).await?
                }
            };
            
            // Add tone metadata to the response
            let mut enhanced_response = response;
            // Note: ReasoningChain doesn't have add_step method, so we'll add to suggestions instead
            enhanced_response.suggestions.insert(0, format!(
                "🎨 Applied {} tone (confidence: {:.2}, intensity: {:.2})",
                tone_result.primary_tone.description(),
                tone_result.confidence,
                tone_result.intensity
            ));
            
            // Add secondary tone suggestions
            for (secondary_tone, confidence) in &tone_result.secondary_tones {
                if *confidence > 0.3 {
                    enhanced_response.suggestions.push(format!(
                        "You might also want to explore this with a {} approach",
                        secondary_tone.description()
                    ));
                }
            }
            
            Ok(Some(enhanced_response))
        }
    }
    
    async fn generate_response(
        &self,
        session_id: &str,
        input: &str,
        intent: &Intent,
        understanding: &CognitiveUnderstanding,
        tasks: &[ExtractedTask],
    ) -> Result<OrchestrationResponse> {
        // Handle greetings and casual conversation directly
        if intent.primary_intent == IntentType::GeneralConversation {
            if let Some(conv_type) = intent.parameters.get("type").and_then(|v| v.as_str()) {
                let response_text = match conv_type {
                    "greeting" => {
                        let greetings = [
                            "Hey there! How can I help you today?",
                            "Hello! What would you like to work on?",
                            "Hi! I'm here to assist you with your project.",
                            "Hey! Ready to help with whatever you need.",
                            "Hello! Let me know what you're working on.",
                        ];
                        // Use timestamp for variety
                        let idx = (chrono::Utc::now().timestamp() as usize) % greetings.len();
                        greetings[idx].to_string()
                    }
                    "farewell" => {
                        "Goodbye! Feel free to come back anytime you need help.".to_string()
                    }
                    "gratitude" => {
                        "You're welcome! Happy to help.".to_string()
                    }
                    _ => {
                        // For other general conversation, process normally
                        return self.generate_model_response(input, intent, understanding, tasks).await;
                    }
                };
                
                return Ok(OrchestrationResponse {
                    primary_response: response_text,
                    intent: Some(intent.clone()),
                    extracted_tasks: tasks.to_vec(),
                    suggestions: vec![],
                    session_context: json!({"conversation_type": conv_type}),
                    follow_up_needed: false,
                    confidence: 0.95,
                });
            }
        }
        
        // For all other intents, use the model orchestrator
        self.generate_model_response(input, intent, understanding, tasks).await
    }
    
    async fn generate_model_response(
        &self,
        input: &str,
        intent: &Intent,
        understanding: &CognitiveUnderstanding,
        tasks: &[ExtractedTask],
    ) -> Result<OrchestrationResponse> {
        // Add context about the intent to help the model respond appropriately
        let contextualized_input = match intent.primary_intent {
            IntentType::AskQuestion => format!("Answer this question: {}", input),
            IntentType::CreateContent => format!("Help create this: {}", input),
            IntentType::FixIssue => format!("Help fix this issue: {}", input),
            IntentType::ExecuteCommand => format!("Execute this command: {}", input),
            IntentType::GeneralConversation => input.to_string(),
            _ => input.to_string(),
        };
        
        let task_request = crate::models::orchestrator::TaskRequest {
            content: contextualized_input,
            task_type: crate::models::TaskType::GeneralChat,
            constraints: Default::default(),
            context_integration: true,  // Enable context
            memory_integration: true,   // Enable memory
            cognitive_enhancement: true, // Enable cognitive enhancement
        };
        
        let task_response = self.model_orchestrator
            .execute_with_fallback(task_request)
            .await?;
        
        let response_text = task_response.content;
        
        Ok(OrchestrationResponse {
            primary_response: response_text,
            intent: Some(intent.clone()),
            extracted_tasks: tasks.to_vec(),
            suggestions: vec![],
            session_context: json!({}),
            follow_up_needed: false,
            confidence: understanding.confidence,
        })
    }
    
    async fn update_session(
        &self,
        session_id: &str,
        input: &str,
        response: &OrchestrationResponse,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            session.conversation_history.push(ConversationTurn {
                timestamp: chrono::Utc::now(),
                user_input: input.to_string(),
                assistant_response: response.primary_response.clone(),
                intent: response.intent.clone(),
                metadata: Default::default(),
            });
            
            session.last_active = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Try to execute tools if the intent requires it
    async fn try_execute_tools(
        &self,
        input: &str,
        intent: &Intent,
        tasks: &[ExtractedTask],
    ) -> Result<Option<OrchestrationResponse>> {
        // Check if this is a tool-related request
        let input_lower = input.to_lowercase();
        
        // Expanded tool detection patterns
        let is_tool_request = 
            // File/folder operations
            input_lower.contains("create") && (input_lower.contains("file") || input_lower.contains("folder") || input_lower.contains("directory"))
            || input_lower.contains("make") && (input_lower.contains("file") || input_lower.contains("folder") || input_lower.contains("directory"))
            || input_lower.contains("new project") || input_lower.contains("set up") && input_lower.contains("project")
            || input_lower.contains("delete") || input_lower.contains("remove") 
            || input_lower.contains("copy") || input_lower.contains("move")
            || input_lower.contains("list") && (input_lower.contains("file") || input_lower.contains("folder"))
            || input_lower.contains("read") && input_lower.contains("file")
            || input_lower.contains("write") && input_lower.contains("file")
            // Search operations
            || input_lower.contains("search") || input_lower.contains("find") || input_lower.contains("look for")
            // Git operations
            || input_lower.contains("git") || input_lower.contains("commit") || input_lower.contains("push") || input_lower.contains("pull")
            // Code operations
            || input_lower.contains("analyze") && input_lower.contains("code")
            || input_lower.contains("run") || input_lower.contains("execute") || input_lower.contains("test")
            // Web operations
            || input_lower.contains("browse") || input_lower.contains("web") || input_lower.contains("url")
            // System operations
            || input_lower.contains("system") || input_lower.contains("process") || input_lower.contains("service")
            // Database operations
            || input_lower.contains("database") || input_lower.contains("query") || input_lower.contains("sql");
            
        // Also check if the intent suggests tool use
        let intent_suggests_tools = matches!(
            intent.primary_intent,
            IntentType::ExecuteCommand | IntentType::RequestInformation | IntentType::PerformTask
        );
        
        // If neither pattern matches and intent doesn't suggest tools, skip tool execution
        if !is_tool_request && !intent_suggests_tools {
            return Ok(None);
        }
        
        // Build tool request
        let tool_request = self.build_tool_request_from_input(input, intent, tasks)?;
        
        // Execute tool
        info!("🛠️ Executing tool request: {} with tool: {}", tool_request.intent, tool_request.tool_name);
        info!("🛠️ Tool parameters: {:?}", tool_request.parameters);
        match self.tool_manager.execute_tool_request(tool_request).await {
            Ok(result) => {
                let response_text = match &result.status {
                    crate::tools::ToolStatus::Success => {
                        format!("✅ Successfully completed: {}\n\n{}", 
                            result.content.get("message").and_then(|v| v.as_str()).unwrap_or("Operation completed"),
                            result.content.get("details").and_then(|v| v.as_str()).unwrap_or("")
                        )
                    }
                    crate::tools::ToolStatus::Partial(msg) => {
                        format!("⚠️ Partially completed: {}", msg)
                    }
                    crate::tools::ToolStatus::Failure(error) => {
                        format!("❌ Failed to complete: {}", error)
                    }
                };
                
                Ok(Some(OrchestrationResponse {
                    primary_response: response_text,
                    intent: Some(intent.clone()),
                    extracted_tasks: tasks.to_vec(),
                    suggestions: result.content.get("suggestions")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default(),
                    session_context: result.content.clone(),
                    follow_up_needed: false,
                    confidence: 0.9,
                }))
            }
            Err(e) => {
                warn!("⚠️ Tool execution failed: {}", e);
                warn!("⚠️ Falling back to regular processing for: {}", input);
                // Fall back to regular processing
                Ok(None)
            }
        }
    }
    
    /// Build a tool request from natural language input
    fn build_tool_request_from_input(
        &self,
        input: &str,
        intent: &Intent,
        tasks: &[ExtractedTask],
    ) -> Result<crate::tools::ToolRequest> {
        let input_lower = input.to_lowercase();
        
        // Extract operation type and parameters
        let (tool_name, parameters) = if input_lower.contains("search") || input_lower.contains("find") || input_lower.contains("look for") {
            // Web search operations
            let query = if input_lower.starts_with("search for") {
                input[11..].trim()
            } else if input_lower.starts_with("search") {
                input[6..].trim()
            } else if input_lower.starts_with("find") {
                input[4..].trim()
            } else if input_lower.contains("look for") {
                input.split("look for").nth(1).unwrap_or("").trim()
            } else {
                input
            };
            ("web_search", json!({
                "query": query,
            }))
        } else if input_lower.contains("create") || input_lower.contains("make") {
            if input_lower.contains("directory") || input_lower.contains("folder") {
                ("create_directory", json!({
                    "path": self.extract_path_from_input(input).unwrap_or_else(|| "new_directory".to_string()),
                }))
            } else if input_lower.contains("file") {
                ("create_file", json!({
                    "path": self.extract_path_from_input(input).unwrap_or_else(|| "new_file.txt".to_string()),
                    "content": ""
                }))
            } else if input_lower.contains("project") {
                // Create project structure
                let project_name = self.extract_project_name(input);
                ("create_directory", json!({
                    "path": project_name.clone(),
                }))
            } else {
                ("file_system", json!({"operation": "unknown"}))
            }
        } else if input_lower.contains("list") {
            ("list_directory", json!({
                "path": self.extract_path_from_input(input).unwrap_or_else(|| ".".to_string()),
            }))
        } else if input_lower.contains("read") {
            ("read_file", json!({
                "path": self.extract_path_from_input(input).unwrap_or_else(|| "file.txt".to_string()),
            }))
        } else if input_lower.contains("git") || input_lower.contains("commit") {
            // Git operations
            ("github", json!({
                "operation": if input_lower.contains("commit") { "commit" } else { "status" },
                "message": input
            }))
        } else if input_lower.contains("analyze") && input_lower.contains("code") {
            // Code analysis
            ("code_analysis", json!({
                "path": self.extract_path_from_input(input).unwrap_or_else(|| ".".to_string()),
            }))
        } else if input_lower.contains("browse") || input_lower.contains("web") || input_lower.contains("url") {
            // Web browsing
            ("web_browser", json!({
                "url": self.extract_url_from_input(input).unwrap_or_else(|| "https://google.com".to_string()),
            }))
        } else {
            // Default to trying to understand as a general query
            ("general_assistant", json!({"query": input}))
        };
        
        Ok(crate::tools::ToolRequest {
            intent: format!("Execute {} operation", tool_name),
            tool_name: tool_name.to_string(),
            context: format!("User request: {}", input),
            parameters,
            priority: 0.8,
            expected_result_type: crate::tools::ResultType::Structured,
            result_type: crate::tools::ResultType::Structured,
            timeout: Some(Duration::from_secs(30)),
            memory_integration: crate::tools::MemoryIntegration {
                store_result: true,
                importance: 0.7,
                tags: vec!["file_operation".to_string()],
                associations: vec![],
            },
        })
    }
    
    /// Extract path from natural language input
    fn extract_path_from_input(&self, input: &str) -> Option<String> {
        // Simple extraction - look for quoted paths or common patterns
        if let Some(start) = input.find('"') {
            if let Some(end) = input[start+1..].find('"') {
                return Some(input[start+1..start+1+end].to_string());
            }
        }
        
        // Look for common patterns like "called X" or "named X"
        for pattern in &["called ", "named ", "at ", "in "] {
            if let Some(pos) = input.to_lowercase().find(pattern) {
                let rest = &input[pos + pattern.len()..];
                let path = rest.split_whitespace().next()?;
                return Some(path.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '_' && c != '-' && c != '.').to_string());
            }
        }
        
        None
    }
    
    /// Extract project name from natural language input
    fn extract_project_name(&self, input: &str) -> String {
        // First try extract_path_from_input
        if let Some(name) = self.extract_path_from_input(input) {
            return name;
        }
        
        // Otherwise generate a default name
        format!("project_{}", chrono::Utc::now().timestamp())
    }
    
    fn extract_url_from_input(&self, input: &str) -> Option<String> {
        // Look for URLs in the input
        if let Some(start) = input.find("http://") {
            let url = &input[start..];
            let end = url.find(char::is_whitespace).unwrap_or(url.len());
            return Some(url[..end].to_string());
        }
        
        if let Some(start) = input.find("https://") {
            let url = &input[start..];
            let end = url.find(char::is_whitespace).unwrap_or(url.len());
            return Some(url[..end].to_string());
        }
        
        // Look for domain patterns
        for word in input.split_whitespace() {
            if word.contains(".com") || word.contains(".org") || word.contains(".net") {
                if word.starts_with("http") {
                    return Some(word.to_string());
                } else {
                    return Some(format!("https://{}", word));
                }
            }
        }
        
        None
    }
    

    /// Get available features
    pub fn available_features() -> Vec<&'static str> {
        let features = vec!["core-orchestration"];
        
        #[cfg(feature = "deep-cognition")]
        features.push("deep-cognition");
        
        #[cfg(feature = "story-enhancement")]
        features.push("story-enhancement");
        
        #[cfg(feature = "agent-streams")]
        features.push("agent-streams");
        
        features
    }
    
    /// Get memory reference
    pub fn memory(&self) -> &Arc<CognitiveMemory> {
        &self.memory
    }
    
    /// Get tool manager reference
    pub fn tool_manager(&self) -> &Arc<IntelligentToolManager> {
        &self.tool_manager
    }
    
    /// Get agent stream manager if available
    #[cfg(feature = "agent-streams")]
    pub fn get_agent_stream_manager(&self) -> Option<Arc<AgentStreamManager>> {
        self.agent_stream_manager.clone()
    }
    
    /// Process a message (compatibility method for NLP integration)
    pub async fn process_message(
        &self,
        message: &str,
        session_id: Option<String>,
    ) -> Result<String> {
        let session_id = session_id.unwrap_or_else(|| "default".to_string());
        
        match self.process_input(&session_id, message).await {
            Ok(response) => Ok(response.primary_response),
            Err(e) => {
                warn!("Failed to process message through orchestrator: {}", e);
                // Fallback to basic processing
                Ok(format!("I understand you're asking about: {}. Let me help you with that.", message))
            }
        }
    }
}

// Core types that are always available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResponse {
    pub primary_response: String,
    pub intent: Option<Intent>,
    pub extracted_tasks: Vec<ExtractedTask>,
    pub suggestions: Vec<String>,
    pub session_context: serde_json::Value,
    pub follow_up_needed: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub primary_intent: IntentType,
    pub confidence: f64,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IntentType {
    AskQuestion,
    ExecuteCommand,
    RequestExplanation,
    CreateContent,
    ModifyCode,
    AnalyzeCode,
    FixIssue,
    SearchInformation,
    ManageTasks,
    GeneralConversation,
    GenerateCode,
    CreateTask,
    RequestInformation,
    PerformTask,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveUnderstanding {
    pub concepts: Vec<String>,
    pub entities: Vec<String>,
    pub complexity: f32,
    pub emotional_tone: f32,
    pub urgency: f32,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTask {
    pub description: String,
    pub priority: TaskPriority,
    pub task_type: ExtractedTaskType,
    pub confidence: f64,
    pub estimated_effort: Option<Duration>,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExtractedTaskType {
    CodingTask,
    ResearchTask,
    DocumentationTask,
    TestingTask,
    DesignTask,
    GeneralTask,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSession {
    pub id: String,
    pub context: ConversationContext,
    pub conversation_history: Vec<ConversationTurn>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    
    #[cfg(feature = "deep-cognition")]
    pub cognitive_modes: CognitiveModes,
    
    #[cfg(feature = "agent-streams")]
    pub active_agents: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationContext {
    pub topics: Vec<String>,
    pub entities: HashMap<String, serde_json::Value>,
    pub user_preferences: HashMap<String, serde_json::Value>,
    pub active_goals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_input: String,
    pub assistant_response: String,
    pub intent: Option<Intent>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[cfg(feature = "deep-cognition")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CognitiveModes {
    pub creative_mode_active: bool,
    pub empathy_mode_active: bool,
    pub deep_thinking_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestratorEvent {
    SessionCreated { session_id: String },
    SessionUpdated { session_id: String },
    TaskExtracted { task: ExtractedTask },
    ToolExecuted { tool_name: String, success: bool },
    SafetyViolation { reason: String },
}

// Re-export for compatibility
pub use self::{
    OrchestrationResponse as HandlerResponse,
};