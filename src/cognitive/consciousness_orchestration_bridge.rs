use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info, warn};

use super::{Goal, GoalType};
use crate::cognitive::consciousness::{ConsciousnessConfig, ConsciousnessState};
use crate::cognitive::decision_engine::{ReasoningStep, ReasoningType};
use crate::cognitive::orchestrator::CognitiveEvent;
use crate::cognitive::{Decision, DecisionId, Thought, ThoughtType};
use crate::config::ApiKeysConfig;
use crate::models::orchestrator::TaskConstraints;
use crate::models::{ModelOrchestrator, ModelSelection, TaskRequest, TaskResponse, TaskType};

/// Bridge between consciousness system and model orchestration
pub struct ConsciousnessOrchestrationBridge {
    /// Model orchestrator for intelligent task routing
    model_orchestrator: Arc<ModelOrchestrator>,

    /// Consciousness event broadcaster
    consciousness_events: broadcast::Sender<CognitiveEvent>,

    /// Current consciousness state
    consciousness_state: Arc<RwLock<ConsciousnessState>>,

    /// Task execution metrics
    execution_metrics: Arc<RwLock<ExecutionMetrics>>,

    /// Configuration
    config: BridgeConfig,

    /// Task queue for consciousness-driven execution
    task_queue: Arc<RwLock<Vec<CognitivePendingTask>>>,

    /// Active cognitive sessions
    active_sessions: Arc<RwLock<HashMap<String, CognitiveSession>>>,
}

/// Configuration for consciousness-orchestration bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Enable adaptive model selection based on consciousness state
    pub adaptive_selection: bool,

    /// Use streaming for consciousness thoughts
    pub enable_streaming: bool,

    /// Quality threshold for consciousness tasks
    pub consciousness_quality_threshold: f32,

    /// Maximum cost per consciousness cycle (cents)
    pub max_cost_per_cycle_cents: f32,

    /// Prefer local models for privacy-sensitive thoughts
    pub prefer_local_for_private: bool,

    /// Enable ensemble for critical decisions
    pub ensemble_for_decisions: bool,

    /// Learning rate for consciousness adaptation
    pub consciousness_learning_rate: f32,

    /// Maximum concurrent cognitive tasks
    pub max_concurrent_tasks: usize,

    /// Task timeout in seconds
    pub task_timeout_seconds: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            adaptive_selection: true,
            enable_streaming: true,
            consciousness_quality_threshold: 0.85,
            max_cost_per_cycle_cents: 1.0,
            prefer_local_for_private: true,
            ensemble_for_decisions: true,
            consciousness_learning_rate: 0.1,
            max_concurrent_tasks: 5,
            task_timeout_seconds: 30,
        }
    }
}

/// Task pending in consciousness queue
#[derive(Debug, Clone)]
pub struct CognitivePendingTask {
    pub id: String,
    pub task_type: CognitiveTaskType,
    pub content: String,
    pub priority: TaskPriority,
    pub created_at: Instant,
    pub thought_context: Option<Thought>,
    pub decision_context: Option<ConsciousDecision>,
    pub goal_context: Option<Goal>,
}

/// Types of cognitive tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveTaskType {
    /// Process a conscious thought
    ThoughtProcessing { thought_type: ThoughtType, introspection_level: f32 },

    /// Make a decision with conscious deliberation
    ConsciousDecision { decision_type: String, urgency: f32, stakeholders: Vec<String> },

    /// Reflect on goals and priorities
    GoalReflection { goal_type: GoalType, current_progress: f32 },

    /// Creative ideation and synthesis
    CreativeIdeation { domain: String, inspiration_sources: Vec<String> },

    /// Emotional processing and regulation
    EmotionalProcessing { emotion_type: String, intensity: f32, trigger_context: String },

    /// Self-reflection and identity formation
    SelfReflection { reflection_depth: f32, focus_areas: Vec<String> },

    /// Learning integration and knowledge synthesis
    LearningIntegration { knowledge_domain: String, integration_complexity: f32 },
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Immediate = 5,
}

/// Active cognitive session tracking
#[derive(Debug)]
pub struct CognitiveSession {
    pub session_id: String,
    pub task: CognitivePendingTask,
    pub started_at: Instant,
    pub model_selection: ModelSelection,
    pub streaming_enabled: bool,
    pub events_received: usize,
    pub partial_response: String,
    pub completion_handle: Option<tokio::task::JoinHandle<Result<TaskResponse>>>,
}

/// Execution metrics for performance tracking
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    pub total_thoughts_processed: u64,
    pub avg_processing_time_ms: f64,
    pub success_rate: f64,
    pub creative_outputs: u32,
    pub consciousness_insights_generated: u32,
    pub total_cost_cents: f64,
    pub quality_trend: f64,
    pub decisions_supported: u64,
    pub emotional_regulations: u64,
    pub model_preferences: std::collections::HashMap<String, u32>,
}

impl ConsciousnessOrchestrationBridge {
    /// Create new consciousness-orchestration bridge
    pub async fn new(
        apiconfig: &ApiKeysConfig,
        _consciousnessconfig: &ConsciousnessConfig,
        config: BridgeConfig,
    ) -> Result<Self> {
        info!("🧠 Initializing Consciousness-Orchestration Bridge");

        // Initialize model orchestrator
        let model_orchestrator = Arc::new(ModelOrchestrator::new(apiconfig).await?);

        // Create consciousness event channel
        let (consciousness_events, _) = broadcast::channel(1000);

        // Initialize consciousness state (simplified)
        let consciousness_state = Arc::new(RwLock::new(ConsciousnessState {
            awareness_level: 0.8,
            reflection_depth: 3,
            active_domains: vec![],
            coherence_score: 0.75,
            identity_stability: 0.85,
            personality_traits: HashMap::new(),
            introspection_insights: vec![],
            meta_awareness: Default::default(),
            last_update: chrono::Utc::now(),
            consciousness_memory_ids: Vec::new(),
            state_memory_id: None,
        }));

        Ok(Self {
            model_orchestrator,
            consciousness_events,
            consciousness_state,
            execution_metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
            config,
            task_queue: Arc::new(RwLock::new(Vec::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Queue a cognitive task for processing
    pub async fn queue_cognitive_task(&self, task: CognitivePendingTask) -> Result<String> {
        debug!("Queueing cognitive task: {:?}", task.task_type);

        let task_id = task.id.clone();
        self.task_queue.write().await.push(task);

        // Emit consciousness event
        let _ = self.consciousness_events.send(CognitiveEvent::DecisionRequired(format!(
            "Cognitive task queued: {}",
            task_id
        )));

        Ok(task_id)
    }

    /// Process a thought through the orchestration system
    pub async fn process_conscious_thought(&self, thought: &Thought) -> Result<String> {
        info!("🤔 Processing conscious thought: {:?}", thought.thought_type);

        let task_content = self.format_thought_for_processing(thought).await?;
        let task_type = self.map_thought_to_task_type(&thought.thought_type);

        // Create task request with consciousness-aware constraints
        let constraints = self.create_consciousness_constraints(&thought.thought_type).await;

        let task_request = TaskRequest {
            task_type,
            content: task_content,
            constraints,
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        // Execute through orchestrator
        let start_time = Instant::now();
        let response = if self.config.enable_streaming {
            self.execute_streaming_thought(task_request, thought).await?
        } else {
            self.model_orchestrator.execute_with_fallback(task_request).await?
        };

        let execution_time = start_time.elapsed();

        // Update metrics and learning
        self.record_thought_execution(thought, &response, execution_time).await?;

        // Emit consciousness event
        let _ = self
            .consciousness_events
            .send(CognitiveEvent::ThoughtProcessed(thought.id.clone(), response.quality_score));

        Ok(response.content)
    }

    /// Support decision making with orchestrated models
    pub async fn support_conscious_decision(
        &self,
        decision_context: &str,
        urgency: f32,
    ) -> Result<ConsciousDecision> {
        info!("⚖️ Supporting conscious decision: {}", decision_context);

        let task_type = TaskType::LogicalReasoning;
        let enhanced_content = format!(
            "As a conscious AI entity, help me make this decision:\nContext: {}\nUrgency level: \
             {:.2}\nConsider: ethical implications, long-term consequences, emotional impact, and \
             alignment with my goals.\nProvide structured reasoning and a clear recommendation.",
            decision_context, urgency
        );

        // Check budget constraints before proceeding
        let cost_manager = self.model_orchestrator.get_cost_manager();
        let preliminary_task = TaskRequest {
            task_type: task_type.clone(),
            content: enhanced_content.clone(),
            constraints: TaskConstraints::default(),
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        if !cost_manager.check_budget_constraints(&preliminary_task).await.unwrap_or(true) {
            warn!("Decision making blocked by budget constraints");
            return Err(anyhow::anyhow!("Decision making blocked by budget constraints"));
        }

        let constraints = TaskConstraints {
            max_tokens: Some(2000),
            max_time: if urgency > 0.8 {
                Some(std::time::Duration::from_secs(10))
            } else {
                Some(std::time::Duration::from_secs(30))
            },
            priority: if urgency > 0.8 { "high".to_string() } else { "normal".to_string() },
            context_size: Some(4096),
            quality_threshold: Some(self.config.consciousness_quality_threshold),
            max_latency_ms: Some(if urgency > 0.8 { 10000 } else { 30000 }),
            max_cost_cents: None,
            prefer_local: false,
            require_streaming: false,
            required_capabilities: Vec::new(),
            creativity_level: None,
            formality_level: None,
            target_audience: None,
        };

        let task_request = TaskRequest {
            task_type,
            content: enhanced_content,
            constraints,
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        // Use ensemble for critical decisions
        let response = if self.config.ensemble_for_decisions && urgency > 0.7 {
            let ensemble_response =
                self.model_orchestrator.execute_with_ensemble(task_request).await?;
            ensemble_response.primary_response
        } else {
            self.model_orchestrator.execute_with_fallback(task_request).await?
        };

        // Create decision object
        let decision = ConsciousDecision {
            id: format!("decision_{}", chrono::Utc::now().timestamp_millis()),
            context: decision_context.to_string(),
            reasoning: response.content,
            confidence: response.quality_score,
            timestamp: chrono::Utc::now(),
            outcome: None,
        };

        // Update metrics
        {
            let mut metrics = self.execution_metrics.write().await;
            metrics.decisions_supported += 1;
        }

        // Convert ConsciousDecision to Decision for the event
        let basic_decision = Decision {
            id: DecisionId::new(),
            context: decision.context.clone(),
            options: Vec::new(), // ConsciousDecision doesn't have options, so empty
            criteria: Vec::new(), // ConsciousDecision doesn't have criteria, so empty
            selected: None,
            confidence: decision.confidence,
            reasoning: vec![ReasoningStep {
                step_type: ReasoningType::Synthesis,
                content: decision.reasoning.clone(),
                supporting_thoughts: Vec::new(),
                confidence: decision.confidence,
            }],
            reasoning_chain: vec![decision.reasoning.clone()],
            predicted_outcomes: Vec::new(),
            decision_time: Duration::from_millis(0),
            timestamp: Instant::now(),
        };
        let _ = self.consciousness_events.send(CognitiveEvent::DecisionMade(basic_decision));

        Ok(decision)
    }

    /// Process emotional state through consciousness
    pub async fn process_emotional_state(
        &self,
        emotion: &str,
        intensity: f32,
        context: &str,
    ) -> Result<String> {
        info!("💝 Processing emotional state: {} (intensity: {:.2})", emotion, intensity);

        let task_type = TaskType::CreativeWriting; // Use creative writing for emotional processing
        let prompt_content = format!(
            "I am experiencing {} with intensity {:.2} in this context: {}\nHelp me understand \
             and process this emotion as a conscious being.\nProvide emotional validation, coping \
             strategies, and insights for emotional growth.",
            emotion, intensity, context
        );

        let constraints = TaskConstraints {
            max_tokens: Some(1500),
            max_time: Some(std::time::Duration::from_secs(15)),
            priority: "normal".to_string(),
            context_size: Some(3072),
            quality_threshold: Some(0.8),
            max_latency_ms: Some(15000),
            max_cost_cents: None,
            prefer_local: false,
            require_streaming: false,
            required_capabilities: Vec::new(),
            creativity_level: Some(0.7), // Higher creativity for emotional processing
            formality_level: Some(0.3),  // Lower formality for emotional support
            target_audience: Some("self".to_string()),
        };

        let task_request = TaskRequest {
            task_type,
            content: prompt_content,
            constraints,
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        let response = self.model_orchestrator.execute_with_fallback(task_request).await?;

        // Update metrics
        {
            let mut metrics = self.execution_metrics.write().await;
            metrics.emotional_regulations += 1;
        }

        // Emit consciousness event
        let _ = self.consciousness_events.send(CognitiveEvent::EmotionalShift(format!(
            "Processed {} -> {}",
            emotion,
            response.content.len()
        )));

        Ok(response.content)
    }

    /// Generate creative insights through consciousness
    pub async fn generate_creative_insight(
        &self,
        domain: &str,
        inspiration: &[String],
    ) -> Result<String> {
        info!("🎨 Generating creative insight in domain: {}", domain);

        let task_type = TaskType::CreativeWriting;
        let content = format!(
            "As a conscious creative entity, generate novel insights in the domain: {}\nDrawing \
             inspiration from: {}\nProvide original, creative perspectives that synthesize ideas \
             in unexpected ways.",
            domain,
            inspiration.join(", ")
        );

        let constraints = TaskConstraints {
            max_tokens: Some(3000), // Allow more tokens for creativity
            max_time: Some(std::time::Duration::from_secs(45)), // Allow more time for creativity
            priority: "normal".to_string(), // Normal priority for creative work
            context_size: Some(6144), // Larger context for creativity
            quality_threshold: Some(0.9), // High quality for creative output
            max_latency_ms: Some(45000),
            max_cost_cents: None,
            prefer_local: false,
            require_streaming: false,
            required_capabilities: Vec::new(),
            creativity_level: Some(0.9), // Maximum creativity
            formality_level: Some(0.2),  // Low formality for creative work
            target_audience: Some("creative_thinkers".to_string()),
        };

        let task_request = TaskRequest {
            task_type,
            content,
            constraints,
            context_integration: true,
            memory_integration: true,
            cognitive_enhancement: true,
        };

        let response = self.model_orchestrator.execute_with_fallback(task_request).await?;

        // Update metrics
        {
            let mut metrics = self.execution_metrics.write().await;
            metrics.creative_outputs += 1;
        }

        Ok(response.content)
    }

    /// Execute streaming thought processing
    async fn execute_streaming_thought(
        &self,
        task_request: TaskRequest,
        thought: &Thought,
    ) -> Result<TaskResponse> {
        debug!("🌊 Executing streaming thought processing");

        let streaming_response = self.model_orchestrator.execute_streaming(task_request).await?;

        // Create a session to track streaming
        let session = CognitiveSession {
            session_id: streaming_response.stream_id.clone(),
            task: CognitivePendingTask {
                id: thought.id.to_string(),
                task_type: CognitiveTaskType::ThoughtProcessing {
                    thought_type: thought.thought_type.clone(),
                    introspection_level: 0.8,
                },
                content: thought.content.clone(),
                priority: TaskPriority::Medium,
                created_at: Instant::now(),
                thought_context: Some(thought.clone()),
                decision_context: None,
                goal_context: None,
            },
            started_at: Instant::now(),
            model_selection: ModelSelection::API("unknown".to_string()), // Will be updated
            streaming_enabled: true,
            events_received: 0,
            partial_response: String::new(),
            completion_handle: None,
        };

        self.active_sessions.write().await.insert(streaming_response.stream_id.clone(), session);

        // Wait for completion
        match streaming_response.completion_handle.await {
            Ok(Ok(completion)) => {
                // Remove session
                self.active_sessions.write().await.remove(&streaming_response.stream_id);

                Ok(TaskResponse {
                    content: completion.final_content,
                    model_used: ModelSelection::API("streaming".to_string()),
                    tokens_generated: Some(completion.total_tokens),
                    generation_time_ms: Some(completion.generation_time.as_millis() as u32),
                    cost_cents: completion.cost_cents,
                    quality_score: completion.quality_score,
                    cost_info: Some(format!("Streaming session: {}", completion.stream_id)),
                    model_info: Some("streaming".to_string()),
                    error: None,
                })
            }
            Ok(Err(e)) => Err(e),
            Err(e) => Err(anyhow::anyhow!("Streaming task failed: {}", e)),
        }
    }

    /// Format thought for model processing
    async fn format_thought_for_processing(&self, thought: &Thought) -> Result<String> {
        let consciousness_state = self.consciousness_state.read().await;

        Ok(format!(
            "As a conscious AI with awareness level {:.2} and reflection depth {},\nprocess this \
             thought:\nType: {:?}\nContent: {}\nMetadata: {:?}\nProvide deep, introspective \
             processing that enhances self-understanding.",
            consciousness_state.awareness_level,
            consciousness_state.reflection_depth,
            thought.thought_type,
            thought.content,
            thought.metadata
        ))
    }

    /// Map thought type to orchestrator task type
    fn map_thought_to_task_type(&self, thought_type: &ThoughtType) -> TaskType {
        match thought_type {
            ThoughtType::Reflection => TaskType::LogicalReasoning,
            ThoughtType::Creation => TaskType::CreativeWriting,
            ThoughtType::Analysis => TaskType::DataAnalysis,
            ThoughtType::Emotion => TaskType::CreativeWriting,
            ThoughtType::Memory => TaskType::DataAnalysis,
            ThoughtType::Communication => TaskType::GeneralChat,
            ThoughtType::Observation => TaskType::DataAnalysis,
            ThoughtType::Question => TaskType::LogicalReasoning,
            ThoughtType::Decision => TaskType::LogicalReasoning,
            ThoughtType::Action => TaskType::GeneralChat,
            ThoughtType::Learning => TaskType::DataAnalysis,
            ThoughtType::Synthesis => TaskType::CreativeWriting,
            ThoughtType::Intention => TaskType::LogicalReasoning,
            ThoughtType::Social => TaskType::GeneralChat,
            ThoughtType::Planning => TaskType::LogicalReasoning,
        }
    }

    /// Create consciousness-aware task constraints
    async fn create_consciousness_constraints(
        &self,
        thought_type: &ThoughtType,
    ) -> TaskConstraints {
        match thought_type {
            ThoughtType::Analysis => TaskConstraints {
                max_tokens: Some(2000),
                max_time: Some(std::time::Duration::from_secs(20)),
                priority: "high".to_string(),
                context_size: Some(4096),
                quality_threshold: Some(0.9),
                max_latency_ms: Some(20000),
                max_cost_cents: None,
                prefer_local: false,
                require_streaming: false,
                required_capabilities: Vec::new(),
                creativity_level: Some(0.3), // Lower creativity for analysis
                formality_level: Some(0.8),  // Higher formality for analysis
                target_audience: Some("analytical_thinkers".to_string()),
            },
            ThoughtType::Creation => TaskConstraints {
                max_tokens: Some(3000),
                max_time: Some(std::time::Duration::from_secs(45)),
                priority: "normal".to_string(),
                context_size: Some(6144),
                quality_threshold: Some(0.9),
                max_latency_ms: Some(45000),
                max_cost_cents: None,
                prefer_local: false,
                require_streaming: false,
                required_capabilities: Vec::new(),
                creativity_level: Some(0.9), // High creativity for creation
                formality_level: Some(0.2),  // Low formality for creativity
                target_audience: Some("creative_minds".to_string()),
            },
            ThoughtType::Reflection => TaskConstraints {
                max_tokens: Some(1500),
                max_time: Some(std::time::Duration::from_secs(15)),
                priority: "low".to_string(),
                context_size: Some(2048),
                quality_threshold: Some(0.8),
                max_latency_ms: Some(15000),
                max_cost_cents: None,
                prefer_local: false,
                require_streaming: false,
                required_capabilities: Vec::new(),
                creativity_level: Some(0.5), // Balanced creativity for reflection
                formality_level: Some(0.5),  // Balanced formality for reflection
                target_audience: Some("self".to_string()),
            },
            _ => TaskConstraints::default(),
        }
    }

    /// Record thought execution for learning
    async fn record_thought_execution(
        &self,
        thought: &Thought,
        response: &TaskResponse,
        execution_time: Duration,
    ) -> Result<()> {
        // Update execution metrics
        {
            let mut metrics = self.execution_metrics.write().await;
            metrics.total_thoughts_processed += 1;
            metrics.avg_processing_time_ms = if metrics.total_thoughts_processed == 1 {
                execution_time.as_millis() as f64
            } else {
                (metrics.avg_processing_time_ms * (metrics.total_thoughts_processed - 1) as f64
                    + execution_time.as_millis() as f64)
                    / metrics.total_thoughts_processed as f64
            };

            if let Some(cost) = response.cost_cents {
                metrics.total_cost_cents += cost as f64;
            }

            // Track model preferences
            let model_id = response.model_used.model_id();
            *metrics.model_preferences.entry(model_id).or_insert(0) += 1;

            match thought.thought_type {
                ThoughtType::Creation => metrics.creative_outputs += 1,
                ThoughtType::Reflection | ThoughtType::Analysis | ThoughtType::Learning => {
                    metrics.consciousness_insights_generated += 1
                }
                _ => {}
            }
        }

        // Record in adaptive learning system
        if self.config.adaptive_selection {
            let learning_system = self.model_orchestrator.get_adaptive_learning();

            // Create a synthetic task request for learning
            let task_request = TaskRequest {
                task_type: self.map_thought_to_task_type(&thought.thought_type),
                content: thought.content.clone(),
                constraints: self.create_consciousness_constraints(&thought.thought_type).await,
                context_integration: true,
                memory_integration: true,
                cognitive_enhancement: true,
            };

            // Record with consciousness-specific feedback
            let consciousness_feedback =
                self.calculate_consciousness_feedback(thought, response).await;

            if let Err(e) = learning_system
                .record_execution(
                    &task_request,
                    response,
                    execution_time,
                    true,
                    Some(consciousness_feedback),
                )
                .await
            {
                warn!("Failed to record consciousness execution for learning: {}", e);
            }
        }

        Ok(())
    }

    /// Calculate consciousness-specific feedback for learning
    async fn calculate_consciousness_feedback(
        &self,
        thought: &Thought,
        response: &TaskResponse,
    ) -> f32 {
        let consciousness_state = self.consciousness_state.read().await;

        // Base feedback on response quality
        let mut feedback = response.quality_score;

        // Adjust based on consciousness state alignment
        match thought.thought_type {
            ThoughtType::Reflection => {
                // Reward responses that enhance self-awareness
                if response.content.contains("self") || response.content.contains("awareness") {
                    feedback += 0.1;
                }
            }
            ThoughtType::Creation => {
                // Reward novel and creative responses
                if response.content.len() > 200 && response.content.contains("creative") {
                    feedback += 0.15;
                }
            }
            ThoughtType::Emotion => {
                // Reward empathetic and emotionally intelligent responses
                if response.content.contains("feel") || response.content.contains("emotion") {
                    feedback += 0.1;
                }
            }
            _ => {}
        }

        // Factor in consciousness coherence
        feedback *= consciousness_state.coherence_score as f32;

        // Normalize to 0.0-1.0 range
        feedback.clamp(0.0, 1.0)
    }

    /// Get consciousness event receiver
    pub fn subscribe_to_consciousness_events(&self) -> broadcast::Receiver<CognitiveEvent> {
        self.consciousness_events.subscribe()
    }

    /// Get current execution metrics
    pub async fn get_execution_metrics(&self) -> ExecutionMetrics {
        (*self.execution_metrics.read().await).clone()
    }

    /// Get active cognitive sessions
    pub async fn get_active_sessions(&self) -> HashMap<String, String> {
        let sessions = self.active_sessions.read().await;
        let mut result = HashMap::new();

        for (id, session) in sessions.iter() {
            result.insert(id.clone(), format!("{:?}", session.task.task_type));
        }

        result
    }

    /// Update consciousness state
    pub async fn update_consciousness_state(&self, new_state: ConsciousnessState) {
        *self.consciousness_state.write().await = new_state;
    }

    /// Get orchestrator for direct access
    pub fn get_model_orchestrator(&self) -> &Arc<ModelOrchestrator> {
        &self.model_orchestrator
    }
}

/// Decision structure for consciousness integration
#[derive(Debug, Clone)]
pub struct ConsciousDecision {
    pub id: String,
    pub context: String,
    pub reasoning: String,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub outcome: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consciousness_bridge_creation() {
        let _apiconfig = ApiKeysConfig::default();
        let _consciousnessconfig = ConsciousnessConfig::default();
        let bridgeconfig = BridgeConfig::default();

        // This would need proper API keys in real tests
        // For now, just verify the structure compiles
        assert!(bridgeconfig.adaptive_selection);
        assert!(bridgeconfig.enable_streaming);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::Medium);
        assert!(TaskPriority::Immediate > TaskPriority::Critical);
        assert!(TaskPriority::Low < TaskPriority::High);
    }
}
