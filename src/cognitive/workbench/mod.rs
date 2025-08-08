//! Cognitive Workbench System
//!
//! Advanced cognitive tools for reasoning, creativity, research, and knowledge
//! work. Provides a comprehensive workbench of cognitive capabilities that
//! leverage the adaptive architecture for optimal performance.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::cognitive::adaptive::{AdaptiveCognitiveArchitecture, TaskType};
use crate::memory::CognitiveMemory;

pub mod creativity;
pub mod knowledge;
pub mod reasoning;
pub mod research;
pub mod visualization;

pub use creativity::{
    ConceptualBlend,
    ConceptualBlendingEngine,
    CreativeConstraintSolver,
    CreativeSolution,
    NarrativeArchitect,
    NarrativeStructure,
};
pub use knowledge::{
    ConceptComposition,
    ConceptualCompositionEngine,
    GeneratedInsight,
    InsightGenerationSystem,
    KnowledgeSynthesis,
    KnowledgeSynthesizer,
};
pub use reasoning::{
    AdvancedLogicEngine,
    AnalogicalReasoning,
    AnalogyEngine,
    CausalNetworkProcessor,
    CausalReasoning,
    LogicalReasoning,
};
pub use research::{
    EvidenceEvaluation,
    EvidenceEvaluationSystem,
    GeneratedHypothesis,
    HypothesisGenerator,
    ResearchOrchestrator,
    ResearchProject,
};
pub use visualization::{
    CognitiveVisualizationEngine,
    KnowledgeGraphVisualizer,
    MemoryArchitectureViewer,
    ReasoningProcessTracer,
    ThoughtStructureMapper,
};

/// Unique identifier for workbench sessions
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkbenchSessionId(String);

impl WorkbenchSessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_task(task_description: &str) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(task_description.as_bytes());
        let hash = format!("{:?}", hasher.finalize());
        Self(format!("session_{}", &hash[..8]))
    }
}

impl std::fmt::Display for WorkbenchSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Types of cognitive work
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum WorkbenchTaskType {
    /// Logical analysis and reasoning
    LogicalAnalysis,
    /// Creative problem solving
    CreativeProblemSolving,
    /// Research and investigation
    ResearchInvestigation,
    /// Knowledge synthesis
    KnowledgeSynthesis,
    /// Concept development
    ConceptDevelopment,
    /// Hypothesis generation
    HypothesisGeneration,
    /// Evidence evaluation
    EvidenceEvaluation,
    /// Narrative construction
    NarrativeConstruction,
    /// Multi-modal reasoning
    MultiModalReasoning,
    /// Complex analysis
    ComplexAnalysis,
}

/// Configuration for workbench operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkbenchConfig {
    /// Enable parallel processing across tools
    pub parallel_processing: bool,

    /// Maximum concurrent tool operations
    pub max_concurrent_tools: u32,

    /// Enable cross-tool information sharing
    pub cross_tool_sharing: bool,

    /// Visualization depth level
    pub visualization_depth: u32,

    /// Memory budget for workbench operations
    pub memory_budget_mb: u64,

    /// Timeout for individual tool operations
    pub tool_timeout_seconds: u64,
}

impl Default for WorkbenchConfig {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            max_concurrent_tools: 4,
            cross_tool_sharing: true,
            visualization_depth: 3,
            memory_budget_mb: 1024,
            tool_timeout_seconds: 300,
        }
    }
}

/// Optimization levels for workbench operations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OptimizationLevel {
    Speed,
    Balanced,
    Quality,
    Creative,
}

/// A cognitive workbench session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkbenchSession {
    /// Session identifier
    pub session_id: WorkbenchSessionId,

    /// Session description
    pub description: String,

    /// Type of cognitive work
    pub task_type: WorkbenchTaskType,

    /// Session configuration
    pub config: WorkbenchConfig,

    /// Active tools in this session
    pub active_tools: Vec<String>,

    /// Session state
    pub state: SessionState,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// Session results
    pub results: Vec<WorkbenchResult>,

    /// Tool interaction history
    pub interaction_history: Vec<ToolInteraction>,

    /// Performance metrics
    pub metrics: SessionMetrics,
}

/// State of a workbench session
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SessionState {
    Initializing,
    ConfiguringArchitecture,
    ToolsLoading,
    Active,
    Processing,
    Completed,
    Failed(String),
    Paused,
}

/// Result from workbench operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkbenchResult {
    /// Result identifier
    pub id: String,

    /// Type of result
    pub result_type: WorkbenchResultType,

    /// Content of the result
    pub content: serde_json::Value,

    /// Confidence score
    pub confidence: f64,

    /// Quality assessment
    pub quality: QualityAssessment,

    /// Tool that generated this result
    pub source_tool: String,

    /// Timestamp of result generation
    pub timestamp: DateTime<Utc>,

    /// Related results
    pub related_results: Vec<String>,
}

/// Types of workbench results
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum WorkbenchResultType {
    LogicalProof,
    AnalogicalMapping,
    CausalModel,
    ConceptualModel,
    Insight,
    CreativeSolution,
    ResearchFinding,
    Hypothesis,
    Evidence,
    Narrative,
    Visualization,
    Synthesis,
}

/// Quality assessment for results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,

    /// Novelty score
    pub novelty: f64,

    /// Coherence score
    pub coherence: f64,

    /// Depth score
    pub depth: f64,

    /// Practical value score
    pub practical_value: f64,

    /// Supporting evidence quality
    pub evidence_quality: f64,
}

impl Default for QualityAssessment {
    fn default() -> Self {
        Self {
            overall_score: 0.7,
            novelty: 0.6,
            coherence: 0.8,
            depth: 0.7,
            practical_value: 0.7,
            evidence_quality: 0.6,
        }
    }
}

/// Interaction between tools
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolInteraction {
    /// Interaction identifier
    pub id: String,

    /// Source tool
    pub source_tool: String,

    /// Target tool
    pub target_tool: String,

    /// Type of interaction
    pub interaction_type: InteractionType,

    /// Data exchanged
    pub data: serde_json::Value,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Success status
    pub success: bool,
}

/// Types of tool interactions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InteractionType {
    DataSharing,
    ResultChaining,
    Collaboration,
    Validation,
    Enhancement,
    Integration,
}

/// Session performance metrics
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SessionMetrics {
    /// Total processing time (milliseconds)
    pub total_processing_time: u64,

    /// Tool activation times
    pub tool_activation_times: HashMap<String, u64>,

    /// Results generated per tool
    pub results_per_tool: HashMap<String, u64>,

    /// Cross-tool interactions
    pub tool_interactions: u64,

    /// Memory usage peak (MB)
    pub peak_memory_usage: u64,

    /// Success rate
    pub success_rate: f64,

    /// Quality score average
    pub avg_quality_score: f64,
}

/// Main cognitive workbench system
pub struct CognitiveWorkbench {
    /// Advanced reasoning tools
    logical_reasoning_engine: Arc<AdvancedLogicEngine>,
    analogical_reasoning_system: Arc<AnalogyEngine>,
    causal_reasoning_network: Arc<CausalNetworkProcessor>,

    /// Knowledge manipulation tools
    concept_composer: Arc<ConceptualCompositionEngine>,
    knowledge_synthesizer: Arc<KnowledgeSynthesizer>,
    insight_generator: Arc<InsightGenerationSystem>,

    /// Creative tools
    creative_constraint_solver: Arc<CreativeConstraintSolver>,
    narrative_architect: Arc<NarrativeArchitect>,
    conceptual_blender: Arc<ConceptualBlendingEngine>,

    /// Research and analysis tools
    research_orchestrator: Arc<ResearchOrchestrator>,
    evidence_evaluator: Arc<EvidenceEvaluationSystem>,
    hypothesis_generator: Arc<HypothesisGenerator>,

    /// Visualization system
    visualization_engine: Arc<CognitiveVisualizationEngine>,

    /// Adaptive architecture integration
    adaptive_architecture: Arc<AdaptiveCognitiveArchitecture>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Active sessions
    active_sessions: Arc<RwLock<HashMap<WorkbenchSessionId, WorkbenchSession>>>,

    /// Global configuration
    config: WorkbenchConfig,

    /// Performance analytics
    analytics: Arc<RwLock<WorkbenchAnalytics>>,
}

/// Analytics for workbench performance
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct WorkbenchAnalytics {
    /// Total sessions conducted
    pub total_sessions: u64,

    /// Successful sessions
    pub successful_sessions: u64,

    /// Tool usage statistics
    pub tool_usage_stats: HashMap<String, ToolUsageStats>,

    /// Average session quality
    pub avg_session_quality: f64,

    /// Most effective tool combinations
    pub effective_combinations: Vec<ToolCombination>,

    /// Performance trends
    pub performance_trends: Vec<PerformanceTrend>,
}

/// Usage statistics for a specific tool
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ToolUsageStats {
    /// Times used
    pub usage_count: u64,

    /// Average processing time
    pub avg_processing_time: f64,

    /// Success rate
    pub success_rate: f64,

    /// Average quality score
    pub avg_quality: f64,

    /// Most common use cases
    pub common_use_cases: Vec<String>,
}

/// Effective combination of tools
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCombination {
    /// Tools in combination
    pub tools: Vec<String>,

    /// Effectiveness score
    pub effectiveness: f64,

    /// Common scenarios
    pub scenarios: Vec<String>,

    /// Usage frequency
    pub frequency: u64,
}

/// Performance trend data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Metric name
    pub metric: String,

    /// Trend direction
    pub direction: TrendDirection,

    /// Change magnitude
    pub magnitude: f64,

    /// Time period
    pub period: String,
}

/// Direction of performance trends
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

impl CognitiveWorkbench {
    /// Create a new cognitive workbench
    pub async fn new(
        adaptive_architecture: Arc<AdaptiveCognitiveArchitecture>,
        memory: Arc<CognitiveMemory>,
        config: WorkbenchConfig,
    ) -> Result<Self> {
        // Initialize reasoning tools
        let logical_reasoning_engine = Arc::new(AdvancedLogicEngine::new().await?);
        let analogical_reasoning_system = Arc::new(AnalogyEngine::new().await?);
        let causal_reasoning_network = Arc::new(CausalNetworkProcessor::new().await?);

        // Initialize knowledge tools
        let concept_composer = Arc::new(ConceptualCompositionEngine::new(memory.clone()).await?);
        let knowledge_synthesizer = Arc::new(KnowledgeSynthesizer::new(memory.clone()).await?);
        let insight_generator = Arc::new(InsightGenerationSystem::new(memory.clone()).await?);

        // Initialize creative tools
        let creative_constraint_solver = Arc::new(CreativeConstraintSolver::new().await?);
        let narrative_architect = Arc::new(NarrativeArchitect::new().await?);
        let conceptual_blender = Arc::new(ConceptualBlendingEngine::new().await?);

        // Initialize research tools
        let research_orchestrator = Arc::new(ResearchOrchestrator::new(memory.clone()).await?);
        let evidence_evaluator = Arc::new(EvidenceEvaluationSystem::new().await?);
        let hypothesis_generator = Arc::new(HypothesisGenerator::new(memory.clone()).await?);

        // Initialize visualization system
        let visualization_engine = Arc::new(CognitiveVisualizationEngine::new().await?);

        Ok(Self {
            logical_reasoning_engine,
            analogical_reasoning_system,
            causal_reasoning_network,
            concept_composer,
            knowledge_synthesizer,
            insight_generator,
            creative_constraint_solver,
            narrative_architect,
            conceptual_blender,
            research_orchestrator,
            evidence_evaluator,
            hypothesis_generator,
            visualization_engine,
            adaptive_architecture,
            memory,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            analytics: Arc::new(RwLock::new(WorkbenchAnalytics::default())),
        })
    }

    /// **Start a new workbench session** - Core capability for advanced
    /// cognitive work
    pub async fn start_session(
        &self,
        description: String,
        task_type: WorkbenchTaskType,
        config: Option<WorkbenchConfig>,
    ) -> Result<WorkbenchSessionId> {
        let session_id = WorkbenchSessionId::from_task(&description);
        let sessionconfig = config.unwrap_or_else(|| self.config.clone());

        // Configure adaptive architecture for this task type
        let arch_task_type = self.map_to_architecture_task_type(&task_type);
        let arch_task_id = self
            .adaptive_architecture
            .reconfigure_for_task(
                &description,
                arch_task_type,
                self.create_performance_requirements(&sessionconfig),
            )
            .await?;

        // Determine required tools for this task type
        let required_tools = self.determine_required_tools(&task_type);

        // Create session
        let session = WorkbenchSession {
            session_id: session_id.clone(),
            description,
            task_type,
            config: sessionconfig,
            active_tools: required_tools,
            state: SessionState::Initializing,
            start_time: Utc::now(),
            results: Vec::new(),
            interaction_history: Vec::new(),
            metrics: SessionMetrics::default(),
        };

        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id.clone(), session);
        }

        tracing::info!("Started workbench session: {} for task: {}", session_id, arch_task_id);

        Ok(session_id)
    }

    /// **Execute multi-modal reasoning** - Blend logical, analogical, and
    /// causal reasoning
    pub async fn execute_multimodal_reasoning(
        &self,
        session_id: &WorkbenchSessionId,
        input: &str,
        reasoning_modes: Vec<ReasoningMode>,
    ) -> Result<MultiModalReasoningResult> {
        // Update session state
        self.update_session_state(session_id, SessionState::Processing).await?;

        let mut results = Vec::new();
        let start_time = std::time::Instant::now();

        // Execute reasoning in parallel if enabled
        if self.config.parallel_processing {
            let mut tasks = Vec::new();

            for mode in reasoning_modes {
                match mode {
                    ReasoningMode::Logical => {
                        let engine = self.logical_reasoning_engine.clone();
                        let input = input.to_string();
                        tasks.push(tokio::spawn(async move { engine.reason(&input).await }));
                    }
                    ReasoningMode::Analogical => {
                        let engine = self.analogical_reasoning_system.clone();
                        let input = input.to_string();
                        tasks
                            .push(tokio::spawn(async move { engine.find_analogies(&input).await }));
                    }
                    ReasoningMode::Causal => {
                        let engine = self.causal_reasoning_network.clone();
                        let input = input.to_string();
                        tasks.push(tokio::spawn(
                            async move { engine.analyze_causality(&input).await },
                        ));
                    }
                }
            }

            // Collect results
            for task in tasks {
                if let Ok(result) = task.await {
                    if let Ok(reasoning_result) = result {
                        results.push(reasoning_result);
                    }
                }
            }
        } else {
            // Sequential execution
            for mode in reasoning_modes {
                let result = match mode {
                    ReasoningMode::Logical => self.logical_reasoning_engine.reason(input).await?,
                    ReasoningMode::Analogical => {
                        self.analogical_reasoning_system.find_analogies(input).await?
                    }
                    ReasoningMode::Causal => {
                        self.causal_reasoning_network.analyze_causality(input).await?
                    }
                };
                results.push(result);
            }
        }

        let processing_time = start_time.elapsed().as_millis() as u64;

        // Synthesize results
        let synthesized_result = self.synthesize_reasoning_results(results).await?;

        // Update session with results
        self.add_session_result(
            session_id,
            WorkbenchResult {
                id: Uuid::new_v4().to_string(),
                result_type: WorkbenchResultType::Synthesis,
                content: serde_json::to_value(&synthesized_result)?,
                confidence: synthesized_result.confidence,
                quality: synthesized_result.quality.clone(),
                source_tool: "multimodal_reasoning".to_string(),
                timestamp: Utc::now(),
                related_results: Vec::new(),
            },
        )
        .await?;

        // Update metrics
        self.update_session_metrics(session_id, processing_time).await?;

        Ok(synthesized_result)
    }

    /// **Generate creative solutions** - Use constraint-based creativity
    pub async fn generate_creative_solution(
        &self,
        session_id: &WorkbenchSessionId,
        problem_description: &str,
        constraints: Vec<CreativeConstraint>,
    ) -> Result<CreativeSolution> {
        let solution = self
            .creative_constraint_solver
            .solve_with_constraints(problem_description, constraints)
            .await?;

        // Add to session results
        self.add_session_result(
            session_id,
            WorkbenchResult {
                id: Uuid::new_v4().to_string(),
                result_type: WorkbenchResultType::CreativeSolution,
                content: serde_json::to_value(&solution)?,
                confidence: solution.confidence,
                quality: solution.quality.clone(),
                source_tool: "creative_constraint_solver".to_string(),
                timestamp: Utc::now(),
                related_results: Vec::new(),
            },
        )
        .await?;

        Ok(solution)
    }

    /// **Conduct research investigation** - Orchestrate research process
    pub async fn conduct_research(
        &self,
        session_id: &WorkbenchSessionId,
        research_question: &str,
        scope: ResearchScope,
    ) -> Result<ResearchProject> {
        let project = self
            .research_orchestrator
            .start_investigation(
                research_question,
                crate::cognitive::workbench::research::ResearchScope {
                    domains: scope.domains,
                    depth_level: scope.depth_level,
                    time_limit: scope.time_limit,
                    resource_limit: scope.resource_limit,
                    boundaries: vec!["relevant_to_question".to_string()],
                },
            )
            .await?;

        // Add to session results
        self.add_session_result(
            session_id,
            WorkbenchResult {
                id: Uuid::new_v4().to_string(),
                result_type: WorkbenchResultType::ResearchFinding,
                content: serde_json::to_value(&project)?,
                confidence: project.confidence,
                quality: project.quality.clone(),
                source_tool: "research_orchestrator".to_string(),
                timestamp: Utc::now(),
                related_results: Vec::new(),
            },
        )
        .await?;

        Ok(project)
    }

    /// Get session status and results
    pub async fn get_session_status(
        &self,
        session_id: &WorkbenchSessionId,
    ) -> Result<WorkbenchSession> {
        let sessions = self.active_sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))
    }

    /// Get workbench analytics
    pub async fn get_analytics(&self) -> Result<WorkbenchAnalytics> {
        Ok(self.analytics.read().await.clone())
    }

    // Private helper methods...
    async fn update_session_state(
        &self,
        session_id: &WorkbenchSessionId,
        state: SessionState,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.state = state;
        }
        Ok(())
    }

    async fn add_session_result(
        &self,
        session_id: &WorkbenchSessionId,
        result: WorkbenchResult,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.results.push(result);
        }
        Ok(())
    }

    async fn update_session_metrics(
        &self,
        session_id: &WorkbenchSessionId,
        processing_time: u64,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.metrics.total_processing_time += processing_time;
        }
        Ok(())
    }

    fn map_to_architecture_task_type(&self, task_type: &WorkbenchTaskType) -> TaskType {
        match task_type {
            WorkbenchTaskType::LogicalAnalysis => TaskType::Analysis,
            WorkbenchTaskType::CreativeProblemSolving => TaskType::CreativeTasks,
            WorkbenchTaskType::ResearchInvestigation => TaskType::Analysis,
            WorkbenchTaskType::KnowledgeSynthesis => TaskType::Synthesis,
            WorkbenchTaskType::ConceptDevelopment => TaskType::CreativeTasks,
            WorkbenchTaskType::HypothesisGeneration => TaskType::Learning,
            WorkbenchTaskType::EvidenceEvaluation => TaskType::Analysis,
            WorkbenchTaskType::NarrativeConstruction => TaskType::CreativeTasks,
            WorkbenchTaskType::MultiModalReasoning => TaskType::Analysis,
            WorkbenchTaskType::ComplexAnalysis => TaskType::Analysis,
        }
    }

    fn create_performance_requirements(
        &self,
        config: &WorkbenchConfig,
    ) -> crate::cognitive::adaptive::PerformanceRequirements {
        use crate::cognitive::adaptive::{
            FaultToleranceLevel,
            PerformanceRequirements,
            ResourceBudget,
        };

        PerformanceRequirements {
            max_latency: config.tool_timeout_seconds * 1000,
            min_accuracy: 0.8,
            max_resources: ResourceBudget {
                memory_mb: config.memory_budget_mb,
                cpu_percent: 70.0,
                energy_units: 100.0,
                network_bandwidth: 50.0,
            },
            min_throughput: 1.0,
            fault_tolerance: FaultToleranceLevel::Moderate,
        }
    }

    fn determine_required_tools(&self, task_type: &WorkbenchTaskType) -> Vec<String> {
        match task_type {
            WorkbenchTaskType::LogicalAnalysis => vec!["logical_reasoning_engine".to_string()],
            WorkbenchTaskType::CreativeProblemSolving => {
                vec!["creative_constraint_solver".to_string(), "conceptual_blender".to_string()]
            }
            WorkbenchTaskType::ResearchInvestigation => vec![
                "research_orchestrator".to_string(),
                "evidence_evaluator".to_string(),
                "hypothesis_generator".to_string(),
            ],
            WorkbenchTaskType::KnowledgeSynthesis => {
                vec!["knowledge_synthesizer".to_string(), "insight_generator".to_string()]
            }
            WorkbenchTaskType::MultiModalReasoning => vec![
                "logical_reasoning_engine".to_string(),
                "analogical_reasoning_system".to_string(),
                "causal_reasoning_network".to_string(),
            ],
            _ => vec!["logical_reasoning_engine".to_string()],
        }
    }

    async fn synthesize_reasoning_results(
        &self,
        _results: Vec<serde_json::Value>,
    ) -> Result<MultiModalReasoningResult> {
        // Simplified synthesis - would use advanced integration logic
        Ok(MultiModalReasoningResult {
            synthesis: "Integrated reasoning result".to_string(),
            confidence: 0.8,
            quality: QualityAssessment::default(),
            contributing_modes: vec!["logical".to_string(), "analogical".to_string()],
        })
    }
}

/// Modes of reasoning
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ReasoningMode {
    Logical,
    Analogical,
    Causal,
}

/// Result from multi-modal reasoning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiModalReasoningResult {
    pub synthesis: String,
    pub confidence: f64,
    pub quality: QualityAssessment,
    pub contributing_modes: Vec<String>,
}

/// Creative constraints for problem solving
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreativeConstraint {
    pub constraint_type: String,
    pub description: String,
    pub strictness: f64, // 0.0 to 1.0
}

/// Scope for research investigations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResearchScope {
    pub domains: Vec<String>,
    pub depth_level: u32,
    pub time_limit: Option<u64>,
    pub resource_limit: Option<u64>,
    pub boundaries: Vec<String>,
}
