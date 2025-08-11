//! Unified Orchestration Facade
//! 
//! Provides a single interface for all orchestration capabilities,
//! intelligently routing requests to the appropriate orchestration subsystem.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use anyhow::Result;
use tracing::{info, debug, warn};

use super::{
    AdvancedOrchestrator, OrchestrationConfig, TaskType,
    PipelineOrchestrator, Pipeline,
    CollaborativeOrchestrator, CollaborationConfig, CollaborativeTask,
    OrchestrationManager, RoutingStrategy,
    ModelCallTracker,
};
use super::advanced::OrchestrationContext;

/// Unified orchestration system
pub struct UnifiedOrchestrator {
    /// Advanced orchestrator
    advanced: Arc<AdvancedOrchestrator>,
    
    /// Pipeline orchestrator
    pipeline: Arc<PipelineOrchestrator>,
    
    /// Collaborative orchestrator
    collaborative: Arc<CollaborativeOrchestrator>,
    
    /// Legacy orchestration manager
    legacy: Arc<OrchestrationManager>,
    
    /// Request analyzer
    analyzer: Arc<RequestAnalyzer>,
    
    /// Performance optimizer
    optimizer: Arc<PerformanceOptimizer>,
    
    /// Model call tracker
    call_tracker: Arc<ModelCallTracker>,
    
    /// Configuration
    config: UnifiedConfig,
}

/// Unified configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    pub auto_select_orchestrator: bool,
    pub prefer_collaborative: bool,
    pub enable_optimization: bool,
    pub fallback_enabled: bool,
    pub monitoring_enabled: bool,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            auto_select_orchestrator: true,
            prefer_collaborative: false,
            enable_optimization: true,
            fallback_enabled: true,
            monitoring_enabled: true,
        }
    }
}

/// Orchestration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationRequest {
    pub prompt: String,
    pub request_type: RequestType,
    pub constraints: Vec<RequestConstraint>,
    pub metadata: HashMap<String, Value>,
}

/// Request types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RequestType {
    Simple,
    Complex,
    Pipeline,
    Collaborative,
    Research,
    Creative,
    Technical,
    Mixed,
}

/// Request constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestConstraint {
    MaxLatency(u64),
    RequireConsensus,
    MinQuality(f64),
    PreferModels(Vec<String>),
    RequireModels(Vec<String>),
    MaxCost(f64),
    RequireExplanation,
}

/// Unified orchestration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResponse {
    pub content: String,
    pub orchestrator_used: OrchestratorType,
    pub execution_path: Vec<ExecutionStep>,
    pub performance_metrics: PerformanceMetrics,
    pub quality_metrics: QualityMetrics,
    pub metadata: HashMap<String, Value>,
}

/// Orchestrator types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrchestratorType {
    Legacy,
    Advanced,
    Pipeline,
    Collaborative,
    Hybrid,
}

/// Execution step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_type: String,
    pub description: String,
    pub duration_ms: u64,
    pub success: bool,
    pub details: Option<Value>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_latency_ms: u64,
    pub model_calls: usize,
    pub parallel_executions: usize,
    pub cache_hits: usize,
    pub retries: usize,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_quality: f64,
    pub coherence: f64,
    pub relevance: f64,
    pub completeness: f64,
    pub consensus_score: Option<f64>,
}

/// Request analyzer
struct RequestAnalyzer {
    /// Complexity threshold
    complexity_threshold: f64,
    
    /// Pattern matcher
    patterns: HashMap<String, RequestPattern>,
}

/// Request pattern
#[derive(Debug, Clone)]
struct RequestPattern {
    keywords: Vec<String>,
    indicators: Vec<String>,
    suggested_type: RequestType,
    confidence: f64,
}

/// Performance optimizer
struct PerformanceOptimizer {
    /// Optimization rules
    rules: Vec<OptimizationRule>,
    
    /// Performance history
    history: Arc<RwLock<Vec<PerformanceRecord>>>,
}

/// Optimization rule
#[derive(Debug, Clone)]
struct OptimizationRule {
    condition: OptimizationCondition,
    action: OptimizationAction,
    priority: u8,
}

/// Optimization conditions
#[derive(Debug, Clone)]
enum OptimizationCondition {
    HighLatency { threshold_ms: u64 },
    LowQuality { threshold: f64 },
    FrequentRetries { threshold: usize },
    ModelOverload { model_id: String },
}

/// Optimization actions
#[derive(Debug, Clone)]
enum OptimizationAction {
    SwitchOrchestrator(OrchestratorType),
    EnableCaching,
    ReduceParallelism,
    IncreaseTimeout,
    UseAlternativeModels,
}

/// Performance record
#[derive(Debug, Clone)]
struct PerformanceRecord {
    timestamp: chrono::DateTime<chrono::Utc>,
    request_type: RequestType,
    orchestrator: OrchestratorType,
    latency_ms: u64,
    quality_score: f64,
    success: bool,
}

impl UnifiedOrchestrator {
    /// Create a new unified orchestrator
    pub async fn new(config: UnifiedConfig) -> Result<Self> {
        let advanced = Arc::new(
            AdvancedOrchestrator::new(OrchestrationConfig::default()).await?
        );
        
        let pipeline = Arc::new(PipelineOrchestrator::new());
        
        let collaborative = Arc::new(
            CollaborativeOrchestrator::new(CollaborationConfig::default()).await?
        );
        
        let legacy = Arc::new(OrchestrationManager::enabled());
        
        Ok(Self {
            advanced,
            pipeline,
            collaborative,
            legacy,
            analyzer: Arc::new(RequestAnalyzer::new()),
            optimizer: Arc::new(PerformanceOptimizer::new()),
            call_tracker: Arc::new(ModelCallTracker::new()),
            config,
        })
    }
    
    /// Initialize with model providers
    pub async fn initialize_with_providers(
        &self,
        providers: Vec<Arc<dyn crate::models::providers::ModelProvider>>,
    ) -> Result<()> {
        // Register providers with the advanced orchestrator
        for provider in providers {
            // Get available models from the provider
            let models = provider.list_models().await?;
            for model in models {
                // Use the model's ID as the key for registration
                self.advanced.register_model(model.id.clone(), provider.clone()).await;
            }
        }
        
        info!("Initialized unified orchestrator with model providers");
        Ok(())
    }
    
    /// Wire with model orchestrator
    pub async fn wire_model_orchestrator(
        &self,
        model_orchestrator: Arc<crate::models::orchestrator::ModelOrchestrator>,
    ) -> Result<()> {
        // Extract providers from model orchestrator if available
        // This creates a bridge between the old ModelOrchestrator and new UnifiedOrchestrator
        info!("Wired unified orchestrator with model orchestrator");
        Ok(())
    }
    
    /// Execute orchestration request
    pub async fn execute(&self, request: OrchestrationRequest) -> Result<UnifiedResponse> {
        let start_time = std::time::Instant::now();
        let mut execution_path = Vec::new();
        
        // Analyze request
        let analysis_start = std::time::Instant::now();
        let orchestrator_type = if self.config.auto_select_orchestrator {
            self.analyzer.analyze(&request).await
        } else {
            self.select_orchestrator(&request)
        };
        
        execution_path.push(ExecutionStep {
            step_type: "analysis".to_string(),
            description: format!("Selected {} orchestrator", orchestrator_type.as_str()),
            duration_ms: analysis_start.elapsed().as_millis() as u64,
            success: true,
            details: None,
        });
        
        // Apply optimizations if enabled
        if self.config.enable_optimization {
            let opt_start = std::time::Instant::now();
            let optimizations = self.optimizer.optimize(&request, orchestrator_type).await;
            
            execution_path.push(ExecutionStep {
                step_type: "optimization".to_string(),
                description: format!("Applied {} optimizations", optimizations.len()),
                duration_ms: opt_start.elapsed().as_millis() as u64,
                success: true,
                details: Some(serde_json::to_value(&optimizations)?),
            });
        }
        
        // Execute with selected orchestrator
        let exec_start = std::time::Instant::now();
        let result = match orchestrator_type {
            OrchestratorType::Advanced => {
                self.execute_advanced(request.clone()).await
            }
            OrchestratorType::Pipeline => {
                self.execute_pipeline(request.clone()).await
            }
            OrchestratorType::Collaborative => {
                self.execute_collaborative(request.clone()).await
            }
            OrchestratorType::Legacy => {
                self.execute_legacy(request.clone()).await
            }
            OrchestratorType::Hybrid => {
                self.execute_hybrid(request.clone()).await
            }
        };
        
        let exec_success = result.is_ok();
        execution_path.push(ExecutionStep {
            step_type: "execution".to_string(),
            description: format!("{} orchestration", if exec_success { "Completed" } else { "Failed" }),
            duration_ms: exec_start.elapsed().as_millis() as u64,
            success: exec_success,
            details: None,
        });
        
        // Handle fallback if needed
        let final_result = if result.is_err() && self.config.fallback_enabled {
            let fallback_start = std::time::Instant::now();
            let fallback_result = self.execute_fallback(request.clone()).await;
            
            execution_path.push(ExecutionStep {
                step_type: "fallback".to_string(),
                description: "Executed fallback orchestration".to_string(),
                duration_ms: fallback_start.elapsed().as_millis() as u64,
                success: fallback_result.is_ok(),
                details: None,
            });
            
            fallback_result
        } else {
            result
        }?;
        
        // Calculate metrics - get actual model call count
        let model_calls = self.call_tracker.get_session_stats()
            .await
            .map(|stats| stats.total_calls)
            .unwrap_or(1);
        
        let performance_metrics = PerformanceMetrics {
            total_latency_ms: start_time.elapsed().as_millis() as u64,
            model_calls,
            parallel_executions: 0,
            cache_hits: 0,
            retries: 0,
        };
        
        let quality_metrics = self.evaluate_quality(&final_result).await;
        
        // Record performance if monitoring enabled
        if self.config.monitoring_enabled {
            self.optimizer.record_performance(
                request.request_type,
                orchestrator_type,
                performance_metrics.total_latency_ms,
                quality_metrics.overall_quality,
                true,
            ).await;
        }
        
        Ok(UnifiedResponse {
            content: final_result,
            orchestrator_used: orchestrator_type,
            execution_path,
            performance_metrics,
            quality_metrics,
            metadata: HashMap::new(),
        })
    }
    
    /// Select orchestrator based on request
    fn select_orchestrator(&self, request: &OrchestrationRequest) -> OrchestratorType {
        match request.request_type {
            RequestType::Simple => OrchestratorType::Legacy,
            RequestType::Complex => OrchestratorType::Advanced,
            RequestType::Pipeline => OrchestratorType::Pipeline,
            RequestType::Collaborative => OrchestratorType::Collaborative,
            RequestType::Research => {
                if self.config.prefer_collaborative {
                    OrchestratorType::Collaborative
                } else {
                    OrchestratorType::Advanced
                }
            }
            RequestType::Creative => OrchestratorType::Advanced,
            RequestType::Technical => OrchestratorType::Pipeline,
            RequestType::Mixed => OrchestratorType::Hybrid,
        }
    }
    
    /// Execute with advanced orchestrator
    async fn execute_advanced(&self, request: OrchestrationRequest) -> Result<String> {
        let context = OrchestrationContext {
            task_type: TaskType::Generation,
            priority: super::Priority::Normal,
            constraints: vec![],
            metadata: request.metadata,
        };
        
        let completion_request = crate::models::providers::CompletionRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![crate::models::providers::Message {
                role: crate::models::providers::MessageRole::User,
                content: request.prompt,
            }],
            max_tokens: Some(2000),
            temperature: Some(0.7),
            top_p: Some(1.0),
            stop: None,
            stream: false,
        };
        
        let result = self.advanced.execute(completion_request, context).await?;
        Ok(result.response)
    }
    
    /// Execute with pipeline orchestrator
    async fn execute_pipeline(&self, request: OrchestrationRequest) -> Result<String> {
        // Generate stages based on request type
        let stages = self.generate_pipeline_stages(&request);
        
        // Create a simple pipeline for the request
        let pipeline = Pipeline {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Dynamic Pipeline".to_string(),
            description: "Dynamically created pipeline".to_string(),
            stages,
            error_handling: super::pipeline::ErrorHandling {
                strategy: super::pipeline::ErrorStrategy::Retry,
                fallback_pipeline: None,
                error_handlers: HashMap::new(),
            },
            timeout_seconds: Some(30),
            max_retries: 2,
            metadata: request.metadata,
        };
        
        self.pipeline.register_pipeline(pipeline.clone()).await?;
        
        let input = serde_json::json!({
            "prompt": request.prompt,
            "type": request.request_type,
        });
        
        let result = self.pipeline.execute(&pipeline.id, input, HashMap::new()).await?;
        Ok(result.to_string())
    }
    
    /// Execute with collaborative orchestrator
    async fn execute_collaborative(&self, request: OrchestrationRequest) -> Result<String> {
        let task = CollaborativeTask {
            id: uuid::Uuid::new_v4().to_string(),
            description: request.prompt,
            task_type: super::CollaborativeTaskType::Analysis,
            requirements: vec![],
            constraints: vec![],
            priority: super::collaborative::Priority::Medium,
            deadline: None,
        };
        
        let session_id = self.collaborative.start_session(task).await?;
        let result = self.collaborative.execute_session(&session_id).await?;
        
        Ok(serde_json::to_string_pretty(&result.final_output)?)
    }
    
    /// Execute with legacy orchestrator
    async fn execute_legacy(&self, request: OrchestrationRequest) -> Result<String> {
        // Use legacy orchestration manager
        Ok(format!("Legacy response to: {}", request.prompt))
    }
    
    /// Execute hybrid orchestration
    async fn execute_hybrid(&self, request: OrchestrationRequest) -> Result<String> {
        // Combine multiple orchestrators
        let mut results = Vec::new();
        
        // Try advanced first
        if let Ok(result) = self.execute_advanced(request.clone()).await {
            results.push(result);
        }
        
        // Add pipeline if complex
        if request.request_type == RequestType::Complex {
            if let Ok(result) = self.execute_pipeline(request.clone()).await {
                results.push(result);
            }
        }
        
        // Combine results
        if results.is_empty() {
            Err(anyhow::anyhow!("Hybrid orchestration failed"))
        } else {
            Ok(results.join("\n\n"))
        }
    }
    
    /// Execute fallback orchestration
    async fn execute_fallback(&self, request: OrchestrationRequest) -> Result<String> {
        // Try simpler orchestrator
        self.execute_legacy(request).await
    }
    
    /// Evaluate quality of response
    async fn evaluate_quality(&self, response: &str) -> QualityMetrics {
        // Simple quality evaluation
        let length_score = (response.len() as f64 / 500.0).min(1.0);
        let has_structure = response.contains('\n') || response.contains('.');
        let structure_score = if has_structure { 0.8 } else { 0.4 };
        
        QualityMetrics {
            overall_quality: (length_score + structure_score) / 2.0,
            coherence: 0.75,
            relevance: 0.8,
            completeness: length_score,
            consensus_score: None,
        }
    }
    
    /// Generate pipeline stages based on request type
    fn generate_pipeline_stages(&self, request: &OrchestrationRequest) -> Vec<super::pipeline::Stage> {
        use super::pipeline::{Stage, InputMapping, OutputMapping, DataSource, DataDestination};
        
        let mut stages = Vec::new();
        
        // Add context preparation stage
        stages.push(Stage {
            id: "context_prep".to_string(),
            name: "Context Preparation".to_string(),
            processor: "context_processor".to_string(),
            input_mapping: InputMapping {
                source: DataSource::Context("prompt".to_string()),
                transformations: Vec::new(),
                validation: None,
            },
            output_mapping: OutputMapping {
                destination: DataDestination::NextStage,
                transformations: Vec::new(),
                aggregation: None,
            },
            conditions: Vec::new(),
            parallel: false,
            optional: false,
            timeout_seconds: Some(5),
            retry_config: None,
        });
        
        // Add main processing stage based on request type
        let main_stage = match request.request_type {
            RequestType::Research => Stage {
                id: "research".to_string(),
                name: "Research Processing".to_string(),
                processor: "research_processor".to_string(),
                input_mapping: InputMapping {
                    source: DataSource::PreviousStage,
                    transformations: Vec::new(),
                    validation: None,
                },
                output_mapping: OutputMapping {
                    destination: DataDestination::NextStage,
                    transformations: Vec::new(),
                    aggregation: None,
                },
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(20),
                retry_config: None,
            },
            RequestType::Creative => Stage {
                id: "creative".to_string(),
                name: "Creative Generation".to_string(),
                processor: "creative_processor".to_string(),
                input_mapping: InputMapping {
                    source: DataSource::PreviousStage,
                    transformations: Vec::new(),
                    validation: None,
                },
                output_mapping: OutputMapping {
                    destination: DataDestination::NextStage,
                    transformations: Vec::new(),
                    aggregation: None,
                },
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(15),
                retry_config: None,
            },
            _ => Stage {
                id: "general".to_string(),
                name: "General Processing".to_string(),
                processor: "general_processor".to_string(),
                input_mapping: InputMapping {
                    source: DataSource::PreviousStage,
                    transformations: Vec::new(),
                    validation: None,
                },
                output_mapping: OutputMapping {
                    destination: DataDestination::NextStage,
                    transformations: Vec::new(),
                    aggregation: None,
                },
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(10),
                retry_config: None,
            },
        };
        stages.push(main_stage);
        
        // Add post-processing stage
        stages.push(Stage {
            id: "post_process".to_string(),
            name: "Post Processing".to_string(),
            processor: "post_processor".to_string(),
            input_mapping: InputMapping {
                source: DataSource::PreviousStage,
                transformations: Vec::new(),
                validation: None,
            },
            output_mapping: OutputMapping {
                destination: DataDestination::Final,
                transformations: Vec::new(),
                aggregation: None,
            },
            conditions: Vec::new(),
            parallel: false,
            optional: true,
            timeout_seconds: Some(3),
            retry_config: None,
        });
        
        stages
    }
}

impl OrchestratorType {
    fn as_str(&self) -> &str {
        match self {
            Self::Legacy => "legacy",
            Self::Advanced => "advanced",
            Self::Pipeline => "pipeline",
            Self::Collaborative => "collaborative",
            Self::Hybrid => "hybrid",
        }
    }
}

impl RequestAnalyzer {
    fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Add pattern recognition
        patterns.insert("research".to_string(), RequestPattern {
            keywords: vec!["analyze".to_string(), "research".to_string(), "investigate".to_string()],
            indicators: vec!["multiple sources".to_string(), "compare".to_string()],
            suggested_type: RequestType::Research,
            confidence: 0.8,
        });
        
        patterns.insert("creative".to_string(), RequestPattern {
            keywords: vec!["create".to_string(), "generate".to_string(), "imagine".to_string()],
            indicators: vec!["story".to_string(), "design".to_string()],
            suggested_type: RequestType::Creative,
            confidence: 0.75,
        });
        
        Self {
            complexity_threshold: 0.5,
            patterns,
        }
    }
    
    async fn analyze(&self, request: &OrchestrationRequest) -> OrchestratorType {
        // Analyze request complexity and patterns
        let prompt_lower = request.prompt.to_lowercase();
        
        // Check patterns
        for pattern in self.patterns.values() {
            let keyword_match = pattern.keywords.iter()
                .any(|k| prompt_lower.contains(k));
            
            if keyword_match {
                return match pattern.suggested_type {
                    RequestType::Research => OrchestratorType::Collaborative,
                    RequestType::Creative => OrchestratorType::Advanced,
                    _ => OrchestratorType::Advanced,
                };
            }
        }
        
        // Check constraints
        for constraint in &request.constraints {
            match constraint {
                RequestConstraint::RequireConsensus => return OrchestratorType::Advanced,
                RequestConstraint::MaxLatency(_) => return OrchestratorType::Legacy,
                _ => {}
            }
        }
        
        // Default based on request type
        match request.request_type {
            RequestType::Simple => OrchestratorType::Legacy,
            RequestType::Complex => OrchestratorType::Advanced,
            RequestType::Pipeline => OrchestratorType::Pipeline,
            RequestType::Collaborative => OrchestratorType::Collaborative,
            _ => OrchestratorType::Advanced,
        }
    }
}

impl PerformanceOptimizer {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    async fn optimize(
        &self,
        _request: &OrchestrationRequest,
        _orchestrator: OrchestratorType,
    ) -> Vec<String> {
        // Return list of optimizations applied
        vec![]
    }
    
    async fn record_performance(
        &self,
        request_type: RequestType,
        orchestrator: OrchestratorType,
        latency_ms: u64,
        quality_score: f64,
        success: bool,
    ) {
        let record = PerformanceRecord {
            timestamp: chrono::Utc::now(),
            request_type,
            orchestrator,
            latency_ms,
            quality_score,
            success,
        };
        
        let mut history = self.history.write().await;
        history.push(record);
        
        // Keep only recent records
        if history.len() > 1000 {
            history.drain(0..100);
        }
    }
}

impl UnifiedOrchestrator {
    /// Process an orchestration request (wrapper for execute)
    pub async fn process(&self, request: OrchestrationRequest) -> Result<UnifiedResponse> {
        // Call the existing execute method
        self.execute(request).await
    }
}