//! Orchestration subsystem module
//! 
//! Manages multi-model orchestration, routing strategies, and ensemble voting

pub mod manager;
pub mod connector;
pub mod routing;
pub mod ensemble;
pub mod models;
pub mod model_validator;
pub mod model_persistence;
pub mod model_selector;
pub mod usage_stats;
pub mod advanced;
pub mod pipeline;
pub mod collaborative;
pub mod unified;
pub mod tracking;
pub mod todo_manager;
pub mod context_aware_execution;
pub mod smart_context_switching;
pub mod execution_feedback_loop;

// Re-export commonly used types
pub use manager::{OrchestrationManager, RoutingStrategy, OrchestrationSetup, ModelCapability};
pub use connector::{OrchestrationConnector, BackendStatus, PerformanceInfo, ModelInfo};
pub use routing::{RoutingConfiguration, ModelRouter, RoutingRequest, RoutingDecision};
pub use ensemble::{EnsembleConfig, VotingStrategy, EnsembleCoordinator, ModelResponse, EnsembleResult};
pub use models::{ModelRegistry, RegisteredModel, ModelConfig, RateLimit, ModelStats};
pub use model_validator::ModelValidator;
pub use model_selector::{ModelSelector, OrchestrationStatus};
pub use usage_stats::{UsageStatsTracker, UsageStats, RequestRecord, UsageSummary};
pub use advanced::{AdvancedOrchestrator, OrchestrationConfig, OrchestrationResult, TaskType, Priority};
pub use pipeline::{PipelineOrchestrator, Pipeline, Stage, ExecutionContext};
pub use collaborative::{
    CollaborativeOrchestrator, CollaborationConfig, CollaborativeTask, CollaborationResult,
    ParticipantRole, Capability, Priority as CollabPriority, CollaborativeTaskType,
};
pub use unified::{
    UnifiedOrchestrator, UnifiedConfig, OrchestrationRequest, UnifiedResponse,
    RequestType, OrchestratorType, RequestConstraint,
};
pub use tracking::{
    ModelCallTracker, TrackingSession, CallRecord, ModelStatistics,
    ModelUsageStats, HourlyStats, CostCalculator, ModelPricing,
};
pub use todo_manager::{
    TodoManager, TodoItem, TodoStatus, TodoPriority, CreateTodoRequest,
    TodoExecution, ExecutionStatus, TodoConfig,
};
pub use context_aware_execution::{
    ContextAwareExecutor, ExecutionContext as ContextAwareExecutionContext, ExecutionResult, 
    AdaptationStrategy, OptimizationGoal,
};
pub use smart_context_switching::{
    SmartContextSwitcher, ActiveContext, ContextType, TransitionTrigger,
    ContextSwitchEvent, ResourceAllocation,
};
pub use execution_feedback_loop::{
    ExecutionFeedbackLoop, FeedbackItem, FeedbackType, Lesson,
    Recommendation, FeedbackEvent, ImpactLevel,
};