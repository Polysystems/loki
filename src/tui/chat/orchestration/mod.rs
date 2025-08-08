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
pub mod usage_stats;
pub mod advanced;
pub mod pipeline;
pub mod collaborative;
pub mod unified;

// Re-export commonly used types
pub use manager::{OrchestrationManager, RoutingStrategy, OrchestrationSetup, ModelCapability};
pub use connector::{OrchestrationConnector, BackendStatus, PerformanceInfo, ModelInfo};
pub use routing::{RoutingConfiguration, ModelRouter, RoutingRequest, RoutingDecision};
pub use ensemble::{EnsembleConfig, VotingStrategy, EnsembleCoordinator, ModelResponse, EnsembleResult};
pub use models::{ModelRegistry, RegisteredModel, ModelConfig, RateLimit, ModelStats};
pub use model_validator::ModelValidator;
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