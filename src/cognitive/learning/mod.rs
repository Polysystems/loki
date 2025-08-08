pub mod adaptive_learning;
pub mod experience_integration;
pub mod knowledge_evolution;
pub mod learning_architecture;
pub mod meta_learning;

// Re-export key types with correct imports based on what's actually defined
pub use adaptive_learning::{AdaptationType, AdaptiveLearningNetwork, NetworkMetrics};
pub use experience_integration::{
    Experience,
    ExperienceCategory,
    ExperienceIntegrator,
    ExperienceOutcome,
    RelationshipType,
};
pub use knowledge_evolution::{
    EvidenceType,
    EvolutionEvent,
    KnowledgeEntity,
    KnowledgeEvolutionEngine,
};
pub use learning_architecture::{
    AdaptiveLearningResult,
    EvolutionResult,
    IntegrationResult,
    LearningArchitecture,
    LearningData,
    LearningObjective,
    LearningResult,
    MetaInsight,
    MetaLearningResult,
    PerformanceAnalysis,
};
pub use meta_learning::{AlgorithmPerformance, LearningStrategy, MetaLearningSystem};
