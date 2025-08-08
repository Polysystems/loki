pub mod abstract_thinking;
pub mod advanced_reasoning;
pub mod analogical_reasoning;
pub mod causal_inference;
pub mod logical_processor;
pub mod multi_modal;
pub mod reasoning_persistence;

// Re-export key types based on what's actually defined
pub use abstract_thinking::{AbstractThinkingModule, AbstractionLevel, ConceptHierarchy};
pub use advanced_reasoning::{
    AdvancedReasoningEngine,
    IntegratedReasoningResult,
    ReasoningChain,
    ReasoningProblem,
    ReasoningResult,
    ReasoningRule,
    ReasoningStep,
    ReasoningType,
};
pub use analogical_reasoning::{AnalogicalReasoningSystem, AnalogyPattern};
pub use causal_inference::{CausalInferenceEngine, CausalRelationship};
pub use logical_processor::LogicalReasoningProcessor;
pub use multi_modal::MultiModalIntegrator;
pub use reasoning_persistence::{ReasoningPersistence, PersistenceConfig};
