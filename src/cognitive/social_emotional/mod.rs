pub mod emotional_intelligence;
pub mod empathy_engine;
pub mod relationship_manager;
pub mod social_context_analyzer;
pub mod social_intelligence;

// Re-export key types with specific imports to avoid ambiguity
pub use emotional_intelligence::{EmotionalIntelligenceSystem, EmotionalState, StateChange};
pub use empathy_engine::{EmpathyEngine, EmpathyResponse};
pub use relationship_manager::{RelationshipManager, RelationshipTracker};
pub use social_context_analyzer::SocialContextAnalyzer;
pub use social_intelligence::{
    RecommendationCategory,
    RecommendationPriority,
    SocialBehaviorAnalyzer,
    SocialIntelligenceSystem,
};
