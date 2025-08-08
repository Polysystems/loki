pub mod artistic_creation;
pub mod creative_intelligence;
pub mod creativity_assessment;
pub mod cross_domain_synthesis;
pub mod idea_generation;
pub mod innovation_discovery;

// Re-export key types based on what's actually defined
pub use artistic_creation::ArtisticCreationEngine;
pub use creative_intelligence::{
    ArtisticCreation,
    ContentType,
    CreativeContent,
    CreativeIdea,
    CreativeIntelligenceSystem,
    CreativeMode,
    CreativeProject,
    CreativePrompt,
    CreativeResult,
    CreativeTechnique,
    Innovation,
    QualityAssessment,
    QualityIndicators,
    QualityRequirements,
    SynthesisResult,
};
pub use creativity_assessment::CreativityAssessmentSystem;
pub use cross_domain_synthesis::CrossDomainSynthesisEngine;
pub use idea_generation::IdeaGenerationEngine;
pub use innovation_discovery::InnovationDiscoverySystem;
