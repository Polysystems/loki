//! Agent management module
//! 
//! Handles multi-agent coordination, collaboration modes, and agent specializations

pub mod manager;
pub mod collaboration;
pub mod specialization;
pub mod coordination;
pub mod creation;
pub mod registry;
pub mod bridge;
pub mod templates;
pub mod templates_impl;
pub mod bindings;
pub mod code_agent;
pub mod story_integration;

// Re-export commonly used types
pub use manager::AgentManager;
pub use collaboration::{CollaborationCoordinator, CollaborationConfig, CollaborativeTask, CollaborationResult};
pub use specialization::{SpecializationRegistry, SpecializationProfile, TaskType};
pub use coordination::{AgentCoordinator, CoordinationConfig, CoordinationTask, AgentStats};
pub use creation::{AgentCreationWizard, AgentConfig, AgentTemplate, PersonalityProfile};
pub use registry::{AgentRegistry, AgentEntry, RuntimeState, AgentStatus};
pub use bridge::AgentBridge;
pub use templates::{TemplateLibrary, TemplateBuilder};
pub use bindings::{BindingManager, ModelBinding, ToolBinding, BindingOptimizer};
pub use code_agent::{CodeAgent, CodeAgentFactory, CodeSpecialization, CodeTask};
pub use story_integration::{
    StoryAgentOrchestrator, NarrativeInfluence, BehaviorModifier,
    StoryCollaboration, NarrativeRole, CollaborationStatus, StoryAgentEvent
};

// Re-export AgentSpecialization from cognitive module
pub use crate::cognitive::agents::AgentSpecialization;