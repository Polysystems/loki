//! Story Engine - Intelligent Context Chaining System
//!
//! The story engine provides automatic context management and synchronization
//! across agents, codebases, and system components through narrative structures.

pub mod engine;
pub mod types;
pub mod context_chain;
pub mod story_sync;
pub mod codebase_story;
pub mod agent_story;
pub mod task_mapper;
pub mod file_watcher;
pub mod context_retrieval;
pub mod visualization;
pub mod agent_coordination;
pub mod templates;
pub mod export_import;
pub mod learning;

#[cfg(test)]
mod tests;

pub use engine::{StoryEngine, StoryEvent};
pub use types::*;
pub use context_chain::ContextChain;
pub use story_sync::StorySynchronizer;
pub use codebase_story::CodebaseStory;
pub use agent_story::AgentStory;
pub use task_mapper::TaskMapper;
pub use file_watcher::{StoryFileWatcher, FileWatcherConfig};
pub use context_retrieval::{StoryContextRetriever, ContextRetrievalConfig, RetrievedContext};
pub use visualization::{StoryAnalytics, StoryStatistics, StoryTimeline, StoryGraph, StoryInsights};
pub use agent_coordination::{
    StoryAgentCoordinator, CoordinationConfig, CoordinationSessionId,
    AgentCapability, AgentStatus, CoordinationEvent,
};
pub use templates::{
    StoryTemplate, StoryTemplateManager, TemplateId, TemplateCategory,
    PlotTemplate, CompletionCondition, TemplateMetadata,
    TemplateInstanceTracker, TemplateInstance,
};
pub use export_import::{
    StoryPorter, StoryExport, ExportedStory, ExportOptions, ExportFormat,
    MergeStrategy, ImportResult, ImportConflict, StoryArchiver, ArchiveInfo,
    ExportMetadata, StoryRelationship,
};
pub use learning::{
    StoryLearningSystem, LearnedPattern, PatternType, PatternOccurrence,
    PatternOutcome, AdaptationStrategy, Adaptation, LearningConfig,
    AdaptationSuggestion, PatternStatistics,
};

use crate::cognitive::context_manager::ContextManager;
use crate::memory::CognitiveMemory;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Story system configuration
#[derive(Debug, Clone)]
pub struct StoryConfig {
    /// Maximum story depth for context chains
    pub max_story_depth: usize,
    /// Auto-sync interval in seconds
    pub sync_interval: u64,
    /// Enable automatic story generation
    pub auto_generate: bool,
    /// Context window size for story segments
    pub context_window: usize,
    /// Enable intelligent todo mapping
    pub enable_task_mapping: bool,
    /// Enable persistence between sessions
    pub enable_persistence: bool,
    /// Path for story persistence
    pub persistence_path: std::path::PathBuf,
    /// Auto-save interval for persistence in seconds
    pub auto_save_interval: u64,
}

impl Default for StoryConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("loki")
            .join("stories");

        Self {
            max_story_depth: 10,
            sync_interval: 30,
            auto_generate: true,
            context_window: 8192,
            enable_task_mapping: true,
            enable_persistence: true,
            persistence_path: data_dir.join("stories.json"),
            auto_save_interval: 300, // 5 minutes
        }
    }
}

/// Initialize the story subsystem
pub async fn init_story_system(
    context_manager: Arc<RwLock<ContextManager>>,
    memory: Arc<CognitiveMemory>,
    config: StoryConfig,
) -> anyhow::Result<Arc<StoryEngine>> {
    let persistence_path = config.persistence_path.clone();
    let auto_save_interval = config.auto_save_interval;
    let enable_persistence = config.enable_persistence;

    let engine = Arc::new(StoryEngine::new(context_manager, memory, config).await?);

    // Initialize template manager with engine reference
    engine.clone().init_template_manager().await;

    // Load existing stories if persistence is enabled
    if enable_persistence {
        if let Err(e) = engine.load_from_persistence(&persistence_path).await {
            tracing::warn!("Failed to load stories from persistence: {}", e);
        }

        // Start auto-persistence
        engine.clone().start_auto_persistence(
            persistence_path,
            std::time::Duration::from_secs(auto_save_interval),
        );
    }

    Ok(engine)
}
