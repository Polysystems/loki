//! Core types for the story engine

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use uuid::Uuid;

/// Unique identifier for a story
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoryId(pub Uuid);

impl StoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for StoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a story arc
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoryArcId(pub Uuid);

impl fmt::Display for StoryArcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a plot point
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlotPointId(pub Uuid);

impl PlotPointId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for PlotPointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a context chain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainId(pub Uuid);

impl ChainId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for ChainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Types of stories in the system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StoryType {
    /// Story for a codebase or project
    Codebase {
        root_path: PathBuf,
        language: String,
    },
    /// Story for a directory
    Directory {
        path: PathBuf,
        parent_story: Option<StoryId>,
    },
    /// Story for a file
    File {
        path: PathBuf,
        file_type: String,
        parent_story: StoryId,
    },
    /// Story for an agent
    Agent {
        agent_id: String,
        agent_type: String,
    },
    /// Story for a task or workflow
    Task {
        task_id: String,
        parent_story: Option<StoryId>,
    },
    /// Story for system interaction
    System {
        component: String,
    },
    /// Story for a bug or issue
    Bug {
        issue_id: String,
        severity: String,
    },
    /// Story for a feature development
    Feature {
        feature_name: String,
        description: String,
    },
    /// Story for an epic or large initiative
    Epic {
        epic_name: String,
        objectives: Vec<String>,
    },
    /// Story for learning or discovery
    Learning {
        topic: String,
        learning_goals: Vec<String>,
    },
    /// Story for performance optimization
    Performance {
        component: String,
        metrics: Vec<String>,
    },
    /// Story for security concerns
    Security {
        security_domain: String,
        threat_level: String,
    },
    /// Story for documentation
    Documentation {
        doc_type: String,
        target_audience: String,
    },
    /// Story for testing activities
    Testing {
        test_type: String,
        coverage_areas: Vec<String>,
    },
    /// Story for refactoring work
    Refactoring {
        component: String,
        refactor_goals: Vec<String>,
    },
    /// Story for dependency management
    Dependencies {
        dependency_type: String,
        updates: Vec<String>,
    },
    /// Story for deployment activities
    Deployment {
        environment: String,
        deployment_type: String,
    },
    /// Story for research activities
    Research {
        research_topic: String,
        hypotheses: Vec<String>,
    },
}

/// A story represents the essential context for understanding a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Story {
    pub id: StoryId,
    pub story_type: StoryType,
    pub title: String,
    pub description: String,
    pub summary: String,
    pub status: StoryStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub arcs: Vec<StoryArc>,
    pub current_arc: Option<StoryArcId>,
    pub metadata: StoryMetadata,
    pub context_chain: ChainId,
    pub segments: Vec<StorySegment>,
    pub context: HashMap<String, serde_json::Value>,
}

impl Story {
    /// Calculate the completion percentage of the story
    pub fn calculate_completion_percentage(&self) -> f32 {
        if self.arcs.is_empty() {
            return 0.0;
        }
        
        let completed = self.arcs.iter()
            .filter(|arc| arc.status == ArcStatus::Completed)
            .count() as f32;
        let total = self.arcs.len() as f32;
        
        (completed / total) * 100.0
    }
    
    /// Create a new story with the given type
    pub fn new(story_type: StoryType) -> Self {
        let story_id = StoryId::new();
        let title = match &story_type {
            StoryType::Feature { feature_name, .. } => format!("Feature: {}", feature_name),
            StoryType::Bug { issue_id, .. } => format!("Bug: {}", issue_id),
            StoryType::Performance { component, .. } => format!("Performance: {}", component),
            StoryType::Documentation { doc_type, .. } => format!("Documentation: {}", doc_type),
            StoryType::Testing { test_type, .. } => format!("Testing: {}", test_type),
            StoryType::Task { task_id, .. } => format!("Task: {}", task_id),
            StoryType::Codebase { root_path, .. } => format!("Codebase: {}", root_path.display()),
            StoryType::Agent { agent_id, .. } => format!("Agent: {}", agent_id),
            StoryType::System { component } => format!("System: {}", component),
            StoryType::Epic { epic_name, .. } => format!("Epic: {}", epic_name),
            StoryType::Learning { topic, .. } => format!("Learning: {}", topic),
            StoryType::Security { security_domain, .. } => format!("Security: {}", security_domain),
            StoryType::Refactoring { component, .. } => format!("Refactoring: {}", component),
            StoryType::Dependencies { dependency_type, .. } => format!("Dependencies: {}", dependency_type),
            StoryType::Directory { path, .. } => format!("Directory: {}", path.display()),
            StoryType::File { path, .. } => format!("File: {}", path.display()),
            StoryType::Deployment { deployment_type, .. } => format!("Deployment: {}", deployment_type),
            StoryType::Research { research_topic, .. } => format!("Research: {}", research_topic),
        };
        
        Self {
            id: story_id,
            story_type,
            title,
            description: String::new(),
            summary: String::new(),
            status: StoryStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            arcs: Vec::new(),
            current_arc: None,
            metadata: StoryMetadata::default(),
            context_chain: ChainId::new(),
            segments: Vec::new(),
            context: HashMap::new(),
        }
    }
}

/// A story arc represents a major phase or chapter in the story
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryArc {
    pub id: StoryArcId,
    pub title: String,
    pub description: String,
    pub sequence_number: i32,
    pub plot_points: Vec<PlotPoint>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: ArcStatus,
}

/// Status of a story arc
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArcStatus {
    Planning,
    Active,
    Paused,
    Completed,
    Abandoned,
}

/// Status of a plot point
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlotPointStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for ArcStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArcStatus::Planning => write!(f, "Planning"),
            ArcStatus::Active => write!(f, "Active"),
            ArcStatus::Paused => write!(f, "Paused"),
            ArcStatus::Completed => write!(f, "Completed"),
            ArcStatus::Abandoned => write!(f, "Abandoned"),
        }
    }
}

/// Status of a story
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StoryStatus {
    NotStarted,
    Draft,
    Active,
    Completed,
    Archived,
}

impl fmt::Display for StoryStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoryStatus::NotStarted => write!(f, "NotStarted"),
            StoryStatus::Draft => write!(f, "Draft"),
            StoryStatus::Active => write!(f, "Active"),
            StoryStatus::Completed => write!(f, "Completed"),
            StoryStatus::Archived => write!(f, "Archived"),
        }
    }
}

/// A plot point represents a significant event or milestone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotPoint {
    pub id: PlotPointId,
    pub title: String,
    pub description: String,
    pub sequence_number: i32,
    pub timestamp: DateTime<Utc>,
    pub plot_type: PlotType,
    pub status: PlotPointStatus,
    pub estimated_duration: Option<chrono::Duration>,
    pub actual_duration: Option<chrono::Duration>,
    pub context_tokens: Vec<String>,
    pub importance: f32,
    pub metadata: PlotMetadata,
    pub tags: Vec<String>,
    pub consequences: Vec<String>,
}

/// Types of plot points
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlotType {
    /// A goal or objective
    Goal { objective: String },
    /// A task or todo item
    Task {
        description: String,
        completed: bool,
    },
    /// A decision point
    Decision {
        question: String,
        choice: String,
    },
    /// A discovery or insight
    Discovery { insight: String },
    /// An error or issue
    Issue {
        error: String,
        resolved: bool,
    },
    /// A transformation or refactoring
    Transformation {
        before: String,
        after: String,
    },
    /// An interaction with another component
    Interaction {
        with: String,
        action: String,
    },
    /// Progress milestone
    Progress {
        milestone: String,
        percentage: f32,
    },
    /// Analysis or investigation
    Analysis {
        subject: String,
        findings: Vec<String>,
    },
    /// Action taken
    Action {
        action_type: String,
        parameters: Vec<String>,
        outcome: String,
    },
    /// Reasoning process
    Reasoning {
        premise: String,
        conclusion: String,
        confidence: f32,
    },
    /// Event occurrence
    Event {
        event_type: String,
        description: String,
        impact: f32,
    },
    /// Context information
    Context {
        context_type: String,
        data: String,
    },
}

/// Metadata associated with a story
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryMetadata {
    pub tags: Vec<String>,
    pub dependencies: Vec<StoryId>,
    pub related_stories: Vec<StoryId>,
    pub priority: Priority,
    pub complexity: f32,
    pub custom_data: HashMap<String, serde_json::Value>,
}

impl Default for StoryMetadata {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            dependencies: Vec::new(),
            related_stories: Vec::new(),
            priority: Priority::Medium,
            complexity: 0.5,
            custom_data: HashMap::new(),
        }
    }
}

/// Plot metadata for story generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotMetadata {
    pub genre: String,
    pub themes: Vec<String>,
    pub tone: String,
    pub style: String,
    pub perspective: String,
    pub pacing: String,
    pub constraints: Vec<String>,
    pub importance: f32,
    pub tags: Vec<String>,
    pub source: String,
    pub references: Vec<String>,
}

impl Default for PlotMetadata {
    fn default() -> Self {
        Self {
            genre: "technical".to_string(),
            themes: Vec::new(),
            tone: "informative".to_string(),
            style: "concise".to_string(),
            perspective: "third-person".to_string(),
            pacing: "steady".to_string(),
            constraints: Vec::new(),
            importance: 0.5,
            tags: Vec::new(),
            source: "system".to_string(),
            references: Vec::new(),
        }
    }
}

/// Priority levels for stories
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
    Background,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Priority::Critical => write!(f, "Critical"),
            Priority::High => write!(f, "High"),
            Priority::Medium => write!(f, "Medium"),
            Priority::Low => write!(f, "Low"),
            Priority::Background => write!(f, "Background"),
        }
    }
}

/// Context synchronization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncEvent {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub source_story: StoryId,
    pub target_stories: Vec<StoryId>,
    pub sync_type: SyncType,
    pub payload: SyncPayload,
}

/// Types of synchronization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SyncType {
    /// Full context synchronization
    Full,
    /// Incremental update
    Delta,
    /// Merge multiple contexts
    Merge,
    /// Broadcast to all related stories
    Broadcast,
}

/// Payload for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPayload {
    pub plot_points: Vec<PlotPoint>,
    pub context_updates: HashMap<String, String>,
    pub metadata_changes: HashMap<String, serde_json::Value>,
}

/// Task mapping for intelligent todo management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMap {
    pub story_id: StoryId,
    pub tasks: Vec<MappedTask>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub completion_percentage: f32,
}

/// A mapped task with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedTask {
    pub id: String,
    pub description: String,
    pub story_context: String,
    pub status: TaskStatus,
    pub assigned_to: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub plot_point: Option<PlotPointId>,
}

/// Task status
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Blocked,
    Completed,
    Cancelled,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "Pending"),
            TaskStatus::InProgress => write!(f, "In Progress"),
            TaskStatus::Blocked => write!(f, "Blocked"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// Context map for working directories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMap {
    pub root: PathBuf,
    pub stories: HashMap<PathBuf, StoryId>,
    pub relationships: Vec<ContextRelationship>,
    pub generated_at: DateTime<Utc>,
}

/// Relationship between contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRelationship {
    pub from: PathBuf,
    pub to: PathBuf,
    pub relationship_type: RelationType,
    pub strength: f32,
}

/// Types of relationships
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationType {
    Parent,
    Child,
    Dependency,
    Reference,
    Similar,
    Conflict,
}

/// Story segment for narrative continuity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorySegment {
    pub id: String,
    pub story_id: StoryId,
    pub content: String,
    pub context: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub segment_type: SegmentType,
    pub tags: Vec<String>,
}

/// Types of story segments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SegmentType {
    Introduction,
    Development,
    Climax,
    Resolution,
    Transition,
}

/// Story context for sharing story state across components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryContext {
    pub story_id: StoryId,
    pub narrative: String,
    pub recent_plot_points: Vec<PlotPoint>,
    pub active_arc: Option<StoryArc>,
    pub current_plot: String,
    pub current_arc: Option<StoryArcId>,
}

/// Task mapping for story-driven tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMapping {
    pub task_id: String,
    pub story_id: StoryId,
    pub plot_point_id: Option<PlotPointId>,
    pub description: String,
    pub context: HashMap<String, String>,
}

/// Narrative context for chat integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeContext {
    pub story_id: StoryId,
    pub current_plot: String,
    pub current_arc: Option<StoryArc>,
    pub current_chapter: String,
    pub active_plot_points: Vec<String>,
    pub story_themes: Vec<String>,
    pub character_state: HashMap<String, String>,
    pub recent_events: Vec<String>,
    pub narrative_tension: f64,
    
    // Additional fields expected by chat
    pub active_characters: Vec<String>,
    pub plot_threads: Vec<String>,
    pub tone: String,
}

impl From<NarrativeContext> for StoryContext {
    fn from(narrative: NarrativeContext) -> Self {
        StoryContext {
            story_id: narrative.story_id,
            narrative: narrative.current_plot.clone(),
            recent_plot_points: Vec::new(), // NarrativeContext doesn't have PlotPoints
            active_arc: narrative.current_arc.clone(),
            current_plot: narrative.current_plot,
            current_arc: narrative.current_arc.as_ref().map(|arc| arc.id),
        }
    }
}

impl From<StoryContext> for NarrativeContext {
    fn from(story: StoryContext) -> Self {
        NarrativeContext {
            story_id: story.story_id,
            current_plot: story.current_plot,
            current_arc: story.active_arc,
            current_chapter: String::new(),
            active_plot_points: story.recent_plot_points.iter()
                .map(|pp| pp.description.clone())
                .collect(),
            story_themes: Vec::new(),
            character_state: HashMap::new(),
            recent_events: Vec::new(),
            narrative_tension: 0.5,
            active_characters: Vec::new(),
            plot_threads: Vec::new(),
            tone: "neutral".to_string(),
        }
    }
}
