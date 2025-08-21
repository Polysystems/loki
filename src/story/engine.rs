//! Core story engine implementation

use super::*;
use crate::cognitive::context_manager::{ContextManager, TokenMetadata};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use anyhow::{Context, Result};
use dashmap::DashMap;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;
use tokio::sync::{RwLock, broadcast, Mutex};
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// The main story engine that manages all stories in the system
#[derive(Debug)]
pub struct StoryEngine {
    /// All active stories indexed by ID
    pub stories: Arc<DashMap<StoryId, Story>>,
    /// Context chains for stories
    pub context_chains: Arc<DashMap<ChainId, ContextChain>>,
    /// Story synchronizer
    synchronizer: Arc<StorySynchronizer>,
    /// Task mapper
    task_mapper: Arc<TaskMapper>,
    /// Reference to context manager
    context_manager: Arc<RwLock<ContextManager>>,
    /// Reference to cognitive memory
    memory: Arc<CognitiveMemory>,
    /// Configuration
    config: StoryConfig,
    /// Broadcast channel for story events
    event_tx: broadcast::Sender<StoryEvent>,
    /// Shutdown signal
    shutdown: broadcast::Sender<()>,
    /// Template manager
    template_manager: Arc<Mutex<StoryTemplateManager>>,
    /// Template instance tracker
    template_tracker: Arc<Mutex<TemplateInstanceTracker>>,
    /// File watcher
    file_watcher: Option<Arc<Mutex<crate::story::file_watcher::StoryFileWatcher>>>,
}

/// Events emitted by the story engine
#[derive(Debug, Clone)]
pub enum StoryEvent {
    StoryCreated(StoryId),
    StoryUpdated(StoryId),
    ArcCompleted(StoryId, StoryArcId),
    PlotPointAdded(StoryId, PlotPointId),
    SyncCompleted(SyncEvent),
    TaskMapped(StoryId, String),
}

impl StoryEngine {
    /// Create a new story engine
    pub async fn new(
        context_manager: Arc<RwLock<ContextManager>>,
        memory: Arc<CognitiveMemory>,
        config: StoryConfig,
    ) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown, _) = broadcast::channel(1);

        let synchronizer = Arc::new(StorySynchronizer::new(event_tx.clone()));
        let task_mapper = Arc::new(TaskMapper::new());

        // Create engine without template manager first
        let engine = Self {
            stories: Arc::new(DashMap::new()),
            context_chains: Arc::new(DashMap::new()),
            synchronizer,
            task_mapper,
            context_manager,
            memory,
            config,
            event_tx,
            shutdown,
            template_manager: Arc::new(Mutex::new(StoryTemplateManager::new_empty())),
            template_tracker: Arc::new(Mutex::new(TemplateInstanceTracker::new())),
            file_watcher: None,
        };

        // Start background tasks if auto-generation is enabled
        if engine.config.auto_generate {
            engine.start_background_tasks().await?;
        }

        Ok(engine)
    }

    /// Start background tasks for automatic story management
    async fn start_background_tasks(&self) -> Result<()> {
        // Auto-sync task
        let sync_interval = self.config.sync_interval;
        let stories = self.stories.clone();
        let synchronizer = self.synchronizer.clone();
        let mut shutdown_rx = self.shutdown.subscribe();

        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(sync_interval));

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        // Perform automatic synchronization
                        for story_ref in stories.iter() {
                            let story = story_ref.value();
                            if let Err(e) = synchronizer.sync_story(&story).await {
                                warn!("Failed to sync story {}: {}", story.id.0, e);
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Story sync task shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Create a new story for a codebase
    pub async fn create_codebase_story(
        &self,
        root_path: PathBuf,
        language: String,
    ) -> Result<StoryId> {
        let story_id = StoryId::new();
        let chain_id = ChainId(Uuid::new_v4());

        // Create initial context chain
        let context_chain = ContextChain::new(chain_id, story_id);
        self.context_chains.insert(chain_id, context_chain);

        // Create the story
        let story = Story {
            id: story_id,
            story_type: StoryType::Codebase {
                root_path: root_path.clone(),
                language: language.clone(),
            },
            title: format!("Codebase: {}", root_path.display()),
            description: format!("Story for {} codebase at {}", language, root_path.display()),
            summary: format!("Story for {} codebase at {}", language, root_path.display()),
            status: StoryStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            arcs: vec![],
            current_arc: None,
            metadata: StoryMetadata {
                tags: vec![language.clone(), "codebase".to_string()],
                dependencies: vec![],
                related_stories: vec![],
                priority: Priority::High,
                complexity: 0.0,
                custom_data: HashMap::new(),
            },
            context_chain: chain_id,
            segments: vec![],
            context: HashMap::new(),
        };

        // Store in memory
        self.stories.insert(story_id, story.clone());

        // Store in cognitive memory
        self.memory
            .store_story_metadata(story_id, &story)
            .await
            .context("Failed to store story metadata")?;

        // Add to context manager
        let ctx_mgr = self.context_manager.write().await;
        ctx_mgr.add_narrative(
            format!("Created codebase story: {}", story.title),
            crate::cognitive::context_manager::TokenMetadata {
                source: "story_engine".to_string(),
                emotional_valence: 0.0,
                attention_weight: 0.8,
                associations: vec![],
                compressed: false,
            },
        ).await?;
        drop(ctx_mgr);

        // Emit event
        let _ = self.event_tx.send(StoryEvent::StoryCreated(story_id));

        info!("Created codebase story {} for {}", story_id.0, root_path.display());

        Ok(story_id)
    }

    /// Get or create a story for an agent
    pub async fn get_or_create_agent_story(
        &self,
        agent_id: String,
    ) -> Result<StoryId> {
        // Check if agent story already exists
        let existing_stories = self.get_stories_by_type(|st| {
            matches!(st, StoryType::Agent { agent_id: aid, .. } if aid == &agent_id)
        });

        if let Some(story) = existing_stories.first() {
            return Ok(story.id);
        }

        // Create new agent story
        self.create_agent_story(agent_id, "Agent".to_string()).await
    }

    /// Create a story for an agent
    pub async fn create_agent_story(
        &self,
        agent_id: String,
        agent_type: String,
    ) -> Result<StoryId> {
        let story_id = StoryId::new();
        let chain_id = ChainId(Uuid::new_v4());

        // Create context chain
        let context_chain = ContextChain::new(chain_id, story_id);
        self.context_chains.insert(chain_id, context_chain);

        let story = Story {
            id: story_id,
            story_type: StoryType::Agent {
                agent_id: agent_id.clone(),
                agent_type: agent_type.clone(),
            },
            title: format!("Agent {}: {}", agent_type, agent_id),
            description: format!("Story tracking {} agent's journey", agent_type),
            summary: format!("Story tracking {} agent's journey", agent_type),
            status: StoryStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            arcs: vec![],
            current_arc: None,
            metadata: StoryMetadata {
                tags: vec![agent_type.clone(), "agent".to_string()],
                dependencies: vec![],
                related_stories: vec![],
                priority: Priority::Medium,
                complexity: 0.0,
                custom_data: HashMap::new(),
            },
            context_chain: chain_id,
            segments: vec![],
            context: HashMap::new(),
        };

        self.stories.insert(story_id, story.clone());

        // Store in memory
        self.memory
            .store_story_metadata(story_id, &story)
            .await?;

        let _ = self.event_tx.send(StoryEvent::StoryCreated(story_id));

        Ok(story_id)
    }
    
    /// Create a generic story with custom type
    pub async fn create_story(
        &self,
        story_type: StoryType,
        title: String,
        summary: String,
        tags: Vec<String>,
        priority: Priority,
    ) -> Result<StoryId> {
        let story_id = StoryId::new();
        let chain_id = ChainId(Uuid::new_v4());

        // Create context chain
        let context_chain = ContextChain::new(chain_id, story_id);
        self.context_chains.insert(chain_id, context_chain);

        let story = Story {
            id: story_id,
            story_type,
            title: title.clone(),
            description: summary.clone(),
            summary,
            status: StoryStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            arcs: vec![],
            current_arc: None,
            metadata: StoryMetadata {
                tags,
                dependencies: vec![],
                related_stories: vec![],
                priority,
                complexity: 0.5,
                custom_data: HashMap::new(),
            },
            context_chain: chain_id,
            segments: vec![],
            context: HashMap::new(),
        };

        self.stories.insert(story_id, story.clone());

        // Store in memory
        self.memory
            .store_story_metadata(story_id, &story)
            .await?;

        let _ = self.event_tx.send(StoryEvent::StoryCreated(story_id));

        Ok(story_id)
    }

    /// Add a plot point to a story
    pub async fn add_plot_point(
        &self,
        story_id: StoryId,
        plot_type: PlotType,
        context_tokens: Vec<String>,
    ) -> Result<PlotPointId> {
        let mut story = self.stories
            .get_mut(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;

        let plot_id = PlotPointId(Uuid::new_v4());
        let importance = self.calculate_importance(&plot_type);

        let plot_point = PlotPoint {
            id: plot_id,
            title: String::from("Plot Point"),
            description: self.describe_plot_type(&plot_type),
            sequence_number: 0,
            timestamp: Utc::now(),
            plot_type,
            status: PlotPointStatus::Pending,
            estimated_duration: None,
            actual_duration: None,
            context_tokens,
            importance,
            metadata: PlotMetadata::default(),
            tags: vec![],
            consequences: vec![],
        };

        // Add to current arc or create new one
        if let Some(arc_id) = story.current_arc {
            if let Some(arc) = story.arcs.iter_mut().find(|a| a.id == arc_id) {
                arc.plot_points.push(plot_point.clone());
            }
        } else {
            // Create new arc
            let arc = StoryArc {
                id: StoryArcId(Uuid::new_v4()),
                title: "Initial Development".to_string(),
                description: "The beginning of the story".to_string(),
                sequence_number: 0,
                plot_points: vec![plot_point.clone()],
                started_at: Utc::now(),
                completed_at: None,
                status: ArcStatus::Active,
            };
            story.current_arc = Some(arc.id);
            story.arcs.push(arc);
        }

        story.updated_at = Utc::now();

        // Update context chain
        if let Some(chain) = self.context_chains.get_mut(&story.context_chain) {
            chain.add_plot_point(&plot_point).await?;
        }

        // Add to context manager
        let ctx_mgr = self.context_manager.write().await;
        ctx_mgr.add_narrative(
            format!("Plot point: {}", plot_point.description),
            TokenMetadata {
                source: "story_engine".to_string(),
                emotional_valence: 0.0,
                attention_weight: importance,
                associations: plot_point.context_tokens.clone(),
                compressed: false,
            },
        ).await?;
        drop(ctx_mgr);

        let _ = self.event_tx.send(StoryEvent::PlotPointAdded(story_id, plot_id));

        Ok(plot_id)
    }

    /// Get a story by ID
    pub fn get_story(&self, story_id: &StoryId) -> Option<Story> {
        self.stories.get(story_id).map(|s| s.clone())
    }

    /// Get all stories of a specific type
    pub fn get_stories_by_type(&self, story_type_filter: impl Fn(&StoryType) -> bool) -> Vec<Story> {
        self.stories
            .iter()
            .filter(|entry| story_type_filter(&entry.value().story_type))
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Create a task mapping for a story
    pub async fn create_task_map(&self, story_id: StoryId) -> Result<TaskMap> {
        let story = self.stories
            .get(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;

        let task_map = self.task_mapper.map_story_to_tasks(&story).await?;

        // Emit event for each mapped task
        for task in &task_map.tasks {
            let _ = self.event_tx.send(StoryEvent::TaskMapped(story_id, task.id.clone()));
        }

        Ok(task_map)
    }

    /// Synchronize stories
    pub async fn sync_stories(&self, story_ids: Vec<StoryId>) -> Result<SyncEvent> {
        let stories: Vec<Story> = story_ids
            .iter()
            .filter_map(|id| self.get_story(id))
            .collect();

        if stories.is_empty() {
            return Err(anyhow::anyhow!("No valid stories to sync"));
        }

        let sync_event = self.synchronizer.sync_multiple(stories).await?;

        let _ = self.event_tx.send(StoryEvent::SyncCompleted(sync_event.clone()));

        Ok(sync_event)
    }

    /// Subscribe to story events
    pub fn subscribe(&self) -> broadcast::Receiver<StoryEvent> {
        self.event_tx.subscribe()
    }


    // Helper methods

    fn calculate_importance(&self, plot_type: &PlotType) -> f32 {
        match plot_type {
            PlotType::Goal { .. } => 0.9,
            PlotType::Decision { .. } => 0.85,
            PlotType::Issue { resolved: false, .. } => 0.8,
            PlotType::Discovery { .. } => 0.75,
            PlotType::Transformation { .. } => 0.7,
            PlotType::Task { completed: false, .. } => 0.65,
            PlotType::Task { completed: true, .. } => 0.5,
            PlotType::Issue { resolved: true, .. } => 0.5,
            PlotType::Interaction { .. } => 0.6,
            PlotType::Progress { .. } => 0.7,
            PlotType::Analysis { .. } => 0.8,
            PlotType::Action { .. } => 0.75,
            PlotType::Reasoning { confidence, .. } => 0.7 * confidence,
            PlotType::Event { impact, .. } => 0.6 * impact,
            PlotType::Context { .. } => 0.5,
        }
    }

    fn describe_plot_type(&self, plot_type: &PlotType) -> String {
        match plot_type {
            PlotType::Goal { objective } => format!("Goal: {}", objective),
            PlotType::Task { description, completed } => {
                format!("Task [{}]: {}", if *completed { "✓" } else { " " }, description)
            }
            PlotType::Decision { question, choice } => {
                format!("Decision: {} → {}", question, choice)
            }
            PlotType::Discovery { insight } => format!("Discovery: {}", insight),
            PlotType::Issue { error, resolved } => {
                format!("Issue [{}]: {}", if *resolved { "resolved" } else { "active" }, error)
            }
            PlotType::Transformation { before, after } => {
                format!("Transform: {} → {}", before, after)
            }
            PlotType::Interaction { with, action } => {
                format!("Interaction with {}: {}", with, action)
            }
            PlotType::Progress { milestone, percentage } => {
                format!("Progress: {} ({:.1}%)", milestone, percentage)
            }
            PlotType::Analysis { subject, findings } => {
                format!("Analysis of {}: {} findings", subject, findings.len())
            }
            PlotType::Action { action_type, parameters, outcome } => {
                format!("Action {}: {} → {}", action_type, parameters.join(", "), outcome)
            }
            PlotType::Reasoning { premise, conclusion, confidence } => {
                format!("Reasoning: {} => {} (confidence: {:.1}%)", premise, conclusion, confidence * 100.0)
            }
            PlotType::Event { event_type, description, impact } => {
                format!("Event [{}]: {} (impact: {:.1})", event_type, description, impact)
            }
            PlotType::Context { context_type, data } => {
                format!("Context [{}]: {}", context_type, data)
            }
        }
    }

    // Template methods

    /// Initialize template manager with engine reference
    pub async fn init_template_manager(self: Arc<Self>) {
        let mut template_manager = self.template_manager.lock().await;
        template_manager.set_story_engine(self.clone());
    }

    /// Get template manager
    pub fn template_manager(&self) -> Arc<Mutex<StoryTemplateManager>> {
        self.template_manager.clone()
    }

    /// Get template tracker
    pub fn template_tracker(&self) -> Arc<Mutex<TemplateInstanceTracker>> {
        self.template_tracker.clone()
    }

    /// Create a story porter for export/import
    pub fn create_porter(self: Arc<Self>) -> StoryPorter {
        StoryPorter::new(self)
    }

    /// Get or create system story
    pub async fn get_or_create_system_story(&self, name: String) -> Result<StoryId> {
        // Check if system story already exists
        for story_ref in self.stories.iter() {
            let story = story_ref.value();
            if matches!(story.story_type, StoryType::System { .. }) && story.title == name {
                return Ok(story.id);
            }
        }

        // Create new system story
        let story_id = StoryId::new();
        let chain_id = ChainId(Uuid::new_v4());

        let story = Story {
            id: story_id,
            story_type: StoryType::System { component: name.clone() },
            title: format!("System Story: {}", name),
            description: format!("Automated story tracking for {} system component", name),
            summary: format!("Automated story tracking for {} system component", name),
            status: StoryStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            arcs: Vec::new(),
            current_arc: None,
            metadata: StoryMetadata {
                priority: Priority::Medium,
                tags: vec!["system".to_string(), name.to_lowercase()],
                complexity: 0.5,
                custom_data: HashMap::new(),
                dependencies: Vec::new(),
                related_stories: Vec::new(),
            },
            context_chain: chain_id,
            segments: vec![],
            context: HashMap::new(),
        };

        self.stories.insert(story_id, story);
        self.event_tx.send(StoryEvent::StoryCreated(story_id)).ok();

        Ok(story_id)
    }

    /// Create story from template
    pub async fn create_story_from_template(
        &self,
        template_id: TemplateId,
        story_type: StoryType,
        context: HashMap<String, String>,
    ) -> Result<StoryId> {
        let mut template_manager = self.template_manager.lock().await;
        let story_id = template_manager.instantiate_template(
            template_id,
            story_type,
            context.clone(),
        ).await?;

        // Track the instance
        let mut tracker = self.template_tracker.lock().await;
        tracker.track_instance(TemplateInstance {
            story_id,
            template_id,
            started_at: chrono::Utc::now(),
            completed_at: None,
            current_plot_index: 0,
            context,
        });

        Ok(story_id)
    }

    /// Get template recommendations
    pub async fn recommend_templates(&self, context: &str) -> Vec<(StoryTemplate, f32)> {
        let template_manager = self.template_manager.lock().await;
        template_manager.recommend_templates(context)
            .into_iter()
            .map(|(t, score)| (t.clone(), score))
            .collect()
    }

    /// Complete template instance
    pub async fn complete_template_instance(
        &self,
        story_id: StoryId,
        success: bool,
    ) -> Result<()> {
        let mut tracker = self.template_tracker.lock().await;

        if let Some(duration) = tracker.complete_instance(story_id) {
            // Find template ID
            let instances = tracker.get_active_instances();
            if let Some(instance) = instances.iter().find(|i| i.story_id == story_id) {
                let mut template_manager = self.template_manager.lock().await;
                template_manager.update_template_metrics(
                    instance.template_id,
                    success,
                    duration,
                ).await?;
            }
        }

        Ok(())
    }

    /// Save all stories to persistent storage
    pub async fn save_to_persistence(&self, path: &PathBuf) -> Result<()> {
        info!("Saving stories to persistent storage at {:?}", path);

        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Export all stories
        let export_options = crate::story::ExportOptions {
            include_completed: true,
            include_context_chains: true,
            compress: true,
            format: crate::story::ExportFormat::Json,
        };

        let export_data = self.export_stories(export_options).await?;

        // Serialize to JSON
        let json_data = serde_json::to_string_pretty(&export_data)?;

        // Write to file
        tokio::fs::write(path, json_data).await?;

        info!("Successfully saved {} stories to {:?}", export_data.stories.len(), path);
        Ok(())
    }

    /// Load stories from persistent storage
    pub async fn load_from_persistence(&self, path: &PathBuf) -> Result<()> {
        info!("Loading stories from persistent storage at {:?}", path);

        // Check if file exists
        if !path.exists() {
            info!("No persistence file found at {:?}, starting fresh", path);
            return Ok(());
        }

        // Read file
        let json_data = tokio::fs::read_to_string(path).await?;

        // Deserialize
        let export_data: crate::story::StoryExport = serde_json::from_str(&json_data)?;

        info!("Found {} stories to import", export_data.stories.len());

        // Import with merge strategy
        let result = self.import_stories(export_data, crate::story::MergeStrategy::Skip).await?;

        info!(
            "Successfully imported {} stories, skipped {}, {} errors",
            result.imported_stories,
            result.skipped_stories,
            result.errors.len()
        );

        if !result.errors.is_empty() {
            warn!("Import errors: {:?}", result.errors);
        }

        Ok(())
    }

    /// Auto-save stories periodically
    pub fn start_auto_persistence(self: Arc<Self>, path: PathBuf, interval: Duration) {
        let engine = self.clone();
        let mut shutdown_rx = self.shutdown.subscribe();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        if let Err(e) = engine.save_to_persistence(&path).await {
                            error!("Failed to auto-save stories: {}", e);
                        } else {
                            debug!("Auto-saved stories to {:?}", path);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Shutting down story persistence, saving final state...");
                        if let Err(e) = engine.save_to_persistence(&path).await {
                            error!("Failed to save stories on shutdown: {}", e);
                        } else {
                            info!("Successfully saved stories on shutdown");
                        }
                        break;
                    }
                }
            }
        });

        info!("Started auto-persistence with interval {:?}", interval);
    }

    /// Start watching a directory for file changes
    pub async fn watch_directory(&self, path: PathBuf) -> Result<()> {
        // Initialize file watcher if not already done
        if self.file_watcher.is_none() {
            // Note: File watcher initialization requires Arc<StoryEngine>
            // This should be initialized from where StoryEngine is created as Arc
            return Err(anyhow::anyhow!("File watcher must be initialized externally with Arc<StoryEngine>"));
        }

        // Start watching the directory
        if let Some(watcher) = &self.file_watcher {
            let mut watcher = watcher.lock().await;
            watcher.watch_path(path).await?;
        }

        Ok(())
    }

    /// Stop watching a directory
    pub async fn unwatch_directory(&self, path: &Path) -> Result<()> {
        if let Some(watcher) = &self.file_watcher {
            let mut watcher = watcher.lock().await;
            watcher.unwatch_path(path).await?;
        }
        Ok(())
    }

    /// Extract all tasks from all stories
    pub async fn extract_all_tasks(&self) -> Result<HashMap<StoryId, Vec<MappedTask>>> {
        let mut all_tasks = HashMap::new();

        // Iterate through all stories
        for story_ref in self.stories.iter() {
            let story_id = story_ref.key().clone();
            let story = story_ref.value();
            let mut story_tasks = Vec::new();

            // Extract tasks from each arc in the story
            for arc in &story.arcs {
                for plot_point in &arc.plot_points {
                    // Check if this plot point represents a task
                    if let PlotType::Task { description, completed } = &plot_point.plot_type {
                        let task = MappedTask {
                            id: plot_point.id.0.to_string(),
                            description: description.clone(),
                            story_context: format!("Story: {}, Arc: {}", story.title, arc.title),
                            status: if *completed {
                                TaskStatus::Completed
                            } else {
                                TaskStatus::InProgress
                            },
                            assigned_to: None, // No assigned_to field in new metadata structure
                            created_at: plot_point.timestamp,
                            updated_at: plot_point.timestamp,
                            plot_point: Some(plot_point.id),
                        };
                        story_tasks.push(task);
                    }
                }
            }

            // Also use the task mapper to extract additional tasks
            if let Ok(task_map) = self.task_mapper.map_story_to_tasks(&story).await {
                for mapped_task in task_map.tasks {
                    // Avoid duplicates by checking task IDs
                    if !story_tasks.iter().any(|t| t.id == mapped_task.id) {
                        story_tasks.push(mapped_task);
                    }
                }
            }

            if !story_tasks.is_empty() {
                all_tasks.insert(story_id, story_tasks);
            }
        }

        info!("Extracted {} stories with tasks", all_tasks.len());
        Ok(all_tasks)
    }

    /// Get a specific segment from a story
    pub async fn get_segment(&self, story_id: &StoryId, segment_id: &str) -> Result<StorySegment> {
        // Get the story
        let story = self.get_story(story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found: {:?}", story_id))?;

        // Search for the segment in story arcs
        for arc in &story.arcs {
            // Check if the arc ID matches the segment ID
            if arc.id.0.to_string() == segment_id {
                return Ok(StorySegment {
                    id: segment_id.to_string(),
                    story_id: story_id.clone(),
                    content: arc.description.clone(),
                    created_at: arc.started_at,
                    context: HashMap::from([
                        ("arc_title".to_string(), arc.title.clone()),
                        ("arc_status".to_string(), format!("{:?}", arc.status)),
                        ("plot_points".to_string(), arc.plot_points.len().to_string()),
                    ]),
                    segment_type: match arc.status {
                        ArcStatus::Planning => SegmentType::Introduction,
                        ArcStatus::Active => SegmentType::Development,
                        ArcStatus::Completed => SegmentType::Resolution,
                        _ => SegmentType::Development,
                    },
                    tags: vec!["arc".to_string(), arc.status.to_string().to_lowercase()],
                });
            }

            // Also check plot points within the arc
            for plot_point in &arc.plot_points {
                if plot_point.id.0.to_string() == segment_id {
                    let content = match &plot_point.plot_type {
                        PlotType::Goal { objective } => objective.clone(),
                        PlotType::Task { description, .. } => description.clone(),
                        PlotType::Decision { question, choice } => format!("{}: {}", question, choice),
                        PlotType::Discovery { insight } => insight.clone(),
                        PlotType::Issue { error, .. } => error.clone(),
                        PlotType::Transformation { before, after } => format!("Transform: {} -> {}", before, after),
                        PlotType::Interaction { with, action } => format!("{} with {}", action, with),
                        PlotType::Analysis { subject, .. } => subject.clone(),
                        PlotType::Progress { milestone, percentage } => format!("{} ({}%)", milestone, percentage * 100.0),
                        PlotType::Action { action_type, outcome, .. } => format!("{}: {}", action_type, outcome),
                        PlotType::Reasoning { premise, conclusion, confidence } => {
                            format!("Reasoning: {} => {} (confidence: {:.1}%)", premise, conclusion, confidence * 100.0)
                        },
                        PlotType::Event { event_type, description, .. } => {
                            format!("{}: {}", event_type, description)
                        },
                        PlotType::Context { context_type, data } => {
                            format!("{}: {}", context_type, data)
                        },
                    };

                    return Ok(StorySegment {
                        id: segment_id.to_string(),
                        story_id: story_id.clone(),
                        content,
                        created_at: plot_point.timestamp,
                        context: HashMap::new(), // Convert metadata to HashMap if needed
                        segment_type: match &plot_point.plot_type {
                            PlotType::Goal { .. } => SegmentType::Introduction,
                            PlotType::Task { .. } => SegmentType::Development,
                            PlotType::Decision { .. } => SegmentType::Development,
                            PlotType::Discovery { .. } => SegmentType::Development,
                            PlotType::Issue { .. } => SegmentType::Development,
                            PlotType::Transformation { .. } => SegmentType::Resolution,
                            PlotType::Interaction { .. } => SegmentType::Introduction,
                            PlotType::Analysis { .. } => SegmentType::Resolution,
                            PlotType::Progress { .. } => SegmentType::Development,
                            PlotType::Action { .. } => SegmentType::Resolution,
                            PlotType::Reasoning { .. } => SegmentType::Development,
                            PlotType::Event { .. } => SegmentType::Development,
                            PlotType::Context { .. } => SegmentType::Introduction,
                        },
                        tags: plot_point.tags.clone(),
                    });
                }
            }
        }

        // If not found in arcs, create a segment from the story summary
        warn!("Segment {} not found in story {:?}, creating from summary", segment_id, story_id);
        Ok(StorySegment {
            id: segment_id.to_string(),
            story_id: story_id.clone(),
            content: story.summary.clone(),
            created_at: story.created_at,
            context: HashMap::from([
                ("source".to_string(), "story_summary".to_string()),
                ("story_title".to_string(), story.title.clone()),
            ]),
            segment_type: SegmentType::Introduction,
            tags: vec!["summary".to_string()],
        })
    }

    /// Extract tasks from a story segment
    pub async fn extract_tasks_from_segment(&self, segment: &StorySegment) -> Result<Vec<MappedTask>> {
        let mut tasks = Vec::new();

        // Extract tasks based on segment content patterns
        let content_lower = segment.content.to_lowercase();

        // Common task indicators
        let task_patterns = [
            ("todo:", "TODO"),
            ("task:", "Task"),
            ("implement", "Implementation"),
            ("fix", "Bug Fix"),
            ("add", "Feature Addition"),
            ("update", "Update"),
            ("create", "Creation"),
            ("refactor", "Refactoring"),
            ("test", "Testing"),
            ("document", "Documentation"),
        ];

        // Check for task patterns in content
        for (pattern, task_type) in &task_patterns {
            if content_lower.contains(pattern) {
                // Extract the task description
                let lines: Vec<&str> = segment.content.lines().collect();
                for (i, line) in lines.iter().enumerate() {
                    if line.to_lowercase().contains(pattern) {
                        // Get the task description from this line and possibly the next
                        let mut description = line.trim().to_string();
                        if description.ends_with(':') && i + 1 < lines.len() {
                            description = format!("{} {}", description, lines[i + 1].trim());
                        }

                        // Create a task
                        let task = MappedTask {
                            id: format!("{}_{}", segment.id, tasks.len()),
                            description: description.clone(),
                            story_context: format!("Segment: {} ({})", segment.id, task_type),
                            status: TaskStatus::Pending,
                            assigned_to: segment.context.get("assigned_to").cloned(),
                            created_at: segment.created_at,
                            updated_at: Utc::now(),
                            plot_point: None,
                        };
                        tasks.push(task);
                        break; // Only one task per pattern to avoid duplicates
                    }
                }
            }
        }

        // Also check segment context for explicit tasks
        if let Some(task_data) = segment.context.get("task") {
            let task = MappedTask {
                id: format!("{}_context", segment.id),
                description: task_data.clone(),
                story_context: format!("Segment: {} (from context)", segment.id),
                status: segment.context.get("status")
                    .and_then(|s| match s.as_str() {
                        "completed" => Some(TaskStatus::Completed),
                        "in_progress" => Some(TaskStatus::InProgress),
                        "blocked" => Some(TaskStatus::Blocked),
                        _ => None,
                    })
                    .unwrap_or(TaskStatus::Pending),
                assigned_to: segment.context.get("assigned_to").cloned(),
                created_at: segment.created_at,
                updated_at: Utc::now(),
                plot_point: None,
            };
            tasks.push(task);
        }

        info!("Extracted {} tasks from segment {}", tasks.len(), segment.id);
        Ok(tasks)
    }

    /// Export stories based on options
    pub async fn export_stories(&self, options: crate::story::ExportOptions) -> Result<crate::story::StoryExport> {
        let mut exported_stories = Vec::new();
        let mut exported_chains = Vec::new();
        let mut total_plot_points = 0;

        // Export all stories
        for story_ref in self.stories.iter() {
            let story = story_ref.value();

            // Skip completed stories if not included
            if !options.include_completed && story.status == StoryStatus::Completed {
                continue;
            }

            // Count plot points
            total_plot_points += story.arcs.iter()
                .map(|arc| arc.plot_points.len())
                .sum::<usize>();

            // Populate context from context chain
            let mut context = HashMap::new();
            if let Some(chain) = self.context_chains.get(&story.context_chain) {
                // Get recent context with a reasonable token limit
                match chain.get_recent_context(1000).await {
                    Ok(recent_context) => {
                        context.insert("recent_context".to_string(), recent_context);

                        // Add chain metadata
                        let chain_metadata = chain.metadata.read().await;
                        context.insert("chain_id".to_string(), chain.id.to_string());
                        context.insert("total_segments".to_string(),
                            chain_metadata.total_segments.to_string());
                        context.insert("compressed_segments".to_string(),
                            chain_metadata.compressed_segments.to_string());
                        context.insert("average_importance".to_string(),
                            chain_metadata.average_importance.to_string());
                        if let Some(last_comp) = chain_metadata.last_compression {
                            context.insert("last_compression".to_string(),
                                last_comp.to_rfc3339());
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to get context for story {}: {}", story.id, e);
                        context.insert("error".to_string(),
                            format!("Failed to retrieve context: {}", e));
                    }
                }
            } else {
                context.insert("warning".to_string(),
                    "No context chain found for this story".to_string());
            }

            // Populate relationships from story metadata
            let mut relationships = Vec::new();

            // Add dependency relationships
            for dep_id in &story.metadata.dependencies {
                relationships.push(crate::story::StoryRelationship {
                    from_story: story.id.clone(),
                    to_story: dep_id.clone(),
                    relationship_type: "depends_on".to_string(),
                    metadata: HashMap::new(),
                });
            }

            // Add related story relationships
            for related_id in &story.metadata.related_stories {
                relationships.push(crate::story::StoryRelationship {
                    from_story: story.id.clone(),
                    to_story: related_id.clone(),
                    relationship_type: "related_to".to_string(),
                    metadata: HashMap::new(),
                });
            }

            // Convert Story to ExportedStory
            let exported_story = crate::story::ExportedStory {
                id: story.id.clone(),
                title: story.title.clone(),
                summary: story.summary.clone(),
                story_type: story.story_type.clone(),
                plot_points: story.arcs.iter()
                    .flat_map(|arc| arc.plot_points.clone())
                    .collect(),
                arcs: story.arcs.clone(),
                metadata: story.metadata.clone(),
                context,
                relationships,
            };

            exported_stories.push(exported_story);
        }

        // Export context chains if requested
        if options.include_context_chains {
            for chain_ref in self.context_chains.iter() {
                let chain = chain_ref.value();
                let exported_chain = crate::story::export_import::ExportedContextChain {
                    id: chain.id,
                    story_id: chain.story_id,
                    segments: chain.segments.read().await.clone().into_iter().collect(),
                    links: chain.links.read().await.clone(),
                    metadata: chain.metadata.read().await.clone(),
                };
                exported_chains.push(exported_chain);
            }
        }

        let export_metadata = ExportMetadata {
            total_stories: exported_stories.len(),
            total_context_chains: exported_chains.len(),
            total_plot_points,
            export_options: options,
        };

        Ok(crate::story::StoryExport {
            version: "1.0".to_string(),
            exported_at: chrono::Utc::now(),
            stories: exported_stories,
            context_chains: exported_chains,
            metadata: export_metadata,
        })
    }

    /// Import stories from export data
    pub async fn import_stories(
        &self,
        export_data: crate::story::StoryExport,
        merge_strategy: crate::story::MergeStrategy,
    ) -> Result<crate::story::ImportResult> {
        let mut imported_stories = 0;
        let mut imported_chains = 0;
        let mut conflicts = Vec::new();

        // Import stories
        for exported_story in export_data.stories {
            let story_id = exported_story.id.clone();
            let story_title = exported_story.title.clone();

            // Convert ExportedStory back to Story
            // Find current arc before moving arcs
            let current_arc = exported_story.arcs.iter()
                .find(|arc| arc.status == ArcStatus::Active)
                .or_else(|| exported_story.arcs.last())
                .map(|arc| arc.id);

            let story = Story {
                id: exported_story.id,
                story_type: exported_story.story_type,
                title: exported_story.title.clone(),
                description: exported_story.summary.clone(),
                summary: exported_story.summary,
                status: StoryStatus::Active, // Default status for imported stories
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                arcs: exported_story.arcs,
                current_arc,
                metadata: exported_story.metadata,
                context_chain: ChainId(Uuid::new_v4()), // Will be remapped if context chains are imported
                segments: vec![], // Initialize empty segments for imported stories
                context: HashMap::new(),
            };

            if self.stories.contains_key(&story_id) {
                // Handle conflict based on merge strategy
                match merge_strategy {
                    crate::story::MergeStrategy::Skip => {
                        conflicts.push(ImportConflict {
                            item_type: "story".to_string(),
                            item_id: story_id.0.to_string(),
                            existing_title: self.stories.get(&story_id).map(|s| s.title.clone()),
                            imported_title: Some(story_title.clone()),
                            conflict_type: "Already exists".to_string(),
                        });
                        continue;
                    }
                    crate::story::MergeStrategy::Replace => {
                        self.stories.insert(story_id, story);
                        imported_stories += 1;
                    }
                    crate::story::MergeStrategy::Merge => {
                        // For now, just replace - full merge logic would be more complex
                        self.stories.insert(story_id, story);
                        imported_stories += 1;
                    }
                    crate::story::MergeStrategy::RenameNew => {
                        // Create a new story with a different ID
                        let new_story_id = StoryId(Uuid::new_v4());
                        let mut new_story = story;
                        new_story.id = new_story_id;
                        new_story.title = format!("{} (imported)", new_story.title);
                        self.stories.insert(new_story_id, new_story);
                        imported_stories += 1;
                    }
                }
            } else {
                self.stories.insert(story_id, story);
                imported_stories += 1;
            }
        }

        // Import context chains
        for exported_chain in export_data.context_chains {
            let chain_id = exported_chain.id;

            // Convert ExportedContextChain back to ContextChain
            let chain = crate::story::context_chain::ContextChain {
                id: exported_chain.id,
                story_id: exported_chain.story_id,
                segments: RwLock::new(exported_chain.segments.into_iter().collect()),
                links: RwLock::new(exported_chain.links),
                metadata: RwLock::new(exported_chain.metadata),
            };

            if self.context_chains.contains_key(&chain_id) {
                conflicts.push(ImportConflict {
                    item_type: "chain".to_string(),
                    item_id: chain_id.0.to_string(),
                    existing_title: None,
                    imported_title: None,
                    conflict_type: "Already exists".to_string(),
                });

                if merge_strategy == crate::story::MergeStrategy::Replace {
                    self.context_chains.insert(chain_id, chain);
                    imported_chains += 1;
                }
            } else {
                self.context_chains.insert(chain_id, chain);
                imported_chains += 1;
            }
        }

        let skipped_stories = conflicts.iter()
            .filter(|c| c.item_type == "story")
            .count();

        Ok(crate::story::ImportResult {
            imported_stories,
            skipped_stories,
            imported_chains,
            conflicts: conflicts.clone(),
            errors: conflicts.into_iter()
                .map(|c| format!("{}: {} - {}", c.item_type, c.item_id, c.conflict_type))
                .collect(),
        })
    }

    /// Shutdown the story engine gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down story engine...");

        // Shutdown file watcher if present
        if let Some(watcher) = &self.file_watcher {
            let watcher = watcher.lock().await;
            watcher.shutdown().await?;
        }

        // Send shutdown signal
        let _ = self.shutdown.send(());

        // Wait a moment for persistence to complete
        tokio::time::sleep(Duration::from_millis(500)).await;

        info!("Story engine shutdown complete");
        Ok(())
    }

    /// Get active node count (stub implementation)
    pub fn get_active_node_count(&self) -> Result<usize> {
        // For now, return the number of active stories
        // In a full implementation, this would count actual story nodes
        Ok(self.stories.len())
    }

    /// Get active story count
    pub fn get_active_story_count(&self) -> Result<usize> {
        let stories = self.stories.iter()
            .filter(|s| matches!(s.status, StoryStatus::Active))
            .count();
        Ok(stories)
    }

    /// Get total story count
    pub fn get_total_story_count(&self) -> Result<usize> {
        Ok(self.stories.len())
    }
    
    /// Get current narrative context across all active stories
    pub async fn get_current_context(&self) -> Result<NarrativeContext> {
        // Find the most relevant active stories
        let mut active_stories = Vec::new();
        for story_ref in self.stories.iter() {
            let story = story_ref.value();
            if matches!(story.status, StoryStatus::Active) {
                active_stories.push(story.clone());
            }
        }
        
        // Sort by last updated to get most recent
        active_stories.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        
        // Get the primary story (most recently updated)
        let primary_story = active_stories.first();
        
        // Build narrative context
        let mut context = NarrativeContext {
            story_id: primary_story.map(|s| s.id).unwrap_or_else(|| StoryId(uuid::Uuid::new_v4())),
            current_plot: String::new(),
            current_arc: None,
            current_chapter: String::new(),
            active_plot_points: Vec::new(),
            story_themes: Vec::new(),
            character_state: HashMap::new(),
            recent_events: Vec::new(),
            narrative_tension: 0.5,
            active_characters: Vec::new(),
            plot_threads: Vec::new(),
            tone: "neutral".to_string(),
        };
        
        if let Some(story) = primary_story {
            // Set current chapter from active arc
            if let Some(arc) = story.arcs.iter().find(|a| matches!(a.status, ArcStatus::Active)) {
                context.current_chapter = arc.title.clone();
                context.current_arc = Some(arc.clone());
                
                // Add active plot points
                for plot_point in &arc.plot_points {
                    context.active_plot_points.push(plot_point.description.clone());
                }
            }
            
            // Extract themes from story metadata
            if let Some(themes_value) = story.metadata.custom_data.get("themes") {
                if let Some(themes_array) = themes_value.as_array() {
                    for theme in themes_array {
                        if let Some(theme_str) = theme.as_str() {
                            context.story_themes.push(theme_str.to_string());
                        }
                    }
                }
            }
            
            // Add recent events from all active stories
            for story in &active_stories[..active_stories.len().min(3)] {
                for arc in &story.arcs {
                    for plot_point in &arc.plot_points {
                        context.recent_events.push(format!(
                            "[{}] {}", 
                            story.title, 
                            plot_point.description
                        ));
                    }
                }
            }
            
            // Calculate narrative tension based on story progress
            let completed_arcs = story.arcs.iter()
                .filter(|a| matches!(a.status, ArcStatus::Completed))
                .count();
            let total_arcs = story.arcs.len();
            if total_arcs > 0 {
                context.narrative_tension = 0.3 + (0.5 * completed_arcs as f64 / total_arcs as f64);
            }
        }
        
        Ok(context)
    }
}

// Extension trait for CognitiveMemory to support stories
#[async_trait::async_trait]
trait StoryMemoryExt {
    async fn store_story_metadata(&self, story_id: StoryId, story: &Story) -> Result<()>;
}

#[async_trait::async_trait]
impl StoryMemoryExt for CognitiveMemory {
    async fn store_story_metadata(&self, story_id: StoryId, story: &Story) -> Result<()> {
        // Store as JSON in memory
        let story_json = serde_json::to_string(story)?;
        let metadata = MemoryMetadata {
            tags: vec!["story".to_string(), "metadata".to_string()],
            importance: 0.8,
            source: "story_engine".to_string(),
            ..Default::default()
        };

        self.store(
            story_json,
            vec![format!("story_id:{}", story_id.0)],
            metadata,
        ).await?;
        Ok(())
    }
}
