//! Story Bridge - Integrates story engine with chat and todo systems
//! 
//! This bridge enables narrative-driven task execution by connecting
//! the story engine with todos, orchestration, and agent systems.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, broadcast};
use anyhow::Result;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tracing::{info, debug, warn, error};

use crate::story::{
    StoryEngine, StoryId, Story, StoryEvent as StoryEngineEvent,
    PlotPoint, PlotPointId, PlotType,
    TaskMapper, StoryContext,
    agent_coordination::StoryAgentCoordinator,
};
use crate::tui::chat::orchestration::todo_manager::{
    TodoManager, TodoItem, CreateTodoRequest, TodoStatus,
};
use crate::tui::event_bus::{EventBus, SystemEvent, TabId};
use crate::cognitive::CognitiveEvent;

/// Story bridge for narrative integration
pub struct StoryBridge {
    /// Story engine reference
    story_engine: Arc<StoryEngine>,
    
    /// Todo manager reference
    todo_manager: Option<Arc<TodoManager>>,
    
    /// Task mapper for story-task conversion
    task_mapper: Arc<TaskMapper>,
    
    /// Agent coordinator for collaborative stories
    agent_coordinator: Option<Arc<StoryAgentCoordinator>>,
    
    /// Event bus for cross-tab communication
    event_bus: Arc<EventBus>,
    
    /// Active story mappings
    story_mappings: Arc<RwLock<StoryMappings>>,
    
    /// Story event receiver
    story_event_rx: Option<broadcast::Receiver<StoryEngineEvent>>,
    
    /// Cognitive event channel
    cognitive_tx: Option<mpsc::Sender<CognitiveEvent>>,
    
    /// Bridge configuration
    config: StoryBridgeConfig,
    
    /// Bridge state
    state: Arc<RwLock<BridgeState>>,
}

/// Story mappings between different systems
#[derive(Debug, Clone, Default)]
struct StoryMappings {
    /// Todo ID to Story ID mapping
    todo_to_story: HashMap<String, StoryId>,
    
    /// Story ID to Todo IDs mapping
    story_to_todos: HashMap<StoryId, Vec<String>>,
    
    /// Plot point to Todo mapping
    plot_to_todo: HashMap<PlotPointId, String>,
    
    /// Agent ID to Story ID mapping
    agent_to_story: HashMap<String, StoryId>,
    
    /// Chat ID to Story ID mapping
    chat_to_story: HashMap<String, StoryId>,
}

/// Bridge configuration
#[derive(Debug, Clone)]
pub struct StoryBridgeConfig {
    pub auto_create_stories: bool,
    pub sync_interval_seconds: u64,
    pub narrative_generation: bool,
    pub plot_point_threshold: f32,
    pub max_story_depth: usize,
    pub enable_collaborative_stories: bool,
    pub story_persistence: bool,
}

impl Default for StoryBridgeConfig {
    fn default() -> Self {
        Self {
            auto_create_stories: true,
            sync_interval_seconds: 30,
            narrative_generation: true,
            plot_point_threshold: 0.7,
            max_story_depth: 10,
            enable_collaborative_stories: true,
            story_persistence: true,
        }
    }
}

/// Bridge state
#[derive(Debug, Clone)]
struct BridgeState {
    initialized: bool,
    active_stories: usize,
    last_sync: DateTime<Utc>,
    events_processed: usize,
    todos_created_from_stories: usize,
    stories_created_from_todos: usize,
}

/// Story event for broadcasting
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum StoryEvent {
    StoryCreated {
        story_id: StoryId,
        title: String,
        description: String,
    },
    PlotPointAdded {
        story_id: StoryId,
        plot_point: PlotPointSummary,
    },
    TodoMapped {
        todo_id: String,
        story_id: StoryId,
        plot_point_id: PlotPointId,
    },
    StoryCompleted {
        story_id: StoryId,
        summary: String,
    },
    NarrativeGenerated {
        story_id: StoryId,
        narrative: String,
    },
}

/// Plot point summary for events
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlotPointSummary {
    pub id: PlotPointId,
    pub description: String,
    pub plot_type: String,
    pub timestamp: DateTime<Utc>,
}

impl StoryBridge {
    /// Create a new story bridge
    pub fn new(
        story_engine: Arc<StoryEngine>,
        event_bus: Arc<EventBus>,
    ) -> Self {
        let task_mapper = Arc::new(TaskMapper::new());
        
        Self {
            story_engine,
            todo_manager: None,
            task_mapper,
            agent_coordinator: None,
            event_bus,
            story_mappings: Arc::new(RwLock::new(StoryMappings::default())),
            story_event_rx: None,
            cognitive_tx: None,
            config: StoryBridgeConfig::default(),
            state: Arc::new(RwLock::new(BridgeState {
                initialized: false,
                active_stories: 0,
                last_sync: Utc::now(),
                events_processed: 0,
                todos_created_from_stories: 0,
                stories_created_from_todos: 0,
            })),
        }
    }
    
    /// Set todo manager
    pub fn set_todo_manager(&mut self, todo_manager: Arc<TodoManager>) {
        self.todo_manager = Some(todo_manager);
        info!("Todo manager connected to story bridge");
    }
    
    /// Set agent coordinator
    pub fn set_agent_coordinator(&mut self, coordinator: Arc<StoryAgentCoordinator>) {
        self.agent_coordinator = Some(coordinator);
        info!("Agent coordinator connected to story bridge");
    }
    
    /// Set cognitive event channel
    pub fn set_cognitive_channel(&mut self, tx: mpsc::Sender<CognitiveEvent>) {
        self.cognitive_tx = Some(tx);
        info!("Cognitive channel connected to story bridge");
    }
    
    /// Initialize the bridge
    pub async fn initialize(&self) -> Result<()> {
        // Subscribe to story engine events
        let mut rx = self.story_engine.subscribe();
        let bridge = self.clone();
        tokio::spawn(async move {
            while let Ok(event) = rx.recv().await {
                if let Err(e) = bridge.handle_story_event(event).await {
                    error!("Error handling story event: {}", e);
                }
            }
        });
        
        // Start sync task
        if self.config.sync_interval_seconds > 0 {
            self.start_sync_task().await;
        }
        
        // Initialize state
        let mut state = self.state.write().await;
        state.initialized = true;
        state.active_stories = self.story_engine.stories.len();
        
        info!("Story bridge initialized with {} active stories", state.active_stories);
        Ok(())
    }
    
    /// Start background sync task
    async fn start_sync_task(&self) {
        let bridge = self.clone();
        let interval = self.config.sync_interval_seconds;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = bridge.sync_stories_and_todos().await {
                    error!("Story sync error: {}", e);
                }
            }
        });
    }
    
    /// Create a story from a todo
    pub async fn create_story_from_todo(&self, todo: &TodoItem) -> Result<StoryId> {
        let story_id = StoryId(Uuid::new_v4());
        
        // Create story with todo context
        let story = Story {
            id: story_id,
            story_type: crate::story::types::StoryType::Task {
                task_id: todo.id.clone(),
                parent_story: None,
            },
            title: format!("Story: {}", todo.title),
            description: todo.description.clone().unwrap_or_default(),
            summary: todo.description.clone().unwrap_or_default(),
            status: crate::story::types::StoryStatus::Active,
            arcs: vec![],
            current_arc: None,
            segments: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: crate::story::types::StoryMetadata::default(),
            context_chain: crate::story::types::ChainId::new(),
            context: HashMap::new(),
        };
        
        // Register story in engine
        self.story_engine.create_story(
            story.story_type.clone(),
            story.title.clone(),
            story.summary.clone(),
            story.metadata.tags.clone(),
            story.metadata.priority,
        ).await?;
        
        // Create initial plot point from todo
        let plot_point = PlotPoint {
            id: PlotPointId(Uuid::new_v4()),
            title: todo.title.clone(),
            description: todo.description.clone().unwrap_or_default(),
            sequence_number: 0,
            plot_type: PlotType::Task {
                description: todo.title.clone(),
                completed: todo.status == TodoStatus::Completed,
            },
            status: crate::story::types::PlotPointStatus::Pending,
            timestamp: Utc::now(),
            estimated_duration: None,
            actual_duration: None,
            context_tokens: vec![],
            importance: todo.priority.score as f32,
            metadata: crate::story::types::PlotMetadata::default(),
            tags: vec![],
            consequences: vec![],
        };
        
        self.story_engine.add_plot_point(
            story_id, 
            plot_point.plot_type.clone(),
            plot_point.context_tokens.clone()
        ).await?;
        
        // Update mappings
        let mut mappings = self.story_mappings.write().await;
        mappings.todo_to_story.insert(todo.id.clone(), story_id);
        mappings.story_to_todos.entry(story_id).or_default().push(todo.id.clone());
        mappings.plot_to_todo.insert(plot_point.id, todo.id.clone());
        
        // Update state
        self.state.write().await.stories_created_from_todos += 1;
        
        // Broadcast event
        self.broadcast_story_event(StoryEvent::StoryCreated {
            story_id,
            title: story.title,
            description: story.description,
        }).await?;
        
        info!("Created story {} from todo {}", story_id.0, todo.id);
        Ok(story_id)
    }
    
    /// Create todos from a story
    pub async fn create_todos_from_story(&self, story_id: StoryId) -> Result<Vec<String>> {
        if self.todo_manager.is_none() {
            return Ok(Vec::new());
        }
        
        let story = self.story_engine.get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
        let task_map = self.task_mapper.map_story_to_tasks(&story).await?;
        
        let mut todo_ids = Vec::new();
        let todo_manager = self.todo_manager.as_ref().unwrap();
        
        for task in task_map.tasks {
            let todo_request = CreateTodoRequest {
                title: task.description.clone(),
                description: Some(task.story_context.clone()),
                creator: "story-engine".to_string(),
                assignee: task.assigned_to,
                due_date: None,
                tags: vec!["story-generated".to_string()],
                parent_id: None,
                dependency_ids: Vec::new(),
                priority_hint: Some(0.5),
                story_context: Some(crate::tui::chat::orchestration::todo_manager::TodoStoryContext {
                    story_id: "unknown".to_string(),
                    plot_point_id: Some("unknown".to_string()),
                    narrative: task.story_context,
                    story_arc: None,
                    related_events: Vec::new(),
                }),
                priority: None,
                energy_required: None,
                focus_required: None,
                context: None,
            };
            
            let todo = todo_manager.create_todo(todo_request).await?;
            todo_ids.push(todo.id.clone());
            
            // Update mappings
            let mut mappings = self.story_mappings.write().await;
            mappings.todo_to_story.insert(todo.id.clone(), story_id);
            mappings.story_to_todos.entry(story_id).or_default().push(todo.id.clone());
            
            if let Some(plot_id) = task.plot_point {
                mappings.plot_to_todo.insert(plot_id, todo.id.clone());
            }
            
            // Broadcast mapping event
            if let Some(plot_id) = task.plot_point {
                self.broadcast_story_event(StoryEvent::TodoMapped {
                    todo_id: todo.id.clone(),
                    story_id,
                    plot_point_id: plot_id,
                }).await?;
            }
        }
        
        // Update state
        self.state.write().await.todos_created_from_stories += todo_ids.len();
        
        info!("Created {} todos from story {}", todo_ids.len(), story_id.0);
        Ok(todo_ids)
    }
    
    /// Extract story context from a message
    pub async fn extract_story_context(&self, message: &str, chat_id: &str) -> Result<Option<StoryContext>> {
        let mappings = self.story_mappings.read().await;
        
        // Check if chat has an associated story
        if let Some(story_id) = mappings.chat_to_story.get(chat_id) {
            let story = self.story_engine.get_story(story_id)
                .ok_or_else(|| anyhow::anyhow!("Story not found: {:?}", story_id))?;
            
            // Get recent plot points
            let recent_points: Vec<PlotPoint> = story.arcs.iter()
                .flat_map(|arc| arc.plot_points.clone())
                .take(5)
                .collect();
            
            // Get narrative context
            let narrative = self.generate_narrative_context(&story, &recent_points).await?;
            
            return Ok(Some(StoryContext {
                story_id: *story_id,
                narrative: narrative.clone(),
                recent_plot_points: recent_points,
                active_arc: story.arcs.last().cloned(),
                current_plot: narrative,
                current_arc: story.arcs.last().map(|arc| arc.id),
            }));
        }
        
        // Check if message references a story or should create one
        if self.should_create_story(message) {
            let story_id = self.create_story_from_message(message, chat_id).await?;
            
            let story = self.story_engine.get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
            let narrative = format!("New story began: {}", story.title);
            
            return Ok(Some(StoryContext {
                story_id,
                narrative: narrative.clone(),
                recent_plot_points: Vec::new(),
                active_arc: None,
                current_plot: narrative,
                current_arc: None,
            }));
        }
        
        Ok(None)
    }
    
    /// Generate plot point from user interaction
    pub async fn generate_plot_point(
        &self,
        interaction: &str,
        story_id: StoryId,
        impact: f32,
    ) -> Result<PlotPointId> {
        let plot_type = self.determine_plot_type(interaction);
        
        let plot_point = PlotPoint {
            id: PlotPointId(Uuid::new_v4()),
            title: interaction.split_whitespace().take(5).collect::<Vec<_>>().join(" "),
            description: interaction.to_string(),
            sequence_number: 0,
            plot_type,
            status: crate::story::types::PlotPointStatus::Pending,
            timestamp: Utc::now(),
            estimated_duration: None,
            actual_duration: None,
            context_tokens: vec![],
            importance: impact,
            metadata: crate::story::types::PlotMetadata::default(),
            tags: vec![],
            consequences: vec![],
        };
        
        self.story_engine.add_plot_point(
            story_id, 
            plot_point.plot_type.clone(),
            plot_point.context_tokens.clone()
        ).await?;
        
        // Broadcast event
        self.broadcast_story_event(StoryEvent::PlotPointAdded {
            story_id,
            plot_point: PlotPointSummary {
                id: plot_point.id,
                description: plot_point.description,
                plot_type: format!("{:?}", plot_point.plot_type),
                timestamp: plot_point.timestamp,
            },
        }).await?;
        
        Ok(plot_point.id)
    }
    
    /// Handle story engine events
    async fn handle_story_event(&self, event: StoryEngineEvent) -> Result<()> {
        self.state.write().await.events_processed += 1;
        
        match event {
            StoryEngineEvent::StoryCreated(story_id) => {
                debug!("Story created: {}", story_id.0);
                self.state.write().await.active_stories += 1;
            }
            StoryEngineEvent::ArcCompleted(story_id, arc_id) => {
                info!("Story arc completed: {} - {}", story_id.0, arc_id.0);
                
                // Check if any todos should be marked complete
                if let Some(todo_manager) = &self.todo_manager {
                    let mappings = self.story_mappings.read().await;
                    if let Some(todo_ids) = mappings.story_to_todos.get(&story_id) {
                        for todo_id in todo_ids {
                            // Check todo status and update if needed
                            debug!("Checking todo {} for arc completion", todo_id);
                        }
                    }
                }
            }
            StoryEngineEvent::PlotPointAdded(story_id, plot_id) => {
                debug!("Plot point added to story {}: {}", story_id.0, plot_id.0);
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Sync stories and todos
    async fn sync_stories_and_todos(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.last_sync = Utc::now();
        
        // Sync todos to stories
        if let Some(todo_manager) = &self.todo_manager {
            // This would need a method to get all todos
            // For now, we'll skip the implementation
            debug!("Syncing todos to stories");
        }
        
        // Sync stories to todos
        for story_ref in self.story_engine.stories.iter() {
            let story_id = *story_ref.key();
            let story = story_ref.value();
            
            // Check if story needs todo generation
            if self.should_generate_todos(&story) {
                self.create_todos_from_story(story_id).await?;
            }
        }
        
        Ok(())
    }
    
    /// Broadcast story event
    async fn broadcast_story_event(&self, event: StoryEvent) -> Result<()> {
        let system_event = SystemEvent::CustomEvent {
            source: TabId::Chat,
            name: "story_event".to_string(),
            data: serde_json::to_value(event)?,
            target: None,
        };
        
        self.event_bus.publish(system_event).await?;
        Ok(())
    }
    
    /// Generate narrative context
    async fn generate_narrative_context(
        &self,
        story: &Story,
        recent_points: &[PlotPoint],
    ) -> Result<String> {
        let mut narrative = format!("{}\n\n", story.description);
        
        if !recent_points.is_empty() {
            narrative.push_str("Recent events:\n");
            for point in recent_points {
                narrative.push_str(&format!("• {}\n", point.description));
            }
        }
        
        Ok(narrative)
    }
    
    /// Check if message should create a story
    fn should_create_story(&self, message: &str) -> bool {
        let keywords = ["let's", "we need to", "project", "epic", "story", "journey"];
        keywords.iter().any(|k| message.to_lowercase().contains(k))
    }
    
    /// Create story from message
    async fn create_story_from_message(&self, message: &str, chat_id: &str) -> Result<StoryId> {
        let story_id = StoryId(Uuid::new_v4());
        
        let story = Story {
            id: story_id,
            story_type: crate::story::types::StoryType::Task {
                task_id: format!("chat-{}", chat_id),
                parent_story: None,
            },
            title: self.extract_title(message),
            description: message.to_string(),
            summary: message.to_string(),
            status: crate::story::types::StoryStatus::Active,
            arcs: vec![],
            current_arc: None,
            segments: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: crate::story::types::StoryMetadata::default(),
            context_chain: crate::story::types::ChainId::new(),
            context: HashMap::new(),
        };
        
        self.story_engine.create_story(
            story.story_type,
            story.title,
            story.summary,
            story.metadata.tags.clone(),
            story.metadata.priority,
        ).await?;
        
        // Map chat to story
        self.story_mappings.write().await
            .chat_to_story.insert(chat_id.to_string(), story_id);
        
        Ok(story_id)
    }
    
    /// Extract title from message
    fn extract_title(&self, message: &str) -> String {
        let words: Vec<&str> = message.split_whitespace().collect();
        let title = words.iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        
        if title.len() > 50 {
            format!("{}...", &title[..47])
        } else {
            title
        }
    }
    
    /// Determine plot type from interaction
    fn determine_plot_type(&self, interaction: &str) -> PlotType {
        let lower = interaction.to_lowercase();
        
        if lower.contains("error") || lower.contains("bug") || lower.contains("issue") {
            PlotType::Issue {
                error: interaction.to_string(),
                resolved: false,
            }
        } else if lower.contains("goal") || lower.contains("objective") {
            PlotType::Goal {
                objective: interaction.to_string(),
            }
        } else if lower.contains("task") || lower.contains("todo") {
            PlotType::Task {
                description: interaction.to_string(),
                completed: false,
            }
        } else if lower.contains("decide") || lower.contains("choice") {
            PlotType::Decision {
                question: interaction.to_string(),
                choice: String::new(),
            }
        } else {
            PlotType::Context {
                context_type: "interaction".to_string(),
                data: interaction.to_string(),
            }
        }
    }
    
    /// Check if story should generate todos
    fn should_generate_todos(&self, story: &Story) -> bool {
        // Check if story has unmapped plot points
        story.arcs.iter()
            .any(|arc| arc.plot_points.iter()
                .any(|p| matches!(p.plot_type, PlotType::Task { completed: false, .. })))
    }
}

/// Story context for message processing
impl Clone for StoryBridge {
    fn clone(&self) -> Self {
        Self {
            story_engine: self.story_engine.clone(),
            todo_manager: self.todo_manager.clone(),
            task_mapper: self.task_mapper.clone(),
            agent_coordinator: self.agent_coordinator.clone(),
            event_bus: self.event_bus.clone(),
            story_mappings: self.story_mappings.clone(),
            story_event_rx: None, // Don't clone receiver
            cognitive_tx: self.cognitive_tx.clone(),
            config: self.config.clone(),
            state: self.state.clone(),
        }
    }
}

impl StoryBridge {
    /// Map agent to story
    pub async fn map_agent_to_story(
        &self,
        agent_id: String,
        story_id: StoryId,
    ) -> Result<()> {
        let mut mappings = self.story_mappings.write().await;
        
        // Check if agent already has a story
        if let Some(existing_story) = mappings.agent_to_story.get(&agent_id) {
            if existing_story != &story_id {
                warn!("Agent {} already mapped to story {:?}, updating to {:?}", 
                    agent_id, existing_story, story_id);
            }
        }
        
        mappings.agent_to_story.insert(agent_id.clone(), story_id);
        
        info!("Mapped agent {} to story {:?}", agent_id, story_id);
        
        // Notify about the mapping
        self.broadcast_story_event(StoryEvent::TodoMapped {
            todo_id: agent_id.clone(),
            story_id,
            plot_point_id: PlotPointId::new(),
        }).await;
        
        // If agent coordinator is available, register the agent
        if let Some(coordinator) = &self.agent_coordinator {
            debug!("Registering agent {} with story coordinator", agent_id);
            // This would call actual coordinator methods
        }
        
        Ok(())
    }
    
    /// Get story for agent
    pub async fn get_story_for_agent(&self, agent_id: &str) -> Option<StoryId> {
        let mappings = self.story_mappings.read().await;
        mappings.agent_to_story.get(agent_id).cloned()
    }
    
    /// Get all agents for a story
    pub async fn get_agents_for_story(&self, story_id: StoryId) -> Vec<String> {
        let mappings = self.story_mappings.read().await;
        mappings.agent_to_story
            .iter()
            .filter(|(_, sid)| **sid == story_id)
            .map(|(aid, _)| aid.clone())
            .collect()
    }
    
    /// Unmap agent from story
    pub async fn unmap_agent_from_story(&self, agent_id: &str) -> Result<()> {
        let mut mappings = self.story_mappings.write().await;
        if let Some(story_id) = mappings.agent_to_story.remove(agent_id) {
            info!("Unmapped agent {} from story {:?}", agent_id, story_id);
        }
        Ok(())
    }
}

impl StoryBridge {
    /// Map a todo item to a story
    pub async fn map_todo_to_story(&self, todo_id: String, description: String) -> Result<()> {
        // Create or find appropriate story for this todo
        let story_id = if let Some(existing_story) = self.find_story_for_todo(&description).await? {
            existing_story
        } else {
            // Create new story for this todo
            let story_id = self.story_engine.create_story(
                crate::story::types::StoryType::Task {
                    task_id: todo_id.clone(),
                    parent_story: None,
                },
                format!("Task: {}", description),
                description.clone(),
                vec!["task".to_string()],
                crate::story::types::Priority::Medium
            ).await?;
            story_id
        };
        
        // Update mappings
        let mut mappings = self.story_mappings.write().await;
        mappings.todo_to_story.insert(todo_id.clone(), story_id);
        mappings.story_to_todos.entry(story_id).or_insert_with(Vec::new).push(todo_id);
        
        info!("Mapped todo to story: {:?}", story_id);
        Ok(())
    }
    
    /// Check if a request is story-driven
    pub async fn is_story_driven_request(&self, content: &str) -> bool {
        // Check for story-related keywords and patterns
        let story_keywords = [
            "story", "narrative", "plot", "chapter", "arc", 
            "journey", "saga", "epic", "quest", "adventure"
        ];
        
        let content_lower = content.to_lowercase();
        
        // Check for story keywords
        if story_keywords.iter().any(|keyword| content_lower.contains(keyword)) {
            return true;
        }
        
        // Check if there's an active story that matches this content
        if self.find_story_for_todo(content).await.is_ok_and(|s| s.is_some()) {
            return true;
        }
        
        // Check configuration for auto-story creation
        self.config.auto_create_stories
    }
    
    /// Create a story stream for processing
    pub async fn create_story_stream(&self, content: &str) -> Result<Box<dyn futures::Stream<Item = crate::tui::chat::processing::unified_streaming::StreamPacket> + Send + Sync + Unpin>> {
        use futures::stream;
        use crate::tui::chat::processing::unified_streaming::{StreamPacket, StreamContent, StreamEventData};
        
        // Create or find story
        let story_id = if let Some(story_id) = self.find_story_for_todo(content).await? {
            story_id
        } else {
            self.story_engine.create_story(
                crate::story::types::StoryType::Task {
                    task_id: Uuid::new_v4().to_string(),
                    parent_story: None,
                },
                format!("Story: {}", content),
                content.to_string(),
                vec!["stream".to_string()],
                crate::story::types::Priority::Medium
            ).await?
        };
        
        let story = self.story_engine.get_story(&story_id)
            .ok_or_else(|| anyhow::anyhow!("Story not found"))?;
        
        // Create stream of story events
        let packets = vec![
            StreamPacket {
                id: crate::tui::chat::processing::unified_streaming::StreamId(Uuid::new_v4()),
                source: crate::tui::chat::processing::unified_streaming::StreamSource::Story(
                    crate::tui::chat::processing::unified_streaming::StorySource::Events
                ),
                timestamp: Utc::now(),
                sequence: 1,
                content: StreamContent::Event(StreamEventData {
                    event_type: "story_start".to_string(),
                    data: serde_json::json!({
                        "story_id": story_id.0,
                        "title": story.title,
                        "message": format!("Starting story: {}", story.title)
                    }),
                }),
                metadata: HashMap::new(),
            },
            StreamPacket {
                id: crate::tui::chat::processing::unified_streaming::StreamId(Uuid::new_v4()),
                source: crate::tui::chat::processing::unified_streaming::StreamSource::Story(
                    crate::tui::chat::processing::unified_streaming::StorySource::Events
                ),
                timestamp: Utc::now(),
                sequence: 2,
                content: StreamContent::Event(StreamEventData {
                    event_type: "progress".to_string(),
                    data: serde_json::json!({
                        "phase": "story_init",
                        "progress": 0.1,
                        "message": format!("Initializing story narrative for: {}", content)
                    }),
                }),
                metadata: HashMap::new(),
            },
        ];
        
        Ok(Box::new(stream::iter(packets)) as Box<dyn futures::Stream<Item = _> + Send + Sync + Unpin>)
    }
    
    /// Find an existing story for a todo description
    async fn find_story_for_todo(&self, description: &str) -> Result<Option<StoryId>> {
        // Search through active stories for matching context
        let stories: Vec<Story> = self.story_engine.stories
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        for story in stories {
            // Check if story description or summary matches
            if story.description.contains(description) || 
               story.summary.contains(description) ||
               description.contains(&story.title) {
                return Ok(Some(story.id));
            }
        }
        
        Ok(None)
    }
}

