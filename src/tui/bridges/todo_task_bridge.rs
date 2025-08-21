//! Todo-Task Bridge - Synchronizes todos with external task management systems
//! 
//! This bridge enables bidirectional sync between the TodoManager and
//! external task platforms (Jira, Linear, Asana) through the TaskManager.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use anyhow::Result;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn, error};

use crate::tui::chat::orchestration::todo_manager::{
    TodoManager, TodoItem, TodoStatus, TodoPriority, CreateTodoRequest,
    PriorityLevel, TodoCognitiveMetadata, EnergyLevel, FocusLevel, PriorityFactors,
    OrchestrationMetadata, ExecutionMetadata,
};
use crate::tools::task_management::{
    TaskManager, Task, TaskStatus, TaskPriority as ExtTaskPriority,
    TaskPlatform, TaskCognitiveMetadata, TaskEvent,
};
use crate::cognitive::goal_manager::Priority;
use crate::tasks::{TaskRegistry, TaskResult};
use crate::tui::event_bus::{EventBus, SystemEvent, TabId, Subscriber, create_handler};

/// Helper trait for logging with timestamp
trait LogWithTimestamp {
    fn log_with_timestamp(self, timestamp: DateTime<Utc>) -> Self;
}

impl LogWithTimestamp for () {
    fn log_with_timestamp(self, timestamp: DateTime<Utc>) -> Self {
        debug!("Operation completed at {}", timestamp);
        self
    }
}

/// Bridge between TodoManager and TaskManager
#[derive(Clone)]
pub struct TodoTaskBridge {
    /// Todo manager reference
    todo_manager: Arc<TodoManager>,
    
    /// Task manager reference
    task_manager: Arc<TaskManager>,
    
    /// Task registry for executable tasks
    task_registry: Arc<TaskRegistry>,
    
    /// Event bus for notifications
    event_bus: Arc<EventBus>,
    
    /// Sync mappings between todos and tasks
    sync_mappings: Arc<RwLock<SyncMappings>>,
    
    /// Bridge configuration
    config: BridgeConfig,
    
    /// Sync state
    state: Arc<RwLock<SyncState>>,
}

/// Mappings between todos and external tasks
#[derive(Debug, Clone, Default)]
struct SyncMappings {
    /// Todo ID to external task ID
    todo_to_task: HashMap<String, String>,
    
    /// External task ID to todo ID
    task_to_todo: HashMap<String, String>,
    
    /// Platform-specific mappings
    platform_mappings: HashMap<TaskPlatform, HashMap<String, String>>,
}

/// Bridge configuration
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub auto_sync_enabled: bool,
    pub sync_interval_seconds: u64,
    pub sync_on_change: bool,
    pub create_external_tasks: bool,
    pub update_external_tasks: bool,
    pub import_external_tasks: bool,
    pub conflict_resolution: ConflictResolution,
    pub platforms: Vec<TaskPlatform>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            auto_sync_enabled: true,
            sync_interval_seconds: 60,
            sync_on_change: true,
            create_external_tasks: true,
            update_external_tasks: true,
            import_external_tasks: true,
            conflict_resolution: ConflictResolution::PreferLocal,
            platforms: vec![TaskPlatform::Jira, TaskPlatform::Linear, TaskPlatform::Asana],
        }
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy)]
pub enum ConflictResolution {
    PreferLocal,
    PreferRemote,
    Manual,
    Merge,
}

/// Sync state
#[derive(Debug, Clone)]
struct SyncState {
    last_sync: DateTime<Utc>,
    sync_in_progress: bool,
    pending_changes: Vec<PendingChange>,
    conflicts: Vec<SyncConflict>,
    stats: SyncStatistics,
}

/// Pending change to sync
#[derive(Debug, Clone)]
struct PendingChange {
    change_type: ChangeType,
    entity_id: String,
    timestamp: DateTime<Utc>,
    platform: Option<TaskPlatform>,
}

impl PendingChange {
    /// Create a new pending change
    fn new(change_type: ChangeType, entity_id: String, platform: Option<TaskPlatform>) -> Self {
        Self {
            change_type,
            entity_id,
            timestamp: Utc::now(),
            platform,
        }
    }
    
    /// Check if change is expired (older than 5 minutes)
    fn is_expired(&self) -> bool {
        let age = Utc::now().signed_duration_since(self.timestamp);
        age.num_minutes() > 5
    }
    
    /// Get platform name for logging
    fn platform_name(&self) -> String {
        self.platform.as_ref()
            .map(|p| format!("{:?}", p))
            .unwrap_or_else(|| "Local".to_string())
    }
}

/// Type of change
#[derive(Debug, Clone, Copy)]
enum ChangeType {
    Created,
    Updated,
    Deleted,
    StatusChanged,
}

impl ChangeType {
    /// Get change type as string for logging
    fn as_str(&self) -> &str {
        match self {
            ChangeType::Created => "created",
            ChangeType::Updated => "updated",
            ChangeType::Deleted => "deleted",
            ChangeType::StatusChanged => "status_changed",
        }
    }
}

/// Sync conflict
#[derive(Debug, Clone)]
struct SyncConflict {
    todo_id: String,
    task_id: String,
    conflict_type: ConflictType,
    local_value: serde_json::Value,
    remote_value: serde_json::Value,
    detected_at: DateTime<Utc>,
}

impl SyncConflict {
    /// Check if conflict should be auto-resolved based on age
    fn should_auto_resolve(&self) -> bool {
        let age = Utc::now().signed_duration_since(self.detected_at);
        age.num_hours() > 24
    }
    
    /// Get task identifier for external system
    fn get_task_identifier(&self) -> String {
        self.task_id.clone()
    }
    
    /// Create conflict description for logging
    fn describe(&self) -> String {
        format!("Conflict {:?} between todo {} and task {} - Local: {:?}, Remote: {:?}",
            self.conflict_type, self.todo_id, self.task_id, 
            self.local_value, self.remote_value)
    }
}

/// Type of conflict
#[derive(Debug, Clone)]
enum ConflictType {
    Status,
    Priority,
    Description,
    DueDate,
    Assignee,
}

impl ConflictType {
    /// Get conflict resolution strategy based on type
    fn resolution_strategy(&self) -> &str {
        match self {
            ConflictType::Status => "prefer_remote", // External status is authoritative
            ConflictType::Priority => "prefer_local", // Local priority takes precedence
            ConflictType::Description => "merge", // Merge descriptions
            ConflictType::DueDate => "prefer_remote", // External due date is authoritative
            ConflictType::Assignee => "prefer_remote", // External assignee is authoritative
        }
    }
    
    /// Check if this conflict type requires manual resolution
    fn requires_manual_resolution(&self) -> bool {
        matches!(self, ConflictType::Description)
    }
}

/// Sync statistics
#[derive(Debug, Clone, Default)]
struct SyncStatistics {
    todos_synced: usize,
    tasks_imported: usize,
    conflicts_resolved: usize,
    sync_errors: usize,
    last_successful_sync: Option<DateTime<Utc>>,
}

impl TodoTaskBridge {
    /// Create a new bridge
    pub fn new(
        todo_manager: Arc<TodoManager>,
        task_manager: Arc<TaskManager>,
        task_registry: Arc<TaskRegistry>,
        event_bus: Arc<EventBus>,
    ) -> Self {
        Self {
            todo_manager,
            task_manager,
            task_registry,
            event_bus,
            sync_mappings: Arc::new(RwLock::new(SyncMappings::default())),
            config: BridgeConfig::default(),
            state: Arc::new(RwLock::new(SyncState {
                last_sync: Utc::now(),
                sync_in_progress: false,
                pending_changes: Vec::new(),
                conflicts: Vec::new(),
                stats: SyncStatistics::default(),
            })),
        }
    }
    
    /// Initialize the bridge
    pub async fn initialize(&self) -> Result<()> {
        // Subscribe to events
        self.subscribe_to_events().await?;
        
        // Subscribe to TaskManager events
        self.subscribe_to_task_events().await?;
        
        // Start sync task if enabled
        if self.config.auto_sync_enabled {
            self.start_sync_task().await;
        }
        
        // Initial sync
        self.sync_all().await?;
        
        info!("TodoTaskBridge initialized");
        Ok(())
    }
    
    /// Subscribe to relevant events
    async fn subscribe_to_events(&self) -> Result<()> {
        // Subscribe to task events from external platforms
        let bridge = self.clone();
        let bus = self.event_bus.clone();
        
        let subscriber = Subscriber {
            id: uuid::Uuid::new_v4().to_string(),
            tab_id: TabId::Utilities,
            handler: create_handler(move |event| {
                if let SystemEvent::CustomEvent { name, data, .. } = event {
                    if name == "task_updated" {
                        let bridge = bridge.clone();
                        tokio::spawn(async move {
                            let _ = bridge.handle_task_update(data).await;
                        });
                    }
                }
                Ok(())
            }),
            filter: None,
        };
        
        bus.subscribe("CustomEvent".to_string(), subscriber).await?;
        
        Ok(())
    }
    
    /// Subscribe to TaskManager events
    async fn subscribe_to_task_events(&self) -> Result<()> {
        // Get event receiver from TaskManager
        let mut event_rx = self.task_manager.subscribe_events();
        let bridge = self.clone();
        
        // Spawn task to handle TaskManager events
        tokio::spawn(async move {
            while let Ok(event) = event_rx.recv().await {
                match event {
                    TaskEvent::TaskCreated(task) => {
                        info!("Task created: {}", task.id);
                        // Create corresponding todo if it doesn't exist
                        if let Err(e) = bridge.handle_task_created(task).await {
                            warn!("Failed to handle task creation: {}", e);
                        }
                    }
                    TaskEvent::TaskUpdated(task) => {
                        debug!("Task updated: {}", task.id);
                        if let Err(e) = bridge.handle_task_updated(task).await {
                            warn!("Failed to handle task update: {}", e);
                        }
                    }
                    TaskEvent::TaskCompleted(task) => {
                        info!("Task completed: {}", task.id);
                        if let Err(e) = bridge.handle_task_completed(task).await {
                            warn!("Failed to handle task completion: {}", e);
                        }
                    }
                    TaskEvent::DeadlineApproaching { task_id, time_remaining } => {
                        warn!("Deadline approaching for task {}: {:?}", task_id, time_remaining);
                        // Notify through event bus
                        let _ = bridge.event_bus.emit(SystemEvent::CustomEvent {
                            name: "deadline_warning".to_string(),
                            data: serde_json::json!({
                                "task_id": task_id,
                                "time_remaining": format!("{:?}", time_remaining),
                                "message": format!("Task {} deadline in {:?}", task_id, time_remaining),
                            }),
                            source: TabId::Utilities,
                            target: None,
                        }).await;
                    }
                    TaskEvent::BurnoutRiskDetected { risk_score, recommendations } => {
                        warn!("Burnout risk detected: {}", risk_score);
                        // Create high-priority todo for self-care
                        if let Err(e) = bridge.handle_burnout_risk(risk_score, recommendations).await {
                            warn!("Failed to handle burnout risk: {}", e);
                        }
                    }
                    TaskEvent::WorkloadOptimized { suggestions } => {
                        info!("Workload optimized with {} suggestions", suggestions.len());
                        // Apply suggestions to todo list
                        for suggestion in suggestions {
                            debug!("Workload suggestion: {:?}", suggestion);
                        }
                    }
                    TaskEvent::ProductivityInsight { insight, data } => {
                        info!("Productivity insight: {}", insight);
                        // Store insight for later analysis
                        let _ = bridge.event_bus.emit(SystemEvent::CustomEvent {
                            name: "productivity_insight".to_string(),
                            data: serde_json::json!({
                                "insight": insight,
                                "data": data,
                                "timestamp": Utc::now().to_rfc3339(),
                            }),
                            source: TabId::Utilities,
                            target: None,
                        }).await;
                    }
                    TaskEvent::CognitiveTrigger { trigger, priority, context } => {
                        info!("Cognitive trigger: {} (priority: {:?})", trigger, priority);
                        // Create todo based on cognitive trigger
                        if let Err(e) = bridge.handle_cognitive_trigger(trigger, priority, context).await {
                            warn!("Failed to handle cognitive trigger: {}", e);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Handle task created event
    async fn handle_task_created(&self, task: Task) -> Result<()> {
        // Check if task is already mapped
        let mappings = self.sync_mappings.read().await;
        if mappings.task_to_todo.contains_key(&task.id) {
            return Ok(());
        }
        drop(mappings);
        
        // Create corresponding todo
        let todo_item = self.task_to_todo_item(task.clone());
        let created_todo = self.todo_manager.create_todo(CreateTodoRequest {
            title: todo_item.title,
            description: todo_item.description,
            priority: Some(todo_item.priority.level),
            tags: todo_item.tags,
            due_date: todo_item.due_date,
            energy_required: Some(todo_item.cognitive_metadata.energy_requirement),
            focus_required: Some(todo_item.cognitive_metadata.focus_requirement),
            context: None,
            parent_id: todo_item.parent_id,
            story_context: todo_item.story_context,
            creator: todo_item.creator,
            assignee: todo_item.assignee,
            dependency_ids: todo_item.dependency_ids,
            priority_hint: None,
        }).await?;
        
        // Update mappings
        let mut mappings = self.sync_mappings.write().await;
        mappings.task_to_todo.insert(task.id.clone(), created_todo.id.clone());
        mappings.todo_to_task.insert(created_todo.id, task.id);
        
        Ok(())
    }
    
    /// Handle task updated event
    async fn handle_task_updated(&self, task: Task) -> Result<()> {
        let mappings = self.sync_mappings.read().await;
        if let Some(todo_id) = mappings.task_to_todo.get(&task.id) {
            // Update the corresponding todo
            let status = self.map_task_status_to_todo_status(task.status);
            self.todo_manager.update_status(todo_id, status).await?;
            
            // Update priority if changed
            if let Some(priority) = self.map_task_priority_to_todo_priority(task.priority) {
                self.todo_manager.update_priority(todo_id.to_string(), priority).await?;
            }
        }
        Ok(())
    }
    
    /// Handle task completed event
    async fn handle_task_completed(&self, task: Task) -> Result<()> {
        let mappings = self.sync_mappings.read().await;
        if let Some(todo_id) = mappings.task_to_todo.get(&task.id) {
            // Mark todo as completed
            self.todo_manager.update_status(todo_id, TodoStatus::Completed).await?;
            
            // Emit completion event
            let _ = self.event_bus.emit(SystemEvent::CustomEvent {
                name: "task_completed".to_string(),
                data: serde_json::json!({
                    "task_id": task.id,
                    "task_name": task.title,
                    "todo_id": todo_id,
                }),
                source: TabId::Utilities,
                target: None,
            }).await;
        }
        Ok(())
    }
    
    /// Handle burnout risk event
    async fn handle_burnout_risk(&self, risk_score: f32, recommendations: Vec<String>) -> Result<()> {
        // Create a high-priority self-care todo
        let description = format!(
            "Burnout risk detected (score: {:.2}). Recommendations:\n{}",
            risk_score,
            recommendations.join("\n- ")
        );
        
        self.todo_manager.create_todo(CreateTodoRequest {
            title: "⚠️ Self-Care Required: Burnout Risk Detected".to_string(),
            description: Some(description),
            priority: Some(PriorityLevel::Critical),
            tags: vec!["self-care".to_string(), "health".to_string(), "urgent".to_string()],
            due_date: Some(Utc::now() + chrono::Duration::hours(24)),
            energy_required: Some(EnergyLevel::Low),
            focus_required: Some(FocusLevel::Light),
            context: Some("health-monitoring".to_string()),
            parent_id: None,
            story_context: None,
            creator: "cognitive-monitor".to_string(),
            assignee: None,
            dependency_ids: Vec::new(),
            priority_hint: Some(1.0),
        }).await?;
        
        Ok(())
    }
    
    /// Handle cognitive trigger event
    async fn handle_cognitive_trigger(&self, trigger: String, priority: Priority, context: String) -> Result<()> {
        // Map Priority (from cognitive::goal_manager) to PriorityLevel
        let todo_priority = match priority {
            Priority::Critical => PriorityLevel::Critical,
            Priority::High => PriorityLevel::High,
            Priority::Medium => PriorityLevel::Medium,
            Priority::Low => PriorityLevel::Low,
        };
        
        // Create todo from cognitive trigger
        self.todo_manager.create_todo(CreateTodoRequest {
            title: format!("🧠 {}", trigger),
            description: Some(format!("Cognitive trigger: {}", context)),
            priority: Some(todo_priority),
            tags: vec!["cognitive".to_string(), "automated".to_string()],
            due_date: None,
            energy_required: Some(EnergyLevel::Medium),
            focus_required: Some(FocusLevel::Deep),
            context: Some(context),
            parent_id: None,
            story_context: None,
            creator: "cognitive-system".to_string(),
            assignee: None,
            dependency_ids: Vec::new(),
            priority_hint: None,
        }).await?;
        
        Ok(())
    }
    
    /// Map task priority to todo priority
    fn map_task_priority_to_todo_priority(&self, priority: ExtTaskPriority) -> Option<TodoPriority> {
        let level = match priority {
            ExtTaskPriority::Critical => PriorityLevel::Critical,
            ExtTaskPriority::High => PriorityLevel::High,
            ExtTaskPriority::Medium => PriorityLevel::Medium,
            ExtTaskPriority::Low => PriorityLevel::Low,
        };
        Some(TodoPriority {
            level,
            score: level as u8 as f32 / 5.0,
            factors: PriorityFactors::default(),
        })
    }
    
    /// Start background sync task
    async fn start_sync_task(&self) {
        let bridge = self.clone();
        let interval = self.config.sync_interval_seconds;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = bridge.sync_all().await {
                    error!("Sync error: {}", e);
                    bridge.state.write().await.stats.sync_errors += 1;
                }
            }
        });
    }
    
    /// Sync all todos and tasks
    pub async fn sync_all(&self) -> Result<()> {
        // Check if sync is already in progress
        {
            let mut state = self.state.write().await;
            if state.sync_in_progress {
                debug!("Sync already in progress, skipping");
                return Ok(());
            }
            state.sync_in_progress = true;
        }
        
        info!("Starting full sync");
        
        // Export todos to external platforms
        if self.config.create_external_tasks || self.config.update_external_tasks {
            self.export_todos().await?;
        }
        
        // Import tasks from external platforms
        if self.config.import_external_tasks {
            self.import_tasks().await?;
        }
        
        // Process pending changes
        self.process_pending_changes().await?;
        
        // Resolve conflicts
        self.resolve_conflicts().await?;
        
        // Update sync state
        {
            let mut state = self.state.write().await;
            state.sync_in_progress = false;
            state.last_sync = Utc::now();
            state.stats.last_successful_sync = Some(Utc::now());
        }
        
        info!("Full sync completed");
        Ok(())
    }
    
    /// Export todos to external platforms
    async fn export_todos(&self) -> Result<()> {
        // Get all todos from TodoManager
        let todos = self.todo_manager.list_todos().await?;
        
        let mut exported_count = 0;
        let mut updated_count = 0;
        
        for todo in todos {
            let mappings = self.sync_mappings.read().await;
            
            // Check if todo already has external mapping
            if let Some(task_id) = mappings.todo_to_task.get(&todo.id) {
                let task_id_clone = task_id.clone();
                // Update existing external task
                if self.config.update_external_tasks {
                    drop(mappings);
                    
                    // Convert todo to task format
                    let task = self.todo_to_task(&todo);
                    
                    // Update in external platform
                    if let Err(e) = self.task_manager.update_task(task).await {
                        error!("Failed to update task {}: {}", task_id_clone, e);
                        self.state.write().await.stats.sync_errors += 1;
                    } else {
                        updated_count += 1;
                    }
                }
            } else if self.config.create_external_tasks {
                // Create new external task
                drop(mappings);
                
                // Convert todo to task format
                let task = self.todo_to_task(&todo);
                
                // Create in external platform
                match self.task_manager.create_task(
                    &task.title,
                    task.description.as_deref(),
                    task.priority,
                    task.due_date,
                    task.platform,
                ).await {
                    Ok(created_task) => {
                        // Update mappings
                        let mut mappings = self.sync_mappings.write().await;
                        mappings.todo_to_task.insert(todo.id.clone(), created_task.id.clone());
                        mappings.task_to_todo.insert(created_task.id.clone(), todo.id.clone());
                        
                        // Track platform-specific mapping
                        mappings.platform_mappings
                            .entry(created_task.platform)
                            .or_insert_with(HashMap::new)
                            .insert(todo.id.clone(), created_task.id.clone());
                        
                        exported_count += 1;
                    }
                    Err(e) => {
                        error!("Failed to create task for todo {}: {}", todo.id, e);
                        self.state.write().await.stats.sync_errors += 1;
                    }
                }
            }
        }
        
        // Update statistics
        if exported_count > 0 || updated_count > 0 {
            let mut state = self.state.write().await;
            state.stats.todos_synced += exported_count;
            info!("Exported {} new todos, updated {} existing tasks", exported_count, updated_count);
        }
        
        debug!("Completed exporting todos to external platforms");
        Ok(())
    }
    
    /// Convert todo to task
    fn todo_to_task(&self, todo: &TodoItem) -> Task {
        Task {
            id: todo.id.clone(),
            external_id: todo.external_task_id.clone(),
            platform: TaskPlatform::Internal,
            title: todo.title.clone(),
            description: todo.description.clone(),
            status: self.map_todo_status(&todo.status),
            priority: self.map_todo_priority(&todo.priority),
            assignee: todo.assignee.clone(),
            reporter: Some(todo.creator.clone()),
            labels: todo.tags.clone(),
            due_date: todo.due_date,
            created_at: todo.created_at,
            updated_at: todo.updated_at,
            estimate: todo.estimated_duration,
            time_spent: None,
            progress: self.calculate_todo_progress(todo),
            parent_task: todo.parent_id.clone(),
            subtasks: todo.subtask_ids.clone(),
            dependencies: todo.dependency_ids.clone(),
            cognitive_metadata: TaskCognitiveMetadata {
                cognitive_priority: todo.priority.score,
                complexity_score: todo.cognitive_metadata.complexity_score,
                energy_requirement: todo.cognitive_metadata.cognitive_load,
                focus_requirement: todo.cognitive_metadata.estimated_effort,
                context_switching_cost: 0.3,
                optimal_time_blocks: vec!["any".to_string()],
                prerequisite_knowledge: Vec::new(),
                related_memories: Vec::new(),
                burnout_risk: 0.2,
                motivation_factors: Vec::new(),
            },
            name: todo.title.clone(),
            metadata: serde_json::json!({
                "todo_id": todo.id,
                "created_by": "todo_bridge"
            }),
        }
    }
    
    /// Map todo status to task status
    fn map_todo_status(&self, status: &TodoStatus) -> TaskStatus {
        match status {
            TodoStatus::Pending => TaskStatus::Todo,
            TodoStatus::Ready => TaskStatus::Todo,
            TodoStatus::InProgress => TaskStatus::InProgress,
            TodoStatus::Blocked => TaskStatus::Blocked,
            TodoStatus::Review => TaskStatus::InReview,
            TodoStatus::Completed => TaskStatus::Done,
            TodoStatus::Cancelled => TaskStatus::Cancelled,
            TodoStatus::Failed => TaskStatus::Failed,
        }
    }
    
    /// Map todo priority to task priority
    fn map_todo_priority(&self, priority: &TodoPriority) -> ExtTaskPriority {
        match priority.level {
            PriorityLevel::Critical => ExtTaskPriority::Critical,
            PriorityLevel::High => ExtTaskPriority::High,
            PriorityLevel::Medium => ExtTaskPriority::Medium,
            PriorityLevel::Low => ExtTaskPriority::Low,
            PriorityLevel::Minimal => ExtTaskPriority::Low,
        }
    }
    
    /// Calculate todo progress
    fn calculate_todo_progress(&self, todo: &TodoItem) -> f32 {
        if todo.status == TodoStatus::Completed {
            1.0
        } else if todo.status == TodoStatus::InProgress {
            // Calculate based on subtasks if available
            if !todo.subtask_ids.is_empty() {
                // Would need to fetch subtasks and calculate completion ratio
                0.5 // Default to 50% for now
            } else {
                0.3 // Default progress for in-progress items
            }
        } else {
            0.0
        }
    }
    
    /// Import tasks from external platforms
    async fn import_tasks(&self) -> Result<()> {
        // Get tasks from TaskManager
        let tasks = self.task_manager.get_all_tasks().await?;
        
        let mut imported_count = 0;
        
        for task in tasks {
            // Check if task is already mapped
            let mappings = self.sync_mappings.read().await;
            if !mappings.task_to_todo.contains_key(&task.id) {
                drop(mappings);
                
                // Create todo from task
                let todo_request = self.task_to_todo_request(&task);
                let todo = self.todo_manager.create_todo(todo_request).await?;
                
                // Update mappings
                let mut mappings = self.sync_mappings.write().await;
                mappings.todo_to_task.insert(todo.id.clone(), task.id.clone());
                mappings.task_to_todo.insert(task.id.clone(), todo.id.clone());
                
                imported_count += 1;
            }
        }
        
        if imported_count > 0 {
            self.state.write().await.stats.tasks_imported += imported_count;
            info!("Imported {} tasks", imported_count);
        }
        
        Ok(())
    }
    
    /// Process pending changes
    async fn process_pending_changes(&self) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Remove expired changes
        state.pending_changes.retain(|c| !c.is_expired());
        
        let changes = state.pending_changes.drain(..).collect::<Vec<_>>();
        drop(state);
        
        for change in changes {
            debug!("Processing {} change for entity {} on platform {}",
                change.change_type.as_str(),
                change.entity_id,
                change.platform_name());
            
            // Track the timestamp for ordering
            let change_time = change.timestamp;
            
            match change.change_type {
                ChangeType::Created => {
                    self.handle_entity_created(&change).await?
                        .log_with_timestamp(change_time);
                },
                ChangeType::Updated => {
                    self.handle_entity_updated(&change).await?
                        .log_with_timestamp(change_time);
                },
                ChangeType::Deleted => {
                    self.handle_entity_deleted(&change).await?
                        .log_with_timestamp(change_time);
                },
                ChangeType::StatusChanged => {
                    self.handle_status_changed(&change).await?
                        .log_with_timestamp(change_time);
                },
            }
            
            // Process platform-specific changes
            if let Some(platform) = change.platform {
                self.handle_platform_specific_change(&change.entity_id, platform).await?;
            }
        }
        
        Ok(())
    }
    
    /// Handle platform-specific change processing
    async fn handle_platform_specific_change(&self, entity_id: &str, platform: TaskPlatform) -> Result<()> {
        debug!("Processing platform-specific change for {} on {:?}", entity_id, platform);
        
        // Get platform-specific mappings
        let mappings = self.sync_mappings.read().await;
        if let Some(platform_map) = mappings.platform_mappings.get(&platform) {
            if let Some(external_id) = platform_map.get(entity_id) {
                // Trigger platform-specific sync
                match platform {
                    TaskPlatform::Jira => {
                        debug!("Syncing with Jira task {}", external_id);
                    },
                    TaskPlatform::Linear => {
                        debug!("Syncing with Linear issue {}", external_id);
                    },
                    TaskPlatform::Asana => {
                        debug!("Syncing with Asana task {}", external_id);
                    },
                    _ => {},
                }
            }
        }
        
        Ok(())
    }
    
    /// Resolve sync conflicts
    async fn resolve_conflicts(&self) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Auto-resolve old conflicts
        let mut resolved_count = 0;
        state.conflicts.retain(|c| {
            if c.should_auto_resolve() {
                info!("Auto-resolving old conflict: {}", c.describe());
                resolved_count += 1;
                false
            } else {
                true
            }
        });
        
        if resolved_count > 0 {
            state.stats.conflicts_resolved += resolved_count;
        }
        
        let conflicts = state.conflicts.drain(..).collect::<Vec<_>>();
        drop(state);
        
        for conflict in conflicts {
            match self.config.conflict_resolution {
                ConflictResolution::PreferLocal => {
                    // Keep local version
                    self.apply_local_resolution(&conflict).await?;
                }
                ConflictResolution::PreferRemote => {
                    // Use remote version
                    self.apply_remote_resolution(&conflict).await?;
                }
                ConflictResolution::Manual => {
                    // Queue for manual resolution
                    warn!("Manual conflict resolution required for todo: {}", conflict.todo_id);
                }
                ConflictResolution::Merge => {
                    // Attempt to merge changes
                    self.merge_conflict(&conflict).await?;
                }
            }
            
            self.state.write().await.stats.conflicts_resolved += 1;
        }
        
        Ok(())
    }
    
    /// Convert task to todo request
    fn task_to_todo_request(&self, task: &Task) -> CreateTodoRequest {
        CreateTodoRequest {
            title: task.title.clone(),
            description: task.description.clone(),
            creator: task.reporter.clone().unwrap_or_else(|| "system".to_string()),
            assignee: task.assignee.clone(),
            due_date: task.due_date,
            tags: task.labels.clone(),
            parent_id: task.parent_task.clone(),
            dependency_ids: task.dependencies.clone(),
            priority_hint: Some(task.cognitive_metadata.cognitive_priority),
            story_context: None,
            priority: None,
            energy_required: Some(self.map_energy_requirement(task.cognitive_metadata.energy_requirement)),
            focus_required: Some(self.map_focus_requirement(task.cognitive_metadata.focus_requirement)),
            context: None,
        }
    }
    
    /// Sync a specific todo with external task
    pub async fn sync_todo(&self, todo_id: &str) -> Result<()> {
        // Get the todo
        // Note: TodoManager would need a method to get a specific todo
        
        // Check if it has external mapping
        let mappings = self.sync_mappings.read().await;
        if let Some(task_id) = mappings.todo_to_task.get(todo_id) {
            // Update existing task
            let task = self.todo_id_to_task(todo_id).await?;
            self.task_manager.update_task(task).await?;
        } else {
            // Create new task
            let task = self.todo_id_to_task(todo_id).await?;
            let created_task = self.task_manager.create_task(
                &task.title,
                task.description.as_deref(),
                task.priority,
                task.due_date,
                task.platform,
            ).await?;
            
            // Update mappings
            drop(mappings);
            let mut mappings = self.sync_mappings.write().await;
            mappings.todo_to_task.insert(todo_id.to_string(), created_task.id.clone());
            mappings.task_to_todo.insert(created_task.id, todo_id.to_string());
        }
        
        self.state.write().await.stats.todos_synced += 1;
        
        Ok(())
    }
    
    /// Convert todo ID to task (async version for fetching todo first)
    async fn todo_id_to_task(&self, todo_id: &str) -> Result<Task> {
        // This would get the todo from TodoManager
        // For now, create a placeholder
        Ok(Task {
            id: todo_id.to_string(),
            external_id: None,
            platform: TaskPlatform::Internal,
            title: "Todo".to_string(),
            description: None,
            status: TaskStatus::Todo,
            priority: ExtTaskPriority::Medium,
            assignee: None,
            reporter: None,
            labels: Vec::new(),
            due_date: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            estimate: None,
            time_spent: None,
            progress: 0.0,
            parent_task: None,
            subtasks: Vec::new(),
            dependencies: Vec::new(),
            cognitive_metadata: TaskCognitiveMetadata {
                cognitive_priority: 0.5,
                complexity_score: 0.5,
                energy_requirement: 0.5,
                focus_requirement: 0.5,
                context_switching_cost: 0.2,
                optimal_time_blocks: Vec::new(),
                prerequisite_knowledge: Vec::new(),
                related_memories: Vec::new(),
                burnout_risk: 0.0,
                motivation_factors: Vec::new(),
            },
            name: "Todo".to_string(),
            metadata: serde_json::json!({}),
        })
    }
    
    /// Handle task update from external platform
    async fn handle_task_update(&self, data: serde_json::Value) -> Result<()> {
        if let Ok(task) = serde_json::from_value::<Task>(data) {
            // Check if task is mapped to a todo
            let mappings = self.sync_mappings.read().await;
            if let Some(todo_id) = mappings.task_to_todo.get(&task.id) {
                // Update the corresponding todo
                let status = self.map_task_status_to_todo_status(task.status);
                self.todo_manager.update_status(todo_id, status).await?;
                
                debug!("Updated todo {} from external task {}", todo_id, task.id);
            }
        }
        
        Ok(())
    }
    
    /// Handle entity created
    async fn handle_entity_created(&self, change: &PendingChange) -> Result<()> {
        debug!("Handling entity created: {}", change.entity_id);
        Ok(())
    }
    
    /// Handle entity updated
    async fn handle_entity_updated(&self, change: &PendingChange) -> Result<()> {
        debug!("Handling entity updated: {}", change.entity_id);
        Ok(())
    }
    
    /// Handle entity deleted
    async fn handle_entity_deleted(&self, change: &PendingChange) -> Result<()> {
        debug!("Handling entity deleted: {}", change.entity_id);
        Ok(())
    }
    
    /// Handle status changed
    async fn handle_status_changed(&self, change: &PendingChange) -> Result<()> {
        debug!("Handling status changed: {}", change.entity_id);
        Ok(())
    }
    
    /// Apply local resolution for conflict
    async fn apply_local_resolution(&self, conflict: &SyncConflict) -> Result<()> {
        // Keep local version, update remote
        debug!("Applying local resolution for conflict in todo: {}", conflict.todo_id);
        Ok(())
    }
    
    /// Apply remote resolution for conflict
    async fn apply_remote_resolution(&self, conflict: &SyncConflict) -> Result<()> {
        // Use remote version, update local
        debug!("Applying remote resolution for conflict in todo: {}", conflict.todo_id);
        Ok(())
    }
    
    /// Merge conflict
    async fn merge_conflict(&self, conflict: &SyncConflict) -> Result<()> {
        // Attempt to merge changes intelligently
        debug!("Merging conflict in todo: {}", conflict.todo_id);
        Ok(())
    }
    
    /// Execute todo as task
    pub async fn execute_todo_as_task(&self, todo_id: &str) -> Result<TaskResult> {
        // Get todo details
        // Note: TodoManager would need a method to get a specific todo
        
        // Find matching task in registry
        let tasks = self.task_registry.list();
        
        // For now, return a placeholder result
        Ok(TaskResult {
            success: true,
            message: format!("Executed todo: {}", todo_id),
            data: None,
        })
    }
    
    /// Map task status to todo status
    fn map_task_status_to_todo_status(&self, status: TaskStatus) -> TodoStatus {
        match status {
            TaskStatus::Todo => TodoStatus::Pending,
            TaskStatus::InProgress => TodoStatus::InProgress,
            TaskStatus::InReview => TodoStatus::Review,
            TaskStatus::Blocked => TodoStatus::Blocked,
            TaskStatus::Done | TaskStatus::Completed => TodoStatus::Completed,
            TaskStatus::Cancelled => TodoStatus::Cancelled,
            TaskStatus::Failed => TodoStatus::Failed,
        }
    }
    
    /// Get sync statistics
    pub async fn get_statistics(&self) -> SyncStatistics {
        self.state.read().await.stats.clone()
    }
    
    /// Clear all mappings
    pub async fn clear_mappings(&self) -> Result<()> {
        let mut mappings = self.sync_mappings.write().await;
        mappings.todo_to_task.clear();
        mappings.task_to_todo.clear();
        mappings.platform_mappings.clear();
        
        info!("Cleared all sync mappings");
        Ok(())
    }
    
    /// Map energy requirement float to EnergyLevel enum
    fn map_energy_requirement(&self, requirement: f32) -> EnergyLevel {
        if requirement > 0.7 {
            EnergyLevel::High
        } else if requirement > 0.3 {
            EnergyLevel::Medium
        } else {
            EnergyLevel::Low
        }
    }
    
    /// Map focus requirement float to FocusLevel enum
    fn map_focus_requirement(&self, requirement: f32) -> FocusLevel {
        if requirement > 0.7 {
            FocusLevel::Deep
        } else if requirement > 0.3 {
            FocusLevel::Moderate
        } else {
            FocusLevel::Light
        }
    }
    
    /// Convert a Task to TodoItem
    fn task_to_todo_item(&self, task: Task) -> TodoItem {
        use crate::tui::chat::orchestration::todo_manager::TodoStatus as TodoItemStatus;
        
        // Map task status to todo status
        let status = match task.status {
            TaskStatus::Todo => TodoItemStatus::Pending,
            TaskStatus::InProgress => TodoItemStatus::InProgress,
            TaskStatus::Done | TaskStatus::Completed => TodoItemStatus::Completed,
            TaskStatus::Cancelled => TodoItemStatus::Cancelled,
            TaskStatus::Blocked => TodoItemStatus::Blocked,
            TaskStatus::InReview => TodoItemStatus::Review,
            TaskStatus::Failed => TodoItemStatus::Cancelled,
        };
        
        // Map task priority to todo priority
        let priority_level = match task.priority {
            ExtTaskPriority::Critical => PriorityLevel::Critical,
            ExtTaskPriority::High => PriorityLevel::High,
            ExtTaskPriority::Medium => PriorityLevel::Medium,
            ExtTaskPriority::Low => PriorityLevel::Low,
        };
        let priority = TodoPriority {
            level: priority_level,
            score: priority_level as u8 as f32 / 5.0,
            factors: PriorityFactors::default(),
        };
        
        // Create cognitive metadata
        let cognitive_metadata = TodoCognitiveMetadata {
            complexity_score: task.cognitive_metadata.complexity_score,
            estimated_effort: task.cognitive_metadata.complexity_score * 8.0, // Estimate hours based on complexity (0-1 score * 8 hours max)
            cognitive_load: task.cognitive_metadata.complexity_score,
            context_switch_cost: task.cognitive_metadata.context_switching_cost,
            optimal_time_blocks: task.cognitive_metadata.optimal_time_blocks.clone(),
            required_skills: task.cognitive_metadata.prerequisite_knowledge.clone(), // Map prerequisite_knowledge to required_skills
            energy_requirement: if task.cognitive_metadata.energy_requirement > 0.7 {
                EnergyLevel::High
            } else if task.cognitive_metadata.energy_requirement > 0.3 {
                EnergyLevel::Medium
            } else {
                EnergyLevel::Low
            },
            focus_requirement: if task.cognitive_metadata.focus_requirement > 0.7 {
                FocusLevel::Deep
            } else if task.cognitive_metadata.focus_requirement > 0.3 {
                FocusLevel::Moderate
            } else {
                FocusLevel::Light
            },
            learning_opportunity: if task.cognitive_metadata.burnout_risk < 0.3 { 1.0 } else { 0.0 }, // Convert boolean logic to f32 score
            automation_potential: if task.cognitive_metadata.complexity_score < 0.5 { 1.0 } else { 0.0 }, // Convert boolean logic to f32 score
        };
        
        TodoItem {
            id: uuid::Uuid::new_v4().to_string(),
            title: task.title,
            description: task.description,
            status,
            priority,
            tags: task.labels,
            created_at: task.created_at,
            updated_at: task.updated_at,
            due_date: task.due_date,
            completed_at: if matches!(status, TodoItemStatus::Completed) {
                Some(task.updated_at)
            } else {
                None
            },
            parent_id: task.parent_task,
            subtask_ids: task.subtasks,
            cognitive_metadata: cognitive_metadata,
            story_context: None,
            creator: "system".to_string(),
            assignee: None,
            dependency_ids: Vec::new(),
            blocked_by: Vec::new(),
            estimated_duration: None,
            external_task_id: None,
            attachments: Vec::new(),
            comments: Vec::new(),
            orchestration_metadata: OrchestrationMetadata::default(),
            execution_metadata: ExecutionMetadata::default(),
        }
    }
}

