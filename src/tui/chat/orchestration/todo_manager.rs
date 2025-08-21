//! Todo Management System with Orchestration Integration
//! 
//! Provides intelligent todo management with AI-powered decomposition,
//! priority calculation, and multi-model execution orchestration.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::pin::Pin;
use std::future::Future;
use std::time::Duration;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::{Result, anyhow};
use uuid::Uuid;
use tracing::{info, debug, warn, error};

use super::{
    PipelineOrchestrator, Pipeline, Stage, ExecutionContext,
    ModelCallTracker, TrackingSession,
    UnifiedOrchestrator, OrchestrationRequest,
};
use crate::tools::task_management::{
    Task, TaskStatus, TaskPriority, TaskCognitiveMetadata,
    TaskPlatform, TaskManager,
};
use crate::tasks::{TaskRegistry, TaskContext};
use crate::tui::chat::agents::manager::AgentManager;
use crate::cognitive::agents::AgentSpecialization;

/// Todo management orchestrator
pub struct TodoManager {
    /// Todo items by ID
    todos: Arc<RwLock<HashMap<String, TodoItem>>>,
    
    /// Todo execution queue
    execution_queue: Arc<RwLock<VecDeque<String>>>,
    
    /// Active todo executions
    active_executions: Arc<RwLock<HashMap<String, TodoExecution>>>,
    
    /// Pipeline orchestrator for complex workflows
    pipeline_orchestrator: Arc<PipelineOrchestrator>,
    
    /// Model call tracker for monitoring
    call_tracker: Arc<ModelCallTracker>,
    
    /// Task manager for external platform sync
    task_manager: Option<Arc<TaskManager>>,
    
    /// Task registry for executable tasks
    task_registry: Option<Arc<TaskRegistry>>,
    
    /// Agent manager for collaborative execution
    agent_manager: Option<Arc<RwLock<AgentManager>>>,
    
    /// Configuration
    config: TodoConfig,
    
    /// Statistics
    stats: Arc<RwLock<TodoStatistics>>,
}

/// Todo item with full metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub status: TodoStatus,
    pub priority: TodoPriority,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub due_date: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub estimated_duration: Option<Duration>,
    pub creator: String,
    pub assignee: Option<String>,
    pub tags: Vec<String>,
    pub parent_id: Option<String>,
    pub subtask_ids: Vec<String>,
    pub dependency_ids: Vec<String>,
    pub blocked_by: Vec<String>,
    pub cognitive_metadata: TodoCognitiveMetadata,
    pub orchestration_metadata: OrchestrationMetadata,
    pub external_task_id: Option<String>,
    pub attachments: Vec<TodoAttachment>,
    pub comments: Vec<TodoComment>,
    pub story_context: Option<TodoStoryContext>,
    pub execution_metadata: ExecutionMetadata,
}

/// Todo status
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TodoStatus {
    Pending,
    Ready,
    InProgress,
    Blocked,
    Review,
    Completed,
    Cancelled,
    Failed,
}

/// Todo priority with intelligent scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoPriority {
    pub level: PriorityLevel,
    pub score: f32, // 0.0 to 1.0
    pub factors: PriorityFactors,
}

/// Priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PriorityLevel {
    Critical = 5,
    High = 4,
    Medium = 3,
    Low = 2,
    Minimal = 1,
}

impl std::fmt::Display for PriorityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PriorityLevel::Critical => write!(f, "Critical"),
            PriorityLevel::High => write!(f, "High"),
            PriorityLevel::Medium => write!(f, "Medium"),
            PriorityLevel::Low => write!(f, "Low"),
            PriorityLevel::Minimal => write!(f, "Minimal"),
        }
    }
}

/// Factors contributing to priority
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PriorityFactors {
    pub urgency: f32,
    pub importance: f32,
    pub complexity: f32,
    pub dependencies: f32,
    pub user_preference: f32,
    pub deadline_proximity: f32,
    pub blocking_others: f32,
}

/// Cognitive metadata for todos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoCognitiveMetadata {
    pub complexity_score: f32,
    pub estimated_effort: f32, // in hours
    pub cognitive_load: f32,
    pub context_switch_cost: f32,
    pub optimal_time_blocks: Vec<String>,
    pub required_skills: Vec<String>,
    pub energy_requirement: EnergyLevel,
    pub focus_requirement: FocusLevel,
    pub learning_opportunity: f32,
    pub automation_potential: f32,
}

/// Energy level required
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EnergyLevel {
    High,
    Medium,
    Low,
}

/// Focus level required
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FocusLevel {
    Deep,
    Moderate,
    Light,
}

/// Orchestration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationMetadata {
    pub execution_strategy: ExecutionStrategy,
    pub model_preferences: Vec<String>,
    pub agent_assignments: Vec<AgentSpecialization>,
    pub pipeline_id: Option<String>,
    pub parallel_subtasks: bool,
    pub auto_decompose: bool,
    pub max_retries: u32,
    pub timeout_seconds: u64,
}

impl Default for OrchestrationMetadata {
    fn default() -> Self {
        Self {
            execution_strategy: ExecutionStrategy::default(),
            model_preferences: Vec::new(),
            agent_assignments: Vec::new(),
            pipeline_id: None,
            parallel_subtasks: false,
            auto_decompose: false,
            max_retries: 3,
            timeout_seconds: 300,
        }
    }
}

/// Execution metadata for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub stages_completed: usize,
    pub execution_progress: f32,
    pub current_stage: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub last_update: Option<DateTime<Utc>>,
    pub error_count: u32,
    pub retry_count: u32,
}

impl Default for ExecutionMetadata {
    fn default() -> Self {
        Self {
            stages_completed: 0,
            execution_progress: 0.0,
            current_stage: None,
            started_at: None,
            last_update: None,
            error_count: 0,
            retry_count: 0,
        }
    }
}

/// Execution strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ExecutionStrategy {
    #[default]
    Simple,      // Direct execution
    Pipeline,    // Multi-stage pipeline
    Collaborative, // Multi-agent collaboration
    Recursive,   // Recursive decomposition
    Adaptive,    // Dynamic strategy selection
}

/// Todo attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoAttachment {
    pub id: String,
    pub name: String,
    pub url: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub uploaded_at: DateTime<Utc>,
}

/// Todo comment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoComment {
    pub id: String,
    pub author: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub edited_at: Option<DateTime<Utc>>,
}

/// Active todo execution
#[derive(Debug, Clone)]
pub struct TodoExecution {
    pub todo_id: String,
    pub execution_id: String,
    pub started_at: DateTime<Utc>,
    pub status: ExecutionStatus,
    pub progress: f32,
    pub current_stage: String,
    pub model_session_id: Option<String>,
    pub agent_sessions: Vec<String>,
    pub logs: Vec<ExecutionLog>,
    pub error: Option<String>,
}

/// Execution status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStatus {
    Initializing,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Execution log entry
#[derive(Debug, Clone)]
pub struct ExecutionLog {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub metadata: HashMap<String, String>,
}

/// Log levels
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

/// Todo configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoConfig {
    pub auto_decompose_threshold: f32,
    pub max_subtask_depth: usize,
    pub default_timeout_seconds: u64,
    pub enable_external_sync: bool,
    pub enable_ai_suggestions: bool,
    pub enable_auto_scheduling: bool,
    pub priority_recalc_interval: u64,
    pub execution_parallelism: usize,
}

impl Default for TodoConfig {
    fn default() -> Self {
        Self {
            auto_decompose_threshold: 0.7,
            max_subtask_depth: 3,
            default_timeout_seconds: 300,
            enable_external_sync: true,
            enable_ai_suggestions: true,
            enable_auto_scheduling: true,
            priority_recalc_interval: 60,
            execution_parallelism: 3,
        }
    }
}

/// Todo statistics
#[derive(Debug, Clone, Default)]
pub struct TodoStatistics {
    pub total_created: usize,
    pub total_completed: usize,
    pub total_cancelled: usize,
    pub total_failed: usize,
    pub total_deleted: usize,
    pub active_todos: usize,
    pub blocked_todos: usize,
    pub average_completion_time: f64,
    pub completion_rate: f32,
    pub decomposition_count: usize,
    pub ai_assistance_count: usize,
    pub priority_changes: usize,
}

impl TodoManager {
    /// Create a new todo manager
    pub fn new(
        pipeline_orchestrator: Arc<PipelineOrchestrator>,
        call_tracker: Arc<ModelCallTracker>,
    ) -> Self {
        Self {
            todos: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            pipeline_orchestrator,
            call_tracker,
            task_manager: None,
            task_registry: None,
            agent_manager: None,
            config: TodoConfig::default(),
            stats: Arc::new(RwLock::new(TodoStatistics::default())),
        }
    }
    
    /// Set task manager for external sync
    pub fn set_task_manager(&mut self, task_manager: Arc<TaskManager>) {
        self.task_manager = Some(task_manager);
    }
    
    /// Set task registry for executable tasks
    pub fn set_task_registry(&mut self, task_registry: Arc<TaskRegistry>) {
        self.task_registry = Some(task_registry);
    }
    
    /// Set agent manager for collaborative execution
    pub fn set_agent_manager(&mut self, agent_manager: Arc<RwLock<AgentManager>>) {
        self.agent_manager = Some(agent_manager);
    }
    
    /// Create a new todo
    pub fn create_todo(&self, request: CreateTodoRequest) -> Pin<Box<dyn Future<Output = Result<TodoItem>> + Send + '_>> {
        Box::pin(async move {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        // Calculate cognitive metadata
        let cognitive_metadata = self.calculate_cognitive_metadata(&request).await?;
        
        // Determine orchestration strategy
        let orchestration_metadata = self.determine_orchestration_strategy(&request, &cognitive_metadata).await?;
        
        // Calculate priority
        let priority = self.calculate_priority(&request, &cognitive_metadata).await;
        
        let todo = TodoItem {
            id: id.clone(),
            title: request.title,
            description: request.description,
            status: TodoStatus::Pending,
            priority,
            created_at: now,
            updated_at: now,
            due_date: request.due_date,
            completed_at: None,
            estimated_duration: None,  // Can be enhanced later with AI estimation
            creator: request.creator,
            assignee: request.assignee,
            tags: request.tags,
            parent_id: request.parent_id,
            subtask_ids: Vec::new(),
            dependency_ids: request.dependency_ids,
            blocked_by: Vec::new(),
            cognitive_metadata,
            orchestration_metadata,
            external_task_id: None,
            attachments: Vec::new(),
            comments: Vec::new(),
            story_context: request.story_context,
            execution_metadata: ExecutionMetadata::default(),
        };
        
        // Store todo
        self.todos.write().await.insert(id.clone(), todo.clone());
        
        // Update statistics
        self.stats.write().await.total_created += 1;
        self.stats.write().await.active_todos += 1;
        
        // Auto-decompose if needed
        if self.should_decompose(&todo) {
            self.decompose_todo(&id).await?;
        }
        
        // Sync with external task manager if configured
        if self.config.enable_external_sync {
            if let Some(task_manager) = &self.task_manager {
                self.sync_to_external(task_manager, &todo).await?;
            }
        }
        
        // Check if ready to execute
        if self.is_ready_to_execute(&todo).await {
            self.queue_for_execution(id).await?;
        }
        
        info!("Created todo: {} - {}", todo.id, todo.title);
        Ok(todo)
        })
    }
    
    /// Update todo status
    pub async fn update_status(&self, todo_id: &str, status: TodoStatus) -> Result<()> {
        let mut todos = self.todos.write().await;
        
        if let Some(todo) = todos.get_mut(todo_id) {
            let old_status = todo.status;
            todo.status = status;
            todo.updated_at = Utc::now();
            
            if status == TodoStatus::Completed {
                todo.completed_at = Some(Utc::now());
                self.stats.write().await.total_completed += 1;
                self.stats.write().await.active_todos -= 1;
            }
            
            // Handle status transitions
            self.handle_status_transition(todo_id, old_status, status).await?;
            
            debug!("Updated todo {} status: {:?} -> {:?}", todo_id, old_status, status);
        }
        
        Ok(())
    }
    
    /// Execute a todo
    pub fn execute_todo<'a>(&'a self, todo_id: &'a str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TodoExecution>> + Send + 'a>> {
        Box::pin(async move {
        let todos = self.todos.read().await;
        let todo = todos.get(todo_id)
            .ok_or_else(|| anyhow::anyhow!("Todo not found: {}", todo_id))?;
        
        let execution_id = Uuid::new_v4().to_string();
        
        // Start model tracking
        let session_id = self.call_tracker.start_tracking(
            "todo-executor",
            &format!("Todo: {}", todo.title),
            1000,
            0.7,
            Some(500),
        ).await?;
        
        let execution = TodoExecution {
            todo_id: todo_id.to_string(),
            execution_id: execution_id.clone(),
            started_at: Utc::now(),
            status: ExecutionStatus::Initializing,
            progress: 0.0,
            current_stage: "initialization".to_string(),
            model_session_id: Some(session_id),
            agent_sessions: Vec::new(),
            logs: Vec::new(),
            error: None,
        };
        
        self.active_executions.write().await.insert(execution_id.clone(), execution.clone());
        
        // Execute based on strategy
        let execution_result = match &todo.orchestration_metadata.execution_strategy {
            ExecutionStrategy::Simple => {
                self.execute_simple(todo).await
            }
            ExecutionStrategy::Pipeline => {
                self.execute_pipeline(todo).await
            }
            ExecutionStrategy::Collaborative => {
                self.execute_collaborative(todo).await
            }
            ExecutionStrategy::Recursive => {
                self.execute_recursive(todo).await
            }
            ExecutionStrategy::Adaptive => {
                self.execute_adaptive(todo).await
            }
        };
        
        // Update execution status
        let mut executions = self.active_executions.write().await;
        if let Some(exec) = executions.get_mut(&execution_id) {
            match execution_result {
                Ok(_) => {
                    exec.status = ExecutionStatus::Completed;
                    exec.progress = 1.0;
                    self.update_status(todo_id, TodoStatus::Completed).await?;
                }
                Err(e) => {
                    exec.status = ExecutionStatus::Failed;
                    exec.error = Some(e.to_string());
                    self.update_status(todo_id, TodoStatus::Failed).await?;
                }
            }
        }
        
        Ok(execution)
        })
    }
    
    /// Calculate cognitive metadata
    async fn calculate_cognitive_metadata(&self, request: &CreateTodoRequest) -> Result<TodoCognitiveMetadata> {
        // Analyze task complexity based on description
        let complexity_score = self.analyze_complexity(&request.title, &request.description).await;
        
        // Estimate effort based on complexity and similar tasks
        let estimated_effort = complexity_score * 2.0; // Basic heuristic
        
        // Determine cognitive requirements
        let (energy_level, focus_level) = if complexity_score > 0.7 {
            (EnergyLevel::High, FocusLevel::Deep)
        } else if complexity_score > 0.4 {
            (EnergyLevel::Medium, FocusLevel::Moderate)
        } else {
            (EnergyLevel::Low, FocusLevel::Light)
        };
        
        Ok(TodoCognitiveMetadata {
            complexity_score,
            estimated_effort,
            cognitive_load: complexity_score * 0.8,
            context_switch_cost: complexity_score * 0.5,
            optimal_time_blocks: vec!["morning".to_string()],
            required_skills: self.extract_required_skills(&request.title, &request.description).await,
            energy_requirement: energy_level,
            focus_requirement: focus_level,
            learning_opportunity: complexity_score * 0.3,
            automation_potential: 1.0 - complexity_score,
        })
    }
    
    /// Determine orchestration strategy
    async fn determine_orchestration_strategy(
        &self,
        request: &CreateTodoRequest,
        cognitive_metadata: &TodoCognitiveMetadata,
    ) -> Result<OrchestrationMetadata> {
        let strategy = if cognitive_metadata.complexity_score > 0.8 {
            ExecutionStrategy::Collaborative
        } else if cognitive_metadata.automation_potential > 0.7 {
            ExecutionStrategy::Pipeline
        } else if request.description.as_ref().map_or(false, |d| d.contains("step")) {
            ExecutionStrategy::Recursive
        } else {
            ExecutionStrategy::Simple
        };
        
        Ok(OrchestrationMetadata {
            execution_strategy: strategy,
            model_preferences: vec!["gpt-4".to_string()],
            agent_assignments: self.determine_agent_assignments(cognitive_metadata).await,
            pipeline_id: None,
            parallel_subtasks: cognitive_metadata.complexity_score < 0.5,
            auto_decompose: cognitive_metadata.complexity_score > 0.6,
            max_retries: 3,
            timeout_seconds: self.config.default_timeout_seconds,
        })
    }
    
    /// Calculate priority
    async fn calculate_priority(
        &self,
        request: &CreateTodoRequest,
        cognitive_metadata: &TodoCognitiveMetadata,
    ) -> TodoPriority {
        let mut factors = PriorityFactors {
            urgency: 0.5,
            importance: 0.5,
            complexity: cognitive_metadata.complexity_score,
            dependencies: 0.0,
            user_preference: request.priority_hint.unwrap_or(0.5),
            deadline_proximity: 0.0,
            blocking_others: 0.0,
        };
        
        // Calculate deadline proximity
        if let Some(due_date) = request.due_date {
            let hours_until_due = (due_date - Utc::now()).num_hours() as f32;
            factors.deadline_proximity = (1.0 / (1.0 + hours_until_due / 24.0)).min(1.0);
            factors.urgency = factors.deadline_proximity;
        }
        
        // Calculate overall score
        let score = (factors.urgency * 0.3
            + factors.importance * 0.3
            + factors.complexity * 0.1
            + factors.dependencies * 0.1
            + factors.user_preference * 0.1
            + factors.deadline_proximity * 0.05
            + factors.blocking_others * 0.05).min(1.0);
        
        let level = if score > 0.8 {
            PriorityLevel::Critical
        } else if score > 0.6 {
            PriorityLevel::High
        } else if score > 0.4 {
            PriorityLevel::Medium
        } else if score > 0.2 {
            PriorityLevel::Low
        } else {
            PriorityLevel::Minimal
        };
        
        TodoPriority { level, score, factors }
    }
    
    /// Check if todo should be decomposed
    fn should_decompose(&self, todo: &TodoItem) -> bool {
        todo.cognitive_metadata.complexity_score > self.config.auto_decompose_threshold
            && todo.orchestration_metadata.auto_decompose
            && todo.subtask_ids.is_empty()
    }
    
    /// Decompose todo into subtasks
    async fn decompose_todo(&self, todo_id: &str) -> Result<Vec<String>> {
        let todos = self.todos.read().await;
        let todo = todos.get(todo_id)
            .ok_or_else(|| anyhow::anyhow!("Todo not found"))?;
        
        // Use AI to decompose the task
        // This would call the model to break down the task
        let subtasks = self.ai_decompose_task(&todo.title, &todo.description).await?;
        
        let mut subtask_ids = Vec::new();
        for subtask in subtasks {
            let subtask_request = CreateTodoRequest {
                title: subtask.title,
                description: subtask.description,
                parent_id: Some(todo_id.to_string()),
                ..Default::default()
            };
            
            let subtask_todo = self.create_todo(subtask_request).await?;
            subtask_ids.push(subtask_todo.id);
        }
        
        // Update parent todo
        drop(todos);
        self.todos.write().await.get_mut(todo_id).unwrap().subtask_ids = subtask_ids.clone();
        
        self.stats.write().await.decomposition_count += 1;
        
        Ok(subtask_ids)
    }
    
    /// AI-powered task decomposition using cognitive analysis
    async fn ai_decompose_task(&self, title: &str, description: &Option<String>) -> Result<Vec<SubtaskSuggestion>> {
        // Use cognitive system to analyze task and generate subtasks
        let mut subtasks = Vec::new();
        
        // Analyze task complexity and requirements
        let task_text = format!("{} {}", title, description.as_ref().unwrap_or(&String::new()));
        let complexity = self.estimate_complexity(&task_text);
        
        // Generate subtasks based on task type and complexity
        if task_text.to_lowercase().contains("implement") || task_text.to_lowercase().contains("build") {
            // Development task decomposition
            subtasks.push(SubtaskSuggestion {
                title: format!("Design architecture for: {}", title),
                description: Some("Plan the technical approach and architecture".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Implement core functionality: {}", title),
                description: Some("Build the main features and logic".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Write tests for: {}", title),
                description: Some("Create unit and integration tests".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Document: {}", title),
                description: Some("Write documentation and usage examples".to_string()),
            });
        } else if task_text.to_lowercase().contains("research") || task_text.to_lowercase().contains("analyze") {
            // Research task decomposition
            subtasks.push(SubtaskSuggestion {
                title: format!("Gather sources for: {}", title),
                description: Some("Collect relevant resources and references".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Analyze data: {}", title),
                description: Some("Process and analyze collected information".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Synthesize findings: {}", title),
                description: Some("Create summary and conclusions".to_string()),
            });
        } else if task_text.to_lowercase().contains("fix") || task_text.to_lowercase().contains("debug") {
            // Bug fix decomposition
            subtasks.push(SubtaskSuggestion {
                title: format!("Reproduce issue: {}", title),
                description: Some("Identify steps to reproduce the problem".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Identify root cause: {}", title),
                description: Some("Debug and find the source of the issue".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Implement fix: {}", title),
                description: Some("Apply the solution".to_string()),
            });
            subtasks.push(SubtaskSuggestion {
                title: format!("Verify fix: {}", title),
                description: Some("Test that the issue is resolved".to_string()),
            });
        } else {
            // Generic task decomposition based on complexity
            if complexity > 0.7 {
                // High complexity - more detailed breakdown
                subtasks.push(SubtaskSuggestion {
                    title: format!("Plan approach: {}", title),
                    description: Some("Define strategy and requirements".to_string()),
                });
                subtasks.push(SubtaskSuggestion {
                    title: format!("Break down: {}", title),
                    description: Some("Identify components and dependencies".to_string()),
                });
            }
            
            subtasks.push(SubtaskSuggestion {
                title: format!("Execute: {}", title),
                description: Some("Perform the main task".to_string()),
            });
            
            subtasks.push(SubtaskSuggestion {
                title: format!("Review: {}", title),
                description: Some("Verify completion and quality".to_string()),
            });
        }
        
        // Add coordination subtask for highly complex tasks
        if complexity > 0.8 {
            subtasks.insert(0, SubtaskSuggestion {
                title: format!("Coordinate: {}", title),
                description: Some("Manage dependencies and communication".to_string()),
            });
        }
        
        Ok(subtasks)
    }
    
    /// Estimate task complexity based on text analysis
    fn estimate_complexity(&self, text: &str) -> f32 {
        let mut complexity: f32 = 0.3; // Base complexity
        
        // Increase complexity based on keywords
        let complex_keywords = ["architecture", "system", "integrate", "migrate", "refactor", 
                                "optimize", "scale", "security", "performance", "distributed"];
        for keyword in &complex_keywords {
            if text.to_lowercase().contains(keyword) {
                complexity += 0.1;
            }
        }
        
        // Increase complexity based on length
        if text.len() > 200 {
            complexity += 0.2;
        } else if text.len() > 100 {
            complexity += 0.1;
        }
        
        // Cap at 1.0
        complexity.min(1.0_f32)
    }
    
    /// Sync todo to external task manager
    async fn sync_to_external(&self, task_manager: &Arc<TaskManager>, todo: &TodoItem) -> Result<()> {
        let task = self.todo_to_task(todo);
        task_manager.create_task(
            &task.title,
            task.description.as_deref(),
            task.priority,
            task.due_date,
            task.platform,
        ).await?;
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
            status: self.map_todo_status_to_task_status(todo.status),
            priority: self.map_priority(todo.priority.level),
            assignee: todo.assignee.clone(),
            reporter: Some(todo.creator.clone()),
            labels: todo.tags.clone(),
            due_date: todo.due_date,
            created_at: todo.created_at,
            updated_at: todo.updated_at,
            estimate: Some(std::time::Duration::from_secs((todo.cognitive_metadata.estimated_effort * 3600.0) as u64)),
            time_spent: None,
            progress: self.calculate_progress(todo),
            parent_task: todo.parent_id.clone(),
            subtasks: todo.subtask_ids.clone(),
            dependencies: todo.dependency_ids.clone(),
            cognitive_metadata: TaskCognitiveMetadata {
                cognitive_priority: todo.priority.score,
                complexity_score: todo.cognitive_metadata.complexity_score,
                energy_requirement: todo.cognitive_metadata.cognitive_load,
                focus_requirement: todo.cognitive_metadata.cognitive_load,
                context_switching_cost: todo.cognitive_metadata.context_switch_cost,
                optimal_time_blocks: todo.cognitive_metadata.optimal_time_blocks.clone(),
                prerequisite_knowledge: todo.cognitive_metadata.required_skills.clone(),
                related_memories: Vec::new(),
                burnout_risk: 0.0,
                motivation_factors: Vec::new(),
            },
            name: todo.title.clone(),
            metadata: serde_json::json!({}),
        }
    }
    
    /// Check if todo is ready to execute
    async fn is_ready_to_execute(&self, todo: &TodoItem) -> bool {
        todo.status == TodoStatus::Pending
            && todo.blocked_by.is_empty()
            && todo.dependency_ids.is_empty()
    }
    
    /// Queue todo for execution
    async fn queue_for_execution(&self, todo_id: String) -> Result<()> {
        self.execution_queue.write().await.push_back(todo_id.clone());
        self.update_status(&todo_id, TodoStatus::Ready).await?;
        Ok(())
    }
    
    /// Handle status transitions
    async fn handle_status_transition(
        &self,
        todo_id: &str,
        old_status: TodoStatus,
        new_status: TodoStatus,
    ) -> Result<()> {
        // Check if this unblocks other todos
        if new_status == TodoStatus::Completed {
            let todos = self.todos.read().await;
            let blocked_todos: Vec<String> = todos.values()
                .filter(|t| t.blocked_by.contains(&todo_id.to_string()))
                .map(|t| t.id.clone())
                .collect();
            drop(todos);
            
            for blocked_id in blocked_todos {
                let mut todos = self.todos.write().await;
                if let Some(blocked_todo) = todos.get_mut(&blocked_id) {
                    blocked_todo.blocked_by.retain(|id| id != todo_id);
                    if blocked_todo.blocked_by.is_empty() && blocked_todo.status == TodoStatus::Blocked {
                        blocked_todo.status = TodoStatus::Pending;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute simple todo
    async fn execute_simple(&self, todo: &TodoItem) -> Result<()> {
        // Simple direct execution
        if let Some(task_registry) = &self.task_registry {
            // Try to find a matching task
            if let Some(task) = task_registry.get(&todo.title) {
                // Execute the task
                // This would need proper TaskArgs and TaskContext
                info!("Executing simple todo: {}", todo.title);
            }
        }
        Ok(())
    }
    
    /// Execute pipeline-based todo
    async fn execute_pipeline(&self, todo: &TodoItem) -> Result<()> {
        // Create pipeline for todo execution
        let pipeline = self.create_todo_pipeline(todo).await?;
        
        // Register and execute pipeline
        self.pipeline_orchestrator.register_pipeline(pipeline.clone()).await?;
        
        let input = serde_json::json!({
            "todo_id": todo.id,
            "title": todo.title,
            "description": todo.description,
        });
        
        let result = self.pipeline_orchestrator.execute(
            &pipeline.id,
            input,
            HashMap::new(),
        ).await?;
        
        info!("Pipeline execution completed for todo: {}", todo.id);
        Ok(())
    }
    
    /// Execute collaborative todo with agents
    async fn execute_collaborative(&self, todo: &TodoItem) -> Result<()> {
        if let Some(agent_manager) = &self.agent_manager {
            let manager = agent_manager.read().await;
            
            if manager.agent_system_enabled {
                // Assign agents based on specialization
                for specialization in &todo.orchestration_metadata.agent_assignments {
                    info!("Assigning {:?} agent to todo: {}", specialization, todo.title);
                    // Agent execution would happen here
                }
            }
        }
        Ok(())
    }
    
    /// Execute recursive todo decomposition
    async fn execute_recursive(&self, todo: &TodoItem) -> Result<()> {
        // Decompose and execute subtasks
        let subtask_ids = self.decompose_todo(&todo.id).await?;
        
        for subtask_id in subtask_ids {
            // Box the recursive call to avoid infinite-sized future
            Box::pin(self.execute_todo(&subtask_id)).await?;
        }
        
        Ok(())
    }
    
    /// Execute with adaptive strategy
    async fn execute_adaptive(&self, todo: &TodoItem) -> Result<()> {
        // Dynamically select best strategy based on current conditions
        let strategy = self.select_adaptive_strategy(todo).await;
        
        match strategy {
            ExecutionStrategy::Simple => self.execute_simple(todo).await,
            ExecutionStrategy::Pipeline => self.execute_pipeline(todo).await,
            ExecutionStrategy::Collaborative => self.execute_collaborative(todo).await,
            ExecutionStrategy::Recursive => self.execute_recursive(todo).await,
            _ => self.execute_simple(todo).await,
        }
    }
    
    /// Select adaptive execution strategy
    async fn select_adaptive_strategy(&self, todo: &TodoItem) -> ExecutionStrategy {
        // Consider current system load, available resources, etc.
        if todo.cognitive_metadata.complexity_score > 0.7 {
            ExecutionStrategy::Collaborative
        } else if !todo.subtask_ids.is_empty() {
            ExecutionStrategy::Recursive
        } else {
            ExecutionStrategy::Simple
        }
    }
    
    /// Create pipeline for todo execution with comprehensive stages
    async fn create_todo_pipeline(&self, todo: &TodoItem) -> Result<Pipeline> {
        use super::pipeline::{Stage, RetryConfig, BackoffStrategy};
        
        let stages = self.define_todo_pipeline_stages(todo).await?;
        
        let pipeline = Pipeline {
            id: format!("todo-pipeline-{}", todo.id),
            name: format!("Pipeline for: {}", todo.title),
            description: format!("Automated pipeline for todo execution with story context"),
            stages,
            error_handling: Default::default(),
            timeout_seconds: Some(todo.orchestration_metadata.timeout_seconds),
            max_retries: todo.orchestration_metadata.max_retries,
            metadata: self.create_pipeline_metadata(todo),
        };
        
        Ok(pipeline)
    }
    
    /// Define comprehensive pipeline stages for todo execution
    async fn define_todo_pipeline_stages(&self, todo: &TodoItem) -> Result<Vec<Stage>> {
        use super::pipeline::{Stage, RetryConfig, BackoffStrategy};
        
        let mut stages = vec![
            // Stage 1: Context Loading
            Stage {
                id: "context_loading".to_string(),
                name: "Load Context".to_string(),
                processor: "context_loader".to_string(),
                input_mapping: Default::default(),
                output_mapping: Default::default(),
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(10),
                retry_config: Some(RetryConfig {
                    max_attempts: 2,
                    backoff_strategy: BackoffStrategy::Fixed { delay_ms: 1000 },
                    retry_on: vec!["timeout".to_string()],
                }),
            },
            
            // Stage 2: Dependency Resolution
            Stage {
                id: "dependency_resolution".to_string(),
                name: "Resolve Dependencies".to_string(),
                processor: "dependency_resolver".to_string(),
                input_mapping: Default::default(),
                output_mapping: Default::default(),
                conditions: Vec::new(),
                parallel: true, // Can check multiple dependencies in parallel
                optional: todo.dependency_ids.is_empty(),
                timeout_seconds: Some(20),
                retry_config: None,
            },
            
            // Stage 3: Resource Allocation
            Stage {
                id: "resource_allocation".to_string(),
                name: "Allocate Resources".to_string(),
                processor: "resource_allocator".to_string(),
                input_mapping: Default::default(),
                output_mapping: Default::default(),
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(15),
                retry_config: Some(RetryConfig {
                    max_attempts: 3,
                    backoff_strategy: BackoffStrategy::Exponential { 
                        initial_ms: 500, 
                        multiplier: 2.0, 
                        max_ms: 5000 
                    },
                    retry_on: vec!["resource_busy".to_string()],
                }),
            },
        ];
        
        // Stage 4: Story Context Integration (if applicable)
        if todo.story_context.is_some() {
            stages.push(Stage {
                id: "story_integration".to_string(),
                name: "Integrate Story Context".to_string(),
                processor: "story_processor".to_string(),
                input_mapping: Default::default(),
                output_mapping: Default::default(),
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(10),
                retry_config: None,
            });
        }
        
        // Stage 5: Agent Assignment (for collaborative execution)
        if !todo.orchestration_metadata.agent_assignments.is_empty() {
            stages.push(Stage {
                id: "agent_assignment".to_string(),
                name: "Assign Agents".to_string(),
                processor: "agent_coordinator".to_string(),
                input_mapping: Default::default(),
                output_mapping: Default::default(),
                conditions: Vec::new(),
                parallel: true, // Multiple agents can be assigned in parallel
                optional: false,
                timeout_seconds: Some(20),
                retry_config: None,
            });
        }
        
        // Stage 6: Main Execution
        stages.push(Stage {
            id: "main_execution".to_string(),
            name: "Execute Task".to_string(),
            processor: self.get_execution_processor(todo),
            input_mapping: Default::default(),
            output_mapping: Default::default(),
            conditions: Vec::new(),
            parallel: todo.orchestration_metadata.parallel_subtasks,
            optional: false,
            timeout_seconds: Some(todo.orchestration_metadata.timeout_seconds),
            retry_config: Some(RetryConfig {
                max_attempts: todo.orchestration_metadata.max_retries,
                backoff_strategy: BackoffStrategy::Linear { 
                    initial_ms: 1000, 
                    increment_ms: 2000 
                },
                retry_on: vec!["transient_error".to_string()],
            }),
        });
        
        // Stage 7: Result Validation
        stages.push(Stage {
            id: "validation".to_string(),
            name: "Validate Results".to_string(),
            processor: "result_validator".to_string(),
            input_mapping: Default::default(),
            output_mapping: Default::default(),
            conditions: Vec::new(),
            parallel: false,
            optional: false,
            timeout_seconds: Some(15),
            retry_config: None,
        });
        
        // Stage 8: Story Update (if applicable)
        if todo.story_context.is_some() {
            stages.push(Stage {
                id: "story_update".to_string(),
                name: "Update Story".to_string(),
                processor: "story_updater".to_string(),
                input_mapping: Default::default(),
                output_mapping: Default::default(),
                conditions: Vec::new(),
                parallel: false,
                optional: false,
                timeout_seconds: Some(10),
                retry_config: None,
            });
        }
        
        // Stage 9: Feedback Collection
        stages.push(Stage {
            id: "feedback".to_string(),
            name: "Collect Feedback".to_string(),
            processor: "feedback_collector".to_string(),
            input_mapping: Default::default(),
            output_mapping: Default::default(),
            conditions: Vec::new(),
            parallel: false,
            optional: true,
            timeout_seconds: Some(5),
            retry_config: None,
        });
        
        // Stage 10: Cleanup
        stages.push(Stage {
            id: "cleanup".to_string(),
            name: "Cleanup Resources".to_string(),
            processor: "resource_cleanup".to_string(),
            input_mapping: Default::default(),
            output_mapping: Default::default(),
            conditions: Vec::new(),
            parallel: false,
            optional: false,
            timeout_seconds: Some(10),
            retry_config: None,
        });
        
        Ok(stages)
    }
    
    /// Get execution processor based on todo type
    fn get_execution_processor(&self, todo: &TodoItem) -> String {
        match &todo.orchestration_metadata.execution_strategy {
            ExecutionStrategy::Simple => "simple_executor".to_string(),
            ExecutionStrategy::Pipeline => "pipeline_executor".to_string(),
            ExecutionStrategy::Collaborative => "collaborative_executor".to_string(),
            ExecutionStrategy::Recursive => "recursive_executor".to_string(),
            ExecutionStrategy::Adaptive => "adaptive_executor".to_string(),
        }
    }
    
    /// Create pipeline metadata
    fn create_pipeline_metadata(&self, todo: &TodoItem) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        
        metadata.insert("todo_id".to_string(), serde_json::json!(todo.id));
        metadata.insert("priority".to_string(), serde_json::json!(todo.priority.score));
        metadata.insert("complexity".to_string(), serde_json::json!(todo.cognitive_metadata.complexity_score));
        
        if let Some(ref story_context) = todo.story_context {
            metadata.insert("story_id".to_string(), serde_json::json!(story_context.story_id));
            metadata.insert("narrative".to_string(), serde_json::json!(story_context.narrative));
        }
        
        metadata
    }
    
    /// Create todo from plot point (story integration)
    pub async fn create_todo_from_plot_point(
        &self,
        plot_point_id: String,
        story_id: String,
        narrative: String,
    ) -> Result<TodoItem> {
        let request = CreateTodoRequest {
            title: format!("Story Task: {}", plot_point_id),
            description: Some(narrative.clone()),
            creator: "story-engine".to_string(),
            assignee: None,
            due_date: None,
            tags: vec!["story-driven".to_string()],
            parent_id: None,
            dependency_ids: Vec::new(),
            priority_hint: Some(0.6),
            story_context: None,
            priority: None,
            energy_required: None,
            focus_required: None,
            context: None,
        };
        
        let mut todo = self.create_todo(request).await?;
        
        // Add story context
        todo.story_context = Some(TodoStoryContext {
            story_id,
            plot_point_id: Some(plot_point_id),
            narrative,
            story_arc: None,
            related_events: Vec::new(),
        });
        
        // Update in storage
        self.todos.write().await.insert(todo.id.clone(), todo.clone());
        
        Ok(todo)
    }
    
    /// Map todo to story arc
    pub async fn map_todo_to_story_arc(
        &self,
        todo_id: &str,
        story_id: String,
        arc_id: String,
    ) -> Result<()> {
        let mut todos = self.todos.write().await;
        
        if let Some(todo) = todos.get_mut(todo_id) {
            if let Some(ref mut story_context) = todo.story_context {
                story_context.story_arc = Some(arc_id);
            } else {
                todo.story_context = Some(TodoStoryContext {
                    story_id,
                    plot_point_id: None,
                    narrative: String::new(),
                    story_arc: Some(arc_id),
                    related_events: Vec::new(),
                });
            }
            
            todo.updated_at = Utc::now();
        }
        
        Ok(())
    }
    
    /// List all todos with optional filtering
    pub async fn list_todos(&self) -> Result<Vec<TodoItem>> {
        let todos = self.todos.read().await;
        Ok(todos.values().cloned().collect())
    }
    
    /// Get todos (alias for list_todos for compatibility)
    pub async fn get_todos(&self) -> Vec<TodoItem> {
        self.list_todos().await.unwrap_or_default()
    }
    
    /// Delete a todo by ID
    pub async fn delete_todo(&self, id: &str) -> Result<()> {
        let mut todos = self.todos.write().await;
        if todos.remove(id).is_some() {
            info!("Deleted todo: {}", id);
            
            // Update statistics
            let mut stats = self.stats.write().await;
            stats.total_deleted += 1;
            
            Ok(())
        } else {
            Err(anyhow!("Todo not found: {}", id))
        }
    }
    
    /// Get pipeline stages for visualization
    pub async fn get_pipeline_stages(&self) -> Result<Vec<String>> {
        // Return default pipeline stages for now
        Ok(vec![
            "Planning".to_string(),
            "Validation".to_string(),
            "Execution".to_string(),
            "Review".to_string(),
            "Completion".to_string(),
        ])
    }
    
    /// Analyze task complexity
    async fn analyze_complexity(&self, title: &str, description: &Option<String>) -> f32 {
        // Simple heuristic based on text analysis
        let text_length = title.len() + description.as_ref().map_or(0, |d| d.len());
        let complexity = (text_length as f32 / 500.0).min(1.0);
        
        // Check for complexity indicators
        let complex_keywords = ["implement", "design", "architect", "refactor", "optimize"];
        let has_complex_keywords = complex_keywords.iter()
            .any(|k| title.to_lowercase().contains(k) || 
                     description.as_ref().map_or(false, |d| d.to_lowercase().contains(k)));
        
        if has_complex_keywords {
            (complexity + 0.3).min(1.0)
        } else {
            complexity
        }
    }
    
    /// Extract required skills
    async fn extract_required_skills(&self, title: &str, description: &Option<String>) -> Vec<String> {
        let mut skills = Vec::new();
        let text = format!("{} {}", title, description.as_ref().unwrap_or(&String::new()));
        
        // Programming languages
        for lang in &["rust", "python", "javascript", "typescript", "go", "java"] {
            if text.to_lowercase().contains(lang) {
                skills.push(lang.to_string());
            }
        }
        
        // Technologies
        for tech in &["docker", "kubernetes", "aws", "git", "database", "api"] {
            if text.to_lowercase().contains(tech) {
                skills.push(tech.to_string());
            }
        }
        
        skills
    }
    
    /// Determine agent assignments
    async fn determine_agent_assignments(&self, metadata: &TodoCognitiveMetadata) -> Vec<AgentSpecialization> {
        let mut assignments = Vec::new();
        
        if metadata.complexity_score > 0.7 {
            assignments.push(AgentSpecialization::Analytical);
        }
        
        if metadata.learning_opportunity > 0.5 {
            assignments.push(AgentSpecialization::Strategic);
        }
        
        if metadata.required_skills.iter().any(|s| s.contains("code") || s.contains("rust")) {
            assignments.push(AgentSpecialization::Technical);
        }
        
        assignments
    }
    
    /// Calculate todo progress
    fn calculate_progress(&self, todo: &TodoItem) -> f32 {
        match todo.status {
            TodoStatus::Pending => 0.0,
            TodoStatus::Ready => 0.1,
            TodoStatus::InProgress => 0.5,
            TodoStatus::Blocked => 0.3,
            TodoStatus::Review => 0.8,
            TodoStatus::Completed => 1.0,
            TodoStatus::Cancelled | TodoStatus::Failed => 0.0,
        }
    }
    
    /// Map todo status to task status
    fn map_todo_status_to_task_status(&self, status: TodoStatus) -> TaskStatus {
        match status {
            TodoStatus::Pending | TodoStatus::Ready => TaskStatus::Todo,
            TodoStatus::InProgress => TaskStatus::InProgress,
            TodoStatus::Blocked => TaskStatus::Blocked,
            TodoStatus::Review => TaskStatus::InReview,
            TodoStatus::Completed => TaskStatus::Completed,
            TodoStatus::Cancelled => TaskStatus::Cancelled,
            TodoStatus::Failed => TaskStatus::Failed,
        }
    }
    
    /// Map priority level
    fn map_priority(&self, level: PriorityLevel) -> TaskPriority {
        match level {
            PriorityLevel::Critical => TaskPriority::Critical,
            PriorityLevel::High => TaskPriority::High,
            PriorityLevel::Medium => TaskPriority::Medium,
            PriorityLevel::Low | PriorityLevel::Minimal => TaskPriority::Low,
        }
    }
    
    /// Update the priority of a todo
    pub async fn update_priority(&self, todo_id: String, priority: TodoPriority) -> Result<()> {
        let mut todos = self.todos.write().await;
        if let Some(todo) = todos.get_mut(&todo_id) {
            todo.priority = priority;
            todo.updated_at = Utc::now();
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.priority_changes += 1;
            
            tracing::info!("Updated priority for todo {}: {:?}", todo_id, todo.priority.level);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Todo {} not found", todo_id))
        }
    }
}

/// Request to create a todo
#[derive(Debug, Clone)]
pub struct CreateTodoRequest {
    pub title: String,
    pub description: Option<String>,
    pub creator: String,
    pub assignee: Option<String>,
    pub due_date: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub parent_id: Option<String>,
    pub dependency_ids: Vec<String>,
    pub priority_hint: Option<f32>,
    pub story_context: Option<TodoStoryContext>,
    pub priority: Option<PriorityLevel>,
    pub energy_required: Option<EnergyLevel>,
    pub focus_required: Option<FocusLevel>,
    pub context: Option<String>,
}

impl Default for CreateTodoRequest {
    fn default() -> Self {
        Self {
            title: String::new(),
            description: None,
            creator: "system".to_string(),
            assignee: None,
            due_date: None,
            tags: Vec::new(),
            parent_id: None,
            dependency_ids: Vec::new(),
            priority_hint: None,
            story_context: None,
            priority: None,
            energy_required: None,
            focus_required: None,
            context: None,
        }
    }
}

/// Subtask suggestion from AI
#[derive(Debug, Clone)]
struct SubtaskSuggestion {
    title: String,
    description: Option<String>,
}

/// Story context for todos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoStoryContext {
    pub story_id: String,
    pub plot_point_id: Option<String>,
    pub narrative: String,
    pub story_arc: Option<String>,
    pub related_events: Vec<String>,
}

// Default implementations for serialization
impl Default for super::pipeline::InputMapping {
    fn default() -> Self {
        use super::pipeline::{InputMapping, DataSource};
        InputMapping {
            source: DataSource::PreviousStage,
            transformations: Vec::new(),
            validation: None,
        }
    }
}

impl Default for super::pipeline::OutputMapping {
    fn default() -> Self {
        use super::pipeline::{OutputMapping, DataDestination};
        OutputMapping {
            destination: DataDestination::NextStage,
            transformations: Vec::new(),
            aggregation: None,
        }
    }
}

impl Default for super::pipeline::ErrorHandling {
    fn default() -> Self {
        use super::pipeline::{ErrorHandling, ErrorStrategy};
        ErrorHandling {
            strategy: ErrorStrategy::Retry,
            fallback_pipeline: None,
            error_handlers: HashMap::new(),
        }
    }
}