use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use base64::prelude::*;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{RwLock, broadcast, mpsc, Mutex};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::cognitive::goal_manager::Priority;
use crate::cognitive::{CognitiveSystem, Thought, ThoughtMetadata, ThoughtType};
use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::tools::calendar::CalendarManager;

/// Task management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Jira configuration
    pub jira_enabled: bool,
    pub jira_url: String,
    pub jira_username: String,
    pub jira_api_token: String,
    pub jira_project_key: String,

    /// Linear configuration
    pub linear_enabled: bool,
    pub linear_api_token: String,
    pub linear_team_id: String,

    /// Asana configuration
    pub asana_enabled: bool,
    pub asana_access_token: String,
    pub asana_workspace_id: String,
    pub asana_project_id: String,

    /// Task processing settings
    pub auto_create_from_conversations: bool,
    pub auto_prioritize: bool,
    pub sync_interval: Duration,
    pub task_extraction_threshold: f32,

    /// Cognitive integration settings
    pub cognitive_awareness_level: f32,
    pub enable_deadline_tracking: bool,
    pub enable_workload_analysis: bool,
    pub enable_burnout_detection: bool,
    pub enable_progress_insights: bool,

    /// Productivity settings
    pub default_estimate_buffer: f32,
    pub context_switching_penalty: Duration,
    pub deep_work_blocks: Vec<String>, // Time blocks for focused work
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            jira_enabled: false,
            jira_url: String::new(),
            jira_username: String::new(),
            jira_api_token: String::new(),
            jira_project_key: String::new(),
            linear_enabled: false,
            linear_api_token: String::new(),
            linear_team_id: String::new(),
            asana_enabled: false,
            asana_access_token: String::new(),
            asana_workspace_id: String::new(),
            asana_project_id: String::new(),
            auto_create_from_conversations: true,
            auto_prioritize: true,
            sync_interval: Duration::from_secs(300), // 5 minutes
            task_extraction_threshold: 0.7,
            cognitive_awareness_level: 0.8,
            enable_deadline_tracking: true,
            enable_workload_analysis: true,
            enable_burnout_detection: true,
            enable_progress_insights: true,
            default_estimate_buffer: 1.2, // 20% buffer
            context_switching_penalty: Duration::from_secs(900), // 15 minutes
            deep_work_blocks: vec!["9:00-11:00".to_string(), "14:00-16:00".to_string()],
        }
    }
}

/// Unified task structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub external_id: Option<String>,
    pub platform: TaskPlatform,
    pub title: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    pub priority: TaskPriority,
    pub assignee: Option<String>,
    pub reporter: Option<String>,
    pub labels: Vec<String>,
    pub due_date: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub estimate: Option<Duration>,
    pub time_spent: Option<Duration>,
    pub progress: f32, // 0.0 to 1.0
    pub parent_task: Option<String>,
    pub subtasks: Vec<String>,
    pub dependencies: Vec<String>,
    pub cognitive_metadata: TaskCognitiveMetadata,
    pub name: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskPlatform {
    Jira,
    Linear,
    Asana,
    Internal, // Created by Loki
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Todo,
    InProgress,
    InReview,
    Blocked,
    Done,
    Cancelled,
    Completed,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Todo => write!(f, "Todo"),
            TaskStatus::InProgress => write!(f, "In Progress"),
            TaskStatus::InReview => write!(f, "In Review"),
            TaskStatus::Blocked => write!(f, "Blocked"),
            TaskStatus::Done => write!(f, "Done"),
            TaskStatus::Cancelled => write!(f, "Cancelled"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Failed => write!(f, "Failed"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCognitiveMetadata {
    pub cognitive_priority: f32,
    pub complexity_score: f32,
    pub energy_requirement: f32,
    pub focus_requirement: f32,
    pub context_switching_cost: f32,
    pub optimal_time_blocks: Vec<String>,
    pub prerequisite_knowledge: Vec<String>,
    pub related_memories: Vec<String>,
    pub burnout_risk: f32,
    pub motivation_factors: Vec<String>,
}

/// Workload analysis
#[derive(Debug, Clone)]
pub struct WorkloadAnalysis {
    pub total_active_tasks: u32,
    pub high_priority_tasks: u32,
    pub overdue_tasks: u32,
    pub estimated_total_time: Duration,
    pub capacity_utilization: f32,
    pub burnout_risk_score: f32,
    pub context_switches_per_day: u32,
    pub productivity_trends: ProductivityTrend,
    pub bottlenecks: Vec<Bottleneck>,
    pub optimization_suggestions: Vec<WorkloadSuggestion>,
}

#[derive(Debug, Clone)]
pub struct ProductivityTrend {
    pub completion_rate: f32,
    pub velocity_trend: f32, // Tasks per week
    pub quality_score: f32,
    pub focus_time_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub description: String,
    pub impact_score: f32,
    pub affected_tasks: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    CapacityOverload,
    DependencyBlocking,
    SkillGap,
    ContextSwitching,
    ExternalWaiting,
}

#[derive(Debug, Clone)]
pub struct WorkloadSuggestion {
    pub suggestion_type: WorkloadSuggestionType,
    pub description: String,
    pub impact: String,
    pub effort_required: f32,
}

#[derive(Debug, Clone)]
pub enum WorkloadSuggestionType {
    RePrioritize,
    Delegate,
    BreakDown,
    ScheduleDeepWork,
    TakeBREAK,
    ReduceScope,
}

/// Statistics for task management
#[derive(Debug, Default, Clone)]
pub struct TaskStats {
    pub tasks_processed: u64,
    pub tasks_created: u64,
    pub tasks_completed: u64,
    pub auto_created_tasks: u64,
    pub cognitive_prioritizations: u64,
    pub workload_analyses: u64,
    pub productivity_insights: u64,
    pub average_completion_time: Duration,
    pub velocity_per_week: f32,
    pub uptime: Duration,
}

/// Main task management system
pub struct TaskManager {
    /// HTTP client for API calls
    http_client: Client,

    /// Configuration
    config: TaskConfig,

    /// Cognitive system integration
    cognitive_system: Arc<CognitiveSystem>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Calendar integration
    calendar_manager: Option<Arc<CalendarManager>>,

    /// Tasks cache
    tasks: Arc<RwLock<HashMap<String, Task>>>,

    /// Current workload analysis
    workload_analysis: Arc<RwLock<Option<WorkloadAnalysis>>>,

    /// Task processing queue
    task_tx: mpsc::Sender<Task>,
    task_rx: Arc<RwLock<Option<mpsc::Receiver<Task>>>>,

    /// Event broadcast
    event_tx: broadcast::Sender<TaskEvent>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<TaskStats>>,

    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Task management events
#[derive(Debug, Clone)]
pub enum TaskEvent {
    TaskCreated(Task),
    TaskUpdated(Task),
    TaskCompleted(Task),
    DeadlineApproaching { task_id: String, time_remaining: Duration },
    BurnoutRiskDetected { risk_score: f32, recommendations: Vec<String> },
    WorkloadOptimized { suggestions: Vec<WorkloadSuggestion> },
    ProductivityInsight { insight: String, data: Value },
    CognitiveTrigger { trigger: String, priority: Priority, context: String },
}

impl TaskManager {
    /// Create a placeholder instance for initialization
    pub fn placeholder() -> Self {
        use std::collections::HashMap;
        use std::sync::Arc;
        use tokio::sync::{RwLock, mpsc, broadcast};
        use reqwest::Client;
        
        let (task_tx, _task_rx) = mpsc::channel(1000);
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);
        
        Self {
            http_client: Client::new(),
            config: TaskConfig::default(),
            cognitive_system: Arc::new(CognitiveSystem::placeholder()),
            memory: Arc::new(CognitiveMemory::placeholder()),
            calendar_manager: None,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            workload_analysis: Arc::new(RwLock::new(None)),
            task_tx,
            task_rx: Arc::new(RwLock::new(None)),
            event_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(TaskStats::default())),
            running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Create new task manager
    pub async fn new(
        config: TaskConfig,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        calendar_manager: Option<Arc<CalendarManager>>,
    ) -> Result<Self> {
        info!("Initializing task management system");

        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Loki-Consciousness/1.0")
            .build()?;

        let (task_tx, task_rx) = mpsc::channel(1000);
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            http_client,
            config,
            cognitive_system,
            memory,
            calendar_manager,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            workload_analysis: Arc::new(RwLock::new(None)),
            task_tx,
            task_rx: Arc::new(RwLock::new(Some(task_rx))),
            event_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(TaskStats::default())),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the task manager
    pub async fn start(&self) -> Result<()> {
        if *self.running.read().await {
            return Ok(());
        }

        info!("Starting task management system");
        *self.running.write().await = true;

        // Start task synchronization
        self.start_sync_loop().await?;

        // Start task processing
        self.start_task_processor().await?;

        // Start cognitive integration
        self.start_cognitive_integration().await?;

        // Start workload analysis
        self.start_workload_analyzer().await?;

        // Start periodic tasks
        self.start_periodic_tasks().await?;

        Ok(())
    }

    /// Create a new task
    pub async fn create_task(
        &self,
        title: &str,
        description: Option<&str>,
        priority: TaskPriority,
        due_date: Option<DateTime<Utc>>,
        platform: TaskPlatform,
    ) -> Result<Task> {
        info!("Creating new task: {}", title);

        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            external_id: None,
            platform,
            title: title.to_string(),
            description: description.map(|s| s.to_string()),
            status: TaskStatus::Todo,
            priority,
            assignee: Some("loki".to_string()),
            reporter: Some("loki".to_string()),
            labels: Vec::new(),
            due_date,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            estimate: None,
            time_spent: None,
            progress: 0.0,
            parent_task: None,
            subtasks: Vec::new(),
            dependencies: Vec::new(),
            cognitive_metadata: self.analyze_task_cognitively(title, description).await?,
            name: title.to_string(),
            metadata: serde_json::json!({}),
        };

        // Store task
        self.tasks.write().await.insert(task.id.clone(), task.clone());

        // Create calendar event if due date exists
        if let (Some(due_date), Some(calendar)) = (&task.due_date, &self.calendar_manager) {
            let _ = calendar
                .create_event(
                    &format!("Task Due: {}", task.title),
                    *due_date - chrono::Duration::hours(1), // 1 hour before due
                    *due_date,
                    Some("Task deadline reminder"),
                    vec![],
                )
                .await;
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.tasks_created += 1;
        }

        // Emit event
        let _ = self.event_tx.send(TaskEvent::TaskCreated(task.clone()));

        Ok(task)
    }

    /// Analyze task cognitively
    async fn analyze_task_cognitively(
        &self,
        title: &str,
        description: Option<&str>,
    ) -> Result<TaskCognitiveMetadata> {
        let content = format!("{} {}", title, description.unwrap_or(""));
        let content_lower = content.to_lowercase();

        // Simple cognitive analysis for now
        let complexity_score =
            if content_lower.contains("research") || content_lower.contains("analyze") {
                0.8
            } else if content_lower.contains("implement") || content_lower.contains("develop") {
                0.7
            } else if content_lower.contains("review") || content_lower.contains("test") {
                0.5
            } else {
                0.3
            };

        let energy_requirement =
            if content_lower.contains("creative") || content_lower.contains("design") {
                0.8
            } else if content_lower.contains("debug") || content_lower.contains("troubleshoot") {
                0.9
            } else {
                0.5
            };

        let focus_requirement =
            if content_lower.contains("deep") || content_lower.contains("complex") {
                0.9
            } else if content_lower.contains("meeting") || content_lower.contains("discuss") {
                0.3
            } else {
                0.6
            };

        Ok(TaskCognitiveMetadata {
            cognitive_priority: complexity_score * 0.5
                + energy_requirement * 0.3
                + focus_requirement * 0.2,
            complexity_score,
            energy_requirement,
            focus_requirement,
            context_switching_cost: if focus_requirement > 0.7 { 0.8 } else { 0.3 },
            optimal_time_blocks: if focus_requirement > 0.7 {
                self.config.deep_work_blocks.clone()
            } else {
                vec!["any".to_string()]
            },
            prerequisite_knowledge: Vec::new(),
            related_memories: Vec::new(),
            burnout_risk: if energy_requirement > 0.8 && complexity_score > 0.7 {
                0.6
            } else {
                0.2
            },
            motivation_factors: vec!["completion".to_string(), "learning".to_string()],
        })
    }

    /// Extract tasks from conversation text
    pub async fn extract_tasks_from_conversation(&self, text: &str) -> Result<Vec<Task>> {
        let mut extracted_tasks = Vec::new();

        if !self.config.auto_create_from_conversations {
            return Ok(extracted_tasks);
        }

        // Simple task extraction patterns
        let task_indicators = [
            "need to",
            "should",
            "must",
            "todo",
            "task",
            "action item",
            "deadline",
            "by",
            "before",
            "complete",
            "finish",
            "deliver",
        ];

        let sentences: Vec<&str> = text.split('.').collect();

        for sentence in sentences {
            let sentence_lower = sentence.to_lowercase();

            // Check if sentence contains task indicators
            let task_score = task_indicators
                .iter()
                .map(|&indicator| if sentence_lower.contains(indicator) { 1.0 } else { 0.0 })
                .sum::<f32>()
                / task_indicators.len() as f32;

            if task_score >= self.config.task_extraction_threshold {
                let title = sentence.trim().to_string();

                if !title.is_empty() && title.len() > 10 {
                    // Minimum meaningful length
                    let task = self
                        .create_task(
                            &title,
                            Some("Extracted from conversation"),
                            TaskPriority::Medium,
                            None,
                            TaskPlatform::Internal,
                        )
                        .await?;

                    extracted_tasks.push(task);

                    // Update statistics
                    {
                        let mut stats = self.stats.write().await;
                        stats.auto_created_tasks += 1;
                    }
                }
            }
        }

        Ok(extracted_tasks)
    }

    /// Start task synchronization loop
    async fn start_sync_loop(&self) -> Result<()> {
        let config = self.config.clone();
        let http_client = self.http_client.clone();
        let tasks = self.tasks.clone();
        let task_tx = self.task_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::sync_loop(config, http_client, tasks, task_tx, shutdown_rx).await;
        });

        Ok(())
    }

    /// Task synchronization loop
    async fn sync_loop(
        config: TaskConfig,
        http_client: Client,
        tasks: Arc<RwLock<HashMap<String, Task>>>,
        task_tx: mpsc::Sender<Task>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Starting task sync loop");
        let mut interval = interval(config.sync_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = Self::sync_tasks(&config, &http_client, &tasks, &task_tx).await {
                        warn!("Task sync error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Task sync loop shutting down");
                    break;
                }
            }
        }
    }

    /// Synchronize tasks from external platforms
    async fn sync_tasks(
        config: &TaskConfig,
        http_client: &Client,
        tasks: &Arc<RwLock<HashMap<String, Task>>>,
        task_tx: &mpsc::Sender<Task>,
    ) -> Result<()> {
        debug!("Synchronizing tasks from external platforms");

        // Sync Jira tasks
        if config.jira_enabled {
            if let Err(e) = Self::sync_jira_tasks(config, http_client, tasks, task_tx).await {
                warn!("Jira sync error: {}", e);
            }
        }

        // Sync Linear tasks
        if config.linear_enabled {
            if let Err(e) = Self::sync_linear_tasks(config, http_client, tasks, task_tx).await {
                warn!("Linear sync error: {}", e);
            }
        }

        // Sync Asana tasks
        if config.asana_enabled {
            if let Err(e) = Self::sync_asana_tasks(config, http_client, tasks, task_tx).await {
                warn!("Asana sync error: {}", e);
            }
        }

        Ok(())
    }

    /// Sync Jira tasks
    async fn sync_jira_tasks(
        config: &TaskConfig,
        http_client: &Client,
        tasks: &Arc<RwLock<HashMap<String, Task>>>,
        task_tx: &mpsc::Sender<Task>,
    ) -> Result<()> {
        let url = format!("{}/rest/api/3/search", config.jira_url);
        let auth = base64::prelude::BASE64_STANDARD
            .encode(format!("{}:{}", config.jira_username, config.jira_api_token));

        let jql = format!("project = {} ORDER BY updated DESC", config.jira_project_key);

        let response = http_client
            .post(&url)
            .header("Authorization", format!("Basic {}", auth))
            .header("Content-Type", "application/json")
            .json(&json!({
                "jql": jql,
                "maxResults": 50,
                "fields": ["summary", "description", "status", "priority", "assignee", "duedate"]
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("Jira API error: {}", response.status()));
        }

        let data: Value = response.json().await?;

        if let Some(issues) = data["issues"].as_array() {
            for issue in issues {
                if let Ok(task) = Self::parse_jira_issue(issue) {
                    tasks.write().await.insert(task.id.clone(), task.clone());
                    let _ = task_tx.send(task).await;
                }
            }
        }

        Ok(())
    }

    /// Parse Jira issue to task
    fn parse_jira_issue(issue: &Value) -> Result<Task> {
        let fields = &issue["fields"];

        Ok(Task {
            id: uuid::Uuid::new_v4().to_string(),
            external_id: issue["key"].as_str().map(|s| s.to_string()),
            platform: TaskPlatform::Jira,
            title: fields["summary"].as_str().unwrap_or("Untitled").to_string(),
            description: fields["description"].as_str().map(|s| s.to_string()),
            status: match fields["status"]["name"].as_str() {
                Some("To Do") => TaskStatus::Todo,
                Some("In Progress") => TaskStatus::InProgress,
                Some("Done") => TaskStatus::Done,
                _ => TaskStatus::Todo,
            },
            priority: match fields["priority"]["name"].as_str() {
                Some("Critical") => TaskPriority::Critical,
                Some("High") => TaskPriority::High,
                Some("Medium") => TaskPriority::Medium,
                Some("Low") => TaskPriority::Low,
                _ => TaskPriority::Medium,
            },
            assignee: fields["assignee"]["displayName"].as_str().map(|s| s.to_string()),
            reporter: None,
            labels: Vec::new(),
            due_date: fields["duedate"]
                .as_str()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            estimate: None,
            time_spent: None,
            progress: if matches!(fields["status"]["name"].as_str(), Some("Done")) {
                1.0
            } else {
                0.0
            },
            parent_task: None,
            subtasks: Vec::new(),
            dependencies: Vec::new(),
            cognitive_metadata: TaskCognitiveMetadata {
                cognitive_priority: 0.5,
                complexity_score: 0.5,
                energy_requirement: 0.5,
                focus_requirement: 0.5,
                context_switching_cost: 0.3,
                optimal_time_blocks: vec!["any".to_string()],
                prerequisite_knowledge: Vec::new(),
                related_memories: Vec::new(),
                burnout_risk: 0.2,
                motivation_factors: Vec::new(),
            },
            name: fields["summary"].as_str().unwrap_or("Untitled").to_string(),
            metadata: issue.clone(),
        })
    }

    /// Sync Linear tasks (simplified)
    async fn sync_linear_tasks(
        _config: &TaskConfig,
        _http_client: &Client,
        _tasks: &Arc<RwLock<HashMap<String, Task>>>,
        _task_tx: &mpsc::Sender<Task>,
    ) -> Result<()> {
        // Linear GraphQL API implementation would go here
        debug!("Syncing Linear tasks");
        Ok(())
    }

    /// Sync Asana tasks (simplified)
    async fn sync_asana_tasks(
        _config: &TaskConfig,
        _http_client: &Client,
        _tasks: &Arc<RwLock<HashMap<String, Task>>>,
        _task_tx: &mpsc::Sender<Task>,
    ) -> Result<()> {
        // Asana REST API implementation would go here
        debug!("Syncing Asana tasks");
        Ok(())
    }

    /// Start task processing loop
    async fn start_task_processor(&self) -> Result<()> {
        let task_rx = {
            let mut rx_lock = self.task_rx.write().await;
            rx_lock.take().ok_or_else(|| anyhow!("Task receiver already taken"))?
        };

        let cognitive_system = self.cognitive_system.clone();
        let memory = self.memory.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let event_tx = self.event_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::task_processing_loop(
                task_rx,
                cognitive_system,
                memory,
                config,
                stats,
                event_tx,
                shutdown_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Task processing loop
    async fn task_processing_loop(
        mut task_rx: mpsc::Receiver<Task>,
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        config: TaskConfig,
        stats: Arc<RwLock<TaskStats>>,
        event_tx: broadcast::Sender<TaskEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Task processing loop started");

        loop {
            tokio::select! {
                Some(task) = task_rx.recv() => {
                    if let Err(e) = Self::process_task(
                        task,
                        &cognitive_system,
                        &memory,
                        &config,
                        &stats,
                        &event_tx,
                    ).await {
                        error!("Task processing error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Task processing loop shutting down");
                    break;
                }
            }
        }
    }

    /// Process individual task with comprehensive configuration-driven features
    async fn process_task(
        task: Task,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        config: &TaskConfig,
        stats: &Arc<RwLock<TaskStats>>,
        event_tx: &broadcast::Sender<TaskEvent>,
    ) -> Result<()> {
        debug!("Processing task: {}", task.title);

        // **CONFIGURATION-DRIVEN TASK FILTERING**

        // Only process tasks if cognitive awareness is enabled
        if config.cognitive_awareness_level < 0.1 {
            debug!(
                "Task cognitive awareness level too low ({:.2}), skipping task processing",
                config.cognitive_awareness_level
            );
            return Ok(());
        }

        // Update statistics
        {
            let mut stats_lock = stats.write().await;
            stats_lock.tasks_processed += 1;
        }

        // **AUTO-PRIORITIZATION CONFIGURATION**
        let mut processed_task = task.clone();
        if config.auto_prioritize {
            // Adjust priority based on cognitive metadata and configuration
            let priority_score =
                task.cognitive_metadata.cognitive_priority * config.cognitive_awareness_level;

            if priority_score > 0.8 {
                processed_task.priority = TaskPriority::Critical;
            } else if priority_score > 0.6 {
                processed_task.priority = TaskPriority::High;
            } else if priority_score > 0.4 {
                processed_task.priority = TaskPriority::Medium;
            } else {
                processed_task.priority = TaskPriority::Low;
            }

            debug!(
                "Auto-prioritized task {} to {:?} (score: {:.2})",
                task.title, processed_task.priority, priority_score
            );
        }

        // **COGNITIVE LOAD AND COMPLEXITY ANALYSIS**
        let context_switch_impact = if task.cognitive_metadata.context_switching_cost > 0.5 {
            config.context_switching_penalty.as_secs_f32() / 3600.0 // Convert to hours
        } else {
            0.0
        };

        let estimated_duration = task.estimate.unwrap_or(Duration::from_secs(3600)); // Default 1 hour
        let buffered_estimate = Duration::from_secs_f32(
            estimated_duration.as_secs_f32() * (1.0 + config.default_estimate_buffer),
        );

        // **DEEP WORK OPTIMIZATION**
        let requires_deep_work = task.cognitive_metadata.focus_requirement > 0.7
            || task.cognitive_metadata.complexity_score > 0.6;

        let optimal_work_blocks = if requires_deep_work && !config.deep_work_blocks.is_empty() {
            task.cognitive_metadata.optimal_time_blocks.clone()
        } else {
            vec!["anytime".to_string()]
        };

        // Store in memory with configuration-aware metadata
        memory
            .store(
                format!("Task: {} (Priority: {:?})", processed_task.title, processed_task.priority),
                vec![
                    processed_task.description.clone().unwrap_or_default(),
                    format!(
                        "Original priority: {:?}, Auto-prioritized: {}",
                        task.priority, config.auto_prioritize
                    ),
                    format!(
                        "Complexity: {:.2}, Focus req: {:.2}",
                        task.cognitive_metadata.complexity_score,
                        task.cognitive_metadata.focus_requirement
                    ),
                    format!(
                        "Context switch cost: {:.2}, Penalty: {:?}",
                        task.cognitive_metadata.context_switching_cost,
                        config.context_switching_penalty
                    ),
                    format!(
                        "Buffered estimate: {:?} (buffer: {:.0}%)",
                        buffered_estimate,
                        config.default_estimate_buffer * 100.0
                    ),
                    format!("Deep work blocks: {:?}", optimal_work_blocks),
                ],
                MemoryMetadata {
                    source: format!("task_management_{:?}", processed_task.platform),
                    tags: vec![
                        "task".to_string(),
                        format!("{:?}", processed_task.platform),
                        format!("{:?}", processed_task.priority),
                        format!("complexity_{:.1}", task.cognitive_metadata.complexity_score),
                        if requires_deep_work { "deep_work_required" } else { "regular_work" }
                            .to_string(),
                        format!("awareness_{:.1}", config.cognitive_awareness_level),
                    ],
                    importance: processed_task.cognitive_metadata.cognitive_priority
                        * config.cognitive_awareness_level,
                    associations: vec![],
                    context: Some("Task management".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "task_management".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        // **COGNITIVE PROCESSING WITH CONFIGURATION AWARENESS**

        // Scale importance by configuration and cognitive factors
        let cognitive_importance =
            processed_task.cognitive_metadata.cognitive_priority * config.cognitive_awareness_level;

        // Create thought for cognitive processing with enhanced metadata
        let thought = Thought {
            id: crate::cognitive::ThoughtId::new(),
            content: format!(
                "Task to consider: {} (Est: {:?}, Complexity: {:.2})",
                processed_task.title, buffered_estimate, task.cognitive_metadata.complexity_score
            ),
            thought_type: match processed_task.priority {
                TaskPriority::Critical => ThoughtType::Decision,
                TaskPriority::High => ThoughtType::Action,
                _ => ThoughtType::Planning,
            },
            metadata: ThoughtMetadata {
                source: format!("task_management_{:?}", processed_task.platform),
                confidence: 0.9 * config.cognitive_awareness_level,
                emotional_valence: if task.cognitive_metadata.burnout_risk > 0.7 {
                    -0.3
                } else if requires_deep_work {
                    0.2 // Positive for challenging work
                } else {
                    0.1
                },
                importance: cognitive_importance,
                tags: vec![
                    "task".to_string(),
                    "productivity".to_string(),
                    format!("priority_{:?}", processed_task.priority),
                    if requires_deep_work { "deep_work" } else { "regular_work" }.to_string(),
                ],
            },
            parent: None,
            children: Vec::new(),
            timestamp: Instant::now(),
        };

        // Send to cognitive system if awareness is high enough
        if cognitive_importance > 0.3 {
            if let Err(e) = cognitive_system.process_query(&thought.content).await {
                warn!("Failed to process task thought: {}", e);
            } else {
                debug!("Task processed cognitively with importance {:.2}", cognitive_importance);
            }
        }

        // **DEADLINE TRACKING CONFIGURATION**
        if config.enable_deadline_tracking {
            if let Some(due_date) = processed_task.due_date {
                let time_remaining = due_date - Utc::now();

                // Calculate deadline urgency based on task complexity and configuration
                let urgency_threshold = if task.cognitive_metadata.complexity_score > 0.7 {
                    chrono::Duration::days(2) // More lead time for complex tasks
                } else {
                    chrono::Duration::days(1)
                };

                if time_remaining < urgency_threshold && time_remaining > chrono::Duration::zero() {
                    let urgency_score = 1.0
                        - (time_remaining.num_hours() as f32
                            / urgency_threshold.num_hours() as f32);

                    let _ = event_tx.send(TaskEvent::DeadlineApproaching {
                        task_id: processed_task.id.clone(),
                        time_remaining: time_remaining.to_std().unwrap_or_default(),
                    });

                    debug!(
                        "Deadline approaching for task {} (urgency: {:.2})",
                        processed_task.title, urgency_score
                    );
                }
            }
        }

        // **BURNOUT DETECTION CONFIGURATION**
        if config.enable_burnout_detection && task.cognitive_metadata.burnout_risk > 0.6 {
            let recommendations = vec![
                format!("Consider breaking down complex task: {}", processed_task.title),
                "Schedule task during high-energy periods".to_string(),
                "Add buffer time for complexity".to_string(),
                if requires_deep_work {
                    "Schedule during deep work blocks".to_string()
                } else {
                    "Consider batching with similar tasks".to_string()
                },
            ];

            let _ = event_tx.send(TaskEvent::BurnoutRiskDetected {
                risk_score: task.cognitive_metadata.burnout_risk,
                recommendations,
            });
        }

        // **PROGRESS INSIGHTS CONFIGURATION**
        if config.enable_progress_insights {
            // Generate insights based on task characteristics and configuration
            let mut insights = Vec::new();

            if requires_deep_work && !config.deep_work_blocks.is_empty() {
                insights.push(format!(
                    "Task '{}' requires deep focus - schedule during: {:?}",
                    processed_task.title, config.deep_work_blocks
                ));
            }

            if context_switch_impact > 0.5 {
                insights.push(format!(
                    "High context switching cost ({:.1}h penalty) - batch similar tasks",
                    context_switch_impact
                ));
            }

            if config.auto_prioritize && processed_task.priority != task.priority {
                insights.push(format!(
                    "Auto-reprioritized from {:?} to {:?} based on cognitive analysis",
                    task.priority, processed_task.priority
                ));
            }

            for insight in insights {
                let _ = event_tx.send(TaskEvent::ProductivityInsight {
                    insight: insight.clone(),
                    data: serde_json::json!({
                        "task_id": processed_task.id,
                        "complexity": task.cognitive_metadata.complexity_score,
                        "focus_requirement": task.cognitive_metadata.focus_requirement,
                        "burnout_risk": task.cognitive_metadata.burnout_risk,
                        "auto_prioritized": config.auto_prioritize,
                        "deep_work_required": requires_deep_work,
                    }),
                });
            }
        }

        // **AUTO-CREATION FROM CONVERSATIONS**
        if config.auto_create_from_conversations
            && processed_task.platform == TaskPlatform::Internal
        {
            // Update statistics for auto-created tasks
            let mut stats_lock = stats.write().await;
            stats_lock.auto_created_tasks += 1;
            stats_lock.cognitive_prioritizations += if config.auto_prioritize { 1 } else { 0 };
        }

        Ok(())
    }

    /// Start cognitive integration loop
    async fn start_cognitive_integration(&self) -> Result<()> {
        let cognitive_system = self.cognitive_system.clone();
        let memory = self.memory.clone();
        let config = self.config.clone();
        let event_rx = self.event_tx.subscribe();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::cognitive_integration_loop(
                cognitive_system,
                memory,
                config,
                event_rx,
                shutdown_rx,
            )
            .await;
        });

        Ok(())
    }

    /// Cognitive integration loop
    async fn cognitive_integration_loop(
        cognitive_system: Arc<CognitiveSystem>,
        memory: Arc<CognitiveMemory>,
        config: TaskConfig,
        mut event_rx: broadcast::Receiver<TaskEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        info!("Task cognitive integration loop started");

        loop {
            tokio::select! {
                Ok(event) = event_rx.recv() => {
                    if let Err(e) = Self::handle_task_cognitive_event(
                        event,
                        &cognitive_system,
                        &memory,
                        &config,
                    ).await {
                        warn!("Task cognitive event handling error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Task cognitive integration loop shutting down");
                    break;
                }
            }
        }
    }

    /// Handle task cognitive events with comprehensive configuration-driven
    /// processing
    async fn handle_task_cognitive_event(
        event: TaskEvent,
        cognitive_system: &Arc<CognitiveSystem>,
        memory: &Arc<CognitiveMemory>,
        config: &TaskConfig,
    ) -> Result<()> {
        // **CONFIGURATION-DRIVEN EVENT FILTERING**

        // Only process events if cognitive awareness is enabled
        if config.cognitive_awareness_level < 0.1 {
            debug!(
                "Task cognitive awareness level too low ({:.2}), skipping event processing",
                config.cognitive_awareness_level
            );
            return Ok(());
        }

        match event {
            TaskEvent::DeadlineApproaching { task_id, time_remaining } => {
                warn!("Deadline approaching for task: {} in {:?}", task_id, time_remaining);

                // **DEADLINE TRACKING CONFIGURATION CHECK**
                if !config.enable_deadline_tracking {
                    debug!("Deadline tracking disabled, skipping deadline processing");
                    return Ok(());
                }

                // Scale urgency by cognitive awareness and remaining time
                let hours_remaining = time_remaining.as_secs_f32() / 3600.0;
                let urgency_score =
                    (1.0 - (hours_remaining / 24.0).min(1.0)) * config.cognitive_awareness_level;

                // Create urgent thought with configuration-aware metadata
                let thought = Thought {
                    id: crate::cognitive::ThoughtId::new(),
                    content: format!(
                        "URGENT: Task {} deadline in {:?} (urgency: {:.2})",
                        task_id, time_remaining, urgency_score
                    ),
                    thought_type: ThoughtType::Decision,
                    metadata: ThoughtMetadata {
                        source: "task_deadline".to_string(),
                        confidence: config.cognitive_awareness_level,
                        emotional_valence: -0.4 * urgency_score, // Stress from deadline
                        importance: 0.9 * urgency_score,
                        tags: vec![
                            "urgent".to_string(),
                            "deadline".to_string(),
                            format!("awareness_{:.1}", config.cognitive_awareness_level),
                        ],
                    },
                    parent: None,
                    children: Vec::new(),
                    timestamp: Instant::now(),
                };

                // Store deadline context with configuration details
                memory
                    .store(
                        format!(
                            "Approaching deadline: {} ({:?} remaining)",
                            task_id, time_remaining
                        ),
                        vec![
                            format!("Urgency score: {:.2}", urgency_score),
                            format!(
                                "Deadline tracking enabled: {}",
                                config.enable_deadline_tracking
                            ),
                            format!("Deep work blocks available: {:?}", config.deep_work_blocks),
                            format!(
                                "Context switching penalty: {:?}",
                                config.context_switching_penalty
                            ),
                        ],
                        MemoryMetadata {
                            source: "task_deadline".to_string(),
                            tags: vec![
                                "deadline".to_string(),
                                "time_management".to_string(),
                                "urgency".to_string(),
                            ],
                            importance: urgency_score,
                            associations: vec![],
                            context: Some("Task deadline tracking".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "task_management".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;

                // Only send to cognitive system if urgency meets threshold
                if urgency_score > 0.3 {
                    cognitive_system.process_query(&thought.content).await?;
                    debug!("Deadline processed with urgency {:.2}", urgency_score);
                }
            }

            TaskEvent::BurnoutRiskDetected { risk_score, recommendations } => {
                warn!("Burnout risk detected: {:.2}", risk_score);

                // **BURNOUT DETECTION CONFIGURATION CHECK**
                if !config.enable_burnout_detection {
                    debug!("Burnout detection disabled, skipping burnout processing");
                    return Ok(());
                }

                // Scale risk importance by configuration awareness
                let adjusted_risk = risk_score * config.cognitive_awareness_level;

                // Add configuration-specific recommendations
                let mut enhanced_recommendations = recommendations.clone();
                if !config.deep_work_blocks.is_empty() {
                    enhanced_recommendations.push(format!(
                        "Schedule demanding tasks during deep work blocks: {:?}",
                        config.deep_work_blocks
                    ));
                }
                if config.context_switching_penalty > Duration::from_secs(900) {
                    // 15 minutes
                    enhanced_recommendations
                        .push("Minimize context switching - batch similar tasks".to_string());
                }
                if config.default_estimate_buffer > 0.2 {
                    enhanced_recommendations.push(format!(
                        "Use time buffers ({:.0}%) to reduce pressure",
                        config.default_estimate_buffer * 100.0
                    ));
                }

                // Store recommendations in memory with configuration context
                memory
                    .store(
                        format!(
                            "Burnout risk detected: {:.2} (adjusted: {:.2})",
                            risk_score, adjusted_risk
                        ),
                        enhanced_recommendations.clone(),
                        MemoryMetadata {
                            source: "burnout_detection".to_string(),
                            tags: vec![
                                "wellbeing".to_string(),
                                "burnout".to_string(),
                                "productivity_health".to_string(),
                                format!("risk_level_{:.1}", risk_score),
                            ],
                            importance: adjusted_risk,
                            associations: vec![],
                            context: Some("Burnout risk detection".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "task_management".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;

                // Create thought for high-risk situations
                if adjusted_risk > 0.5 {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!(
                            "BURNOUT RISK: Score {:.2} - immediate intervention needed",
                            adjusted_risk
                        ),
                        thought_type: ThoughtType::Decision,
                        metadata: ThoughtMetadata {
                            source: "burnout_alert".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: -0.5, // Negative for health concern
                            importance: adjusted_risk,
                            tags: vec![
                                "health".to_string(),
                                "burnout_prevention".to_string(),
                                "urgent_action".to_string(),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                }

                debug!(
                    "Burnout risk processed: original {:.2}, adjusted {:.2}",
                    risk_score, adjusted_risk
                );
            }

            TaskEvent::WorkloadOptimized { suggestions } => {
                info!("Workload optimization completed with {} suggestions", suggestions.len());

                // **WORKLOAD ANALYSIS CONFIGURATION CHECK**
                if !config.enable_workload_analysis {
                    debug!("Workload analysis disabled, skipping optimization processing");
                    return Ok(());
                }

                // Process suggestions based on configuration preferences
                for suggestion in &suggestions {
                    let suggestion_importance =
                        (1.0 - suggestion.effort_required) * config.cognitive_awareness_level;

                    // Apply configuration-specific filtering
                    let should_implement = match suggestion.suggestion_type {
                        WorkloadSuggestionType::ScheduleDeepWork => {
                            !config.deep_work_blocks.is_empty()
                        }
                        WorkloadSuggestionType::TakeBREAK => config.enable_burnout_detection,
                        WorkloadSuggestionType::RePrioritize => config.auto_prioritize,
                        _ => true, // Other suggestions always considered
                    };

                    if should_implement && suggestion_importance > 0.2 {
                        memory
                            .store(
                                format!("Workload optimization: {}", suggestion.description),
                                vec![
                                    format!("Type: {:?}", suggestion.suggestion_type),
                                    format!("Impact: {}", suggestion.impact),
                                    format!("Effort: {:.2}", suggestion.effort_required),
                                    format!("Configuration supports: {}", should_implement),
                                    format!("Deep work blocks: {:?}", config.deep_work_blocks),
                                ],
                                MemoryMetadata {
                                    source: "workload_optimization".to_string(),
                                    tags: vec![
                                        "optimization".to_string(),
                                        "productivity".to_string(),
                                        format!("{:?}", suggestion.suggestion_type),
                                    ],
                                    importance: suggestion_importance,
                                    associations: vec![],
                                    context: Some("Workload optimization".to_string()),
                                    created_at: chrono::Utc::now(),
                                    accessed_count: 0,
                                    last_accessed: None,
                                    version: 1,
                                    category: "task_management".to_string(),
                                    timestamp: chrono::Utc::now(),
                                    expiration: None,
                                },
                            )
                            .await?;

                        debug!(
                            "Workload suggestion stored: {} (importance: {:.2})",
                            suggestion.description, suggestion_importance
                        );
                    }
                }

                // Create thought for high-impact optimization suggestions
                let high_impact_suggestions: Vec<_> =
                    suggestions.iter().filter(|s| s.effort_required < 0.5).collect();

                if !high_impact_suggestions.is_empty() && config.cognitive_awareness_level > 0.4 {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!(
                            "Workload optimization opportunities: {} high-impact suggestions \
                             available",
                            high_impact_suggestions.len()
                        ),
                        thought_type: ThoughtType::Analysis,
                        metadata: ThoughtMetadata {
                            source: "workload_optimization".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: 0.2, // Positive for improvement opportunities
                            importance: 0.7 * config.cognitive_awareness_level,
                            tags: vec![
                                "optimization".to_string(),
                                "productivity_improvement".to_string(),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                }
            }

            TaskEvent::ProductivityInsight { insight, data } => {
                info!("Productivity insight: {}", insight);

                // **PROGRESS INSIGHTS CONFIGURATION CHECK**
                if !config.enable_progress_insights {
                    debug!("Progress insights disabled, skipping insight processing");
                    return Ok(());
                }

                // Store insight with configuration context
                memory
                    .store(
                        format!("Productivity insight: {}", insight),
                        vec![
                            format!("Data: {}", data),
                            format!("Auto-prioritization: {}", config.auto_prioritize),
                            format!(
                                "Deep work blocks configured: {}",
                                !config.deep_work_blocks.is_empty()
                            ),
                            format!(
                                "Context switching penalty: {:?}",
                                config.context_switching_penalty
                            ),
                        ],
                        MemoryMetadata {
                            source: "productivity_insights".to_string(),
                            tags: vec![
                                "productivity".to_string(),
                                "insights".to_string(),
                                "optimization".to_string(),
                            ],
                            importance: 0.6 * config.cognitive_awareness_level,
                            associations: vec![],
                            context: Some("Productivity insights".to_string()),
                            created_at: chrono::Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: "task_management".to_string(),
                            timestamp: chrono::Utc::now(),
                            expiration: None,
                        },
                    )
                    .await?;

                debug!(
                    "Productivity insight processed with awareness level {:.2}",
                    config.cognitive_awareness_level
                );
            }

            TaskEvent::CognitiveTrigger { trigger, priority, context } => {
                info!("Task cognitive trigger: {} (priority: {:?})", trigger, priority);

                // Scale trigger importance by priority and configuration awareness
                let trigger_importance = match priority {
                    Priority::High => 0.9 * config.cognitive_awareness_level,
                    Priority::Medium => 0.6 * config.cognitive_awareness_level,
                    Priority::Low => 0.3 * config.cognitive_awareness_level,
                    Priority::Critical => 1.0 * config.cognitive_awareness_level,
                };

                // Only process if importance meets configuration-based threshold
                let min_threshold = 1.0 - config.cognitive_awareness_level; // Higher awareness = lower threshold
                if trigger_importance > min_threshold {
                    let thought = Thought {
                        id: crate::cognitive::ThoughtId::new(),
                        content: format!("Task cognitive trigger: {} - {}", trigger, context),
                        thought_type: ThoughtType::Analysis,
                        metadata: ThoughtMetadata {
                            source: "task_trigger".to_string(),
                            confidence: config.cognitive_awareness_level,
                            emotional_valence: 0.0,
                            importance: trigger_importance,
                            tags: vec![
                                "cognitive_trigger".to_string(),
                                format!("priority_{:?}", priority),
                                "task_intelligence".to_string(),
                            ],
                        },
                        parent: None,
                        children: Vec::new(),
                        timestamp: Instant::now(),
                    };

                    cognitive_system.process_query(&thought.content).await?;
                    debug!(
                        "Task cognitive trigger processed with importance {:.2}",
                        trigger_importance
                    );
                } else {
                    debug!(
                        "Task trigger importance ({:.2}) below threshold ({:.2}), skipping",
                        trigger_importance, min_threshold
                    );
                }
            }

            _ => {
                debug!(
                    "Handling other task event with awareness level {:.2}",
                    config.cognitive_awareness_level
                );
            }
        }

        Ok(())
    }

    /// Start workload analyzer
    async fn start_workload_analyzer(&self) -> Result<()> {
        let tasks = self.tasks.clone();
        let workload_analysis = self.workload_analysis.clone();
        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::workload_analysis_loop(tasks, workload_analysis, config, event_tx, shutdown_rx)
                .await;
        });

        Ok(())
    }

    /// Workload analysis loop
    async fn workload_analysis_loop(
        tasks: Arc<RwLock<HashMap<String, Task>>>,
        workload_analysis: Arc<RwLock<Option<WorkloadAnalysis>>>,
        config: TaskConfig,
        event_tx: broadcast::Sender<TaskEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut interval = interval(Duration::from_secs(3600)); // Every hour

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if config.enable_workload_analysis {
                        if let Err(e) = Self::analyze_workload(
                            &tasks,
                            &workload_analysis,
                            &config,
                            &event_tx,
                        ).await {
                            warn!("Workload analysis error: {}", e);
                        }
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Workload analysis loop shutting down");
                    break;
                }
            }
        }
    }

    /// Analyze current workload
    async fn analyze_workload(
        tasks: &Arc<RwLock<HashMap<String, Task>>>,
        workload_analysis: &Arc<RwLock<Option<WorkloadAnalysis>>>,
        config: &TaskConfig,
        event_tx: &broadcast::Sender<TaskEvent>,
    ) -> Result<()> {
        debug!("Analyzing workload");

        let tasks_lock = tasks.read().await;
        let active_tasks: Vec<_> = tasks_lock
            .values()
            .filter(|t| !matches!(t.status, TaskStatus::Done | TaskStatus::Cancelled))
            .cloned()
            .collect();

        let total_active_tasks = active_tasks.len() as u32;
        let high_priority_tasks = active_tasks
            .iter()
            .filter(|t| matches!(t.priority, TaskPriority::Critical | TaskPriority::High))
            .count() as u32;

        let overdue_tasks = active_tasks
            .iter()
            .filter(|t| t.due_date.map_or(false, |due| due < Utc::now()))
            .count() as u32;

        let estimated_total_time = active_tasks.iter().filter_map(|t| t.estimate).sum();

        // Calculate burnout risk
        let burnout_risk_score =
            active_tasks.iter().map(|t| t.cognitive_metadata.burnout_risk).sum::<f32>()
                / active_tasks.len().max(1) as f32;

        let analysis = WorkloadAnalysis {
            total_active_tasks,
            high_priority_tasks,
            overdue_tasks,
            estimated_total_time,
            capacity_utilization: (estimated_total_time.as_secs() as f32 / (8.0 * 3600.0)).min(2.0), /* 8 hours per day */
            burnout_risk_score,
            context_switches_per_day: (total_active_tasks as f32 * 0.3) as u32, // Estimate
            productivity_trends: ProductivityTrend {
                completion_rate: 0.7, // Would calculate from historical data
                velocity_trend: 5.0,  // Tasks per week
                quality_score: 0.8,
                focus_time_ratio: 0.6,
            },
            bottlenecks: Vec::new(),
            optimization_suggestions: Self::generate_workload_suggestions(
                &active_tasks,
                burnout_risk_score,
            )
            .await,
        };

        // Check for burnout risk
        if config.enable_burnout_detection && burnout_risk_score > 0.7 {
            let recommendations = vec![
                "Take regular breaks".to_string(),
                "Reduce task complexity".to_string(),
                "Delegate where possible".to_string(),
            ];

            let _ = event_tx.send(TaskEvent::BurnoutRiskDetected {
                risk_score: burnout_risk_score,
                recommendations,
            });
        }

        // Emit optimization suggestions
        if !analysis.optimization_suggestions.is_empty() {
            let _ = event_tx.send(TaskEvent::WorkloadOptimized {
                suggestions: analysis.optimization_suggestions.clone(),
            });
        }

        // Store analysis
        *workload_analysis.write().await = Some(analysis);

        Ok(())
    }

    /// Generate workload optimization suggestions
    async fn generate_workload_suggestions(
        active_tasks: &[Task],
        burnout_risk: f32,
    ) -> Vec<WorkloadSuggestion> {
        let mut suggestions = Vec::new();

        if active_tasks.len() > 20 {
            suggestions.push(WorkloadSuggestion {
                suggestion_type: WorkloadSuggestionType::RePrioritize,
                description: "High task count detected".to_string(),
                impact: "Focus on highest value tasks".to_string(),
                effort_required: 0.3,
            });
        }

        if burnout_risk > 0.7 {
            suggestions.push(WorkloadSuggestion {
                suggestion_type: WorkloadSuggestionType::TakeBREAK,
                description: "High burnout risk detected".to_string(),
                impact: "Prevent productivity decline".to_string(),
                effort_required: 0.1,
            });
        }

        let high_complexity_tasks =
            active_tasks.iter().filter(|t| t.cognitive_metadata.complexity_score > 0.7).count();

        if high_complexity_tasks > 5 {
            suggestions.push(WorkloadSuggestion {
                suggestion_type: WorkloadSuggestionType::BreakDown,
                description: "Many high-complexity tasks".to_string(),
                impact: "Improve progress visibility".to_string(),
                effort_required: 0.5,
            });
        }

        suggestions
    }

    /// Start periodic tasks
    async fn start_periodic_tasks(&self) -> Result<()> {
        let stats = self.stats.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            Self::periodic_tasks_loop(stats, shutdown_rx).await;
        });

        Ok(())
    }

    /// Periodic tasks loop
    async fn periodic_tasks_loop(
        stats: Arc<RwLock<TaskStats>>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Update statistics
                    {
                        let mut stats_lock = stats.write().await;
                        stats_lock.uptime += Duration::from_secs(300);
                    }

                    debug!("Task management periodic tasks completed");
                }

                _ = shutdown_rx.recv() => {
                    info!("Task management periodic tasks shutting down");
                    break;
                }
            }
        }
    }

    /// Get all tasks
    pub async fn get_tasks(&self) -> HashMap<String, Task> {
        self.tasks.read().await.clone()
    }

    /// Get tasks by status
    pub async fn get_tasks_by_status(&self, status: TaskStatus) -> Vec<Task> {
        self.tasks
            .read()
            .await
            .values()
            .filter(|t| std::mem::discriminant(&t.status) == std::mem::discriminant(&status))
            .cloned()
            .collect()
    }

    /// Get current workload analysis
    pub async fn get_workload_analysis(&self) -> Option<WorkloadAnalysis> {
        self.workload_analysis.read().await.clone()
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> TaskStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to task events
    pub fn subscribe_events(&self) -> broadcast::Receiver<TaskEvent> {
        self.event_tx.subscribe()
    }

    /// Shutdown the task manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down task management system");

        *self.running.write().await = false;
        let _ = self.shutdown_tx.send(());

        Ok(())
    }
    
    /// Update the status of a task
    pub async fn update_task_status(&self, task_id: &str, status: TaskStatus) -> Result<()> {
        let mut tasks = self.tasks.write().await;
        
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = status.clone();
            task.updated_at = chrono::Utc::now();
            
            // Log the status update
            info!("Updated task {} status to {:?}", task_id, status);
            
            // Store in memory for learning
            let memory_entry = serde_json::json!({
                "event": "task_status_update",
                "task_id": task_id,
                "new_status": format!("{:?}", status),
                "timestamp": chrono::Utc::now().to_rfc3339(),
            });
            
            let metadata = crate::memory::MemoryMetadata {
                source: "System".to_string(),
                tags: vec!["task".to_string(), "status_update".to_string()],
                importance: 0.7, // Medium priority
                associations: vec![],
                context: Some("Task status update".to_string()),
                created_at: chrono::Utc::now(),
                accessed_count: 0,
                last_accessed: Some(chrono::Utc::now()),
                version: 1,
                category: "task_management".to_string(),
                timestamp: chrono::Utc::now(),
                expiration: None,
            };
            let _ = self.memory.store(
                serde_json::to_string(&memory_entry)?,
                vec![format!("task_status_{}", task_id)],
                metadata
            ).await;
            
            Ok(())
        } else {
            Err(anyhow::anyhow!("Task with ID {} not found", task_id))
        }
    }
}
