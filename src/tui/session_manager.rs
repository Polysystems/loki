use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Local, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::compute::ComputeManager;
use crate::models::MultiAgentOrchestrator;
use crate::streaming::StreamManager;

/// Session identifier
pub type SessionId = String;

#[derive(Clone)]
/// Comprehensive session management system with persistence and state tracking
pub struct SessionManager {
    /// Active sessions in memory
    active_sessions: Arc<RwLock<HashMap<SessionId, SessionState>>>,

    /// Session persistence handler
    persistence: SessionPersistence,

    
    /// Session analytics and metrics
    analytics: SessionAnalytics,

    /// External dependencies
    model_orchestrator: Arc<MultiAgentOrchestrator>,
    stream_manager: Arc<StreamManager>,
    compute_manager: Arc<ComputeManager>,

    /// Configuration
    config: SessionConfig,
}

/// Persistent session state with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Unique session identifier
    pub id: SessionId,

    /// Session metadata
    pub metadata: SessionMetadata,

    /// Current session status
    pub status: SessionStatus,

    /// Session runtime data
    pub runtime: SessionRuntime,

    /// Session configuration
    pub config: SessionConfiguration,

    /// Session analytics
    pub analytics: SessionAnalyticsData,

    /// Session checkpoints for recovery
    pub checkpoints: Vec<SessionCheckpoint>,

    /// Creation and modification times
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}

/// Session metadata and identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Human-readable session name
    pub name: String,

    /// Session description
    pub description: String,

    /// Session tags for organization
    pub tags: Vec<String>,

    /// Associated project or context
    pub project: Option<String>,

    /// User who created the session
    pub created_by: String,

    /// Session priority level
    pub priority: SessionPriority,

    /// Session category
    pub category: SessionCategory,

    /// Associated template ID if created from template
    pub template_id: Option<String>,
}

/// Current session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// Session is initializing
    Initializing,

    /// Session is active and running
    Active,

    /// Session is paused
    Paused { reason: String },

    /// Session is suspended (can be resumed)
    Suspended { reason: String },

    /// Session is stopping
    Stopping,

    /// Session has been stopped
    Stopped { reason: String },

    /// Session encountered an error
    Error { error: String, recoverable: bool },

    /// Session has completed successfully
    Completed,

    /// Session is being archived
    Archiving,

    /// Session has been archived
    Archived,
}

/// Session runtime data (not persisted frequently)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRuntime {
    /// Currently active models
    pub active_models: Vec<ModelInstanceInfo>,

    /// Current resource usage
    pub resource_usage: ResourceUsage,

    /// Active streams and connections
    pub active_streams: Vec<StreamInfo>,

    /// Current processing queue
    pub processing_queue: Vec<TaskInfo>,

    /// Recent activities
    pub recent_activities: Vec<ActivityInfo>,

    /// Runtime statistics
    pub stats: RuntimeStats,
    
    /// Session participants
    pub participants: Vec<String>,
    
    /// Message count
    pub message_count: usize,
    
    /// Active tasks
    pub active_tasks: Vec<String>,
    
    /// Shared state
    pub shared_state: HashMap<String, serde_json::Value>,
}

/// Session configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfiguration {
    /// Auto-save interval in seconds
    pub auto_save_interval: u64,

    /// Maximum runtime before auto-suspend (seconds)
    pub max_runtime: Option<u64>,

    /// Resource limits
    pub resource_limits: ResourceLimits,

    /// Session preferences
    pub preferences: SessionPreferences,

    /// Backup and recovery settings
    pub backup_settings: BackupSettings,
}

/// Session analytics and performance data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionAnalyticsData {
    /// Total runtime in seconds
    pub total_runtime: u64,

    /// Total requests processed
    pub total_requests: u64,

    /// Total cost incurred
    pub total_cost: f64,

    /// Average response time
    pub avg_response_time: f64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Resource efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,

    /// Performance trends
    pub performance_trends: PerformanceTrends,

    /// Error tracking
    pub error_tracking: ErrorTracking,
}

/// Session checkpoint for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    /// Checkpoint identifier
    pub id: String,

    /// Checkpoint creation time
    pub created_at: DateTime<Utc>,

    /// Checkpoint type
    pub checkpoint_type: CheckpointType,

    /// State snapshot
    pub state_snapshot: StateSnapshot,

    /// Description of checkpoint
    pub description: String,

    /// Whether this is an automatic checkpoint
    pub automatic: bool,
}

/// Session priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Session categories for organization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionCategory {
    Development,
    Testing,
    Production,
    Research,
    Training,
    Demo,
    Maintenance,
    Other(String),
}

/// Model instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstanceInfo {
    pub model_id: String,
    pub model_name: String,
    pub provider: String,
    pub status: String,
    pub uptime: u64,
    pub request_count: u64,
    pub last_activity: DateTime<Utc>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub gpu_usage_percent: f64,
    pub storage_usage_mb: f64,
    pub network_usage_mbps: f64,
}

/// Stream information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    pub stream_id: String,
    pub stream_type: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub data_processed: u64,
}

/// Task information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub task_id: String,
    pub task_type: String,
    pub status: String,
    pub priority: u8,
    pub created_at: DateTime<Utc>,
    pub estimated_completion: Option<DateTime<Utc>>,
}

/// Activity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityInfo {
    pub activity_type: String,
    pub description: String,
    pub timestamp: DateTime<Utc>,
    pub details: HashMap<String, serde_json::Value>,
}

/// Runtime statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeStats {
    pub uptime_seconds: u64,
    pub requests_per_minute: f64,
    pub average_latency_ms: f64,
    pub peak_memory_mb: f64,
    pub peak_cpu_percent: f64,
    pub total_errors: u64,
    pub total_warnings: u64,
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: Option<u64>,
    pub max_cpu_percent: Option<f64>,
    pub max_gpu_percent: Option<f64>,
    pub max_storage_mb: Option<u64>,
    pub max_concurrent_requests: Option<u32>,
}

/// Session preferences
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionPreferences {
    pub auto_pause_on_idle: bool,
    pub idle_timeout_minutes: u32,
    pub auto_checkpoint: bool,
    pub checkpoint_interval_minutes: u32,
    pub log_level: String,
    pub metrics_collection: bool,
}

/// Backup and recovery settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupSettings {
    pub auto_backup: bool,
    pub backup_interval_hours: u32,
    pub max_backups: u32,
    pub backup_location: Option<PathBuf>,
    pub compression_enabled: bool,
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EfficiencyMetrics {
    pub resource_utilization: f64,
    pub cost_efficiency: f64,
    pub throughput_efficiency: f64,
    pub quality_score: f64,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceTrends {
    pub response_time_trend: f64,
    pub throughput_trend: f64,
    pub error_rate_trend: f64,
    pub resource_usage_trend: f64,
}

/// Error tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorTracking {
    pub total_errors: u64,
    pub error_rate_percent: f64,
    pub last_error: Option<DateTime<Utc>>,
    pub error_categories: HashMap<String, u64>,
    pub critical_errors: u64,
}

/// Checkpoint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointType {
    Automatic,
    Manual,
    BeforeOperation,
    Emergency,
    Scheduled,
}

/// State snapshot for checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub runtime_state: SessionRuntime,
    pub configuration: SessionConfiguration,
    pub timestamp: DateTime<Utc>,
    pub size_bytes: u64,
}

#[derive(Clone)]
/// Session persistence handler
pub struct SessionPersistence {
    storage_path: PathBuf,
    backup_path: PathBuf,
}

#[derive(Clone)]
/// Session analytics engine
pub struct SessionAnalytics {
    metrics_history: HashMap<SessionId, Vec<MetricsSnapshot>>,
}

/// Metrics snapshot for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub metrics: SessionAnalyticsData,
    pub resource_usage: ResourceUsage,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub storage_path: PathBuf,
    pub auto_save_interval: u64,
    pub max_sessions: usize,
    pub cleanup_after_days: u32,
    pub enable_analytics: bool,
    pub backup_enabled: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./data/sessions"),
            auto_save_interval: 30, // 30 seconds
            max_sessions: 100,
            cleanup_after_days: 30,
            enable_analytics: true,
            backup_enabled: true,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(8192), // 8GB default
            max_cpu_percent: Some(80.0),
            max_gpu_percent: Some(90.0),
            max_storage_mb: Some(10240), // 10GB default
            max_concurrent_requests: Some(100),
        }
    }
}

impl Default for BackupSettings {
    fn default() -> Self {
        Self {
            auto_backup: true,
            backup_interval_hours: 6,
            max_backups: 10,
            backup_location: None,
            compression_enabled: true,
        }
    }
}

impl SessionManager {
    /// Create a new session manager
    pub async fn new(
        model_orchestrator: Arc<MultiAgentOrchestrator>,
        stream_manager: Arc<StreamManager>,
        compute_manager: Arc<ComputeManager>,
        config: Option<SessionConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        // Ensure storage directories exist
        fs::create_dir_all(&config.storage_path)
            .context("Failed to create session storage directory")?;

        let backup_path = config.storage_path.join("backups");
        fs::create_dir_all(&backup_path).context("Failed to create session backup directory")?;

        let manager = Self {
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            persistence: SessionPersistence {
                storage_path: config.storage_path.clone(),
                backup_path,
            },
            analytics: SessionAnalytics { metrics_history: HashMap::new() },
            model_orchestrator,
            stream_manager,
            compute_manager,
            config,
        };

        // Load existing sessions
        manager.load_sessions().await?;

        // Start background tasks
        manager.start_background_tasks().await;

        info!("Session manager initialized successfully");
        Ok(manager)
    }

    /// Create a new session
    pub async fn create_session(&self, metadata: SessionMetadata) -> Result<SessionId> {
        let session_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let session = SessionState {
            id: session_id.clone(),
            metadata,
            status: SessionStatus::Initializing,
            runtime: SessionRuntime {
                active_models: Vec::new(),
                resource_usage: ResourceUsage::default(),
                active_streams: Vec::new(),
                processing_queue: Vec::new(),
                recent_activities: Vec::new(),
                stats: RuntimeStats::default(),
                participants: Vec::new(),
                message_count: 0,
                active_tasks: Vec::new(),
                shared_state: HashMap::new(),
            },
            config: SessionConfiguration {
                auto_save_interval: self.config.auto_save_interval,
                max_runtime: None,
                resource_limits: ResourceLimits::default(),
                preferences: SessionPreferences::default(),
                backup_settings: BackupSettings::default(),
            },
            analytics: SessionAnalyticsData::default(),
            checkpoints: Vec::new(),
            created_at: now,
            modified_at: now,
            last_accessed: now,
        };

        // Add to active sessions
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id.clone(), session);
        }

        // Create initial checkpoint
        self.create_checkpoint(&session_id, CheckpointType::Automatic, "Session created").await?;

        // Persist session
        self.save_session(&session_id).await?;

        info!("Created new session: {}", session_id);
        Ok(session_id)
    }

    /// Get session by ID
    pub async fn get_session(&self, session_id: &str) -> Option<SessionState> {
        let sessions = self.active_sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<SessionState> {
        let sessions = self.active_sessions.read().await;
        sessions.values().cloned().collect()
    }

    /// Update session status
    pub async fn update_session_status(
        &self,
        session_id: &str,
        status: SessionStatus,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.status = status;
            session.modified_at = Utc::now();
            session.last_accessed = Utc::now();

            info!("Updated session {} status to {:?}", session_id, session.status);

            // Auto-save on status change
            drop(sessions);
            self.save_session(session_id).await?;
        }
        Ok(())
    }

    /// Start a session
    pub async fn start_session(&self, session_id: &str) -> Result<()> {
        self.update_session_status(session_id, SessionStatus::Active).await?;

        // Initialize session resources
        self.initialize_session_resources(session_id).await?;

        info!("Started session: {}", session_id);
        Ok(())
    }

    /// Pause a session
    pub async fn pause_session(&self, session_id: &str, reason: String) -> Result<()> {
        // Create checkpoint before pausing
        self.create_checkpoint(session_id, CheckpointType::BeforeOperation, "Before pause").await?;

        self.update_session_status(session_id, SessionStatus::Paused { reason }).await?;

        info!("Paused session: {}", session_id);
        Ok(())
    }

    /// Resume a session
    pub async fn resume_session(&self, session_id: &str) -> Result<()> {
        self.update_session_status(session_id, SessionStatus::Active).await?;

        // Restore session resources if needed
        self.restore_session_resources(session_id).await?;

        info!("Resumed session: {}", session_id);
        Ok(())
    }

    /// Stop a session
    pub async fn stop_session(&self, session_id: &str, reason: String) -> Result<()> {
        // Create final checkpoint
        self.create_checkpoint(session_id, CheckpointType::Manual, "Final checkpoint").await?;

        // Clean up session resources
        self.cleanup_session_resources(session_id).await?;

        self.update_session_status(session_id, SessionStatus::Stopped { reason }).await?;

        info!("Stopped session: {}", session_id);
        Ok(())
    }

    /// Delete a session
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        // Stop session first if active
        if let Some(session) = self.get_session(session_id).await {
            match session.status {
                SessionStatus::Active | SessionStatus::Paused { .. } => {
                    self.stop_session(session_id, "Session deleted".to_string()).await?;
                }
                _ => {}
            }
        }

        // Remove from active sessions
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(session_id);
        }

        // Delete persisted data
        self.delete_session_files(session_id).await?;

        info!("Deleted session: {}", session_id);
        Ok(())
    }

    /// Create a checkpoint
    pub async fn create_checkpoint(
        &self,
        session_id: &str,
        checkpoint_type: CheckpointType,
        description: &str,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            let checkpoint_id = Uuid::new_v4().to_string();
            let now = Utc::now();

            let state_snapshot = StateSnapshot {
                runtime_state: session.runtime.clone(),
                configuration: session.config.clone(),
                timestamp: now,
                size_bytes: 0, // Calculate actual size if needed
            };

            let automatic =
                matches!(checkpoint_type, CheckpointType::Automatic | CheckpointType::Scheduled);
            let checkpoint = SessionCheckpoint {
                id: checkpoint_id,
                created_at: now,
                checkpoint_type,
                state_snapshot,
                description: description.to_string(),
                automatic,
            };

            session.checkpoints.push(checkpoint);
            session.modified_at = now;

            // Keep only last 10 checkpoints
            if session.checkpoints.len() > 10 {
                session.checkpoints.remove(0);
            }

            debug!("Created checkpoint for session {}: {}", session_id, description);
        }
        Ok(())
    }

    /// Update session analytics
    pub async fn update_session_analytics(
        &self,
        session_id: &str,
        request_count: u64,
        response_time: f64,
        cost: f64,
        success: bool,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            let analytics = &mut session.analytics;

            // Update counters
            analytics.total_requests += request_count;
            analytics.total_cost += cost;

            // Update averages
            let total_responses = analytics.total_requests;
            if total_responses > 0 {
                analytics.avg_response_time =
                    (analytics.avg_response_time * (total_responses - 1) as f64 + response_time)
                        / total_responses as f64;
            }

            // Update success rate
            if success {
                analytics.success_rate = (analytics.success_rate * (total_responses - 1) as f64
                    + 1.0)
                    / total_responses as f64;
            } else {
                analytics.success_rate = (analytics.success_rate * (total_responses - 1) as f64)
                    / total_responses as f64;
                analytics.error_tracking.total_errors += 1;
            }

            session.modified_at = Utc::now();
            session.last_accessed = Utc::now();
        }
        Ok(())
    }

    /// Update session resource usage
    pub async fn update_resource_usage(
        &self,
        session_id: &str,
        usage: ResourceUsage,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            // Update peak values before moving usage
            let stats = &mut session.runtime.stats;
            stats.peak_memory_mb = stats.peak_memory_mb.max(usage.memory_usage_mb);
            stats.peak_cpu_percent = stats.peak_cpu_percent.max(usage.cpu_usage_percent);

            // Now move usage into the session
            session.runtime.resource_usage = usage;
            session.modified_at = Utc::now();
            session.last_accessed = Utc::now();
        }
        Ok(())
    }

    /// Load sessions from persistence
    async fn load_sessions(&self) -> Result<()> {
        let session_files = fs::read_dir(&self.persistence.storage_path)
            .context("Failed to read session storage directory")?;

        let mut loaded_count = 0;
        for entry in session_files {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.load_session_file(&path).await {
                    Ok(session) => {
                        let mut sessions = self.active_sessions.write().await;
                        sessions.insert(session.id.clone(), session);
                        loaded_count += 1;
                    }
                    Err(e) => {
                        warn!("Failed to load session from {}: {}", path.display(), e);
                    }
                }
            }
        }

        info!("Loaded {} sessions from persistence", loaded_count);
        Ok(())
    }

    /// Load a single session file
    async fn load_session_file(&self, path: &PathBuf) -> Result<SessionState> {
        let content = fs::read_to_string(path).context("Failed to read session file")?;

        let session: SessionState =
            serde_json::from_str(&content).context("Failed to deserialize session")?;

        Ok(session)
    }

    /// Save a session to persistence
    async fn save_session(&self, session_id: &str) -> Result<()> {
        let session = {
            let sessions = self.active_sessions.read().await;
            sessions.get(session_id).cloned()
        };

        if let Some(session) = session {
            let file_path = self.persistence.storage_path.join(format!("{}.json", session_id));
            let content =
                serde_json::to_string_pretty(&session).context("Failed to serialize session")?;

            fs::write(&file_path, content).context("Failed to write session file")?;

            debug!("Saved session {} to {}", session_id, file_path.display());
        }

        Ok(())
    }

    /// Delete session files
    async fn delete_session_files(&self, session_id: &str) -> Result<()> {
        let file_path = self.persistence.storage_path.join(format!("{}.json", session_id));
        if file_path.exists() {
            fs::remove_file(&file_path).context("Failed to delete session file")?;
        }

        // Also delete any backup files
        let _backup_pattern = format!("{}_*.json", session_id);
        if let Ok(entries) = fs::read_dir(&self.persistence.backup_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let file_name = entry.file_name();
                    if let Some(name) = file_name.to_str() {
                        if name.starts_with(&session_id) && name.ends_with(".json") {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Initialize session resources with local timestamp logging
    async fn initialize_session_resources(&self, session_id: &str) -> Result<()> {
        let local_time = Local::now();
        info!(
            "🚀 Initializing session resources for {} at {}",
            session_id,
            local_time.format("%Y-%m-%d %H:%M:%S")
        );

        // Try to initialize resources with error handling
        match self.stream_manager.create_session_stream(session_id).await {
            Ok(_) => {
                info!(
                    "✅ Stream initialized for session {} at local time {}",
                    session_id, local_time
                );
            }
            Err(e) => {
                error!(
                    "❌ Failed to initialize stream for session {} at {}: {}",
                    session_id, local_time, e
                );
                return Err(e.context("Stream initialization failed"));
            }
        }

        // Initialize with model orchestrator
        if let Err(e) = self.model_orchestrator.prepare_session(session_id).await {
            error!(
                "❌ Model orchestrator setup failed for session {} at {}: {}",
                session_id, local_time, e
            );
            return Err(e.context("Model orchestrator setup failed"));
        }

        info!("✅ Session {} resources fully initialized at {}", session_id, local_time);
        Ok(())
    }

    /// Restore session resources
    async fn restore_session_resources(&self, _session_id: &str) -> Result<()> {
        // Restore session state from checkpoints if needed
        Ok(())
    }

    /// Cleanup session resources with error logging
    async fn cleanup_session_resources(&self, session_id: &str) -> Result<()> {
        let local_time = Local::now();
        info!(
            "🧹 Cleaning up session resources for {} at {}",
            session_id,
            local_time.format("%Y-%m-%d %H:%M:%S")
        );

        // Clean up streams
        if let Err(e) = self.stream_manager.cleanup_session_streams(session_id).await {
            error!(
                "❌ Failed to cleanup streams for session {} at {}: {}",
                session_id, local_time, e
            );
            // Continue cleanup despite stream cleanup failure
        }

        // Clean up model resources
        if let Err(e) = self.model_orchestrator.cleanup_session(session_id).await {
            error!(
                "❌ Failed to cleanup models for session {} at {}: {}",
                session_id, local_time, e
            );
            // Continue cleanup despite model cleanup failure
        }

        info!("✅ Session {} resources cleaned up at {}", session_id, local_time);
        Ok(())
    }

    /// Start background tasks
    async fn start_background_tasks(&self) {
        let _sessions = self.active_sessions.clone();
        let config = self.config.clone();

        // Auto-save task
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(config.auto_save_interval));

            loop {
                interval.tick().await;

                // Auto-save all modified sessions
                // Implementation would check modified timestamps and save as needed
                debug!("Auto-save task tick");
            }
        });

        // Cleanup task
        let _sessions_cleanup = self.active_sessions.clone();
        let _config_cleanup = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // Every hour

            loop {
                interval.tick().await;

                // Cleanup old sessions
                debug!("Session cleanup task tick");
            }
        });
    }
}

impl Default for SessionPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl Default for SessionCategory {
    fn default() -> Self {
        Self::Development
    }
}
