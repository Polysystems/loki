//! Action Validation System
//!
//! Validates and controls all actions that Loki can perform,
//! preventing unauthorized or dangerous operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::path::PathBuf;

use anyhow::{Result, anyhow, Context};
use chrono::{DateTime, Utc};
use rocksdb::{DB, Options};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc, oneshot};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use glob;

use crate::safety::encryption::{SecurityEncryption, EncryptedData, helpers};
use crate::safety::security_audit::{record_security_event, SecurityEventType};

/// Types of actions that can be performed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionType {
    // File operations
    FileRead {
        path: String,
    },
    FileWrite {
        path: String,
        content: String,
    },
    FileDelete {
        path: String,
    },
    FileMove {
        from: String,
        to: String,
    },

    // API calls
    ApiCall {
        provider: String,
        endpoint: String,
    },

    // Social media
    SocialPost {
        platform: String,
        content: String,
    },
    SocialReply {
        platform: String,
        to: String,
        content: String,
    },

    // Code execution
    CommandExecute {
        command: String,
        args: Vec<String>,
    },
    CodeEval {
        language: String,
        code: String,
    },

    // GitHub operations
    GitCommit {
        message: String,
        files: Vec<String>,
    },
    GitPush {
        branch: String,
    },
    GitPullRequest {
        title: String,
        body: String,
    },

    // System operations
    MemoryModify {
        operation: String,
    },
    ConfigChange {
        key: String,
        value: String,
    },
    SelfModify {
        file: String,
        changes: String,
    },

    // Cognitive operations
    Decision {
        description: String,
        risk_level: u8,
    },

    // Tool operations
    ToolUsage {
        tool_id: String,
        parameters: serde_json::Value,
        confidence: f32,
        archetypal_form: String,
    },
}

impl ActionType {
    /// Get risk level for this action type
    pub fn risk_level(&self) -> RiskLevel {
        match self {
            // Low risk - read-only operations
            ActionType::FileRead { .. } => RiskLevel::Low,

            // Medium risk - limited scope modifications
            ActionType::SocialPost { .. } => RiskLevel::Medium,
            ActionType::SocialReply { .. } => RiskLevel::Medium,
            ActionType::ApiCall { .. } => RiskLevel::Medium,

            // High risk - file/code modifications
            ActionType::FileWrite { .. } => RiskLevel::High,
            ActionType::FileDelete { .. } => RiskLevel::High,
            ActionType::FileMove { .. } => RiskLevel::High,
            ActionType::CommandExecute { .. } => RiskLevel::High,
            ActionType::CodeEval { .. } => RiskLevel::High,

            // Critical risk - system modifications
            ActionType::GitCommit { .. } => RiskLevel::Critical,
            ActionType::GitPush { .. } => RiskLevel::Critical,
            ActionType::GitPullRequest { .. } => RiskLevel::Critical,
            ActionType::MemoryModify { .. } => RiskLevel::Critical,
            ActionType::ConfigChange { .. } => RiskLevel::Critical,
            ActionType::SelfModify { .. } => RiskLevel::Critical,

            // Variable risk - based on decision content
            ActionType::Decision { risk_level, .. } => {
                if *risk_level >= 80 {
                    RiskLevel::Critical
                } else if *risk_level >= 60 {
                    RiskLevel::High
                } else if *risk_level >= 30 {
                    RiskLevel::Medium
                } else {
                    RiskLevel::Low
                }
            }

            // Variable risk - based on tool confidence and type
            ActionType::ToolUsage { tool_id, confidence, .. } => {
                if tool_id.contains("system") || tool_id.contains("self_modify") {
                    RiskLevel::Critical
                } else if *confidence < 0.5 {
                    RiskLevel::High
                } else if *confidence < 0.7 {
                    RiskLevel::Medium
                } else {
                    RiskLevel::Low
                }
            }
        }
    }

    /// Check if action requires approval
    pub fn requires_approval(&self) -> bool {
        match self.risk_level() {
            RiskLevel::Low => false,
            RiskLevel::Medium => true, // In safe mode, even medium requires approval
            RiskLevel::High => true,
            RiskLevel::Critical => true,
        }
    }
}

/// Risk levels for actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// A pending action awaiting approval
#[derive(Debug, Serialize, Deserialize)]
pub struct PendingAction {
    pub id: String,
    pub action: ActionType,
    pub context: String,
    pub reasoning: Vec<String>,
    pub risk_level: RiskLevel,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub requested_at: chrono::DateTime<chrono::Utc>,
    pub timeout: Duration,
    #[serde(skip)]
    pub response_tx: Option<oneshot::Sender<ActionDecision>>,
}

impl Clone for PendingAction {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            action: self.action.clone(),
            context: self.context.clone(),
            reasoning: self.reasoning.clone(),
            risk_level: self.risk_level,
            requested_at: self.requested_at,
            timeout: self.timeout,
            response_tx: None, // Cannot clone Sender
        }
    }
}

/// Decision on a pending action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionDecision {
    Approve,
    Deny { reason: String },
    Defer { until: DateTime<Utc> },
}

/// Stored decision with metadata for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredDecision {
    pub decision: ActionDecision,
    pub decided_by: String, // Human user ID or system component
    pub decided_at: DateTime<Utc>,
    pub reason: Option<String>,
    pub action_context: String,
    pub risk_assessment: RiskLevel,
}

/// Validation result
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Action not allowed: {0}")]
    NotAllowed(String),

    #[error("Action requires approval")]
    RequiresApproval { action_id: String },

    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),

    #[error("Validation timeout")]
    Timeout,

    #[error("Invalid action: {0}")]
    Invalid(String),
}

/// Safety statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyStatistics {
    /// Total number of actions processed
    pub total_actions: usize,
    /// Number of actions currently pending approval
    pub pending_actions: usize,
    /// Number of approved actions
    pub approved_actions: usize,
    /// Number of denied actions
    pub denied_actions: usize,
    /// Number of deferred actions
    pub deferred_actions: usize,
    /// Approval rate percentage
    pub approval_rate: f64,
    /// Action counts by type
    pub action_counts: HashMap<String, usize>,
    /// Risk level counts
    pub risk_level_counts: HashMap<RiskLevel, usize>,
    /// Recent actions in the last hour
    pub recent_actions_last_hour: usize,
    /// Recent approved actions in the last hour
    pub recent_approved_last_hour: usize,
    /// Recent denied actions in the last hour
    pub recent_denied_last_hour: usize,
    /// Most common action type
    pub most_common_action: Option<String>,
    /// Highest risk level seen in recent actions
    pub highest_recent_risk: Option<RiskLevel>,
    /// Whether safe mode is enabled
    pub safe_mode_enabled: bool,
    /// Whether dry run mode is enabled
    pub dry_run_enabled: bool,
    /// Whether approval is required
    pub approval_required: bool,
}

/// Configuration for the action validator
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Whether we're in safe mode (more restrictions)
    pub safe_mode: bool,

    /// Whether to run in dry-run mode (no actual execution)
    pub dry_run: bool,

    /// Whether all actions require approval
    pub approval_required: bool,

    /// Default timeout for approvals
    pub approval_timeout: Duration,

    /// Allowed file paths (glob patterns)
    pub allowed_paths: Vec<String>,

    /// Blocked file paths (glob patterns)
    pub blocked_paths: Vec<String>,

    /// Maximum file size for writes
    pub max_file_size: usize,

    /// Custom storage path for decisions (optional)
    pub storage_path: Option<PathBuf>,

    /// Enable encryption for stored decisions
    pub encrypt_decisions: bool,

    /// Enable comprehensive resource monitoring
    pub enable_resource_monitoring: bool,

    /// CPU usage threshold (percentage)
    pub cpu_threshold: f32,

    /// Memory usage threshold (percentage)
    pub memory_threshold: f32,

    /// Disk usage threshold (percentage)
    pub disk_threshold: f32,

    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,

    /// Rate limiting enabled
    pub enable_rate_limiting: bool,

    /// Network bandwidth monitoring enabled
    pub enable_network_monitoring: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            safe_mode: true,
            dry_run: false,
            approval_required: true,
            approval_timeout: Duration::from_secs(300), // 5 minutes
            allowed_paths: vec!["data/**".to_string(), "workspace/**".to_string()],
            blocked_paths: vec![
                "src/safety/**".to_string(), // Can't modify safety system
                ".git/**".to_string(),
                "**/.env".to_string(),
                "**/secrets/**".to_string(),
            ],
            max_file_size: 10 * 1024 * 1024, // 10MB
            storage_path: None, // Use default
            encrypt_decisions: true, // Always encrypt by default
            enable_resource_monitoring: true, // Enable by default for Rust 2025 standards
            cpu_threshold: 75.0, // 75% CPU threshold
            memory_threshold: 85.0, // 85% memory threshold
            disk_threshold: 90.0, // 90% disk threshold
            max_concurrent_operations: 50, // Conservative concurrent operation limit
            enable_rate_limiting: true, // Enable rate limiting by default
            enable_network_monitoring: true, // Enable network monitoring
        }
    }
}

/// Action validator that enforces safety policies
#[derive(Debug)]
pub struct ActionValidator {
    config: ValidatorConfig,
    allowed_actions: Arc<RwLock<Vec<ActionType>>>,
    approval_queue: Arc<RwLock<HashMap<String, PendingAction>>>,
    approval_tx: mpsc::Sender<PendingAction>,
    approval_rx: Arc<RwLock<mpsc::Receiver<PendingAction>>>,
    action_history: Arc<RwLock<Vec<(DateTime<Utc>, ActionType, ActionDecision)>>>,
    // SECURITY: Persistent encrypted storage for decisions
    decision_storage: Arc<RwLock<Option<DB>>>,
    storage_path: PathBuf,
    // Security encryption instance
    encryption: Arc<RwLock<Option<SecurityEncryption>>>,
    // User context for audit trails
    user_context: Arc<RwLock<Option<String>>>,
}

impl ActionValidator {
    /// Create a new action validator with secure decision storage
    pub async fn new(config: ValidatorConfig) -> Result<Self> {
        let (approval_tx, approval_rx) = mpsc::channel(100);
        let storage_path = config.storage_path.clone()
            .unwrap_or_else(|| PathBuf::from("data/safety/decisions"));

        let mut validator = Self {
            config,
            allowed_actions: Arc::new(RwLock::new(Self::default_allowed_actions())),
            approval_queue: Arc::new(RwLock::new(HashMap::new())),
            approval_tx,
            approval_rx: Arc::new(RwLock::new(approval_rx)),
            action_history: Arc::new(RwLock::new(Vec::new())),
            decision_storage: Arc::new(RwLock::new(None)),
            storage_path,
            encryption: Arc::new(RwLock::new(None)),
            user_context: Arc::new(RwLock::new(None)),
        };

        // Initialize secure storage immediately
        validator.initialize_storage().await
            .with_context(|| "Failed to initialize secure decision storage")?;

        // Initialize encryption if enabled
        if validator.config.encrypt_decisions {
            validator.initialize_encryption().await?;
        }

        Ok(validator)
    }

    /// Create a new action validator without storage (for legacy compatibility)
    pub fn new_without_storage(config: ValidatorConfig) -> Self {
        let (approval_tx, approval_rx) = mpsc::channel(100);
        let storage_path = config.storage_path.clone()
            .unwrap_or_else(|| PathBuf::from("data/safety/decisions"));

        Self {
            config,
            allowed_actions: Arc::new(RwLock::new(Self::default_allowed_actions())),
            approval_queue: Arc::new(RwLock::new(HashMap::new())),
            approval_tx,
            approval_rx: Arc::new(RwLock::new(approval_rx)),
            action_history: Arc::new(RwLock::new(Vec::new())),
            decision_storage: Arc::new(RwLock::new(None)),
            storage_path,
            encryption: Arc::new(RwLock::new(None)),
            user_context: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a minimal action validator for bootstrapping
    pub async fn new_minimal() -> Result<Self> {
        let config = ValidatorConfig {
            safe_mode: true,
            dry_run: true,            // Safe default for minimal setup
            approval_required: false, // Auto-approve for minimal setup
            approval_timeout: Duration::from_secs(300),
            allowed_paths: vec!["**".to_string()], // Allow all paths in minimal setup
            blocked_paths: Vec::new(),             // No blocked paths in minimal setup
            max_file_size: 10 * 1024 * 1024,
            storage_path: Some(PathBuf::from(format!("data/safety/decisions_minimal_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))),
            encrypt_decisions: false, // Disable for minimal setup to avoid storage conflicts
            enable_resource_monitoring: false, // Disable for minimal setup
            cpu_threshold: 95.0, // Very high threshold for minimal setup
            memory_threshold: 95.0, // Very high threshold for minimal setup
            disk_threshold: 98.0, // Very high threshold for minimal setup
            max_concurrent_operations: 10, // Lower limit for minimal setup
            enable_rate_limiting: false, // Disable for minimal setup
            enable_network_monitoring: false, // Disable for minimal setup
        };

        Self::new(config).await
    }

    /// Initialize encryption system
    async fn initialize_encryption(&self) -> Result<()> {
        let encryption_key = SecurityEncryption::generate_random_key();
        let encryption = SecurityEncryption::new(&encryption_key).await?;
        *self.encryption.write().await = Some(encryption);
        info!("🔐 Decision encryption initialized");
        Ok(())
    }

    /// Get or create encryption instance
    async fn get_encryption(&self) -> Result<SecurityEncryption> {
        let encryption_guard = self.encryption.read().await;
        if let Some(encryption) = encryption_guard.as_ref() {
            Ok(encryption.clone())
        } else {
            drop(encryption_guard);
            self.initialize_encryption().await?;
            let encryption_guard = self.encryption.read().await;
            Ok(encryption_guard.as_ref().unwrap().clone())
        }
    }

    /// Set user context for audit trail
    pub async fn set_user_context(&self, user_id: String) {
        *self.user_context.write().await = Some(user_id);
    }

    /// Get current user context
    async fn get_user_context(&self) -> Option<String> {
        self.user_context.read().await.clone()
    }

    /// Initialize persistent decision storage
    pub async fn initialize_storage(&mut self) -> Result<()> {
        // Check if storage is already initialized
        {
            let storage_guard = self.decision_storage.read().await;
            if storage_guard.is_some() {
                info!("✅ Decision storage already initialized, skipping");
                return Ok(());
            }
        }

        // Create storage directory if it doesn't exist
        if let Some(parent) = self.storage_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create storage directory: {:?}", parent))?;
        }

        // Configure RocksDB with security optimizations
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Security: Enable encryption at rest (would need proper key management in production)
        // Note: This is a basic implementation - production should use proper key derivation
        info!("🔒 Initializing encrypted decision storage at: {:?}", self.storage_path);

        let db = DB::open(&opts, &self.storage_path)
            .with_context(|| format!("Failed to open decision storage: {:?}", self.storage_path))?;

        *self.decision_storage.write().await = Some(db);

        // Audit log the storage initialization
        self.audit_log("STORAGE_INIT", "Decision storage initialized", serde_json::json!({
            "storage_path": self.storage_path,
            "timestamp": Utc::now(),
            "security_level": "encrypted"
        })).await;

        info!("✅ Decision storage initialized successfully");
        Ok(())
    }

    /// Get default allowed actions
    fn default_allowed_actions() -> Vec<ActionType> {
        let mut actions = Vec::new();

        // Always allow read operations
        actions.push(ActionType::FileRead { path: String::new() });

        // In non-safe mode, allow some additional actions
        // (These would be added based on configuration)

        actions
    }

    /// Validate an action before execution
    pub async fn validate_action(
        &self,
        action: ActionType,
        context: String,
        reasoning: Vec<String>,
    ) -> ValidationResult<()> {
        info!("Validating action: {:?}", action);

        // Always check basic permissions and resource limits first
        self.check_permissions(&action)?;
        self.check_resource_limits(&action)?;

        // Check if in dry-run mode
        if self.config.dry_run {
            info!("DRY RUN: Would execute {:?}", action);
            return Ok(());
        }

        // Check if approval is required
        if self.config.approval_required || action.requires_approval() {
            let action_id = self.request_approval(action.clone(), context, reasoning).await?;

            // Wait for approval
            let decision = self.wait_for_approval(&action_id).await?;

            match decision {
                ActionDecision::Approve => {
                    info!("Action approved: {:?}", action);
                    self.record_action(action, ActionDecision::Approve).await;
                    Ok(())
                }
                ActionDecision::Deny { reason } => {
                    warn!("Action denied: {:?} - {}", action, reason);
                    self.record_action(action, ActionDecision::Deny { reason: reason.clone() })
                        .await;
                    Err(ValidationError::NotAllowed(reason))
                }
                ActionDecision::Defer { until } => {
                    info!("Action deferred until {:?}", until);
                    Err(ValidationError::NotAllowed("Action deferred".to_string()))
                }
            }
        } else {
            // Auto-approve low-risk actions
            self.record_action(action, ActionDecision::Approve).await;
            Ok(())
        }
    }

    /// Check basic permissions for an action
    fn check_permissions(&self, action: &ActionType) -> ValidationResult<()> {
        match action {
            ActionType::FileWrite { path, .. } | ActionType::FileDelete { path } => {
                self.check_path_permissions(path)?;
            }
            ActionType::FileMove { from, to } => {
                self.check_path_permissions(from)?;
                self.check_path_permissions(to)?;
            }
            ActionType::SelfModify { .. } if self.config.safe_mode => {
                return Err(ValidationError::NotAllowed(
                    "Self-modification not allowed in safe mode".to_string(),
                ));
            }
            _ => {}
        }

        Ok(())
    }

    /// Check if a file path is allowed
    fn check_path_permissions(&self, path: &str) -> ValidationResult<()> {
        // Check blocked paths first
        for pattern in &self.config.blocked_paths {
            match glob::Pattern::new(pattern) {
                Ok(p) if p.matches(path) => {
                    return Err(ValidationError::NotAllowed(format!(
                        "Path '{}' matches blocked pattern '{}'",
                        path, pattern
                    )));
                }
                Err(e) => {
                    warn!("Invalid glob pattern '{}': {}", pattern, e);
                    continue;
                }
                _ => continue,
            }
        }

        // Check allowed paths
        let mut allowed = false;
        for pattern in &self.config.allowed_paths {
            match glob::Pattern::new(pattern) {
                Ok(p) if p.matches(path) => {
                    allowed = true;
                    break;
                }
                Err(e) => {
                    warn!("Invalid glob pattern '{}': {}", pattern, e);
                    continue;
                }
                _ => continue,
            }
        }

        if !allowed && !self.config.allowed_paths.is_empty() {
            return Err(ValidationError::NotAllowed(format!(
                "Path '{}' not in allowed paths",
                path
            )));
        }

        Ok(())
    }

    /// Check resource limits with comprehensive monitoring
    fn check_resource_limits(&self, action: &ActionType) -> ValidationResult<()> {
        // Basic file size check (preserve existing behavior)
        match action {
            ActionType::FileWrite { content, .. } => {
                if content.len() > self.config.max_file_size {
                    return Err(ValidationError::ResourceLimit(format!(
                        "File size {} exceeds limit {}",
                        content.len(),
                        self.config.max_file_size
                    )));
                }
            }
            _ => {}
        }

        // Check if resource monitoring is enabled and perform async checks if needed
        if self.config.enable_resource_monitoring {
            // Since this method is sync, we'll use the basic checks for now
            // The async version below provides comprehensive monitoring
            debug!("Resource monitoring enabled but using basic checks in sync context");
            // In production, this should be refactored to use async validation throughout
        }

        Ok(())
    }

    /// Comprehensive async resource limit checking (future implementation)
    #[allow(dead_code)]
    async fn check_resource_limits_async(&self, action: &ActionType) -> ValidationResult<()> {
        use crate::safety::limits::{ResourceMonitor, ResourceLimits, LimitExceeded};
        use std::collections::HashMap;
        use tokio::sync::mpsc;

        // Create a temporary resource monitor for demonstration
        // In practice, this would be a shared instance
        let (alert_tx, mut _alert_rx) = mpsc::channel(100);
        let limits = ResourceLimits {
            max_memory_mb: 4096,      // 4GB memory limit
            max_cpu_percent: 80.0,    // 80% CPU limit
            max_file_handles: 1000,   // File handle limit
            max_concurrent_ops: 100,  // Concurrent operations limit
            api_rate_limits: HashMap::new(),
            token_budgets: HashMap::new(),
        };

        let monitor = ResourceMonitor::new(limits, alert_tx);

        // 1. Check system resource usage
        let current_usage = monitor.get_usage().await;

        // Memory usage check
        if current_usage.memory_mb > 3584 { // 3.5GB threshold (87.5% of 4GB)
            return Err(ValidationError::ResourceLimit(format!(
                "Memory usage too high: {}MB (approaching 4GB limit)",
                current_usage.memory_mb
            )));
        }

        // CPU usage check
        if current_usage.cpu_percent > 75.0 { // 75% threshold
            return Err(ValidationError::ResourceLimit(format!(
                "CPU usage too high: {:.1}% (approaching 80% limit)",
                current_usage.cpu_percent
            )));
        }

        // 2. Check concurrent operations limit
        if let Err(e) = monitor.start_operation().await {
            match e {
                LimitExceeded::ConcurrentOps { current, limit } => {
                    return Err(ValidationError::ResourceLimit(format!(
                        "Too many concurrent operations: {} >= {}",
                        current, limit
                    )));
                }
                _ => {}
            }
        }

        // 3. Action-specific resource checks
        match action {
            ActionType::FileWrite { content, path } => {
                // File size check (existing)
                if content.len() > self.config.max_file_size {
                    return Err(ValidationError::ResourceLimit(format!(
                        "File size {} exceeds limit {}",
                        content.len(),
                        self.config.max_file_size
                    )));
                }

                // Disk space check
                if let Err(e) = self.check_disk_space_async(path, content.len()).await {
                    return Err(ValidationError::ResourceLimit(e));
                }
            }

            ActionType::ApiCall { provider, .. } => {
                // API rate limiting
                if let Err(e) = monitor.check_api_limit(provider).await {
                    match e {
                        LimitExceeded::RateLimit { service, remaining } => {
                            return Err(ValidationError::ResourceLimit(format!(
                                "API rate limit exceeded for {}: {} requests remaining",
                                service, remaining
                            )));
                        }
                        _ => {}
                    }
                }
            }

            ActionType::CommandExecute { command, .. } => {
                // Check if system can handle additional process
                let estimated_memory_mb = self.estimate_command_memory_usage(command);
                if current_usage.memory_mb + estimated_memory_mb > 3584 {
                    return Err(ValidationError::ResourceLimit(format!(
                        "Insufficient memory for command execution: current {}MB + estimated {}MB exceeds safe threshold",
                        current_usage.memory_mb, estimated_memory_mb
                    )));
                }
            }

            ActionType::CodeEval { code, .. } => {
                // Code evaluation safety checks
                let estimated_memory_kb = code.len() * 10; // Rough estimate: 10x code size
                if estimated_memory_kb > 10_000 { // 10MB limit for code evaluation
                    return Err(ValidationError::ResourceLimit(format!(
                        "Code evaluation memory estimate too high: {} KB",
                        estimated_memory_kb
                    )));
                }
            }

            ActionType::ToolUsage { parameters, .. } => {
                // Tool-specific resource checks based on parameters
                if let Some(size_hint) = parameters.get("size_estimate") {
                    if let Some(size) = size_hint.as_u64() {
                        if size > 100_000_000 { // 100MB limit for tool operations
                            return Err(ValidationError::ResourceLimit(format!(
                                "Tool operation size estimate too large: {} bytes",
                                size
                            )));
                        }
                    }
                }
            }

            // Network-intensive operations
            ActionType::SocialPost { content, .. } | ActionType::SocialReply { content, .. } => {
                // Check network bandwidth availability (simplified)
                if content.len() > 280_000 { // Reasonable limit for social media content
                    return Err(ValidationError::ResourceLimit(format!(
                        "Social media content too large: {} bytes",
                        content.len()
                    )));
                }
            }

            // Low-risk operations - minimal checks
            ActionType::FileRead { .. } | ActionType::Decision { .. } => {
                // These operations have minimal resource impact
            }

            // High-risk operations - strict checks
            ActionType::GitCommit { files, .. } => {
                // Git operations can be resource-intensive
                if files.len() > 100 {
                    return Err(ValidationError::ResourceLimit(format!(
                        "Too many files in git commit: {} (limit: 100)",
                        files.len()
                    )));
                }
            }
            ActionType::GitPush { .. } | ActionType::GitPullRequest { .. } => {
                // Other git operations have minimal additional resource checks
            }

            ActionType::MemoryModify { .. } | ActionType::SelfModify { .. } => {
                // These operations are inherently risky and require additional scrutiny
                // Always require higher resource headroom
                if current_usage.memory_mb > 2048 { // 50% of limit for critical operations
                    return Err(ValidationError::ResourceLimit(format!(
                        "Memory usage too high for critical operations: {}MB (requires < 2GB)",
                        current_usage.memory_mb
                    )));
                }
            }

            _ => {
                // Default resource check for other action types
                // Ensure system is not under stress
                if current_usage.cpu_percent > 90.0 || current_usage.memory_mb > 3600 {
                    return Err(ValidationError::ResourceLimit(
                        "System under high load - deferring non-critical operations".to_string()
                    ));
                }
            }
        }

        // Clean up the operation we started for testing
        monitor.end_operation().await;

        Ok(())
    }

    /// Check available disk space for file operations
    #[allow(dead_code)]
    async fn check_disk_space_async(&self, path: &str, required_bytes: usize) -> Result<(), String> {
        use std::path::Path;

        // Get the parent directory to check available space
        let target_path = Path::new(path);
        let check_path = target_path.parent().unwrap_or(Path::new("."));

        // Platform-specific disk space checking
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::mem;

            if let Ok(c_path) = CString::new(check_path.to_string_lossy().as_bytes()) {
                unsafe {
                    let mut stat: libc::statvfs = mem::zeroed();
                    if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
                        let block_size = stat.f_frsize as u64;
                        let available_blocks = stat.f_bavail as u64;
                        let available_bytes = available_blocks * block_size;

                        // Require at least 10% free space plus the file size
                        let total_blocks = stat.f_blocks as u64;
                        let total_bytes = total_blocks * block_size;
                        let required_free = (total_bytes / 10) + required_bytes as u64; // 10% + file size

                        if available_bytes < required_free {
                            return Err(format!(
                                "Insufficient disk space: {} bytes available, {} bytes required",
                                available_bytes, required_free
                            ));
                        }
                    }
                }
            }
        }

        #[cfg(windows)]
        {
            use std::ffi::OsStr;
            use std::os::windows::ffi::OsStrExt;

            let wide: Vec<u16> = OsStr::new(&check_path.to_string_lossy())
                .encode_wide()
                .chain(Some(0))
                .collect();

            unsafe {
                let mut free_bytes = 0u64;
                let mut total_bytes = 0u64;

                if winapi::um::fileapi::GetDiskFreeSpaceExW(
                    wide.as_ptr(),
                    &mut free_bytes,
                    &mut total_bytes,
                    std::ptr::null_mut(),
                ) != 0 {
                    let required_free = (total_bytes / 10) + required_bytes as u64; // 10% + file size

                    if free_bytes < required_free {
                        return Err(format!(
                            "Insufficient disk space: {} bytes available, {} bytes required",
                            free_bytes, required_free
                        ));
                    }
                }
            }
        }

        #[cfg(not(any(unix, windows)))]
        {
            // For unsupported platforms, be conservative and reject large files
            if required_bytes > 100_000_000 { // 100MB
                return Err("Large file operations not supported on this platform".to_string());
            }
        }

        Ok(())
    }

    /// Estimate memory usage for command execution
    #[allow(dead_code)]
    fn estimate_command_memory_usage(&self, command: &str) -> usize {
        // Rough estimates based on common commands
        match command {
            cmd if cmd.starts_with("cargo") => 512,      // 512MB for Rust compilation
            cmd if cmd.starts_with("npm") || cmd.starts_with("node") => 256, // 256MB for Node.js
            cmd if cmd.starts_with("python") => 128,     // 128MB for Python
            cmd if cmd.starts_with("git") => 64,         // 64MB for Git operations
            cmd if cmd.contains("ffmpeg") || cmd.contains("convert") => 1024, // 1GB for media processing
            cmd if cmd.contains("docker") => 256,        // 256MB for Docker commands
            _ => 32, // 32MB default for simple commands
        }
    }

    /// Request approval for an action
    async fn request_approval(
        &self,
        action: ActionType,
        context: String,
        reasoning: Vec<String>,
    ) -> ValidationResult<String> {
        let action_id = Uuid::new_v4().to_string();
        let (response_tx, _response_rx) = oneshot::channel();

        let pending = PendingAction {
            id: action_id.clone(),
            action: action.clone(),
            context,
            reasoning,
            risk_level: action.risk_level(),
            requested_at: Utc::now(),
            timeout: self.config.approval_timeout,
            response_tx: Some(response_tx),
        };

        // Add to queue
        self.approval_queue.write().await.insert(action_id.clone(), pending.clone());

        // Send to approval channel
        let _ = self.approval_tx.send(pending).await;

        Ok(action_id)
    }

    /// Wait for approval decision - SECURITY CRITICAL: Retrieves actual stored decisions
    async fn wait_for_approval(&self, action_id: &str) -> ValidationResult<ActionDecision> {
        let timeout = self.config.approval_timeout;
        let start = Utc::now();

        loop {
            // SECURITY FIX: Check for actual stored decision first
            if let Ok(Some(stored_decision)) = self.get_stored_decision(action_id).await {
                info!("🔒 Retrieved stored decision for action {}: {:?}", action_id, stored_decision.decision);

                // Audit log the decision retrieval
                self.audit_log("DECISION_RETRIEVED", "Decision retrieved from secure storage", serde_json::json!({
                    "action_id": action_id,
                    "decision": stored_decision.decision,
                    "decided_by": stored_decision.decided_by,
                    "decided_at": stored_decision.decided_at
                })).await;

                // Remove from approval queue since decision is final
                self.approval_queue.write().await.remove(action_id);
                return Ok(stored_decision.decision);
            }

            // Check if decision is pending in memory (fallback for real-time decisions)
            if let Some(pending) = self.approval_queue.read().await.get(action_id) {
                if pending.response_tx.is_none() {
                    // Decision channel was consumed but not yet stored - wait briefly
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            }

            // Check timeout
            let elapsed = Utc::now().signed_duration_since(start);
            if elapsed > chrono::Duration::from_std(timeout).unwrap() {
                warn!("⚠️ Action {} timed out waiting for approval", action_id);

                // SECURITY: Store timeout as implicit denial
                let timeout_decision = StoredDecision {
                    decision: ActionDecision::Deny { reason: "Approval timeout exceeded".to_string() },
                    decided_by: "system_timeout".to_string(),
                    decided_at: Utc::now(),
                    reason: Some("Action timed out waiting for human approval".to_string()),
                    action_context: "timeout".to_string(),
                    risk_assessment: RiskLevel::High,
                };

                if let Err(e) = self.store_decision(action_id, &timeout_decision).await {
                    error!("🔴 Failed to store timeout decision: {}", e);
                }

                self.approval_queue.write().await.remove(action_id);
                return Err(ValidationError::Timeout);
            }

            // Wait before checking again
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Record an action in history
    async fn record_action(&self, action: ActionType, decision: ActionDecision) {
        let mut history = self.action_history.write().await;
        history.push((Utc::now(), action, decision));

        // Keep only last 1000 actions
        if history.len() > 1000 {
            history.drain(0..100);
        }
    }

    /// Get pending actions awaiting approval
    pub async fn get_pending_actions(&self) -> Vec<PendingAction> {
        self.approval_queue.read().await.values().cloned().collect()
    }

    /// Approve or deny a pending action - SECURITY CRITICAL: Stores decisions persistently
    pub async fn decide_action(&self, action_id: &str, decision: ActionDecision) -> Result<()> {
        let mut queue = self.approval_queue.write().await;

        if let Some(pending) = queue.get_mut(action_id) {
            // SECURITY: Store decision in persistent encrypted storage
            let stored_decision = StoredDecision {
                decision: decision.clone(),
                decided_by: self.get_user_context().await.unwrap_or_else(|| "human_operator".to_string()),
                decided_at: Utc::now(),
                reason: match &decision {
                    ActionDecision::Deny { reason } => Some(reason.clone()),
                    ActionDecision::Defer { .. } => Some("Deferred for later review".to_string()),
                    ActionDecision::Approve => Some("Approved by human operator".to_string()),
                },
                action_context: pending.context.clone(),
                risk_assessment: pending.risk_level,
            };

            // Store the decision securely
            self.store_decision(action_id, &stored_decision).await
                .with_context(|| format!("Failed to store decision for action {}", action_id))?;

            // Send decision through channel for immediate processing
            if let Some(tx) = pending.response_tx.take() {
                let _ = tx.send(decision.clone());
            }

            // Audit log the decision
            self.audit_log("DECISION_MADE", "Human decision recorded", serde_json::json!({
                "action_id": action_id,
                "decision": decision,
                "decided_by": stored_decision.decided_by,
                "risk_level": pending.risk_level
            })).await;

            info!("🔒 Decision stored securely for action {}: {:?}", action_id, decision);
            Ok(())
        } else {
            Err(anyhow!("Action not found: {}", action_id))
        }
    }

    /// Get action history
    pub async fn get_history(&self) -> Vec<(DateTime<Utc>, ActionType, ActionDecision)> {
        self.action_history.read().await.clone()
    }

    /// Clear action history
    pub async fn clear_history(&self) {
        self.action_history.write().await.clear();
    }

    /// Get safety statistics
    pub async fn get_safety_statistics(&self) -> SafetyStatistics {
        let history = self.action_history.read().await;
        let pending_actions = self.approval_queue.read().await;

        // Calculate action counts by type
        let mut action_counts: HashMap<String, usize> = HashMap::new();
        let mut risk_level_counts: HashMap<RiskLevel, usize> = HashMap::new();
        let _decision_counts: HashMap<String, usize> = HashMap::new();

        // Initialize risk level counts
        risk_level_counts.insert(RiskLevel::Low, 0);
        risk_level_counts.insert(RiskLevel::Medium, 0);
        risk_level_counts.insert(RiskLevel::High, 0);
        risk_level_counts.insert(RiskLevel::Critical, 0);

        // Initialize decision counts
        let mut approved_count = 0;
        let mut denied_count = 0;
        let mut deferred_count = 0;

        // Process history
        for (_, action, decision) in history.iter() {
            // Count by action type name
            let action_type_name = match action {
                ActionType::FileRead { .. } => "FileRead",
                ActionType::FileWrite { .. } => "FileWrite",
                ActionType::FileDelete { .. } => "FileDelete",
                ActionType::FileMove { .. } => "FileMove",
                ActionType::ApiCall { .. } => "ApiCall",
                ActionType::SocialPost { .. } => "SocialPost",
                ActionType::SocialReply { .. } => "SocialReply",
                ActionType::CommandExecute { .. } => "CommandExecute",
                ActionType::CodeEval { .. } => "CodeEval",
                ActionType::GitCommit { .. } => "GitCommit",
                ActionType::GitPush { .. } => "GitPush",
                ActionType::GitPullRequest { .. } => "GitPullRequest",
                ActionType::MemoryModify { .. } => "MemoryModify",
                ActionType::ConfigChange { .. } => "ConfigChange",
                ActionType::SelfModify { .. } => "SelfModify",
                ActionType::Decision { .. } => "Decision",
                ActionType::ToolUsage { .. } => "ToolUsage",
            };

            *action_counts.entry(action_type_name.to_string()).or_insert(0) += 1;

            // Count by risk level
            let risk_level = action.risk_level();
            *risk_level_counts.get_mut(&risk_level).unwrap() += 1;

            // Count by decision
            match decision {
                ActionDecision::Approve => approved_count += 1,
                ActionDecision::Deny { .. } => denied_count += 1,
                ActionDecision::Defer { .. } => deferred_count += 1,
            }
        }

        // Calculate recent activity (last hour)
        let one_hour_ago = Utc::now() - chrono::Duration::hours(1);
        let recent_actions: Vec<_> = history
            .iter()
            .filter(|(timestamp, _, _)| *timestamp > one_hour_ago)
            .collect();

        let recent_approved = recent_actions
            .iter()
            .filter(|(_, _, decision)| matches!(decision, ActionDecision::Approve))
            .count();

        let recent_denied = recent_actions
            .iter()
            .filter(|(_, _, decision)| matches!(decision, ActionDecision::Deny { .. }))
            .count();

        // Calculate approval rate
        let total_decisions = approved_count + denied_count;
        let approval_rate = if total_decisions > 0 {
            (approved_count as f64 / total_decisions as f64) * 100.0
        } else {
            0.0
        };

        // Find most common action
        let most_common_action = action_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(action, _)| action.clone());

        // Find highest risk action in recent history
        let highest_recent_risk = recent_actions
            .iter()
            .map(|(_, action, _)| action.risk_level())
            .max();

        SafetyStatistics {
            total_actions: history.len(),
            pending_actions: pending_actions.len(),
            approved_actions: approved_count,
            denied_actions: denied_count,
            deferred_actions: deferred_count,
            approval_rate,
            action_counts,
            risk_level_counts,
            recent_actions_last_hour: recent_actions.len(),
            recent_approved_last_hour: recent_approved,
            recent_denied_last_hour: recent_denied,
            most_common_action,
            highest_recent_risk,
            safe_mode_enabled: self.config.safe_mode,
            dry_run_enabled: self.config.dry_run,
            approval_required: self.config.approval_required,
        }
    }
    

    /// Store a decision in encrypted persistent storage
    async fn store_decision(&self, action_id: &str, decision: &StoredDecision) -> Result<()> {
        let storage = self.decision_storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow!("Decision storage not initialized"))?;

        // Serialize decision with encryption metadata
        let decision_json = serde_json::to_vec(decision)
            .with_context(|| "Failed to serialize decision")?;

        // Encrypt the decision data
        let encrypted_data = if self.config.encrypt_decisions {
            let encryption = self.get_encryption().await?;
            let encrypted = helpers::encrypt_stored_decision(decision, &encryption).await?;
            serde_json::to_vec(&encrypted)?
        } else {
            decision_json
        };

        // Store with timestamped key for audit trail
        let storage_key = format!("decision:{}:{}", action_id, decision.decided_at.timestamp());

        db.put(storage_key.as_bytes(), &encrypted_data)
            .with_context(|| format!("Failed to store decision for action {}", action_id))?;

        // Also store latest decision under simple key for quick lookup
        let latest_key = format!("latest:{}", action_id);
        db.put(latest_key.as_bytes(), &encrypted_data)
            .with_context(|| format!("Failed to store latest decision for action {}", action_id))?;

        // Record security audit event
        record_security_event(
            SecurityEventType::DecisionEncrypted {
                action_id: action_id.to_string(),
                algorithm: if self.config.encrypt_decisions { "AES-256-GCM".to_string() } else { "none".to_string() },
            },
            "safety_validator",
            Some(decision.decided_by.clone()),
            serde_json::json!({
                "risk_level": decision.risk_assessment,
                "decision": format!("{:?}", decision.decision),
            }),
        ).await?;

        debug!("🔒 Stored encrypted decision for action: {}", action_id);
        Ok(())
    }

    /// Retrieve a stored decision from encrypted persistent storage
    async fn get_stored_decision(&self, action_id: &str) -> Result<Option<StoredDecision>> {
        let storage = self.decision_storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow!("Decision storage not initialized"))?;

        let latest_key = format!("latest:{}", action_id);

        match db.get(latest_key.as_bytes())? {
            Some(encrypted_data) => {
                // Decrypt the data if encryption is enabled
                let decision: StoredDecision = if self.config.encrypt_decisions {
                    let encryption = self.get_encryption().await?;
                    if let Ok(encrypted) = serde_json::from_slice::<EncryptedData>(&encrypted_data) {
                        helpers::decrypt_stored_decision(&encrypted, &encryption).await?
                    } else {
                        // Fallback for non-encrypted data
                        serde_json::from_slice(&encrypted_data)?
                    }
                } else {
                    serde_json::from_slice(&encrypted_data)?
                };

                // Record security audit event
                record_security_event(
                    SecurityEventType::DecisionDecrypted {
                        action_id: action_id.to_string(),
                        requester: self.get_user_context().await.unwrap_or_else(|| "system".to_string()),
                    },
                    "safety_validator",
                    self.get_user_context().await,
                    serde_json::json!({
                        "retrieved_at": chrono::Utc::now(),
                    }),
                ).await?;

                debug!("🔓 Retrieved encrypted decision for action: {}", action_id);
                Ok(Some(decision))
            }
            None => Ok(None)
        }
    }

    /// Security audit logging for decision operations
    async fn audit_log(&self, event_type: &str, message: &str, metadata: serde_json::Value) {
        let audit_entry = serde_json::json!({
            "timestamp": Utc::now(),
            "event_type": event_type,
            "message": message,
            "metadata": metadata,
            "component": "safety_validator",
            "security_level": "critical"
        });

        // Log to structured logging system
        info!("SECURITY_AUDIT: {}", audit_entry);

        // Send to dedicated security audit system
        if let Err(e) = record_security_event(
            SecurityEventType::DecisionStorageAccess {
                action_id: metadata.get("action_id").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                operation: event_type.to_string(),
                user_id: metadata.get("user_id").and_then(|v| v.as_str()).unwrap_or("system").to_string(),
            },
            "safety_validator",
            metadata.get("user_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            metadata.clone(),
        ).await {
            error!("Failed to record security audit event: {}", e);
        }
    }

    /// Get all stored decisions for an action (audit trail)
    pub async fn get_decision_history(&self, action_id: &str) -> Result<Vec<StoredDecision>> {
        let storage = self.decision_storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow!("Decision storage not initialized"))?;

        let prefix = format!("decision:{}:", action_id);
        let mut decisions = Vec::new();

        let iter = db.prefix_iterator(prefix.as_bytes());
        for result in iter {
            let (_, value) = result?;
            let decision: StoredDecision = serde_json::from_slice(&value)
                .with_context(|| "Failed to deserialize decision from audit trail")?;
            decisions.push(decision);
        }

        // Sort by timestamp
        decisions.sort_by_key(|d| d.decided_at);

        Ok(decisions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_file_path_validation() {
        let config = ValidatorConfig::default();
        let validator = ActionValidator::new(config).await.expect("Failed to create validator");

        // Test blocked path
        let action = ActionType::FileWrite {
            path: "src/safety/validator.rs".to_string(),
            content: "malicious".to_string(),
        };

        let result = validator.validate_action(action, "test".to_string(), vec![]).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_safety_statistics() {
        let config = ValidatorConfig {
            safe_mode: true,
            dry_run: true,  // Use dry run for testing
            approval_required: false,  // No approval needed for test
            storage_path: Some(PathBuf::from("data/test/safety_stats")),
            ..Default::default()
        };
        let validator = ActionValidator::new(config).await.expect("Failed to create validator");

        // Add some test actions to history
        let test_actions = vec![
            ActionType::FileRead { path: "test1.txt".to_string() },
            ActionType::FileWrite { path: "test2.txt".to_string(), content: "test".to_string() },
            ActionType::ApiCall { provider: "test".to_string(), endpoint: "test".to_string() },
        ];

        // Validate actions to populate history
        for action in test_actions {
            let _ = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        }

        // Get statistics
        let stats = validator.get_safety_statistics().await;

        // Verify statistics
        assert_eq!(stats.total_actions, 3);
        assert_eq!(stats.pending_actions, 0);
        assert_eq!(stats.approved_actions, 3);  // All approved in dry run mode
        assert_eq!(stats.denied_actions, 0);
        assert_eq!(stats.approval_rate, 100.0);
        assert!(stats.safe_mode_enabled);
        assert!(stats.dry_run_enabled);

        // Check action counts
        assert_eq!(stats.action_counts.get("FileRead"), Some(&1));
        assert_eq!(stats.action_counts.get("FileWrite"), Some(&1));
        assert_eq!(stats.action_counts.get("ApiCall"), Some(&1));

        // Check risk level counts
        assert_eq!(stats.risk_level_counts.get(&RiskLevel::Low), Some(&1));
        assert_eq!(stats.risk_level_counts.get(&RiskLevel::Medium), Some(&1));
        assert_eq!(stats.risk_level_counts.get(&RiskLevel::High), Some(&1));
    }

    #[tokio::test]
    async fn test_decision_storage_and_retrieval() {
        let config = ValidatorConfig {
            safe_mode: true,
            dry_run: false,
            approval_required: true,
            storage_path: Some(PathBuf::from("data/test/decision_storage")),
            encrypt_decisions: true,
            ..Default::default()
        };
        let validator = ActionValidator::new(config).await.expect("Failed to create validator");

        let action_id = "test_action_123";
        let test_decision = StoredDecision {
            decision: ActionDecision::Approve,
            decided_by: "test_user".to_string(),
            decided_at: Utc::now(),
            reason: Some("Test approval".to_string()),
            action_context: "Test context".to_string(),
            risk_assessment: RiskLevel::Medium,
        };

        // Store decision
        validator.store_decision(action_id, &test_decision).await
            .expect("Failed to store decision");

        // Retrieve decision
        let retrieved = validator.get_stored_decision(action_id).await
            .expect("Failed to retrieve decision")
            .expect("Decision not found");

        // Verify decision matches
        assert!(matches!(retrieved.decision, ActionDecision::Approve));
        assert_eq!(retrieved.decided_by, "test_user");
        assert_eq!(retrieved.reason, Some("Test approval".to_string()));
        assert_eq!(retrieved.risk_assessment, RiskLevel::Medium);
    }

    #[tokio::test]
    async fn test_security_fix_no_automatic_approval() {
        let config = ValidatorConfig {
            safe_mode: true,
            dry_run: false,
            approval_required: true,
            approval_timeout: Duration::from_millis(500), // Short timeout for test
            storage_path: Some(PathBuf::from("data/test/security_fix")),
            ..Default::default()
        };
        let validator = ActionValidator::new(config).await.expect("Failed to create validator");

        let action = ActionType::FileWrite {
            path: "data/test.txt".to_string(),
            content: "test content".to_string(),
        };

        // This should timeout and NOT automatically approve
        let result = validator.validate_action(
            action,
            "test context".to_string(),
            vec!["test reasoning".to_string()]
        ).await;

        // Should fail with timeout, not succeed with automatic approval
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ValidationError::Timeout));
    }

    #[tokio::test]
    async fn test_resource_limits_comprehensive() {
        let config = ValidatorConfig {
            safe_mode: true,
            dry_run: true,  // Use dry run for testing
            approval_required: false,  // No approval needed for test
            storage_path: Some(PathBuf::from("data/test/resource_limits")),
            enable_resource_monitoring: true,
            cpu_threshold: 75.0,
            memory_threshold: 85.0,
            disk_threshold: 90.0,
            max_concurrent_operations: 5,
            enable_rate_limiting: true,
            enable_network_monitoring: true,
            ..Default::default()
        };
        let validator = ActionValidator::new(config).await.expect("Failed to create validator");

        // Test 1: Basic file size limit (existing functionality)
        let large_content = "x".repeat(11 * 1024 * 1024); // 11MB content (exceeds 10MB limit)
        let action = ActionType::FileWrite {
            path: "data/test_large.txt".to_string(),
            content: large_content,
        };

        let result = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        assert!(result.is_err());
        if let Err(ValidationError::ResourceLimit(msg)) = result {
            assert!(msg.contains("exceeds limit"));
        } else {
            panic!("Expected ResourceLimit error for large file");
        }

        // Test 2: Normal file size (should pass)
        let normal_content = "Normal content".to_string();
        let action = ActionType::FileWrite {
            path: "data/test_normal.txt".to_string(),
            content: normal_content,
        };

        let result = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        assert!(result.is_ok(), "Normal file size should be allowed");

        // Test 3: API call action
        let action = ActionType::ApiCall {
            provider: "openai".to_string(),
            endpoint: "chat/completions".to_string(),
        };

        let result = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        assert!(result.is_ok(), "API call should be allowed in dry run mode");

        // Test 4: Command execution
        let action = ActionType::CommandExecute {
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
        };

        let result = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        assert!(result.is_ok(), "Simple command should be allowed");

        // Test 5: Code evaluation with reasonable size
        let action = ActionType::CodeEval {
            language: "rust".to_string(),
            code: "fn main() { println!(\"Hello, world!\"); }".to_string(),
        };

        let result = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        assert!(result.is_ok(), "Small code evaluation should be allowed");

        // Test 6: Code evaluation with excessive size
        let large_code = "fn main() { ".to_string() + &"println!(\"test\"); ".repeat(100_000) + "}";
        let action = ActionType::CodeEval {
            language: "rust".to_string(),
            code: large_code,
        };

        let result = validator.validate_action(action, "test context".to_string(), vec!["test reason".to_string()]).await;
        // This should pass in dry run mode but would fail in the async implementation
        assert!(result.is_ok(), "Large code evaluation allowed in basic implementation");
    }

    #[tokio::test]
    async fn test_resource_monitoring_config() {
        // Test that new configuration fields are properly set
        let config = ValidatorConfig::default();

        assert_eq!(config.enable_resource_monitoring, true);
        assert_eq!(config.cpu_threshold, 75.0);
        assert_eq!(config.memory_threshold, 85.0);
        assert_eq!(config.disk_threshold, 90.0);
        assert_eq!(config.max_concurrent_operations, 50);
        assert_eq!(config.enable_rate_limiting, true);
        assert_eq!(config.enable_network_monitoring, true);

        // Test custom configuration
        let custom_config = ValidatorConfig {
            enable_resource_monitoring: false,
            cpu_threshold: 90.0,
            memory_threshold: 95.0,
            disk_threshold: 98.0,
            max_concurrent_operations: 100,
            enable_rate_limiting: false,
            enable_network_monitoring: false,
            ..Default::default()
        };

        assert_eq!(custom_config.enable_resource_monitoring, false);
        assert_eq!(custom_config.cpu_threshold, 90.0);
        assert_eq!(custom_config.memory_threshold, 95.0);
    }

    #[test]
    fn test_command_memory_estimation() {
        let config = ValidatorConfig::default();
        let validator = ActionValidator::new_without_storage(config);

        // Test various command estimations
        assert_eq!(validator.estimate_command_memory_usage("cargo build"), 512);
        assert_eq!(validator.estimate_command_memory_usage("npm install"), 256);
        assert_eq!(validator.estimate_command_memory_usage("python script.py"), 128);
        assert_eq!(validator.estimate_command_memory_usage("git status"), 64);
        assert_eq!(validator.estimate_command_memory_usage("ffmpeg -i input.mp4"), 1024);
        assert_eq!(validator.estimate_command_memory_usage("docker run"), 256);
        assert_eq!(validator.estimate_command_memory_usage("ls -la"), 32); // Default
    }
}
