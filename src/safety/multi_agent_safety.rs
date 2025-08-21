//! Multi-Agent Safety Coordination System
//!
//! This module provides comprehensive safety mechanisms for multi-agent
//! coordination, including validation of collective decisions, monitoring
//! of emergent behaviors, and enforcement of agent-specific permissions.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use crate::models::agent_specialization_router::AgentId;
use crate::safety::{
    ActionType,
    ActionValidator,
    AuditLogger,
    ResourceMonitor,
    ResourceUsage,
    ValidationError,
    ValidationResult,
};

/// Configuration for multi-agent safety coordination
#[derive(Debug, Clone)]
pub struct MultiAgentSafetyConfig {
    /// Maximum number of agents that can coordinate simultaneously
    pub max_coordinated_agents: usize,

    /// Consensus threshold for safety-critical decisions
    pub safety_consensus_threshold: f32,

    /// Maximum allowed resource usage per agent
    pub max_agent_resource_usage: f32,

    /// Timeout for safety validations
    pub validation_timeout: Duration,

    /// Enable emergent behavior monitoring
    pub monitor_emergent_behavior: bool,

    /// Quarantine threshold for suspicious agents
    pub quarantine_threshold: f32,

    /// Maximum collective goal complexity
    pub max_goal_complexity: usize,

    /// Agent interaction rate limits
    pub interaction_rate_limit: u32,

    /// Enable distributed safety validation
    pub distributed_validation: bool,

    /// Safety audit retention period
    pub audit_retention_days: u32,
}

impl Default for MultiAgentSafetyConfig {
    fn default() -> Self {
        Self {
            max_coordinated_agents: 10,
            safety_consensus_threshold: 0.8,
            max_agent_resource_usage: 0.7,
            validation_timeout: Duration::from_secs(30),
            monitor_emergent_behavior: true,
            quarantine_threshold: 0.9,
            max_goal_complexity: 20,
            interaction_rate_limit: 100,
            distributed_validation: true,
            audit_retention_days: 365,
        }
    }
}

/// Agent-specific permissions and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPermissions {

    /// Blocked action types
    pub blocked_actions: Vec<ActionType>,

    /// Resource usage limits
    pub resource_limits: HashMap<String, f32>,

    /// Trust level (0.0 to 1.0)
    pub trust_level: f32,

    /// Can initiate collective goals
    pub can_initiate_goals: bool,

    /// Can vote in consensus processes
    pub can_vote: bool,

    /// Can communicate with other agents
    pub can_communicate: bool,

    /// Quarantine status
    pub quarantined: bool,

    /// Permission expiry
    pub expires_at: Option<DateTime<Utc>>,
}

impl Default for AgentPermissions {
    fn default() -> Self {
        Self {
            blocked_actions: Vec::new(),
            resource_limits: HashMap::from([
                ("cpu".to_string(), 0.5),
                ("memory".to_string(), 0.5),
                ("network".to_string(), 0.5),
            ]),
            trust_level: 0.5,
            can_initiate_goals: false,
            can_vote: true,
            can_communicate: true,
            quarantined: false,
            expires_at: None,
        }
    }
}

/// Safety profile for an agent
#[derive(Debug, Clone)]
pub struct AgentSafetyProfile {
    /// Agent permissions
    pub permissions: AgentPermissions,

    /// Recent safety violations
    pub violations: VecDeque<SafetyViolation>,

    /// Current safety score (0.0 to 1.0)
    pub safety_score: f32,

    /// Resource usage history
    pub resource_history: VecDeque<ResourceUsage>,

    /// Communication patterns
    pub communication_patterns: HashMap<AgentId, u32>,

    /// Last safety check
    pub last_safety_check: Instant,

    /// Behavioral anomalies detected
    pub anomalies: Vec<BehavioralAnomaly>,
}

/// Safety violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    /// Violation ID
    pub id: String,

    /// Agent that violated safety
    pub agent_id: AgentId,

    /// Type of violation
    pub violation_type: SafetyViolationType,

    /// Severity level
    pub severity: ViolationSeverity,

    /// Description
    pub description: String,

    /// Action taken
    pub action_taken: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Additional context
    pub context: Value,
}

/// Types of safety violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyViolationType {
    /// Unauthorized action attempt
    UnauthorizedAction,

    /// Resource limit exceeded
    ResourceOveruse,

    /// Suspicious communication pattern
    SuspiciousCommunication,

    /// Emergent behavior concern
    EmergentBehavior,

    /// Consensus manipulation attempt
    ConsensusManipulation,

    /// Role permission violation
    RoleViolation,

    /// Goal manipulation
    GoalManipulation,

    /// Agent impersonation
    Impersonation,

    /// Data corruption attempt
    DataCorruption,

    /// System compromise attempt
    SystemCompromise,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Behavioral anomaly detection
#[derive(Debug, Clone)]
pub struct BehavioralAnomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,

    /// Confidence level
    pub confidence: f32,

    /// Description
    pub description: String,

    /// Detected at
    pub detected_at: Instant,

    /// Evidence
    pub evidence: Vec<String>,
}

/// Types of behavioral anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    /// Unusual communication frequency
    CommunicationAnomaly,

    /// Abnormal resource usage
    ResourceAnomaly,

    /// Unexpected decision patterns
    DecisionAnomaly,

    /// Role behavior inconsistency
    RoleAnomaly,

    /// Goal pursuit deviation
    GoalAnomaly,

    /// Interaction pattern change
    InteractionAnomaly,
}

/// Collective action validator for multi-agent decisions
pub struct CollectiveActionValidator {
    /// Individual action validator
    action_validator: Arc<ActionValidator>,

    /// Multi-agent safety config
    config: MultiAgentSafetyConfig,

    /// Agent permissions
    agent_permissions: Arc<RwLock<HashMap<AgentId, AgentPermissions>>>,

    /// Active collective validations
    active_validations: Arc<RwLock<HashMap<String, CollectiveValidation>>>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,
}

/// A collective validation process
#[derive(Debug, Clone)]
pub struct CollectiveValidation {
    /// Validation ID
    pub id: String,

    /// Participating agents
    pub agents: Vec<AgentId>,

    /// Action being validated
    pub action: CollectiveAction,

    /// Individual validations
    pub individual_validations: HashMap<AgentId, ValidationResult<()>>,

    /// Consensus requirement
    pub consensus_required: bool,

    /// Started at
    pub started_at: Instant,

    /// Timeout
    pub timeout: Duration,

    /// Status
    pub status: ValidationStatus,
}

/// Collective actions that require validation
#[derive(Debug, Clone)]
pub enum CollectiveAction {
    /// Resource allocation
    ResourceAllocation { agent: AgentId, resource_type: String, amount: f32 },

    /// Agent coordination change
    CoordinationChange { change_type: String, affected_agents: Vec<AgentId> },

    /// Emergency action
    EmergencyAction { action_type: String, triggered_by: AgentId, reason: String },
}

/// Status of a collective validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation in progress
    InProgress,

    /// Approved by consensus
    Approved,

    /// Denied by consensus
    Denied,

    /// Timed out
    TimedOut,

    /// Cancelled
    Cancelled,
}

/// Multi-agent audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiAgentAuditEvent {
    /// Agent joined coordination
    AgentJoined { agent_id: AgentId},

    /// Agent left coordination
    AgentLeft { agent_id: AgentId, reason: String },

    /// Collective validation started
    CollectiveValidationStarted { validation_id: String, action: String, agents: Vec<AgentId> },

    /// Collective validation completed
    CollectiveValidationCompleted {
        validation_id: String,
        result: ValidationStatus,
        consensus_level: f32,
    },

    /// Safety violation detected
    SafetyViolationDetected { violation: SafetyViolation },

    /// Agent quarantined
    AgentQuarantined { agent_id: AgentId, reason: String, duration: Duration },

    /// Emergent behavior detected
    EmergentBehaviorDetected { description: String, agents: Vec<AgentId>, risk_level: String },

    /// Resource limit exceeded
    ResourceLimitExceeded { agent_id: AgentId, resource: String, usage: f32, limit: f32 },

    /// Consensus manipulation detected
    ConsensusManipulationDetected { agent_id: AgentId, process_id: String, evidence: Vec<String> },
}

/// Emergent behavior monitor for detecting concerning patterns
pub struct EmergentBehaviorMonitor {
    /// Configuration
    #[allow(dead_code)]
    config: MultiAgentSafetyConfig,

    /// Agent behavior patterns
    behavior_patterns: Arc<RwLock<HashMap<AgentId, BehaviorPattern>>>,

    /// Collective behavior history
    collective_history: Arc<RwLock<VecDeque<CollectiveBehaviorEvent>>>,

    /// Anomaly detectors
    anomaly_detectors: Vec<Box<dyn AnomalyDetector + Send + Sync>>,

    /// Alert channel
    alert_tx: mpsc::Sender<BehaviorAlert>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,
}

/// Individual agent behavior pattern
#[derive(Debug, Clone)]
pub struct BehaviorPattern {
    /// Agent ID
    pub agent_id: AgentId,

    /// Communication frequency (messages per minute)
    pub communication_frequency: f32,

    /// Decision making patterns
    pub decision_patterns: HashMap<String, f32>,

    /// Resource usage patterns
    pub resource_patterns: HashMap<String, f32>,

    /// Interaction preferences
    pub interaction_preferences: HashMap<AgentId, f32>,

    /// Last updated
    pub last_updated: Instant,
}

/// Collective behavior event
#[derive(Debug, Clone)]
pub struct CollectiveBehaviorEvent {
    /// Event type
    pub event_type: CollectiveBehaviorType,

    /// Participating agents
    pub agents: Vec<AgentId>,

    /// Event details
    pub details: Value,

    /// Timestamp
    pub timestamp: Instant,

    /// Risk assessment
    pub risk_level: f32,
}

/// Types of collective behavior
#[derive(Debug, Clone)]
pub enum CollectiveBehaviorType {
    /// Agents forming coalitions
    Coalition,

    /// Synchronized decision making
    SynchronizedDecisions,

    /// Resource competition
    ResourceCompetition,

    /// Communication clustering
    CommunicationClustering,

    /// Goal alignment patterns
    GoalAlignment,

    /// Emergent specialization
    EmergentSpecialization,

    /// Collective learning
    CollectiveLearning,

    /// Consensus manipulation
    ConsensusManipulation,
}

/// Behavior alert
#[derive(Debug, Clone)]
pub struct BehaviorAlert {
    /// Alert type
    pub alert_type: AlertType,

    /// Affected agents
    pub agents: Vec<AgentId>,

    /// Risk level
    pub risk_level: f32,

    /// Description
    pub description: String,

    /// Evidence
    pub evidence: Vec<String>,

    /// Recommended actions
    pub recommended_actions: Vec<String>,

    /// Timestamp
    pub timestamp: Instant,
}

/// Types of behavior alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    /// Suspicious coordination pattern
    SuspiciousCoordination,

    /// Resource abuse detected
    ResourceAbuse,

    /// Potential security threat
    SecurityThreat,

    /// Communication anomaly
    CommunicationAnomaly,

    /// Performance degradation
    PerformanceDegradation,

    /// Goal manipulation
    GoalManipulation,

    /// System compromise attempt
    SystemCompromise,
}

/// Trait for anomaly detection algorithms
pub trait AnomalyDetector: Send + Sync {
    /// Analyze behavior patterns and detect anomalies
    fn detect_anomalies(
        &self,
        patterns: &HashMap<AgentId, BehaviorPattern>,
        collective_history: &VecDeque<CollectiveBehaviorEvent>,
    ) -> Vec<BehavioralAnomaly>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get confidence threshold
    fn confidence_threshold(&self) -> f32;
}

/// Main multi-agent safety coordinator
pub struct MultiAgentSafetyCoordinator {
    /// Configuration
    config: MultiAgentSafetyConfig,

    /// Individual action validator
    action_validator: Arc<ActionValidator>,

    /// Collective action validator
    collective_validator: Arc<CollectiveActionValidator>,

    /// Emergent behavior monitor
    behavior_monitor: Arc<EmergentBehaviorMonitor>,

    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,

    /// Agent safety profiles
    agent_profiles: Arc<RwLock<HashMap<AgentId, AgentSafetyProfile>>>,

    /// Safety event channel
    safety_event_tx: broadcast::Sender<SafetyEvent>,

    /// Background task handles
    task_handles: Vec<tokio::task::JoinHandle<()>>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
}

/// Safety events broadcasted by the coordinator
#[derive(Debug, Clone)]
pub enum SafetyEvent {
    /// Agent safety profile updated
    ProfileUpdated { agent_id: AgentId, old_score: f32, new_score: f32 },

    /// Safety violation detected
    ViolationDetected { violation: SafetyViolation },

    /// Agent quarantined
    AgentQuarantined { agent_id: AgentId, reason: String },

    /// Emergency stop triggered
    EmergencyStop { triggered_by: String, reason: String },

    /// Behavior alert issued
    BehaviorAlert { alert: BehaviorAlert },

    /// System integrity check
    IntegrityCheck { status: String, issues: Vec<String> },
}

impl MultiAgentSafetyCoordinator {
    /// Create a new multi-agent safety coordinator
    pub async fn new(
        config: MultiAgentSafetyConfig,
        action_validator: Arc<ActionValidator>,
        resource_monitor: Arc<ResourceMonitor>,
        audit_logger: Arc<AuditLogger>,
    ) -> Result<Self> {
        info!("Initializing multi-agent safety coordinator");

        // Create collective action validator
        let collective_validator = Arc::new(
            CollectiveActionValidator::new(
                action_validator.clone(),
                config.clone(),
                audit_logger.clone(),
            )
            .await?,
        );

        // Create emergent behavior monitor
        let (alert_tx, _alert_rx) = mpsc::channel(100);
        let behavior_monitor = Arc::new(
            EmergentBehaviorMonitor::new(config.clone(), alert_tx, audit_logger.clone()).await?,
        );

        // Create safety event channel
        let (safety_event_tx, _safety_event_rx) = broadcast::channel(100);

        // Create shutdown channel
        let (shutdown_tx, _shutdown_rx) = broadcast::channel(1);

        Ok(Self {
            config,
            action_validator,
            collective_validator,
            behavior_monitor,
            resource_monitor,
            audit_logger,
            agent_profiles: Arc::new(RwLock::new(HashMap::new())),
            safety_event_tx,
            task_handles: Vec::new(),
            shutdown_tx,
        })
    }

    /// Start the safety coordinator
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting multi-agent safety coordinator");

        // Start behavior monitoring
        let behavior_monitor = self.behavior_monitor.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
            behavior_monitor.start_monitoring(shutdown_rx).await;
        });
        self.task_handles.push(handle);

        // Start safety profile updates
        let profiles = self.agent_profiles.clone();
        let event_tx = self.safety_event_tx.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
            Self::safety_profile_update_loop(profiles, event_tx, shutdown_rx).await;
        });
        self.task_handles.push(handle);

        // Start integrity checking
        let coordinator = self.clone();
        let shutdown_rx = self.shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
            coordinator.integrity_check_loop(shutdown_rx).await;
        });
        self.task_handles.push(handle);

        // Log startup - Use a memory modify action as a proxy for system startup
        self.audit_logger
            .log_action_request(
                "multi_agent_safety",
                &ActionType::MemoryModify {
                    operation: "multi_agent_safety_coordinator_startup".to_string(),
                },
                "Multi-agent safety coordinator started",
            )
            .await?;

        Ok(())
    }

    /// Validate agent action
    pub async fn validate_agent_action(
        &self,
        agent_id: &AgentId,
        action: ActionType,
        context: String,
        reasoning: Vec<String>,
    ) -> ValidationResult<()> {
        debug!("Validating action for agent {}: {:?}", agent_id, action);

        // Check agent permissions
        self.check_agent_permissions(agent_id, &action).await?;

        // Check agent safety profile
        self.check_agent_safety_profile(agent_id).await?;

        // Validate with individual action validator
        self.action_validator
            .validate_action(action.clone(), context.clone(), reasoning.clone())
            .await?;

        // Update agent profile
        self.update_agent_activity(agent_id, &action).await.map_err(|e| {
            ValidationError::Invalid(format!("Failed to update agent activity: {}", e))
        })?;

        info!("Action validated for agent {}: {:?}", agent_id, action);
        Ok(())
    }

    /// Validate collective action
    pub async fn validate_collective_action(
        &self,
        action: CollectiveAction,
        participating_agents: Vec<AgentId>,
    ) -> ValidationResult<String> {
        info!("Validating collective action: {:?}", action);

        // Check participating agents
        for agent_id in &participating_agents {
            self.check_agent_safety_profile(agent_id).await?;
        }

        // Validate with collective validator
        let validation_id = self
            .collective_validator
            .validate_collective_action(action, participating_agents)
            .await?;

        Ok(validation_id)
    }

    /// Check agent permissions for specific action
    async fn check_agent_permissions(
        &self,
        agent_id: &AgentId,
        action: &ActionType,
    ) -> ValidationResult<()> {
        let profiles = self.agent_profiles.read().await;

        if let Some(profile) = profiles.get(agent_id) {
            // Check quarantine status
            if profile.permissions.quarantined {
                return Err(ValidationError::NotAllowed("Agent is quarantined".to_string()));
            }

            // Check blocked actions
            if profile.permissions.blocked_actions.iter().any(|blocked| blocked == action) {
                return Err(ValidationError::NotAllowed(
                    "Action is blocked for this agent".to_string(),
                ));
            }

            // Check permission expiry
            if let Some(expires_at) = profile.permissions.expires_at {
                if Utc::now() > expires_at {
                    return Err(ValidationError::NotAllowed(
                        "Agent permissions have expired".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check agent safety profile
    async fn check_agent_safety_profile(&self, agent_id: &AgentId) -> ValidationResult<()> {
        let profiles = self.agent_profiles.read().await;

        if let Some(profile) = profiles.get(agent_id) {
            // Check safety score
            if profile.safety_score < 0.3 {
                return Err(ValidationError::NotAllowed(format!(
                    "Agent safety score too low: {}",
                    profile.safety_score
                )));
            }

            // Check recent violations
            let recent_critical_violations = profile
                .violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Critical)
                .filter(|v| v.timestamp > Utc::now() - chrono::Duration::hours(24))
                .count();

            if recent_critical_violations > 0 {
                return Err(ValidationError::NotAllowed(
                    "Agent has recent critical safety violations".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Update agent activity tracking
    async fn update_agent_activity(&self, agent_id: &AgentId, _action: &ActionType) -> Result<()> {
        let mut profiles = self.agent_profiles.write().await;

        let profile = profiles.entry(agent_id.clone()).or_insert_with(|| AgentSafetyProfile {
            permissions: AgentPermissions::default(),
            violations: VecDeque::new(),
            safety_score: 0.8,
            resource_history: VecDeque::new(),
            communication_patterns: HashMap::new(),
            last_safety_check: Instant::now(),
            anomalies: Vec::new(),
        });

        // Update last activity
        profile.last_safety_check = Instant::now();

        // Track action patterns (simplified)
        // In a real implementation, this would be more sophisticated

        Ok(())
    }

    /// Safety profile update loop
    async fn safety_profile_update_loop(
        profiles: Arc<RwLock<HashMap<AgentId, AgentSafetyProfile>>>,
        event_tx: broadcast::Sender<SafetyEvent>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut interval = interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    // Update safety profiles
                    let mut profiles_guard = profiles.write().await;
                    for (agent_id, profile) in profiles_guard.iter_mut() {
                        let old_score = profile.safety_score;

                        // Recalculate safety score based on recent activity
                        let new_score = Self::calculate_safety_score(profile);
                        profile.safety_score = new_score;

                        // Broadcast if significant change
                        if (new_score - old_score).abs() > 0.1 {
                            let _ = event_tx.send(SafetyEvent::ProfileUpdated {
                                agent_id: agent_id.clone(),
                                old_score,
                                new_score,
                            });
                        }
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Safety profile update loop shutting down");
                    break;
                }
            }
        }
    }

    /// Calculate safety score for an agent
    fn calculate_safety_score(profile: &AgentSafetyProfile) -> f32 {
        let mut score = 1.0;

        // Deduct for recent violations
        for violation in &profile.violations {
            let age_hours =
                Utc::now().signed_duration_since(violation.timestamp).num_hours() as f32;
            let age_factor = (age_hours / 168.0).min(1.0); // Week decay

            let penalty = match violation.severity {
                ViolationSeverity::Low => 0.05,
                ViolationSeverity::Medium => 0.1,
                ViolationSeverity::High => 0.2,
                ViolationSeverity::Critical => 0.4,
            };

            score -= penalty * (1.0 - age_factor);
        }

        // Deduct for anomalies
        for anomaly in &profile.anomalies {
            let age_factor = profile.last_safety_check.elapsed().as_secs() as f32 / 86400.0; // Day decay
            score -= anomaly.confidence * 0.1 * (1.0 - age_factor.min(1.0));
        }

        // Trust level factor
        score *= 0.5 + profile.permissions.trust_level * 0.5;

        score.max(0.0).min(1.0)
    }

    /// Integrity check loop
    async fn integrity_check_loop(self, mut shutdown_rx: broadcast::Receiver<()>) {
        let mut interval = interval(Duration::from_secs(300)); // 5 minutes

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.perform_integrity_check().await {
                        error!("Integrity check failed: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Integrity check loop shutting down");
                    break;
                }
            }
        }
    }

    /// Perform system integrity check
    async fn perform_integrity_check(&self) -> Result<()> {
        debug!("Performing system integrity check");

        let mut issues = Vec::new();

        // Check agent profiles for consistency
        let profiles = self.agent_profiles.read().await;
        for (agent_id, profile) in profiles.iter() {
            if profile.safety_score < 0.0 || profile.safety_score > 1.0 {
                issues.push(format!(
                    "Invalid safety score for agent {}: {}",
                    agent_id, profile.safety_score
                ));
            }

            if profile.permissions.trust_level < 0.0 || profile.permissions.trust_level > 1.0 {
                issues.push(format!(
                    "Invalid trust level for agent {}: {}",
                    agent_id, profile.permissions.trust_level
                ));
            }
        }

        // Broadcast integrity status
        let _ = self.safety_event_tx.send(SafetyEvent::IntegrityCheck {
            status: if issues.is_empty() { "OK".to_string() } else { "ISSUES".to_string() },
            issues: issues.clone(),
        });

        if !issues.is_empty() {
            warn!("Integrity check found {} issues", issues.len());
            for issue in &issues {
                warn!("Integrity issue: {}", issue);
            }
        }

        Ok(())
    }

    /// Shutdown the safety coordinator
    pub async fn shutdown(self) -> Result<()> {
        info!("Shutting down multi-agent safety coordinator");

        // Send shutdown signal
        let _ = self.shutdown_tx.send(());

        // Wait for tasks to complete
        for handle in self.task_handles {
            let _ = handle.await;
        }

        info!("Multi-agent safety coordinator shutdown complete");
        Ok(())
    }
}

impl Clone for MultiAgentSafetyCoordinator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            action_validator: self.action_validator.clone(),
            collective_validator: self.collective_validator.clone(),
            behavior_monitor: self.behavior_monitor.clone(),
            resource_monitor: self.resource_monitor.clone(),
            audit_logger: self.audit_logger.clone(),
            agent_profiles: self.agent_profiles.clone(),
            safety_event_tx: self.safety_event_tx.clone(),
            task_handles: Vec::new(), // Don't clone handles
            shutdown_tx: self.shutdown_tx.clone(),
        }
    }
}

impl CollectiveActionValidator {
    /// Create a new collective action validator
    pub async fn new(
        action_validator: Arc<ActionValidator>,
        config: MultiAgentSafetyConfig,
        audit_logger: Arc<AuditLogger>,
    ) -> Result<Self> {
        Ok(Self {
            action_validator,
            config,
            agent_permissions: Arc::new(RwLock::new(HashMap::new())),
            active_validations: Arc::new(RwLock::new(HashMap::new())),
            audit_logger,
        })
    }

    /// Validate a collective action
    pub async fn validate_collective_action(
        &self,
        action: CollectiveAction,
        participating_agents: Vec<AgentId>,
    ) -> ValidationResult<String> {
        let validation_id = Uuid::new_v4().to_string();

        info!("Starting collective validation {}: {:?}", validation_id, action);

        // Check agent count limits
        if participating_agents.len() > self.config.max_coordinated_agents {
            return Err(ValidationError::NotAllowed(format!(
                "Too many agents for collective action: {} > {}",
                participating_agents.len(),
                self.config.max_coordinated_agents
            )));
        }

        // Check agent permissions
        for agent_id in &participating_agents {
            self.check_collective_permissions(agent_id, &action).await?;
        }

        // Create validation record
        let validation = CollectiveValidation {
            id: validation_id.clone(),
            agents: participating_agents.clone(),
            action: action.clone(),
            individual_validations: HashMap::new(),
            consensus_required: self.requires_consensus(&action),
            started_at: Instant::now(),
            timeout: self.config.validation_timeout,
            status: ValidationStatus::InProgress,
        };

        // Store validation
        self.active_validations.write().await.insert(validation_id.clone(), validation);

        // Log validation start
        self.audit_logger
            .log_action_request(
                "collective_validator",
                &ActionType::MemoryModify {
                    operation: format!("collective_validation_{}", validation_id),
                },
                &format!("Collective action validation: {:?}", action),
            )
            .await
            .map_err(|e| ValidationError::Invalid(format!("Failed to log validation: {}", e)))?;

        // Perform validation based on action type
        match action {
            CollectiveAction::ResourceAllocation { ref agent, ref resource_type, amount } => {
                self.validate_resource_allocation(agent, resource_type, amount).await?;
            }
            CollectiveAction::CoordinationChange { .. } => {
                self.validate_coordination_change(&participating_agents).await?;
            }
            CollectiveAction::EmergencyAction { ref action_type, ref triggered_by, ref reason } => {
                self.validate_emergency_action(action_type, triggered_by, reason).await?;
            }
        }

        // Mark as approved if validation passes
        let mut validations = self.active_validations.write().await;
        if let Some(validation) = validations.get_mut(&validation_id) {
            validation.status = ValidationStatus::Approved;
        }

        info!("Collective validation {} approved", validation_id);
        Ok(validation_id)
    }

    /// Check agent permissions for collective actions
    async fn check_collective_permissions(
        &self,
        agent_id: &AgentId,
        action: &CollectiveAction,
    ) -> ValidationResult<()> {
        let permissions = self.agent_permissions.read().await;

        if let Some(agent_perms) = permissions.get(agent_id) {
            // Check quarantine
            if agent_perms.quarantined {
                return Err(ValidationError::NotAllowed(format!(
                    "Agent {} is quarantined",
                    agent_id
                )));
            }

            // Check specific permissions based on action
            match action {
                _ => {}
            }

            // Check trust level
            if agent_perms.trust_level < 0.5 {
                return Err(ValidationError::NotAllowed(format!(
                    "Agent {} trust level too low: {}",
                    agent_id, agent_perms.trust_level
                )));
            }
        }

        Ok(())
    }

    /// Check if action requires consensus
    fn requires_consensus(&self, action: &CollectiveAction) -> bool {
        match action {
            CollectiveAction::ResourceAllocation { .. } => true,
            CollectiveAction::CoordinationChange { .. } => true,
            CollectiveAction::EmergencyAction { .. } => false, // Emergency actions bypass consensus
        }
    }


    /// Validate resource allocation
    async fn validate_resource_allocation(
        &self,
        agent: &AgentId,
        resource_type: &str,
        amount: f32,
    ) -> ValidationResult<()> {
        let permissions = self.agent_permissions.read().await;
        if let Some(perms) = permissions.get(agent) {
            if let Some(&limit) = perms.resource_limits.get(resource_type) {
                if amount > limit {
                    return Err(ValidationError::ResourceLimit(format!(
                        "Resource allocation {} > limit {} for agent {}",
                        amount, limit, agent
                    )));
                }
            }
        }

        Ok(())
    }

    /// Validate coordination change
    async fn validate_coordination_change(
        &self,
        participating_agents: &[AgentId],
    ) -> ValidationResult<()> {
        // Check communication permissions
        let permissions = self.agent_permissions.read().await;
        for agent_id in participating_agents {
            if let Some(perms) = permissions.get(agent_id) {
                if !perms.can_communicate {
                    return Err(ValidationError::NotAllowed(format!(
                        "Agent {} cannot communicate",
                        agent_id
                    )));
                }
            }
        }

        Ok(())
    }

    /// Validate emergency action
    async fn validate_emergency_action(
        &self,
        _action_type: &str,
        triggered_by: &AgentId,
        _reason: &str,
    ) -> ValidationResult<()> {
        let permissions = self.agent_permissions.read().await;
        if let Some(perms) = permissions.get(triggered_by) {
            if perms.trust_level < 0.8 {
                return Err(ValidationError::NotAllowed(format!(
                    "Agent {} trust level too low for emergency actions: {}",
                    triggered_by, perms.trust_level
                )));
            }
        }

        Ok(())
    }
}

impl EmergentBehaviorMonitor {
    /// Create a new emergent behavior monitor
    pub async fn new(
        config: MultiAgentSafetyConfig,
        alert_tx: mpsc::Sender<BehaviorAlert>,
        audit_logger: Arc<AuditLogger>,
    ) -> Result<Self> {
        // Initialize anomaly detectors
        let mut anomaly_detectors: Vec<Box<dyn AnomalyDetector + Send + Sync>> = Vec::new();

        // Add basic detectors (simplified for this implementation)
        anomaly_detectors.push(Box::new(CommunicationAnomalyDetector::new()));
        anomaly_detectors.push(Box::new(ResourceAnomalyDetector::new()));
        anomaly_detectors.push(Box::new(DecisionAnomalyDetector::new()));

        Ok(Self {
            config,
            behavior_patterns: Arc::new(RwLock::new(HashMap::new())),
            collective_history: Arc::new(RwLock::new(VecDeque::new())),
            anomaly_detectors,
            alert_tx,
            audit_logger,
        })
    }

    /// Start monitoring behavior patterns
    pub async fn start_monitoring(&self, mut shutdown_rx: broadcast::Receiver<()>) {
        info!("Starting emergent behavior monitoring");

        let mut interval = interval(Duration::from_secs(60)); // Check every minute

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.analyze_behavior_patterns().await {
                        error!("Behavior analysis failed: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Emergent behavior monitor shutting down");
                    break;
                }
            }
        }
    }

    /// Analyze current behavior patterns for anomalies
    async fn analyze_behavior_patterns(&self) -> Result<()> {
        debug!("Analyzing behavior patterns");

        let patterns = self.behavior_patterns.read().await;
        let history = self.collective_history.read().await;

        // Run anomaly detection
        for detector in &self.anomaly_detectors {
            let anomalies = detector.detect_anomalies(&patterns, &history);

            for anomaly in anomalies {
                if anomaly.confidence > detector.confidence_threshold() {
                    self.handle_anomaly(anomaly).await?;
                }
            }
        }

        Ok(())
    }

    /// Handle detected anomaly
    async fn handle_anomaly(&self, anomaly: BehavioralAnomaly) -> Result<()> {
        warn!("Behavioral anomaly detected: {:?}", anomaly);

        // Create behavior alert
        let alert = BehaviorAlert {
            alert_type: match anomaly.anomaly_type {
                AnomalyType::CommunicationAnomaly => AlertType::CommunicationAnomaly,
                AnomalyType::ResourceAnomaly => AlertType::ResourceAbuse,
                AnomalyType::DecisionAnomaly => AlertType::SuspiciousCoordination,
                AnomalyType::RoleAnomaly => AlertType::SuspiciousCoordination,
                AnomalyType::GoalAnomaly => AlertType::GoalManipulation,
                AnomalyType::InteractionAnomaly => AlertType::SuspiciousCoordination,
            },
            agents: Vec::new(), // Would be populated with affected agents
            risk_level: anomaly.confidence,
            description: anomaly.description.clone(),
            evidence: anomaly.evidence.clone(),
            recommended_actions: self.get_recommended_actions(&anomaly),
            timestamp: Instant::now(),
        };

        // Send alert
        if let Err(e) = self.alert_tx.send(alert.clone()).await {
            error!("Failed to send behavior alert: {}", e);
        }

        // Log to audit
        self.audit_logger
            .log_suspicious_activity("behavior_monitor", &anomaly.description, anomaly.evidence)
            .await?;

        Ok(())
    }

    /// Get recommended actions for an anomaly
    fn get_recommended_actions(&self, anomaly: &BehavioralAnomaly) -> Vec<String> {
        match anomaly.anomaly_type {
            AnomalyType::CommunicationAnomaly => vec![
                "Investigate communication patterns".to_string(),
                "Review agent interaction logs".to_string(),
                "Consider rate limiting".to_string(),
            ],
            AnomalyType::ResourceAnomaly => vec![
                "Review resource allocation".to_string(),
                "Check for resource abuse".to_string(),
                "Implement stricter limits".to_string(),
            ],
            AnomalyType::DecisionAnomaly => vec![
                "Analyze decision patterns".to_string(),
                "Review consensus processes".to_string(),
                "Check for manipulation".to_string(),
            ],
            _ => vec!["Investigate further".to_string(), "Monitor closely".to_string()],
        }
    }
}

// Simple anomaly detector implementations
struct CommunicationAnomalyDetector;

impl CommunicationAnomalyDetector {
    fn new() -> Self {
        Self
    }
}

impl AnomalyDetector for CommunicationAnomalyDetector {
    fn detect_anomalies(
        &self,
        patterns: &HashMap<AgentId, BehaviorPattern>,
        _collective_history: &VecDeque<CollectiveBehaviorEvent>,
    ) -> Vec<BehavioralAnomaly> {
        let mut anomalies = Vec::new();

        // Calculate average communication frequency
        let avg_freq = patterns.values().map(|p| p.communication_frequency).sum::<f32>()
            / patterns.len() as f32;

        // Detect outliers
        for (agent_id, pattern) in patterns {
            if pattern.communication_frequency > avg_freq * 3.0 {
                anomalies.push(BehavioralAnomaly {
                    anomaly_type: AnomalyType::CommunicationAnomaly,
                    confidence: 0.8,
                    description: format!(
                        "High communication frequency for agent {}: {}",
                        agent_id, pattern.communication_frequency
                    ),
                    detected_at: Instant::now(),
                    evidence: vec![
                        format!("Frequency: {}", pattern.communication_frequency),
                        format!("Average: {}", avg_freq),
                    ],
                });
            }
        }

        anomalies
    }

    fn name(&self) -> &str {
        "CommunicationAnomalyDetector"
    }

    fn confidence_threshold(&self) -> f32 {
        0.7
    }
}

struct ResourceAnomalyDetector;

impl ResourceAnomalyDetector {
    fn new() -> Self {
        Self
    }
}

impl AnomalyDetector for ResourceAnomalyDetector {
    fn detect_anomalies(
        &self,
        patterns: &HashMap<AgentId, BehaviorPattern>,
        _collective_history: &VecDeque<CollectiveBehaviorEvent>,
    ) -> Vec<BehavioralAnomaly> {
        let mut anomalies = Vec::new();

        // Check for resource usage anomalies
        for (agent_id, pattern) in patterns {
            for (resource, usage) in &pattern.resource_patterns {
                if *usage > 0.9 {
                    // High resource usage
                    anomalies.push(BehavioralAnomaly {
                        anomaly_type: AnomalyType::ResourceAnomaly,
                        confidence: 0.7,
                        description: format!(
                            "High {} usage for agent {}: {}",
                            resource, agent_id, usage
                        ),
                        detected_at: Instant::now(),
                        evidence: vec![
                            format!("Resource: {}", resource),
                            format!("Usage: {}", usage),
                        ],
                    });
                }
            }
        }

        anomalies
    }

    fn name(&self) -> &str {
        "ResourceAnomalyDetector"
    }

    fn confidence_threshold(&self) -> f32 {
        0.6
    }
}

struct DecisionAnomalyDetector;

impl DecisionAnomalyDetector {
    fn new() -> Self {
        Self
    }
}

impl AnomalyDetector for DecisionAnomalyDetector {
    fn detect_anomalies(
        &self,
        patterns: &HashMap<AgentId, BehaviorPattern>,
        _collective_history: &VecDeque<CollectiveBehaviorEvent>,
    ) -> Vec<BehavioralAnomaly> {
        let mut anomalies = Vec::new();

        // Detect unusual decision patterns (simplified)
        for (agent_id, pattern) in patterns {
            let decision_variance =
                pattern.decision_patterns.values().map(|&v| (v - 0.5).abs()).sum::<f32>()
                    / pattern.decision_patterns.len() as f32;

            if decision_variance > 0.3 {
                anomalies.push(BehavioralAnomaly {
                    anomaly_type: AnomalyType::DecisionAnomaly,
                    confidence: 0.6,
                    description: format!("Unusual decision patterns for agent {}", agent_id),
                    detected_at: Instant::now(),
                    evidence: vec![format!("Decision variance: {}", decision_variance)],
                });
            }
        }

        anomalies
    }

    fn name(&self) -> &str {
        "DecisionAnomalyDetector"
    }

    fn confidence_threshold(&self) -> f32 {
        0.5
    }
}
