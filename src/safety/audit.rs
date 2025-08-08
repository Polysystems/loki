//! Audit Logging System
//!
//! Provides comprehensive logging of all actions for security,
//! debugging, and compliance purposes.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::{RwLock, mpsc};
use tracing::{error, info};
use uuid::Uuid;

use crate::safety::validator::{ActionDecision, ActionType};

/// Severity levels for audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// An audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: AuditEventType,

    /// Severity
    pub severity: AuditSeverity,

    /// Actor (who triggered this)
    pub actor: String,

    /// Details
    pub details: serde_json::Value,

    /// Associated request ID (if any)
    pub request_id: Option<String>,

    /// IP address (if applicable)
    pub ip_address: Option<String>,
}

/// Types of audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuditEventType {
    // Action events
    ActionRequested { action: ActionType, context: String },
    ActionApproved { action: ActionType, approver: String },
    ActionDenied { action: ActionType, reason: String },
    ActionExecuted { action: ActionType, result: String },
    ActionFailed { action: ActionType, error: String },

    // Resource events
    ResourceLimitExceeded { resource: String, limit: String, current: String },
    ResourceUsageHigh { resource: String, usage_percent: f32 },

    // Security events
    AuthenticationFailed { method: String, reason: String },
    UnauthorizedAccess { resource: String, attempted_action: String },
    SuspiciousActivity { description: String, indicators: Vec<String> },

    // System events
    SystemStartup { version: String, config: serde_json::Value },
    SystemShutdown { reason: String },
    ConfigurationChanged { setting: String, old_value: String, new_value: String },
    EmergencyStop { trigger: String, reason: String },
}

/// Configuration for audit logger
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Directory to store audit logs
    pub log_dir: PathBuf,

    /// Maximum events in memory
    pub max_memory_events: usize,

    /// Whether to log to file
    pub file_logging: bool,

    /// Whether to encrypt logs
    pub encrypt_logs: bool,

    /// Retention period in days
    pub retention_days: u32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("data/audit"),
            max_memory_events: 10000,
            file_logging: true,
            encrypt_logs: false, // Would be true in production
            retention_days: 90,
        }
    }
}

/// Audit trail for querying historical events
#[derive(Debug)]
pub struct AuditTrail {
    events: Arc<RwLock<VecDeque<AuditEvent>>>,
    config: AuditConfig,
}

impl AuditTrail {
    /// Create new audit trail
    pub fn new(config: AuditConfig) -> Self {
        Self { events: Arc::new(RwLock::new(VecDeque::new())), config }
    }

    /// Add event to trail
    pub async fn add_event(&self, event: AuditEvent) -> Result<()> {
        let mut events = self.events.write().await;
        events.push_back(event);

        // Limit memory usage
        while events.len() > self.config.max_memory_events {
            events.pop_front();
        }

        Ok(())
    }

    /// Query events by time range
    pub async fn query_by_time(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<AuditEvent> {
        let events = self.events.read().await;
        events.iter().filter(|e| e.timestamp >= start && e.timestamp <= end).cloned().collect()
    }

    /// Query events by severity
    pub async fn query_by_severity(&self, min_severity: AuditSeverity) -> Vec<AuditEvent> {
        let events = self.events.read().await;
        events.iter().filter(|e| e.severity >= min_severity).cloned().collect()
    }

    /// Query events by actor
    pub async fn query_by_actor(&self, actor: &str) -> Vec<AuditEvent> {
        let events = self.events.read().await;
        events.iter().filter(|e| e.actor == actor).cloned().collect()
    }

    /// Get recent events
    pub async fn get_recent(&self, count: usize) -> Vec<AuditEvent> {
        let events = self.events.read().await;
        events.iter().rev().take(count).cloned().collect()
    }
}

/// Main audit logger
pub struct AuditLogger {
    config: AuditConfig,
    trail: Arc<AuditTrail>,
    event_tx: mpsc::Sender<AuditEvent>,
    event_rx: Arc<RwLock<mpsc::Receiver<AuditEvent>>>,
    shutdown_tx: tokio::sync::broadcast::Sender<()>,
}

impl AuditLogger {
    /// Create new audit logger
    pub async fn new(config: AuditConfig) -> Result<Self> {
        // Create log directory
        if config.file_logging {
            tokio::fs::create_dir_all(&config.log_dir).await?;
        }

        let (event_tx, event_rx) = mpsc::channel(1000);
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);

        Ok(Self {
            trail: Arc::new(AuditTrail::new(config.clone())),
            config,
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
            shutdown_tx,
        })
    }

    /// Start the audit logger
    pub async fn start(&self) -> Result<()> {
        if self.config.file_logging {
            let logger = self.clone();
            tokio::spawn(async move {
                logger.file_logging_loop().await;
            });
        }

        // Log startup
        self.log_system_startup().await?;

        Ok(())
    }

    /// Log an action request
    pub async fn log_action_request(
        &self,
        actor: &str,
        action: &ActionType,
        context: &str,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::ActionRequested {
                action: action.clone(),
                context: context.to_string(),
            },
            severity: AuditSeverity::Info,
            actor: actor.to_string(),
            details: serde_json::json!({
                "risk_level": format!("{:?}", action.risk_level()),
            }),
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await
    }

    /// Log an action decision
    pub async fn log_action_decision(
        &self,
        actor: &str,
        action: &ActionType,
        decision: &ActionDecision,
    ) -> Result<()> {
        let (event_type, severity) = match decision {
            ActionDecision::Approve => (
                AuditEventType::ActionApproved {
                    action: action.clone(),
                    approver: actor.to_string(),
                },
                AuditSeverity::Info,
            ),
            ActionDecision::Deny { reason } => (
                AuditEventType::ActionDenied { action: action.clone(), reason: reason.clone() },
                AuditSeverity::Warning,
            ),
            ActionDecision::Defer { .. } => return Ok(()), // Don't log defers
        };

        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type,
            severity,
            actor: actor.to_string(),
            details: serde_json::Value::Null,
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await
    }

    /// Log a resource limit exceeded
    pub async fn log_resource_limit(
        &self,
        resource: &str,
        limit: &str,
        current: &str,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::ResourceLimitExceeded {
                resource: resource.to_string(),
                limit: limit.to_string(),
                current: current.to_string(),
            },
            severity: AuditSeverity::Error,
            actor: "system".to_string(),
            details: serde_json::Value::Null,
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await
    }

    /// Log suspicious activity
    pub async fn log_suspicious_activity(
        &self,
        actor: &str,
        description: &str,
        indicators: Vec<String>,
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::SuspiciousActivity {
                description: description.to_string(),
                indicators,
            },
            severity: AuditSeverity::Critical,
            actor: actor.to_string(),
            details: serde_json::Value::Null,
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await
    }

    /// Log emergency stop
    pub async fn log_emergency_stop(&self, trigger: &str, reason: &str) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::EmergencyStop {
                trigger: trigger.to_string(),
                reason: reason.to_string(),
            },
            severity: AuditSeverity::Critical,
            actor: "system".to_string(),
            details: serde_json::Value::Null,
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await
    }

    /// Core logging function
    async fn log_event(&self, event: AuditEvent) -> Result<()> {
        // Add to in-memory trail
        self.trail.add_event(event.clone()).await?;

        // Send to file logger
        if self.config.file_logging {
            let _ = self.event_tx.send(event.clone()).await;
        }

        // Log to tracing based on severity
        match event.severity {
            AuditSeverity::Debug => {
                tracing::debug!("Audit: {:?}", event.event_type);
            }
            AuditSeverity::Info => {
                tracing::info!("Audit: {:?}", event.event_type);
            }
            AuditSeverity::Warning => {
                tracing::warn!("Audit: {:?}", event.event_type);
            }
            AuditSeverity::Error => {
                tracing::error!("Audit: {:?}", event.event_type);
            }
            AuditSeverity::Critical => {
                tracing::error!("CRITICAL AUDIT: {:?}", event.event_type);
            }
        }

        Ok(())
    }

    /// File logging loop
    async fn file_logging_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut event_rx = self.event_rx.write().await;

        loop {
            tokio::select! {
                Some(event) = event_rx.recv() => {
                    if let Err(e) = self.write_event_to_file(&event).await {
                        error!("Failed to write audit event to file: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Audit file logger shutting down");
                    break;
                }
            }
        }
    }

    /// Write event to file
    async fn write_event_to_file(&self, event: &AuditEvent) -> Result<()> {
        let date = event.timestamp.format("%Y-%m-%d");
        let filename = self.config.log_dir.join(format!("audit-{}.json", date));

        let mut file = OpenOptions::new().create(true).append(true).open(&filename).await?;

        let json = serde_json::to_string(event)?;
        file.write_all(json.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;

        Ok(())
    }

    /// Log system startup
    async fn log_system_startup(&self) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::SystemStartup {
                version: std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string()),
                config: serde_json::json!({
                    "safe_mode": true,
                    "audit_enabled": true,
                }),
            },
            severity: AuditSeverity::Info,
            actor: "system".to_string(),
            details: serde_json::Value::Null,
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await
    }

    /// Get the audit trail
    pub fn trail(&self) -> &Arc<AuditTrail> {
        &self.trail
    }

    /// Shutdown the logger
    pub async fn shutdown(&self) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::SystemShutdown { reason: "Normal shutdown".to_string() },
            severity: AuditSeverity::Info,
            actor: "system".to_string(),
            details: serde_json::Value::Null,
            request_id: None,
            ip_address: None,
        };

        self.log_event(event).await?;
        let _ = self.shutdown_tx.send(());

        Ok(())
    }
}

// Make AuditLogger cloneable
impl Clone for AuditLogger {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            trail: Arc::clone(&self.trail),
            event_tx: self.event_tx.clone(),
            event_rx: Arc::clone(&self.event_rx),
            shutdown_tx: self.shutdown_tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_trail_queries() {
        let config = AuditConfig { file_logging: false, ..Default::default() };

        let logger = AuditLogger::new(config).await.unwrap();

        // Log some events
        logger
            .log_action_request(
                "test_user",
                &ActionType::FileRead { path: "test.txt".to_string() },
                "testing",
            )
            .await
            .unwrap();

        // Query recent
        let recent = logger.trail().get_recent(10).await;
        assert!(!recent.is_empty());
    }
}
