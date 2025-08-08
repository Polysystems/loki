//! Dedicated Security Audit System
//!
//! Provides specialized security event logging and monitoring
//! separate from general audit logging for enhanced security visibility.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Security event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    /// Decision encryption/decryption events
    DecisionEncrypted {
        action_id: String,
        algorithm: String,
    },
    DecisionDecrypted {
        action_id: String,
        requester: String,
    },
    DecisionStorageAccess {
        action_id: String,
        operation: String,
        user_id: String,
    },

    /// Resource security events
    ResourceThresholdBreached {
        resource: String,
        current_value: f64,
        threshold: f64,
    },
    RateLimitExceeded {
        service: String,
        user_id: String,
        limit: u32,
    },

    /// Authentication events
    AuthenticationAttempt {
        user_id: String,
        success: bool,
        method: String,
    },
    PrivilegeEscalation {
        user_id: String,
        from_role: String,
        to_role: String,
    },

    /// Anomaly detection
    AnomalousPattern {
        pattern_type: String,
        confidence: f64,
        indicators: Vec<String>,
    },
    SecurityViolation {
        violation_type: String,
        severity: SecuritySeverity,
        details: String,
    },

    /// System security events
    EncryptionKeyRotation {
        key_type: String,
        rotation_reason: String,
    },
    SecurityConfigChange {
        setting: String,
        old_value: String,
        new_value: String,
    },
}

/// Security event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Security audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditEvent {
    /// Event ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: SecurityEventType,
    /// Severity
    pub severity: SecuritySeverity,
    /// Source component
    pub source: String,
    /// User context (if applicable)
    pub user_context: Option<String>,
    /// IP address (if applicable)
    pub ip_address: Option<String>,
    /// Additional metadata
    pub metadata: serde_json::Value,
    /// Correlation ID for related events
    pub correlation_id: Option<String>,
}

/// Security audit configuration
#[derive(Debug, Clone)]
pub struct SecurityAuditConfig {
    /// Storage path for security events
    pub storage_path: PathBuf,
    /// Enable real-time alerting
    pub enable_alerts: bool,
    /// Alert channel capacity
    pub alert_channel_size: usize,
    /// Retention period in days
    pub retention_days: u32,
    /// Enable encryption for stored events
    pub encrypt_storage: bool,
}

impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("data/security_audit"),
            enable_alerts: true,
            alert_channel_size: 1000,
            retention_days: 365, // 1 year for security events
            encrypt_storage: true,
        }
    }
}

/// Security alert
#[derive(Debug, Clone)]
pub struct SecurityAlert {
    pub event: SecurityAuditEvent,
    pub alert_time: DateTime<Utc>,
    pub requires_immediate_action: bool,
}

/// Dedicated security audit system
pub struct SecurityAuditSystem {
    /// Configuration
    config: SecurityAuditConfig,
    /// Event storage
    storage: Arc<RwLock<Option<DB>>>,
    /// Alert channel
    alert_tx: mpsc::Sender<SecurityAlert>,
    alert_rx: Arc<RwLock<mpsc::Receiver<SecurityAlert>>>,
    /// Event statistics
    statistics: Arc<RwLock<SecurityStatistics>>,
}

/// Security statistics
#[derive(Debug, Default)]
struct SecurityStatistics {
    total_events: u64,
    events_by_type: HashMap<String, u64>,
    events_by_severity: HashMap<SecuritySeverity, u64>,
    last_critical_event: Option<DateTime<Utc>>,
    active_alerts: usize,
}

impl SecurityAuditSystem {
    /// Create new security audit system
    pub async fn new(config: SecurityAuditConfig) -> Result<Self> {
        // Create storage directory
        tokio::fs::create_dir_all(&config.storage_path)
            .await
            .with_context(|| format!("Failed to create security audit directory: {:?}", config.storage_path))?;

        let (alert_tx, alert_rx) = mpsc::channel(config.alert_channel_size);

        let mut system = Self {
            config,
            storage: Arc::new(RwLock::new(None)),
            alert_tx,
            alert_rx: Arc::new(RwLock::new(alert_rx)),
            statistics: Arc::new(RwLock::new(SecurityStatistics::default())),
        };

        // Initialize storage
        system.initialize_storage().await?;

        info!("🛡️ Security audit system initialized");
        Ok(system)
    }

    /// Initialize secure storage
    async fn initialize_storage(&mut self) -> Result<()> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let db_path = self.config.storage_path.join("events.db");
        let db = DB::open(&opts, &db_path)
            .with_context(|| format!("Failed to open security audit database: {:?}", db_path))?;

        *self.storage.write().await = Some(db);

        Ok(())
    }

    /// Record a security event
    pub async fn record_event(
        &self,
        event_type: SecurityEventType,
        source: &str,
        user_context: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<String> {
        let severity = self.determine_severity(&event_type);
        let event_id = Uuid::new_v4().to_string();

        let event = SecurityAuditEvent {
            id: event_id.clone(),
            timestamp: Utc::now(),
            event_type: event_type.clone(),
            severity,
            source: source.to_string(),
            user_context,
            ip_address: None, // Would be populated from request context
            metadata,
            correlation_id: None,
        };

        // Store event
        self.store_event(&event).await?;

        // Update statistics
        self.update_statistics(&event).await;

        // Check if alert is needed
        if self.should_alert(&event) {
            self.send_alert(event.clone()).await?;
        }

        // Log based on severity
        match severity {
            SecuritySeverity::Emergency => {
                error!("🚨 SECURITY EMERGENCY: {:?}", event_type);
            }
            SecuritySeverity::Critical => {
                error!("🔴 SECURITY CRITICAL: {:?}", event_type);
            }
            SecuritySeverity::High => {
                warn!("🟠 SECURITY HIGH: {:?}", event_type);
            }
            SecuritySeverity::Medium => {
                warn!("🟡 SECURITY MEDIUM: {:?}", event_type);
            }
            SecuritySeverity::Low => {
                info!("🟢 SECURITY LOW: {:?}", event_type);
            }
        }

        Ok(event_id)
    }

    /// Record decision-related security event
    pub async fn record_decision_event(
        &self,
        action_id: &str,
        operation: &str,
        user_id: &str,
        metadata: serde_json::Value,
    ) -> Result<()> {
        let event_type = SecurityEventType::DecisionStorageAccess {
            action_id: action_id.to_string(),
            operation: operation.to_string(),
            user_id: user_id.to_string(),
        };

        self.record_event(
            event_type,
            "safety_validator",
            Some(user_id.to_string()),
            metadata,
        ).await?;

        Ok(())
    }

    /// Store event to database
    async fn store_event(&self, event: &SecurityAuditEvent) -> Result<()> {
        let storage = self.storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Security audit storage not initialized"))?;

        // Serialize event
        let event_data = if self.config.encrypt_storage {
            // In production, encrypt the event data
            serde_json::to_vec(event)?
        } else {
            serde_json::to_vec(event)?
        };

        // Store with timestamp-based key for ordering
        let key = format!("event:{:?}:{}", event.timestamp.timestamp_nanos_opt(), event.id);
        db.put(key.as_bytes(), &event_data)?;

        // Also store by event ID for quick lookup
        let id_key = format!("id:{}", event.id);
        db.put(id_key.as_bytes(), key.as_bytes())?;

        Ok(())
    }

    /// Update statistics
    async fn update_statistics(&self, event: &SecurityAuditEvent) {
        let mut stats = self.statistics.write().await;

        stats.total_events += 1;

        // Count by type
        let type_name = format!("{:?}", event.event_type);
        *stats.events_by_type.entry(type_name).or_insert(0) += 1;

        // Count by severity
        *stats.events_by_severity.entry(event.severity).or_insert(0) += 1;

        // Track last critical event
        if event.severity >= SecuritySeverity::Critical {
            stats.last_critical_event = Some(event.timestamp);
        }
    }

    /// Determine event severity
    fn determine_severity(&self, event_type: &SecurityEventType) -> SecuritySeverity {
        match event_type {
            SecurityEventType::DecisionEncrypted { .. } => SecuritySeverity::Low,
            SecurityEventType::DecisionDecrypted { .. } => SecuritySeverity::Medium,
            SecurityEventType::DecisionStorageAccess { .. } => SecuritySeverity::Low,

            SecurityEventType::ResourceThresholdBreached { .. } => SecuritySeverity::High,
            SecurityEventType::RateLimitExceeded { .. } => SecuritySeverity::Medium,

            SecurityEventType::AuthenticationAttempt { success, .. } => {
                if *success {
                    SecuritySeverity::Low
                } else {
                    SecuritySeverity::Medium
                }
            }
            SecurityEventType::PrivilegeEscalation { .. } => SecuritySeverity::Critical,

            SecurityEventType::AnomalousPattern { confidence, .. } => {
                if *confidence > 0.9 {
                    SecuritySeverity::Critical
                } else if *confidence > 0.7 {
                    SecuritySeverity::High
                } else {
                    SecuritySeverity::Medium
                }
            }
            SecurityEventType::SecurityViolation { severity, .. } => *severity,

            SecurityEventType::EncryptionKeyRotation { .. } => SecuritySeverity::Medium,
            SecurityEventType::SecurityConfigChange { .. } => SecuritySeverity::High,
        }
    }

    /// Check if event should trigger an alert
    fn should_alert(&self, event: &SecurityAuditEvent) -> bool {
        if !self.config.enable_alerts {
            return false;
        }

        // Alert on high severity and above
        event.severity >= SecuritySeverity::High
    }

    /// Send security alert
    async fn send_alert(&self, event: SecurityAuditEvent) -> Result<()> {
        let requires_immediate_action = event.severity >= SecuritySeverity::Critical;
        let alert = SecurityAlert {
            event,
            alert_time: Utc::now(),
            requires_immediate_action,
        };

        self.alert_tx.send(alert).await
            .map_err(|e| anyhow::anyhow!("Failed to send security alert: {}", e))?;

        Ok(())
    }

    /// Get recent security events
    pub async fn get_recent_events(&self, count: usize) -> Result<Vec<SecurityAuditEvent>> {
        let storage = self.storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Security audit storage not initialized"))?;

        let mut events = Vec::new();
        let iter = db.iterator(rocksdb::IteratorMode::End);
        let mut collected = 0;

        for item in iter {
            if collected >= count {
                break;
            }
            match item {
                Ok((key, value)) => {
                    if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                        if key_str.starts_with("event:") {
                            if let Ok(event) = serde_json::from_slice::<SecurityAuditEvent>(&value) {
                                events.push(event);
                                collected += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error reading from iterator: {}", e);
                    break;
                }
            }
        }

        Ok(events)
    }

    /// Get events by severity
    pub async fn get_events_by_severity(
        &self,
        severity: SecuritySeverity,
        since: DateTime<Utc>,
    ) -> Result<Vec<SecurityAuditEvent>> {
        let storage = self.storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Security audit storage not initialized"))?;

        let mut events = Vec::new();
        let start_key = format!("event:{:?}:", since.timestamp_nanos_opt());
        let iter = db.iterator(rocksdb::IteratorMode::From(start_key.as_bytes(), rocksdb::Direction::Forward));

        for item in iter {
            match item {
                Ok((key, value)) => {
                    if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                        if key_str.starts_with("event:") {
                            if let Ok(event) = serde_json::from_slice::<SecurityAuditEvent>(&value) {
                                if event.severity >= severity {
                                    events.push(event);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error reading from iterator: {}", e);
                    break;
                }
            }
        }

        Ok(events)
    }

    /// Get security statistics
    pub async fn get_statistics(&self) -> SecurityAuditStatistics {
        let stats = self.statistics.read().await;

        SecurityAuditStatistics {
            total_events: stats.total_events,
            events_by_type: stats.events_by_type.clone(),
            events_by_severity: stats.events_by_severity.clone(),
            last_critical_event: stats.last_critical_event,
            active_alerts: stats.active_alerts,
        }
    }

    /// Clean up old events
    pub async fn cleanup_old_events(&self) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(self.config.retention_days as i64);
        let storage = self.storage.read().await;
        let db = storage.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Security audit storage not initialized"))?;

        let mut batch = WriteBatch::default();
        let mut count = 0;

        let cutoff_key = format!("event:{:?}:", cutoff.timestamp_nanos_opt());
        let iter = db.iterator(rocksdb::IteratorMode::Start);

        for item in iter {
            match item {
                Ok((key, _)) => {
                    if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                        if key_str.starts_with("event:") && key_str < cutoff_key {
                            batch.delete(&key);
                            count += 1;
                        }
                    }
                }
                Err(e) => {
                    warn!("Error reading from iterator: {}", e);
                    break;
                }
            }
        }

        if count > 0 {
            db.write(batch)?;
            info!("Cleaned up {} old security events", count);
        }

        Ok(count)
    }
}

/// Public statistics structure
#[derive(Debug, Clone)]
pub struct SecurityAuditStatistics {
    pub total_events: u64,
    pub events_by_type: HashMap<String, u64>,
    pub events_by_severity: HashMap<SecuritySeverity, u64>,
    pub last_critical_event: Option<DateTime<Utc>>,
    pub active_alerts: usize,
}

/// Global security audit instance
static SECURITY_AUDIT: once_cell::sync::OnceCell<Arc<SecurityAuditSystem>> = once_cell::sync::OnceCell::new();

/// Initialize global security audit system
pub async fn init_security_audit(config: SecurityAuditConfig) -> Result<()> {
    let system = Arc::new(SecurityAuditSystem::new(config).await?);
    SECURITY_AUDIT.set(system)
        .map_err(|_| anyhow::anyhow!("Security audit system already initialized"))?;
    Ok(())
}

/// Get global security audit system
pub fn security_audit() -> Option<&'static Arc<SecurityAuditSystem>> {
    SECURITY_AUDIT.get()
}

/// Record security event using global instance
pub async fn record_security_event(
    event_type: SecurityEventType,
    source: &str,
    user_context: Option<String>,
    metadata: serde_json::Value,
) -> Result<String> {
    if let Some(audit) = security_audit() {
        audit.record_event(event_type, source, user_context, metadata).await
    } else {
        Err(anyhow::anyhow!("Security audit system not initialized"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_audit_system() {
        let config = SecurityAuditConfig {
            storage_path: PathBuf::from("data/test/security_audit"),
            ..Default::default()
        };

        let system = SecurityAuditSystem::new(config).await
            .expect("Failed to create security audit system");

        // Record various events
        let event_id = system.record_event(
            SecurityEventType::AuthenticationAttempt {
                user_id: "test_user".to_string(),
                success: true,
                method: "password".to_string(),
            },
            "auth_system",
            Some("test_user".to_string()),
            serde_json::json!({
                "ip": "127.0.0.1",
                "user_agent": "test"
            }),
        ).await.expect("Failed to record event");

        assert!(!event_id.is_empty());

        // Test decision event
        system.record_decision_event(
            "test_action_123",
            "encrypt",
            "test_user",
            serde_json::json!({
                "algorithm": "AES-256-GCM"
            }),
        ).await.expect("Failed to record decision event");

        // Get recent events
        let events = system.get_recent_events(10).await
            .expect("Failed to get recent events");

        assert!(!events.is_empty());
    }
}
