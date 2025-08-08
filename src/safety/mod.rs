//! Safety and Security Module
//!
//! This module provides critical safety mechanisms for Loki including:
//! - Action validation and approval
//! - Resource limits and monitoring
//! - Emergency shutdown procedures
//! - Audit logging
//! - Encryption for sensitive data

pub mod audit;
pub mod encryption;
pub mod integration_test;
pub mod limits;
pub mod multi_agent_safety;
pub mod security_audit;
pub mod validator;

pub use audit::{AuditConfig, AuditEvent, AuditEventType, AuditLogger, AuditSeverity, AuditTrail};
pub use encryption::{EncryptedData, EncryptionAlgorithm, SecurityEncryption};
pub use security_audit::{
    SecurityAuditConfig, SecurityAuditEvent, SecurityAuditSystem, SecurityEventType, SecuritySeverity,
    init_security_audit, record_security_event, security_audit,
};
pub use limits::{
    LimitExceeded,
    RateLimit,
    ResourceLimits,
    ResourceMonitor,
    ResourceUsage,
    TokenBudget,
};
pub use multi_agent_safety::{
    AgentPermissions,
    AgentSafetyProfile,
    CollectiveActionValidator,
    EmergentBehaviorMonitor,
    MultiAgentAuditEvent,
    MultiAgentSafetyConfig,
    MultiAgentSafetyCoordinator,
    SafetyViolation,
    SafetyViolationType,
};
pub use validator::{
    ActionDecision,
    ActionType,
    ActionValidator,
    PendingAction,
    RiskLevel,
    SafetyStatistics,
    ValidationError,
    ValidationResult,
    ValidatorConfig,
};
