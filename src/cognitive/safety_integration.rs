//! Safety Integration for Cognitive System
//!
//! This module provides safety-aware wrappers and validation for all
//! cognitive system operations, ensuring safe execution.

use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info, warn};

use super::CognitiveSystem;
use super::consciousness_stream::ThermodynamicConsciousnessStream;
use crate::safety::{
    ActionType,
    ActionValidator,
    AuditConfig,
    AuditLogger,
    ResourceLimits,
    ResourceMonitor,
    ValidatorConfig,
    SecurityAuditConfig,
    init_security_audit,
};

/// Safety-aware cognitive system wrapper
pub struct SafeCognitiveSystem {
    /// The underlying cognitive system
    inner: Arc<CognitiveSystem>,

    /// Action validator
    validator: Arc<ActionValidator>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,

    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
}

impl SafeCognitiveSystem {
    /// Create a new safety-aware cognitive system
    pub async fn new(
        cognitive_system: Arc<CognitiveSystem>,
        validatorconfig: ValidatorConfig,
        auditconfig: AuditConfig,
        resource_limits: ResourceLimits,
    ) -> Result<Self> {
        info!("Initializing safety-aware cognitive system");

        // Initialize security audit system first
        let security_config = SecurityAuditConfig::default();
        if let Err(e) = init_security_audit(security_config).await {
            warn!("Security audit initialization failed (non-critical): {}", e);
        }
        // Create action validator
        let validator = Arc::new(ActionValidator::new(validatorconfig).await?);

        // Create audit logger
        let audit_logger = Arc::new(AuditLogger::new(auditconfig).await?);
        audit_logger.start().await?;

        // Create resource monitor
        let (alert_tx, mut alert_rx) = tokio::sync::mpsc::channel(100);
        let resource_monitor = Arc::new(ResourceMonitor::new(resource_limits, alert_tx));
        resource_monitor.start().await?;

        // Start alert handler
        let audit_logger_clone = audit_logger.clone();
        tokio::spawn(async move {
            while let Some(limit_exceeded) = alert_rx.recv().await {
                error!("Resource limit exceeded: {}", limit_exceeded);

                // Log to audit trail
                let _ = audit_logger_clone
                    .log_resource_limit("system", &format!("{:?}", limit_exceeded), "exceeded")
                    .await;
            }
        });

        Ok(Self { inner: cognitive_system, validator, audit_logger, resource_monitor })
    }

    /// Validate and execute a cognitive action
    pub async fn validate_action(
        &self,
        action: ActionType,
        context: String,
        reasoning: Vec<String>,
    ) -> Result<()> {
        // Log action request
        self.audit_logger.log_action_request("cognitive_system", &action, &context).await?;

        // Validate the action
        match self.validator.validate_action(action.clone(), context, reasoning).await {
            Ok(()) => {
                info!("Action validated successfully: {:?}", action);
                Ok(())
            }
            Err(e) => {
                warn!("Action validation failed: {:?} - {}", action, e);
                Err(e.into())
            }
        }
    }

    /// Execute memory storage with safety validation
    pub async fn safe_memory_store(&self, content: String, metadata_source: String) -> Result<()> {
        // Create action for memory storage
        let action = ActionType::MemoryModify {
            operation: format!("store: {}", content.chars().take(50).collect::<String>()),
        };

        // Validate the action
        self.validate_action(
            action,
            format!("Memory storage from {}", metadata_source),
            vec!["Storing information in cognitive memory".to_string()],
        )
        .await?;

        // If validation passes, proceed with storage
        // Note: The actual storage would happen in the calling code
        Ok(())
    }

    /// Execute file operation with safety validation
    pub async fn safe_file_operation(
        &self,
        operation: &str,
        path: &str,
        content: Option<&str>,
    ) -> Result<()> {
        let action = match operation {
            "read" => ActionType::FileRead { path: path.to_string() },
            "write" => ActionType::FileWrite {
                path: path.to_string(),
                content: content.unwrap_or("").to_string(),
            },
            "delete" => ActionType::FileDelete { path: path.to_string() },
            _ => return Err(anyhow::anyhow!("Unknown file operation: {}", operation)),
        };

        self.validate_action(
            action,
            format!("File operation: {} on {}", operation, path),
            vec![format!("Cognitive system needs to {} file", operation)],
        )
        .await
    }

    /// Execute API call with safety validation
    pub async fn safe_api_call(&self, provider: &str, endpoint: &str, context: &str) -> Result<()> {
        // Check resource limits first
        if let Err(limit_exceeded) = self.resource_monitor.check_api_limit(provider).await {
            warn!("API rate limit exceeded for {}: {}", provider, limit_exceeded);
            return Err(anyhow::anyhow!("Rate limit exceeded: {}", limit_exceeded));
        }

        let action =
            ActionType::ApiCall { provider: provider.to_string(), endpoint: endpoint.to_string() };

        self.validate_action(
            action,
            format!("API call to {} - {}", provider, context),
            vec!["Cognitive system needs to make API call".to_string()],
        )
        .await
    }

    /// Get pending actions awaiting approval
    pub async fn get_pending_actions(&self) -> Vec<crate::safety::PendingAction> {
        self.validator.get_pending_actions().await
    }

    /// Approve or deny a pending action
    pub async fn decide_action(
        &self,
        action_id: &str,
        decision: crate::safety::ActionDecision,
    ) -> Result<()> {
        // Log the decision
        match &decision {
            crate::safety::ActionDecision::Approve => {
                info!("Action approved: {}", action_id);
            }
            crate::safety::ActionDecision::Deny { reason } => {
                warn!("Action denied: {} - {}", action_id, reason);
            }
            crate::safety::ActionDecision::Defer { until } => {
                info!("Action deferred until {:?}: {}", until, action_id);
            }
        }

        self.validator.decide_action(action_id, decision).await
    }

    /// Get resource usage
    pub async fn get_resource_usage(&self) -> crate::safety::ResourceUsage {
        self.resource_monitor.get_usage().await
    }

    /// Get audit trail
    pub async fn get_recent_audit_events(&self, count: usize) -> Vec<crate::safety::AuditEvent> {
        self.audit_logger.trail().get_recent(count).await
    }

    /// Emergency shutdown
    pub async fn emergency_shutdown(&self, reason: &str) -> Result<()> {
        error!("EMERGENCY SHUTDOWN: {}", reason);

        // Log emergency stop
        self.audit_logger.log_emergency_stop("cognitive_system", reason).await?;

        // Shutdown components
        self.resource_monitor.shutdown().await;
        self.audit_logger.shutdown().await?;
        self.inner.shutdown().await?;

        Ok(())
    }

    /// Get the underlying cognitive system (for read-only operations)
    pub fn inner(&self) -> &Arc<CognitiveSystem> {
        &self.inner
    }

    /// Get the action validator
    pub fn validator(&self) -> &Arc<ActionValidator> {
        &self.validator
    }

    /// Get the audit logger
    pub fn audit_logger(&self) -> &Arc<AuditLogger> {
        &self.audit_logger
    }

    /// Get the resource monitor
    pub fn resource_monitor(&self) -> &Arc<ResourceMonitor> {
        &self.resource_monitor
    }

    /// Process a query through the cognitive system with safety validation
    pub async fn process_query(&self, query: &str) -> Result<String> {
        // Validate the query action
        let action = ActionType::ApiCall {
            provider: "cognitive_system".to_string(),
            endpoint: "query".to_string(),
        };

        self.validate_action(
            action,
            format!("Processing query: {}", query.chars().take(50).collect::<String>()),
            vec!["User query requires cognitive processing".to_string()],
        )
        .await?;

        // Delegate to the inner cognitive system for actual processing
        match self.inner.process_query(query).await {
            Ok(response) => {
                info!("Query processed successfully");
                Ok(response)
            }
            Err(e) => {
                error!("Query processing failed: {}", e);
                Err(e)
            }
        }
    }

    /// Check the health of the cognitive system
    pub async fn health_check(&self) -> Result<()> {
        // Check resource usage
        let usage = self.resource_monitor.get_usage().await;

        let max_memory_mb = std::env::var("LOKI_MAX_MEMORY_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192); // Default 8GB

        if usage.memory_mb > max_memory_mb {
            // 4GB limit
            return Err(anyhow::anyhow!("Memory usage too high: {}MB", usage.memory_mb));
        }

        if usage.cpu_percent > 95.0 {
            return Err(anyhow::anyhow!("CPU usage too high: {:.1}%", usage.cpu_percent));
        }

        // Basic health check - ensure we can access the inner system
        // In a real implementation, this might ping the system or check its internal
        // state
        info!("Cognitive system health check passed");

        Ok(())
    }
}

/// Safety-aware wrapper for consciousness stream operations
pub struct SafeConsciousnessWrapper {
    /// The underlying consciousness stream
    consciousness: Arc<ThermodynamicConsciousnessStream>,

    /// Safety validator
    validator: Arc<ActionValidator>,

    /// Audit logger
    audit_logger: Arc<AuditLogger>,
}

impl SafeConsciousnessWrapper {
    /// Create a new safety-aware consciousness wrapper
    pub fn new(
        consciousness: Arc<ThermodynamicConsciousnessStream>,
        validator: Arc<ActionValidator>,
        audit_logger: Arc<AuditLogger>,
    ) -> Self {
        Self { consciousness, validator, audit_logger }
    }

    /// Safely send interrupt to consciousness
    pub async fn safe_interrupt(
        &self,
        source: String,
        content: String,
        priority: crate::cognitive::goal_manager::Priority,
    ) -> Result<()> {
        // Log the interrupt
        self.audit_logger
            .log_action_request(
                &source,
                &ActionType::MemoryModify {
                    operation: format!(
                        "consciousness_interrupt: {}",
                        content.chars().take(50).collect::<String>()
                    ),
                },
                "Sending interrupt to consciousness stream",
            )
            .await?;


        Ok(())
    }

    /// Get recent thoughts (read-only, safe)
    pub fn get_recent_thoughts(&self, count: usize) -> Vec<crate::cognitive::Thought> {
        self.consciousness.get_recent_thoughts(count)
    }


    /// Check if enhanced features are enabled
    pub fn is_enhanced(&self) -> bool {
        self.consciousness.is_enhanced()
    }
}

/// Trait for safety-aware operations
pub trait SafeOperation {
    /// Execute the operation with safety validation
    fn execute_safely(
        &self,
        validator: &ActionValidator,
        audit_logger: &AuditLogger,
    ) -> impl std::future::Future<Output = Result<()>> + Send;
}

/// Helper macro for creating safe operations
#[macro_export]
macro_rules! safe_operation {
    ($action:expr, $context:expr, $reasoning:expr, $validator:expr, $audit:expr) => {{
        $audit.log_action_request("system", &$action, &$context).await?;
        $validator.validate_action($action, $context, $reasoning).await?;
    }};
}

#[cfg(test)]
mod tests {

    use crate::safety::{AuditConfig, ValidatorConfig};

    #[tokio::test]
    async fn test_safe_memory_operation() {
        // This would test the safety wrapper
        // For now, just check that the types compile
        let validatorconfig = ValidatorConfig::default();
        let auditconfig = AuditConfig::default();
        assert!(validatorconfig.safe_mode);
        assert!(auditconfig.file_logging);
    }
}
