//! Error Handling and Recovery System
//! 
//! Provides comprehensive error handling, recovery strategies, and graceful degradation
//! for the TUI system.

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Error};
use tracing::{error, warn, info};

use crate::tui::event_bus::{EventBus, SystemEvent, TabId, ErrorSeverity};

/// Error handler for TUI system
#[derive(Clone)]
pub struct ErrorHandler {
    /// Error recovery strategies
    strategies: Arc<RwLock<HashMap<ErrorCategory, Box<dyn RecoveryStrategy>>>>,
    
    /// Error history
    history: Arc<RwLock<ErrorHistory>>,
    
    /// Circuit breakers
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    
    /// Retry policies
    retry_policies: Arc<RwLock<HashMap<String, RetryPolicy>>>,
    
    /// Event bus for notifications
    event_bus: Arc<EventBus>,
    
    /// Error metrics
    metrics: Arc<RwLock<ErrorMetrics>>,
}

/// Error categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    Network,
    Model,
    Database,
    FileSystem,
    Memory,
    Timeout,
    Validation,
    Configuration,
    Unknown,
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorCategory::Network => write!(f, "network"),
            ErrorCategory::Model => write!(f, "model"),
            ErrorCategory::Database => write!(f, "database"),
            ErrorCategory::FileSystem => write!(f, "filesystem"),
            ErrorCategory::Memory => write!(f, "memory"),
            ErrorCategory::Timeout => write!(f, "timeout"),
            ErrorCategory::Validation => write!(f, "validation"),
            ErrorCategory::Configuration => write!(f, "configuration"),
            ErrorCategory::Unknown => write!(f, "unknown"),
        }
    }
}

/// Recovery strategy trait
#[async_trait::async_trait]
pub trait RecoveryStrategy: Send + Sync {
    /// Attempt recovery
    async fn recover(&self, error: &TUIError) -> Result<RecoveryAction>;
    
    /// Strategy name
    fn name(&self) -> &str;
}

/// TUI system error
#[derive(Debug, Clone)]
pub struct TUIError {
    pub category: ErrorCategory,
    pub message: String,
    pub source: TabId,
    pub context: HashMap<String, serde_json::Value>,
    pub timestamp: Instant,
    pub retryable: bool,
}

impl std::fmt::Display for TUIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.category, self.message)
    }
}

impl std::error::Error for TUIError {}

/// Recovery actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    Retry { delay_ms: u64 },
    Fallback { alternative: String },
    Skip,
    Abort,
    Degrade { feature: String },
    Restart { component: String },
}

/// Error history
struct ErrorHistory {
    errors: VecDeque<ErrorRecord>,
    max_size: usize,
}

/// Error record
#[derive(Debug, Clone)]
struct ErrorRecord {
    error: TUIError,
    recovery_attempted: bool,
    recovery_result: Option<RecoveryResult>,
    timestamp: Instant,
}

/// Recovery result
#[derive(Debug, Clone)]
enum RecoveryResult {
    Success,
    Failed(String),
    Partial(String),
}

/// Circuit breaker for preventing cascading failures
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub name: String,
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure: Option<Instant>,
    pub config: CircuitBreakerConfig,
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing, reject requests
    HalfOpen,   // Testing recovery
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
}

/// Retry policy
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff: BackoffStrategy,
    pub jitter: bool,
}

/// Backoff strategies
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    Fixed(Duration),
    Linear { initial: Duration, increment: Duration },
    Exponential { initial: Duration, multiplier: f64, max: Duration },
}

/// Error metrics
#[derive(Debug, Default)]
struct ErrorMetrics {
    total_errors: u64,
    recovered_errors: u64,
    failed_recoveries: u64,
    errors_by_category: HashMap<ErrorCategory, u64>,
    circuit_breaker_trips: u64,
}

impl ErrorHandler {
    /// Create a new error handler
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        let handler = Self {
            strategies: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(ErrorHistory::new(1000))),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            retry_policies: Arc::new(RwLock::new(HashMap::new())),
            event_bus,
            metrics: Arc::new(RwLock::new(ErrorMetrics::default())),
        };
        
        // Register default strategies
        let init_handler = handler.clone();
        tokio::spawn(async move {
            init_handler.register_default_strategies().await;
            init_handler.register_default_policies().await;
        });
        
        handler
    }
    
    /// Register default recovery strategies
    async fn register_default_strategies(&self) {
        let mut strategies = self.strategies.write().await;
        
        strategies.insert(ErrorCategory::Network, Box::new(NetworkRecoveryStrategy));
        strategies.insert(ErrorCategory::Model, Box::new(ModelRecoveryStrategy));
        strategies.insert(ErrorCategory::Database, Box::new(DatabaseRecoveryStrategy));
        strategies.insert(ErrorCategory::Timeout, Box::new(TimeoutRecoveryStrategy));
        strategies.insert(ErrorCategory::Memory, Box::new(MemoryRecoveryStrategy));
    }
    
    /// Register default retry policies
    async fn register_default_policies(&self) {
        let mut policies = self.retry_policies.write().await;
        
        // Network retry policy
        policies.insert("network".to_string(), RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                initial: Duration::from_millis(100),
                multiplier: 2.0,
                max: Duration::from_secs(10),
            },
            jitter: true,
        });
        
        // Model retry policy
        policies.insert("model".to_string(), RetryPolicy {
            max_attempts: 2,
            backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            jitter: false,
        });
        
        // Database retry policy
        policies.insert("database".to_string(), RetryPolicy {
            max_attempts: 5,
            backoff: BackoffStrategy::Linear {
                initial: Duration::from_millis(500),
                increment: Duration::from_millis(500),
            },
            jitter: true,
        });
    }
    
    /// Handle an error
    pub async fn handle_error(&self, error: TUIError) -> Result<()> {
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_errors += 1;
            *metrics.errors_by_category.entry(error.category).or_insert(0) += 1;
        }
        
        // Check circuit breaker
        if let Some(breaker) = self.get_circuit_breaker(&error.category.to_string()).await {
            if !self.check_circuit_breaker(breaker).await? {
                warn!("Circuit breaker open for {:?}", error.category);
                return Err(anyhow::anyhow!("Circuit breaker open"));
            }
        }
        
        // Record error
        self.record_error(error.clone()).await;
        
        // Attempt recovery
        let recovery_result = self.attempt_recovery(&error).await;
        
        // Update history
        self.update_recovery_result(&error, recovery_result.clone()).await;
        
        // Publish error event
        self.publish_error_event(&error, recovery_result).await?;
        
        Ok(())
    }
    
    /// Attempt recovery for an error
    async fn attempt_recovery(&self, error: &TUIError) -> Option<RecoveryResult> {
        let strategies = self.strategies.read().await;
        
        if let Some(strategy) = strategies.get(&error.category) {
            match strategy.recover(error).await {
                Ok(action) => {
                    info!("Recovery action for {:?}: {:?}", error.category, action);
                    
                    // Execute recovery action
                    match self.execute_recovery_action(action, error).await {
                        Ok(_) => {
                            let mut metrics = self.metrics.write().await;
                            metrics.recovered_errors += 1;
                            Some(RecoveryResult::Success)
                        }
                        Err(e) => {
                            let mut metrics = self.metrics.write().await;
                            metrics.failed_recoveries += 1;
                            Some(RecoveryResult::Failed(e.to_string()))
                        }
                    }
                }
                Err(e) => {
                    error!("Recovery strategy failed: {}", e);
                    Some(RecoveryResult::Failed(e.to_string()))
                }
            }
        } else {
            warn!("No recovery strategy for {:?}", error.category);
            None
        }
    }
    
    /// Execute recovery action
    async fn execute_recovery_action(
        &self,
        action: RecoveryAction,
        error: &TUIError,
    ) -> Result<()> {
        match action {
            RecoveryAction::Retry { delay_ms } => {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                info!("Retrying after {} ms", delay_ms);
                Ok(())
            }
            RecoveryAction::Fallback { alternative } => {
                info!("Using fallback: {}", alternative);
                Ok(())
            }
            RecoveryAction::Skip => {
                info!("Skipping failed operation");
                Ok(())
            }
            RecoveryAction::Abort => {
                error!("Aborting due to error");
                Err(anyhow::anyhow!("Operation aborted"))
            }
            RecoveryAction::Degrade { feature } => {
                warn!("Degrading feature: {}", feature);
                Ok(())
            }
            RecoveryAction::Restart { component } => {
                info!("Restarting component: {}", component);
                Ok(())
            }
        }
    }
    
    /// Get circuit breaker for a component
    async fn get_circuit_breaker(&self, name: &str) -> Option<CircuitBreaker> {
        let breakers = self.circuit_breakers.read().await;
        breakers.get(name).cloned()
    }
    
    /// Check circuit breaker state
    async fn check_circuit_breaker(&self, mut breaker: CircuitBreaker) -> Result<bool> {
        match breaker.state {
            CircuitState::Closed => Ok(true),
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = breaker.last_failure {
                    if last_failure.elapsed() > breaker.config.timeout {
                        // Transition to half-open
                        breaker.state = CircuitState::HalfOpen;
                        breaker.success_count = 0;
                        
                        let mut breakers = self.circuit_breakers.write().await;
                        breakers.insert(breaker.name.clone(), breaker);
                        
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited calls in half-open state
                Ok(breaker.success_count < breaker.config.half_open_max_calls)
            }
        }
    }
    
    /// Record error in history
    async fn record_error(&self, error: TUIError) {
        let mut history = self.history.write().await;
        history.add(ErrorRecord {
            error,
            recovery_attempted: false,
            recovery_result: None,
            timestamp: Instant::now(),
        });
    }
    
    /// Update recovery result
    async fn update_recovery_result(&self, error: &TUIError, result: Option<RecoveryResult>) {
        let mut history = self.history.write().await;
        // Find and update the most recent matching error
        for record in history.errors.iter_mut().rev() {
            if record.error.message == error.message && record.recovery_result.is_none() {
                record.recovery_attempted = true;
                record.recovery_result = result;
                break;
            }
        }
    }
    
    /// Publish error event
    async fn publish_error_event(&self, error: &TUIError, recovery: Option<RecoveryResult>) -> Result<()> {
        let severity = match error.category {
            ErrorCategory::Memory => ErrorSeverity::Critical,
            ErrorCategory::Network | ErrorCategory::Model => ErrorSeverity::Warning,
            _ => ErrorSeverity::Error,
        };
        
        let mut message = error.message.clone();
        if let Some(result) = recovery {
            message.push_str(&format!(" (Recovery: {:?})", result));
        }
        
        self.event_bus.publish(SystemEvent::ErrorOccurred {
            source: error.source.clone(),
            error: message,
            severity,
        }).await?;
        
        Ok(())
    }
    
    /// Get error metrics
    pub async fn get_metrics(&self) -> ErrorMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Clear error history
    pub async fn clear_history(&self) {
        let mut history = self.history.write().await;
        history.clear();
    }
}

impl ErrorHistory {
    fn new(max_size: usize) -> Self {
        Self {
            errors: VecDeque::with_capacity(max_size),
            max_size,
        }
    }
    
    fn add(&mut self, record: ErrorRecord) {
        if self.errors.len() >= self.max_size {
            self.errors.pop_front();
        }
        self.errors.push_back(record);
    }
    
    fn clear(&mut self) {
        self.errors.clear();
    }
}

// Recovery Strategies

struct NetworkRecoveryStrategy;

#[async_trait::async_trait]
impl RecoveryStrategy for NetworkRecoveryStrategy {
    async fn recover(&self, error: &TUIError) -> Result<RecoveryAction> {
        if error.retryable {
            Ok(RecoveryAction::Retry { delay_ms: 1000 })
        } else {
            Ok(RecoveryAction::Fallback {
                alternative: "offline_mode".to_string(),
            })
        }
    }
    
    fn name(&self) -> &str {
        "network_recovery"
    }
}

struct ModelRecoveryStrategy;

#[async_trait::async_trait]
impl RecoveryStrategy for ModelRecoveryStrategy {
    async fn recover(&self, error: &TUIError) -> Result<RecoveryAction> {
        Ok(RecoveryAction::Fallback {
            alternative: "alternative_model".to_string(),
        })
    }
    
    fn name(&self) -> &str {
        "model_recovery"
    }
}

struct DatabaseRecoveryStrategy;

#[async_trait::async_trait]
impl RecoveryStrategy for DatabaseRecoveryStrategy {
    async fn recover(&self, error: &TUIError) -> Result<RecoveryAction> {
        if error.retryable {
            Ok(RecoveryAction::Retry { delay_ms: 500 })
        } else {
            Ok(RecoveryAction::Degrade {
                feature: "database_operations".to_string(),
            })
        }
    }
    
    fn name(&self) -> &str {
        "database_recovery"
    }
}

struct TimeoutRecoveryStrategy;

#[async_trait::async_trait]
impl RecoveryStrategy for TimeoutRecoveryStrategy {
    async fn recover(&self, _error: &TUIError) -> Result<RecoveryAction> {
        Ok(RecoveryAction::Skip)
    }
    
    fn name(&self) -> &str {
        "timeout_recovery"
    }
}

struct MemoryRecoveryStrategy;

#[async_trait::async_trait]
impl RecoveryStrategy for MemoryRecoveryStrategy {
    async fn recover(&self, _error: &TUIError) -> Result<RecoveryAction> {
        Ok(RecoveryAction::Restart {
            component: "memory_intensive_component".to_string(),
        })
    }
    
    fn name(&self) -> &str {
        "memory_recovery"
    }
}

impl Clone for ErrorMetrics {
    fn clone(&self) -> Self {
        Self {
            total_errors: self.total_errors,
            recovered_errors: self.recovered_errors,
            failed_recoveries: self.failed_recoveries,
            errors_by_category: self.errors_by_category.clone(),
            circuit_breaker_trips: self.circuit_breaker_trips,
        }
    }
}

/// Create TUI error from any error
pub fn tui_error_from(error: Error, category: ErrorCategory, source: TabId) -> TUIError {
    TUIError {
        category,
        message: error.to_string(),
        source,
        context: HashMap::new(),
        timestamp: Instant::now(),
        retryable: matches!(category, ErrorCategory::Network | ErrorCategory::Database),
    }
}