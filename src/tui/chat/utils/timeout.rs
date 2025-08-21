//! Timeout utilities for async operations
//! 
//! Provides configurable timeout handling with retry logic
//! and exponential backoff for resilient async operations.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::{timeout, sleep};
use tracing::{warn, debug};

use crate::tui::chat::error::{ChatError, ChatResult};

/// Default timeout duration for operations
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default timeout for quick operations
pub const QUICK_TIMEOUT: Duration = Duration::from_secs(5);

/// Default timeout for long operations
pub const LONG_TIMEOUT: Duration = Duration::from_secs(120);

/// Configuration for timeout behavior
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Maximum duration to wait
    pub timeout: Duration,
    
    /// Number of retry attempts
    pub retries: u32,
    
    /// Initial backoff duration
    pub initial_backoff: Duration,
    
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Maximum backoff duration
    pub max_backoff: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            retries: 3,
            initial_backoff: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(10),
        }
    }
}

impl TimeoutConfig {
    /// Create a quick timeout configuration
    pub fn quick() -> Self {
        Self {
            timeout: QUICK_TIMEOUT,
            retries: 1,
            initial_backoff: Duration::from_millis(50),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(1),
        }
    }
    
    /// Create a long timeout configuration
    pub fn long() -> Self {
        Self {
            timeout: LONG_TIMEOUT,
            retries: 5,
            initial_backoff: Duration::from_millis(500),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(30),
        }
    }
    
    /// Create a configuration with no retries
    pub fn no_retry() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            retries: 0,
            initial_backoff: Duration::from_millis(0),
            backoff_multiplier: 1.0,
            max_backoff: Duration::from_millis(0),
        }
    }
}

/// Execute an async operation with timeout
pub async fn with_timeout<F, T>(
    duration: Duration,
    operation: F,
) -> ChatResult<T>
where
    F: Future<Output = ChatResult<T>>,
{
    match timeout(duration, operation).await {
        Ok(result) => result,
        Err(_) => Err(ChatError::Timeout(duration)),
    }
}

/// Execute an async operation with configurable timeout and retry
pub async fn with_timeout_retry<F, Fut, T>(
    config: &TimeoutConfig,
    mut operation: F,
) -> ChatResult<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = ChatResult<T>>,
{
    let mut backoff = config.initial_backoff;
    
    for attempt in 0..=config.retries {
        debug!("Attempt {} of {}", attempt + 1, config.retries + 1);
        
        match timeout(config.timeout, operation()).await {
            Ok(Ok(result)) => return Ok(result),
            Ok(Err(e)) => {
                if attempt < config.retries {
                    warn!("Operation failed (attempt {}): {}", attempt + 1, e);
                    sleep(backoff).await;
                    
                    // Calculate next backoff
                    backoff = Duration::from_secs_f64(
                        (backoff.as_secs_f64() * config.backoff_multiplier)
                            .min(config.max_backoff.as_secs_f64())
                    );
                } else {
                    return Err(e);
                }
            }
            Err(_) => {
                if attempt < config.retries {
                    warn!("Operation timed out (attempt {})", attempt + 1);
                    sleep(backoff).await;
                    
                    // Calculate next backoff
                    backoff = Duration::from_secs_f64(
                        (backoff.as_secs_f64() * config.backoff_multiplier)
                            .min(config.max_backoff.as_secs_f64())
                    );
                } else {
                    return Err(ChatError::Timeout(config.timeout));
                }
            }
        }
    }
    
    Err(ChatError::Timeout(config.timeout))
}

/// Execute multiple async operations with individual timeouts
pub async fn with_timeout_all<T>(
    operations: Vec<(Duration, impl Future<Output = ChatResult<T>>)>,
) -> Vec<ChatResult<T>> {
    let mut results = Vec::new();
    
    for (duration, operation) in operations {
        results.push(with_timeout(duration, operation).await);
    }
    
    results
}

/// Execute multiple async operations concurrently with timeout
pub async fn with_timeout_concurrent<T>(
    duration: Duration,
    operations: Vec<impl Future<Output = ChatResult<T>>>,
) -> ChatResult<Vec<T>> {
    match timeout(duration, futures::future::try_join_all(operations)).await {
        Ok(results) => Ok(results?),
        Err(_) => Err(ChatError::Timeout(duration)),
    }
}

/// Race multiple async operations with timeout
pub async fn race_with_timeout<T>(
    duration: Duration,
    operations: Vec<Pin<Box<dyn Future<Output = ChatResult<T>> + Send>>>,
) -> ChatResult<T> {
    match timeout(duration, futures::future::select_ok(operations)).await {
        Ok(Ok((result, _))) => Ok(result),
        Ok(Err(e)) => Err(ChatError::Internal(format!("All operations failed: {}", e))),
        Err(_) => Err(ChatError::Timeout(duration)),
    }
}

/// Adaptive timeout that adjusts based on operation history
pub struct AdaptiveTimeout {
    base_timeout: Duration,
    history: Vec<Duration>,
    max_history: usize,
}

impl AdaptiveTimeout {
    pub fn new(base_timeout: Duration) -> Self {
        Self {
            base_timeout,
            history: Vec::new(),
            max_history: 10,
        }
    }
    
    /// Record an operation duration
    pub fn record(&mut self, duration: Duration) {
        self.history.push(duration);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
    
    /// Get the adaptive timeout based on history
    pub fn get_timeout(&self) -> Duration {
        if self.history.is_empty() {
            return self.base_timeout;
        }
        
        // Calculate average duration
        let total: Duration = self.history.iter().sum();
        let avg = total / self.history.len() as u32;
        
        // Add 50% buffer to average
        avg + avg / 2
    }
    
    /// Execute with adaptive timeout
    pub async fn execute<F, T>(&mut self, operation: F) -> ChatResult<T>
    where
        F: Future<Output = ChatResult<T>>,
    {
        let start = std::time::Instant::now();
        let timeout_duration = self.get_timeout();
        
        let result = with_timeout(timeout_duration, operation).await;
        
        if result.is_ok() {
            self.record(start.elapsed());
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_with_timeout_success() {
        let result = with_timeout(Duration::from_secs(1), async {
            Ok::<_, ChatError>(42)
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
    
    #[tokio::test]
    async fn test_with_timeout_timeout() {
        let result = with_timeout(Duration::from_millis(100), async {
            sleep(Duration::from_secs(1)).await;
            Ok::<_, ChatError>(42)
        }).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ChatError::Timeout(_)));
    }
    
    #[tokio::test]
    async fn test_adaptive_timeout() {
        let mut adaptive = AdaptiveTimeout::new(Duration::from_secs(1));
        
        // Record some fast operations
        adaptive.record(Duration::from_millis(100));
        adaptive.record(Duration::from_millis(150));
        adaptive.record(Duration::from_millis(200));
        
        // Adaptive timeout should be around 225ms (150ms avg + 50% buffer)
        let timeout = adaptive.get_timeout();
        assert!(timeout < Duration::from_secs(1));
        assert!(timeout > Duration::from_millis(200));
    }
}