//! Zero-Panic Infrastructure
//!
//! This module provides panic-safe wrappers and executors to ensure
//! Loki never panics in production, achieving true 24/7 autonomous operation.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use tokio::sync::{RwLock, oneshot};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// Macro for wrapping expressions in panic recovery
#[macro_export]
macro_rules! panic_safe {
    ($expr:expr) => {
        $crate::core::panic_safe::catch_panic(|| $expr)
    };
    ($expr:expr, $default:expr) => {
        $crate::core::panic_safe::catch_panic_or(|| $expr, $default)
    };
    ($expr:expr, $msg:expr, $default:expr) => {
        $crate::core::panic_safe::catch_panic_with_msg(|| $expr, $msg, $default)
    };
}

/// Catch panics and return Result
pub fn catch_panic<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> T + panic::UnwindSafe,
{
    panic::catch_unwind(f).map_err(|e| anyhow!("Panic caught: {:?}", panic_message(&e)))
}

/// Catch panics and return default value
pub fn catch_panic_or<F, T>(f: F, default: T) -> T
where
    F: FnOnce() -> T + panic::UnwindSafe,
{
    panic::catch_unwind(f).unwrap_or_else(|e| {
        error!("Panic caught, using default: {:?}", panic_message(&e));
        default
    })
}

/// Catch panics with custom message
pub fn catch_panic_with_msg<F, T>(f: F, msg: &str, default: T) -> T
where
    F: FnOnce() -> T + panic::UnwindSafe,
{
    panic::catch_unwind(f).unwrap_or_else(|e| {
        error!("{}: {:?}", msg, panic_message(&e));
        default
    })
}

/// Extract panic message from Any
fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = panic.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "Unknown panic".to_string()
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, reject requests
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for external calls
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<u32>>,
    last_failure: Arc<RwLock<Option<Instant>>>,
    config: CircuitBreakerConfig,
}

#[derive(Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            config,
        }
    }

    pub async fn call<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        // Check state and potentially transition
        {
            let state = self.state.read().await;

            match *state {
                CircuitState::Open => {
                    // Check if we should transition to half-open
                    if let Some(last) = *self.last_failure.read().await {
                        if last.elapsed() >= self.config.recovery_timeout {
                            drop(state);
                            *self.state.write().await = CircuitState::HalfOpen;
                            *self.failure_count.write().await = 0;
                        } else {
                            return Err(anyhow!("Circuit breaker is open"));
                        }
                    } else {
                        return Err(anyhow!("Circuit breaker is open"));
                    }
                }
                CircuitState::HalfOpen => {
                    // Allow limited calls in half-open state
                    let count = *self.failure_count.read().await;
                    if count >= self.config.half_open_max_calls {
                        return Err(anyhow!("Circuit breaker half-open limit reached"));
                    }
                }
                CircuitState::Closed => {}
            }
        }

        // Execute the function
        match f() {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(e)
            }
        }
    }

    async fn on_success(&self) {
        let mut state = self.state.write().await;
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
            *self.failure_count.write().await = 0;
            info!("Circuit breaker closed after successful recovery");
        }
    }

    async fn on_failure(&self) {
        let mut count = self.failure_count.write().await;
        *count += 1;
        *self.last_failure.write().await = Some(Instant::now());

        if *count >= self.config.failure_threshold {
            *self.state.write().await = CircuitState::Open;
            warn!("Circuit breaker opened after {} failures", count);
        }
    }
}

/// Dead letter queue for failed operations
pub struct DeadLetterQueue<T: Send + Sync + 'static> {
    queue: Arc<RwLock<Vec<DeadLetter<T>>>>,
    max_size: usize,
}

#[derive(Clone)]
pub struct DeadLetter<T> {
    pub item: T,
    pub error: String,
    pub timestamp: Instant,
    pub retry_count: u32,
}

impl<T: Send + Sync + Clone + 'static> DeadLetterQueue<T> {
    pub fn new(max_size: usize) -> Self {
        Self { queue: Arc::new(RwLock::new(Vec::new())), max_size }
    }

    pub async fn push(&self, item: T, error: String) {
        let mut queue = self.queue.write().await;

        // Remove oldest if at capacity
        if queue.len() >= self.max_size {
            queue.remove(0);
        }

        queue.push(DeadLetter { item, error, timestamp: Instant::now(), retry_count: 0 });
    }

    pub async fn replay<F>(&self, mut handler: F) -> Vec<Result<()>>
    where
        F: FnMut(T) -> Result<()>,
    {
        let mut queue = self.queue.write().await;
        let mut results = Vec::new();
        let mut processed = Vec::new();

        for (idx, letter) in queue.iter_mut().enumerate() {
            match handler(letter.item.clone()) {
                Ok(()) => {
                    results.push(Ok(()));
                    processed.push(idx);
                }
                Err(e) => {
                    letter.retry_count += 1;
                    results.push(Err(e));
                }
            }
        }

        // Remove successfully processed items (in reverse order)
        for idx in processed.into_iter().rev() {
            queue.remove(idx);
        }

        results
    }

    pub async fn size(&self) -> usize {
        self.queue.read().await.len()
    }
}

/// Panic-safe executor for async tasks
pub struct PanicSafeExecutor {
    circuit_breakers: DashMap<String, Arc<CircuitBreaker>>,
    dead_letter_queues: DashMap<String, Arc<DeadLetterQueue<Vec<u8>>>>,
}

impl PanicSafeExecutor {
    pub fn new() -> Self {
        Self { circuit_breakers: DashMap::new(), dead_letter_queues: DashMap::new() }
    }

    /// Execute a future with panic recovery
    pub async fn execute<F, T>(&self, name: &str, future: F) -> Result<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let name = name.to_string();
        let task_name = name.clone();

        // Spawn with panic catching
        let _handle: JoinHandle<()> = tokio::spawn(async move {
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                tokio::task::block_in_place(move || {
                    tokio::runtime::Handle::current().block_on(future)
                })
            }));

            let _ = match result {
                Ok(value) => tx.send(Ok(value)),
                Err(panic) => {
                    let msg = panic_message(&panic);
                    error!("Task {} panicked: {}", task_name, msg);
                    let _ = tx.send(Err(anyhow!("Task panicked: {}", msg)));
                    Ok(())
                }
            };
        });

        // Wait for result
        match rx.await {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Task {} failed to complete", name)),
        }
    }

    /// Execute with circuit breaker
    pub async fn execute_with_breaker<F, T>(&self, name: &str, future: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Get or create circuit breaker
        let breaker = self
            .circuit_breakers
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(CircuitBreaker::new(Default::default())))
            .clone();

        // Execute through circuit breaker
        breaker
            .call(|| {
                tokio::task::block_in_place(move || {
                    tokio::runtime::Handle::current().block_on(future)
                })
            })
            .await
    }

    /// Execute with dead letter queue on failure
    pub async fn execute_with_dlq<F, T>(&self, name: &str, data: Vec<u8>, future: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        match self.execute_with_breaker(name, future).await {
            Ok(result) => Ok(result),
            Err(e) => {
                // Store in dead letter queue
                let dlq = self
                    .dead_letter_queues
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(DeadLetterQueue::new(1000)))
                    .clone();

                dlq.push(data, e.to_string()).await;
                Err(e)
            }
        }
    }

    /// Replay dead letter queue
    pub async fn replay_dlq<F>(&self, name: &str, handler: F) -> Vec<Result<()>>
    where
        F: FnMut(Vec<u8>) -> Result<()>,
    {
        if let Some(dlq) = self.dead_letter_queues.get(name) {
            dlq.replay(handler).await
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panic_safe_macro() {
        // Should catch panic and return error
        let result: Result<i32> = panic_safe!(panic!("test panic"));
        assert!(result.is_err());

        // Should return value on success
        let result: Result<i32> = panic_safe!(42);
        assert_eq!(result.unwrap(), 42);

        // Should return default on panic
        let result = panic_safe!(panic!("test"), 0);
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            half_open_max_calls: 1,
        });

        // First call succeeds
        assert!(breaker.call(|| Ok::<_, anyhow::Error>(42)).await.is_ok());

        // Two failures open the breaker
        assert!(breaker.call(|| Err::<i32, _>(anyhow!("fail"))).await.is_err());
        assert!(breaker.call(|| Err::<i32, _>(anyhow!("fail"))).await.is_err());

        // Next call should fail immediately (breaker open)
        assert!(breaker.call(|| Ok::<_, anyhow::Error>(42)).await.is_err());

        // Wait for recovery timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should transition to half-open and allow one call
        assert!(breaker.call(|| Ok::<_, anyhow::Error>(42)).await.is_ok());
    }

    #[tokio::test]
    async fn test_dead_letter_queue() {
        let dlq = DeadLetterQueue::new(3);

        // Add some failures
        dlq.push("item1".to_string(), "error1".to_string()).await;
        dlq.push("item2".to_string(), "error2".to_string()).await;

        assert_eq!(dlq.size().await, 2);

        // Replay with partial success
        let mut processed = Vec::new();
        let results = dlq
            .replay(|item| {
                processed.push(item.clone());
                if item == "item1" { Ok(()) } else { Err(anyhow!("still failing")) }
            })
            .await;

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert_eq!(dlq.size().await, 1); // item2 still in queue
    }

    #[tokio::test]
    async fn test_panic_safe_executor() {
        let executor = PanicSafeExecutor::new();

        // Normal execution
        let result = executor.execute("test", async { 42 }).await;
        assert_eq!(result.unwrap(), 42);

        // Panic recovery
        let result = executor.execute::<_, i32>("test", async { panic!("oh no!") }).await;
        assert!(result.is_err());
    }
}
