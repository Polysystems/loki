use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub service_name: String,
    pub requests_per_second: f32,
    pub burst_size: usize,
    pub retry_after: Duration,
    pub backoff_multiplier: f32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            service_name: "default".to_string(),
            requests_per_second: 1.0,
            burst_size: 10,
            retry_after: Duration::from_secs(1),
            backoff_multiplier: 2.0,
        }
    }
}

/// Global rate limiter for all services
pub struct GlobalRateLimiter {
    /// Service-specific limiters
    limiters: Arc<RwLock<HashMap<String, ServiceRateLimiter>>>,

    /// Global semaphore for total concurrent requests
    global_semaphore: Arc<Semaphore>,

    /// Configuration
    configs: Arc<RwLock<HashMap<String, RateLimitConfig>>>,
}

impl GlobalRateLimiter {
    /// Create a new global rate limiter
    pub fn new(max_concurrent_requests: usize) -> Self {
        info!(
            "Initializing global rate limiter with {} max concurrent requests",
            max_concurrent_requests
        );

        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
            global_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a service with rate limit configuration
    pub fn register_service(&self, config: RateLimitConfig) {
        let service_name = config.service_name.clone();

        // Create limiter
        let limiter = ServiceRateLimiter::new(&config);

        // Store
        self.limiters.write().insert(service_name.clone(), limiter);
        self.configs.write().insert(service_name.clone(), config);

        info!("Registered rate limiter for service: {}", service_name);
    }

    /// Acquire permission to make a request
    pub async fn acquire(&self, service_name: &str) -> Result<RateLimitGuard> {
        // Get global permit first
        let global_permit = self
            .global_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to acquire global permit: {}", e))?;

        // Get or create service limiter
        let mut limiters = self.limiters.write();
        let limiter = limiters.entry(service_name.to_string()).or_insert_with(|| {
            let config = self.configs.read().get(service_name).cloned().unwrap_or_else(|| {
                RateLimitConfig { service_name: service_name.to_string(), ..Default::default() }
            });
            ServiceRateLimiter::new(&config)
        });

        // Wait for service rate limit
        let wait_time = limiter.acquire();
        if wait_time > Duration::ZERO {
            debug!("Rate limit for {}: waiting {:?}", service_name, wait_time);
            tokio::time::sleep(wait_time).await;
        }

        Ok(RateLimitGuard {
            _permit: global_permit,
            service_name: service_name.to_string(),
            start_time: Instant::now(),
        })
    }

    /// Report a failed request (for backoff)
    pub fn report_failure(&self, service_name: &str) {
        if let Some(limiter) = self.limiters.write().get_mut(service_name) {
            limiter.report_failure();
        }
    }

    /// Report a successful request (reset backoff)
    pub fn report_success(&self, service_name: &str) {
        if let Some(limiter) = self.limiters.write().get_mut(service_name) {
            limiter.report_success();
        }
    }

    /// Get current statistics
    pub fn get_stats(&self, service_name: &str) -> Option<RateLimitStats> {
        self.limiters.read().get(service_name).map(|l| l.get_stats())
    }

    /// Get all service names
    pub fn list_services(&self) -> Vec<String> {
        self.limiters.read().keys().cloned().collect()
    }
}

/// Guard that releases rate limit when dropped
pub struct RateLimitGuard {
    _permit: tokio::sync::OwnedSemaphorePermit,
    service_name: String,
    start_time: Instant,
}

impl Drop for RateLimitGuard {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        debug!("Request to {} completed in {:?}", self.service_name, duration);
    }
}

/// Per-service rate limiter
struct ServiceRateLimiter {
    /// Token bucket
    tokens: f32,
    max_tokens: f32,
    refill_rate: f32,
    last_refill: Instant,

    /// Backoff state
    consecutive_failures: usize,
    backoff_until: Option<Instant>,
    backoff_multiplier: f32,
    base_retry_after: Duration,

    /// Statistics
    total_requests: usize,
    total_failures: usize,
    total_wait_time: Duration,
}

impl ServiceRateLimiter {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            tokens: config.burst_size as f32,
            max_tokens: config.burst_size as f32,
            refill_rate: config.requests_per_second,
            last_refill: Instant::now(),
            consecutive_failures: 0,
            backoff_until: None,
            backoff_multiplier: config.backoff_multiplier,
            base_retry_after: config.retry_after,
            total_requests: 0,
            total_failures: 0,
            total_wait_time: Duration::ZERO,
        }
    }

    /// Acquire permission (returns wait time)
    fn acquire(&mut self) -> Duration {
        let now = Instant::now();

        // Check backoff
        if let Some(backoff_until) = self.backoff_until {
            if now < backoff_until {
                let wait = backoff_until - now;
                self.total_wait_time += wait;
                return wait;
            } else {
                self.backoff_until = None;
            }
        }

        // Refill tokens
        self.refill(now);

        // Check if we have tokens
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            self.total_requests += 1;
            Duration::ZERO
        } else {
            // Calculate wait time
            let tokens_needed = 1.0 - self.tokens;
            let wait_seconds = tokens_needed / self.refill_rate;
            let wait = Duration::from_secs_f32(wait_seconds);

            self.total_wait_time += wait;
            self.total_requests += 1;

            // Consume the token for future
            self.tokens = 0.0;

            wait
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_refill).as_secs_f32();
        let new_tokens = elapsed * self.refill_rate;

        self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Report a failure
    fn report_failure(&mut self) {
        self.consecutive_failures += 1;
        self.total_failures += 1;

        // Exponential backoff
        let backoff_duration = self
            .base_retry_after
            .mul_f32(self.backoff_multiplier.powi(self.consecutive_failures as i32 - 1));

        self.backoff_until = Some(Instant::now() + backoff_duration);

        warn!(
            "Service failure #{}, backing off for {:?}",
            self.consecutive_failures, backoff_duration
        );
    }

    /// Report a success
    fn report_success(&mut self) {
        if self.consecutive_failures > 0 {
            info!("Service recovered after {} failures", self.consecutive_failures);
        }
        self.consecutive_failures = 0;
        self.backoff_until = None;
    }

    /// Get statistics
    fn get_stats(&self) -> RateLimitStats {
        RateLimitStats {
            total_requests: self.total_requests,
            total_failures: self.total_failures,
            consecutive_failures: self.consecutive_failures,
            average_wait_time: if self.total_requests > 0 {
                self.total_wait_time / self.total_requests as u32
            } else {
                Duration::ZERO
            },
            tokens_available: self.tokens,
            in_backoff: self.backoff_until.is_some(),
        }
    }
}

/// Rate limit statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub total_requests: usize,
    pub total_failures: usize,
    pub consecutive_failures: usize,
    pub average_wait_time: Duration,
    pub tokens_available: f32,
    pub in_backoff: bool,
}

/// Pre-configured rate limits for common services
impl GlobalRateLimiter {
    /// Configure common web services
    pub fn configure_common_services(&self) {
        // Search engines
        self.register_service(RateLimitConfig {
            service_name: "duckduckgo".to_string(),
            requests_per_second: 1.0,
            burst_size: 5,
            retry_after: Duration::from_secs(2),
            backoff_multiplier: 2.0,
        });

        self.register_service(RateLimitConfig {
            service_name: "brave_search".to_string(),
            requests_per_second: 2.0,
            burst_size: 10,
            retry_after: Duration::from_secs(1),
            backoff_multiplier: 1.5,
        });

        // Social media
        self.register_service(RateLimitConfig {
            service_name: "twitter".to_string(),
            requests_per_second: 5.0,
            burst_size: 20,
            retry_after: Duration::from_secs(15),
            backoff_multiplier: 2.0,
        });

        // AI services
        self.register_service(RateLimitConfig {
            service_name: "openai".to_string(),
            requests_per_second: 1.0,
            burst_size: 3,
            retry_after: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        });

        self.register_service(RateLimitConfig {
            service_name: "anthropic".to_string(),
            requests_per_second: 0.5,
            burst_size: 2,
            retry_after: Duration::from_secs(20),
            backoff_multiplier: 2.0,
        });

        // GitHub
        self.register_service(RateLimitConfig {
            service_name: "github".to_string(),
            requests_per_second: 1.0,
            burst_size: 5,
            retry_after: Duration::from_secs(60),
            backoff_multiplier: 1.5,
        });

        info!("Configured rate limits for common services");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = GlobalRateLimiter::new(10);

        limiter.register_service(RateLimitConfig {
            service_name: "test".to_string(),
            requests_per_second: 2.0,
            burst_size: 2,
            retry_after: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        });

        // Should allow burst
        let start = Instant::now();
        let _g1 = limiter.acquire("test").await.unwrap();
        let _g2 = limiter.acquire("test").await.unwrap();

        // Third should wait
        let _g3 = limiter.acquire("test").await.unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(400)); // ~0.5 seconds for refill
    }

    #[test]
    fn test_backoff() {
        let config = RateLimitConfig {
            service_name: "test".to_string(),
            requests_per_second: 1.0,
            burst_size: 1,
            retry_after: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        };

        let mut limiter = ServiceRateLimiter::new(&config);

        // First request succeeds
        assert_eq!(limiter.acquire(), Duration::ZERO);

        // Report failures
        limiter.report_failure();
        let wait1 = limiter.acquire();
        assert!(wait1 >= Duration::from_millis(100));

        limiter.report_failure();
        let wait2 = limiter.acquire();
        assert!(wait2 >= Duration::from_millis(200)); // Exponential backoff
    }
}
