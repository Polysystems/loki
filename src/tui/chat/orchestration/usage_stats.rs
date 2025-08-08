//! Usage statistics tracking for orchestration
//! 
//! Tracks model usage, performance metrics, and cost analysis

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration, Datelike, Timelike};
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Usage statistics tracker
#[derive(Clone)]
pub struct UsageStatsTracker {
    /// Statistics storage
    stats: Arc<RwLock<UsageStats>>,
    
    /// Historical data points
    history: Arc<RwLock<Vec<UsageSnapshot>>>,
    
    /// Maximum history size
    max_history_size: usize,
}

/// Comprehensive usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total requests processed
    pub total_requests: u64,
    
    /// Successful requests
    pub successful_requests: u64,
    
    /// Failed requests
    pub failed_requests: u64,
    
    /// Total tokens used
    pub total_tokens: u64,
    
    /// Total cost in cents
    pub total_cost_cents: u64,
    
    /// Statistics per model
    pub model_stats: HashMap<String, ModelUsageStats>,
    
    /// Statistics per provider
    pub provider_stats: HashMap<String, ProviderUsageStats>,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Time-based statistics
    pub time_stats: TimeBasedStats,
    
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Per-model usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsageStats {
    /// Model identifier
    pub model_id: String,
    
    /// Number of requests
    pub request_count: u64,
    
    /// Success count
    pub success_count: u64,
    
    /// Total tokens
    pub tokens_used: u64,
    
    /// Total cost
    pub cost_cents: u64,
    
    /// Average response time
    pub avg_response_time_ms: f64,
    
    /// Error rate
    pub error_rate: f64,
    
    /// Last used timestamp
    pub last_used: DateTime<Utc>,
}

/// Per-provider usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderUsageStats {
    /// Provider name
    pub provider: String,
    
    /// Active models count
    pub active_models: usize,
    
    /// Total requests
    pub request_count: u64,
    
    /// Total tokens
    pub tokens_used: u64,
    
    /// Total cost
    pub cost_cents: u64,
    
    /// Provider health status
    pub health_status: HealthStatus,
    
    /// Last error if any
    pub last_error: Option<String>,
}

/// Provider health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unavailable,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average response time across all requests
    pub avg_response_time_ms: f64,
    
    /// P50 response time
    pub p50_response_time_ms: f64,
    
    /// P95 response time
    pub p95_response_time_ms: f64,
    
    /// P99 response time
    pub p99_response_time_ms: f64,
    
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    
    /// Success rate
    pub success_rate: f64,
}

/// Time-based statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedStats {
    /// Requests in last hour
    pub requests_last_hour: u64,
    
    /// Requests today
    pub requests_today: u64,
    
    /// Peak hour (0-23)
    pub peak_hour: u32,
    
    /// Peak requests per hour
    pub peak_requests_per_hour: u64,
}

/// Usage snapshot for historical tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSnapshot {
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    
    /// Request count at snapshot
    pub request_count: u64,
    
    /// Token count at snapshot
    pub token_count: u64,
    
    /// Cost at snapshot
    pub cost_cents: u64,
    
    /// Active models count
    pub active_models: usize,
    
    /// Average response time
    pub avg_response_time_ms: f64,
}

/// Request record for tracking
#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// Model used
    pub model_id: String,
    
    /// Provider
    pub provider: String,
    
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Response time in milliseconds
    pub response_time_ms: u64,
    
    /// Tokens used
    pub tokens_used: u64,
    
    /// Cost in cents
    pub cost_cents: u64,
    
    /// Success status
    pub success: bool,
    
    /// Error message if failed
    pub error: Option<String>,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_tokens: 0,
            total_cost_cents: 0,
            model_stats: HashMap::new(),
            provider_stats: HashMap::new(),
            performance: PerformanceMetrics {
                avg_response_time_ms: 0.0,
                p50_response_time_ms: 0.0,
                p95_response_time_ms: 0.0,
                p99_response_time_ms: 0.0,
                avg_tokens_per_second: 0.0,
                success_rate: 100.0,
            },
            time_stats: TimeBasedStats {
                requests_last_hour: 0,
                requests_today: 0,
                peak_hour: 0,
                peak_requests_per_hour: 0,
            },
            last_updated: Utc::now(),
        }
    }
}

impl UsageStatsTracker {
    /// Create a new usage stats tracker
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(UsageStats::default())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history_size: 1000,
        }
    }
    
    /// Record a request
    pub async fn record_request(&self, record: RequestRecord) -> Result<()> {
        let mut stats = self.stats.write().await;
        
        // Update totals
        stats.total_requests += 1;
        stats.total_tokens += record.tokens_used;
        stats.total_cost_cents += record.cost_cents;
        
        if record.success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }
        
        // Update model stats
        let model_stats = stats.model_stats
            .entry(record.model_id.clone())
            .or_insert_with(|| ModelUsageStats {
                model_id: record.model_id.clone(),
                request_count: 0,
                success_count: 0,
                tokens_used: 0,
                cost_cents: 0,
                avg_response_time_ms: 0.0,
                error_rate: 0.0,
                last_used: record.timestamp,
            });
        
        model_stats.request_count += 1;
        if record.success {
            model_stats.success_count += 1;
        }
        model_stats.tokens_used += record.tokens_used;
        model_stats.cost_cents += record.cost_cents;
        model_stats.last_used = record.timestamp;
        
        // Update average response time
        let new_avg = ((model_stats.avg_response_time_ms * (model_stats.request_count - 1) as f64) 
            + record.response_time_ms as f64) / model_stats.request_count as f64;
        model_stats.avg_response_time_ms = new_avg;
        
        // Update error rate
        model_stats.error_rate = (model_stats.request_count - model_stats.success_count) as f64 
            / model_stats.request_count as f64 * 100.0;
        
        // Calculate provider error rate first (before getting mutable reference)
        let provider_error_rate = if !record.success {
            let provider_name = record.provider.clone();
            let active_models = stats.provider_stats
                .get(&provider_name)
                .map(|p| p.active_models.max(1) as f64)
                .unwrap_or(1.0);
            
            stats.model_stats.values()
                .filter(|m| self.get_provider_for_model(&m.model_id) == provider_name)
                .map(|m| m.error_rate)
                .sum::<f64>() / active_models
        } else {
            0.0
        };
        
        // Now update provider stats
        let provider_stats = stats.provider_stats
            .entry(record.provider.clone())
            .or_insert_with(|| ProviderUsageStats {
                provider: record.provider.clone(),
                active_models: 0,
                request_count: 0,
                tokens_used: 0,
                cost_cents: 0,
                health_status: HealthStatus::Healthy,
                last_error: None,
            });
        
        provider_stats.request_count += 1;
        provider_stats.tokens_used += record.tokens_used;
        provider_stats.cost_cents += record.cost_cents;
        
        if !record.success {
            provider_stats.last_error = record.error.clone();
            // Update health status based on error rate
            provider_stats.health_status = if provider_error_rate > 50.0 {
                HealthStatus::Unavailable
            } else if provider_error_rate > 10.0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };
        }
        
        // Update performance metrics
        self.update_performance_metrics(&mut stats, record.response_time_ms).await;
        
        // Update time-based stats
        self.update_time_stats(&mut stats, &record.timestamp).await;
        
        stats.last_updated = Utc::now();
        
        // Take snapshot if needed
        self.maybe_take_snapshot(&stats).await;
        
        Ok(())
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(
        &self, 
        stats: &mut UsageStats, 
        response_time_ms: u64
    ) {
        // Update average response time
        let total_time = stats.performance.avg_response_time_ms * (stats.total_requests - 1) as f64;
        stats.performance.avg_response_time_ms = (total_time + response_time_ms as f64) 
            / stats.total_requests as f64;
        
        // Update success rate
        stats.performance.success_rate = 
            (stats.successful_requests as f64 / stats.total_requests as f64) * 100.0;
        
        // For percentiles, we'd need to store response times
        // For now, use approximations
        stats.performance.p50_response_time_ms = stats.performance.avg_response_time_ms * 0.8;
        stats.performance.p95_response_time_ms = stats.performance.avg_response_time_ms * 1.5;
        stats.performance.p99_response_time_ms = stats.performance.avg_response_time_ms * 2.0;
        
        // Calculate tokens per second
        if stats.performance.avg_response_time_ms > 0.0 {
            let avg_tokens_per_request = stats.total_tokens as f64 / stats.total_requests as f64;
            stats.performance.avg_tokens_per_second = 
                avg_tokens_per_request / (stats.performance.avg_response_time_ms / 1000.0);
        }
    }
    
    /// Update time-based statistics
    async fn update_time_stats(&self, stats: &mut UsageStats, timestamp: &DateTime<Utc>) {
        let now = Utc::now();
        let hour_ago = now - Duration::hours(1);
        let today_start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
        
        // This is simplified - in production you'd track requests by time
        if *timestamp > hour_ago {
            stats.time_stats.requests_last_hour += 1;
        }
        
        if *timestamp > today_start {
            stats.time_stats.requests_today += 1;
        }
        
        // Update peak hour (simplified)
        let current_hour = timestamp.hour();
        stats.time_stats.peak_hour = current_hour;
    }
    
    /// Take a snapshot of current stats
    async fn maybe_take_snapshot(&self, stats: &UsageStats) {
        let mut history = self.history.write().await;
        
        // Take snapshot every 100 requests or every hour
        let should_snapshot = stats.total_requests % 100 == 0 || 
            history.last().map(|s| {
                Utc::now().signed_duration_since(s.timestamp).num_hours() >= 1
            }).unwrap_or(true);
        
        if should_snapshot {
            let snapshot = UsageSnapshot {
                timestamp: Utc::now(),
                request_count: stats.total_requests,
                token_count: stats.total_tokens,
                cost_cents: stats.total_cost_cents,
                active_models: stats.model_stats.len(),
                avg_response_time_ms: stats.performance.avg_response_time_ms,
            };
            
            history.push(snapshot);
            
            // Trim history if too large
            if history.len() > self.max_history_size {
                history.remove(0);
            }
        }
    }
    
    /// Get current usage statistics
    pub async fn get_stats(&self) -> UsageStats {
        self.stats.read().await.clone()
    }
    
    /// Get usage history
    pub async fn get_history(&self) -> Vec<UsageSnapshot> {
        self.history.read().await.clone()
    }
    
    /// Get stats for a specific model
    pub async fn get_model_stats(&self, model_id: &str) -> Option<ModelUsageStats> {
        let stats = self.stats.read().await;
        stats.model_stats.get(model_id).cloned()
    }
    
    /// Get stats for a specific provider
    pub async fn get_provider_stats(&self, provider: &str) -> Option<ProviderUsageStats> {
        let stats = self.stats.read().await;
        stats.provider_stats.get(provider).cloned()
    }
    
    /// Get usage summary
    pub async fn get_summary(&self) -> UsageSummary {
        let stats = self.stats.read().await;
        
        UsageSummary {
            total_requests: stats.total_requests,
            success_rate: stats.performance.success_rate,
            total_cost_dollars: stats.total_cost_cents as f64 / 100.0,
            avg_response_time_ms: stats.performance.avg_response_time_ms,
            active_models: stats.model_stats.len(),
            active_providers: stats.provider_stats.len(),
            requests_last_hour: stats.time_stats.requests_last_hour,
        }
    }
    
    /// Reset statistics
    pub async fn reset(&self) {
        *self.stats.write().await = UsageStats::default();
        self.history.write().await.clear();
    }
    
    /// Helper to determine provider from model ID
    fn get_provider_for_model(&self, model_id: &str) -> String {
        if model_id.contains("gpt") {
            "openai".to_string()
        } else if model_id.contains("claude") {
            "anthropic".to_string()
        } else if model_id.contains("gemini") {
            "google".to_string()
        } else if model_id.contains("llama") || model_id.contains("mistral") {
            "ollama".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

/// Usage summary for quick overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub total_requests: u64,
    pub success_rate: f64,
    pub total_cost_dollars: f64,
    pub avg_response_time_ms: f64,
    pub active_models: usize,
    pub active_providers: usize,
    pub requests_last_hour: u64,
}

impl Default for UsageStatsTracker {
    fn default() -> Self {
        Self::new()
    }
}