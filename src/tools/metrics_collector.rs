use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Serialize, Deserialize};
use tracing::debug;

/// Tool metrics for tracking performance and usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetrics {
    /// Tool identifier
    pub tool_id: String,
    /// Tool display name
    pub tool_name: String,
    /// Total number of invocations
    pub total_calls: u64,
    /// Successful invocations
    pub successful_calls: u64,
    /// Failed invocations
    pub failed_calls: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Minimum response time
    pub min_response_time_ms: u64,
    /// Maximum response time
    pub max_response_time_ms: u64,
    /// Success rate percentage
    pub success_rate: f64,
    /// Current status
    pub status: ToolStatus,
    /// Last invocation time
    pub last_invocation: Option<DateTime<Utc>>,
    /// Error rate percentage
    pub error_rate: f64,
    /// Retry count
    pub retry_count: u64,
    /// Cache hits (if applicable)
    pub cache_hits: u64,
    /// Cache misses (if applicable)
    pub cache_misses: u64,
}

/// Tool status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolStatus {
    Active,
    Idle,
    Processing,
    Error,
    Disabled,
}

/// Active tool session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveToolSession {
    pub session_id: String,
    pub tool_id: String,
    pub tool_name: String,
    pub status: SessionStatus,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub request_type: String,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Processing,
    Idle,
    Active,
    Completing,
}

/// Overall tool system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_active_tools: usize,
    pub total_sessions: usize,
    pub overall_success_rate: f64,
    pub avg_response_time: f64,
    pub total_calls_today: u64,
    pub average_load: f64,
    pub peak_load: f64,
    pub peak_load_time: DateTime<Utc>,
    pub retry_rate: f64,
    pub cache_hit_rate: f64,
    pub parallel_efficiency: f64,
    pub error_rate: f64,
}

/// Tool metrics collector
pub struct ToolMetricsCollector {
    /// Individual tool metrics
    tool_metrics: Arc<RwLock<HashMap<String, ToolMetrics>>>,
    /// Active sessions
    active_sessions: Arc<RwLock<HashMap<String, ActiveToolSession>>>,
    /// System metrics
    system_metrics: Arc<RwLock<SystemMetrics>>,
    /// Historical data for trends
    historical_data: Arc<RwLock<Vec<SystemMetrics>>>,
    /// Last daily reset timestamp
    last_daily_reset: Arc<RwLock<DateTime<Utc>>>,
    /// Daily call counter
    daily_call_counter: Arc<RwLock<u64>>,
}

impl ToolMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            tool_metrics: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            system_metrics: Arc::new(RwLock::new(SystemMetrics {
                total_active_tools: 0,
                total_sessions: 0,
                overall_success_rate: 0.0,
                avg_response_time: 0.0,
                total_calls_today: 0,
                average_load: 0.0,
                peak_load: 0.0,
                peak_load_time: Utc::now(),
                retry_rate: 0.0,
                cache_hit_rate: 0.0,
                parallel_efficiency: 0.0,
                error_rate: 0.0,
            })),
            historical_data: Arc::new(RwLock::new(Vec::new())),
            last_daily_reset: Arc::new(RwLock::new(Utc::now())),
            daily_call_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Initialize metrics for a tool
    pub async fn initialize_tool(&self, tool_id: String, tool_name: String) {
        let mut metrics = self.tool_metrics.write().await;
        metrics.insert(
            tool_id.clone(),
            ToolMetrics {
                tool_id,
                tool_name,
                total_calls: 0,
                successful_calls: 0,
                failed_calls: 0,
                avg_response_time_ms: 0.0,
                min_response_time_ms: u64::MAX,
                max_response_time_ms: 0,
                success_rate: 100.0,
                status: ToolStatus::Idle,
                last_invocation: None,
                error_rate: 0.0,
                retry_count: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
        );
    }

    /// Reset daily metrics (called externally if needed)
    pub async fn reset_daily_metrics(&self) {
        let mut daily_counter = self.daily_call_counter.write().await;
        let mut last_reset = self.last_daily_reset.write().await;
        
        *daily_counter = 0;
        *last_reset = Utc::now();
        
        let mut system_metrics = self.system_metrics.write().await;
        system_metrics.total_calls_today = 0;
        
        debug!("Daily metrics reset at {}", *last_reset);
    }

    /// Record a tool invocation start
    pub async fn record_invocation_start(
        &self,
        tool_id: &str,
        session_id: String,
        request_type: String,
    ) -> anyhow::Result<()> {
        debug!("Recording invocation start for tool: {}", tool_id);

        // Update tool status
        {
            let mut metrics = self.tool_metrics.write().await;
            if let Some(tool_metric) = metrics.get_mut(tool_id) {
                tool_metric.status = ToolStatus::Processing;
                tool_metric.last_invocation = Some(Utc::now());
            }
        }

        // Create active session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(
                session_id.clone(),
                ActiveToolSession {
                    session_id,
                    tool_id: tool_id.to_string(),
                    tool_name: self.get_tool_name(tool_id).await,
                    status: SessionStatus::Processing,
                    start_time: Utc::now(),
                    duration: Duration::zero(),
                    request_type,
                },
            );
        }

        // Update system metrics
        self.update_system_metrics().await;

        Ok(())
    }

    /// Record a tool invocation completion
    pub async fn record_invocation_complete(
        &self,
        tool_id: &str,
        session_id: &str,
        success: bool,
        response_time_ms: u64,
    ) -> anyhow::Result<()> {
        debug!("Recording invocation completion for tool: {}", tool_id);

        // Update tool metrics
        {
            let mut metrics = self.tool_metrics.write().await;
            if let Some(tool_metric) = metrics.get_mut(tool_id) {
                tool_metric.total_calls += 1;
                
                if success {
                    tool_metric.successful_calls += 1;
                } else {
                    tool_metric.failed_calls += 1;
                }

                // Update response times
                tool_metric.min_response_time_ms = tool_metric.min_response_time_ms.min(response_time_ms);
                tool_metric.max_response_time_ms = tool_metric.max_response_time_ms.max(response_time_ms);
                
                // Calculate new average
                let total_time = tool_metric.avg_response_time_ms * (tool_metric.total_calls - 1) as f64 
                    + response_time_ms as f64;
                tool_metric.avg_response_time_ms = total_time / tool_metric.total_calls as f64;

                // Update rates
                tool_metric.success_rate = (tool_metric.successful_calls as f64 / tool_metric.total_calls as f64) * 100.0;
                tool_metric.error_rate = (tool_metric.failed_calls as f64 / tool_metric.total_calls as f64) * 100.0;

                // Update status
                tool_metric.status = ToolStatus::Active;
            }
        }

        // Remove active session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(session_id);
        }

        // Update system metrics
        self.update_system_metrics().await;

        Ok(())
    }

    /// Record a cache hit
    pub async fn record_cache_hit(&self, tool_id: &str) {
        let mut metrics = self.tool_metrics.write().await;
        if let Some(tool_metric) = metrics.get_mut(tool_id) {
            tool_metric.cache_hits += 1;
        }
    }

    /// Record a cache miss
    pub async fn record_cache_miss(&self, tool_id: &str) {
        let mut metrics = self.tool_metrics.write().await;
        if let Some(tool_metric) = metrics.get_mut(tool_id) {
            tool_metric.cache_misses += 1;
        }
    }

    /// Record a retry
    pub async fn record_retry(&self, tool_id: &str) {
        let mut metrics = self.tool_metrics.write().await;
        if let Some(tool_metric) = metrics.get_mut(tool_id) {
            tool_metric.retry_count += 1;
        }
    }

    /// Get all tool metrics
    pub async fn get_all_tool_metrics(&self) -> Vec<ToolMetrics> {
        let metrics = self.tool_metrics.read().await;
        metrics.values().cloned().collect()
    }

    /// Get active sessions
    pub async fn get_active_sessions(&self) -> Vec<ActiveToolSession> {
        let sessions = self.active_sessions.read().await;
        let now = Utc::now();
        
        // Update durations
        sessions.values().map(|session| {
            let mut updated_session = session.clone();
            updated_session.duration = now.signed_duration_since(session.start_time);
            updated_session
        }).collect()
    }

    /// Get system metrics
    pub async fn get_system_metrics(&self) -> SystemMetrics {
        self.system_metrics.read().await.clone()
    }

    /// Update system metrics
    async fn update_system_metrics(&self) {
        let metrics = self.tool_metrics.read().await;
        let sessions = self.active_sessions.read().await;

        let total_calls: u64 = metrics.values().map(|m| m.total_calls).sum();
        let successful_calls: u64 = metrics.values().map(|m| m.successful_calls).sum();
        let total_response_time: f64 = metrics.values()
            .map(|m| m.avg_response_time_ms * m.total_calls as f64)
            .sum();

        let active_tools = metrics.values()
            .filter(|m| m.status == ToolStatus::Active || m.status == ToolStatus::Processing)
            .count();

        let overall_success_rate = if total_calls > 0 {
            (successful_calls as f64 / total_calls as f64) * 100.0
        } else {
            100.0
        };

        let avg_response_time = if total_calls > 0 {
            total_response_time / total_calls as f64
        } else {
            0.0
        };

        // Calculate cache hit rate
        let total_cache_attempts: u64 = metrics.values()
            .map(|m| m.cache_hits + m.cache_misses)
            .sum();
        let total_cache_hits: u64 = metrics.values().map(|m| m.cache_hits).sum();
        let cache_hit_rate = if total_cache_attempts > 0 {
            (total_cache_hits as f64 / total_cache_attempts as f64) * 100.0
        } else {
            0.0
        };

        // Calculate retry rate
        let total_retries: u64 = metrics.values().map(|m| m.retry_count).sum();
        let retry_rate = if total_calls > 0 {
            (total_retries as f64 / total_calls as f64) * 100.0
        } else {
            0.0
        };

        // Calculate error rate
        let failed_calls: u64 = metrics.values().map(|m| m.failed_calls).sum();
        let error_rate = if total_calls > 0 {
            (failed_calls as f64 / total_calls as f64) * 100.0
        } else {
            0.0
        };

        // Update system metrics
        let mut system_metrics = self.system_metrics.write().await;
        system_metrics.total_active_tools = active_tools;
        system_metrics.total_sessions = sessions.len();
        system_metrics.overall_success_rate = overall_success_rate;
        system_metrics.avg_response_time = avg_response_time;
        
        // Check if we need to reset daily counter
        let now = Utc::now();
        let mut last_reset = self.last_daily_reset.write().await;
        let mut daily_counter = self.daily_call_counter.write().await;
        
        // Reset if it's a new day (UTC)
        if now.date_naive() != last_reset.date_naive() {
            debug!("Resetting daily call counter for new day: {} -> {}", 
                   last_reset.date_naive(), now.date_naive());
            *daily_counter = 0;
            *last_reset = now;
        }
        
        // Add today's calls to the daily counter
        let new_calls_today = total_calls.saturating_sub(*daily_counter);
        *daily_counter = (*daily_counter).saturating_add(new_calls_today);
        system_metrics.total_calls_today = *daily_counter;
        
        system_metrics.retry_rate = retry_rate;
        system_metrics.cache_hit_rate = cache_hit_rate;
        system_metrics.error_rate = error_rate;

        // Update load metrics (simplified calculation)
        let current_load = (sessions.len() as f64 / 10.0) * 100.0; // Assume max 10 concurrent
        system_metrics.average_load = (system_metrics.average_load * 0.95) + (current_load * 0.05);
        
        if current_load > system_metrics.peak_load {
            system_metrics.peak_load = current_load;
            system_metrics.peak_load_time = Utc::now();
        }

        // Estimate parallel efficiency
        system_metrics.parallel_efficiency = if sessions.len() > 1 {
            90.0 + (10.0 / sessions.len() as f64) // Simplified calculation
        } else {
            100.0
        };
    }

    /// Get tool name from ID
    async fn get_tool_name(&self, tool_id: &str) -> String {
        let metrics = self.tool_metrics.read().await;
        metrics.get(tool_id)
            .map(|m| m.tool_name.clone())
            .unwrap_or_else(|| tool_id.to_string())
    }

    /// Get metrics for display in TUI
    pub async fn get_tui_display_data(&self) -> TuiToolMetrics {
        let tool_metrics = self.get_all_tool_metrics().await;
        let active_sessions = self.get_active_sessions().await;
        let system_metrics = self.get_system_metrics().await;

        TuiToolMetrics {
            tool_metrics,
            active_sessions,
            system_metrics,
        }
    }
}

/// Data structure for TUI display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiToolMetrics {
    pub tool_metrics: Vec<ToolMetrics>,
    pub active_sessions: Vec<ActiveToolSession>,
    pub system_metrics: SystemMetrics,
}

impl Default for ToolMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}