//! Session monitoring service
//! 
//! Collects performance metrics and updates session analytics

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::interval;
use anyhow::Result;
use sysinfo::System;

use crate::tui::chat::state::session::{
    SessionManager, PerformanceMetrics, MetricType,
};
use crate::tui::chat::agents::{AgentManager, coordination::AgentCoordinator};

/// Monitoring service for session performance
pub struct MonitoringService {
    /// Session manager reference
    session_manager: Arc<RwLock<SessionManager>>,
    
    /// Agent manager reference for queue stats
    agent_manager: Option<Arc<RwLock<AgentManager>>>,
    
    /// System info collector
    system: System,
    
    /// Collection interval
    interval: Duration,
    
    /// Whether monitoring is active
    active: bool,
    
    /// Last collection time
    last_collection: Instant,
    
    /// Request counter for throughput calculation
    request_count: u64,
    
    /// Cache hit counter
    cache_hits: u64,
    
    /// Cache miss counter
    cache_misses: u64,
    
    /// Network latency accumulator for averaging
    latency_samples: Vec<Duration>,
}

impl MonitoringService {
    /// Create a new monitoring service
    pub fn new(session_manager: Arc<RwLock<SessionManager>>) -> Self {
        Self {
            session_manager,
            agent_manager: None,
            system: System::new_all(),
            interval: Duration::from_secs(5),
            active: false,
            last_collection: Instant::now(),
            request_count: 0,
            cache_hits: 0,
            cache_misses: 0,
            latency_samples: Vec::new(),
        }
    }
    
    /// Set the agent manager for queue tracking
    pub fn set_agent_manager(&mut self, agent_manager: Arc<RwLock<AgentManager>>) {
        self.agent_manager = Some(agent_manager);
    }
    
    /// Start monitoring
    pub async fn start(&mut self) -> Result<()> {
        self.active = true;
        self.last_collection = Instant::now();
        
        // Spawn monitoring task
        let session_manager = self.session_manager.clone();
        let agent_manager = self.agent_manager.clone();
        let interval_duration = self.interval;
        let cache_hit_rate = self.get_cache_hit_rate();
        let avg_latency = self.get_average_latency();
        
        // Note: System is not Send, so we create it inside the spawned task
        tokio::spawn(async move {
            let mut ticker = interval(interval_duration);
            let mut service = MonitoringServiceTask::new(
                session_manager,
                agent_manager,
                cache_hit_rate,
                avg_latency,
            );
            
            loop {
                ticker.tick().await;
                if let Err(e) = service.collect_metrics().await {
                    tracing::warn!("Failed to collect metrics: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Stop monitoring
    pub fn stop(&mut self) {
        self.active = false;
    }
    
    /// Record a request
    pub fn record_request(&mut self) {
        self.request_count += 1;
    }
    
    /// Record a cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    /// Record a cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
    
    /// Record network latency sample
    pub fn record_latency(&mut self, latency: Duration) {
        self.latency_samples.push(latency);
        // Keep only last 100 samples to avoid unbounded growth
        if self.latency_samples.len() > 100 {
            self.latency_samples.remove(0);
        }
    }
    
    /// Get average latency
    pub fn get_average_latency(&self) -> Duration {
        if self.latency_samples.is_empty() {
            return Duration::from_millis(0);
        }
        let total: Duration = self.latency_samples.iter().sum();
        total / self.latency_samples.len() as u32
    }
    
    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / total as f32
    }
}

/// Internal monitoring task
struct MonitoringServiceTask {
    session_manager: Arc<RwLock<SessionManager>>,
    agent_manager: Option<Arc<RwLock<AgentManager>>>,
    system: System,
    last_collection: Instant,
    request_count: u64,
    initial_cache_hit_rate: f32,
    initial_avg_latency: Duration,
}

impl MonitoringServiceTask {
    fn new(
        session_manager: Arc<RwLock<SessionManager>>,
        agent_manager: Option<Arc<RwLock<AgentManager>>>,
        initial_cache_hit_rate: f32,
        initial_avg_latency: Duration,
    ) -> Self {
        Self {
            session_manager,
            agent_manager,
            system: System::new_all(),
            last_collection: Instant::now(),
            request_count: 0,
            initial_cache_hit_rate,
            initial_avg_latency,
        }
    }
    
    async fn collect_metrics(&mut self) -> Result<()> {
        // Refresh system info
        self.system.refresh_all();
        
        // Calculate CPU usage
        let cpu_usage = self.system.global_cpu_usage();
        
        // Calculate memory usage
        let used_memory = self.system.used_memory() as f32 / 1024.0 / 1024.0; // Convert to MB
        
        // Calculate throughput
        let elapsed = self.last_collection.elapsed().as_secs_f32();
        let throughput = if elapsed > 0.0 {
            self.request_count as f32 / elapsed
        } else {
            0.0
        };
        
        // Get process-specific metrics if available
        let current_pid = sysinfo::get_current_pid().ok();
        let process_memory = if let Some(pid) = current_pid {
            self.system.processes().get(&pid)
                .map(|p| p.memory() as f32 / 1024.0 / 1024.0)
                .unwrap_or(used_memory)
        } else {
            used_memory
        };
        
        // Get queue depth from agent coordinator if available
        let queue_depth = if let Some(agent_manager) = &self.agent_manager {
            let manager = agent_manager.read().await;
            if let Some(coordinator) = &manager.coordinator {
                let coord = coordinator.read().await;
                let stats = coord.get_queue_stats().await;
                stats.queued_tasks
            } else {
                0
            }
        } else {
            0
        };
        
        // Use initial values passed from main service (which tracks across runs)
        // In a full implementation, we'd have a shared cache stats collector
        let cache_hit_rate = self.initial_cache_hit_rate;
        let network_latency = self.initial_avg_latency;
        
        // Create performance metrics
        let metrics = PerformanceMetrics {
            cpu_usage,
            memory_usage: process_memory,
            throughput,
            active_sessions: 0, // Will be set by session manager
            queue_depth,
            cache_hit_rate,
            network_latency,
        };
        
        // Update session manager
        let mut manager = self.session_manager.write().await;
        manager.update_performance_metrics(metrics);
        
        // Apply adaptive optimization if needed
        if manager.get_optimization_settings().strategy == 
            crate::tui::chat::state::session::OptimizationStrategy::Adaptive {
            manager.set_optimization_strategy(
                crate::tui::chat::state::session::OptimizationStrategy::Adaptive
            );
        }
        
        // Reset counters
        self.request_count = 0;
        self.last_collection = Instant::now();
        
        Ok(())
    }
}

/// Analytics aggregator for multiple sessions
pub struct AnalyticsAggregator;

impl AnalyticsAggregator {
    /// Aggregate analytics across all sessions
    pub async fn aggregate(
        session_manager: &SessionManager,
    ) -> AggregatedAnalytics {
        let all_analytics = session_manager.get_all_analytics();
        
        let mut total_messages = 0;
        let mut total_tokens = 0;
        let mut total_errors = 0;
        let mut total_duration = Duration::ZERO;
        let mut model_usage = std::collections::HashMap::new();
        let mut tool_usage = std::collections::HashMap::new();
        
        for (_, analytics) in &all_analytics {
            total_messages += analytics.message_count;
            total_tokens += analytics.tokens_used;
            total_errors += analytics.error_count;
            total_duration = total_duration + analytics.total_duration;
            
            // Aggregate model usage
            for (model, count) in &analytics.model_usage {
                *model_usage.entry(model.clone()).or_insert(0) += count;
            }
            
            // Aggregate tool usage
            for (tool, count) in &analytics.tool_usage {
                *tool_usage.entry(tool.clone()).or_insert(0) += count;
            }
        }
        
        let overall_success_rate = if total_messages > 0 {
            (total_messages - total_errors) as f32 / total_messages as f32
        } else {
            1.0
        };
        
        AggregatedAnalytics {
            total_sessions: all_analytics.len(),
            total_messages,
            total_tokens,
            total_errors,
            total_duration,
            overall_success_rate,
            model_usage,
            tool_usage,
        }
    }
    
    /// Generate analytics report
    pub fn generate_report(analytics: &AggregatedAnalytics) -> String {
        format!(
            r#"
📊 Session Analytics Report
═══════════════════════════

📈 Overview:
  • Total Sessions: {}
  • Total Messages: {}
  • Total Tokens: {}
  • Total Errors: {}
  • Success Rate: {:.1}%
  • Total Duration: {:?}

🤖 Model Usage:
{}

🔧 Tool Usage:
{}
"#,
            analytics.total_sessions,
            analytics.total_messages,
            analytics.total_tokens,
            analytics.total_errors,
            analytics.overall_success_rate * 100.0,
            analytics.total_duration,
            Self::format_usage_stats(&analytics.model_usage),
            Self::format_usage_stats(&analytics.tool_usage),
        )
    }
    
    fn format_usage_stats(usage: &std::collections::HashMap<String, usize>) -> String {
        if usage.is_empty() {
            return "  No data available".to_string();
        }
        
        let mut sorted: Vec<_> = usage.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        
        sorted.iter()
            .take(10)
            .map(|(name, count)| format!("  • {}: {}", name, count))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Aggregated analytics across sessions
#[derive(Debug, Clone)]
pub struct AggregatedAnalytics {
    pub total_sessions: usize,
    pub total_messages: usize,
    pub total_tokens: usize,
    pub total_errors: usize,
    pub total_duration: Duration,
    pub overall_success_rate: f32,
    pub model_usage: std::collections::HashMap<String, usize>,
    pub tool_usage: std::collections::HashMap<String, usize>,
}