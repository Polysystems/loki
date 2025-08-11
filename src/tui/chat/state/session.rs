//! Chat session management
//! 
//! Manages multiple chat sessions and active chat switching

use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::chat_state::ChatState;

/// Session analytics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAnalytics {
    /// Session creation time
    pub created_at: DateTime<Utc>,
    
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    
    /// Total duration active
    pub total_duration: Duration,
    
    /// Message count
    pub message_count: usize,
    
    /// Token usage
    pub tokens_used: usize,
    
    /// Model usage statistics
    pub model_usage: HashMap<String, usize>,
    
    /// Tool usage statistics
    pub tool_usage: HashMap<String, usize>,
    
    /// Error count
    pub error_count: usize,
    
    /// Success rate
    pub success_rate: f32,
    
    /// Average response time
    pub avg_response_time: Duration,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f32,
    
    /// Memory usage in MB
    pub memory_usage: f32,
    
    /// Request throughput (requests/sec)
    pub throughput: f32,
    
    /// Active sessions count
    pub active_sessions: usize,
    
    /// Queue depth
    pub queue_depth: usize,
    
    /// Cache hit rate
    pub cache_hit_rate: f32,
    
    /// Network latency
    pub network_latency: Duration,
}

/// Optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Minimize latency
    LowLatency,
    
    /// Maximize throughput
    HighThroughput,
    
    /// Balance cost and performance
    CostOptimized,
    
    /// Prioritize quality
    QualityFirst,
    
    /// Adaptive based on load
    Adaptive,
}

/// Session optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// Current strategy
    pub strategy: OptimizationStrategy,
    
    /// Enable caching
    pub enable_cache: bool,
    
    /// Max cache size in MB
    pub max_cache_size: usize,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Timeout for operations
    pub operation_timeout: Duration,
    
    /// Max concurrent requests
    pub max_concurrent: usize,
    
    /// Enable prefetching
    pub enable_prefetch: bool,
}

impl Default for SessionAnalytics {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            last_activity: Utc::now(),
            total_duration: Duration::ZERO,
            message_count: 0,
            tokens_used: 0,
            model_usage: HashMap::new(),
            tool_usage: HashMap::new(),
            error_count: 0,
            success_rate: 1.0,
            avg_response_time: Duration::ZERO,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            throughput: 0.0,
            active_sessions: 0,
            queue_depth: 0,
            cache_hit_rate: 0.0,
            network_latency: Duration::ZERO,
        }
    }
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Adaptive,
            enable_cache: true,
            max_cache_size: 100, // MB
            enable_compression: true,
            batch_size: 10,
            operation_timeout: Duration::from_secs(30),
            max_concurrent: 5,
            enable_prefetch: true,
        }
    }
}

/// Manages multiple chat sessions
#[derive(Debug)]
pub struct SessionManager {
    /// All chat sessions
    pub chats: HashMap<usize, ChatState>,
    
    /// Currently active chat ID
    pub active_chat: usize,
    
    /// Next chat ID to assign
    next_chat_id: usize,
    
    /// Session analytics by chat ID
    analytics: HashMap<usize, SessionAnalytics>,
    
    /// Global performance metrics
    performance_metrics: PerformanceMetrics,
    
    /// Optimization settings
    optimization_settings: OptimizationSettings,
    
    /// Session start times for duration tracking
    session_starts: HashMap<usize, Instant>,
    
    /// Response time samples for averaging
    response_time_samples: Vec<Duration>,
}

impl Default for SessionManager {
    fn default() -> Self {
        let mut chats = HashMap::new();
        chats.insert(0, ChatState::new(0, "Main Chat".to_string()));
        
        let mut analytics = HashMap::new();
        analytics.insert(0, SessionAnalytics::default());
        
        let mut session_starts = HashMap::new();
        session_starts.insert(0, Instant::now());
        
        Self {
            chats,
            active_chat: 0,
            next_chat_id: 1,
            analytics,
            performance_metrics: PerformanceMetrics::default(),
            optimization_settings: OptimizationSettings::default(),
            session_starts,
            response_time_samples: Vec::new(),
        }
    }
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self::default()
    }
    
    
    /// Create a new chat session
    pub fn create_chat(&mut self, name: String) -> usize {
        let chat_id = self.next_chat_id;
        self.next_chat_id += 1;
        
        self.chats.insert(chat_id, ChatState::new(chat_id, name));
        self.analytics.insert(chat_id, SessionAnalytics::default());
        self.session_starts.insert(chat_id, Instant::now());
        
        // Update performance metrics
        self.performance_metrics.active_sessions = self.chats.len();
        
        chat_id
    }
    
    /// Switch to a different chat
    pub fn switch_to_chat(&mut self, chat_id: usize) -> Result<()> {
        if self.chats.contains_key(&chat_id) {
            self.active_chat = chat_id;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Chat {} not found", chat_id))
        }
    }
    
    /// Get the active chat
    pub fn active_chat(&self) -> Option<&ChatState> {
        self.chats.get(&self.active_chat)
    }
    
    /// Get the active chat mutably
    pub fn active_chat_mut(&mut self) -> Option<&mut ChatState> {
        self.chats.get_mut(&self.active_chat)
    }
    
    /// Delete a chat session
    pub fn delete_chat(&mut self, chat_id: usize) -> Result<()> {
        if chat_id == 0 {
            return Err(anyhow::anyhow!("Cannot delete main chat"));
        }
        
        self.chats.remove(&chat_id);
        
        // If we deleted the active chat, switch to main
        if self.active_chat == chat_id {
            self.active_chat = 0;
        }
        
        Ok(())
    }
    
    /// Get all chat sessions
    pub fn all_chats(&self) -> Vec<(usize, &ChatState)> {
        let mut chats: Vec<_> = self.chats.iter()
            .map(|(id, chat)| (*id, chat))
            .collect();
        chats.sort_by_key(|(id, _)| *id);
        chats
    }
    
    /// Count total messages across all chats
    pub fn total_message_count(&self) -> usize {
        self.chats.values()
            .map(|chat| chat.total_message_count())
            .sum()
    }
    
    // ============ Analytics Methods ============
    
    /// Track a message in analytics
    pub fn track_message(&mut self, chat_id: usize, tokens: usize) {
        if let Some(analytics) = self.analytics.get_mut(&chat_id) {
            analytics.message_count += 1;
            analytics.tokens_used += tokens;
            analytics.last_activity = Utc::now();
            
            // Update duration if session is tracked
            if let Some(start_time) = self.session_starts.get(&chat_id) {
                analytics.total_duration = start_time.elapsed();
            }
        }
    }
    
    /// Track model usage
    pub fn track_model_usage(&mut self, chat_id: usize, model: String) {
        if let Some(analytics) = self.analytics.get_mut(&chat_id) {
            *analytics.model_usage.entry(model).or_insert(0) += 1;
        }
    }
    
    /// Track tool usage
    pub fn track_tool_usage(&mut self, chat_id: usize, tool: String) {
        if let Some(analytics) = self.analytics.get_mut(&chat_id) {
            *analytics.tool_usage.entry(tool).or_insert(0) += 1;
        }
    }
    
    /// Track an error
    pub fn track_error(&mut self, chat_id: usize) {
        if let Some(analytics) = self.analytics.get_mut(&chat_id) {
            analytics.error_count += 1;
            // Recalculate success rate
            let total = analytics.message_count as f32;
            if total > 0.0 {
                analytics.success_rate = (total - analytics.error_count as f32) / total;
            }
        }
    }
    
    /// Track response time
    pub fn track_response_time(&mut self, duration: Duration) {
        self.response_time_samples.push(duration);
        
        // Keep only last 100 samples
        if self.response_time_samples.len() > 100 {
            self.response_time_samples.remove(0);
        }
        
        // Update average for active chat
        if let Some(analytics) = self.analytics.get_mut(&self.active_chat) {
            let total: Duration = self.response_time_samples.iter().sum();
            analytics.avg_response_time = total / self.response_time_samples.len() as u32;
        }
    }
    
    /// Get analytics for a session
    pub fn get_analytics(&self, chat_id: usize) -> Option<&SessionAnalytics> {
        self.analytics.get(&chat_id)
    }
    
    /// Get analytics for all sessions
    pub fn get_all_analytics(&self) -> Vec<(usize, &SessionAnalytics)> {
        let mut analytics: Vec<_> = self.analytics.iter()
            .map(|(id, data)| (*id, data))
            .collect();
        analytics.sort_by_key(|(id, _)| *id);
        analytics
    }
    
    // ============ Monitoring Methods ============
    
    /// Update performance metrics
    pub fn update_performance_metrics(&mut self, metrics: PerformanceMetrics) {
        self.performance_metrics = metrics;
        self.performance_metrics.active_sessions = self.chats.len();
    }
    
    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Update specific metric
    pub fn update_metric(&mut self, metric: MetricType, value: f32) {
        match metric {
            MetricType::CpuUsage => self.performance_metrics.cpu_usage = value,
            MetricType::MemoryUsage => self.performance_metrics.memory_usage = value,
            MetricType::Throughput => self.performance_metrics.throughput = value,
            MetricType::QueueDepth => self.performance_metrics.queue_depth = value as usize,
            MetricType::CacheHitRate => self.performance_metrics.cache_hit_rate = value,
            MetricType::NetworkLatency => {
                self.performance_metrics.network_latency = Duration::from_millis(value as u64);
            }
        }
    }
    
    // ============ Optimization Methods ============
    
    /// Get current optimization settings
    pub fn get_optimization_settings(&self) -> &OptimizationSettings {
        &self.optimization_settings
    }
    
    /// Update optimization strategy
    pub fn set_optimization_strategy(&mut self, strategy: OptimizationStrategy) {
        self.optimization_settings.strategy = strategy;
        
        // Adjust settings based on strategy
        match strategy {
            OptimizationStrategy::LowLatency => {
                self.optimization_settings.batch_size = 1;
                self.optimization_settings.enable_prefetch = true;
                self.optimization_settings.max_concurrent = 10;
                self.optimization_settings.operation_timeout = Duration::from_secs(10);
            }
            OptimizationStrategy::HighThroughput => {
                self.optimization_settings.batch_size = 20;
                self.optimization_settings.enable_cache = true;
                self.optimization_settings.max_concurrent = 20;
                self.optimization_settings.operation_timeout = Duration::from_secs(60);
            }
            OptimizationStrategy::CostOptimized => {
                self.optimization_settings.batch_size = 10;
                self.optimization_settings.enable_compression = true;
                self.optimization_settings.max_cache_size = 50;
                self.optimization_settings.max_concurrent = 5;
            }
            OptimizationStrategy::QualityFirst => {
                self.optimization_settings.batch_size = 1;
                self.optimization_settings.enable_cache = false;
                self.optimization_settings.operation_timeout = Duration::from_secs(120);
                self.optimization_settings.max_concurrent = 3;
            }
            OptimizationStrategy::Adaptive => {
                // Keep current settings, will adjust based on load
                self.apply_adaptive_optimization();
            }
        }
    }
    
    /// Apply adaptive optimization based on current metrics
    fn apply_adaptive_optimization(&mut self) {
        let metrics = &self.performance_metrics;
        
        // High CPU usage - reduce batch size
        if metrics.cpu_usage > 80.0 {
            self.optimization_settings.batch_size = 
                (self.optimization_settings.batch_size / 2).max(1);
        }
        
        // Low cache hit rate - increase cache size
        if metrics.cache_hit_rate < 0.5 && self.optimization_settings.enable_cache {
            self.optimization_settings.max_cache_size = 
                (self.optimization_settings.max_cache_size * 2).min(500);
        }
        
        // High queue depth - increase concurrent requests
        if metrics.queue_depth > 50 {
            self.optimization_settings.max_concurrent = 
                (self.optimization_settings.max_concurrent * 2).min(50);
        }
        
        // High network latency - enable compression
        if metrics.network_latency > Duration::from_millis(500) {
            self.optimization_settings.enable_compression = true;
        }
    }
    
    /// Optimize session for specific workload
    pub fn optimize_for_workload(&mut self, workload: WorkloadType) {
        match workload {
            WorkloadType::Interactive => {
                self.set_optimization_strategy(OptimizationStrategy::LowLatency);
            }
            WorkloadType::Batch => {
                self.set_optimization_strategy(OptimizationStrategy::HighThroughput);
            }
            WorkloadType::Analytics => {
                self.set_optimization_strategy(OptimizationStrategy::CostOptimized);
            }
            WorkloadType::Development => {
                self.set_optimization_strategy(OptimizationStrategy::QualityFirst);
            }
        }
    }
    
    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let metrics = &self.performance_metrics;
        
        if metrics.cpu_usage > 90.0 {
            recommendations.push("High CPU usage detected. Consider reducing batch size or concurrent requests.".to_string());
        }
        
        if metrics.memory_usage > 1000.0 {
            recommendations.push("High memory usage. Consider reducing cache size or enabling compression.".to_string());
        }
        
        if metrics.cache_hit_rate < 0.3 && self.optimization_settings.enable_cache {
            recommendations.push("Low cache hit rate. Consider increasing cache size or adjusting cache strategy.".to_string());
        }
        
        if metrics.network_latency > Duration::from_secs(1) {
            recommendations.push("High network latency. Consider enabling compression or using a different endpoint.".to_string());
        }
        
        if metrics.queue_depth > 100 {
            recommendations.push("Large queue depth. Consider increasing concurrent request limit.".to_string());
        }
        
        recommendations
    }
}

/// Metric types for updates
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    CpuUsage,
    MemoryUsage,
    Throughput,
    QueueDepth,
    CacheHitRate,
    NetworkLatency,
}

/// Workload types for optimization
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    Interactive,
    Batch,
    Analytics,
    Development,
}