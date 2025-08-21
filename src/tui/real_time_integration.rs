use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tokio::sync::{broadcast, RwLock};
use tokio::time::interval;
use tracing::{debug, error};

use crate::cognitive::CognitiveSystem;
use crate::memory::CognitiveMemory;
use crate::monitoring::cost_analytics::{CostAlert, CostMetrics, CostAlertType, AlertSeverity};
use crate::monitoring::distributed_safety::DistributedSafetyEvent;
use crate::monitoring::real_time::{SystemAlert, SystemMetrics, AlertLevel};
use crate::tools::IntelligentToolManager;
use crate::tui::components::CognitiveSystemMetrics;
use crate::tui::monitoring::real_time_metrics_collector::RealTimeMetricsCollector;

/// Real-time metrics aggregator for TUI integration
pub struct RealTimeMetricsAggregator {
    /// Cognitive system reference
    cognitive_system: Arc<CognitiveSystem>,

    /// Memory system reference
    memory_system: Arc<CognitiveMemory>,

    /// Tool manager reference
    tool_manager: Arc<IntelligentToolManager>,

    /// System metrics broadcaster
    system_metrics_tx: broadcast::Sender<SystemMetrics>,

    /// System alerts broadcaster
    system_alerts_tx: broadcast::Sender<SystemAlert>,

    /// Cost metrics broadcaster
    cost_metrics_tx: broadcast::Sender<CostMetrics>,

    /// Cost alerts broadcaster
    cost_alerts_tx: broadcast::Sender<CostAlert>,

    /// Safety events broadcaster
    safety_events_tx: broadcast::Sender<DistributedSafetyEvent>,

    /// Cognitive metrics broadcaster
    cognitive_metrics_tx: broadcast::Sender<CognitiveSystemMetrics>,

    /// Update interval
    update_interval: Duration,

    /// Metrics history for trend analysis
    metrics_history: Arc<RwLock<VecDeque<SystemMetrics>>>,

    /// Last update timestamp
    last_update: Arc<RwLock<Instant>>,
    
    /// Real-time metrics collector
    metrics_collector: Arc<RealTimeMetricsCollector>,
}

impl RealTimeMetricsAggregator {
    /// Create a new real-time metrics aggregator
    pub fn new(
        cognitive_system: Arc<CognitiveSystem>,
        memory_system: Arc<CognitiveMemory>,
        tool_manager: Arc<IntelligentToolManager>,
    ) -> Self {
        let (system_metrics_tx, _) = broadcast::channel(1000);
        let (system_alerts_tx, _) = broadcast::channel(1000);
        let (cost_metrics_tx, _) = broadcast::channel(1000);
        let (cost_alerts_tx, _) = broadcast::channel(1000);
        let (safety_events_tx, _) = broadcast::channel(1000);
        let (cognitive_metrics_tx, _) = broadcast::channel(1000);

        Self {
            cognitive_system,
            memory_system,
            tool_manager,
            system_metrics_tx,
            system_alerts_tx,
            cost_metrics_tx,
            cost_alerts_tx,
            safety_events_tx,
            cognitive_metrics_tx,
            update_interval: Duration::from_millis(1000), // 1 second updates
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(3600))), // 1 hour history
            last_update: Arc::new(RwLock::new(Instant::now())),
            metrics_collector: Arc::new(RealTimeMetricsCollector::new()),
        }
    }

    /// Get system metrics receiver
    pub fn get_system_metrics_receiver(&self) -> broadcast::Receiver<SystemMetrics> {
        self.system_metrics_tx.subscribe()
    }

    /// Get system alerts receiver
    pub fn get_system_alerts_receiver(&self) -> broadcast::Receiver<SystemAlert> {
        self.system_alerts_tx.subscribe()
    }

    /// Get cost metrics receiver
    pub fn get_cost_metrics_receiver(&self) -> broadcast::Receiver<CostMetrics> {
        self.cost_metrics_tx.subscribe()
    }

    /// Get cost alerts receiver
    pub fn get_cost_alerts_receiver(&self) -> broadcast::Receiver<CostAlert> {
        self.cost_alerts_tx.subscribe()
    }

    /// Get safety events receiver
    pub fn get_safety_events_receiver(&self) -> broadcast::Receiver<DistributedSafetyEvent> {
        self.safety_events_tx.subscribe()
    }

    /// Get cognitive metrics receiver
    pub fn get_cognitive_metrics_receiver(&self) -> broadcast::Receiver<CognitiveSystemMetrics> {
        self.cognitive_metrics_tx.subscribe()
    }

    /// Start the real-time metrics collection
    pub async fn start(&self) -> Result<()> {
        let mut interval = interval(self.update_interval);
        let cognitive_system = self.cognitive_system.clone();
        let memory_system = self.memory_system.clone();
        let tool_manager = self.tool_manager.clone();
        let system_metrics_tx = self.system_metrics_tx.clone();
        let system_alerts_tx = self.system_alerts_tx.clone();
        let cost_metrics_tx = self.cost_metrics_tx.clone();
        let cost_alerts_tx = self.cost_alerts_tx.clone();
        let cognitive_metrics_tx = self.cognitive_metrics_tx.clone();
        let metrics_history = self.metrics_history.clone();
        let last_update = self.last_update.clone();
        let metrics_collector = self.metrics_collector.clone();

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                // Collect system metrics
                match metrics_collector.collect_metrics().await {
                    Ok(metrics) => {
                        // Store in history
                        {
                            let mut history = metrics_history.write().await;
                            history.push_back(metrics.clone());
                            if history.len() > 3600 {
                                history.pop_front();
                            }
                        }

                        // Broadcast metrics
                        if let Err(e) = system_metrics_tx.send(metrics.clone()) {
                            debug!("No TUI listeners for system metrics: {}", e);
                        }

                        // Check for alerts
                        if let Some(alert) = Self::check_system_alerts(&metrics).await {
                            if let Err(e) = system_alerts_tx.send(alert) {
                                debug!("No TUI listeners for system alerts: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect system metrics: {}", e);
                    }
                }

                // Collect cognitive metrics
                match Self::collect_cognitive_metrics(&cognitive_system, &memory_system, &tool_manager).await {
                    Ok(cognitive_metrics) => {
                        if let Err(e) = cognitive_metrics_tx.send(cognitive_metrics) {
                            debug!("No TUI listeners for cognitive metrics: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect cognitive metrics: {}", e);
                    }
                }

                // Collect cost metrics
                match Self::collect_cost_metrics(&tool_manager).await {
                    Ok(cost_metrics) => {
                        if let Err(e) = cost_metrics_tx.send(cost_metrics.clone()) {
                            debug!("No TUI listeners for cost metrics: {}", e);
                        }

                        // Check for cost alerts
                        if let Some(alert) = Self::check_cost_alerts(&cost_metrics).await {
                            if let Err(e) = cost_alerts_tx.send(alert) {
                                debug!("No TUI listeners for cost alerts: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to collect cost metrics: {}", e);
                    }
                }

                // Update last update timestamp
                *last_update.write().await = Instant::now();
            }
        });

        Ok(())
    }


    /// Collect cognitive system metrics
    async fn collect_cognitive_metrics(
        cognitive_system: &Arc<CognitiveSystem>,
        _memory_system: &Arc<CognitiveMemory>,
        _tool_manager: &Arc<IntelligentToolManager>,
    ) -> Result<CognitiveSystemMetrics> {
        // Get consciousness level (simplified)
        let consciousness_level = if let Some(consciousness) = cognitive_system.consciousness() {
            let stats = consciousness.get_stats_guard().await;
            stats.average_awareness_level as f32
        } else {
            0.5
        };

        // Get memory utilization (placeholder)
        let memory_utilization = 0.6;

        // Get active tasks (placeholder)
        let active_tasks = 4;

        // Get narrative coherence (placeholder)
        let narrative_coherence = 0.85;

        // Get attention flows (placeholder)
        let attention_flows = 3;

        // Get fractal depth (placeholder)
        let fractal_depth = 2;

        // Get coordination mode
        let coordination_mode = "Autonomous".to_string();

        Ok(CognitiveSystemMetrics {
            consciousness_level: consciousness_level as f64,
            memory_utilization,
            active_tasks,
            narrative_coherence,
            attention_flows,
            fractal_depth,
            coordination_mode,
        })
    }

    /// Collect cost metrics
    async fn collect_cost_metrics(tool_manager: &Arc<IntelligentToolManager>) -> Result<CostMetrics> {
        // Create placeholder cost metrics
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Get tool statistics for cost estimation
        let tool_stats = tool_manager.get_tool_statistics().await?;
        
        // Estimate costs based on tool usage
        // Typical costs per execution (in USD cents)
        const OPENAI_COST_PER_CALL: f64 = 0.02;  // 2 cents per call
        const ANTHROPIC_COST_PER_CALL: f64 = 0.03;  // 3 cents per call
        const LOCAL_COST_PER_CALL: f64 = 0.001;  // 0.1 cent for local compute
        
        // Calculate costs based on tool statistics
        let mut provider_costs = std::collections::HashMap::new();
        let mut total_cost = 0.0;
        
        for (tool_name, stats) in &tool_stats.per_tool_stats {
            let cost_per_call = if tool_name.contains("openai") || tool_name.contains("gpt") {
                OPENAI_COST_PER_CALL
            } else if tool_name.contains("anthropic") || tool_name.contains("claude") {
                ANTHROPIC_COST_PER_CALL
            } else {
                LOCAL_COST_PER_CALL
            };
            
            let tool_cost = stats.total_executions as f64 * cost_per_call;
            total_cost += tool_cost;
            
            // Group by provider
            let provider = if tool_name.contains("openai") || tool_name.contains("gpt") {
                "OpenAI"
            } else if tool_name.contains("anthropic") || tool_name.contains("claude") {
                "Anthropic"
            } else {
                "Local"
            };
            
            *provider_costs.entry(provider.to_string()).or_insert(0.0) += tool_cost;
        }
        
        // Convert provider costs to the expected format
        let providers: std::collections::HashMap<String, crate::monitoring::cost_analytics::ProviderCost> = 
            provider_costs.into_iter().map(|(provider, cost)| {
                (provider.clone(), crate::monitoring::cost_analytics::ProviderCost {
                    name: provider,
                    total_cost_usd: cost,
                    input_tokens: (tool_stats.total_executions * 500) as u64, // Estimate
                    output_tokens: (tool_stats.total_executions * 500) as u64, // Estimate
                    input_cost_usd: cost * 0.5,
                    output_cost_usd: cost * 0.5,
                    requests_count: tool_stats.total_executions as u64,
                    avg_cost_per_request: cost / tool_stats.total_executions.max(1) as f64,
                    daily_cost_usd: cost,
                    hourly_cost_usd: cost / 24.0,
                    rate_limits_hit: 0,
                    fallbacks_triggered: 0,
                    efficiency_score: 0.8,
                })
            }).collect();

        Ok(CostMetrics {
            providers,
            total: crate::monitoring::cost_analytics::TotalCost {
                today_usd: total_cost,
                this_hour_usd: total_cost / 24.0,  // Rough estimate
                this_month_usd: total_cost * 30.0,  // Project monthly
                all_time_usd: total_cost * 365.0,  // Project yearly
                projected_monthly_usd: total_cost * 30.0,
                cost_per_hour_avg: total_cost / 24.0,
                savings_from_optimization_usd: total_cost * 0.1,  // Assume 10% savings possible
            },
            models: std::collections::HashMap::new(),
            tasks: std::collections::HashMap::new(),
            trends: crate::monitoring::cost_analytics::CostTrends {
                hourly_spend_rate: 0.80,
                cost_acceleration: 0.05,
                predicted_daily_cost: 20.00,
                predicted_monthly_cost: 600.00,
                savings_opportunity: 15.0,
                efficiency_trend: 0.12,
            },
            timestamp,
        })
    }

    /// Check for system alerts
    async fn check_system_alerts(metrics: &SystemMetrics) -> Option<SystemAlert> {
        // Check CPU usage
        if metrics.cpu.usage_percent > 90.0 {
            return Some(SystemAlert {
                level: AlertLevel::Critical,
                metric: "cpu_usage".to_string(),
                value: metrics.cpu.usage_percent as f64,
                threshold: 90.0,
                message: format!("CPU usage is at {:.1}%", metrics.cpu.usage_percent),
                timestamp: metrics.timestamp,
                resolved: false,
            });
        }

        // Check memory usage
        if metrics.memory.usage_percent > 95.0 {
            return Some(SystemAlert {
                level: AlertLevel::Critical,
                metric: "memory_usage".to_string(),
                value: metrics.memory.usage_percent as f64,
                threshold: 95.0,
                message: format!("Memory usage is at {:.1}%", metrics.memory.usage_percent),
                timestamp: metrics.timestamp,
                resolved: false,
            });
        }

        // Check disk usage
        if metrics.disk.usage_percent > 90.0 {
            return Some(SystemAlert {
                level: AlertLevel::Warning,
                metric: "disk_usage".to_string(),
                value: metrics.disk.usage_percent as f64,
                threshold: 90.0,
                message: format!("Disk usage is at {:.1}%", metrics.disk.usage_percent),
                timestamp: metrics.timestamp,
                resolved: false,
            });
        }

        None
    }

    /// Check for cost alerts
    async fn check_cost_alerts(metrics: &CostMetrics) -> Option<CostAlert> {
        // Check hourly cost threshold
        if metrics.total.this_hour_usd > 10.0 {
            return Some(CostAlert {
                alert_type: CostAlertType::HourlyBudgetExceeded,
                threshold: 10.0,
                current_value: metrics.total.this_hour_usd,
                message: format!("Hourly cost is ${:.2}", metrics.total.this_hour_usd),
                timestamp: metrics.timestamp,
                provider: Some("aggregate".to_string()),
                severity: AlertSeverity::Critical,
                resolved: false,
            });
        }

        // Check daily cost threshold
        if metrics.total.today_usd > 100.0 {
            return Some(CostAlert {
                alert_type: CostAlertType::DailyBudgetExceeded,
                threshold: 100.0,
                current_value: metrics.total.today_usd,
                message: format!("Daily cost is ${:.2}", metrics.total.today_usd),
                timestamp: metrics.timestamp,
                provider: Some("aggregate".to_string()),
                severity: AlertSeverity::Critical,
                resolved: false,
            });
        }

        None
    }

    /// Get current metrics snapshot
    pub async fn get_current_metrics(&self) -> Option<SystemMetrics> {
        let history = self.metrics_history.read().await;
        history.back().cloned()
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self) -> Vec<SystemMetrics> {
        let history = self.metrics_history.read().await;
        history.iter().cloned().collect()
    }

    /// Get last update timestamp
    pub async fn get_last_update(&self) -> Instant {
        *self.last_update.read().await
    }

    /// Get health monitor (stub implementation)
    pub async fn get_health_monitor(&self) -> Option<Arc<crate::monitoring::health::HealthMonitor>> {
        // For now, return None - would return actual health monitor reference
        None
    }
}
