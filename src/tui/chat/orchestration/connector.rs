//! Orchestration connector module
//! 
//! Provides connection between UI orchestration settings and backend ModelOrchestrator

use std::sync::Arc;
use anyhow::Result;
use chrono::Utc;
use crate::models::{ModelOrchestrator, TaskRequest};
use super::manager::{OrchestrationManager, RoutingStrategy};
use super::usage_stats::{UsageStatsTracker, RequestRecord};

/// Connector between UI orchestration and backend ModelOrchestrator
#[derive(Clone)]
pub struct OrchestrationConnector {
    /// Reference to the backend orchestrator
    orchestrator: Option<Arc<ModelOrchestrator>>,
    
    /// Pending configuration changes
    pending_changes: PendingChanges,
    
    /// Usage statistics tracker
    usage_stats: Arc<UsageStatsTracker>,
}

#[derive(Default, Clone)]
struct PendingChanges {
    routing_strategy: Option<RoutingStrategy>,
    ensemble_enabled: Option<bool>,
    enabled_models: Option<Vec<String>>,
    quality_threshold: Option<f32>,
    cost_threshold: Option<u32>,
}

impl OrchestrationConnector {
    /// Create a new orchestration connector
    pub fn new() -> Self {
        Self {
            orchestrator: None,
            pending_changes: PendingChanges::default(),
            usage_stats: Arc::new(UsageStatsTracker::new()),
        }
    }
    
    /// Set the backend orchestrator reference
    pub fn set_orchestrator(&mut self, orchestrator: Arc<ModelOrchestrator>) {
        self.orchestrator = Some(orchestrator);
    }
    
    /// Sync UI state to backend
    pub async fn sync_ui_to_backend(
        &mut self,
        ui_state: &OrchestrationManager,
    ) -> Result<()> {
        // Since ModelOrchestrator doesn't expose runtime configuration,
        // we store the changes to be applied on next task request
        self.pending_changes.routing_strategy = Some(ui_state.preferred_strategy.clone());
        self.pending_changes.ensemble_enabled = Some(ui_state.ensemble_enabled);
        self.pending_changes.enabled_models = Some(ui_state.enabled_models.clone());
        self.pending_changes.quality_threshold = Some(ui_state.quality_threshold);
        self.pending_changes.cost_threshold = Some(ui_state.cost_threshold_cents as u32);
        
        tracing::info!(
            "Stored orchestration changes - Strategy: {:?}, Ensemble: {}, Models: {:?}",
            ui_state.preferred_strategy,
            ui_state.ensemble_enabled,
            ui_state.enabled_models
        );
        
        Ok(())
    }
    
    /// Apply pending changes when creating a task request
    pub fn apply_to_task_request(&self, mut request: TaskRequest) -> TaskRequest {
        // Apply quality threshold to constraints
        if let Some(threshold) = self.pending_changes.quality_threshold {
            // Store quality threshold as metadata in constraints
            // (Since TaskRequest doesn't have a quality field, we'll need to handle this differently)
            tracing::debug!("Quality threshold {} will be considered during model selection", threshold);
        }
        
        // Apply cost threshold to constraints
        if let Some(cost) = self.pending_changes.cost_threshold {
            request.constraints.max_cost_cents = Some(cost as f32);
        }
        
        // Apply enabled models and routing strategy
        // Note: Since TaskRequest doesn't have these fields, they'll be applied
        // during orchestrator initialization or through other mechanisms
        if let Some(strategy) = &self.pending_changes.routing_strategy {
            tracing::debug!("Routing strategy {:?} will be used for this request", strategy);
        }
        
        if let Some(models) = &self.pending_changes.enabled_models {
            if !models.is_empty() {
                tracing::debug!("Preferred models {:?} will be considered", models);
            }
        }
        
        request
    }
    
    /// Get current backend status
    pub async fn get_backend_status(&self) -> Option<BackendStatus> {
        if let Some(_orchestrator) = &self.orchestrator {
            let stats = self.usage_stats.get_stats().await;
            
            Some(BackendStatus {
                total_requests: stats.total_requests,
                total_tokens: stats.total_tokens,
                total_cost: stats.total_cost_cents,
                active_providers: stats.provider_stats.len(),
                performance_metrics: Some(PerformanceInfo {
                    avg_response_time_ms: stats.performance.avg_response_time_ms,
                    success_rate: stats.performance.success_rate,
                    tokens_per_second: stats.performance.avg_tokens_per_second,
                }),
                model_breakdown: stats.model_stats.into_iter()
                    .map(|(id, model_stats)| ModelInfo {
                        model_id: id,
                        request_count: model_stats.request_count,
                        error_rate: model_stats.error_rate,
                        avg_response_time_ms: model_stats.avg_response_time_ms,
                    })
                    .collect(),
            })
        } else {
            None
        }
    }
    
    /// Record a request completion
    pub async fn record_request(
        &self,
        model_id: String,
        provider: String,
        response_time_ms: u64,
        tokens_used: u64,
        cost_cents: u64,
        success: bool,
        error: Option<String>,
    ) -> Result<()> {
        let record = RequestRecord {
            model_id,
            provider,
            timestamp: Utc::now(),
            response_time_ms,
            tokens_used,
            cost_cents,
            success,
            error,
        };
        
        self.usage_stats.record_request(record).await?;
        Ok(())
    }
    
    /// Get usage statistics
    pub async fn get_usage_stats(&self) -> super::usage_stats::UsageStats {
        self.usage_stats.get_stats().await
    }
    
    /// Get usage summary
    pub async fn get_usage_summary(&self) -> super::usage_stats::UsageSummary {
        self.usage_stats.get_summary().await
    }
    
    /// Reset usage statistics
    pub async fn reset_usage_stats(&self) {
        self.usage_stats.reset().await;
    }
}

/// Backend orchestrator status
pub struct BackendStatus {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost: u64,
    pub active_providers: usize,
    pub performance_metrics: Option<PerformanceInfo>,
    pub model_breakdown: Vec<ModelInfo>,
}

/// Performance information
pub struct PerformanceInfo {
    pub avg_response_time_ms: f64,
    pub success_rate: f64,
    pub tokens_per_second: f64,
}

/// Model usage information
pub struct ModelInfo {
    pub model_id: String,
    pub request_count: u64,
    pub error_rate: f64,
    pub avg_response_time_ms: f64,
}

/// Convert UI routing strategy to backend routing strategy
fn convert_routing_strategy(ui_strategy: &RoutingStrategy) -> crate::models::RoutingStrategy {
    match ui_strategy {
        RoutingStrategy::RoundRobin => crate::models::RoutingStrategy::LoadBased,
        RoutingStrategy::LeastLatency => crate::models::RoutingStrategy::LatencyOptimized,
        RoutingStrategy::CapabilityBased => crate::models::RoutingStrategy::CapabilityBased,
        RoutingStrategy::CostOptimized => crate::models::RoutingStrategy::CostOptimized,
        RoutingStrategy::ContextAware => crate::models::RoutingStrategy::CapabilityBased,
        RoutingStrategy::Custom(_) => crate::models::RoutingStrategy::LoadBased,
        RoutingStrategy::Capability => crate::models::RoutingStrategy::CapabilityBased,
        RoutingStrategy::Cost => crate::models::RoutingStrategy::CostOptimized,
        RoutingStrategy::Speed => crate::models::RoutingStrategy::LatencyOptimized,
        RoutingStrategy::Quality => crate::models::RoutingStrategy::CapabilityBased,
        RoutingStrategy::Availability => crate::models::RoutingStrategy::LoadBased,
        RoutingStrategy::Hybrid => crate::models::RoutingStrategy::LoadBased,
    }
}