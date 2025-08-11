//! Model Call Tracking System
//! 
//! Tracks all model API calls including tokens used, cost, response time,
//! and success/failure status for analytics and optimization.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::Result;

/// Session statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub total_calls: usize,
    pub total_tokens: usize,
    pub total_cost: f64,
}

/// Model call tracker
pub struct ModelCallTracker {
    /// Active tracking sessions
    active_sessions: Arc<RwLock<HashMap<String, TrackingSession>>>,
    
    /// Historical call records
    history: Arc<RwLock<Vec<CallRecord>>>,
    
    /// Aggregated statistics
    statistics: Arc<RwLock<ModelStatistics>>,
    
    /// Cost calculator
    cost_calculator: Arc<CostCalculator>,
    
    /// Maximum history size
    max_history_size: usize,
}

/// Tracking session for an active model call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingSession {
    pub session_id: String,
    pub model_id: String,
    pub model_name: String,
    pub start_time: DateTime<Utc>,
    pub request_tokens: Option<usize>,
    pub context_size: usize,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub metadata: HashMap<String, String>,
}

/// Completed call record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRecord {
    pub session_id: String,
    pub model_id: String,
    pub model_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_ms: u64,
    pub request_tokens: usize,
    pub response_tokens: usize,
    pub total_tokens: usize,
    pub cost_usd: f64,
    pub success: bool,
    pub error: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Aggregated model statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelStatistics {
    pub total_calls: usize,
    pub successful_calls: usize,
    pub failed_calls: usize,
    pub total_tokens: usize,
    pub total_cost_usd: f64,
    pub average_response_time_ms: f64,
    pub models_used: HashMap<String, ModelUsageStats>,
    pub hourly_stats: Vec<HourlyStats>,
    pub error_types: HashMap<String, usize>,
}

/// Per-model usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelUsageStats {
    pub calls: usize,
    pub tokens: usize,
    pub cost_usd: f64,
    pub avg_response_time_ms: f64,
    pub success_rate: f64,
    pub last_used: Option<DateTime<Utc>>,
}

/// Hourly statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyStats {
    pub hour: DateTime<Utc>,
    pub calls: usize,
    pub tokens: usize,
    pub cost_usd: f64,
    pub errors: usize,
}

/// Cost calculator for different models
pub struct CostCalculator {
    /// Pricing per model (cost per 1K tokens)
    pricing: HashMap<String, ModelPricing>,
}

/// Model pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_cost_per_1k: f64,
    pub output_cost_per_1k: f64,
    pub currency: String,
}

impl ModelCallTracker {
    /// Create a new model call tracker
    pub fn new() -> Self {
        Self {
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(ModelStatistics::default())),
            cost_calculator: Arc::new(CostCalculator::new()),
            max_history_size: 10000,
        }
    }
    
    /// Start tracking a model call
    pub async fn start_tracking(
        &self,
        model_id: &str,
        model_name: &str,
        context_size: usize,
        temperature: f32,
        max_tokens: Option<usize>,
    ) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        let session = TrackingSession {
            session_id: session_id.clone(),
            model_id: model_id.to_string(),
            model_name: model_name.to_string(),
            start_time: Utc::now(),
            request_tokens: None,
            context_size,
            temperature,
            max_tokens,
            metadata: HashMap::new(),
        };
        
        self.active_sessions.write().await.insert(session_id.clone(), session);
        
        tracing::debug!("Started tracking session {} for model {}", session_id, model_name);
        Ok(session_id)
    }
    
    /// Update request token count
    pub async fn update_request_tokens(&self, session_id: &str, tokens: usize) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.request_tokens = Some(tokens);
            tracing::debug!("Updated request tokens for session {}: {}", session_id, tokens);
        }
        Ok(())
    }
    
    /// Complete tracking with success
    pub async fn complete_tracking(
        &self,
        session_id: &str,
        response_tokens: usize,
    ) -> Result<CallRecord> {
        let mut sessions = self.active_sessions.write().await;
        let session = sessions.remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        let end_time = Utc::now();
        let duration_ms = (end_time - session.start_time).num_milliseconds() as u64;
        let request_tokens = session.request_tokens.unwrap_or(session.context_size);
        let total_tokens = request_tokens + response_tokens;
        
        // Calculate cost
        let cost_usd = self.cost_calculator.calculate_cost(
            &session.model_id,
            request_tokens,
            response_tokens,
        ).await;
        
        let record = CallRecord {
            session_id: session.session_id.clone(),
            model_id: session.model_id.clone(),
            model_name: session.model_name.clone(),
            start_time: session.start_time,
            end_time,
            duration_ms,
            request_tokens,
            response_tokens,
            total_tokens,
            cost_usd,
            success: true,
            error: None,
            metadata: session.metadata,
        };
        
        // Update history and statistics
        self.add_to_history(record.clone()).await;
        self.update_statistics(&record).await;
        
        tracing::info!(
            "Completed tracking session {} - {} tokens, ${:.4} USD, {}ms",
            session_id, total_tokens, cost_usd, duration_ms
        );
        
        Ok(record)
    }
    
    /// Complete tracking with error
    pub async fn fail_tracking(&self, session_id: &str, error: String) -> Result<CallRecord> {
        let mut sessions = self.active_sessions.write().await;
        let session = sessions.remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        let end_time = Utc::now();
        let duration_ms = (end_time - session.start_time).num_milliseconds() as u64;
        let request_tokens = session.request_tokens.unwrap_or(0);
        
        let record = CallRecord {
            session_id: session.session_id.clone(),
            model_id: session.model_id.clone(),
            model_name: session.model_name.clone(),
            start_time: session.start_time,
            end_time,
            duration_ms,
            request_tokens,
            response_tokens: 0,
            total_tokens: request_tokens,
            cost_usd: 0.0,
            success: false,
            error: Some(error.clone()),
            metadata: session.metadata,
        };
        
        // Update history and statistics
        self.add_to_history(record.clone()).await;
        self.update_statistics(&record).await;
        
        tracing::warn!(
            "Failed tracking session {} for model {}: {}",
            session_id, session.model_name, error
        );
        
        Ok(record)
    }
    
    /// Add metadata to a session
    pub async fn add_metadata(&self, session_id: &str, key: String, value: String) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.metadata.insert(key, value);
        }
        Ok(())
    }
    
    /// Add record to history
    async fn add_to_history(&self, record: CallRecord) {
        let mut history = self.history.write().await;
        history.push(record);
        
        // Trim history if it exceeds max size
        if history.len() > self.max_history_size {
            let trim_count = history.len() - self.max_history_size;
            history.drain(0..trim_count);
        }
    }
    
    /// Update statistics with a new record
    async fn update_statistics(&self, record: &CallRecord) {
        let mut stats = self.statistics.write().await;
        
        // Update total counts
        stats.total_calls += 1;
        if record.success {
            stats.successful_calls += 1;
        } else {
            stats.failed_calls += 1;
            
            // Track error types
            if let Some(ref error) = record.error {
                let error_type = self.categorize_error(error);
                *stats.error_types.entry(error_type).or_insert(0) += 1;
            }
        }
        
        // Update tokens and cost
        stats.total_tokens += record.total_tokens;
        stats.total_cost_usd += record.cost_usd;
        
        // Update average response time
        let total_time = stats.average_response_time_ms * (stats.total_calls - 1) as f64;
        stats.average_response_time_ms = (total_time + record.duration_ms as f64) / stats.total_calls as f64;
        
        // Update per-model stats
        let model_stats = stats.models_used.entry(record.model_id.clone())
            .or_insert_with(ModelUsageStats::default);
        
        model_stats.calls += 1;
        model_stats.tokens += record.total_tokens;
        model_stats.cost_usd += record.cost_usd;
        model_stats.last_used = Some(record.end_time);
        
        // Update success rate
        let successful = if record.success { 1.0 } else { 0.0 };
        model_stats.success_rate = 
            (model_stats.success_rate * (model_stats.calls - 1) as f64 + successful) 
            / model_stats.calls as f64;
        
        // Update average response time
        let model_total_time = model_stats.avg_response_time_ms * (model_stats.calls - 1) as f64;
        model_stats.avg_response_time_ms = 
            (model_total_time + record.duration_ms as f64) / model_stats.calls as f64;
        
        // Update hourly stats
        self.update_hourly_stats(&mut stats, record);
    }
    
    /// Update hourly statistics
    fn update_hourly_stats(&self, stats: &mut ModelStatistics, record: &CallRecord) {
        use chrono::{Datelike, Timelike};
        let dt = record.start_time;
        let hour = dt.date_naive().and_hms_opt(dt.hour(), 0, 0)
            .and_then(|naive| naive.and_local_timezone(chrono::Utc).single())
            .unwrap_or_else(|| dt);
        
        // Find or create hourly stat entry
        let hourly_stat = stats.hourly_stats.iter_mut()
            .find(|s| s.hour == hour);
        
        if let Some(stat) = hourly_stat {
            stat.calls += 1;
            stat.tokens += record.total_tokens;
            stat.cost_usd += record.cost_usd;
            if !record.success {
                stat.errors += 1;
            }
        } else {
            stats.hourly_stats.push(HourlyStats {
                hour,
                calls: 1,
                tokens: record.total_tokens,
                cost_usd: record.cost_usd,
                errors: if record.success { 0 } else { 1 },
            });
        }
        
        // Keep only last 24 hours
        if stats.hourly_stats.len() > 24 {
            stats.hourly_stats.remove(0);
        }
    }
    
    /// Categorize error for statistics
    fn categorize_error(&self, error: &str) -> String {
        let error_lower = error.to_lowercase();
        
        if error_lower.contains("timeout") {
            "timeout".to_string()
        } else if error_lower.contains("rate") || error_lower.contains("limit") {
            "rate_limit".to_string()
        } else if error_lower.contains("auth") || error_lower.contains("api_key") {
            "authentication".to_string()
        } else if error_lower.contains("network") || error_lower.contains("connection") {
            "network".to_string()
        } else if error_lower.contains("invalid") || error_lower.contains("validation") {
            "validation".to_string()
        } else {
            "other".to_string()
        }
    }
    
    /// Get current statistics
    pub async fn get_statistics(&self) -> ModelStatistics {
        self.statistics.read().await.clone()
    }
    
    /// Get session statistics (alias for get_statistics with SessionStats wrapper)
    pub async fn get_session_stats(&self) -> Result<SessionStats> {
        let stats = self.statistics.read().await;
        Ok(SessionStats {
            total_calls: stats.total_calls,
            total_tokens: stats.total_tokens,
            total_cost: stats.total_cost_usd,
        })
    }
    
    /// Get recent call history
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<CallRecord> {
        let history = self.history.read().await;
        let limit = limit.unwrap_or(100).min(history.len());
        
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get active sessions
    pub async fn get_active_sessions(&self) -> Vec<TrackingSession> {
        self.active_sessions.read().await
            .values()
            .cloned()
            .collect()
    }
    
    /// Clear all tracking data
    pub async fn clear_all(&self) {
        self.active_sessions.write().await.clear();
        self.history.write().await.clear();
        *self.statistics.write().await = ModelStatistics::default();
        
        tracing::info!("Cleared all model tracking data");
    }
    
    /// Export statistics as JSON
    pub async fn export_statistics(&self) -> Result<String> {
        let stats = self.statistics.read().await;
        Ok(serde_json::to_string_pretty(&*stats)?)
    }
    
    /// Get cost breakdown by model
    pub async fn get_cost_breakdown(&self) -> HashMap<String, f64> {
        let stats = self.statistics.read().await;
        stats.models_used.iter()
            .map(|(model, usage)| (model.clone(), usage.cost_usd))
            .collect()
    }
}

impl CostCalculator {
    /// Create a new cost calculator with default pricing
    pub fn new() -> Self {
        let mut pricing = HashMap::new();
        
        // OpenAI models
        pricing.insert("gpt-4".to_string(), ModelPricing {
            input_cost_per_1k: 0.03,
            output_cost_per_1k: 0.06,
            currency: "USD".to_string(),
        });
        pricing.insert("gpt-4-turbo".to_string(), ModelPricing {
            input_cost_per_1k: 0.01,
            output_cost_per_1k: 0.03,
            currency: "USD".to_string(),
        });
        pricing.insert("gpt-3.5-turbo".to_string(), ModelPricing {
            input_cost_per_1k: 0.0005,
            output_cost_per_1k: 0.0015,
            currency: "USD".to_string(),
        });
        
        // Anthropic models
        pricing.insert("claude-3-opus".to_string(), ModelPricing {
            input_cost_per_1k: 0.015,
            output_cost_per_1k: 0.075,
            currency: "USD".to_string(),
        });
        pricing.insert("claude-3-sonnet".to_string(), ModelPricing {
            input_cost_per_1k: 0.003,
            output_cost_per_1k: 0.015,
            currency: "USD".to_string(),
        });
        pricing.insert("claude-3-haiku".to_string(), ModelPricing {
            input_cost_per_1k: 0.00025,
            output_cost_per_1k: 0.00125,
            currency: "USD".to_string(),
        });
        
        // Google models
        pricing.insert("gemini-pro".to_string(), ModelPricing {
            input_cost_per_1k: 0.00025,
            output_cost_per_1k: 0.0005,
            currency: "USD".to_string(),
        });
        
        // Mistral models
        pricing.insert("mistral-large".to_string(), ModelPricing {
            input_cost_per_1k: 0.004,
            output_cost_per_1k: 0.012,
            currency: "USD".to_string(),
        });
        pricing.insert("mistral-medium".to_string(), ModelPricing {
            input_cost_per_1k: 0.0027,
            output_cost_per_1k: 0.0081,
            currency: "USD".to_string(),
        });
        
        // Local models (no cost)
        pricing.insert("llama2".to_string(), ModelPricing {
            input_cost_per_1k: 0.0,
            output_cost_per_1k: 0.0,
            currency: "USD".to_string(),
        });
        pricing.insert("codellama".to_string(), ModelPricing {
            input_cost_per_1k: 0.0,
            output_cost_per_1k: 0.0,
            currency: "USD".to_string(),
        });
        
        Self { pricing }
    }
    
    /// Calculate cost for a model call
    pub async fn calculate_cost(
        &self,
        model_id: &str,
        input_tokens: usize,
        output_tokens: usize,
    ) -> f64 {
        // Try exact match first
        if let Some(pricing) = self.pricing.get(model_id) {
            return self.calculate_with_pricing(pricing, input_tokens, output_tokens);
        }
        
        // Try to find a matching model by prefix
        for (key, pricing) in &self.pricing {
            if model_id.starts_with(key) || key.starts_with(model_id) {
                return self.calculate_with_pricing(pricing, input_tokens, output_tokens);
            }
        }
        
        // Default to no cost for unknown models
        0.0
    }
    
    /// Calculate cost with specific pricing
    fn calculate_with_pricing(
        &self,
        pricing: &ModelPricing,
        input_tokens: usize,
        output_tokens: usize,
    ) -> f64 {
        let input_cost = (input_tokens as f64 / 1000.0) * pricing.input_cost_per_1k;
        let output_cost = (output_tokens as f64 / 1000.0) * pricing.output_cost_per_1k;
        input_cost + output_cost
    }
    
    /// Update pricing for a model
    pub fn update_pricing(&mut self, model_id: String, pricing: ModelPricing) {
        self.pricing.insert(model_id, pricing);
    }
    
    /// Get pricing for a model
    pub fn get_pricing(&self, model_id: &str) -> Option<&ModelPricing> {
        self.pricing.get(model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_tracking_lifecycle() {
        let tracker = ModelCallTracker::new();
        
        // Start tracking
        let session_id = tracker.start_tracking(
            "gpt-4",
            "GPT-4",
            1000,
            0.7,
            Some(500),
        ).await.unwrap();
        
        // Update tokens
        tracker.update_request_tokens(&session_id, 1200).await.unwrap();
        
        // Complete tracking
        let record = tracker.complete_tracking(&session_id, 300).await.unwrap();
        
        assert_eq!(record.request_tokens, 1200);
        assert_eq!(record.response_tokens, 300);
        assert_eq!(record.total_tokens, 1500);
        assert!(record.success);
        assert!(record.cost_usd > 0.0);
    }
    
    #[tokio::test]
    async fn test_cost_calculation() {
        let calculator = CostCalculator::new();
        
        // Test GPT-4 pricing
        let cost = calculator.calculate_cost("gpt-4", 1000, 500).await;
        assert_eq!(cost, 0.03 + 0.03); // $0.03 input + $0.03 output
        
        // Test local model (free)
        let cost = calculator.calculate_cost("llama2", 10000, 5000).await;
        assert_eq!(cost, 0.0);
    }
    
    #[tokio::test]
    async fn test_statistics_update() {
        let tracker = ModelCallTracker::new();
        
        // Track multiple calls
        for i in 0..3 {
            let session_id = tracker.start_tracking(
                "gpt-3.5-turbo",
                "GPT-3.5 Turbo",
                500,
                0.7,
                None,
            ).await.unwrap();
            
            if i < 2 {
                tracker.complete_tracking(&session_id, 200).await.unwrap();
            } else {
                tracker.fail_tracking(&session_id, "Test error".to_string()).await.unwrap();
            }
        }
        
        let stats = tracker.get_statistics().await;
        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.successful_calls, 2);
        assert_eq!(stats.failed_calls, 1);
        assert!(stats.models_used.contains_key("gpt-3.5-turbo"));
    }
}