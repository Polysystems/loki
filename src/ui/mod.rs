// UI Module - Seamless User Interface for Model Orchestration
//
// This module provides a simplified, user-friendly interface on top of our
// sophisticated model orchestration backend.

pub mod api;
pub mod profiles;
pub mod sessions;
pub mod templates;
pub mod web_server;

use std::sync::Arc;

use anyhow::Result;
pub use api::SimplifiedAPI;
pub use profiles::{UserPreferences, UserProfile, UserProfileManager};
use serde::{Deserialize, Serialize};
pub use sessions::{ActiveSession, SessionManager, SessionStats};
pub use templates::{SetupCategory, SetupTemplate, SetupTemplateManager};
pub use web_server::WebUIServer;

/// Unique identifier for UI sessions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub const fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for setup templates
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SetupId(String);

impl SetupId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub const fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SetupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for users
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(String);

impl UserId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub const fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for UserId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Performance profile for setup templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub response_time: ResponseTimeClass,
    pub quality_level: QualityLevel,
    pub cost_class: CostClass,
    pub resource_usage: ResourceUsageClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseTimeClass {
    Lightning, // < 500ms
    Fast,      // < 2s
    Moderate,  // < 5s
    Thorough,  // > 5s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityLevel {
    Good,       // 0.7-0.8
    High,       // 0.8-0.9
    Premium,    // 0.9-0.95
    Excellence, // 0.95+
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostClass {
    Free,   // Local only
    Low,    // < $0.10/hour
    Medium, // $0.10-$0.50/hour
    High,   // > $0.50/hour
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceUsageClass {
    Light,     // < 4GB RAM
    Moderate,  // 4-8GB RAM
    Heavy,     // 8-16GB RAM
    Intensive, // > 16GB RAM
}

/// Cost estimation for setup templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub hourly_cost_cents: f32,
    pub per_request_cost_cents: f32,
    pub cost_class: CostClass,
    pub explanation: String,
}

/// Recommendation for users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub setup_template: SetupTemplate,
    pub reason: String,
    pub confidence_score: f32,
    pub expected_benefit: String,
}

/// Optimization suggestion for active setups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub current_setup: SetupId,
    pub suggested_changes: Vec<SetupChange>,
    pub expected_improvement: ImprovementMetrics,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SetupChange {
    AddModel { model_id: String, reason: String },
    RemoveModel { model_id: String, reason: String },
    ReplaceModel { old_model: String, new_model: String, reason: String },
    ChangeRouting { new_strategy: String, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    pub cost_change_percent: f32,
    pub speed_change_percent: f32,
    pub quality_change_percent: f32,
}

/// Usage record for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub session_id: SessionId,
    pub setup_id: SetupId,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub duration_seconds: u64,
    pub messages_sent: u32,
    pub total_cost_cents: f32,
    pub average_quality_score: f32,
    pub user_satisfaction: Option<f32>, // 1-5 rating
}

/// Real-time notification for UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UINotification {
    pub id: String,
    pub notification_type: NotificationType,
    pub title: String,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub actions: Vec<NotificationAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    Info,
    Success,
    Warning,
    Error,
    Optimization,
    Achievement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    pub label: String,
    pub action_type: String,
    pub payload: serde_json::Value,
}

/// Initialize the UI system
pub async fn initialize_ui_system(
    orchestrator: Arc<crate::models::IntegratedModelSystem>,
) -> Result<SimplifiedAPI> {
    let profile_manager = Arc::new(UserProfileManager::new().await?);
    let template_manager = Arc::new(SetupTemplateManager::new().await?);
    let session_manager = Arc::new(SessionManager::new().await?);

    // Load default setup templates
    template_manager.load_default_templates().await?;

    let api = SimplifiedAPI::new(orchestrator, profile_manager, template_manager, session_manager)
        .await?;

    Ok(api)
}
