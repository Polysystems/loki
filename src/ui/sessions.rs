// Session Management - Active session tracking and management
//
// This manages active user sessions with setup templates.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{SessionId, SetupTemplate, UsageRecord, UserPreferences};
use crate::models::{IntegratedModelSystem, TaskResponse};

/// Active session with a setup template
#[derive(Debug, Clone)]
pub struct ActiveSession {
    pub id: SessionId,
    pub template: SetupTemplate,
    pub user_preferences: UserPreferences,
    pub stats: SessionStats,
    pub start_time: Instant,
    pub last_activity: Instant,
    pub is_initializing: bool,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub message_count: u32,
    pub total_cost_cents: f32,
    pub average_response_time_ms: f32,
    pub average_quality_score: f32,
    pub error_count: u32,
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            message_count: 0,
            total_cost_cents: 0.0,
            average_response_time_ms: 0.0,
            average_quality_score: 0.0,
            error_count: 0,
        }
    }
}

impl ActiveSession {
    pub fn new(template: SetupTemplate, user_preferences: UserPreferences) -> Self {
        Self {
            id: SessionId::new(),
            template,
            user_preferences,
            stats: SessionStats::default(),
            start_time: Instant::now(),
            last_activity: Instant::now(),
            is_initializing: true,
        }
    }

    pub fn get_uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    pub async fn get_active_models(&self) -> Vec<String> {
        self.template.models.iter().map(|m| m.model_id.clone()).collect()
    }

    pub fn get_request_constraints(&self) -> crate::models::TaskConstraints {
        crate::models::TaskConstraints {
            max_tokens: Some(2000),
            context_size: Some(4096),
            max_time: Some(std::time::Duration::from_secs(30)),
            max_latency_ms: Some(30000), // Max 30 seconds
            max_cost_cents: Some(10.0),  // Max 10 cents per request
            quality_threshold: Some(0.7),
            priority: "normal".to_string(), // Normal priority
            prefer_local: false,
            require_streaming: false,
            required_capabilities: Vec::new(),
            task_hint: None,
            creativity_level: None,
            formality_level: None,
            target_audience: None,
        }
    }

    pub async fn get_health_status(&self) -> super::api::SessionHealth {
        if self.is_initializing {
            super::api::SessionHealth::Warning("Session initializing".to_string())
        } else if self.stats.error_count > 5 {
            super::api::SessionHealth::Error("High error rate".to_string())
        } else {
            super::api::SessionHealth::Healthy
        }
    }

    pub fn update_stats(&mut self, response: &TaskResponse) {
        self.stats.message_count += 1;
        self.last_activity = Instant::now();

        if let Some(cost) = response.cost_cents {
            if cost > 0.0 {
                self.stats.total_cost_cents += cost;
            }
        }

        if let Some(time_ms) = response.generation_time_ms {
            let new_avg = (self.stats.average_response_time_ms
                * (self.stats.message_count - 1) as f32
                + time_ms as f32)
                / self.stats.message_count as f32;
            self.stats.average_response_time_ms = new_avg;
        }

        // Quality score calculation based on response characteristics
        let quality = if response.tokens_generated.unwrap_or(0) > 100
            && response.generation_time_ms.unwrap_or(0) < 10000
        {
            0.8
        } else if response.tokens_generated.unwrap_or(0) > 50 {
            0.6
        } else {
            0.4
        };

        if quality > 0.5 {
            let new_avg = (self.stats.average_quality_score
                * (self.stats.message_count - 1) as f32
                + quality)
                / self.stats.message_count as f32;
            self.stats.average_quality_score = new_avg;
        }
    }
}

/// Manages active sessions
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, ActiveSession>>>,
}

impl SessionManager {
    pub async fn new() -> Result<Self> {
        Ok(Self { sessions: Arc::new(RwLock::new(HashMap::new())) })
    }

    /// Create a new session
    pub async fn create_session(
        &self,
        template: SetupTemplate,
        user_preferences: UserPreferences,
    ) -> Result<SessionId> {
        let session = ActiveSession::new(template, user_preferences);
        let session_id = session.id.clone();

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    /// Initialize session with models
    pub async fn initialize_session(
        &self,
        session_id: &SessionId,
        _orchestrator: &IntegratedModelSystem,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(session_id) {
            // Initialize models in the background
            // This is where we'd warm up local models, check API connections, etc.
            session.is_initializing = false;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found: {}", session_id))
        }
    }

    /// Get active session
    pub async fn get_session(&self, session_id: &SessionId) -> Result<Option<ActiveSession>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned())
    }

    /// Record interaction in session
    pub async fn record_interaction(
        &self,
        session_id: &SessionId,
        response: &TaskResponse,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(session_id) {
            session.update_stats(response);
            debug!("Recorded interaction for session {}", session_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found: {}", session_id))
        }
    }

    /// Stop a session
    pub async fn stop_session(&self, session_id: &SessionId) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if sessions.remove(session_id).is_some() {
            info!("Stopped session: {}", session_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found: {}", session_id))
        }
    }

    /// Get all active sessions
    pub async fn list_active_sessions(&self) -> Result<Vec<SessionId>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.keys().cloned().collect())
    }

    /// Clean up inactive sessions
    pub async fn cleanup_inactive_sessions(&self, max_idle_minutes: u64) -> Result<u32> {
        let mut sessions = self.sessions.write().await;
        let max_idle_duration = std::time::Duration::from_secs(max_idle_minutes * 60);

        let initial_count = sessions.len();
        sessions.retain(|session_id, session| {
            let is_active = session.last_activity.elapsed() < max_idle_duration;
            if !is_active {
                info!("Cleaning up inactive session: {}", session_id);
            }
            is_active
        });

        let cleaned_count = initial_count - sessions.len();
        if cleaned_count > 0 {
            info!("Cleaned up {} inactive sessions", cleaned_count);
        }

        Ok(cleaned_count as u32)
    }

    /// Get session usage record for analytics
    pub async fn get_usage_record(&self, session_id: &SessionId) -> Result<Option<UsageRecord>> {
        let sessions = self.sessions.read().await;

        if let Some(session) = sessions.get(session_id) {
            Ok(Some(UsageRecord {
                session_id: session_id.clone(),
                setup_id: session.template.id.clone(),
                start_time: chrono::DateTime::from_timestamp(
                    session.start_time.elapsed().as_secs() as i64,
                    0,
                )
                .unwrap_or_else(chrono::Utc::now),
                duration_seconds: session.get_uptime_seconds(),
                messages_sent: session.stats.message_count,
                total_cost_cents: session.stats.total_cost_cents,
                average_quality_score: session.stats.average_quality_score,
                user_satisfaction: None, // To be set by user feedback
            }))
        } else {
            Ok(None)
        }
    }
}
