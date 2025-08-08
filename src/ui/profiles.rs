// User Profiles - Personal preferences and saved configurations
//
// This manages user preferences, favorite setups, and usage history
// to provide a personalized experience.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{SetupId, UsageRecord, UserId};

/// User profile with preferences and history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub id: UserId,
    pub name: Option<String>,
    pub email: Option<String>,
    pub preferences: UserPreferences,
    pub favorite_setups: Vec<SetupId>,
    pub usage_history: Vec<UsageRecord>,
    pub custom_setups: Vec<super::SetupTemplate>,
    pub budget_limits: Option<BudgetLimits>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
}

/// User preferences for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub preferred_response_time: ResponseTimePreference,
    pub budget_limit_cents_per_hour: Option<f32>,
    pub prefer_local_models: bool,
    pub auto_optimize_setups: bool,
    pub default_setup_id: Option<SetupId>,
    pub notification_preferences: NotificationPreferences,
    pub ui_theme: UITheme,
    pub language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseTimePreference {
    Speed,    // Prefer fastest response
    Quality,  // Prefer highest quality
    Balanced, // Balance speed and quality
}

impl std::fmt::Display for ResponseTimePreference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponseTimePreference::Speed => write!(f, "speed"),
            ResponseTimePreference::Quality => write!(f, "quality"),
            ResponseTimePreference::Balanced => write!(f, "balanced"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub setup_optimization_suggestions: bool,
    pub cost_alerts: bool,
    pub new_setup_recommendations: bool,
    pub performance_reports: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UITheme {
    Light,
    Dark,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetLimits {
    pub daily_limit_cents: Option<f32>,
    pub monthly_limit_cents: Option<f32>,
    pub per_session_limit_cents: Option<f32>,
    pub alert_threshold_percent: f32, // Alert when reaching X% of limit
}

/// Manages user profiles
pub struct UserProfileManager {
    profiles: Arc<RwLock<HashMap<UserId, UserProfile>>>,
}

impl UserProfileManager {
    pub async fn new() -> Result<Self> {
        Ok(Self { profiles: Arc::new(RwLock::new(HashMap::new())) })
    }

    /// Get or create a user profile
    pub async fn get_or_create_profile(&self, user_id: &UserId) -> Result<UserProfile> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get(user_id) {
            let mut updated_profile = profile.clone();
            updated_profile.last_active = chrono::Utc::now();
            profiles.insert(user_id.clone(), updated_profile.clone());
            Ok(updated_profile)
        } else {
            info!("Creating new user profile for {}", user_id);
            let new_profile = self.create_default_profile(user_id.clone());
            profiles.insert(user_id.clone(), new_profile.clone());
            Ok(new_profile)
        }
    }

    /// Update user preferences
    pub async fn update_preferences(
        &self,
        user_id: &UserId,
        preferences: UserPreferences,
    ) -> Result<()> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(user_id) {
            profile.preferences = preferences;
            profile.last_active = chrono::Utc::now();
            debug!("Updated preferences for user {}", user_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Add a setup to user's favorites
    pub async fn add_favorite_setup(&self, user_id: &UserId, setup_id: &SetupId) -> Result<()> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(user_id) {
            if !profile.favorite_setups.contains(setup_id) {
                profile.favorite_setups.push(setup_id.clone());
                profile.last_active = chrono::Utc::now();
                info!("Added setup {} to favorites for user {}", setup_id, user_id);
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Remove a setup from user's favorites
    pub async fn remove_favorite_setup(&self, user_id: &UserId, setup_id: &SetupId) -> Result<()> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(user_id) {
            profile.favorite_setups.retain(|id| id != setup_id);
            profile.last_active = chrono::Utc::now();
            info!("Removed setup {} from favorites for user {}", setup_id, user_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Add a recent setup to user's history
    pub async fn add_recent_setup(&self, user_id: &UserId, setup_id: &SetupId) -> Result<()> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(user_id) {
            // Create a usage record
            let usage_record = UsageRecord {
                session_id: super::SessionId::new(),
                setup_id: setup_id.clone(),
                start_time: chrono::Utc::now(),
                duration_seconds: 0, // Will be updated when session ends
                messages_sent: 0,
                total_cost_cents: 0.0,
                average_quality_score: 0.0,
                user_satisfaction: None,
            };

            profile.usage_history.push(usage_record);

            // Keep only recent 100 records
            if profile.usage_history.len() > 100 {
                profile.usage_history.remove(0);
            }

            profile.last_active = chrono::Utc::now();
            debug!("Added recent setup {} for user {}", setup_id, user_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Record completed session
    pub async fn record_session_completion(
        &self,
        user_id: &UserId,
        usage_record: UsageRecord,
    ) -> Result<()> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(user_id) {
            // Update existing record or add new one
            if let Some(existing) =
                profile.usage_history.iter_mut().find(|r| r.session_id == usage_record.session_id)
            {
                *existing = usage_record;
            } else {
                profile.usage_history.push(usage_record);
            }

            profile.last_active = chrono::Utc::now();
            debug!("Recorded session completion for user {}", user_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Get user's usage statistics
    pub async fn get_user_stats(&self, user_id: &UserId) -> Result<UserStats> {
        let profiles = self.profiles.read().await;

        if let Some(profile) = profiles.get(user_id) {
            let total_sessions = profile.usage_history.len();
            let total_cost: f32 = profile.usage_history.iter().map(|r| r.total_cost_cents).sum();
            let total_messages: u32 = profile.usage_history.iter().map(|r| r.messages_sent).sum();
            let average_session_duration = if total_sessions > 0 {
                profile.usage_history.iter().map(|r| r.duration_seconds).sum::<u64>()
                    / total_sessions as u64
            } else {
                0
            };

            // Most used setups
            let mut setup_usage: HashMap<SetupId, u32> = HashMap::new();
            for record in &profile.usage_history {
                *setup_usage.entry(record.setup_id.clone()).or_insert(0) += 1;
            }
            let mut most_used: Vec<_> = setup_usage.into_iter().collect();
            most_used.sort_by(|a, b| b.1.cmp(&a.1));
            let most_used_setups: Vec<SetupId> =
                most_used.into_iter().take(5).map(|(setup_id, _)| setup_id).collect();

            Ok(UserStats {
                total_sessions: total_sessions as u64,
                total_cost_cents: total_cost,
                total_messages,
                average_session_duration_seconds: average_session_duration,
                favorite_setups_count: profile.favorite_setups.len() as u32,
                most_used_setups,
                days_active: self.calculate_days_active(profile),
            })
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Set user's default setup
    pub async fn set_default_setup(&self, user_id: &UserId, setup_id: SetupId) -> Result<()> {
        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(user_id) {
            profile.preferences.default_setup_id = Some(setup_id.clone());
            profile.last_active = chrono::Utc::now();
            info!("Set default setup {} for user {}", setup_id, user_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("User profile not found: {}", user_id))
        }
    }

    /// Get personalized recommendations based on user history
    pub async fn get_personalized_recommendations(&self, user_id: &UserId) -> Result<Vec<SetupId>> {
        let profiles = self.profiles.read().await;

        if let Some(profile) = profiles.get(user_id) {
            // Simple recommendation based on usage patterns
            let mut setup_scores: HashMap<SetupId, f32> = HashMap::new();

            // Score based on recent usage
            for (i, record) in profile.usage_history.iter().rev().enumerate().take(10) {
                let recency_weight = 1.0 - (i as f32 * 0.1);
                let quality_weight = record.average_quality_score;
                let satisfaction_weight = record.user_satisfaction.unwrap_or(0.5);

                let score = recency_weight * quality_weight * satisfaction_weight;
                *setup_scores.entry(record.setup_id.clone()).or_insert(0.0) += score;
            }

            // Boost favorites
            for favorite in &profile.favorite_setups {
                *setup_scores.entry(favorite.clone()).or_insert(0.0) += 1.0;
            }

            // Sort by score and return top recommendations
            let mut recommendations: Vec<_> = setup_scores.into_iter().collect();
            recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            Ok(recommendations.into_iter().take(5).map(|(setup_id, _)| setup_id).collect())
        } else {
            Ok(Vec::new())
        }
    }

    // Helper methods

    fn create_default_profile(&self, user_id: UserId) -> UserProfile {
        UserProfile {
            id: user_id,
            name: None,
            email: None,
            preferences: UserPreferences {
                preferred_response_time: ResponseTimePreference::Balanced,
                budget_limit_cents_per_hour: Some(50.0), // $0.50 per hour default
                prefer_local_models: true,
                auto_optimize_setups: true,
                default_setup_id: None,
                notification_preferences: NotificationPreferences {
                    setup_optimization_suggestions: true,
                    cost_alerts: true,
                    new_setup_recommendations: true,
                    performance_reports: false,
                },
                ui_theme: UITheme::Auto,
                language: "en".to_string(),
            },
            favorite_setups: Vec::new(),
            usage_history: Vec::new(),
            custom_setups: Vec::new(),
            budget_limits: Some(BudgetLimits {
                daily_limit_cents: Some(500.0),       // $5 per day
                monthly_limit_cents: Some(5000.0),    // $50 per month
                per_session_limit_cents: Some(100.0), // $1 per session
                alert_threshold_percent: 80.0,
            }),
            created_at: chrono::Utc::now(),
            last_active: chrono::Utc::now(),
        }
    }

    fn calculate_days_active(&self, profile: &UserProfile) -> u32 {
        let mut active_dates = std::collections::HashSet::new();

        for record in &profile.usage_history {
            let date = record.start_time.date_naive();
            active_dates.insert(date);
        }

        active_dates.len() as u32
    }
}

/// User statistics for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStats {
    pub total_sessions: u64,
    pub total_cost_cents: f32,
    pub total_messages: u32,
    pub average_session_duration_seconds: u64,
    pub favorite_setups_count: u32,
    pub most_used_setups: Vec<SetupId>,
    pub days_active: u32,
}
