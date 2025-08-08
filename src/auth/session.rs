//! Session Management
//!
//! This module provides session management functionality including
//! session creation, validation, and cleanup.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

use super::{AuthSession, User, UserRole};

/// Session context for current user
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub user: User,
    pub session: AuthSession,
    pub permissions: Vec<String>,
}

impl SessionContext {
    /// Check if user has specific permission
    pub fn has_permission(&self, permission: &str) -> bool {
        // Admin users have all permissions
        if self.user.role == UserRole::Admin {
            return true;
        }

        self.permissions.contains(&permission.to_string())
    }

    /// Check if user can access resource
    pub fn can_access_resource(&self, resource_type: &str, resource_id: &str) -> bool {
        match resource_type {
            "session" => {
                // Users can access their own sessions
                self.has_permission("manage_own_sessions") || 
                self.has_permission("manage_all_sessions")
            }
            "analytics" => {
                self.has_permission("view_analytics")
            }
            "tools" => {
                self.has_permission("use_tools")
            }
            _ => false,
        }
    }

    /// Get user display name
    pub fn display_name(&self) -> String {
        self.user.metadata.get("display_name")
            .cloned()
            .unwrap_or_else(|| self.user.username.clone())
    }
}

/// Session manager for handling active sessions
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, SessionContext>>>,
    config: SessionConfig,
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub cleanup_interval: Duration,
    pub max_idle_time: Duration,
    pub session_timeout: Duration,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            max_idle_time: Duration::from_secs(1800),   // 30 minutes
            session_timeout: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl SessionManager {
    /// Create new session manager
    pub fn new(config: SessionConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Add session to manager
    pub async fn add_session(&self, session_context: SessionContext) -> Result<()> {
        let session_id = session_context.session.session_id.clone();
        self.sessions.write().await.insert(session_id.clone(), session_context);
        debug!("Added session to manager: {}", session_id);
        Ok(())
    }

    /// Get session context
    pub async fn get_session(&self, session_id: &str) -> Option<SessionContext> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    /// Remove session
    pub async fn remove_session(&self, session_id: &str) -> Result<()> {
        self.sessions.write().await.remove(session_id);
        debug!("Removed session from manager: {}", session_id);
        Ok(())
    }

    /// Update session activity
    pub async fn update_session_activity(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(context) = sessions.get_mut(session_id) {
            context.session.last_accessed = SystemTime::now();
            debug!("Updated session activity: {}", session_id);
        }
        Ok(())
    }

    /// Get active sessions count
    pub async fn active_sessions_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Get sessions for user
    pub async fn get_user_sessions(&self, user_id: &Uuid) -> Vec<SessionContext> {
        let sessions = self.sessions.read().await;
        sessions.values()
            .filter(|context| context.user.id == *user_id)
            .cloned()
            .collect()
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let now = SystemTime::now();
        let mut sessions = self.sessions.write().await;
        let mut expired_sessions = Vec::new();

        for (session_id, context) in sessions.iter() {
            let is_expired = now > context.session.expires_at ||
                (now.duration_since(context.session.last_accessed).unwrap_or(Duration::ZERO) > self.config.max_idle_time);

            if is_expired {
                expired_sessions.push(session_id.clone());
            }
        }

        for session_id in &expired_sessions {
            sessions.remove(session_id);
        }

        let count = expired_sessions.len();
        if count > 0 {
            info!("Cleaned up {} expired sessions", count);
        }

        Ok(count)
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let sessions = Arc::clone(&self.sessions);
        let cleanup_interval = self.config.cleanup_interval;
        let max_idle_time = self.config.max_idle_time;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                let now = SystemTime::now();
                let mut sessions_guard = sessions.write().await;
                let mut expired_sessions = Vec::new();

                for (session_id, context) in sessions_guard.iter() {
                    let is_expired = now > context.session.expires_at ||
                        (now.duration_since(context.session.last_accessed).unwrap_or(Duration::ZERO) > max_idle_time);

                    if is_expired {
                        expired_sessions.push(session_id.clone());
                    }
                }

                for session_id in &expired_sessions {
                    sessions_guard.remove(session_id);
                }

                if !expired_sessions.is_empty() {
                    debug!("Background cleanup removed {} expired sessions", expired_sessions.len());
                }
            }
        })
    }

    /// Get session statistics
    pub async fn get_stats(&self) -> SessionStats {
        let sessions = self.sessions.read().await;
        let mut stats = SessionStats {
            total_sessions: sessions.len(),
            ..Default::default()
        };

        let now = SystemTime::now();
        
        for context in sessions.values() {
            // Count by role
            match context.user.role {
                UserRole::Admin => stats.admin_sessions += 1,
                UserRole::User => stats.user_sessions += 1,
                UserRole::ReadOnly => stats.readonly_sessions += 1,
                UserRole::Guest => stats.guest_sessions += 1,
            }

            // Check if session is active (accessed within last 5 minutes)
            if now.duration_since(context.session.last_accessed).unwrap_or(Duration::MAX) < Duration::from_secs(300) {
                stats.active_sessions += 1;
            }
        }

        stats
    }
}

/// Session statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SessionStats {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub admin_sessions: usize,
    pub user_sessions: usize,
    pub readonly_sessions: usize,
    pub guest_sessions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_user() -> User {
        User {
            id: Uuid::new_v4(),
            username: "testuser".to_string(),
            email: Some("test@example.com".to_string()),
            role: UserRole::User,
            created_at: SystemTime::now(),
            last_login: Some(SystemTime::now()),
            active: true,
            metadata: HashMap::new(),
        }
    }

    fn create_test_session(user: &User) -> AuthSession {
        AuthSession {
            session_id: "test_session".to_string(),
            user_id: user.id,
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            last_accessed: SystemTime::now(),
            ip_address: None,
            user_agent: None,
        }
    }

    #[tokio::test]
    async fn test_session_context() {
        let user = create_test_user();
        let session = create_test_session(&user);
        
        let context = SessionContext {
            user: user.clone(),
            session,
            permissions: vec!["view_analytics".to_string()],
        };

        assert!(context.has_permission("view_analytics"));
        assert!(!context.has_permission("manage_all_sessions"));
        assert_eq!(context.display_name(), user.username);
    }

    #[tokio::test]
    async fn test_session_manager() -> Result<()> {
        let manager = SessionManager::new(SessionConfig::default());
        
        let user = create_test_user();
        let session = create_test_session(&user);
        let context = SessionContext {
            user: user.clone(),
            session: session.clone(),
            permissions: vec!["view_analytics".to_string()],
        };

        // Add session
        manager.add_session(context.clone()).await?;
        assert_eq!(manager.active_sessions_count().await, 1);

        // Get session
        let retrieved = manager.get_session(&session.session_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().user.id, user.id);

        // Remove session
        manager.remove_session(&session.session_id).await?;
        assert_eq!(manager.active_sessions_count().await, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_cleanup() -> Result<()> {
        let mut config = SessionConfig::default();
        config.max_idle_time = Duration::from_secs(1); // Very short for testing
        
        let manager = SessionManager::new(config);
        
        let user = create_test_user();
        let mut session = create_test_session(&user);
        session.last_accessed = SystemTime::now() - Duration::from_secs(2); // Already expired
        
        let context = SessionContext {
            user: user.clone(),
            session,
            permissions: vec![],
        };

        manager.add_session(context).await?;
        assert_eq!(manager.active_sessions_count().await, 1);

        // Cleanup should remove expired session
        let cleaned = manager.cleanup_expired_sessions().await?;
        assert_eq!(cleaned, 1);
        assert_eq!(manager.active_sessions_count().await, 0);

        Ok(())
    }
}