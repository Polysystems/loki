//! User Authentication System
//!
//! This module provides comprehensive user authentication and session management
//! for the TUI application with support for multiple authentication methods,
//! role-based access control, and secure credential storage.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use chacha20poly1305::{
    aead::KeyInit,
    ChaCha20Poly1305, Key,
};
use rand::RngCore;
use rand::prelude::*;
use rand::distributions::Alphanumeric;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

pub mod providers;
pub mod session;
pub mod storage;
pub mod tokens;

pub use providers::*;
pub use session::*;
pub use storage::*;
pub use tokens::*;

/// User roles for access control
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum UserRole {
    /// Full system access
    Admin,
    /// Standard user access
    User,
    /// Read-only access
    ReadOnly,
    /// Guest access (limited functionality)
    Guest,
}

impl UserRole {
    /// Check if role has permission for specific action
    pub fn has_permission(&self, action: &Permission) -> bool {
        match (self, action) {
            (UserRole::Admin, _) => true,
            (UserRole::User, Permission::CreateSession) => true,
            (UserRole::User, Permission::ManageOwnSessions) => true,
            (UserRole::User, Permission::ViewAnalytics) => true,
            (UserRole::User, Permission::UseTools) => true,
            (UserRole::ReadOnly, Permission::ViewAnalytics) => true,
            (UserRole::ReadOnly, Permission::ViewSessions) => true,
            (UserRole::Guest, Permission::ViewSessions) => true,
            _ => false,
        }
    }
}

/// System permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    CreateSession,
    ManageOwnSessions,
    ManageAllSessions,
    ViewSessions,
    ViewAnalytics,
    ManageUsers,
    UseTools,
    ModifySystem,
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: Option<String>,
    pub role: UserRole,
    pub created_at: SystemTime,
    pub last_login: Option<SystemTime>,
    pub active: bool,
    pub metadata: HashMap<String, String>,
}

/// Authentication credentials
#[derive(Debug)]
pub struct Credentials {
    pub username: String,
    pub password: String,
}

/// Authentication session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSession {
    pub session_id: String,
    pub user_id: Uuid,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub last_accessed: SystemTime,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub session_timeout: Duration,
    pub max_sessions_per_user: usize,
    pub password_min_length: usize,
    pub enable_oauth: bool,
    pub enable_api_keys: bool,
    pub storage_path: String,
    pub encryption_key: Option<[u8; 32]>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::from_secs(24 * 60 * 60),
            max_sessions_per_user: 5,
            password_min_length: 8,
            enable_oauth: true,
            enable_api_keys: true,
            storage_path: "./data/auth".to_string(),
            encryption_key: None,
        }
    }
}

/// Main authentication system
pub struct AuthSystem {
    storage: Arc<AuthStorage>,
    sessions: Arc<RwLock<HashMap<String, AuthSession>>>,
    config: AuthConfig,
    cipher: ChaCha20Poly1305,
    oauth_providers: HashMap<String, Box<dyn OAuthProvider>>,
}

impl AuthSystem {
    /// Create new authentication system
    pub async fn new(config: AuthConfig) -> Result<Self> {
        let encryption_key = config.encryption_key.unwrap_or_else(|| {
            let mut key = [0u8; 32];
            let mut rng = OsRng;
            rng.fill_bytes(&mut key);
            key
        });

        let cipher = ChaCha20Poly1305::new(Key::from_slice(&encryption_key));
        let storage = Arc::new(AuthStorage::new(&config.storage_path).await?);

        let mut auth_system = Self {
            storage,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            cipher,
            oauth_providers: HashMap::new(),
        };

        // Initialize OAuth providers if enabled
        if auth_system.config.enable_oauth {
            auth_system.init_oauth_providers().await?;
        }

        // Create default admin user if none exists
        auth_system.ensure_admin_user().await?;

        info!("Authentication system initialized");
        Ok(auth_system)
    }

    /// Register a new user
    pub async fn register_user(
        &self,
        username: &str,
        password: &str,
        email: Option<String>,
        role: UserRole,
    ) -> Result<Uuid> {
        // Validate password strength
        if password.len() < self.config.password_min_length {
            return Err(anyhow::anyhow!("Password too short"));
        }

        // Check if username already exists
        if self.storage.get_user_by_username(username).await?.is_some() {
            return Err(anyhow::anyhow!("Username already exists"));
        }

        // Hash password
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow::Error::msg(format!("Failed to hash password: {}", e)))?
            .to_string();

        // Create user
        let user = User {
            id: Uuid::new_v4(),
            username: username.to_string(),
            email,
            role,
            created_at: SystemTime::now(),
            last_login: None,
            active: true,
            metadata: HashMap::new(),
        };

        // Store user and password
        self.storage.store_user(&user).await?;
        self.storage.store_password_hash(&user.id, &password_hash).await?;

        info!("User registered: {}", username);
        Ok(user.id)
    }

    /// Authenticate user with credentials
    pub async fn authenticate(&self, credentials: &Credentials) -> Result<Option<User>> {
        let user = match self.storage.get_user_by_username(&credentials.username).await? {
            Some(user) => user,
            None => return Ok(None),
        };

        if !user.active {
            return Ok(None);
        }

        // Verify password
        let stored_hash = match self.storage.get_password_hash(&user.id).await? {
            Some(hash) => hash,
            None => return Ok(None),
        };

        let parsed_hash = PasswordHash::new(&stored_hash)
            .map_err(|e| anyhow::Error::msg(format!("Failed to parse stored password hash: {}", e)))?;

        let argon2 = Argon2::default();
        if argon2.verify_password(credentials.password.as_bytes(), &parsed_hash).is_ok() {
            Ok(Some(user))
        } else {
            Ok(None)
        }
    }

    /// Create authentication session
    pub async fn create_session(&self, user: &User) -> Result<String> {
        let session_id = generate_session_token();
        let now = SystemTime::now();
        let expires_at = now + self.config.session_timeout;

        let session = AuthSession {
            session_id: session_id.clone(),
            user_id: user.id,
            created_at: now,
            expires_at,
            last_accessed: now,
            ip_address: None,
            user_agent: None,
        };

        // Store session
        self.storage.store_session(&session).await?;
        self.sessions.write().await.insert(session_id.clone(), session);

        // Update user last login
        let mut updated_user = user.clone();
        updated_user.last_login = Some(now);
        self.storage.store_user(&updated_user).await?;

        info!("Session created for user: {}", user.username);
        Ok(session_id)
    }

    /// Validate session and return user
    pub async fn validate_session(&self, session_id: &str) -> Result<Option<User>> {
        let sessions = self.sessions.read().await;
        let session = match sessions.get(session_id) {
            Some(session) => session.clone(),
            None => {
                // Try loading from storage
                match self.storage.get_session(session_id).await? {
                    Some(session) => session,
                    None => return Ok(None),
                }
            }
        };

        // Check if session is expired
        if SystemTime::now() > session.expires_at {
            self.cleanup_session(session_id).await?;
            return Ok(None);
        }

        // Update last accessed time
        let mut updated_session = session.clone();
        updated_session.last_accessed = SystemTime::now();
        self.storage.store_session(&updated_session).await?;

        // Get user
        self.storage.get_user_by_id(&session.user_id).await
    }

    /// Logout user (invalidate session)
    pub async fn logout(&self, session_id: &str) -> Result<()> {
        self.cleanup_session(session_id).await?;
        info!("User logged out");
        Ok(())
    }

    /// Update session last accessed time
    pub async fn update_session(&self, session: &AuthSession) -> Result<()> {
        self.storage.store_session(session).await?;
        self.sessions.write().await.insert(session.session_id.clone(), session.clone());
        Ok(())
    }

    /// Get current user from session
    pub async fn get_current_user(&self, session_id: Option<&str>) -> Result<Option<User>> {
        match session_id {
            Some(session_id) => self.validate_session(session_id).await,
            None => Ok(None),
        }
    }

    /// Initialize OAuth providers
    async fn init_oauth_providers(&mut self) -> Result<()> {
        // Add OAuth providers based on configuration
        if let Ok(api_keys) = crate::config::ApiKeysConfig::from_env() {
            if let Some(github_config) = api_keys.github {
                let provider = Box::new(GitHubOAuthProvider::new(github_config)?);
                self.oauth_providers.insert("github".to_string(), provider);
            }

            if let Some(twitter_config) = api_keys.x_twitter {
                let provider = Box::new(TwitterOAuthProvider::new(twitter_config)?);
                self.oauth_providers.insert("twitter".to_string(), provider);
            }
        }

        Ok(())
    }

    /// Ensure admin user exists
    async fn ensure_admin_user(&self) -> Result<()> {
        // Check if any admin user exists
        let users = self.storage.list_users().await?;
        let has_admin = users.iter().any(|u| u.role == UserRole::Admin);

        if !has_admin {
            // Create default admin user
            let admin_password = generate_password(16);
            let admin_id = self.register_user(
                "admin",
                &admin_password,
                None,
                UserRole::Admin,
            ).await?;

            warn!("Created default admin user with password: {}", admin_password);
            warn!("Please change the admin password immediately!");
        }

        Ok(())
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let now = SystemTime::now();
        let mut sessions = self.sessions.write().await;
        let mut expired_sessions = Vec::new();

        for (session_id, session) in sessions.iter() {
            if now > session.expires_at {
                expired_sessions.push(session_id.clone());
            }
        }

        for session_id in &expired_sessions {
            sessions.remove(session_id);
            self.storage.delete_session(session_id).await?;
        }

        let count = expired_sessions.len();
        if count > 0 {
            debug!("Cleaned up {} expired sessions", count);
        }

        Ok(count)
    }

    /// Clean up specific session
    async fn cleanup_session(&self, session_id: &str) -> Result<()> {
        self.sessions.write().await.remove(session_id);
        self.storage.delete_session(session_id).await?;
        Ok(())
    }
}

/// Generate secure session token
pub fn generate_session_token() -> String {
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| rng.sample(Alphanumeric) as char)
        .collect()
}

/// Generate secure password
pub fn generate_password(length: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| rng.sample(Alphanumeric) as char)
        .collect()
}

/// Authentication result
#[derive(Debug)]
pub enum AuthResult {
    Success(User),
    InvalidCredentials,
    UserNotFound,
    AccountDisabled,
    SessionExpired,
}
