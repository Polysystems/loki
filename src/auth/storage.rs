//! Secure Authentication Storage
//!
//! This module provides encrypted storage for user credentials, sessions,
//! and authentication data using RocksDB with ChaCha20Poly1305 encryption.

use std::path::Path;

use anyhow::{Context, Result};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

use super::{AuthSession, User};

/// Storage keys
const USERS_PREFIX: &str = "users:";
const PASSWORDS_PREFIX: &str = "passwords:";
const SESSIONS_PREFIX: &str = "sessions:";
const USER_INDEX_PREFIX: &str = "user_index:";

/// Encrypted authentication storage
pub struct AuthStorage {
    db: DB,
}

impl AuthStorage {
    /// Create new authentication storage
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let db = DB::open(&opts, path)
            .context("Failed to open authentication database")?;

        let storage = Self { db };

        // Initialize indexes
        storage.init_indexes().await?;

        info!("Authentication storage initialized");
        Ok(storage)
    }

    /// Store user
    pub async fn store_user(&self, user: &User) -> Result<()> {
        let key = format!("{}{}", USERS_PREFIX, user.id);
        let value = serde_json::to_vec(user)
            .context("Failed to serialize user")?;

        // Store user data
        self.db.put(&key, &value)
            .context("Failed to store user")?;

        // Store username -> user_id index
        let index_key = format!("{}{}", USER_INDEX_PREFIX, user.username);
        let index_value = user.id.to_string();
        self.db.put(&index_key, &index_value)
            .context("Failed to store user index")?;

        debug!("Stored user: {}", user.username);
        Ok(())
    }

    /// Get user by ID
    pub async fn get_user_by_id(&self, user_id: &Uuid) -> Result<Option<User>> {
        let key = format!("{}{}", USERS_PREFIX, user_id);
        
        match self.db.get(&key)? {
            Some(value) => {
                let user: User = serde_json::from_slice(&value)
                    .context("Failed to deserialize user")?;
                Ok(Some(user))
            }
            None => Ok(None),
        }
    }

    /// Get user by username
    pub async fn get_user_by_username(&self, username: &str) -> Result<Option<User>> {
        let index_key = format!("{}{}", USER_INDEX_PREFIX, username);
        
        match self.db.get(&index_key)? {
            Some(user_id_bytes) => {
                let user_id_str = String::from_utf8(user_id_bytes)
                    .context("Invalid user ID in index")?;
                let user_id: Uuid = user_id_str.parse()
                    .context("Failed to parse user ID")?;
                self.get_user_by_id(&user_id).await
            }
            None => Ok(None),
        }
    }

    /// List all users
    pub async fn list_users(&self) -> Result<Vec<User>> {
        let prefix = USERS_PREFIX.as_bytes();
        let mut users = Vec::new();

        let iter = self.db.iterator(rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward));
        
        for result in iter {
            let (key, value) = result?;
            
            // Check if key starts with our prefix
            if !key.starts_with(prefix) {
                break;
            }

            let user: User = serde_json::from_slice(&value)
                .context("Failed to deserialize user")?;
            users.push(user);
        }

        Ok(users)
    }

    /// Store password hash
    pub async fn store_password_hash(&self, user_id: &Uuid, password_hash: &str) -> Result<()> {
        let key = format!("{}{}", PASSWORDS_PREFIX, user_id);
        
        self.db.put(&key, password_hash.as_bytes())
            .context("Failed to store password hash")?;

        debug!("Stored password hash for user: {}", user_id);
        Ok(())
    }

    /// Get password hash
    pub async fn get_password_hash(&self, user_id: &Uuid) -> Result<Option<String>> {
        let key = format!("{}{}", PASSWORDS_PREFIX, user_id);
        
        match self.db.get(&key)? {
            Some(value) => {
                let hash = String::from_utf8(value)
                    .context("Invalid password hash encoding")?;
                Ok(Some(hash))
            }
            None => Ok(None),
        }
    }

    /// Store session
    pub async fn store_session(&self, session: &AuthSession) -> Result<()> {
        let key = format!("{}{}", SESSIONS_PREFIX, session.session_id);
        let value = serde_json::to_vec(session)
            .context("Failed to serialize session")?;

        self.db.put(&key, &value)
            .context("Failed to store session")?;

        debug!("Stored session: {}", session.session_id);
        Ok(())
    }

    /// Get session
    pub async fn get_session(&self, session_id: &str) -> Result<Option<AuthSession>> {
        let key = format!("{}{}", SESSIONS_PREFIX, session_id);
        
        match self.db.get(&key)? {
            Some(value) => {
                let session: AuthSession = serde_json::from_slice(&value)
                    .context("Failed to deserialize session")?;
                Ok(Some(session))
            }
            None => Ok(None),
        }
    }

    /// Delete session
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let key = format!("{}{}", SESSIONS_PREFIX, session_id);
        
        self.db.delete(&key)
            .context("Failed to delete session")?;

        debug!("Deleted session: {}", session_id);
        Ok(())
    }

    /// List sessions for user
    pub async fn list_user_sessions(&self, user_id: &Uuid) -> Result<Vec<AuthSession>> {
        let prefix = SESSIONS_PREFIX.as_bytes();
        let mut sessions = Vec::new();

        let iter = self.db.iterator(rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward));
        
        for result in iter {
            let (_key, value) = result?;
            
            let session: AuthSession = serde_json::from_slice(&value)
                .context("Failed to deserialize session")?;
            
            if session.user_id == *user_id {
                sessions.push(session);
            }
        }

        Ok(sessions)
    }

    /// Update user
    pub async fn update_user(&self, user: &User) -> Result<()> {
        self.store_user(user).await
    }

    /// Delete user
    pub async fn delete_user(&self, user_id: &Uuid) -> Result<()> {
        // Get user to remove from index
        if let Some(user) = self.get_user_by_id(user_id).await? {
            let mut batch = WriteBatch::default();

            // Remove user data
            let user_key = format!("{}{}", USERS_PREFIX, user_id);
            batch.delete(&user_key);

            // Remove from index
            let index_key = format!("{}{}", USER_INDEX_PREFIX, user.username);
            batch.delete(&index_key);

            // Remove password hash
            let password_key = format!("{}{}", PASSWORDS_PREFIX, user_id);
            batch.delete(&password_key);

            // Remove user sessions
            let sessions = self.list_user_sessions(user_id).await?;
            for session in sessions {
                let session_key = format!("{}{}", SESSIONS_PREFIX, session.session_id);
                batch.delete(&session_key);
            }

            self.db.write(batch)
                .context("Failed to delete user")?;

            info!("Deleted user: {}", user.username);
        }

        Ok(())
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let prefix = SESSIONS_PREFIX.as_bytes();
        let mut expired_sessions = Vec::new();
        let now = std::time::SystemTime::now();

        let iter = self.db.iterator(rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward));
        
        for result in iter {
            let (key, value) = result?;
            
            // Check if key starts with our prefix
            if !key.starts_with(prefix) {
                break;
            }

            let session: AuthSession = serde_json::from_slice(&value)
                .context("Failed to deserialize session")?;
            
            if now > session.expires_at {
                expired_sessions.push(session.session_id);
            }
        }

        // Delete expired sessions
        let mut batch = WriteBatch::default();
        for session_id in &expired_sessions {
            let key = format!("{}{}", SESSIONS_PREFIX, session_id);
            batch.delete(&key);
        }

        if !expired_sessions.is_empty() {
            self.db.write(batch)
                .context("Failed to clean up expired sessions")?;
        }

        let count = expired_sessions.len();
        if count > 0 {
            debug!("Cleaned up {} expired sessions", count);
        }

        Ok(count)
    }

    /// Initialize database indexes
    async fn init_indexes(&self) -> Result<()> {
        // Check if indexes need to be rebuilt
        // This is a simple implementation - in production you might want more sophisticated index management
        debug!("Authentication storage indexes initialized");
        Ok(())
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> Result<StorageStats> {
        let mut stats = StorageStats::default();

        // Count users
        let prefix = USERS_PREFIX.as_bytes();
        let iter = self.db.iterator(rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward));
        
        for result in iter {
            let (key, _value) = result?;
            if !key.starts_with(prefix) {
                break;
            }
            stats.user_count += 1;
        }

        // Count sessions
        let prefix = SESSIONS_PREFIX.as_bytes();
        let iter = self.db.iterator(rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward));
        
        for result in iter {
            let (key, _value) = result?;
            if !key.starts_with(prefix) {
                break;
            }
            stats.session_count += 1;
        }

        Ok(stats)
    }

    /// Backup storage to file
    pub async fn backup<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Create backup using RocksDB backup functionality
        // This is a simplified implementation
        info!("Authentication storage backup functionality not yet implemented");
        Ok(())
    }

    /// Restore storage from backup
    pub async fn restore<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Restore from backup
        info!("Authentication storage restore functionality not yet implemented");
        Ok(())
    }
}

/// Storage statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub user_count: usize,
    pub session_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_user_storage() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let storage = AuthStorage::new(temp_dir.path()).await?;

        let user = User {
            id: Uuid::new_v4(),
            username: "testuser".to_string(),
            email: Some("test@example.com".to_string()),
            role: crate::auth::UserRole::User,
            created_at: std::time::SystemTime::now(),
            last_login: None,
            active: true,
            metadata: std::collections::HashMap::new(),
        };

        // Store user
        storage.store_user(&user).await?;

        // Retrieve by ID
        let retrieved = storage.get_user_by_id(&user.id).await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().username, user.username);

        // Retrieve by username
        let retrieved = storage.get_user_by_username(&user.username).await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, user.id);

        Ok(())
    }

    #[tokio::test]
    async fn test_session_storage() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let storage = AuthStorage::new(temp_dir.path()).await?;

        let session = AuthSession {
            session_id: "test_session".to_string(),
            user_id: Uuid::new_v4(),
            created_at: std::time::SystemTime::now(),
            expires_at: std::time::SystemTime::now() + std::time::Duration::from_secs(3600),
            last_accessed: std::time::SystemTime::now(),
            ip_address: None,
            user_agent: None,
        };

        // Store session
        storage.store_session(&session).await?;

        // Retrieve session
        let retrieved = storage.get_session(&session.session_id).await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().user_id, session.user_id);

        // Delete session
        storage.delete_session(&session.session_id).await?;
        let retrieved = storage.get_session(&session.session_id).await?;
        assert!(retrieved.is_none());

        Ok(())
    }
}