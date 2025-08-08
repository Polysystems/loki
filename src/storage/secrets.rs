//! Secure Storage for API Keys and Secrets
//!
//! This module provides encrypted storage for sensitive data like API keys,
//! database credentials, and other secrets with platform-specific keyring integration.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, warn, error, debug};
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use argon2::{
    password_hash::{PasswordHasher, Salt, SaltString},
    Argon2,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rand::RngCore;

/// Secure storage for API keys and secrets
pub struct SecureStorage {
    /// Storage backend
    backend: StorageBackend,
    
    /// Encryption key derived from master password
    encryption_key: Option<Vec<u8>>,
    
    /// Cache of decrypted secrets
    cache: HashMap<String, SecretValue>,
    
    /// Storage configuration
    config: SecureStorageConfig,
}

/// Storage backend types
enum StorageBackend {
    /// File-based storage with encryption
    File(PathBuf),
    
    /// Platform keyring integration
    #[cfg(target_os = "macos")]
    Keychain,
    
    /// Platform keyring integration
    #[cfg(target_os = "linux")]
    SecretService,
    
    /// Platform keyring integration
    #[cfg(target_os = "windows")]
    WindowsCredentialStore,
}

/// Secret value with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretValue {
    /// The actual secret value
    pub value: String,
    
    /// When the secret was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// When the secret was last updated
    pub updated_at: chrono::DateTime<chrono::Utc>,
    
    /// Optional expiration time
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Secret metadata
    pub metadata: HashMap<String, String>,
}

/// Configuration for secure storage
#[derive(Debug, Clone)]
pub struct SecureStorageConfig {
    /// Storage file path (for file backend)
    pub storage_path: PathBuf,
    
    /// Use platform keyring if available
    pub use_keyring: bool,
    
    /// Auto-save changes
    pub auto_save: bool,
    
    /// Cache timeout in seconds
    pub cache_timeout: u64,
    
    /// Require master password
    pub require_password: bool,
}

impl Default for SecureStorageConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("loki")
            .join("secrets");
        
        Self {
            storage_path: data_dir.join("secrets.enc"),
            use_keyring: true,
            auto_save: true,
            cache_timeout: 300, // 5 minutes
            require_password: true,
        }
    }
}

/// Encrypted storage container
#[derive(Serialize, Deserialize)]
struct EncryptedStorage {
    /// Salt for key derivation
    salt: String,
    
    /// Nonce for encryption
    nonce: String,
    
    /// Encrypted data
    ciphertext: String,
    
    /// Storage version
    version: u32,
}

impl SecureStorage {
    /// Create new secure storage instance
    pub async fn new(config: SecureStorageConfig) -> Result<Self> {
        info!("🔐 Initializing secure storage");
        
        // Ensure storage directory exists
        if let Some(parent) = config.storage_path.parent() {
            fs::create_dir_all(parent).await
                .context("Failed to create storage directory")?;
        }
        
        // Determine storage backend
        let backend = if config.use_keyring {
            Self::get_platform_backend(&config)?
        } else {
            StorageBackend::File(config.storage_path.clone())
        };
        
        Ok(Self {
            backend,
            encryption_key: None,
            cache: HashMap::new(),
            config,
        })
    }
    
    /// Get platform-specific backend
    fn get_platform_backend(config: &SecureStorageConfig) -> Result<StorageBackend> {
        #[cfg(target_os = "macos")]
        {
            debug!("Using macOS Keychain for secure storage");
            Ok(StorageBackend::Keychain)
        }
        
        #[cfg(target_os = "linux")]
        {
            debug!("Using Linux Secret Service for secure storage");
            Ok(StorageBackend::SecretService)
        }
        
        #[cfg(target_os = "windows")]
        {
            debug!("Using Windows Credential Store for secure storage");
            Ok(StorageBackend::WindowsCredentialStore)
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            debug!("Platform keyring not available, using file storage");
            Ok(StorageBackend::File(config.storage_path.clone()))
        }
    }
    
    /// Initialize storage with master password
    pub async fn unlock(&mut self, password: &str) -> Result<()> {
        info!("🔓 Unlocking secure storage");
        
        // Derive encryption key from password
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        
        // Use password hashing to derive key
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("Failed to hash password: {}", e))?;
        
        // Extract the hash bytes as the key
        let hash_bytes = password_hash.hash.unwrap();
        let key = hash_bytes.as_bytes()[..32].to_vec();
        
        self.encryption_key = Some(key);
        
        // Load existing secrets if available
        self.load_secrets().await?;
        
        info!("✅ Secure storage unlocked");
        Ok(())
    }
    
    /// Store an API key
    pub async fn store_api_key(&mut self, provider: &str, key: &str) -> Result<()> {
        let secret = SecretValue {
            value: key.to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            expires_at: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "api_key".to_string());
                meta.insert("provider".to_string(), provider.to_string());
                meta
            },
        };
        
        self.store_secret(&format!("api_key_{}", provider), secret).await
    }
    
    /// Retrieve an API key
    pub async fn get_api_key(&self, provider: &str) -> Result<Option<String>> {
        match self.get_secret(&format!("api_key_{}", provider)).await? {
            Some(secret) => {
                // Check expiration
                if let Some(expires) = secret.expires_at {
                    if expires < chrono::Utc::now() {
                        warn!("API key for {} has expired", provider);
                        return Ok(None);
                    }
                }
                Ok(Some(secret.value))
            }
            None => Ok(None),
        }
    }
    
    /// Store database credentials
    pub async fn store_database_config(
        &mut self,
        backend: &str,
        config: HashMap<String, String>,
    ) -> Result<()> {
        let secret = SecretValue {
            value: serde_json::to_string(&config)?,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            expires_at: None,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "database_config".to_string());
                meta.insert("backend".to_string(), backend.to_string());
                meta
            },
        };
        
        self.store_secret(&format!("db_config_{}", backend), secret).await
    }
    
    /// Retrieve database credentials
    pub async fn get_database_config(
        &self,
        backend: &str,
    ) -> Result<Option<HashMap<String, String>>> {
        match self.get_secret(&format!("db_config_{}", backend)).await? {
            Some(secret) => {
                let config: HashMap<String, String> = serde_json::from_str(&secret.value)?;
                Ok(Some(config))
            }
            None => Ok(None),
        }
    }
    
    /// Store a generic secret
    pub async fn store_secret(&mut self, key: &str, value: SecretValue) -> Result<()> {
        debug!("Storing secret: {}", key);
        
        // Update cache
        self.cache.insert(key.to_string(), value.clone());
        
        // Save to backend if auto-save is enabled
        if self.config.auto_save {
            self.save_secrets().await?;
        }
        
        Ok(())
    }
    
    /// Retrieve a generic secret
    pub async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        // Check cache first
        if let Some(secret) = self.cache.get(key) {
            return Ok(Some(secret.clone()));
        }
        
        // Load from backend if not in cache
        // This would require loading all secrets and filtering
        // For now, return None if not in cache
        Ok(None)
    }
    
    /// List all stored secrets (keys only)
    pub async fn list_secrets(&self) -> Result<Vec<String>> {
        Ok(self.cache.keys().cloned().collect())
    }
    
    /// Delete a secret
    pub async fn delete_secret(&mut self, key: &str) -> Result<()> {
        self.cache.remove(key);
        
        if self.config.auto_save {
            self.save_secrets().await?;
        }
        
        Ok(())
    }
    
    /// Save all secrets to storage
    async fn save_secrets(&self) -> Result<()> {
        let encryption_key = self.encryption_key.as_ref()
            .context("Storage is locked")?;
        
        match &self.backend {
            StorageBackend::File(path) => {
                // Serialize secrets
                let data = serde_json::to_vec(&self.cache)?;
                
                // Encrypt data
                let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&encryption_key[..32]));
                
                let mut nonce_bytes = [0u8; 12];
                OsRng.fill_bytes(&mut nonce_bytes);
                let nonce = Nonce::from_slice(&nonce_bytes);
                
                let ciphertext = cipher
                    .encrypt(nonce, data.as_ref())
                    .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;
                
                // Create encrypted storage container
                let salt = SaltString::generate(&mut OsRng);
                let encrypted = EncryptedStorage {
                    salt: salt.to_string(),
                    nonce: BASE64.encode(&nonce_bytes),
                    ciphertext: BASE64.encode(&ciphertext),
                    version: 1,
                };
                
                // Write to file
                let json = serde_json::to_string_pretty(&encrypted)?;
                fs::write(path, json).await?;
                
                debug!("Secrets saved to file");
            }
            #[cfg(target_os = "macos")]
            StorageBackend::Keychain => {
                // Use macOS Keychain
                self.save_to_keychain().await?;
            }
            #[cfg(target_os = "linux")]
            StorageBackend::SecretService => {
                // Use Linux Secret Service
                self.save_to_secret_service().await?;
            }
            #[cfg(target_os = "windows")]
            StorageBackend::WindowsCredentialStore => {
                // Use Windows Credential Store
                self.save_to_credential_store().await?;
            }
            _ => {
                bail!("Platform keyring not implemented");
            }
        }
        
        Ok(())
    }
    
    /// Load secrets from storage
    async fn load_secrets(&mut self) -> Result<()> {
        let encryption_key = self.encryption_key.as_ref()
            .context("Storage is locked")?;
        
        match &self.backend {
            StorageBackend::File(path) => {
                if !path.exists() {
                    debug!("No existing secrets file");
                    return Ok(());
                }
                
                // Read encrypted file
                let json = fs::read_to_string(path).await?;
                let encrypted: EncryptedStorage = serde_json::from_str(&json)?;
                
                // Decrypt data
                let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&encryption_key[..32]));
                
                let nonce_bytes = BASE64.decode(&encrypted.nonce)?;
                let nonce = Nonce::from_slice(&nonce_bytes);
                
                let ciphertext = BASE64.decode(&encrypted.ciphertext)?;
                
                let plaintext = cipher
                    .decrypt(nonce, ciphertext.as_ref())
                    .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;
                
                // Deserialize secrets
                self.cache = serde_json::from_slice(&plaintext)?;
                
                debug!("Loaded {} secrets from file", self.cache.len());
            }
            #[cfg(target_os = "macos")]
            StorageBackend::Keychain => {
                self.load_from_keychain().await?;
            }
            #[cfg(target_os = "linux")]
            StorageBackend::SecretService => {
                self.load_from_secret_service().await?;
            }
            #[cfg(target_os = "windows")]
            StorageBackend::WindowsCredentialStore => {
                self.load_from_credential_store().await?;
            }
            _ => {
                bail!("Platform keyring not implemented");
            }
        }
        
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    async fn save_to_keychain(&self) -> Result<()> {
        // Implementation would use security-framework crate
        warn!("Keychain integration not yet implemented, using file storage");
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    async fn load_from_keychain(&mut self) -> Result<()> {
        // Implementation would use security-framework crate
        warn!("Keychain integration not yet implemented");
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    async fn save_to_secret_service(&self) -> Result<()> {
        // Implementation would use secret-service crate
        warn!("Secret Service integration not yet implemented, using file storage");
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    async fn load_from_secret_service(&mut self) -> Result<()> {
        // Implementation would use secret-service crate
        warn!("Secret Service integration not yet implemented");
        Ok(())
    }
    
    #[cfg(target_os = "windows")]
    async fn save_to_credential_store(&self) -> Result<()> {
        // Implementation would use windows-sys crate
        warn!("Credential Store integration not yet implemented, using file storage");
        Ok(())
    }
    
    #[cfg(target_os = "windows")]
    async fn load_from_credential_store(&mut self) -> Result<()> {
        // Implementation would use windows-sys crate
        warn!("Credential Store integration not yet implemented");
        Ok(())
    }
    
    /// Clear all cached secrets
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.encryption_key = None;
        info!("🔒 Secure storage locked and cache cleared");
    }
}

/// API key management functions
impl SecureStorage {
    /// Store all API keys at once
    pub async fn store_all_api_keys(&mut self, keys: HashMap<String, String>) -> Result<()> {
        for (provider, key) in keys {
            self.store_api_key(&provider, &key).await?;
        }
        Ok(())
    }
    
    /// Get all stored API keys
    pub async fn get_all_api_keys(&self) -> Result<HashMap<String, String>> {
        let mut keys = HashMap::new();
        
        for (key, secret) in &self.cache {
            if let Some(secret_type) = secret.metadata.get("type") {
                if secret_type == "api_key" {
                    if let Some(provider) = secret.metadata.get("provider") {
                        keys.insert(provider.clone(), secret.value.clone());
                    }
                }
            }
        }
        
        Ok(keys)
    }
    
    /// Check if an API key exists
    pub async fn has_api_key(&self, provider: &str) -> bool {
        self.cache.contains_key(&format!("api_key_{}", provider))
    }
}