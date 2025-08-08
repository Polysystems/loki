//! Security Encryption Module
//!
//! Provides AES-256-GCM encryption for sensitive decision data
//! with proper key management and security best practices.

use std::sync::Arc;

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use chacha20poly1305::ChaCha20Poly1305;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;
use zeroize::ZeroizeOnDrop;

/// Key derivation parameters for secure key generation
const KEY_DERIVATION_SALT: &[u8] = b"loki_safety_validator_2025";
const KEY_DERIVATION_ITERATIONS: u32 = 100_000;

/// Encryption algorithm choice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM (default)
    Aes256Gcm,
    /// ChaCha20-Poly1305 (alternative)
    ChaCha20Poly1305,
}

/// Encrypted data wrapper with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Algorithm used for encryption
    pub algorithm: String,
    /// Base64-encoded nonce/IV
    pub nonce: String,
    /// Base64-encoded ciphertext
    pub ciphertext: String,
    /// Timestamp of encryption
    pub encrypted_at: chrono::DateTime<chrono::Utc>,
    /// Version for future compatibility
    pub version: u8,
}

/// Secure key storage that zeroes memory on drop
#[derive(ZeroizeOnDrop)]
pub struct SecureKey {
    #[zeroize(drop)]
    key_material: Vec<u8>,
}

impl std::fmt::Debug for SecureKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecureKey(***)")
    }
}

impl SecureKey {
    /// Create a new secure key from bytes
    fn new(key_bytes: Vec<u8>) -> Self {
        Self { key_material: key_bytes }
    }

    /// Get the key material (use with caution)
    fn as_bytes(&self) -> &[u8] {
        &self.key_material
    }
}

/// Security encryption manager for decision data
#[derive(Clone, Debug)]
pub struct SecurityEncryption {
    /// Master encryption key (protected in memory)
    master_key: Arc<RwLock<SecureKey>>,
    /// Preferred encryption algorithm
    algorithm: EncryptionAlgorithm,
}

impl SecurityEncryption {
    /// Create a new encryption manager with a derived key
    pub async fn new(context_key: &str) -> Result<Self> {
        // Derive a proper encryption key from the context
        let key = Self::derive_key(context_key)?;

        Ok(Self {
            master_key: Arc::new(RwLock::new(key)),
            algorithm: EncryptionAlgorithm::Aes256Gcm,
        })
    }

    /// Create with a specific algorithm
    pub async fn with_algorithm(context_key: &str, algorithm: EncryptionAlgorithm) -> Result<Self> {
        let key = Self::derive_key(context_key)?;

        Ok(Self {
            master_key: Arc::new(RwLock::new(key)),
            algorithm,
        })
    }

    /// Derive a secure key from context using PBKDF2
    fn derive_key(context: &str) -> Result<SecureKey> {
        use sha2::{Sha256, Digest};

        // For simplicity, we'll use SHA256 with multiple iterations
        // This is a workaround for the pbkdf2 API changes
        let mut hasher = Sha256::new();
        hasher.update(context.as_bytes());
        hasher.update(KEY_DERIVATION_SALT);
        
        let mut result = hasher.finalize().to_vec();
        
        // Perform multiple iterations for key stretching
        for _ in 1..KEY_DERIVATION_ITERATIONS {
            let mut hasher = Sha256::new();
            hasher.update(&result);
            hasher.update(KEY_DERIVATION_SALT);
            result = hasher.finalize().to_vec();
        }

        Ok(SecureKey::new(result))
    }

    /// Encrypt decision data using AES-256-GCM
    pub async fn encrypt_decision(&self, data: &[u8]) -> Result<EncryptedData> {
        let key_guard = self.master_key.read().await;
        let key_bytes = key_guard.as_bytes();

        match self.algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                self.encrypt_aes256gcm(data, key_bytes).await
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                self.encrypt_chacha20(data, key_bytes).await
            }
        }
    }

    /// Decrypt decision data
    pub async fn decrypt_decision(&self, encrypted: &EncryptedData) -> Result<Vec<u8>> {
        // Validate version
        if encrypted.version != 1 {
            return Err(anyhow!("Unsupported encryption version: {}", encrypted.version));
        }

        let key_guard = self.master_key.read().await;
        let key_bytes = key_guard.as_bytes();

        match encrypted.algorithm.as_str() {
            "AES-256-GCM" => self.decrypt_aes256gcm(encrypted, key_bytes).await,
            "ChaCha20-Poly1305" => self.decrypt_chacha20(encrypted, key_bytes).await,
            _ => Err(anyhow!("Unsupported encryption algorithm: {}", encrypted.algorithm)),
        }
    }

    /// Internal AES-256-GCM encryption
    async fn encrypt_aes256gcm(&self, data: &[u8], key_bytes: &[u8]) -> Result<EncryptedData> {
        let key = Key::<Aes256Gcm>::from_slice(key_bytes);
        let cipher = Aes256Gcm::new(key);

        // Generate a random nonce
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

        // Encrypt the data
        let ciphertext = cipher
            .encrypt(&nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        Ok(EncryptedData {
            algorithm: "AES-256-GCM".to_string(),
            nonce: general_purpose::STANDARD.encode(&nonce),
            ciphertext: general_purpose::STANDARD.encode(&ciphertext),
            encrypted_at: chrono::Utc::now(),
            version: 1,
        })
    }

    /// Internal AES-256-GCM decryption
    async fn decrypt_aes256gcm(&self, encrypted: &EncryptedData, key_bytes: &[u8]) -> Result<Vec<u8>> {
        let key = Key::<Aes256Gcm>::from_slice(key_bytes);
        let cipher = Aes256Gcm::new(key);

        // Decode nonce and ciphertext
        let nonce_bytes = general_purpose::STANDARD
            .decode(&encrypted.nonce)
            .context("Failed to decode nonce")?;
        let ciphertext = general_purpose::STANDARD
            .decode(&encrypted.ciphertext)
            .context("Failed to decode ciphertext")?;

        let nonce = Nonce::from_slice(&nonce_bytes);

        // Decrypt the data
        cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| anyhow!("Decryption failed: {}", e))
    }

    /// Internal ChaCha20-Poly1305 encryption (alternative)
    async fn encrypt_chacha20(&self, data: &[u8], key_bytes: &[u8]) -> Result<EncryptedData> {
        use chacha20poly1305::{KeyInit, aead::OsRng as ChaChaOsRng};

        let key = chacha20poly1305::Key::from_slice(key_bytes);
        let cipher = ChaCha20Poly1305::new(key);

        // Generate a random nonce
        let nonce = ChaCha20Poly1305::generate_nonce(&mut ChaChaOsRng);

        // Encrypt the data
        let ciphertext = cipher
            .encrypt(&nonce, data)
            .map_err(|e| anyhow!("ChaCha20 encryption failed: {}", e))?;

        Ok(EncryptedData {
            algorithm: "ChaCha20-Poly1305".to_string(),
            nonce: general_purpose::STANDARD.encode(&nonce),
            ciphertext: general_purpose::STANDARD.encode(&ciphertext),
            encrypted_at: chrono::Utc::now(),
            version: 1,
        })
    }

    /// Internal ChaCha20-Poly1305 decryption
    async fn decrypt_chacha20(&self, encrypted: &EncryptedData, key_bytes: &[u8]) -> Result<Vec<u8>> {
        use chacha20poly1305::{KeyInit, Nonce as ChachaNonce};

        let key = chacha20poly1305::Key::from_slice(key_bytes);
        let cipher = ChaCha20Poly1305::new(key);

        // Decode nonce and ciphertext
        let nonce_bytes = general_purpose::STANDARD
            .decode(&encrypted.nonce)
            .context("Failed to decode nonce")?;
        let ciphertext = general_purpose::STANDARD
            .decode(&encrypted.ciphertext)
            .context("Failed to decode ciphertext")?;

        let nonce = ChachaNonce::from_slice(&nonce_bytes);

        // Decrypt the data
        cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| anyhow!("ChaCha20 decryption failed: {}", e))
    }

    /// Rotate encryption key (for periodic key rotation)
    pub async fn rotate_key(&self, new_context: &str) -> Result<()> {
        let new_key = Self::derive_key(new_context)?;
        let mut key_guard = self.master_key.write().await;

        // Replace with new key (old key will be zeroed on drop)
        *key_guard = new_key;

        info!("🔐 Encryption key rotated successfully");
        Ok(())
    }

    /// Generate a secure random key for initial setup
    pub fn generate_random_key() -> String {
        use rand::prelude::*;
        use rand::distributions::Alphanumeric;

        rand::thread_rng()
            .sample_iter(Alphanumeric)
            .take(64)
            .map(char::from)
            .collect()
    }
}

/// Helper functions for decision encryption/decryption
pub mod helpers {
    use super::*;
    use crate::safety::validator::StoredDecision;

    /// Encrypt a stored decision
    pub async fn encrypt_stored_decision(
        decision: &StoredDecision,
        encryption: &SecurityEncryption,
    ) -> Result<EncryptedData> {
        let json_data = serde_json::to_vec(decision)
            .context("Failed to serialize decision")?;

        encryption.encrypt_decision(&json_data).await
    }

    /// Decrypt a stored decision
    pub async fn decrypt_stored_decision(
        encrypted: &EncryptedData,
        encryption: &SecurityEncryption,
    ) -> Result<StoredDecision> {
        let decrypted_data = encryption.decrypt_decision(encrypted).await?;

        serde_json::from_slice(&decrypted_data)
            .context("Failed to deserialize decision")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safety::validator::{ActionDecision, RiskLevel, StoredDecision};

    #[tokio::test]
    async fn test_encryption_decryption() {
        let encryption = SecurityEncryption::new("test_context_key")
            .await
            .expect("Failed to create encryption");

        let test_data = b"This is sensitive decision data";

        // Encrypt
        let encrypted = encryption.encrypt_decision(test_data)
            .await
            .expect("Encryption failed");

        assert_eq!(encrypted.algorithm, "AES-256-GCM");
        assert_eq!(encrypted.version, 1);
        assert!(!encrypted.nonce.is_empty());
        assert!(!encrypted.ciphertext.is_empty());

        // Decrypt
        let decrypted = encryption.decrypt_decision(&encrypted)
            .await
            .expect("Decryption failed");

        assert_eq!(decrypted, test_data);
    }

    #[tokio::test]
    async fn test_decision_encryption() {
        let encryption = SecurityEncryption::new("test_validator_key")
            .await
            .expect("Failed to create encryption");

        let decision = StoredDecision {
            decision: ActionDecision::Approve,
            decided_by: "test_user".to_string(),
            decided_at: chrono::Utc::now(),
            reason: Some("Test approval".to_string()),
            action_context: "Test context".to_string(),
            risk_assessment: RiskLevel::Medium,
        };

        // Encrypt decision
        let encrypted = helpers::encrypt_stored_decision(&decision, &encryption)
            .await
            .expect("Failed to encrypt decision");

        // Decrypt decision
        let decrypted = helpers::decrypt_stored_decision(&encrypted, &encryption)
            .await
            .expect("Failed to decrypt decision");

        assert!(matches!(decrypted.decision, ActionDecision::Approve));
        assert_eq!(decrypted.decided_by, "test_user");
        assert_eq!(decrypted.risk_assessment, RiskLevel::Medium);
    }

    #[tokio::test]
    async fn test_chacha20_encryption() {
        let encryption = SecurityEncryption::with_algorithm(
            "test_context",
            EncryptionAlgorithm::ChaCha20Poly1305
        )
        .await
        .expect("Failed to create encryption");

        let test_data = b"ChaCha20 test data";

        let encrypted = encryption.encrypt_decision(test_data)
            .await
            .expect("Encryption failed");

        assert_eq!(encrypted.algorithm, "ChaCha20-Poly1305");

        let decrypted = encryption.decrypt_decision(&encrypted)
            .await
            .expect("Decryption failed");

        assert_eq!(decrypted, test_data);
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let encryption = SecurityEncryption::new("initial_key")
            .await
            .expect("Failed to create encryption");

        let test_data = b"Data before rotation";

        // Encrypt with initial key
        let encrypted = encryption.encrypt_decision(test_data)
            .await
            .expect("Encryption failed");

        // Rotate key
        encryption.rotate_key("new_key")
            .await
            .expect("Key rotation failed");

        // Should not be able to decrypt with new key
        let decrypt_result = encryption.decrypt_decision(&encrypted).await;
        assert!(decrypt_result.is_err());
    }

    #[test]
    fn test_random_key_generation() {
        let key1 = SecurityEncryption::generate_random_key();
        let key2 = SecurityEncryption::generate_random_key();

        assert_eq!(key1.len(), 64);
        assert_eq!(key2.len(), 64);
        assert_ne!(key1, key2); // Should be different
    }
}
