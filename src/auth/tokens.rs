//! Token Management
//!
//! This module provides JWT token generation and validation for
//! stateless authentication and API access.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
// use chrono::{DateTime, Utc}; // Unused imports
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{User, UserRole};

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Username
    pub username: String,
    /// User role
    pub role: String,
    /// Issued at (timestamp)
    pub iat: u64,
    /// Expiration time (timestamp)
    pub exp: u64,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// Custom claims
    pub custom: HashMap<String, String>,
}

/// Token configuration
#[derive(Debug, Clone)]
pub struct TokenConfig {
    pub secret: String,
    pub issuer: String,
    pub audience: String,
    pub access_token_lifetime: Duration,
    pub refresh_token_lifetime: Duration,
    pub algorithm: Algorithm,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            secret: "your-secret-key".to_string(), // Should be loaded from secure config
            issuer: "loki-ai-system".to_string(),
            audience: "loki-tui".to_string(),
            access_token_lifetime: Duration::from_secs(3600), // 1 hour
            refresh_token_lifetime: Duration::from_secs(604800), // 1 week
            algorithm: Algorithm::HS256,
        }
    }
}

/// Token pair (access + refresh)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: u64,
}

/// Refresh token claims
#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshClaims {
    pub sub: String,
    pub iat: u64,
    pub exp: u64,
    pub iss: String,
    pub aud: String,
    pub token_type: String,
}

/// Token manager for JWT operations
pub struct TokenManager {
    config: TokenConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
}

impl TokenManager {
    /// Create new token manager
    pub fn new(config: TokenConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(config.secret.as_ref());
        let decoding_key = DecodingKey::from_secret(config.secret.as_ref());
        
        let mut validation = Validation::new(config.algorithm);
        validation.set_issuer(&[&config.issuer]);
        validation.set_audience(&[&config.audience]);

        Self {
            config,
            encoding_key,
            decoding_key,
            validation,
        }
    }

    /// Generate access token for user
    pub fn generate_access_token(&self, user: &User) -> Result<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        let claims = Claims {
            sub: user.id.to_string(),
            username: user.username.clone(),
            role: format!("{:?}", user.role).to_lowercase(),
            iat: now,
            exp: now + self.config.access_token_lifetime.as_secs(),
            iss: self.config.issuer.clone(),
            aud: self.config.audience.clone(),
            custom: user.metadata.clone(),
        };

        encode(&Header::new(self.config.algorithm), &claims, &self.encoding_key)
            .context("Failed to encode access token")
    }

    /// Generate refresh token for user
    pub fn generate_refresh_token(&self, user: &User) -> Result<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        let claims = RefreshClaims {
            sub: user.id.to_string(),
            iat: now,
            exp: now + self.config.refresh_token_lifetime.as_secs(),
            iss: self.config.issuer.clone(),
            aud: self.config.audience.clone(),
            token_type: "refresh".to_string(),
        };

        encode(&Header::new(self.config.algorithm), &claims, &self.encoding_key)
            .context("Failed to encode refresh token")
    }

    /// Generate token pair (access + refresh)
    pub fn generate_token_pair(&self, user: &User) -> Result<TokenPair> {
        let access_token = self.generate_access_token(user)?;
        let refresh_token = self.generate_refresh_token(user)?;

        Ok(TokenPair {
            access_token,
            refresh_token,
            token_type: "Bearer".to_string(),
            expires_in: self.config.access_token_lifetime.as_secs(),
        })
    }

    /// Validate and decode access token
    pub fn validate_access_token(&self, token: &str) -> Result<Claims> {
        let token_data = decode::<Claims>(token, &self.decoding_key, &self.validation)
            .context("Failed to decode access token")?;

        Ok(token_data.claims)
    }

    /// Validate and decode refresh token
    pub fn validate_refresh_token(&self, token: &str) -> Result<RefreshClaims> {
        let token_data = decode::<RefreshClaims>(token, &self.decoding_key, &self.validation)
            .context("Failed to decode refresh token")?;

        // Additional validation for refresh tokens
        if token_data.claims.token_type != "refresh" {
            return Err(anyhow::anyhow!("Invalid token type"));
        }

        Ok(token_data.claims)
    }

    /// Extract user info from token claims
    pub fn extract_user_info(&self, claims: &Claims) -> Result<UserInfo> {
        let user_id: Uuid = claims.sub.parse()
            .context("Invalid user ID in token")?;

        let role = match claims.role.as_str() {
            "admin" => UserRole::Admin,
            "user" => UserRole::User,
            "readonly" => UserRole::ReadOnly,
            "guest" => UserRole::Guest,
            _ => return Err(anyhow::anyhow!("Invalid role in token")),
        };

        Ok(UserInfo {
            id: user_id,
            username: claims.username.clone(),
            role,
            metadata: claims.custom.clone(),
        })
    }

    /// Check if token is expired
    pub fn is_token_expired(&self, claims: &Claims) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now >= claims.exp
    }

    /// Get token remaining lifetime
    pub fn get_token_remaining_lifetime(&self, claims: &Claims) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if claims.exp > now {
            Duration::from_secs(claims.exp - now)
        } else {
            Duration::ZERO
        }
    }

    /// Create API key token (long-lived)
    pub fn generate_api_key(&self, user: &User, name: &str, expires_in: Option<Duration>) -> Result<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        let expiration = expires_in
            .map(|d| now + d.as_secs())
            .unwrap_or(now + Duration::from_secs(31536000).as_secs()); // 1 year default

        let mut custom_claims = user.metadata.clone();
        custom_claims.insert("api_key_name".to_string(), name.to_string());
        custom_claims.insert("token_type".to_string(), "api_key".to_string());

        let claims = Claims {
            sub: user.id.to_string(),
            username: user.username.clone(),
            role: format!("{:?}", user.role).to_lowercase(),
            iat: now,
            exp: expiration,
            iss: self.config.issuer.clone(),
            aud: self.config.audience.clone(),
            custom: custom_claims,
        };

        encode(&Header::new(self.config.algorithm), &claims, &self.encoding_key)
            .context("Failed to encode API key")
    }

    /// Validate API key
    pub fn validate_api_key(&self, api_key: &str) -> Result<Claims> {
        let claims = self.validate_access_token(api_key)?;

        // Check if it's actually an API key
        if claims.custom.get("token_type") != Some(&"api_key".to_string()) {
            return Err(anyhow::anyhow!("Token is not an API key"));
        }

        Ok(claims)
    }
}

/// Extracted user information from token
#[derive(Debug, Clone)]
pub struct UserInfo {
    pub id: Uuid,
    pub username: String,
    pub role: UserRole,
    pub metadata: HashMap<String, String>,
}

/// Token validation result
#[derive(Debug)]
pub enum TokenValidationResult {
    Valid(Claims),
    Expired,
    Invalid,
    Malformed,
}

impl From<Result<Claims>> for TokenValidationResult {
    fn from(result: Result<Claims>) -> Self {
        match result {
            Ok(claims) => TokenValidationResult::Valid(claims),
            Err(e) => {
                let error_msg = e.to_string().to_lowercase();
                if error_msg.contains("expired") {
                    TokenValidationResult::Expired
                } else if error_msg.contains("invalid") {
                    TokenValidationResult::Invalid
                } else {
                    TokenValidationResult::Malformed
                }
            }
        }
    }
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

    #[test]
    fn test_token_generation_and_validation() -> Result<()> {
        let config = TokenConfig::default();
        let token_manager = TokenManager::new(config);
        let user = create_test_user();

        // Generate access token
        let access_token = token_manager.generate_access_token(&user)?;
        assert!(!access_token.is_empty());

        // Validate access token
        let claims = token_manager.validate_access_token(&access_token)?;
        assert_eq!(claims.username, user.username);
        assert_eq!(claims.sub, user.id.to_string());

        Ok(())
    }

    #[test]
    fn test_token_pair_generation() -> Result<()> {
        let config = TokenConfig::default();
        let token_manager = TokenManager::new(config);
        let user = create_test_user();

        // Generate token pair
        let token_pair = token_manager.generate_token_pair(&user)?;
        assert!(!token_pair.access_token.is_empty());
        assert!(!token_pair.refresh_token.is_empty());
        assert_eq!(token_pair.token_type, "Bearer");

        // Validate both tokens
        let access_claims = token_manager.validate_access_token(&token_pair.access_token)?;
        let refresh_claims = token_manager.validate_refresh_token(&token_pair.refresh_token)?;

        assert_eq!(access_claims.sub, refresh_claims.sub);

        Ok(())
    }

    #[test]
    fn test_api_key_generation() -> Result<()> {
        let config = TokenConfig::default();
        let token_manager = TokenManager::new(config);
        let user = create_test_user();

        // Generate API key
        let api_key = token_manager.generate_api_key(&user, "test-key", None)?;
        assert!(!api_key.is_empty());

        // Validate API key
        let claims = token_manager.validate_api_key(&api_key)?;
        assert_eq!(claims.username, user.username);
        assert_eq!(claims.custom.get("api_key_name"), Some(&"test-key".to_string()));
        assert_eq!(claims.custom.get("token_type"), Some(&"api_key".to_string()));

        Ok(())
    }

    #[test]
    fn test_user_info_extraction() -> Result<()> {
        let config = TokenConfig::default();
        let token_manager = TokenManager::new(config);
        let user = create_test_user();

        let access_token = token_manager.generate_access_token(&user)?;
        let claims = token_manager.validate_access_token(&access_token)?;
        let user_info = token_manager.extract_user_info(&claims)?;

        assert_eq!(user_info.id, user.id);
        assert_eq!(user_info.username, user.username);
        assert_eq!(user_info.role, user.role);

        Ok(())
    }
}