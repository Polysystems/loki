//! OAuth2 Authentication Flow for X/Twitter
//!
//! This module implements the OAuth2 authentication flow for X/Twitter API v2
//! including PKCE (Proof Key for Code Exchange) for enhanced security.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use rand::{self, Rng};

use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// OAuth2 configuration for X/Twitter
#[derive(Debug, Clone)]
pub struct OAuth2Config {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub redirect_uri: String,
    pub scopes: Vec<String>,
}

impl OAuth2Config {
    /// Create config from environment variables
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            client_id: std::env::var("X_OAUTH_CLIENT_ID").context("X_OAUTH_CLIENT_ID not set")?,
            client_secret: std::env::var("X_OAUTH_CLIENT_SECRET").ok(),
            redirect_uri: std::env::var("X_OAUTH_REDIRECT_URI")
                .unwrap_or_else(|_| "http://localhost:8080/callback".to_string()),
            scopes: vec![
                "tweet.read".to_string(),
                "tweet.write".to_string(),
                "users.read".to_string(),
                "follows.read".to_string(),
                "follows.write".to_string(),
                "offline.access".to_string(),
            ],
        })
    }
}

/// OAuth2 tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Token {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
    #[serde(skip)]
    pub expires_at: Option<SystemTime>,
}

impl OAuth2Token {
    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at { SystemTime::now() > expires_at } else { false }
    }
}

/// OAuth2 client for X/Twitter
pub struct OAuth2Client {
    client: Client,
    config: OAuth2Config,
    token: Arc<RwLock<Option<OAuth2Token>>>,
    pkce_verifier: Arc<RwLock<Option<String>>>,
}

impl OAuth2Client {
    /// Create new OAuth2 client
    pub fn new(config: OAuth2Config) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config,
            token: Arc::new(RwLock::new(None)),
            pkce_verifier: Arc::new(RwLock::new(None)),
        }
    }

    /// Generate PKCE challenge
    fn generate_pkce() -> (String, String) {
        let mut rng = rand::thread_rng();
        let code_verifier: String = (0..128)
            .map(|_| {
                let idx = rng.gen_range(0..62);
                match idx {
                    0..26 => (b'A' + idx as u8) as char,
                    26..52 => (b'a' + (idx - 26) as u8) as char,
                    _ => (b'0' + (idx - 52) as u8) as char,
                }
            })
            .collect();

        let challenge = sha256::digest(&code_verifier);
        let code_challenge = URL_SAFE_NO_PAD.encode(challenge.as_bytes());

        (code_verifier, code_challenge)
    }

    /// Generate authorization URL
    pub async fn get_authorization_url(&self, state: &str) -> Result<String> {
        let (verifier, challenge) = Self::generate_pkce();

        // Store verifier for later use
        *self.pkce_verifier.write().await = Some(verifier);

        let scope_string = self.config.scopes.join(" ");
        let params = vec![
            ("response_type", "code"),
            ("client_id", &self.config.client_id),
            ("redirect_uri", &self.config.redirect_uri),
            ("scope", &scope_string),
            ("state", state),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
        ];

        let url = format!(
            "https://twitter.com/i/oauth2/authorize?{}",
            serde_urlencoded::to_string(&params)?
        );

        Ok(url)
    }

    /// Exchange authorization code for tokens
    pub async fn exchange_code(&self, code: &str) -> Result<OAuth2Token> {
        let verifier =
            self.pkce_verifier.read().await.as_ref().context("PKCE verifier not found")?.clone();

        let mut params = HashMap::new();
        params.insert("grant_type", "authorization_code");
        params.insert("code", code);
        params.insert("redirect_uri", &self.config.redirect_uri);
        params.insert("code_verifier", &verifier);

        let mut request = self.client.post("https://api.twitter.com/2/oauth2/token").form(&params);

        // Add client credentials if available
        if let Some(client_secret) = &self.config.client_secret {
            request = request.basic_auth(&self.config.client_id, Some(client_secret));
        } else {
            params.insert("client_id", &self.config.client_id);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Token exchange failed: {}", error_text);
        }

        let mut token: OAuth2Token = response.json().await?;

        // Calculate expiration time
        if let Some(expires_in) = token.expires_in {
            token.expires_at = Some(SystemTime::now() + Duration::from_secs(expires_in));
        }

        // Store token
        *self.token.write().await = Some(token.clone());

        // Clear PKCE verifier
        *self.pkce_verifier.write().await = None;

        info!("Successfully obtained OAuth2 token");
        Ok(token)
    }

    /// Refresh access token
    pub async fn refresh_token(&self) -> Result<OAuth2Token> {
        let current_token =
            self.token.read().await.as_ref().context("No token to refresh")?.clone();

        let refresh_token = current_token.refresh_token.context("No refresh token available")?;

        let mut params = HashMap::new();
        params.insert("grant_type", "refresh_token");
        params.insert("refresh_token", &refresh_token);

        let mut request = self.client.post("https://api.twitter.com/2/oauth2/token").form(&params);

        // Add client credentials
        if let Some(client_secret) = &self.config.client_secret {
            request = request.basic_auth(&self.config.client_id, Some(client_secret));
        } else {
            params.insert("client_id", &self.config.client_id);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Token refresh failed: {}", error_text);
        }

        let mut token: OAuth2Token = response.json().await?;

        // Calculate expiration time
        if let Some(expires_in) = token.expires_in {
            token.expires_at = Some(SystemTime::now() + Duration::from_secs(expires_in));
        }

        // Store new token
        *self.token.write().await = Some(token.clone());

        info!("Successfully refreshed OAuth2 token");
        Ok(token)
    }

    /// Get current token, refreshing if necessary
    pub async fn get_valid_token(&self) -> Result<OAuth2Token> {
        let token = self.token.read().await.clone();

        match token {
            Some(t) => {
                if t.is_expired() {
                    debug!("Token expired, refreshing...");
                    self.refresh_token().await
                } else {
                    Ok(t)
                }
            }
            None => Err(anyhow::anyhow!("No OAuth2 token available")),
        }
    }

    /// Revoke token
    pub async fn revoke_token(&self) -> Result<()> {
        let token = self.token.read().await.as_ref().context("No token to revoke")?.clone();

        let params =
            vec![("token", token.access_token.as_str()), ("token_type_hint", "access_token")];

        let mut request = self.client.post("https://api.twitter.com/2/oauth2/revoke").form(&params);

        if let Some(client_secret) = &self.config.client_secret {
            request = request.basic_auth(&self.config.client_id, Some(client_secret));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            warn!("Token revocation may have failed: {}", response.status());
        }

        // Clear stored token
        *self.token.write().await = None;

        info!("OAuth2 token revoked");
        Ok(())
    }

    /// Save token to file
    pub async fn save_token(&self, path: &std::path::Path) -> Result<()> {
        let token = self.token.read().await.as_ref().context("No token to save")?.clone();

        let json = serde_json::to_string_pretty(&token)?;
        tokio::fs::write(path, json).await?;

        debug!("Token saved to {:?}", path);
        Ok(())
    }

    /// Load token from file
    pub async fn load_token(&self, path: &std::path::Path) -> Result<()> {
        let json = tokio::fs::read_to_string(path).await?;
        let mut token: OAuth2Token = serde_json::from_str(&json)?;

        // Recalculate expiration time
        if let Some(expires_in) = token.expires_in {
            // This is approximate - better to store issued_at time
            token.expires_at = Some(SystemTime::now() + Duration::from_secs(expires_in / 2));
        }

        *self.token.write().await = Some(token);

        debug!("Token loaded from {:?}", path);
        Ok(())
    }
}

/// SHA256 module for PKCE
mod sha256 {
    use sha2::{Digest, Sha256};

    pub fn digest(input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:?}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkce_generation() {
        let (verifier, challenge) = OAuth2Client::generate_pkce();
        assert_eq!(verifier.len(), 128);
        assert!(!challenge.is_empty());
    }

    #[test]
    fn test_token_expiration() {
        let mut token = OAuth2Token {
            access_token: "test".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: Some(3600),
            refresh_token: None,
            scope: None,
            expires_at: Some(SystemTime::now() - Duration::from_secs(1)),
        };

        assert!(token.is_expired());

        token.expires_at = Some(SystemTime::now() + Duration::from_secs(3600));
        assert!(!token.is_expired());
    }
}
