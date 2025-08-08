//! Authentication Providers
//!
//! This module implements various authentication providers including
//! OAuth2 providers for GitHub, Twitter, and other services.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use rand::distributions::Alphanumeric;
use reqwest::Client;
use serde::{Deserialize, Serialize};
// use tracing::debug; // Unused import - commented out to reduce warnings

use crate::config::{GitHubConfig, XTwitterConfig};

/// OAuth provider trait
#[async_trait]
pub trait OAuthProvider: Send + Sync {
    /// Get authorization URL for OAuth flow
    async fn get_auth_url(&self, state: &str) -> Result<String>;

    /// Exchange authorization code for access token
    async fn exchange_code(&self, code: &str, state: &str) -> Result<OAuthToken>;

    /// Get user info using access token
    async fn get_user_info(&self, token: &OAuthToken) -> Result<OAuthUserInfo>;

    /// Refresh access token
    async fn refresh_token(&self, refresh_token: &str) -> Result<OAuthToken>;
}

/// OAuth token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthToken {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub scope: Option<String>,
}

/// OAuth user information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthUserInfo {
    pub id: String,
    pub username: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub avatar_url: Option<String>,
    pub provider: String,
}

/// GitHub OAuth provider
pub struct GitHubOAuthProvider {
    client: Client,
    config: GitHubOAuthConfig,
}

#[derive(Debug, Clone)]
struct GitHubOAuthConfig {
    client_id: String,
    client_secret: String,
    redirect_uri: String,
}

impl GitHubOAuthProvider {
    pub fn new(github_config: GitHubConfig) -> Result<Self> {
        let config = GitHubOAuthConfig {
            client_id: std::env::var("GITHUB_OAUTH_CLIENT_ID")
                .context("GITHUB_OAUTH_CLIENT_ID not set")?,
            client_secret: std::env::var("GITHUB_OAUTH_CLIENT_SECRET")
                .context("GITHUB_OAUTH_CLIENT_SECRET not set")?,
            redirect_uri: std::env::var("GITHUB_OAUTH_REDIRECT_URI")
                .unwrap_or_else(|_| "http://localhost:8080/auth/github/callback".to_string()),
        };

        Ok(Self {
            client: Client::new(),
            config,
        })
    }
}

#[async_trait]
impl OAuthProvider for GitHubOAuthProvider {
    async fn get_auth_url(&self, state: &str) -> Result<String> {
        let scopes = "user:email";
        let auth_url = format!(
            "https://github.com/login/oauth/authorize?client_id={}&redirect_uri={}&scope={}&state={}",
            self.config.client_id,
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(scopes),
            state
        );
        Ok(auth_url)
    }

    async fn exchange_code(&self, code: &str, _state: &str) -> Result<OAuthToken> {
        let params = HashMap::from([
            ("client_id", self.config.client_id.as_str()),
            ("client_secret", self.config.client_secret.as_str()),
            ("code", code),
            ("redirect_uri", self.config.redirect_uri.as_str()),
        ]);

        let response = self.client
            .post("https://github.com/login/oauth/access_token")
            .header("Accept", "application/json")
            .form(&params)
            .send()
            .await
            .context("Failed to exchange code for token")?;

        #[derive(Deserialize)]
        struct GitHubTokenResponse {
            access_token: String,
            token_type: String,
            scope: String,
        }

        let token_response: GitHubTokenResponse = response
            .json()
            .await
            .context("Failed to parse token response")?;

        Ok(OAuthToken {
            access_token: token_response.access_token,
            refresh_token: None,
            token_type: token_response.token_type,
            expires_in: None,
            scope: Some(token_response.scope),
        })
    }

    async fn get_user_info(&self, token: &OAuthToken) -> Result<OAuthUserInfo> {
        let response = self.client
            .get("https://api.github.com/user")
            .header("Authorization", format!("token {}", token.access_token))
            .header("User-Agent", "Loki-AI")
            .send()
            .await
            .context("Failed to get user info")?;

        #[derive(Deserialize)]
        struct GitHubUser {
            id: u64,
            login: String,
            email: Option<String>,
            name: Option<String>,
            avatar_url: String,
        }

        let user: GitHubUser = response
            .json()
            .await
            .context("Failed to parse user info")?;

        Ok(OAuthUserInfo {
            id: user.id.to_string(),
            username: user.login,
            email: user.email,
            name: user.name,
            avatar_url: Some(user.avatar_url),
            provider: "github".to_string(),
        })
    }

    async fn refresh_token(&self, _refresh_token: &str) -> Result<OAuthToken> {
        // GitHub doesn't use refresh tokens
        Err(anyhow::anyhow!("GitHub doesn't support token refresh"))
    }
}

/// Twitter OAuth provider
pub struct TwitterOAuthProvider {
    client: Client,
    config: TwitterOAuthConfig,
}

#[derive(Debug, Clone)]
struct TwitterOAuthConfig {
    client_id: String,
    client_secret: String,
    redirect_uri: String,
}

impl TwitterOAuthProvider {
    pub fn new(twitter_config: XTwitterConfig) -> Result<Self> {
        let config = TwitterOAuthConfig {
            client_id: std::env::var("X_OAUTH_CLIENT_ID")
                .context("X_OAUTH_CLIENT_ID not set").unwrap_or_default(),
            client_secret: std::env::var("X_OAUTH_CLIENT_SECRET")
                .context("X_OAUTH_CLIENT_SECRET not set").unwrap_or_default(),
            redirect_uri: std::env::var("X_OAUTH_REDIRECT_URI")
                .unwrap_or_else(|_| "http://localhost:8080/auth/twitter/callback".to_string()),
        };

        Ok(Self {
            client: Client::new(),
            config,
        })
    }
}

#[async_trait]
impl OAuthProvider for TwitterOAuthProvider {
    async fn get_auth_url(&self, state: &str) -> Result<String> {
        let scopes = "tweet.read tweet.write users.read offline.access";
        let auth_url = format!(
            "https://twitter.com/i/oauth2/authorize?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}&code_challenge=challenge&code_challenge_method=plain",
            self.config.client_id,
            urlencoding::encode(&self.config.redirect_uri),
            urlencoding::encode(scopes),
            state
        );
        Ok(auth_url)
    }

    async fn exchange_code(&self, code: &str, _state: &str) -> Result<OAuthToken> {
        let params = HashMap::from([
            ("code", code),
            ("grant_type", "authorization_code"),
            ("client_id", self.config.client_id.as_str()),
            ("redirect_uri", self.config.redirect_uri.as_str()),
            ("code_verifier", "challenge"),
        ]);

        #[allow(deprecated)]
        let auth_header = base64::encode(format!("{}:{}", self.config.client_id, self.config.client_secret));

        let response = self.client
            .post("https://api.twitter.com/2/oauth2/token")
            .header("Authorization", format!("Basic {}", auth_header))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await
            .context("Failed to exchange code for token")?;

        #[derive(Deserialize)]
        struct TwitterTokenResponse {
            access_token: String,
            refresh_token: Option<String>,
            token_type: String,
            expires_in: Option<u64>,
            scope: Option<String>,
        }

        let token_response: TwitterTokenResponse = response
            .json()
            .await
            .context("Failed to parse token response")?;

        Ok(OAuthToken {
            access_token: token_response.access_token,
            refresh_token: token_response.refresh_token,
            token_type: token_response.token_type,
            expires_in: token_response.expires_in,
            scope: token_response.scope,
        })
    }

    async fn get_user_info(&self, token: &OAuthToken) -> Result<OAuthUserInfo> {
        let response = self.client
            .get("https://api.twitter.com/2/users/me?user.fields=profile_image_url")
            .header("Authorization", format!("Bearer {}", token.access_token))
            .send()
            .await
            .context("Failed to get user info")?;

        #[derive(Deserialize)]
        struct TwitterUserResponse {
            data: TwitterUser,
        }

        #[derive(Deserialize)]
        struct TwitterUser {
            id: String,
            username: String,
            name: String,
            profile_image_url: Option<String>,
        }

        let user_response: TwitterUserResponse = response
            .json()
            .await
            .context("Failed to parse user info")?;

        Ok(OAuthUserInfo {
            id: user_response.data.id,
            username: user_response.data.username,
            email: None, // Twitter doesn't provide email in basic scope
            name: Some(user_response.data.name),
            avatar_url: user_response.data.profile_image_url,
            provider: "twitter".to_string(),
        })
    }

    async fn refresh_token(&self, refresh_token: &str) -> Result<OAuthToken> {
        let params = HashMap::from([
            ("refresh_token", refresh_token),
            ("grant_type", "refresh_token"),
            ("client_id", self.config.client_id.as_str()),
        ]);

        #[allow(deprecated)]
        let auth_header = base64::encode(format!("{}:{}", self.config.client_id, self.config.client_secret));

        let response = self.client
            .post("https://api.twitter.com/2/oauth2/token")
            .header("Authorization", format!("Basic {}", auth_header))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await
            .context("Failed to refresh token")?;

        #[derive(Deserialize)]
        struct TwitterTokenResponse {
            access_token: String,
            refresh_token: Option<String>,
            token_type: String,
            expires_in: Option<u64>,
            scope: Option<String>,
        }

        let token_response: TwitterTokenResponse = response
            .json()
            .await
            .context("Failed to parse refresh response")?;

        Ok(OAuthToken {
            access_token: token_response.access_token,
            refresh_token: token_response.refresh_token,
            token_type: token_response.token_type,
            expires_in: token_response.expires_in,
            scope: token_response.scope,
        })
    }
}

/// Local authentication provider (username/password)
pub struct LocalAuthProvider {
    // Local auth doesn't need external configuration
}

impl LocalAuthProvider {
    pub fn new() -> Self {
        Self {}
    }
}

/// API Key authentication provider
pub struct ApiKeyProvider {
    // Configuration for API key validation
}

impl ApiKeyProvider {
    pub fn new() -> Self {
        Self {}
    }

    /// Validate API key
    pub async fn validate_api_key(&self, api_key: &str) -> Result<bool> {
        // Implement API key validation logic
        // This could check against a database of valid API keys
        // For now, we'll implement a simple validation
        Ok(!api_key.is_empty() && api_key.len() >= 32)
    }

    /// Generate new API key
    pub async fn generate_api_key(&self) -> Result<String> {
        use rand::Rng;
        let api_key: String = rand::thread_rng()
            .sample_iter(Alphanumeric)
            .take(64)
            .map(char::from)
            .collect();
        Ok(api_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oauth_token_serialization() {
        let token = OAuthToken {
            access_token: "test_token".to_string(),
            refresh_token: Some("refresh_token".to_string()),
            token_type: "Bearer".to_string(),
            expires_in: Some(3600),
            scope: Some("read write".to_string()),
        };

        let serialized = serde_json::to_string(&token).unwrap();
        let deserialized: OAuthToken = serde_json::from_str(&serialized).unwrap();

        assert_eq!(token.access_token, deserialized.access_token);
        assert_eq!(token.refresh_token, deserialized.refresh_token);
    }

    #[tokio::test]
    async fn test_api_key_provider() {
        let provider = ApiKeyProvider::new();

        // Test API key generation
        let api_key = provider.generate_api_key().await.unwrap();
        assert_eq!(api_key.len(), 64);

        // Test API key validation
        let valid = provider.validate_api_key(&api_key).await.unwrap();
        assert!(valid);

        let invalid = provider.validate_api_key("short").await.unwrap();
        assert!(!invalid);
    }
}
