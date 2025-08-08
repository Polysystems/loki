//! Secure Environment Variable Handling
//!
//! This module provides secure methods for accessing environment variables
//! without relying on dotenv files which can expose secrets.

use std::env;
use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use tracing::{warn, info, debug};

/// Secure environment variable provider that validates and sanitizes access
pub struct SecureEnv {
    /// Cache of validated environment variables
    cache: HashMap<String, String>,
    /// Whether to allow fallback to .env files (disabled by default)
    allow_dotenv: bool,
}

impl SecureEnv {
    /// Create a new secure environment provider
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            allow_dotenv: false,
        }
    }

    /// Get an environment variable with validation
    pub fn get(&mut self, key: &str) -> Option<String> {
        // Check cache first
        if let Some(value) = self.cache.get(key) {
            return Some(value.clone());
        }

        // Validate key format
        if !Self::is_valid_key(key) {
            warn!("Invalid environment variable key format: {}", key);
            return None;
        }

        // Get from environment
        match env::var(key) {
            Ok(value) => {
                // Validate value
                if Self::is_valid_value(&value) {
                    debug!("Loaded {} from environment", key);
                    self.cache.insert(key.to_string(), value.clone());
                    Some(value)
                } else {
                    warn!("Invalid value format for {}", key);
                    None
                }
            }
            Err(_) => {
                debug!("{} not found in environment", key);
                None
            }
        }
    }

    /// Get a required environment variable
    pub fn require(&mut self, key: &str) -> Result<String> {
        self.get(key)
            .with_context(|| format!("Required environment variable {} not found", key))
    }

    /// Check if a key is valid (alphanumeric with underscores)
    fn is_valid_key(key: &str) -> bool {
        !key.is_empty() && key.chars().all(|c| c.is_alphanumeric() || c == '_')
    }

    /// Validate value doesn't contain suspicious patterns
    fn is_valid_value(value: &str) -> bool {
        // Check for common injection patterns
        !value.contains('\0') && // No null bytes
        !value.contains("${") && // No variable expansion
        !value.contains("$(") && // No command substitution
        value.len() < 10_000 // Reasonable length limit
    }

    /// Load API keys with validation
    pub fn load_api_keys(&mut self) -> HashMap<String, Option<String>> {
        let api_keys = vec![
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN",
            "DEEPSEEK_API_KEY",
            "MISTRAL_API_KEY",
            "CODESTRAL_API_KEY",
            "GEMINI_API_KEY",
            "GROK_API_KEY",
            "COHERE_API_KEY",
            "PERPLEXITY_API_KEY",
            "X_API_KEY",
            "X_API_SECRET",
            "X_ACCESS_TOKEN",
            "X_ACCESS_TOKEN_SECRET",
            "X_BEARER_TOKEN",
        ];

        let mut loaded = HashMap::new();
        for key in api_keys {
            loaded.insert(key.to_string(), self.get(key));
        }

        // Log summary without exposing values
        let configured = loaded.values().filter(|v| v.is_some()).count();
        if configured > 0 {
            info!("Loaded {} API keys from secure environment", configured);
        }

        loaded
    }

    /// Validate that required keys are present
    pub fn validate_required(&mut self, required: &[&str]) -> Result<()> {
        let mut missing = Vec::new();
        
        for key in required {
            if self.get(key).is_none() {
                missing.push(*key);
            }
        }

        if !missing.is_empty() {
            bail!(
                "Missing required environment variables: {}. \
                Please set these in your shell environment, not in .env files.",
                missing.join(", ")
            );
        }

        Ok(())
    }

    /// Clear the cache (useful for testing)
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for SecureEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to check if running in a secure environment
pub fn check_secure_environment() -> Result<()> {
    // Check for .env file presence
    if std::path::Path::new(".env").exists() {
        warn!(
            "⚠️  Found .env file in project directory. \
            For production use, please set environment variables in your shell instead."
        );
    }

    // Check for common insecure practices
    if env::var("NODE_ENV").as_deref() == Ok("production") {
        // In production, we should not use .env files
        if std::path::Path::new(".env").exists() {
            bail!(
                "Security Error: .env file found in production environment. \
                Please use proper secret management."
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_keys() {
        assert!(SecureEnv::is_valid_key("OPENAI_API_KEY"));
        assert!(SecureEnv::is_valid_key("TEST_123"));
        assert!(!SecureEnv::is_valid_key(""));
        assert!(!SecureEnv::is_valid_key("TEST-KEY"));
        assert!(!SecureEnv::is_valid_key("TEST KEY"));
    }

    #[test]
    fn test_valid_values() {
        assert!(SecureEnv::is_valid_value("sk-1234567890"));
        assert!(SecureEnv::is_valid_value("normal_value_123"));
        assert!(!SecureEnv::is_valid_value("value\0with\0nulls"));
        assert!(!SecureEnv::is_valid_value("${INJECTED}"));
        assert!(!SecureEnv::is_valid_value("$(command)"));
    }

    #[test]
    fn test_cache() {
        let mut env = SecureEnv::new();
        
        // Set a test variable
        env::set_var("TEST_CACHE_VAR", "test_value");
        
        // First access should load from environment
        assert_eq!(env.get("TEST_CACHE_VAR"), Some("test_value".to_string()));
        
        // Second access should use cache
        assert_eq!(env.get("TEST_CACHE_VAR"), Some("test_value".to_string()));
        
        // Clear cache
        env.clear_cache();
        
        // Should load from environment again
        assert_eq!(env.get("TEST_CACHE_VAR"), Some("test_value".to_string()));
        
        // Clean up
        env::remove_var("TEST_CACHE_VAR");
    }
}