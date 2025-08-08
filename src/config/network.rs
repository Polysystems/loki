//! Network configuration module
//! 
//! Centralizes all network-related configuration including hosts, ports, and URLs.
//! This replaces hardcoded localhost references throughout the codebase.

use std::env;
use once_cell::sync::Lazy;

/// Global network configuration
pub static NETWORK_CONFIG: Lazy<NetworkConfig> = Lazy::new(|| NetworkConfig::from_env());

/// Network configuration for all services
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    // Database connections
    pub postgres_host: String,
    pub postgres_port: u16,
    pub mysql_host: String,
    pub mysql_port: u16,
    pub redis_host: String,
    pub redis_port: u16,
    pub mongodb_host: String,
    pub mongodb_port: u16,
    
    // Model serving
    pub ollama_host: String,
    pub ollama_port: u16,
    pub model_server_host: String,
    pub model_server_port: u16,
    
    // OAuth and authentication
    pub oauth_callback_host: String,
    pub oauth_callback_port: u16,
    
    // API and web services
    pub api_host: String,
    pub api_port: u16,
    pub web_host: String,
    pub web_port: u16,
    
    // Plugin marketplace
    pub marketplace_host: String,
    pub marketplace_port: u16,
    
    // Cluster coordination
    pub cluster_host: String,
    pub cluster_port: u16,
    
    // Development mode flag
    pub development_mode: bool,
}

impl NetworkConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let development_mode = env::var("LOKI_DEV_MODE")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);
        
        // Use localhost in development, proper hosts in production
        let default_host = if development_mode {
            "localhost".to_string()
        } else {
            "0.0.0.0".to_string()
        };
        
        Self {
            // Database connections
            postgres_host: env::var("POSTGRES_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            postgres_port: env::var("POSTGRES_PORT")
                .unwrap_or_else(|_| "5432".to_string())
                .parse()
                .unwrap_or(5432),
            
            mysql_host: env::var("MYSQL_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            mysql_port: env::var("MYSQL_PORT")
                .unwrap_or_else(|_| "3306".to_string())
                .parse()
                .unwrap_or(3306),
            
            redis_host: env::var("REDIS_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            redis_port: env::var("REDIS_PORT")
                .unwrap_or_else(|_| "6379".to_string())
                .parse()
                .unwrap_or(6379),
            
            mongodb_host: env::var("MONGODB_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            mongodb_port: env::var("MONGODB_PORT")
                .unwrap_or_else(|_| "27017".to_string())
                .parse()
                .unwrap_or(27017),
            
            // Model serving
            ollama_host: env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            ollama_port: env::var("OLLAMA_PORT")
                .unwrap_or_else(|_| "11434".to_string())
                .parse()
                .unwrap_or(11434),
            
            model_server_host: env::var("MODEL_SERVER_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            model_server_port: env::var("MODEL_SERVER_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
            
            // OAuth and authentication
            oauth_callback_host: env::var("OAUTH_CALLBACK_HOST")
                .unwrap_or_else(|_| {
                    if development_mode {
                        "localhost".to_string()
                    } else {
                        env::var("PUBLIC_HOST").unwrap_or_else(|_| default_host.clone())
                    }
                }),
            oauth_callback_port: env::var("OAUTH_CALLBACK_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .unwrap_or(3000),
            
            // API and web services
            api_host: env::var("API_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            api_port: env::var("API_PORT")
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .unwrap_or(8000),
            
            web_host: env::var("WEB_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            web_port: env::var("WEB_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .unwrap_or(3000),
            
            // Plugin marketplace
            marketplace_host: env::var("MARKETPLACE_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            marketplace_port: env::var("MARKETPLACE_PORT")
                .unwrap_or_else(|_| "9000".to_string())
                .parse()
                .unwrap_or(9000),
            
            // Cluster coordination
            cluster_host: env::var("CLUSTER_HOST")
                .unwrap_or_else(|_| default_host.clone()),
            cluster_port: env::var("CLUSTER_PORT")
                .unwrap_or_else(|_| "7000".to_string())
                .parse()
                .unwrap_or(7000),
            
            development_mode,
        }
    }
    
    /// Get PostgreSQL connection URL
    pub fn postgres_url(&self, username: &str, password: &str, database: &str) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            username, password, self.postgres_host, self.postgres_port, database
        )
    }
    
    /// Get MySQL connection URL
    pub fn mysql_url(&self, username: &str, password: &str, database: &str) -> String {
        format!(
            "mysql://{}:{}@{}:{}/{}",
            username, password, self.mysql_host, self.mysql_port, database
        )
    }
    
    /// Get Redis connection URL
    pub fn redis_url(&self) -> String {
        format!("redis://{}:{}", self.redis_host, self.redis_port)
    }
    
    /// Get MongoDB connection URL
    pub fn mongodb_url(&self) -> String {
        format!("mongodb://{}:{}", self.mongodb_host, self.mongodb_port)
    }
    
    /// Get Ollama base URL
    pub fn ollama_url(&self) -> String {
        format!("http://{}:{}", self.ollama_host, self.ollama_port)
    }
    
    /// Get model server URL
    pub fn model_server_url(&self) -> String {
        format!("http://{}:{}", self.model_server_host, self.model_server_port)
    }
    
    /// Get OAuth callback URL
    pub fn oauth_callback_url(&self, path: &str) -> String {
        format!(
            "http://{}:{}/{}",
            self.oauth_callback_host, self.oauth_callback_port, path.trim_start_matches('/')
        )
    }
    
    /// Get API base URL
    pub fn api_url(&self) -> String {
        format!("http://{}:{}", self.api_host, self.api_port)
    }
    
    /// Get web server URL
    pub fn web_url(&self) -> String {
        format!("http://{}:{}", self.web_host, self.web_port)
    }
    
    /// Get marketplace URL
    pub fn marketplace_url(&self) -> String {
        format!("http://{}:{}", self.marketplace_host, self.marketplace_port)
    }
    
    /// Get cluster coordination URL
    pub fn cluster_url(&self) -> String {
        format!("http://{}:{}", self.cluster_host, self.cluster_port)
    }
}

/// Get the global network configuration
pub fn get_network_config() -> &'static NetworkConfig {
    &NETWORK_CONFIG
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_configuration() {
        let config = NetworkConfig::from_env();
        
        // In test environment, should default to safe values
        assert!(!config.postgres_host.is_empty());
        assert_eq!(config.postgres_port, 5432);
        assert_eq!(config.redis_port, 6379);
        assert_eq!(config.ollama_port, 11434);
    }
    
    #[test]
    fn test_url_generation() {
        let config = NetworkConfig::from_env();
        
        let postgres_url = config.postgres_url("user", "pass", "db");
        assert!(postgres_url.contains("postgresql://"));
        assert!(postgres_url.contains(":5432/"));
        
        let redis_url = config.redis_url();
        assert!(redis_url.starts_with("redis://"));
        
        let ollama_url = config.ollama_url();
        assert!(ollama_url.starts_with("http://"));
        assert!(ollama_url.contains(":11434"));
    }
}