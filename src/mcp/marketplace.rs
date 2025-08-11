//! MCP Marketplace Integration

use serde::{Deserialize, Serialize};
use anyhow::Result;
use tracing::info;

/// MCP marketplace entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpMarketplaceEntry {
    /// Server name
    pub name: String,
    
    /// Server description
    pub description: String,
    
    /// Author/organization
    pub author: String,
    
    /// Version
    pub version: String,
    
    /// Category
    pub category: String,
    
    /// Installation command
    pub command: String,
    
    /// Command arguments
    pub args: Vec<String>,
    
    /// Required environment variables
    pub env_vars: Vec<String>,
    
    /// Supported platforms
    pub platforms: Vec<String>,
    
    /// Requires API key
    pub requires_api_key: bool,
    
    /// API key instructions
    pub api_key_instructions: String,
    
    /// Installation URL
    pub installation_url: String,
    
    /// Documentation URL
    pub documentation_url: String,
    
    /// User rating (0-5)
    pub rating: f32,
    
    /// Number of downloads
    pub downloads: u64,
}

/// MCP marketplace service
pub struct McpMarketplace {
    /// Cached marketplace entries
    entries: Vec<McpMarketplaceEntry>,
}

impl McpMarketplace {
    /// Create a new marketplace service
    pub fn new() -> Self {
        Self {
            entries: Self::default_entries(),
        }
    }
    
    /// Get default marketplace entries
    fn default_entries() -> Vec<McpMarketplaceEntry> {
        vec![
            McpMarketplaceEntry {
                name: "filesystem".to_string(),
                description: "File system operations with sandboxing".to_string(),
                author: "Anthropic".to_string(),
                version: "1.0.0".to_string(),
                category: "System".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-filesystem".to_string()],
                env_vars: vec![],
                platforms: vec!["macos".to_string(), "linux".to_string(), "windows".to_string()],
                requires_api_key: false,
                api_key_instructions: String::new(),
                installation_url: "https://github.com/modelcontextprotocol/servers".to_string(),
                documentation_url: "https://modelcontextprotocol.io/docs".to_string(),
                rating: 4.8,
                downloads: 10000,
            },
            McpMarketplaceEntry {
                name: "github".to_string(),
                description: "GitHub API integration for repository management".to_string(),
                author: "Anthropic".to_string(),
                version: "1.0.0".to_string(),
                category: "Development".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-github".to_string()],
                env_vars: vec!["GITHUB_TOKEN".to_string()],
                platforms: vec!["macos".to_string(), "linux".to_string(), "windows".to_string()],
                requires_api_key: true,
                api_key_instructions: "Set GITHUB_TOKEN environment variable with your GitHub Personal Access Token".to_string(),
                installation_url: "https://github.com/modelcontextprotocol/servers".to_string(),
                documentation_url: "https://modelcontextprotocol.io/docs".to_string(),
                rating: 4.9,
                downloads: 8500,
            },
            McpMarketplaceEntry {
                name: "memory".to_string(),
                description: "Knowledge graph and memory management".to_string(),
                author: "Anthropic".to_string(),
                version: "1.0.0".to_string(),
                category: "AI/ML".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-memory".to_string()],
                env_vars: vec![],
                platforms: vec!["macos".to_string(), "linux".to_string(), "windows".to_string()],
                requires_api_key: false,
                api_key_instructions: String::new(),
                installation_url: "https://github.com/modelcontextprotocol/servers".to_string(),
                documentation_url: "https://modelcontextprotocol.io/docs".to_string(),
                rating: 4.7,
                downloads: 6000,
            },
        ]
    }
    
    /// Refresh marketplace data
    pub async fn refresh(&mut self) -> Result<()> {
        info!("Refreshing MCP marketplace data");
        // In a real implementation, this would fetch from a remote API
        // For now, we use the default entries
        self.entries = Self::default_entries();
        Ok(())
    }
    
    /// Search marketplace entries
    pub fn search(&self, query: &str) -> Vec<McpMarketplaceEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .filter(|entry| {
                entry.name.to_lowercase().contains(&query_lower)
                    || entry.description.to_lowercase().contains(&query_lower)
                    || entry.category.to_lowercase().contains(&query_lower)
            })
            .cloned()
            .collect()
    }
    
    /// Get entries by category
    pub fn get_by_category(&self, category: &str) -> Vec<McpMarketplaceEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.category == category)
            .cloned()
            .collect()
    }
    
    /// Get all marketplace entries
    pub fn get_all(&self) -> Vec<McpMarketplaceEntry> {
        self.entries.clone()
    }
    
    /// Get available categories
    pub fn get_categories(&self) -> Vec<String> {
        let mut categories: Vec<String> = self.entries
            .iter()
            .map(|e| e.category.clone())
            .collect();
        categories.dedup();
        categories.sort();
        categories
    }
}