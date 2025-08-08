// 🏪 Loki AI Plugin Marketplace & Discovery System
// Advanced plugin discovery, installation, and management

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, error, info, warn};
use url::Url;

use crate::plugins::{PluginCapability, PluginMetadata};

/// Plugin marketplace client for discovering and installing plugins
pub struct PluginMarketplace {
    client: Client,
    registry_url: Url,
    local_cache: PathBuf,
    auth_token: Option<String>,
    trusted_publishers: Vec<String>,
}

/// Plugin marketplace entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplacePlugin {
    pub metadata: PluginMetadata,
    pub download_url: String,
    pub download_count: u64,
    pub rating: f32,
    pub reviews: Vec<PluginReview>,
    pub screenshots: Vec<String>,
    pub tags: Vec<String>,
    pub publisher: PublisherInfo,
    pub verified: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub size_bytes: u64,
    pub checksum: String,
}

/// Plugin review
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginReview {
    pub user: String,
    pub rating: u8,
    pub comment: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub verified_download: bool,
}

/// Publisher information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublisherInfo {
    pub name: String,
    pub email: String,
    pub website: Option<String>,
    pub verified: bool,
    pub trust_score: f32,
    pub published_plugins: u32,
}

/// Plugin search filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub query: Option<String>,
    pub category: Option<PluginCategory>,
    pub capabilities: Vec<PluginCapability>,
    pub min_rating: Option<f32>,
    pub verified_only: bool,
    pub compatible_version: Option<String>,
    pub tags: Vec<String>,
    pub sort_by: SortBy,
    pub limit: usize,
    pub offset: usize,
}

/// Plugin categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCategory {
    AI,
    Cognitive,
    Memory,
    Social,
    Productivity,
    Development,
    Communication,
    Security,
    Monitoring,
    Integration,
    Custom(String),
}

/// Sort criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortBy {
    Relevance,
    Rating,
    Downloads,
    Recent,
    Name,
    Updated,
}

/// Plugin installation result
#[derive(Debug, Clone)]
pub struct InstallationResult {
    pub plugin_id: String,
    pub success: bool,
    pub message: String,
    pub installed_version: Option<String>,
    pub dependencies_installed: Vec<String>,
}

impl Default for SortBy {
    fn default() -> Self {
        SortBy::Relevance
    }
}

impl PluginMarketplace {
    /// Create a new marketplace client
    pub fn new<U: Into<Url>>(registry_url: U, cache_dir: PathBuf) -> Result<Self> {
        let client = Client::builder()
            .user_agent("loki-ai/0.2.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            registry_url: registry_url.into(),
            local_cache: cache_dir,
            auth_token: None,
            trusted_publishers: vec![
                "loki-ai".to_string(),
                "official".to_string(),
                "community".to_string(),
            ],
        })
    }

    /// Set authentication token for premium features
    pub fn set_auth_token(&mut self, token: String) {
        self.auth_token = Some(token);
    }

    /// Search for plugins in the marketplace
    pub async fn search(&self, filters: SearchFilters) -> Result<Vec<MarketplacePlugin>> {
        let mut url = self.registry_url.clone();
        url.set_path("/api/v1/plugins/search");

        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(token) = &self.auth_token {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", token).parse().unwrap(),
            );
        }

        let response = self
            .client
            .post(url)
            .headers(headers)
            .json(&filters)
            .send()
            .await
            .context("Failed to search plugins")?;

        if response.status().is_success() {
            let plugins: Vec<MarketplacePlugin> = response
                .json()
                .await
                .context("Failed to parse search results")?;

            info!("Found {} plugins matching search criteria", plugins.len());
            Ok(plugins)
        } else {
            let error_msg = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            error!("Plugin search failed: {}", error_msg);
            anyhow::bail!("Search failed: {}", error_msg);
        }
    }

    /// Get featured/recommended plugins
    pub async fn get_featured(&self) -> Result<Vec<MarketplacePlugin>> {
        let mut url = self.registry_url.clone();
        url.set_path("/api/v1/plugins/featured");

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to get featured plugins")?;

        let plugins: Vec<MarketplacePlugin> = response
            .json()
            .await
            .context("Failed to parse featured plugins")?;

        info!("Retrieved {} featured plugins", plugins.len());
        Ok(plugins)
    }

    /// Get plugin details by ID
    pub async fn get_plugin(&self, plugin_id: &str) -> Result<MarketplacePlugin> {
        let mut url = self.registry_url.clone();
        url.set_path(&format!("/api/v1/plugins/{}", plugin_id));

        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to get plugin details")?;

        let plugin: MarketplacePlugin = response
            .json()
            .await
            .context("Failed to parse plugin details")?;

        debug!("Retrieved plugin details for {}", plugin_id);
        Ok(plugin)
    }

    /// Install a plugin from the marketplace
    pub async fn install_plugin(
        &self,
        plugin_id: &str,
        target_dir: &Path,
        force: bool,
    ) -> Result<InstallationResult> {
        info!("Installing plugin: {}", plugin_id);

        // Get plugin metadata
        let plugin = self.get_plugin(plugin_id).await?;

        // Verify plugin integrity and security
        if !self.verify_plugin_security(&plugin).await? {
            return Ok(InstallationResult {
                plugin_id: plugin_id.to_string(),
                success: false,
                message: "Plugin failed security verification".to_string(),
                installed_version: None,
                dependencies_installed: vec![],
            });
        }

        // Check if already installed
        let plugin_dir = target_dir.join(&plugin.metadata.id);
        if plugin_dir.exists() && !force {
            return Ok(InstallationResult {
                plugin_id: plugin_id.to_string(),
                success: false,
                message: "Plugin already installed (use --force to overwrite)".to_string(),
                installed_version: None,
                dependencies_installed: vec![],
            });
        }

        // Download plugin
        let plugin_path = self.download_plugin(&plugin).await?;

        // Install dependencies first
        let mut dependencies_installed = Vec::new();
        for dep in &plugin.metadata.dependencies {
            if !dep.optional {
                match Box::pin(self.install_plugin(&dep.id, target_dir, false)).await {
                    Ok(result) if result.success => {
                        dependencies_installed.push(dep.id.clone());
                    }
                    Ok(_) => {
                        warn!("Failed to install dependency: {}", dep.id);
                    }
                    Err(e) => {
                        error!("Dependency installation error: {}", e);
                        return Ok(InstallationResult {
                            plugin_id: plugin_id.to_string(),
                            success: false,
                            message: format!("Failed to install dependency: {}", dep.id),
                            installed_version: None,
                            dependencies_installed,
                        });
                    }
                }
            }
        }

        // Extract plugin to target directory
        fs::create_dir_all(&plugin_dir).await?;
        self.extract_plugin(&plugin_path, &plugin_dir).await?;

        // Verify installation
        let metadata_path = plugin_dir.join("plugin.toml");
        if metadata_path.exists() {
            info!("Successfully installed plugin: {}", plugin_id);
            Ok(InstallationResult {
                plugin_id: plugin_id.to_string(),
                success: true,
                message: "Plugin installed successfully".to_string(),
                installed_version: Some(plugin.metadata.version),
                dependencies_installed,
            })
        } else {
            error!("Plugin installation verification failed");
            Ok(InstallationResult {
                plugin_id: plugin_id.to_string(),
                success: false,
                message: "Installation verification failed".to_string(),
                installed_version: None,
                dependencies_installed,
            })
        }
    }

    /// Update a plugin to the latest version
    pub async fn update_plugin(
        &self,
        plugin_id: &str,
        target_dir: &Path,
    ) -> Result<InstallationResult> {
        info!("Updating plugin: {}", plugin_id);

        // Check current version
        let plugin_dir = target_dir.join(plugin_id);
        let metadata_path = plugin_dir.join("plugin.toml");

        let current_version = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path).await?;
            // Parse TOML to get current version
            // This is simplified - you'd use a proper TOML parser
            content
                .lines()
                .find(|line| line.starts_with("version"))
                .and_then(|line| line.split('=').nth(1))
                .map(|v| v.trim().trim_matches('"'))
                .unwrap_or("unknown")
                .to_string()
        } else {
            return Ok(InstallationResult {
                plugin_id: plugin_id.to_string(),
                success: false,
                message: "Plugin not currently installed".to_string(),
                installed_version: None,
                dependencies_installed: vec![],
            });
        };

        // Get latest version from marketplace
        let latest_plugin = self.get_plugin(plugin_id).await?;

        if current_version == latest_plugin.metadata.version {
            return Ok(InstallationResult {
                plugin_id: plugin_id.to_string(),
                success: true,
                message: "Plugin is already up to date".to_string(),
                installed_version: Some(current_version),
                dependencies_installed: vec![],
            });
        }

        // Install latest version (force overwrite)
        self.install_plugin(plugin_id, target_dir, true).await
    }

    /// List installed plugins and check for updates
    pub async fn check_updates(&self, plugins_dir: &Path) -> Result<Vec<UpdateInfo>> {
        let mut updates = Vec::new();

        // Scan installed plugins
        let mut entries = fs::read_dir(plugins_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                let plugin_dir = entry.path();
                let metadata_path = plugin_dir.join("plugin.toml");

                if metadata_path.exists() {
                    if let Ok(update_info) = self.check_plugin_update(&plugin_dir).await {
                        updates.push(update_info);
                    }
                }
            }
        }

        info!("Checked {} plugins for updates", updates.len());
        Ok(updates)
    }

    /// Publish a plugin to the marketplace
    pub async fn publish_plugin(
        &self,
        plugin_path: &Path,
        publisher_token: &str,
    ) -> Result<PublishResult> {
        info!("Publishing plugin: {}", plugin_path.display());

        // Validate plugin structure
        self.validate_plugin_structure(plugin_path).await?;

        // Create plugin package
        let package_path = self.create_plugin_package(plugin_path).await?;

        // Upload to marketplace
        let mut url = self.registry_url.clone();
        url.set_path("/api/v1/plugins/publish");

        let form = reqwest::multipart::Form::new()
            .file("plugin", &package_path)
            .await?;

        let response = self
            .client
            .post(url)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {}", publisher_token))
            .multipart(form)
            .send()
            .await?;

        let result: PublishResult = response.json().await?;

        info!("Plugin publication result: {:?}", result);
        Ok(result)
    }

    // Private helper methods

    async fn verify_plugin_security(&self, plugin: &MarketplacePlugin) -> Result<bool> {
        // Check if publisher is trusted
        if !self.trusted_publishers.contains(&plugin.publisher.name) && !plugin.verified {
            warn!("Plugin from untrusted publisher: {}", plugin.publisher.name);
            return Ok(false);
        }

        // Check capabilities against security policy
        for capability in &plugin.metadata.capabilities {
            if self.is_dangerous_capability(capability) && !plugin.verified {
                warn!("Plugin requests dangerous capability: {}", capability);
                return Ok(false);
            }
        }

        // Additional security checks could include:
        // - Binary analysis
        // - Sandboxing validation
        // - Code signing verification

        Ok(true)
    }

    fn is_dangerous_capability(&self, capability: &PluginCapability) -> bool {
        match capability {
            PluginCapability::FileSystemWrite |
            PluginCapability::CodeModification => true,
            PluginCapability::Custom(s) => {
                s.contains("admin") || s.contains("system")
            }
            _ => false,
        }
    }

    async fn download_plugin(&self, plugin: &MarketplacePlugin) -> Result<PathBuf> {
        let cache_path = self.local_cache.join(format!("{}-{}.tar.gz", plugin.metadata.id, plugin.metadata.version));

        if !cache_path.exists() {
            fs::create_dir_all(&self.local_cache).await?;

            let response = self.client.get(&plugin.download_url).send().await?;
            let content = response.bytes().await?;

            // Verify checksum
            let computed_checksum = sha256::digest(content.as_ref());
            if computed_checksum != plugin.checksum {
                anyhow::bail!("Plugin checksum verification failed");
            }

            fs::write(&cache_path, content).await?;
        }

        Ok(cache_path)
    }

    async fn extract_plugin(&self, archive_path: &Path, target_dir: &Path) -> Result<()> {
        // This would use a proper archive extraction library
        // For now, we'll assume it's a simple tar.gz
        debug!("Extracting plugin from {} to {}", archive_path.display(), target_dir.display());

        // Simplified implementation - would use tar/zip library
        // tar::Archive::new(GzDecoder::new(File::open(archive_path)?))
        //     .unpack(target_dir)?;

        Ok(())
    }

    async fn check_plugin_update(&self, plugin_dir: &Path) -> Result<UpdateInfo> {
        // Read current plugin metadata
        let metadata_path = plugin_dir.join("plugin.toml");
        let content = fs::read_to_string(&metadata_path).await?;

        // Parse plugin ID and version (simplified)
        let plugin_id = content
            .lines()
            .find(|line| line.starts_with("id"))
            .and_then(|line| line.split('=').nth(1))
            .map(|v| v.trim().trim_matches('"'))
            .unwrap_or("unknown")
            .to_string();

        let current_version = content
            .lines()
            .find(|line| line.starts_with("version"))
            .and_then(|line| line.split('=').nth(1))
            .map(|v| v.trim().trim_matches('"'))
            .unwrap_or("unknown")
            .to_string();

        // Check marketplace for latest version
        match self.get_plugin(&plugin_id).await {
            Ok(latest) => {
                let latest_version = latest.metadata.version.clone();
                let update_available = current_version != latest_version;
                Ok(UpdateInfo {
                    plugin_id,
                    current_version,
                    latest_version,
                    update_available,
                    security_update: false, // Would check security advisories
                })
            }
            Err(_) => {
                Ok(UpdateInfo {
                    plugin_id,
                    current_version,
                    latest_version: "unknown".to_string(),
                    update_available: false,
                    security_update: false,
                })
            }
        }
    }

    async fn validate_plugin_structure(&self, plugin_path: &Path) -> Result<()> {
        // Validate plugin has required files
        let metadata_path = plugin_path.join("plugin.toml");
        if !metadata_path.exists() {
            anyhow::bail!("Plugin missing plugin.toml");
        }

        let src_path = plugin_path.join("src");
        if !src_path.exists() {
            anyhow::bail!("Plugin missing src directory");
        }

        // Additional validation could include:
        // - TOML parsing
        // - Rust compilation check
        // - Security scanning

        Ok(())
    }

    async fn create_plugin_package(&self, plugin_path: &Path) -> Result<PathBuf> {
        let package_path = self.local_cache.join("temp_package.tar.gz");

        // Create tar.gz package (simplified)
        debug!("Creating package from {}", plugin_path.display());

        // Would use tar crate to create actual package
        Ok(package_path)
    }
}

/// Plugin update information
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub plugin_id: String,
    pub current_version: String,
    pub latest_version: String,
    pub update_available: bool,
    pub security_update: bool,
}

/// Plugin publish result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResult {
    pub success: bool,
    pub plugin_id: String,
    pub version: String,
    pub message: String,
    pub review_url: Option<String>,
}

/// Marketplace API trait for different implementations
#[async_trait]
pub trait MarketplaceApi: Send + Sync {
    async fn search_plugins(&self, filters: SearchFilters) -> Result<Vec<MarketplacePlugin>>;
    async fn get_plugin(&self, id: &str) -> Result<MarketplacePlugin>;
    async fn install_plugin(&self, id: &str, target: &Path) -> Result<InstallationResult>;
    async fn publish_plugin(&self, path: &Path, token: &str) -> Result<PublishResult>;
    async fn check_updates(&self, plugins_dir: &Path) -> Result<Vec<UpdateInfo>>;
}

#[async_trait]
impl MarketplaceApi for PluginMarketplace {
    async fn search_plugins(&self, filters: SearchFilters) -> Result<Vec<MarketplacePlugin>> {
        self.search(filters).await
    }

    async fn get_plugin(&self, id: &str) -> Result<MarketplacePlugin> {
        self.get_plugin(id).await
    }

    async fn install_plugin(&self, id: &str, target: &Path) -> Result<InstallationResult> {
        self.install_plugin(id, target, false).await
    }

    async fn publish_plugin(&self, path: &Path, token: &str) -> Result<PublishResult> {
        self.publish_plugin(path, token).await
    }

    async fn check_updates(&self, plugins_dir: &Path) -> Result<Vec<UpdateInfo>> {
        self.check_updates(plugins_dir).await
    }
}

// For testing and local development
pub struct LocalMarketplace {
    local_registry: PathBuf,
}

impl LocalMarketplace {
    pub fn new(registry_path: PathBuf) -> Self {
        Self {
            local_registry: registry_path,
        }
    }
}

#[async_trait]
impl MarketplaceApi for LocalMarketplace {
    async fn search_plugins(&self, _filters: SearchFilters) -> Result<Vec<MarketplacePlugin>> {
        // Scan local registry directory
        let mut plugins = Vec::new();

        let mut entries = fs::read_dir(&self.local_registry).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                // Try to load plugin metadata
                if let Ok(metadata) = self.load_local_metadata(&entry.path()).await {
                    plugins.push(metadata);
                }
            }
        }

        Ok(plugins)
    }

    async fn get_plugin(&self, id: &str) -> Result<MarketplacePlugin> {
        let plugin_path = self.local_registry.join(id);
        self.load_local_metadata(&plugin_path).await
    }

    async fn install_plugin(&self, id: &str, target: &Path) -> Result<InstallationResult> {
        let source = self.local_registry.join(id);
        let dest = target.join(id);

        // Copy plugin directory
        if source.exists() {
            fs::create_dir_all(&dest).await?;
            // Would use proper directory copying

            Ok(InstallationResult {
                plugin_id: id.to_string(),
                success: true,
                message: "Plugin installed from local registry".to_string(),
                installed_version: Some("local".to_string()),
                dependencies_installed: vec![],
            })
        } else {
            Ok(InstallationResult {
                plugin_id: id.to_string(),
                success: false,
                message: "Plugin not found in local registry".to_string(),
                installed_version: None,
                dependencies_installed: vec![],
            })
        }
    }

    async fn publish_plugin(&self, path: &Path, _token: &str) -> Result<PublishResult> {
        // Copy to local registry
        let metadata_path = path.join("plugin.toml");
        let content = fs::read_to_string(&metadata_path).await?;

        let plugin_id = content
            .lines()
            .find(|line| line.starts_with("id"))
            .and_then(|line| line.split('=').nth(1))
            .map(|v| v.trim().trim_matches('"'))
            .unwrap_or("unknown")
            .to_string();

        let dest = self.local_registry.join(&plugin_id);
        fs::create_dir_all(&dest).await?;

        // Would copy entire directory

        Ok(PublishResult {
            success: true,
            plugin_id,
            version: "local".to_string(),
            message: "Plugin published to local registry".to_string(),
            review_url: None,
        })
    }

    async fn check_updates(&self, _plugins_dir: &Path) -> Result<Vec<UpdateInfo>> {
        // Local marketplace doesn't support update checking
        // Return empty list for local development
        Ok(vec![])
    }
}

impl LocalMarketplace {
    async fn load_local_metadata(&self, plugin_path: &Path) -> Result<MarketplacePlugin> {
        let metadata_path = plugin_path.join("plugin.toml");
        let content = fs::read_to_string(&metadata_path).await?;

        // Parse TOML (simplified - would use proper TOML parser)
        let metadata = PluginMetadata {
            id: self.extract_toml_value(&content, "id")?,
            name: self.extract_toml_value(&content, "name")?,
            version: self.extract_toml_value(&content, "version")?,
            author: self.extract_toml_value(&content, "author")?,
            description: self.extract_toml_value(&content, "description")?,
            loki_version: self.extract_toml_value(&content, "loki_version")?,
            dependencies: vec![], // Would parse properly
            capabilities: vec![], // Would parse properly
            homepage: None,
            repository: None,
            license: None,
        };

        Ok(MarketplacePlugin {
            metadata,
            download_url: plugin_path.to_string_lossy().to_string(),
            download_count: 0,
            rating: 0.0,
            reviews: vec![],
            screenshots: vec![],
            tags: vec![],
            publisher: PublisherInfo {
                name: "local".to_string(),
                email: "local@localhost".to_string(),
                website: None,
                verified: true,
                trust_score: 1.0,
                published_plugins: 1,
            },
            verified: true,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            size_bytes: 0,
            checksum: "local".to_string(),
        })
    }

    fn extract_toml_value(&self, content: &str, key: &str) -> Result<String> {
        content
            .lines()
            .find(|line| line.trim().starts_with(key))
            .and_then(|line| line.split('=').nth(1))
            .map(|v| v.trim().trim_matches('"').to_string())
            .ok_or_else(|| anyhow::anyhow!("Missing TOML key: {}", key))
    }
}
