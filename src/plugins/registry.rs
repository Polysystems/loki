use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{PluginMetadata, PluginCapability};

/// Plugin registry for discovering and installing plugins
pub struct PluginRegistry {
    /// Registry URL
    registry_url: String,

    /// Local cache directory
    cache_dir: PathBuf,

    /// HTTP client
    client: reqwest::Client,

    /// Cached registry data
    cache: Arc<RwLock<RegistryCache>>,
}

/// Registry cache
#[derive(Default)]
struct RegistryCache {
    /// Available plugins
    plugins: HashMap<String, RegistryPlugin>,

    /// Last update time
    last_update: Option<std::time::Instant>,

    /// Cache duration
    cache_duration: std::time::Duration,
}

/// Plugin entry in registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryPlugin {
    /// Plugin metadata
    pub metadata: PluginMetadata,

    /// Download URL
    pub download_url: String,

    /// File size (bytes)
    pub size: u64,

    /// SHA256 checksum
    pub checksum: String,

    /// Installation instructions
    pub install_instructions: Option<String>,

    /// Screenshots/demos
    pub screenshots: Vec<String>,

    /// User ratings
    pub rating: f32,

    /// Download count
    pub downloads: u64,

    /// Last updated
    pub last_updated: String,

    /// Verified publisher
    pub verified: bool,
}

/// Registry search filters
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Search query
    pub query: Option<String>,

    /// Filter by capabilities
    pub capabilities: Vec<PluginCapability>,

    /// Minimum rating
    pub min_rating: Option<f32>,

    /// Only verified publishers
    pub verified_only: bool,

    /// Sort order
    pub sort_by: SortOrder,

    /// Maximum results
    pub limit: Option<usize>,
}

/// Sort order for search results
#[derive(Debug, Clone)]
pub enum SortOrder {
    Relevance,
    Downloads,
    Rating,
    Recent,
    Name,
}

impl Default for SortOrder {
    fn default() -> Self {
        Self::Relevance
    }
}

impl PluginRegistry {
    pub fn new(registry_url: String, cache_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Loki-AI/1.0")
            .build()
            .unwrap();

        Self {
            registry_url,
            cache_dir,
            client,
            cache: Arc::new(RwLock::new(RegistryCache {
                plugins: HashMap::new(),
                last_update: None,
                cache_duration: std::time::Duration::from_secs(3600), // 1 hour
            })),
        }
    }

    /// Refresh registry data
    pub async fn refresh(&self) -> Result<()> {
        info!("Refreshing plugin registry");

        // Check if cache is still valid
        {
            let cache = self.cache.read().await;
            if let Some(last_update) = cache.last_update {
                if last_update.elapsed() < cache.cache_duration {
                    debug!("Registry cache is still valid");
                    return Ok(());
                }
            }
        }

        // Fetch registry data
        let url = format!("{}/plugins.json", self.registry_url);
        let response = self.client.get(&url)
            .send()
            .await
            .context("Failed to fetch registry data")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Registry returned error: {}",
                response.status()
            ));
        }

        let plugins: Vec<RegistryPlugin> = response.json().await
            .context("Failed to parse registry data")?;

        // Update cache
        let mut cache = self.cache.write().await;
        cache.plugins.clear();

        for plugin in plugins {
            cache.plugins.insert(plugin.metadata.id.clone(), plugin);
        }

        cache.last_update = Some(std::time::Instant::now());

        info!("Registry refreshed with {} plugins", cache.plugins.len());

        Ok(())
    }

    /// Search for plugins
    pub async fn search(&self, filters: SearchFilters) -> Result<Vec<RegistryPlugin>> {
        // Ensure registry is up to date
        self.refresh().await?;

        let cache = self.cache.read().await;
        let mut results: Vec<RegistryPlugin> = cache.plugins.values()
            .cloned()
            .collect();

        // Apply filters
        if let Some(query) = &filters.query {
            let query_lower = query.to_lowercase();
            results.retain(|p| {
                p.metadata.name.to_lowercase().contains(&query_lower) ||
                p.metadata.description.to_lowercase().contains(&query_lower) ||
                p.metadata.id.to_lowercase().contains(&query_lower)
            });
        }

        // Filter by capabilities
        if !filters.capabilities.is_empty() {
            results.retain(|p| {
                filters.capabilities.iter().all(|cap| {
                    p.metadata.capabilities.contains(cap)
                })
            });
        }

        // Filter by rating
        if let Some(min_rating) = filters.min_rating {
            results.retain(|p| p.rating >= min_rating);
        }

        // Filter by verified status
        if filters.verified_only {
            results.retain(|p| p.verified);
        }

        // Sort results
        match filters.sort_by {
            SortOrder::Relevance => {
                // Already sorted by relevance if query was provided
            }
            SortOrder::Downloads => {
                results.sort_by(|a, b| b.downloads.cmp(&a.downloads));
            }
            SortOrder::Rating => {
                results.sort_by(|a, b| {
            b.rating.partial_cmp(&a.rating)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
            }
            SortOrder::Recent => {
                results.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
            }
            SortOrder::Name => {
                results.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
            }
        }

        // Apply limit
        if let Some(limit) = filters.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get plugin details
    pub async fn get_plugin(&self, plugin_id: &str) -> Result<RegistryPlugin> {
        // Ensure registry is up to date
        self.refresh().await?;

        let cache = self.cache.read().await;
        cache.plugins.get(plugin_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Plugin not found in registry: {}", plugin_id))
    }

    /// Install a plugin
    pub async fn install_plugin(
        &self,
        plugin_id: &str,
        target_dir: &Path,
    ) -> Result<PathBuf> {
        info!("Installing plugin: {}", plugin_id);

        // Get plugin info
        let plugin = self.get_plugin(plugin_id).await?;

        // Create plugin directory
        let plugin_dir = target_dir.join(&plugin.metadata.id);
        tokio::fs::create_dir_all(&plugin_dir).await
            .context("Failed to create plugin directory")?;

        // Download plugin archive
        let archive_path = self.download_plugin(&plugin).await?;

        // Verify checksum
        self.verify_checksum(&archive_path, &plugin.checksum).await?;

        // Extract plugin
        self.extract_plugin(&archive_path, &plugin_dir).await?;

        // Clean up archive
        tokio::fs::remove_file(&archive_path).await?;

        info!("Plugin {} installed successfully", plugin_id);

        Ok(plugin_dir)
    }

    /// Download plugin archive
    async fn download_plugin(&self, plugin: &RegistryPlugin) -> Result<PathBuf> {
        info!("Downloading plugin from: {}", plugin.download_url);

        // Create cache directory
        tokio::fs::create_dir_all(&self.cache_dir).await?;

        // Download to temporary file
        let temp_path = self.cache_dir.join(format!("{}.tmp", plugin.metadata.id));

        let response = self.client.get(&plugin.download_url)
            .send()
            .await
            .context("Failed to download plugin")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download plugin: {}",
                response.status()
            ));
        }

        // Stream to file
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut stream = response.bytes_stream();

        use tokio::io::AsyncWriteExt;
        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Failed to read download chunk")?;
            file.write_all(&chunk).await
                .context("Failed to write plugin data")?;
        }

        file.flush().await?;

        Ok(temp_path)
    }

    /// Verify plugin checksum
    async fn verify_checksum(&self, path: &Path, expected: &str) -> Result<()> {
        use sha2::{Sha256, Digest};

        let data = tokio::fs::read(path).await
            .context("Failed to read plugin file")?;

        let mut hasher = Sha256::new();
        hasher.update(&data);
        let checksum = format!("{:?}", hasher.finalize());

        if checksum != expected {
            return Err(anyhow::anyhow!(
                "Checksum mismatch: expected {}, got {}",
                expected,
                checksum
            ));
        }

        Ok(())
    }

    /// Extract plugin archive
    async fn extract_plugin(&self, archive_path: &Path, target_dir: &Path) -> Result<()> {
        info!("Extracting plugin to: {:?}", target_dir);

        // Determine archive type by extension
        let extension = archive_path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow::anyhow!("Unknown archive format"))?;

        match extension {
            "zip" => self.extract_zip(archive_path, target_dir).await,
            "tar" => self.extract_tar(archive_path, target_dir, None).await,
            "gz" | "tgz" => self.extract_tar(archive_path, target_dir, Some("gz")).await,
            "bz2" => self.extract_tar(archive_path, target_dir, Some("bz2")).await,
            "xz" => self.extract_tar(archive_path, target_dir, Some("xz")).await,
            _ => Err(anyhow::anyhow!("Unsupported archive format: {}", extension)),
        }
    }

    /// Extract ZIP archive
    async fn extract_zip(&self, archive_path: &Path, target_dir: &Path) -> Result<()> {
        use zip::ZipArchive;

        let file = std::fs::File::open(archive_path)?;
        let mut archive = ZipArchive::new(file)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = target_dir.join(file.name());

            if file.name().ends_with('/') {
                tokio::fs::create_dir_all(&outpath).await?;
            } else {
                if let Some(parent) = outpath.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }

                // Read file contents into memory first (zip crate doesn't support async)
                let mut contents = Vec::new();
                std::io::Read::read_to_end(&mut file, &mut contents)?;

                // Write asynchronously
                tokio::fs::write(&outpath, contents).await?;
            }
        }

        Ok(())
    }

    /// Extract TAR archive
    async fn extract_tar(
        &self,
        archive_path: &Path,
        target_dir: &Path,
        compression: Option<&str>,
    ) -> Result<()> {
        use tokio::process::Command;

        let mut cmd = Command::new("tar");

        // Add decompression flag if needed
        match compression {
            Some("gz") => cmd.arg("-xzf"),
            Some("bz2") => cmd.arg("-xjf"),
            Some("xz") => cmd.arg("-xJf"),
            _ => cmd.arg("-xf"),
        };

        cmd.arg(archive_path);
        cmd.arg("-C").arg(target_dir);

        let output = cmd.output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to extract archive: {}", stderr));
        }

        Ok(())
    }

    /// Update a plugin
    pub async fn update_plugin(
        &self,
        plugin_id: &str,
        current_version: &str,
        target_dir: &Path,
    ) -> Result<bool> {
        info!("Checking for updates for plugin: {}", plugin_id);

        // Get latest plugin info
        let plugin = self.get_plugin(plugin_id).await?;

        // Check if update is available
        if plugin.metadata.version == current_version {
            info!("Plugin {} is up to date", plugin_id);
            return Ok(false);
        }

        info!(
            "Update available for {}: {} -> {}",
            plugin_id,
            current_version,
            plugin.metadata.version
        );

        // Install new version
        self.install_plugin(plugin_id, target_dir).await?;

        Ok(true)
    }

    /// Get featured plugins
    pub async fn get_featured(&self) -> Result<Vec<RegistryPlugin>> {
        let filters = SearchFilters {
            verified_only: true,
            min_rating: Some(4.0),
            sort_by: SortOrder::Downloads,
            limit: Some(10),
            ..Default::default()
        };

        self.search(filters).await
    }

    /// Get recently updated plugins
    pub async fn get_recent(&self) -> Result<Vec<RegistryPlugin>> {
        let filters = SearchFilters {
            sort_by: SortOrder::Recent,
            limit: Some(10),
            ..Default::default()
        };

        self.search(filters).await
    }
}

/// Example registry entry for documentation
pub fn create_example_registry_entry() -> RegistryPlugin {
    RegistryPlugin {
        metadata: PluginMetadata {
            id: "example-plugin".to_string(),
            name: "Example Plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Loki Community".to_string(),
            description: "An example plugin demonstrating the plugin API".to_string(),
            loki_version: "0.2.0".to_string(),
            dependencies: vec![],
            capabilities: vec![
                PluginCapability::MemoryRead,
                PluginCapability::NetworkAccess,
            ],
            homepage: Some("https://example.com".to_string()),
            repository: Some("https://github.com/example/plugin".to_string()),
            license: Some("MIT".to_string()),
        },
        download_url: "https://plugins.loki.ai/example-plugin-1.0.0.zip".to_string(),
        size: 1024 * 1024, // 1 MB
        checksum: "abcdef1234567890".to_string(),
        install_instructions: Some("Extract and run 'plugin install'".to_string()),
        screenshots: vec![
            "https://plugins.loki.ai/screenshots/example-1.png".to_string(),
        ],
        rating: 4.5,
        downloads: 1000,
        last_updated: "2024-01-01T00:00:00Z".to_string(),
        verified: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_search_filters_default() {
        let filters = SearchFilters::default();
        assert!(filters.query.is_none());
        assert!(filters.capabilities.is_empty());
        assert!(!filters.verified_only);
    }

    #[test]
    fn test_example_registry_entry() {
        let entry = create_example_registry_entry();
        assert_eq!(entry.metadata.id, "example-plugin");
        assert_eq!(entry.rating, 4.5);
        assert!(entry.verified);
    }
}
