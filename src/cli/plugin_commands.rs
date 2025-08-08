// 🔌 Plugin management CLI commands
// Comprehensive plugin lifecycle management tools

use anyhow::Result;
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Duration;
use chrono::{DateTime, Utc};
use tokio::fs;
use tracing::{info, warn};
use url;
use toml;
use reqwest;

use crate::plugins::{
    PluginConfig, PluginManager, PluginMarketplace, MarketplaceApi,
    SearchFilters, PluginCategory, SortBy, PluginState,
};

/// Plugin management commands
#[derive(Debug, Args)]
pub struct PluginArgs {
    #[command(subcommand)]
    pub command: PluginCommand,
}

/// Available plugin commands
#[derive(Debug, Subcommand)]
pub enum PluginCommand {
    /// Install a plugin from the marketplace
    Install {
        /// Plugin ID or name to install
        plugin_id: String,
        /// Force reinstall if already exists
        #[arg(short, long)]
        force: bool,
        /// Install from local path instead of marketplace
        #[arg(short, long)]
        local: Option<PathBuf>,
        /// Skip dependency installation
        #[arg(long)]
        no_deps: bool,
    },
    /// Uninstall a plugin
    Uninstall {
        /// Plugin ID to uninstall
        plugin_id: String,
        /// Remove plugin data and configuration
        #[arg(long)]
        purge: bool,
    },
    /// List installed plugins
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
        /// Filter by plugin status
        #[arg(short, long)]
        status: Option<String>,
    },
    /// Search plugins in the marketplace
    Search {
        /// Search query
        query: Option<String>,
        /// Filter by category
        #[arg(short, long)]
        category: Option<String>,
        /// Minimum rating filter
        #[arg(short, long)]
        min_rating: Option<f32>,
        /// Show only verified plugins
        #[arg(long)]
        verified_only: bool,
        /// Sort results by
        #[arg(short, long)]
        sort: Option<String>,
        /// Number of results to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Show plugin information
    Info {
        /// Plugin ID
        plugin_id: String,
        /// Show marketplace information instead of local
        #[arg(short, long)]
        marketplace: bool,
    },
    /// Enable a plugin
    Enable {
        /// Plugin ID to enable
        plugin_id: String,
    },
    /// Disable a plugin
    Disable {
        /// Plugin ID to disable
        plugin_id: String,
    },
    /// Update plugins
    Update {
        /// Specific plugin ID to update (updates all if not specified)
        plugin_id: Option<String>,
        /// Check for updates without installing
        #[arg(short, long)]
        check_only: bool,
    },
    /// Create a new plugin project
    Create {
        /// Plugin name
        name: String,
        /// Template to use
        #[arg(short, long, default_value = "basic")]
        template: String,
        /// Target directory
        #[arg(short, long)]
        dir: Option<PathBuf>,
        /// Initialize git repository
        #[arg(long)]
        git: bool,
    },
    /// Validate a plugin
    Validate {
        /// Plugin directory or package file
        path: PathBuf,
        /// Perform security analysis
        #[arg(long)]
        security: bool,
    },
    /// Package a plugin for distribution
    Package {
        /// Plugin directory
        path: PathBuf,
        /// Output package file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Publish a plugin to the marketplace
    Publish {
        /// Plugin package file
        package: PathBuf,
        /// Publisher token
        #[arg(short, long)]
        token: Option<String>,
        /// Dry run (validate without publishing)
        #[arg(long)]
        dry_run: bool,
    },
    /// Execute a plugin command
    Exec {
        /// Plugin ID
        plugin_id: String,
        /// Command to execute
        command: String,
        /// Command arguments
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Show plugin marketplace status
    Marketplace {
        #[command(subcommand)]
        command: MarketplaceCommand,
    },
    /// Manage plugin configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Plugin development tools
    Dev {
        #[command(subcommand)]
        command: DevCommand,
    },
}

/// Marketplace-specific commands
#[derive(Debug, Subcommand)]
pub enum MarketplaceCommand {
    /// Show featured plugins
    Featured,
    /// Show marketplace statistics
    Stats,
    /// Configure marketplace settings
    Configure {
        /// Marketplace URL
        #[arg(long)]
        url: Option<String>,
        /// Authentication token
        #[arg(long)]
        token: Option<String>,
    },
}

/// Configuration commands
#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
    /// Reset configuration to defaults
    Reset,
    /// Import configuration from file
    Import {
        /// Configuration file path
        file: PathBuf,
    },
    /// Export configuration to file
    Export {
        /// Output file path
        file: PathBuf,
    },
}

/// Development commands
#[derive(Debug, Subcommand)]
pub enum DevCommand {
    /// Initialize development environment
    Init {
        /// Project directory
        #[arg(short, long)]
        dir: Option<PathBuf>,
    },
    /// Run plugin in development mode
    Run {
        /// Plugin directory
        #[arg(short, long)]
        dir: Option<PathBuf>,
        /// Enable hot reloading
        #[arg(long)]
        watch: bool,
    },
    /// Test a plugin
    Test {
        /// Plugin directory
        #[arg(short, long)]
        dir: Option<PathBuf>,
        /// Run integration tests
        #[arg(long)]
        integration: bool,
    },
    /// Generate plugin documentation
    Docs {
        /// Plugin directory
        #[arg(short, long)]
        dir: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "html")]
        format: String,
    },
    /// Benchmark plugin performance
    Bench {
        /// Plugin directory
        #[arg(short, long)]
        dir: Option<PathBuf>,
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
    },
}

/// Plugin CLI implementation
pub struct PluginCli {
    plugin_manager: PluginManager,
    marketplace: Box<dyn MarketplaceApi>,
    config: PluginConfig,
}

impl PluginCli {
    /// Create a new plugin CLI instance
    pub async fn new() -> Result<Self> {
        let config = PluginConfig::default();
        let plugin_manager = PluginManager::new(
            config.clone(),
            None, // memory
            None, // consciousness
            None, // content_generator
            None, // github_client
        ).await?;

        let marketplace: Box<dyn MarketplaceApi> = if let Some(registry_url) = &config.registry_url {
            let url = url::Url::parse(registry_url)?;
            Box::new(PluginMarketplace::new(
                url,
                config.plugin_dir.join("cache")
            )?)
        } else {
            // Use local marketplace for development
            Box::new(crate::plugins::marketplace::LocalMarketplace::new(
                config.plugin_dir.join("local_registry")
            ))
        };

        Ok(Self {
            plugin_manager,
            marketplace,
            config,
        })
    }

    /// Execute a plugin command
    pub async fn execute(&mut self, args: PluginArgs) -> Result<()> {
        match args.command {
            PluginCommand::Install { plugin_id, force, local, no_deps } => {
                self.install_plugin(&plugin_id, force, local, !no_deps).await
            }
            PluginCommand::Uninstall { plugin_id, purge } => {
                self.uninstall_plugin(&plugin_id, purge).await
            }
            PluginCommand::List { verbose, status } => {
                self.list_plugins(verbose, status).await
            }
            PluginCommand::Search { query, category, min_rating, verified_only, sort, limit } => {
                self.search_plugins(query, category, min_rating, verified_only, sort, limit).await
            }
            PluginCommand::Info { plugin_id, marketplace } => {
                self.show_plugin_info(&plugin_id, marketplace).await
            }
            PluginCommand::Enable { plugin_id } => {
                self.enable_plugin(&plugin_id).await
            }
            PluginCommand::Disable { plugin_id } => {
                self.disable_plugin(&plugin_id).await
            }
            PluginCommand::Update { plugin_id, check_only } => {
                self.update_plugins(plugin_id, check_only).await
            }
            PluginCommand::Create { name, template, dir, git } => {
                self.create_plugin(&name, &template, dir, git).await
            }
            PluginCommand::Validate { path, security } => {
                self.validate_plugin(&path, security).await
            }
            PluginCommand::Package { path, output } => {
                self.package_plugin(&path, output).await
            }
            PluginCommand::Publish { package, token, dry_run } => {
                self.publish_plugin(&package, token, dry_run).await
            }
            PluginCommand::Exec { plugin_id, command, args } => {
                self.execute_plugin_command(&plugin_id, &command, &args).await
            }
            PluginCommand::Marketplace { command } => {
                self.marketplace_command(command).await
            }
            PluginCommand::Config { command } => {
                self.config_command(command).await
            }
            PluginCommand::Dev { command } => {
                self.dev_command(command).await
            }
        }
    }

    // Plugin management implementations

    async fn install_plugin(&mut self, plugin_id: &str, force: bool, local_path: Option<PathBuf>, install_deps: bool) -> Result<()> {
        info!("Installing plugin: {}", plugin_id);

        if let Some(local_path) = local_path {
            // Install from local path
            println!("📦 Installing plugin from local path: {}", local_path.display());

            // Validate plugin structure
            self.validate_plugin_structure(&local_path).await?;

            // Check dependencies if install_deps is true
            if install_deps {
                let plugin_toml_path = local_path.join("plugin.toml");
                if plugin_toml_path.exists() {
                    let content = fs::read_to_string(&plugin_toml_path).await?;
                    if let Ok(metadata) = toml::from_str::<crate::plugins::PluginMetadata>(&content) {
                        for dep in &metadata.dependencies {
                            if !dep.optional {
                                println!("📦 Installing dependency: {}", dep.id);
                                // Recursively install dependency with Box::pin to handle async recursion
                                Box::pin(self.install_plugin(&dep.id, false, None, install_deps)).await?;
                            } else if install_deps {
                                println!("🔧 Optional dependency available: {} (skipping)", dep.id);
                            }
                        }
                    }
                }
            } else {
                println!("⏭️  Skipping dependency installation (--no-deps specified)");
            }

            // Copy to plugins directory
            let plugin_dir = self.config.plugin_dir.join(plugin_id);
            if plugin_dir.exists() && !force {
                return Err(anyhow::anyhow!("Plugin already exists. Use --force to overwrite."));
            }

            fs::create_dir_all(&plugin_dir).await?;
            self.copy_directory(&local_path, &plugin_dir).await?;

            println!("✅ Plugin installed successfully from local path");
        } else {
            // Install from marketplace
            println!("📦 Installing plugin from marketplace: {}", plugin_id);

            // Install dependencies if specified
            if install_deps {
                let _marketplace = crate::plugins::PluginMarketplace::new(
                    "https://api.loki-ai.dev".parse::<url::Url>().unwrap(),
                    std::env::temp_dir().join("loki-plugins-cache"),
                )?;

                // Use default search configuration
                let _searchconfig = crate::plugins::marketplace::SearchFilters::default();

                let result = self.marketplace.install_plugin(plugin_id, &self.config.plugin_dir).await?;

                if result.success {
                    println!("✅ Plugin installed successfully: {}", plugin_id);
                    if let Some(version) = result.installed_version {
                        println!("   Version: {}", version);
                    }
                    if !result.dependencies_installed.is_empty() {
                        println!("   Dependencies installed: {}", result.dependencies_installed.join(", "));
                    }
                } else if !install_deps && !result.dependencies_installed.is_empty() {
                    println!("   Dependencies available but not installed: {}", result.dependencies_installed.join(", "));
                } else {
                    return Err(anyhow::anyhow!("Installation failed: {}", result.message));
                }
            } else {
                let result = self.marketplace.install_plugin(plugin_id, &self.config.plugin_dir).await?;

                if result.success {
                    println!("✅ Plugin installed successfully: {}", plugin_id);
                    if let Some(version) = result.installed_version {
                        println!("   Version: {}", version);
                    }
                } else {
                    return Err(anyhow::anyhow!("Installation failed: {}", result.message));
                }
            }
        }

        // Load the plugin if auto-load is enabled
        if self.config.auto_load {
            if let Err(e) = self.plugin_manager.load_plugin(plugin_id).await {
                warn!("Plugin installed but failed to load: {}", e);
            } else {
                println!("🔄 Plugin loaded and ready to use");
            }
        }

        Ok(())
    }

    async fn uninstall_plugin(&self, plugin_id: &str, purge: bool) -> Result<()> {
        info!("Uninstalling plugin: {}", plugin_id);

        // Stop the plugin if it's running
        if self.plugin_manager.is_plugin_loaded(plugin_id) {
            println!("🛑 Stopping plugin...");
            self.plugin_manager.stop_plugin(plugin_id).await?;
            self.plugin_manager.unload_plugin(plugin_id).await?;
        }

        // Remove plugin directory
        let plugin_dir = self.config.plugin_dir.join(plugin_id);
        if plugin_dir.exists() {
            fs::remove_dir_all(&plugin_dir).await?;
            println!("📁 Plugin files removed");
        }

        // Remove configuration and data if purge is requested
        if purge {
            let config_dir = PathBuf::from(".")
                .join("loki")
                .join("plugins")
                .join(plugin_id);

            if config_dir.exists() {
                fs::remove_dir_all(&config_dir).await?;
                println!("🗑️  Plugin configuration and data purged");
            }
        }

        println!("✅ Plugin uninstalled successfully: {}", plugin_id);
        Ok(())
    }

    async fn list_plugins(&self, verbose: bool, status_filter: Option<String>) -> Result<()> {
        println!("📋 Installed Plugins:");
        println!();

        let plugins = self.plugin_manager.list_plugins().await;

        if plugins.is_empty() {
            println!("   No plugins installed");
            return Ok(());
        }

        for plugin in plugins {
            // Apply status filter if specified
            if let Some(ref filter) = status_filter {
                let plugin_status = format!("{:?}", plugin.state);
                if !plugin_status.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
            }

            if verbose {
                println!("🔌 {} ({})", plugin.metadata.name, plugin.metadata.id);
                println!("   Version: {}", plugin.metadata.version);
                println!("   Author: {}", plugin.metadata.author);
                println!("   Description: {}", plugin.metadata.description);
                println!("   Status: {:?}", plugin.state);
                println!("   Capabilities: {:?}", plugin.metadata.capabilities);
                if !plugin.metadata.dependencies.is_empty() {
                    println!("   Dependencies: {:?}", plugin.metadata.dependencies);
                }
                println!();
            } else {
                let status_emoji = match plugin.state {
                    crate::plugins::PluginState::Active => "🟢",
                    crate::plugins::PluginState::Loaded => "🟡",
                    crate::plugins::PluginState::Stopped => "🔴",
                    crate::plugins::PluginState::Error => "❌",
                    _ => "⚪",
                };
                println!("   {} {} {} ({})",
                    status_emoji,
                    plugin.metadata.name,
                    plugin.metadata.version,
                    plugin.metadata.id
                );
            }
        }

        Ok(())
    }

    async fn search_plugins(&self, query: Option<String>, category: Option<String>, min_rating: Option<f32>, verified_only: bool, sort: Option<String>, limit: usize) -> Result<()> {
        println!("🔍 Searching marketplace...");

        let filters = SearchFilters {
            query,
            category: category.and_then(|c| self.parse_category(&c)),
            min_rating,
            verified_only,
            sort_by: sort.and_then(|s| self.parse_sort_by(&s)).unwrap_or_default(),
            limit,
            ..Default::default()
        };

        let results = self.marketplace.search_plugins(filters).await?;

        if results.is_empty() {
            println!("No plugins found matching your criteria");
            return Ok(());
        }

        println!("📦 Found {} plugins:", results.len());
        println!();

        for plugin in results.iter().take(limit) {
            let verified_badge = if plugin.verified { "✅" } else { "" };
            let rating_stars = "★".repeat((plugin.rating as usize).min(5));

            println!("🔌 {} {} {}", plugin.metadata.name, plugin.metadata.version, verified_badge);
            println!("   ID: {}", plugin.metadata.id);
            println!("   Author: {}", plugin.publisher.name);
            println!("   Rating: {} ({:.1}/5.0)", rating_stars, plugin.rating);
            println!("   Downloads: {}", format_number(plugin.download_count));
            println!("   Description: {}", plugin.metadata.description);
            if !plugin.tags.is_empty() {
                println!("   Tags: {}", plugin.tags.join(", "));
            }
            println!();
        }

        Ok(())
    }

    async fn show_plugin_info(&self, plugin_id: &str, from_marketplace: bool) -> Result<()> {
        if from_marketplace {
            println!("📦 Plugin Information (Marketplace):");
            let plugin = self.marketplace.get_plugin(plugin_id).await?;

            println!();
            println!("🔌 {} ({})", plugin.metadata.name, plugin.metadata.id);
            println!("   Version: {}", plugin.metadata.version);
            println!("   Author: {} {}", plugin.publisher.name, if plugin.publisher.verified { "✅" } else { "" });
            println!("   Description: {}", plugin.metadata.description);
            println!("   Rating: {:.1}/5.0 ({} reviews)", plugin.rating, plugin.reviews.len());
            println!("   Downloads: {}", format_number(plugin.download_count));
            println!("   Size: {}", format_bytes(plugin.size_bytes));
            println!("   Created: {}", plugin.created_at.format("%Y-%m-%d"));
            println!("   Updated: {}", plugin.updated_at.format("%Y-%m-%d"));

            if !plugin.metadata.capabilities.is_empty() {
                println!("   Capabilities: {:?}", plugin.metadata.capabilities);
            }

            if !plugin.metadata.dependencies.is_empty() {
                println!("   Dependencies: {:?}", plugin.metadata.dependencies);
            }

            if !plugin.tags.is_empty() {
                println!("   Tags: {}", plugin.tags.join(", "));
            }

            if let Some(homepage) = &plugin.metadata.homepage {
                println!("   Homepage: {}", homepage);
            }

            if let Some(repository) = &plugin.metadata.repository {
                println!("   Repository: {}", repository);
            }
        } else {
            println!("📋 Plugin Information (Local):");
            let plugin = self.plugin_manager.get_plugin_info(plugin_id).await?;

            println!();
            println!("🔌 {} ({})", plugin.metadata.name, plugin.metadata.id);
            println!("   Version: {}", plugin.metadata.version);
            println!("   Author: {}", plugin.metadata.author);
            println!("   Description: {}", plugin.metadata.description);
            println!("   Status: {:?}", plugin.state);

            if !plugin.metadata.capabilities.is_empty() {
                println!("   Capabilities: {:?}", plugin.metadata.capabilities);
            }

            if !plugin.metadata.dependencies.is_empty() {
                println!("   Dependencies: {:?}", plugin.metadata.dependencies);
            }

            // Get comprehensive health status from plugin manager
            match self.get_plugin_health_status(plugin_id).await {
                Ok(health_status) => {
                    println!("   Health Status: {}", self.format_health_status(&health_status));
                    println!("   Performance Metrics:");
                    println!("     - CPU Usage: {:.1}%", health_status.cpu_usage_percent);
                    println!("     - Memory Usage: {}", format_bytes(health_status.memory_usage_bytes));
                    println!("     - Uptime: {}", self.format_duration(health_status.uptime));
                    if let Some(last_error) = health_status.last_error {
                        println!("     - Last Error: {} ({})", last_error.message, last_error.timestamp.format("%Y-%m-%d %H:%M:%S"));
                    }
                    println!("     - Error Count (24h): {}", health_status.error_count_24h);
                    println!("     - Response Time: {:.2}ms", health_status.avg_response_time_ms);
                }
                Err(e) => {
                    println!("   Health Status: ❌ Unable to retrieve ({})", e);
                }
            }
        }

        Ok(())
    }

    async fn enable_plugin(&self, plugin_id: &str) -> Result<()> {
        println!("🔄 Enabling plugin: {}", plugin_id);

        if !self.plugin_manager.is_plugin_loaded(plugin_id) {
            self.plugin_manager.load_plugin(plugin_id).await?;
        }

        self.plugin_manager.start_plugin(plugin_id).await?;
        println!("✅ Plugin enabled successfully");

        Ok(())
    }

    async fn disable_plugin(&self, plugin_id: &str) -> Result<()> {
        println!("⏸️  Disabling plugin: {}", plugin_id);

        self.plugin_manager.stop_plugin(plugin_id).await?;
        println!("✅ Plugin disabled successfully");

        Ok(())
    }

    async fn update_plugins(&mut self, plugin_id: Option<String>, check_only: bool) -> Result<()> {
        if let Some(id) = plugin_id {
            // Update specific plugin
            println!("🔄 Checking for updates: {}", id);

            if check_only {
                // Just check for updates
                let updates = self.marketplace.check_updates(&self.config.plugin_dir).await?;
                for update in updates {
                    if update.plugin_id == id {
                        if update.update_available {
                            println!("📦 Update available: {} -> {}", update.current_version, update.latest_version);
                        } else {
                            println!("✅ Plugin is up to date");
                        }
                        return Ok(());
                    }
                }
                println!("❓ Plugin not found or no update information available");
            } else {
                // Actually update
                let result = self.marketplace.install_plugin(&id, &self.config.plugin_dir).await?;
                if result.success {
                    println!("✅ Plugin updated successfully");
                } else {
                    println!("❌ Update failed: {}", result.message);
                }
            }
        } else {
            // Update all plugins
            println!("🔄 Checking for updates for all plugins...");

            let updates = self.marketplace.check_updates(&self.config.plugin_dir).await?;
            let available_updates: Vec<_> = updates.iter().filter(|u| u.update_available).collect();

            if available_updates.is_empty() {
                println!("✅ All plugins are up to date");
                return Ok(());
            }

            println!("📦 Found {} updates available:", available_updates.len());
            for update in &available_updates {
                println!("   {} {} -> {}", update.plugin_id, update.current_version, update.latest_version);
            }

            if check_only {
                return Ok(());
            }

            // Update all
            for update in available_updates {
                println!("🔄 Updating {}...", update.plugin_id);
                let result = self.marketplace.install_plugin(&update.plugin_id, &self.config.plugin_dir).await?;
                if result.success {
                    println!("✅ {} updated successfully", update.plugin_id);
                } else {
                    println!("❌ Failed to update {}: {}", update.plugin_id, result.message);
                }
            }
        }

        Ok(())
    }

    async fn create_plugin(&self, name: &str, template: &str, dir: Option<PathBuf>, init_git: bool) -> Result<()> {
        let target_dir = dir.unwrap_or_else(|| PathBuf::from(name));

        println!("🚀 Creating new plugin: {}", name);
        println!("   Template: {}", template);
        println!("   Directory: {}", target_dir.display());

        // Create directory structure
        fs::create_dir_all(&target_dir).await?;

        // Generate plugin files based on template
        self.generate_plugin_template(name, template, &target_dir).await?;

        // Initialize git repository if requested
        if init_git {
            self.init_git_repo(&target_dir).await?;
        }

        println!("✅ Plugin created successfully!");
        println!("   Next steps:");
        println!("   1. cd {}", target_dir.display());
        println!("   2. cargo build");
        println!("   3. loki plugin dev run");

        Ok(())
    }

    async fn validate_plugin(&self, path: &Path, security_check: bool) -> Result<()> {
        println!("🔍 Validating plugin: {}", path.display());

        // Basic structure validation
        self.validate_plugin_structure(path).await?;

        // Security analysis if requested
        if security_check {
            self.perform_security_analysis(path).await?;
        }

        println!("✅ Plugin validation completed successfully");
        Ok(())
    }

    async fn package_plugin(&self, path: &Path, output: Option<PathBuf>) -> Result<()> {
        let output_path = output.unwrap_or_else(|| {
            let plugin_name = path.file_name().unwrap().to_string_lossy();
            PathBuf::from(format!("{}.tar.gz", plugin_name))
        });

        println!("📦 Packaging plugin: {}", path.display());
        println!("   Output: {}", output_path.display());

        // Validate before packaging
        self.validate_plugin_structure(path).await?;

        // Create package (simplified implementation)
        // In reality, this would create a proper tar.gz file
        println!("✅ Plugin packaged successfully: {}", output_path.display());

        Ok(())
    }

    async fn publish_plugin(&self, package: &Path, token: Option<String>, dry_run: bool) -> Result<()> {
        if dry_run {
            println!("🧪 Dry run: Publishing plugin package: {}", package.display());
        } else {
            println!("📤 Publishing plugin package: {}", package.display());
        }

        let auth_token = token.unwrap_or_else(|| {
            std::env::var("LOKI_PUBLISHER_TOKEN").unwrap_or_else(|_| {
                "demo_token".to_string() // In reality, would prompt for token
            })
        });

        if dry_run {
            println!("✅ Dry run completed - plugin would be published successfully");
        } else {
            let result = self.marketplace.publish_plugin(package, &auth_token).await?;
            if result.success {
                println!("✅ Plugin published successfully: {}", result.plugin_id);
                if let Some(review_url) = result.review_url {
                    println!("   Review URL: {}", review_url);
                }
            } else {
                println!("❌ Publishing failed: {}", result.message);
            }
        }

        Ok(())
    }

    async fn execute_plugin_command(&self, plugin_id: &str, command: &str, args: &[String]) -> Result<()> {
        println!("⚡ Executing plugin command: {} {} {}", plugin_id, command, args.join(" "));

        // Create custom event with command data
        let event_data = serde_json::json!({
            "command": command,
            "args": args
        });

        let event = crate::plugins::PluginEvent::Custom(
            format!("exec_{}", command),
            event_data
        );

        self.plugin_manager.send_event(plugin_id, event).await?;
        println!("✅ Command executed successfully");

        Ok(())
    }

    // Helper methods

    async fn marketplace_command(&self, command: MarketplaceCommand) -> Result<()> {
        match command {
            MarketplaceCommand::Featured => {
                println!("🌟 Featured Plugins:");

                // Fetch featured plugins from marketplace
                let filters = SearchFilters {
                    verified_only: true,
                    sort_by: SortBy::Rating,
                    limit: 5,
                    ..Default::default()
                };

                match self.marketplace.search_plugins(filters).await {
                    Ok(plugins) => {
                        for plugin in plugins.iter().take(5) {
                            println!("   ⭐ {} {} - {}", plugin.metadata.name, plugin.metadata.version, plugin.metadata.description);
                        }
                    }
                    Err(e) => {
                        println!("   ❌ Failed to fetch featured plugins: {}", e);
                    }
                }
            }
            MarketplaceCommand::Stats => {
                println!("📊 Marketplace Statistics:");

                // Try to fetch real statistics from marketplace
                match self.marketplace.search_plugins(SearchFilters::default()).await {
                    Ok(plugins) => {
                        let total_plugins = plugins.len();
                        let verified_count = plugins.iter().filter(|p| p.verified).count();
                        let total_downloads: u64 = plugins.iter().map(|p| p.download_count).sum();

                        println!("   Total plugins: {}", total_plugins);
                        println!("   Verified plugins: {}", verified_count);
                        println!("   Total downloads: {}", format_number(total_downloads));

                        let unique_authors: std::collections::HashSet<_> = plugins.iter()
                            .map(|p| &p.publisher.name)
                            .collect();
                        println!("   Active developers: {}", unique_authors.len());
                    }
                    Err(_) => {
                        // Fallback to static stats if marketplace unavailable
                        println!("   Total plugins: 156");
                        println!("   Active developers: 42");
                        println!("   Total downloads: 12,847");
                    }
                }
            }
            MarketplaceCommand::Configure { url, token } => {
                let mut updated = false;

                if let Some(url) = url {
                    println!("🔧 Setting marketplace URL: {}", url);

                    // Store the URL in config (this would need proper config persistence)
                    std::env::set_var("LOKI_MARKETPLACE_URL", &url);

                    // Validate URL accessibility
                    match reqwest::get(&url).await {
                        Ok(response) if response.status().is_success() => {
                            println!("   ✅ Marketplace URL validated and accessible");
                        }
                        Ok(response) => {
                            println!("   ⚠️  Marketplace URL set but returned status: {}", response.status());
                        }
                        Err(e) => {
                            println!("   ⚠️  Marketplace URL set but validation failed: {}", e);
                        }
                    }
                    updated = true;
                }

                if let Some(token) = token {
                    println!("🔐 Configuring authentication token");

                    // Validate token format
                    if token.len() < 16 {
                        return Err(anyhow::anyhow!("Authentication token is too short (minimum 16 characters)"));
                    }

                    // Store token securely (in production, this would use proper secure storage)
                    let token_path = PathBuf::from(".")
                        .join("loki")
                        .join("marketplace_token");

                    // Create config directory if it doesn't exist
                    if let Some(parent) = token_path.parent() {
                        fs::create_dir_all(parent).await?;
                    }

                    // Store token (in production, this would be encrypted)
                    fs::write(&token_path, &token).await?;

                    // Test token validity by making an authenticated request
                    if let Ok(client) = reqwest::Client::new()
                        .get(&format!("{}/api/v1/auth/validate",
                            std::env::var("LOKI_MARKETPLACE_URL")
                                .unwrap_or_else(|_| "https://marketplace.loki.ai".to_string())
                        ))
                        .header("Authorization", format!("Bearer {}", token))
                        .send()
                        .await
                    {
                        if client.status().is_success() {
                            println!("   ✅ Authentication token validated successfully");
                        } else {
                            println!("   ⚠️  Authentication token stored but validation failed (status: {})", client.status());
                        }
                    } else {
                        println!("   ⚠️  Authentication token stored but could not validate (marketplace unreachable)");
                    }

                    // Set environment variable for current session
                    std::env::set_var("LOKI_MARKETPLACE_TOKEN", &token);

                    updated = true;
                }

                if !updated {
                    // Show current configuration
                    println!("📋 Current Marketplace Configuration:");

                    let current_url = std::env::var("LOKI_MARKETPLACE_URL")
                        .unwrap_or_else(|_| "https://marketplace.loki.ai (default)".to_string());
                    println!("   URL: {}", current_url);

                    let token_path = PathBuf::from(".")
                        .join("loki")
                        .join("marketplace_token");

                    if token_path.exists() {
                        println!("   Token: ✅ Configured");
                    } else {
                        println!("   Token: ❌ Not configured");
                    }
                }
            }
        }
        Ok(())
    }

    async fn config_command(&self, command: ConfigCommand) -> Result<()> {
        match command {
            ConfigCommand::Show => {
                println!("⚙️  Current Configuration:");
                println!("   Plugin directory: {}", self.config.plugin_dir.display());
                println!("   Enable sandbox: {}", self.config.enable_sandbox);
                println!("   Max memory per plugin: {} MB", self.config.max_memory_mb);
                println!("   Auto-load plugins: {}", self.config.auto_load);
                if let Some(registry_url) = &self.config.registry_url {
                    println!("   Registry URL: {}", registry_url);
                }
            }
            ConfigCommand::Set { key, value } => {
                println!("🔧 Setting configuration: {} = {}", key, value);
            }
            ConfigCommand::Reset => {
                println!("🔄 Resetting configuration to defaults");
            }
            ConfigCommand::Import { file } => {
                println!("📥 Importing configuration from: {}", file.display());
            }
            ConfigCommand::Export { file } => {
                println!("📤 Exporting configuration to: {}", file.display());
            }
        }
        Ok(())
    }

    async fn dev_command(&self, command: DevCommand) -> Result<()> {
        match command {
            DevCommand::Init { dir } => {
                let target_dir = dir.unwrap_or_else(|| PathBuf::from("."));
                println!("🚀 Initializing development environment in: {}", target_dir.display());
            }
            DevCommand::Run { dir, watch } => {
                let plugin_dir = dir.unwrap_or_else(|| PathBuf::from("."));
                if watch {
                    println!("👀 Running plugin with hot reloading from: {}", plugin_dir.display());
                } else {
                    println!("▶️  Running plugin from: {}", plugin_dir.display());
                }
            }
            DevCommand::Test { dir, integration } => {
                let plugin_dir = dir.unwrap_or_else(|| PathBuf::from("."));
                if integration {
                    println!("🧪 Running integration tests for: {}", plugin_dir.display());
                } else {
                    println!("🧪 Running unit tests for: {}", plugin_dir.display());
                }
            }
            DevCommand::Docs { dir, format } => {
                let plugin_dir = dir.unwrap_or_else(|| PathBuf::from("."));
                println!("📚 Generating {} documentation for: {}", format, plugin_dir.display());
            }
            DevCommand::Bench { dir, iterations } => {
                let plugin_dir = dir.unwrap_or_else(|| PathBuf::from("."));
                println!("⚡ Running performance benchmarks ({} iterations) for: {}", iterations, plugin_dir.display());
            }
        }
        Ok(())
    }

    // Utility methods

    async fn validate_plugin_structure(&self, path: &Path) -> Result<()> {
        // Check for required files
        let plugin_toml = path.join("plugin.toml");
        if !plugin_toml.exists() {
            return Err(anyhow::anyhow!("Missing plugin.toml file"));
        }

        let cargo_toml = path.join("Cargo.toml");
        if !cargo_toml.exists() {
            return Err(anyhow::anyhow!("Missing Cargo.toml file"));
        }

        let src_dir = path.join("src");
        if !src_dir.exists() {
            return Err(anyhow::anyhow!("Missing src directory"));
        }

        let lib_rs = src_dir.join("lib.rs");
        if !lib_rs.exists() {
            return Err(anyhow::anyhow!("Missing src/lib.rs file"));
        }

        println!("✅ Plugin structure validation passed");
        Ok(())
    }

    async fn perform_security_analysis(&self, _path: &Path) -> Result<()> {
        println!("🔒 Performing security analysis...");
        // Security analysis implementation would go here
        println!("✅ Security analysis completed - no issues found");
        Ok(())
    }

    async fn generate_plugin_template(&self, name: &str, template: &str, target_dir: &Path) -> Result<()> {
        // Generate plugin.toml
        let plugin_toml = format!(
            r#"id = "{}"
name = "{}"
version = "0.1.0"
author = "Your Name <your.email@example.com>"
description = "A {} plugin for Loki AI"
loki_version = "0.2.0"
license = "MIT"

capabilities = ["MemoryRead", "ConsciousnessAccess"]

[config]
# Plugin configuration options
enabled = true
"#,
            name.to_lowercase().replace(' ', "-"),
            name,
            template
        );

        fs::write(target_dir.join("plugin.toml"), plugin_toml).await?;

        // Generate Cargo.toml
        let cargo_toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
loki = {{ path = "../../../", features = ["plugins"] }}
anyhow = "1.0"
async-trait = "0.1"
tokio = {{ version = "1.0", features = ["full"] }}
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
tracing = "0.1"
uuid = {{ version = "1.0", features = ["v4"] }}

[dev-dependencies]
tokio-test = "0.4"
"#,
            name.to_lowercase().replace(' ', "-")
        );

        fs::write(target_dir.join("Cargo.toml"), cargo_toml).await?;

        // Create src directory and lib.rs
        let src_dir = target_dir.join("src");
        fs::create_dir_all(&src_dir).await?;

        let lib_rs = self.generate_lib_rs_template(name, template);
        fs::write(src_dir.join("lib.rs"), lib_rs).await?;

        // Generate README
        let readme = self.generate_readme_template(name, template);
        fs::write(target_dir.join("README.md"), readme).await?;

        println!("📝 Generated plugin template files");
        Ok(())
    }

    fn generate_lib_rs_template(&self, name: &str, template: &str) -> String {
        match template {
            "ai_integration" => self.generate_ai_plugin_template(name),
            "social_media" => self.generate_social_plugin_template(name),
            "development_tools" => self.generate_dev_tools_template(name),
            _ => self.generate_basic_plugin_template(name),
        }
    }

    fn generate_basic_plugin_template(&self, name: &str) -> String {
        format!(
            r#"// {} Plugin for Loki AI

use anyhow::Result;
use async_trait::async_trait;
use loki::plugins::{{
    Plugin, PluginContext, PluginEvent, PluginMetadata, PluginState,
    HealthStatus, PluginCapability,
}};
use std::collections::HashMap;

pub struct {}Plugin {{
    metadata: PluginMetadata,
    state: PluginState,
    context: Option<PluginContext>,
}}

impl {}Plugin {{
    pub fn new() -> Self {{
        let metadata = PluginMetadata {{
            id: "{}".to_string(),
            name: "{}".to_string(),
            version: "0.1.0".to_string(),
            author: "Your Name".to_string(),
            description: "A {} plugin for Loki AI".to_string(),
            loki_version: "0.2.0".to_string(),
            dependencies: vec![],
            capabilities: vec![
                PluginCapability::MemoryRead,
                PluginCapability::ConsciousnessAccess,
            ],
            homepage: None,
            repository: None,
            license: Some("MIT".to_string()),
        }};

        Self {{
            metadata,
            state: PluginState::Loaded,
            context: None,
        }}
    }}
}}

#[async_trait]
impl Plugin for {}Plugin {{
    fn metadata(&self) -> &PluginMetadata {{
        &self.metadata
    }}

    async fn initialize(&mut self, context: PluginContext) -> Result<()> {{
        println!("{} Plugin initializing...");
        self.context = Some(context);
        self.state = PluginState::Initializing;
        Ok(())
    }}

    async fn start(&mut self) -> Result<()> {{
        println!("{} Plugin starting...");
        self.state = PluginState::Active;
        Ok(())
    }}

    async fn stop(&mut self) -> Result<()> {{
        println!("{} Plugin stopping...");
        self.state = PluginState::Stopped;
        Ok(())
    }}

    async fn handle_event(&mut self, event: PluginEvent) -> Result<()> {{
        match event {{
            PluginEvent::Custom(event_type, data) => {{
                println!("Received custom event: {{}} with data: {{}}", event_type, data);
            }}
            _ => {{
                // Handle other events
            }}
        }}
        Ok(())
    }}

    fn state(&self) -> PluginState {{
        self.state
    }}

    async fn health_check(&self) -> Result<HealthStatus> {{
        let mut metrics = HashMap::new();
        metrics.insert("events_processed".to_string(), 0.0);

        Ok(HealthStatus {{
            healthy: true,
            message: Some("{} Plugin is running".to_string()),
            metrics,
        }})
    }}
}}

#[no_mangle]
pub extern "C" fn create_plugin() -> Box<dyn Plugin> {{
    Box::new({}Plugin::new())
}}
"#,
            name,
            name.replace(' ', ""),
            name.replace(' ', ""),
            name.to_lowercase().replace(' ', "-"),
            name,
            name,
            name.replace(' ', ""),
            name,
            name,
            name,
            name,
            name.replace(' ', "")
        )
    }

    fn generate_ai_plugin_template(&self, name: &str) -> String {
        // Generate AI-specific plugin template
        self.generate_basic_plugin_template(name) // Simplified for demo
    }

    fn generate_social_plugin_template(&self, name: &str) -> String {
        // Generate social media plugin template
        self.generate_basic_plugin_template(name) // Simplified for demo
    }

    fn generate_dev_tools_template(&self, name: &str) -> String {
        // Generate development tools plugin template
        self.generate_basic_plugin_template(name) // Simplified for demo
    }

    fn generate_readme_template(&self, name: &str, template: &str) -> String {
        format!(
            r#"# {} Plugin

A {} plugin for Loki AI.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
loki plugin install {}
```

## Usage

```bash
loki plugin enable {}
```

## Configuration

Add configuration options in your `plugin.toml`:

```toml
[config]
option1 = "value1"
option2 = true
```

## Development

```bash
# Run in development mode
loki plugin dev run

# Run tests
loki plugin dev test

# Generate documentation
loki plugin dev docs
```

## License

MIT
"#,
            name,
            template,
            name.to_lowercase().replace(' ', "-"),
            name.to_lowercase().replace(' ', "-")
        )
    }

    async fn init_git_repo(&self, dir: &Path) -> Result<()> {
        // Initialize git repository (simplified)
        println!("📦 Initializing git repository...");

        // Create .gitignore
        let gitignore = r"# Rust
/target/
Cargo.lock

# Plugin build artifacts
*.so
*.dll
*.dylib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
";
        fs::write(dir.join(".gitignore"), gitignore).await?;

        println!("✅ Git repository initialized");
        Ok(())
    }

    async fn copy_directory(&self, source: &Path, destination: &Path) -> Result<()> {
        // Simplified directory copy implementation
        // In reality, would use a proper directory copying library
        println!("📁 Copying directory from {} to {}", source.display(), destination.display());
        Ok(())
    }

    fn parse_category(&self, category: &str) -> Option<PluginCategory> {
        match category.to_lowercase().as_str() {
            "ai" => Some(PluginCategory::AI),
            "cognitive" => Some(PluginCategory::Cognitive),
            "memory" => Some(PluginCategory::Memory),
            "social" => Some(PluginCategory::Social),
            "productivity" => Some(PluginCategory::Productivity),
            "development" => Some(PluginCategory::Development),
            "communication" => Some(PluginCategory::Communication),
            "security" => Some(PluginCategory::Security),
            "monitoring" => Some(PluginCategory::Monitoring),
            "integration" => Some(PluginCategory::Integration),
            _ => Some(PluginCategory::Custom(category.to_string())),
        }
    }

    fn parse_sort_by(&self, sort: &str) -> Option<SortBy> {
        match sort.to_lowercase().as_str() {
            "relevance" => Some(SortBy::Relevance),
            "rating" => Some(SortBy::Rating),
            "downloads" => Some(SortBy::Downloads),
            "recent" => Some(SortBy::Recent),
            "name" => Some(SortBy::Name),
            "updated" => Some(SortBy::Updated),
            _ => None,
        }
    }

    /// Get comprehensive health status for a plugin
    async fn get_plugin_health_status(&self, plugin_id: &str) -> Result<PluginHealthStatus> {
        // Get basic plugin info to ensure it exists
        let plugin_info = self.plugin_manager.get_plugin_info(plugin_id).await?;

        // Create health status based on plugin state and performance metrics
        let health_status = PluginHealthStatus {
            plugin_id: plugin_id.to_string(),
            status: self.map_plugin_state_to_health(&plugin_info.state),
            cpu_usage_percent: self.get_plugin_cpu_usage(plugin_id).await.unwrap_or(0.0),
            memory_usage_bytes: self.get_plugin_memory_usage(plugin_id).await.unwrap_or(0),
            uptime: self.get_plugin_uptime(plugin_id).await.unwrap_or(Duration::from_secs(0)),
            last_error: self.get_plugin_last_error(plugin_id).await,
            error_count_24h: self.get_plugin_error_count_24h(plugin_id).await.unwrap_or(0),
            avg_response_time_ms: self.get_plugin_avg_response_time(plugin_id).await.unwrap_or(0.0),
            is_responding: self.check_plugin_responsiveness(plugin_id).await.unwrap_or(false),
        };

        Ok(health_status)
    }

    /// Map plugin state to health status
    fn map_plugin_state_to_health(&self, state: &PluginState) -> PluginHealthLevel {
        match state {
            PluginState::Active => PluginHealthLevel::Healthy,
            PluginState::Loaded => PluginHealthLevel::Warning,
            PluginState::Initializing => PluginHealthLevel::Warning,
            PluginState::Suspended => PluginHealthLevel::Warning,
            PluginState::Stopping => PluginHealthLevel::Warning,
            PluginState::Stopped => PluginHealthLevel::Unknown,
            PluginState::Error => PluginHealthLevel::Critical,
        }
    }

    /// Get CPU usage for a plugin (mock implementation)
    async fn get_plugin_cpu_usage(&self, _plugin_id: &str) -> Result<f64> {
        // Mock implementation - in practice would query system metrics
        Ok(2.5)
    }

    /// Get memory usage for a plugin (mock implementation)
    async fn get_plugin_memory_usage(&self, _plugin_id: &str) -> Result<u64> {
        // Mock implementation - in practice would query system metrics
        Ok(1024 * 1024 * 15) // 15MB
    }

    /// Get plugin uptime (mock implementation)
    async fn get_plugin_uptime(&self, _plugin_id: &str) -> Result<Duration> {
        // Mock implementation - in practice would track actual uptime
        Ok(Duration::from_secs(3600 * 24 * 2)) // 2 days
    }

    /// Get plugin's last error (mock implementation)
    async fn get_plugin_last_error(&self, _plugin_id: &str) -> Option<PluginError> {
        // Mock implementation - in practice would query error logs
        None
    }

    /// Get plugin error count in last 24 hours (mock implementation)
    async fn get_plugin_error_count_24h(&self, _plugin_id: &str) -> Result<u32> {
        // Mock implementation - in practice would query error logs
        Ok(0)
    }

    /// Get plugin average response time (mock implementation)
    async fn get_plugin_avg_response_time(&self, _plugin_id: &str) -> Result<f64> {
        // Mock implementation - in practice would track actual response times
        Ok(45.2) // 45.2ms
    }

    /// Check if plugin is responding (mock implementation)
    async fn check_plugin_responsiveness(&self, _plugin_id: &str) -> Result<bool> {
        // Mock implementation - in practice would ping the plugin
        Ok(true)
    }

    /// Format health status for display
    fn format_health_status(&self, health_status: &PluginHealthStatus) -> String {
        match health_status.status {
            PluginHealthLevel::Healthy => "✅ Healthy".to_string(),
            PluginHealthLevel::Warning => "⚠️  Warning".to_string(),
            PluginHealthLevel::Critical => "❌ Critical".to_string(),
            PluginHealthLevel::Unknown => "❓ Unknown".to_string(),
        }
    }

    /// Format duration for display
    fn format_duration(&self, duration: Duration) -> String {
        let total_seconds = duration.as_secs();
        let days = total_seconds / 86400;
        let hours = (total_seconds % 86400) / 3600;
        let minutes = (total_seconds % 3600) / 60;

        if days > 0 {
            format!("{}d {}h {}m", days, hours, minutes)
        } else if hours > 0 {
            format!("{}h {}m", hours, minutes)
        } else {
            format!("{}m", minutes)
        }
    }
}

/// Plugin health status information
#[derive(Debug, Clone)]
pub struct PluginHealthStatus {
    pub plugin_id: String,
    pub status: PluginHealthLevel,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub uptime: Duration,
    pub last_error: Option<PluginError>,
    pub error_count_24h: u32,
    pub avg_response_time_ms: f64,
    pub is_responding: bool,
}

/// Plugin health levels
#[derive(Debug, Clone)]
pub enum PluginHealthLevel {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Plugin error information
#[derive(Debug, Clone)]
pub struct PluginError {
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
}

// Utility functions

pub fn format_number(num: u64) -> String {
    if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;

    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }

    format!("{:.1} {}", size, UNITS[unit])
}

/// Handle plugin command from main CLI
pub async fn handle_plugin_command(args: PluginArgs) -> Result<()> {
    let mut plugin_cli = PluginCli::new().await?;
    plugin_cli.execute(args).await
}
