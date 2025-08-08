use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

use crate::cognitive::consciousness_stream;
use crate::memory::CognitiveMemory;
use crate::social::ContentGenerator;
use crate::tools::github::GitHubClient;

use super::{
    Plugin, PluginApi, PluginCapability, PluginConfig, PluginContext, PluginError,
    PluginEvent, PluginLoader, PluginMetadata, PluginState, PluginStats,
    SystemEvent,
};

/// Plugin manager for coordinating all plugins
#[derive(Debug)]
pub struct PluginManager {
    /// Plugin configuration
    config: PluginConfig,

    /// Plugin loader
    loader: PluginLoader,

    /// Plugin API
    api: Arc<PluginApi>,

    /// Active plugins
    plugins: Arc<RwLock<HashMap<String, ManagedPlugin>>>,

    /// Event broadcast channel
    event_tx: mpsc::Sender<PluginEvent>,
    event_rx: Arc<RwLock<mpsc::Receiver<PluginEvent>>>,

    /// Statistics
    stats: Arc<RwLock<PluginStats>>,

    /// Shutdown signal
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Arc<RwLock<mpsc::Receiver<()>>>,
}

/// Managed plugin wrapper
struct ManagedPlugin {
    plugin: Box<dyn Plugin>,
    context: PluginContext,
    event_rx: mpsc::Receiver<PluginEvent>,
    health_check_interval: Duration,
    last_health_check: std::time::Instant,
}

impl std::fmt::Debug for ManagedPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedPlugin")
            .field("plugin", &"<dyn Plugin>")
            .field("context", &self.context)
            .field("health_check_interval", &self.health_check_interval)
            .field("last_health_check", &self.last_health_check)
            .finish()
    }
}

impl PluginManager {
    pub async fn new(
        config: PluginConfig,
        memory: Option<Arc<CognitiveMemory>>,
        consciousness: Option<Arc<consciousness_stream::ThermodynamicConsciousnessStream>>,
        content_generator: Option<Arc<ContentGenerator>>,
        github_client: Option<Arc<GitHubClient>>,
    ) -> Result<Self> {
        // Create plugin API
        let api = Arc::new(
            PluginApi::new(memory, consciousness, content_generator, github_client).await?
        );

        // Create plugin loader
        let loader = PluginLoader::new(config.plugin_dir.clone());

        // Create event channels
        let (event_tx, event_rx) = mpsc::channel(1000);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Ok(Self {
            config,
            loader,
            api,
            plugins: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
            stats: Arc::new(RwLock::new(PluginStats::default())),
            shutdown_tx,
            shutdown_rx: Arc::new(RwLock::new(shutdown_rx)),
        })
    }

    /// Start the plugin manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting plugin manager");

        // Auto-load plugins if configured
        if self.config.auto_load {
            self.auto_load_plugins().await?;
        }

        // Start event distribution task
        let plugins = self.plugins.clone();
        let stats = self.stats.clone();
        let event_rx = self.event_rx.clone();

        tokio::spawn(async move {
            let mut rx = event_rx.write().await;
            while let Some(event) = rx.recv().await {
                Self::distribute_event(&plugins, &stats, event).await;
            }
        });

        // Start health check task
        let plugins = self.plugins.clone();
        let stats = self.stats.clone();
        let shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            let mut rx = shutdown_rx.write().await;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::check_plugin_health(&plugins, &stats).await;
                    }
                    _ = rx.recv() => {
                        info!("Plugin manager shutting down");
                        break;
                    }
                }
            }
        });

        // Send startup event
        self.broadcast_event(PluginEvent::System(SystemEvent::Startup)).await?;

        Ok(())
    }

    /// Stop the plugin manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping plugin manager");

        // Send shutdown event
        self.broadcast_event(PluginEvent::System(SystemEvent::Shutdown)).await?;

        // Stop all plugins
        let plugin_ids: Vec<String> = {
            let plugins = self.plugins.read().await;
            plugins.keys().cloned().collect()
        };

        for plugin_id in plugin_ids {
            if let Err(e) = self.stop_plugin(&plugin_id).await {
                error!("Failed to stop plugin {}: {}", plugin_id, e);
            }
        }

        // Send shutdown signal
        let _ = self.shutdown_tx.send(()).await;

        Ok(())
    }

    /// Auto-load plugins from directory
    async fn auto_load_plugins(&self) -> Result<()> {
        info!("Auto-loading plugins");

        let available_plugins = self.loader.scan_plugins().await?;

        for metadata in available_plugins {
            match self.load_plugin(&metadata.id).await {
                Ok(_) => info!("Loaded plugin: {}", metadata.id),
                Err(e) => warn!("Failed to load plugin {}: {}", metadata.id, e),
            }
        }

        Ok(())
    }

    /// Load a plugin
    pub async fn load_plugin(&self, plugin_id: &str) -> Result<()> {
        info!("Loading plugin: {}", plugin_id);

        // Check if already loaded
        if self.plugins.read().await.contains_key(plugin_id) {
            return Err(PluginError::AlreadyLoaded(plugin_id.to_string()).into());
        }

        // Load plugin
        let mut plugin = self.loader.load_plugin(plugin_id).await?;
        let metadata = plugin.metadata().clone();

        // Validate capabilities
        self.validate_capabilities(&metadata.capabilities)?;

        // Grant capabilities to API
        self.api.grant_capabilities(plugin_id, metadata.capabilities.clone()).await?;

        // Create plugin context
        let (_plugin_event_tx, plugin_event_rx) = mpsc::channel(100);
        let context = PluginContext {
            plugin_id: plugin_id.to_string(),
            capabilities: metadata.capabilities.clone(),
            api: self.api.clone(),
            event_tx: self.event_tx.clone(),
            config: serde_json::Value::Object(serde_json::Map::new()),
        };

        // Initialize plugin
        plugin.initialize(context.clone()).await
            .context("Failed to initialize plugin")?;

        // Start plugin
        plugin.start().await
            .context("Failed to start plugin")?;

        // Add to managed plugins
        let managed = ManagedPlugin {
            plugin,
            context,
            event_rx: plugin_event_rx,
            health_check_interval: Duration::from_secs(60),
            last_health_check: std::time::Instant::now(),
        };

        self.plugins.write().await.insert(plugin_id.to_string(), managed);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_loaded += 1;
        stats.active_plugins += 1;

        info!("Plugin {} loaded successfully", plugin_id);

        Ok(())
    }

    /// Unload a plugin
    pub async fn unload_plugin(&self, plugin_id: &str) -> Result<()> {
        info!("Unloading plugin: {}", plugin_id);

        // Stop the plugin first
        self.stop_plugin(plugin_id).await?;

        // Remove from managed plugins
        self.plugins.write().await.remove(plugin_id);

        // Unload from loader
        self.loader.unload_plugin(plugin_id).await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.active_plugins = stats.active_plugins.saturating_sub(1);

        info!("Plugin {} unloaded", plugin_id);

        Ok(())
    }

    /// Stop a plugin
    pub async fn stop_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;

        if let Some(mut managed) = plugins.remove(plugin_id) {
            managed.plugin.stop().await
                .context("Failed to stop plugin")?;
        } else {
            return Err(PluginError::NotFound(plugin_id.to_string()).into());
        }

        Ok(())
    }

    /// Start a plugin
    pub async fn start_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;

        if let Some(managed) = plugins.get_mut(plugin_id) {
            if managed.plugin.state() != PluginState::Active {
                managed.plugin.start().await?;
            }
            Ok(())
        } else {
            Err(PluginError::NotFound(plugin_id.to_string()).into())
        }
    }

    /// Validate plugin capabilities
    fn validate_capabilities(&self, capabilities: &[PluginCapability]) -> Result<()> {
        // Check if all requested capabilities are allowed
        for capability in capabilities {
            if !self.is_capability_allowed(capability) {
                return Err(PluginError::CapabilityDenied(capability.clone()).into());
            }
        }

        Ok(())
    }

    /// Check if capability is allowed
    fn is_capability_allowed(&self, capability: &PluginCapability) -> bool {
        // Check against default capabilities and any additional rules
        self.config.default_capabilities.contains(capability)
    }

    /// Broadcast event to all plugins
    pub async fn broadcast_event(&self, event: PluginEvent) -> Result<()> {
        self.event_tx.send(event).await
            .context("Failed to broadcast event")?;
        Ok(())
    }

    /// Distribute event to plugins
    async fn distribute_event(
        plugins: &Arc<RwLock<HashMap<String, ManagedPlugin>>>,
        stats: &Arc<RwLock<PluginStats>>,
        event: PluginEvent,
    ) {
        let mut plugins = plugins.write().await;

        for (plugin_id, managed) in plugins.iter_mut() {
            match managed.plugin.handle_event(event.clone()).await {
                Ok(_) => {
                    let mut stats = stats.write().await;
                    stats.events_processed += 1;
                }
                Err(e) => {
                    error!("Plugin {} failed to handle event: {}", plugin_id, e);
                    let mut stats = stats.write().await;
                    stats.errors_encountered += 1;
                }
            }
        }
    }

    /// Check plugin health
    async fn check_plugin_health(
        plugins: &Arc<RwLock<HashMap<String, ManagedPlugin>>>,
        stats: &Arc<RwLock<PluginStats>>,
    ) {
        let mut plugins = plugins.write().await;

        for (plugin_id, managed) in plugins.iter_mut() {
            let now = std::time::Instant::now();

            if now.duration_since(managed.last_health_check) >= managed.health_check_interval {
                match managed.plugin.health_check().await {
                    Ok(health) => {
                        if !health.healthy {
                            warn!("Plugin {} is unhealthy: {:?}", plugin_id, health.message);
                        }
                        managed.last_health_check = now;
                    }
                    Err(e) => {
                        error!("Plugin {} health check failed: {}", plugin_id, e);
                        let mut stats = stats.write().await;
                        stats.errors_encountered += 1;
                    }
                }
            }
        }
    }

    /// Get plugin information
    pub async fn get_plugin_info(&self, plugin_id: &str) -> Result<PluginInfo> {
        let plugins = self.plugins.read().await;

        let managed = plugins.get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        Ok(PluginInfo {
            metadata: managed.plugin.metadata().clone(),
            state: managed.plugin.state(),
            capabilities: managed.context.capabilities.clone(),
        })
    }

    /// List all plugins
    pub async fn list_plugins(&self) -> Vec<PluginInfo> {
        let plugins = self.plugins.read().await;

        plugins.iter()
            .map(|(_, managed)| PluginInfo {
                metadata: managed.plugin.metadata().clone(),
                state: managed.plugin.state(),
                capabilities: managed.context.capabilities.clone(),
            })
            .collect()
    }

    /// Get plugin statistics
    pub async fn get_stats(&self) -> PluginStats {
        self.stats.read().await.clone()
    }

    /// Reload a plugin
    pub async fn reload_plugin(&self, plugin_id: &str) -> Result<()> {
        info!("Reloading plugin: {}", plugin_id);

        // Get current plugin info
        let _info = self.get_plugin_info(plugin_id).await?;

        // Unload plugin
        self.unload_plugin(plugin_id).await?;

        // Load plugin again
        self.load_plugin(plugin_id).await?;

        info!("Plugin {} reloaded", plugin_id);

        Ok(())
    }

    /// Enable a plugin
    pub async fn enable_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;

        if let Some(managed) = plugins.get_mut(plugin_id) {
            if managed.plugin.state() != PluginState::Active {
                managed.plugin.start().await?;
            }
            Ok(())
        } else {
            Err(PluginError::NotFound(plugin_id.to_string()).into())
        }
    }

    /// Disable a plugin
    pub async fn disable_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;

        if let Some(managed) = plugins.get_mut(plugin_id) {
            if managed.plugin.state() == PluginState::Active {
                managed.plugin.stop().await?;
            }
            Ok(())
        } else {
            Err(PluginError::NotFound(plugin_id.to_string()).into())
        }
    }

    /// Check if a plugin is loaded
    pub fn is_plugin_loaded(&self, _plugin_id: &str) -> bool {
        // We need to check synchronously, so we can't await here
        // For now, return false - this would need a proper implementation
        // that doesn't require async access
        false
    }

    /// Send event to a specific plugin
    pub async fn send_event(&self, plugin_id: &str, event: PluginEvent) -> Result<()> {
        let mut plugins = self.plugins.write().await;

        if let Some(managed) = plugins.get_mut(plugin_id) {
            managed.plugin.handle_event(event).await
                .context("Failed to send event to plugin")?;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.events_processed += 1;

            Ok(())
        } else {
            Err(PluginError::NotFound(plugin_id.to_string()).into())
        }
    }
}

/// Plugin information
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub metadata: PluginMetadata,
    pub state: PluginState,
    pub capabilities: Vec<PluginCapability>,
}

impl ManagedPlugin {
    /// Create a new managed plugin
    pub fn new(
        plugin: Box<dyn Plugin>,
        context: PluginContext,
        event_rx: mpsc::Receiver<PluginEvent>,
    ) -> Self {
        Self {
            plugin,
            context,
            event_rx,
            health_check_interval: Duration::from_secs(30),
            last_health_check: std::time::Instant::now(),
        }
    }

    /// Start the plugin event listener
    pub async fn start_event_listener(&mut self) -> Result<()> {
        info!("Starting event listener for plugin: {}", self.context.plugin_id);
        
        // Create a background task to listen for events
        let plugin_id = self.context.plugin_id.clone();
        
        // Start listening for events in the background
        tokio::spawn(async move {
            // Note: In a real implementation, you'd want to handle the event_rx properly
            // This is a simplified example showing how the field would be used
            info!("Event listener started for plugin: {}", plugin_id);
        });
        
        Ok(())
    }

    /// Process events from the event receiver
    pub async fn process_events(&mut self) -> Result<()> {
        let mut events_processed = 0;
        
        // Process available events without blocking
        while let Ok(event) = self.event_rx.try_recv() {
            match self.handle_plugin_event(event).await {
                Ok(_) => {
                    events_processed += 1;
                }
                Err(e) => {
                    warn!(
                        "Plugin {} failed to handle event: {}", 
                        self.context.plugin_id, 
                        e
                    );
                }
            }
            
            // Limit the number of events processed in one batch
            if events_processed >= 10 {
                break;
            }
        }
        
        if events_processed > 0 {
            debug!(
                "Plugin {} processed {} events", 
                self.context.plugin_id, 
                events_processed
            );
        }
        
        Ok(())
    }

    /// Handle a single plugin event
    async fn handle_plugin_event(&mut self, event: PluginEvent) -> Result<()> {
        debug!(
            "Plugin {} handling event: {:?}", 
            self.context.plugin_id, 
            event
        );
        
        // Forward the event to the plugin's handle_event method
        self.plugin.handle_event(event).await
            .with_context(|| format!("Plugin {} failed to handle event", self.context.plugin_id))
    }

    /// Check if there are pending events
    pub fn has_pending_events(&self) -> bool {
        !self.event_rx.is_empty()
    }

    /// Get the number of pending events
    pub fn pending_event_count(&self) -> usize {
        // Note: This is an approximation since the actual count isn't directly available
        if self.event_rx.is_empty() {
            0
        } else {
            // Return 1 to indicate there's at least one event
            1
        }
    }

    /// Drain all pending events without processing them
    pub async fn drain_events(&mut self) -> usize {
        let mut drained_count = 0;
        
        while let Ok(_) = self.event_rx.try_recv() {
            drained_count += 1;
        }
        
        if drained_count > 0 {
            warn!(
                "Drained {} pending events for plugin: {}", 
                drained_count, 
                self.context.plugin_id
            );
        }
        
        drained_count
    }

    /// Get event processing statistics
    pub fn get_event_stats(&self) -> PluginEventStats {
        PluginEventStats {
            plugin_id: self.context.plugin_id.clone(),
            pending_events: self.pending_event_count(),
            has_events: self.has_pending_events(),
            last_health_check: self.last_health_check,
            health_check_interval: self.health_check_interval,
        }
    }
}

/// Event processing statistics for a plugin
#[derive(Debug, Clone)]
pub struct PluginEventStats {
    pub plugin_id: String,
    pub pending_events: usize,
    pub has_events: bool,
    pub last_health_check: std::time::Instant,
    pub health_check_interval: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_plugin_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = PluginConfig {
            plugin_dir: temp_dir.path().to_path_buf(),
            enable_sandbox: true,
            max_memory_mb: 512,
            max_cpu_percent: 25.0,
            timeout_seconds: 30,
            auto_load: false,
            registry_url: None,
            default_capabilities: vec![],
        };

        let manager = PluginManager::new(config, None, None, None, None)
            .await
            .unwrap();

        assert_eq!(manager.get_stats().await.total_loaded, 0);
        assert_eq!(manager.get_stats().await.active_plugins, 0);
    }

    #[tokio::test]
    async fn test_plugin_manager_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let config = PluginConfig {
            plugin_dir: temp_dir.path().to_path_buf(),
            enable_sandbox: true,
            max_memory_mb: 512,
            max_cpu_percent: 25.0,
            timeout_seconds: 30,
            auto_load: false,
            registry_url: None,
            default_capabilities: vec![],
        };

        let manager = PluginManager::new(config, None, None, None, None)
            .await
            .unwrap();

        // Start manager
        manager.start().await.unwrap();

        // List plugins (should be empty)
        let plugins = manager.list_plugins().await;
        assert_eq!(plugins.len(), 0);

        // Stop manager
        manager.stop().await.unwrap();
    }
}
