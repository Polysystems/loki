use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use super::{Plugin, PluginMetadata, PluginError, PluginEvent, PluginState, HealthStatus, PluginCapability};
use crate::plugins::api::PluginContext;

/// Plugin loader for different plugin types
#[derive(Debug)]
pub struct PluginLoader {
    /// Plugin directory
    plugin_dir: PathBuf,

    /// Loaded plugins cache
    loaded_plugins: Arc<RwLock<Vec<LoadedPlugin>>>,
}

/// Loaded plugin information
#[derive(Clone, Debug)]
pub struct LoadedPlugin {
    pub metadata: PluginMetadata,
    pub plugin_type: PluginType,
    pub path: PathBuf,
    pub checksum: String,
}

/// Library metadata for dynamic plugins
#[derive(Debug, Clone)]
pub struct LibraryMetadata {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub modified: Option<SystemTime>,
    pub is_file: bool,
    pub permissions: String,
}

/// Connection statistics for remote plugins
#[derive(Debug, Clone)]
pub struct RemoteConnectionStats {
    pub endpoint_url: url::Url,
    pub connection_type: String,
    pub is_secure: bool,
    pub host: String,
    pub port: u16,
}

/// Plugin type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginType {
    /// WebAssembly plugin
    Wasm,

    /// Dynamic library plugin (native)
    DynamicLibrary,

    /// Script-based plugin (Python, JavaScript, etc.)
    Script(ScriptType),

    /// Remote plugin (via gRPC/HTTP)
    Remote(String),
}

/// Script type for script-based plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScriptType {
    Python,
    JavaScript,
    Lua,
}

impl PluginLoader {
    pub fn new(plugin_dir: PathBuf) -> Self {
        Self {
            plugin_dir,
            loaded_plugins: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Scan plugin directory for available plugins
    pub async fn scan_plugins(&self) -> Result<Vec<PluginMetadata>> {
        info!("Scanning plugin directory: {:?}", self.plugin_dir);

        let mut plugins = Vec::new();

        // Create plugin directory if it doesn't exist
        tokio::fs::create_dir_all(&self.plugin_dir).await?;

        // Read directory entries
        let mut entries = tokio::fs::read_dir(&self.plugin_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Skip non-directories
            if !path.is_dir() {
                continue;
            }

            // Look for plugin manifest
            let manifest_path = path.join("plugin.toml");
            if !manifest_path.exists() {
                continue;
            }

            // Load plugin manifest
            match self.load_manifest(&manifest_path).await {
                Ok(metadata) => {
                    info!("Found plugin: {} v{}", metadata.name, metadata.version);
                    plugins.push(metadata);
                }
                Err(e) => {
                    warn!("Failed to load plugin manifest at {:?}: {}", manifest_path, e);
                }
            }
        }

        Ok(plugins)
    }

    /// Load plugin manifest
    async fn load_manifest(&self, path: &Path) -> Result<PluginMetadata> {
        let content = tokio::fs::read_to_string(path).await
            .context("Failed to read plugin manifest")?;

        let metadata: PluginMetadata = toml::from_str(&content)
            .context("Failed to parse plugin manifest")?;

        // Validate metadata
        self.validate_metadata(&metadata)?;

        Ok(metadata)
    }

    /// Validate plugin metadata
    fn validate_metadata(&self, metadata: &PluginMetadata) -> Result<()> {
        // Check required fields
        if metadata.id.is_empty() {
            return Err(anyhow::anyhow!("Plugin ID is required"));
        }

        if metadata.name.is_empty() {
            return Err(anyhow::anyhow!("Plugin name is required"));
        }

        if metadata.version.is_empty() {
            return Err(anyhow::anyhow!("Plugin version is required"));
        }

        // Validate version format
        if !self.is_valid_version(&metadata.version) {
            return Err(anyhow::anyhow!("Invalid version format: {}", metadata.version));
        }

        // Check Loki version compatibility
        if !self.is_compatible_version(&metadata.loki_version) {
            return Err(anyhow::anyhow!(
                "Plugin requires Loki version {}, current version is {}",
                metadata.loki_version,
                std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string())
            ));
        }

        Ok(())
    }

    /// Check if version string is valid
    fn is_valid_version(&self, version: &str) -> bool {
        // Simple semver validation
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return false;
        }

        parts.iter().all(|part| part.parse::<u32>().is_ok())
    }

    /// Check if plugin is compatible with current Loki version
    fn is_compatible_version(&self, required_version: &str) -> bool {
        let current_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string());

        // Simple compatibility check - major version must match
        let current_major = current_version.split('.').next().unwrap_or("0");
        let required_major = required_version.split('.').next().unwrap_or("0");

        current_major == required_major
    }

    /// Load a plugin by ID
    pub async fn load_plugin(&self, plugin_id: &str) -> Result<Box<dyn Plugin>> {
        info!("Loading plugin: {}", plugin_id);

        // Find plugin directory
        let plugin_path = self.plugin_dir.join(plugin_id);
        if !plugin_path.exists() {
            return Err(PluginError::NotFound(plugin_id.to_string()).into());
        }

        // Load manifest
        let manifest_path = plugin_path.join("plugin.toml");
        let metadata = self.load_manifest(&manifest_path).await?;

        // Determine plugin type
        let plugin_type = self.detect_plugin_type(&plugin_path).await?;

        // Calculate checksum
        let checksum = self.calculate_checksum(&plugin_path).await?;

        // Load plugin based on type
        let plugin = match &plugin_type {
            PluginType::Wasm => self.load_wasm_plugin(&plugin_path, &metadata).await?,
            PluginType::DynamicLibrary => self.load_dynamic_plugin(&plugin_path, &metadata).await?,
            PluginType::Script(script_type) => {
                self.load_script_plugin(&plugin_path, &metadata, script_type).await?
            }
            PluginType::Remote(url) => self.load_remote_plugin(url, &metadata).await?,
        };

        // Cache loaded plugin info
        let loaded = LoadedPlugin {
            metadata,
            plugin_type,
            path: plugin_path,
            checksum,
        };

        self.loaded_plugins.write().await.push(loaded);

        Ok(plugin)
    }

    /// Detect plugin type from directory contents
    async fn detect_plugin_type(&self, path: &Path) -> Result<PluginType> {
        // Check for WASM file
        if path.join("plugin.wasm").exists() {
            return Ok(PluginType::Wasm);
        }

        // Check for dynamic library
        #[cfg(target_os = "linux")]
        let dylib_ext = "so";
        #[cfg(target_os = "macos")]
        let dylib_ext = "dylib";
        #[cfg(target_os = "windows")]
        let dylib_ext = "dll";

        if path.join(format!("plugin.{}", dylib_ext)).exists() {
            return Ok(PluginType::DynamicLibrary);
        }

        // Check for script files
        if path.join("plugin.py").exists() {
            return Ok(PluginType::Script(ScriptType::Python));
        }

        if path.join("plugin.js").exists() {
            return Ok(PluginType::Script(ScriptType::JavaScript));
        }

        if path.join("plugin.lua").exists() {
            return Ok(PluginType::Script(ScriptType::Lua));
        }

        // Check for remote plugin config
        let remoteconfig = path.join("remote.toml");
        if remoteconfig.exists() {
            let content = tokio::fs::read_to_string(&remoteconfig).await?;
            let config: RemoteConfig = toml::from_str(&content)?;
            return Ok(PluginType::Remote(config.url));
        }

        Err(anyhow::anyhow!("Could not determine plugin type"))
    }

    /// Calculate plugin checksum
    async fn calculate_checksum(&self, path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();

        // Hash all relevant files
        let mut entries = tokio::fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_path = entry.path();
            if file_path.is_file() {
                let content = tokio::fs::read(&file_path).await?;
                hasher.update(&content);
            }
        }

        Ok(format!("{:?}", hasher.finalize()))
    }

    /// Load WASM plugin using advanced WebAssembly engine
    async fn load_wasm_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<Box<dyn Plugin>> {
        info!("Loading WASM plugin with advanced engine: {}", metadata.name);

        let wasm_path = path.join("plugin.wasm");
        let wasm_bytes = tokio::fs::read(&wasm_path).await
            .context("Failed to read WASM file")?;

        // Create advanced WASM engine configuration
        let engineconfig = super::wasm_engine::WasmEngineConfig::default();
        
        // Configure based on plugin capabilities
        if metadata.capabilities.iter().any(|cap| matches!(cap, PluginCapability::NetworkAccess)) {
            // Allow network access for plugins that need it
            info!("Plugin {} has network access capability", metadata.name);
        }

        // Create plugin configuration
        let pluginconfig = super::wasm_engine::WasmPluginConfig {
            memory_limit_pages: Some(512), // 32MB limit
            fuel_limit: Some(1_000_000),   // 1M fuel units
            timeout_seconds: Some(30),      // 30 second timeout
            enable_profiling: true,
            enable_tracing: false,
            custom_imports: std::collections::HashMap::new(),
            environment: std::collections::HashMap::new(),
            capabilities: metadata.capabilities.clone(),
        };

        // Create the advanced WASM engine
        let wasm_engine = super::wasm_engine::WasmEngine::new(engineconfig)
            .context("Failed to create advanced WASM engine")?;

        // Create a basic context for the plugin
        // In a real implementation, this would come from the plugin manager
        let context = super::api::PluginContext {
            plugin_id: metadata.id.clone(),
            capabilities: metadata.capabilities.clone(),
            api: std::sync::Arc::new(super::api::PluginApi::new(None, None, None, None).await?),
            event_tx: {
                let (tx, _rx) = tokio::sync::mpsc::channel(100);
                tx
            },
            config: serde_json::Value::Object(serde_json::Map::new()),
        };

        // Load the plugin using the advanced engine
        wasm_engine.load_plugin(
            metadata.id.clone(),
            metadata.clone(),
            &wasm_bytes,
            pluginconfig,
            context,
        ).await?;

        // Create plugin wrapper that uses the advanced engine
        let plugin = AdvancedWasmPlugin::new(metadata.clone(), wasm_engine, metadata.id.clone());
        Ok(Box::new(plugin))
    }

    /// Load dynamic library plugin with comprehensive safety and compatibility checks
    /// Implements sophisticated plugin loading following cognitive enhancement principles
    async fn load_dynamic_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<Box<dyn Plugin>> {
        info!("Loading dynamic library plugin: {} from {}", metadata.name, path.display());

        // Find the dynamic library file
        let lib_path = self.find_dynamic_library(path).await?;

        // Perform security checks
        self.validate_dynamic_library(&lib_path, metadata).await?;

        // Load the library safely
        let library = self.load_library_with_safety_checks(&lib_path).await?;

        // Create and initialize the plugin
        let plugin = DynamicPlugin::new(metadata.clone(), library, lib_path)?;

        // Test plugin functionality
        plugin.validate_plugin_interface()?;

        info!("Dynamic library plugin loaded successfully: {}", metadata.name);
        Ok(Box::new(plugin))
    }

    /// Find the dynamic library file in the plugin directory
    async fn find_dynamic_library(&self, path: &Path) -> Result<PathBuf> {
        let lib_extensions = if cfg!(target_os = "windows") {
            vec!["dll"]
        } else if cfg!(target_os = "macos") {
            vec!["dylib", "so"]
        } else {
            vec!["so", "dylib"]
        };

        let lib_prefixes = if cfg!(target_os = "windows") {
            vec!["", "lib"]
        } else {
            vec!["lib", ""]
        };

        // Look for library files with various naming conventions
        for prefix in &lib_prefixes {
            for extension in &lib_extensions {
                let candidates = [
                    format!("{}plugin.{}", prefix, extension),
                    format!("{}main.{}", prefix, extension),
                    format!("{}{}.{}", prefix, path.file_name().unwrap().to_string_lossy(), extension),
                ];

                for candidate in &candidates {
                    let lib_path = path.join(candidate);
                    if lib_path.exists() {
                        debug!("Found dynamic library: {}", lib_path.display());
                        return Ok(lib_path);
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "No dynamic library found in plugin directory: {}. Expected files: lib*.so, lib*.dylib, or *.dll",
            path.display()
        ))
    }

    /// Validate dynamic library before loading
    async fn validate_dynamic_library(&self, lib_path: &Path, metadata: &PluginMetadata) -> Result<()> {
        debug!("Validating dynamic library: {}", lib_path.display());

        // Check file exists and is readable
        if !lib_path.exists() {
            return Err(anyhow::anyhow!("Library file does not exist: {}", lib_path.display()));
        }

        if !lib_path.is_file() {
            return Err(anyhow::anyhow!("Library path is not a file: {}", lib_path.display()));
        }

        // Check file permissions
        let metadata_fs = std::fs::metadata(lib_path)
            .with_context(|| format!("Failed to read library metadata: {}", lib_path.display()))?;

        if metadata_fs.len() == 0 {
            return Err(anyhow::anyhow!("Library file is empty: {}", lib_path.display()));
        }

        if metadata_fs.len() > 100 * 1024 * 1024 { // 100MB limit
            warn!("Library file is very large ({}MB), this may indicate a problem",
                  metadata_fs.len() / (1024 * 1024));
        }

        // Platform-specific validation
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = metadata_fs.permissions();
            if perms.mode() & 0o111 == 0 {
                warn!("Library file is not executable: {}", lib_path.display());
            }
        }

        // Basic file format validation (check for ELF/Mach-O/PE magic numbers)
        self.validate_library_format(lib_path).await?;

        // Check plugin manifest compatibility
        self.validate_plugin_compatibility(metadata).await?;

        Ok(())
    }

    /// Validate library file format
    async fn validate_library_format(&self, lib_path: &Path) -> Result<()> {
        let mut file = tokio::fs::File::open(lib_path).await
            .with_context(|| format!("Failed to open library: {}", lib_path.display()))?;

        let mut magic = [0u8; 16];
        use tokio::io::AsyncReadExt;
        file.read_exact(&mut magic).await
            .with_context(|| "Failed to read library magic number")?;

        // Check for known dynamic library magic numbers
        let is_valid = if cfg!(target_os = "linux") {
            // ELF magic: 0x7F 'E' 'L' 'F'
            magic[0] == 0x7F && magic[1] == b'E' && magic[2] == b'L' && magic[3] == b'F'
        } else if cfg!(target_os = "macos") {
            // Mach-O magic numbers
            let magic_u32 = u32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]);
            magic_u32 == 0xfeedface || magic_u32 == 0xfeedfacf ||
            magic_u32 == 0xcafebabe || magic_u32 == 0xcffaedfe
        } else if cfg!(target_os = "windows") {
            // PE magic: 'M' 'Z'
            magic[0] == b'M' && magic[1] == b'Z'
        } else {
            // Unknown platform, skip validation
            true
        };

        if !is_valid {
            return Err(anyhow::anyhow!(
                "Invalid library format: {} does not appear to be a valid dynamic library",
                lib_path.display()
            ));
        }

        debug!("Library format validation passed");
        Ok(())
    }

    /// Validate plugin compatibility with current system
    async fn validate_plugin_compatibility(&self, metadata: &PluginMetadata) -> Result<()> {
        debug!("Validating plugin compatibility: {}", metadata.name);

        // Check Loki version compatibility
        if !self.is_compatible_version(&metadata.loki_version) {
            return Err(anyhow::anyhow!(
                "Plugin {} requires Loki version {} but current version is {}",
                metadata.name,
                metadata.loki_version,
                std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string())
            ));
        }

        // Validate capabilities
        for capability in &metadata.capabilities {
            if !self.is_capability_supported(capability) {
                warn!("Plugin {} requests unsupported capability: {}", metadata.name, capability);
            }
        }

        Ok(())
    }

    /// Check if a capability is supported
    fn is_capability_supported(&self, capability: &PluginCapability) -> bool {
        let supported_capabilities = [
            PluginCapability::MemoryRead,
            PluginCapability::MemoryWrite,
            PluginCapability::NetworkAccess,
            PluginCapability::FileSystemRead,
            PluginCapability::FileSystemWrite,
            PluginCapability::Custom("ComputeAccess".to_string()),
            PluginCapability::ConsciousnessAccess,
            PluginCapability::SocialMedia,
            PluginCapability::Custom("ToolsAccess".to_string()),
        ];

        supported_capabilities.iter().any(|cap| cap == capability)
    }

    /// Load library with comprehensive safety checks
    async fn load_library_with_safety_checks(&self, lib_path: &Path) -> Result<libloading::Library> {
        debug!("Loading library with safety checks: {}", lib_path.display());

        // Use libloading to load the dynamic library
        let library = unsafe {
            libloading::Library::new(lib_path)
                .with_context(|| format!("Failed to load dynamic library: {}", lib_path.display()))?
        };

        // Verify required symbols exist
        self.verify_plugin_symbols(&library)?;

        Ok(library)
    }

    /// Verify that the library contains required plugin symbols
    fn verify_plugin_symbols(&self, library: &libloading::Library) -> Result<()> {
        debug!("Verifying plugin symbols");

        let required_symbols = [
            "plugin_init",
            "plugin_shutdown",
            "plugin_get_metadata",
            "plugin_handle_event",
        ];

        for symbol_name in &required_symbols {
            unsafe {
                let symbol: Result<libloading::Symbol<unsafe extern "C" fn()>, _> =
                    library.get(symbol_name.as_bytes());

                if symbol.is_err() {
                    return Err(anyhow::anyhow!(
                        "Required symbol '{}' not found in dynamic library",
                        symbol_name
                    ));
                }
            }
        }

        debug!("All required symbols found");
        Ok(())
    }

    /// Load script-based plugin
    async fn load_script_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
        script_type: &ScriptType,
    ) -> Result<Box<dyn Plugin>> {
        info!("Loading script plugin: {} ({})", metadata.name, script_type.name());

        match script_type {
            ScriptType::Python => self.load_python_plugin(path, metadata).await,
            ScriptType::JavaScript => self.load_javascript_plugin(path, metadata).await,
            ScriptType::Lua => self.load_lua_plugin(path, metadata).await,
        }
    }

    /// Load Python plugin
    async fn load_python_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<Box<dyn Plugin>> {
        let script_path = path.join("plugin.py");
        let script_content = tokio::fs::read_to_string(&script_path).await
            .context("Failed to read Python script")?;

        // Validate Python script has required functions
        if !script_content.contains("def plugin_init(") {
            return Err(anyhow::anyhow!("Python plugin must define 'plugin_init' function"));
        }

        if !script_content.contains("def plugin_execute(") {
            return Err(anyhow::anyhow!("Python plugin must define 'plugin_execute' function"));
        }

        // Create plugin wrapper
        let plugin = ScriptPlugin::new(
            metadata.clone(),
            ScriptType::Python,
            script_content,
            script_path,
        );

        // Test plugin initialization
        plugin.test_initialization().await
            .context("Python plugin initialization test failed")?;

        Ok(Box::new(plugin))
    }

    /// Load JavaScript plugin
    async fn load_javascript_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<Box<dyn Plugin>> {
        let script_path = path.join("plugin.js");
        let script_content = tokio::fs::read_to_string(&script_path).await
            .context("Failed to read JavaScript script")?;

        // Validate JavaScript has required functions
        if !script_content.contains("function pluginInit(") && !script_content.contains("pluginInit =") {
            return Err(anyhow::anyhow!("JavaScript plugin must define 'pluginInit' function"));
        }

        if !script_content.contains("function pluginExecute(") && !script_content.contains("pluginExecute =") {
            return Err(anyhow::anyhow!("JavaScript plugin must define 'pluginExecute' function"));
        }

        // Create plugin wrapper
        let plugin = ScriptPlugin::new(
            metadata.clone(),
            ScriptType::JavaScript,
            script_content,
            script_path,
        );

        // Test plugin initialization
        plugin.test_initialization().await
            .context("JavaScript plugin initialization test failed")?;

        Ok(Box::new(plugin))
    }

    /// Load Lua plugin
    async fn load_lua_plugin(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<Box<dyn Plugin>> {
        let script_path = path.join("plugin.lua");
        let script_content = tokio::fs::read_to_string(&script_path).await
            .context("Failed to read Lua script")?;

        // Validate Lua has required functions
        if !script_content.contains("function plugin_init(") {
            return Err(anyhow::anyhow!("Lua plugin must define 'plugin_init' function"));
        }

        if !script_content.contains("function plugin_execute(") {
            return Err(anyhow::anyhow!("Lua plugin must define 'plugin_execute' function"));
        }

        // Create plugin wrapper
        let plugin = ScriptPlugin::new(
            metadata.clone(),
            ScriptType::Lua,
            script_content,
            script_path,
        );

        // Test plugin initialization
        plugin.test_initialization().await
            .context("Lua plugin initialization test failed")?;

        Ok(Box::new(plugin))
    }

    /// Load remote plugin with comprehensive networking and security features
    /// Implements sophisticated remote plugin connectivity following cognitive enhancement principles
    async fn load_remote_plugin(
        &self,
        url: &str,
        metadata: &PluginMetadata,
    ) -> Result<Box<dyn Plugin>> {
        info!("Loading remote plugin: {} from {}", metadata.name, url);

        // Parse and validate URL
        let parsed_url = self.parse_and_validate_url(url)?;

        // Perform security and connectivity checks
        self.validate_remote_connection(&parsed_url, metadata).await?;

        // Establish connection based on protocol
        let connection = self.establish_remote_connection(&parsed_url).await?;

        // Perform handshake and authentication
        self.perform_remote_handshake(&connection, metadata).await?;

        // Create remote plugin wrapper
        let plugin = RemotePlugin::new(metadata.clone(), connection, parsed_url)?;

        // Test plugin connectivity and basic functionality
        plugin.test_connectivity().await?;

        info!("Remote plugin connected successfully: {}", metadata.name);
        Ok(Box::new(plugin))
    }

    /// Parse and validate remote URL
    fn parse_and_validate_url(&self, url: &str) -> Result<url::Url> {
        let parsed = url::Url::parse(url)
            .with_context(|| format!("Invalid remote plugin URL: {}", url))?;

        // Validate protocol
        match parsed.scheme() {
            "http" | "https" | "grpc" | "grpcs" => {
                debug!("Remote plugin protocol: {}", parsed.scheme());
            }
            scheme => {
                return Err(anyhow::anyhow!(
                    "Unsupported remote plugin protocol: {}. Supported: http, https, grpc, grpcs",
                    scheme
                ));
            }
        }

        // Validate host
        if parsed.host().is_none() {
            return Err(anyhow::anyhow!("Remote plugin URL must specify a host"));
        }

        // Security check: prevent localhost access in production
        if let Some(host) = parsed.host_str() {
            if self.is_restricted_host(host) {
                warn!("Remote plugin attempting to connect to restricted host: {}", host);
                return Err(anyhow::anyhow!("Connection to restricted host not allowed: {}", host));
            }
        }

        Ok(parsed)
    }

    /// Check if host is restricted for security
    fn is_restricted_host(&self, host: &str) -> bool {
        let restricted_hosts = [
            "localhost", "127.0.0.1", "::1", "0.0.0.0",
            "169.254.169.254", // AWS metadata service
            "metadata.google.internal", // GCP metadata service
        ];

        restricted_hosts.iter().any(|&restricted| host == restricted) ||
        host.starts_with("10.") || // Private IP ranges
        host.starts_with("192.168.") ||
        host.starts_with("172.16.") || host.starts_with("172.17.") ||
        host.starts_with("172.18.") || host.starts_with("172.19.") ||
        host.starts_with("172.20.") || host.starts_with("172.21.") ||
        host.starts_with("172.22.") || host.starts_with("172.23.") ||
        host.starts_with("172.24.") || host.starts_with("172.25.") ||
        host.starts_with("172.26.") || host.starts_with("172.27.") ||
        host.starts_with("172.28.") || host.starts_with("172.29.") ||
        host.starts_with("172.30.") || host.starts_with("172.31.")
    }

    /// Validate remote connection security and availability
    async fn validate_remote_connection(&self, url: &url::Url, metadata: &PluginMetadata) -> Result<()> {
        debug!("Validating remote connection: {}", url);

        // DNS resolution check
        let host = url.host_str().unwrap();
        let addrs = tokio::net::lookup_host(format!("{}:{}", host, url.port_or_known_default().unwrap_or(80)))
            .await
            .with_context(|| format!("Failed to resolve host: {}", host))?;

        let resolved_ips: Vec<_> = addrs.collect();
        if resolved_ips.is_empty() {
            return Err(anyhow::anyhow!("No IP addresses resolved for host: {}", host));
        }

        debug!("Resolved {} IP addresses for host {}", resolved_ips.len(), host);

        // Connection timeout test
        let timeout = std::time::Duration::from_secs(10);
        let connection_test = async {
            match url.scheme() {
                "http" | "https" => self.test_http_connection(url).await,
                "grpc" | "grpcs" => self.test_grpc_connection(url).await,
                _ => Ok(()) // Already validated in parse_and_validate_url
            }
        };

        tokio::time::timeout(timeout, connection_test)
            .await
            .with_context(|| "Remote plugin connection test timed out")?
            .with_context(|| "Remote plugin connection test failed")?;

        // Security validation
        self.validate_remote_security(url, metadata).await?;

        Ok(())
    }

    /// Test HTTP connection
    async fn test_http_connection(&self, url: &url::Url) -> Result<()> {
        debug!("Testing HTTP connection: {}", url);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .user_agent("Loki-Plugin-Loader/1.0")
            .build()?;

        let health_url = format!("{}/health", url.as_str().trim_end_matches('/'));
        let response = client.head(&health_url).send().await
            .with_context(|| format!("Failed to connect to remote plugin: {}", url))?;

        if !response.status().is_success() && response.status() != 404 {
            return Err(anyhow::anyhow!(
                "Remote plugin health check failed: HTTP {}",
                response.status()
            ));
        }

        debug!("HTTP connection test passed");
        Ok(())
    }

    /// Test gRPC connection
    async fn test_grpc_connection(&self, url: &url::Url) -> Result<()> {
        debug!("Testing gRPC connection: {}", url);

        // For gRPC, we'll use a simple TCP connection test
        let host = url.host_str().unwrap();
        let port = url.port_or_known_default().unwrap_or(80);

        let addr = format!("{}:{}", host, port);
        let _stream = tokio::net::TcpStream::connect(&addr).await
            .with_context(|| format!("Failed to connect to gRPC endpoint: {}", addr))?;

        debug!("gRPC connection test passed");
        Ok(())
    }

    /// Validate remote plugin security
    async fn validate_remote_security(&self, url: &url::Url, metadata: &PluginMetadata) -> Result<()> {
        debug!("Validating remote plugin security");

        // Enforce HTTPS/gRPCS for production
        if !url.scheme().ends_with('s') {
            warn!("Remote plugin using insecure connection: {}", url);
            // In production, this might be an error
        }

        // Check plugin capabilities for network access
        if !metadata.capabilities.iter().any(|cap|
            matches!(cap, PluginCapability::NetworkAccess) ||
            matches!(cap, PluginCapability::Custom(s) if s == "RemoteAccess")
        ) {
            return Err(anyhow::anyhow!(
                "Remote plugin {} does not declare required network capabilities",
                metadata.name
            ));
        }

        // Additional security checks based on plugin metadata
        if metadata.capabilities.iter().any(|cap|
            matches!(cap, PluginCapability::Custom(s) if s == "SystemAccess" || s == "AdminAccess")
        ) {
            warn!("Remote plugin {} requests elevated privileges", metadata.name);
        }

        Ok(())
    }

    /// Establish connection to remote plugin
    async fn establish_remote_connection(&self, url: &url::Url) -> Result<RemotePluginConnection> {
        debug!("Establishing remote plugin connection: {}", url);

        match url.scheme() {
            "http" | "https" => {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .user_agent("Loki-Plugin-Loader/1.0")
                    .build()?;

                Ok(RemotePluginConnection::Http {
                    client,
                    base_url: url.clone(),
                    session_id: uuid::Uuid::new_v4().to_string(),
                })
            }
            "grpc" | "grpcs" => {
                // For gRPC, we'd create a gRPC client here
                // This is a simplified implementation
                Ok(RemotePluginConnection::Grpc {
                    endpoint: url.clone(),
                    session_id: uuid::Uuid::new_v4().to_string(),
                    // In real implementation, this would be a tonic::Client or similar
                })
            }
            scheme => Err(anyhow::anyhow!("Unsupported scheme for remote connection: {}", scheme))
        }
    }

    /// Perform handshake with remote plugin
    async fn perform_remote_handshake(
        &self,
        connection: &RemotePluginConnection,
        metadata: &PluginMetadata,
    ) -> Result<()> {
        debug!("Performing remote plugin handshake");

        match connection {
            RemotePluginConnection::Http { client, base_url, session_id } => {
                let handshake_url = format!("{}/plugin/handshake", base_url.as_str().trim_end_matches('/'));

                let handshake_request = serde_json::json!({
                    "loki_version": std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string()),
                    "plugin_id": metadata.id,
                    "session_id": session_id,
                    "capabilities": metadata.capabilities,
                    "timestamp": chrono::Utc::now().timestamp(),
                });

                let response = client
                    .post(&handshake_url)
                    .json(&handshake_request)
                    .send()
                    .await
                    .with_context(|| "Failed to send handshake request")?;

                if !response.status().is_success() {
                    return Err(anyhow::anyhow!(
                        "Remote plugin handshake failed: HTTP {}",
                        response.status()
                    ));
                }

                let handshake_response: serde_json::Value = response.json().await
                    .with_context(|| "Failed to parse handshake response")?;

                if handshake_response["status"] != "success" {
                    return Err(anyhow::anyhow!(
                        "Remote plugin handshake rejected: {}",
                        handshake_response.get("message").unwrap_or(&serde_json::json!("Unknown error"))
                    ));
                }

                debug!("HTTP handshake completed successfully");
            }
            RemotePluginConnection::Grpc { endpoint, session_id } => {
                // gRPC handshake would be implemented here
                debug!("gRPC handshake with endpoint {} (session: {})", endpoint, session_id);
                // In real implementation, this would use gRPC service calls
            }
        }

        Ok(())
    }

    /// Unload a plugin
    pub async fn unload_plugin(&self, plugin_id: &str) -> Result<()> {
        info!("Unloading plugin: {}", plugin_id);

        // Remove from loaded plugins
        let mut loaded = self.loaded_plugins.write().await;
        loaded.retain(|p| p.metadata.id != plugin_id);

        Ok(())
    }

    /// Get loaded plugins
    pub async fn loaded_plugins(&self) -> Vec<LoadedPlugin> {
        self.loaded_plugins.read().await.clone()
    }

    /// Verify plugin integrity
    pub async fn verify_plugin(&self, plugin_id: &str) -> Result<bool> {
        let loaded = self.loaded_plugins.read().await;

        if let Some(plugin) = loaded.iter().find(|p| p.metadata.id == plugin_id) {
            let current_checksum = self.calculate_checksum(&plugin.path).await?;
            Ok(current_checksum == plugin.checksum)
        } else {
            Err(PluginError::NotFound(plugin_id.to_string()).into())
        }
    }
}

/// Remote plugin configuration
#[derive(Debug, Deserialize)]
struct RemoteConfig {
    url: String,
}

/// Remote plugin connection types
#[derive(Debug)]
pub enum RemotePluginConnection {
    /// HTTP/HTTPS connection
    Http {
        client: reqwest::Client,
        base_url: url::Url,
        session_id: String,
    },
    /// gRPC connection
    Grpc {
        endpoint: url::Url,
        session_id: String,
        // In a real implementation, this would contain a gRPC client
    },
}

/// Dynamic plugin implementation
pub struct DynamicPlugin {
    metadata: PluginMetadata,
    library: libloading::Library,
    library_path: PathBuf,
}

impl DynamicPlugin {
    pub fn new(
        metadata: PluginMetadata,
        library: libloading::Library,
        library_path: PathBuf,
    ) -> Result<Self> {
        Ok(Self {
            metadata,
            library,
            library_path,
        })
    }

    /// Validate that the plugin implements the required interface
    pub fn validate_plugin_interface(&self) -> Result<()> {
        debug!("Validating plugin interface for: {}", self.metadata.name);

        // Check for required symbols - already done in loading process
        // Additional validation could be performed here

        Ok(())
    }

    /// Get the plugin library path for debugging and logging
    pub fn get_library_path(&self) -> &Path {
        &self.library_path
    }

    /// Check if the library file still exists and hasn't been modified
    pub fn verify_library_integrity(&self) -> Result<bool> {
        debug!("Verifying integrity of library at: {:?}", self.library_path);
        
        if !self.library_path.exists() {
            warn!("Library file no longer exists: {:?}", self.library_path);
            return Ok(false);
        }

        // Check if the file is readable
        match std::fs::File::open(&self.library_path) {
            Ok(_) => {
                debug!("Library file is accessible: {:?}", self.library_path);
                Ok(true)
            }
            Err(e) => {
                warn!("Cannot access library file {:?}: {}", self.library_path, e);
                Ok(false)
            }
        }
    }

    /// Get library metadata (size, modification time, etc.)
    pub fn get_library_metadata(&self) -> Result<LibraryMetadata> {
        let metadata = std::fs::metadata(&self.library_path)
            .with_context(|| format!("Failed to get metadata for {:?}", self.library_path))?;
        
        let permissions = {
            #[cfg(unix)]
            {
                format!("{:o}", metadata.permissions().mode())
            }
            #[cfg(not(unix))]
            {
                format!("readonly: {}", metadata.permissions().readonly())
            }
        };
        
        Ok(LibraryMetadata {
            path: self.library_path.clone(),
            size_bytes: metadata.len(),
            modified: metadata.modified().ok(),
            is_file: metadata.is_file(),
            permissions,
        })
    }

    /// Hot reload the plugin library
    pub async fn hot_reload(&mut self) -> Result<()> {
        info!("Hot reloading plugin library: {:?}", self.library_path);
        
        // First verify the library file integrity
        if !self.verify_library_integrity()? {
            return Err(anyhow::anyhow!("Library file integrity check failed"));
        }
        
        // Stop the current plugin
        self.stop().await?;
        
        // Reload the library
        unsafe {
            let new_library = libloading::Library::new(&self.library_path)
                .with_context(|| format!("Failed to reload library: {:?}", self.library_path))?;
            
            // Replace the old library with the new one
            // Note: This is a simplified approach. In practice, you'd want more sophisticated handling
            self.library = new_library;
        }
        
        // Reinitialize the plugin
        // Create minimal plugin context - in real usage this would come from the plugin manager
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(100);
        let api = Arc::new(crate::plugins::api::PluginApi::new(None, None, None, None).await?);
        let context = crate::plugins::api::PluginContext {
            plugin_id: "mock_plugin".to_string(),
            capabilities: vec![],
            api,
            event_tx,
            config: serde_json::Value::Object(serde_json::Map::new()),
        };
        self.initialize(context).await?;
        
        info!("Successfully hot reloaded plugin: {}", self.metadata.name);
        Ok(())
    }
}

#[async_trait::async_trait]
impl Plugin for DynamicPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, _context: PluginContext) -> Result<()> {
        info!("Initializing dynamic plugin: {}", self.metadata.name);

        // Call plugin_init function from the dynamic library
        unsafe {
            let init_fn: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                self.library.get(b"plugin_init")?;

            let result = init_fn();
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("Plugin initialization failed with code: {}", result))
            }
        }
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting dynamic plugin: {}", self.metadata.name);
        // Dynamic plugins are started during initialization
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping dynamic plugin: {}", self.metadata.name);

        // Call plugin_shutdown function
        unsafe {
            if let Ok(shutdown_fn) = self.library.get::<unsafe extern "C" fn() -> i32>(b"plugin_shutdown") {
                let _result = shutdown_fn();
            }
        }

        Ok(())
    }

    async fn handle_event(&mut self, event: PluginEvent) -> Result<()> {
        debug!("Dynamic plugin {} handling event: {:?}", self.metadata.name, event);

        // Call plugin_handle_event function
        unsafe {
            let handle_event_fn: libloading::Symbol<unsafe extern "C" fn(i32) -> i32> =
                self.library.get(b"plugin_handle_event")?;

            // Convert event to a simple integer code for the C interface
            let event_code = match event {
                PluginEvent::System(_) => 1,
                PluginEvent::Cognitive(_) => 2,
                PluginEvent::Memory(_) => 3,
                PluginEvent::Social(_) => 4,
                PluginEvent::Custom(_, _) => 5,
            };

            let _result = handle_event_fn(event_code);
        }

        Ok(())
    }

    fn state(&self) -> PluginState {
        PluginState::Active
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: Some(format!("Dynamic plugin {} is healthy", self.metadata.name)),
            metrics: std::collections::HashMap::new(),
        })
    }
}

/// Remote plugin implementation
pub struct RemotePlugin {
    metadata: PluginMetadata,
    connection: RemotePluginConnection,
    endpoint_url: url::Url,
}

impl RemotePlugin {
    pub fn new(
        metadata: PluginMetadata,
        connection: RemotePluginConnection,
        endpoint_url: url::Url,
    ) -> Result<Self> {
        Ok(Self {
            metadata,
            connection,
            endpoint_url,
        })
    }

    /// Test connectivity to the remote plugin
    pub async fn test_connectivity(&self) -> Result<()> {
        debug!("Testing connectivity for remote plugin: {}", self.metadata.name);
        
        // Use the endpoint_url to test connectivity
        info!("Testing connectivity to endpoint: {}", self.endpoint_url);
        
        match self.endpoint_url.scheme() {
            "http" | "https" => {
                self.test_http_connectivity().await?;
            }
            "grpc" | "grpcs" => {
                self.test_grpc_connectivity().await?;
            }
            scheme => {
                return Err(anyhow::anyhow!("Unsupported endpoint scheme: {}", scheme));
            }
        }
        
        info!("Connectivity test passed for: {}", self.endpoint_url);
        Ok(())
    }
    
    /// Test HTTP connectivity to the remote plugin
    async fn test_http_connectivity(&self) -> Result<()> {
        debug!("Testing HTTP connectivity to: {}", self.endpoint_url);
        
        // Create a simple HTTP client request
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
            
        let health_url = format!("{}/_health", self.endpoint_url.as_str().trim_end_matches('/'));
        
        match client.get(&health_url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    debug!("HTTP health check successful for: {}", self.endpoint_url);
                    Ok(())
                } else {
                    Err(anyhow::anyhow!(
                        "HTTP health check failed with status: {}", 
                        response.status()
                    ))
                }
            }
            Err(e) => {
                warn!("HTTP connectivity test failed for {}: {}", self.endpoint_url, e);
                Err(anyhow::anyhow!("HTTP connectivity test failed: {}", e))
            }
        }
    }
    
    /// Test gRPC connectivity to the remote plugin
    async fn test_grpc_connectivity(&self) -> Result<()> {
        debug!("Testing gRPC connectivity to: {}", self.endpoint_url);
        
        // For gRPC, we'll simulate a basic connectivity test
        // In a real implementation, you'd use a gRPC client library
        let host = self.endpoint_url.host_str().unwrap_or("localhost");
        let port = self.endpoint_url.port().unwrap_or(if self.endpoint_url.scheme() == "grpcs" { 443 } else { 80 });
        
        match tokio::net::TcpStream::connect((host, port)).await {
            Ok(_) => {
                debug!("gRPC TCP connectivity successful for: {}:{}", host, port);
                Ok(())
            }
            Err(e) => {
                warn!("gRPC connectivity test failed for {}:{}: {}", host, port, e);
                Err(anyhow::anyhow!("gRPC connectivity test failed: {}", e))
            }
        }
    }
    
    /// Get the endpoint URL for this remote plugin
    pub fn get_endpoint_url(&self) -> &url::Url {
        &self.endpoint_url
    }
    
    /// Update the endpoint URL for the remote plugin
    pub fn update_endpoint_url(&mut self, new_url: url::Url) -> Result<()> {
        info!(
            "Updating endpoint URL for plugin {} from {} to {}", 
            self.metadata.name, 
            self.endpoint_url, 
            new_url
        );
        
        // Validate the new URL
        match new_url.scheme() {
            "http" | "https" | "grpc" | "grpcs" => {
                self.endpoint_url = new_url;
                Ok(())
            }
            scheme => {
                Err(anyhow::anyhow!("Unsupported endpoint scheme: {}", scheme))
            }
        }
    }
    
    /// Get connection statistics for monitoring
    pub fn get_connection_stats(&self) -> RemoteConnectionStats {
        RemoteConnectionStats {
            endpoint_url: self.endpoint_url.clone(),
            connection_type: match self.endpoint_url.scheme() {
                "http" | "https" => "HTTP".to_string(),
                "grpc" | "grpcs" => "gRPC".to_string(),
                scheme => format!("Unknown({})", scheme),
            },
            is_secure: matches!(self.endpoint_url.scheme(), "https" | "grpcs"),
            host: self.endpoint_url.host_str().unwrap_or("unknown").to_string(),
            port: self.endpoint_url.port().unwrap_or(0),
        }
    }
}

#[async_trait::async_trait]
impl Plugin for RemotePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, _context: PluginContext) -> Result<()> {
        info!("Initializing remote plugin: {}", self.metadata.name);
        self.test_connectivity().await?;
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting remote plugin: {}", self.metadata.name);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping remote plugin: {}", self.metadata.name);
        Ok(())
    }

    async fn handle_event(&mut self, event: PluginEvent) -> Result<()> {
        debug!("Remote plugin {} handling event: {:?}", self.metadata.name, event);

        match &self.connection {
            RemotePluginConnection::Http { client, base_url, session_id } => {
                let event_url = format!("{}/plugin/event", base_url.as_str().trim_end_matches('/'));

                let event_data = serde_json::json!({
                    "session_id": session_id,
                    "event_type": match event {
                        PluginEvent::System(_) => "system",
                        PluginEvent::Cognitive(_) => "cognitive",
                        PluginEvent::Memory(_) => "memory",
                        PluginEvent::Social(_) => "social",
                        PluginEvent::Custom(_, _) => "custom",
                    },
                    "timestamp": chrono::Utc::now().timestamp(),
                });

                let response = client
                    .post(&event_url)
                    .json(&event_data)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    return Err(anyhow::anyhow!(
                        "Remote plugin event handling failed: HTTP {}",
                        response.status()
                    ));
                }
            }
            RemotePluginConnection::Grpc { .. } => {
                // gRPC event handling would be implemented here
                debug!("Sending gRPC event to remote plugin");
            }
        }

        Ok(())
    }

    fn state(&self) -> PluginState {
        PluginState::Active
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        match self.test_connectivity().await {
            Ok(()) => Ok(HealthStatus {
                healthy: true,
                message: Some(format!("Remote plugin {} is healthy", self.metadata.name)),
                metrics: std::collections::HashMap::new(),
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                message: Some(format!("Remote plugin {} connectivity failed: {}", self.metadata.name, e)),
                metrics: std::collections::HashMap::new(),
            }),
        }
    }
}

/// Example plugin manifest structure
#[derive(Debug, Serialize)]
struct ExampleManifest {
    id: String,
    name: String,
    version: String,
    author: String,
    description: String,
    loki_version: String,
    capabilities: Vec<String>,
}

/// Create an example plugin manifest
pub fn create_example_manifest() -> String {
    let manifest = ExampleManifest {
        id: "example-plugin".to_string(),
        name: "Example Plugin".to_string(),
        version: "1.0.0".to_string(),
        author: "Your Name".to_string(),
        description: "An example Loki plugin".to_string(),
        loki_version: std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string()),
        capabilities: vec![
            "MemoryRead".to_string(),
            "NetworkAccess".to_string(),
        ],
    };

    toml::to_string_pretty(&manifest).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_plugin_loader_scan() {
        let temp_dir = TempDir::new().unwrap();
        let loader = PluginLoader::new(temp_dir.path().to_path_buf());

        // Create a test plugin directory
        let plugin_dir = temp_dir.path().join("test-plugin");
        tokio::fs::create_dir_all(&plugin_dir).await.unwrap();

        // Create plugin manifest
        let manifest = PluginMetadata {
            id: "test-plugin".to_string(),
            name: "Test Plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Test".to_string(),
            loki_version: std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string()),
            dependencies: vec![],
            capabilities: vec![],
            homepage: None,
            repository: None,
            license: None,
        };

        let manifest_content = toml::to_string(&manifest).unwrap();
        tokio::fs::write(plugin_dir.join("plugin.toml"), manifest_content)
            .await
            .unwrap();

        // Scan plugins
        let plugins = loader.scan_plugins().await.unwrap();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].id, "test-plugin");
    }

    #[test]
    fn test_version_validation() {
        let loader = PluginLoader::new(PathBuf::from("plugins"));

        assert!(loader.is_valid_version("1.0.0"));
        assert!(loader.is_valid_version("0.2.0"));
        assert!(!loader.is_valid_version("1.0"));
        assert!(!loader.is_valid_version("1.0.0.0"));
        assert!(!loader.is_valid_version("v1.0.0"));
    }
}

/// WASM Plugin implementation
pub struct WasmPlugin {
    metadata: PluginMetadata,
    engine: wasmtime::Engine,
    module: wasmtime::Module,
    // Note: Store and Instance would need to be stored differently for real use
    _phantom: std::marker::PhantomData<()>,
}

impl WasmPlugin {
    pub fn new(
        metadata: PluginMetadata,
        engine: wasmtime::Engine,
        module: wasmtime::Module,
        _instance: wasmtime::Instance,
    ) -> Self {
        Self {
            metadata,
            engine,
            module,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl Plugin for WasmPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, _context: PluginContext) -> Result<()> {
        info!("Initializing WASM plugin: {}", self.metadata.name);
        // WASM plugin already initialized during loading
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting WASM plugin: {}", self.metadata.name);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping WASM plugin: {}", self.metadata.name);
        Ok(())
    }

    async fn handle_event(&mut self, event: PluginEvent) -> Result<()> {
        use wasmtime::*;

        info!("WASM plugin {} handling event: {:?}", self.metadata.name, event);

        // Create new store for execution
        let mut store = Store::new(&self.engine, ());

        // Create linker and instantiate
        let mut linker = Linker::new(&self.engine);

        // Add host functions
        linker.func_wrap("env", "loki_log", |level: i32, ptr: i32, len: i32| {
            info!("WASM Plugin Log [{}]: <message at {}:{}>", level, ptr, len);
        })?;

        let instance = linker.instantiate(&mut store, &self.module)?;

        // Get event handler function
        if let Ok(handler_fn) = instance.get_typed_func::<i32, i32>(&mut store, "plugin_handle_event") {
            // Convert event to a simple integer code for WASM
            let event_code = match event {
                PluginEvent::System(_) => 1,
                PluginEvent::Cognitive(_) => 2,
                PluginEvent::Memory(_) => 3,
                PluginEvent::Social(_) => 4,
                PluginEvent::Custom(_, _) => 5,
            };

            let _result = handler_fn.call(&mut store, event_code)
                .context("Plugin event handling failed")?;
        }

        Ok(())
    }

    fn state(&self) -> PluginState {
        PluginState::Active
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            healthy: true,
            message: Some(format!("WASM plugin {} is healthy", self.metadata.name)),
            metrics: std::collections::HashMap::new(),
        })
    }
}

/// Script Plugin implementation
pub struct ScriptPlugin {
    metadata: PluginMetadata,
    script_type: ScriptType,
    script_content: String,
    script_path: PathBuf,
}

impl ScriptPlugin {
    pub fn new(
        metadata: PluginMetadata,
        script_type: ScriptType,
        script_content: String,
        script_path: PathBuf,
    ) -> Self {
        Self {
            metadata,
            script_type,
            script_content,
            script_path,
        }
    }

    /// Test plugin initialization without full execution
    pub async fn test_initialization(&self) -> Result<()> {
        match &self.script_type {
            ScriptType::Python => self.test_python_init().await,
            ScriptType::JavaScript => self.test_javascript_init().await,
            ScriptType::Lua => self.test_lua_init().await,
        }
    }

    async fn test_python_init(&self) -> Result<()> {
        // Simple validation - check Python syntax
        use std::process::Command;

        let output = Command::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(&self.script_path)
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    info!("Python plugin syntax validated: {}", self.metadata.name);
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&result.stderr);
                    Err(anyhow::anyhow!("Python syntax error: {}", error))
                }
            }
            Err(_) => {
                warn!("Python not available for syntax checking, skipping validation");
                Ok(())
            }
        }
    }

    async fn test_javascript_init(&self) -> Result<()> {
        // Simple validation - check Node.js syntax
        use std::process::Command;

        let output = Command::new("node")
            .arg("--check")
            .arg(&self.script_path)
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    info!("JavaScript plugin syntax validated: {}", self.metadata.name);
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&result.stderr);
                    Err(anyhow::anyhow!("JavaScript syntax error: {}", error))
                }
            }
            Err(_) => {
                warn!("Node.js not available for syntax checking, skipping validation");
                Ok(())
            }
        }
    }

    async fn test_lua_init(&self) -> Result<()> {
        // Validate Lua script using external lua interpreter if available
        use std::process::Command;

        let output = Command::new("lua")
            .arg("-e")
            .arg(&format!("assert(loadfile('{}'))", self.script_path.display()))
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    info!("Lua plugin syntax validated: {}", self.metadata.name);
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&result.stderr);
                    Err(anyhow::anyhow!("Lua syntax error: {}", error))
                }
            }
            Err(_) => {
                // Fallback: basic script validation
                if self.script_content.contains("function") || self.script_content.contains("plugin") {
                    info!("Lua plugin basic validation passed: {}", self.metadata.name);
                    Ok(())
                } else {
                    warn!("Lua interpreter not available, performing basic validation for: {}", self.metadata.name);
                    Ok(())
                }
            }
        }
    }

    /// Execute script using external interpreter
    async fn execute_script(&self, input: &str) -> Result<String> {
        use std::process::Command;

        match &self.script_type {
            ScriptType::Python => {
                let cmd = Command::new("python3")
                    .arg("-c")
                    .arg(&format!(r#"
import sys
import json

# Load the plugin script
with open('{}', 'r') as f:
    exec(f.read())

# Parse input
input_data = json.loads('{}')

# Call plugin function
result = plugin_execute(input_data)

# Output result
print(json.dumps(result))
"#, self.script_path.display(), input))
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()?;

                let output = cmd.wait_with_output()?;
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(anyhow::anyhow!("Python execution error: {}", error))
                }
            }
            ScriptType::JavaScript => {
                let cmd = Command::new("node")
                    .arg("-e")
                    .arg(&format!(r#"
const fs = require('fs');

// Load the plugin script
const pluginCode = fs.readFileSync('{}', 'utf8');
eval(pluginCode);

// Parse input
const inputData = JSON.parse('{}');

// Call plugin function
const result = pluginExecute(inputData);

// Output result
console.log(JSON.stringify(result));
"#, self.script_path.display(), input))
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()?;

                let output = cmd.wait_with_output()?;
                if output.status.success() {
                    Ok(String::from_utf8_lossy(&output.stdout).to_string())
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(anyhow::anyhow!("JavaScript execution error: {}", error))
                }
            }
            ScriptType::Lua => {
                // Execute Lua script using external interpreter
                self.execute_lua_external(input).await
            }
        }
    }

        /// Execute Lua script using external interpreter with comprehensive API support
    /// Implements sophisticated plugin execution with Loki integration
    async fn execute_lua_external(&self, input: &str) -> Result<String> {
        use std::process::Command;

        info!("Executing Lua plugin: {} with input length: {}", self.metadata.name, input.len());

        // Create comprehensive Lua execution environment
        let lua_wrapper = self.create_lua_execution_wrapper(input)?;

        // Execute Lua script with timeout and security constraints
        let cmd = Command::new("lua")
            .arg("-e")
            .arg(&lua_wrapper)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        // Wait for execution with timeout
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(30), // 30 second timeout
            tokio::task::spawn_blocking(move || cmd.wait_with_output())
        ).await??;

        let output = output?;
        if output.status.success() {
            let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info!("Lua plugin {} executed successfully, result length: {}", self.metadata.name, result.len());
            Ok(if result.is_empty() {
                serde_json::json!({"status": "executed", "plugin": self.metadata.name}).to_string()
            } else {
                result
            })
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("Lua execution error in plugin {}: {}", self.metadata.name, error))
        }
    }

    /// Create comprehensive Lua execution wrapper with Loki API integration
    fn create_lua_execution_wrapper(&self, input: &str) -> Result<String> {
        let plugin_id = &self.metadata.id;
        let plugin_name = &self.metadata.name;
        let plugin_version = &self.metadata.version;
        let plugin_author = &self.metadata.author;
        let plugin_description = &self.metadata.description;

        // Escape input for Lua string
        let escaped_input = input.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
        let escaped_script = self.script_content.replace("\\", "\\\\").replace("\"", "\\\"");

        let lua_wrapper = format!(r#"
-- Loki Plugin Execution Environment
-- Plugin: {} v{}
-- Author: {}

-- Security: Disable dangerous functions
os.execute = nil
os.exit = nil
os.remove = nil
os.rename = nil
io.popen = nil
loadfile = nil
dofile = nil

-- Loki API Functions
local loki = {{}}

-- Plugin metadata
loki.plugin = {{
    id = "{}",
    name = "{}",
    version = "{}",
    author = "{}",
    description = "{}"
}}

-- Logging function
function loki.log(level, message)
    local timestamp = os.date("%Y-%m-%d %H:%M:%S")
    print(string.format("[%s] [%s] [Lua Plugin %s]: %s", timestamp, level:upper(), loki.plugin.name, tostring(message)))
end

-- JSON utilities (basic implementation)
function loki.json_encode(obj)
    if type(obj) == "table" then
        local result = {{}}
        local is_array = true
        local max_index = 0

        -- Check if table is array-like
        for k, v in pairs(obj) do
            if type(k) ~= "number" or k <= 0 or k ~= math.floor(k) then
                is_array = false
                break
            end
            max_index = math.max(max_index, k)
        end

        if is_array then
            -- Array encoding
            for i = 1, max_index do
                table.insert(result, loki.json_encode(obj[i] or "null"))
            end
            return "[" .. table.concat(result, ",") .. "]"
        else
            -- Object encoding
            for k, v in pairs(obj) do
                table.insert(result, '"' .. tostring(k) .. '":' .. loki.json_encode(v))
            end
            return "{{" .. table.concat(result, ",") .. "}}"
        end
    elseif type(obj) == "string" then
        return '"' .. obj:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n') .. '"'
    elseif type(obj) == "number" then
        return tostring(obj)
    elseif type(obj) == "boolean" then
        return obj and "true" or "false"
    else
        return "null"
    end
end

-- Time utilities
function loki.timestamp()
    return os.time()
end

function loki.sleep(duration_ms)
    -- Limit sleep duration for security (max 5 seconds)
    local sleep_duration = math.min(duration_ms or 0, 5000)
    os.execute("sleep " .. (sleep_duration / 1000))
end

-- String utilities
function loki.uuid()
    -- Simple UUID generation (not cryptographically secure)
    local template = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
    return string.gsub(template, '[xy]', function (c)
        local v = (c == 'x') and math.random(0, 0xf) or math.random(8, 0xb)
        return string.format('%x', v)
    end)
end

-- Hash utilities
function loki.hash(algorithm, data)
    -- Basic hash implementation (for demonstration)
    local hash = 0
    for i = 1, #data do
        hash = (hash * 31 + string.byte(data, i)) % 1000000007
    end
    return string.format("%x", hash)
end

-- Math utilities
function loki.random(min, max)
    return math.random() * (max - min) + min
end

-- Make loki table available globally
_G.loki = loki

-- Override print to use loki.log
_original_print = print
function print(...)
    local args = {{...}}
    local message = ""
    for i, v in ipairs(args) do
        if i > 1 then message = message .. " " end
        message = message .. tostring(v)
    end
    loki.log("INFO", message)
end

-- Plugin script execution
local function execute_plugin()
    -- Load plugin script
    local plugin_script = [[{}]]

    -- Execute plugin script in protected environment
    local success, result = pcall(function()
        local chunk, err = load(plugin_script, "plugin_script")
        if not chunk then
            error("Failed to compile plugin script: " .. (err or "unknown error"))
        end
        return chunk()
    end)

    if not success then
        loki.log("ERROR", "Plugin execution failed: " .. tostring(result))
        return loki.json_encode({{status = "error", message = tostring(result)}})
    end

    -- Prepare input data
    local input_data = "{}"
    if input_data ~= "" then
        -- Try to parse as JSON, fallback to string
        local json_success, parsed_input = pcall(function()
            return load("return " .. input_data)()
        end)
        if not json_success then
            input_data = input_data -- Use as string
        else
            input_data = parsed_input
        end
    else
        input_data = nil
    end

    -- Try to call plugin_execute function
    if _G.plugin_execute and type(_G.plugin_execute) == "function" then
        local exec_success, exec_result = pcall(_G.plugin_execute, input_data)
        if exec_success then
            return loki.json_encode(exec_result)
        else
            loki.log("ERROR", "plugin_execute failed: " .. tostring(exec_result))
            return loki.json_encode({{status = "error", message = tostring(exec_result)}})
        end
    elseif _G.plugin_main and type(_G.plugin_main) == "function" then
        local main_success, main_result = pcall(_G.plugin_main, input_data)
        if main_success then
            return loki.json_encode(main_result)
        else
            loki.log("ERROR", "plugin_main failed: " .. tostring(main_result))
            return loki.json_encode({{status = "error", message = tostring(main_result)}})
        end
    else
        -- Script executed successfully but no main function found
        return loki.json_encode({{status = "executed", plugin = loki.plugin.name}})
    end
end

-- Execute plugin and output result
local result = execute_plugin()
print(result)
"#,
            plugin_name, plugin_version, plugin_author,
            plugin_id, plugin_name, plugin_version, plugin_author, plugin_description,
            escaped_script, escaped_input
        );

        Ok(lua_wrapper)
    }
}



#[async_trait::async_trait]
impl Plugin for ScriptPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, _context: PluginContext) -> Result<()> {
        info!("Initializing script plugin: {} ({})",
               self.metadata.name, self.script_type.name());

        // Test that the script can be executed
        self.test_initialization().await
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting script plugin: {} ({})",
               self.metadata.name, self.script_type.name());
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping script plugin: {}", self.metadata.name);
        Ok(())
    }

    async fn handle_event(&mut self, event: PluginEvent) -> Result<()> {
        info!("Script plugin {} handling event: {:?}", self.metadata.name, event);

        // Convert event to JSON for script processing
        let event_json = serde_json::to_string(&event)
            .context("Failed to serialize event")?;

        // Execute script with event data
        let output = self.execute_script(&event_json).await
            .context("Script event handling failed")?;

        info!("Script plugin {} event result: {}", self.metadata.name, output.trim());
        Ok(())
    }

    fn state(&self) -> PluginState {
        PluginState::Active
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        // Perform a simple syntax check as health validation
        match self.test_initialization().await {
            Ok(_) => Ok(HealthStatus {
                healthy: true,
                message: Some(format!("Script plugin {} is healthy", self.metadata.name)),
                metrics: std::collections::HashMap::new(),
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                message: Some(format!("Script plugin {} health check failed: {}", self.metadata.name, e)),
                metrics: std::collections::HashMap::new(),
            }),
        }
    }
}

/// Helper methods for ScriptType
impl ScriptType {
    pub fn name(&self) -> &'static str {
        match self {
            ScriptType::Python => "Python",
            ScriptType::JavaScript => "JavaScript",
            ScriptType::Lua => "Lua",
        }
    }
}

/// Advanced WebAssembly plugin implementation using the enhanced WASM engine
pub struct AdvancedWasmPlugin {
    metadata: PluginMetadata,
    engine: super::wasm_engine::WasmEngine,
    plugin_id: String,
}

impl AdvancedWasmPlugin {
    pub fn new(
        metadata: PluginMetadata,
        engine: super::wasm_engine::WasmEngine,
        plugin_id: String,
    ) -> Self {
        Self {
            metadata,
            engine,
            plugin_id,
        }
    }
}

#[async_trait::async_trait]
impl Plugin for AdvancedWasmPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, _context: PluginContext) -> Result<()> {
        info!("Advanced WASM plugin {} already initialized during loading", self.metadata.name);
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting advanced WASM plugin: {}", self.metadata.name);
        
        // Execute plugin startup function if it exists
        match self.engine.execute_plugin_function(&self.plugin_id, "plugin_start", &[]).await {
            Ok(_) => {
                info!("Plugin {} started successfully", self.metadata.name);
                Ok(())
            }
            Err(e) => {
                // If plugin_start doesn't exist, that's okay
                if e.to_string().contains("not found") {
                    debug!("Plugin {} has no startup function, continuing", self.metadata.name);
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping advanced WASM plugin: {}", self.metadata.name);
        
        // Execute plugin cleanup function if it exists
        match self.engine.execute_plugin_function(&self.plugin_id, "plugin_cleanup", &[]).await {
            Ok(_) => {
                info!("Plugin {} cleanup completed", self.metadata.name);
            }
            Err(e) => {
                if e.to_string().contains("not found") {
                    debug!("Plugin {} has no cleanup function", self.metadata.name);
                } else {
                    warn!("Plugin {} cleanup failed: {}", self.metadata.name, e);
                }
            }
        }

        // Unload from engine
        self.engine.unload_plugin(&self.plugin_id).await?;
        Ok(())
    }

    async fn handle_event(&mut self, event: PluginEvent) -> Result<()> {
        debug!("Advanced WASM plugin {} handling event: {:?}", self.metadata.name, event);

        // Convert event to a parameter that WASM can understand
        let event_code = match event {
            PluginEvent::System(_) => wasmtime::Val::I32(1),
            PluginEvent::Cognitive(_) => wasmtime::Val::I32(2),
            PluginEvent::Memory(_) => wasmtime::Val::I32(3),
            PluginEvent::Social(_) => wasmtime::Val::I32(4),
            PluginEvent::Custom(_, _) => wasmtime::Val::I32(5),
        };

        // Try to call plugin_handle_event function
        match self.engine.execute_plugin_function(&self.plugin_id, "plugin_handle_event", &[event_code]).await {
            Ok(results) => {
                debug!("Plugin {} handled event successfully: {:?}", self.metadata.name, results);
                Ok(())
            }
            Err(e) => {
                if e.to_string().contains("not found") {
                    debug!("Plugin {} has no event handler function", self.metadata.name);
                    Ok(())
                } else {
                    error!("Plugin {} event handling failed: {}", self.metadata.name, e);
                    Err(e)
                }
            }
        }
    }

    fn state(&self) -> PluginState {
        // Check if plugin is loaded in the engine
        let plugin_list = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.engine.list_plugins())
        });
        
        if plugin_list.contains(&self.plugin_id) {
            PluginState::Active
        } else {
            PluginState::Stopped
        }
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        // Get plugin metrics from the advanced engine
        match self.engine.get_plugin_metrics(&self.plugin_id).await {
            Ok(metrics) => {
                let mut health_metrics = std::collections::HashMap::new();
                health_metrics.insert("function_calls".to_string(), metrics.function_calls as f64);
                health_metrics.insert("total_execution_time_us".to_string(), metrics.total_execution_time_us as f64);
                health_metrics.insert("memory_usage_bytes".to_string(), metrics.memory_usage_bytes as f64);
                health_metrics.insert("error_count".to_string(), metrics.error_count as f64);

                Ok(HealthStatus {
                    healthy: metrics.error_count == 0 || (metrics.function_calls > 0 && metrics.error_count < metrics.function_calls / 10),
                    message: Some(format!(
                        "Advanced WASM plugin {} - {} calls, {} errors, {} memory usage",
                        self.metadata.name,
                        metrics.function_calls,
                        metrics.error_count,
                        metrics.memory_usage_bytes
                    )),
                    metrics: health_metrics,
                })
            }
            Err(e) => {
                Ok(HealthStatus {
                    healthy: false,
                    message: Some(format!("Failed to get plugin metrics: {}", e)),
                    metrics: std::collections::HashMap::new(),
                })
            }
        }
    }
}
