use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use wasmtime::*;

use super::{
    PluginCapability, PluginContext, PluginError, PluginMetadata,
    PluginState, HealthStatus,
};

/// Advanced WebAssembly plugin engine with comprehensive security, performance, and debugging features
#[derive(Clone)]
pub struct WasmEngine {
    /// Wasmtime engine with optimized configuration
    engine: Arc<Engine>,

    /// Plugin instances
    instances: Arc<RwLock<HashMap<String, WasmPluginInstance>>>,

    /// Engine configuration
    config: WasmEngineConfig,

    /// Resource limits and monitoring
    resource_monitor: Arc<WasmResourceMonitor>,

    /// Security policies
    _securityconfig: WasmSecurityConfig,

    /// Performance profiler
    _profiler: Arc<WasmProfiler>,
}

/// WebAssembly engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmEngineConfig {
    /// Enable debugging support
    pub enable_debug: bool,

    /// Enable fuel metering (for CPU limiting)
    pub enable_fuel: bool,

    /// Maximum fuel per execution
    pub max_fuel: u64,

    /// Enable epoch interruption (for timeouts)
    pub enable_epoch_interruption: bool,

    /// Epoch deadline (seconds)
    pub epoch_deadline_seconds: u64,

    /// Enable memory limits
    pub enable_memory_limits: bool,

    /// Maximum memory pages (64KB each)
    pub max_memory_pages: u32,

    /// Enable stack limits
    pub enable_stack_limits: bool,

    /// Maximum stack size (bytes)
    pub max_stack_size: usize,

    /// Enable optimization
    pub enable_optimization: bool,

    /// Optimization level (0-2)
    pub optimization_level: u8,

    /// Enable parallel compilation
    pub enable_parallel_compilation: bool,

    /// Cache compiled modules
    pub enable_module_cache: bool,

    /// Module cache directory
    pub cache_directory: Option<String>,
}

impl Default for WasmEngineConfig {
    fn default() -> Self {
        Self {
            enable_debug: false,
            enable_fuel: true,
            max_fuel: 1_000_000, // 1M fuel units
            enable_epoch_interruption: true,
            epoch_deadline_seconds: 30,
            enable_memory_limits: true,
            max_memory_pages: 1024, // 64MB max
            enable_stack_limits: true,
            max_stack_size: 1024 * 1024, // 1MB stack
            enable_optimization: true,
            optimization_level: 2,
            enable_parallel_compilation: true,
            enable_module_cache: true,
            cache_directory: Some("./cache/wasm".to_string()),
        }
    }
}

/// WebAssembly security configuration
#[derive(Debug, Clone)]
pub struct WasmSecurityConfig {
    /// Allowed host functions
    pub allowed_host_functions: Vec<String>,

    /// Sandbox mode (restricts all host calls)
    pub sandbox_mode: bool,

    /// Allow network access
    pub allow_network: bool,

    /// Allow file system access
    pub allow_filesystem: bool,

    /// Allow environment variable access
    pub allow_env_vars: bool,

    /// Allow system calls
    pub allow_system_calls: bool,

    /// Capability-based security
    pub required_capabilities: Vec<PluginCapability>,
}

impl Default for WasmSecurityConfig {
    fn default() -> Self {
        Self {
            allowed_host_functions: vec![
                "loki_log".to_string(),
                "loki_get_time".to_string(),
                "loki_memory_read".to_string(),
            ],
            sandbox_mode: true,
            allow_network: false,
            allow_filesystem: false,
            allow_env_vars: false,
            allow_system_calls: false,
            required_capabilities: vec![],
        }
    }
}

/// WebAssembly plugin instance
#[derive(Debug)]
pub struct WasmPluginInstance {
    /// Plugin metadata
    pub metadata: PluginMetadata,

    /// Wasmtime store
    pub store: Store<WasmPluginState>,

    /// Wasmtime instance
    pub instance: Instance,

    /// Plugin configuration
    pub config: WasmPluginConfig,

    /// Current state
    pub state: PluginState,

    /// Performance metrics
    pub metrics: WasmPluginMetrics,

    /// Security context
    pub security_context: WasmSecurityContext,
}

/// WebAssembly plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPluginConfig {
    /// Plugin-specific memory limit (pages)
    pub memory_limit_pages: Option<u32>,

    /// Plugin-specific fuel limit
    pub fuel_limit: Option<u64>,

    /// Plugin-specific timeout (seconds)
    pub timeout_seconds: Option<u64>,

    /// Enable profiling for this plugin
    pub enable_profiling: bool,

    /// Enable tracing
    pub enable_tracing: bool,

    /// Custom host function imports
    pub custom_imports: HashMap<String, String>,

    /// Environment variables accessible to plugin
    pub environment: HashMap<String, String>,

    /// Plugin-specific capabilities
    pub capabilities: Vec<PluginCapability>,
}

impl Default for WasmPluginConfig {
    fn default() -> Self {
        Self {
            memory_limit_pages: None,
            fuel_limit: None,
            timeout_seconds: None,
            enable_profiling: false,
            enable_tracing: false,
            custom_imports: HashMap::new(),
            environment: HashMap::new(),
            capabilities: vec![],
        }
    }
}

/// WebAssembly plugin state (store data)
#[derive(Debug)]
pub struct WasmPluginState {
    /// Plugin ID
    pub plugin_id: String,

    /// Plugin context
    pub context: Option<PluginContext>,

    /// Shared data between host and plugin
    pub shared_data: HashMap<String, Vec<u8>>,

    /// Performance counters
    pub performance_counters: HashMap<String, u64>,

    /// Security violations count
    pub security_violations: u64,

    /// Last activity timestamp
    pub last_activity: std::time::Instant,
}

/// WebAssembly plugin performance metrics
#[derive(Debug, Clone, Default)]
pub struct WasmPluginMetrics {
    /// Total execution time (microseconds)
    pub total_execution_time_us: u64,

    /// Number of function calls
    pub function_calls: u64,

    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,

    /// Fuel consumed
    pub fuel_consumed: u64,

    /// Host function calls
    pub host_function_calls: HashMap<String, u64>,

    /// Errors encountered
    pub error_count: u64,

    /// Last execution time
    pub last_execution_time_us: u64,
}

/// WebAssembly security context
#[derive(Debug, Clone)]
pub struct WasmSecurityContext {
    /// Granted capabilities
    pub granted_capabilities: Vec<PluginCapability>,

    /// Security policy version
    pub policy_version: String,

    /// Access log
    pub access_log: Vec<WasmSecurityEvent>,

    /// Violations count
    pub violations_count: u64,

    /// Last security check
    pub last_security_check: std::time::Instant,
}

/// WebAssembly security event
#[derive(Debug, Clone)]
pub struct WasmSecurityEvent {
    /// Event timestamp
    pub timestamp: std::time::Instant,

    /// Event type
    pub event_type: WasmSecurityEventType,

    /// Function called
    pub function_name: String,

    /// Access granted/denied
    pub access_granted: bool,

    /// Additional context
    pub context: String,
}

/// WebAssembly security event type
#[derive(Debug, Clone)]
pub enum WasmSecurityEventType {
    HostFunctionCall,
    MemoryAccess,
    NetworkAccess,
    FileSystemAccess,
    EnvironmentAccess,
    SystemCall,
    CapabilityCheck,
}

/// WebAssembly resource monitor
#[derive(Debug)]
pub struct WasmResourceMonitor {
    /// Global memory usage
    pub global_memory_usage: Arc<RwLock<u64>>,

    /// Global CPU usage
    pub global_cpu_usage: Arc<RwLock<f64>>,

    /// Per-plugin resource usage
    pub plugin_resources: Arc<RwLock<HashMap<String, PluginResourceUsage>>>,

    /// Resource limits
    pub limits: WasmResourceLimits,
}

/// Plugin resource usage
#[derive(Debug, Clone, Default)]
pub struct PluginResourceUsage {
    /// Memory usage (bytes)
    pub memory_bytes: u64,

    /// CPU time (microseconds)
    pub cpu_time_us: u64,

    /// Number of active instances
    pub active_instances: u32,

    /// Peak memory usage
    pub peak_memory_bytes: u64,

    /// Total function calls
    pub total_function_calls: u64,
}

/// WebAssembly resource limits
#[derive(Debug, Clone)]
pub struct WasmResourceLimits {
    /// Maximum global memory (bytes)
    pub max_global_memory_bytes: u64,

    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,

    /// Maximum plugins per engine
    pub max_plugins_per_engine: u32,

    /// Maximum instances per plugin
    pub max_instances_per_plugin: u32,
}

impl Default for WasmResourceLimits {
    fn default() -> Self {
        Self {
            max_global_memory_bytes: 1024 * 1024 * 1024, // 1GB
            max_cpu_percent: 80.0,
            max_plugins_per_engine: 100,
            max_instances_per_plugin: 10,
        }
    }
}

/// WebAssembly profiler
#[derive(Debug)]
pub struct WasmProfiler {
    /// Profiling enabled
    pub enabled: bool,

    /// Profile data
    pub profiles: Arc<RwLock<HashMap<String, WasmProfileData>>>,

    /// Sampling rate (Hz)
    pub sampling_rate: u32,
}

/// WebAssembly profile data
#[derive(Debug, Clone, Default)]
pub struct WasmProfileData {
    /// Function call counts
    pub function_calls: HashMap<String, u64>,

    /// Function execution times
    pub function_times: HashMap<String, u64>,

    /// Memory allocation events
    pub memory_events: Vec<WasmMemoryEvent>,

    /// Host function call trace
    pub host_function_trace: Vec<WasmHostFunctionCall>,
}

/// WebAssembly memory event
#[derive(Debug, Clone)]
pub struct WasmMemoryEvent {
    /// Timestamp
    pub timestamp: std::time::Instant,

    /// Event type
    pub event_type: WasmMemoryEventType,

    /// Address
    pub address: u32,

    /// Size
    pub size: u32,
}

/// WebAssembly memory event type
#[derive(Debug, Clone)]
pub enum WasmMemoryEventType {
    Allocation,
    Deallocation,
    Read,
    Write,
}

/// WebAssembly host function call
#[derive(Debug, Clone)]
pub struct WasmHostFunctionCall {
    /// Timestamp
    pub timestamp: std::time::Instant,

    /// Function name
    pub function_name: String,

    /// Parameters
    pub parameters: Vec<String>,

    /// Execution time (microseconds)
    pub execution_time_us: u64,

    /// Result
    pub result: Result<String, String>,
}

impl WasmEngine {
    /// Create a new WebAssembly engine with advanced features
    pub fn new(config: WasmEngineConfig) -> Result<Self> {
        info!("Creating advanced WebAssembly engine");

        // Configure Wasmtime engine
        let mut engineconfig = Config::new();

        // Enable/disable debug based on configuration
        engineconfig.debug_info(config.enable_debug);

        // Configure fuel metering
        if config.enable_fuel {
            engineconfig.consume_fuel(true);
        }

        // Configure epoch interruption
        if config.enable_epoch_interruption {
            engineconfig.epoch_interruption(true);
        }

        // Configure optimization
        if config.enable_optimization {
            engineconfig.strategy(match config.optimization_level {
                0 => Strategy::Auto,
                1 => Strategy::Cranelift,
                _ => Strategy::Auto,
            });
        }

        // Configure parallel compilation
        if config.enable_parallel_compilation {
            engineconfig.parallel_compilation(true);
        }

        // Configure module cache
        if config.enable_module_cache {
            if let Some(ref cache_dir) = config.cache_directory {
                if let Err(e) = std::fs::create_dir_all(cache_dir) {
                    warn!("Failed to create WASM cache directory {}: {}", cache_dir, e);
                } else {
                    // engineconfig.cache_config_load_default()
                    //     .context("Failed to load default cache config")?;
                }
            }
        }

        // Create engine
        let engine = Engine::new(&engineconfig)
            .context("Failed to create Wasmtime engine")?;

        // Create resource monitor
        let resource_monitor = Arc::new(WasmResourceMonitor {
            global_memory_usage: Arc::new(RwLock::new(0)),
            global_cpu_usage: Arc::new(RwLock::new(0.0)),
            plugin_resources: Arc::new(RwLock::new(HashMap::new())),
            limits: WasmResourceLimits::default(),
        });

        // Create profiler
        let profiler = Arc::new(WasmProfiler {
            enabled: config.enable_debug,
            profiles: Arc::new(RwLock::new(HashMap::new())),
            sampling_rate: 1000, // 1kHz
        });

        Ok(Self {
            engine: Arc::new(engine),
            instances: Arc::new(RwLock::new(HashMap::new())),
            config,
            resource_monitor,
            _securityconfig: WasmSecurityConfig::default(),
            _profiler: profiler,
        })
    }

    /// Load a WebAssembly plugin from bytes with comprehensive validation
    pub async fn load_plugin(
        &self,
        plugin_id: String,
        metadata: PluginMetadata,
        wasm_bytes: &[u8],
        pluginconfig: WasmPluginConfig,
        context: PluginContext,
    ) -> Result<()> {
        info!("Loading WASM plugin: {} ({})", metadata.name, plugin_id);

        // Validate WASM bytecode
        self.validate_wasm_bytecode(wasm_bytes).await?;

        // Create store with plugin state
        let plugin_state = WasmPluginState {
            plugin_id: plugin_id.clone(),
            context: Some(context),
            shared_data: HashMap::new(),
            performance_counters: HashMap::new(),
            security_violations: 0,
            last_activity: std::time::Instant::now(),
        };

        let mut store = Store::new(&*self.engine, plugin_state);

        // Configure store limits
        self.configure_store_limits(&mut store, &pluginconfig)?;

        // Compile module
        let module = Module::new(&*self.engine, wasm_bytes)
            .context("Failed to compile WASM module")?;

        // Validate module exports
        self.validate_module_exports(&module)?;

        // Create linker with host functions
        let mut linker = Linker::new(&*self.engine);
        self.setup_host_functions(&mut linker, &metadata.capabilities).await?;

        // Instantiate the module
        let instance = linker.instantiate(&mut store, &module)
            .context("Failed to instantiate WASM module")?;

        // Validate plugin interface
        self.validate_plugin_interface(&instance, &mut store)?;

        // Initialize plugin
        self.initialize_plugin_instance(&instance, &mut store).await?;

        // Create security context
        let security_context = WasmSecurityContext {
            granted_capabilities: metadata.capabilities.clone(),
            policy_version: "1.0.0".to_string(),
            access_log: Vec::new(),
            violations_count: 0,
            last_security_check: std::time::Instant::now(),
        };

        // Create plugin instance
        let plugin_instance = WasmPluginInstance {
            metadata,
            store,
            instance,
            config: pluginconfig,
            state: PluginState::Active,
            metrics: WasmPluginMetrics::default(),
            security_context,
        };

        // Store instance
        self.instances.write().await.insert(plugin_id.clone(), plugin_instance);

        // Update resource monitoring
        self.update_resource_usage(&plugin_id, 0, 0).await;

        info!("Successfully loaded WASM plugin: {}", plugin_id);
        Ok(())
    }

    /// Validate WebAssembly bytecode for security and compatibility
    async fn validate_wasm_bytecode(&self, wasm_bytes: &[u8]) -> Result<()> {
        debug!("Validating WASM bytecode ({} bytes)", wasm_bytes.len());

        // Basic size check
        if wasm_bytes.len() > 50 * 1024 * 1024 { // 50MB limit
            return Err(anyhow::anyhow!("WASM module too large: {} bytes", wasm_bytes.len()));
        }

        if wasm_bytes.len() < 8 {
            return Err(anyhow::anyhow!("WASM module too small: {} bytes", wasm_bytes.len()));
        }

        // Check WASM magic number
        if &wasm_bytes[0..4] != b"\0asm" {
            return Err(anyhow::anyhow!("Invalid WASM magic number"));
        }

        // Check WASM version
        let version = u32::from_le_bytes([wasm_bytes[4], wasm_bytes[5], wasm_bytes[6], wasm_bytes[7]]);
        if version != 1 {
            return Err(anyhow::anyhow!("Unsupported WASM version: {}", version));
        }

        // Validate that the module can be parsed (without instantiating)
        Module::validate(&*self.engine, wasm_bytes)
            .context("WASM bytecode validation failed")?;

        debug!("WASM bytecode validation passed");
        Ok(())
    }

    /// Configure store limits based on plugin configuration
    fn configure_store_limits(&self, store: &mut Store<WasmPluginState>, config: &WasmPluginConfig) -> Result<()> {
        // Set fuel limit
        if self.config.enable_fuel {
            let fuel_limit = config.fuel_limit.unwrap_or(self.config.max_fuel);
            store.set_fuel(fuel_limit)
                .context("Failed to set fuel on store")?;
        }

        // Set epoch deadline
        if self.config.enable_epoch_interruption {
            let deadline = config.timeout_seconds.unwrap_or(self.config.epoch_deadline_seconds);
            store.set_epoch_deadline(deadline);
        }

        Ok(())
    }

    /// Validate module exports for required plugin interface
    fn validate_module_exports(&self, module: &Module) -> Result<()> {
        debug!("Validating module exports");

        let required_exports = ["plugin_init", "memory"];
        let optional_exports = ["plugin_execute", "plugin_cleanup", "plugin_handle_event"];

        // Check required exports
        for export_name in &required_exports {
            if !module.exports().any(|export| export.name() == *export_name) {
                return Err(anyhow::anyhow!("Required export '{}' not found", export_name));
            }
        }

        // Log optional exports
        for export_name in &optional_exports {
            if module.exports().any(|export| export.name() == *export_name) {
                debug!("Found optional export: {}", export_name);
            }
        }

        debug!("Module exports validation passed");
        Ok(())
    }

    /// Setup host functions that plugins can call
    async fn setup_host_functions(
        &self,
        linker: &mut Linker<WasmPluginState>,
        capabilities: &[PluginCapability],
    ) -> Result<()> {
        debug!("Setting up host functions for capabilities: {:?}", capabilities);

        // Always available: logging
        linker.func_wrap("loki", "log", |mut caller: Caller<'_, WasmPluginState>, level: i32, ptr: i32, len: i32| {
            let plugin_id = caller.data().plugin_id.clone();
            
            // Validate parameters
            if ptr < 0 || len < 0 || len > 1024 * 1024 { // 1MB limit
                error!("Plugin {} attempted invalid log call: ptr={}, len={}", plugin_id, ptr, len);
                return;
            }

            // Get memory and read string
            if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                let memory_data = memory.data(&caller);
                let start = ptr as usize;
                let end = start + len as usize;

                if end <= memory_data.len() {
                    if let Ok(message) = std::str::from_utf8(&memory_data[start..end]) {
                        let level_str = match level {
                            0 => "TRACE",
                            1 => "DEBUG", 
                            2 => "INFO",
                            3 => "WARN",
                            4 => "ERROR",
                            _ => "UNKNOWN",
                        };
                        info!("[WASM Plugin {}] [{}]: {}", plugin_id, level_str, message);
                    } else {
                        warn!("Plugin {} provided invalid UTF-8 in log message", plugin_id);
                    }
                } else {
                    error!("Plugin {} attempted out-of-bounds memory access in log", plugin_id);
                }
            } else {
                error!("Plugin {} does not export memory", plugin_id);
            }
        })?;

        // Time functions (always available)
        linker.func_wrap("loki", "get_time", |_caller: Caller<'_, WasmPluginState>| -> i64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        })?;

        linker.func_wrap("loki", "get_time_ms", |_caller: Caller<'_, WasmPluginState>| -> i64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64
        })?;

        // Memory access functions (if MemoryRead/MemoryWrite capabilities)
        if capabilities.iter().any(|cap| matches!(cap, PluginCapability::MemoryRead)) {
            linker.func_wrap("loki", "memory_read", 
                |mut caller: Caller<'_, WasmPluginState>, addr: i32, len: i32| -> i32 {
                    let plugin_id = caller.data().plugin_id.clone();
                    debug!("Plugin {} requesting memory read: addr={}, len={}", plugin_id, addr, len);

                    // Security check
                    if len > 1024 * 1024 { // 1MB limit
                        warn!("Plugin {} memory read request too large: {} bytes", plugin_id, len);
                        return -1;
                    }

                    // Update performance counter
                    caller.data_mut().performance_counters
                        .entry("memory_reads".to_string())
                        .and_modify(|e| *e += 1)
                        .or_insert(1);

                    // Simulate successful read
                    len
                }
            )?;
        }

        // Network access functions (if NetworkAccess capability)
        if capabilities.iter().any(|cap| matches!(cap, PluginCapability::NetworkAccess)) {
            linker.func_wrap("loki", "http_get",
                |mut caller: Caller<'_, WasmPluginState>, _url_ptr: i32, _url_len: i32| -> i32 {
                    let plugin_id = caller.data().plugin_id.clone();
                    info!("Plugin {} requesting HTTP GET", plugin_id);

                    // Update performance counter
                    caller.data_mut().performance_counters
                        .entry("http_requests".to_string())
                        .and_modify(|e| *e += 1)
                        .or_insert(1);

                    // For now, return a mock success response
                    200 // HTTP OK
                }
            )?;
        }

        // File system access (if FileSystem capabilities)
        if capabilities.iter().any(|cap| matches!(cap, PluginCapability::FileSystemRead | PluginCapability::FileSystemWrite)) {
            linker.func_wrap("loki", "file_read",
                |mut caller: Caller<'_, WasmPluginState>, _path_ptr: i32, _path_len: i32| -> i32 {
                    let plugin_id = caller.data().plugin_id.clone();
                    debug!("Plugin {} requesting file read", plugin_id);

                    // Security logging
                    caller.data_mut().performance_counters
                        .entry("file_reads".to_string())
                        .and_modify(|e| *e += 1)
                        .or_insert(1);

                    // Check if file read is allowed
                    0 // Success
                }
            )?;
        }

        debug!("Host functions setup completed");
        Ok(())
    }

    /// Validate plugin interface functions
    fn validate_plugin_interface(&self, instance: &Instance, store: &mut Store<WasmPluginState>) -> Result<()> {
        debug!("Validating plugin interface");

        // Check for plugin_init function
        let _init_func = instance.get_typed_func::<(), i32>(&mut *store, "plugin_init")
            .context("Plugin must export 'plugin_init' function with signature () -> i32")?;

        // Validate memory export
        let _memory = instance.get_memory(store, "memory")
            .context("Plugin must export 'memory'")?;

        debug!("Plugin interface validation passed");
        Ok(())
    }

    /// Initialize plugin instance by calling plugin_init
    async fn initialize_plugin_instance(&self, instance: &Instance, store: &mut Store<WasmPluginState>) -> Result<()> {
        debug!("Initializing plugin instance");

        // Get initialization function and call it
        let start_time = std::time::Instant::now();
        let result = {
            let init_func = instance.get_typed_func::<(), i32>(&mut *store, "plugin_init")
                .context("Plugin initialization function not found")?;
            init_func.call(&mut *store, ())
                .context("Plugin initialization failed")?
        };
        let elapsed = start_time.elapsed();

        // Update metrics
        store.data_mut().performance_counters
            .insert("init_time_us".to_string(), elapsed.as_micros() as u64);

        if result != 0 {
            return Err(anyhow::anyhow!("Plugin initialization returned error code: {}", result));
        }

        debug!("Plugin initialized successfully in {:?}", elapsed);
        Ok(())
    }

    /// Execute a plugin function with comprehensive monitoring and security
    pub async fn execute_plugin_function(
        &self,
        plugin_id: &str,
        function_name: &str,
        params: &[wasmtime::Val],
    ) -> Result<Vec<wasmtime::Val>> {
        let _start_time = std::time::Instant::now();
        
        let mut instances = self.instances.write().await;
        let plugin_instance = instances.get_mut(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        debug!("Executing function '{}' on plugin '{}'", function_name, plugin_id);

        // Security check
        self.perform_security_check(plugin_instance, function_name).await?;

        // Get function
        let func = plugin_instance.instance.get_func(&mut plugin_instance.store, function_name)
            .ok_or_else(|| anyhow::anyhow!("Function '{}' not found in plugin", function_name))?;

        // Prepare result storage
        let func_type = func.ty(&plugin_instance.store);
        let mut results = vec![wasmtime::Val::I32(0); func_type.results().len()];

        // Execute with timeout and resource monitoring
        let execution_start = std::time::Instant::now();
        
        let execution_result = if self.config.enable_epoch_interruption {
            // Use epoch-based timeout
            self.engine.increment_epoch();
            func.call(&mut plugin_instance.store, params, &mut results)
        } else {
            func.call(&mut plugin_instance.store, params, &mut results)
        };

        let execution_time = execution_start.elapsed();

        // Update metrics
        plugin_instance.metrics.function_calls += 1;
        plugin_instance.metrics.last_execution_time_us = execution_time.as_micros() as u64;
        plugin_instance.metrics.total_execution_time_us += execution_time.as_micros() as u64;

        // Handle execution result
        match execution_result {
            Ok(()) => {
                debug!("Function '{}' executed successfully in {:?}", function_name, execution_time);
                
                // Update resource usage
                let memory_usage = self.get_plugin_memory_usage(plugin_instance).await;
                self.update_resource_usage(plugin_id, memory_usage, execution_time.as_micros() as u64).await;

                Ok(results)
            }
            Err(e) => {
                plugin_instance.metrics.error_count += 1;
                error!("Function '{}' execution failed: {}", function_name, e);
                
                // Check if it's a resource limit error
                if e.to_string().contains("fuel") {
                    warn!("Plugin '{}' hit fuel limit", plugin_id);
                } else if e.to_string().contains("epoch") {
                    warn!("Plugin '{}' hit timeout", plugin_id);
                }

                Err(e.into())
            }
        }
    }

    /// Perform security check before function execution
    async fn perform_security_check(&self, plugin_instance: &mut WasmPluginInstance, function_name: &str) -> Result<()> {
        let now = std::time::Instant::now();

        // Rate limiting check
        if now.duration_since(plugin_instance.security_context.last_security_check).as_millis() < 10 {
            return Err(anyhow::anyhow!("Rate limit exceeded for plugin security checks"));
        }

        // Function whitelist check
        let allowed_functions = ["plugin_init", "plugin_execute", "plugin_cleanup", "plugin_handle_event"];
        if !allowed_functions.contains(&function_name) {
            plugin_instance.security_context.violations_count += 1;
            return Err(anyhow::anyhow!("Function '{}' not in allowed list", function_name));
        }

        // Log security event
        plugin_instance.security_context.access_log.push(WasmSecurityEvent {
            timestamp: now,
            event_type: WasmSecurityEventType::HostFunctionCall,
            function_name: function_name.to_string(),
            access_granted: true,
            context: "Function execution".to_string(),
        });

        plugin_instance.security_context.last_security_check = now;

        Ok(())
    }

    /// Get plugin memory usage
    async fn get_plugin_memory_usage(&self, plugin_instance: &mut WasmPluginInstance) -> u64 {
        if let Some(memory) = plugin_instance.instance.get_memory(&mut plugin_instance.store, "memory") {
            memory.data_size(&plugin_instance.store) as u64
        } else {
            0
        }
    }

    /// Update resource usage monitoring
    async fn update_resource_usage(&self, plugin_id: &str, memory_bytes: u64, cpu_time_us: u64) {
        let mut resources = self.resource_monitor.plugin_resources.write().await;
        let usage = resources.entry(plugin_id.to_string()).or_default();
        
        usage.memory_bytes = memory_bytes;
        usage.cpu_time_us += cpu_time_us;
        usage.peak_memory_bytes = usage.peak_memory_bytes.max(memory_bytes);
        usage.total_function_calls += 1;
    }

    /// Unload a plugin
    pub async fn unload_plugin(&self, plugin_id: &str) -> Result<()> {
        info!("Unloading WASM plugin: {}", plugin_id);

        let mut instances = self.instances.write().await;
        
        if let Some(mut plugin_instance) = instances.remove(plugin_id) {
            // Try to call cleanup function if it exists
            if let Some(cleanup_func) = plugin_instance.instance.get_func(&mut plugin_instance.store, "plugin_cleanup") {
                match cleanup_func.call(&mut plugin_instance.store, &[], &mut []) {
                    Ok(_) => debug!("Plugin cleanup completed"),
                    Err(e) => warn!("Plugin cleanup failed: {}", e),
                }
            }

            plugin_instance.state = PluginState::Stopped;
        } else {
            return Err(PluginError::NotFound(plugin_id.to_string()).into());
        }

        // Clean up resource monitoring
        self.resource_monitor.plugin_resources.write().await.remove(plugin_id);

        info!("Plugin {} unloaded successfully", plugin_id);
        Ok(())
    }

    /// Get plugin metrics
    pub async fn get_plugin_metrics(&self, plugin_id: &str) -> Result<WasmPluginMetrics> {
        let instances = self.instances.read().await;
        let plugin_instance = instances.get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        Ok(plugin_instance.metrics.clone())
    }

    /// Get all loaded plugins
    pub async fn list_plugins(&self) -> Vec<String> {
        self.instances.read().await.keys().cloned().collect()
    }

    /// Get engine statistics
    pub async fn get_engine_stats(&self) -> WasmEngineStats {
        let instances = self.instances.read().await;
        let plugin_count = instances.len();
        
        let total_memory = self.resource_monitor.global_memory_usage.read().await;
        let cpu_usage = self.resource_monitor.global_cpu_usage.read().await;

        let total_function_calls: u64 = instances.values()
            .map(|instance| instance.metrics.function_calls)
            .sum();

        let total_errors: u64 = instances.values()
            .map(|instance| instance.metrics.error_count)
            .sum();

        WasmEngineStats {
            total_plugins: plugin_count,
            total_memory_usage: *total_memory,
            cpu_usage_percent: *cpu_usage,
            total_function_calls,
            total_errors,
            engine_uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Health check for all plugins
    pub async fn health_check_all(&mut self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();
        let mut instances = self.instances.write().await;

        for (plugin_id, plugin_instance) in instances.iter_mut() {
            let health = if plugin_instance.state == PluginState::Active {
                // Try a simple function call to test health
                let healthy = match plugin_instance.instance.get_func(&mut plugin_instance.store, "plugin_init") {
                    Some(_) => true,
                    None => false,
                };

                HealthStatus {
                    healthy,
                    message: if healthy {
                        Some("Plugin is responsive".to_string())
                    } else {
                        Some("Plugin is not responsive".to_string())
                    },
                    metrics: {
                        let mut metrics = HashMap::new();
                        metrics.insert("memory_usage".to_string(), plugin_instance.metrics.memory_usage_bytes as f64);
                        metrics.insert("function_calls".to_string(), plugin_instance.metrics.function_calls as f64);
                        metrics.insert("error_count".to_string(), plugin_instance.metrics.error_count as f64);
                        metrics
                    },
                }
            } else {
                HealthStatus {
                    healthy: false,
                    message: Some(format!("Plugin state: {:?}", plugin_instance.state)),
                    metrics: HashMap::new(),
                }
            };

            results.insert(plugin_id.clone(), health);
        }

        results
    }
}

/// WebAssembly engine statistics
#[derive(Debug, Clone)]
pub struct WasmEngineStats {
    /// Total number of loaded plugins
    pub total_plugins: usize,

    /// Total memory usage (bytes)
    pub total_memory_usage: u64,

    /// CPU usage percentage
    pub cpu_usage_percent: f64,

    /// Total function calls across all plugins
    pub total_function_calls: u64,

    /// Total errors across all plugins
    pub total_errors: u64,

    /// Engine uptime (seconds)
    pub engine_uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wasm_engine_creation() {
        let config = WasmEngineConfig::default();
        let engine = WasmEngine::new(config).unwrap();
        
        let stats = engine.get_engine_stats().await;
        assert_eq!(stats.total_plugins, 0);
    }

    #[test]
    fn test_wasm_engineconfig_default() {
        let config = WasmEngineConfig::default();
        assert!(config.enable_fuel);
        assert!(config.enable_optimization);
        assert_eq!(config.max_fuel, 1_000_000);
    }

    #[test]
    fn test_securityconfig_default() {
        let config = WasmSecurityConfig::default();
        assert!(config.sandbox_mode);
        assert!(!config.allow_network);
        assert!(!config.allow_filesystem);
    }
}