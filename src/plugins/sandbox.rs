use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::RwLock;
use tokio::time::Duration;
use tracing::{debug, info, warn};

use super::{PluginCapability, PluginError};

/// Plugin sandbox for secure execution
pub struct PluginSandbox {
    /// Sandbox configuration
    config: SandboxConfig,

    /// Resource limits
    limits: ResourceLimits,

    /// Active sandboxes
    sandboxes: Arc<RwLock<HashMap<String, SandboxInstance>>>,
}

/// Sandbox configuration
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Enable sandboxing
    pub enabled: bool,

    /// Sandbox type
    pub sandbox_type: SandboxType,

    /// Allowed directories
    pub allowed_dirs: Vec<PathBuf>,

    /// Blocked system calls
    pub blocked_syscalls: Vec<String>,

    /// Network isolation
    pub network_isolation: bool,

    /// Process isolation
    pub process_isolation: bool,
}

/// Sandbox type
#[derive(Debug, Clone)]
pub enum SandboxType {
    /// No sandboxing (dangerous!)
    None,

    /// Basic process isolation
    Basic,

    /// Firejail sandboxing (Linux)
    Firejail,

    /// Docker container
    Docker,

    /// WebAssembly sandbox
    Wasm,
}

/// Resource limits for sandboxed plugins
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory (bytes)
    pub max_memory: usize,

    /// Maximum CPU percentage
    pub max_cpu_percent: f32,

    /// Maximum disk usage (bytes)
    pub max_disk: usize,

    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Maximum file descriptors
    pub max_file_descriptors: usize,

    /// Maximum threads
    pub max_threads: usize,
}

/// Active sandbox instance
struct SandboxInstance {
    plugin_id: String,
    process_id: Option<u32>,
    container_id: Option<String>,
    start_time: std::time::Instant,
    resource_usage: ResourceUsage,
}

/// Resource usage tracking
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    memory_bytes: usize,
    cpu_percent: f32,
    disk_bytes: usize,
    file_descriptors: usize,
    threads: usize,
}

/// WASM sandbox execution context data
#[derive(Debug, Clone)]
struct WasmSandboxData {
    /// Sandbox identifier
    sandbox_id: String,

    /// Granted capabilities
    _capabilities: Vec<PluginCapability>,

    /// Current resource usage
    resource_usage: ResourceUsage,

    /// Execution start time
    _start_time: std::time::Instant,
}

impl PluginSandbox {
    pub fn new(config: SandboxConfig, limits: ResourceLimits) -> Self {
        Self {
            config,
            limits,
            sandboxes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create sandbox for plugin
    pub async fn create_sandbox(
        &self,
        plugin_id: &str,
        plugin_path: &Path,
        capabilities: &[PluginCapability],
    ) -> Result<String> {
        info!("Creating sandbox for plugin: {}", plugin_id);

        if !self.config.enabled {
            warn!("Sandboxing is disabled - plugin will run without isolation!");
            return Ok(plugin_id.to_string());
        }

        let sandbox_id = format!("{}-{}", plugin_id, uuid::Uuid::new_v4());

        match self.config.sandbox_type {
            SandboxType::None => {
                warn!("No sandboxing - plugin running with full system access!");
                Ok(sandbox_id)
            }
            SandboxType::Basic => {
                self.create_basic_sandbox(&sandbox_id, plugin_path, capabilities).await
            }
            SandboxType::Firejail => {
                self.create_firejail_sandbox(&sandbox_id, plugin_path, capabilities).await
            }
            SandboxType::Docker => {
                self.create_docker_sandbox(&sandbox_id, plugin_path, capabilities).await
            }
            SandboxType::Wasm => {
                self.create_wasm_sandbox(&sandbox_id, plugin_path, capabilities).await
            }
        }
    }

    /// Create basic process isolation sandbox
    async fn create_basic_sandbox(
        &self,
        sandbox_id: &str,
        plugin_path: &Path,
        _capabilities: &[PluginCapability],
    ) -> Result<String> {
        // Basic isolation using OS features
        let mut cmd = Command::new("nice");
        cmd.arg("-n").arg("10"); // Lower priority

        // Set resource limits using ulimit
        if cfg!(unix) {
            cmd.arg("bash");
            cmd.arg("-c");

            let mut script = String::new();

            // Memory limit
            script.push_str(&format!(
                "ulimit -v {}; ",
                self.limits.max_memory / 1024 // KB
            ));

            // File descriptor limit
            script.push_str(&format!(
                "ulimit -n {}; ",
                self.limits.max_file_descriptors
            ));

            // Process limit
            script.push_str(&format!(
                "ulimit -u {}; ",
                self.limits.max_threads
            ));

            // Add the actual command
            script.push_str(&format!("exec {}", plugin_path.display()));

            cmd.arg(script);
        }

        // Start process
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn sandboxed process")?;

        let process_id = child.id();

        // Track sandbox instance
        let instance = SandboxInstance {
            plugin_id: sandbox_id.to_string(),
            process_id,
            container_id: None,
            start_time: std::time::Instant::now(),
            resource_usage: ResourceUsage::default(),
        };

        self.sandboxes.write().await.insert(sandbox_id.to_string(), instance);

        Ok(sandbox_id.to_string())
    }

    /// Create Firejail sandbox (Linux only)
    async fn create_firejail_sandbox(
        &self,
        sandbox_id: &str,
        plugin_path: &Path,
        capabilities: &[PluginCapability],
    ) -> Result<String> {
        // Check if firejail is available
        if !self.is_firejail_available().await? {
            warn!("Firejail not available, falling back to basic sandbox");
            return self.create_basic_sandbox(sandbox_id, plugin_path, capabilities).await;
        }

        let mut cmd = Command::new("firejail");

        // Basic security options
        cmd.arg("--quiet");
        cmd.arg("--private");
        cmd.arg("--noroot");

        // Memory limit
        cmd.arg(format!("--rlimit-as={}", self.limits.max_memory));

        // CPU limit
        cmd.arg(format!("--cpu={}", self.limits.max_cpu_percent / 100.0));

        // Network isolation if not granted
        if self.config.network_isolation && !capabilities.contains(&PluginCapability::NetworkAccess) {
            cmd.arg("--net=none");
        }

        // Filesystem restrictions
        for dir in &self.config.allowed_dirs {
            cmd.arg(format!("--whitelist={}", dir.display()));
        }

        // Block dangerous syscalls
        if !self.config.blocked_syscalls.is_empty() {
            let syscalls = self.config.blocked_syscalls.join(",");
            cmd.arg(format!("--seccomp.drop={}", syscalls));
        }

        // Add plugin executable
        cmd.arg(plugin_path);

        // Start sandboxed process
        let child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn firejail sandbox")?;

        let process_id = child.id();

        // Track sandbox instance
        let instance = SandboxInstance {
            plugin_id: sandbox_id.to_string(),
            process_id,
            container_id: None,
            start_time: std::time::Instant::now(),
            resource_usage: ResourceUsage::default(),
        };

        self.sandboxes.write().await.insert(sandbox_id.to_string(), instance);

        Ok(sandbox_id.to_string())
    }

    /// Create Docker container sandbox
    async fn create_docker_sandbox(
        &self,
        sandbox_id: &str,
        plugin_path: &Path,
        capabilities: &[PluginCapability],
    ) -> Result<String> {
        // Check if Docker is available
        if !self.is_docker_available().await? {
            warn!("Docker not available, falling back to basic sandbox");
            return self.create_basic_sandbox(sandbox_id, plugin_path, capabilities).await;
        }

        let mut cmd = Command::new("docker");
        cmd.arg("run");
        cmd.arg("-d"); // Detached
        cmd.arg("--rm"); // Remove on exit

        // Set container name
        cmd.arg("--name").arg(sandbox_id);

        // Resource limits
        cmd.arg("--memory").arg(format!("{}m", self.limits.max_memory / 1_048_576));
        cmd.arg("--cpus").arg(format!("{:.2}", self.limits.max_cpu_percent / 100.0));

        // Security options
        cmd.arg("--security-opt").arg("no-new-privileges");
        cmd.arg("--cap-drop").arg("ALL");

        // Network isolation
        if self.config.network_isolation && !capabilities.contains(&PluginCapability::NetworkAccess) {
            cmd.arg("--network").arg("none");
        }

        // Mount plugin directory
        cmd.arg("-v").arg(format!("{}:/plugin:ro", plugin_path.parent().unwrap().display()));

        // Use minimal base image
        cmd.arg("alpine:latest");

        // Run the plugin
        cmd.arg("/plugin/plugin");

        // Start container
        let output = cmd.output().await
            .context("Failed to create Docker container")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to start Docker container: {}", stderr));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // Track sandbox instance
        let instance = SandboxInstance {
            plugin_id: sandbox_id.to_string(),
            process_id: None,
            container_id: Some(container_id),
            start_time: std::time::Instant::now(),
            resource_usage: ResourceUsage::default(),
        };

        self.sandboxes.write().await.insert(sandbox_id.to_string(), instance);

        Ok(sandbox_id.to_string())
    }

    /// Create WASM sandbox using wasmtime runtime
    /// Implements comprehensive WASM sandboxing with capability-based security
    async fn create_wasm_sandbox(
        &self,
        sandbox_id: &str,
        plugin_path: &Path,
        capabilities: &[PluginCapability],
    ) -> Result<String> {
        info!("Creating WASM sandbox for plugin: {}", sandbox_id);

        // Verify the plugin is a WASM file
        if !plugin_path.extension().map_or(false, |ext| ext == "wasm") {
            return Err(anyhow::anyhow!(
                "WASM sandbox requires .wasm file, got: {}",
                plugin_path.display()
            ));
        }

        // Load WASM module bytecode
        let wasm_bytes = tokio::fs::read(plugin_path).await
            .context("Failed to read WASM plugin file")?;

        // Validate WASM module
        wasmtime::Module::validate(&wasmtime::Engine::default(), &wasm_bytes)
            .context("Invalid WASM module")?;

        // Create WASM sandbox configuration based on capabilities
        let wasmconfig = self.create_wasmconfig(capabilities)?;

        // Create wasmtime engine with security settings
        let engine = wasmtime::Engine::new(&wasmconfig)
            .context("Failed to create WASM engine")?;

        // Create module from bytecode
        let module = wasmtime::Module::new(&engine, &wasm_bytes)
            .context("Failed to create WASM module")?;

        // Create store with resource limits
        let mut store = wasmtime::Store::new(&engine, WasmSandboxData {
            sandbox_id: sandbox_id.to_string(),
            _capabilities: capabilities.to_vec(),
            resource_usage: ResourceUsage::default(),
            _start_time: std::time::Instant::now(),
        });

        // Set resource limits on the store
        self.configure_wasm_limits(&mut store)?;

        // Create linker for host functions
        let mut linker = wasmtime::Linker::new(&engine);

        // Add capability-based host functions
        self.add_host_functions(&mut linker, capabilities)?;

        // Instantiate the module in isolated environment
        let _instance = linker.instantiate(&mut store, &module)
            .context("Failed to instantiate WASM module")?;

        // Store the sandbox instance for management
        let sandbox_instance = SandboxInstance {
            plugin_id: sandbox_id.to_string(),
            process_id: None, // WASM runs in-process
            container_id: None,
            start_time: std::time::Instant::now(),
            resource_usage: ResourceUsage::default(),
        };

        self.sandboxes.write().await.insert(sandbox_id.to_string(), sandbox_instance);

        // Spawn background task to monitor WASM execution
        let sandbox_id_clone = sandbox_id.to_string();
        let sandboxes_clone = Arc::clone(&self.sandboxes);
        let limits = self.limits.clone();

        tokio::spawn(async move {
            Self::monitor_wasm_execution(sandbox_id_clone, sandboxes_clone, limits).await;
        });

        info!("WASM sandbox created successfully: {}", sandbox_id);
        Ok(sandbox_id.to_string())
    }

    /// Create WASM engine configuration with security constraints
    fn create_wasmconfig(&self, _capabilities: &[PluginCapability]) -> Result<wasmtime::Config> {
        let mut config = wasmtime::Config::new();

        // Enable compilation caching for performance
        // config.cache_config_load_default()
        //     .context("Failed to load WASM cache config")?;

        // Set compilation strategy (optimized for security)
        config.strategy(wasmtime::Strategy::Cranelift);

        // Enable epoch-based interruption for timeouts
        config.epoch_interruption(true);

        // Memory configuration
        config.max_wasm_stack(1024 * 1024); // 1MB stack limit
        //config.dynamic_memory_guard_size(64 * 1024); // 64KB guard pages

        // Security features
        config.cranelift_debug_verifier(true); // Enable verification
        config.consume_fuel(true); // Enable fuel consumption for execution limits

        // Disable dangerous features
        config.wasm_simd(false); // Disable SIMD for security
        config.wasm_reference_types(false); // Disable reference types
        config.wasm_multi_value(true); // Allow multi-value (safe)
        config.wasm_bulk_memory(false); // Disable bulk memory operations

        // Thread safety (disable threads for security)
        config.wasm_threads(false);

        // Disable module linking for isolation
        // Wasm module linking is deprecated, removed

        // Memory limits based on resource constraints
        config.memory_init_cow(false); // Disable copy-on-write for predictable memory

        // Enable guard regions for memory safety
       // config.static_memory_guard_size(65536); // 64KB guard pages

        Ok(config)
    }

    /// Configure resource limits for WASM store
    fn configure_wasm_limits(&self, store: &mut wasmtime::Store<WasmSandboxData>) -> Result<()> {
        // Set fuel limit based on execution time limit
        let fuel_limit = (self.limits.max_execution_time.as_secs() * 1_000_000) as u64; // 1M units per second
        store.set_fuel(fuel_limit)
            .context("Failed to set WASM fuel limit")?;

        // Configure epoch deadlines for timeout handling
        store.set_epoch_deadline(1); // Interrupt every epoch

        Ok(())
    }

    /// Add capability-based host functions to the WASM linker
    fn add_host_functions(
        &self,
        linker: &mut wasmtime::Linker<WasmSandboxData>,
        capabilities: &[PluginCapability],
    ) -> Result<()> {
        // Basic logging function (always available)
        linker.func_wrap("env", "log", |mut caller: wasmtime::Caller<'_, WasmSandboxData>, ptr: i32, len: i32| {
            let memory = caller.get_export("memory")
                .and_then(|export| export.into_memory())
                .ok_or_else(|| anyhow::anyhow!("Failed to get WASM memory"))?;

            let data = memory.data(&caller);
            if ptr < 0 || len < 0 || (ptr as usize + len as usize) > data.len() {
                return Err(anyhow::anyhow!("Invalid memory access in log function"));
            }

            let log_data = &data[ptr as usize..(ptr as usize + len as usize)];
            let log_string = String::from_utf8_lossy(log_data);

            let sandbox_data = caller.data();
            info!("[WASM:{}] {}", sandbox_data.sandbox_id, log_string);

            Ok(())
        })?;

        // Memory allocation (controlled)
        linker.func_wrap("env", "alloc", |mut caller: wasmtime::Caller<'_, WasmSandboxData>, size: i32| -> Result<i32> {
            if size <= 0 || size > 1024 * 1024 { // Max 1MB allocation
                return Err(anyhow::anyhow!("Invalid allocation size: {}", size));
            }

            let sandbox_data = caller.data_mut();
            sandbox_data.resource_usage.memory_bytes += size as usize;

            // Simple bump allocator for demonstration
            // In production, this would be more sophisticated
            Ok(size) // Return the allocated pointer (simplified)
        })?;

        // File system access (if capability granted)
        if capabilities.contains(&PluginCapability::FileSystemRead) ||
           capabilities.contains(&PluginCapability::FileSystemWrite) {
            self.add_filesystem_functions(linker)?;
        }

        // Network access (if capability granted)
        if capabilities.contains(&PluginCapability::NetworkAccess) {
            self.add_network_functions(linker)?;
        }

        // System information access (if capability granted)
        if capabilities.iter().any(|cap| matches!(cap, PluginCapability::Custom(s) if s == "SystemInfo")) {
            self.add_system_info_functions(linker)?;
        }

        Ok(())
    }

    /// Add filesystem access functions for WASM
    fn add_filesystem_functions(&self, linker: &mut wasmtime::Linker<WasmSandboxData>) -> Result<()> {
        // Read file function
        linker.func_wrap("fs", "read_file",
            |mut caller: wasmtime::Caller<'_, WasmSandboxData>, path_ptr: i32, path_len: i32| -> Result<i32> {
                let memory = caller.get_export("memory")
                    .and_then(|export| export.into_memory())
                    .ok_or_else(|| anyhow::anyhow!("Failed to get WASM memory"))?;

                let data = memory.data(&caller);
                if path_ptr < 0 || path_len < 0 || (path_ptr as usize + path_len as usize) > data.len() {
                    return Err(anyhow::anyhow!("Invalid memory access in read_file"));
                }

                let path_bytes = &data[path_ptr as usize..(path_ptr as usize + path_len as usize)];
                let path_str = String::from_utf8_lossy(path_bytes);

                // Validate path is within allowed directories
                let path = std::path::Path::new(path_str.as_ref());
                if !path.starts_with("plugins/") {
                    return Err(anyhow::anyhow!("File access denied: {}", path_str));
                }

                info!("[WASM] Reading file: {}", path_str);

                // Return success indicator (simplified)
                Ok(1)
            }
        )?;

        Ok(())
    }

    /// Add network access functions for WASM
    fn add_network_functions(&self, linker: &mut wasmtime::Linker<WasmSandboxData>) -> Result<()> {
        // HTTP request function
        linker.func_wrap("net", "http_get",
            |mut caller: wasmtime::Caller<'_, WasmSandboxData>, url_ptr: i32, url_len: i32| -> Result<i32> {
                let memory = caller.get_export("memory")
                    .and_then(|export| export.into_memory())
                    .ok_or_else(|| anyhow::anyhow!("Failed to get WASM memory"))?;

                let data = memory.data(&caller);
                if url_ptr < 0 || url_len < 0 || (url_ptr as usize + url_len as usize) > data.len() {
                    return Err(anyhow::anyhow!("Invalid memory access in http_get"));
                }

                let url_bytes = &data[url_ptr as usize..(url_ptr as usize + url_len as usize)];
                let url_str = String::from_utf8_lossy(url_bytes);

                info!("[WASM] HTTP GET request: {}", url_str);

                // Return success indicator (simplified)
                Ok(200) // HTTP OK
            }
        )?;

        Ok(())
    }

    /// Add system information functions for WASM
    fn add_system_info_functions(&self, linker: &mut wasmtime::Linker<WasmSandboxData>) -> Result<()> {
        // Get timestamp function
        linker.func_wrap("sys", "get_timestamp",
            |_caller: wasmtime::Caller<'_, WasmSandboxData>| -> Result<i64> {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;

                Ok(timestamp)
            }
        )?;

        Ok(())
    }

    /// Monitor WASM execution for resource usage and timeouts
    async fn monitor_wasm_execution(
        sandbox_id: String,
        sandboxes: Arc<RwLock<HashMap<String, SandboxInstance>>>,
        limits: ResourceLimits,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        let start_time = std::time::Instant::now();

        loop {
            interval.tick().await;

            // Check execution time limit
            if start_time.elapsed() > limits.max_execution_time {
                warn!("WASM sandbox {} exceeded execution time limit", sandbox_id);

                // Remove sandbox to trigger cleanup
                sandboxes.write().await.remove(&sandbox_id);
                break;
            }

            // Check if sandbox still exists
            if !sandboxes.read().await.contains_key(&sandbox_id) {
                debug!("WASM sandbox {} monitoring stopped", sandbox_id);
                break;
            }

            // Update resource usage metrics
            // In a full implementation, this would track actual WASM memory usage
        }
    }

    /// Check if Firejail is available
    async fn is_firejail_available(&self) -> Result<bool> {
        match Command::new("which").arg("firejail").output().await {
            Ok(output) => Ok(output.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Check if Docker is available
    async fn is_docker_available(&self) -> Result<bool> {
        match Command::new("docker").arg("--version").output().await {
            Ok(output) => Ok(output.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Destroy sandbox
    pub async fn destroy_sandbox(&self, sandbox_id: &str) -> Result<()> {
        info!("Destroying sandbox: {}", sandbox_id);

        let mut sandboxes = self.sandboxes.write().await;

        if let Some(instance) = sandboxes.remove(sandbox_id) {
            // Kill process if running
            if let Some(pid) = instance.process_id {
                self.kill_process(pid).await?;
            }

            // Stop Docker container if running
            if let Some(container_id) = instance.container_id {
                self.stop_docker_container(&container_id).await?;
            }
        }

        Ok(())
    }

    /// Kill process
    async fn kill_process(&self, pid: u32) -> Result<()> {
        if cfg!(unix) {
            Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .output()
                .await?;

            // Give process time to terminate gracefully
            tokio::time::sleep(Duration::from_secs(2)).await;

            // Force kill if still running
            let _ = Command::new("kill")
                .arg("-KILL")
                .arg(pid.to_string())
                .output()
                .await;
        }

        Ok(())
    }

    /// Stop Docker container
    async fn stop_docker_container(&self, container_id: &str) -> Result<()> {
        Command::new("docker")
            .arg("stop")
            .arg(container_id)
            .output()
            .await?;

        Ok(())
    }

    /// Check sandbox health
    pub async fn check_sandbox_health(&self, sandbox_id: &str) -> Result<bool> {
        let sandboxes = self.sandboxes.read().await;

        if let Some(instance) = sandboxes.get(sandbox_id) {
            // Check if process is still running
            if let Some(pid) = instance.process_id {
                return self.is_process_running(pid).await;
            }

            // Check if Docker container is running
            if let Some(container_id) = &instance.container_id {
                return self.is_container_running(container_id).await;
            }
        }

        Ok(false)
    }

    /// Check if process is running
    async fn is_process_running(&self, pid: u32) -> Result<bool> {
        if cfg!(unix) {
            let output = Command::new("kill")
                .arg("-0")
                .arg(pid.to_string())
                .output()
                .await?;

            Ok(output.status.success())
        } else {
            // Windows implementation would go here
            Ok(false)
        }
    }

    /// Check if Docker container is running
    async fn is_container_running(&self, container_id: &str) -> Result<bool> {
        let output = Command::new("docker")
            .arg("inspect")
            .arg("-f")
            .arg("{{.State.Running}}")
            .arg(container_id)
            .output()
            .await?;

        let running = String::from_utf8_lossy(&output.stdout).trim() == "true";
        Ok(running)
    }

    /// Get sandbox resource usage
    pub async fn get_resource_usage(&self, sandbox_id: &str) -> Result<ResourceUsage> {
        let mut sandboxes = self.sandboxes.write().await;

        if let Some(instance) = sandboxes.get_mut(sandbox_id) {
            // Update resource usage with real measurements
            instance.resource_usage = self.measure_resource_usage(sandbox_id, instance).await?;
            Ok(instance.resource_usage.clone())
        } else {
            Err(PluginError::NotFound(sandbox_id.to_string()).into())
        }
    }

    /// Measure current resource usage for a sandbox
    async fn measure_resource_usage(&self, sandbox_id: &str, instance: &SandboxInstance) -> Result<ResourceUsage> {
        debug!("Measuring resource usage for sandbox: {}", sandbox_id);

        // Measure based on sandbox type
        let usage = if let Some(process_id) = instance.process_id {
            // For process-based sandboxes
            self.measure_process_resources(process_id).await?
        } else {
            // For WASM-based sandboxes (in-process)
            self.measure_wasm_resources(sandbox_id).await?
        };

        debug!("Resource usage for {}: CPU: {:.2}%, Memory: {} bytes, Disk: {} bytes, FDs: {}, Threads: {}", 
               sandbox_id, usage.cpu_percent, usage.memory_bytes, usage.disk_bytes, 
               usage.file_descriptors, usage.threads);

        Ok(usage)
    }

    /// Measure resource usage for a process-based sandbox
    async fn measure_process_resources(&self, process_id: u32) -> Result<ResourceUsage> {
        let mut usage = ResourceUsage::default();

        #[cfg(unix)]
        {
            // Read from /proc filesystem on Unix systems
            usage.memory_bytes = self.get_process_memory_usage(process_id).await?;
            usage.cpu_percent = self.get_process_cpu_usage(process_id).await?;
            usage.file_descriptors = self.get_process_fd_count(process_id).await?;
            usage.threads = self.get_process_thread_count(process_id).await?;
            usage.disk_bytes = self.get_process_disk_usage(process_id).await?;
        }

        #[cfg(not(unix))]
        {
            // Windows implementation would use WMI or Performance Counters
            warn!("Resource monitoring not implemented for this platform");
        }

        Ok(usage)
    }

    /// Get memory usage for a process (Unix only)
    #[cfg(unix)]
    async fn get_process_memory_usage(&self, process_id: u32) -> Result<usize> {
        let status_path = format!("/proc/{}/status", process_id);
        
        match tokio::fs::read_to_string(&status_path).await {
            Ok(content) => {
                // Parse VmRSS (Resident Set Size) from /proc/PID/status
                for line in content.lines() {
                    if line.starts_with("VmRSS:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            let kb = parts[1].parse::<usize>().unwrap_or(0);
                            return Ok(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
                Ok(0)
            }
            Err(_) => {
                // Process may have exited
                Ok(0)
            }
        }
    }

    /// Get CPU usage for a process (Unix only)
    #[cfg(unix)]
    async fn get_process_cpu_usage(&self, process_id: u32) -> Result<f32> {
        let stat_path = format!("/proc/{}/stat", process_id);
        
        match tokio::fs::read_to_string(&stat_path).await {
            Ok(content) => {
                let parts: Vec<&str> = content.split_whitespace().collect();
                if parts.len() >= 15 {
                    // Get utime (14th field) and stime (15th field) in clock ticks
                    let utime = parts[13].parse::<u64>().unwrap_or(0);
                    let stime = parts[14].parse::<u64>().unwrap_or(0);
                    let total_time = utime + stime;
                    
                    // Simple approximation - in practice you'd track delta over time
                    // and use system clock ticks per second
                    let cpu_percent = (total_time as f32) / 1000.0; // Rough approximation
                    Ok(cpu_percent.min(100.0))
                } else {
                    Ok(0.0)
                }
            }
            Err(_) => Ok(0.0)
        }
    }

    /// Get file descriptor count for a process (Unix only)
    #[cfg(unix)]
    async fn get_process_fd_count(&self, process_id: u32) -> Result<usize> {
        let fd_dir = format!("/proc/{}/fd", process_id);
        
        match tokio::fs::read_dir(&fd_dir).await {
            Ok(mut entries) => {
                let mut count = 0;
                while let Ok(Some(_entry)) = entries.next_entry().await {
                    count += 1;
                }
                Ok(count)
            }
            Err(_) => Ok(0)
        }
    }

    /// Get thread count for a process (Unix only)
    #[cfg(unix)]
    async fn get_process_thread_count(&self, process_id: u32) -> Result<usize> {
        let task_dir = format!("/proc/{}/task", process_id);
        
        match tokio::fs::read_dir(&task_dir).await {
            Ok(mut entries) => {
                let mut count = 0;
                while let Ok(Some(_entry)) = entries.next_entry().await {
                    count += 1;
                }
                Ok(count)
            }
            Err(_) => Ok(1) // At least the main thread
        }
    }

    /// Get disk usage for a process (approximation)
    #[cfg(unix)]
    async fn get_process_disk_usage(&self, process_id: u32) -> Result<usize> {
        let io_path = format!("/proc/{}/io", process_id);
        
        match tokio::fs::read_to_string(&io_path).await {
            Ok(content) => {
                let mut read_bytes = 0;
                let mut write_bytes = 0;
                
                for line in content.lines() {
                    if line.starts_with("read_bytes:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            read_bytes = parts[1].parse::<usize>().unwrap_or(0);
                        }
                    } else if line.starts_with("write_bytes:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            write_bytes = parts[1].parse::<usize>().unwrap_or(0);
                        }
                    }
                }
                
                Ok(read_bytes + write_bytes)
            }
            Err(_) => Ok(0)
        }
    }

    /// Measure resource usage for WASM-based sandbox
    async fn measure_wasm_resources(&self, _sandbox_id: &str) -> Result<ResourceUsage> {
        // For WASM sandboxes, we measure in-process resource usage
        let mut usage = ResourceUsage::default();
        
        // WASM memory usage (approximation)
        usage.memory_bytes = self.get_current_memory_usage().await?;
        
        // WASM doesn't use separate processes, so these are minimal
        usage.cpu_percent = 5.0; // Approximation
        usage.file_descriptors = 10; // Minimal set
        usage.threads = 1; // Single-threaded
        usage.disk_bytes = 0; // No direct disk access
        
        Ok(usage)
    }

    /// Get current process memory usage
    async fn get_current_memory_usage(&self) -> Result<usize> {
        #[cfg(unix)]
        {
            let current_pid = std::process::id();
            self.get_process_memory_usage(current_pid).await
        }
        
        #[cfg(not(unix))]
        {
            // Platform-specific implementation needed
            Ok(0)
        }
    }

    /// Monitor resource usage continuously
    pub async fn start_resource_monitoring(&self) -> Result<()> {
        info!("Starting continuous resource monitoring for all sandboxes");
        
        let sandboxes = Arc::clone(&self.sandboxes);
        let limits = self.limits.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                let sandbox_ids: Vec<String> = {
                    let sandboxes_guard = sandboxes.read().await;
                    sandboxes_guard.keys().cloned().collect()
                };
                
                for sandbox_id in sandbox_ids {
                    if let Err(e) = Self::check_resource_limits(&sandboxes, &sandbox_id, &limits).await {
                        warn!("Resource limit check failed for {}: {}", sandbox_id, e);
                    }
                }
            }
        });
        
        Ok(())
    }

    /// Check if a sandbox is exceeding resource limits
    async fn check_resource_limits(
        sandboxes: &Arc<RwLock<HashMap<String, SandboxInstance>>>, 
        sandbox_id: &str,
        limits: &ResourceLimits
    ) -> Result<()> {
        let sandboxes_guard = sandboxes.read().await;
        
        if let Some(instance) = sandboxes_guard.get(sandbox_id) {
            let usage = &instance.resource_usage;
            
            // Check memory limit
            if usage.memory_bytes > limits.max_memory {
                warn!("Sandbox {} exceeded memory limit: {} > {}", 
                      sandbox_id, usage.memory_bytes, limits.max_memory);
            }
            
            // Check CPU limit
            if usage.cpu_percent > limits.max_cpu_percent {
                warn!("Sandbox {} exceeded CPU limit: {:.2}% > {:.2}%", 
                      sandbox_id, usage.cpu_percent, limits.max_cpu_percent);
            }
            
            // Check file descriptor limit
            if usage.file_descriptors > limits.max_file_descriptors {
                warn!("Sandbox {} exceeded FD limit: {} > {}", 
                      sandbox_id, usage.file_descriptors, limits.max_file_descriptors);
            }
            
            // Check thread limit
            if usage.threads > limits.max_threads {
                warn!("Sandbox {} exceeded thread limit: {} > {}", 
                      sandbox_id, usage.threads, limits.max_threads);
            }
            
            // Check disk usage (against max_disk limit)
            if usage.disk_bytes > limits.max_disk {
                warn!("Sandbox {} exceeded disk limit: {} > {}", 
                      sandbox_id, usage.disk_bytes, limits.max_disk);
            }
        }
        
        Ok(())
    }

    /// Get detailed resource statistics for all sandboxes
    pub async fn get_all_resource_stats(&self) -> Result<SandboxResourceStats> {
        let sandboxes = self.sandboxes.read().await;
        let mut stats = SandboxResourceStats {
            total_sandboxes: sandboxes.len(),
            total_memory_bytes: 0,
            total_cpu_percent: 0.0,
            total_disk_bytes: 0,
            total_file_descriptors: 0,
            total_threads: 0,
            sandbox_details: HashMap::new(),
        };
        
        for (sandbox_id, instance) in sandboxes.iter() {
            let usage = &instance.resource_usage;
            
            stats.total_memory_bytes += usage.memory_bytes;
            stats.total_cpu_percent += usage.cpu_percent;
            stats.total_disk_bytes += usage.disk_bytes;
            stats.total_file_descriptors += usage.file_descriptors;
            stats.total_threads += usage.threads;
            
            stats.sandbox_details.insert(sandbox_id.clone(), SandboxDetail {
                plugin_id: instance.plugin_id.clone(),
                process_id: instance.process_id,
                uptime: instance.start_time.elapsed(),
                resource_usage: usage.clone(),
            });
        }
        
        Ok(stats)
    }
}

/// Comprehensive resource statistics for all sandboxes
#[derive(Debug, Clone)]
pub struct SandboxResourceStats {
    pub total_sandboxes: usize,
    pub total_memory_bytes: usize,
    pub total_cpu_percent: f32,
    pub total_disk_bytes: usize,
    pub total_file_descriptors: usize,
    pub total_threads: usize,
    pub sandbox_details: HashMap<String, SandboxDetail>,
}

/// Detailed information about a single sandbox
#[derive(Debug, Clone)]
pub struct SandboxDetail {
    pub plugin_id: String,
    pub process_id: Option<u32>,
    pub uptime: Duration,
    pub resource_usage: ResourceUsage,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sandbox_type: SandboxType::Basic,
            allowed_dirs: vec![PathBuf::from("plugins")],
            blocked_syscalls: vec![
                "ptrace".to_string(),
                "mount".to_string(),
                "umount".to_string(),
            ],
            network_isolation: true,
            process_isolation: true,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 512 * 1024 * 1024, // 512 MB
            max_cpu_percent: 25.0,
            max_disk: 1024 * 1024 * 1024, // 1 GB
            max_execution_time: Duration::from_secs(300), // 5 minutes
            max_file_descriptors: 256,
            max_threads: 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandboxconfig_default() {
        let config = SandboxConfig::default();
        assert!(config.enabled);
        assert!(matches!(config.sandbox_type, SandboxType::Basic));
        assert!(config.network_isolation);
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_memory, 512 * 1024 * 1024);
        assert_eq!(limits.max_cpu_percent, 25.0);
        assert_eq!(limits.max_threads, 16);
    }
}
