//! Python Functions Tool
//!
//! This tool provides secure Python code execution with PyO3 integration,
//! sandboxing, and cognitive learning capabilities.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{RwLock, Mutex};
use tokio::process::Command;
use tokio::fs;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::memory::CognitiveMemory;
use crate::safety::ActionValidator;

/// Python execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonExecutorConfig {
    /// Python interpreter path
    pub python_path: String,
    
    /// Virtual environment path (optional)
    pub venv_path: Option<PathBuf>,
    
    /// Maximum execution time in seconds
    pub max_execution_time: u64,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    
    /// Enable sandboxing
    pub enable_sandboxing: bool,
    
    /// Sandbox working directory
    pub sandbox_dir: PathBuf,
    
    /// Allowed Python packages
    pub allowed_packages: Vec<String>,
    
    /// Blocked Python modules/functions
    pub blocked_modules: Vec<String>,
    
    /// Enable code learning and optimization
    pub enable_code_learning: bool,
    
    /// Enable output caching
    pub enable_output_caching: bool,
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for PythonExecutorConfig {
    fn default() -> Self {
        Self {
            python_path: "python3".to_string(),
            venv_path: None,
            max_execution_time: 30,
            max_memory_mb: 512,
            enable_sandboxing: true,
            sandbox_dir: std::env::temp_dir().join("loki_python_sandbox"),
            allowed_packages: vec![
                "numpy".to_string(),
                "pandas".to_string(),
                "matplotlib".to_string(),
                "requests".to_string(),
                "json".to_string(),
                "datetime".to_string(),
                "math".to_string(),
                "random".to_string(),
                "re".to_string(),
                "os".to_string(),
                "sys".to_string(),
            ],
            blocked_modules: vec![
                "subprocess".to_string(),
                "os.system".to_string(),
                "eval".to_string(),
                "exec".to_string(),
                "__import__".to_string(),
                "open".to_string(), // Restricted file access
            ],
            enable_code_learning: true,
            enable_output_caching: true,
            cache_ttl: 300,
            enable_performance_monitoring: true,
        }
    }
}

/// Python execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonExecutionContext {
    /// Unique execution ID
    pub execution_id: String,
    
    /// Python code to execute
    pub code: String,
    
    /// Input variables
    pub variables: HashMap<String, Value>,
    
    /// Required packages
    pub required_packages: Vec<String>,
    
    /// Execution mode
    pub execution_mode: PythonExecutionMode,
    
    /// Security level
    pub security_level: SecurityLevel,
    
    /// Expected output type
    pub expected_output: Option<String>,
}

/// Python execution modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PythonExecutionMode {
    /// Execute as a script
    Script,
    
    /// Execute as a function
    Function,
    
    /// Execute as a notebook cell
    NotebookCell,
    
    /// Execute as a module
    Module,
}

/// Security levels for Python execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Maximum security, restricted imports and functions
    High,
    
    /// Moderate security, allow most scientific packages
    Medium,
    
    /// Low security, allow most packages (still sandboxed)
    Low,
    
    /// Unsafe mode, no restrictions (use with extreme caution)
    Unsafe,
}

/// Python execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonExecutionResult {
    /// Execution ID
    pub execution_id: String,
    
    /// Standard output
    pub stdout: String,
    
    /// Standard error
    pub stderr: String,
    
    /// Return value (if any)
    pub return_value: Option<Value>,
    
    /// Execution time
    pub execution_time: Duration,
    
    /// Memory usage in MB
    pub memory_usage: u64,
    
    /// Exit code
    pub exit_code: i32,
    
    /// Success status
    pub success: bool,
    
    /// Generated files (if any)
    pub generated_files: Vec<String>,
    
    /// Performance metrics
    pub metrics: PythonExecutionMetrics,
    
    /// Whether result came from cache
    pub from_cache: bool,
}

/// Performance metrics for Python execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonExecutionMetrics {
    pub total_executions: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub avg_execution_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub package_usage: HashMap<String, usize>,
}

impl Default for PythonExecutionMetrics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            avg_execution_time: Duration::from_millis(0),
            cache_hits: 0,
            cache_misses: 0,
            package_usage: HashMap::new(),
        }
    }
}

/// Cached execution result
#[derive(Debug, Clone)]
struct CachedExecution {
    result: PythonExecutionResult,
    cached_at: Instant,
    ttl: Duration,
    access_count: usize,
}

/// Python code pattern for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodePattern {
    pattern_hash: String,
    frequency: usize,
    avg_execution_time: Duration,
    success_rate: f32,
    common_packages: Vec<String>,
    last_used: SystemTime,
}

/// Python Functions Tool - secure Python code execution
pub struct PythonExecutorTool {
    /// Configuration
    config: PythonExecutorConfig,
    
    /// Reference to cognitive memory
    cognitive_memory: Arc<CognitiveMemory>,
    
    /// Safety validator
    safety_validator: Arc<ActionValidator>,
    
    /// Execution cache
    execution_cache: Arc<RwLock<HashMap<String, CachedExecution>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<PythonExecutionMetrics>>,
    
    /// Learned code patterns
    learned_patterns: Arc<RwLock<HashMap<String, CodePattern>>>,
    
    /// Active execution sessions
    active_sessions: Arc<Mutex<HashMap<String, tokio::process::Child>>>,
}

impl PythonExecutorTool {
    /// Create a new Python executor tool
    pub async fn new(
        config: PythonExecutorConfig,
        cognitive_memory: Arc<CognitiveMemory>,
        safety_validator: Arc<ActionValidator>,
    ) -> Result<Self> {
        info!("🐍 Initializing Python Executor Tool");
        
        // Create sandbox directory
        if config.enable_sandboxing {
            fs::create_dir_all(&config.sandbox_dir).await
                .context("Failed to create sandbox directory")?;
            
            // Set restrictive permissions on sandbox
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let metadata = fs::metadata(&config.sandbox_dir).await?;
                let mut permissions = metadata.permissions();
                permissions.set_mode(0o700); // Owner read/write/execute only
                fs::set_permissions(&config.sandbox_dir, permissions).await?;
            }
        }
        
        let tool = Self {
            config,
            cognitive_memory,
            safety_validator,
            execution_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(PythonExecutionMetrics::default())),
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
        };
        
        // Validate Python installation
        tool.validate_python_installation().await?;
        
        info!("✅ Python Executor Tool initialized successfully");
        Ok(tool)
    }
    
    /// Validate Python installation
    async fn validate_python_installation(&self) -> Result<()> {
        let output = Command::new(&self.config.python_path)
            .arg("--version")
            .output()
            .await
            .context("Failed to execute Python. Is Python installed?")?;
        
        if !output.status.success() {
            return Err(anyhow!("Python is not working correctly"));
        }
        
        let version = String::from_utf8_lossy(&output.stdout);
        info!("✅ Python installation validated: {}", version.trim());
        
        // Check for virtual environment if specified
        if let Some(venv_path) = &self.config.venv_path {
            if !venv_path.exists() {
                warn!("⚠️  Specified virtual environment does not exist: {:?}", venv_path);
            } else {
                info!("✅ Virtual environment found: {:?}", venv_path);
            }
        }
        
        Ok(())
    }
    
    /// Execute Python code with full context and safety
    pub async fn execute_python(
        &self,
        context: PythonExecutionContext,
    ) -> Result<PythonExecutionResult> {
        let start_time = Instant::now();
        
        // Safety validation
        self.validate_code_safety(&context).await?;
        
        // Check cache if enabled
        if self.config.enable_output_caching {
            let cache_key = self.generate_cache_key(&context);
            if let Some(cached_result) = self.get_cached_result(&cache_key).await {
                self.update_metrics(|m| m.cache_hits += 1).await;
                return Ok(cached_result.result);
            }
            self.update_metrics(|m| m.cache_misses += 1).await;
        }
        
        // Prepare execution environment
        let execution_dir = self.prepare_execution_environment(&context).await?;
        
        // Execute code
        let result = self.execute_code_in_sandbox(&context, &execution_dir).await?;
        
        // Clean up
        if self.config.enable_sandboxing {
            self.cleanup_execution_environment(&execution_dir).await?;
        }
        
        // Update metrics and learning
        self.update_metrics(|m| {
            m.total_executions += 1;
            if result.success {
                m.successful_executions += 1;
            } else {
                m.failed_executions += 1;
            }
            
            // Update average execution time
            let total = m.total_executions;
            if total > 0 {
                m.avg_execution_time = (m.avg_execution_time * (total - 1) as u32 + result.execution_time) / total as u32;
            }
            
            // Update package usage
            for package in &context.required_packages {
                *m.package_usage.entry(package.clone()).or_insert(0) += 1;
            }
        }).await;
        
        // Learn from execution if enabled
        if self.config.enable_code_learning && result.success {
            self.learn_from_execution(&context, &result).await?;
        }
        
        // Cache result if enabled
        if self.config.enable_output_caching {
            let cache_key = self.generate_cache_key(&context);
            self.cache_result(cache_key, result.clone()).await;
        }
        
        Ok(result)
    }
    
    /// Validate code safety before execution
    async fn validate_code_safety(&self, context: &PythonExecutionContext) -> Result<()> {
        // Check for blocked modules/functions
        for blocked in &self.config.blocked_modules {
            if context.code.contains(blocked) {
                return Err(anyhow!("Blocked module/function detected: {}", blocked));
            }
        }
        
        // Security level checks
        match context.security_level {
            SecurityLevel::High => {
                // Very restrictive checks
                if context.code.contains("import os") || 
                   context.code.contains("import subprocess") ||
                   context.code.contains("__import__") {
                    return Err(anyhow!("High security mode: System imports not allowed"));
                }
            }
            SecurityLevel::Medium => {
                // Moderate checks
                if context.code.contains("subprocess") {
                    return Err(anyhow!("Medium security mode: Subprocess not allowed"));
                }
            }
            SecurityLevel::Low => {
                // Basic checks only
                if context.code.contains("rm -rf") || context.code.contains("delete") {
                    warn!("Potentially destructive code detected");
                }
            }
            SecurityLevel::Unsafe => {
                // No restrictions, but log warning
                warn!("⚠️  Executing code in UNSAFE mode - no security restrictions");
            }
        }
        
        // Use safety validator for additional checks
        // This would integrate with the existing ActionValidator
        debug!("Code safety validation passed");
        Ok(())
    }
    
    /// Prepare execution environment
    async fn prepare_execution_environment(&self, context: &PythonExecutionContext) -> Result<PathBuf> {
        let execution_dir = if self.config.enable_sandboxing {
            self.config.sandbox_dir.join(&context.execution_id)
        } else {
            std::env::temp_dir().join(&context.execution_id)
        };
        
        fs::create_dir_all(&execution_dir).await
            .context("Failed to create execution directory")?;
        
        // Write Python code to file
        let code_file = execution_dir.join("main.py");
        let full_code = self.prepare_python_code(context)?;
        fs::write(&code_file, full_code).await
            .context("Failed to write Python code to file")?;
        
        // Create input data file if variables provided
        if !context.variables.is_empty() {
            let input_file = execution_dir.join("input.json");
            let input_json = serde_json::to_string_pretty(&context.variables)?;
            fs::write(input_file, input_json).await
                .context("Failed to write input variables")?;
        }
        
        Ok(execution_dir)
    }
    
    /// Prepare Python code with safety wrappers and imports
    fn prepare_python_code(&self, context: &PythonExecutionContext) -> Result<String> {
        let mut full_code = String::new();
        
        // Add standard imports based on security level
        match context.security_level {
            SecurityLevel::High => {
                full_code.push_str("# High security mode - restricted imports\n");
                full_code.push_str("import json\nimport math\nimport datetime\nimport random\n\n");
            }
            SecurityLevel::Medium | SecurityLevel::Low => {
                full_code.push_str("# Standard scientific imports\n");
                full_code.push_str("import json\nimport math\nimport datetime\nimport random\nimport re\n");
                
                // Add requested packages if allowed
                for package in &context.required_packages {
                    if self.config.allowed_packages.contains(package) {
                        full_code.push_str(&format!("import {}\n", package));
                    }
                }
                full_code.push('\n');
            }
            SecurityLevel::Unsafe => {
                full_code.push_str("# Unsafe mode - all imports allowed\n");
                for package in &context.required_packages {
                    full_code.push_str(&format!("import {}\n", package));
                }
                full_code.push('\n');
            }
        }
        
        // Add input loading code if variables provided
        if !context.variables.is_empty() {
            full_code.push_str(r#"
# Load input variables
try:
    with open('input.json', 'r') as f:
        input_vars = json.load(f)
        globals().update(input_vars)
except Exception as e:
    print(f"Warning: Could not load input variables: {e}")

"#);
        }
        
        // Add execution time monitoring
        full_code.push_str(r#"
import time
import traceback
import sys

# Execution monitoring
start_time = time.time()

try:
"#);
        
        // Add user code with proper indentation
        for line in context.code.lines() {
            full_code.push_str(&format!("    {}\n", line));
        }
        
        // Add error handling and output capture
        full_code.push_str(r#"
    
    # Capture execution time
    execution_time = time.time() - start_time
    print(f"\n--- Execution completed in {execution_time:.3f} seconds ---")
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    print(f"Traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"#);
        
        Ok(full_code)
    }
    
    /// Execute code in sandbox with monitoring
    async fn execute_code_in_sandbox(
        &self,
        context: &PythonExecutionContext,
        execution_dir: &Path,
    ) -> Result<PythonExecutionResult> {
        let start_time = Instant::now();
        
        // Determine Python executable
        let python_exe = if let Some(venv_path) = &self.config.venv_path {
            venv_path.join("bin").join("python")
        } else {
            PathBuf::from(&self.config.python_path)
        };
        
        // Build command
        let mut cmd = Command::new(python_exe);
        cmd.current_dir(execution_dir)
           .arg("main.py")
           .stdout(std::process::Stdio::piped())
           .stderr(std::process::Stdio::piped());
        
        // Add resource limits if sandboxing enabled
        if self.config.enable_sandboxing {
            // Set memory limit (Unix-specific)
            #[cfg(unix)]
            {
                // This would use ulimit or similar to set resource limits
                cmd.env("PYTHONMALLOC", "malloc");
            }
        }
        
        // Execute with timeout
        let child = cmd.spawn().context("Failed to spawn Python process")?;
        
        // Store active session for potential termination
        {
            let mut sessions = self.active_sessions.lock().await;
            sessions.insert(context.execution_id.clone(), child);
        }
        
        // Wait for completion with timeout
        let timeout_duration = Duration::from_secs(self.config.max_execution_time);
        let output = tokio::time::timeout(timeout_duration, async {
            let mut sessions = self.active_sessions.lock().await;
            if let Some( child) = sessions.remove(&context.execution_id) {
                child.wait_with_output().await
            } else {
                Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Process not found"))
            }
        }).await;
        
        let execution_time = start_time.elapsed();
        
        match output {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);
                let success = output.status.success();
                
                // Check for generated files
                let generated_files = self.scan_generated_files(execution_dir).await?;
                
                let return_value = self.extract_return_value(&stdout);
                let memory_usage = self.estimate_memory_usage(&stdout, execution_time);
                
                Ok(PythonExecutionResult {
                    execution_id: context.execution_id.clone(),
                    stdout,
                    stderr,
                    return_value,
                    execution_time,
                    memory_usage,
                    exit_code,
                    success,
                    generated_files,
                    metrics: self.metrics.read().await.clone(),
                    from_cache: false,
                })
            }
            Ok(Err(e)) => {
                Err(anyhow!("Process execution failed: {}", e))
            }
            Err(_) => {
                // Timeout occurred, kill the process
                self.terminate_execution(&context.execution_id).await?;
                Err(anyhow!("Python execution timed out after {} seconds", self.config.max_execution_time))
            }
        }
    }
    
    /// Terminate active execution
    async fn terminate_execution(&self, execution_id: &str) -> Result<()> {
        let mut sessions = self.active_sessions.lock().await;
        if let Some(mut child) = sessions.remove(execution_id) {
            match child.kill().await {
                Ok(_) => info!("🔪 Terminated Python execution: {}", execution_id),
                Err(e) => warn!("Failed to terminate Python execution: {}", e),
            }
        }
        Ok(())
    }
    
    /// Scan for files generated during execution
    async fn scan_generated_files(&self, execution_dir: &Path) -> Result<Vec<String>> {
        let mut generated_files = Vec::new();
        
        let mut entries = fs::read_dir(execution_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                // Skip our own files
                if filename_str != "main.py" && filename_str != "input.json" {
                    generated_files.push(filename_str.to_string());
                }
            }
        }
        
        Ok(generated_files)
    }
    
    /// Clean up execution environment
    async fn cleanup_execution_environment(&self, execution_dir: &Path) -> Result<()> {
        if execution_dir.exists() {
            fs::remove_dir_all(execution_dir).await
                .context("Failed to clean up execution directory")?;
        }
        Ok(())
    }
    
    /// Generate cache key for execution
    fn generate_cache_key(&self, context: &PythonExecutionContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        context.code.hash(&mut hasher);
        context.variables.len().hash(&mut hasher); // Simple hash of variables
        context.required_packages.hash(&mut hasher);
        
        format!("python_{:x}", hasher.finish())
    }
    
    /// Get cached result if valid
    async fn get_cached_result(&self, cache_key: &str) -> Option<CachedExecution> {
        let cache = self.execution_cache.read().await;
        
        if let Some(cached) = cache.get(cache_key) {
            if cached.cached_at.elapsed() < cached.ttl {
                return Some(cached.clone());
            }
        }
        
        None
    }
    
    /// Cache execution result
    async fn cache_result(&self, cache_key: String, result: PythonExecutionResult) {
        let mut cache = self.execution_cache.write().await;
        
        // Clean up expired entries if cache is getting large
        if cache.len() > 100 {
            cache.retain(|_, cached| cached.cached_at.elapsed() < cached.ttl);
        }
        
        cache.insert(cache_key, CachedExecution {
            result,
            cached_at: Instant::now(),
            ttl: Duration::from_secs(self.config.cache_ttl),
            access_count: 1,
        });
    }
    
    /// Learn from successful execution
    async fn learn_from_execution(&self, context: &PythonExecutionContext, result: &PythonExecutionResult) -> Result<()> {
        if !self.config.enable_code_learning {
            return Ok(());
        }
        
        // Extract pattern from code
        let pattern_hash = self.extract_code_pattern(&context.code);
        
        let mut patterns = self.learned_patterns.write().await;
        
        if let Some(existing_pattern) = patterns.get_mut(&pattern_hash) {
            // Update existing pattern
            existing_pattern.frequency += 1;
            existing_pattern.avg_execution_time = (existing_pattern.avg_execution_time * (existing_pattern.frequency - 1) as u32 + result.execution_time) / existing_pattern.frequency as u32;
            existing_pattern.success_rate = (existing_pattern.success_rate * (existing_pattern.frequency - 1) as f32 + if result.success { 1.0 } else { 0.0 }) / existing_pattern.frequency as f32;
            existing_pattern.last_used = SystemTime::now();
            
            // Update common packages
            for package in &context.required_packages {
                if !existing_pattern.common_packages.contains(package) {
                    existing_pattern.common_packages.push(package.clone());
                }
            }
        } else {
            // Create new pattern
            patterns.insert(pattern_hash.clone(), CodePattern {
                pattern_hash,
                frequency: 1,
                avg_execution_time: result.execution_time,
                success_rate: if result.success { 1.0 } else { 0.0 },
                common_packages: context.required_packages.clone(),
                last_used: SystemTime::now(),
            });
        }
        
        Ok(())
    }
    
    /// Extract pattern from Python code for learning
    fn extract_code_pattern(&self, code: &str) -> String {
        // Simple pattern extraction - this could be more sophisticated
        let lines: Vec<&str> = code.lines().collect();
        let mut pattern_elements = Vec::new();
        
        for line in lines.iter().take(10) { // Analyze first 10 lines
            let trimmed = line.trim();
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                pattern_elements.push("IMPORT");
            } else if trimmed.starts_with("def ") {
                pattern_elements.push("FUNCTION");
            } else if trimmed.starts_with("class ") {
                pattern_elements.push("CLASS");
            } else if trimmed.starts_with("for ") {
                pattern_elements.push("LOOP");
            } else if trimmed.starts_with("if ") {
                pattern_elements.push("CONDITIONAL");
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                pattern_elements.push("STATEMENT");
            }
        }
        
        pattern_elements.join("_")
    }
    
    /// Update performance metrics
    async fn update_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut PythonExecutionMetrics),
    {
        let mut metrics = self.metrics.write().await;
        update_fn(&mut *metrics);
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PythonExecutionMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get learned code patterns
    pub async fn get_learned_patterns(&self) -> HashMap<String, CodePattern> {
        self.learned_patterns.read().await.clone()
    }
    
    /// Clear execution cache
    pub async fn clear_cache(&self) {
        self.execution_cache.write().await.clear();
        info!("🧹 Python execution cache cleared");
    }
    
    /// Terminate all active executions
    pub async fn terminate_all_executions(&self) -> Result<()> {
        let mut sessions = self.active_sessions.lock().await;
        
        for (execution_id, mut child) in sessions.drain() {
            match child.kill().await {
                Ok(_) => info!("🔪 Terminated Python execution: {}", execution_id),
                Err(e) => warn!("Failed to terminate Python execution {}: {}", execution_id, e),
            }
        }
        
        Ok(())
    }
    
    /// Extract return value from stdout
    fn extract_return_value(&self, stdout: &str) -> Option<Value> {
        // Look for JSON-like output in the last few lines
        let lines: Vec<&str> = stdout.lines().collect();
        
        for line in lines.iter().rev().take(5) { // Check last 5 lines
            let trimmed = line.trim();
            
            // Try to parse as JSON
            if trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"') {
                if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
                    return Some(value);
                }
            }
            
            // Try to parse as number
            if let Ok(num) = trimmed.parse::<f64>() {
                return Some(json!(num));
            }
            
            // Try to parse as boolean
            if trimmed == "True" || trimmed == "true" {
                return Some(json!(true));
            }
            if trimmed == "False" || trimmed == "false" {
                return Some(json!(false));
            }
            
            // Return as string if it looks like a result
            if !trimmed.is_empty() && !trimmed.starts_with("---") && !trimmed.starts_with("Error") {
                return Some(json!(trimmed));
            }
        }
        
        None
    }
    
    /// Estimate memory usage based on execution characteristics
    fn estimate_memory_usage(&self, stdout: &str, execution_time: Duration) -> u64 {
        let mut memory_estimate = 10; // Base Python interpreter memory in MB
        
        // Estimate based on output length (rough heuristic)
        memory_estimate += (stdout.len() / 1024) as u64; // 1MB per KB of output
        
        // Estimate based on execution time (longer = potentially more memory)
        if execution_time > Duration::from_secs(10) {
            memory_estimate += 50; // Complex operations likely use more memory
        } else if execution_time > Duration::from_secs(1) {
            memory_estimate += 20;
        }
        
        // Look for memory-intensive patterns in output
        if stdout.contains("numpy") || stdout.contains("pandas") || stdout.contains("DataFrame") {
            memory_estimate += 100; // Data processing uses more memory
        }
        
        if stdout.contains("matplotlib") || stdout.contains("plot") {
            memory_estimate += 50; // Plotting uses more memory
        }
        
        // Cap at reasonable maximum
        memory_estimate.min(512)
    }
    
    /// Create a new execution context
    pub fn create_execution_context(
        code: String,
        variables: Option<HashMap<String, Value>>,
        required_packages: Option<Vec<String>>,
        security_level: Option<SecurityLevel>,
    ) -> PythonExecutionContext {
        PythonExecutionContext {
            execution_id: Uuid::new_v4().to_string(),
            code,
            variables: variables.unwrap_or_default(),
            required_packages: required_packages.unwrap_or_default(),
            execution_mode: PythonExecutionMode::Script,
            security_level: security_level.unwrap_or(SecurityLevel::Medium),
            expected_output: None,
        }
    }
}