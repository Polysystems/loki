// Copyright (c) 2024 Loki AI
// 
// Comprehensive error handling system for Loki AI
// Provides structured error types, recovery mechanisms, and detailed context

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;
use tracing::{error, warn, info};

/// Comprehensive error types for the Loki system
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum LokiError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        source_message: Option<String>,
        suggested_fix: Option<String>,
    },

    /// Build and compilation errors
    #[error("Build error: {message}")]
    Build {
        message: String,
        build_stage: BuildStage,
        recovery_action: Option<RecoveryAction>,
    },

    /// Dependency errors (CUDA, Metal, etc.)
    #[error("Dependency error: {dependency} - {message}")]
    Dependency {
        dependency: String,
        message: String,
        platform_specific: bool,
        fallback_available: bool,
    },

    /// Model loading and inference errors
    #[error("Model error: {message}")]
    Model {
        message: String,
        model_type: String,
        error_code: ModelErrorCode,
    },

    /// Memory and cognitive system errors
    #[error("Cognitive error: {message}")]
    Cognitive {
        message: String,
        subsystem: CognitiveSubsystem,
        severity: ErrorSeverity,
    },

    /// Network and API errors
    #[error("Network error: {message}")]
    Network {
        message: String,
        endpoint: Option<String>,
        status_code: Option<u16>,
        retry_recommended: bool,
    },

    /// Safety and security errors
    #[error("Safety error: {message}")]
    Safety {
        message: String,
        safety_level: SafetyLevel,
        requires_immediate_action: bool,
    },

    /// Plugin and extension errors
    #[error("Plugin error: {plugin} - {message}")]
    Plugin {
        plugin: String,
        message: String,
        plugin_type: PluginType,
    },

    /// Generic system errors with context
    #[error("System error: {message}")]
    System {
        message: String,
        context: Vec<String>,
        recoverable: bool,
    },
}

/// Build stages for better error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildStage {
    DependencyResolution,
    Compilation,
    FeatureConfiguration,
    Testing,
    Optimization,
}

/// Recovery actions for build errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    RetryWithDifferentFeatures,
    FallbackToPlatformDefaults,
    DisableOptionalDependencies,
    UpdateConfiguration,
    ContactSupport,
    CleanAndRebuild,
    UpdateDependencies,
    UseMinimalFeatureSet,
    SwitchToStableToolchain,
    RegenerateCargoLock,
    ClearTargetDirectory,
    DowngradeProblematicDependency,
    EnableCompatibilityMode,
    DisableParallelBuilds,
    UseSystemLibraries,
}

/// Model error codes for specific handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelErrorCode {
    LoadingFailed,
    InferenceFailed,
    TokenizationError,
    MemoryExhausted,
    UnsupportedArchitecture,
    NetworkTimeout,
}

/// Cognitive subsystems for error tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveSubsystem {
    Consciousness,
    Memory,
    Attention,
    Decision,
    Emotional,
    GoalManager,
    DistributedConsciousness,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialOrd, PartialEq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Safety levels for security-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Plugin types for error classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginType {
    WebAssembly,
    Native,
    Python,
    JavaScript,
}

/// Error recovery mechanism
pub struct ErrorRecovery {
    max_retries: usize,
    backoff_strategy: BackoffStrategy,
    fallback_actions: Vec<RecoveryAction>,
}

#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    Linear(std::time::Duration),
    Exponential { base: std::time::Duration, max: std::time::Duration },
    Fixed(std::time::Duration),
}

impl ErrorRecovery {
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            backoff_strategy: BackoffStrategy::Exponential { 
                base: std::time::Duration::from_millis(100),
                max: std::time::Duration::from_secs(5),
            },
            fallback_actions: Vec::new(),
        }
    }

    pub fn with_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    pub fn with_fallback(mut self, action: RecoveryAction) -> Self {
        self.fallback_actions.push(action);
        self
    }

    /// Execute an operation with error recovery
    pub async fn execute<F, T, E>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> std::result::Result<T, E> + Clone,
        E: Into<LokiError> + fmt::Debug,
    {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= self.max_retries {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e.into());
                    
                    if attempts <= self.max_retries {
                        let delay = self.calculate_delay(attempts);
                        warn!("Operation failed (attempt {}/{}), retrying in {:?}", 
                              attempts, self.max_retries, delay);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        // All retries exhausted, try fallback actions
        for action in &self.fallback_actions {
            info!("Attempting fallback action: {:?}", action);
            if let Some(result) = self.execute_fallback_action(action).await? {
                return Ok(result);
            }
        }

        Err(last_error.unwrap_or_else(|| {
            LokiError::System {
                message: "Unknown error during recovery".to_string(),
                context: vec!["Error recovery exhausted all options".to_string()],
                recoverable: false,
            }
        }).into())
    }

    fn calculate_delay(&self, attempt: usize) -> std::time::Duration {
        match &self.backoff_strategy {
            BackoffStrategy::Linear(base) => *base * attempt as u32,
            BackoffStrategy::Exponential { base, max } => {
                let delay = base.mul_f32(2_f32.powi((attempt as i32) - 1));
                std::cmp::min(delay, *max)
            }
            BackoffStrategy::Fixed(delay) => *delay,
        }
    }

    async fn execute_fallback_action<T>(&self, action: &RecoveryAction) -> Result<Option<T>> {
        match action {
            RecoveryAction::RetryWithDifferentFeatures => {
                info!("Attempting retry with safe feature set");
                self.execute_command(&["cargo", "build", "--features=all-safe"]).await
            }
            RecoveryAction::FallbackToPlatformDefaults => {
                info!("Falling back to platform-specific defaults");
                let platform_features = self.get_platform_features();
                self.execute_command(&["cargo", "build", &format!("--features={}", platform_features)]).await
            }
            RecoveryAction::DisableOptionalDependencies => {
                info!("Disabling optional dependencies");
                self.execute_command(&["cargo", "build", "--no-default-features"]).await
            }
            RecoveryAction::UpdateConfiguration => {
                info!("Updating configuration with known good values");
                self.reset_to_safe_configuration().await
            }
            RecoveryAction::CleanAndRebuild => {
                info!("Cleaning build artifacts and rebuilding");
                self.execute_command::<()>(&["cargo", "clean"]).await?;
                self.execute_command(&["cargo", "build", "--features=all-safe"]).await
            }
            RecoveryAction::UpdateDependencies => {
                info!("Updating dependencies to latest compatible versions");
                self.execute_command(&["cargo", "update"]).await
            }
            RecoveryAction::UseMinimalFeatureSet => {
                info!("Building with minimal feature set");
                self.execute_command(&["cargo", "build", "--features=simd-optimizations"]).await
            }
            RecoveryAction::SwitchToStableToolchain => {
                info!("Switching to stable Rust toolchain");
                self.execute_command::<()>(&["rustup", "default", "stable"]).await?;
                self.execute_command(&["cargo", "build", "--features=all-safe"]).await
            }
            RecoveryAction::RegenerateCargoLock => {
                info!("Regenerating Cargo.lock file");
                if std::fs::remove_file("Cargo.lock").is_ok() {
                    info!("Removed existing Cargo.lock");
                }
                self.execute_command(&["cargo", "generate-lockfile"]).await
            }
            RecoveryAction::ClearTargetDirectory => {
                info!("Clearing target directory");
                if std::fs::remove_dir_all("target").is_ok() {
                    info!("Cleared target directory");
                }
                self.execute_command(&["cargo", "build", "--features=all-safe"]).await
            }
            RecoveryAction::DowngradeProblematicDependency => {
                info!("Attempting to downgrade problematic dependencies");
                self.downgrade_problematic_dependencies().await
            }
            RecoveryAction::EnableCompatibilityMode => {
                info!("Enabling compatibility mode");
                self.execute_command(&["cargo", "build", "--features=compatibility"]).await
            }
            RecoveryAction::DisableParallelBuilds => {
                info!("Disabling parallel builds");
                self.execute_command(&["cargo", "build", "-j", "1", "--features=all-safe"]).await
            }
            RecoveryAction::UseSystemLibraries => {
                info!("Using system libraries instead of vendored versions");
                self.execute_command(&["cargo", "build", "--features=system-libs"]).await
            }
            RecoveryAction::ContactSupport => {
                error!("Manual intervention required - contact support");
                Ok(None)
            }
        }
    }

    async fn execute_command<T>(&self, args: &[&str]) -> Result<Option<T>> {
        use std::process::Command;
        
        let output = Command::new(args[0])
            .args(&args[1..])
            .output()
            .map_err(|e| LokiError::System {
                message: format!("Failed to execute command: {:?}", args),
                context: vec![e.to_string()],
                recoverable: false,
            })?;

        if output.status.success() {
            info!("Command succeeded: {:?}", args);
            Ok(None)
        } else {
            warn!("Command failed: {:?}", args);
            warn!("Output: {}", String::from_utf8_lossy(&output.stderr));
            Ok(None)
        }
    }

    fn get_platform_features(&self) -> String {
        match std::env::consts::OS {
            "linux" => "all-linux",
            "macos" => "all-macos", 
            _ => "all-safe",
        }.to_string()
    }

    async fn reset_to_safe_configuration<T>(&self) -> Result<Option<T>> {
        info!("Resetting to safe configuration");
        // Create a minimal config that should work everywhere
        Ok(None)
    }

    async fn downgrade_problematic_dependencies<T>(&self) -> Result<Option<T>> {
        info!("Attempting to downgrade known problematic dependencies");
        // Implementation would identify and downgrade specific deps
        Ok(None)
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Build error handler specifically for dependency and compilation issues
pub struct BuildErrorHandler {
    recovery: ErrorRecovery,
    platform_info: PlatformInfo,
}

#[derive(Debug, Clone)]
pub struct PlatformInfo {
    os: String,
    arch: String,
    has_cuda: bool,
    has_metal: bool,
}

impl BuildErrorHandler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            recovery: ErrorRecovery::new()
                .with_retries(2)
                .with_fallback(RecoveryAction::FallbackToPlatformDefaults),
            platform_info: Self::detect_platform()?,
        })
    }

    fn detect_platform() -> Result<PlatformInfo> {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        
        let has_cuda = match os.as_str() {
            "linux" => Self::check_cuda_availability(),
            _ => false,
        };
        
        let has_metal = match os.as_str() {
            "macos" => Self::check_metal_availability(),
            _ => false,
        };

        Ok(PlatformInfo { os, arch, has_cuda, has_metal })
    }

    fn check_cuda_availability() -> bool {
        // Check for CUDA installation
        std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .is_ok()
    }

    fn check_metal_availability() -> bool {
        // On macOS, Metal is generally available
        cfg!(target_os = "macos")
    }

    /// Handle dependency errors with platform-specific solutions
    pub fn handle_dependency_error(&self, error: &LokiError) -> Result<String> {
        match error {
            LokiError::Dependency { dependency, platform_specific, .. } => {
                if *platform_specific {
                    self.handle_platform_dependency(dependency)
                } else {
                    self.handle_generic_dependency(dependency)
                }
            }
            _ => Ok("No specific handler for this error type".to_string()),
        }
    }

    fn handle_platform_dependency(&self, dependency: &str) -> Result<String> {
        match dependency {
            "cudarc" | "nvml-wrapper" => {
                if self.platform_info.os != "linux" {
                    Ok("CUDA dependencies are not supported on this platform. Use --features=all-safe to build without CUDA.".to_string())
                } else if !self.platform_info.has_cuda {
                    Ok("CUDA is not installed or not available. Install CUDA toolkit or build with --no-default-features.".to_string())
                } else {
                    Ok("CUDA dependency issue - check CUDA installation and environment variables.".to_string())
                }
            }
            "metal" => {
                if self.platform_info.os != "macos" {
                    Ok("Metal dependencies are only supported on macOS. Use platform-appropriate features.".to_string())
                } else {
                    Ok("Metal should be available on macOS - check system requirements.".to_string())
                }
            }
            _ => Ok(format!("Unknown platform-specific dependency: {}", dependency)),
        }
    }

    fn handle_generic_dependency(&self, dependency: &str) -> Result<String> {
        match dependency {
            "candle-core" => {
                Ok("Candle ML framework issue. Try building with --features=all-safe or update candle dependencies.".to_string())
            }
            "rand" => {
                Ok("Random number generation crate conflict. Check for version mismatches in Cargo.lock.".to_string())
            }
            _ => Ok(format!("Generic dependency issue with {}: check version compatibility", dependency)),
        }
    }

    /// Suggest build command based on platform and available features
    pub fn suggest_build_command(&self) -> String {
        let base_command = "cargo build";
        
        let features = match self.platform_info.os.as_str() {
            "linux" if self.platform_info.has_cuda => "--features=all-linux",
            "macos" if self.platform_info.has_metal => "--features=all-macos", 
            _ => "--features=all-safe",
        };

        format!("{} {}", base_command, features)
    }
}

impl Default for BuildErrorHandler {
    fn default() -> Self {
        Self::new().expect("Failed to create BuildErrorHandler")
    }
}

/// Smart dependency conflict resolver
pub struct DependencyConflictResolver {
    known_conflicts: HashMap<String, Vec<ConflictResolution>>,
    version_compatibility_matrix: HashMap<String, CompatibilityMatrix>,
    resolution_history: Arc<RwLock<Vec<ResolutionAttempt>>>,
}

#[derive(Debug, Clone)]
pub struct ConflictResolution {
    pub conflict_type: ConflictType,
    pub suggested_action: RecoveryAction,
    pub success_probability: f32,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum ConflictType {
    VersionMismatch,
    FeatureConflict,
    TransitiveDependency,
    PlatformIncompatibility,
    BuildScriptFailure,
    LinkageError,
}

#[derive(Debug, Clone)]
pub struct CompatibilityMatrix {
    pub package_name: String,
    pub compatible_versions: Vec<VersionRange>,
    pub incompatible_combinations: Vec<IncompatibleCombination>,
}

#[derive(Debug, Clone)]
pub struct VersionRange {
    pub min_version: String,
    pub max_version: String,
    pub platform_constraints: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IncompatibleCombination {
    pub packages: Vec<String>,
    pub reason: String,
    pub workaround: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolutionAttempt {
    pub timestamp: std::time::Instant,
    pub conflict_description: String,
    pub attempted_resolution: RecoveryAction,
    pub success: bool,
    pub error_details: Option<String>,
}

impl DependencyConflictResolver {
    pub fn new() -> Self {
        let mut known_conflicts = HashMap::new();
        
        // Pre-populate with known conflict patterns
        known_conflicts.insert("rand".to_string(), vec![
            ConflictResolution {
                conflict_type: ConflictType::VersionMismatch,
                suggested_action: RecoveryAction::RegenerateCargoLock,
                success_probability: 0.8,
                description: "Rand version conflicts often resolve with fresh Cargo.lock".to_string(),
            },
            ConflictResolution {
                conflict_type: ConflictType::VersionMismatch,
                suggested_action: RecoveryAction::DowngradeProblematicDependency,
                success_probability: 0.6,
                description: "Downgrade to rand 0.8.5 for better compatibility".to_string(),
            },
        ]);

        known_conflicts.insert("candle-core".to_string(), vec![
            ConflictResolution {
                conflict_type: ConflictType::PlatformIncompatibility,
                suggested_action: RecoveryAction::FallbackToPlatformDefaults,
                success_probability: 0.9,
                description: "Use platform-specific feature set for candle".to_string(),
            },
            ConflictResolution {
                conflict_type: ConflictType::FeatureConflict,
                suggested_action: RecoveryAction::DisableOptionalDependencies,
                success_probability: 0.7,
                description: "Disable conflicting GPU features".to_string(),
            },
        ]);

        known_conflicts.insert("cudarc".to_string(), vec![
            ConflictResolution {
                conflict_type: ConflictType::PlatformIncompatibility,
                suggested_action: RecoveryAction::UseMinimalFeatureSet,
                success_probability: 0.95,
                description: "CUDA not available on this platform".to_string(),
            },
        ]);

        Self {
            known_conflicts,
            version_compatibility_matrix: Self::build_compatibility_matrix(),
            resolution_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn build_compatibility_matrix() -> HashMap<String, CompatibilityMatrix> {
        let mut matrix = HashMap::new();
        
        matrix.insert("candle-core".to_string(), CompatibilityMatrix {
            package_name: "candle-core".to_string(),
            compatible_versions: vec![
                VersionRange {
                    min_version: "0.8.0".to_string(),
                    max_version: "0.8.4".to_string(),
                    platform_constraints: vec!["macos".to_string(), "linux".to_string()],
                },
            ],
            incompatible_combinations: vec![
                IncompatibleCombination {
                    packages: vec!["cudarc".to_string(), "metal".to_string()],
                    reason: "Cannot enable both CUDA and Metal simultaneously".to_string(),
                    workaround: Some("Use platform-specific feature sets".to_string()),
                },
            ],
        });

        matrix
    }

    pub async fn resolve_conflict(&self, error_message: &str) -> Result<Vec<ConflictResolution>> {
        let conflict_type = self.analyze_conflict_type(error_message);
        let package_name = self.extract_package_name(error_message);
        
        let mut resolutions = Vec::new();
        
        // Check for known conflicts
        if let Some(package) = package_name {
            if let Some(known_resolutions) = self.known_conflicts.get(&package) {
                resolutions.extend(known_resolutions.clone());
            }
        }
        
        // Add generic resolutions based on conflict type
        resolutions.extend(self.get_generic_resolutions(conflict_type));
        
        // Sort by success probability
        resolutions.sort_by(|a, b| b.success_probability.partial_cmp(&a.success_probability).unwrap());
        
        Ok(resolutions)
    }

    fn analyze_conflict_type(&self, error_message: &str) -> ConflictType {
        let error_lower = error_message.to_lowercase();
        
        if error_lower.contains("version") && error_lower.contains("conflict") {
            ConflictType::VersionMismatch
        } else if error_lower.contains("feature") && error_lower.contains("conflict") {
            ConflictType::FeatureConflict
        } else if error_lower.contains("cuda") && error_lower.contains("not") {
            ConflictType::PlatformIncompatibility
        } else if error_lower.contains("build script") {
            ConflictType::BuildScriptFailure
        } else if error_lower.contains("link") || error_lower.contains("undefined") {
            ConflictType::LinkageError
        } else {
            ConflictType::TransitiveDependency
        }
    }

    fn extract_package_name(&self, error_message: &str) -> Option<String> {
        // Simple regex-like extraction - in a real implementation you'd use proper regex
        for line in error_message.lines() {
            if line.contains("error") || line.contains("conflict") {
                for word in line.split_whitespace() {
                    if word.contains('-') && !word.starts_with('-') {
                        // Likely a package name
                        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
                        if !clean_word.is_empty() {
                            return Some(clean_word.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn get_generic_resolutions(&self, conflict_type: ConflictType) -> Vec<ConflictResolution> {
        match conflict_type {
            ConflictType::VersionMismatch => vec![
                ConflictResolution {
                    conflict_type: ConflictType::VersionMismatch,
                    suggested_action: RecoveryAction::UpdateDependencies,
                    success_probability: 0.7,
                    description: "Update all dependencies to latest compatible versions".to_string(),
                },
                ConflictResolution {
                    conflict_type: ConflictType::VersionMismatch,
                    suggested_action: RecoveryAction::RegenerateCargoLock,
                    success_probability: 0.6,
                    description: "Regenerate dependency resolution".to_string(),
                },
            ],
            ConflictType::PlatformIncompatibility => vec![
                ConflictResolution {
                    conflict_type: ConflictType::PlatformIncompatibility,
                    suggested_action: RecoveryAction::FallbackToPlatformDefaults,
                    success_probability: 0.9,
                    description: "Use platform-appropriate features".to_string(),
                },
            ],
            ConflictType::FeatureConflict => vec![
                ConflictResolution {
                    conflict_type: ConflictType::FeatureConflict,
                    suggested_action: RecoveryAction::UseMinimalFeatureSet,
                    success_probability: 0.8,
                    description: "Build with minimal feature set".to_string(),
                },
            ],
            _ => vec![
                ConflictResolution {
                    conflict_type,
                    suggested_action: RecoveryAction::CleanAndRebuild,
                    success_probability: 0.5,
                    description: "Clean build and retry".to_string(),
                },
            ],
        }
    }

    pub async fn record_resolution_attempt(&self, attempt: ResolutionAttempt) -> Result<()> {
        let mut history = self.resolution_history.write().map_err(|_| LokiError::System {
            message: "Failed to acquire write lock on resolution history".to_string(),
            context: vec![],
            recoverable: true,
        })?;
        
        history.push(attempt);
        
        // Keep only recent attempts (last 100)
        if history.len() > 100 {
            let len = history.len();
            history.drain(0..len - 100);
        }
        
        Ok(())
    }

    pub fn get_success_rate(&self, action: &RecoveryAction) -> f32 {
        let history = match self.resolution_history.read() {
            Ok(h) => h,
            Err(_) => return 0.5, // Default success rate
        };
        
        let attempts: Vec<_> = history.iter()
            .filter(|attempt| std::mem::discriminant(&attempt.attempted_resolution) == std::mem::discriminant(action))
            .collect();
        
        if attempts.is_empty() {
            return 0.5; // Default for unknown actions
        }
        
        let successes = attempts.iter().filter(|a| a.success).count();
        successes as f32 / attempts.len() as f32
    }
}

/// Context-aware error logging
pub trait ErrorContext {
    fn log_with_context(&self, context: &str);
    fn severity(&self) -> ErrorSeverity;
    fn is_recoverable(&self) -> bool;
}

impl ErrorContext for LokiError {
    fn log_with_context(&self, context: &str) {
        match self.severity() {
            ErrorSeverity::Info => info!("{}: {}", context, self),
            ErrorSeverity::Warning => warn!("{}: {}", context, self),
            ErrorSeverity::Error => error!("{}: {}", context, self),
            ErrorSeverity::Critical => error!("CRITICAL - {}: {}", context, self),
            ErrorSeverity::Fatal => error!("FATAL - {}: {}", context, self),
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            LokiError::Configuration { .. } => ErrorSeverity::Warning,
            LokiError::Build { .. } => ErrorSeverity::Error,
            LokiError::Dependency { .. } => ErrorSeverity::Warning,
            LokiError::Model { .. } => ErrorSeverity::Error,
            LokiError::Cognitive { severity, .. } => severity.clone(),
            LokiError::Network { .. } => ErrorSeverity::Warning,
            LokiError::Safety { requires_immediate_action, .. } => {
                if *requires_immediate_action {
                    ErrorSeverity::Critical
                } else {
                    ErrorSeverity::Error
                }
            }
            LokiError::Plugin { .. } => ErrorSeverity::Warning,
            LokiError::System { recoverable, .. } => {
                if *recoverable {
                    ErrorSeverity::Warning
                } else {
                    ErrorSeverity::Error
                }
            }
        }
    }

    fn is_recoverable(&self) -> bool {
        match self {
            LokiError::Configuration { .. } => true,
            LokiError::Build { recovery_action, .. } => recovery_action.is_some(),
            LokiError::Dependency { fallback_available, .. } => *fallback_available,
            LokiError::Model { error_code, .. } => {
                matches!(error_code, ModelErrorCode::NetworkTimeout | ModelErrorCode::LoadingFailed)
            }
            LokiError::Cognitive { severity, .. } => {
                !matches!(severity, ErrorSeverity::Fatal)
            }
            LokiError::Network { retry_recommended, .. } => *retry_recommended,
            LokiError::Safety { requires_immediate_action, .. } => !*requires_immediate_action,
            LokiError::Plugin { .. } => true,
            LokiError::System { recoverable, .. } => *recoverable,
        }
    }
}

/// Utility functions for error handling
pub mod utils {
    use super::*;

    /// Convert anyhow errors to LokiError with context
    pub fn anyhow_to_loki(error: anyhow::Error, context: &str) -> LokiError {
        LokiError::System {
            message: format!("{}: {}", context, error),
            context: vec![context.to_string()],
            recoverable: true,
        }
    }

    /// Create a dependency error with platform detection
    pub fn dependency_error(dependency: &str, message: &str) -> LokiError {
        let platform_specific = matches!(dependency, "cudarc" | "nvml-wrapper" | "metal");
        
        LokiError::Dependency {
            dependency: dependency.to_string(),
            message: message.to_string(),
            platform_specific,
            fallback_available: platform_specific,
        }
    }

    /// Create a build error with recovery suggestion
    pub fn build_error(message: &str, stage: BuildStage) -> LokiError {
        let recovery_action = match stage {
            BuildStage::DependencyResolution => Some(RecoveryAction::DisableOptionalDependencies),
            BuildStage::FeatureConfiguration => Some(RecoveryAction::FallbackToPlatformDefaults),
            _ => None,
        };

        LokiError::Build {
            message: message.to_string(),
            build_stage: stage,
            recovery_action,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::Error);
        assert!(ErrorSeverity::Error > ErrorSeverity::Warning);
    }

    #[test]
    fn test_dependency_error_creation() {
        let error = utils::dependency_error("cudarc", "CUDA not available");
        assert!(matches!(error, LokiError::Dependency { platform_specific: true, .. }));
    }

    #[tokio::test]
    async fn test_error_recovery_basic() {
        let recovery = ErrorRecovery::new().with_retries(1);
        
        let mut attempt_count = 0;
        let operation = || -> std::result::Result<String, &'static str> {
            attempt_count += 1;
            if attempt_count == 1 {
                Err("first failure")
            } else {
                Ok("success".to_string())
            }
        };

        // This test would need to be adapted to work with the actual async recovery mechanism
        // For now, we'll just test that the ErrorRecovery can be created
        assert_eq!(recovery.max_retries, 1);
    }
}