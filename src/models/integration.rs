use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use super::config::{LocalModelConfig, ModelConfigManager, ModelOrchestrationConfig};
use super::local_manager::{LocalModelManager, ModelLoadConfig};
use super::orchestrator::ModelOrchestrator;
use super::registry::{
    ModelCapabilities,
    ModelSpecialization,
    QuantizationType,
    ResourceRequirements,
};
use crate::config::ApiKeysConfig;

/// Integrated model management system with configuration support
pub struct IntegratedModelSystem {
    config_manager: ModelConfigManager,
    local_manager: Arc<LocalModelManager>,
    orchestrator: Arc<ModelOrchestrator>,
    loaded_models: Arc<RwLock<HashMap<String, ModelStatus>>>,
}

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub model_id: String,
    pub is_loaded: bool,
    pub load_time: Option<std::time::Instant>,
    pub error_count: u32,
    pub last_error: Option<String>,
}

impl IntegratedModelSystem {
    /// Create a new integrated system from configuration
    pub async fn fromconfig(
        config_manager: ModelConfigManager,
        api_keys: &ApiKeysConfig,
    ) -> Result<Self> {
        info!("Initializing integrated model system");

        // Validate hardware requirements
        let (ram_gb, gpu_gb) = Self::detect_hardware().await?;
        let validation = config_manager.validate_hardware(ram_gb, gpu_gb)?;

        if !validation.is_valid {
            error!("Hardware validation failed:");
            for error in &validation.errors {
                error!("  - {}", error);
            }
            return Err(anyhow::anyhow!("Hardware requirements not met"));
        }

        if !validation.warnings.is_empty() {
            warn!("Hardware validation warnings:");
            for warning in &validation.warnings {
                warn!("  - {}", warning);
            }
        }

        // Initialize components
        let local_manager = Arc::new(LocalModelManager::new().await?);
        let orchestrator = Arc::new(ModelOrchestrator::new(api_keys).await?);
        let loaded_models = Arc::new(RwLock::new(HashMap::new()));

        let system = Self { config_manager, local_manager, orchestrator, loaded_models };

        // Load auto-load models
        system.load_auto_load_models().await?;

        info!("Integrated model system initialized successfully");
        Ok(system)
    }

    /// Load from configuration directory
    pub async fn fromconfig_dir(
        config_dir: &std::path::Path,
        api_keys: &ApiKeysConfig,
    ) -> Result<Self> {
        let config_manager = ModelConfigManager::load_from_dir(config_dir).await?;
        Self::fromconfig(config_manager, api_keys).await
    }

    /// Detect available hardware
    async fn detect_hardware() -> Result<(f32, Option<f32>)> {
        // Try to detect system memory
        let ram_gb = tokio::task::spawn_blocking(|| {
            #[cfg(feature = "sys-info")]
            {
                let mut system = sysinfo::System::new_all();
                system.refresh_memory();
                let total_kb = system.total_memory();
                if total_kb > 0 {
                    return total_kb as f32 / (1024.0 * 1024.0 * 1024.0);
                }
            }

            // Fallback detection methods
            #[cfg(target_os = "macos")]
            {
                if let Ok(output) =
                    std::process::Command::new("sysctl").args(&["-n", "hw.memsize"]).output()
                {
                    if let Ok(memsize_str) = String::from_utf8(output.stdout) {
                        if let Ok(memsize) = memsize_str.trim().parse::<u64>() {
                            return memsize as f32 / (1024.0 * 1024.0 * 1024.0);
                        }
                    }
                }
            }

            #[cfg(target_os = "linux")]
            {
                if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                    for line in meminfo.lines() {
                        if line.starts_with("MemTotal:") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 2 {
                                if let Ok(kb) = parts[1].parse::<u64>() {
                                    return kb as f32 / (1024.0 * 1024.0);
                                }
                            }
                        }
                    }
                }
            }

            // Conservative fallback
            16.0
        })
        .await?;

        // Detect GPU memory using multiple methods
        let gpu_gb = Self::detect_gpu_memory().await;

        info!("Detected hardware: {:.1}GB RAM, GPU: {:?}", ram_gb, gpu_gb);
        Ok((ram_gb, gpu_gb))
    }

    /// Comprehensive GPU memory detection across platforms
    async fn detect_gpu_memory() -> Option<f32> {
        // Try multiple detection methods in order of preference
        tokio::task::spawn_blocking(|| {
            // Method 1: NVIDIA GPU detection via nvidia-smi
            if let Some(nvidia_memory) = Self::detect_nvidia_gpu_memory() {
                info!("🎮 Detected NVIDIA GPU: {:.1}GB VRAM", nvidia_memory);
                return Some(nvidia_memory);
            }

            // Method 2: AMD GPU detection via rocm-smi
            if let Some(amd_memory) = Self::detect_amd_gpu_memory() {
                info!("🎮 Detected AMD GPU: {:.1}GB VRAM", amd_memory);
                return Some(amd_memory);
            }

            // Method 3: Apple Silicon GPU detection
            if let Some(apple_memory) = Self::detect_apple_silicon_gpu() {
                info!("🍎 Detected Apple Silicon GPU: {:.1}GB unified memory", apple_memory);
                return Some(apple_memory);
            }

            // Method 4: Intel GPU detection
            if let Some(intel_memory) = Self::detect_intel_gpu_memory() {
                info!("💻 Detected Intel GPU: {:.1}GB memory", intel_memory);
                return Some(intel_memory);
            }

            // Method 5: Generic GPU detection via system info
            if let Some(generic_memory) = Self::detect_generic_gpu_memory() {
                info!("🎯 Detected Generic GPU: {:.1}GB memory", generic_memory);
                return Some(generic_memory);
            }

            debug!("No GPU memory detected");
            None
        })
        .await
        .unwrap_or(None)
    }

    /// Detect NVIDIA GPU memory via nvidia-smi
    fn detect_nvidia_gpu_memory() -> Option<f32> {
        // Try nvidia-smi command
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // Parse the first GPU's memory (in MB)
                    let memory_mb = output_str.lines().next()?.trim().parse::<f32>().ok()?;

                    return Some(memory_mb / 1024.0); // Convert to GB
                }
            }
        }

        // Try nvidia-ml-py alternative method
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("python3")
                .args(&[
                    "-c",
                    "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); \
                     print(pynvml.nvmlDeviceGetMemoryInfo(h).total//1024//1024)",
                ])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        if let Ok(memory_mb) = output_str.trim().parse::<f32>() {
                            return Some(memory_mb / 1024.0);
                        }
                    }
                }
            }
        }

        None
    }

    /// Detect AMD GPU memory via rocm-smi
    fn detect_amd_gpu_memory() -> Option<f32> {
        // Try rocm-smi command
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .args(&["--showmeminfo", "vram", "--csv"])
            .output()
        {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // Parse CSV output for VRAM info
                    for line in output_str.lines().skip(1) {
                        // Skip header
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 2 {
                            if let Ok(memory_mb) = parts[1].trim().parse::<f32>() {
                                return Some(memory_mb / 1024.0);
                            }
                        }
                    }
                }
            }
        }

        // Try alternative ROCm detection
        if let Ok(output) = std::process::Command::new("rocminfo").output() {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // Look for GPU memory information
                    for line in output_str.lines() {
                        if line.contains("Size:") && line.contains("MB") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            for (i, part) in parts.iter().enumerate() {
                                if part == &"Size:" && i + 1 < parts.len() {
                                    if let Ok(memory_mb) = parts[i + 1].parse::<f32>() {
                                        return Some(memory_mb / 1024.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Detect Apple Silicon GPU (unified memory)
    fn detect_apple_silicon_gpu() -> Option<f32> {
        #[cfg(target_os = "macos")]
        {
            // Check if running on Apple Silicon
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(&["-n", "machdep.cpu.brand_string"])
                .output()
            {
                if let Ok(cpu_brand) = String::from_utf8(output.stdout) {
                    if cpu_brand.contains("Apple") {
                        // On Apple Silicon, GPU shares unified memory with CPU
                        // Try to get total memory and estimate GPU portion
                        if let Ok(output) = std::process::Command::new("sysctl")
                            .args(&["-n", "hw.memsize"])
                            .output()
                        {
                            if let Ok(memsize_str) = String::from_utf8(output.stdout) {
                                if let Ok(memsize) = memsize_str.trim().parse::<u64>() {
                                    let total_gb = memsize as f32 / (1024.0 * 1024.0 * 1024.0);

                                    // Estimate GPU available memory (typically 75% of total on
                                    // Apple Silicon)
                                    let gpu_memory = total_gb * 0.75;
                                    return Some(gpu_memory);
                                }
                            }
                        }
                    }
                }
            }

            // Try system_profiler for more detailed GPU info
            if let Ok(output) = std::process::Command::new("system_profiler")
                .args(&["SPDisplaysDataType", "-xml"])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        // Parse XML for GPU memory information
                        if output_str.contains("spdisplays_vram") {
                            // Simple regex-like parsing for VRAM
                            for line in output_str.lines() {
                                if line.contains("spdisplays_vram") {
                                    // Extract memory value
                                    if let Some(start) = line.find(">") {
                                        if let Some(end) = line[start + 1..].find("<") {
                                            let memory_str = &line[start + 1..start + 1 + end];
                                            if let Ok(memory_mb) = memory_str.parse::<f32>() {
                                                return Some(memory_mb / 1024.0);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Detect Intel GPU memory
    fn detect_intel_gpu_memory() -> Option<f32> {
        // Try Intel GPU Top tool
        if let Ok(output) = std::process::Command::new("intel_gpu_top").args(&["-L"]).output() {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // Parse memory information
                    for line in output_str.lines() {
                        if line.contains("Memory") && line.contains("MB") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            for (i, part) in parts.iter().enumerate() {
                                if part.contains("MB") && i > 0 {
                                    if let Ok(memory_mb) = parts[i - 1].parse::<f32>() {
                                        return Some(memory_mb / 1024.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Try vainfo for Intel GPU detection
        if let Ok(output) = std::process::Command::new("vainfo").output() {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    if output_str.contains("Intel") {
                        // Intel integrated GPUs typically share system memory
                        // Estimate based on system memory
                        return Some(2.0); // Conservative estimate for Intel integrated
                    }
                }
            }
        }

        None
    }

    /// Generic GPU memory detection via system tools
    fn detect_generic_gpu_memory() -> Option<f32> {
        // Try lspci for PCI GPU detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("lspci").args(&["-v"]).output() {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let mut in_gpu_section = false;

                        for line in output_str.lines() {
                            // Check for GPU device
                            if line.contains("VGA compatible controller")
                                || line.contains("3D controller")
                                || line.contains("Display controller")
                            {
                                in_gpu_section = true;
                                continue;
                            }

                            // If in GPU section, look for memory information
                            if in_gpu_section {
                                if line.is_empty() {
                                    in_gpu_section = false;
                                    continue;
                                }

                                if line.contains("Memory") && line.contains("prefetchable") {
                                    // Extract memory size
                                    if let Some(start) = line.find("size=") {
                                        let size_part = &line[start + 5..];
                                        if let Some(end) = size_part.find("]") {
                                            let size_str = &size_part[..end];

                                            // Parse different size formats (K, M, G)
                                            if let Some(memory_gb) =
                                                Self::parse_memory_size(size_str)
                                            {
                                                return Some(memory_gb);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Try Windows GPU detection via wmic
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = std::process::Command::new("wmic")
                .args(&["path", "win32_VideoController", "get", "AdapterRAM"])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        for line in output_str.lines() {
                            if let Ok(memory_bytes) = line.trim().parse::<u64>() {
                                if memory_bytes > 0 {
                                    return Some(memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0));
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Parse memory size string (e.g., "4G", "512M", "2048K")
    fn parse_memory_size(size_str: &str) -> Option<f32> {
        let size_str = size_str.trim();

        if size_str.ends_with('G') {
            let number_part = &size_str[..size_str.len() - 1];
            if let Ok(gb) = number_part.parse::<f32>() {
                return Some(gb);
            }
        } else if size_str.ends_with('M') {
            let number_part = &size_str[..size_str.len() - 1];
            if let Ok(mb) = number_part.parse::<f32>() {
                return Some(mb / 1024.0);
            }
        } else if size_str.ends_with('K') {
            let number_part = &size_str[..size_str.len() - 1];
            if let Ok(kb) = number_part.parse::<f32>() {
                return Some(kb / (1024.0 * 1024.0));
            }
        }

        None
    }

    /// Load models marked for auto-loading
    async fn load_auto_load_models(&self) -> Result<()> {
        let auto_load_models = self.config_manager.get_auto_load_models();

        if auto_load_models.is_empty() {
            info!("No models configured for auto-loading");
            return Ok(());
        }

        info!("Loading {} auto-load models", auto_load_models.len());

        // Sort by priority
        let mut sorted_models = auto_load_models;
        sorted_models.sort_by_key(|(_, config)| config.priority);

        for (model_id, config) in sorted_models {
            match self.load_local_model(model_id, config).await {
                Ok(_) => {
                    info!("Successfully loaded model: {}", model_id);
                    self.update_model_status(model_id, true, None).await;
                }
                Err(e) => {
                    error!("Failed to load model {}: {}", model_id, e);
                    self.update_model_status(model_id, false, Some(e.to_string())).await;
                }
            }
        }

        Ok(())
    }

    /// Load a specific local model
    async fn load_local_model(&self, model_id: &str, config: &LocalModelConfig) -> Result<()> {
        debug!("Loading local model: {} ({})", model_id, config.ollama_name);

        let loadconfig = ModelLoadConfig {
            model_id: model_id.to_string(),
            ollama_name: config.ollama_name.clone(),
            capabilities: config.capability_overrides.clone().unwrap_or_else(|| {
                self.estimate_capabilities_from_specializations(&config.specializations)
            }),
            specializations: config.specializations.clone(),
            requirements: self.create_resource_requirements(config),
            max_concurrent_requests: config.max_concurrent_requests.unwrap_or(3),
        };

        self.local_manager.load_model(loadconfig).await
    }

    /// Create resource requirements from configuration
    fn create_resource_requirements(&self, config: &LocalModelConfig) -> ResourceRequirements {
        // Estimate resource requirements based on model configuration
        let base_memory = match config.ollama_name.as_str() {
            name if name.contains("34b") => 20.0,
            name if name.contains("22b") => 14.0,
            name if name.contains("13b") => 8.0,
            name if name.contains("7b") => 4.0,
            name if name.contains("3b") => 2.0,
            name if name.contains("1b") => 1.0,
            _ => 4.0, // Default
        };

        // Adjust for quantization
        let quantization_factor =
            match config.quantization.as_ref().unwrap_or(&QuantizationType::Q4KM) {
                QuantizationType::None => 1.0,
                QuantizationType::Q80 => 0.8,
                QuantizationType::Q5KM => 0.6,
                QuantizationType::Q4KM => 0.5,
                QuantizationType::Q3KM => 0.4,
                QuantizationType::Custom(_) => 0.5,
            };

        let memory_gb = base_memory * quantization_factor;

        ResourceRequirements {
            min_memory_gb: memory_gb,
            recommended_memory_gb: memory_gb * 1.2,
            min_gpu_memory_gb: config.gpu_layers.map(|_| memory_gb * 0.8),
            recommended_gpu_memory_gb: config.gpu_layers.map(|_| memory_gb),
            cpu_cores: 4,
            gpu_layers: config.gpu_layers,
            quantization: config.quantization.clone().unwrap_or(QuantizationType::Q4KM),
        }
    }

    /// Estimate capabilities from specializations
    fn estimate_capabilities_from_specializations(
        &self,
        specializations: &[ModelSpecialization],
    ) -> ModelCapabilities {
        let mut capabilities = ModelCapabilities::default();

        for spec in specializations {
            match spec {
                ModelSpecialization::CodeGeneration => {
                    capabilities.code_generation = 0.9;
                    capabilities.code_review = 0.7;
                }
                ModelSpecialization::CodeReview => {
                    capabilities.code_review = 0.9;
                    capabilities.code_generation = 0.6;
                }
                ModelSpecialization::LogicalReasoning => {
                    capabilities.reasoning = 0.9;
                    capabilities.mathematical_computation = 0.8;
                }
                ModelSpecialization::CreativeWriting => {
                    capabilities.creative_writing = 0.9;
                    capabilities.language_translation = 0.7;
                }
                ModelSpecialization::DataAnalysis => {
                    capabilities.data_analysis = 0.9;
                    capabilities.mathematical_computation = 0.8;
                }
                _ => {}
            }
        }

        capabilities
    }

    /// Update model status
    async fn update_model_status(&self, model_id: &str, is_loaded: bool, error: Option<String>) {
        let mut loaded_models = self.loaded_models.write().await;

        let status = loaded_models.entry(model_id.to_string()).or_insert_with(|| ModelStatus {
            model_id: model_id.to_string(),
            is_loaded: false,
            load_time: None,
            error_count: 0,
            last_error: None,
        });

        status.is_loaded = is_loaded;

        if is_loaded {
            status.load_time = Some(std::time::Instant::now());
            status.last_error = None;
        } else if let Some(err) = error {
            status.error_count += 1;
            status.last_error = Some(err);
        }
    }

    /// Get system status
    pub async fn get_system_status(&self) -> SystemStatus {
        let orchestration_status = self.orchestrator.get_status().await;
        let loaded_models = self.loaded_models.read().await.clone();
        let config = self.config_manager.getconfig();

        SystemStatus {
            orchestration: orchestration_status,
            loaded_models,
            totalconfigured_local: config.models.local.len(),
            totalconfigured_api: config.models.api.len(),
            auto_load_count: self.config_manager.get_auto_load_models().len(),
        }
    }

    /// Get the orchestrator for task execution
    pub fn get_orchestrator(&self) -> &Arc<ModelOrchestrator> {
        &self.orchestrator
    }

    /// Get the local manager
    pub fn get_local_manager(&self) -> &Arc<LocalModelManager> {
        &self.local_manager
    }

    /// Get the configuration
    pub fn getconfig(&self) -> &ModelOrchestrationConfig {
        self.config_manager.getconfig()
    }

    /// Reload configuration
    pub async fn reloadconfig(&mut self) -> Result<()> {
        info!("Reloading model configuration");

        // Reload from file if available
        if let Some(config_path) = self.config_manager.getconfig_path() {
            let oldconfig = self.config_manager.getconfig().clone();
            let newconfig_manager = ModelConfigManager::load_from_file(config_path).await?;
            let newconfig = newconfig_manager.getconfig();

            // Apply configuration changes by comparing old vs new config
            self.applyconfiguration_changes(&oldconfig, newconfig).await?;

            // Update the config manager after successful application
            self.config_manager = newconfig_manager;

            info!("Configuration reloaded and applied successfully");
        } else {
            warn!("No configuration file path available for reload");
        }

        Ok(())
    }

    /// Apply configuration changes by comparing old and new configurations
    async fn applyconfiguration_changes(
        &self,
        oldconfig: &ModelOrchestrationConfig,
        newconfig: &ModelOrchestrationConfig,
    ) -> Result<()> {
        info!("Applying configuration changes");

        // Track configuration changes
        let mut models_to_unload = Vec::new();
        let mut models_to_load = Vec::new();
        let mut models_to_update = Vec::new();

        // Analyze local model changes
        for (model_id, old_modelconfig) in &oldconfig.models.local {
            match newconfig.models.local.get(model_id) {
                Some(new_modelconfig) => {
                    // Model exists in both configs - check if configuration changed
                    if self.has_local_modelconfig_changed(old_modelconfig, new_modelconfig) {
                        info!("Detected configuration change for model: {}", model_id);
                        models_to_update.push((model_id.clone(), new_modelconfig.clone()));
                    }
                }
                None => {
                    // Model was removed
                    info!("Model removed from configuration: {}", model_id);
                    models_to_unload.push(model_id.clone());
                }
            }
        }

        // Find new models to load
        for (model_id, new_modelconfig) in &newconfig.models.local {
            if !oldconfig.models.local.contains_key(model_id) {
                info!("New model added to configuration: {}", model_id);
                models_to_load.push((model_id.clone(), new_modelconfig.clone()));
            }
        }

        // Apply changes in order: unload, update, load

        // 1. Unload removed models
        for model_id in models_to_unload {
            if let Err(e) = self.unload_model(&model_id).await {
                warn!("Failed to unload model {}: {}", model_id, e);
            } else {
                info!("Successfully unloaded model: {}", model_id);
            }
        }

        // 2. Update existing models (unload and reload with new config)
        for (model_id, newconfig) in models_to_update {
            info!("Updating configuration for model: {}", model_id);

            // Unload the model first
            if let Err(e) = self.unload_model(&model_id).await {
                warn!("Failed to unload model {} for update: {}", model_id, e);
                continue;
            }

            // Reload with new configuration
            match self.load_local_model(&model_id, &newconfig).await {
                Ok(_) => {
                    info!("Successfully updated model: {}", model_id);
                    self.update_model_status(&model_id, true, None).await;
                }
                Err(e) => {
                    error!("Failed to reload model {} with new config: {}", model_id, e);
                    self.update_model_status(&model_id, false, Some(e.to_string())).await;
                }
            }
        }

        // 3. Load new models (only if they are configured for auto-loading)
        for (model_id, modelconfig) in models_to_load {
            if modelconfig.auto_load {
                match self.load_local_model(&model_id, &modelconfig).await {
                    Ok(_) => {
                        info!("Successfully loaded new model: {}", model_id);
                        self.update_model_status(&model_id, true, None).await;
                    }
                    Err(e) => {
                        error!("Failed to load new model {}: {}", model_id, e);
                        self.update_model_status(&model_id, false, Some(e.to_string())).await;
                    }
                }
            } else {
                info!("New model {} configured but not set for auto-load", model_id);
            }
        }

        // 4. Check for orchestrator configuration changes
        if self.has_orchestrationconfig_changed(
            &super::OrchestrationConfig::from(&oldconfig.orchestration),
            &super::OrchestrationConfig::from(&newconfig.orchestration),
        ) {
            warn!(
                "Orchestration configuration changed - full restart recommended for changes to \
                 take effect"
            );
            // Note: Some orchestration config changes may require system
            // restart Future enhancement: implement hot-reload for
            // orchestrator configuration
        }

        info!("Configuration changes applied successfully");
        Ok(())
    }

    /// Check if local model configuration has changed significantly
    fn has_local_modelconfig_changed(
        &self,
        oldconfig: &super::LocalModelConfig,
        newconfig: &super::LocalModelConfig,
    ) -> bool {
        // Check critical configuration parameters that require reload
        oldconfig.ollama_name != newconfig.ollama_name
            || oldconfig.gpu_layers != newconfig.gpu_layers
            || oldconfig.quantization != newconfig.quantization
            || oldconfig.max_concurrent_requests != newconfig.max_concurrent_requests
            || oldconfig.specializations != newconfig.specializations
            || oldconfig.capability_overrides != newconfig.capability_overrides
    }

    /// Check if orchestration configuration has changed
    fn has_orchestrationconfig_changed(
        &self,
        oldconfig: &super::OrchestrationConfig,
        newconfig: &super::OrchestrationConfig,
    ) -> bool {
        // Check key orchestration parameters
        oldconfig.selection_strategy != newconfig.selection_strategy
            || oldconfig.fallback_strategy != newconfig.fallback_strategy
            || oldconfig.load_balancing != newconfig.load_balancing
            || oldconfig.request_timeout != newconfig.request_timeout
            || oldconfig.retry_attempts != newconfig.retry_attempts
    }

    /// Load a model on demand
    pub async fn load_model_on_demand(&self, model_id: &str) -> Result<()> {
        if let Some(config) = self.config_manager.get_local_models().get(model_id) {
            info!("Loading model on demand: {}", model_id);
            self.load_local_model(model_id, config).await?;
            self.update_model_status(model_id, true, None).await;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Model configuration not found: {}", model_id))
        }
    }

    /// Unload a model to free resources
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        info!("Unloading model: {}", model_id);
        self.local_manager.unload_model(model_id).await?;
        self.update_model_status(model_id, false, None).await;
        Ok(())
    }
}

/// Comprehensive system status
#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub orchestration: super::orchestrator::OrchestrationStatus,
    pub loaded_models: HashMap<String, ModelStatus>,
    pub totalconfigured_local: usize,
    pub totalconfigured_api: usize,
    pub auto_load_count: usize,
}
