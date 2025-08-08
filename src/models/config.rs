use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::orchestrator::RoutingStrategy;
use super::registry::{ModelCapabilities, ModelSpecialization, QuantizationType};

/// Complete model orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOrchestrationConfig {
    pub models: ModelConfigs,
    pub orchestration: OrchestrationSettings,
    pub resource_management: ResourceManagementConfig,
    pub hardware_requirements: HardwareRequirements,
}

/// Model configurations for local and API models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigs {
    pub local: HashMap<String, LocalModelConfig>,
    pub api: HashMap<String, ApiModelConfig>,
}

/// Configuration for a local model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelConfig {
    pub ollama_name: String,
    pub specializations: Vec<ModelSpecialization>,
    pub priority: u8,
    pub auto_load: bool,

    // Resource configuration
    pub quantization: Option<QuantizationType>,
    pub gpu_layers: Option<usize>,
    pub context_window: Option<usize>,
    pub max_concurrent_requests: Option<u32>,

    // Performance tuning
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,

    // Custom capability overrides
    pub capability_overrides: Option<ModelCapabilities>,
}

/// Configuration for an API model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiModelConfig {
    pub provider: String,
    pub model: String,
    pub role: Option<String>, // e.g., "orchestrator", "specialist"
    pub specializations: Vec<ModelSpecialization>,
    pub priority: u8,
    pub cost_per_token: Option<f32>,

    // Custom capability overrides
    pub capability_overrides: Option<ModelCapabilities>,
}

/// Orchestration behavior settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationSettings {
    pub default_strategy: RoutingStrategy,
    pub fallback_enabled: bool,
    pub prefer_local: bool,
    pub ensemble_mode: Option<String>,

    // Task-specific routing preferences
    pub task_routing: HashMap<String, TaskRoutingConfig>,

    // Performance thresholds
    pub quality_threshold: f32,
    pub latency_threshold_ms: Option<u64>,
    pub cost_threshold_cents: Option<f32>,
}

/// Task-specific routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRoutingConfig {
    pub primary: Vec<String>,
    pub fallback: Vec<String>,
    pub require_streaming: Option<bool>,
    pub max_cost_cents: Option<f32>,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagementConfig {
    pub auto_scaling: bool,
    pub memory_threshold: f32,
    pub gpu_memory_threshold: f32,
    pub load_balancing: String,

    // Model loading behavior
    pub preload_priority_models: bool,
    pub unload_idle_models: bool,
    pub idle_timeout_minutes: u32,
}

/// Hardware requirements and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    pub minimum: HardwareSpec,
    pub recommended: HardwareSpec,

    // Optional hardware detection overrides
    pub force_cpu_only: Option<bool>,
    pub max_gpu_memory_gb: Option<f32>,
    pub max_concurrent_models: Option<usize>,
}

/// Hardware specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub ram_gb: f32,
    pub gpu_memory_gb: Option<f32>,
    pub cpu_cores: usize,
    pub storage_gb: Option<f32>,
}

/// Configuration manager for model orchestration
pub struct ModelConfigManager {
    config: ModelOrchestrationConfig,
    config_path: Option<std::path::PathBuf>,
}

impl ModelConfigManager {
    /// Load configuration from file
    pub async fn load_from_file(config_path: &Path) -> Result<Self> {
        info!("Loading model configuration from: {:?}", config_path);

        let content = tokio::fs::read_to_string(config_path).await?;
        let config: ModelOrchestrationConfig = serde_yaml::from_str(&content)?;

        debug!(
            "Loaded configuration with {} local models and {} API models",
            config.models.local.len(),
            config.models.api.len()
        );

        Ok(Self { config, config_path: Some(config_path.to_path_buf()) })
    }

    /// Load configuration from directory (looks for models.yaml)
    pub async fn load_from_dir(config_dir: &Path) -> Result<Self> {
        let config_path = config_dir.join("models.yaml");

        if config_path.exists() {
            Self::load_from_file(&config_path).await
        } else {
            warn!("No models.yaml found in {:?}, creating default configuration", config_dir);
            let config = Self::create_default();
            config.save_to_file(&config_path).await?;
            Ok(config)
        }
    }

    /// Create default configuration
    pub fn create_default() -> Self {
        let config = ModelOrchestrationConfig {
            models: ModelConfigs {
                local: Self::create_default_local_models(),
                api: Self::create_default_api_models(),
            },
            orchestration: OrchestrationSettings {
                default_strategy: RoutingStrategy::CapabilityBased,
                fallback_enabled: true,
                prefer_local: true,
                ensemble_mode: None,
                task_routing: Self::create_default_task_routing(),
                quality_threshold: 0.7,
                latency_threshold_ms: Some(5000),
                cost_threshold_cents: Some(10.0),
            },
            resource_management: ResourceManagementConfig {
                auto_scaling: true,
                memory_threshold: 0.85,
                gpu_memory_threshold: 0.90,
                load_balancing: "least_connections".to_string(),
                preload_priority_models: true,
                unload_idle_models: true,
                idle_timeout_minutes: 30,
            },
            hardware_requirements: HardwareRequirements {
                minimum: HardwareSpec {
                    ram_gb: 16.0,
                    gpu_memory_gb: Some(6.0),
                    cpu_cores: 4,
                    storage_gb: Some(50.0),
                },
                recommended: HardwareSpec {
                    ram_gb: 32.0,
                    gpu_memory_gb: Some(12.0),
                    cpu_cores: 8,
                    storage_gb: Some(100.0),
                },
                force_cpu_only: None,
                max_gpu_memory_gb: None,
                max_concurrent_models: Some(5),
            },
        };

        Self { config, config_path: None }
    }

    fn create_default_local_models() -> HashMap<String, LocalModelConfig> {
        let mut models = HashMap::new();

        // DeepSeek Coder - Primary code generation model
        models.insert(
            "deepseek_coder_v2".to_string(),
            LocalModelConfig {
                ollama_name: "deepseek-coder:6.7b-instruct".to_string(),
                specializations: vec![
                    ModelSpecialization::CodeGeneration,
                    ModelSpecialization::CodeReview,
                ],
                priority: 1,
                auto_load: true,
                quantization: Some(QuantizationType::Q4KM),
                gpu_layers: Some(32),
                context_window: Some(16384),
                max_concurrent_requests: Some(3),
                temperature: Some(0.1),
                top_p: Some(0.95),
                top_k: Some(40),
                capability_overrides: Some(ModelCapabilities {
                    code_generation: 0.95,
                    code_review: 0.85,
                    reasoning: 0.75,
                    creative_writing: 0.3,
                    data_analysis: 0.6,
                    mathematical_computation: 0.7,
                    language_translation: 0.4,
                    context_window: 16384,
                    max_tokens_per_second: 50.0,
                    supports_streaming: true,
                    supports_function_calling: false,
                }),
            },
        );

        // MagiCoder - Fast code completion
        models.insert(
            "magicoder_7b".to_string(),
            LocalModelConfig {
                ollama_name: "magicoder:7b".to_string(),
                specializations: vec![ModelSpecialization::CodeGeneration],
                priority: 2,
                auto_load: true,
                quantization: Some(QuantizationType::Q4KM),
                gpu_layers: Some(28),
                context_window: Some(8192),
                max_concurrent_requests: Some(5),
                temperature: Some(0.1),
                top_p: Some(0.9),
                top_k: Some(50),
                capability_overrides: Some(ModelCapabilities {
                    code_generation: 0.85,
                    code_review: 0.7,
                    reasoning: 0.6,
                    creative_writing: 0.2,
                    data_analysis: 0.5,
                    mathematical_computation: 0.6,
                    language_translation: 0.3,
                    context_window: 8192,
                    max_tokens_per_second: 75.0,
                    supports_streaming: true,
                    supports_function_calling: false,
                }),
            },
        );

        // WizardCoder - Complex reasoning and code
        models.insert(
            "wizardcoder_34b".to_string(),
            LocalModelConfig {
                ollama_name: "wizardcoder:34b".to_string(),
                specializations: vec![
                    ModelSpecialization::CodeGeneration,
                    ModelSpecialization::LogicalReasoning,
                ],
                priority: 3,
                auto_load: false, // Large model, load on demand
                quantization: Some(QuantizationType::Q3KM),
                gpu_layers: Some(16),
                context_window: Some(8192),
                max_concurrent_requests: Some(2),
                temperature: Some(0.2),
                top_p: Some(0.95),
                top_k: Some(40),
                capability_overrides: Some(ModelCapabilities {
                    code_generation: 0.9,
                    code_review: 0.85,
                    reasoning: 0.85,
                    creative_writing: 0.4,
                    data_analysis: 0.7,
                    mathematical_computation: 0.8,
                    language_translation: 0.5,
                    context_window: 8192,
                    max_tokens_per_second: 25.0,
                    supports_streaming: true,
                    supports_function_calling: false,
                }),
            },
        );

        models
    }

    fn create_default_api_models() -> HashMap<String, ApiModelConfig> {
        let mut models = HashMap::new();

        // Claude 4 - Primary orchestrator for complex reasoning
        models.insert(
            "claude_4".to_string(),
            ApiModelConfig {
                provider: "anthropic".to_string(),
                model: "claude-3-5-sonnet-20241022".to_string(),
                role: Some("orchestrator".to_string()),
                specializations: vec![
                    ModelSpecialization::LogicalReasoning,
                    ModelSpecialization::CreativeWriting,
                    ModelSpecialization::GeneralPurpose,
                ],
                priority: 1,
                cost_per_token: Some(0.000015),
                capability_overrides: Some(ModelCapabilities {
                    code_generation: 0.85,
                    code_review: 0.9,
                    reasoning: 0.95,
                    creative_writing: 0.95,
                    data_analysis: 0.9,
                    mathematical_computation: 0.85,
                    language_translation: 0.9,
                    context_window: 200000,
                    max_tokens_per_second: 100.0,
                    supports_streaming: true,
                    supports_function_calling: true,
                }),
            },
        );

        models
    }

    fn create_default_task_routing() -> HashMap<String, TaskRoutingConfig> {
        let mut routing = HashMap::new();

        routing.insert(
            "code_generation".to_string(),
            TaskRoutingConfig {
                primary: vec!["deepseek_coder_v2".to_string(), "magicoder_7b".to_string()],
                fallback: vec!["claude_4".to_string()],
                require_streaming: Some(false),
                max_cost_cents: Some(5.0),
            },
        );

        routing.insert(
            "code_review".to_string(),
            TaskRoutingConfig {
                primary: vec!["deepseek_coder_v2".to_string()],
                fallback: vec!["claude_4".to_string()],
                require_streaming: Some(false),
                max_cost_cents: Some(3.0),
            },
        );

        routing.insert(
            "logical_reasoning".to_string(),
            TaskRoutingConfig {
                primary: vec!["claude_4".to_string()],
                fallback: vec!["wizardcoder_34b".to_string()],
                require_streaming: Some(false),
                max_cost_cents: Some(10.0),
            },
        );

        routing.insert(
            "creative_writing".to_string(),
            TaskRoutingConfig {
                primary: vec!["claude_4".to_string()],
                fallback: vec![],
                require_streaming: Some(false),
                max_cost_cents: Some(8.0),
            },
        );

        routing
    }

    /// Save configuration to file
    pub async fn save_to_file(&self, config_path: &Path) -> Result<()> {
        info!("Saving model configuration to: {:?}", config_path);

        // Ensure directory exists
        if let Some(parent) = config_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let yaml_content = serde_yaml::to_string(&self.config)?;
        tokio::fs::write(config_path, yaml_content).await?;

        debug!("Configuration saved successfully");
        Ok(())
    }

    /// Save to the original file if loaded from file
    pub async fn save(&self) -> Result<()> {
        match &self.config_path {
            Some(path) => self.save_to_file(path).await,
            None => Err(anyhow::anyhow!("No file path set for configuration")),
        }
    }

    /// Get the configuration
    pub fn getconfig(&self) -> &ModelOrchestrationConfig {
        &self.config
    }

    /// Update configuration
    pub fn updateconfig(&mut self, config: ModelOrchestrationConfig) {
        self.config = config;
    }

    /// Get local model configurations
    pub fn get_local_models(&self) -> &HashMap<String, LocalModelConfig> {
        &self.config.models.local
    }

    /// Get API model configurations
    pub fn get_api_models(&self) -> &HashMap<String, ApiModelConfig> {
        &self.config.models.api
    }

    /// Get models that should be auto-loaded
    pub fn get_auto_load_models(&self) -> Vec<(&String, &LocalModelConfig)> {
        self.config.models.local.iter().filter(|(_, config)| config.auto_load).collect()
    }

    /// Get models by specialization
    pub fn get_models_by_specialization(
        &self,
        specialization: &ModelSpecialization,
    ) -> Vec<String> {
        let mut models = Vec::new();

        // Add local models
        for (name, config) in &self.config.models.local {
            if config.specializations.contains(specialization) {
                models.push(name.clone());
            }
        }

        // Add API models
        for (name, config) in &self.config.models.api {
            if config.specializations.contains(specialization) {
                models.push(name.clone());
            }
        }

        // Sort by priority
        models.sort_by(|a, b| {
            let priority_a = self.get_model_priority(a).unwrap_or(255);
            let priority_b = self.get_model_priority(b).unwrap_or(255);
            priority_a.cmp(&priority_b)
        });

        models
    }

    /// Get model priority
    pub fn get_model_priority(&self, model_name: &str) -> Option<u8> {
        if let Some(config) = self.config.models.local.get(model_name) {
            Some(config.priority)
        } else if let Some(config) = self.config.models.api.get(model_name) {
            Some(config.priority)
        } else {
            None
        }
    }

    /// Get the configuration file path if available
    pub fn getconfig_path(&self) -> Option<&std::path::Path> {
        self.config_path.as_deref()
    }

    /// Validate configuration against hardware requirements
    pub fn validate_hardware(
        &self,
        available_ram_gb: f32,
        available_gpu_gb: Option<f32>,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            recommendations: Vec::new(),
        };

        // Check minimum requirements
        if available_ram_gb < self.config.hardware_requirements.minimum.ram_gb {
            report.is_valid = false;
            report.errors.push(format!(
                "Insufficient RAM: {:.1}GB available, {:.1}GB required",
                available_ram_gb, self.config.hardware_requirements.minimum.ram_gb
            ));
        }

        // Check GPU requirements
        if let Some(min_gpu) = self.config.hardware_requirements.minimum.gpu_memory_gb {
            match available_gpu_gb {
                Some(gpu_gb) if gpu_gb < min_gpu => {
                    report.warnings.push(format!(
                        "Limited GPU memory: {:.1}GB available, {:.1}GB recommended",
                        gpu_gb, min_gpu
                    ));
                }
                None => {
                    report
                        .warnings
                        .push("No GPU detected, local models will use CPU only".to_string());
                }
                _ => {}
            }
        }

        // Check recommended requirements
        if available_ram_gb < self.config.hardware_requirements.recommended.ram_gb {
            report.recommendations.push(format!(
                "Consider upgrading RAM to {:.1}GB for optimal performance",
                self.config.hardware_requirements.recommended.ram_gb
            ));
        }

        Ok(report)
    }
}

/// Hardware validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub recommendations: Vec<String>,
}

impl Default for ModelOrchestrationConfig {
    fn default() -> Self {
        ModelConfigManager::create_default().config
    }
}

/// Orchestration configuration (subset of OrchestrationSettings)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    /// Model selection strategy
    pub selection_strategy: String,

    /// Fallback strategy when primary model fails
    pub fallback_strategy: String,

    /// Load balancing strategy
    pub load_balancing: String,

    /// Request timeout in milliseconds
    pub request_timeout: u64,

    /// Number of retry attempts
    pub retry_attempts: u32,

    /// Enable/disable fallback
    pub fallback_enabled: bool,

    /// Prefer local models
    pub prefer_local: bool,

    /// Quality threshold
    pub quality_threshold: f32,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            selection_strategy: "capability_based".to_string(),
            fallback_strategy: "best_available".to_string(),
            load_balancing: "least_connections".to_string(),
            request_timeout: 30000,
            retry_attempts: 3,
            fallback_enabled: true,
            prefer_local: true,
            quality_threshold: 0.7,
        }
    }
}

impl From<&OrchestrationSettings> for OrchestrationConfig {
    fn from(settings: &OrchestrationSettings) -> Self {
        Self {
            selection_strategy: format!("{:?}", settings.default_strategy).to_lowercase(),
            fallback_strategy: if settings.fallback_enabled {
                "best_available".to_string()
            } else {
                "none".to_string()
            },
            load_balancing: "least_connections".to_string(),
            request_timeout: settings.latency_threshold_ms.unwrap_or(30000),
            retry_attempts: 3,
            fallback_enabled: settings.fallback_enabled,
            prefer_local: settings.prefer_local,
            quality_threshold: settings.quality_threshold,
        }
    }
}
