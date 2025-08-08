use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::cli::{ConfigArgs, ConfigCommands};

pub mod api_keys;
pub mod network;
pub mod secure_env;
pub mod setup;

pub use api_keys::{
    AiModelsConfig,
    ApiKeysConfig,
    EmbeddingModelsConfig,
    GitHubConfig,
    GoogleSearchConfig,
    OptionalServicesConfig,
    SearchApisConfig,
    VectorDbConfig,
    VectorDbProvider,
    XTwitterConfig,
};
pub use network::{NetworkConfig, get_network_config, NETWORK_CONFIG};
pub use setup::{
    ModelAssignment,
    ModelManagementConfig,
    ModelPreferences,
    ModelReference,
    ModelSize,
    SecureConfig,
    SetupWizard,
    TaskType,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default model to use
    pub default_model: String,

    /// Model configuration
    pub models: ModelConfig,

    /// Task configuration
    pub tasks: TaskConfig,

    /// Streaming configuration
    pub streaming: StreamingConfig,

    /// Device configuration
    pub device: DeviceConfig,

    /// Memory configuration
    pub memory: MemoryConfig,

    /// General settings
    pub settings: Settings,

    /// API keys configuration
    #[serde(default)]
    pub api_keys: ApiKeysConfig,

    #[serde(skip)]
    config_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to store downloaded models
    pub model_dir: PathBuf,

    /// Default model identifier
    pub default_model: String,

    /// Maximum model size in GB
    pub max_model_size: f64,

    /// Preferred quantization level
    pub quantization: String,

    /// Model-specific settings
    pub model_settings: std::collections::HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Enable task caching
    pub cache_enabled: bool,

    /// Cache directory
    pub cache_dir: PathBuf,

    /// Task timeout in seconds
    pub timeout: u64,

    /// Task-specific settings
    pub task_settings: std::collections::HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Chunk size for streaming processing
    pub chunk_size: Option<usize>,

    /// Processing mode (fast, balanced, quality)
    pub processing_mode: Option<String>,

    /// Quality threshold (0.0-1.0)
    pub quality_threshold: Option<f32>,

    /// Buffer size for stream processing
    pub buffer_size: usize,

    /// Enable parallel streaming
    pub enable_parallel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Preferred device for computation (cpu, cuda, metal, etc.)
    pub preferred_device: Option<String>,

    /// Device-specific settings
    pub device_settings: std::collections::HashMap<String, toml::Value>,

    /// Enable device auto-selection
    pub auto_select: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable context memory
    pub enable_context_memory: bool,

    /// Context window size
    pub context_window_size: usize,

    /// Memory cache size in MB
    pub cache_size_mb: usize,

    /// Enable memory compression
    pub enable_compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Enable telemetry (anonymized usage data)
    pub telemetry: bool,

    /// Auto-update models
    pub auto_update: bool,

    /// Number of parallel tasks
    pub parallelism: usize,

    /// Memory limit in MB
    pub memory_limit: usize,
}

impl Config {
    /// Ensure all local storage directories exist
    pub fn ensure_directories() -> Result<()> {
        let dirs = ProjectDirs::from("dev", "loki", "loki")
            .context("Failed to determine project directories")?;

        // Create all necessary local directories
        fs::create_dir_all(dirs.config_dir()).context("Failed to create config directory")?;
        fs::create_dir_all(dirs.data_dir()).context("Failed to create data directory")?;
        fs::create_dir_all(dirs.cache_dir()).context("Failed to create cache directory")?;
        fs::create_dir_all(dirs.data_dir().join("models"))
            .context("Failed to create models directory")?;
        fs::create_dir_all(dirs.data_dir().join("memory"))
            .context("Failed to create memory directory")?;
        fs::create_dir_all(dirs.data_dir().join("logs"))
            .context("Failed to create logs directory")?;
        fs::create_dir_all(dirs.data_dir().join("temp"))
            .context("Failed to create temp directory")?;

        Ok(())
    }

    /// Load configuration from the default location or create if not exists
    pub fn load() -> Result<Self> {
        // First ensure all directories exist
        Self::ensure_directories()?;
        let config_path = Self::config_file_path()?;

        if config_path.exists() {
            let contents =
                fs::read_to_string(&config_path).context("Failed to read configuration file")?;

            // Try to parse the config, but if it fails, regenerate it
            match toml::from_str::<Config>(&contents) {
                Ok(mut config) => {
                    config.config_path = config_path;
                    Ok(config)
                }
                Err(_) => {
                    // Config file exists but has invalid structure, recreate it
                    eprintln!(
                        "Warning: Configuration file has invalid structure, recreating with \
                         defaults..."
                    );
                    let config = Self::default_with_path(config_path);
                    config.save()?;
                    Ok(config)
                }
            }
        } else {
            let config = Self::default_with_path(config_path);
            config.save()?;
            Ok(config)
        }
    }

    /// Save configuration to disk
    pub fn save(&self) -> Result<()> {
        let contents = toml::to_string_pretty(self).context("Failed to serialize configuration")?;

        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create configuration directory")?;
        }

        fs::write(&self.config_path, contents).context("Failed to write configuration file")?;

        Ok(())
    }

    /// Get the configuration file path
    fn config_file_path() -> Result<PathBuf> {
        let dirs = ProjectDirs::from("dev", "loki", "loki")
            .context("Failed to determine configuration directory")?;
        Ok(dirs.config_dir().join("config.toml"))
    }

    /// Create default configuration with specified path
    fn default_with_path(config_path: PathBuf) -> Self {
        let dirs = ProjectDirs::from("dev", "loki", "loki")
            .expect("Failed to determine project directories");

        // Ensure all necessary directories exist
        let _ = fs::create_dir_all(dirs.config_dir());
        let _ = fs::create_dir_all(dirs.data_dir());
        let _ = fs::create_dir_all(dirs.cache_dir());
        let _ = fs::create_dir_all(dirs.data_dir().join("models"));
        let _ = fs::create_dir_all(dirs.data_dir().join("memory"));
        let _ = fs::create_dir_all(dirs.data_dir().join("logs"));

        Config {
            default_model: "phi-3.5-mini".to_string(),
            models: ModelConfig {
                model_dir: dirs.data_dir().join("models"),
                max_model_size: 10.0,
                quantization: "q4_k_m".to_string(),
                model_settings: std::collections::HashMap::new(),
                default_model: "phi-3.5-mini".to_string(),
            },
            tasks: TaskConfig {
                cache_enabled: true,
                cache_dir: dirs.cache_dir().to_path_buf(),
                timeout: 300,
                task_settings: std::collections::HashMap::new(),
            },
            streaming: StreamingConfig {
                chunk_size: Some(1024),
                processing_mode: Some("balanced".to_string()),
                quality_threshold: Some(0.8),
                buffer_size: 1024,
                enable_parallel: true,
            },
            device: DeviceConfig {
                preferred_device: None,
                device_settings: std::collections::HashMap::new(),
                auto_select: true,
            },
            memory: MemoryConfig {
                enable_context_memory: true,
                context_window_size: 8192,
                cache_size_mb: 512,
                enable_compression: true,
            },
            settings: Settings {
                telemetry: false,
                auto_update: true,
                parallelism: num_cpus::get(),
                memory_limit: 4096,
            },
            api_keys: ApiKeysConfig::from_env().unwrap_or_default(),
            config_path,
        }
    }

    /// Get the config path
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
}

impl Default for Config {
    fn default() -> Self {
        let config_path = Self::config_file_path().unwrap_or_else(|_| PathBuf::from("config.toml"));
        Self::default_with_path(config_path)
    }
}

/// Handle config command
pub fn handleconfig_command(args: ConfigArgs, mut config: Config) -> Result<()> {
    match args.command {
        ConfigCommands::Show => {
            println!("{}", toml::to_string_pretty(&config)?);
        }
        ConfigCommands::Set { key, value } => {
            setconfig_value(&mut config, &key, &value)?;
            config.save()?;
            println!("Set {} = {}", key, value);
        }
        ConfigCommands::Get { key } => {
            let value = getconfig_value(&config, &key)?;
            println!("{}", value);
        }
        ConfigCommands::Reset => {
            config = Config::default_with_path(config.config_path.clone());
            config.save()?;
            println!("Configuration reset to defaults");
        }
    }
    Ok(())
}

fn setconfig_value(config: &mut Config, key: &str, value: &str) -> Result<()> {
    match key {
        "default_model" => config.default_model = value.to_string(),
        "models.quantization" => config.models.quantization = value.to_string(),
        "tasks.cache_enabled" => config.tasks.cache_enabled = value.parse()?,
        "tasks.timeout" => config.tasks.timeout = value.parse()?,
        "settings.telemetry" => config.settings.telemetry = value.parse()?,
        "settings.auto_update" => config.settings.auto_update = value.parse()?,
        "settings.parallelism" => config.settings.parallelism = value.parse()?,
        "settings.memory_limit" => config.settings.memory_limit = value.parse()?,
        _ => anyhow::bail!("Unknown configuration key: {}", key),
    }
    Ok(())
}

fn getconfig_value(config: &Config, key: &str) -> Result<String> {
    let value = match key {
        "default_model" => config.default_model.clone(),
        "models.quantization" => config.models.quantization.clone(),
        "tasks.cache_enabled" => config.tasks.cache_enabled.to_string(),
        "tasks.timeout" => config.tasks.timeout.to_string(),
        "settings.telemetry" => config.settings.telemetry.to_string(),
        "settings.auto_update" => config.settings.auto_update.to_string(),
        "settings.parallelism" => config.settings.parallelism.to_string(),
        "settings.memory_limit" => config.settings.memory_limit.to_string(),
        _ => anyhow::bail!("Unknown configuration key: {}", key),
    };
    Ok(value)
}
