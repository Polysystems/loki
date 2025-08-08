use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::config::SetupWizard;

/// Enhanced model information with orchestration capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    // Basic information
    pub name: String,
    pub description: String,
    pub size: u64,
    pub file_name: String,
    pub quantization: String,
    pub parameters: u64,
    pub license: String,
    pub url: Option<String>,
    pub version: Option<String>,

    // Orchestration enhancements
    #[serde(default)]
    pub provider_type: ProviderType,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    #[serde(default)]
    pub specializations: Vec<ModelSpecialization>,
    pub resource_requirements: Option<ResourceRequirements>,
    #[serde(default)]
    pub performance_metrics: RegistryPerformanceMetrics,
}

impl ModelInfo {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: "".to_string(),
            size: 0,
            file_name: "".to_string(),
            quantization: "".to_string(),
            parameters: 0,
            license: "".to_string(),
            url: None,
            version: None,
            provider_type: Default::default(),
            capabilities: Default::default(),
            specializations: vec![],
            resource_requirements: None,
            performance_metrics: Default::default(),
        }
    }
}

/// Model provider type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProviderType {
    Local,
    API,
    Hybrid,
}

/// Model capabilities with scores (0.0-1.0)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCapabilities {
    pub code_generation: f32,
    pub code_review: f32,
    pub reasoning: f32,
    pub creative_writing: f32,
    pub data_analysis: f32,
    pub mathematical_computation: f32,
    pub language_translation: f32,
    pub context_window: usize,
    pub max_tokens_per_second: f32,
    pub supports_streaming: bool,
    pub supports_function_calling: bool,
}

/// Model specializations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelSpecialization {
    CodeGeneration,
    CodeReview,
    LogicalReasoning,
    CreativeWriting,
    DataAnalysis,
    ConversationalAI,
    MultiModal,
    Embedding,
    GeneralPurpose,
}

/// Resource requirements for local models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_memory_gb: f32,
    pub recommended_memory_gb: f32,
    pub min_gpu_memory_gb: Option<f32>,
    pub recommended_gpu_memory_gb: Option<f32>,
    pub cpu_cores: usize,
    pub gpu_layers: Option<usize>,
    pub quantization: QuantizationType,
}

/// Quantization types for local models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    None,
    Q80,
    Q5KM,
    Q4KM,
    Q3KM,
    Custom(String),
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryPerformanceMetrics {
    pub average_latency_ms: f32,
    pub tokens_per_second: f32,
    pub quality_score: f32,
    pub reliability_score: f32,
    pub cost_per_1k_tokens: Option<f32>,
    pub last_updated: Option<String>,
}

/// Registry of available models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self { models: HashMap::new() }
    }

    /// Load registry from disk
    pub async fn load(model_dir: &Path) -> Result<Self> {
        let registry_path = model_dir.join("registry.json");

        if registry_path.exists() {
            debug!("Loading model registry from: {:?}", registry_path);
            let content = tokio::fs::read_to_string(&registry_path).await?;
            let registry: Self = serde_json::from_str(&content)?;
            Ok(registry)
        } else {
            debug!("No existing registry found, creating new one");
            Ok(Self::new())
        }
    }

    /// Save registry to disk
    pub async fn save(&self, model_dir: &Path) -> Result<()> {
        // Ensure directory exists
        tokio::fs::create_dir_all(model_dir).await?;

        let registry_path = model_dir.join("registry.json");
        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(&registry_path, content).await?;

        debug!("Saved model registry to: {:?}", registry_path);
        Ok(())
    }

    /// Add a model to the registry
    pub fn add_model(&mut self, model: ModelInfo) -> Result<()> {
        info!("Adding model to registry: {} ({})", model.name, model.provider_type.as_str());
        self.models.insert(model.name.clone(), model);
        Ok(())
    }

    /// Find models by specialization
    pub fn find_by_specialization(&self, specialization: &ModelSpecialization) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|model| model.specializations.contains(specialization))
            .collect()
    }

    /// Find models by provider type
    pub fn find_by_provider_type(&self, provider_type: &ProviderType) -> Vec<&ModelInfo> {
        self.models.values().filter(|model| &model.provider_type == provider_type).collect()
    }

    /// Get models sorted by capability score for a specific task
    pub fn get_ranked_by_capability(
        &self,
        capability: fn(&ModelCapabilities) -> f32,
    ) -> Vec<(&ModelInfo, f32)> {
        let mut models: Vec<_> =
            self.models.values().map(|model| (model, capability(&model.capabilities))).collect();

        models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        models
    }

    /// Remove a model from the registry
    pub fn remove_model(&mut self, name: &str) -> Result<()> {
        info!("Removing model from registry: {}", name);
        self.models.remove(name).ok_or_else(|| anyhow::anyhow!("Model not found: {}", name))?;
        Ok(())
    }

    /// Get a model by name
    pub async fn get_model(&self, name: &str) -> Option<ModelInfo> {
        match self.models.get(name) {
            Some(model) => Some(model.clone()),
            None => {
                let mut wizzard = SetupWizard::new().unwrap();
                wizzard.scan_local_models(false).await.unwrap();
                wizzard.config.modelconfig.local_models
                    .iter().map(|model| ModelInfo::new(model.clone()))
                    .find(|model| model.name == name.to_string())
                    .clone()
            }
        }
    }

    /// Check if a model exists
    pub fn has_model(&self, name: &str) -> bool {
        self.models.contains_key(name)
    }

    /// List all models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let mut wizzard = SetupWizard::new().unwrap();
        wizzard.scan_local_models(false).await.unwrap();
        let mut models = wizzard
            .config
            .modelconfig
            .local_models
            .iter()
            .map(|i| ModelInfo::new(i.clone()))
            .collect::<Vec<ModelInfo>>();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        models
    }

    /// List models with capability scores for debugging
    pub fn list_models_with_capabilities(&self) -> Vec<(ModelInfo, String)> {
        let mut models: Vec<_> = self
            .models
            .values()
            .map(|model| {
                let caps = format!(
                    "Code:{:.1} Reasoning:{:.1} Creative:{:.1}",
                    model.capabilities.code_generation,
                    model.capabilities.reasoning,
                    model.capabilities.creative_writing
                );
                (model.clone(), caps)
            })
            .collect();
        models.sort_by(|a, b| a.0.name.cmp(&b.0.name));
        models
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            code_generation: 0.5,
            code_review: 0.5,
            reasoning: 0.5,
            creative_writing: 0.5,
            data_analysis: 0.5,
            mathematical_computation: 0.5,
            language_translation: 0.5,
            context_window: 2048,
            max_tokens_per_second: 10.0,
            supports_streaming: false,
            supports_function_calling: false,
        }
    }
}

impl Default for RegistryPerformanceMetrics {
    fn default() -> Self {
        Self {
            average_latency_ms: 1000.0,
            tokens_per_second: 10.0,
            quality_score: 0.7,
            reliability_score: 0.9,
            cost_per_1k_tokens: None,
            last_updated: None,
        }
    }
}

impl RegistryPerformanceMetrics {
    /// Get current metrics as atomic data for anomaly detection
    pub async fn get_current_metrics(&self) -> CurrentMetricsData {
        // Convert the performance metrics to atomic data
        // Using approximate calculations based on the available data
        let estimated_requests = (self.tokens_per_second * 60.0) as u64; // Estimate requests per minute
        let estimated_tokens = (self.tokens_per_second * 3600.0) as u64; // Estimate tokens per hour

        CurrentMetricsData {
            total_requests: estimated_requests,
            total_tokens: estimated_tokens,
            average_latency_ms: self.average_latency_ms,
            quality_score: self.quality_score,
            reliability_score: self.reliability_score,
            cost_per_1k_tokens: self.cost_per_1k_tokens.unwrap_or(0.0),
            last_updated: self.last_updated.clone(),
        }
    }
}

/// Current metrics data for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentMetricsData {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub average_latency_ms: f32,
    pub quality_score: f32,
    pub reliability_score: f32,
    pub cost_per_1k_tokens: f32,
    pub last_updated: Option<String>,
}

impl ProviderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderType::Local => "local",
            ProviderType::API => "api",
            ProviderType::Hybrid => "hybrid",
        }
    }
}

impl Default for ProviderType {
    fn default() -> Self {
        ProviderType::Local
    }
}
