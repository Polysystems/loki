use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::process::Command as TokioCommand;
use tracing::{debug, info, warn};

use super::registry::{ModelCapabilities, ModelSpecialization};
use crate::ollama::{OllamaClient, OllamaManager};

/// Local model discovery service that automatically detects and activates local
/// models
#[derive(Debug, Clone)]
pub struct LocalModelDiscoveryService {
    ollama_client: Option<OllamaClient>,
    lm_studio_client: Option<LMStudioClient>,
    discovered_models: HashMap<String, DiscoveredModel>,
}

/// Represents a discovered local model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub id: String,
    pub name: String,
    pub source: ModelSource,
    pub capabilities: ModelCapabilities,
    pub specializations: Vec<ModelSpecialization>,
    pub size_gb: f64,
    pub context_window: usize,
    pub parameters: u64,
    pub is_available: bool,
    pub is_activated: bool,
    pub activation_error: Option<String>,
}

/// Model source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    Ollama { model_name: String, tags: Vec<String> },
    LMStudio { model_path: PathBuf, model_file: String },
}

/// LM Studio client for detecting and communicating with LM Studio models
#[derive(Debug, Clone)]
pub struct LMStudioClient {
    base_url: String,
    models_path: PathBuf,
}

impl LMStudioClient {
    pub fn new() -> Self {
        let base_url =
            std::env::var("LM_STUDIO_URL").unwrap_or_else(|_| "http://localhost:1234".to_string());

        // Common LM Studio model paths
        let models_path = if cfg!(target_os = "macos") {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".cache/lm-studio/models")
        } else if cfg!(target_os = "windows") {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("AppData/Roaming/LM Studio/models")
        } else {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".cache/lm-studio/models")
        };

        Self { base_url, models_path }
    }

    /// Check if LM Studio is running
    pub async fn is_running(&self) -> bool {
        match reqwest::get(&format!("{}/v1/models", self.base_url)).await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    /// Get models from LM Studio API
    pub async fn get_api_models(&self) -> Result<Vec<LMStudioModel>> {
        let url = format!("{}/v1/models", self.base_url);
        let response = reqwest::get(&url).await?;

        if !response.status().is_success() {
            return Ok(Vec::new());
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<LMStudioModel>,
        }

        let models_response: ModelsResponse = response.json().await?;
        Ok(models_response.data)
    }

    /// Scan for locally downloaded models
    pub async fn scan_local_models(&self) -> Result<Vec<LMStudioModel>> {
        let mut models = Vec::new();

        if !self.models_path.exists() {
            return Ok(models);
        }

        let mut entries = tokio::fs::read_dir(&self.models_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(model) = self.scan_model_directory(&path).await {
                    models.push(model);
                }
            }
        }

        Ok(models)
    }

    /// Scan a single model directory
    async fn scan_model_directory(&self, dir: &Path) -> Result<LMStudioModel> {
        let model_name =
            dir.file_name().and_then(|name| name.to_str()).unwrap_or("unknown").to_string();

        // Look for GGUF files
        let mut gguf_files = Vec::new();
        let mut entries = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "gguf") {
                gguf_files.push(path);
            }
        }

        let model_file = gguf_files
            .first()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("model.gguf")
            .to_string();

        // Estimate parameters from file size
        let size_bytes = if let Some(gguf_file) = gguf_files.first() {
            tokio::fs::metadata(gguf_file).await?.len()
        } else {
            0
        };

        let size_gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let estimated_params = Self::estimate_parameters_from_size(size_gb);

        Ok(LMStudioModel {
            id: model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "lm-studio".to_string(),
            model_name,
            model_file,
            size_gb,
            estimated_params,
        })
    }

    /// Estimate model parameters from file size
    fn estimate_parameters_from_size(size_gb: f64) -> u64 {
        // Rough estimates for common quantization levels
        if size_gb < 1.0 {
            1_000_000_000 // 1B
        } else if size_gb < 2.5 {
            3_000_000_000 // 3B
        } else if size_gb < 5.0 {
            7_000_000_000 // 7B
        } else if size_gb < 8.0 {
            13_000_000_000 // 13B
        } else if size_gb < 15.0 {
            20_000_000_000 // 20B
        } else if size_gb < 25.0 {
            34_000_000_000 // 34B
        } else {
            70_000_000_000 // 70B+
        }
    }
}

/// LM Studio model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMStudioModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    pub model_name: String,
    pub model_file: String,
    pub size_gb: f64,
    pub estimated_params: u64,
}

impl LocalModelDiscoveryService {
    /// Create a new discovery service
    pub async fn new() -> Result<Self> {
        info!("🔍 Initializing Local Model Discovery Service");

        // Try to connect to Ollama
        let ollama_client = match OllamaClient::new("http://localhost:11434") {
            Ok(client) => match client.health_check().await {
                Ok(_) => {
                    info!("✅ Ollama service detected and available");
                    Some(client)
                }
                Err(_) => {
                    info!("⚠️  Ollama service not running, checking if installed...");
                    if OllamaManager::is_installed() {
                        info!("📦 Ollama is installed but not running");
                        None
                    } else {
                        info!("❌ Ollama not installed");
                        None
                    }
                }
            },
            Err(_) => {
                info!("❌ Failed to initialize Ollama client");
                None
            }
        };

        // Try to connect to LM Studio
        let lm_studio_client = LMStudioClient::new();
        let lm_studio_available = lm_studio_client.is_running().await;

        if lm_studio_available {
            info!("✅ LM Studio service detected and available");
        } else {
            info!("⚠️  LM Studio service not running, will check for local models");
        }

        Ok(Self {
            ollama_client,
            lm_studio_client: Some(lm_studio_client),
            discovered_models: HashMap::new(),
        })
    }

    /// Discover all available local models
    pub async fn discover_models(&mut self) -> Result<Vec<DiscoveredModel>> {
        info!("🔍 Starting local model discovery...");

        self.discovered_models.clear();

        // Discover Ollama models
        if let Some(ref ollama_client) = self.ollama_client {
            match self.discover_ollama_models(ollama_client).await {
                Ok(models) => {
                    info!("📦 Discovered {} Ollama models", models.len());
                    for model in models {
                        self.discovered_models.insert(model.id.clone(), model);
                    }
                }
                Err(e) => {
                    warn!("Failed to discover Ollama models: {}", e);
                }
            }
        }

        // Discover LM Studio models
        if let Some(ref lm_studio_client) = self.lm_studio_client {
            match self.discover_lm_studio_models(lm_studio_client).await {
                Ok(models) => {
                    info!("🏠 Discovered {} LM Studio models", models.len());
                    for model in models {
                        self.discovered_models.insert(model.id.clone(), model);
                    }
                }
                Err(e) => {
                    warn!("Failed to discover LM Studio models: {}", e);
                }
            }
        }

        let total_models = self.discovered_models.len();
        info!("✅ Discovery complete: {} total models found", total_models);

        Ok(self.discovered_models.values().cloned().collect())
    }

    /// Discover Ollama models
    async fn discover_ollama_models(&self, client: &OllamaClient) -> Result<Vec<DiscoveredModel>> {
        let mut models = Vec::new();

        let ollama_models = client.list_models().await?;

        for model_info in ollama_models {
            let model_name = &model_info.name; // Keep a reference for multiple uses
            let capabilities = self.estimate_capabilities_from_name(model_name);
            let specializations = self.estimate_specializations_from_name(model_name);
            let size_gb = model_info.size as f64 / (1024.0 * 1024.0 * 1024.0);

            let discovered_model = DiscoveredModel {
                id: format!("ollama_{}", model_name.replace([':', '/'], "_")),
                name: model_info.name.clone(),
                source: ModelSource::Ollama {
                    model_name: model_info.name.clone(), // Clone to avoid move
                    tags: vec![],                        // Could be enhanced to parse tags
                },
                capabilities,
                specializations,
                size_gb,
                context_window: self.estimate_context_window_from_name(model_name),
                parameters: Self::estimate_parameters_from_size(size_gb),
                is_available: true,
                is_activated: false,
                activation_error: None,
            };

            models.push(discovered_model);
        }

        Ok(models)
    }

    /// Discover LM Studio models
    async fn discover_lm_studio_models(
        &self,
        client: &LMStudioClient,
    ) -> Result<Vec<DiscoveredModel>> {
        let mut models = Vec::new();

        // Try API models first (running models)
        if let Ok(api_models) = client.get_api_models().await {
            for lm_model in api_models {
                let capabilities = self.estimate_capabilities_from_name(&lm_model.model_name);
                let specializations = self.estimate_specializations_from_name(&lm_model.model_name);

                let discovered_model = DiscoveredModel {
                    id: format!("lmstudio_{}", lm_model.id.replace([':', '/', ' '], "_")),
                    name: lm_model.model_name.clone(),
                    source: ModelSource::LMStudio {
                        model_path: client.models_path.clone(),
                        model_file: lm_model.model_file,
                    },
                    capabilities,
                    specializations,
                    size_gb: lm_model.size_gb,
                    context_window: self.estimate_context_window_from_name(&lm_model.model_name),
                    parameters: lm_model.estimated_params,
                    is_available: true,
                    is_activated: false,
                    activation_error: None,
                };

                models.push(discovered_model);
            }
        }

        // Scan local models directory
        if let Ok(local_models) = client.scan_local_models().await {
            for lm_model in local_models {
                // Skip if already found via API
                if models.iter().any(|m| m.name == lm_model.model_name) {
                    continue;
                }

                let capabilities = self.estimate_capabilities_from_name(&lm_model.model_name);
                let specializations = self.estimate_specializations_from_name(&lm_model.model_name);

                let discovered_model = DiscoveredModel {
                    id: format!("lmstudio_{}", lm_model.id.replace([':', '/', ' '], "_")),
                    name: lm_model.model_name.clone(),
                    source: ModelSource::LMStudio {
                        model_path: client.models_path.clone(),
                        model_file: lm_model.model_file,
                    },
                    capabilities,
                    specializations,
                    size_gb: lm_model.size_gb,
                    context_window: self.estimate_context_window_from_name(&lm_model.model_name),
                    parameters: lm_model.estimated_params,
                    is_available: true,
                    is_activated: false,
                    activation_error: None,
                };

                models.push(discovered_model);
            }
        }

        Ok(models)
    }

    /// Automatically activate the best available models
    pub async fn auto_activate_models(&mut self) -> Result<Vec<String>> {
        info!("⚡ Auto-activating best available local models...");

        let mut activated = Vec::new();

        // Collect model IDs and scores to avoid borrowing issues
        let mut model_scores: Vec<(String, f64)> = self
            .discovered_models
            .values()
            .map(|model| (model.id.clone(), self.calculate_activation_score(model)))
            .collect();

        // Sort by score (highest first)
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Activate up to 3 best models (to avoid resource exhaustion)
        for (model_id, _score) in model_scores.iter().take(3) {
            if let Ok(()) = self.activate_model(model_id).await {
                activated.push(model_id.clone());
                if let Some(model) = self.discovered_models.get(model_id) {
                    info!("✅ Auto-activated model: {}", model.name);
                }
            }
        }

        if activated.is_empty() {
            info!("⚠️  No models were auto-activated");
        } else {
            info!("🎉 Auto-activated {} models: {:?}", activated.len(), activated);
        }

        Ok(activated)
    }

    /// Calculate activation score for a model (higher is better)
    fn calculate_activation_score(&self, model: &DiscoveredModel) -> f64 {
        let mut score = 0.0;

        // Prefer smaller models for auto-activation
        if model.size_gb < 2.0 {
            score += 3.0;
        } else if model.size_gb < 5.0 {
            score += 2.0;
        } else if model.size_gb < 10.0 {
            score += 1.0;
        }

        // Prefer coding models
        if model.specializations.contains(&ModelSpecialization::CodeGeneration) {
            score += 2.0;
        }

        // Prefer models with good general capabilities
        if model.capabilities.reasoning > 0.7 {
            score += 1.0;
        }

        // Prefer Ollama models (easier to activate)
        if matches!(model.source, ModelSource::Ollama { .. }) {
            score += 1.0;
        }

        score
    }

    /// Activate a specific model
    pub async fn activate_model(&mut self, model_id: &str) -> Result<()> {
        let model = self
            .discovered_models
            .get_mut(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if model.is_activated {
            return Ok(());
        }

        info!("⚡ Activating model: {}", model.name);

        match &model.source {
            ModelSource::Ollama { model_name, .. } => {
                // For Ollama, we just need to ensure the model is available
                // The actual activation will be handled by the LocalModelManager
                debug!("Model {} is available in Ollama", model_name);
                model.is_activated = true;
                model.activation_error = None;
            }
            ModelSource::LMStudio { .. } => {
                // For LM Studio, check if the service is running
                if let Some(ref client) = self.lm_studio_client {
                    if client.is_running().await {
                        model.is_activated = true;
                        model.activation_error = None;
                    } else {
                        let error = "LM Studio service is not running".to_string();
                        model.activation_error = Some(error.clone());
                        return Err(anyhow::anyhow!(error));
                    }
                }
            }
        }

        Ok(())
    }

    /// Get all discovered models
    pub fn get_discovered_models(&self) -> &HashMap<String, DiscoveredModel> {
        &self.discovered_models
    }

    /// Get activated models
    pub fn get_activated_models(&self) -> Vec<&DiscoveredModel> {
        self.discovered_models.values().filter(|m| m.is_activated).collect()
    }

    /// Estimate capabilities from model name
    fn estimate_capabilities_from_name(&self, name: &str) -> ModelCapabilities {
        let name_lower = name.to_lowercase();

        // Base capabilities
        let mut capabilities = ModelCapabilities {
            code_generation: 0.3,
            code_review: 0.2,
            reasoning: 0.4,
            creative_writing: 0.3,
            data_analysis: 0.3,
            mathematical_computation: 0.3,
            language_translation: 0.3,
            context_window: 4096,
            max_tokens_per_second: 30.0,
            supports_streaming: true,
            supports_function_calling: false,
        };

        // Adjust based on model name patterns
        if name_lower.contains("code") || name_lower.contains("coder") {
            capabilities.code_generation = 0.9;
            capabilities.code_review = 0.8;
            capabilities.reasoning = 0.7;
        }

        if name_lower.contains("llama") || name_lower.contains("mistral") {
            capabilities.reasoning = 0.8;
            capabilities.creative_writing = 0.7;
        }

        if name_lower.contains("3b") || name_lower.contains("3.8b") {
            capabilities.max_tokens_per_second = 60.0;
        } else if name_lower.contains("7b") {
            capabilities.max_tokens_per_second = 40.0;
        } else if name_lower.contains("13b") || name_lower.contains("15b") {
            capabilities.max_tokens_per_second = 25.0;
        }

        capabilities
    }

    /// Estimate specializations from model name
    fn estimate_specializations_from_name(&self, name: &str) -> Vec<ModelSpecialization> {
        let name_lower = name.to_lowercase();
        let mut specializations = Vec::new();

        if name_lower.contains("code") || name_lower.contains("coder") {
            specializations.push(ModelSpecialization::CodeGeneration);
            specializations.push(ModelSpecialization::CodeReview);
        }

        if name_lower.contains("chat") || name_lower.contains("instruct") {
            specializations.push(ModelSpecialization::GeneralPurpose);
        }

        if name_lower.contains("reasoning") || name_lower.contains("logic") {
            specializations.push(ModelSpecialization::LogicalReasoning);
        }

        // Default to general purpose if no specific specialization detected
        if specializations.is_empty() {
            specializations.push(ModelSpecialization::GeneralPurpose);
        }

        specializations
    }

    /// Estimate context window from model name
    fn estimate_context_window_from_name(&self, name: &str) -> usize {
        let name_lower = name.to_lowercase();

        if name_lower.contains("32k") || name_lower.contains("32768") {
            32768
        } else if name_lower.contains("16k") || name_lower.contains("16384") {
            16384
        } else if name_lower.contains("8k") || name_lower.contains("8192") {
            8192
        } else if name_lower.contains("4k") || name_lower.contains("4096") {
            4096
        } else if name_lower.contains("2k") || name_lower.contains("2048") {
            2048
        } else {
            // Default context window
            4096
        }
    }

    /// Estimate parameter count from model size
    fn estimate_parameters_from_size(size_gb: f64) -> u64 {
        // Rough estimates for common quantization levels
        if size_gb < 1.0 {
            1_000_000_000 // 1B
        } else if size_gb < 2.5 {
            3_000_000_000 // 3B
        } else if size_gb < 5.0 {
            7_000_000_000 // 7B
        } else if size_gb < 8.0 {
            13_000_000_000 // 13B
        } else if size_gb < 15.0 {
            20_000_000_000 // 20B
        } else if size_gb < 25.0 {
            34_000_000_000 // 34B
        } else {
            70_000_000_000 // 70B+
        }
    }
}

/// Auto-start Ollama service if installed but not running
pub async fn auto_start_ollama() -> Result<bool> {
    if !OllamaManager::is_installed() {
        return Ok(false);
    }

    // Check if already running
    if let Ok(client) = OllamaClient::new("http://localhost:11434") {
        if client.health_check().await.is_ok() {
            return Ok(true);
        }
    }

    info!("🚀 Starting Ollama service...");

    // Try to start Ollama service
    let output = TokioCommand::new("ollama").arg("serve").spawn();

    match output {
        Ok(mut child) => {
            // Give it time to start
            tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

            // Check if it's running
            if let Ok(client) = OllamaClient::new("http://localhost:11434") {
                if client.health_check().await.is_ok() {
                    info!("✅ Ollama service started successfully");
                    return Ok(true);
                }
            }

            // If not running, try to kill the process
            let _ = child.kill().await;
            warn!("⚠️  Failed to start Ollama service");
        }
        Err(e) => {
            warn!("⚠️  Failed to spawn Ollama service: {}", e);
        }
    }

    Ok(false)
}
