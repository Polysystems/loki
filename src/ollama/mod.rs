use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use reqwest::Client;
use tokio::process::Command as TokioCommand;
use tracing::{info, warn};

pub mod client;
pub mod models;

pub use client::{DetailedGenerationResponse, GenerationOptions, GenerationRequest, OllamaClient};
pub use models::{ModelInfo, ModelRequirements};

/// Ollama service manager
#[allow(dead_code)]
#[derive(Debug)]
pub struct OllamaManager {
    client: OllamaClient,
    models_dir: PathBuf,
    api_url: String,
}

impl OllamaManager {
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        let api_url = std::env::var("OLLAMA_API_URL")
            .unwrap_or_else(|_| crate::config::get_network_config().ollama_url());

        Ok(Self { client: OllamaClient::new(&api_url)?, models_dir, api_url })
    }

    /// Check if Ollama is installed
    pub fn is_installed() -> bool {
        Command::new("ollama").arg("--version").output().is_ok()
    }

    /// Install Ollama if not present
    pub async fn ensure_installed() -> Result<()> {
        if Self::is_installed() {
            info!("Ollama is already installed");
            return Ok(());
        }

        info!("Installing Ollama...");

        #[cfg(target_os = "macos")]
        {
            Self::install_macos().await?;
        }

        #[cfg(target_os = "linux")]
        {
            Self::install_linux().await?;
        }

        #[cfg(target_os = "windows")]
        {
            Self::install_windows().await?;
        }

        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn install_macos() -> Result<()> {
        // Check if homebrew is available
        if Command::new("brew").arg("--version").output().is_ok() {
            let mut cmd = TokioCommand::new("brew").args(&["install", "ollama"]).spawn()?;
            cmd.wait().await?;
        } else {
            // Download and install directly
            let installer_url = "https://ollama.ai/download/Ollama-darwin.zip";
            Self::download_and_install(installer_url).await?;
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn install_linux() -> Result<()> {
        let cmd = TokioCommand::new("curl")
            .args(&["-fsSL", "https://ollama.ai/install.sh"])
            .stdout(std::process::Stdio::piped())
            .spawn()?;

        let output = cmd.wait_with_output().await?;

        let mut install_cmd =
            TokioCommand::new("sh").stdin(std::process::Stdio::piped()).spawn()?;

        if let Some(mut stdin) = install_cmd.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(&output.stdout).await?;
        }

        install_cmd.wait().await?;
        Ok(())
    }

    #[cfg(target_os = "windows")]
    async fn install_windows() -> Result<()> {
        let installer_url = "https://ollama.ai/download/OllamaSetup.exe";
        Self::download_and_install(installer_url).await?;
        Ok(())
    }

    async fn download_and_install(url: &str) -> Result<()> {
        let client = Client::new();
        let response = client.get(url).send().await?;
        let bytes = response.bytes().await?;

        let temp_path = std::env::temp_dir().join("ollama_installer");
        tokio::fs::write(&temp_path, bytes).await?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&temp_path).await?.permissions();
            perms.set_mode(0o755);
            tokio::fs::set_permissions(&temp_path, perms).await?;
        }

        let mut cmd = TokioCommand::new(&temp_path).spawn()?;
        cmd.wait().await?;

        tokio::fs::remove_file(temp_path).await?;
        Ok(())
    }

    /// Start Ollama service
    pub async fn start_service(&self) -> Result<()> {
        // Check if already running
        if self.client.health_check().await.is_ok() {
            info!("Ollama service is already running");
            return Ok(());
        }

        info!("Starting Ollama service...");

        let _cmd = TokioCommand::new("ollama")
            .arg("serve")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("Failed to start Ollama service")?;

        // Give it time to start
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Verify it's running
        for _ in 0..10 {
            if self.client.health_check().await.is_ok() {
                info!("Ollama service started successfully");
                return Ok(());
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        Err(anyhow::anyhow!("Failed to start Ollama service"))
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.client.list_models().await
    }

    /// Pull a model
    pub async fn pull_model(&self, model_name: &str) -> Result<()> {
        // Safety check for empty model name
        if model_name.is_empty() {
            let default_model = "llama3.2:3b-instruct"; // Lightweight default model
            warn!("⚠️ Empty model name provided, using default: {}", default_model);
            info!("Pulling model: {}", default_model);
            return self.client.pull_model(default_model).await;
        }
        
        info!("Pulling model: {}", model_name);
        self.client.pull_model(model_name).await
    }

    /// Check if a model exists locally
    pub async fn has_model(&self, model_name: &str) -> Result<bool> {
        let models = self.list_models().await?;
        Ok(models.iter().any(|m| m.name == model_name))
    }

    /// Get model requirements
    pub async fn get_model_requirements(&self, model_name: &str) -> Result<ModelRequirements> {
        ModelRequirements::estimate(model_name)
    }

    /// Select best model for available resources
    pub async fn select_optimal_model(
        &self,
        available_memory_gb: f32,
        gpu_available: bool,
    ) -> Result<String> {
        let candidates = if gpu_available {
            vec![
                ("llama3.2:3b", 3.0),
                ("phi3.5:3.8b", 4.0),
                ("mistral:7b", 6.0),
                ("llama3.1:8b", 8.0),
                ("codellama:13b", 12.0),
                ("mixtral:8x7b", 24.0),
            ]
        } else {
            vec![
                ("llama3.2:1b", 1.5),
                ("phi3:mini", 2.0),
                ("llama3.2:3b", 3.0),
                ("mistral:7b-q4_0", 4.0),
            ]
        };

        // Find the best model that fits in memory
        let selected = candidates
            .into_iter()
            .filter(|(_, req_gb)| *req_gb <= available_memory_gb * 0.8) // Leave 20% headroom
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(name, _)| name.to_string())
            .unwrap_or_else(|| "llama3.2:1b".to_string());

        info!(
            "Selected model '{}' for {:.1}GB available memory (GPU: {})",
            selected, available_memory_gb, gpu_available
        );

        Ok(selected)
    }

    /// Deploy a model with cognitive capabilities
    pub async fn deploy_cognitive_model(
        &self,
        model_name: &str,
        context_window: usize,
    ) -> Result<CognitiveModel> {
        // Ensure model is available
        if !self.has_model(model_name).await? {
            self.pull_model(model_name).await?;
        }

        Ok(CognitiveModel {
            name: model_name.to_string(),
            client: self.client.clone(),
            context_window,
        })
    }
}

/// A model with cognitive memory capabilities
#[derive(Clone)]
pub struct CognitiveModel {
    name: String,
    client: OllamaClient,
    context_window: usize,
}

impl CognitiveModel {
    /// Generate with context
    pub async fn generate_with_context(&self, prompt: &str, context: &[String]) -> Result<String> {
        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("Context:\n{}\n\nQuery: {}", context.join("\n"), prompt)
        };

        self.client.generate(&self.name, &full_prompt).await
    }

    /// Stream generation with context
    pub async fn stream_with_context(
        &self,
        prompt: &str,
        context: &[String],
    ) -> Result<impl tokio_stream::Stream<Item = Result<String>>> {
        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("Context:\n{}\n\nQuery: {}", context.join("\n"), prompt)
        };

        self.client.stream_generate(&self.name, &full_prompt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_detection() {
        let installed = OllamaManager::is_installed();
        println!("Ollama installed: {}", installed);
    }
}
