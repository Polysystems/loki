use std::time::Duration;

use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::{Stream, StreamExt};
use tracing::info;

use super::models::ModelInfo;

/// Ollama API client
#[derive(Debug, Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let client = Client::builder().timeout(Duration::from_secs(300)).build()?;

        Ok(Self { client, base_url: base_url.to_string() })
    }

    /// Health check
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("Ollama service is not healthy"))
        }
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await?;

        #[derive(Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelInfo>,
        }

        let models_response: ModelsResponse = response.json().await?;
        Ok(models_response.models)
    }

    /// Pull a model with progress monitoring
    pub async fn pull_model(&self, model_name: &str) -> Result<()> {
        self.pull_model_with_progress(model_name, None::<fn(f32)>).await
    }

    /// Pull a model with optional progress callback
    pub async fn pull_model_with_progress<F>(
        &self,
        model_name: &str,
        progress_callback: Option<F>,
    ) -> Result<()>
    where
        F: Fn(f32) + Send + Sync,
    {
        let url = format!("{}/api/pull", self.base_url);
        let body = json!({
            "name": model_name,
            "stream": true // Enable streaming for progress updates
        });

        let response = self.client.post(&url).json(&body).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Failed to pull model: {}", error_text));
        }

        // Process streaming response for progress updates
        let mut stream = response.bytes_stream();
        let mut last_progress = 0.0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let chunk_str = String::from_utf8_lossy(&chunk);

            // Parse each line as JSON (Ollama sends newline-delimited JSON)
            for line in chunk_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                match serde_json::from_str::<PullProgressResponse>(line) {
                    Ok(progress) => {
                        let progress_percent = extract_progress_percentage(&progress);

                        // Only call callback if progress has changed significantly
                        if (progress_percent - last_progress).abs() > 1.0 {
                            if let Some(ref callback) = progress_callback {
                                callback(progress_percent);
                            }
                            last_progress = progress_percent;
                        }

                        // Check if pull is complete
                        if progress.status.as_deref() == Some("success")
                            || progress_percent >= 100.0
                        {
                            info!("Model {} pulled successfully", model_name);
                            return Ok(());
                        }
                    }
                    Err(_) => {
                        // Skip malformed JSON lines (common in streaming responses)
                        continue;
                    }
                }
            }
        }

        info!("Model {} pulled successfully", model_name);
        Ok(())
    }

    /// Generate completion
    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": model,
            "prompt": prompt,
            "stream": false
        });

        let response = self.client.post(&url).json(&body).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Generation failed: {}", error_text));
        }

        let gen_response: GenerateResponse = response.json().await?;
        Ok(gen_response.response)
    }

    /// Generate text with advanced options
    pub async fn generate_with_options(&self, request: &GenerationRequest) -> Result<String> {
        let url = format!("{}/api/generate", self.base_url);

        let mut body = json!({
            "model": request.model,
            "prompt": request.prompt,
            "stream": false
        });

        // Add generation options if provided
        if let Some(ref options) = request.options {
            let mut options_obj = serde_json::Map::new();

            if let Some(temp) = options.temperature {
                options_obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = options.top_p {
                options_obj.insert("top_p".to_string(), json!(top_p));
            }
            if let Some(top_k) = options.top_k {
                options_obj.insert("top_k".to_string(), json!(top_k));
            }
            if let Some(max_tokens) = options.max_tokens {
                options_obj.insert("num_predict".to_string(), json!(max_tokens));
            }
            if let Some(ref stop) = options.stop {
                options_obj.insert("stop".to_string(), json!(stop));
            }
            if let Some(presence) = options.presence_penalty {
                options_obj.insert("presence_penalty".to_string(), json!(presence));
            }
            if let Some(frequency) = options.frequency_penalty {
                options_obj.insert("frequency_penalty".to_string(), json!(frequency));
            }

            body.as_object_mut().unwrap().insert("options".to_string(), json!(options_obj));
        }

        let response = self.client.post(&url).json(&body).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Generation with options failed: {}", error_text));
        }

        let gen_response: GenerateResponse = response.json().await?;
        Ok(gen_response.response)
    }

    /// Generate text with comprehensive metrics
    pub async fn generate_detailed(
        &self,
        request: &GenerationRequest,
    ) -> Result<DetailedGenerationResponse> {
        let url = format!("{}/api/generate", self.base_url);

        let mut body = json!({
            "model": request.model,
            "prompt": request.prompt,
            "stream": false
        });

        // Add generation options if provided
        if let Some(ref options) = request.options {
            let mut options_obj = serde_json::Map::new();

            if let Some(temp) = options.temperature {
                options_obj.insert("temperature".to_string(), json!(temp));
            }
            if let Some(top_p) = options.top_p {
                options_obj.insert("top_p".to_string(), json!(top_p));
            }
            if let Some(top_k) = options.top_k {
                options_obj.insert("top_k".to_string(), json!(top_k));
            }
            if let Some(max_tokens) = options.max_tokens {
                options_obj.insert("num_predict".to_string(), json!(max_tokens));
            }
            if let Some(ref stop) = options.stop {
                options_obj.insert("stop".to_string(), json!(stop));
            }
            if let Some(presence) = options.presence_penalty {
                options_obj.insert("presence_penalty".to_string(), json!(presence));
            }
            if let Some(frequency) = options.frequency_penalty {
                options_obj.insert("frequency_penalty".to_string(), json!(frequency));
            }

            body.as_object_mut().unwrap().insert("options".to_string(), json!(options_obj));
        }

        let response = self.client.post(&url).json(&body).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Detailed generation failed: {}", error_text));
        }

        let gen_response: GenerateResponse = response.json().await?;

        // Extract performance metrics
        let generation_time_ms = gen_response
            .total_duration
            .map(|d| (d as f64 / 1_000_000.0) as u32) // Convert nanoseconds to milliseconds
            .unwrap_or(0);

        let prompt_tokens = gen_response.prompt_eval_count.unwrap_or(0);
        let tokens_generated = gen_response.eval_count.unwrap_or(0);
        let total_tokens = prompt_tokens + tokens_generated;

        let tokens_per_second = if generation_time_ms > 0 && tokens_generated > 0 {
            (tokens_generated as f32) / (generation_time_ms as f32 / 1000.0)
        } else {
            0.0
        };

        Ok(DetailedGenerationResponse {
            text: gen_response.response,
            tokens_generated,
            generation_time_ms,
            prompt_tokens,
            total_tokens,
            tokens_per_second,
        })
    }

    /// Stream generation
    pub async fn stream_generate(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": model,
            "prompt": prompt,
            "stream": true
        });

        let response = self.client.post(&url).json(&body).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Stream generation failed: {}", error_text));
        }

        let stream = response.bytes_stream().map(|chunk| {
            chunk.map_err(|e| anyhow::anyhow!("Stream error: {}", e)).and_then(|bytes| {
                let text = String::from_utf8_lossy(&bytes);
                let response: StreamResponse = serde_json::from_str(&text)?;
                Ok(response.response)
            })
        });

        Ok(stream)
    }

    /// Create embeddings
    pub async fn create_embeddings(&self, model: &str, prompt: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = json!({
            "model": model,
            "prompt": prompt
        });

        let response = self.client.post(&url).json(&body).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Embeddings failed: {}", error_text));
        }

        let embed_response: EmbeddingsResponse = response.json().await?;
        Ok(embed_response.embedding)
    }
}

#[derive(Deserialize)]
struct PullResponse {
    status: String,
}

#[derive(Deserialize)]
struct PullProgressResponse {
    status: Option<String>,
    digest: Option<String>,
    total: Option<u64>,
    completed: Option<u64>,
}

/// Extract progress percentage from pull response
fn extract_progress_percentage(progress: &PullProgressResponse) -> f32 {
    match (progress.completed, progress.total) {
        (Some(completed), Some(total)) if total > 0 => (completed as f32 / total as f32) * 100.0,
        _ => {
            // If no progress data, estimate based on status
            match progress.status.as_deref() {
                Some("pulling manifest") => 10.0,
                Some("downloading") => 50.0,
                Some("verifying sha256 digest") => 90.0,
                Some("writing manifest") => 95.0,
                Some("removing any unused layers") => 98.0,
                Some("success") => 100.0,
                _ => 0.0,
            }
        }
    }
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    done: bool,
    context: Option<Vec<i32>>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u32>,
    eval_duration: Option<u64>,
}

#[derive(Deserialize)]
struct StreamResponse {
    response: String,
    done: bool,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    embedding: Vec<f32>,
}

/// Request for text generation with options
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub model: String,
    pub prompt: String,
    pub options: Option<GenerationOptions>,
}

/// Generation options for fine-tuning model behavior
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(40),
            max_tokens: Some(2048),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
        }
    }
}

/// Comprehensive generation response with performance metrics
#[derive(Debug, Clone)]
pub struct DetailedGenerationResponse {
    pub text: String,
    pub tokens_generated: u32,
    pub generation_time_ms: u32,
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    pub tokens_per_second: f32,
}
