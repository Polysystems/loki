//! Mistral/Codestral Provider Implementation

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::{Client, header};
use serde::Deserialize;
use serde_json::json;
use tokio_stream::{Stream, StreamExt};
use tracing::debug;

use super::{
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    MessageRole,
    ModelInfo,
    ModelProvider,
    Usage,
};

pub struct MistralProvider {
    api_key: String,
    client: Client,
    base_url: String,
    is_codestral: bool,
}

impl MistralProvider {
    pub fn new(api_key: String, is_codestral: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        let base_url = if is_codestral {
            "https://codestral.mistral.ai/v1".to_string()
        } else {
            "https://api.mistral.ai/v1".to_string()
        };

        Self { api_key, client, base_url, is_codestral }
    }
}

#[async_trait]
impl ModelProvider for MistralProvider {
    fn name(&self) -> &str {
        if self.is_codestral { "codestral" } else { "mistral" }
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        if self.is_codestral {
            Ok(vec![
                ModelInfo {
                    id: "codestral-latest".to_string(),
                    name: "Codestral Latest".to_string(),
                    description: "Most advanced code generation and completion model".to_string(),
                    context_length: 256000,
                    capabilities: vec![
                        "code".to_string(), 
                        "completion".to_string(),
                        "refactoring".to_string(),
                        "debugging".to_string(),
                        "multi-language".to_string(),
                    ],
                },
                ModelInfo {
                    id: "codestral-2025.01".to_string(),
                    name: "Codestral January 2025".to_string(),
                    description: "Latest stable version with enhanced code understanding".to_string(),
                    context_length: 256000,
                    capabilities: vec![
                        "code".to_string(), 
                        "completion".to_string(),
                        "refactoring".to_string(),
                        "testing".to_string(),
                    ],
                },
                ModelInfo {
                    id: "codestral-mamba".to_string(),
                    name: "Codestral Mamba".to_string(),
                    description: "Fast code model with Mamba architecture".to_string(),
                    context_length: 128000,
                    capabilities: vec![
                        "code".to_string(), 
                        "completion".to_string(),
                        "fast-inference".to_string(),
                    ],
                },
            ])
        } else {
            Ok(vec![
                ModelInfo {
                    id: "mistral-large-2025".to_string(),
                    name: "Mistral Large 2025".to_string(),
                    description: "Most capable Mistral model with enhanced reasoning".to_string(),
                    context_length: 256000,
                    capabilities: vec![
                        "chat".to_string(),
                        "code".to_string(),
                        "analysis".to_string(),
                        "reasoning".to_string(),
                        "function-calling".to_string(),
                    ],
                },
                ModelInfo {
                    id: "mistral-medium-2025".to_string(),
                    name: "Mistral Medium 2025".to_string(),
                    description: "Balanced performance model with excellent efficiency".to_string(),
                    context_length: 128000,
                    capabilities: vec![
                        "chat".to_string(), 
                        "code".to_string(),
                        "analysis".to_string(),
                    ],
                },
                ModelInfo {
                    id: "mistral-small-2025".to_string(),
                    name: "Mistral Small 2025".to_string(),
                    description: "Ultra-fast and efficient model".to_string(),
                    context_length: 65536,
                    capabilities: vec!["chat".to_string(), "code".to_string(), "fast-inference".to_string()],
                },
                ModelInfo {
                    id: "mistral-nemo".to_string(),
                    name: "Mistral Nemo".to_string(),
                    description: "12B parameter efficient model".to_string(),
                    context_length: 128000,
                    capabilities: vec!["chat".to_string(), "code".to_string()],
                },
                ModelInfo {
                    id: "mixtral-8x22b".to_string(),
                    name: "Mixtral 8x22B".to_string(),
                    description: "Large mixture-of-experts model".to_string(),
                    context_length: 65536,
                    capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
                },
            ])
        }
    }

    async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = if self.is_codestral {
                "codestral-latest".to_string()
            } else {
                "mistral-medium-2025".to_string()
            };
        }

        let url = format!("{}/chat/completions", self.base_url);

        let mut body = json!({
            "model": request.model,
            "messages": request.messages.iter().map(|m| json!({
                "role": match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                },
                "content": m.content,
            })).collect::<Vec<_>>(),
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(stop) = request.stop {
            body["stop"] = json!(stop);
        }

        debug!("Mistral request: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Mistral")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Mistral API error: {}", error_text);
        }

        let api_response: MistralResponse =
            response.json().await.context("Failed to parse Mistral response")?;

        Ok(CompletionResponse {
            id: api_response.id,
            model: api_response.model,
            content: api_response
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default(),
            usage: Usage {
                prompt_tokens: api_response.usage.prompt_tokens,
                completion_tokens: api_response.usage.completion_tokens,
                total_tokens: api_response.usage.total_tokens,
            },
        })
    }

    async fn stream_complete(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Box<dyn Stream<Item = Result<CompletionChunk>> + Send + Unpin>> {
        request.stream = true;

        // Set default model if not specified
        if request.model.is_empty() {
            request.model = if self.is_codestral {
                "codestral-latest".to_string()
            } else {
                "mistral-medium-2025".to_string()
            };
        }

        let url = format!("{}/chat/completions", self.base_url);

        let mut body = json!({
            "model": request.model,
            "messages": request.messages.iter().map(|m| json!({
                "role": match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                },
                "content": m.content,
            })).collect::<Vec<_>>(),
            "stream": true,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(stop) = request.stop {
            body["stop"] = json!(stop);
        }

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Mistral")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Mistral API error: {}", error_text);
        }

        let stream = response.bytes_stream().map(|chunk| {
            chunk.map_err(|e| anyhow::anyhow!("Stream error: {}", e)).and_then(|bytes| {
                let text = String::from_utf8_lossy(&bytes);
                // Parse SSE format
                if text.starts_with("data: ") {
                    let json_str = text.trim_start_matches("data: ").trim();
                    if json_str == "[DONE]" {
                        return Ok(CompletionChunk {
                            id: String::new(),
                            delta: String::new(),
                            finish_reason: Some("stop".to_string()),
                        });
                    }

                    if let Ok(chunk) = serde_json::from_str::<MistralStreamChunk>(json_str) {
                        return Ok(CompletionChunk {
                            id: chunk.id,
                            delta: chunk
                                .choices
                                .first()
                                .and_then(|c| c.delta.content.clone())
                                .unwrap_or_default(),
                            finish_reason: chunk
                                .choices
                                .first()
                                .and_then(|c| c.finish_reason.clone()),
                        });
                    }
                }

                Ok(CompletionChunk { id: String::new(), delta: String::new(), finish_reason: None })
            })
        });

        Ok(Box::new(stream))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if self.is_codestral {
            anyhow::bail!("Codestral does not support embeddings. Use Mistral API instead.");
        }

        let url = format!("{}/embeddings", self.base_url);

        let body = json!({
            "model": "mistral-embed",
            "input": texts,
        });

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send embedding request to Mistral")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Mistral embeddings API error: {}", error_text);
        }

        let api_response: EmbeddingResponse =
            response.json().await.context("Failed to parse Mistral embedding response")?;

        Ok(api_response.data.into_iter().map(|d| d.embedding).collect())
    }
}

// Mistral API response structures
#[derive(Deserialize)]
struct MistralResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: MistralUsage,
}

#[derive(Deserialize)]
struct Choice {
    message: MistralMessage,
}

#[derive(Deserialize)]
struct MistralMessage {
    content: String,
}

#[derive(Deserialize)]
struct MistralUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
struct MistralStreamChunk {
    id: String,
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}
