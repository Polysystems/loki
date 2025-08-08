//! OpenAI Provider Implementation

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

pub struct OpenAIProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self { api_key, client, base_url: "https://api.openai.com/v1".to_string() }
    }
}

#[async_trait]
impl ModelProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "gpt-5".to_string(),
                name: "GPT-5".to_string(),
                description: "Most advanced OpenAI model with superior reasoning and creativity".to_string(),
                context_length: 256000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string(), "reasoning".to_string(), "vision".to_string()],
            },
            ModelInfo {
                id: "gpt-5-turbo".to_string(),
                name: "GPT-5 Turbo".to_string(),
                description: "Faster variant of GPT-5 optimized for speed".to_string(),
                context_length: 256000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string(), "fast-inference".to_string()],
            },
            ModelInfo {
                id: "gpt-5-mini".to_string(),
                name: "GPT-5 Mini".to_string(),
                description: "Efficient GPT-5 variant for high-volume tasks".to_string(),
                context_length: 128000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "efficient".to_string()],
            },
            ModelInfo {
                id: "o1-preview".to_string(),
                name: "O1 Preview".to_string(),
                description: "Advanced reasoning model with chain-of-thought capabilities".to_string(),
                context_length: 128000,
                capabilities: vec!["reasoning".to_string(), "math".to_string(), "code".to_string(), "analysis".to_string()],
            },
            ModelInfo {
                id: "o1-mini".to_string(),
                name: "O1 Mini".to_string(),
                description: "Efficient reasoning model for faster problem solving".to_string(),
                context_length: 128000,
                capabilities: vec!["reasoning".to_string(), "code".to_string(), "math".to_string()],
            },
            ModelInfo {
                id: "gpt-4o".to_string(),
                name: "GPT-4 Optimized".to_string(),
                description: "Previous generation optimized model".to_string(),
                context_length: 128000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "analysis".to_string()],
            },
            ModelInfo {
                id: "gpt-4o-mini".to_string(),
                name: "GPT-4 Optimized Mini".to_string(),
                description: "Smaller, faster GPT-4 variant".to_string(),
                context_length: 128000,
                capabilities: vec!["chat".to_string(), "code".to_string()],
            },
        ])
    }

    async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "gpt-5-mini".to_string();
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
        if let Some(stop) = &request.stop {
            if !stop.is_empty() {
                body["stop"] = json!(stop);
            }
        }

        debug!("OpenAI request: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("OpenAI API error: {}", error_text);
        }

        let api_response: OpenAIResponse =
            response.json().await.context("Failed to parse OpenAI response")?;

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
            request.model = "gpt-4o-mini".to_string();
        }

        let url = format!("{}/chat/completions", self.base_url);

        let body = json!({
            "model": request.model,
            "messages": request.messages.iter().map(|m| json!({
                "role": match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                },
                "content": m.content,
            })).collect::<Vec<_>>(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop,
            "stream": true,
        });

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to OpenAI")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("OpenAI API error: {}", error_text);
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

                    let chunk: OpenAIStreamChunk = serde_json::from_str(json_str)?;
                    Ok(CompletionChunk {
                        id: chunk.id,
                        delta: chunk
                            .choices
                            .first()
                            .and_then(|c| c.delta.content.clone())
                            .unwrap_or_default(),
                        finish_reason: chunk.choices.first().and_then(|c| c.finish_reason.clone()),
                    })
                } else {
                    Ok(CompletionChunk {
                        id: String::new(),
                        delta: String::new(),
                        finish_reason: None,
                    })
                }
            })
        });

        Ok(Box::new(stream))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);

        let body = json!({
            "model": "text-embedding-3-small",
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
            .context("Failed to send embedding request to OpenAI")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("OpenAI embeddings API error: {}", error_text);
        }

        let api_response: EmbeddingResponse =
            response.json().await.context("Failed to parse OpenAI embedding response")?;

        Ok(api_response.data.into_iter().map(|d| d.embedding).collect())
    }
}

// OpenAI API response structures
#[derive(Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct Choice {
    message: OpenAIMessage,
}

#[derive(Deserialize)]
struct OpenAIMessage {
    content: String,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Deserialize)]
struct OpenAIStreamChunk {
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
