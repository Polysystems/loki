//! Anthropic (Claude) Provider Implementation

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

pub struct AnthropicProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client");

        Self { api_key, client, base_url: "https://api.anthropic.com/v1".to_string() }
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "claude-4.1-opus".to_string(),
                name: "Claude 4.1 Opus".to_string(),
                description: "Latest and most capable Claude model with enhanced reasoning".to_string(),
                context_length: 500000,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "analysis".to_string(),
                    "vision".to_string(),
                    "reasoning".to_string(),
                    "multimodal".to_string(),
                ],
            },
            ModelInfo {
                id: "claude-4-opus".to_string(),
                name: "Claude 4 Opus".to_string(),
                description: "Most powerful Claude 4 model for complex tasks".to_string(),
                context_length: 400000,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "analysis".to_string(),
                    "vision".to_string(),
                    "reasoning".to_string(),
                ],
            },
            ModelInfo {
                id: "claude-4-sonnet".to_string(),
                name: "Claude 4 Sonnet".to_string(),
                description: "Balanced Claude 4 model for general use".to_string(),
                context_length: 400000,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "analysis".to_string(),
                    "fast-inference".to_string(),
                ],
            },
            ModelInfo {
                id: "claude-4-haiku".to_string(),
                name: "Claude 4 Haiku".to_string(),
                description: "Fast and efficient Claude 4 model".to_string(),
                context_length: 300000,
                capabilities: vec!["chat".to_string(), "code".to_string(), "efficient".to_string()],
            },
            ModelInfo {
                id: "claude-3-5-sonnet-20241022".to_string(),
                name: "Claude 3.5 Sonnet".to_string(),
                description: "Previous generation high-performance model".to_string(),
                context_length: 200000,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "analysis".to_string(),
                    "vision".to_string(),
                ],
            },
            ModelInfo {
                id: "claude-3-5-haiku-20241022".to_string(),
                name: "Claude 3.5 Haiku".to_string(),
                description: "Previous generation fast model".to_string(),
                context_length: 200000,
                capabilities: vec!["chat".to_string(), "code".to_string()],
            },
        ])
    }

    async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "claude-4-sonnet".to_string();
        }

        let url = format!("{}/messages", self.base_url);

        // Convert messages to Anthropic format
        let mut system_prompt = None;
        let messages: Vec<_> = request
            .messages
            .iter()
            .filter_map(|m| match m.role {
                MessageRole::System => {
                    system_prompt = Some(m.content.clone());
                    None
                }
                _ => Some(json!({
                    "role": match m.role {
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::System => "system", // Should not reach here as System is handled above
                    },
                    "content": m.content,
                })),
            })
            .collect();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
        });

        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(stop) = &request.stop {
            if !stop.is_empty() {
                body["stop_sequences"] = json!(stop);
            }
        }

        if let Some(system) = system_prompt {
            body["system"] = json!(system);
        }

        debug!("Anthropic request: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "prompt-caching-2024-07-31")
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Anthropic")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Anthropic API error: {}", error_text);
        }

        let api_response: AnthropicResponse =
            response.json().await.context("Failed to parse Anthropic response")?;

        Ok(CompletionResponse {
            id: api_response.id,
            model: api_response.model,
            content: api_response.content.first().map(|c| c.text.clone()).unwrap_or_default(),
            usage: Usage {
                prompt_tokens: api_response.usage.input_tokens,
                completion_tokens: api_response.usage.output_tokens,
                total_tokens: api_response.usage.input_tokens + api_response.usage.output_tokens,
            },
        })
    }

    async fn stream_complete(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Box<dyn Stream<Item = Result<CompletionChunk>> + Send + Unpin>> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "claude-3-5-sonnet-20241022".to_string();
        }

        let url = format!("{}/messages", self.base_url);

        // Convert messages to Anthropic format
        let mut system_prompt = None;
        let messages: Vec<_> = request
            .messages
            .iter()
            .filter_map(|m| match m.role {
                MessageRole::System => {
                    system_prompt = Some(m.content.clone());
                    None
                }
                _ => Some(json!({
                    "role": match m.role {
                        MessageRole::User => "user",
                        MessageRole::Assistant => "assistant",
                        MessageRole::System => "system", // Should not reach here as System is handled above
                    },
                    "content": m.content,
                })),
            })
            .collect();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "stream": true,
        });

        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(stop) = &request.stop {
            if !stop.is_empty() {
                body["stop_sequences"] = json!(stop);
            }
        }

        if let Some(system) = system_prompt {
            body["system"] = json!(system);
        }

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "prompt-caching-2024-07-31")
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Anthropic")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Anthropic API error: {}", error_text);
        }

        let stream = response.bytes_stream().map(|chunk| {
            chunk.map_err(|e| anyhow::anyhow!("Stream error: {}", e)).and_then(|bytes| {
                let text = String::from_utf8_lossy(&bytes);

                // Parse SSE format
                for line in text.lines() {
                    if line.starts_with("data: ") {
                        let json_str = line.trim_start_matches("data: ").trim();

                        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(json_str) {
                            return match event.event_type.as_str() {
                                "content_block_delta" => Ok(CompletionChunk {
                                    id: String::new(),
                                    delta: event
                                        .delta
                                        .as_ref()
                                        .and_then(|d| d.text.clone())
                                        .unwrap_or_default(),
                                    finish_reason: None,
                                }),
                                "message_stop" => Ok(CompletionChunk {
                                    id: String::new(),
                                    delta: String::new(),
                                    finish_reason: Some("stop".to_string()),
                                }),
                                _ => Ok(CompletionChunk {
                                    id: String::new(),
                                    delta: String::new(),
                                    finish_reason: None,
                                }),
                            };
                        }
                    }
                }

                Ok(CompletionChunk { id: String::new(), delta: String::new(), finish_reason: None })
            })
        });

        Ok(Box::new(stream))
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Anthropic doesn't have a dedicated embeddings API
        // We could use their messages API to generate embeddings via prompting
        // For now, return an error
        anyhow::bail!(
            "Anthropic does not provide a native embeddings API. Consider using OpenAI or another \
             provider for embeddings."
        )
    }
}

// Anthropic API response structures
#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<TextDelta>,
}

#[derive(Deserialize)]
struct TextDelta {
    text: Option<String>,
}
