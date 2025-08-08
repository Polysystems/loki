//! Grok (xAI) Provider Implementation
//!
//! Provides integration with xAI's Grok models including Grok 4 and variants

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::{Stream, StreamExt};
use tracing::{debug, info};

use super::{
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    MessageRole,
    ModelInfo,
    ModelProvider,
    Usage,
};

/// Grok provider for xAI's language models
pub struct GrokProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl GrokProvider {
    /// Create a new Grok provider instance
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key,
            client,
            base_url: "https://api.x.ai/v1".to_string(),
        }
    }
}

#[async_trait]
impl ModelProvider for GrokProvider {
    fn name(&self) -> &str {
        "grok"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "grok-4".to_string(),
                name: "Grok 4".to_string(),
                description: "Most capable Grok model with advanced reasoning and analysis".to_string(),
                context_length: 131072,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "analysis".to_string(),
                    "reasoning".to_string(),
                    "real-time".to_string(),
                ],
            },
            ModelInfo {
                id: "grok-4-vision".to_string(),
                name: "Grok 4 Vision".to_string(),
                description: "Multimodal Grok model with image understanding".to_string(),
                context_length: 131072,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "vision".to_string(),
                    "analysis".to_string(),
                    "multimodal".to_string(),
                ],
            },
            ModelInfo {
                id: "grok-4-mini".to_string(),
                name: "Grok 4 Mini".to_string(),
                description: "Fast and efficient Grok model for quick responses".to_string(),
                context_length: 65536,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "fast-inference".to_string(),
                ],
            },
        ])
    }

    async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "grok-4".to_string();
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

        // Add optional parameters
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

        debug!("Sending request to Grok API: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Grok API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("Grok API error ({}): {}", status, error_text));
        }

        let response_json: serde_json::Value = response.json().await?;

        debug!("Received response from Grok API: {}", serde_json::to_string_pretty(&response_json)?);

        // Parse the response
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = Usage {
            prompt_tokens: response_json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
            completion_tokens: response_json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize,
            total_tokens: response_json["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
        };

        Ok(CompletionResponse {
            id: response_json["id"].as_str().unwrap_or("").to_string(),
            model: request.model,
            content,
            usage,
        })
    }

    async fn stream_complete(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Box<dyn Stream<Item = Result<CompletionChunk>> + Send + Unpin>> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "grok-4".to_string();
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

        // Add optional parameters
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
            .context("Failed to send streaming request to Grok API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("Grok API streaming error ({}): {}", status, error_text));
        }

        // Convert response to stream
        let stream = response.bytes_stream();
        let stream = stream.map(move |chunk| {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    
                    // Parse SSE format
                    if text.starts_with("data: ") {
                        let json_str = &text[6..];
                        if json_str.trim() == "[DONE]" {
                            return Ok(CompletionChunk {
                                id: String::new(),
                                delta: String::new(),
                                finish_reason: Some("stop".to_string()),
                            });
                        }
                        
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                            let delta = json["choices"][0]["delta"]["content"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            
                            return Ok(CompletionChunk {
                                id: json["id"].as_str().unwrap_or("").to_string(),
                                delta,
                                finish_reason: json["choices"][0]["finish_reason"]
                                    .as_str()
                                    .map(|s| s.to_string()),
                            });
                        }
                    }
                    
                    Ok(CompletionChunk {
                        id: String::new(),
                        delta: String::new(),
                        finish_reason: None,
                    })
                }
                Err(e) => Err(anyhow::anyhow!("Stream error: {}", e)),
            }
        });

        Ok(Box::new(stream) as Box<dyn Stream<Item = Result<CompletionChunk>> + Send + Unpin>)
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Grok doesn't currently offer a public embeddings API
        // This is a placeholder that returns empty embeddings
        info!("Grok embeddings not yet available, returning placeholder");
        Ok(texts.iter().map(|_| vec![0.0; 1536]).collect())
    }
}

impl Default for GrokProvider {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grok_provider_creation() {
        let provider = GrokProvider::new("test_api_key".to_string());
        assert_eq!(provider.name(), "grok");
        assert!(provider.is_available());
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = GrokProvider::new("test_api_key".to_string());
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 3);
        assert!(models.iter().any(|m| m.id == "grok-4"));
        assert!(models.iter().any(|m| m.id == "grok-4-vision"));
        assert!(models.iter().any(|m| m.id == "grok-4-mini"));
    }
}