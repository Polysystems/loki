//! DeepSeek Provider Implementation
//!
//! Provides integration with DeepSeek's advanced reasoning and coding models

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

/// DeepSeek provider for advanced reasoning and code models
pub struct DeepSeekProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl DeepSeekProvider {
    /// Create a new DeepSeek provider instance
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(600)) // Longer timeout for reasoning models
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key,
            client,
            base_url: "https://api.deepseek.com/v1".to_string(),
        }
    }
}

#[async_trait]
impl ModelProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "deepseek-r1-0528".to_string(),
                name: "DeepSeek R1 (May 2028)".to_string(),
                description: "Latest reasoning model with advanced problem-solving capabilities".to_string(),
                context_length: 128000,
                capabilities: vec![
                    "reasoning".to_string(),
                    "math".to_string(),
                    "code".to_string(),
                    "analysis".to_string(),
                    "chain-of-thought".to_string(),
                ],
            },
            ModelInfo {
                id: "deepseek-coder-v3".to_string(),
                name: "DeepSeek Coder V3".to_string(),
                description: "Specialized model for code generation and analysis".to_string(),
                context_length: 64000,
                capabilities: vec![
                    "code".to_string(),
                    "debugging".to_string(),
                    "refactoring".to_string(),
                    "documentation".to_string(),
                    "multi-language".to_string(),
                ],
            },
            ModelInfo {
                id: "deepseek-chat".to_string(),
                name: "DeepSeek Chat".to_string(),
                description: "General-purpose conversational model".to_string(),
                context_length: 32000,
                capabilities: vec![
                    "chat".to_string(),
                    "general-knowledge".to_string(),
                    "multilingual".to_string(),
                ],
            },
            ModelInfo {
                id: "deepseek-math".to_string(),
                name: "DeepSeek Math".to_string(),
                description: "Specialized model for mathematical reasoning and proofs".to_string(),
                context_length: 32000,
                capabilities: vec![
                    "math".to_string(),
                    "proofs".to_string(),
                    "symbolic-reasoning".to_string(),
                    "calculus".to_string(),
                ],
            },
        ])
    }

    async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "deepseek-r1-0528".to_string();
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

        // Add reasoning-specific parameters for R1 model
        if request.model.contains("r1") {
            body["reasoning_mode"] = json!(true);
            body["max_thinking_tokens"] = json!(32000);
        }

        debug!("Sending request to DeepSeek API: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to DeepSeek API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("DeepSeek API error ({}): {}", status, error_text));
        }

        let response_json: serde_json::Value = response.json().await?;

        debug!("Received response from DeepSeek API: {}", serde_json::to_string_pretty(&response_json)?);

        // Parse the response
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // For R1 models, check if there's reasoning content
        let reasoning_content = if request.model.contains("r1") {
            response_json["choices"][0]["message"]["reasoning_content"]
                .as_str()
                .map(|s| format!("\n\n<reasoning>\n{}\n</reasoning>", s))
                .unwrap_or_default()
        } else {
            String::new()
        };

        let final_content = format!("{}{}", content, reasoning_content);

        let usage = Usage {
            prompt_tokens: response_json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
            completion_tokens: response_json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize,
            total_tokens: response_json["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
        };

        Ok(CompletionResponse {
            id: response_json["id"].as_str().unwrap_or("").to_string(),
            model: request.model,
            content: final_content,
            usage,
        })
    }

    async fn stream_complete(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Box<dyn Stream<Item = Result<CompletionChunk>> + Send + Unpin>> {
        // Set default model if not specified
        if request.model.is_empty() {
            request.model = "deepseek-r1-0528".to_string();
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

        // Add reasoning-specific parameters for R1 model
        if request.model.contains("r1") {
            body["reasoning_mode"] = json!(true);
            body["max_thinking_tokens"] = json!(32000);
            body["stream_reasoning"] = json!(true);
        }

        let response = self
            .client
            .post(&url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send streaming request to DeepSeek API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("DeepSeek API streaming error ({}): {}", status, error_text));
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
                            
                            // Check for reasoning content in stream
                            let reasoning_delta = json["choices"][0]["delta"]["reasoning_content"]
                                .as_str()
                                .map(|s| format!("<reasoning>{}</reasoning>", s))
                                .unwrap_or_default();
                            
                            let full_delta = format!("{}{}", delta, reasoning_delta);
                            
                            return Ok(CompletionChunk {
                                id: json["id"].as_str().unwrap_or("").to_string(),
                                delta: full_delta,
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
        let url = format!("{}/embeddings", self.base_url);

        let body = json!({
            "model": "deepseek-embed",
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
            .context("Failed to send embeddings request to DeepSeek API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("DeepSeek embeddings API error ({}): {}", status, error_text));
        }

        let response_json: serde_json::Value = response.json().await?;
        
        let embeddings = response_json["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid embeddings response"))?
            .iter()
            .map(|item| -> Result<Vec<f32>> {
                let embedding = item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<f32>>();
                Ok(embedding)
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;

        Ok(embeddings)
    }
}

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self::new(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_provider_creation() {
        let provider = DeepSeekProvider::new("test_api_key".to_string());
        assert_eq!(provider.name(), "deepseek");
        assert!(provider.is_available());
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = DeepSeekProvider::new("test_api_key".to_string());
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 4);
        assert!(models.iter().any(|m| m.id == "deepseek-r1-0528"));
        assert!(models.iter().any(|m| m.id == "deepseek-coder-v3"));
        assert!(models.iter().any(|m| m.id == "deepseek-chat"));
        assert!(models.iter().any(|m| m.id == "deepseek-math"));
    }
}