//! Google Gemini Provider Implementation

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use tokio_stream::{Stream, StreamExt};
use tracing::{debug, error, info, warn};

use super::{CompletionChunk, CompletionRequest, CompletionResponse, ModelInfo, ModelProvider};
use crate::models::Usage;

/// Google Gemini API provider
pub struct GeminiProvider {
    api_key: String,
    client: Client,
    base_url: String,
    rate_limiter: Option<tokio::sync::Semaphore>,
    request_timeout: Duration,
}

/// Gemini API request format
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generationconfig: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<Vec<SafetySetting>>,
}

/// Gemini API response format
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<Candidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
}

/// Gemini streaming response chunk
#[derive(Debug, Deserialize)]
struct GeminiStreamChunk {
    candidates: Option<Vec<Candidate>>,
    #[serde(default)]
    #[allow(dead_code)]
    usage_metadata: Option<UsageMetadata>,
}

/// Content part of the request
#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
}

/// Part of the content (text, image, etc.)
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Part {
    Text { text: String },
    InlineData { inline_data: InlineData },
}

/// Inline data for images/files
#[derive(Debug, Serialize)]
struct InlineData {
    mime_type: String,
    data: String, // Base64 encoded
}

/// Generation configuration
#[derive(Debug, Serialize)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

/// Safety setting for content filtering
#[derive(Debug, Serialize)]
struct SafetySetting {
    category: String,
    threshold: String,
}

/// Response candidate
#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<ResponseContent>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    safety_ratings: Vec<SafetyRating>,
    #[serde(default)]
    #[allow(dead_code)]
    citation_metadata: Option<CitationMetadata>,
}

/// Response content
#[derive(Debug, Deserialize)]
struct ResponseContent {
    parts: Vec<ResponsePart>,
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
}

/// Response part
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponsePart {
    Text { text: String },
    // Add other types as needed
}

/// Safety rating
#[derive(Debug, Deserialize)]
struct SafetyRating {
    #[allow(dead_code)]
    category: String,
    #[allow(dead_code)]
    probability: String,
}

/// Citation metadata
#[derive(Debug, Deserialize)]
struct CitationMetadata {
    #[serde(default)]
    #[allow(dead_code)]
    citation_sources: Vec<CitationSource>,
}

/// Citation source
#[derive(Debug, Deserialize)]
struct CitationSource {
    #[serde(default)]
    #[allow(dead_code)]
    start_index: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    end_index: Option<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    uri: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    license: Option<String>,
}

/// Usage metadata
#[derive(Debug, Deserialize)]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: Option<i32>,
    #[serde(default)]
    candidates_token_count: Option<i32>,
    #[serde(default)]
    total_token_count: Option<i32>,
}

/// Embedding request
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    content: EmbeddingContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
}

/// Embedding content
#[derive(Debug, Serialize)]
struct EmbeddingContent {
    parts: Vec<Part>,
}

/// Embedding response
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    embedding: EmbeddingData,
}

/// Embedding data
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    values: Vec<f32>,
}

impl GeminiProvider {
    /// Create a new Gemini provider with configuration
    pub fn new(api_key: String) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .unwrap_or_else(|_| HeaderValue::from_static("")),
        );

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key,
            client,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            rate_limiter: Some(tokio::sync::Semaphore::new(10)), // 10 concurrent requests
            request_timeout: Duration::from_secs(120),
        }
    }

    /// Create provider with custom configuration
    pub fn withconfig(
        api_key: String,
        base_url: Option<String>,
        max_concurrent: Option<usize>,
        timeout: Option<Duration>,
    ) -> Self {
        let mut provider = Self::new(api_key);

        if let Some(url) = base_url {
            provider.base_url = url;
        }

        if let Some(max) = max_concurrent {
            provider.rate_limiter = Some(tokio::sync::Semaphore::new(max));
        }

        if let Some(timeout) = timeout {
            provider.request_timeout = timeout;
        }

        provider
    }

    /// Build URL for API endpoints
    fn build_url(&self, endpoint: &str, model: &str) -> String {
        format!("{}/models/{}:{}", self.base_url, model, endpoint)
    }

    /// Convert completion request to Gemini format
    fn convert_request(&self, request: &CompletionRequest) -> Result<GeminiRequest> {
        let mut contents = Vec::new();

        // Convert messages to Gemini format
        for message in &request.messages {
            let role = match message.role {
                super::MessageRole::System => Some("user".to_string()), /* Gemini doesn't have
                                                                          * system role */
                super::MessageRole::User => Some("user".to_string()),
                super::MessageRole::Assistant => Some("model".to_string()),
            };

            contents
                .push(Content { parts: vec![Part::Text { text: message.content.clone() }], role });
        }

        // Handle generation configuration
        let generationconfig = if request.temperature.is_some()
            || request.top_p.is_some()
            || request.max_tokens.is_some()
        {
            Some(GenerationConfig {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: None, // Not exposed in our interface
                max_output_tokens: request.max_tokens.map(|t| t as i32),
                stop_sequences: request.stop.clone(),
            })
        } else {
            None
        };

        // Set up safety settings (permissive for now)
        let safety_settings = Some(vec![
            SafetySetting {
                category: "HARM_CATEGORY_HARASSMENT".to_string(),
                threshold: "BLOCK_ONLY_HIGH".to_string(),
            },
            SafetySetting {
                category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
                threshold: "BLOCK_ONLY_HIGH".to_string(),
            },
            SafetySetting {
                category: "HARM_CATEGORY_SEXUALLY_EXPLICIT".to_string(),
                threshold: "BLOCK_ONLY_HIGH".to_string(),
            },
            SafetySetting {
                category: "HARM_CATEGORY_DANGEROUS_CONTENT".to_string(),
                threshold: "BLOCK_ONLY_HIGH".to_string(),
            },
        ]);

        Ok(GeminiRequest { contents, generationconfig, safety_settings })
    }

    /// Convert Gemini response to our format
    fn convert_response(&self, response: GeminiResponse) -> Result<CompletionResponse> {
        let candidate = response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No candidates in response"))?;

        let content =
            candidate.content.ok_or_else(|| anyhow::anyhow!("No content in candidate"))?;

        let text = content
            .parts
            .into_iter()
            .filter_map(|part| match part {
                ResponsePart::Text { text } => Some(text),
                #[allow(unreachable_patterns)]
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: "gemini".to_string(),
            content: text,
            usage: Usage {
                prompt_tokens: response
                    .usage_metadata
                    .as_ref()
                    .and_then(|u| u.prompt_token_count)
                    .unwrap_or(0) as usize,
                completion_tokens: response
                    .usage_metadata
                    .as_ref()
                    .and_then(|u| u.candidates_token_count)
                    .unwrap_or(0) as usize,
                total_tokens: response
                    .usage_metadata
                    .as_ref()
                    .and_then(|u| u.total_token_count)
                    .unwrap_or(0) as usize,
            },
        })
    }

    /// Convert streaming chunk to our format
    fn convert_chunk(chunk: GeminiStreamChunk) -> Option<CompletionChunk> {
        let candidate = chunk.candidates?.into_iter().next()?;
        let content = candidate.content?;

        let text = content
            .parts
            .into_iter()
            .filter_map(|part| match part {
                ResponsePart::Text { text } => Some(text),
                #[allow(unreachable_patterns)]
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return None;
        }

        Some(CompletionChunk {
            id: uuid::Uuid::new_v4().to_string(),
            delta: text,
            finish_reason: candidate.finish_reason,
        })
    }

    /// Determine the appropriate model name for Gemini API
    fn resolve_model_name<'a>(&self, model: &'a str) -> &'a str {
        match model {
            "gemini-2.5-ultra" => "gemini-2.5-ultra-latest",
            "gemini-2.5-pro" => "gemini-2.5-pro-latest",
            "gemini-2.5-flash" => "gemini-2.5-flash-latest",
            "gemini-2.0-flash-exp" => "gemini-2.0-flash-exp",
            "gemini-1.5-pro" => "gemini-1.5-pro-latest",
            "gemini-1.5-flash" => "gemini-1.5-flash-latest",
            "gemini-pro" => "gemini-pro",
            "gemini-pro-vision" => "gemini-pro-vision",
            _ => model, // Use as-is for custom models
        }
    }

    /// Acquire rate limiting permit
    async fn acquire_permit(&self) -> Result<Option<tokio::sync::SemaphorePermit<'_>>> {
        if let Some(ref semaphore) = self.rate_limiter {
            let permit =
                semaphore.acquire().await.context("Failed to acquire rate limit permit")?;
            Ok(Some(permit))
        } else {
            Ok(None)
        }
    }
}

#[async_trait]
impl ModelProvider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty() && !self.api_key.starts_with("your-api-key")
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        // Static model list - in production, this could query the API
        Ok(vec![
            ModelInfo {
                id: "gemini-2.5-ultra".to_string(),
                name: "Gemini 2.5 Ultra".to_string(),
                description: "Most advanced Gemini model with 2M token context and superior capabilities".to_string(),
                context_length: 2097152,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "vision".to_string(),
                    "function_calling".to_string(),
                    "reasoning".to_string(),
                    "multimodal".to_string(),
                    "video".to_string(),
                ],
            },
            ModelInfo {
                id: "gemini-2.5-pro".to_string(),
                name: "Gemini 2.5 Pro".to_string(),
                description: "Flagship Gemini model with 2M token context".to_string(),
                context_length: 2097152,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "vision".to_string(),
                    "function_calling".to_string(),
                    "reasoning".to_string(),
                    "multimodal".to_string(),
                ],
            },
            ModelInfo {
                id: "gemini-2.5-flash".to_string(),
                name: "Gemini 2.5 Flash".to_string(),
                description: "Ultra-fast Gemini model with excellent performance".to_string(),
                context_length: 1048576,
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "vision".to_string(),
                    "fast-inference".to_string(),
                ],
            },
            ModelInfo {
                id: "gemini-2.0-flash-exp".to_string(),
                name: "Gemini 2.0 Flash Experimental".to_string(),
                description: "Previous generation experimental model".to_string(),
                context_length: 1048576,
                capabilities: vec!["chat".to_string(), "code".to_string(), "vision".to_string()],
            },
            ModelInfo {
                id: "gemini-1.5-pro".to_string(),
                name: "Gemini 1.5 Pro".to_string(),
                description: "Legacy model for compatibility".to_string(),
                context_length: 1048576,
                capabilities: vec!["chat".to_string(), "code".to_string(), "vision".to_string()],
            },
        ])
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let _permit = self.acquire_permit().await?;

        debug!("Making Gemini completion request for model: {}", request.model);

        let model_name = self.resolve_model_name(&request.model);
        let gemini_request =
            self.convert_request(&request).context("Failed to convert request to Gemini format")?;

        let url = self.build_url("generateContent", model_name);

        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.api_key)])
            .json(&gemini_request)
            .timeout(self.request_timeout)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Gemini API error {}: {}", status, error_text));
        }

        let gemini_response: GeminiResponse =
            response.json().await.context("Failed to parse Gemini response")?;

        self.convert_response(gemini_response).context("Failed to convert Gemini response")
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
    ) -> Result<Box<dyn Stream<Item = Result<CompletionChunk>> + Send + Unpin>> {
        let _permit = self.acquire_permit().await?;

        debug!("Making Gemini streaming completion request for model: {}", request.model);

        let model_name = self.resolve_model_name(&request.model);
        let gemini_request =
            self.convert_request(&request).context("Failed to convert request to Gemini format")?;

        let url = self.build_url("streamGenerateContent", model_name);

        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.api_key)])
            .json(&gemini_request)
            .timeout(self.request_timeout)
            .send()
            .await
            .context("Failed to send streaming request to Gemini API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Gemini streaming API error {}: {}", status, error_text));
        }

        let byte_stream = response.bytes_stream();
        let line_stream = tokio_util::codec::FramedRead::new(
            tokio_util::io::StreamReader::new(byte_stream.map(|result| {
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            })),
            tokio_util::codec::LinesCodec::new(),
        );

        let chunk_stream = line_stream.filter_map(|line_result| match line_result {
            Ok(line) => {
                if line.trim().is_empty() || !line.trim().starts_with("{") {
                    return None;
                }

                match serde_json::from_str::<GeminiStreamChunk>(&line) {
                    Ok(chunk) => Self::convert_chunk(chunk).map(Ok),
                    Err(e) => {
                        warn!("Failed to parse Gemini stream chunk: {} - Line: {}", e, line);
                        None
                    }
                }
            }
            Err(e) => {
                error!("Stream read error: {}", e);
                Some(Err(anyhow::anyhow!("Stream read error: {}", e)))
            }
        });

        Ok(Box::new(Box::pin(chunk_stream)))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let _permit = self.acquire_permit().await?;

        debug!("Making Gemini embeddings request for {} texts", texts.len());

        let mut embeddings = Vec::new();

        // Process texts in batches (Gemini typically supports one at a time)
        for text in texts {
            let embedding_request = EmbeddingRequest {
                model: "models/embedding-001".to_string(),
                content: EmbeddingContent { parts: vec![Part::Text { text }] },
                task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
                title: None,
            };

            let url = format!("{}/models/embedding-001:embedContent", self.base_url);

            let response = self
                .client
                .post(&url)
                .query(&[("key", &self.api_key)])
                .json(&embedding_request)
                .timeout(self.request_timeout)
                .send()
                .await
                .context("Failed to send embedding request to Gemini API")?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                return Err(anyhow::anyhow!(
                    "Gemini embeddings API error {}: {}",
                    status,
                    error_text
                ));
            }

            let embedding_response: EmbeddingResponse =
                response.json().await.context("Failed to parse Gemini embedding response")?;

            embeddings.push(embedding_response.embedding.values);

            // Small delay to avoid rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        info!("Generated {} embeddings using Gemini", embeddings.len());
        Ok(embeddings)
    }
}
