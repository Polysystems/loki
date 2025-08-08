//! Model Providers Module
//!
//! This module provides a unified interface for different AI model providers
//! including OpenAI, Anthropic, Mistral, Codestral, Gemini, Grok, and more.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::ApiKeysConfig;
use crate::zero_cost_validation::{ZeroCostValidator, validation_levels, generic_specialization::SpecializationValidator};

pub mod anthropic;
pub mod deepseek;
pub mod gemini;
pub mod grok;
pub mod mistral;
pub mod openai;
pub mod stubs;

pub use anthropic::AnthropicProvider;
pub use deepseek::DeepSeekProvider;
pub use gemini::GeminiProvider;
pub use grok::GrokProvider;
pub use mistral::MistralProvider;
pub use openai::OpenAIProvider;
pub use stubs::{
    HuggingFaceProvider,
    OllamaProvider,
    ReplicateProvider,
};

/// Model provider trait for unified interface
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;

    /// Check if provider is available
    fn is_available(&self) -> bool;

    /// Check if provider supports streaming
    fn supports_streaming(&self) -> bool {
        true // Default to true, providers can override
    }

    /// List available models
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Generate completion
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// Stream completion
    async fn stream_complete(
        &self,
        request: CompletionRequest,
    ) -> Result<Box<dyn tokio_stream::Stream<Item = Result<CompletionChunk>> + Send + Unpin>>;

    /// Create embeddings
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Get API key for the provider (optional - may not be applicable for all
    /// providers)
    fn get_api_key(&self) -> Option<String> {
        None // Default implementation returns None
    }

    /// Execute streaming request (alternative interface)
    async fn execute_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<Box<dyn tokio_stream::Stream<Item = Result<CompletionChunk>> + Send + Unpin>> {
        // Default implementation delegates to stream_complete
        self.stream_complete(request).await
    }
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub context_length: usize,
    pub capabilities: Vec<String>,
}

/// Completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub stream: bool,
}

/// Message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub content: String,
    pub usage: Usage,
}

/// Streaming completion chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub id: String,
    pub delta: String,
    pub finish_reason: Option<String>,
}

/// Token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Model provider factory with generic specialization support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFactory;

/// Specialized provider factory for compile-time optimization
/// This enables zero-cost abstractions for known provider types
pub struct SpecializedProviderFactory;

impl SpecializedProviderFactory {
    /// Create OpenAI provider with compile-time optimization
    #[inline(always)]
    pub fn create_openai_provider(api_key: String) -> Arc<OpenAIProvider> {
        // Zero-cost validation: ensure monomorphization for concrete types
        SpecializationValidator::<OpenAIProvider>::validate_monomorphization(|| {
            Arc::new(OpenAIProvider::new(api_key))
        })
    }
    
    /// Create Anthropic provider with compile-time optimization  
    #[inline(always)]
    pub fn create_anthropic_provider(api_key: String) -> Arc<AnthropicProvider> {
        // Zero-cost validation: ensure monomorphization for concrete types
        SpecializationValidator::<AnthropicProvider>::validate_monomorphization(|| {
            Arc::new(AnthropicProvider::new(api_key))
        })
    }
    
    /// Create Mistral provider with compile-time optimization
    #[inline(always)]
    pub fn create_mistral_provider(api_key: String, is_codestral: bool) -> Arc<MistralProvider> {
        Arc::new(MistralProvider::new(api_key, is_codestral))
    }
    
    /// Create Gemini provider with compile-time optimization
    #[inline(always)] 
    pub fn create_gemini_provider(api_key: String) -> Arc<GeminiProvider> {
        Arc::new(GeminiProvider::new(api_key))
    }
    
    /// Generic factory method with compile-time dispatch for known types
    /// Zero-cost validation ensures proper specialization
    #[inline(always)]
    pub fn create_provider<T>() -> Option<Arc<T>>
    where
        T: ModelProvider + Default + 'static,
    {
        // Validate that generic specialization is occurring
        SpecializationValidator::<T>::assert_specialized();
        ZeroCostValidator::<T, {validation_levels::EXPERT}>::mark_zero_cost(|| {
            Some(Arc::new(T::default()))
        })
    }
    
    /// Fast batch creation for specific provider combinations
    /// Zero-cost validation ensures optimal struct layout and initialization
    #[inline(always)]
    pub fn create_core_providers(apiconfig: &ApiKeysConfig) -> CoreProviders {
        // Validate that CoreProviders has optimal memory layout
        crate::zero_cost_validation::memory_layout_validation::MemoryLayoutValidator::<CoreProviders>::validate_cache_layout();
        
        ZeroCostValidator::<CoreProviders, {validation_levels::ADVANCED}>::mark_zero_cost(|| {
            CoreProviders {
                openai: apiconfig.ai_models.openai.as_ref()
                    .map(|key| Self::create_openai_provider(key.clone())),
                anthropic: apiconfig.ai_models.anthropic.as_ref()
                    .map(|key| Self::create_anthropic_provider(key.clone())),
                mistral: apiconfig.ai_models.mistral.as_ref()
                    .map(|key| Self::create_mistral_provider(key.clone(), false)),
                gemini: apiconfig.ai_models.gemini.as_ref()
                    .map(|key| Self::create_gemini_provider(key.clone())),
            }
        })
    }
}

/// Strongly-typed provider collection for zero-cost abstractions
pub struct CoreProviders {
    pub openai: Option<Arc<OpenAIProvider>>,
    pub anthropic: Option<Arc<AnthropicProvider>>,
    pub mistral: Option<Arc<MistralProvider>>,
    pub gemini: Option<Arc<GeminiProvider>>,
}

impl CoreProviders {
    /// Convert to dynamic provider vector (only when needed)
    pub fn to_dynamic_providers(self) -> Vec<Arc<dyn ModelProvider>> {
        let mut providers: Vec<Arc<dyn ModelProvider>> = Vec::new();
        
        if let Some(openai) = self.openai {
            providers.push(openai);
        }
        if let Some(anthropic) = self.anthropic {
            providers.push(anthropic);
        }
        if let Some(mistral) = self.mistral {
            providers.push(mistral);
        }
        if let Some(gemini) = self.gemini {
            providers.push(gemini);
        }
        
        providers
    }
    
    /// Get provider count without dynamic allocation
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.openai.is_some() as usize +
        self.anthropic.is_some() as usize +
        self.mistral.is_some() as usize +
        self.gemini.is_some() as usize
    }
}

impl ProviderFactory {
    /// Create all available providers from API configuration
    pub fn create_providers(apiconfig: &ApiKeysConfig) -> Vec<Arc<dyn ModelProvider>> {
        let mut providers: Vec<Arc<dyn ModelProvider>> = Vec::new();

        // OpenAI
        if let Some(api_key) = &apiconfig.ai_models.openai {
            providers.push(Arc::new(OpenAIProvider::new(api_key.clone())));
        }

        // Anthropic
        if let Some(api_key) = &apiconfig.ai_models.anthropic {
            providers.push(Arc::new(AnthropicProvider::new(api_key.clone())));
        }

        // Mistral
        if let Some(api_key) = &apiconfig.ai_models.mistral {
            providers.push(Arc::new(MistralProvider::new(api_key.clone(), false)));
        }

        // Codestral (Mistral variant)
        if let Some(api_key) = &apiconfig.ai_models.codestral {
            providers.push(Arc::new(MistralProvider::new(api_key.clone(), true)));
        }

        // Gemini
        if let Some(api_key) = &apiconfig.ai_models.gemini {
            providers.push(Arc::new(GeminiProvider::new(api_key.clone())));
        }

        // Grok (now using full implementation)
        if let Some(api_key) = &apiconfig.ai_models.grok {
            providers.push(Arc::new(GrokProvider::new(api_key.clone())));
        }

        // DeepSeek (now using full implementation)
        if let Some(api_key) = &apiconfig.ai_models.deepseek {
            providers.push(Arc::new(DeepSeekProvider::new(api_key.clone())));
        }

        // Replicate
        if let Some(api_token) = &apiconfig.optional_services.replicate {
            providers.push(Arc::new(ReplicateProvider::new(api_token.clone())));
        }

        // HuggingFace
        if let Some(token) = &apiconfig.optional_services.huggingface {
            providers.push(Arc::new(HuggingFaceProvider::new(token.clone())));
        }

        // Always add Ollama as fallback
        providers.push(Arc::new(OllamaProvider::new("http://localhost:11434".to_string())));

        providers
    }

    /// Get provider by name (optimized hot path for provider lookup)
    #[inline(always)] // Critical path for model selection
    pub fn get_provider(
        providers: &[Arc<dyn ModelProvider>],
        name: &str,
    ) -> Option<Arc<dyn ModelProvider>> {
        // Critical hot path for provider resolution
        crate::compiler_backend_optimization::critical_path_optimization::ultra_fast_path(|| {
            // Optimize for common provider names (OpenAI, Anthropic likely to be first)
            
            // Fast path for exact name matches (case sensitive first)
            for provider in providers {
                if std::hint::likely(provider.name() == name) {
                    return Some(provider.clone());
                }
            }
            
            // Slow path for case-insensitive search
            if std::hint::unlikely(true) {
                providers.iter().find(|p| p.name().eq_ignore_ascii_case(name)).cloned()
            } else {
                None
            }
        })
    }
}

/// Convert from old InferenceRequest to CompletionRequest
impl From<crate::models::InferenceRequest> for CompletionRequest {
    fn from(req: crate::models::InferenceRequest) -> Self {
        CompletionRequest {
            model: String::new(), // Will be set by provider
            messages: vec![Message { role: MessageRole::User, content: req.prompt }],
            max_tokens: Some(req.max_tokens),
            temperature: Some(req.temperature),
            top_p: Some(req.top_p),
            stop: if req.stop_sequences.is_empty() { None } else { Some(req.stop_sequences) },
            stream: false,
        }
    }
}
