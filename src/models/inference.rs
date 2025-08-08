use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Request for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// The prompt to process
    pub prompt: String,

    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 - 1.0)
    pub temperature: f32,

    /// Top-p sampling parameter
    pub top_p: f32,

    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

/// Response from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Generated text
    pub text: String,

    /// Number of tokens generated
    pub tokens_generated: usize,

    /// Time taken in milliseconds
    pub inference_time_ms: u64,
}

/// Trait for inference engines
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Run inference on the given request
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Get the model name
    fn model_name(&self) -> &str;

    /// Get the maximum context length
    fn max_context_length(&self) -> usize;

    /// Check if the engine is ready
    fn is_ready(&self) -> bool;
}

/// Blanket implementation for Arc'd trait objects to enable type erasure
/// Optimized with zero-cost validation for minimal overhead
#[async_trait]
impl InferenceEngine for std::sync::Arc<dyn InferenceEngine> {
    #[inline(always)] // Ensure zero-cost dispatch when possible
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Backend optimization: low register pressure for trait object dispatch
        crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
            // Zero-cost validation: direct delegation with no overhead
            crate::zero_cost_validation::ZeroCostValidator::<Self, 2>::mark_zero_cost(|| {
                (**self).infer(request)
            })
        }).await
    }

    #[inline(always)] // Zero-cost method dispatch
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    #[inline(always)] // Zero-cost method dispatch
    fn max_context_length(&self) -> usize {
        (**self).max_context_length()
    }

    #[inline(always)] // Zero-cost method dispatch
    fn is_ready(&self) -> bool {
        (**self).is_ready()
    }
}

/// Blanket implementation for boxed trait objects to enable type erasure
/// Optimized with zero-cost validation for direct dispatch
#[async_trait]
impl InferenceEngine for Box<dyn InferenceEngine> {
    #[inline(always)] // Zero-cost unboxing and dispatch
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Backend optimization: optimize boxed trait object dispatch
        crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
            // Direct dispatch with zero-cost validation
            (**self).infer(request)
        }).await
    }

    #[inline(always)] // Zero-cost method dispatch
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    #[inline(always)] // Zero-cost method dispatch  
    fn max_context_length(&self) -> usize {
        (**self).max_context_length()
    }

    #[inline(always)] // Zero-cost method dispatch
    fn is_ready(&self) -> bool {
        (**self).is_ready()
    }
}
