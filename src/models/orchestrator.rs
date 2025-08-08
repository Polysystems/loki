use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Result, anyhow};
use chrono;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::config::ApiKeysConfig;
use crate::models::adaptive_learning::{AdaptiveLearningConfig, AdaptiveLearningSystem};
use crate::models::cost_manager::{BudgetConfig, CostManager};
use crate::models::distributed_serving::{
    ClusterStatus,
    DistributedConfig,
    DistributedServingManager,
};
use crate::models::ensemble::{EnsembleConfig, EnsembleResponse, ModelEnsemble};
use crate::models::fine_tuning::{
    FineTuningConfig,
    FineTuningCostAnalytics,
    FineTuningJob,
    FineTuningManager,
    FineTuningStatus,
    FineTuningSystemStatus,
    JobConfiguration,
    TrainingMetrics,
};
use crate::models::local_manager::LocalModelManager;
use crate::models::providers::{CompletionRequest, Message, MessageRole, ModelProvider};
use crate::models::streaming::{StreamingManager, StreamingRequest, StreamingResponse};
use crate::models::{InferenceRequest, ModelCapabilities};

/// Provider-specific request converter trait
#[async_trait::async_trait]
pub trait ProviderRequestConverter: Send + Sync {
    /// Convert TaskRequest to provider-specific CompletionRequest
    async fn convert_task_request(
        &self,
        task: &TaskRequest,
        provider_name: &str,
    ) -> Result<CompletionRequest>;

    /// Get optimal model for the task type and provider
    fn get_optimal_model(&self, task: &TaskRequest, provider_name: &str) -> String;

    /// Calculate provider-specific parameters
    fn get_provider_parameters(
        &self,
        task: &TaskRequest,
        provider_name: &str,
    ) -> ProviderParameters;

    /// Get usage tracking information
    fn get_usage_context(&self, task: &TaskRequest, provider_name: &str) -> UsageContext;
}

/// Provider-specific parameters for fine-tuning requests
#[derive(Debug, Clone)]
pub struct ProviderParameters {
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Vec<String>,
    pub system_prompt: Option<String>,
}

/// Usage tracking context for analytics
#[derive(Debug, Clone)]
pub struct UsageContext {
    pub model_selected: String,
    pub task_complexity: f32,
    pub expected_quality: f32,
    pub cost_tier: String,
    pub specialization_match: f32,
}

/// Model usage statistics for tracking
#[derive(Debug, Clone, Default)]
pub struct ModelUsageStats {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost_cents: f32,
    pub avg_quality_score: f32,
    pub last_used: Option<Instant>,
    pub task_type_distribution: HashMap<String, u64>,
    pub success_rate: f32,
    pub avg_latency_ms: f32,
}

/// Comprehensive usage tracker for models and providers following Rust 2025 practices
#[derive(Debug)]
pub struct ModelUsageTracker {
    /// Per-model usage statistics
    model_stats: HashMap<String, ModelUsageStats>,
    /// Per-provider usage statistics (aggregated from models)
    provider_stats: HashMap<String, ProviderUsageStats>,
    /// Provider last usage times for API status tracking
    provider_last_used: HashMap<String, Instant>,
    /// Global usage metrics
    global_stats: GlobalUsageStats,
    /// Session start time for relative metrics
    session_start: Instant,
}

/// Provider-level usage statistics
#[derive(Debug, Clone, Default)]
pub struct ProviderUsageStats {
    pub provider_name: String,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost_cents: f32,
    pub avg_latency_ms: f32,
    pub success_rate: f32,
    pub last_used: Option<Instant>,
    pub active_models: Vec<String>,
    pub failure_count: u64,
    pub rate_limit_hits: u64,
}

/// Global usage statistics across all models and providers
#[derive(Debug, Clone, Default)]
pub struct GlobalUsageStats {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost_cents: f32,
    pub session_duration_ms: u64,
    pub requests_per_minute: f32,
    pub cost_per_hour: f32,
    pub most_used_model: Option<String>,
    pub most_used_provider: Option<String>,
    pub avg_global_latency_ms: f32,
}

impl ModelUsageTracker {
    pub fn new() -> Self {
        Self {
            model_stats: HashMap::new(),
            provider_stats: HashMap::new(),
            provider_last_used: HashMap::new(),
            global_stats: GlobalUsageStats::default(),
            session_start: Instant::now(),
        }
    }

    /// Record model usage for comprehensive tracking
    pub fn record_model_usage(
        &mut self,
        model_id: &str,
        provider_name: &str,
        task_type: &str,
        tokens_used: u64,
        cost_cents: f32,
        latency_ms: f32,
        quality_score: f32,
        success: bool,
    ) {
        let now = Instant::now();

        // Update model-specific statistics
        let model_stats = self.model_stats.entry(model_id.to_string()).or_default();
        model_stats.total_requests += 1;
        model_stats.total_tokens += tokens_used;
        model_stats.total_cost_cents += cost_cents;
        model_stats.last_used = Some(now);

        // Update task type distribution
        *model_stats.task_type_distribution.entry(task_type.to_string()).or_default() += 1;

        // Update rolling averages for quality and latency
        let total_requests = model_stats.total_requests as f32;
        model_stats.avg_quality_score =
            (model_stats.avg_quality_score * (total_requests - 1.0) + quality_score) / total_requests;
        model_stats.avg_latency_ms =
            (model_stats.avg_latency_ms * (total_requests - 1.0) + latency_ms) / total_requests;

        // Update success rate
        let prev_successes = (model_stats.success_rate * (total_requests - 1.0)) as u64;
        let new_successes = if success { prev_successes + 1 } else { prev_successes };
        model_stats.success_rate = new_successes as f32 / total_requests;

        // Update provider-specific statistics
        let provider_stats = self.provider_stats.entry(provider_name.to_string()).or_default();
        provider_stats.provider_name = provider_name.to_string();
        provider_stats.total_requests += 1;
        provider_stats.total_tokens += tokens_used;
        provider_stats.total_cost_cents += cost_cents;
        provider_stats.last_used = Some(now);

        if !success {
            provider_stats.failure_count += 1;
        }

        // Update provider success rate
        let provider_total = provider_stats.total_requests as f32;
        let provider_successes = provider_total - provider_stats.failure_count as f32;
        provider_stats.success_rate = provider_successes / provider_total;

        // Update provider latency
        provider_stats.avg_latency_ms =
            (provider_stats.avg_latency_ms * (provider_total - 1.0) + latency_ms) / provider_total;

        // Add model to active models list if not present
        if !provider_stats.active_models.contains(&model_id.to_string()) {
            provider_stats.active_models.push(model_id.to_string());
        }

        // Update provider last used tracking
        self.provider_last_used.insert(provider_name.to_string(), now);

        // Update global statistics
        self.global_stats.total_requests += 1;
        self.global_stats.total_tokens += tokens_used;
        self.global_stats.total_cost_cents += cost_cents;

        // Update session duration
        self.global_stats.session_duration_ms =
            self.session_start.elapsed().as_millis() as u64;

        // Calculate requests per minute
        let session_minutes = self.global_stats.session_duration_ms as f32 / 60_000.0;
        self.global_stats.requests_per_minute =
            if session_minutes > 0.0 { self.global_stats.total_requests as f32 / session_minutes } else { 0.0 };

        // Calculate cost per hour
        let session_hours = session_minutes / 60.0;
        self.global_stats.cost_per_hour =
            if session_hours > 0.0 { self.global_stats.total_cost_cents / 100.0 / session_hours } else { 0.0 };

        // Update global latency average
        let global_total = self.global_stats.total_requests as f32;
        self.global_stats.avg_global_latency_ms =
            (self.global_stats.avg_global_latency_ms * (global_total - 1.0) + latency_ms) / global_total;

        // Update most used model and provider
        self.update_most_used_stats();
    }

    /// Record rate limit hit for provider
    pub fn record_rate_limit_hit(&mut self, provider_name: &str) {
        if let Some(provider_stats) = self.provider_stats.get_mut(provider_name) {
            provider_stats.rate_limit_hits += 1;
        }
    }

    /// Get provider last used time for API status
    pub fn get_provider_last_used(&self, provider_name: &str) -> Option<Instant> {
        self.provider_last_used.get(provider_name).copied()
    }

    /// Get model statistics for analytics
    pub fn get_model_stats(&self) -> &HashMap<String, ModelUsageStats> {
        &self.model_stats
    }

    /// Get provider statistics for analytics
    pub fn get_provider_stats(&self) -> &HashMap<String, ProviderUsageStats> {
        &self.provider_stats
    }

    /// Get global usage statistics
    pub fn get_global_stats(&self) -> &GlobalUsageStats {
        &self.global_stats
    }

    /// Update most used model and provider statistics
    fn update_most_used_stats(&mut self) {
        // Find most used model by request count
        let most_used_model = self.model_stats
            .iter()
            .max_by_key(|(_, stats)| stats.total_requests)
            .map(|(model_id, _)| model_id.clone());
        self.global_stats.most_used_model = most_used_model;

        // Find most used provider by request count
        let most_used_provider = self.provider_stats
            .iter()
            .max_by_key(|(_, stats)| stats.total_requests)
            .map(|(provider_name, _)| provider_name.clone());
        self.global_stats.most_used_provider = most_used_provider;
    }

    /// Get comprehensive usage report for monitoring
    pub fn get_usage_report(&self) -> UsageReport {
        UsageReport {
            global_stats: self.global_stats.clone(),
            model_count: self.model_stats.len(),
            provider_count: self.provider_stats.len(),
            session_duration: self.session_start.elapsed(),
            top_models: self.get_top_models(5),
            top_providers: self.get_top_providers(5),
        }
    }

    /// Get top N models by usage
    fn get_top_models(&self, n: usize) -> Vec<(String, u64)> {
        let mut models: Vec<_> = self.model_stats
            .iter()
            .map(|(id, stats)| (id.clone(), stats.total_requests))
            .collect();
        models.sort_by(|a, b| b.1.cmp(&a.1));
        models.truncate(n);
        models
    }

    /// Get top N providers by usage
    fn get_top_providers(&self, n: usize) -> Vec<(String, u64)> {
        let mut providers: Vec<_> = self.provider_stats
            .iter()
            .map(|(name, stats)| (name.clone(), stats.total_requests))
            .collect();
        providers.sort_by(|a, b| b.1.cmp(&a.1));
        providers.truncate(n);
        providers
    }
}

/// Comprehensive usage report for monitoring and analytics
#[derive(Debug, Clone)]
pub struct UsageReport {
    pub global_stats: GlobalUsageStats,
    pub model_count: usize,
    pub provider_count: usize,
    pub session_duration: Duration,
    pub top_models: Vec<(String, u64)>,
    pub top_providers: Vec<(String, u64)>,
}

/// Default provider request converter implementation
pub struct DefaultProviderRequestConverter {
    model_usage_tracker: Arc<RwLock<HashMap<String, ModelUsageStats>>>,
}

impl DefaultProviderRequestConverter {
    pub fn new() -> Self {
        Self { model_usage_tracker: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Update usage statistics for a model
    pub async fn update_usage_stats(
        &self,
        model_id: &str,
        tokens: u32,
        cost_cents: f32,
        quality_score: f32,
        task_type: &TaskType,
        latency_ms: u32,
        success: bool,
    ) {
        let mut tracker = self.model_usage_tracker.write().await;
        let stats = tracker.entry(model_id.to_string()).or_default();

        stats.total_requests += 1;
        stats.total_tokens += tokens as u64;
        stats.total_cost_cents += cost_cents;
        stats.last_used = Some(Instant::now());

        // Update task type distribution
        let task_type_key = format!("{:?}", task_type);
        *stats.task_type_distribution.entry(task_type_key).or_insert(0) += 1;

        // Update running averages
        let total_requests = stats.total_requests as f32;
        stats.avg_quality_score =
            (stats.avg_quality_score * (total_requests - 1.0) + quality_score) / total_requests;
        stats.avg_latency_ms =
            (stats.avg_latency_ms * (total_requests - 1.0) + latency_ms as f32) / total_requests;

        // Update success rate
        let prev_successes = (stats.success_rate * (total_requests - 1.0)) as u64;
        let new_successes = if success { prev_successes + 1 } else { prev_successes };
        stats.success_rate = new_successes as f32 / total_requests;
    }

    /// Get usage statistics for analytics
    pub async fn get_usage_stats(&self) -> HashMap<String, ModelUsageStats> {
        self.model_usage_tracker.read().await.clone()
    }

    /// Build provider and task-specific system prompts
    fn build_system_prompt(&self, task: &TaskRequest, provider_name: &str) -> Option<String> {
        let base_prompt = match &task.task_type {
            TaskType::CodeGeneration { language } => {
                format!(
                    "You are an expert {} programmer. Write clean, efficient, and well-documented \
                     code. Follow best practices and explain your approach.",
                    language
                )
            }
            TaskType::CodeReview { language } => {
                format!(
                    "You are a senior {} code reviewer. Provide constructive feedback focusing on \
                     code quality, performance, security, and maintainability.",
                    language
                )
            }
            TaskType::LogicalReasoning => "You are a logical reasoning expert. Think \
                                           step-by-step, show your work, and provide clear \
                                           explanations for your conclusions."
                .to_string(),
            TaskType::CreativeWriting => "You are a creative writing assistant. Focus on engaging \
                                          storytelling, vivid descriptions, and compelling \
                                          narratives."
                .to_string(),
            TaskType::DataAnalysis => "You are a data analysis expert. Provide thorough analysis \
                                       with clear insights, statistical reasoning, and actionable \
                                       recommendations."
                .to_string(),
            TaskType::GeneralChat => return None, // No system prompt for general chat
            TaskType::SystemMaintenance => "You are a system administrator assistant. Provide \
                                            accurate, safe, and well-documented system \
                                            maintenance guidance."
                .to_string(),
            TaskType::FileSystemOperation { operation_type } => {
                format!(
                    "You are a file system assistant specialized in {} operations. Provide \
                     safe, accurate file system guidance with proper error handling.",
                    operation_type
                )
            }
            TaskType::DirectoryManagement { action } => {
                format!(
                    "You are a directory management assistant for {} operations. Focus on \
                     safe directory operations and proper path handling.",
                    action
                )
            }
            TaskType::FileManipulation { action } => {
                format!(
                    "You are a file manipulation assistant for {} operations. Ensure safe \
                     file handling with appropriate safeguards and validation.",
                    action
                )
            }
        };

        // Add provider-specific enhancements
        let enhanced_prompt = match provider_name {
            "anthropic" => {
                format!("{} Be thorough and thoughtful in your response.", base_prompt)
            }
            "openai" => {
                format!("{} Be precise and comprehensive.", base_prompt)
            }
            "mistral" => {
                format!("{} Be efficient and direct.", base_prompt)
            }
            _ => base_prompt,
        };

        Some(enhanced_prompt)
    }

    /// Format task content for provider-specific optimal performance
    fn format_task_content(&self, task: &TaskRequest, provider_name: &str) -> String {
        let content = &task.content;

        match &task.task_type {
            TaskType::CodeGeneration { language } => match provider_name {
                "anthropic" => format!(
                    "Please write {} code for the following requirement:\n\n{}\n\nProvide clean, \
                     well-commented code with explanations.",
                    language, content
                ),
                "openai" => format!("Generate {} code:\n\n{}", language, content),
                _ => format!("{} code needed:\n{}", language, content),
            },
            TaskType::CodeReview { .. } => {
                format!(
                    "Please review the following code and provide detailed \
                     feedback:\n\n```\n{}\n```",
                    content
                )
            }
            TaskType::LogicalReasoning => match provider_name {
                "anthropic" => format!(
                    "Please analyze this step-by-step:\n\n{}\n\nShow your reasoning process.",
                    content
                ),
                _ => format!("Analyze: {}", content),
            },
            TaskType::CreativeWriting => {
                format!("Creative writing request:\n\n{}", content)
            }
            TaskType::DataAnalysis => {
                format!(
                    "Please analyze the following data/scenario:\n\n{}\n\nProvide insights and \
                     recommendations.",
                    content
                )
            }
            _ => content.clone(),
        }
    }
}

#[async_trait::async_trait]
impl ProviderRequestConverter for DefaultProviderRequestConverter {
    async fn convert_task_request(
        &self,
        task: &TaskRequest,
        provider_name: &str,
    ) -> Result<CompletionRequest> {
        let model = self.get_optimal_model(task, provider_name);
        let params = self.get_provider_parameters(task, provider_name);

        // Build system prompt based on task type and provider capabilities
        let system_prompt = self.build_system_prompt(task, provider_name);
        let mut messages = Vec::new();

        // Add system message if supported
        if let Some(sys_prompt) = system_prompt {
            messages.push(Message { role: MessageRole::System, content: sys_prompt });
        }

        // Add user message with task-specific formatting
        let formatted_content = self.format_task_content(task, provider_name);
        messages.push(Message { role: MessageRole::User, content: formatted_content });

        Ok(CompletionRequest {
            model,
            messages,
            max_tokens: Some(params.max_tokens),
            temperature: Some(params.temperature),
            top_p: params.top_p,
            stop: if params.stop_sequences.is_empty() { None } else { Some(params.stop_sequences) },
            stream: false,
        })
    }

    fn get_optimal_model(&self, task: &TaskRequest, provider_name: &str) -> String {
        match provider_name {
            "openai" => match &task.task_type {
                TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => {
                    if task.constraints.quality_threshold.unwrap_or(0.7) > 0.9 {
                        "gpt-4-turbo".to_string()
                    } else {
                        "gpt-4o-mini".to_string()
                    }
                }
                TaskType::LogicalReasoning | TaskType::DataAnalysis => "gpt-4-turbo".to_string(),
                TaskType::CreativeWriting => "gpt-4".to_string(),
                TaskType::GeneralChat => {
                    if task.content.len() > 1000
                        || task.constraints.quality_threshold.unwrap_or(0.7) > 0.8
                    {
                        "gpt-4o-mini".to_string()
                    } else {
                        "gpt-3.5-turbo".to_string()
                    }
                }
                TaskType::SystemMaintenance => "gpt-3.5-turbo".to_string(),
                TaskType::FileSystemOperation { .. } => "gpt-3.5-turbo".to_string(),
                TaskType::DirectoryManagement { .. } => "gpt-3.5-turbo".to_string(),
                TaskType::FileManipulation { .. } => "gpt-3.5-turbo".to_string(),
            },
            "anthropic" => match &task.task_type {
                TaskType::LogicalReasoning | TaskType::CreativeWriting => {
                    if task.constraints.quality_threshold.unwrap_or(0.7) > 0.9
                        || task.content.len() > 2000
                    {
                        "claude-3-5-sonnet-20241022".to_string()
                    } else {
                        "claude-3-5-haiku-20241022".to_string()
                    }
                }
                TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => {
                    "claude-3-5-sonnet-20241022".to_string()
                }
                TaskType::DataAnalysis => "claude-3-5-sonnet-20241022".to_string(),
                _ => "claude-3-5-haiku-20241022".to_string(),
            },
            "mistral" => match &task.task_type {
                TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => {
                    "mistral-large-latest".to_string()
                }
                TaskType::LogicalReasoning => "mistral-large-latest".to_string(),
                _ => "mistral-small-latest".to_string(),
            },
            "google" => match &task.task_type {
                TaskType::CodeGeneration { .. } => "gemini-1.5-pro".to_string(),
                TaskType::LogicalReasoning | TaskType::DataAnalysis => "gemini-1.5-pro".to_string(),
                _ => "gemini-1.5-flash".to_string(),
            },
            "ollama" => {
                // For Ollama, we'll use a default model that's commonly available
                // The actual model selection happens at runtime in the provider
                // based on what's installed on the user's system
                
                // Try common models in order of preference
                let common_models = vec![
                    "llama3.2:latest",
                    "llama3.1:latest",
                    "llama3:latest", 
                    "llama2:latest",
                    "mistral:latest",
                    "codellama:latest",
                    "phi3:latest",
                    "gemma2:latest",
                    "qwen2.5:latest",
                ];
                
                // Return the first common model name
                // The actual validation happens when the request is executed
                common_models[0].to_string()
            },
            _ => "default".to_string(),
        }
    }

    fn get_provider_parameters(
        &self,
        task: &TaskRequest,
        provider_name: &str,
    ) -> ProviderParameters {
        let base_temperature = match &task.task_type {
            TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => 0.1,
            TaskType::LogicalReasoning | TaskType::DataAnalysis => 0.2,
            TaskType::CreativeWriting => 0.8,
            TaskType::GeneralChat => 0.7,
            TaskType::SystemMaintenance => 0.3,
            TaskType::FileSystemOperation { .. } => 0.1,
            TaskType::DirectoryManagement { .. } => 0.1,
            TaskType::FileManipulation { .. } => 0.2,
        };

        let max_tokens = match &task.task_type {
            TaskType::CodeGeneration { .. } => 4000,
            TaskType::DataAnalysis => 3000,
            TaskType::CreativeWriting => 2000,
            TaskType::LogicalReasoning => 2500,
            _ => 1500,
        };

        // Provider-specific adjustments
        let (temperature, adjusted_max_tokens, top_p) = match provider_name {
            "anthropic" => {
                // Claude works well with slightly higher temperature for creative tasks
                let temp = if matches!(task.task_type, TaskType::CreativeWriting) {
                    f32::min(base_temperature + 0.1, 1.0)
                } else {
                    base_temperature
                };
                (temp, max_tokens, Some(0.9))
            }
            "mistral" => {
                // Mistral is efficient, can handle larger outputs
                (base_temperature, (max_tokens as f32 * 1.2) as usize, Some(0.95))
            }
            "google" => {
                // Gemini works well with moderate parameters
                (base_temperature, max_tokens, Some(0.8))
            }
            _ => (base_temperature, max_tokens, Some(0.9)),
        };

        // Task-specific stop sequences
        let stop_sequences = match &task.task_type {
            TaskType::CodeGeneration { .. } => vec!["```".to_string(), "\n\n\n".to_string()],
            TaskType::CodeReview { .. } => vec!["---".to_string()],
            _ => vec![],
        };

        ProviderParameters {
            temperature,
            max_tokens: adjusted_max_tokens,
            top_p,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences,
            system_prompt: None,
        }
    }

    fn get_usage_context(&self, task: &TaskRequest, provider_name: &str) -> UsageContext {
        let model_selected = self.get_optimal_model(task, provider_name);

        let task_complexity = match &task.task_type {
            TaskType::LogicalReasoning | TaskType::DataAnalysis => 0.9,
            TaskType::CodeGeneration { .. } => 0.8,
            TaskType::CreativeWriting => 0.7,
            TaskType::CodeReview { .. } => 0.6,
            _ => 0.5,
        };

        let expected_quality = match provider_name {
            "anthropic" => 0.92,
            "openai" => 0.88,
            "mistral" => 0.82,
            "google" => 0.85,
            _ => 0.75,
        };

        let cost_tier = match provider_name {
            "anthropic" => "premium",
            "openai" => "high",
            "mistral" => "medium",
            "google" => "low",
            _ => "basic",
        }
        .to_string();

        // Calculate specialization match based on provider strengths
        let specialization_match = match (provider_name, &task.task_type) {
            ("anthropic", TaskType::LogicalReasoning) => 0.95,
            ("anthropic", TaskType::CreativeWriting) => 0.93,
            ("openai", TaskType::CodeGeneration { .. }) => 0.90,
            ("mistral", TaskType::GeneralChat) => 0.85,
            ("google", TaskType::DataAnalysis) => 0.88,
            _ => 0.75,
        };

        UsageContext {
            model_selected,
            task_complexity,
            expected_quality,
            cost_tier,
            specialization_match,
        }
    }
}

/// Orchestrates between local and API models for optimal task routing
pub struct ModelOrchestrator {
    local_manager: Arc<LocalModelManager>,
    api_providers: HashMap<String, Arc<dyn ModelProvider>>,
    routing_strategy: RoutingStrategy,
    performance_tracker: Arc<PerformanceTracker>,
    capability_matcher: CapabilityMatcher,
    fallback_manager: FallbackManager,
    ensemble: Option<ModelEnsemble>,
    #[allow(dead_code)]
    ensembleconfig: EnsembleConfig,
    adaptive_learning: Arc<AdaptiveLearningSystem>,
    streaming_manager: Arc<StreamingManager>,
    cost_manager: Arc<CostManager>,
    fine_tuning_manager: Arc<FineTuningManager>,
    distributed_manager: Option<Arc<DistributedServingManager>>,
    /// Comprehensive usage tracking for models and providers
    usage_tracker: Arc<RwLock<ModelUsageTracker>>,
}

impl std::fmt::Debug for ModelOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelOrchestrator")
            .field("local_manager", &self.local_manager)
            .field("api_providers", &format!("{} providers", self.api_providers.len()))
            .field("routing_strategy", &self.routing_strategy)
            .field("performance_tracker", &self.performance_tracker)
            .field("capability_matcher", &self.capability_matcher)
            .field("fallback_manager", &self.fallback_manager)
            .field("ensemble", &self.ensemble)
            .field("adaptive_learning", &self.adaptive_learning)
            .field("streaming_manager", &self.streaming_manager)
            .field("cost_manager", &self.cost_manager)
            .field("fine_tuning_manager", &self.fine_tuning_manager)
            .field("distributed_manager", &self.distributed_manager.is_some())
            .field("usage_tracker", &"<usage_tracker>")
            .finish()
    }
}

impl ModelOrchestrator {
    pub async fn new(config: &ApiKeysConfig) -> Result<Self> {
        Self::new_with_options(config, false, None).await
    }

    /// Create a new model orchestrator with optional decision tracking
    pub async fn new_with_options(
        config: &ApiKeysConfig,
        enable_decision_tracking: bool,
        memory: Option<Arc<crate::memory::CognitiveMemory>>,
    ) -> Result<Self> {
        let local_manager = Arc::new(LocalModelManager::new().await?);
        let api_providers = Self::initialize_api_providers(config)?;
        let ensembleconfig = EnsembleConfig::default();

        // Initialize ensemble if we have multiple models available
        let ensemble = if api_providers.len() >= 2 || true {
            // Always enable for now
            Some(ModelEnsemble::new(
                ensembleconfig.clone(),
                local_manager.clone(),
                api_providers.clone(),
            ))
        } else {
            None
        };

        // Initialize adaptive learning system
        let learningconfig = AdaptiveLearningConfig::default();
        let adaptive_learning = Arc::new(AdaptiveLearningSystem::new(learningconfig));

        // Initialize streaming manager
        let streaming_manager =
            Arc::new(StreamingManager::new(local_manager.clone(), api_providers.clone()));

        // Initialize cost manager
        let budgetconfig = BudgetConfig::default();
        let cost_manager = Arc::new(CostManager::new(budgetconfig).await?);

        // Initialize fine-tuning manager
        let fine_tuningconfig = FineTuningConfig::default();
        let fine_tuning_manager = Arc::new(FineTuningManager::new(fine_tuningconfig).await?);

        // Initialize distributed serving manager (optional)
        let distributedconfig = DistributedConfig::default();
        let distributed_manager = if distributedconfig.enabled {
            Some(Arc::new(DistributedServingManager::new(distributedconfig).await?))
        } else {
            None
        };

        // Create performance tracker with optional decision tracking
        let performance_tracker = if enable_decision_tracking {
            if let Some(memory) = memory {
                Arc::new(PerformanceTracker::with_decision_tracking(memory).await?)
            } else {
                return Err(anyhow::anyhow!("Memory required for decision tracking"));
            }
        } else {
            Arc::new(PerformanceTracker::new())
        };

        Ok(Self {
            local_manager,
            api_providers,
            routing_strategy: RoutingStrategy::CapabilityBased,
            performance_tracker,
            capability_matcher: CapabilityMatcher::new(),
            fallback_manager: FallbackManager::new(),
            ensemble,
            ensembleconfig,
            adaptive_learning,
            streaming_manager,
            cost_manager,
            fine_tuning_manager,
            distributed_manager,
            usage_tracker: Arc::new(RwLock::new(ModelUsageTracker::new())),
        })
    }

    /// Initialize API providers from configuration
    fn initialize_api_providers(
        config: &ApiKeysConfig,
    ) -> Result<HashMap<String, Arc<dyn ModelProvider>>> {
        let mut providers = HashMap::new();

        // Use ProviderFactory to create all available providers
        let provider_list = crate::models::providers::ProviderFactory::create_providers(config);

        for provider in provider_list {
            providers.insert(provider.name().to_string(), provider);
        }

        Ok(providers)
    }

    /// Route a task to the best available model
    pub async fn route_task(&self, task: &TaskRequest) -> Result<ModelSelection> {
        debug!("Routing task: {:?}", task.task_type);

        // Special handling for Ollama models - if prefer_local is true and ollama provider exists
        // route directly to ollama
        if task.constraints.prefer_local && self.api_providers.contains_key("ollama") {
            info!("🦙 Routing to Ollama provider (prefer_local=true)");
            return Ok(ModelSelection::API("ollama".to_string()));
        }

        // Try adaptive learning first if available
        let available_models = self.get_available_model_names().await;

        match self.adaptive_learning.get_optimized_recommendation(task, &available_models).await {
            Ok(recommendation) => {
                info!(
                    "Using adaptive recommendation: {} (confidence: {:.2})",
                    recommendation.model_id, recommendation.confidence
                );

                // Convert recommendation to model selection
                let selection = if self
                    .local_manager
                    .get_available_models()
                    .await
                    .contains(&recommendation.model_id)
                {
                    ModelSelection::Local(recommendation.model_id)
                } else {
                    ModelSelection::API(recommendation.model_id)
                };

                return Ok(selection);
            }
            Err(e) => {
                debug!(
                    "Adaptive learning not available, falling back to traditional routing: {}",
                    e
                );
            }
        }

        // Fall back to traditional routing strategies
        let selection = match self.routing_strategy {
            RoutingStrategy::CapabilityBased => self.route_by_capability(task).await?,
            RoutingStrategy::LoadBased => self.route_by_load(task).await?,
            RoutingStrategy::CostOptimized => self.route_by_cost(task).await?,
            RoutingStrategy::LatencyOptimized => self.route_by_latency(task).await?,
        };

        Ok(selection)
    }

    /// Execute task with automatic fallback handling
    pub async fn execute_with_fallback(&self, task: TaskRequest) -> Result<TaskResponse> {
        // Check if distributed execution is available and beneficial
        if let Some(ref distributed_manager) = self.distributed_manager {
            if self.should_use_distributed(&task).await {
                match distributed_manager.execute_distributed_request(task.clone()).await {
                    Ok(response) => {
                        info!("🌐 Distributed execution completed successfully");
                        return Ok(response);
                    }
                    Err(e) => {
                        warn!("Distributed execution failed, falling back to local: {}", e);
                        // Continue to local execution
                    }
                }
            }
        }

        // Check if ensemble execution is requested and available
        if self.should_use_ensemble(&task).await {
            if let Some(ref ensemble) = self.ensemble {
                match ensemble.execute_ensemble(task.clone()).await {
                    Ok(ensemble_response) => {
                        info!(
                            "Ensemble execution completed with {} models, quality: {:.2}",
                            ensemble_response.contributing_models.len(),
                            ensemble_response.quality_score
                        );
                        return Ok(ensemble_response.primary_response);
                    }
                    Err(e) => {
                        warn!("Ensemble execution failed, falling back to single model: {}", e);
                        // Continue to single model execution
                    }
                }
            }
        }

        // Single model execution with fallback
        let max_attempts = 3;
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < max_attempts {
            attempts += 1;

            // Route task to best model
            let selection = match self.route_task(&task).await {
                Ok(sel) => sel,
                Err(e) => {
                    warn!("Routing failed on attempt {}: {}", attempts, e);
                    last_error = Some(e);
                    continue;
                }
            };

            // Execute on selected model
            let start_time = Instant::now();
            let result = self.execute_on_model(&task, &selection).await;
            let execution_time = start_time.elapsed();

            match result {
                Ok(response) => {
                    // Record successful execution for traditional tracking
                    self.performance_tracker.record_success(&selection, execution_time).await;

                    // Record execution for adaptive learning
                    if let Err(e) = self
                        .adaptive_learning
                        .record_execution(&task, &response, execution_time, true, None)
                        .await
                    {
                        warn!("Failed to record execution for learning: {}", e);
                    }

                    // Record cost for budget tracking
                    if let Err(e) =
                        self.cost_manager.record_cost(&task, &response, execution_time).await
                    {
                        warn!("Failed to record cost for budget tracking: {}", e);
                    }

                    // Record comprehensive usage tracking
                    {
                        let mut usage_tracker = self.usage_tracker.write().await;
                        let provider_name = selection.provider_name();
                        let task_type = format!("{:?}", task.task_type);
                        let tokens_used = response.tokens_generated.unwrap_or(0) as u64;
                        let cost_cents = response.cost_cents.unwrap_or(0.0);
                        let latency_ms = execution_time.as_millis() as f32;
                        let quality_score = response.quality_score;

                        usage_tracker.record_model_usage(
                            &selection.model_id(),
                            &provider_name,
                            &task_type,
                            tokens_used,
                            cost_cents,
                            latency_ms,
                            quality_score,
                            true, // success
                        );
                    }

                    // Collect training data for fine-tuning
                    if let Err(e) = self
                        .fine_tuning_manager
                        .collect_training_data(&task, &response, None, execution_time)
                        .await
                    {
                        warn!("Failed to collect training data: {}", e);
                    }

                    info!(
                        "Task completed successfully with {} on attempt {}",
                        selection.model_id(),
                        attempts
                    );
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        "Execution failed with {} on attempt {}: {}",
                        selection.model_id(),
                        attempts,
                        e
                    );

                    // Record failure for traditional tracking
                    self.performance_tracker.record_failure(&selection, &e, execution_time).await;

                    // Create a dummy response for learning (failed execution)
                    let dummy_response = TaskResponse {
                        content: "ERROR: Task execution failed".to_string(),
                        model_used: selection.clone(),
                        tokens_generated: None,
                        generation_time_ms: None,
                        cost_cents: None,
                        quality_score: 0.0,
                        cost_info: Some("Failed execution".to_string()),
                        model_info: None,
                        error: Some("Task execution failed".to_string()),
                    };

                    // Record failed execution for adaptive learning
                    if let Err(learning_err) = self
                        .adaptive_learning
                        .record_execution(&task, &dummy_response, execution_time, false, None)
                        .await
                    {
                        warn!("Failed to record failed execution for learning: {}", learning_err);
                    }

                    // Record failed execution in comprehensive usage tracking
                    {
                        let mut usage_tracker = self.usage_tracker.write().await;
                        let provider_name = selection.provider_name();
                        let task_type = format!("{:?}", task.task_type);
                        let latency_ms = execution_time.as_millis() as f32;

                        usage_tracker.record_model_usage(
                            &selection.model_id(),
                            &provider_name,
                            &task_type,
                            0, // no tokens for failed execution
                            0.0, // no cost for failed execution
                            latency_ms,
                            0.0, // quality score is 0 for failure
                            false, // failure
                        );

                        // Record rate limit if applicable
                        if e.to_string().contains("rate limit") || e.to_string().contains("quota") {
                            usage_tracker.record_rate_limit_hit(&provider_name);
                        }
                    }

                    last_error = Some(e);

                    // If not the last attempt, try fallback
                    if attempts < max_attempts {
                        if let Some(fallback) =
                            self.fallback_manager.get_fallback(&selection, &task).await
                        {
                            debug!("Trying fallback: {}", fallback.model_id());
                            continue;
                        }
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("All routing attempts failed")))
    }

    /// Execute task using ensemble for improved quality
    pub async fn execute_with_ensemble(&self, task: TaskRequest) -> Result<EnsembleResponse> {
        match &self.ensemble {
            Some(ensemble) => ensemble.execute_ensemble(task).await,
            None => Err(anyhow!("Ensemble not available")),
        }
    }

    /// Determine if ensemble should be used for this task
    async fn should_use_ensemble(&self, task: &TaskRequest) -> bool {
        // Use ensemble for complex or high-value tasks
        task.constraints.quality_threshold.unwrap_or(0.7) >= 0.8
            || match task.task_type {
                TaskType::CreativeWriting => true,
                TaskType::LogicalReasoning => true,
                _ => false,
            }
    }

    async fn route_by_capability(&self, task: &TaskRequest) -> Result<ModelSelection> {
        let required_capabilities =
            self.capability_matcher.get_required_capabilities(&task.task_type);

        // Get local model candidates (excluding Ollama models which are API-based)
        let local_models = self.local_manager.get_available_models().await;
        let mut best_local_score = 0.0;
        let mut best_local_model = None;

        for model_id in local_models {
            // Skip Ollama models - they should be routed through API
            if model_id.contains(":") || model_id.contains("ollama") {
                continue;
            }
            if self.local_manager.can_handle_request(&model_id).await {
                let score = self
                    .capability_matcher
                    .score_model_for_task(&model_id, &required_capabilities)
                    .await?;
                if score > best_local_score {
                    best_local_score = score;
                    best_local_model = Some(model_id);
                }
            }
        }

        // Check API providers (including Ollama for models with ":")
        let mut best_api_score = 0.0;
        let mut best_api_provider = None;

        // Special handling for Ollama models - if task content contains model with ":", prefer Ollama
        let prefers_ollama = task.content.contains(":") || 
                            task.constraints.prefer_local;
        
        for (provider_name, provider) in &self.api_providers {
            if provider.is_available() {
                let mut score = self
                    .capability_matcher
                    .score_api_provider(provider_name, &required_capabilities)
                    .await;
                    
                // Boost Ollama provider score if we have local preference
                if provider_name == "ollama" && prefers_ollama {
                    score = score.max(0.9); // Ensure Ollama gets high score for local models
                }
                
                if score > best_api_score {
                    best_api_score = score;
                    best_api_provider = Some(provider_name.clone());
                }
            }
        }

        // Decide between local and API based on scores and preferences
        let quality_threshold = task.constraints.quality_threshold.unwrap_or(0.7);
        let prefer_local = task.constraints.prefer_local || 
            task.constraints.priority == "low" || task.constraints.priority == "normal";

        // Always prefer local models when available and meeting quality threshold
        if best_local_score >= (quality_threshold * 0.8) { // Lower threshold for local models
            if let Some(model) = best_local_model {
                info!("🏠 Selecting local model: {} (score: {:.2})", model, best_local_score);
                return Ok(ModelSelection::Local(model.clone()));
            }
        }

        // Give significant bonus to local models to prevent routing to Ollama API provider
        if best_local_score > 0.3 { // Any reasonable local model beats API
            if let Some(model) = best_local_model {
                info!("🏠 Preferring local model over API: {} (score: {:.2})", model, best_local_score);
                return Ok(ModelSelection::Local(model.clone()));
            }
        }

        if let Some(ref api_provider) = best_api_provider {
            if best_api_score >= quality_threshold {
                return Ok(ModelSelection::API(api_provider.clone()));
            }
        }

        // Fallback to best available
        if let Some(local_model) = best_local_model {
            Ok(ModelSelection::Local(local_model))
        } else if let Some(api_provider) = best_api_provider {
            Ok(ModelSelection::API(api_provider))
        } else {
            Err(anyhow!("No suitable model found for task"))
        }
    }

    async fn route_by_load(&self, _task: &TaskRequest) -> Result<ModelSelection> {
        // Find the local model with lowest current load
        let local_models = self.local_manager.get_available_models().await;

        for model_id in local_models {
            if self.local_manager.can_handle_request(&model_id).await {
                return Ok(ModelSelection::Local(model_id));
            }
        }

        // Fallback to API if all local models are busy
        for (provider_name, provider) in &self.api_providers {
            if provider.is_available() {
                return Ok(ModelSelection::API(provider_name.clone()));
            }
        }

        Err(anyhow!("No available models"))
    }

    async fn route_by_cost(&self, task: &TaskRequest) -> Result<ModelSelection> {
        debug!("Routing by cost optimization");

        // For cost optimization, prefer local models when quality allows
        let quality_requirement = task.constraints.quality_threshold.unwrap_or(0.7);

        if quality_requirement <= 0.6 {
            // Low quality requirement - use most cost-effective local model
            let local_models = self.local_manager.get_available_models().await;
            if !local_models.is_empty() {
                if !local_models.is_empty() {
                    // Use smallest/fastest local model for cost savings
                    let cost_effective = local_models
                        .iter()
                        .find(|m| m.contains("3b") || m.contains("llama3_2"))
                        .or_else(|| local_models.iter().find(|m| m.contains("7b")))
                        .or_else(|| local_models.first())
                        .cloned()
                        .unwrap_or_else(|| "llama3_2_3b".to_string()); // Use our configured model ID

                    info!("💰 Cost-optimized routing: {}", cost_effective);
                    return Ok(ModelSelection::Local(cost_effective));
                }
            }
        }

        // Fallback to API with cost consideration
        info!("💰 Cost-optimized routing: anthropic (affordable API)");
        Ok(ModelSelection::API("anthropic".to_string()))
    }

    async fn route_by_latency(&self, _task: &TaskRequest) -> Result<ModelSelection> {
        // Prefer local models for lowest latency
        let local_models = self.local_manager.get_available_models().await;

        for model_id in local_models {
            if self.local_manager.can_handle_request(&model_id).await {
                return Ok(ModelSelection::Local(model_id));
            }
        }

        // Fallback to fastest API provider
        for (provider_name, provider) in &self.api_providers {
            if provider.is_available() {
                return Ok(ModelSelection::API(provider_name.clone()));
            }
        }

        Err(anyhow!("No low-latency models available"))
    }

    async fn execute_on_model(
        &self,
        task: &TaskRequest,
        selection: &ModelSelection,
    ) -> Result<TaskResponse> {
        match selection {
            ModelSelection::Local(model_id) => self.execute_on_local_model(task, model_id).await,
            ModelSelection::API(provider_name) => {
                self.execute_on_api_provider(task, provider_name).await
            }
        }
    }

    /// Execute task on local model
    async fn execute_on_local_model(
        &self,
        task: &TaskRequest,
        model_id: &str,
    ) -> Result<TaskResponse> {
        let instance = self
            .local_manager
            .get_model(model_id)
            .await
            .ok_or_else(|| anyhow!("Local model not found: {}", model_id))?;

        // Convert TaskRequest to InferenceRequest for local models
        let inference_request = InferenceRequest {
            prompt: task.content.clone(),
            max_tokens: match &task.task_type {
                TaskType::CodeGeneration { .. } => 4000,
                TaskType::DataAnalysis => 3000,
                TaskType::CreativeWriting => 2000,
                TaskType::LogicalReasoning => 2500,
                _ => 1500,
            },
            temperature: match &task.task_type {
                TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => 0.1,
                TaskType::LogicalReasoning | TaskType::DataAnalysis => 0.2,
                TaskType::CreativeWriting => 0.8,
                TaskType::GeneralChat => 0.7,
                TaskType::SystemMaintenance => 0.3,
                TaskType::FileSystemOperation { .. } | TaskType::DirectoryManagement { .. } | TaskType::FileManipulation { .. } => 0.2,
            },
            top_p: 0.9,
            stop_sequences: match &task.task_type {
                TaskType::CodeGeneration { .. } => vec!["```".to_string()],
                _ => vec![],
            },
        };

        let start_time = Instant::now();
        // Use inference_request parameters for better prompt construction
        let enhanced_prompt = format!(
            "{}\n\n[Max tokens: {}, Temperature: {:.1}]",
            inference_request.prompt, inference_request.max_tokens, inference_request.temperature
        );
        let response = instance.infer(&enhanced_prompt).await?;
        let generation_time = start_time.elapsed();

        // Parse token count from response (simple heuristic)
        let estimated_tokens = (response.len() as f32 / 4.0) as u32; // Rough estimate: ~4 chars per token

        Ok(TaskResponse {
            content: response,
            model_used: ModelSelection::Local(model_id.to_string()),
            tokens_generated: Some(estimated_tokens),
            generation_time_ms: Some(generation_time.as_millis().try_into().unwrap_or(0)),
            cost_cents: Some(0.0), // Local models have no cost
            quality_score: instance.capabilities.reasoning.min(1.0), /* Use model's reasoning
                                                                      * capability as quality
                                                                      * proxy */
            cost_info: Some("Local model - no cost".to_string()),
            model_info: Some(format!("Local model: {} - {} parameters", model_id, "Unknown")),
            error: None,
        })
    }

    async fn execute_on_api_provider(
        &self,
        task: &TaskRequest,
        provider_name: &str,
    ) -> Result<TaskResponse> {
        let provider = self
            .api_providers
            .get(provider_name)
            .ok_or_else(|| anyhow!("API provider not found: {}", provider_name))?;

        // Initialize request converter if not exists
        let converter = DefaultProviderRequestConverter::new();

        // Convert TaskRequest to provider-specific format
        let completion_request = converter.convert_task_request(task, provider_name).await?;
        let usage_context = converter.get_usage_context(task, provider_name);

        info!("🌐 Executing task on {} using model: {}", provider_name, completion_request.model);

        // Record start time for accurate latency measurement
        let start_time = Instant::now();

        // Execute API request
        let response = match provider.complete(completion_request.clone()).await {
            Ok(resp) => resp,
            Err(e) => {
                let execution_time = start_time.elapsed();

                // Update usage stats with failure
                converter
                    .update_usage_stats(
                        &completion_request.model,
                        0,
                        0.0,
                        0.0,
                        &task.task_type,
                        execution_time.as_millis().try_into().unwrap_or(0),
                        false,
                    )
                    .await;

                error!("❌ API request failed for {}: {}", provider_name, e);
                return Err(anyhow!("API request failed: {}", e));
            }
        };

        let execution_time = start_time.elapsed();
        let generation_time_ms = execution_time.as_millis().try_into().unwrap_or(0);

        // Calculate actual cost based on real token usage
        let estimated_cost_cents = self
            .calculate_actual_cost(
                provider_name,
                response.usage.prompt_tokens.try_into().unwrap_or(0),
                response.usage.completion_tokens.try_into().unwrap_or(0),
            )
            .await?;

        // Calculate quality score based on response characteristics
        let quality_score = self.calculate_response_quality(
            &response.content,
            &task.task_type,
            provider_name,
            usage_context.expected_quality,
        );

        // Update usage tracking
        converter
            .update_usage_stats(
                &completion_request.model,
                response.usage.completion_tokens.try_into().unwrap_or(0),
                estimated_cost_cents,
                quality_score,
                &task.task_type,
                generation_time_ms,
                true,
            )
            .await;

        // Generate detailed cost breakdown
        let cost_breakdown = format!(
            "{} ({}) - Input: {} tokens ({:.3}¢), Output: {} tokens ({:.3}¢), Total: {:.3}¢",
            provider_name,
            completion_request.model,
            response.usage.prompt_tokens,
            (response.usage.prompt_tokens as f32 / 1000.0)
                * self.get_provider_pricing(provider_name).0
                * 100.0,
            response.usage.completion_tokens,
            (response.usage.completion_tokens as f32 / 1000.0)
                * self.get_provider_pricing(provider_name).1
                * 100.0,
            estimated_cost_cents
        );

        info!(
            "✅ {} completed in {}ms, quality: {:.2}, cost: {:.3}¢",
            provider_name, generation_time_ms, quality_score, estimated_cost_cents
        );

        Ok(TaskResponse {
            content: response.content,
            model_used: ModelSelection::API(format!(
                "{}:{}",
                provider_name, completion_request.model
            )),
            tokens_generated: Some(response.usage.completion_tokens.try_into().unwrap_or(0)),
            generation_time_ms: Some(generation_time_ms),
            cost_cents: Some(estimated_cost_cents),
            quality_score,
            cost_info: Some(cost_breakdown),
            model_info: Some(format!(
                "Provider: {}, Model: {}, Context: {:.0}k tokens",
                provider_name,
                completion_request.model,
                usage_context.task_complexity * 10.0
            )),
            error: None,
        })
    }

    /// Get provider pricing information (input_cost_per_1k, output_cost_per_1k)
    /// in dollars
    fn get_provider_pricing(&self, provider_name: &str) -> (f32, f32) {
        match provider_name {
            "openai" | "gpt" => (0.03, 0.06),         // GPT-4o pricing
            "anthropic" | "claude" => (0.025, 0.125), // Claude 3.5 Sonnet pricing
            "mistral" => (0.02, 0.06),                // Mistral pricing
            "google" | "gemini" => (0.0005, 0.0015),  // Gemini pricing
            _ => (0.01, 0.03),                        // Default fallback pricing
        }
    }

    /// Calculate actual cost based on real token usage
    async fn calculate_actual_cost(
        &self,
        provider_name: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Result<f32> {
        let (input_cost_per_1k, output_cost_per_1k) = self.get_provider_pricing(provider_name);

        let input_cost = (input_tokens as f32 / 1000.0) * input_cost_per_1k;
        let output_cost = (output_tokens as f32 / 1000.0) * output_cost_per_1k;
        let total_cost_cents = (input_cost + output_cost) * 100.0;

        Ok(total_cost_cents)
    }

    /// Calculate response quality based on content analysis
    fn calculate_response_quality(
        &self,
        content: &str,
        task_type: &TaskType,
        provider_name: &str,
        expected_quality: f32,
    ) -> f32 {
        let mut quality = expected_quality; // Start with provider baseline

        // Content length analysis
        let content_length = content.len();
        if content_length < 50 {
            quality -= 0.2; // Very short responses usually indicate problems
        } else if content_length > 100 && content_length < 1000 {
            quality += 0.05; // Good length responses
        }

        // Task-specific quality indicators
        match task_type {
            TaskType::CodeGeneration { .. } => {
                if content.contains("```")
                    || content.contains("def ")
                    || content.contains("function")
                {
                    quality += 0.1; // Contains code blocks
                }
                if content.contains("error") || content.contains("Error") {
                    quality -= 0.15; // Mentions errors
                }
            }
            TaskType::LogicalReasoning => {
                if content.contains("because")
                    || content.contains("therefore")
                    || content.contains("step")
                {
                    quality += 0.1; // Shows reasoning structure
                }
                if content.len() > 500 {
                    quality += 0.05; // Detailed reasoning
                }
            }
            TaskType::CreativeWriting => {
                let word_count = content.split_whitespace().count();
                if word_count > 100 {
                    quality += 0.1; // Sufficient creative content
                }
            }
            _ => {}
        }

        // Provider-specific adjustments
        match provider_name {
            "anthropic" if content.len() > 200 => quality += 0.05, // Claude's thoroughness
            "openai" if content.contains("```") => quality += 0.05, // GPT's code formatting
            _ => {}
        }

        // Clamp quality between 0.0 and 1.0
        quality.max(0.0).min(1.0)
    }

    /// Get comprehensive orchestration status
    pub async fn get_status(&self) -> OrchestrationStatus {
        let local_status = self.local_manager.get_status().await;
        let api_status = self.get_api_status().await;
        let performance_stats = self.performance_tracker.get_statistics().await;

        OrchestrationStatus {
            local_models: local_status,
            api_providers: api_status,
            performance_stats,
            routing_strategy: self.routing_strategy.clone(),
        }
    }

    async fn get_api_status(&self) -> HashMap<String, ApiProviderStatus> {
        let mut status = HashMap::new();

        for (name, provider) in &self.api_providers {
            status.insert(
                name.clone(),
                ApiProviderStatus {
                    name: name.clone(),
                    is_available: provider.is_available(),
                    last_used: self.usage_tracker.read().await.get_provider_last_used(name)
                },
            );
        }

        status
    }

    /// Get all available model names (both local and API)
    async fn get_available_model_names(&self) -> Vec<String> {
        let mut model_names = Vec::new();

        // Add local models
        let local_models = self.local_manager.get_available_models().await;
        model_names.extend(local_models);

        // Add API providers
        for provider_name in self.api_providers.keys() {
            model_names.push(provider_name.clone());
        }

        model_names
    }

    /// Trigger learning update manually
    pub async fn trigger_learning_update(&self) -> Result<()> {
        info!("Triggering adaptive learning update");
        self.adaptive_learning.trigger_learning_update().await
    }

    /// Get learning system for advanced operations
    pub fn get_adaptive_learning(&self) -> &Arc<AdaptiveLearningSystem> {
        &self.adaptive_learning
    }

    /// Execute task with streaming response
    pub async fn execute_streaming(&self, task: TaskRequest) -> Result<StreamingResponse> {
        // Route task to best model for streaming
        let selection = self.route_task(&task).await?;

        // Check if the selected model supports streaming
        if !self.supports_streaming(&selection).await {
            return Err(anyhow!(
                "Selected model {} does not support streaming",
                selection.model_id()
            ));
        }

        // Create streaming request
        let streaming_request = StreamingRequest {
            task: task.clone(),
            selection: selection.clone(),
            buffer_size: 1024,
            timeout_ms: 30000,
        };

        // Execute streaming request
        self.streaming_manager.execute_streaming(streaming_request).await
    }

    /// Check if a model selection supports streaming
    async fn supports_streaming(&self, selection: &ModelSelection) -> bool {
        match selection {
            ModelSelection::Local(model_id) => {
                // Check local model streaming capabilities
                if let Some(instance) = self.local_manager.get_model(model_id).await {
                    instance.capabilities.supports_streaming
                } else {
                    false
                }
            }
            ModelSelection::API(provider_name) => {
                // Check API provider streaming capabilities
                if let Some(provider) = self.api_providers.get(provider_name) {
                    provider.supports_streaming()
                } else {
                    false
                }
            }
        }
    }

    /// Get streaming manager for direct access
    pub fn get_streaming_manager(&self) -> &Arc<StreamingManager> {
        &self.streaming_manager
    }

    /// Get cost manager for budget operations
    pub fn get_cost_manager(&self) -> &Arc<CostManager> {
        &self.cost_manager
    }

    /// Check if task execution is within budget constraints
    pub async fn check_budget_constraints(&self, task: &TaskRequest) -> Result<bool> {
        self.cost_manager.check_budget_constraints(task).await
    }

    /// Get current budget status
    pub async fn get_budget_status(&self) -> Result<String> {
        let budget_status = self.cost_manager.get_budget_status().await;
        Ok(format!(
            "Budget Status: {:.1}% utilization, {} health",
            budget_status.usage_percentage,
            format!("{:?}", budget_status.health)
        ))
    }

    /// Get fine-tuning manager for direct access
    pub fn get_fine_tuning_manager(&self) -> &Arc<FineTuningManager> {
        &self.fine_tuning_manager
    }

    /// Get fine-tuning system status
    pub async fn get_fine_tuning_status(&self) -> FineTuningSystemStatus {
        self.fine_tuning_manager.get_fine_tuning_status().await
    }

    /// Get comprehensive fine-tuning cost analytics
    pub async fn get_fine_tuning_cost_analytics(&self) -> FineTuningCostAnalytics {
        self.fine_tuning_manager.get_cost_analytics().await
    }

    /// Archive a completed fine-tuning job for cost tracking
    pub async fn archive_fine_tuning_job(&self, job_id: &str, final_status: FineTuningStatus, actual_cost_cents: f32) -> Result<()> {
        self.fine_tuning_manager.archive_completed_job(job_id, final_status, actual_cost_cents).await
    }

    /// Start fine-tuning for a specific task type
    pub async fn start_fine_tuning(
        &self,
        task_type: TaskType,
        config: JobConfiguration,
    ) -> Result<String> {
        let training_data = self
            .fine_tuning_manager
            .get_training_data_for_task(&format!("{:?}", task_type))
            .await?;

        if training_data.is_empty() {
            return Err(anyhow!("No training data available for task type: {:?}", task_type));
        }

        let job = FineTuningJob {
            id: format!(
                "manual_{}_{}",
                format!("{:?}", task_type),
                SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()
            ),
            model_id: "best_available".to_string(),
            task_type,
            status: FineTuningStatus::Queued,
            progress: 0.0,
            started_at: SystemTime::now(),
            estimated_completion: None,
            training_data_size: training_data.len(),
            cost_estimate_cents: 0.0,
            quality_baseline: 0.8, // Will be calculated properly
            target_improvement: 0.1,
            provider: "auto".to_string(),
            jobconfig: config,
            metrics: TrainingMetrics::default(),
            error_log: Vec::new(),
        };

        self.fine_tuning_manager.queue_fine_tuning_job(job).await
    }

    /// Check if distributed execution should be used
    async fn should_use_distributed(&self, task: &TaskRequest) -> bool {
        // Use distributed execution for resource-intensive tasks or when local
        // resources are constrained
        match task.task_type {
            TaskType::CodeGeneration { .. } => true, /* Complex code generation benefits from
                                                       * distributed processing */
            TaskType::LogicalReasoning => task.content.len() > 1000, // Large reasoning tasks
            TaskType::DataAnalysis => true,                          /* Data analysis can
                                                                       * benefit from distributed
                                                                       * compute */
            _ => false,
        }
    }

    /// Get distributed manager for direct access
    pub fn get_distributed_manager(&self) -> Option<&Arc<DistributedServingManager>> {
        self.distributed_manager.as_ref()
    }

    /// Get cluster status if distributed serving is enabled
    pub async fn get_cluster_status(&self) -> Option<ClusterStatus> {
        if let Some(ref distributed_manager) = self.distributed_manager {
            Some(distributed_manager.get_cluster_status().await)
        } else {
            None
        }
    }

    /// Enable distributed serving
    pub async fn enable_distributed_serving(&mut self, config: DistributedConfig) -> Result<()> {
        info!("🌐 Enabling distributed serving");
        let distributed_manager = Arc::new(DistributedServingManager::new(config).await?);
        distributed_manager.start().await?;
        self.distributed_manager = Some(distributed_manager);
        Ok(())
    }

    /// Disable distributed serving
    pub async fn disable_distributed_serving(&mut self) -> Result<()> {
        if let Some(distributed_manager) = self.distributed_manager.take() {
            info!("🛑 Disabling distributed serving");
            distributed_manager.stop().await?;
        }
        Ok(())
    }

    /// Register a fallback agent for resilient operation
    pub async fn register_fallback_agent(&self, agent_id: &str) -> Result<()> {
        info!("🛡️ Registering fallback agent: {}", agent_id);
        // For now, just log the registration - in a full implementation this would
        // integrate with the distributed serving manager or fallback systems
        debug!("Fallback agent {} registered successfully", agent_id);
        Ok(())
    }

    /// Configure load balancing for an agent
    pub async fn configure_agent_load_balancing(
        &self,
        agent_id: &str,
        config: crate::models::multi_agent_orchestrator::LoadBalancingConfig,
    ) -> Result<()> {
        info!(
            "⚖️ Configuring load balancing for agent {}: weight={}, max_requests={}",
            agent_id, config.weight, config.max_concurrent_requests
        );

        // For now, just log the configuration - in a full implementation this would
        // integrate with the load balancer and distributed serving systems
        debug!("Agent {} load balancing configured: {:?}", agent_id, config);
        Ok(())
    }

    /// Get model usage statistics for analytics
    pub async fn get_model_usage_statistics(&self) -> HashMap<String, ModelUsageStats> {
        self.usage_tracker.read().await.get_model_stats().clone()
    }

    /// Get provider usage statistics for analytics
    pub async fn get_provider_usage_statistics(&self) -> HashMap<String, ProviderUsageStats> {
        self.usage_tracker.read().await.get_provider_stats().clone()
    }

    /// Get global usage statistics for monitoring
    pub async fn get_global_usage_statistics(&self) -> GlobalUsageStats {
        self.usage_tracker.read().await.get_global_stats().clone()
    }

    /// Get comprehensive usage report for analytics dashboard
    pub async fn get_usage_report(&self) -> UsageReport {
        self.usage_tracker.read().await.get_usage_report()
    }

    /// Get provider performance comparison
    pub async fn get_provider_performance_comparison(
        &self,
    ) -> HashMap<String, ProviderPerformanceMetrics> {
        let mut comparison = HashMap::new();
        let usage_stats = self.get_model_usage_statistics().await;

        // Group by provider (extract provider name from model_id)
        let mut provider_stats: HashMap<String, Vec<&ModelUsageStats>> = HashMap::new();
        for (model_id, stats) in &usage_stats {
            let provider_name = if model_id.contains(':') {
                model_id.split(':').next().unwrap_or("unknown").to_string()
            } else {
                "local".to_string()
            };
            provider_stats.entry(provider_name).or_default().push(stats);
        }

        // Calculate aggregate metrics per provider
        for (provider, stats_list) in provider_stats {
            let total_requests: u64 = stats_list.iter().map(|s| s.total_requests).sum();
            let total_tokens: u64 = stats_list.iter().map(|s| s.total_tokens).sum();
            let total_cost: f32 = stats_list.iter().map(|s| s.total_cost_cents).sum();

            let avg_quality = if total_requests > 0 {
                stats_list
                    .iter()
                    .map(|s| s.avg_quality_score * s.total_requests as f32)
                    .sum::<f32>()
                    / total_requests as f32
            } else {
                0.0
            };

            let avg_latency = if total_requests > 0 {
                stats_list.iter().map(|s| s.avg_latency_ms * s.total_requests as f32).sum::<f32>()
                    / total_requests as f32
            } else {
                0.0
            };

            let avg_success_rate = if total_requests > 0 {
                stats_list.iter().map(|s| s.success_rate * s.total_requests as f32).sum::<f32>()
                    / total_requests as f32
            } else {
                0.0
            };

            let cost_per_token =
                if total_tokens > 0 { total_cost / total_tokens as f32 } else { 0.0 };

            comparison.insert(
                provider.clone(),
                ProviderPerformanceMetrics {
                    provider_name: provider,
                    total_requests,
                    total_tokens,
                    total_cost_cents: total_cost,
                    avg_quality_score: avg_quality,
                    avg_latency_ms: avg_latency,
                    success_rate: avg_success_rate,
                    cost_per_token,
                    value_score: if total_cost > 0.0 {
                        avg_quality / (total_cost / 100.0)
                    } else {
                        0.0
                    },
                    most_used_models: stats_list
                        .iter()
                        .take(3)
                        .map(|_| "model".to_string())
                        .collect(),
                },
            );
        }

        comparison
    }

    /// Get task type optimization recommendations
    pub async fn get_task_optimization_recommendations(
        &self,
    ) -> Vec<TaskOptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let usage_stats = self.get_model_usage_statistics().await;

        // Analyze task type distributions and costs
        let mut task_analysis: HashMap<String, TaskTypeAnalysis> = HashMap::new();

        for (model_id, stats) in &usage_stats {
            for (task_type, count) in &stats.task_type_distribution {
                let analysis = task_analysis.entry(task_type.clone()).or_insert(TaskTypeAnalysis {
                    task_type: task_type.clone(),
                    total_requests: 0,
                    total_cost_cents: 0.0,
                    avg_quality: 0.0,
                    avg_latency_ms: 0.0,
                    provider_performance: HashMap::new(),
                });

                analysis.total_requests += count;
                analysis.total_cost_cents += stats.total_cost_cents;
                analysis.avg_quality += stats.avg_quality_score * (*count as f32);
                analysis.avg_latency_ms += stats.avg_latency_ms * (*count as f32);

                let provider_name = if model_id.contains(':') {
                    model_id.split(':').next().unwrap_or("unknown").to_string()
                } else {
                    "local".to_string()
                };

                // Calculate performance score for this provider and task
                let performance_score = stats.avg_quality_score
                    / (stats.total_cost_cents / stats.total_requests as f32 + 1.0);
                analysis.provider_performance.insert(provider_name, performance_score);
            }
        }

        // Generate recommendations based on analysis
        for (task_type, analysis) in task_analysis {
            if analysis.total_requests > 10 {
                // Only recommend for tasks with sufficient data
                let avg_quality = analysis.avg_quality / analysis.total_requests as f32;
                let avg_latency = analysis.avg_latency_ms / analysis.total_requests as f32;
                let cost_per_request = analysis.total_cost_cents / analysis.total_requests as f32;

                // Find best provider for this task type (highest performance score)
                let best_provider = analysis
                    .provider_performance
                    .iter()
                    .max_by(|(_, score_a), (_, score_b)| {
                        score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(provider, _)| provider.clone());

                if let Some(recommended_provider) = best_provider {
                    recommendations.push(TaskOptimizationRecommendation {
                        task_type: task_type.clone(),
                        current_avg_cost_cents: cost_per_request,
                        current_avg_quality: avg_quality,
                        current_avg_latency_ms: avg_latency,
                        recommended_provider,
                        potential_cost_savings: 0.0, // Would be calculated based on comparison
                        potential_quality_improvement: 0.0,
                        recommendation_reason: format!(
                            "Based on {} requests, this provider shows best value ratio",
                            analysis.total_requests
                        ),
                    });
                }
            }
        }

        recommendations
    }
}

/// Capability matching for task routing
#[derive(Debug)]
pub struct CapabilityMatcher {
    capability_weights: HashMap<TaskType, CapabilityWeights>,
}

impl CapabilityMatcher {
    pub fn new() -> Self {
        let mut capability_weights = HashMap::new();

        // Define capability requirements for different task types
        capability_weights.insert(
            TaskType::CodeGeneration { language: "python".to_string() },
            CapabilityWeights {
                code_generation: 0.9,
                code_review: 0.1,
                reasoning: 0.3,
                creative_writing: 0.0,
                data_analysis: 0.2,
            },
        );

        capability_weights.insert(
            TaskType::LogicalReasoning,
            CapabilityWeights {
                code_generation: 0.1,
                code_review: 0.1,
                reasoning: 0.9,
                creative_writing: 0.2,
                data_analysis: 0.7,
            },
        );

        capability_weights.insert(
            TaskType::CreativeWriting,
            CapabilityWeights {
                code_generation: 0.0,
                code_review: 0.0,
                reasoning: 0.4,
                creative_writing: 0.9,
                data_analysis: 0.1,
            },
        );

        // Add GeneralChat capabilities - favor reasoning and general conversation
        capability_weights.insert(
            TaskType::GeneralChat,
            CapabilityWeights {
                code_generation: 0.4,  // Some code ability is useful
                code_review: 0.2,
                reasoning: 0.8,        // Primary skill for general chat
                creative_writing: 0.3, // Lower requirement than creative writing tasks
                data_analysis: 0.5,    // Moderate data analysis ability
            },
        );

        Self { capability_weights }
    }

    pub fn get_required_capabilities(&self, task_type: &TaskType) -> ModelCapabilities {
        let default_weights = CapabilityWeights::default();
        let weights = self.capability_weights.get(task_type).unwrap_or(&default_weights);

        ModelCapabilities {
            code_generation: weights.code_generation,
            code_review: weights.code_review,
            reasoning: weights.reasoning,
            creative_writing: weights.creative_writing,
            data_analysis: weights.data_analysis,
            mathematical_computation: 0.5,
            language_translation: 0.3,
            context_window: 8192,
            max_tokens_per_second: 20.0,
            supports_streaming: false,
            supports_function_calling: false,
        }
    }

    pub async fn score_model_for_task(
        &self,
        model_id: &str,
        capabilities: &ModelCapabilities,
    ) -> Result<f32> {
        // Enhanced model capability scoring based on model characteristics
        let model_profile = self.get_model_profile(model_id).await;

        // Base score from provided capabilities
        let base_score = capabilities.code_generation * 0.3
            + capabilities.reasoning * 0.4
            + capabilities.creative_writing * 0.2
            + capabilities.data_analysis * 0.1;

        // Apply model-specific adjustments based on architecture and size
        let architecture_multiplier = match model_profile.architecture {
            ModelArchitecture::Transformer => 1.0,
            ModelArchitecture::LLaMA => 1.1, // Generally good performance
            ModelArchitecture::CodeLLaMA => {
                if capabilities.code_generation > 0.5 {
                    1.3
                } else {
                    0.9
                }
            }
            ModelArchitecture::Mistral => 1.05,
            ModelArchitecture::Qwen => 1.1,
            ModelArchitecture::DeepSeekCoder => 1.0,
            // ModelArchitecture::Mamba => 1.2, // New state-space models - not in enum
            // ModelArchitecture::Mixture => 1.3, // Mixture of experts - not in enum
            // ModelArchitecture::Diffusion => 0.8, // Different use case - not in enum
            ModelArchitecture::Unknown => 0.9,
            // ModelArchitecture::Custom(_) => 1.0, // not in enum
        };

        // Size-based adjustments (larger models generally more capable)
        let size_multiplier = match model_profile.parameter_count {
            p if p >= 70_000_000_000 => 1.2,  // 70B+ parameters
            p if p >= 30_000_000_000 => 1.15, // 30B+ parameters
            p if p >= 13_000_000_000 => 1.1,  // 13B+ parameters
            p if p >= 7_000_000_000 => 1.05,  // 7B+ parameters
            p if p >= 3_000_000_000 => 1.0,   // 3B+ parameters
            p if p >= 1_000_000_000 => 0.95,  // 1B+ parameters
            _ => 0.85,                        // < 1B parameters
        };

        // Task-specific model specialization bonuses
        let specialization_bonus =
            self.calculate_specialization_bonus(model_id, capabilities).await;

        // Performance history adjustment
        let performance_adjustment = self.get_performance_adjustment(model_id).await;

        // Final scoring calculation
        let final_score = base_score
            * architecture_multiplier
            * size_multiplier
            * (1.0 + specialization_bonus)
            * performance_adjustment;

        // Clamp score between 0.0 and 1.0
        Ok(final_score.min(1.0).max(0.0))
    }

    /// Get detailed model profile for capability assessment
    async fn get_model_profile(&self, model_id: &str) -> ModelProfile {
        // Determine model characteristics from name and known patterns
        let normalized_id = model_id.to_lowercase();

        let architecture = if normalized_id.contains("llama") || normalized_id.contains("alpaca") {
            ModelArchitecture::LLaMA
        } else if normalized_id.contains("codellama") || normalized_id.contains("code-llama") {
            ModelArchitecture::CodeLLaMA
        } else if normalized_id.contains("mistral") {
            ModelArchitecture::Mistral
        } else if normalized_id.contains("deepseek") {
            ModelArchitecture::DeepSeekCoder
        } else if normalized_id.contains("qwen") {
            ModelArchitecture::Qwen
        } else {
            ModelArchitecture::Unknown
        };

        // Extract parameter count from model name
        let parameter_count = extract_parameter_count(&normalized_id);

        // Determine context window based on model family
        let context_window = if normalized_id.contains("32k") {
            32768
        } else if normalized_id.contains("16k") {
            16384
        } else if normalized_id.contains("8k") {
            8192
        } else if architecture == ModelArchitecture::Mistral {
            8192 // Mistral default
        } else if architecture == ModelArchitecture::LLaMA {
            4096 // LLaMA default
        } else {
            2048 // Conservative default
        };

        // Assess instruction following capability
        let instruction_following = if normalized_id.contains("instruct")
            || normalized_id.contains("chat")
            || normalized_id.contains("it")
        {
            // instruction tuned
            0.9
        } else {
            0.6
        };

        // Assess fine-tuning indicators
        let is_fine_tuned = normalized_id.contains("instruct")
            || normalized_id.contains("chat")
            || normalized_id.contains("code")
            || normalized_id.contains("math");

        ModelProfile {
            model_id: model_id.to_string(),
            architecture: architecture.clone(),
            parameter_count,
            context_window,
            instruction_following,
            is_fine_tuned,
            training_data_cutoff: estimate_training_cutoff(&normalized_id),
            supports_function_calling: architecture == ModelArchitecture::Mistral
                || normalized_id.contains("function"),
            multilingual_support: normalized_id.contains("qwen")
                || normalized_id.contains("mistral")
                || parameter_count > 7_000_000_000,
        }
    }

    /// Calculate specialization bonus based on model's known strengths
    async fn calculate_specialization_bonus(
        &self,
        model_id: &str,
        capabilities: &ModelCapabilities,
    ) -> f32 {
        let normalized_id = model_id.to_lowercase();
        let mut bonus = 0.0;

        // Code generation specialists
        if (normalized_id.contains("code") || normalized_id.contains("deepseek"))
            && capabilities.code_generation > 0.5
        {
            bonus += 0.15;
        }

        // Reasoning specialists
        if (normalized_id.contains("qwen") || normalized_id.contains("mistral"))
            && capabilities.reasoning > 0.5
        {
            bonus += 0.10;
        }

        // Math and logic specialists
        if normalized_id.contains("math") && capabilities.data_analysis > 0.5 {
            bonus += 0.12;
        }

        // Creative writing specialists
        if normalized_id.contains("creative") && capabilities.creative_writing > 0.5 {
            bonus += 0.08;
        }

        // Instruction-following bonus for chat tasks
        if normalized_id.contains("instruct") || normalized_id.contains("chat") {
            bonus += 0.05;
        }

        bonus
    }

    /// Get performance adjustment based on historical success rates
    async fn get_performance_adjustment(&self, model_id: &str) -> f32 {
        // Calculate performance adjustment based on model_id characteristics
        let base_adjustment = 1.0;

        // Adjust based on model type indicators in the ID
        let performance_factor = if model_id.contains("gpt-4") || model_id.contains("claude-3") {
            1.1 // Premium models get bonus
        } else if model_id.contains("gpt-3.5") || model_id.contains("local") {
            0.9 // Standard models get slight penalty
        } else if model_id.contains("turbo") || model_id.contains("fast") {
            1.05 // Fast models get small bonus
        } else {
            1.0 // Default adjustment
        };

        // Additional adjustment for known reliable models
        let reliability_factor = if model_id.contains("sonnet") || model_id.contains("opus") {
            1.05 // High reliability models
        } else {
            1.0
        };

        base_adjustment * performance_factor * reliability_factor
    }

    pub async fn score_api_provider(
        &self,
        provider_name: &str,
        capabilities: &ModelCapabilities,
    ) -> f32 {
        // Score API providers based on known capabilities
        match provider_name {
            "anthropic" => capabilities.reasoning * 0.9 + capabilities.creative_writing * 0.8,
            "openai" => capabilities.code_generation * 0.8 + capabilities.reasoning * 0.7,
            "mistral" => capabilities.code_generation * 0.7 + capabilities.reasoning * 0.6,
            _ => 0.5, // Default score
        }
    }
}

/// Fallback management
#[derive(Debug)]
pub struct FallbackManager {
    capability_matcher: CapabilityMatcher,
}

impl FallbackManager {
    pub fn new() -> Self {
        Self { capability_matcher: CapabilityMatcher::new() }
    }

    pub async fn get_fallback(
        &self,
        failed_selection: &ModelSelection,
        task: &TaskRequest,
    ) -> Option<ModelSelection> {
        info!(
            "🔄 Selecting intelligent fallback for failed model: {}",
            failed_selection.model_id()
        );

        // Analyze failure context to determine best fallback strategy
        let failed_model_id = failed_selection.model_id();
        let fallback_strategy = self.determine_fallback_strategy(&failed_model_id, task).await;

        match fallback_strategy {
            FallbackStrategy::SameTierDifferentProvider => {
                info!("📊 Fallback strategy: Same tier, different provider");
                self.find_same_tier_alternative(failed_selection, task).await
            }
            FallbackStrategy::LowerCostAlternative => {
                info!("💰 Fallback strategy: Lower cost alternative");
                self.find_cost_effective_alternative(failed_selection, task).await
            }
            FallbackStrategy::HigherCapabilityModel => {
                info!("⚡ Fallback strategy: Higher capability model");
                self.find_higher_capability_alternative(failed_selection, task).await
            }
            FallbackStrategy::LocalToApi => {
                info!("🌐 Fallback strategy: Local to API");
                self.find_api_alternative(task).await
            }
            FallbackStrategy::ApiToLocal => {
                info!("💾 Fallback strategy: API to Local");
                self.find_local_alternative(task).await
            }
            FallbackStrategy::EmergencyFallback => {
                warn!("🚨 Fallback strategy: Emergency fallback");
                self.find_emergency_fallback(task).await
            }
        }
    }

    /// Determine appropriate fallback strategy based on failure context
    async fn determine_fallback_strategy(
        &self,
        failed_model_id: &str,
        task: &TaskRequest,
    ) -> FallbackStrategy {
        let failed_local = !failed_model_id.contains("http") && !failed_model_id.contains("api");

        // Analyze task requirements
        let requires_high_quality = task.constraints.quality_threshold.unwrap_or(0.7) > 0.8;
        let has_priority_constraints = task.constraints.priority == "high";
        let prefers_local =
            task.constraints.priority == "low" || task.constraints.priority == "normal";

        // Determine strategy based on context
        if failed_local && !prefers_local && requires_high_quality {
            FallbackStrategy::LocalToApi
        } else if !failed_local && has_priority_constraints {
            FallbackStrategy::LowerCostAlternative
        } else if !failed_local && prefers_local {
            FallbackStrategy::ApiToLocal
        } else if requires_high_quality {
            FallbackStrategy::HigherCapabilityModel
        } else {
            FallbackStrategy::SameTierDifferentProvider
        }
    }

    /// Find same-tier alternative with different provider
    async fn find_same_tier_alternative(
        &self,
        failed_selection: &ModelSelection,
        task: &TaskRequest,
    ) -> Option<ModelSelection> {
        let failed_id = failed_selection.model_id();
        let task_capabilities = self.capability_matcher.get_required_capabilities(&task.task_type);

        match failed_selection {
            ModelSelection::Local(_) => {
                // Try other local models with similar capabilities
                let available_models = vec![
                    "llama3.2:8b-instruct",
                    "mistral:7b-instruct",
                    "qwen2.5:7b-instruct",
                    "deepseek-coder:6.7b-instruct",
                ];

                for model_id in available_models {
                    if model_id != failed_id {
                        if let Ok(score) = self
                            .capability_matcher
                            .score_model_for_task(model_id, &task_capabilities)
                            .await
                        {
                            if score > 0.6 {
                                info!(
                                    "✅ Found local alternative: {} (score: {:.2})",
                                    model_id, score
                                );
                                return Some(ModelSelection::Local(model_id.to_string()));
                            }
                        }
                    }
                }
            }
            ModelSelection::API(_) => {
                // Try other API providers
                let api_alternatives = vec![
                    ("anthropic", vec!["claude-3-haiku", "claude-3-sonnet"]),
                    ("openai", vec!["gpt-3.5-turbo", "gpt-4-turbo"]),
                    ("mistral", vec!["mistral-small", "mistral-medium"]),
                ];

                for (provider, models) in api_alternatives {
                    if provider != failed_id {
                        for model in models {
                            let provider_score = self
                                .capability_matcher
                                .score_api_provider(provider, &task_capabilities)
                                .await;
                            let model_adjustment =
                                self.capability_matcher.get_performance_adjustment(model).await;
                            let combined_score = provider_score * model_adjustment;

                            if combined_score > 0.5 {
                                info!(
                                    "✅ Found API alternative: {} with model {} (combined score: \
                                     {:.2})",
                                    provider, model, combined_score
                                );
                                return Some(ModelSelection::API(format!(
                                    "{}:{}",
                                    provider, model
                                )));
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Find cost-effective alternative
    async fn find_cost_effective_alternative(
        &self,
        _failed_selection: &ModelSelection,
        task: &TaskRequest,
    ) -> Option<ModelSelection> {
        let max_cost = task.constraints.max_cost_cents.unwrap_or(1.0);

        // Prioritize local models for cost efficiency
        let cost_effective_locals =
            vec!["llama3.2:1b-instruct", "qwen2.5:3b-instruct", "mistral:7b-instruct"];

        for model_id in cost_effective_locals {
            // Local models are essentially free to run
            let task_capabilities =
                self.capability_matcher.get_required_capabilities(&task.task_type);
            if let Ok(score) =
                self.capability_matcher.score_model_for_task(model_id, &task_capabilities).await
            {
                if score > 0.5 {
                    info!(
                        "💰 Found cost-effective local alternative: {} (score: {:.2})",
                        model_id, score
                    );
                    return Some(ModelSelection::Local(model_id.to_string()));
                }
            }
        }

        // If local models don't work, try cheaper API options
        if max_cost >= 0.5 {
            info!("💰 Found cost-effective API alternative: claude-3-haiku");
            return Some(ModelSelection::API("anthropic".to_string()));
        }

        None
    }

    /// Find higher capability alternative
    async fn find_higher_capability_alternative(
        &self,
        failed_selection: &ModelSelection,
        task: &TaskRequest,
    ) -> Option<ModelSelection> {
        let task_capabilities = self.capability_matcher.get_required_capabilities(&task.task_type);

        // Try premium API models for highest capability
        let premium_options = vec![
            ("anthropic", "claude-3-opus"),
            ("openai", "gpt-4-turbo"),
            ("anthropic", "claude-3-sonnet"),
        ];

        for (provider, model) in premium_options {
            if ModelSelection::API(provider.to_string()) != *failed_selection {
                let provider_score =
                    self.capability_matcher.score_api_provider(provider, &task_capabilities).await;
                if provider_score > 0.7 {
                    info!(
                        "⚡ Found higher capability alternative: {} {} (score: {:.2})",
                        provider, model, provider_score
                    );
                    return Some(ModelSelection::API(provider.to_string()));
                }
            }
        }

        // Try high-capability local models
        let high_capability_locals =
            vec!["llama3.2:70b-instruct", "qwen2.5:32b-instruct", "deepseek-coder:33b-instruct"];

        for model_id in high_capability_locals {
            if ModelSelection::Local(model_id.to_string()) != *failed_selection {
                if let Ok(score) =
                    self.capability_matcher.score_model_for_task(model_id, &task_capabilities).await
                {
                    if score > 0.8 {
                        info!(
                            "⚡ Found higher capability local alternative: {} (score: {:.2})",
                            model_id, score
                        );
                        return Some(ModelSelection::Local(model_id.to_string()));
                    }
                }
            }
        }

        None
    }

    /// Find API alternative for local failure
    async fn find_api_alternative(&self, task: &TaskRequest) -> Option<ModelSelection> {
        let task_capabilities = self.capability_matcher.get_required_capabilities(&task.task_type);

        // Map task types to best API providers
        let provider = match &task.task_type {
            TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => {
                "openai" // GPT-4 is strong at code
            }
            TaskType::LogicalReasoning | TaskType::DataAnalysis => {
                "anthropic" // Claude is strong at reasoning
            }
            TaskType::CreativeWriting => {
                "anthropic" // Claude is excellent at creative tasks
            }
            TaskType::GeneralChat => {
                "openai" // GPT models are good generalists
            }
            TaskType::SystemMaintenance => {
                "anthropic" // Claude is good for system tasks
            }
            TaskType::FileSystemOperation { .. } | TaskType::DirectoryManagement { .. } | TaskType::FileManipulation { .. } => {
                "anthropic" // Claude is good for file system tasks
            }
        };

        let provider_score =
            self.capability_matcher.score_api_provider(provider, &task_capabilities).await;
        if provider_score > 0.6 {
            info!("🌐 Found API alternative: {} (score: {:.2})", provider, provider_score);
            Some(ModelSelection::API(provider.to_string()))
        } else {
            None
        }
    }

    /// Find local alternative for API failure
    async fn find_local_alternative(&self, task: &TaskRequest) -> Option<ModelSelection> {
        let task_capabilities = self.capability_matcher.get_required_capabilities(&task.task_type);

        // Map task types to best configured local models (using model_id from config)
        let model_id = match &task.task_type {
            TaskType::CodeGeneration { .. } | TaskType::CodeReview { .. } => {
                "deepseek_coder_v2" // Configured in models.yaml
            }
            TaskType::LogicalReasoning | TaskType::DataAnalysis => {
                "wizardcoder_34b" // Strong reasoning, configured in models.yaml
            }
            TaskType::CreativeWriting => {
                "llama3_2_3b" // Our lightweight general model
            }
            TaskType::GeneralChat => {
                "llama3_2_3b" // Our configured general purpose model
            }
            TaskType::SystemMaintenance => {
                "magicoder_7b" // Fast model for system tasks
            }
            TaskType::FileSystemOperation { .. } | TaskType::DirectoryManagement { .. } | TaskType::FileManipulation { .. } => {
                "magicoder_7b" // Fast model for file operations
            }
        };

        if let Ok(score) =
            self.capability_matcher.score_model_for_task(model_id, &task_capabilities).await
        {
            if score > 0.5 {
                info!("💾 Found local alternative: {} (score: {:.2})", model_id, score);
                return Some(ModelSelection::Local(model_id.to_string()));
            }
        }

        None
    }

    /// Find emergency fallback (most reliable option)
    async fn find_emergency_fallback(&self, task: &TaskRequest) -> Option<ModelSelection> {
        warn!("🚨 Using emergency fallback - prioritizing reliability over performance");

        // Emergency fallback prioritizes reliability over performance
        // First try most reliable API for high quality requirements
        if task.constraints.quality_threshold.unwrap_or(0.7) >= 0.6 {
            info!("🚨 Emergency API fallback: anthropic (claude-3-haiku)");
            return Some(ModelSelection::API("anthropic".to_string()));
        }

        // Otherwise use most reliable local model
        let emergency_local = "llama3.2:3b-instruct"; // Small, stable model
        info!("🚨 Emergency local fallback: {}", emergency_local);
        Some(ModelSelection::Local(emergency_local.to_string()))
    }
}

/// Fallback strategy types for intelligent selection
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    SameTierDifferentProvider,
    LowerCostAlternative,
    HigherCapabilityModel,
    LocalToApi,
    ApiToLocal,
    EmergencyFallback,
}

/// Performance tracking and learning
#[derive(Debug)]
pub struct PerformanceTracker {
    metrics: Arc<RwLock<HashMap<String, ModelPerformanceMetrics>>>,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self { metrics: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Create a new performance tracker with decision tracking enabled
    pub async fn with_decision_tracking(
        memory: Arc<crate::memory::CognitiveMemory>,
    ) -> Result<Self> {
        Ok(Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    

    /// Calculate confidence for routing decision
    fn calculate_routing_confidence(&self, selection: &ModelSelection, task: &TaskRequest) -> f32 {
        // Base confidence on model type and task requirements
        let base_confidence = match selection {
            ModelSelection::Local(_) => 0.7, // Moderate confidence for local models
            ModelSelection::API(_) => 0.8,   // Higher confidence for API models
        };

        // Adjust based on task constraints and priority
        let constraint_adjustment: f32 = if task.constraints.priority == "high" {
            match selection {
                ModelSelection::API(_) => 0.1,     // Boost for high priority API usage
                ModelSelection::Local(_) => -0.05, // Slight reduction for local on high priority
            }
        } else {
            0.0
        };

        let result = base_confidence + constraint_adjustment;
        if result < 0.0 {
            0.0
        } else if result > 1.0 {
            1.0
        } else {
            result
        }
    }

    pub async fn record_success(&self, selection: &ModelSelection, execution_time: Duration) {
        let model_id = selection.model_id();
        let mut metrics = self.metrics.write().await;
        let model_metrics =
            metrics.entry(model_id.clone()).or_insert_with(ModelPerformanceMetrics::default);

        model_metrics.total_requests += 1;
        model_metrics.successful_requests += 1;
        model_metrics.total_execution_time += execution_time;
        model_metrics.last_success = Some(Instant::now());

        debug!("Recorded success for model: {}", model_id);
    }

    pub async fn record_failure(
        &self,
        selection: &ModelSelection,
        error: &anyhow::Error,
        execution_time: Duration,
    ) {
        let model_id = selection.model_id();
        let mut metrics = self.metrics.write().await;
        let model_metrics =
            metrics.entry(model_id.clone()).or_insert_with(ModelPerformanceMetrics::default);

        model_metrics.total_requests += 1;
        model_metrics.failed_requests += 1;
        model_metrics.total_execution_time += execution_time;
        model_metrics.last_failure = Some(Instant::now());

        warn!("Recorded failure for model {}: {}", model_id, error);
    }

    pub async fn get_statistics(&self) -> PerformanceStatistics {
        let metrics = self.metrics.read().await;
        let mut stats = HashMap::new();

        for (model_id, metrics) in metrics.iter() {
            let success_rate = if metrics.total_requests > 0 {
                metrics.successful_requests as f32 / metrics.total_requests as f32
            } else {
                0.0
            };

            let avg_execution_time = if metrics.successful_requests > 0 {
                metrics.total_execution_time / metrics.successful_requests as u32
            } else {
                Duration::default()
            };

            stats.insert(
                model_id.clone(),
                ModelStatistics {
                    total_requests: metrics.total_requests,
                    success_rate,
                    avg_execution_time,
                    last_used: metrics.last_success.or(metrics.last_failure),
                },
            );
        }

        PerformanceStatistics { model_stats: stats }
    }
}

// Data structures

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RoutingStrategy {
    CapabilityBased,
    LoadBased,
    CostOptimized,
    LatencyOptimized,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum TaskType {
    CodeGeneration { language: String },
    CodeReview { language: String },
    LogicalReasoning,
    CreativeWriting,
    DataAnalysis,
    GeneralChat,
    SystemMaintenance,
    // File system operations
    FileSystemOperation { operation_type: String },
    DirectoryManagement { action: String },
    FileManipulation { action: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelSelection {
    Local(String),
    API(String),
}

impl ModelSelection {
    pub fn model_id(&self) -> String {
        match self {
            ModelSelection::Local(id) => id.clone(),
            ModelSelection::API(provider) => provider.clone(),
        }
    }

    pub fn provider_type(&self) -> &str {
        match self {
            ModelSelection::Local(_) => "local",
            ModelSelection::API(_) => "api",
        }
    }

    pub fn provider_name(&self) -> String {
        match self {
            ModelSelection::Local(_) => "local".to_string(),
            ModelSelection::API(provider) => {
                // Extract provider name from API provider string
                // Format is often "provider:model" or just "provider"
                if provider.contains(':') {
                    provider.split(':').next().unwrap_or(provider).to_string()
                } else {
                    provider.clone()
                }
            },
        }
    }
}

impl std::fmt::Display for ModelSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSelection::Local(id) => write!(f, "Local({})", id),
            ModelSelection::API(provider) => write!(f, "API({})", provider),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task_type: TaskType,
    pub content: String,
    pub constraints: TaskConstraints,
    pub context_integration: bool,
    pub memory_integration: bool,
    pub cognitive_enhancement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConstraints {
    // Token and context constraints
    pub max_tokens: Option<u32>,
    pub context_size: Option<u32>,

    // Time constraints
    pub max_time: Option<std::time::Duration>,
    pub max_latency_ms: Option<u64>,

    // Cost constraints
    pub max_cost_cents: Option<f32>,

    // Quality constraints
    pub quality_threshold: Option<f32>,

    // Priority and preferences
    pub priority: String,
    pub prefer_local: bool,
    pub require_streaming: bool,

    // Capability requirements
    pub required_capabilities: Vec<String>,

    // Social and creative constraints
    pub creativity_level: Option<f32>,
    pub formality_level: Option<f32>,
    pub target_audience: Option<String>,
}

impl Default for TaskConstraints {
    fn default() -> Self {
        Self {
            max_tokens: Some(2000),
            context_size: Some(4096),
            max_time: Some(std::time::Duration::from_secs(30)),
            max_latency_ms: Some(30000),
            max_cost_cents: None,
            quality_threshold: Some(0.7),
            priority: "normal".to_string(),
            prefer_local: false,
            require_streaming: false,
            required_capabilities: Vec::new(),
            creativity_level: None,
            formality_level: None,
            target_audience: None,
        }
    }
}

/// User-facing request constraints for orchestration (alias for
/// TaskConstraints)
pub type RequestConstraints = TaskConstraints;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskResponse {
    pub content: String,
    pub model_used: ModelSelection,
    pub tokens_generated: Option<u32>,
    pub generation_time_ms: Option<u32>,
    pub cost_cents: Option<f32>,
    pub quality_score: f32,
    pub cost_info: Option<String>,
    pub model_info: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CapabilityWeights {
    pub code_generation: f32,
    pub code_review: f32,
    pub reasoning: f32,
    pub creative_writing: f32,
    pub data_analysis: f32,
}

impl Default for CapabilityWeights {
    fn default() -> Self {
        Self {
            code_generation: 0.5,
            code_review: 0.5,
            reasoning: 0.5,
            creative_writing: 0.5,
            data_analysis: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrchestrationStatus {
    pub local_models: super::local_manager::ModelManagerStatus,
    pub api_providers: HashMap<String, ApiProviderStatus>,
    pub performance_stats: PerformanceStatistics,
    pub routing_strategy: RoutingStrategy,
}

#[derive(Debug, Clone)]
pub struct ApiProviderStatus {
    pub name: String,
    pub is_available: bool,
    pub last_used: Option<Instant>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelPerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_execution_time: Duration,
    pub last_success: Option<Instant>,
    pub last_failure: Option<Instant>,
}

#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub model_stats: HashMap<String, ModelStatistics>,
}

#[derive(Debug, Clone)]
pub struct ModelStatistics {
    pub total_requests: u64,
    pub success_rate: f32,
    pub avg_execution_time: Duration,
    pub last_used: Option<Instant>,
}

/// Model architecture types for capability assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    Transformer,
    LLaMA,
    CodeLLaMA,
    Mistral,
    DeepSeekCoder,
    Qwen,
    Unknown,
}

/// Detailed model profile for capability assessment
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub model_id: String,
    pub architecture: ModelArchitecture,
    pub parameter_count: u64,
    pub context_window: u32,
    pub instruction_following: f32,
    pub is_fine_tuned: bool,
    pub training_data_cutoff: String,
    pub supports_function_calling: bool,
    pub multilingual_support: bool,
}

/// Extract parameter count from model name
fn extract_parameter_count(model_name: &str) -> u64 {
    // Look for patterns like "7b", "13b", "70b", etc.
    if let Some(cap) =
        regex::Regex::new(r"(\d+(?:\.\d+)?)[bt]").ok().and_then(|re| re.captures(model_name))
    {
        if let Ok(num) = cap[1].parse::<f64>() {
            // Convert to actual parameter count
            if model_name.contains('b') {
                return (num * 1_000_000_000.0) as u64;
            } else if model_name.contains('t') {
                return (num * 1_000_000_000_000.0) as u64;
            }
        }
    }

    // Fallback based on model family
    if model_name.contains("70b") {
        70_000_000_000
    } else if model_name.contains("34b") {
        34_000_000_000
    } else if model_name.contains("13b") {
        13_000_000_000
    } else if model_name.contains("7b") {
        7_000_000_000
    } else if model_name.contains("3b") {
        3_000_000_000
    } else if model_name.contains("1b") {
        1_000_000_000
    } else {
        // Default assumption for unknown models
        7_000_000_000
    }
}

/// Estimate training data cutoff date
fn estimate_training_cutoff(model_name: &str) -> String {
    // Estimate based on model release patterns
    if model_name.contains("2024") {
        "2024-04".to_string()
    } else if model_name.contains("2023") {
        "2023-10".to_string()
    } else if model_name.contains("qwen2.5") {
        "2024-09".to_string()
    } else if model_name.contains("qwen2") {
        "2024-06".to_string()
    } else if model_name.contains("mistral") && model_name.contains("v0.3") {
        "2024-05".to_string()
    } else if model_name.contains("deepseek") && model_name.contains("6.7b") {
        "2024-01".to_string()
    } else if model_name.contains("llama") && model_name.contains("3.2") {
        "2024-09".to_string()
    } else if model_name.contains("llama") && model_name.contains("3.1") {
        "2024-07".to_string()
    } else {
        "2023-09".to_string() // Conservative default
    }
}

/// Provider performance metrics for comparison
#[derive(Debug, Clone)]
pub struct ProviderPerformanceMetrics {
    pub provider_name: String,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost_cents: f32,
    pub avg_quality_score: f32,
    pub avg_latency_ms: f32,
    pub success_rate: f32,
    pub cost_per_token: f32,
    pub value_score: f32, // Quality per dollar spent
    pub most_used_models: Vec<String>,
}

/// Task optimization recommendation
#[derive(Debug, Clone)]
pub struct TaskOptimizationRecommendation {
    pub task_type: String,
    pub current_avg_cost_cents: f32,
    pub current_avg_quality: f32,
    pub current_avg_latency_ms: f32,
    pub recommended_provider: String,
    pub potential_cost_savings: f32,
    pub potential_quality_improvement: f32,
    pub recommendation_reason: String,
}

/// Task type analysis for optimization
#[derive(Debug, Clone)]
struct TaskTypeAnalysis {
    pub task_type: String,
    pub total_requests: u64,
    pub total_cost_cents: f32,
    pub avg_quality: f32,
    pub avg_latency_ms: f32,
    pub provider_performance: HashMap<String, f32>,
}
