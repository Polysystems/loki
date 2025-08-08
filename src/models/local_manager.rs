use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::registry::{
    ModelCapabilities,
    ModelSpecialization,
    RegistryPerformanceMetrics,
    ResourceRequirements,
};
use crate::ollama::OllamaClient;

/// Manages multiple local models running in parallel
#[derive(Debug)]
pub struct LocalModelManager {
    active_models: Arc<RwLock<HashMap<String, LocalModelInstance>>>,
    ollama_client: OllamaClient,
    resource_allocator: ResourceAllocator,
    health_monitor: HealthMonitor,
}

impl LocalModelManager {
    pub async fn new() -> Result<Self> {
        let ollama_client = OllamaClient::new("http://localhost:11434")?;

        Ok(Self {
            active_models: Arc::new(RwLock::new(HashMap::new())),
            ollama_client,
            resource_allocator: ResourceAllocator::new(),
            health_monitor: HealthMonitor::new(),
        })
    }

    /// Load a model for parallel execution
    pub async fn load_model(&self, modelconfig: ModelLoadConfig) -> Result<()> {
        info!("Loading local model: {}", modelconfig.model_id);

        // Check if we have sufficient resources
        if !self.resource_allocator.can_allocate(&modelconfig.requirements) {
            return Err(anyhow::anyhow!(
                "Insufficient resources for model {}. Available: {:.1}GB, Required: {:.1}GB",
                modelconfig.model_id,
                self.resource_allocator.available_memory_gb(),
                modelconfig.requirements.min_memory_gb
            ));
        }

        // Check if model exists in Ollama, pull if needed
        let available_models = self.ollama_client.list_models().await?;
        if !available_models.iter().any(|m| m.name == modelconfig.ollama_name) {
            info!("Model {} not found locally, pulling...", modelconfig.ollama_name);
            self.ollama_client.pull_model(&modelconfig.ollama_name).await?;
        }

        // Create model instance
        let instance = LocalModelInstance::new(modelconfig.clone(), &self.ollama_client).await?;

        // Allocate resources
        self.resource_allocator.allocate(&modelconfig.model_id, &modelconfig.requirements)?;

        // Store in active models
        self.active_models.write().await.insert(modelconfig.model_id.clone(), instance);

        // Start health monitoring
        self.health_monitor.start_monitoring(&modelconfig.model_id).await;

        info!("Successfully loaded model: {}", modelconfig.model_id);
        Ok(())
    }

    /// Unload a model and free resources
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        info!("Unloading model: {}", model_id);

        // Remove from active models
        let _instance = self
            .active_models
            .write()
            .await
            .remove(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        // Free resources
        self.resource_allocator.deallocate(model_id)?;

        // Stop health monitoring
        self.health_monitor.stop_monitoring(model_id).await;

        info!("Successfully unloaded model: {}", model_id);
        Ok(())
    }

    /// Get available models for task routing
    pub async fn get_available_models(&self) -> Vec<String> {
        self.active_models.read().await.keys().cloned().collect()
    }

    /// Get model instance for execution
    pub async fn get_model(&self, model_id: &str) -> Option<LocalModelInstance> {
        self.active_models.read().await.get(model_id).cloned()
    }

    /// Check if model can handle additional requests
    pub async fn can_handle_request(&self, model_id: &str) -> bool {
        if let Some(instance) = self.active_models.read().await.get(model_id) {
            instance.can_handle_request()
        } else {
            false
        }
    }

    /// Get comprehensive status of all models
    pub async fn get_status(&self) -> ModelManagerStatus {
        let models = self.active_models.read().await;
        let mut model_statuses = HashMap::new();

        for (id, instance) in models.iter() {
            model_statuses.insert(
                id.clone(),
                ModelInstanceStatus {
                    model_id: id.clone(),
                    current_load: instance.current_load(),
                    max_concurrent: instance.max_concurrent,
                    memory_usage_mb: instance.estimated_memory_mb(),
                    health: instance.health.clone(),
                    last_request_time: instance.last_request_time.clone(),
                },
            );
        }

        ModelManagerStatus {
            total_models: models.len(),
            total_memory_allocated_gb: self.resource_allocator.allocated_memory_gb(),
            available_memory_gb: self.resource_allocator.available_memory_gb(),
            model_statuses,
        }
    }
}

/// Individual local model instance
#[derive(Clone)]
#[derive(Debug)]
pub struct LocalModelInstance {
    pub model_id: String,
    pub ollama_name: String,
    pub capabilities: ModelCapabilities,
    pub specializations: Vec<ModelSpecialization>,
    pub current_load: Arc<AtomicU32>,
    pub max_concurrent: u32,
    pub health: ModelHealth,
    pub last_request_time: Option<String>,
    pub performance_metrics: RegistryPerformanceMetrics,
    ollama_client: OllamaClient,

    // Lock-free atomic performance metrics
    total_requests: Arc<AtomicU32>,
    successful_requests: Arc<AtomicU32>,
    total_tokens_generated: Arc<AtomicU32>,
    total_generation_time_ms: Arc<AtomicU32>,
    average_tokens_per_second: Arc<std::sync::atomic::AtomicU64>, // Store as bits for f32
}

impl LocalModelInstance {
    pub async fn new(config: ModelLoadConfig, ollama_client: &OllamaClient) -> Result<Self> {
        let instance = Self {
            model_id: config.model_id.clone(),
            ollama_name: config.ollama_name.clone(),
            capabilities: config.capabilities,
            specializations: config.specializations,
            current_load: Arc::new(AtomicU32::new(0)),
            max_concurrent: config.max_concurrent_requests,
            health: ModelHealth::Healthy,
            last_request_time: None,
            performance_metrics: RegistryPerformanceMetrics::default(),
            ollama_client: ollama_client.clone(),

            // Initialize atomic performance metrics
            total_requests: Arc::new(AtomicU32::new(0)),
            successful_requests: Arc::new(AtomicU32::new(0)),
            total_tokens_generated: Arc::new(AtomicU32::new(0)),
            total_generation_time_ms: Arc::new(AtomicU32::new(0)),
            average_tokens_per_second: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };

        Ok(instance)
    }

    /// Check if instance can handle another request
    pub fn can_handle_request(&self) -> bool {
        self.current_load.load(Ordering::Relaxed) < self.max_concurrent
    }

    /// Get current load (0-max_concurrent)
    pub fn current_load(&self) -> u32 {
        self.current_load.load(Ordering::Relaxed)
    }

    /// Estimate memory usage in MB
    pub fn estimated_memory_mb(&self) -> u32 {
        // Rough estimation based on model parameters
        // This would be enhanced with actual memory monitoring
        1024 // Placeholder
    }

    /// Generate response using this model
    pub async fn generate(
        &self,
        request: LocalGenerationRequest,
    ) -> Result<LocalGenerationResponse> {
        // Increment load counter
        let previous_load = self.current_load.fetch_add(1, Ordering::Relaxed);

        if previous_load >= self.max_concurrent {
            self.current_load.fetch_sub(1, Ordering::Relaxed);
            return Err(anyhow::anyhow!(
                "Model at capacity: {}/{}",
                previous_load,
                self.max_concurrent
            ));
        }

        debug!(
            "Generating with model {} (load: {}/{})",
            self.model_id,
            previous_load + 1,
            self.max_concurrent
        );

        // Safety check for empty ollama_name - use model_id as fallback
        let model_name = if self.ollama_name.is_empty() {
            warn!("⚠️ Empty ollama_name for model {}, using model_id as fallback", self.model_id);
            &self.model_id
        } else {
            &self.ollama_name
        };
        
        let gen_request = crate::ollama::GenerationRequest {
            model: model_name.clone(),
            prompt: request.prompt.clone(),
            options: request.options.clone(),
        };

        let result = self.ollama_client.generate_detailed(&gen_request).await;

        // Decrement load counter
        self.current_load.fetch_sub(1, Ordering::Relaxed);

        match result {
            Ok(detailed_response) => {
                // Update performance metrics
                let tokens_per_second = if detailed_response.generation_time_ms > 0 {
                    (detailed_response.tokens_generated as f32 * 1000.0)
                        / detailed_response.generation_time_ms as f32
                } else {
                    0.0
                };

                // Update performance metrics using lock-free atomic operations
                self.update_atomic_performance_metrics(&detailed_response).await;
                // For now, we'll track basic metrics
                debug!(
                    "Generation completed: {} tokens in {}ms ({:.1} tokens/sec)",
                    detailed_response.tokens_generated,
                    detailed_response.generation_time_ms,
                    tokens_per_second
                );

                Ok(LocalGenerationResponse {
                    text: detailed_response.text,
                    model_id: self.model_id.clone(),
                    tokens_generated: detailed_response.tokens_generated,
                    generation_time_ms: detailed_response.generation_time_ms,
                })
            }
            Err(e) => {
                warn!("Generation failed for model {}: {}", self.model_id, e);
                Err(e)
            }
        }
    }

    /// Infer method for compatibility with other interfaces
    pub async fn infer(&self, prompt: &str) -> Result<String> {
        let request = LocalGenerationRequest { prompt: prompt.to_string(), options: None };

        let response = self.generate(request).await?;
        Ok(response.text)
    }

    /// Update performance metrics using atomic operations for lock-free
    /// performance
    async fn update_atomic_performance_metrics(
        &self,
        response: &crate::ollama::DetailedGenerationResponse,
    ) {
        // Update atomic counters
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_generated.fetch_add(response.tokens_generated, Ordering::Relaxed);
        self.total_generation_time_ms.fetch_add(response.generation_time_ms, Ordering::Relaxed);

        // Calculate and store tokens per second as atomic f32 (using bit
        // representation)
        let tokens_per_second = if response.generation_time_ms > 0 {
            (response.tokens_generated as f32 * 1000.0) / response.generation_time_ms as f32
        } else {
            0.0
        };

        // Store as u64 bit representation for atomic access
        let tps_bits = tokens_per_second.to_bits() as u64;
        self.average_tokens_per_second.store(tps_bits, Ordering::Relaxed);

        debug!("Updated atomic metrics for {}: {:.1} tokens/sec", self.model_id, tokens_per_second);
    }

    /// Get current performance metrics (lock-free)
    pub fn get_performance_metrics(&self) -> (u32, u32, u32, u32, f32) {
        let total = self.total_requests.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);
        let tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        let time_ms = self.total_generation_time_ms.load(Ordering::Relaxed);
        let tps_bits = self.average_tokens_per_second.load(Ordering::Relaxed);
        let tokens_per_second = f32::from_bits(tps_bits as u32);

        (total, successful, tokens, time_ms, tokens_per_second)
    }
}

/// Resource allocation and tracking
#[derive(Debug)]
pub struct ResourceAllocator {
    total_memory_gb: f32,
    allocated_memory: Arc<AtomicUsize>, // in MB
    model_assignments: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
}

impl ResourceAllocator {
    pub fn new() -> Self {
        // Detect system memory
        let total_memory_gb = Self::detect_system_memory();

        Self {
            total_memory_gb,
            allocated_memory: Arc::new(AtomicUsize::new(0)),
            model_assignments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn detect_system_memory() -> f32 {
        // Try to detect actual system memory
        #[cfg(feature = "sys-info")]
        {
            // Use system information if available
            8.0 // Default to 8GB as fallback
        }
        #[cfg(not(feature = "sys-info"))]
        {
            // Fallback to conservative estimate
            16.0
        }
    }

    pub fn can_allocate(&self, requirements: &ResourceRequirements) -> bool {
        let required_mb = (requirements.min_memory_gb * 1024.0) as usize;
        let current_allocated = self.allocated_memory.load(Ordering::Relaxed);
        let total_mb = (self.total_memory_gb * 1024.0) as usize;

        current_allocated + required_mb <= (total_mb * 85 / 100) // Leave 15% free
    }

    pub fn allocate(&self, model_id: &str, requirements: &ResourceRequirements) -> Result<()> {
        let required_mb = (requirements.min_memory_gb * 1024.0) as usize;

        if !self.can_allocate(requirements) {
            return Err(anyhow::anyhow!(
                "Cannot allocate {:.1}GB for model {}",
                requirements.min_memory_gb,
                model_id
            ));
        }

        // Record allocation
        let allocation = ResourceAllocation {
            model_id: model_id.to_string(),
            allocated_memory_mb: required_mb,
            gpu_memory_mb: requirements.min_gpu_memory_gb.map(|gb| (gb * 1024.0) as usize),
        };

        // Update totals
        self.allocated_memory.fetch_add(required_mb, Ordering::Relaxed);

        // Record assignment
        tokio::spawn({
            let assignments = self.model_assignments.clone();
            let model_id = model_id.to_string();
            let allocation = allocation.clone();
            async move {
                assignments.write().await.insert(model_id, allocation);
            }
        });

        info!("Allocated {:.1}GB for model {}", requirements.min_memory_gb, model_id);
        Ok(())
    }

    pub fn deallocate(&self, model_id: &str) -> Result<()> {
        tokio::spawn({
            let assignments = self.model_assignments.clone();
            let allocated_memory = self.allocated_memory.clone();
            let model_id = model_id.to_string();

            async move {
                if let Some(allocation) = assignments.write().await.remove(&model_id) {
                    allocated_memory.fetch_sub(allocation.allocated_memory_mb, Ordering::Relaxed);
                    info!(
                        "Deallocated {:.1}GB from model {}",
                        allocation.allocated_memory_mb as f32 / 1024.0,
                        model_id
                    );
                }
            }
        });

        Ok(())
    }

    pub fn available_memory_gb(&self) -> f32 {
        let allocated_mb = self.allocated_memory.load(Ordering::Relaxed);
        self.total_memory_gb - (allocated_mb as f32 / 1024.0)
    }

    pub fn allocated_memory_gb(&self) -> f32 {
        self.allocated_memory.load(Ordering::Relaxed) as f32 / 1024.0
    }
}

/// Health monitoring for models
#[derive(Debug)]
pub struct HealthMonitor {
    monitoring_tasks: Arc<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self { monitoring_tasks: Arc::new(RwLock::new(HashMap::new())) }
    }

    pub async fn start_monitoring(&self, model_id: &str) {
        let model_id = model_id.to_string();
        let model_id_for_task = model_id.clone();
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                interval.tick().await;
                // Implement health check logic here
                debug!("Health check for model: {}", model_id_for_task);
            }
        });

        self.monitoring_tasks.write().await.insert(model_id, task);
    }

    pub async fn stop_monitoring(&self, model_id: &str) {
        if let Some(task) = self.monitoring_tasks.write().await.remove(model_id) {
            task.abort();
        }
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub model_id: String,
    pub ollama_name: String,
    pub capabilities: ModelCapabilities,
    pub specializations: Vec<ModelSpecialization>,
    pub requirements: ResourceRequirements,
    pub max_concurrent_requests: u32,
}

#[derive(Debug, Clone)]
pub struct LocalGenerationRequest {
    pub prompt: String,
    pub options: Option<crate::ollama::GenerationOptions>,
}

#[derive(Debug, Clone)]
pub struct LocalGenerationResponse {
    pub text: String,
    pub model_id: String,
    pub tokens_generated: u32,
    pub generation_time_ms: u32,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub model_id: String,
    pub allocated_memory_mb: usize,
    pub gpu_memory_mb: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ModelInstanceStatus {
    pub model_id: String,
    pub current_load: u32,
    pub max_concurrent: u32,
    pub memory_usage_mb: u32,
    pub health: ModelHealth,
    pub last_request_time: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelManagerStatus {
    pub total_models: usize,
    pub total_memory_allocated_gb: f32,
    pub available_memory_gb: f32,
    pub model_statuses: HashMap<String, ModelInstanceStatus>,
}
