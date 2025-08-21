//! Agent Registry System
//! 
//! Provides persistent storage and management for agent configurations,
//! enabling CRUD operations, versioning, and runtime agent instantiation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};

use super::creation::{AgentConfig, AgentTemplate, ValidationRules};
use crate::cognitive::agents::AgentSpecialization;

/// Agent Registry for managing agent configurations
pub struct AgentRegistry {
    /// Storage backend
    storage: Box<dyn AgentStorage>,
    
    /// In-memory cache of agents
    cache: Arc<RwLock<HashMap<String, AgentEntry>>>,
    
    /// Available models cache
    available_models: Arc<RwLock<Vec<String>>>,
    
    /// Validation rules
    validation_rules: ValidationRules,
    
    /// Registry metadata
    metadata: RegistryMetadata,
}

/// Entry in the agent registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEntry {
    /// Agent configuration
    pub config: AgentConfig,
    
    /// Entry metadata
    pub metadata: EntryMetadata,
    
    /// Runtime state
    pub runtime_state: RuntimeState,
}

/// Metadata for a registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    
    /// Version number
    pub version: u32,
    
    /// Creator identifier
    pub created_by: String,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Usage statistics
    pub usage_stats: UsageStats,
}

/// Usage statistics for an agent
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageStats {
    /// Total number of activations
    pub activation_count: u64,
    
    /// Total runtime in seconds
    pub total_runtime_secs: u64,
    
    /// Number of successful tasks
    pub successful_tasks: u64,
    
    /// Number of failed tasks
    pub failed_tasks: u64,
    
    /// Average response time in milliseconds
    pub avg_response_time_ms: u64,
    
    /// Last activation timestamp
    pub last_activated: Option<DateTime<Utc>>,
}

/// Runtime state of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    /// Whether the agent is currently active
    pub is_active: bool,
    
    /// Current status
    pub status: AgentStatus,
    
    /// Assigned resources
    pub resources: ResourceAllocation,
    
    /// Active session ID if running
    pub session_id: Option<String>,
}

/// Agent status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    /// Agent is not instantiated
    Inactive,
    
    /// Agent is starting up
    Starting,
    
    /// Agent is ready for tasks
    Ready,
    
    /// Agent is processing a task
    Busy,
    
    /// Agent is in error state
    Error(String),
    
    /// Agent is shutting down
    Stopping,
}

/// Resource allocation for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Allocated memory in MB
    pub memory_mb: u32,
    
    /// CPU allocation percentage
    pub cpu_percent: u8,
    
    /// Token budget
    pub token_budget: u32,
    
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            memory_mb: 256,
            cpu_percent: 10,
            token_budget: 10000,
            rate_limit: 60,
        }
    }
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Registry version
    pub version: String,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last sync timestamp
    pub last_sync: DateTime<Utc>,
    
    /// Total agent count
    pub agent_count: usize,
    
    /// Storage location
    pub storage_path: Option<PathBuf>,
}

/// Storage backend trait
#[async_trait::async_trait]
pub trait AgentStorage: Send + Sync {
    /// Load all agents from storage
    async fn load_all(&self) -> Result<HashMap<String, AgentEntry>>;
    
    /// Save an agent to storage
    async fn save(&self, id: &str, entry: &AgentEntry) -> Result<()>;
    
    /// Delete an agent from storage
    async fn delete(&self, id: &str) -> Result<()>;
    
    /// Check if an agent exists
    async fn exists(&self, id: &str) -> Result<bool>;
    
    /// Get storage metadata
    async fn get_metadata(&self) -> Result<RegistryMetadata>;
    
    /// Update storage metadata
    async fn update_metadata(&self, metadata: &RegistryMetadata) -> Result<()>;
}

/// JSON file-based storage implementation
pub struct JsonFileStorage {
    base_path: PathBuf,
}

impl JsonFileStorage {
    /// Create a new JSON file storage
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&base_path)
            .context("Failed to create storage directory")?;
        
        Ok(Self { base_path })
    }
    
    fn agent_file_path(&self, id: &str) -> PathBuf {
        self.base_path.join(format!("{}.json", id))
    }
    
    fn metadata_file_path(&self) -> PathBuf {
        self.base_path.join("_metadata.json")
    }
}

#[async_trait::async_trait]
impl AgentStorage for JsonFileStorage {
    async fn load_all(&self) -> Result<HashMap<String, AgentEntry>> {
        let mut agents = HashMap::new();
        
        let entries = tokio::fs::read_dir(&self.base_path).await?;
        let mut entries = tokio_stream::wrappers::ReadDirStream::new(entries);
        
        use tokio_stream::StreamExt;
        while let Some(entry) = entries.next().await {
            let entry = entry?;
            let path = entry.path();
            
            // Skip non-JSON files and metadata
            if path.extension() != Some(std::ffi::OsStr::new("json")) {
                continue;
            }
            if path.file_name() == Some(std::ffi::OsStr::new("_metadata.json")) {
                continue;
            }
            
            // Load agent entry
            let content = tokio::fs::read_to_string(&path).await?;
            let agent_entry: AgentEntry = serde_json::from_str(&content)
                .context(format!("Failed to parse agent file: {:?}", path))?;
            
            let id = path.file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?
                .to_string();
            
            agents.insert(id, agent_entry);
        }
        
        Ok(agents)
    }
    
    async fn save(&self, id: &str, entry: &AgentEntry) -> Result<()> {
        let path = self.agent_file_path(id);
        let content = serde_json::to_string_pretty(entry)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }
    
    async fn delete(&self, id: &str) -> Result<()> {
        let path = self.agent_file_path(id);
        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }
    
    async fn exists(&self, id: &str) -> Result<bool> {
        Ok(self.agent_file_path(id).exists())
    }
    
    async fn get_metadata(&self) -> Result<RegistryMetadata> {
        let path = self.metadata_file_path();
        
        if path.exists() {
            let content = tokio::fs::read_to_string(path).await?;
            Ok(serde_json::from_str(&content)?)
        } else {
            // Create default metadata
            Ok(RegistryMetadata {
                version: "1.0.0".to_string(),
                created_at: Utc::now(),
                last_sync: Utc::now(),
                agent_count: 0,
                storage_path: Some(self.base_path.clone()),
            })
        }
    }
    
    async fn update_metadata(&self, metadata: &RegistryMetadata) -> Result<()> {
        let path = self.metadata_file_path();
        let content = serde_json::to_string_pretty(metadata)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }
}

impl AgentRegistry {
    /// Create a new agent registry with file storage
    pub async fn new(storage_path: impl AsRef<Path>) -> Result<Self> {
        let storage = Box::new(JsonFileStorage::new(storage_path)?);
        Self::with_storage(storage).await
    }
    
    /// Create a registry with a custom storage backend
    pub async fn with_storage(storage: Box<dyn AgentStorage>) -> Result<Self> {
        // Load metadata
        let mut metadata = storage.get_metadata().await?;
        
        // Load all agents into cache
        let agents = storage.load_all().await?;
        metadata.agent_count = agents.len();
        
        // Update sync time
        metadata.last_sync = Utc::now();
        storage.update_metadata(&metadata).await?;
        
        Ok(Self {
            storage,
            cache: Arc::new(RwLock::new(agents)),
            available_models: Arc::new(RwLock::new(Vec::new())),
            validation_rules: ValidationRules::default(),
            metadata,
        })
    }
    
    /// Register a new agent
    pub async fn register(&mut self, config: AgentConfig) -> Result<String> {
        let id = config.id.clone();
        
        // Validate configuration
        self.validate_config(&config)?;
        
        // Check if agent already exists
        if self.exists(&id).await? {
            return Err(anyhow::anyhow!("Agent with ID {} already exists", id));
        }
        
        // Create entry
        let entry = AgentEntry {
            config,
            metadata: EntryMetadata {
                created_at: Utc::now(),
                modified_at: Utc::now(),
                version: 1,
                created_by: "system".to_string(),
                tags: Vec::new(),
                usage_stats: UsageStats::default(),
            },
            runtime_state: RuntimeState {
                is_active: false,
                status: AgentStatus::Inactive,
                resources: ResourceAllocation::default(),
                session_id: None,
            },
        };
        
        // Save to storage
        self.storage.save(&id, &entry).await?;
        
        // Update cache
        self.cache.write().await.insert(id.clone(), entry);
        
        // Update metadata
        self.metadata.agent_count += 1;
        self.metadata.last_sync = Utc::now();
        self.storage.update_metadata(&self.metadata).await?;
        
        tracing::info!("Registered agent: {}", id);
        Ok(id)
    }
    
    /// Get an agent by ID
    pub async fn get(&self, id: &str) -> Result<Option<AgentEntry>> {
        Ok(self.cache.read().await.get(id).cloned())
    }
    
    /// Update an agent configuration
    pub async fn update(&mut self, id: &str, config: AgentConfig) -> Result<()> {
        // Validate configuration
        self.validate_config(&config)?;
        
        // Get existing entry
        let mut cache = self.cache.write().await;
        let entry = cache.get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found", id))?;
        
        // Update entry
        entry.config = config;
        entry.metadata.modified_at = Utc::now();
        entry.metadata.version += 1;
        
        // Save to storage
        self.storage.save(id, entry).await?;
        
        // Update metadata
        self.metadata.last_sync = Utc::now();
        self.storage.update_metadata(&self.metadata).await?;
        
        tracing::info!("Updated agent: {}", id);
        Ok(())
    }
    
    /// Delete an agent
    pub async fn delete(&mut self, id: &str) -> Result<()> {
        // Remove from cache
        let removed = self.cache.write().await.remove(id);
        
        if removed.is_none() {
            return Err(anyhow::anyhow!("Agent {} not found", id));
        }
        
        // Delete from storage
        self.storage.delete(id).await?;
        
        // Update metadata
        self.metadata.agent_count = self.metadata.agent_count.saturating_sub(1);
        self.metadata.last_sync = Utc::now();
        self.storage.update_metadata(&self.metadata).await?;
        
        tracing::info!("Deleted agent: {}", id);
        Ok(())
    }
    
    /// List all agents
    pub async fn list(&self) -> Result<Vec<(String, AgentEntry)>> {
        let cache = self.cache.read().await;
        Ok(cache.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }
    
    /// Check if an agent exists
    pub async fn exists(&self, id: &str) -> Result<bool> {
        Ok(self.cache.read().await.contains_key(id))
    }
    
    /// Update agent runtime state
    pub async fn update_runtime_state(&self, id: &str, state: RuntimeState) -> Result<()> {
        let mut cache = self.cache.write().await;
        let entry = cache.get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found", id))?;
        
        entry.runtime_state = state;
        entry.metadata.modified_at = Utc::now();
        
        // Save to storage
        self.storage.save(id, entry).await?;
        
        Ok(())
    }
    
    /// Update agent usage statistics
    pub async fn update_usage_stats<F>(&self, id: &str, updater: F) -> Result<()>
    where
        F: FnOnce(&mut UsageStats),
    {
        let mut cache = self.cache.write().await;
        let entry = cache.get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found", id))?;
        
        updater(&mut entry.metadata.usage_stats);
        entry.metadata.modified_at = Utc::now();
        
        // Save to storage
        self.storage.save(id, entry).await?;
        
        Ok(())
    }
    
    /// Get available models
    pub async fn get_available_models(&self) -> Vec<String> {
        self.available_models.read().await.clone()
    }
    
    /// Add an available model
    pub async fn add_available_model(&self, model_id: String) -> Result<()> {
        let mut models = self.available_models.write().await;
        if !models.contains(&model_id) {
            models.push(model_id);
            tracing::debug!("Added available model: {}", models.last().unwrap());
        }
        Ok(())
    }
    
    /// Set available models
    pub async fn set_available_models(&self, models: Vec<String>) -> Result<()> {
        *self.available_models.write().await = models;
        Ok(())
    }
    
    /// Validate agent configuration
    fn validate_config(&self, config: &AgentConfig) -> Result<()> {
        // Basic validation
        if config.name.is_empty() {
            return Err(anyhow::anyhow!("Agent name cannot be empty"));
        }
        
        if config.name.len() > 100 {
            return Err(anyhow::anyhow!("Agent name too long (max 100 characters)"));
        }
        
        if config.skills.is_empty() {
            return Err(anyhow::anyhow!("Agent must have at least one skill"));
        }
        
        Ok(())
    }
    
    /// Find agents by specialization
    pub async fn find_by_specialization(&self, spec: &AgentSpecialization) -> Result<Vec<(String, AgentEntry)>> {
        let cache = self.cache.read().await;
        Ok(cache.iter()
            .filter(|(_, entry)| entry.config.specialization == *spec)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
    
    /// Find agents by tag
    pub async fn find_by_tag(&self, tag: &str) -> Result<Vec<(String, AgentEntry)>> {
        let cache = self.cache.read().await;
        Ok(cache.iter()
            .filter(|(_, entry)| entry.metadata.tags.contains(&tag.to_string()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
    
    /// Get registry statistics
    pub async fn get_statistics(&self) -> RegistryStatistics {
        let cache = self.cache.read().await;
        
        let mut stats = RegistryStatistics::default();
        stats.total_agents = cache.len();
        
        for (_, entry) in cache.iter() {
            // Count by status
            match entry.runtime_state.status {
                AgentStatus::Ready => stats.ready_agents += 1,
                AgentStatus::Busy => stats.busy_agents += 1,
                AgentStatus::Error(_) => stats.error_agents += 1,
                _ => {}
            }
            
            // Count by specialization
            *stats.by_specialization.entry(entry.config.specialization.clone()).or_insert(0) += 1;
            
            // Aggregate usage stats
            stats.total_activations += entry.metadata.usage_stats.activation_count;
            stats.total_successful_tasks += entry.metadata.usage_stats.successful_tasks;
            stats.total_failed_tasks += entry.metadata.usage_stats.failed_tasks;
        }
        
        stats
    }
}

/// Registry statistics
#[derive(Debug, Clone, Default)]
pub struct RegistryStatistics {
    pub total_agents: usize,
    pub ready_agents: usize,
    pub busy_agents: usize,
    pub error_agents: usize,
    pub by_specialization: HashMap<AgentSpecialization, usize>,
    pub total_activations: u64,
    pub total_successful_tasks: u64,
    pub total_failed_tasks: u64,
}