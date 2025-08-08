//! Model Catalog
//! 
//! Stores and manages the collection of discovered AI models with search,
//! filtering, and comparison capabilities.

use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use anyhow::Result;

use super::discovery::AvailabilityStatus;

/// Model catalog for storing and querying models
#[derive(Debug, Clone)]
pub struct ModelCatalog {
    /// All models indexed by ID
    models: HashMap<String, ModelEntry>,
    
    /// Category index
    by_category: HashMap<ModelCategory, HashSet<String>>,
    
    /// Provider index
    by_provider: HashMap<String, HashSet<String>>,
    
    /// Capability index
    by_capability: HashMap<String, HashSet<String>>,
    
    /// Tag index
    by_tag: HashMap<String, HashSet<String>>,
    
    /// Currently selected model
    selected_model: Option<String>,
    
    /// Catalog metadata
    metadata: CatalogMetadata,
}

/// Catalog metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogMetadata {
    pub total_models: usize,
    pub total_providers: usize,
    pub last_updated: DateTime<Utc>,
    pub version: String,
}

/// Model entry in the catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub provider: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub category: ModelCategory,
    pub capabilities: Vec<String>,
    pub context_window: usize,
    pub pricing: PricingInfo,
    pub performance: PerformanceMetrics,
    pub availability: AvailabilityStatus,
    pub custom_config: Option<CustomConfig>,
    pub tags: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

/// Model categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCategory {
    Chat,
    Code,
    Instruct,
    Embedding,
    Vision,
    Audio,
    Video,
    General,
}

impl ModelCategory {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Chat,
            Self::Code,
            Self::Instruct,
            Self::Embedding,
            Self::Vision,
            Self::Audio,
            Self::Video,
            Self::General,
        ]
    }
    
    pub fn display_name(&self) -> &str {
        match self {
            Self::Chat => "Chat",
            Self::Code => "Code",
            Self::Instruct => "Instruction",
            Self::Embedding => "Embedding",
            Self::Vision => "Vision",
            Self::Audio => "Audio",
            Self::Video => "Video",
            Self::General => "General",
        }
    }
    
    pub fn icon(&self) -> &str {
        match self {
            Self::Chat => "💬",
            Self::Code => "🔧",
            Self::Instruct => "📝",
            Self::Embedding => "🔢",
            Self::Vision => "👁️",
            Self::Audio => "🎵",
            Self::Video => "🎬",
            Self::General => "🤖",
        }
    }
}

/// Pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingInfo {
    pub currency: String,
    pub input_per_1k_tokens: f64,
    pub output_per_1k_tokens: f64,
    pub fine_tuning_per_1k: Option<f64>,
    pub embedding_per_1k: Option<f64>,
}

impl Default for PricingInfo {
    fn default() -> Self {
        Self {
            currency: "USD".to_string(),
            input_per_1k_tokens: 0.0,
            output_per_1k_tokens: 0.0,
            fine_tuning_per_1k: None,
            embedding_per_1k: None,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub latency_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub accuracy_score: Option<f64>,
    pub benchmark_scores: HashMap<String, f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            latency_ms: None,
            tokens_per_second: None,
            accuracy_score: None,
            benchmark_scores: HashMap::new(),
        }
    }
}

/// Custom model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConfig {
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Filter criteria for model search
#[derive(Debug, Clone, Default)]
pub struct ModelFilter {
    pub category: Option<ModelCategory>,
    pub provider: Option<String>,
    pub min_context_window: Option<usize>,
    pub max_input_price: Option<f64>,
    pub capabilities: Vec<String>,
    pub tags: Vec<String>,
    pub availability: Option<AvailabilityStatus>,
}

/// Sort options for models
#[derive(Debug, Clone, Copy)]
pub enum SortBy {
    Name,
    Provider,
    ContextWindow,
    InputPrice,
    OutputPrice,
    Performance,
    LastUpdated,
}

impl ModelCatalog {
    /// Create a new empty catalog
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            by_category: HashMap::new(),
            by_provider: HashMap::new(),
            by_capability: HashMap::new(),
            by_tag: HashMap::new(),
            selected_model: None,
            metadata: CatalogMetadata {
                total_models: 0,
                total_providers: 0,
                last_updated: Utc::now(),
                version: "1.0.0".to_string(),
            },
        }
    }
    
    /// Register a model in the catalog (async version for compatibility)
    pub async fn register_model(&mut self, entry: ModelEntry) -> Result<()> {
        self.add_model(entry)
    }
    
    /// Mark a model as selected
    pub async fn mark_as_selected(&mut self, model_id: &str) {
        if self.models.contains_key(model_id) {
            self.selected_model = Some(model_id.to_string());
        }
    }
    
    /// Get the currently selected model
    pub fn get_selected_model(&self) -> Option<&str> {
        self.selected_model.as_deref()
    }
    
    /// Add a model to the catalog
    pub fn add_model(&mut self, entry: ModelEntry) -> Result<()> {
        let model_id = entry.id.clone();
        
        // Update category index
        self.by_category
            .entry(entry.category)
            .or_insert_with(HashSet::new)
            .insert(model_id.clone());
        
        // Update provider index
        self.by_provider
            .entry(entry.provider.clone())
            .or_insert_with(HashSet::new)
            .insert(model_id.clone());
        
        // Update capability index
        for capability in &entry.capabilities {
            self.by_capability
                .entry(capability.clone())
                .or_insert_with(HashSet::new)
                .insert(model_id.clone());
        }
        
        // Update tag index
        for tag in &entry.tags {
            self.by_tag
                .entry(tag.clone())
                .or_insert_with(HashSet::new)
                .insert(model_id.clone());
        }
        
        // Add model
        self.models.insert(model_id, entry);
        
        // Update metadata
        self.metadata.total_models = self.models.len();
        self.metadata.total_providers = self.by_provider.len();
        self.metadata.last_updated = Utc::now();
        
        Ok(())
    }
    
    /// Remove a model from the catalog
    pub fn remove_model(&mut self, model_id: &str) -> Result<()> {
        if let Some(entry) = self.models.remove(model_id) {
            // Remove from indices
            if let Some(set) = self.by_category.get_mut(&entry.category) {
                set.remove(model_id);
            }
            
            if let Some(set) = self.by_provider.get_mut(&entry.provider) {
                set.remove(model_id);
            }
            
            for capability in &entry.capabilities {
                if let Some(set) = self.by_capability.get_mut(capability) {
                    set.remove(model_id);
                }
            }
            
            for tag in &entry.tags {
                if let Some(set) = self.by_tag.get_mut(tag) {
                    set.remove(model_id);
                }
            }
            
            // Update metadata
            self.metadata.total_models = self.models.len();
            self.metadata.total_providers = self.by_provider.len();
            self.metadata.last_updated = Utc::now();
        }
        
        Ok(())
    }
    
    /// Get a model by ID
    pub fn get_model(&self, model_id: &str) -> Option<&ModelEntry> {
        self.models.get(model_id)
    }
    
    /// Get all models
    pub fn get_all_models(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }
    
    /// Get models by category
    pub fn get_by_category(&self, category: ModelCategory) -> Vec<ModelEntry> {
        self.by_category
            .get(&category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.models.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get models by provider
    pub fn get_by_provider(&self, provider: &str) -> Vec<ModelEntry> {
        self.by_provider
            .get(provider)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.models.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Search models by query
    pub fn search(&self, query: &str) -> Vec<ModelEntry> {
        let query_lower = query.to_lowercase();
        
        self.models
            .values()
            .filter(|entry| {
                entry.name.to_lowercase().contains(&query_lower) ||
                entry.description.to_lowercase().contains(&query_lower) ||
                entry.provider.to_lowercase().contains(&query_lower) ||
                entry.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect()
    }
    
    /// Filter models by criteria
    pub fn filter(&self, filter: &ModelFilter) -> Vec<ModelEntry> {
        self.models
            .values()
            .filter(|entry| {
                // Category filter
                if let Some(cat) = filter.category {
                    if entry.category != cat {
                        return false;
                    }
                }
                
                // Provider filter
                if let Some(ref provider) = filter.provider {
                    if entry.provider != *provider {
                        return false;
                    }
                }
                
                // Context window filter
                if let Some(min) = filter.min_context_window {
                    if entry.context_window < min {
                        return false;
                    }
                }
                
                // Price filter
                if let Some(max_price) = filter.max_input_price {
                    if entry.pricing.input_per_1k_tokens > max_price {
                        return false;
                    }
                }
                
                // Capability filter
                if !filter.capabilities.is_empty() {
                    let entry_caps: HashSet<_> = entry.capabilities.iter().collect();
                    for required_cap in &filter.capabilities {
                        if !entry_caps.contains(required_cap) {
                            return false;
                        }
                    }
                }
                
                // Tag filter
                if !filter.tags.is_empty() {
                    let entry_tags: HashSet<_> = entry.tags.iter().collect();
                    for required_tag in &filter.tags {
                        if !entry_tags.contains(required_tag) {
                            return false;
                        }
                    }
                }
                
                // Availability filter
                if let Some(availability) = &filter.availability {
                    if entry.availability != *availability {
                        return false;
                    }
                }
                
                true
            })
            .cloned()
            .collect()
    }
    
    /// Sort models
    pub fn sort(&self, models: &mut Vec<ModelEntry>, sort_by: SortBy, ascending: bool) {
        models.sort_by(|a, b| {
            let ordering = match sort_by {
                SortBy::Name => a.name.cmp(&b.name),
                SortBy::Provider => a.provider.cmp(&b.provider),
                SortBy::ContextWindow => a.context_window.cmp(&b.context_window),
                SortBy::InputPrice => {
                    a.pricing.input_per_1k_tokens
                        .partial_cmp(&b.pricing.input_per_1k_tokens)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                SortBy::OutputPrice => {
                    a.pricing.output_per_1k_tokens
                        .partial_cmp(&b.pricing.output_per_1k_tokens)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                SortBy::Performance => {
                    let a_score = a.performance.tokens_per_second.unwrap_or(0.0);
                    let b_score = b.performance.tokens_per_second.unwrap_or(0.0);
                    a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortBy::LastUpdated => a.last_updated.cmp(&b.last_updated),
            };
            
            if ascending {
                ordering
            } else {
                ordering.reverse()
            }
        });
    }
    
    /// Compare two models
    pub fn compare(&self, model1_id: &str, model2_id: &str) -> Option<ModelComparison> {
        let model1 = self.models.get(model1_id)?;
        let model2 = self.models.get(model2_id)?;
        
        Some(ModelComparison {
            model1: model1.clone(),
            model2: model2.clone(),
            context_window_diff: model1.context_window as i32 - model2.context_window as i32,
            input_price_diff: model1.pricing.input_per_1k_tokens - model2.pricing.input_per_1k_tokens,
            output_price_diff: model1.pricing.output_per_1k_tokens - model2.pricing.output_per_1k_tokens,
            performance_diff: {
                let perf1 = model1.performance.tokens_per_second.unwrap_or(0.0);
                let perf2 = model2.performance.tokens_per_second.unwrap_or(0.0);
                perf1 - perf2
            },
            common_capabilities: {
                let caps1: HashSet<_> = model1.capabilities.iter().collect();
                let caps2: HashSet<_> = model2.capabilities.iter().collect();
                caps1.intersection(&caps2).map(|s| s.to_string()).collect()
            },
            unique_to_model1: {
                let caps1: HashSet<_> = model1.capabilities.iter().collect();
                let caps2: HashSet<_> = model2.capabilities.iter().collect();
                caps1.difference(&caps2).map(|s| s.to_string()).collect()
            },
            unique_to_model2: {
                let caps1: HashSet<_> = model1.capabilities.iter().collect();
                let caps2: HashSet<_> = model2.capabilities.iter().collect();
                caps2.difference(&caps1).map(|s| s.to_string()).collect()
            },
        })
    }
    
    /// Get catalog statistics
    pub fn get_stats(&self) -> CatalogStats {
        let mut stats = CatalogStats::default();
        
        stats.total_models = self.models.len();
        stats.total_providers = self.by_provider.len();
        
        for category in ModelCategory::all() {
            let count = self.by_category.get(&category).map(|s| s.len()).unwrap_or(0);
            stats.models_by_category.insert(category, count);
        }
        
        for (provider, models) in &self.by_provider {
            stats.models_by_provider.insert(provider.clone(), models.len());
        }
        
        // Calculate price ranges
        let prices: Vec<f64> = self.models
            .values()
            .map(|m| m.pricing.input_per_1k_tokens)
            .filter(|p| *p > 0.0)
            .collect();
        
        if !prices.is_empty() {
            stats.min_input_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
            stats.max_input_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            stats.avg_input_price = prices.iter().sum::<f64>() / prices.len() as f64;
        }
        
        // Context window stats
        let contexts: Vec<usize> = self.models.values().map(|m| m.context_window).collect();
        if !contexts.is_empty() {
            stats.min_context_window = *contexts.iter().min().unwrap();
            stats.max_context_window = *contexts.iter().max().unwrap();
            stats.avg_context_window = contexts.iter().sum::<usize>() / contexts.len();
        }
        
        stats
    }
}

/// Model comparison result
#[derive(Debug, Clone)]
pub struct ModelComparison {
    pub model1: ModelEntry,
    pub model2: ModelEntry,
    pub context_window_diff: i32,
    pub input_price_diff: f64,
    pub output_price_diff: f64,
    pub performance_diff: f64,
    pub common_capabilities: Vec<String>,
    pub unique_to_model1: Vec<String>,
    pub unique_to_model2: Vec<String>,
}

/// Catalog statistics
#[derive(Debug, Clone, Default)]
pub struct CatalogStats {
    pub total_models: usize,
    pub total_providers: usize,
    pub models_by_category: HashMap<ModelCategory, usize>,
    pub models_by_provider: HashMap<String, usize>,
    pub min_input_price: f64,
    pub max_input_price: f64,
    pub avg_input_price: f64,
    pub min_context_window: usize,
    pub max_context_window: usize,
    pub avg_context_window: usize,
}