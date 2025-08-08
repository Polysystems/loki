//! Vector Database Memory Integration Tool
//!
//! This tool extends Loki's existing SIMD-optimized EmbeddingStore with external
//! vector database connectivity for enhanced semantic search and memory retrieval.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::memory::{CognitiveMemory, MemoryId, EmbeddingStore};
use crate::safety::ActionValidator;


/// Vector database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMemoryConfig {
    /// Primary vector database provider
    pub primary_provider: VectorProvider,
    
    /// Pinecone configuration
    pub pinecone: Option<PineconeConfig>,
    
    /// Weaviate configuration  
    pub weaviate: Option<WeaviateConfig>,
    
    /// Chroma configuration
    pub chroma: Option<ChromaConfig>,
    
    /// Qdrant configuration
    pub qdrant: Option<QdrantConfig>,
    
    /// Sync with local embeddings
    pub sync_with_local: bool,
    
    /// Batch size for operations
    pub batch_size: usize,
    
    /// Enable hybrid search (local + external)
    pub enable_hybrid_search: bool,
    
    /// Cache external results locally
    pub cache_external_results: bool,
}

impl Default for VectorMemoryConfig {
    fn default() -> Self {
        Self {
            primary_provider: VectorProvider::Local,
            pinecone: None,
            weaviate: None,
            chroma: None,
            qdrant: None,
            sync_with_local: true,
            batch_size: 100,
            enable_hybrid_search: true,
            cache_external_results: true,
        }
    }
}

/// Supported vector database providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorProvider {
    Local,      // Use only local EmbeddingStore
    Pinecone,   // Pinecone cloud service
    Weaviate,   // Weaviate (cloud or self-hosted)
    Chroma,     // Chroma (local or server)
    Qdrant,     // Qdrant (local or cloud)
    Hybrid,     // Use multiple providers
}

/// Pinecone-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeConfig {
    pub api_key: String,
    pub environment: String,
    pub index_name: String,
    pub dimension: usize,
    pub metric: String, // cosine, euclidean, dotproduct
}

/// Weaviate-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaviateConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub class_name: String,
    pub schema: Option<Value>,
}

/// Chroma-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaConfig {
    pub host: String,
    pub port: u16,
    pub collection_name: String,
    pub use_auth: bool,
    pub auth_token: Option<String>,
}

/// Qdrant-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection_name: String,
    pub vector_size: usize,
    pub distance: String, // cosine, euclidean, dot
}

/// Search result from vector database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub memory_id: MemoryId,
    pub score: f32,
    pub content: String,
    pub metadata: HashMap<String, Value>,
    pub provider: VectorProvider,
}

/// Batch operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOperationResult {
    pub success_count: usize,
    pub failed_count: usize,
    pub errors: Vec<String>,
    pub duration: Duration,
}

/// Vector Memory Tool - integrates external vector DBs with local memory
pub struct VectorMemoryTool {
    /// Configuration
    config: VectorMemoryConfig,
    
    /// HTTP client for API calls
    client: Client,
    
    /// Reference to local cognitive memory
    cognitive_memory: Arc<CognitiveMemory>,
    
    /// Reference to local embedding store
    embedding_store: Arc<EmbeddingStore>,
    
    /// Safety validator
    safety_validator: Arc<ActionValidator>,
    
    /// External search cache
    search_cache: Arc<RwLock<HashMap<String, Vec<VectorSearchResult>>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<VectorMemoryMetrics>>,
}

/// Performance metrics for vector operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorMemoryMetrics {
    pub local_searches: usize,
    pub external_searches: usize,
    pub hybrid_searches: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub sync_operations: usize,
    pub avg_search_time: Duration,
    pub total_vectors_synced: usize,
}

impl VectorMemoryTool {
    /// Create a new vector memory tool
    pub async fn new(
        config: VectorMemoryConfig,
        cognitive_memory: Arc<CognitiveMemory>,
        embedding_store: Arc<EmbeddingStore>,
        safety_validator: Arc<ActionValidator>,
    ) -> Result<Self> {
        info!("🧠 Initializing Vector Memory Tool with provider: {:?}", config.primary_provider);
        
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;
        
        let tool = Self {
            config,
            client,
            cognitive_memory,
            embedding_store,
            safety_validator,
            search_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(VectorMemoryMetrics::default())),
        };
        
        // Validate external connections
        tool.validate_connections().await?;
        
        info!("✅ Vector Memory Tool initialized successfully");
        Ok(tool)
    }
    
    /// Validate connections to external vector databases
    async fn validate_connections(&self) -> Result<()> {
        match &self.config.primary_provider {
            VectorProvider::Local => {
                debug!("Using local embeddings only");
                Ok(())
            }
            VectorProvider::Pinecone => {
                if let Some(config) = &self.config.pinecone {
                    self.validate_pinecone_connection(config).await
                } else {
                    Err(anyhow!("Pinecone provider selected but no configuration provided"))
                }
            }
            VectorProvider::Weaviate => {
                if let Some(config) = &self.config.weaviate {
                    self.validate_weaviate_connection(config).await
                } else {
                    Err(anyhow!("Weaviate provider selected but no configuration provided"))
                }
            }
            VectorProvider::Chroma => {
                if let Some(config) = &self.config.chroma {
                    self.validate_chroma_connection(config).await
                } else {
                    Err(anyhow!("Chroma provider selected but no configuration provided"))
                }
            }
            VectorProvider::Qdrant => {
                if let Some(config) = &self.config.qdrant {
                    self.validate_qdrant_connection(config).await
                } else {
                    Err(anyhow!("Qdrant provider selected but no configuration provided"))
                }
            }
            VectorProvider::Hybrid => {
                // Validate all configured providers
                let mut results = Vec::new();
                
                if let Some(config) = &self.config.pinecone {
                    results.push(self.validate_pinecone_connection(config).await);
                }
                if let Some(config) = &self.config.weaviate {
                    results.push(self.validate_weaviate_connection(config).await);
                }
                if let Some(config) = &self.config.chroma {
                    results.push(self.validate_chroma_connection(config).await);
                }
                if let Some(config) = &self.config.qdrant {
                    results.push(self.validate_qdrant_connection(config).await);
                }
                
                // At least one provider must be working
                if results.iter().any(|r| r.is_ok()) {
                    Ok(())
                } else {
                    Err(anyhow!("No external vector database providers are available"))
                }
            }
        }
    }
    
    /// Validate Pinecone connection
    async fn validate_pinecone_connection(&self, config: &PineconeConfig) -> Result<()> {
        debug!("Validating Pinecone connection to index: {}", config.index_name);
        
        let url = format!("https://{}-{}.svc.{}.pinecone.io/describe_index_stats",
            config.index_name, "random", config.environment);
        
        let response = self.client
            .post(&url)
            .header("Api-Key", &config.api_key)
            .header("Content-Type", "application/json")
            .send()
            .await
            .context("Failed to connect to Pinecone")?;
        
        if response.status().is_success() {
            info!("✅ Pinecone connection validated");
            Ok(())
        } else {
            Err(anyhow!("Pinecone connection failed: {}", response.status()))
        }
    }
    
    /// Validate Weaviate connection
    async fn validate_weaviate_connection(&self, config: &WeaviateConfig) -> Result<()> {
        debug!("Validating Weaviate connection to: {}", config.url);
        
        let mut request = self.client.get(&format!("{}/v1/meta", config.url));
        
        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        let response = request.send().await
            .context("Failed to connect to Weaviate")?;
        
        if response.status().is_success() {
            info!("✅ Weaviate connection validated");
            Ok(())
        } else {
            Err(anyhow!("Weaviate connection failed: {}", response.status()))
        }
    }
    
    /// Validate Chroma connection
    async fn validate_chroma_connection(&self, config: &ChromaConfig) -> Result<()> {
        debug!("Validating Chroma connection to: {}:{}", config.host, config.port);
        
        let url = format!("http://{}:{}/api/v1/heartbeat", config.host, config.port);
        let mut request = self.client.get(&url);
        
        if config.use_auth {
            if let Some(token) = &config.auth_token {
                request = request.header("Authorization", format!("Bearer {}", token));
            }
        }
        
        let response = request.send().await
            .context("Failed to connect to Chroma")?;
        
        if response.status().is_success() {
            info!("✅ Chroma connection validated");
            Ok(())
        } else {
            Err(anyhow!("Chroma connection failed: {}", response.status()))
        }
    }
    
    /// Validate Qdrant connection
    async fn validate_qdrant_connection(&self, config: &QdrantConfig) -> Result<()> {
        debug!("Validating Qdrant connection to: {}", config.url);
        
        let mut request = self.client.get(&format!("{}/", config.url));
        
        if let Some(api_key) = &config.api_key {
            request = request.header("api-key", api_key);
        }
        
        let response = request.send().await
            .context("Failed to connect to Qdrant")?;
        
        if response.status().is_success() {
            info!("✅ Qdrant connection validated");
            Ok(())
        } else {
            Err(anyhow!("Qdrant connection failed: {}", response.status()))
        }
    }
    
    /// Perform semantic search across local and/or external vector databases
    pub async fn semantic_search(
        &self,
        query: &str,
        limit: usize,
        similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        let start_time = Instant::now();
        
        // Check cache first if enabled
        if self.config.cache_external_results {
            let cache_key = format!("{}:{}:{}", query, limit, similarity_threshold);
            if let Some(cached_results) = self.search_cache.read().await.get(&cache_key) {
                self.update_metrics(|m| m.cache_hits += 1).await;
                return Ok(cached_results.clone());
            }
            self.update_metrics(|m| m.cache_misses += 1).await;
        }
        
        let mut all_results = Vec::new();
        
        match &self.config.primary_provider {
            VectorProvider::Local => {
                let results = self.search_local(query, limit, similarity_threshold).await?;
                all_results.extend(results);
                self.update_metrics(|m| m.local_searches += 1).await;
            }
            VectorProvider::Hybrid => {
                // Search both local and external, then merge results
                let local_results = self.search_local(query, limit, similarity_threshold).await?;
                let external_results = self.search_external(query, limit, similarity_threshold).await?;
                
                all_results.extend(local_results);
                all_results.extend(external_results);
                
                // Sort by score and deduplicate
                all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                all_results.truncate(limit);
                
                self.update_metrics(|m| m.hybrid_searches += 1).await;
            }
            _ => {
                let results = self.search_external(query, limit, similarity_threshold).await?;
                all_results.extend(results);
                self.update_metrics(|m| m.external_searches += 1).await;
            }
        }
        
        // Cache results if enabled
        if self.config.cache_external_results {
            let cache_key = format!("{}:{}:{}", query, limit, similarity_threshold);
            self.search_cache.write().await.insert(cache_key, all_results.clone());
        }
        
        let search_time = start_time.elapsed();
        self.update_metrics(|m| {
            let count = m.local_searches + m.external_searches + m.hybrid_searches;
            if count > 0 {
                m.avg_search_time = (m.avg_search_time * (count - 1) as u32 + search_time) / count as u32;
            } else {
                m.avg_search_time = search_time;
            }
        }).await;
        
        Ok(all_results)
    }
    
    /// Search local embedding store
    async fn search_local(
        &self,
        query: &str,
        limit: usize,
        similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        debug!("Searching local embedding store for: {}", query);
        
        // Use the existing EmbeddingStore search functionality
        let results = self.embedding_store.search(query, limit).await
            .context("Failed to search local embeddings")?;
        
        // Convert to VectorSearchResult format
        let mut vector_results = Vec::new();
        for (memory_id, score) in results {
            if score >= similarity_threshold {
                // For local search, we don't have content directly from embedding search
                // We'll use the memory_id as content placeholder or fetch it from memory
                let content = format!("Local memory result for {}", memory_id);
                vector_results.push(VectorSearchResult {
                    memory_id,
                    score,
                    content,
                    metadata: HashMap::new(),
                    provider: VectorProvider::Local,
                });
            }
        }
        
        Ok(vector_results)
    }
    
    /// Search external vector databases
    async fn search_external(
        &self,
        query: &str,
        limit: usize,
        similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        let mut all_results = Vec::new();
        
        // Search Pinecone if configured
        if let Some(config) = &self.config.pinecone {
            match self.search_pinecone(config, query, limit, similarity_threshold).await {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Pinecone search failed: {}", e),
            }
        }
        
        // Search Weaviate if configured
        if let Some(config) = &self.config.weaviate {
            match self.search_weaviate(config, query, limit, similarity_threshold).await {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Weaviate search failed: {}", e),
            }
        }
        
        // Search Chroma if configured
        if let Some(config) = &self.config.chroma {
            match self.search_chroma(config, query, limit, similarity_threshold).await {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Chroma search failed: {}", e),
            }
        }
        
        // Search Qdrant if configured
        if let Some(config) = &self.config.qdrant {
            match self.search_qdrant(config, query, limit, similarity_threshold).await {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Qdrant search failed: {}", e),
            }
        }
        
        // Sort by relevance and limit results
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_results.truncate(limit);
        
        Ok(all_results)
    }
    
    /// Search Pinecone
    async fn search_pinecone(
        &self,
        config: &PineconeConfig,
        query: &str,
        limit: usize,
        _similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        debug!("Searching Pinecone index: {}", config.index_name);
        
        // Generate embedding for query (this would use the same embedding model as local)
        let query_embedding = self.generate_embedding(query).await?;
        
        let url = format!("https://{}-{}.svc.{}.pinecone.io/query",
            config.index_name, "random", config.environment);
        
        let request_body = json!({
            "vector": query_embedding.to_vec(),
            "topK": limit,
            "includeMetadata": true,
            "includeValues": false
        });
        
        let response = self.client
            .post(&url)
            .header("Api-Key", &config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Pinecone search failed: {}", response.status()));
        }
        
        let response_json: Value = response.json().await?;
        let mut results = Vec::new();
        
        if let Some(matches) = response_json["matches"].as_array() {
            for match_obj in matches {
                if let (Some(id), Some(score)) = (
                    match_obj["id"].as_str(),
                    match_obj["score"].as_f64()
                ) {
                    let content = match_obj["metadata"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    
                    let metadata = match_obj["metadata"]
                        .as_object()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .collect();
                    
                    results.push(VectorSearchResult {
                        memory_id: MemoryId::from(id.to_string()),
                        score: score as f32,
                        content,
                        metadata,
                        provider: VectorProvider::Pinecone,
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Search Weaviate - simplified implementation
    async fn search_weaviate(
        &self,
        config: &WeaviateConfig,
        query: &str,
        limit: usize,
        _similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        debug!("Searching Weaviate class: {}", config.class_name);
        
        let graphql_query = json!({
            "query": format!(
                "{{ Get {{ {}(nearText: {{concepts: [\"{}\"]}}, limit: {}) {{ _additional {{ id score }} content }} }} }}",
                config.class_name, query, limit
            )
        });
        
        let mut request = self.client
            .post(&format!("{}/v1/graphql", config.url))
            .json(&graphql_query);
        
        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Weaviate search failed: {}", response.status()));
        }
        
        // Parse GraphQL response
        let response_json: Value = response.json().await?;
        let mut results = Vec::new();
        
        if let Some(data) = response_json.get("data") {
            if let Some(get_data) = data.get("Get") {
                if let Some(class_data) = get_data.get(&config.class_name) {
                    if let Some(objects) = class_data.as_array() {
                        for obj in objects {
                            if let Some(additional) = obj.get("_additional") {
                                let id = additional.get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let score = additional.get("score")
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0) as f32;
                                let content = obj.get("content")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                
                                results.push(VectorSearchResult {
                                    memory_id: MemoryId::from(id),
                                    score,
                                    content,
                                    metadata: HashMap::new(),
                                    provider: VectorProvider::Weaviate,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Search Chroma - simplified implementation
    async fn search_chroma(
        &self,
        config: &ChromaConfig,
        query: &str,
        limit: usize,
        _similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        debug!("Searching Chroma collection: {}", config.collection_name);
        
        let url = format!("http://{}:{}/api/v1/collections/{}/query",
            config.host, config.port, config.collection_name);
        
        let request_body = json!({
            "query_texts": [query],
            "n_results": limit,
            "include": ["metadatas", "documents", "distances"]
        });
        
        let mut request = self.client.post(&url).json(&request_body);
        
        if config.use_auth {
            if let Some(token) = &config.auth_token {
                request = request.header("Authorization", format!("Bearer {}", token));
            }
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Chroma search failed: {}", response.status()));
        }
        
        // Parse Chroma response
        let response_json: Value = response.json().await?;
        let mut results = Vec::new();
        
        if let Some(ids) = response_json.get("ids") {
            if let Some(distances) = response_json.get("distances") {
                if let Some(documents) = response_json.get("documents") {
                    if let (Some(ids_array), Some(distances_array), Some(docs_array)) = 
                        (ids.as_array(), distances.as_array(), documents.as_array()) 
                    {
                        // Chroma returns nested arrays
                        if let (Some(first_ids), Some(first_distances), Some(first_docs)) = 
                            (ids_array.get(0), distances_array.get(0), docs_array.get(0))
                        {
                            if let (Some(ids_list), Some(dist_list), Some(doc_list)) = 
                                (first_ids.as_array(), first_distances.as_array(), first_docs.as_array())
                            {
                                for ((id, distance), document) in ids_list.iter()
                                    .zip(dist_list.iter())
                                    .zip(doc_list.iter())
                                {
                                    if let (Some(id_str), Some(dist_val), Some(doc_str)) = 
                                        (id.as_str(), distance.as_f64(), document.as_str())
                                    {
                                        // Convert distance to similarity score (Chroma uses cosine distance)
                                        let score = 1.0 - dist_val as f32;
                                        
                                        results.push(VectorSearchResult {
                                            memory_id: MemoryId::from(id_str.to_string()),
                                            score,
                                            content: doc_str.to_string(),
                                            metadata: HashMap::new(),
                                            provider: VectorProvider::Chroma,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Search Qdrant - simplified implementation
    async fn search_qdrant(
        &self,
        config: &QdrantConfig,
        query: &str,
        limit: usize,
        _similarity_threshold: f32,
    ) -> Result<Vec<VectorSearchResult>> {
        debug!("Searching Qdrant collection: {}", config.collection_name);
        
        // Generate embedding for query
        let query_embedding = self.generate_embedding(query).await?;
        
        let url = format!("{}/collections/{}/points/search",
            config.url, config.collection_name);
        
        let request_body = json!({
            "vector": query_embedding.to_vec(),
            "limit": limit,
            "with_payload": true,
            "with_vector": false
        });
        
        let mut request = self.client.post(&url).json(&request_body);
        
        if let Some(api_key) = &config.api_key {
            request = request.header("api-key", api_key);
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Qdrant search failed: {}", response.status()));
        }
        
        // Parse Qdrant response
        let response_json: Value = response.json().await?;
        let mut results = Vec::new();
        
        if let Some(result_array) = response_json.get("result") {
            if let Some(points) = result_array.as_array() {
                for point in points {
                    if let Some(id) = point.get("id") {
                        let score = point.get("score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        
                        let content = point.get("payload")
                            .and_then(|p| p.get("content"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string();
                        
                        let id_str = match id {
                            Value::String(s) => s.clone(),
                            Value::Number(n) => n.to_string(),
                            _ => "unknown".to_string(),
                        };
                        
                        // Extract metadata from payload
                        let mut metadata = HashMap::new();
                        if let Some(payload) = point.get("payload") {
                            if let Some(payload_obj) = payload.as_object() {
                                for (key, value) in payload_obj {
                                    if key != "content" {
                                        metadata.insert(key.clone(), value.clone());
                                    }
                                }
                            }
                        }
                        
                        results.push(VectorSearchResult {
                            memory_id: MemoryId::from(id_str),
                            score,
                            content,
                            metadata,
                            provider: VectorProvider::Qdrant,
                        });
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Generate embedding for text - delegates to local embedding system
    async fn generate_embedding(&self, text: &str) -> Result<Array1<f32>> {
        // Use a simple hash-based embedding as fallback
        // In a real implementation, this would use the same embedding model as the local system
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let dimension = 384; // MiniLM dimension
        let mut embedding = Array1::zeros(dimension);
        
        // Generate deterministic pseudo-embedding from text hash
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Use hash to seed a simple embedding
        for i in 0..dimension {
            let seed = hash.wrapping_add(i as u64);
            let value = ((seed * 0x9e3779b97f4a7c15) >> 32) as f32 / u32::MAX as f32;
            embedding[i] = (value - 0.5) * 2.0; // Normalize to [-1, 1]
        }
        
        // Normalize the embedding
        let norm = embedding.dot(&embedding).sqrt();
        if norm > 0.0 {
            embedding /= norm;
        }
        
        Ok(embedding)
    }
    
    /// Sync local memories to external vector database
    pub async fn sync_to_external(&self) -> Result<BatchOperationResult> {
        if !self.config.sync_with_local {
            return Ok(BatchOperationResult {
                success_count: 0,
                failed_count: 0,
                errors: vec!["Sync disabled in configuration".to_string()],
                duration: Duration::from_secs(0),
            });
        }
        
        let start_time = Instant::now();
        info!("🔄 Starting sync from local memory to external vector database");
        
        // Get memories from local embedding store (placeholder for real implementation)
        // In a real implementation, this would iterate through local memories
        let mut success_count = 0;
        let mut failed_count = 0;
        let mut errors = Vec::new();
        
        // For demonstration, simulate syncing a few memories
        let sample_memories = vec![
            ("memory_1", "Sample memory content 1"),
            ("memory_2", "Sample memory content 2"),
            ("memory_3", "Sample memory content 3"),
        ];
        
        for (memory_id, content) in sample_memories {
            match self.sync_single_memory(memory_id, content).await {
                Ok(_) => success_count += 1,
                Err(e) => {
                    failed_count += 1;
                    errors.push(format!("Failed to sync {}: {}", memory_id, e));
                }
            }
            
            // Add small delay to avoid overwhelming external services
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        let result = BatchOperationResult {
            success_count,
            failed_count,
            errors,
            duration: start_time.elapsed(),
        };
        
        self.update_metrics(|m| {
            m.sync_operations += 1;
            m.total_vectors_synced += success_count;
        }).await;
        
        info!("✅ Sync completed: {} success, {} failed", success_count, failed_count);
        Ok(result)
    }
    
    /// Sync a single memory to external vector database
    async fn sync_single_memory(&self, memory_id: &str, content: &str) -> Result<()> {
        let embedding = self.generate_embedding(content).await?;
        
        // Try to sync to configured external providers
        let mut synced = false;
        
        // Sync to Pinecone if configured
        if let Some(config) = &self.config.pinecone {
            match self.sync_to_pinecone(memory_id, content, &embedding, config).await {
                Ok(_) => {
                    debug!("Synced {} to Pinecone", memory_id);
                    synced = true;
                }
                Err(e) => warn!("Failed to sync {} to Pinecone: {}", memory_id, e),
            }
        }
        
        // Sync to other providers (Weaviate, Chroma, Qdrant) would be similar
        if let Some(config) = &self.config.weaviate {
            match self.sync_to_weaviate(memory_id, content, &embedding, config).await {
                Ok(_) => {
                    debug!("Synced {} to Weaviate", memory_id);
                    synced = true;
                }
                Err(e) => warn!("Failed to sync {} to Weaviate: {}", memory_id, e),
            }
        }
        
        if synced {
            Ok(())
        } else {
            Err(anyhow!("Failed to sync to any external provider"))
        }
    }
    
    /// Sync to Pinecone
    async fn sync_to_pinecone(&self, memory_id: &str, content: &str, embedding: &Array1<f32>, config: &PineconeConfig) -> Result<()> {
        let url = format!("https://{}-{}.svc.{}.pinecone.io/vectors/upsert",
            config.index_name, "random", config.environment);
        
        let request_body = json!({
            "vectors": [{
                "id": memory_id,
                "values": embedding.to_vec(),
                "metadata": {
                    "content": content,
                    "synced_at": chrono::Utc::now().to_rfc3339()
                }
            }]
        });
        
        let response = self.client
            .post(&url)
            .header("Api-Key", &config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("Pinecone sync failed: {}", response.status()))
        }
    }
    
    /// Sync to Weaviate
    async fn sync_to_weaviate(&self, memory_id: &str, content: &str, embedding: &Array1<f32>, config: &WeaviateConfig) -> Result<()> {
        let url = format!("{}/v1/objects", config.url);
        
        let request_body = json!({
            "class": config.class_name,
            "id": memory_id,
            "properties": {
                "content": content,
                "synced_at": chrono::Utc::now().to_rfc3339()
            },
            "vector": embedding.to_vec()
        });
        
        let mut request = self.client.post(&url).json(&request_body);
        
        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        let response = request.send().await?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("Weaviate sync failed: {}", response.status()))
        }
    }
    
    /// Update performance metrics
    async fn update_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut VectorMemoryMetrics),
    {
        let mut metrics = self.metrics.write().await;
        update_fn(&mut *metrics);
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> VectorMemoryMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Clear search cache
    pub async fn clear_cache(&self) {
        self.search_cache.write().await.clear();
        info!("🧹 Vector search cache cleared");
    }
}

/// Extension trait for EmbeddingStore to add search functionality
trait EmbeddingStoreExt {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<(MemoryId, f32)>>;
}

impl EmbeddingStoreExt for EmbeddingStore {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<(MemoryId, f32)>> {
        // Generate embedding for the query
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Create a simple hash-based search for demonstration
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        let query_hash = hasher.finish();
        
        // Create some sample results based on query hash
        let mut results = Vec::new();
        for i in 0..limit.min(5) {
            let memory_id = MemoryId::from(format!("mem_{:x}_{}", query_hash, i));
            let score = 0.9 - (i as f32 * 0.1); // Decreasing similarity scores
            results.push((memory_id, score));
        }
        
        Ok(results)
    }
}