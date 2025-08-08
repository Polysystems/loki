//! Memory Bridge - Connects Memory tab storage to Chat tab context retrieval
//! 
//! This bridge enables the Chat tab to query and retrieve relevant context
//! from the Memory tab's knowledge graph, vector store, and story database.

use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use anyhow::Result;

use crate::tui::event_bus::{SystemEvent, TabId};
use crate::memory::MemoryItem;

/// Retrieval result from memory system
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub metadata: Value,
}

/// Bridge for memory-related cross-tab communication
pub struct MemoryBridge {
    event_bridge: Arc<super::EventBridge>,
    context_sync: Arc<RwLock<super::ContextSync>>,
    
    /// Cache of recent retrievals for performance
    retrieval_cache: Arc<RwLock<Vec<CachedRetrieval>>>,
    
    /// Active context window
    active_context: Arc<RwLock<Vec<ContextItem>>>,
    
    /// Reference to actual memory system if available
    memory_system: Arc<RwLock<Option<Arc<crate::memory::CognitiveMemory>>>>,
}

/// Cached retrieval result
#[derive(Debug, Clone)]
pub struct CachedRetrieval {
    pub query: String,
    pub results: Vec<RetrievalResult>,
    pub timestamp: std::time::Instant,
    pub source_tab: TabId,
}

/// Context item for chat
#[derive(Debug, Clone)]
pub struct ContextItem {
    pub id: String,
    pub content: String,
    pub relevance_score: f32,
    pub source: ContextSource,
    pub metadata: Value,
}

/// Source of context
#[derive(Debug, Clone)]
pub enum ContextSource {
    KnowledgeGraph,
    VectorStore,
    StoryMemory,
    ConversationHistory,
    CognitiveInsight,
}

impl MemoryBridge {
    /// Create a new memory bridge
    pub fn new(
        event_bridge: Arc<super::EventBridge>,
        context_sync: Arc<RwLock<super::ContextSync>>,
    ) -> Self {
        Self {
            event_bridge,
            context_sync,
            retrieval_cache: Arc::new(RwLock::new(Vec::new())),
            active_context: Arc::new(RwLock::new(Vec::new())),
            memory_system: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Set the memory system for actual storage/retrieval
    pub async fn set_memory_system(&self, memory: Arc<crate::memory::CognitiveMemory>) {
        let mut mem_sys = self.memory_system.write().await;
        *mem_sys = Some(memory);
        tracing::info!("Memory system connected to bridge");
    }
    
    /// Initialize the memory bridge
    pub async fn initialize(&self) -> Result<()> {
        tracing::debug!("Initializing memory bridge");
        
        // Subscribe to memory-related events
        self.subscribe_to_events().await?;
        
        // Initialize context window
        self.initialize_context().await?;
        
        Ok(())
    }
    
    /// Subscribe to memory-related events
    async fn subscribe_to_events(&self) -> Result<()> {
        let cache = self.retrieval_cache.clone();
        let context = self.active_context.clone();
        
        // Subscribe to memory storage events
        self.event_bridge.subscribe_handler(
            "MemoryStored",
            move |event| {
                let context = context.clone();
                
                Box::pin(async move {
                    if let SystemEvent::MemoryStored { key, value_type, source } = event {
                        tracing::debug!("Memory stored: {} ({}) from {:?}", key, value_type, source);
                        
                        // Potentially add to active context if relevant
                        // This would involve checking relevance to current conversation
                    }
                    Ok(())
                })
            }
        ).await?;
        
        // Subscribe to context retrieval events
        self.event_bridge.subscribe_handler(
            "ContextRetrieved",
            move |event| {
                let cache = cache.clone();
                
                Box::pin(async move {
                    if let SystemEvent::ContextRetrieved { query, result_count, source } = event {
                        tracing::debug!("Context retrieved: {} results for '{}' from {:?}", 
                            result_count, query, source);
                        
                        // Cache the retrieval for reuse
                        // In a real implementation, this would store actual results
                    }
                    Ok(())
                })
            }
        ).await?;
        
        Ok(())
    }
    
    /// Initialize context window
    async fn initialize_context(&self) -> Result<()> {
        // Load initial context items
        // This would connect to the actual memory system
        tracing::debug!("Context window initialized");
        Ok(())
    }
    
    /// Retrieve context for a query from Chat tab
    pub async fn retrieve_context(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<ContextItem>> {
        tracing::debug!("Retrieving context for query: {}", query);
        
        // Check cache first
        if let Some(cached) = self.get_cached_retrieval(query).await {
            tracing::debug!("Using cached retrieval for query: {}", query);
            return Ok(self.convert_to_context_items(cached.results));
        }
        
        // Perform actual retrieval (would connect to memory system)
        let results = self.perform_retrieval(query, max_results).await?;
        
        // Cache the results
        self.cache_retrieval(query.to_string(), results.clone(), TabId::Chat).await;
        
        // Publish retrieval event
        self.event_bridge.publish(SystemEvent::ContextRetrieved {
            query: query.to_string(),
            result_count: results.len(),
            source: TabId::Chat,
        }).await?;
        
        Ok(results)
    }
    
    /// Get cached retrieval if available and fresh
    async fn get_cached_retrieval(&self, query: &str) -> Option<CachedRetrieval> {
        let cache = self.retrieval_cache.read().await;
        
        cache.iter()
            .find(|r| r.query == query && r.timestamp.elapsed().as_secs() < 60)
            .cloned()
    }
    
    /// Perform actual memory retrieval
    async fn perform_retrieval(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<ContextItem>> {
        let memory_system = self.memory_system.read().await;
        
        if let Some(ref memory) = *memory_system {
            // Real retrieval from memory system
            match memory.retrieve_similar(query, max_results).await {
                Ok(results) => {
                    // Convert memory results to context items
                    let context_items: Vec<ContextItem> = results.into_iter()
                        .map(|item| ContextItem {
                            id: item.id.to_string(),
                            content: item.content.clone(),
                            relevance_score: item.relevance_score,
                            source: ContextSource::KnowledgeGraph,
                            metadata: serde_json::json!({
                                "source": item.metadata.source,
                                "tags": item.metadata.tags,
                                "importance": item.metadata.importance,
                                "category": item.metadata.category,
                            }),
                        })
                        .collect();
                    
                    Ok(context_items)
                }
                Err(e) => {
                    tracing::error!("Memory retrieval failed: {}", e);
                    // Return empty results on error
                    Ok(Vec::new())
                }
            }
        } else {
            // Fallback when no memory system is connected
            tracing::debug!("No memory system connected, returning simulated results");
            
            let mock_results = vec![
                ContextItem {
                    id: uuid::Uuid::new_v4().to_string(),
                    content: format!("Simulated context for: {}", query),
                    relevance_score: 0.75,
                    source: ContextSource::KnowledgeGraph,
                    metadata: serde_json::json!({
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "simulated": true,
                    }),
                },
            ];
            
            Ok(mock_results.into_iter().take(max_results).collect())
        }
    }
    
    /// Convert retrieval results to context items
    fn convert_to_context_items(&self, results: Vec<RetrievalResult>) -> Vec<ContextItem> {
        results.into_iter()
            .map(|r| ContextItem {
                id: r.id,
                content: r.content,
                relevance_score: r.score,
                source: ContextSource::KnowledgeGraph,
                metadata: r.metadata,
            })
            .collect()
    }
    
    /// Cache retrieval results
    async fn cache_retrieval(
        &self,
        query: String,
        results: Vec<ContextItem>,
        source: TabId,
    ) {
        let mut cache = self.retrieval_cache.write().await;
        
        // Convert context items to retrieval results for caching
        let retrieval_results: Vec<RetrievalResult> = results.iter()
            .map(|item| RetrievalResult {
                id: item.id.clone(),
                content: item.content.clone(),
                score: item.relevance_score,
                metadata: item.metadata.clone(),
            })
            .collect();
        
        cache.push(CachedRetrieval {
            query,
            results: retrieval_results,
            timestamp: std::time::Instant::now(),
            source_tab: source,
        });
        
        // Keep only last 50 cached retrievals
        if cache.len() > 50 {
            let drain_count = cache.len() - 50;
            cache.drain(0..drain_count);
        }
    }
    
    /// Get active context window
    pub async fn get_active_context(&self) -> Vec<ContextItem> {
        let context = self.active_context.read().await;
        context.clone()
    }
    
    /// Update active context with new items
    pub async fn update_active_context(&self, items: Vec<ContextItem>) -> Result<()> {
        let mut context = self.active_context.write().await;
        
        // Add new items
        for item in items {
            // Check if item already exists
            if !context.iter().any(|c| c.id == item.id) {
                context.push(item);
            }
        }
        
        // Sort by relevance and keep top items
        context.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        context.truncate(20); // Keep top 20 context items
        
        // Sync with context sync service
        let context_sync = self.context_sync.write().await;
        context_sync.update_context(TabId::Chat, context.clone()).await?;
        
        Ok(())
    }
    
    /// Store new memory from Chat tab
    pub async fn store_from_chat(
        &self,
        key: String,
        content: String,
        metadata: Value,
    ) -> Result<()> {
        tracing::debug!("Storing memory from chat: {}", key);
        
        let memory_system = self.memory_system.read().await;
        
        if let Some(ref memory) = *memory_system {
            // Real storage in memory system
            let now = chrono::Utc::now();
            let memory_metadata = crate::memory::MemoryMetadata {
                source: "chat_tab".to_string(),
                tags: vec!["chat".to_string()],
                importance: 0.5,
                associations: vec![],
                context: Some(serde_json::to_string(&metadata).unwrap_or_default()),
                created_at: now,
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                category: "conversation".to_string(),
                timestamp: now,
                expiration: None,
            };
            
            match memory.store(content.clone(), vec![key.clone()], memory_metadata).await {
                Ok(_) => {
                    tracing::info!("✅ Memory stored successfully: {}", key);
                }
                Err(e) => {
                    tracing::error!("Failed to store memory: {}", e);
                    return Err(e.into());
                }
            }
        } else {
            tracing::warn!("No memory system connected, memory not persisted");
        }
        
        // Always publish the event for other tabs to know
        self.event_bridge.publish(SystemEvent::MemoryStored {
            key,
            value_type: "chat_memory".to_string(),
            source: TabId::Chat,
        }).await?;
        
        Ok(())
    }
    
    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let cache = self.retrieval_cache.read().await;
        let context = self.active_context.read().await;
        
        MemoryStats {
            cached_retrievals: cache.len(),
            active_context_items: context.len(),
            cache_hit_rate: self.calculate_cache_hit_rate(&cache),
        }
    }
    
    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self, cache: &[CachedRetrieval]) -> f32 {
        // This would track actual hits vs misses
        // For now, return a mock value
        0.75
    }
}

/// Memory system statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub cached_retrievals: usize,
    pub active_context_items: usize,
    pub cache_hit_rate: f32,
}