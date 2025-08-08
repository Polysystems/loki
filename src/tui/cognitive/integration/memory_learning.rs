//! Cognitive Memory Learning Integration
//!
//! This module enables Loki to learn from cognitive insights and interactions,
//! updating its memory and understanding based on new experiences.

use std::sync::Arc;
use anyhow::{Result};
use tokio::sync::RwLock;
use tracing::{info, debug};
use serde::{Serialize, Deserialize};
use chrono::{Utc};

use crate::memory::{CognitiveMemory, MemoryMetadata};
use crate::cognitive::{
    CognitiveSystem,
};
use crate::tui::{
    chat::integrations::cognitive::CognitiveResponse,
    cognitive::persistence::state::{PersistedInsight, LearningOutcome},
    cognitive::integration::background_processor::{BackgroundInsight, Connection, Learning},
};

/// Memory learning system
pub struct CognitiveMemoryLearning {
    /// Memory system reference
    memory: Arc<CognitiveMemory>,
    
    /// Cognitive system reference
    cognitive_system: Arc<CognitiveSystem>,
    
    /// Learning configuration
    config: LearningConfig,
    
    /// Learning statistics
    stats: Arc<RwLock<LearningStatistics>>,
    
    /// Knowledge graph updater
    knowledge_updater: Arc<KnowledgeGraphUpdater>,
}

/// Learning configuration
#[derive(Debug, Clone)]
pub struct LearningConfig {
    /// Minimum confidence for memory storage
    pub confidence_threshold: f64,
    
    /// Minimum relevance for insight storage
    pub relevance_threshold: f64,
    
    /// Enable automatic concept extraction
    pub extract_concepts: bool,
    
    /// Enable relationship discovery
    pub discover_relationships: bool,
    
    /// Maximum memories per concept
    pub max_memories_per_concept: usize,
    
    /// Memory decay rate (0.0 = no decay, 1.0 = full decay)
    pub memory_decay_rate: f64,
    
    /// Reinforcement learning rate
    pub learning_rate: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            relevance_threshold: 0.6,
            extract_concepts: true,
            discover_relationships: true,
            max_memories_per_concept: 100,
            memory_decay_rate: 0.1,
            learning_rate: 0.3,
        }
    }
}

/// Learning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningStatistics {
    pub total_insights_processed: u64,
    pub insights_stored: u64,
    pub concepts_learned: u64,
    pub relationships_discovered: u64,
    pub memory_reinforcements: u64,
    pub knowledge_graph_updates: u64,
}

/// Knowledge graph updater
pub struct KnowledgeGraphUpdater {
    memory: Arc<CognitiveMemory>,
}

impl KnowledgeGraphUpdater {
    pub fn new(memory: Arc<CognitiveMemory>) -> Self {
        Self { memory }
    }
    
    /// Update knowledge graph with new connection
    pub async fn add_connection(
        &self,
        concept_a: &str,
        concept_b: &str,
        relationship: &str,
        strength: f64,
    ) -> Result<()> {
        let connection_data = serde_json::json!({
            "type": "knowledge_connection",
            "concept_a": concept_a,
            "concept_b": concept_b,
            "relationship": relationship,
            "strength": strength,
            "timestamp": Utc::now(),
        });
        
        self.memory.store(
            connection_data.to_string(),
            vec!["knowledge_graph".to_string(), "connections".to_string()],
            MemoryMetadata {
                source: "cognitive_learning".to_string(),
                tags: vec!["connection".to_string(), concept_a.to_string(), concept_b.to_string()],
                timestamp: Utc::now(),
                expiration: None,
                importance: strength as f32,
                associations: vec![],
                context: Some(format!("Connecting {} to {}", concept_a, concept_b)),
                created_at: Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                category: "knowledge_graph".to_string(),
            },
        ).await?;
        
        Ok(())
    }
    
    /// Strengthen existing connection
    pub async fn reinforce_connection(
        &self,
        concept_a: &str,
        concept_b: &str,
        reinforcement: f64,
    ) -> Result<()> {
        // Implementation would update existing connection strength
        Ok(())
    }
}

impl CognitiveMemoryLearning {
    /// Create new memory learning system
    pub fn new(
        memory: Arc<CognitiveMemory>,
        cognitive_system: Arc<CognitiveSystem>,
        config: LearningConfig,
    ) -> Self {
        let knowledge_updater = Arc::new(KnowledgeGraphUpdater::new(memory.clone()));
        
        Self {
            memory,
            cognitive_system,
            config,
            stats: Arc::new(RwLock::new(LearningStatistics::default())),
            knowledge_updater,
        }
    }
    
    /// Learn from cognitive response
    pub async fn learn_from_response(
        &self,
        response: &CognitiveResponse,
        user_input: &str,
        session_id: &str,
    ) -> Result<LearningResult> {
        info!("🧠 Learning from cognitive response");
        
        let mut result = LearningResult::default();
        let mut stats = self.stats.write().await;
        
        // Learn from insights
        for insight in &response.cognitive_insights {
            if let Ok(learned) = self.process_insight_text(insight, session_id).await {
                result.merge(learned);
            }
        }
        
        // Extract concepts from interaction
        if self.config.extract_concepts {
            result.merge(self.extract_and_learn_concepts(user_input, &response.content).await?);
        }
        
        // Update statistics
        stats.total_insights_processed += response.cognitive_insights.len() as u64;
        stats.insights_stored += result.insights_stored;
        stats.concepts_learned += result.concepts_learned;
        
        Ok(result)
    }
    
    /// Learn from background insights
    pub async fn learn_from_background(
        &self,
        insights: Vec<BackgroundInsight>,
        connections: Vec<Connection>,
        learnings: Vec<Learning>,
    ) -> Result<LearningResult> {
        info!("🌙 Learning from background processing");
        
        let mut result = LearningResult::default();
        let mut stats = self.stats.write().await;
        
        // Process background insights
        for insight in insights {
            if insight.confidence >= self.config.confidence_threshold
                && insight.relevance >= self.config.relevance_threshold {
                
                let memory_data = serde_json::json!({
                    "type": "background_insight",
                    "content": insight.content,
                    "category": insight.category,
                    "novelty": insight.novelty,
                    "relevance": insight.relevance,
                    "confidence": insight.confidence,
                    "evidence": insight.supporting_evidence,
                    "timestamp": Utc::now(),
                });
                
                self.memory.store(
                    memory_data.to_string(),
                    vec!["insights".to_string(), "background".to_string(), insight.category.clone()],
                    MemoryMetadata {
                        source: "background_processing".to_string(),
                        tags: vec!["insight".to_string(), "background".to_string()],
                        timestamp: Utc::now(),
                        expiration: None,
                        importance: insight.confidence as f32,
                        associations: vec![],
                        context: Some("Background cognitive processing".to_string()),
                        created_at: Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: insight.category.clone(),
                    },
                ).await?;
                
                result.insights_stored += 1;
            }
        }
        
        // Learn connections
        for connection in connections {
            if connection.strength >= self.config.confidence_threshold {
                self.knowledge_updater.add_connection(
                    &connection.concept_a,
                    &connection.concept_b,
                    &connection.relationship,
                    connection.strength,
                ).await?;
                
                result.relationships_discovered += 1;
            }
        }
        
        // Process learnings
        for learning in learnings {
            if learning.understanding_after > learning.understanding_before {
                let improvement = learning.understanding_after - learning.understanding_before;
                
                let learning_data = serde_json::json!({
                    "type": "cognitive_learning",
                    "topic": learning.topic,
                    "improvement": improvement,
                    "realization": learning.key_realization,
                    "applications": learning.application_ideas,
                    "timestamp": Utc::now(),
                });
                
                self.memory.store(
                    learning_data.to_string(),
                    vec!["learnings".to_string(), "improvements".to_string()],
                    MemoryMetadata {
                        source: "cognitive_integration".to_string(),
                        tags: vec!["learning".to_string(), learning.topic.clone()],
                        timestamp: Utc::now(),
                        expiration: None,
                        importance: improvement as f32,
                        associations: vec![],
                        context: Some(learning.topic.clone()),
                        created_at: Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "learning".to_string(),
                    },
                ).await?;
                
                result.concepts_learned += 1;
            }
        }
        
        // Update statistics
        stats.insights_stored += result.insights_stored;
        stats.concepts_learned += result.concepts_learned;
        stats.relationships_discovered += result.relationships_discovered;
        stats.knowledge_graph_updates += result.relationships_discovered;
        
        Ok(result)
    }
    
    /// Learn from persisted insights (session restoration)
    pub async fn learn_from_persisted(
        &self,
        insights: Vec<PersistedInsight>,
        learnings: Vec<LearningOutcome>,
        session_id: &str,
    ) -> Result<()> {
        debug!("Restoring learned knowledge from session: {}", session_id);
        
        // Restore insights with decay
        for insight in insights {
            let age = Utc::now() - insight.timestamp;
            let decay_factor = 1.0 - (age.num_hours() as f64 * self.config.memory_decay_rate / 24.0);
            
            if decay_factor > 0.1 {
                let adjusted_relevance = insight.relevance * decay_factor;
                
                if adjusted_relevance >= self.config.relevance_threshold {
                    let memory_data = serde_json::json!({
                        "type": "restored_insight",
                        "content": insight.content,
                        "category": insight.category,
                        "original_relevance": insight.relevance,
                        "adjusted_relevance": adjusted_relevance,
                        "source_modalities": insight.source_modalities,
                        "restoration_session": session_id,
                        "original_timestamp": insight.timestamp,
                        "restored_timestamp": Utc::now(),
                    });
                    
                    self.memory.store(
                        memory_data.to_string(),
                        vec!["insights".to_string(), "restored".to_string(), insight.category.clone()],
                        MemoryMetadata {
                            source: "session_restoration".to_string(),
                            tags: vec!["restored".to_string(), "insight".to_string()],
                            timestamp: insight.timestamp,
                            expiration: None,
                            importance: adjusted_relevance as f32,
                            associations: vec![],
                            context: Some(insight.content.clone()),
                            created_at: Utc::now(),
                            accessed_count: 0,
                            last_accessed: None,
                            version: 1,
                            category: insight.category.clone(),
                        },
                    ).await?;
                }
            }
        }
        
        // Restore learnings
        for learning in learnings {
            if learning.confidence >= self.config.confidence_threshold && learning.applied {
                let learning_data = serde_json::json!({
                    "type": "restored_learning",
                    "learning_type": learning.learning_type,
                    "content": learning.content,
                    "confidence": learning.confidence,
                    "applied": learning.applied,
                    "restoration_session": session_id,
                    "timestamp": learning.timestamp,
                });
                
                self.memory.store(
                    learning_data.to_string(),
                    vec!["learnings".to_string(), "restored".to_string()],
                    MemoryMetadata {
                        source: "session_restoration".to_string(),
                        tags: vec!["restored".to_string(), "learning".to_string()],
                        timestamp: Utc::now(),
                        expiration: None,
                        importance: learning.confidence as f32,
                        associations: vec![],
                        context: Some(learning.content.clone()),
                        created_at: Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: format!("{:?}", learning.learning_type),
                    },
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Reinforce existing memory
    pub async fn reinforce_memory(
        &self,
        concept: &str,
        context: &str,
        reinforcement_strength: f64,
    ) -> Result<()> {
        let mut stats = self.stats.write().await;
        
        // Search for related memories
        let query = format!("concept:{}", concept);
        
        if let Ok(memories) = self.memory.search_memories(&query, 10, None).await {
            for memory in memories {
                // Reinforce memory relevance
                // Reinforce memory by updating its relevance
                let new_relevance = (memory.relevance_score + (reinforcement_strength * self.config.learning_rate) as f32)
                    .min(1.0);
                
                // Store reinforcement information in metadata
                let mut updated_metadata = memory.metadata.clone();
                updated_metadata.context = Some(format!(
                    "Reinforced at {} with context: {}", 
                    Utc::now().format("%Y-%m-%d %H:%M:%S"),
                    context
                ));
                
                // TODO: Memory update not available - need to re-store with updated metadata
                // self.memory.update_memory_metadata(&memory.id, updated_metadata).await?;
                
                stats.memory_reinforcements += 1;
            }
        }
        
        Ok(())
    }
    
    
    /// Process insight text
    async fn process_insight_text(
        &self,
        insight: &str,
        session_id: &str,
    ) -> Result<LearningResult> {
        let mut result = LearningResult::default();
        
        let insight_data = serde_json::json!({
            "type": "text_insight",
            "content": insight,
            "session_id": session_id,
            "timestamp": Utc::now(),
        });
        
        self.memory.store(
            insight_data.to_string(),
            vec!["insights".to_string(), "text".to_string()],
            MemoryMetadata {
                source: "cognitive_insight".to_string(),
                tags: vec!["insight".to_string(), "text".to_string()],
                timestamp: Utc::now(),
                expiration: None,
                importance: 0.7,
                associations: vec![],
                context: Some(session_id.to_string()),
                created_at: Utc::now(),
                accessed_count: 0,
                last_accessed: None,
                version: 1,
                category: "text_insight".to_string(),
            },
        ).await?;
        
        result.insights_stored += 1;
        
        Ok(result)
    }
    
    /// Extract and learn concepts
    async fn extract_and_learn_concepts(
        &self,
        user_input: &str,
        ai_response: &str,
    ) -> Result<LearningResult> {
        let mut result = LearningResult::default();
        
        // Simple concept extraction (would be more sophisticated)
        let concepts = self.extract_concepts(user_input)
            .into_iter()
            .chain(self.extract_concepts(ai_response))
            .collect::<Vec<_>>();
        
        for concept in concepts {
            let concept_data = serde_json::json!({
                "type": "extracted_concept",
                "concept": concept,
                "context": format!("{} -> {}", 
                    &user_input[..user_input.len().min(100)],
                    &ai_response[..ai_response.len().min(100)]
                ),
                "timestamp": Utc::now(),
            });
            
            self.memory.store(
                concept_data.to_string(),
                vec!["concepts".to_string(), "extracted".to_string()],
                MemoryMetadata {
                    source: "concept_extraction".to_string(),
                    tags: vec!["concept".to_string(), concept.clone()],
                    timestamp: Utc::now(),
                    expiration: None,
                    importance: 0.6,
                    associations: vec![],
                    context: Some("Extracted from cognitive response".to_string()),
                    created_at: Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "concept".to_string(),
                },
            ).await?;
            
            result.concepts_learned += 1;
        }
        
        Ok(result)
    }
    
    /// Simple concept extraction
    fn extract_concepts(&self, text: &str) -> Vec<String> {
        // Very simple implementation - would use NLP in production
        text.split_whitespace()
            .filter(|word| word.len() > 5 && word.chars().all(|c| c.is_alphabetic()))
            .map(|w| w.to_lowercase())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(5)
            .collect()
    }
    
    /// Get learning statistics
    pub async fn get_statistics(&self) -> LearningStatistics {
        self.stats.read().await.clone()
    }
    
    /// Store an insight directly (public method for external use)
    pub async fn store_insight(&self, insight_data: serde_json::Value) -> Result<()> {
        // Extract key information from the insight
        let content = insight_data.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("No content");
        
        let insight_type = insight_data.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("general");
        
        let confidence = insight_data.get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8);
        
        // Check if insight meets threshold
        if confidence < self.config.confidence_threshold {
            debug!("Insight below confidence threshold: {}", confidence);
            return Ok(());
        }
        
        // Create memory metadata
        let metadata = MemoryMetadata {
            source: "story_memory_integration".to_string(),
            context: Some(insight_type.to_string()),
            tags: vec!["insight".to_string(), insight_type.to_string()],
            importance: (confidence * 100.0) as f32,
            associations: vec![],
            created_at: Utc::now(),
            accessed_count: 0,
            last_accessed: None,
            version: 1,
            category: "insight".to_string(),
            timestamp: Utc::now(),
            expiration: None,
        };
        
        // Store in memory
        self.memory.store(
            content.to_string(),
            vec![], // empty context
            metadata,
        ).await?;
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.insights_stored += 1;
        
        info!("✅ Stored insight: {}", content);
        
        Ok(())
    }
}

/// Learning result
#[derive(Debug, Default)]
pub struct LearningResult {
    pub insights_stored: u64,
    pub concepts_learned: u64,
    pub relationships_discovered: u64,
    pub memories_reinforced: u64,
}

impl LearningResult {
    fn merge(&mut self, other: LearningResult) {
        self.insights_stored += other.insights_stored;
        self.concepts_learned += other.concepts_learned;
        self.relationships_discovered += other.relationships_discovered;
        self.memories_reinforced += other.memories_reinforced;
    }
}

/// Memory decay scheduler
pub struct MemoryDecayScheduler {
    memory_learning: Arc<CognitiveMemoryLearning>,
}

impl MemoryDecayScheduler {
    pub fn new(memory_learning: Arc<CognitiveMemoryLearning>) -> Self {
        Self { memory_learning }
    }
    
    /// Run memory decay process
    pub async fn run_decay_cycle(&self) -> Result<()> {
        info!("🌙 Running memory decay cycle");
        
        // Implementation would:
        // 1. Query old memories
        // 2. Apply decay based on age and access patterns
        // 3. Remove memories below threshold
        // 4. Update statistics
        
        Ok(())
    }
}