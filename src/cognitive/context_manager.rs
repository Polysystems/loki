//! Context Manager for Extended Consciousness
//!
//! This module implements a sophisticated context management system that
//! maintains a 128k token window with compression, prioritization, and
//! checkpointing.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use lz4_flex::decompress_size_prepended;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, info, warn};
use uuid;

use crate::cognitive::{
    orchestrator::CognitiveEvent,
    Goal,
    GoalId,
    Thought,
    ThoughtId,
};
use crate::cognitive::attention_manager::AttentionManager;
use crate::cognitive::decision_engine::{Decision, DecisionId};
use crate::cognitive::emotional_core::EmotionalBlend;
use crate::memory::{CognitiveMemory, MemoryMetadata};

/// Token representation for context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextToken {
    pub content: String,
    pub token_type: TokenType,
    pub importance: f32, // 0.0 to 1.0
    pub timestamp: DateTime<Utc>,
    pub metadata: TokenMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TokenType {
    Thought(ThoughtId),
    Decision(DecisionId),
    Goal(GoalId),
    Event(String),
    Narrative(String),
    Memory(String),
    Social(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenMetadata {
    pub source: String,
    pub emotional_valence: f32,
    pub attention_weight: f32,
    pub associations: Vec<String>,
    pub compressed: bool,
}

/// Context segment for efficient management
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextSegment {
    pub id: String,
    pub tokens: Vec<ContextToken>,
    pub total_size: usize,     // In tokens
    pub importance_score: f32, // Average importance
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    pub compressed: bool,
    pub compression_ratio: f32,
}

/// Compression strategy for context
#[derive(Clone, Debug)]
pub enum CompressionStrategy {
    /// Remove low-importance tokens
    ImportanceThreshold(f32),

    /// Summarize similar tokens
    Summarization { similarity_threshold: f32, max_summary_length: usize },

    /// Temporal decay - older tokens compressed more
    TemporalDecay { half_life: Duration, min_importance: f32 },

    /// Semantic clustering and representative selection
    SemanticClustering { num_clusters: usize, representatives_per_cluster: usize },
}

/// Priority system for context retention
#[derive(Clone, Debug)]
pub struct PrioritySystem {
    /// Weight factors for different aspects
    weights: PriorityWeights,

    /// Minimum retention thresholds
    thresholds: RetentionThresholds,

    /// Special retention rules
    rules: Vec<RetentionRule>,
}

#[derive(Clone, Debug)]
pub struct PriorityWeights {
    pub importance: f32,
    pub recency: f32,
    pub emotional_significance: f32,
    pub attention_correlation: f32,
    pub goal_relevance: f32,
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            importance: 0.3,
            recency: 0.2,
            emotional_significance: 0.2,
            attention_correlation: 0.2,
            goal_relevance: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RetentionThresholds {
    pub min_importance: f32,
    pub max_age: Duration,
    pub min_emotional_significance: f32,
}

impl Default for RetentionThresholds {
    fn default() -> Self {
        Self {
            min_importance: 0.2,
            max_age: Duration::from_secs(86400), // 24 hours
            min_emotional_significance: -1.0,    // Keep all
        }
    }
}

#[derive(Clone, Debug)]
pub struct RetentionRule {
    pub name: String,
    pub condition: RetentionCondition,
    pub priority_boost: f32,
}

#[derive(Clone, Debug)]
pub enum RetentionCondition {
    /// Always retain tokens of this type
    TokenType(TokenType),

    /// Retain if source matches
    Source(String),

    /// Retain if associated with active goal
    ActiveGoal,

    /// Retain if part of decision chain
    DecisionChain,

    /// Custom condition
    Custom(String),
}

/// Checkpoint for state persistence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub segments: Vec<ContextSegment>,
    pub metadata: CheckpointMetadata,
    pub compressed_state: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub total_tokens: usize,
    pub compressed_tokens: usize,
    pub compression_ratio: f32,
    pub emotional_summary: EmotionalBlend,
    pub active_goals: Vec<GoalId>,
    pub recent_decisions: Vec<DecisionId>,
    pub checkpoint_reason: String,
}

/// Configuration for context manager
#[derive(Clone, Debug)]
pub struct ContextConfig {
    /// Maximum token window size
    pub max_tokens: usize,

    /// Target token count after compression
    pub target_tokens: usize,

    /// Segment size for chunking
    pub segment_size: usize,

    /// Compression trigger threshold
    pub compression_threshold: f32,

    /// Checkpoint interval
    pub checkpoint_interval: Duration,

    /// Maximum checkpoint storage
    pub max_checkpoints: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128_000,
            target_tokens: 64_000,
            segment_size: 1_000,
            compression_threshold: 0.8, // Compress at 80% full
            checkpoint_interval: Duration::from_secs(3600), // 1 hour
            max_checkpoints: 24,
        }
    }
}

#[derive(Debug)]
/// Main context manager
pub struct ContextManager {
    /// Active context segments
    segments: Arc<RwLock<VecDeque<ContextSegment>>>,

    /// Token buffer for incoming context
    token_buffer: Arc<RwLock<Vec<ContextToken>>>,

    /// Priority system
    priority_system: Arc<PrioritySystem>,

    /// Compression strategies
    compression_strategies: Arc<RwLock<Vec<CompressionStrategy>>>,

    /// Checkpoints
    checkpoints: Arc<RwLock<BTreeMap<DateTime<Utc>, ContextCheckpoint>>>,

    /// Attention manager reference
    attention_manager: Option<Arc<AttentionManager>>,

    /// Memory system reference
    memory: Arc<CognitiveMemory>,

    /// Configuration
    config: ContextConfig,

    /// Update channel
    update_tx: mpsc::Sender<ContextUpdate>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<ContextStats>>,
}

#[derive(Clone, Debug)]
pub enum ContextUpdate {
    TokenAdded(ContextToken),
    SegmentCreated(String),         // Segment ID
    SegmentCompressed(String, f32), // Segment ID, compression ratio
    CheckpointCreated(String),      // Checkpoint ID
    ContextRestored(String),        // Checkpoint ID
}

#[derive(Debug, Default, Clone)]
pub struct ContextStats {
    pub total_tokens: usize,
    pub compressed_tokens: usize,
    pub segments_created: u64,
    pub compressions_performed: u64,
    pub checkpoints_created: u64,
    pub avg_compression_ratio: f32,
    pub memory_usage_mb: f32,
}

impl ContextManager {
    pub async fn new(memory: Arc<CognitiveMemory>, config: ContextConfig) -> Result<Self> {
        info!("Initializing Context Manager with {}k token window", config.max_tokens / 1000);

        let (update_tx, _) = mpsc::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);

        let priority_system = Arc::new(PrioritySystem {
            weights: PriorityWeights::default(),
            thresholds: RetentionThresholds::default(),
            rules: Self::default_retention_rules(),
        });

        let compression_strategies = vec![
            CompressionStrategy::ImportanceThreshold(0.4),
            CompressionStrategy::TemporalDecay {
                half_life: Duration::from_secs(3600),
                min_importance: 0.2,
            },
            CompressionStrategy::Summarization {
                similarity_threshold: 0.8,
                max_summary_length: 100,
            },
        ];

        Ok(Self {
            segments: Arc::new(RwLock::new(VecDeque::new())),
            token_buffer: Arc::new(RwLock::new(Vec::with_capacity(config.segment_size))),
            priority_system,
            compression_strategies: Arc::new(RwLock::new(compression_strategies)),
            checkpoints: Arc::new(RwLock::new(BTreeMap::new())),
            attention_manager: None,
            memory,
            config,
            update_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(ContextStats::default())),
        })
    }

    /// Set attention manager reference
    pub fn set_attention_manager(&mut self, attention_manager: Arc<AttentionManager>) {
        self.attention_manager = Some(attention_manager);
    }

    /// Start the context manager
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting Context Manager");

        // Segment creation loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.segment_creation_loop().await;
            });
        }

        // Compression loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.compression_loop().await;
            });
        }

        // Checkpoint loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.checkpoint_loop().await;
            });
        }

        Ok(())
    }

    /// Add a thought to context
    pub async fn add_thought(&self, thought: &Thought) -> Result<()> {
        let token = ContextToken {
            content: thought.content.clone(),
            token_type: TokenType::Thought(thought.id.clone()),
            importance: thought.metadata.importance,
            timestamp: Utc::now(),
            metadata: TokenMetadata {
                source: thought.metadata.source.clone(),
                emotional_valence: thought.metadata.emotional_valence,
                attention_weight: 0.5, // Would calculate from attention manager
                associations: thought.metadata.tags.clone(),
                compressed: false,
            },
        };

        self.add_token(token).await
    }

    /// Add a decision to context
    pub async fn add_decision(&self, decision: &Decision) -> Result<()> {
        let token = ContextToken {
            content: format!(
                "Decision: {} (confidence: {:.2})",
                decision.context, decision.confidence
            ),
            token_type: TokenType::Decision(decision.id.clone()),
            importance: decision.confidence,
            timestamp: Utc::now(),
            metadata: TokenMetadata {
                source: "decision_engine".to_string(),
                emotional_valence: 0.0,
                attention_weight: 0.8,
                associations: decision.reasoning_chain.clone(),
                compressed: false,
            },
        };

        self.add_token(token).await
    }

    /// Add a goal to context
    pub async fn add_goal(&self, goal: &Goal) -> Result<()> {
        let token = ContextToken {
            content: format!("Goal: {} (priority: {:.2})", goal.name, goal.priority.to_f32()),
            token_type: TokenType::Goal(goal.id.clone()),
            importance: goal.priority.to_f32(),
            timestamp: Utc::now(),
            metadata: TokenMetadata {
                source: "goal_manager".to_string(),
                emotional_valence: goal.emotional_significance,
                attention_weight: goal.priority.to_f32(),
                associations: vec![format!("{:?}", goal.goal_type)],
                compressed: false,
            },
        };

        self.add_token(token).await
    }

    /// Add an event to context
    pub async fn add_event(&self, event: &CognitiveEvent) -> Result<()> {
        let (content, importance) = match event {
            CognitiveEvent::ThoughtGenerated(thought) => {
                (format!("Thought generated: {}", thought.content), 0.5)
            }
            CognitiveEvent::DecisionMade(decision) => {
                (format!("Decision made: {}", decision.context), 0.8)
            }
            CognitiveEvent::GoalCompleted(goal_id) => {
                (format!("Goal completed: {}", goal_id), 0.9)
            }
            CognitiveEvent::PatternDetected(pattern, confidence) => (
                format!("Pattern detected: {} (confidence: {:.2})", pattern, confidence),
                *confidence,
            ),
            _ => (format!("Event: {:?}", event), 0.4),
        };

        let token = ContextToken {
            content,
            token_type: TokenType::Event(format!("{:?}", event)),
            importance,
            timestamp: Utc::now(),
            metadata: TokenMetadata {
                source: "consciousness_orchestrator".to_string(),
                emotional_valence: 0.0,
                attention_weight: importance,
                associations: vec![],
                compressed: false,
            },
        };

        self.add_token(token).await
    }

    /// Add a narrative to context
    pub async fn add_narrative(&self, narrative: String, metadata: TokenMetadata) -> Result<()> {
        let token = ContextToken {
            content: narrative.clone(),
            token_type: TokenType::Narrative(narrative),
            importance: 0.6,
            timestamp: Utc::now(),
            metadata,
        };

        self.add_token(token).await
    }

    /// Core method to add any token
    async fn add_token(&self, token: ContextToken) -> Result<()> {
        let mut buffer = self.token_buffer.write().await;
        buffer.push(token.clone());

        // Check if we need to create a segment
        if buffer.len() >= self.config.segment_size {
            drop(buffer); // Release lock
            self.create_segment().await?;
        }

        // Update stats
        self.stats.write().await.total_tokens += 1;

        // Send update
        let _ = self.update_tx.send(ContextUpdate::TokenAdded(token)).await;

        // Check if compression needed
        let total_tokens = self.calculate_total_tokens().await;
        if total_tokens as f32 > self.config.max_tokens as f32 * self.config.compression_threshold {
            self.trigger_compression().await?;
        }

        Ok(())
    }

    /// Create a new segment from buffer
    async fn create_segment(&self) -> Result<()> {
        let mut buffer = self.token_buffer.write().await;
        if buffer.is_empty() {
            return Ok(());
        }

        let tokens: Vec<ContextToken> = buffer.drain(..).collect();
        let segment_id = uuid::Uuid::new_v4().to_string();

        let importance_sum: f32 = tokens.iter().map(|t| t.importance).sum();
        let importance_avg = importance_sum / tokens.len() as f32;

        let time_range = (tokens.first().unwrap().timestamp, tokens.last().unwrap().timestamp);

        let segment = ContextSegment {
            id: segment_id.clone(),
            total_size: tokens.len(),
            tokens,
            importance_score: importance_avg,
            time_range,
            compressed: false,
            compression_ratio: 1.0,
        };

        self.segments.write().await.push_back(segment);
        self.stats.write().await.segments_created += 1;

        let _ = self.update_tx.send(ContextUpdate::SegmentCreated(segment_id)).await;

        Ok(())
    }

    /// Calculate total tokens across all segments
    async fn calculate_total_tokens(&self) -> usize {
        let segments = self.segments.read().await;
        let buffer = self.token_buffer.read().await;

        let segment_tokens: usize = segments.iter().map(|s| s.total_size).sum();
        segment_tokens + buffer.len()
    }

    /// Trigger compression when needed
    async fn trigger_compression(&self) -> Result<()> {
        info!("Triggering context compression");

        let strategies = self.compression_strategies.read().await.clone();

        for strategy in strategies {
            if let Err(e) = self.apply_compression_strategy(&strategy).await {
                warn!("Compression strategy failed: {}", e);
            }
        }

        Ok(())
    }

    /// Apply a specific compression strategy
    async fn apply_compression_strategy(&self, strategy: &CompressionStrategy) -> Result<()> {
        match strategy {
            CompressionStrategy::ImportanceThreshold(threshold) => {
                self.compress_by_importance(*threshold).await
            }
            CompressionStrategy::TemporalDecay { half_life, min_importance } => {
                self.compress_by_temporal_decay(*half_life, *min_importance).await
            }
            CompressionStrategy::Summarization { similarity_threshold, max_summary_length } => {
                self.compress_by_summarization(*similarity_threshold, *max_summary_length).await
            }
            CompressionStrategy::SemanticClustering {
                num_clusters,
                representatives_per_cluster,
            } => self.compress_by_clustering(*num_clusters, *representatives_per_cluster).await,
        }
    }

    /// Compress by removing low-importance tokens
    async fn compress_by_importance(&self, threshold: f32) -> Result<()> {
        let mut segments = self.segments.write().await;
        let mut total_removed = 0;

        for segment in segments.iter_mut() {
            if !segment.compressed {
                let original_size = segment.tokens.len();
                segment.tokens.retain(|token| self.calculate_token_priority(token) >= threshold);

                let removed = original_size - segment.tokens.len();
                total_removed += removed;

                if removed > 0 {
                    segment.compressed = true;
                    segment.compression_ratio = segment.tokens.len() as f32 / original_size as f32;
                    segment.total_size = segment.tokens.len();

                    let _ = self
                        .update_tx
                        .send(ContextUpdate::SegmentCompressed(
                            segment.id.clone(),
                            segment.compression_ratio,
                        ))
                        .await;
                }
            }
        }

        self.stats.write().await.compressed_tokens += total_removed;
        self.stats.write().await.compressions_performed += 1;

        debug!("Removed {} low-importance tokens", total_removed);

        Ok(())
    }

    /// Compress using temporal decay
    async fn compress_by_temporal_decay(
        &self,
        half_life: Duration,
        min_importance: f32,
    ) -> Result<()> {
        let now = Utc::now();
        let mut segments = self.segments.write().await;

        for segment in segments.iter_mut() {
            let mut remaining_tokens = Vec::new();

            for token in &segment.tokens {
                // Calculate age-adjusted importance
                let age = now.signed_duration_since(token.timestamp).num_seconds() as f32;
                let decay_factor = (-age / half_life.as_secs_f32()).exp();
                let adjusted_importance = token.importance * decay_factor;

                if adjusted_importance >= min_importance {
                    remaining_tokens.push(token.clone());
                }
            }

            if remaining_tokens.len() < segment.tokens.len() {
                segment.tokens = remaining_tokens;
                segment.compressed = true;
                segment.compression_ratio = segment.tokens.len() as f32 / segment.total_size as f32;
            }
        }

        Ok(())
    }

    /// Compress by summarizing similar tokens
    async fn compress_by_summarization(
        &self,
        similarity_threshold: f32, /* TODO: Use similarity_threshold for intelligent token
                                    * similarity detection and grouping */
        max_summary_length: usize,
    ) -> Result<()> {
        // Enhanced similarity-based compression using the threshold parameter
        info!("Starting similarity-based compression with threshold: {:.3}", similarity_threshold);

        let mut segments = self.segments.write().await;
        let mut total_tokens_before = 0;
        let mut total_tokens_after = 0;

        for segment in segments.iter_mut() {
            if segment.compressed {
                continue;
            }

            let original_size = segment.tokens.len();
            total_tokens_before += original_size;

            // Group tokens by similarity using the threshold
            let similarity_groups =
                self.group_tokens_by_similarity(&segment.tokens, similarity_threshold).await?;

            let mut compressed_tokens: Vec<ContextToken> = Vec::new();

            for group in similarity_groups {
                if group.len() == 1 {
                    // Single token - keep as is
                    compressed_tokens.push(group[0].clone());
                } else {
                    // Multiple similar tokens - create summary
                    let summary_token =
                        self.create_summary_token(&group, max_summary_length).await?;
                    compressed_tokens.push(summary_token);
                }
            }

            // Only apply compression if it's beneficial
            if compressed_tokens.len() < original_size {
                let compression_ratio = compressed_tokens.len() as f32 / original_size as f32;

                segment.tokens = compressed_tokens;
                segment.compressed = true;
                segment.compression_ratio = compression_ratio;
                segment.total_size = segment.tokens.len();

                total_tokens_after += segment.tokens.len();

                debug!(
                    "Compressed segment {} from {} to {} tokens (ratio: {:.3}, similarity \
                     threshold: {:.3})",
                    segment.id,
                    original_size,
                    segment.tokens.len(),
                    compression_ratio,
                    similarity_threshold
                );
            } else {
                total_tokens_after += original_size;
            }
        }

        // Update statistics
        if total_tokens_before > 0 {
            let mut stats = self.stats.write().await;
            stats.compressions_performed += 1;
            let overall_ratio = total_tokens_after as f32 / total_tokens_before as f32;
            stats.avg_compression_ratio = (stats.avg_compression_ratio + overall_ratio) / 2.0;

            info!(
                "Similarity compression complete: {} -> {} tokens (ratio: {:.3})",
                total_tokens_before, total_tokens_after, overall_ratio
            );
        }

        Ok(())
    }

    /// Group tokens by similarity based on threshold
    async fn group_tokens_by_similarity(
        &self,
        tokens: &[ContextToken],
        similarity_threshold: f32,
    ) -> Result<Vec<Vec<ContextToken>>> {
        let mut groups: Vec<Vec<ContextToken>> = Vec::new();
        let mut used_indices = std::collections::HashSet::new();

        for (i, token) in tokens.iter().enumerate() {
            if used_indices.contains(&i) {
                continue;
            }

            let mut group = vec![token.clone()];
            used_indices.insert(i);

            // Find similar tokens
            for (j, other_token) in tokens.iter().enumerate().skip(i + 1) {
                if used_indices.contains(&j) {
                    continue;
                }

                let similarity = self.calculate_token_similarity(token, other_token).await;
                if similarity >= similarity_threshold {
                    group.push(other_token.clone());
                    used_indices.insert(j);
                }
            }

            groups.push(group);
        }

        debug!(
            "Grouped {} tokens into {} similarity groups (threshold: {:.3})",
            tokens.len(),
            groups.len(),
            similarity_threshold
        );

        Ok(groups)
    }

    /// Calculate similarity between two tokens
    async fn calculate_token_similarity(
        &self,
        token1: &ContextToken,
        token2: &ContextToken,
    ) -> f32 {
        let mut similarity = 0.0;

        // Content similarity (semantic)
        let content_sim = self.calculate_content_similarity(&token1.content, &token2.content);
        similarity += content_sim * 0.4;

        // Type similarity
        let type_sim = if std::mem::discriminant(&token1.token_type)
            == std::mem::discriminant(&token2.token_type)
        {
            1.0
        } else {
            0.0
        };
        similarity += type_sim * 0.2;

        // Temporal similarity (closer in time = more similar)
        let time_diff = (token1.timestamp - token2.timestamp).num_seconds().abs() as f32;
        let temporal_sim = (-time_diff / 3600.0).exp(); // Decay over hours
        similarity += temporal_sim * 0.2;

        // Metadata similarity
        let metadata_sim = self.calculate_metadata_similarity(&token1.metadata, &token2.metadata);
        similarity += metadata_sim * 0.2;

        similarity.clamp(0.0, 1.0)
    }

    /// Calculate content similarity between two strings
    #[inline(always)] // Backend optimization: frequent similarity computation
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        // Backend optimization: optimized for string processing hot path
        crate::compiler_backend_optimization::register_optimization::low_register_pressure(|| {
        // Enhanced similarity calculation
        let words1: std::collections::HashSet<String> =
            content1.split_whitespace().map(|w| w.to_lowercase()).collect();
        let words2: std::collections::HashSet<String> =
            content2.split_whitespace().map(|w| w.to_lowercase()).collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        // Jaccard similarity
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        let jaccard = if union == 0 { 0.0 } else { intersection as f32 / union as f32 };

        // Add semantic similarity bonus for related concepts
        let semantic_bonus = self.calculate_semantic_bonus(&words1, &words2);

        (jaccard + semantic_bonus * 0.3).clamp(0.0, 1.0)
        })
    }

    /// Calculate semantic bonus for related concepts
    fn calculate_semantic_bonus(
        &self,
        words1: &std::collections::HashSet<String>,
        words2: &std::collections::HashSet<String>,
    ) -> f32 {
        // Simple semantic relationships - in a full implementation, this would use
        // embeddings
        let semantic_pairs = [
            ("thinking", "thought"),
            ("memory", "remember"),
            ("decision", "choose"),
            ("emotion", "feeling"),
            ("attention", "focus"),
            ("goal", "objective"),
            ("create", "generate"),
            ("analyze", "examine"),
            ("learn", "study"),
            ("social", "interaction"),
            ("empathy", "understanding"),
            ("conscious", "aware"),
        ];

        let mut bonus = 0.0;
        let mut matches = 0;

        for word1 in words1 {
            for word2 in words2 {
                for (related1, related2) in semantic_pairs {
                    if (word1.contains(related1) && word2.contains(related2))
                        || (word1.contains(related2) && word2.contains(related1))
                    {
                        bonus += 0.5;
                        matches += 1;
                    }
                }
            }
        }

        if matches > 0 { bonus / matches as f32 } else { 0.0 }
    }

    /// Calculate metadata similarity
    fn calculate_metadata_similarity(&self, meta1: &TokenMetadata, meta2: &TokenMetadata) -> f32 {
        let mut similarity = 0.0;

        // Source similarity
        if meta1.source == meta2.source {
            similarity += 0.3;
        }

        // Emotional valence similarity
        let valence_diff = (meta1.emotional_valence - meta2.emotional_valence).abs();
        similarity += (1.0 - valence_diff) * 0.2;

        // Attention weight similarity
        let attention_diff = (meta1.attention_weight - meta2.attention_weight).abs();
        similarity += (1.0 - attention_diff) * 0.2;

        // Association overlap
        let associations1: std::collections::HashSet<_> = meta1.associations.iter().collect();
        let associations2: std::collections::HashSet<_> = meta2.associations.iter().collect();
        let intersection = associations1.intersection(&associations2).count();
        let union = associations1.union(&associations2).count();
        if union > 0 {
            similarity += (intersection as f32 / union as f32) * 0.3;
        }

        similarity.clamp(0.0, 1.0)
    }

    /// Create a summary token from a group of similar tokens
    async fn create_summary_token(
        &self,
        group: &[ContextToken],
        max_summary_length: usize,
    ) -> Result<ContextToken> {
        if group.is_empty() {
            return Err(anyhow::anyhow!("Cannot create summary from empty group"));
        }

        // Use the most important token as the base
        let base_token = group
            .iter()
            .max_by(|a, b| {
                a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Create summary content
        let mut summary_parts: Vec<String> =
            group.iter().map(|token| token.content.clone()).collect();

        // Remove duplicates and sort by length (shortest first for core concepts)
        summary_parts.sort_by_key(|s| s.len());
        summary_parts.dedup();

        let summary_content = if summary_parts.join(" ").len() <= max_summary_length {
            summary_parts.join(" ")
        } else {
            // Truncate to max length while preserving important parts
            let mut result = String::new();
            for part in summary_parts {
                if result.len() + part.len() < max_summary_length {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(&part);
                } else {
                    break;
                }
            }
            if result.is_empty() {
                base_token.content.chars().take(max_summary_length).collect()
            } else {
                result
            }
        };

        // Aggregate metadata
        let avg_importance = group.iter().map(|t| t.importance).sum::<f32>() / group.len() as f32;
        let avg_emotional_valence =
            group.iter().map(|t| t.metadata.emotional_valence).sum::<f32>() / group.len() as f32;
        let avg_attention_weight =
            group.iter().map(|t| t.metadata.attention_weight).sum::<f32>() / group.len() as f32;

        // Collect all unique associations
        let mut all_associations = std::collections::HashSet::new();
        for token in group {
            for assoc in &token.metadata.associations {
                all_associations.insert(assoc.clone());
            }
        }

        // Create summary token
        Ok(ContextToken {
            content: summary_content,
            token_type: base_token.token_type.clone(),
            importance: avg_importance * 1.1, // Slight boost for being a summary
            timestamp: base_token.timestamp,  // Use base token's timestamp
            metadata: TokenMetadata {
                source: format!("summary_of_{}_tokens", group.len()),
                emotional_valence: avg_emotional_valence,
                attention_weight: avg_attention_weight,
                associations: all_associations.into_iter().collect(),
                compressed: true,
            },
        })
    }

    /// Compress using semantic clustering
    /// Implements fractal-based semantic clustering following the cognitive
    /// enhancement plan
    async fn compress_by_clustering(
        &self,
        num_clusters: usize,
        representatives_per_cluster: usize,
    ) -> Result<()> {
        info!(
            "Starting semantic clustering compression: {} clusters, {} reps per cluster",
            num_clusters, representatives_per_cluster
        );

        let mut segments = self.segments.write().await;
        if segments.is_empty() {
            return Ok(());
        }

        let mut total_compression_ratio = 0.0;
        let mut segments_compressed = 0;

        // Process segments that aren't already compressed
        for segment in segments.iter_mut().filter(|s| !s.compressed) {
            if segment.tokens.len() < num_clusters * 2 {
                continue; // Skip segments too small for meaningful clustering
            }

            let original_size = segment.tokens.len();

            // Extract semantic features for clustering
            let token_features = self.extract_semantic_features(&segment.tokens).await?;

            // Perform k-means clustering with semantic similarity
            let clusters = self
                .perform_semantic_clustering(&segment.tokens, &token_features, num_clusters)
                .await?;

            // Select representatives from each cluster
            let mut compressed_tokens = Vec::new();
            for cluster in clusters {
                let representatives = self
                    .select_cluster_representatives(&cluster, representatives_per_cluster)
                    .await?;
                compressed_tokens.extend(representatives);
            }

            // Only apply compression if it's beneficial
            if compressed_tokens.len() < original_size {
                let compression_ratio = compressed_tokens.len() as f32 / original_size as f32;

                segment.tokens = compressed_tokens;
                segment.total_size = segment.tokens.len();
                segment.compressed = true;
                segment.compression_ratio = compression_ratio;

                total_compression_ratio += compression_ratio;
                segments_compressed += 1;

                debug!(
                    "Compressed segment {} from {} to {} tokens (ratio: {:.3})",
                    segment.id,
                    original_size,
                    segment.tokens.len(),
                    compression_ratio
                );
            }
        }

        // Update statistics
        if segments_compressed > 0 {
            let mut stats = self.stats.write().await;
            stats.compressions_performed += 1;
            stats.avg_compression_ratio = total_compression_ratio / segments_compressed as f32;

            info!(
                "Semantic clustering complete: compressed {} segments, avg ratio: {:.3}",
                segments_compressed, stats.avg_compression_ratio
            );
        }

        Ok(())
    }

    /// Extract semantic features for tokens using cognitive-aware embeddings
    async fn extract_semantic_features(&self, tokens: &[ContextToken]) -> Result<Vec<Vec<f32>>> {
        let mut features = Vec::new();

        for token in tokens {
            let mut feature_vector = vec![0.0f32; 128]; // Standard embedding dimension

            // Content-based features (simplified word embeddings)
            let content_hash = self.simple_hash(&token.content);
            let content_features = self.hash_to_features(content_hash, 64);
            feature_vector[0..64].copy_from_slice(&content_features);

            // Metadata features
            feature_vector[64] = token.importance;
            feature_vector[65] = token.metadata.emotional_valence;
            feature_vector[66] = token.metadata.attention_weight;
            feature_vector[67] = self.encode_token_type(&token.token_type);

            // Temporal features
            let age_hours =
                chrono::Utc::now().signed_duration_since(token.timestamp).num_hours() as f32;
            feature_vector[68] = (-age_hours / 24.0).exp(); // Recency factor

            // Source encoding
            let source_hash = self.simple_hash(&token.metadata.source);
            let source_features = self.hash_to_features(source_hash, 32);
            feature_vector[69..101].copy_from_slice(&source_features);

            // Association features (connectivity)
            feature_vector[101] = token.metadata.associations.len() as f32 / 10.0; // Normalized

            // Remaining dimensions for future extensions
            for i in 102..128 {
                feature_vector[i] = 0.0;
            }

            features.push(feature_vector);
        }

        Ok(features)
    }

    /// Perform semantic clustering using k-means with cognitive weighting
    async fn perform_semantic_clustering(
        &self,
        tokens: &[ContextToken],
        features: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<ContextToken>>> {
        if tokens.len() < k {
            // If we have fewer tokens than clusters, each token forms its own cluster
            return Ok(tokens.iter().map(|t| vec![t.clone()]).collect());
        }

        // Initialize centroids using k-means++ for better initialization
        let mut centroids = self.initialize_centroids_kmeans_plus_plus(features, k).await?;
        let mut clusters: Vec<Vec<ContextToken>> = vec![Vec::new(); k];

        const MAX_ITERATIONS: usize = 20;
        const CONVERGENCE_THRESHOLD: f32 = 0.001;

        for iteration in 0..MAX_ITERATIONS {
            // Clear previous assignments
            for cluster in &mut clusters {
                cluster.clear();
            }

            // Assign tokens to nearest centroids with position-aware clustering
            for (token_idx, (token, feature)) in tokens.iter().zip(features.iter()).enumerate() {
                // Implement position-aware clustering with advanced weighting
                let mut best_cluster = 0;
                let mut best_distance = f32::INFINITY;

                for (cluster_idx, centroid) in centroids.iter().enumerate() {
                    let mut distance =
                        self.calculate_weighted_distance(feature, centroid, token).await;

                    // Enhanced position-aware weighting with multiple factors
                    let position_factor = self.calculate_enhanced_position_factor(token_idx, tokens.len());
                    // Extract content strings for semantic locality calculation
                    let token_contents: Vec<String> = tokens.iter().map(|t| t.content.clone()).collect();
                    let semantic_locality_factor = self.calculate_semantic_locality_factor(token_idx, &token_contents, cluster_idx);
                    let temporal_factor = self.calculate_temporal_clustering_factor(token_idx);

                    // Combined weighting for optimal clustering
                    let combined_factor = position_factor * semantic_locality_factor * temporal_factor;
                    distance *= combined_factor;

                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = cluster_idx;
                    }
                }

                clusters[best_cluster].push(token.clone());
            }

            // Update centroids
            let mut centroid_shift = 0.0;
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid =
                        self.calculate_cluster_centroid(cluster, features, tokens).await?;

                    // Calculate shift for convergence check
                    let shift = self.euclidean_distance(&centroids[cluster_idx], &new_centroid);
                    centroid_shift += shift;

                    centroids[cluster_idx] = new_centroid;
                }
            }

            // Check for convergence
            let avg_shift = centroid_shift / k as f32;
            if avg_shift < CONVERGENCE_THRESHOLD {
                debug!(
                    "K-means converged after {} iterations (shift: {:.6})",
                    iteration + 1,
                    avg_shift
                );
                break;
            }
        }

        // Remove empty clusters
        clusters.retain(|cluster| !cluster.is_empty());

        debug!(
            "Semantic clustering produced {} non-empty clusters from {} tokens",
            clusters.len(),
            tokens.len()
        );

        Ok(clusters)
    }

    /// Calculate position factor for position-aware clustering
    /// Enhanced position-aware clustering factor with multiple weighting strategies
    fn calculate_enhanced_position_factor(&self, token_idx: usize, total_tokens: usize) -> f32 {
        if total_tokens <= 1 {
            return 1.0;
        }

        let relative_position = token_idx as f32 / (total_tokens - 1) as f32;

        // Multi-strategy position weighting
        let attention_weight = self.calculate_attention_based_position_weight(relative_position);
        let recency_weight = self.calculate_recency_weight(token_idx, total_tokens);
        let structural_weight = self.calculate_structural_position_weight(relative_position);

        // Combine weights with learned importance
        (attention_weight * 0.4 + recency_weight * 0.35 + structural_weight * 0.25).max(0.1)
    }

    /// Calculate attention-based position weight (U-shaped curve)
    fn calculate_attention_based_position_weight(&self, relative_position: f32) -> f32 {
        // Attention mechanism: higher weights at beginning and end
        let attention_curve = 1.0 - 4.0 * (relative_position - 0.5).powi(2);
        attention_curve.max(0.2)
    }

    /// Calculate recency weight (exponential decay)
    fn calculate_recency_weight(&self, token_idx: usize, total_tokens: usize) -> f32 {
        // More recent tokens (later in sequence) get higher weight
        let recency_factor = (token_idx as f32 / total_tokens as f32).powi(2);
        0.5 + recency_factor * 0.5
    }

    /// Calculate structural position weight based on text structure
    fn calculate_structural_position_weight(&self, relative_position: f32) -> f32 {
        // Apply different weighting strategies based on position
        match relative_position {
            // Beginning tokens (first 20%) - slightly higher importance
            p if p < 0.2 => 0.9,
            // Middle tokens (20-80%) - standard weighting
            p if p < 0.8 => 1.0,
            // End tokens (last 20%) - higher importance (recency effect)
            _ => 0.85,
        }
    }

    /// Calculate semantic locality factor for clustering
    fn calculate_semantic_locality_factor(&self, token_idx: usize, tokens: &[String], cluster_idx: usize) -> f32 {
        // Consider semantic similarity to nearby tokens
        let window_size = 3; // Look at ±3 tokens around current position
        let mut locality_score = 1.0;

        // Check semantic coherence within local window
        let start_idx = token_idx.saturating_sub(window_size);
        let end_idx = (token_idx + window_size + 1).min(tokens.len());

        let local_tokens = &tokens[start_idx..end_idx];

        // Calculate semantic density in local window
        let semantic_density = self.calculate_local_semantic_density(local_tokens);

        // Adjust clustering factor based on semantic density
        if semantic_density > 0.7 {
            locality_score *= 0.8; // Tighter clustering for semantically dense areas
        } else if semantic_density < 0.3 {
            locality_score *= 1.2; // Looser clustering for sparse areas
        }

        // Add cluster-specific adjustment
        let cluster_affinity = self.calculate_cluster_affinity(cluster_idx);
        locality_score *= cluster_affinity;

        locality_score.max(0.5).min(1.5)
    }

    /// Calculate temporal clustering factor
    fn calculate_temporal_clustering_factor(&self, token_idx: usize) -> f32 {
        // Temporal patterns in clustering - simulate temporal importance
        let temporal_cycle = (token_idx as f32 * 0.1).sin().abs();
        let temporal_base = 0.8 + (temporal_cycle * 0.4);

        // Add exponential decay for very distant tokens
        let decay_factor = if token_idx > 100 {
            (-0.005 * (token_idx - 100) as f32).exp()
        } else {
            1.0
        };

        (temporal_base * decay_factor).max(0.3)
    }

    /// Calculate local semantic density
    fn calculate_local_semantic_density(&self, local_tokens: &[String]) -> f32 {
        if local_tokens.len() < 2 {
            return 0.5;
        }

        let mut similarity_sum = 0.0;
        let mut pair_count = 0;

        // Calculate pairwise semantic similarity in local window
        for i in 0..local_tokens.len() {
            for j in (i + 1)..local_tokens.len() {
                let similarity = self.calculate_string_similarity(&local_tokens[i], &local_tokens[j]);
                similarity_sum += similarity;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            similarity_sum / pair_count as f32
        } else {
            0.5
        }
    }

    /// Calculate cluster affinity factor
    fn calculate_cluster_affinity(&self, cluster_idx: usize) -> f32 {
        // Simulate cluster-specific preferences
        match cluster_idx % 4 {
            0 => 1.0,  // Neutral cluster
            1 => 0.9,  // Slightly prefer other clusters
            2 => 1.1,  // Slightly prefer this cluster
            3 => 0.95, // Moderately prefer other clusters
            _ => 1.0,
        }
    }

    /// Calculate string similarity using Levenshtein distance
    fn calculate_string_similarity(&self, token1: &str, token2: &str) -> f32 {
        if token1 == token2 {
            return 1.0;
        }

        // Simple Levenshtein-based similarity
        let max_len = token1.len().max(token2.len());
        if max_len == 0 {
            return 1.0;
        }

        let distance = self.levenshtein_distance(token1, token2);
        1.0 - (distance as f32 / max_len as f32)
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let m = s1_chars.len();
        let n = s2_chars.len();

        if m == 0 { return n; }
        if n == 0 { return m; }

        let mut matrix = vec![vec![0; n + 1]; m + 1];

        for i in 0..=m { matrix[i][0] = i; }
        for j in 0..=n { matrix[0][j] = j; }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[m][n]
    }

    /// Select representative tokens from a cluster based on cognitive
    /// importance
    async fn select_cluster_representatives(
        &self,
        cluster: &[ContextToken],
        max_representatives: usize,
    ) -> Result<Vec<ContextToken>> {
        if cluster.len() <= max_representatives {
            return Ok(cluster.to_vec());
        }

        // Score tokens by representative quality with advanced selection strategies
        let mut scored_tokens: Vec<(f32, &ContextToken)> = cluster
            .iter()
            .map(|token| {
                let score = self.calculate_representative_score(token, cluster);
                (score, token)
            })
            .collect();

        // Sort by score (highest first)
        scored_tokens.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Select top representatives with advanced strategies
        let mut representatives = Vec::new();
        let mut selected_content_hashes = std::collections::HashSet::new();

        for (score, token) in &scored_tokens {
            // Advanced representative selection using score-based strategies
            let content_hash = self.simple_hash(&token.content);

            // Calculate score-based inclusion probability
            let score_factor = score / scored_tokens.first().map(|(s, _)| *s).unwrap_or(1.0);
            let quality_threshold = 0.7; // High-quality tokens get preference
            let diversity_bonus = if selected_content_hashes.contains(&content_hash) { 0.0 } else { 0.2 };
            let adjusted_score = score_factor + diversity_bonus;

            // Advanced score-based selection strategy
            let should_include = if representatives.is_empty() {
                // Always include the highest-scoring token
                true
            } else if selected_content_hashes.contains(&content_hash) {
                // Skip duplicates regardless of score
                false
            } else {
                // Score-based inclusion with quality and diversity factors
                let final_score = adjusted_score;
                let meets_quality = final_score >= quality_threshold;
                let meets_relative_quality = final_score >= (score_factor * 0.6); // Relative to best score

                meets_quality && meets_relative_quality && representatives.len() < max_representatives
            };

            if should_include {
                representatives.push((*token).clone());
                selected_content_hashes.insert(content_hash);

                if representatives.len() >= max_representatives {
                    break;
                }
            }
        }

        // Ensure we have at least one representative
        if representatives.is_empty() && !cluster.is_empty() {
            representatives.push(cluster[0].clone());
        }

        debug!(
            "Selected {} representatives from cluster of {} tokens using advanced strategies",
            representatives.len(),
            cluster.len()
        );

        Ok(representatives)
    }

    /// Calculate representative quality score for a token within its cluster
    fn calculate_representative_score(
        &self,
        token: &ContextToken,
        cluster: &[ContextToken],
    ) -> f32 {
        let mut score = 0.0;

        // Base importance
        score += token.importance * 0.3;

        // Centrality within cluster (how typical it is)
        let content_similarities = cluster
            .iter()
            .map(|other| self.simple_content_similarity(&token.content, &other.content))
            .collect::<Vec<_>>();
        let avg_similarity =
            content_similarities.iter().sum::<f32>() / content_similarities.len() as f32;
        score += avg_similarity * 0.2;

        // Recency factor
        let age_hours =
            chrono::Utc::now().signed_duration_since(token.timestamp).num_hours() as f32;
        let recency = (-age_hours / 168.0).exp(); // Decay over a week
        score += recency * 0.2;

        // Emotional significance
        score += token.metadata.emotional_valence.abs() * 0.1;

        // Attention weight
        score += token.metadata.attention_weight * 0.1;

        // Association richness
        score += (token.metadata.associations.len() as f32 / 10.0).min(1.0) * 0.1;

        score.clamp(0.0, 1.0)
    }

    /// Helper functions for clustering
    ///
    /// Initialize centroids using k-means++ algorithm
    async fn initialize_centroids_kmeans_plus_plus(
        &self,
        features: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<f32>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::new();

        if features.is_empty() {
            return Ok(centroids);
        }

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..features.len());
        centroids.push(features[first_idx].clone());

        // Choose remaining centroids with probability proportional to squared distance
        for _ in 1..k {
            let mut distances = Vec::new();
            let mut total_distance = 0.0;

            for feature in features {
                let min_distance = centroids
                    .iter()
                    .map(|centroid| self.euclidean_distance(feature, centroid))
                    .fold(f32::INFINITY, f32::min);
                let squared_distance = min_distance * min_distance;
                distances.push(squared_distance);
                total_distance += squared_distance;
            }

            if total_distance > 0.0 {
                let threshold = rng.gen::<f32>() * total_distance;
                let mut cumulative = 0.0;

                for (idx, &distance) in distances.iter().enumerate() {
                    cumulative += distance;
                    if cumulative >= threshold {
                        centroids.push(features[idx].clone());
                        break;
                    }
                }
            } else {
                // Fallback: choose random point
                let idx = rng.gen_range(0..features.len());
                centroids.push(features[idx].clone());
            }
        }

        Ok(centroids)
    }

    /// Calculate weighted distance considering cognitive factors
    async fn calculate_weighted_distance(
        &self,
        feature: &[f32],
        centroid: &[f32],
        token: &ContextToken,
    ) -> f32 {
        let euclidean_dist = self.euclidean_distance(feature, centroid);

        // Apply cognitive weighting
        let importance_weight = 1.0 + token.importance; // More important tokens get stronger affinity
        let recency_weight = 1.0
            + (-chrono::Utc::now().signed_duration_since(token.timestamp).num_hours() as f32
                / 24.0)
                .exp()
                * 0.5;

        euclidean_dist / (importance_weight * recency_weight)
    }

    /// Calculate cluster centroid
    async fn calculate_cluster_centroid(
        &self,
        cluster: &[ContextToken],
        all_features: &[Vec<f32>],
        all_tokens: &[ContextToken],
    ) -> Result<Vec<f32>> {
        if cluster.is_empty() {
            return Ok(vec![0.0; 128]);
        }

        let feature_dim = all_features.first().map(|f| f.len()).unwrap_or(128);
        let mut centroid = vec![0.0; feature_dim];

        // Find features for cluster tokens and compute weighted average
        let mut total_weight = 0.0;

        for cluster_token in cluster {
            // Find corresponding feature vector
            if let Some(token_idx) = all_tokens.iter().position(|t| {
                t.content == cluster_token.content && t.timestamp == cluster_token.timestamp
            }) {
                if let Some(feature) = all_features.get(token_idx) {
                    let weight = cluster_token.importance + 0.1; // Minimum weight to avoid zero
                    total_weight += weight;

                    for (i, &value) in feature.iter().enumerate() {
                        if i < centroid.len() {
                            centroid[i] += value * weight;
                        }
                    }
                }
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for value in &mut centroid {
                *value /= total_weight;
            }
        }

        Ok(centroid)
    }

    /// Calculate euclidean distance between two vectors
    #[inline(always)] // Backend optimization: critical math operation
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Backend optimization: vectorized distance calculation
        crate::compiler_backend_optimization::instruction_selection::fast_math::vectorized_euclidean_distance(a, b)
    }

    /// Hash to feature vector conversion
    fn hash_to_features(&self, hash: u64, size: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(size);
        let mut working_hash = hash;

        for _ in 0..size {
            features.push((working_hash as f32 % 1000.0) / 1000.0); // Normalize to 0-1
            working_hash = working_hash.wrapping_mul(1_103_515_245).wrapping_add(12_345); // Linear congruential generator
        }

        features
    }

    /// Encode token type as a numeric feature
    fn encode_token_type(&self, token_type: &TokenType) -> f32 {
        match token_type {
            TokenType::Thought(_) => 0.1,
            TokenType::Decision(_) => 0.2,
            TokenType::Goal(_) => 0.3,
            TokenType::Event(_) => 0.4,
            TokenType::Narrative(_) => 0.5,
            TokenType::Memory(_) => 0.6,
            TokenType::Social(_) => 0.7,
        }
    }

    /// Simple content similarity metric
    fn simple_content_similarity(&self, content1: &str, content2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = content1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = content2.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }

    /// Calculate priority for a token
    fn calculate_token_priority(&self, token: &ContextToken) -> f32 {
        let weights = &self.priority_system.weights;

        let mut priority = 0.0;

        // Base importance
        priority += token.importance * weights.importance;

        // Recency factor
        let age = Utc::now().signed_duration_since(token.timestamp).num_seconds() as f32 / 3600.0; // Hours
        let recency = (-age / 24.0).exp(); // Decay over 24 hours
        priority += recency * weights.recency;

        // Emotional significance
        let emotional_abs = token.metadata.emotional_valence.abs();
        priority += emotional_abs * weights.emotional_significance;

        // Attention correlation
        priority += token.metadata.attention_weight * weights.attention_correlation;

        // Apply retention rules
        for rule in &self.priority_system.rules {
            if self.matches_retention_rule(token, &rule.condition) {
                priority += rule.priority_boost;
            }
        }

        priority.clamp(0.0, 1.0)
    }

    /// Check if token matches retention rule
    fn matches_retention_rule(&self, token: &ContextToken, condition: &RetentionCondition) -> bool {
        match condition {
            RetentionCondition::TokenType(token_type) => {
                // Compare token types (simplified)
                matches!(
                    (&token.token_type, token_type),
                    (TokenType::Decision(_), TokenType::Decision(_))
                        | (TokenType::Goal(_), TokenType::Goal(_))
                )
            }
            RetentionCondition::Source(source) => token.metadata.source == *source,
            RetentionCondition::ActiveGoal => {
                matches!(token.token_type, TokenType::Goal(_))
            }
            RetentionCondition::DecisionChain => {
                matches!(token.token_type, TokenType::Decision(_))
            }
            RetentionCondition::Custom(_) => false,
        }
    }

    /// Create a checkpoint
    pub async fn create_checkpoint(&self, reason: String) -> Result<String> {
        info!("Creating context checkpoint: {}", reason);

        // Ensure all buffers are flushed to segments
        self.create_segment().await?;

        let segments = self.segments.read().await.clone();
        let checkpoint_id = uuid::Uuid::new_v4().to_string();

        // Calculate metadata
        let total_tokens: usize = segments.iter().map(|s| s.total_size).sum();
        let compressed_tokens: usize =
            segments.iter().filter(|s| s.compressed).map(|s| s.total_size).sum();

        let compression_ratio =
            if total_tokens > 0 { compressed_tokens as f32 / total_tokens as f32 } else { 1.0 };

        // Serialize segments

        // let serialized = bincode::serialize(&segments)?;
        // let compressed_state = compress_prepend_size(&serialized);
        let compressed_state = Vec::new();

        let metadata = CheckpointMetadata {
            total_tokens,
            compressed_tokens,
            compression_ratio,
            emotional_summary: EmotionalBlend::default(), // Would get from emotional core
            active_goals: vec![],                         // Would get from goal manager
            recent_decisions: vec![],                     // Would get from decision engine
            checkpoint_reason: reason,
        };

        let checkpoint = ContextCheckpoint {
            id: checkpoint_id.clone(),
            timestamp: Utc::now(),
            segments: segments.into_iter().collect(),
            metadata,
            compressed_state,
        };

        // Store checkpoint
        self.checkpoints.write().await.insert(checkpoint.timestamp, checkpoint);

        // Prune old checkpoints
        self.prune_checkpoints().await?;

        // Update stats
        self.stats.write().await.checkpoints_created += 1;

        // Store in memory
        self.memory
            .store(
                format!("Context checkpoint created: {}", checkpoint_id),
                vec![],
                MemoryMetadata {
                    source: "context_manager".to_string(),
                    tags: vec!["checkpoint".to_string(), "context".to_string()],
                    importance: 0.8,
                    associations: vec![],
                    context: Some("context checkpoint creation".to_string()),
                    created_at: chrono::Utc::now(),
                    accessed_count: 0,
                    last_accessed: None,
                    version: 1,
                    category: "cognitive".to_string(),
                    timestamp: chrono::Utc::now(),
                    expiration: None,
                },
            )
            .await?;

        let _ = self.update_tx.send(ContextUpdate::CheckpointCreated(checkpoint_id.clone())).await;

        Ok(checkpoint_id)
    }

    /// Restore from checkpoint
    pub async fn restore_checkpoint(&self, checkpoint_id: &str) -> Result<()> {
        info!("Restoring context from checkpoint: {}", checkpoint_id);

        let checkpoints = self.checkpoints.read().await;
        let checkpoint = checkpoints
            .values()
            .find(|c| c.id == checkpoint_id)
            .ok_or_else(|| anyhow!("Checkpoint not found"))?;

        // Decompress and deserialize
        let _decompressed = decompress_size_prepended(&checkpoint.compressed_state)?;
        let segments: VecDeque<ContextSegment> = VecDeque::new(); /* bincode::deserialize(&decompressed)?; */

        // Replace current context
        *self.segments.write().await = segments;
        self.token_buffer.write().await.clear();

        // Update stats
        self.stats.write().await.total_tokens = checkpoint.metadata.total_tokens;
        self.stats.write().await.compressed_tokens = checkpoint.metadata.compressed_tokens;

        let _ =
            self.update_tx.send(ContextUpdate::ContextRestored(checkpoint_id.to_string())).await;

        info!("Restored {} tokens from checkpoint", checkpoint.metadata.total_tokens);

        Ok(())
    }

    /// Prune old checkpoints
    async fn prune_checkpoints(&self) -> Result<()> {
        let mut checkpoints = self.checkpoints.write().await;

        while checkpoints.len() > self.config.max_checkpoints {
            if let Some((timestamp, _)) = checkpoints.iter().next() {
                let ts = *timestamp;
                checkpoints.remove(&ts);
            }
        }

        Ok(())
    }

    /// Get current context summary
    pub async fn get_context_summary(&self) -> Result<ContextSummary> {
        let segments = self.segments.read().await;
        let buffer = self.token_buffer.read().await;

        let total_tokens = self.calculate_total_tokens().await;
        let compressed_segments = segments.iter().filter(|s| s.compressed).count();

        let recent_tokens: Vec<ContextToken> = segments
            .iter()
            .rev()
            .take(3)
            .flat_map(|s| s.tokens.iter().cloned())
            .chain(buffer.iter().cloned())
            .collect();

        let important_tokens: Vec<ContextToken> = segments
            .iter()
            .flat_map(|s| s.tokens.iter())
            .filter(|t| t.importance > 0.8)
            .take(10)
            .cloned()
            .collect();

        Ok(ContextSummary {
            total_tokens,
            compressed_segments,
            total_segments: segments.len(),
            buffer_size: buffer.len(),
            recent_tokens,
            important_tokens,
            memory_usage_mb: self.estimate_memory_usage().await,
        })
    }

    /// Get tokens within a time range
    pub async fn get_tokens_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<ContextToken> {
        let segments = self.segments.read().await;

        segments
            .iter()
            .filter(|s| {
                let (seg_start, seg_end) = s.time_range;
                seg_end >= start && seg_start <= end
            })
            .flat_map(|s| s.tokens.iter())
            .filter(|t| t.timestamp >= start && t.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Search tokens by pattern
    pub async fn search_tokens(&self, pattern: &str) -> Vec<ContextToken> {
        let segments = self.segments.read().await;
        let pattern_lower = pattern.to_lowercase();

        segments
            .iter()
            .flat_map(|s| s.tokens.iter())
            .filter(|t| t.content.to_lowercase().contains(&pattern_lower))
            .cloned()
            .collect()
    }

    /// Estimate memory usage
    async fn estimate_memory_usage(&self) -> f32 {
        let segments = self.segments.read().await;
        let buffer = self.token_buffer.read().await;

        // Rough estimation: 100 bytes per token average
        let token_count = segments.iter().map(|s| s.total_size).sum::<usize>() + buffer.len();
        let bytes = token_count * 100;

        bytes as f32 / 1_000_000.0 // Convert to MB
    }

    /// Default retention rules
    fn default_retention_rules() -> Vec<RetentionRule> {
        vec![
            RetentionRule {
                name: "Decisions".to_string(),
                condition: RetentionCondition::DecisionChain,
                priority_boost: 0.3,
            },
            RetentionRule {
                name: "Active Goals".to_string(),
                condition: RetentionCondition::ActiveGoal,
                priority_boost: 0.2,
            },
            RetentionRule {
                name: "Emotional Core".to_string(),
                condition: RetentionCondition::Source("emotional_core".to_string()),
                priority_boost: 0.15,
            },
        ]
    }

    /// Simple hash function for content
    #[inline(always)] // Backend optimization: frequent hashing operation
    fn simple_hash(&self, content: &str) -> u64 {
        // Backend optimization: fast hash using bit operations
        crate::compiler_backend_optimization::instruction_selection::bit_operations::fast_hash(content)
    }

    /// Segment creation loop
    async fn segment_creation_loop(&self) {
        let mut interval = interval(Duration::from_secs(30));
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.create_segment().await {
                        warn!("Segment creation error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Segment creation loop shutting down");
                    break;
                }
            }
        }
    }

    /// Compression loop
    async fn compression_loop(&self) {
        let mut interval = interval(Duration::from_secs(300)); // 5 minutes
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let total_tokens = self.calculate_total_tokens().await;
                    let threshold = self.config.max_tokens as f32 * self.config.compression_threshold;

                    if total_tokens as f32 > threshold {
                        if let Err(e) = self.trigger_compression().await {
                            warn!("Compression error: {}", e);
                        }
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Compression loop shutting down");
                    break;
                }
            }
        }
    }

    /// Checkpoint loop
    async fn checkpoint_loop(&self) {
        let mut interval = interval(self.config.checkpoint_interval);
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.create_checkpoint("Scheduled checkpoint".to_string()).await {
                        warn!("Checkpoint creation error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Checkpoint loop shutting down");
                    break;
                }
            }
        }
    }

    /// Get statistics
    pub async fn get_stats(&self) -> ContextStats {
        let mut stats = self.stats.read().await.clone();
        stats.memory_usage_mb = self.estimate_memory_usage().await;

        // Calculate average compression ratio
        let segments = self.segments.read().await;
        let compressed_count = segments.iter().filter(|s| s.compressed).count();
        if compressed_count > 0 {
            let total_ratio: f32 =
                segments.iter().filter(|s| s.compressed).map(|s| s.compression_ratio).sum();
            stats.avg_compression_ratio = total_ratio / compressed_count as f32;
        }

        stats
    }

    /// Shutdown the context manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Context Manager");

        // Create final checkpoint
        self.create_checkpoint("Shutdown checkpoint".to_string()).await?;

        let _ = self.shutdown_tx.send(());
        Ok(())
    }
}

/// Summary of current context state
#[derive(Clone, Debug)]
pub struct ContextSummary {
    pub total_tokens: usize,
    pub compressed_segments: usize,
    pub total_segments: usize,
    pub buffer_size: usize,
    pub recent_tokens: Vec<ContextToken>,
    pub important_tokens: Vec<ContextToken>,
    pub memory_usage_mb: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::{ThoughtMetadata, ThoughtType};

    #[tokio::test]
    async fn test_context_creation() {
        let memory = Arc::new(CognitiveMemory::new_for_test().await.unwrap());
        let context_manager =
            Arc::new(ContextManager::new(memory, ContextConfig::default()).await.unwrap());

        // Add some test tokens
        for i in 0..10 {
            let thought = Thought {
                id: ThoughtId::new(),
                content: format!("Test thought {}", i),
                thought_type: ThoughtType::Observation,
                metadata: ThoughtMetadata { importance: 0.5, ..Default::default() },
                ..Default::default()
            };

            context_manager.add_thought(&thought).await.unwrap();
        }

        let stats = context_manager.get_stats().await;
        assert_eq!(stats.total_tokens, 10);
    }

    #[tokio::test]
    async fn test_compression() {
        let memory = Arc::new(CognitiveMemory::new_for_test().await.unwrap());
        let mut config = ContextConfig::default();
        config.segment_size = 5;
        config.max_tokens = 20;
        config.compression_threshold = 0.5;

        let context_manager = Arc::new(ContextManager::new(memory, config).await.unwrap());

        // Add tokens with varying importance
        for i in 0..15 {
            let importance = if i % 3 == 0 { 0.8 } else { 0.3 };
            let thought = Thought {
                id: ThoughtId::new(),
                content: format!("Thought {}", i),
                thought_type: ThoughtType::Observation,
                metadata: ThoughtMetadata { importance, ..Default::default() },
                ..Default::default()
            };

            context_manager.add_thought(&thought).await.unwrap();
        }

        // Should trigger compression
        tokio::time::sleep(Duration::from_millis(100)).await;

        let stats = context_manager.get_stats().await;
        assert!(stats.compressed_tokens > 0);
    }
}
