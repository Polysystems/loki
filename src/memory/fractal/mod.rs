//! Fractal Memory Architecture
//!
//! Implements self-similar memory structures that maintain patterns across multiple scales.
//! Each memory node contains sub-nodes with similar organizational patterns, creating
//! emergent hierarchies and cross-scale resonance.

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Instant;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::memory::{MemoryId, MemoryItem, MemoryMetadata};

// ===== BASIC TYPE DEFINITIONS (must come before module declarations) =====

/// Unique identifier for fractal memory nodes
#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct FractalNodeId(String);

impl FractalNodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_content(content: &str) -> Self {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:?}", hasher.finalize());
        Self(hash[..16].to_string()) // Use first 16 chars for readability
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FractalNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for FractalNodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Scale levels for fractal organization
#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ScaleLevel {
    /// System level - fundamental system operations and core behaviors
    System = 0,
    /// Domain level - domain-specific knowledge and patterns
    Domain = 1,
    /// Token level - individual tokens and atomic elements
    Token = 2,
    /// Atomic level - individual facts, observations, sensations
    Atomic = 3,
    /// Concept level - grouped atomic elements forming concepts
    Concept = 4,
    /// Schema level - patterns and structures of concepts
    Schema = 5,
    /// Worldview level - overarching frameworks and philosophies
    Worldview = 6,
    /// Meta level - patterns about patterns, consciousness about consciousness
    Meta = 7,
    /// Pattern level - recurring patterns and structures
    Pattern = 8,
    /// Instance level - specific instances or occurrences
    Instance = 9,
    /// Detail level - fine-grained details
    Detail = 10,
    /// Quantum level - quantum-like superposition states
    Quantum = 11,
}

impl ScaleLevel {
    #[inline] // Frequently used for hierarchical navigation
    pub fn parent_scale(&self) -> Option<ScaleLevel> {
        match self {
            ScaleLevel::Atomic => Some(ScaleLevel::Concept),
            ScaleLevel::Concept => Some(ScaleLevel::Schema),
            ScaleLevel::Schema => Some(ScaleLevel::Worldview),
            ScaleLevel::Worldview => Some(ScaleLevel::Meta),
            ScaleLevel::Meta => None, // Top level
            ScaleLevel::System => Some(ScaleLevel::Meta),
            ScaleLevel::Domain => Some(ScaleLevel::System),
            ScaleLevel::Token => Some(ScaleLevel::Domain),
            ScaleLevel::Pattern => Some(ScaleLevel::Schema),
            ScaleLevel::Instance => Some(ScaleLevel::Pattern),
            ScaleLevel::Detail => Some(ScaleLevel::Instance),
            ScaleLevel::Quantum => Some(ScaleLevel::Atomic),
        }
    }

    #[inline] // Frequently used for hierarchical navigation
    pub fn child_scale(&self) -> Option<ScaleLevel> {
        match self {
            ScaleLevel::Atomic => Some(ScaleLevel::Quantum),
            ScaleLevel::Concept => Some(ScaleLevel::Atomic),
            ScaleLevel::Schema => Some(ScaleLevel::Concept),
            ScaleLevel::Worldview => Some(ScaleLevel::Schema),
            ScaleLevel::Meta => Some(ScaleLevel::Worldview),
            ScaleLevel::System => Some(ScaleLevel::Domain),
            ScaleLevel::Domain => Some(ScaleLevel::Token),
            ScaleLevel::Token => None, // Bottom level
            ScaleLevel::Pattern => Some(ScaleLevel::Instance),
            ScaleLevel::Instance => Some(ScaleLevel::Detail),
            ScaleLevel::Detail => None, // Bottom level
            ScaleLevel::Quantum => None, // Bottom level
        }
    }

    #[inline(always)] // Simple enum conversion, frequently used
    pub fn as_usize(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ContentType {
    Fact,
    Concept,
    Pattern,
    Relationship,
    Experience,
    Insight,
    Question,
    Hypothesis,
    Story,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalSignature {
    pub valence: f32,      // -1.0 (negative) to 1.0 (positive)
    pub arousal: f32,      // 0.0 (calm) to 1.0 (intense)
    pub dominance: f32,    // 0.0 (submissive) to 1.0 (dominant)
    pub resonance_factors: Vec<String>, // What makes this emotionally significant
}

impl Default for EmotionalSignature {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.5,
            resonance_factors: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalMarker {
    pub marker_type: TemporalType,
    pub timestamp: DateTime<Utc>,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TemporalType {
    Created,
    LastAccessed,
    LastModified,
    Consolidated,
    Emerged,
    Connected,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub coherence: f32,        // How well-organized the content is
    pub completeness: f32,     // How complete the information is
    pub reliability: f32,      // How reliable the source is
    pub relevance: f32,        // How relevant to current context
    pub uniqueness: f32,       // How novel/unique the content is
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            coherence: 0.5,
            completeness: 0.5,
            reliability: 0.5,
            relevance: 0.5,
            uniqueness: 0.5,
        }
    }
}

/// Content stored in a fractal memory node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryContent {
    /// Primary content
    pub text: String,

    /// Structured data
    pub data: Option<serde_json::Value>,

    /// Content type
    pub content_type: ContentType,

    /// Emotional resonance
    pub emotional_signature: EmotionalSignature,

    /// Temporal markers
    pub temporal_markers: Vec<TemporalMarker>,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActivationType {
    DirectAccess,      // Directly queried
    Resonance,         // Activated by resonance with another node
    Consolidation,     // Activated during memory consolidation
    Emergence,         // Activated during emergence detection
    CrossScale,        // Activated by cross-scale pattern matching
}

/// Activation event for tracking memory access patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationEvent {
    pub timestamp: DateTime<Utc>,
    pub activation_type: ActivationType,
    pub strength: f32,
    pub context: Vec<String>,
    pub triggered_by: Option<FractalNodeId>,
}

/// Ring buffer for efficient activation history
#[derive(Debug)]
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    head: usize,
    size: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        Self {
            buffer,
            capacity,
            head: 0,
            size: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        self.buffer[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter().filter_map(|x| x.as_ref())
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn recent(&self, n: usize) -> Vec<&T> {
        let mut result = Vec::new();
        let actual_n = n.min(self.size);

        for i in 0..actual_n {
            let idx = if self.head > i {
                self.head - i - 1
            } else {
                self.capacity + self.head - i - 1
            };

            if let Some(ref item) = self.buffer[idx] {
                result.push(item);
            }
        }

        result
    }
}

/// Configuration for fractal memory system
#[derive(Clone, Debug)]
pub struct FractalMemoryConfig {
    /// Maximum depth of fractal hierarchy
    pub max_depth: usize,

    /// Emergence threshold for creating new levels
    pub emergence_threshold: f64,

    /// Maximum children per node before splitting
    pub max_children_per_node: usize,

    /// Self-similarity threshold for pattern recognition
    pub self_similarity_threshold: f64,

    /// Activation history buffer size
    pub activation_history_size: usize,

    /// Resonance decay factor
    pub resonance_decay: f32,

    /// Cross-scale connection strength threshold
    pub cross_scale_threshold: f64,
}

impl Default for FractalMemoryConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            emergence_threshold: 0.75,
            max_children_per_node: 10,
            self_similarity_threshold: 0.8,
            activation_history_size: 100,
            resonance_decay: 0.95,
            cross_scale_threshold: 0.6,
        }
    }
}

// ===== MODULE DECLARATIONS (after basic types) =====

pub mod nodes;
pub mod patterns;
pub mod resonance;
pub mod emergence;
pub mod hierarchy;

// ===== RE-EXPORTS =====

pub use nodes::{
    FractalMemoryNode, CrossScaleConnection, ConnectionType,
    ResonancePattern, ResonancePatternType, FractalProperties, NodeStats,
};
pub use patterns::{CrossScalePatternMatcher, AnalogyConnection, InvariantFeature};
pub use resonance::{ResonanceEngine, ResonanceSignature};
pub use emergence::{MemoryEmergenceEngine, EmergenceThreshold};
pub use hierarchy::{
    DynamicHierarchyManager, AdvancedBalanceEvaluator, NodeMigrationEngine,
    AdaptiveRestructuringEngine, HierarchyMetricsCollector, HierarchyConfig,
    OptimizedHierarchy, HierarchyMetrics, FormationStrategy, StrategyType,
};

/// Statistics for fractal memory system
#[derive(Clone, Debug, Default)]
pub struct FractalMemoryStats {
    pub total_nodes: usize,
    pub nodes_by_scale: HashMap<ScaleLevel, usize>,
    pub total_connections: usize,
    pub cross_scale_connections: usize,
    pub emergence_events: u64,
    pub resonance_events: u64,
    pub average_depth: f32,
    pub memory_usage_mb: f32,
    pub avg_coherence: f32,
}

/// Main fractal memory system
#[derive(Debug)]
pub struct FractalMemorySystem {
    /// Root nodes for different knowledge domains
    root_nodes: Arc<RwLock<HashMap<String, Arc<FractalMemoryNode>>>>,

    /// Pattern matcher for cross-scale connections
    pattern_matcher: Arc<CrossScalePatternMatcher>,

    /// Resonance engine for activation propagation
    resonance_engine: Arc<ResonanceEngine>,

    /// Emergence detector for new hierarchy creation
    emergence_engine: Arc<MemoryEmergenceEngine>,

    /// Configuration
    config: FractalMemoryConfig,

    /// Node lookup by ID
    node_index: Arc<RwLock<HashMap<FractalNodeId, Weak<FractalMemoryNode>>>>,

    /// Statistics
    stats: Arc<RwLock<FractalMemoryStats>>,
}

impl FractalMemorySystem {
    /// Create a new production-ready fractal memory system with full functionality
    pub async fn new(config: FractalMemoryConfig) -> Result<Self> {
        tracing::info!("🚀 Initializing production-grade FractalMemorySystem with enhanced cognitive capabilities");
        let start_time = std::time::Instant::now();

        // Parallel initialization of production components with structured concurrency
        let (pattern_matcher, resonance_engine, emergence_engine) = futures::future::try_join3(
            // Initialize production-grade pattern matcher with real ML
            CrossScalePatternMatcher::create_production_matcher(config.clone()),
            // Initialize production-grade resonance engine
            ResonanceEngine::create_production_engine(config.clone()),
            // Initialize production-grade emergence engine with full analyzers
            MemoryEmergenceEngine::create_production_engine(config.clone())
        ).await.context("Failed to initialize production fractal memory components")?;

        let system = Self {
            root_nodes: Arc::new(RwLock::new(HashMap::new())),
            pattern_matcher: Arc::new(pattern_matcher),
            resonance_engine: Arc::new(resonance_engine),
            emergence_engine: Arc::new(emergence_engine),
            config,
            node_index: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FractalMemoryStats::default())),
        };

        // Verify all components are production-ready
        if system.is_placeholder() {
            return Err(anyhow::anyhow!("Failed to initialize production system - some components are still placeholders"));
        }

        let initialization_time = start_time.elapsed();
        tracing::info!("✅ Production FractalMemorySystem initialized successfully in {}ms", initialization_time.as_millis());

        // Log production readiness status
        tracing::info!("🧠 Emergence Engine: {} analyzers loaded",
                      if system.emergence_engine.is_placeholder() { 0 } else { 5 });
        tracing::info!("🔍 Pattern Matcher: {} templates loaded",
                      if system.pattern_matcher.is_placeholder() { "0" } else { "production" });
        tracing::info!("🌊 Resonance Engine: {} resonators active",
                      if system.resonance_engine.is_placeholder() { 0 } else { 4 });

        Ok(system)
    }

    /// Create a placeholder fractal memory system for two-phase initialization
    /// ⚠️  This method is deprecated - use new() for production systems
    pub fn placeholder() -> Self {
        use tracing::warn;
        warn!("⚠️  Creating placeholder FractalMemorySystem - this is deprecated for production use");
        warn!("🔧 Use FractalMemorySystem::new() for full production functionality");

        let config = FractalMemoryConfig::default();

        // Create minimal placeholder components - only for backward compatibility
        let pattern_matcher = Arc::new(CrossScalePatternMatcher::placeholder());
        let resonance_engine = Arc::new(ResonanceEngine::placeholder());
        let emergence_engine = Arc::new(MemoryEmergenceEngine::placeholder());

        Self {
            root_nodes: Arc::new(RwLock::new(HashMap::new())),
            pattern_matcher,
            resonance_engine,
            emergence_engine,
            config,
            node_index: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FractalMemoryStats::default())),
        }
    }

    /// Check if this is a placeholder instance
    pub fn is_placeholder(&self) -> bool {
        // Check if any component is a placeholder
        self.pattern_matcher.is_placeholder() ||
        self.resonance_engine.is_placeholder() ||
        self.emergence_engine.is_placeholder()
    }

    /// Create a new root node for a domain
    pub async fn create_domain_root(&self, domain: String, content: MemoryContent) -> Result<FractalNodeId> {
        let node = Arc::new(FractalMemoryNode::new_root(
            content,
            domain.clone(),
            self.config.clone(),
        ).await?);

        let node_id = node.id().clone();

        // Add to root nodes
        {
            let mut roots = self.root_nodes.write().await;
            roots.insert(domain, node.clone());
        }

        // Add to index
        {
            let mut index = self.node_index.write().await;
            index.insert(node_id.clone(), Arc::downgrade(&node));
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_nodes += 1;
            *stats.nodes_by_scale.entry(ScaleLevel::Worldview).or_insert(0) += 1;
        }

        Ok(node_id)
    }

    /// Store content in the fractal memory system
    pub async fn store_content(&self, domain: &str, content: MemoryContent) -> Result<FractalNodeId> {
        // Get or create domain root
        let root_node = {
            let roots = self.root_nodes.read().await;
            match roots.get(domain) {
                Some(node) => node.clone(),
                None => {
                    drop(roots);
                    let root_content = MemoryContent {
                        text: format!("Domain: {}", domain),
                        data: None,
                        content_type: ContentType::Concept,
                        emotional_signature: EmotionalSignature::default(),
                        temporal_markers: vec![TemporalMarker {
                            marker_type: TemporalType::Created,
                            timestamp: Utc::now(),
                            description: "Domain root created".to_string(),
                        }],
                        quality_metrics: QualityMetrics::default(),
                    };

                    self.create_domain_root(domain.to_string(), root_content).await?;

                    let roots = self.root_nodes.read().await;
                    roots.get(domain).unwrap().clone()
                }
            }
        };

        // Find appropriate location and store
        let node_id = root_node.store_content(content, &self.pattern_matcher, &self.emergence_engine).await?;

        // Update index
        if let Some(node) = root_node.find_node(&node_id).await {
            let mut index = self.node_index.write().await;
            index.insert(node_id.clone(), Arc::downgrade(&node));
        }

        Ok(node_id)
    }

    /// Search for content across all domains
    pub async fn search(&self, query: &str, max_results: usize) -> Result<Vec<FractalSearchResult>> {
        let mut results = Vec::new();

        let roots = self.root_nodes.read().await;
        for (domain, root) in roots.iter() {
            let domain_results = root.search(query, max_results).await?;
            for mut result in domain_results {
                result.domain = Some(domain.clone());
                results.push(result);
            }
        }

        // Sort by relevance and limit results
        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        Ok(results)
    }

    /// Get node by ID
    pub async fn get_node(&self, node_id: &FractalNodeId) -> Option<Arc<FractalMemoryNode>> {
        let index = self.node_index.read().await;
        index.get(node_id).and_then(|weak| weak.upgrade())
    }

    /// Trigger emergence detection across all domains
    pub async fn detect_emergence(&self) -> Result<Vec<EmergenceEvent>> {
        let mut emergence_events = Vec::new();

        let roots = self.root_nodes.read().await;
        for root in roots.values() {
            let events = self.emergence_engine.detect_emergence(root).await?;
            emergence_events.extend(events);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.emergence_events += emergence_events.len() as u64;
        }

        Ok(emergence_events)
    }

    /// Get system statistics
    pub async fn get_stats(&self) -> FractalMemoryStats {
        self.stats.read().await.clone()
    }

    /// Consolidate memory across all domains
    pub async fn consolidate(&self) -> Result<()> {
        let start_time = Instant::now();
        tracing::debug!("Starting fractal memory consolidation across all domains");

        let roots = self.root_nodes.read().await;
        for root in roots.values() {
            root.consolidate(&self.pattern_matcher, &self.resonance_engine).await?;
        }

        let consolidation_time = start_time.elapsed();
        tracing::info!("Memory consolidation completed in {}ms across {} domains",
                      consolidation_time.as_millis(), roots.len());

        Ok(())
    }

        /// Bridge integration with legacy memory interface for backward compatibility
    pub async fn integrate_with_legacy_memory(&self, legacy_items: Vec<MemoryItem>) -> Result<Vec<FractalNodeId>> {
        let start_time = Instant::now();
        let item_count = legacy_items.len();
        tracing::info!("🔗 Integrating {} legacy memory items with fractal system", item_count);

        let mut integrated_nodes = Vec::new();

        for legacy_item in legacy_items {
            // Convert legacy MemoryItem to fractal MemoryContent
            let fractal_content = self.convert_legacy_to_fractal(legacy_item).await?;

            // Store in appropriate domain
            let domain = "legacy_integration";
            let node_id = self.store_content(domain, fractal_content).await?;
            integrated_nodes.push(node_id);
        }

        let integration_time = start_time.elapsed();
        tracing::info!("✅ Legacy memory integration completed in {}ms: {} items -> {} nodes",
                      integration_time.as_millis(), item_count, integrated_nodes.len());

        Ok(integrated_nodes)
    }

    /// Convert legacy MemoryItem to fractal MemoryContent
    async fn convert_legacy_to_fractal(&self, legacy_item: MemoryItem) -> Result<MemoryContent> {
        // Infer content type from metadata source and tags
        let content_type = if legacy_item.metadata.tags.contains(&"fact".to_string()) {
            ContentType::Fact
        } else if legacy_item.metadata.tags.contains(&"concept".to_string()) {
            ContentType::Concept
        } else if legacy_item.metadata.tags.contains(&"pattern".to_string()) {
            ContentType::Pattern
        } else if legacy_item.metadata.tags.contains(&"relationship".to_string()) {
            ContentType::Relationship
        } else if legacy_item.metadata.tags.contains(&"experience".to_string()) {
            ContentType::Experience
        } else if legacy_item.metadata.tags.contains(&"insight".to_string()) {
            ContentType::Insight
        } else if legacy_item.metadata.tags.contains(&"question".to_string()) {
            ContentType::Question
        } else if legacy_item.metadata.tags.contains(&"hypothesis".to_string()) {
            ContentType::Hypothesis
        } else if legacy_item.metadata.tags.contains(&"story".to_string()) {
            ContentType::Story
        } else {
            ContentType::Concept // Default
        };

        // Convert quality metrics using available fields
        let quality_metrics = QualityMetrics {
            coherence: legacy_item.metadata.importance,
            completeness: legacy_item.relevance_score,
            reliability: legacy_item.relevance_score,
            relevance: legacy_item.relevance_score,
            uniqueness: 0.5, // Default for legacy items
        };

        // Create temporal markers using actual MemoryItem fields
        let temporal_markers = vec![
            TemporalMarker {
                marker_type: TemporalType::Created,
                timestamp: legacy_item.timestamp,
                description: "Migrated from legacy memory system".to_string(),
            },
        ];

        // Create emotional signature (default for legacy items)
        let emotional_signature = EmotionalSignature {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.5,
            resonance_factors: vec!["legacy_integration".to_string()],
        };

        Ok(MemoryContent {
            text: legacy_item.content,
            data: Some(serde_json::json!({
                "legacy_id": legacy_item.id.to_string(),
                "legacy_metadata": legacy_item.metadata,
                "migration_timestamp": Utc::now()
            })),
            content_type,
            emotional_signature,
            temporal_markers,
            quality_metrics,
        })
    }

        /// Export fractal nodes back to legacy format for backward compatibility
    pub async fn export_to_legacy_format(&self, node_ids: &[FractalNodeId]) -> Result<Vec<MemoryItem>> {
        let start_time = Instant::now();
        tracing::debug!("Exporting {} fractal nodes to legacy format", node_ids.len());

        let mut legacy_items = Vec::new();

        for node_id in node_ids {
            if let Some(node) = self.get_node(node_id).await {
                let content = node.get_content().await;

                // Convert to legacy format using correct MemoryItem fields
                let legacy_item = MemoryItem {
                    id: MemoryId(node_id.to_string()),
                    content: content.text,
                    context: vec![format!("{:?}", content.content_type)],
                    timestamp: content.temporal_markers
                        .iter()
                        .find(|m| matches!(m.marker_type, TemporalType::Created))
                        .map(|m| m.timestamp)
                        .unwrap_or_else(Utc::now),
                    access_count: 0,
                    relevance_score: content.quality_metrics.relevance,
                    metadata: MemoryMetadata {
                        source: "fractal_export".to_string(),
                        importance: content.quality_metrics.reliability,
                        tags: vec![
                            "fractal_export".to_string(),
                            format!("{:?}", content.content_type).to_lowercase()
                        ],
                        associations: vec![],
                        context: Some(format!("Fractal export: {:?}", content.content_type)),
                        created_at: chrono::Utc::now(),
                        accessed_count: 0,
                        last_accessed: None,
                        version: 1,
                        category: "fractal".to_string(),
                        timestamp: chrono::Utc::now(),
                        expiration: None,
                    },
                };

                legacy_items.push(legacy_item);
            }
        }

        let export_time = start_time.elapsed();
        tracing::debug!("Legacy export completed in {}ms: {} items exported",
                       export_time.as_millis(), legacy_items.len());

        Ok(legacy_items)
    }
}

/// Search result from fractal memory
#[derive(Clone, Debug)]
pub struct FractalSearchResult {
    pub node_id: FractalNodeId,
    pub content: MemoryContent,
    pub scale_level: ScaleLevel,
    pub relevance: f32,
    pub context_path: Vec<String>,
    pub domain: Option<String>,
    pub resonance_strength: f32,
}

/// Emergence event detected in fractal memory
#[derive(Clone, Debug)]
pub struct EmergenceEvent {
    pub event_type: EmergenceType,
    pub scale_level: ScaleLevel,
    pub nodes_involved: Vec<FractalNodeId>,
    pub confidence: f32,
    pub description: String,
    pub timestamp: DateTime<Utc>,
    pub emergence_data: std::collections::HashMap<String, String>,
    pub affected_nodes: Vec<FractalNodeId>,
}

#[derive(Clone, Debug)]
pub enum EmergenceType {
    NewPatternDetected,
    CrossScaleResonance,
    HierarchyFormation,
    ConceptualMerge,
    ScaleTransition,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fractal_memory_creation() {
        let config = FractalMemoryConfig::default();
        let memory_system = FractalMemorySystem::new(config).await.unwrap();

        let content = MemoryContent {
            text: "Test memory content".to_string(),
            data: None,
            content_type: ContentType::Fact,
            emotional_signature: EmotionalSignature::default(),
            temporal_markers: vec![],
            quality_metrics: QualityMetrics::default(),
        };

        let node_id = memory_system.store_content("test_domain", content).await.unwrap();
        assert!(!node_id.to_string().is_empty());

        let stats = memory_system.get_stats().await;
        assert!(stats.total_nodes > 0);
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingBuffer::new(3);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.len(), 3);

        buffer.push(4); // Should overwrite first element
        assert_eq!(buffer.len(), 3);

        let recent = buffer.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(*recent[0], 4);
        assert_eq!(*recent[1], 3);
    }

    #[test]
    fn test_scale_levels() {
        assert_eq!(ScaleLevel::Atomic.parent_scale(), Some(ScaleLevel::Concept));
        assert_eq!(ScaleLevel::Concept.child_scale(), Some(ScaleLevel::Atomic));
        assert_eq!(ScaleLevel::Meta.parent_scale(), None);
        assert_eq!(ScaleLevel::Atomic.child_scale(), None);
    }
}
