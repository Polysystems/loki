//! Fractal Memory Node Implementation
//!
//! Implements true self-similar memory structures with hierarchical organization,
//! cross-scale connections, and emergent pattern formation.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, Weak};
use std::time::Instant;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

// Import from parent module (mod.rs) - basic types are now defined there
use super::{
    FractalNodeId, ScaleLevel, MemoryContent, FractalMemoryConfig,
    ActivationEvent, ActivationType, RingBuffer,
    ContentType, EmotionalSignature, TemporalMarker, TemporalType, QualityMetrics,
    FractalSearchResult,
};

// Import from sibling modules
use super::patterns::{CrossScalePatternMatcher, AnalogyConnection, InvariantFeature};
use super::resonance::ResonanceEngine;
use super::emergence::MemoryEmergenceEngine;

// Import from other crate modules

/// Core fractal memory node with full self-similar structure
#[derive(Clone, Debug)]
pub struct FractalMemoryNode {
    /// Unique identifier
    id: FractalNodeId,

    /// Content stored in this node
    content: Arc<RwLock<MemoryContent>>,

    /// Scale level of this node
    scale_level: ScaleLevel,

    /// Domain this node belongs to
    domain: String,

    /// Parent node (weak reference to prevent cycles)
    parent: Arc<RwLock<Option<Weak<FractalMemoryNode>>>>,

    /// Child nodes (self-similar structure)
    children: Arc<RwLock<BTreeMap<FractalNodeId, Arc<FractalMemoryNode>>>>,

    /// Cross-scale connections (analogies and invariant features)
    cross_scale_connections: Arc<RwLock<Vec<CrossScaleConnection>>>,

    /// Activation history for resonance patterns
    activation_history: Arc<RwLock<RingBuffer<ActivationEvent>>>,

    /// Resonance patterns with other nodes
    resonance_patterns: Arc<RwLock<HashMap<String, ResonancePattern>>>,

    /// Fractal properties (self-similarity, emergence, etc.)
    fractal_properties: Arc<RwLock<FractalProperties>>,

    /// Node statistics and metrics
    node_stats: Arc<RwLock<NodeStats>>,

    /// Configuration
    config: FractalMemoryConfig,

    /// Creation timestamp
    created_at: Instant,
}

/// Cross-scale connection between nodes at different levels
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossScaleConnection {
    pub connection_id: String,
    pub target_node_id: FractalNodeId,
    pub target_scale: ScaleLevel,
    pub connection_type: ConnectionType,
    pub strength: f64,
    pub analogy_mapping: Option<AnalogyConnection>,
    pub invariant_features: Vec<InvariantFeature>,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub last_activated: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    /// Same pattern at different scales
    ScaleInvariant,
    /// Functional similarity across scales
    FunctionalAnalogy,
    /// Structural similarity across scales
    StructuralAnalogy,
    /// Causal relationship across scales
    CausalMapping,
    /// Emergent property connection
    EmergentProperty,
    /// Resonance-based connection
    ResonanceAlignment,
}

impl std::fmt::Display for ConnectionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionType::ScaleInvariant => write!(f, "scale-invariant"),
            ConnectionType::FunctionalAnalogy => write!(f, "functional-analogy"),
            ConnectionType::StructuralAnalogy => write!(f, "structural-analogy"),
            ConnectionType::CausalMapping => write!(f, "causal-mapping"),
            ConnectionType::EmergentProperty => write!(f, "emergent-property"),
            ConnectionType::ResonanceAlignment => write!(f, "resonance-alignment"),
        }
    }
}

/// Resonance pattern between nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResonancePattern {
    pub pattern_id: String,
    pub participating_nodes: Vec<FractalNodeId>,
    pub resonance_frequency: f64,
    pub synchronization_level: f64,
    pub pattern_type: ResonancePatternType,
    pub stability: f32,
    pub last_update: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ResonancePatternType {
    Harmonic,        // Nodes resonate at harmonic frequencies
    Synchronous,     // Nodes resonate in perfect sync
    Antiphase,       // Nodes resonate in opposition
    Chaotic,         // Complex, non-linear resonance
    Emergent,        // Spontaneous resonance pattern
}

/// Comprehensive fractal properties
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FractalProperties {
    /// How similar this node's structure is to its parent
    pub self_similarity_score: f64,

    /// Threshold for triggering emergence of new levels
    pub emergence_threshold: f64,

    /// Current depth in the hierarchy
    pub hierarchy_depth: usize,

    /// How well this node's children are organized
    pub sibling_coherence: f64,

    /// Strength of connections across scales
    pub cross_scale_resonance: f64,

    /// Complexity measure of contained patterns
    pub pattern_complexity: f64,

    /// How stable this node is over time
    pub temporal_stability: f64,

    /// Information entropy of this node's content and connections
    pub information_entropy: f64,

    /// Measure of emergent properties
    pub emergence_potential: f64,

    /// How many scales this node spans connections across
    pub scale_spanning: usize,
}

/// Node metadata for fractal memory nodes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Node creation source
    pub source: String,
    /// Tags associated with the node
    pub tags: Vec<String>,
    /// Importance score
    pub importance: f32,
    /// Node version
    pub version: u32,
    /// Additional attributes
    pub attributes: HashMap<String, String>,
    /// Node domain
    pub domain: String,
    /// Scale level
    pub scale_level: String,
    /// Last modification timestamp
    pub last_modified: DateTime<Utc>,
}

/// Comprehensive node statistics
#[derive(Clone, Debug)]
pub struct NodeStats {
    /// Number of times this node has been accessed
    pub access_count: u64,

    /// Last time this node was accessed
    pub last_access: Option<Instant>,

    /// When this node was created
    pub creation_time: Instant,

    /// Number of times content has been modified
    pub modification_count: u32,

    /// Number of resonance events involving this node
    pub resonance_events: u32,

    /// Number of times this node triggered emergence
    pub emergence_triggers: u32,

    /// Cross-scale activations through this node
    pub cross_scale_activations: u32,

    /// Average activation strength
    pub average_activation_strength: f32,

    /// Number of child nodes
    pub child_count: usize,

    /// Number of cross-scale connections
    pub cross_scale_connection_count: usize,

    /// Total information processed through this node
    pub information_throughput: u64,

    /// Quality score based on various metrics
    pub quality_score: f32,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            access_count: 0,
            last_access: None,
            creation_time: Instant::now(),
            modification_count: 0,
            resonance_events: 0,
            emergence_triggers: 0,
            cross_scale_activations: 0,
            average_activation_strength: 0.0,
            child_count: 0,
            cross_scale_connection_count: 0,
            information_throughput: 0,
            quality_score: 0.5,
        }
    }
}

impl FractalMemoryNode {
    /// Create a new fractal memory node (simplified constructor for compatibility)
    pub fn new(
        id: String,
        content: String,
        _metadata: HashMap<String, String>,
    ) -> Self {
        let memory_content = MemoryContent {
            text: content,
            data: None,
            content_type: ContentType::Fact,
            emotional_signature: EmotionalSignature {
                valence: 0.0,
                arousal: 0.0,
                dominance: 0.0,
                resonance_factors: Vec::new(),
            },
            temporal_markers: vec![TemporalMarker {
                marker_type: TemporalType::Created,
                timestamp: Utc::now(),
                description: "Node created".to_string(),
            }],
            quality_metrics: QualityMetrics {
                coherence: 0.7,
                completeness: 0.8,
                reliability: 0.9,
                relevance: 0.8,
                uniqueness: 0.5,
            },
        };

        let fractal_properties = FractalProperties {
            self_similarity_score: 0.5,
            emergence_threshold: 0.8,
            hierarchy_depth: 0,
            sibling_coherence: 0.5,
            cross_scale_resonance: 0.0,
            pattern_complexity: 0.5,
            temporal_stability: 0.8,
            information_entropy: 0.0,
            emergence_potential: 0.6,
            scale_spanning: 1,
        };

        Self {
            id: FractalNodeId::from_string(id),
            content: Arc::new(RwLock::new(memory_content)),
            scale_level: ScaleLevel::Concept,
            domain: "default".to_string(),
            parent: Arc::new(RwLock::new(None)),
            children: Arc::new(RwLock::new(BTreeMap::new())),
            cross_scale_connections: Arc::new(RwLock::new(Vec::new())),
            activation_history: Arc::new(RwLock::new(RingBuffer::new(100))),
            resonance_patterns: Arc::new(RwLock::new(HashMap::new())),
            fractal_properties: Arc::new(RwLock::new(fractal_properties)),
            node_stats: Arc::new(RwLock::new(NodeStats::default())),
            config: FractalMemoryConfig::default(),
            created_at: Instant::now(),
        }
    }

    /// Create a new root node for a domain
    pub async fn new_root(
        content: MemoryContent,
        domain: String,
        config: FractalMemoryConfig,
    ) -> Result<Self> {
        let id = FractalNodeId::from_content(&content.text);

        let fractal_properties = FractalProperties {
            self_similarity_score: 1.0, // Root is perfectly self-similar to itself
            emergence_threshold: config.emergence_threshold,
            hierarchy_depth: 0,
            sibling_coherence: 1.0,
            cross_scale_resonance: 0.0,
            pattern_complexity: 0.5,
            temporal_stability: 1.0,
            information_entropy: 0.0,
            emergence_potential: 0.8,
            scale_spanning: 1,
        };

        let node = Self {
            id,
            content: Arc::new(RwLock::new(content)),
            scale_level: ScaleLevel::Worldview, // Root nodes are worldview level
            domain,
            parent: Arc::new(RwLock::new(None)),
            children: Arc::new(RwLock::new(BTreeMap::new())),
            cross_scale_connections: Arc::new(RwLock::new(Vec::new())),
            activation_history: Arc::new(RwLock::new(RingBuffer::new(config.activation_history_size))),
            resonance_patterns: Arc::new(RwLock::new(HashMap::new())),
            fractal_properties: Arc::new(RwLock::new(fractal_properties)),
            node_stats: Arc::new(RwLock::new(NodeStats::default())),
            config,
            created_at: Instant::now(),
        };

        Ok(node)
    }

    /// Create a child node with self-similar structure
    pub async fn create_child(
        parent: Arc<Self>,
        content: MemoryContent,
        suggested_scale: Option<ScaleLevel>,
    ) -> Result<Arc<Self>> {
        let id = FractalNodeId::from_content(&content.text);

        // Determine child scale level
        let child_scale = suggested_scale.unwrap_or_else(|| {
            parent.scale_level.child_scale().unwrap_or(ScaleLevel::Atomic)
        });

        // Calculate hierarchy depth
        let parent_props = parent.fractal_properties.read().await;
        let hierarchy_depth = parent_props.hierarchy_depth + 1;
        drop(parent_props);

        // Initialize fractal properties with inheritance from parent
        let parent_similarity = parent.calculate_content_similarity(&content).await?;

        let fractal_properties = FractalProperties {
            self_similarity_score: parent_similarity,
            emergence_threshold: parent.config.emergence_threshold,
            hierarchy_depth,
            sibling_coherence: 0.5, // Will be calculated based on siblings
            cross_scale_resonance: 0.0,
            pattern_complexity: content.quality_metrics.coherence as f64,
            temporal_stability: 0.8,
            information_entropy: Self::calculate_information_entropy(&content),
            emergence_potential: 0.6,
            scale_spanning: 1,
        };

        let child = Arc::new(Self {
            id: id.clone(),
            content: Arc::new(RwLock::new(content)),
            scale_level: child_scale,
            domain: parent.domain.clone(),
            parent: Arc::new(RwLock::new(Some(Arc::downgrade(&parent)))),
            children: Arc::new(RwLock::new(BTreeMap::new())),
            cross_scale_connections: Arc::new(RwLock::new(Vec::new())),
            activation_history: Arc::new(RwLock::new(RingBuffer::new(parent.config.activation_history_size))),
            resonance_patterns: Arc::new(RwLock::new(HashMap::new())),
            fractal_properties: Arc::new(RwLock::new(fractal_properties)),
            node_stats: Arc::new(RwLock::new(NodeStats::default())),
            config: parent.config.clone(),
            created_at: Instant::now(),
        });

        // Add child to parent
        {
            let mut parent_children = parent.children.write().await;
            parent_children.insert(id.clone(), child.clone());
        }

        // Update parent statistics
        {
            let mut parent_stats = parent.node_stats.write().await;
            parent_stats.child_count = parent.children.read().await.len();
        }

        // Trigger sibling coherence recalculation
        parent.update_sibling_coherence().await?;

        Ok(child)
    }

    /// Store content in this node or appropriate child
    pub async fn store_content(
        &self,
        content: MemoryContent,
        pattern_matcher: &CrossScalePatternMatcher,
        emergence_engine: &MemoryEmergenceEngine,
    ) -> Result<FractalNodeId> {
        // Record access
        self.record_access(ActivationType::DirectAccess, 1.0, vec!["store_content".to_string()]).await?;

        // Check if content should be stored here or in a child
        let content_scale = self.determine_content_scale(&content).await?;

        if content_scale == self.scale_level {
            // Store as sibling by creating new child under parent
            if let Some(parent_weak) = &*self.parent.read().await {
                if let Some(parent) = parent_weak.upgrade() {
                    let child = Self::create_child(parent, content, Some(content_scale)).await?;
                    return Ok(child.id.clone());
                }
            }

            // If no parent, create as child of this node at lower scale
            if let Some(child_scale) = self.scale_level.child_scale() {
                let child = Self::create_child(Arc::new(self.clone()), content, Some(child_scale)).await?;
                return Ok(child.id.clone());
            }
        }

        // Find best matching child or create new one
        let best_child = self.find_best_matching_child(&content, pattern_matcher).await?;

        match best_child {
            Some(child) => {
                // Recursively store in child with Box::pin for async recursion
                Box::pin(child.store_content(content, pattern_matcher, emergence_engine)).await
            }
            None => {
                // Create new child
                let child_scale = self.scale_level.child_scale().unwrap_or(ScaleLevel::Atomic);
                let child = Self::create_child(Arc::new(self.clone()), content, Some(child_scale)).await?;

                // Check for emergence after adding child
                if self.children.read().await.len() > self.config.max_children_per_node {
                    emergence_engine.should_trigger_emergence(self).await?;
                }

                Ok(child.id.clone())
            }
        }
    }

    /// Search for content within this node and children
    pub async fn search(&self, query: &str, max_results: usize) -> Result<Vec<FractalSearchResult>> {
        let mut results = Vec::new();

        // Search this node
        let content = self.content.read().await;
        let relevance = self.calculate_relevance(&content.text, query).await?;

        if relevance > 0.1 {
            results.push(FractalSearchResult {
                node_id: self.id.clone(),
                content: content.clone(),
                scale_level: self.scale_level,
                relevance,
                context_path: self.get_context_path().await?,
                domain: Some(self.domain.clone()),
                resonance_strength: self.get_current_resonance_strength().await?,
            });
        }
        drop(content);

        // Record search access
        self.record_access(ActivationType::DirectAccess, relevance, vec![query.to_string()]).await?;

        // Search children recursively
        let children = self.children.read().await;
        for child in children.values() {
            let child_results = Box::pin(child.search(query, max_results - results.len())).await?;
            results.extend(child_results);

            if results.len() >= max_results {
                break;
            }
        }

        // Sort by relevance and limit
        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        Ok(results)
    }

    /// Find a specific node by ID in this subtree
    pub async fn find_node(&self, node_id: &FractalNodeId) -> Option<Arc<FractalMemoryNode>> {
        if &self.id == node_id {
            return Some(Arc::new(self.clone()));
        }

        // Search children
        let children = self.children.read().await;
        for child in children.values() {
            if let Some(found) = Box::pin(child.find_node(node_id)).await {
                return Some(found);
            }
        }

        None
    }

    /// Consolidate memory in this subtree
    pub async fn consolidate(
        &self,
        pattern_matcher: &CrossScalePatternMatcher,
        resonance_engine: &ResonanceEngine,
    ) -> Result<()> {
        // Update cross-scale connections
        pattern_matcher.update_cross_scale_connections(self).await?;

        // Update resonance patterns
        resonance_engine.update_resonance_patterns(self).await?;

        // Consolidate children
        let children = self.children.read().await;
        for child in children.values() {
            Box::pin(child.consolidate(pattern_matcher, resonance_engine)).await?;
        }

        // Update fractal properties
        self.update_fractal_properties().await?;

        Ok(())
    }

    /// Get comprehensive node statistics
    pub async fn get_stats(&self) -> NodeStats {
        let mut stats = self.node_stats.read().await.clone();

        // Update dynamic stats
        stats.child_count = self.children.read().await.len();
        stats.cross_scale_connection_count = self.cross_scale_connections.read().await.len();

        // Calculate quality score without recursion - use simplified version
        stats.quality_score = self.calculate_simple_quality_score().await.unwrap_or(0.5);

        stats
    }

    /// Calculate simple quality score (non-recursive)
    async fn calculate_simple_quality_score(&self) -> Result<f32> {
        let props = self.fractal_properties.read().await;
        let content = self.content.read().await;

        // Combine quality factors without calling get_stats to avoid recursion
        let structure_quality = (props.sibling_coherence + props.temporal_stability) / 2.0;
        let content_quality = (content.quality_metrics.coherence + content.quality_metrics.completeness + content.quality_metrics.reliability) / 3.0;
        let connection_count = self.cross_scale_connections.read().await.len();
        let connection_quality = if connection_count > 0 { props.cross_scale_resonance } else { 0.5 };

        let overall_quality = (structure_quality * 0.4 + content_quality as f64 * 0.4 + connection_quality * 0.2) as f32;

        Ok(overall_quality.max(0.0).min(1.0))
    }

    /// Get current fractal properties
    pub async fn get_fractal_properties(&self) -> FractalProperties {
        self.fractal_properties.read().await.clone()
    }

    // === PUBLIC ACCESSOR METHODS FOR INTEGRATION ===

    /// Get read-only access to content for pattern analysis
    pub async fn get_content(&self) -> MemoryContent {
        self.content.read().await.clone()
    }

    /// Get current resonance patterns for analysis
    pub async fn get_resonance_patterns(&self) -> HashMap<String, ResonancePattern> {
        self.resonance_patterns.read().await.clone()
    }

    /// Get cross-scale connections for pattern matching
    pub async fn get_cross_scale_connections(&self) -> Vec<CrossScaleConnection> {
        self.cross_scale_connections.read().await.clone()
    }

    /// Get children nodes for tree traversal
    pub async fn get_children(&self) -> Vec<Arc<FractalMemoryNode>> {
        self.children.read().await.values().cloned().collect()
    }

    /// Get parent node if it exists
    pub async fn get_parent(&self) -> Option<Arc<FractalMemoryNode>> {
        let parent_ref = self.parent.read().await;
        parent_ref.as_ref().and_then(|weak| weak.upgrade())
    }

    /// Get activation history for resonance analysis
    pub async fn get_recent_activations(&self, count: usize) -> Vec<ActivationEvent> {
        let history = self.activation_history.read().await;
        history.recent(count).into_iter().cloned().collect()
    }

    /// Add a cross-scale connection (for pattern matcher integration)
    pub async fn add_cross_scale_connection(&self, connection: CrossScaleConnection) -> Result<()> {
        let mut connections = self.cross_scale_connections.write().await;
        connections.push(connection);

        // Update stats
        let mut stats = self.node_stats.write().await;
        stats.cross_scale_connection_count = connections.len();

        Ok(())
    }

    /// Update resonance pattern (for resonance engine integration)
    pub async fn update_resonance_pattern(&self, pattern_id: String, pattern: ResonancePattern) -> Result<()> {
        let mut patterns = self.resonance_patterns.write().await;
        patterns.insert(pattern_id, pattern);

        // Update stats
        let mut stats = self.node_stats.write().await;
        stats.resonance_events += 1;

        Ok(())
    }

    /// Record an activation event (public version)
    pub async fn record_activation(&self, activation_type: ActivationType, strength: f32, context: Vec<String>, triggered_by: Option<FractalNodeId>) -> Result<()> {
        let event = ActivationEvent {
            timestamp: Utc::now(),
            activation_type,
            strength,
            context,
            triggered_by,
        };

        // Add to history
        {
            let mut history = self.activation_history.write().await;
            history.push(event);
        }

        // Update stats
        {
            let mut stats = self.node_stats.write().await;
            stats.access_count += 1;
            stats.last_access = Some(Instant::now());

            // Update average activation strength
            let new_total = stats.average_activation_strength * (stats.access_count - 1) as f32 + strength;
            stats.average_activation_strength = new_total / stats.access_count as f32;
        }

        Ok(())
    }

    /// Create a new child node (public version for emergence engine)
    pub async fn create_child_node(self: &Arc<Self>, content: MemoryContent, suggested_scale: Option<ScaleLevel>) -> Result<Arc<FractalMemoryNode>> {
        Self::create_child(self.clone(), content, suggested_scale).await
    }

    /// Update content (for content synthesis in emergence)
    pub async fn update_content(&self, new_content: MemoryContent) -> Result<()> {
        *self.content.write().await = new_content;

        // Update modification stats
        let mut stats = self.node_stats.write().await;
        stats.modification_count += 1;

        Ok(())
    }

    /// Calculate content similarity with another piece of content (public helper)
    pub async fn calculate_content_similarity_with(&self, other_content: &MemoryContent) -> Result<f64> {
        self.calculate_content_similarity(other_content).await
    }

    /// Reparent this node to a new parent (for emergence reorganization)
    pub async fn reparent_to(&self, new_parent: Option<Arc<FractalMemoryNode>>) -> Result<()> {
        // Remove from current parent if it exists
        if let Some(current_parent) = self.get_parent().await {
            let mut parent_children = current_parent.children.write().await;
            parent_children.remove(&self.id);

            // Update parent stats
            let mut parent_stats = current_parent.node_stats.write().await;
            parent_stats.child_count = parent_children.len();
        }

        // Set new parent
        {
            let mut parent = self.parent.write().await;
            *parent = new_parent.as_ref().map(|p| Arc::downgrade(p));
        }

        // Add to new parent's children
        if let Some(new_parent) = new_parent {
            let mut parent_children = new_parent.children.write().await;
            parent_children.insert(self.id.clone(), Arc::new(self.clone()));

            // Update parent stats
            let mut parent_stats = new_parent.node_stats.write().await;
            parent_stats.child_count = parent_children.len();
        }

        Ok(())
    }

    /// Get node metadata for visualization
    pub async fn get_visualization_metadata(&self) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();

        let stats = self.get_stats().await;
        let props = self.get_fractal_properties().await;
        let content = self.get_content().await;

        metadata.insert("id".to_string(), self.id.to_string());
        metadata.insert("domain".to_string(), self.domain.clone());
        metadata.insert("scale_level".to_string(), format!("{:?}", self.scale_level));
        metadata.insert("access_count".to_string(), stats.access_count.to_string());
        metadata.insert("child_count".to_string(), stats.child_count.to_string());
        metadata.insert("quality_score".to_string(), stats.quality_score.to_string());
        metadata.insert("self_similarity".to_string(), props.self_similarity_score.to_string());
        metadata.insert("pattern_complexity".to_string(), props.pattern_complexity.to_string());
        metadata.insert("emergence_potential".to_string(), props.emergence_potential.to_string());
        metadata.insert("content_type".to_string(), format!("{:?}", content.content_type));
        metadata.insert("emotional_valence".to_string(), content.emotional_signature.valence.to_string());

        Ok(metadata)
    }

    // === PRIVATE HELPER METHODS ===

    /// Record an access event
    async fn record_access(&self, activation_type: ActivationType, strength: f32, context: Vec<String>) -> Result<()> {
        let event = ActivationEvent {
            timestamp: Utc::now(),
            activation_type,
            strength,
            context,
            triggered_by: None,
        };

        // Add to activation history
        {
            let mut history = self.activation_history.write().await;
            history.push(event);
        }

        // Update statistics
        {
            let mut stats = self.node_stats.write().await;
            stats.access_count += 1;
            stats.last_access = Some(Instant::now());
            stats.average_activation_strength =
                (stats.average_activation_strength * (stats.access_count - 1) as f32 + strength) / stats.access_count as f32;
        }

        Ok(())
    }

    /// Calculate content similarity between this node and new content
    async fn calculate_content_similarity(&self, other_content: &MemoryContent) -> Result<f64> {
        let my_content = self.content.read().await;

        // Use multiple similarity measures
        let text_similarity = self.text_similarity(&my_content.text, &other_content.text);
        let type_similarity = if my_content.content_type == other_content.content_type { 1.0 } else { 0.5 };
        let emotional_similarity = self.emotional_similarity(&my_content.emotional_signature, &other_content.emotional_signature);

        // Weighted combination
        let similarity = text_similarity * 0.5 + type_similarity * 0.2 + emotional_similarity * 0.3;

        Ok(similarity)
    }

    /// Text similarity using Jaccard coefficient
    fn text_similarity(&self, text1: &str, text2: &str) -> f64 {
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();
        let words1: std::collections::HashSet<_> = text1_lower.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = text2_lower.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Calculate emotional signature similarity
    fn emotional_similarity(&self, sig1: &super::EmotionalSignature, sig2: &super::EmotionalSignature) -> f64 {
        let valence_diff = (sig1.valence - sig2.valence).abs() as f64;
        let arousal_diff = (sig1.arousal - sig2.arousal).abs() as f64;
        let dominance_diff = (sig1.dominance - sig2.dominance).abs() as f64;

        1.0 - (valence_diff + arousal_diff + dominance_diff) / 3.0
    }

    /// Determine appropriate scale level for content
    async fn determine_content_scale(&self, content: &MemoryContent) -> Result<ScaleLevel> {
        // Analyze content complexity and type to determine scale
        let complexity = content.quality_metrics.coherence as f64;
        let text_length = content.text.len();

        match content.content_type {
            super::ContentType::Fact if text_length < 100 => Ok(ScaleLevel::Atomic),
            super::ContentType::Concept | super::ContentType::Pattern => Ok(ScaleLevel::Concept),
            super::ContentType::Relationship if complexity > 0.7 => Ok(ScaleLevel::Schema),
            super::ContentType::Insight | super::ContentType::Story => Ok(ScaleLevel::Worldview),
            _ => Ok(self.scale_level.child_scale().unwrap_or(ScaleLevel::Atomic))
        }
    }

    /// Find best matching child for new content
    async fn find_best_matching_child(&self, content: &MemoryContent, pattern_matcher: &CrossScalePatternMatcher) -> Result<Option<Arc<FractalMemoryNode>>> {
        let children = self.children.read().await;
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for child in children.values() {
            let child_content = child.content.read().await;
            let similarity = pattern_matcher.calculate_content_similarity(&child_content, content).await?;

            if similarity > best_similarity && similarity > self.config.self_similarity_threshold {
                best_similarity = similarity;
                best_match = Some(child.clone());
            }
        }

        Ok(best_match)
    }

    /// Calculate relevance of this node to a query
    async fn calculate_relevance(&self, content: &str, query: &str) -> Result<f32> {
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();

        // Simple keyword matching - could be enhanced with semantic analysis
        let words: Vec<&str> = query_lower.split_whitespace().collect();
        let matches = words.iter().filter(|word| content_lower.contains(*word)).count();

        let relevance = matches as f32 / words.len() as f32;
        Ok(relevance)
    }

    /// Get context path from root to this node
    async fn get_context_path(&self) -> Result<Vec<String>> {
        let path = vec![self.content.read().await.text.clone()];

        if let Some(parent_weak) = &*self.parent.read().await {
            if let Some(parent) = parent_weak.upgrade() {
                let mut parent_path = Box::pin(parent.get_context_path()).await?;
                parent_path.extend(path);
                return Ok(parent_path);
            }
        }

        Ok(path)
    }

    /// Get current resonance strength
    async fn get_current_resonance_strength(&self) -> Result<f32> {
        let patterns = self.resonance_patterns.read().await;
        let total_strength = patterns.values()
            .map(|p| p.synchronization_level as f32)
            .sum::<f32>();

        let average_strength = if patterns.is_empty() {
            0.0
        } else {
            total_strength / patterns.len() as f32
        };

        Ok(average_strength)
    }

    /// Update sibling coherence based on children
    async fn update_sibling_coherence(&self) -> Result<()> {
        let children = self.children.read().await;

        if children.len() < 2 {
            let mut props = self.fractal_properties.write().await;
            props.sibling_coherence = 1.0;
            return Ok(());
        }

        // Calculate average similarity between all child pairs
        let mut total_similarity = 0.0;
        let mut comparison_count = 0;

        for child1 in children.values() {
            for child2 in children.values() {
                if child1.id != child2.id {
                    let similarity = child1.calculate_content_similarity(&*child2.content.read().await).await?;
                    total_similarity += similarity;
                    comparison_count += 1;
                }
            }
        }

        let average_coherence = if comparison_count > 0 {
            total_similarity / comparison_count as f64
        } else {
            1.0
        };

        {
            let mut props = self.fractal_properties.write().await;
            props.sibling_coherence = average_coherence;
        }

        Ok(())
    }

    /// Update all fractal properties
    async fn update_fractal_properties(&self) -> Result<()> {
        let mut props = self.fractal_properties.write().await;

        // Update hierarchy depth
        props.hierarchy_depth = {
            if let Some(parent_weak) = &*self.parent.read().await {
                if let Some(parent) = parent_weak.upgrade() {
                    parent.fractal_properties.read().await.hierarchy_depth + 1
                } else {
                    0
                }
            } else {
                0
            }
        };

        // Update cross-scale resonance
        props.cross_scale_resonance = {
            let connections = self.cross_scale_connections.read().await;
            if connections.is_empty() {
                0.0
            } else {
                connections.iter().map(|c| c.strength).sum::<f64>() / connections.len() as f64
            }
        };

        // Update scale spanning
        props.scale_spanning = {
            let connections = self.cross_scale_connections.read().await;
            let unique_scales: std::collections::HashSet<_> = connections.iter().map(|c| c.target_scale).collect();
            unique_scales.len() + 1 // +1 for this node's scale
        };

        // Update pattern complexity based on children and connections
        props.pattern_complexity = {
            let child_count = self.children.read().await.len();
            let connection_count = self.cross_scale_connections.read().await.len();
            ((child_count + connection_count) as f64 / 20.0).min(1.0) // Normalize to 0-1
        };

        // Update emergence potential
        props.emergence_potential = {
            let complexity = props.pattern_complexity;
            let resonance = props.cross_scale_resonance;
            let coherence = props.sibling_coherence;
            (complexity + resonance + coherence) / 3.0
        };

        Ok(())
    }

    /// Calculate information entropy
    fn calculate_information_entropy(content: &MemoryContent) -> f64 {
        let text = &content.text;
        let mut char_counts: HashMap<char, usize> = HashMap::new();

        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        let total_chars = text.len() as f64;
        let mut entropy = 0.0;

        for count in char_counts.values() {
            let probability = *count as f64 / total_chars;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        entropy / 8.0 // Normalize to approximate 0-1 range
    }

    // === PUBLIC ACCESSORS ===

    /// Get node ID
    pub fn id(&self) -> &FractalNodeId {
        &self.id
    }

    /// Get scale level
    pub fn scale_level(&self) -> ScaleLevel {
        self.scale_level
    }

    /// Get domain
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// Get children lock for direct access (used in hierarchy traversal)
    pub fn children_lock(&self) -> &Arc<RwLock<BTreeMap<FractalNodeId, Arc<FractalMemoryNode>>>> {
        &self.children
    }

    /// Get number of children
    pub async fn child_count(&self) -> usize {
        self.children.read().await.len()
    }

    /// Get creation time
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Get the depth of this node in the hierarchy
    pub fn get_depth(&self) -> usize {
        // The scale_level field represents the depth in the fractal hierarchy
        match self.scale_level {
            ScaleLevel::Worldview => 0,
            ScaleLevel::Domain => 1,
            ScaleLevel::Concept => 2,
            ScaleLevel::Pattern => 3,
            ScaleLevel::Instance => 4,
            ScaleLevel::Detail => 5,
            ScaleLevel::Atomic => 6,
            ScaleLevel::Quantum => 7,
            ScaleLevel::System => 1,  // System level is similar to Domain
            ScaleLevel::Token => 6,   // Token level is similar to Atomic
            ScaleLevel::Schema => 2,  // Schema level is similar to Concept
            ScaleLevel::Meta => 0,    // Meta level is highest, like Worldview
        }
    }

    /// Get current activation level
    pub async fn get_activation_level(&self) -> f32 {
        let stats = self.node_stats.read().await;
        stats.average_activation_strength
    }

    /// Get total connection count (children + cross-scale connections)
    pub async fn get_connection_count(&self) -> usize {
        let children_count = self.children.read().await.len();
        let cross_scale_count = self.cross_scale_connections.read().await.len();
        children_count + cross_scale_count
    }

    /// Get last access time
    pub async fn get_last_access_time(&self) -> Option<std::time::Instant> {
        let stats = self.node_stats.read().await;
        stats.last_access
    }

    /// Get semantic coherence score
    pub async fn get_semantic_coherence(&self) -> f32 {
        let content = self.content.read().await;
        content.quality_metrics.coherence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::fractal::{ContentType, EmotionalSignature, QualityMetrics};

    #[tokio::test]
    async fn test_fractal_node_creation() {
        let content = MemoryContent {
            text: "Test content".to_string(),
            data: None,
            content_type: ContentType::Fact,
            emotional_signature: EmotionalSignature::default(),
            temporal_markers: vec![],
            quality_metrics: QualityMetrics::default(),
        };

        let config = FractalMemoryConfig::default();
        let node = FractalMemoryNode::new_root(content, "test_domain".to_string(), config).await.unwrap();

        assert_eq!(node.scale_level(), ScaleLevel::Worldview);
        assert_eq!(node.domain(), "test_domain");
        assert!(!node.id().to_string().is_empty());
    }
}
