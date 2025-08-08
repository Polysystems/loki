//! Neuroprocessor with B-Tree Thought Tracing
//!
//! This module implements the neural processing engine that manages thoughts
//! in a B-tree structure, enabling efficient pathway tracing and pattern
//! detection.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::cognitive::{Thought, ThoughtId, ThoughtType};
use crate::memory::{CacheKey, SimdSmartCache};

/// B-tree node representing a thought
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ThoughtNode {
    /// Unique thought identifier
    pub thought_id: ThoughtId,

    /// Thought content and metadata
    pub thought: Thought,

    /// Parent thought (if any)
    pub parent: Option<ThoughtId>,

    /// Child thoughts
    pub children: Vec<ThoughtId>,

    /// Neural activation level (0.0-1.0)
    pub activation: f32,

    /// Number of times this thought was activated
    pub activation_count: u32,

    /// Last activation time
    #[serde(skip)]
    pub last_activation: Instant,

    /// Associated neural weights
    pub weights: Vec<f32>,

    /// Fractal depth level
    pub fractal_depth: u32,

    /// Pattern signature for fractal detection
    pub pattern_signature: u64,
}

impl ThoughtNode {
    pub fn new(thought: Thought) -> Self {
        let thought_id = thought.id.clone();
        Self {
            thought_id,
            thought,
            parent: None,
            children: Vec::new(),
            activation: 0.0,
            activation_count: 0,
            last_activation: Instant::now(),
            weights: vec![0.0; 128], // Initialize with 128 weights
            fractal_depth: 0,
            pattern_signature: 0,
        }
    }

    /// Activate this node
    pub fn activate(&mut self, strength: f32) {
        self.activation = (self.activation + strength).min(1.0);
        self.activation_count += 1;
        self.last_activation = Instant::now();
    }

    /// Decay activation over time
    pub fn decay(&mut self, rate: f32) {
        self.activation *= rate;
        if self.activation < 0.01 {
            self.activation = 0.0;
        }
    }

    /// Calculate pattern signature for fractal detection
    pub fn calculate_pattern_signature(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.thought.thought_type.hash(&mut hasher);
        self.children.len().hash(&mut hasher);
        self.fractal_depth.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for ThoughtNode {
    fn default() -> Self {
        Self::new(Thought::default())
    }
}

/// Activation pattern for neural pathways
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationPattern {
    /// Pattern identifier
    pub id: PatternId,

    /// Pattern identifier string (for compatibility)
    pub pattern_id: String,

    /// Thoughts involved in this pattern
    pub thoughts: Vec<ThoughtId>,

    /// Pattern strength (0.0-1.0)
    pub strength: f32,

    /// Number of times pattern activated
    pub activation_count: u32,

    /// Pattern type
    pub pattern_type: PatternType,

    /// Fractal properties
    pub fractal_properties: FractalProperties,

    /// Timestamp of pattern detection
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PatternId(pub String);

impl PatternId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    Sequential,  // A -> B -> C
    Branching,   // A -> (B | C)
    Convergent,  // (A & B) -> C
    Cyclic,      // A -> B -> C -> A
    Fractal,     // Self-similar at different scales
    Emergent,    // Novel pattern combination
    Hierarchical, // Nested hierarchical patterns
    Associative, // Associative connections between patterns
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FractalProperties {
    /// Self-similarity score (0.0-1.0)
    pub self_similarity: f32,

    /// Scale invariance factor
    pub scale_invariance: f32,

    /// Recursion depth
    pub recursion_depth: u32,

    /// Dimensional complexity
    pub complexity: f32,
}

/// Main Neuroprocessor
#[derive(Debug, Clone)]
pub struct NeuroProcessor {
    /// B-tree of all thoughts
    pub thought_tree: Arc<RwLock<BTreeMap<ThoughtId, ThoughtNode>>>,

    /// Activation patterns
    pub activation_patterns: Arc<DashMap<PatternId, ActivationPattern>>,

    /// Pattern detection engine
    pattern_detector: Arc<PatternDetector>,

    /// Fractal analyzer
    fractal_analyzer: Arc<FractalAnalyzer>,

    /// Neural cache for fast access
    cache: Arc<SimdSmartCache>,

    /// Processing statistics
    stats: Arc<RwLock<ProcessingStats>>,
}

#[derive(Default, Debug, Clone)]
pub struct ProcessingStats {
    pub thoughts_processed: u64,
    pub patterns_detected: u64,
    pub fractals_found: u64,
    pub avg_activation_time_us: f64,
}

impl NeuroProcessor {
    pub async fn new(cache: Arc<SimdSmartCache>) -> Result<Self> {
        info!("Initializing NeuroProcessor with B-tree thought storage");

        Ok(Self {
            thought_tree: Arc::new(RwLock::new(BTreeMap::new())),
            activation_patterns: Arc::new(DashMap::new()),
            pattern_detector: Arc::new(PatternDetector::new()),
            fractal_analyzer: Arc::new(FractalAnalyzer::new()),
            cache,
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        })
    }

    /// Process a thought through the neural network
    pub async fn process_thought(&self, thought: &Thought) -> Result<f32> {
        let start = Instant::now();

        // Create thought node
        let mut node = ThoughtNode::new(thought.clone());

        // Find parent in tree if exists
        if let Some(parent_id) = &thought.parent {
            if let Some(mut parent_node) = self.get_node(parent_id).await {
                parent_node.children.push(thought.id.clone());
                node.parent = Some(parent_id.clone());
                node.fractal_depth = parent_node.fractal_depth + 1;
                self.update_node(parent_node).await?;
            }
        }

        // Calculate pattern signature
        node.pattern_signature = node.calculate_pattern_signature();

        // Process through neural network
        let activation = self.calculate_activation(&node).await?;
        node.activate(activation);

        // Store in B-tree
        self.thought_tree.write().await.insert(thought.id.clone(), node.clone());

        // Cache for fast access
        self.cache_thought(&node).await?;

        // Detect patterns
        if activation > 0.5 {
            self.detect_patterns(&thought.id).await?;
        }

        // Check for fractals
        if node.fractal_depth > 2 {
            self.check_fractal_patterns(&node).await?;
        }

        // Update stats
        let elapsed = start.elapsed().as_micros() as f64;
        self.update_stats(elapsed).await;

        Ok(activation)
    }

    /// Calculate neural activation for a thought
    async fn calculate_activation(&self, node: &ThoughtNode) -> Result<f32> {
        // Get related thoughts from cache
        let similar = self.cache.find_similar(&node.weights, 5).await;

        let mut activation = 0.0;

        // Base activation from thought type
        activation += match node.thought.thought_type {
            ThoughtType::Decision => 0.3,
            ThoughtType::Analysis => 0.4,
            ThoughtType::Synthesis => 0.5,
            ThoughtType::Creation => 0.6,
            _ => 0.2,
        };

        // Boost from similar thoughts
        for (_, similarity) in similar {
            activation += similarity * 0.1;
        }

        // Boost from parent activation
        if let Some(parent_id) = &node.parent {
            if let Some(parent) = self.get_node(parent_id).await {
                activation += parent.activation * 0.2;
            }
        }

        // Normalize to 0.0-1.0
        Ok(activation.min(1.0))
    }

    /// Cache thought for fast retrieval
    async fn cache_thought(&self, node: &ThoughtNode) -> Result<()> {
        use crate::memory::{AlignedVec, CachedNeuron};

        let key = CacheKey(node.thought_id.to_bytes());

        let mut activations = AlignedVec::new(node.weights.len());
        for weight in &node.weights {
            if let Err(e) = activations.push(*weight) {
                tracing::debug!("Failed to push weight to activations: {}", e);
                break; // Stop if we can't add more weights
            }
        }

        let neuron = CachedNeuron {
            id: node.thought_id.0.clone(),
            key: key.clone(),
            weights: AlignedVec::new(0), // Empty for now
            bias: 0.0,
            activation: node.activation,
            activations,
            layer_index: 0,
            neuron_index: 0,
            last_updated: Instant::now(),
            last_access: Instant::now(),
            activation_history: vec![node.activation],
            gradient_cache: Vec::new(),
            momentum: Vec::new(),
            access_count: std::sync::atomic::AtomicUsize::new(1),
        };

        self.cache.insert(key, neuron).await;
        Ok(())
    }

    /// Detect activation patterns
    async fn detect_patterns(&self, thought_id: &ThoughtId) -> Result<()> {
        let patterns =
            self.pattern_detector.detect_patterns(thought_id, &self.thought_tree).await?;

        for pattern in patterns {
            self.activation_patterns.insert(pattern.id.clone(), pattern);

            let mut stats = self.stats.write().await;
            stats.patterns_detected += 1;
        }

        Ok(())
    }

    /// Check for fractal patterns
    async fn check_fractal_patterns(&self, node: &ThoughtNode) -> Result<()> {
        let fractals = self.fractal_analyzer.analyze_node(node, &self.thought_tree).await?;

        if fractals.self_similarity > 0.7 {
            debug!("Fractal pattern detected: {:?}", fractals);

            let mut stats = self.stats.write().await;
            stats.fractals_found += 1;

            // Create fractal pattern
            let pattern_id = PatternId::new();
            let pattern = ActivationPattern {
                id: pattern_id.clone(),
                pattern_id: pattern_id.0.clone(),
                thoughts: vec![node.thought_id.clone()],
                strength: fractals.self_similarity,
                activation_count: 1,
                pattern_type: PatternType::Fractal,
                fractal_properties: fractals,
                timestamp: Instant::now(),
            };

            self.activation_patterns.insert(pattern.id.clone(), pattern);
        }

        Ok(())
    }

    /// Get a thought node
    async fn get_node(&self, id: &ThoughtId) -> Option<ThoughtNode> {
        self.thought_tree.read().await.get(id).cloned()
    }

    /// Update a thought node
    async fn update_node(&self, node: ThoughtNode) -> Result<()> {
        self.thought_tree.write().await.insert(node.thought_id.clone(), node);
        Ok(())
    }

    /// Update processing statistics
    async fn update_stats(&self, activation_time_us: f64) {
        let mut stats = self.stats.write().await;
        stats.thoughts_processed += 1;

        // Running average
        let n = stats.thoughts_processed as f64;
        stats.avg_activation_time_us =
            (stats.avg_activation_time_us * (n - 1.0) + activation_time_us) / n;
    }

    /// Get processing statistics
    pub async fn get_stats(&self) -> ProcessingStats {
        self.stats.read().await.clone()
    }

    /// Apply decay to all activations
    pub async fn decay_activations(&self, rate: f32) -> Result<()> {
        let mut tree = self.thought_tree.write().await;

        for (_, node) in tree.iter_mut() {
            node.decay(rate);
        }

        Ok(())
    }

    /// Get active thoughts (activation > threshold)
    pub async fn get_active_thoughts(&self, threshold: f32) -> Vec<ThoughtNode> {
        let tree = self.thought_tree.read().await;

        tree.values().filter(|node| node.activation > threshold).cloned().collect()
    }

    /// Get thought pathways from a starting thought
    pub async fn get_pathways(&self, start: &ThoughtId, depth: u32) -> Vec<Vec<ThoughtId>> {
        let tree = self.thought_tree.read().await;
        let mut pathways = Vec::new();
        let mut current_path = Vec::new();

        self.trace_pathways(&tree, start, depth, &mut current_path, &mut pathways);

        pathways
    }

    /// Get recent activation patterns
    pub async fn get_recent_activations(&self, count: usize) -> Vec<ActivationPattern> {
        let patterns =
            self.activation_patterns.iter().map(|entry| entry.value().clone()).collect::<Vec<_>>();

        // Sort by timestamp (most recent first)
        let mut sorted: Vec<ActivationPattern> = patterns;
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Return requested count
        sorted.into_iter().take(count).collect()
    }

    /// Get recent patterns with their types and confidence (public interface)
    pub async fn get_pattern_types(&self, count: usize) -> Vec<(PatternType, f32)> {
        let recent = self.get_recent_activations(count).await;

        recent.into_iter().map(|pattern| (pattern.pattern_type, pattern.strength)).collect()
    }

    /// Recursive pathway tracing
    fn trace_pathways(
        &self,
        tree: &BTreeMap<ThoughtId, ThoughtNode>,
        current: &ThoughtId,
        remaining_depth: u32,
        current_path: &mut Vec<ThoughtId>,
        pathways: &mut Vec<Vec<ThoughtId>>,
    ) {
        if remaining_depth == 0 {
            if !current_path.is_empty() {
                pathways.push(current_path.clone());
            }
            return;
        }

        current_path.push(current.clone());

        if let Some(node) = tree.get(current) {
            if node.children.is_empty() {
                // Leaf node - save path
                pathways.push(current_path.clone());
            } else {
                // Recurse into children
                for child in &node.children {
                    self.trace_pathways(tree, child, remaining_depth - 1, current_path, pathways);
                }
            }
        }

        current_path.pop();
    }

    /// Set processing mode based on emotional state
    pub async fn set_processing_mode(&self, mode: &str) -> Result<()> {
        info!("Setting neural processing mode to: {}", mode);

        // Adjust processing based on mode
        match mode {
            "conservative" => {
                // Reduce exploration, increase stability
                info!("Neural processor in conservative mode - prioritizing stability");
            }
            "exploratory" => {
                // Increase exploration, creative thinking
                info!("Neural processor in exploratory mode - prioritizing creativity");
            }
            "balanced" => {
                // Default balanced processing
                info!("Neural processor in balanced mode - normal operation");
            }
            _ => {
                warn!("Unknown processing mode: {}, using balanced", mode);
            }
        }

        Ok(())
    }
}

/// Pattern detection engine
#[derive(Debug, Clone)]
struct PatternDetector {
    /// Minimum pattern length
    min_length: usize,

    /// Pattern history
    history: Arc<Mutex<VecDeque<ThoughtId>>>,

    /// Detected sequences
    sequences: Arc<RwLock<HashMap<Vec<ThoughtId>, u32>>>,
}

impl PatternDetector {
    fn new() -> Self {
        Self {
            min_length: 3,
            history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            sequences: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn detect_patterns(
        &self,
        thought_id: &ThoughtId,
        _tree: &Arc<RwLock<BTreeMap<ThoughtId, ThoughtNode>>>,
    ) -> Result<Vec<ActivationPattern>> {
        let mut patterns = Vec::new();

        // Add to history
        {
            let mut history = self.history.lock().await;
            history.push_back(thought_id.clone());
            if history.len() > 100 {
                history.pop_front();
            }
        }

        // Look for sequential patterns
        let history = self.history.lock().await.clone();
        let history_vec: Vec<ThoughtId> = history.into_iter().collect();
        if history_vec.len() >= self.min_length {
            for window_size in self.min_length..=history_vec.len().min(10) {
                for window in history_vec.windows(window_size) {
                    let sequence: Vec<ThoughtId> = window.to_vec();

                    let mut sequences = self.sequences.write().await;
                    *sequences.entry(sequence.clone()).or_insert(0) += 1;

                    let count = sequences[&sequence];

                    if count >= 3 {
                        // Pattern detected
                        let pattern_id = PatternId::new();
                        let pattern = ActivationPattern {
                            id: pattern_id.clone(),
                            pattern_id: pattern_id.0.clone(),
                            thoughts: sequence,
                            strength: (count as f32 / 10.0).min(1.0),
                            activation_count: count,
                            pattern_type: PatternType::Sequential,
                            fractal_properties: FractalProperties {
                                self_similarity: 0.0,
                                scale_invariance: 0.0,
                                recursion_depth: 0,
                                complexity: window_size as f32,
                            },
                            timestamp: Instant::now(),
                        };
                        patterns.push(pattern);
                    }
                }
            }
        }

        Ok(patterns)
    }
}

/// Fractal pattern analyzer
#[derive(Debug, Clone)]
struct FractalAnalyzer {
    /// Similarity threshold
    similarity_threshold: f32,
}

impl FractalAnalyzer {
    fn new() -> Self {
        Self { similarity_threshold: 0.7 }
    }

    async fn analyze_node(
        &self,
        node: &ThoughtNode,
        tree: &Arc<RwLock<BTreeMap<ThoughtId, ThoughtNode>>>,
    ) -> Result<FractalProperties> {
        let tree_map = tree.read().await;

        // Check for self-similarity
        let self_similarity = self.calculate_self_similarity(node, &tree_map);

        // Check scale invariance
        let scale_invariance = self.calculate_scale_invariance(node, &tree_map);

        // Calculate complexity
        let complexity = self.calculate_complexity(node, &tree_map);

        Ok(FractalProperties {
            self_similarity,
            scale_invariance,
            recursion_depth: node.fractal_depth,
            complexity,
        })
    }

    fn calculate_self_similarity(
        &self,
        node: &ThoughtNode,
        tree: &BTreeMap<ThoughtId, ThoughtNode>,
    ) -> f32 {
        let mut similarity_sum = 0.0;
        let mut count = 0;

        // Compare with ancestors
        let mut current = node.parent.as_ref();
        while let Some(parent_id) = current {
            if let Some(parent) = tree.get(parent_id) {
                if parent.pattern_signature == node.pattern_signature {
                    similarity_sum += 1.0;
                }
                count += 1;
                current = parent.parent.as_ref();
            } else {
                break;
            }
        }

        if count > 0 { similarity_sum / count as f32 } else { 0.0 }
    }

    fn calculate_scale_invariance(
        &self,
        _node: &ThoughtNode,
        _tree: &BTreeMap<ThoughtId, ThoughtNode>,
    ) -> f32 {
        // Simplified for now - would check pattern preservation across scales
        0.5
    }

    fn calculate_complexity(
        &self,
        node: &ThoughtNode,
        tree: &BTreeMap<ThoughtId, ThoughtNode>,
    ) -> f32 {
        // Count descendants
        let mut complexity: f32 = 0.0;
        let mut queue = VecDeque::new();
        queue.push_back(&node.thought_id);

        while let Some(current_id) = queue.pop_front() {
            if let Some(current) = tree.get(current_id) {
                complexity += 1.0;
                for child in &current.children {
                    queue.push_back(child);
                }
            }
        }

        complexity.log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::{Thought, ThoughtMetadata, ThoughtType};
    use crate::memory::SimdCacheConfig;

    #[tokio::test]
    async fn test_thought_processing() {
        let cache = Arc::new(SimdSmartCache::new(SimdCacheConfig::default()));
        let processor = NeuroProcessor::new(cache).await.unwrap();

        let thought = Thought {
            id: ThoughtId::new(),
            content: "Test thought".to_string(),
            thought_type: ThoughtType::Analysis,
            metadata: ThoughtMetadata::default(),
            parent: None,
            children: Vec::new(),
            timestamp: Instant::now(),
        };

        let activation = processor.process_thought(&thought).await.unwrap();
        assert!(activation > 0.0);
        assert!(activation <= 1.0);

        // Check stats
        let stats = processor.get_stats().await;
        assert_eq!(stats.thoughts_processed, 1);
    }

    #[tokio::test]
    async fn test_pathway_tracing() {
        let cache = Arc::new(SimdSmartCache::new(SimdCacheConfig::default()));
        let processor = NeuroProcessor::new(cache).await.unwrap();

        // Create a chain of thoughts
        let mut parent_id = None;
        let mut first_id = None;

        for i in 0..5 {
            let thought = Thought {
                id: ThoughtId::new(),
                content: format!("Thought {}", i),
                thought_type: ThoughtType::Analysis,
                metadata: ThoughtMetadata::default(),
                parent: parent_id.clone(),
                children: Vec::new(),
                timestamp: Instant::now(),
            };

            if i == 0 {
                first_id = Some(thought.id.clone());
            }

            processor.process_thought(&thought).await.unwrap();
            parent_id = Some(thought.id);
        }

        // Get pathways
        let pathways = processor.get_pathways(&first_id.unwrap(), 5).await;
        assert!(!pathways.is_empty());
    }
}
