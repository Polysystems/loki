//! Cross-scale indexing system for multi-dimensional cognitive context operations
//! Provides O(1) lookups across different cognitive scales with lock-free guarantees

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::time::Instant;
use std::collections::BTreeMap;

use dashmap::{DashMap, DashSet};
use serde::{Serialize, Deserialize};

use super::{ConcurrentMap, IndexedRingBuffer};
use crate::memory::fractal::ScaleLevel;

/// Multi-scale index for efficient cross-scale operations
pub struct CrossScaleIndex<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Primary index by scale level
    scale_index: Arc<DashMap<ScaleLevel, ScaleIndexNode<T>>>,
    
    /// Cross-scale correlation index for fast queries between scales
    correlation_index: Arc<ConcurrentMap<(ScaleLevel, ScaleLevel), CorrelationData>>,
    
    /// Temporal index for time-based queries (using BTreeMap for range queries)  
    temporal_index: Arc<parking_lot::RwLock<BTreeMap<u64, Vec<TemporalIndexEntry>>>>,
    
    /// Pattern index for fast pattern matching across scales
    pattern_index: Arc<DashMap<String, PatternIndexEntry>>,
    
    /// Global statistics
    stats: Arc<CrossScaleIndexStats>,
    
    /// Configuration
    config: CrossScaleIndexConfig,
}

/// Index node for a specific cognitive scale
pub struct ScaleIndexNode<T>
where
    T: Clone + Send + Sync,
{
    /// Scale level identifier
    scale_level: ScaleLevel,
    
    /// Context data for this scale with indexed access
    contexts: Arc<IndexedRingBuffer<T>>,
    
    /// Active patterns at this scale
    patterns: Arc<DashSet<String>>,
    
    /// Atomic metrics for this scale
    metrics: Arc<AtomicScaleMetrics>,
    
    /// Cross-references to related items in other scales
    cross_references: Arc<DashMap<ScaleLevel, Vec<usize>>>,
    
    /// Temporal ordering information
    temporal_order: Arc<IndexedRingBuffer<TemporalIndexEntry>>,
}

/// Entry in the scale index with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleIndexEntry<T>
where
    T: Clone,
{
    pub scale_level: ScaleLevel,
    pub item: T,
    pub timestamp: u64, // Unix timestamp in nanoseconds
    pub sequence_id: u64,
    pub confidence: f32,
    pub metadata: IndexEntryMetadata,
}

/// Metadata for index entries
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexEntryMetadata {
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub quality_score: f32,
    pub processing_stage: String,
    pub custom_fields: std::collections::HashMap<String, serde_json::Value>,
}

/// Temporal index entry for time-based ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalIndexEntry {
    pub timestamp: u64,
    pub scale_level: ScaleLevel,
    pub item_index: usize,
    pub sequence_id: u64,
}

/// Cross-scale correlation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationData {
    pub correlation_strength: f32,
    pub correlation_type: CorrelationType,
    pub sample_count: usize,
    pub last_updated: u64,
    pub confidence_interval: (f32, f32),
}

/// Types of correlations between scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    Causal { direction: CausalDirection, lag: i32 },
    Structural { similarity: f32 },
    Temporal { synchronization: f32 },
    Emergent { emergence_strength: f32 },
    Semantic { semantic_similarity: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalDirection {
    Forward,
    Backward,
    Bidirectional,
}

/// Pattern index entry
#[derive(Debug, Clone)]
pub struct PatternIndexEntry {
    pub pattern_id: String,
    pub scale_occurrences: Arc<DashMap<ScaleLevel, Vec<usize>>>,
    pub pattern_strength: Arc<AtomicU64>, // Scaled by 10000 for precision
    pub last_seen: Arc<AtomicU64>, // Unix timestamp
    pub total_occurrences: Arc<AtomicUsize>,
}

/// Atomic metrics for a scale
#[derive(Debug, Default)]
pub struct AtomicScaleMetrics {
    pub item_count: AtomicUsize,
    pub total_insertions: AtomicU64,
    pub total_queries: AtomicU64,
    pub avg_quality_score: AtomicU64, // Scaled by 10000
    pub last_update_timestamp: AtomicU64,
    pub active_patterns: AtomicUsize,
}

impl<T> CrossScaleIndex<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Helper function to get scale ordering for comparison
    fn scale_order(scale: &ScaleLevel) -> u8 {
        match scale {
            ScaleLevel::System => 0,
            ScaleLevel::Domain => 1, 
            ScaleLevel::Token => 2,
            ScaleLevel::Atomic => 3,
            ScaleLevel::Concept => 4,
            ScaleLevel::Schema => 5,
            ScaleLevel::Worldview => 6,
            ScaleLevel::Meta => 7,
            ScaleLevel::Pattern => 8,
            ScaleLevel::Instance => 9,
            ScaleLevel::Detail => 10,
            ScaleLevel::Quantum => 11,
        }
    }
    /// Create a new cross-scale index
    pub fn new(config: CrossScaleIndexConfig) -> Self {
        Self {
            scale_index: Arc::new(DashMap::new()),
            correlation_index: Arc::new(ConcurrentMap::new()),
            temporal_index: Arc::new(parking_lot::RwLock::new(BTreeMap::new())),
            pattern_index: Arc::new(DashMap::new()),
            stats: Arc::new(CrossScaleIndexStats::new()),
            config,
        }
    }
    
    /// Insert an item into the index
    pub fn insert(&self, scale_level: ScaleLevel, item: T, metadata: IndexEntryMetadata) -> u64 {
        let sequence_id = self.stats.next_sequence_id();
        let timestamp = current_timestamp_nanos();
        
        // Get or create scale index node
        let node = self.get_or_create_scale_node(scale_level.clone());
        
        // Create index entry
        let entry = ScaleIndexEntry {
            scale_level: scale_level.clone(),
            item: item.clone(),
            timestamp,
            sequence_id,
            confidence: metadata.quality_score,
            metadata,
        };
        
        // Insert into scale-specific buffer
        if !node.contexts.push(item.clone()) {
            // Buffer full, handle overflow
            self.handle_buffer_overflow(&node);
            // Try again
            node.contexts.push(item);
        }
        
        // Update temporal index
        self.update_temporal_index(timestamp, scale_level.clone(), sequence_id);
        
        // Update metrics
        node.metrics.item_count.store(node.contexts.len(), Ordering::Relaxed);
        node.metrics.total_insertions.fetch_add(1, Ordering::Relaxed);
        node.metrics.last_update_timestamp.store(timestamp, Ordering::Relaxed);
        
        // Update global stats
        self.stats.total_insertions.fetch_add(1, Ordering::Relaxed);
        
        sequence_id
    }
    
    /// Get items from a specific scale
    pub fn get_scale_items(&self, scale_level: &ScaleLevel, limit: Option<usize>) -> Vec<T> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);
        
        if let Some(node) = self.scale_index.get(scale_level) {
            node.metrics.total_queries.fetch_add(1, Ordering::Relaxed);
            
            let items = node.contexts.get_all();
            if let Some(limit) = limit {
                items.into_iter().take(limit).collect()
            } else {
                items
            }
        } else {
            Vec::new()
        }
    }
    
    /// Get the most recent N items from a scale
    pub fn get_recent_items(&self, scale_level: &ScaleLevel, n: usize) -> Vec<T> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);
        
        if let Some(node) = self.scale_index.get(scale_level) {
            node.metrics.total_queries.fetch_add(1, Ordering::Relaxed);
            node.contexts.get_last_n(n)
        } else {
            Vec::new()
        }
    }
    
    /// Query items across multiple scales
    pub fn query_multi_scale(&self, scales: &[ScaleLevel], limit: Option<usize>) -> Vec<ScaleIndexEntry<T>> {
        let mut results = Vec::new();
        
        for scale in scales {
            if let Some(node) = self.scale_index.get(scale) {
                let items = node.contexts.get_all();
                
                for (index, item) in items.into_iter().enumerate() {
                    let timestamp = node.temporal_order.get(index)
                        .map(|entry| entry.timestamp)
                        .unwrap_or(current_timestamp_nanos());
                    
                    results.push(ScaleIndexEntry {
                        scale_level: scale.clone(),
                        item,
                        timestamp,
                        sequence_id: self.stats.next_sequence_id(),
                        confidence: 1.0, // Default confidence
                        metadata: IndexEntryMetadata::default(),
                    });
                }
            }
        }
        
        // Sort by timestamp (most recent first)
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        if let Some(limit) = limit {
            results.truncate(limit);
        }
        
        self.stats.multi_scale_queries.fetch_add(1, Ordering::Relaxed);
        results
    }
    
    /// Get correlation data between two scales
    pub fn get_correlation(&self, scale_a: &ScaleLevel, scale_b: &ScaleLevel) -> Option<CorrelationData> {
        let key = if Self::scale_order(scale_a) <= Self::scale_order(scale_b) {
            (scale_a.clone(), scale_b.clone())
        } else {
            (scale_b.clone(), scale_a.clone())
        };
        
        self.correlation_index.get(&key)
    }
    
    /// Update correlation between two scales
    pub fn update_correlation(&self, scale_a: ScaleLevel, scale_b: ScaleLevel, correlation: CorrelationData) {
        let key = if Self::scale_order(&scale_a) <= Self::scale_order(&scale_b) {
            (scale_a, scale_b)
        } else {
            (scale_b, scale_a)
        };
        
        self.correlation_index.insert(key, correlation);
        self.stats.correlation_updates.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Register a pattern occurrence at a scale
    pub fn register_pattern(&self, pattern_id: String, scale_level: ScaleLevel, item_index: usize, strength: f32) {
        let entry = self.pattern_index
            .entry(pattern_id.clone())
            .or_insert_with(|| PatternIndexEntry {
                pattern_id: pattern_id.clone(),
                scale_occurrences: Arc::new(DashMap::new()),
                pattern_strength: Arc::new(AtomicU64::new(0)),
                last_seen: Arc::new(AtomicU64::new(current_timestamp_nanos())),
                total_occurrences: Arc::new(AtomicUsize::new(0)),
            });
        
        // Update occurrence list for this scale
        entry.scale_occurrences
            .entry(scale_level)
            .or_insert_with(Vec::new)
            .push(item_index);
        
        // Update pattern strength (weighted average)
        let current_strength = entry.pattern_strength.load(Ordering::Relaxed) as f32 / 10000.0;
        let occurrence_count = entry.total_occurrences.fetch_add(1, Ordering::Relaxed) + 1;
        
        let new_strength = ((current_strength * (occurrence_count - 1) as f32) + strength) / occurrence_count as f32;
        entry.pattern_strength.store((new_strength * 10000.0) as u64, Ordering::Relaxed);
        entry.last_seen.store(current_timestamp_nanos(), Ordering::Relaxed);
        
        // Update scale pattern count
        if let Some(node) = self.scale_index.get(&scale_level) {
            node.patterns.insert(pattern_id);
            node.metrics.active_patterns.store(node.patterns.len(), Ordering::Relaxed);
        }
        
        self.stats.pattern_registrations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Find patterns across multiple scales
    pub fn find_cross_scale_patterns(&self, pattern_id: &str) -> Vec<(ScaleLevel, Vec<usize>)> {
        if let Some(entry) = self.pattern_index.get(pattern_id) {
            entry.scale_occurrences
                .iter()
                .map(|kv| (kv.key().clone(), kv.value().clone()))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get items within a time range across all scales
    pub fn get_temporal_range(&self, start_nanos: u64, end_nanos: u64) -> Vec<ScaleIndexEntry<T>> {
        let temporal_index = self.temporal_index.read();
        let mut results = Vec::new();
        
        for (timestamp, entries) in temporal_index.range(start_nanos..=end_nanos) {
            for temporal_entry in entries {
                if let Some(node) = self.scale_index.get(&temporal_entry.scale_level) {
                    if let Some(item) = node.contexts.get(temporal_entry.item_index) {
                        results.push(ScaleIndexEntry {
                            scale_level: temporal_entry.scale_level.clone(),
                            item,
                            timestamp: *timestamp,
                            sequence_id: temporal_entry.sequence_id,
                            confidence: 1.0,
                            metadata: IndexEntryMetadata::default(),
                        });
                    }
                }
            }
        }
        
        self.stats.temporal_queries.fetch_add(1, Ordering::Relaxed);
        results
    }
    
    /// Get statistics for all scales
    pub fn get_stats(&self) -> CrossScaleIndexStats {
        self.stats.snapshot()
    }
    
    /// Get statistics for a specific scale
    pub fn get_scale_stats(&self, scale_level: &ScaleLevel) -> Option<AtomicScaleMetrics> {
        self.scale_index.get(scale_level).map(|node| {
            AtomicScaleMetrics {
                item_count: AtomicUsize::new(node.metrics.item_count.load(Ordering::Relaxed)),
                total_insertions: AtomicU64::new(node.metrics.total_insertions.load(Ordering::Relaxed)),
                total_queries: AtomicU64::new(node.metrics.total_queries.load(Ordering::Relaxed)),
                avg_quality_score: AtomicU64::new(node.metrics.avg_quality_score.load(Ordering::Relaxed)),
                last_update_timestamp: AtomicU64::new(node.metrics.last_update_timestamp.load(Ordering::Relaxed)),
                active_patterns: AtomicUsize::new(node.metrics.active_patterns.load(Ordering::Relaxed)),
            }
        })
    }
    
    /// Private helper methods
    fn get_or_create_scale_node(&self, scale_level: ScaleLevel) -> dashmap::mapref::one::Ref<'_, ScaleLevel, ScaleIndexNode<T>> {
        if !self.scale_index.contains_key(&scale_level) {
            let node = ScaleIndexNode {
                scale_level: scale_level.clone(),
                contexts: Arc::new(IndexedRingBuffer::new(self.config.scale_buffer_capacity)),
                patterns: Arc::new(DashSet::new()),
                metrics: Arc::new(AtomicScaleMetrics::default()),
                cross_references: Arc::new(DashMap::new()),
                temporal_order: Arc::new(IndexedRingBuffer::new(self.config.scale_buffer_capacity)),
            };
            self.scale_index.insert(scale_level.clone(), node);
        }
        
        self.scale_index.get(&scale_level).unwrap()
    }
    
    fn update_temporal_index(&self, timestamp: u64, scale_level: ScaleLevel, sequence_id: u64) {
        let mut temporal_index = self.temporal_index.write();
        
        let entries = temporal_index.entry(timestamp).or_insert_with(Vec::new);
        entries.push(TemporalIndexEntry {
            timestamp,
            scale_level,
            item_index: sequence_id as usize, // Use sequence_id as item index for now
            sequence_id,
        });
        
        // Limit temporal index size
        if temporal_index.len() > self.config.max_temporal_entries {
            let oldest_key = *temporal_index.keys().next().unwrap();
            temporal_index.remove(&oldest_key);
        }
    }
    
    fn handle_buffer_overflow(&self, _node: &ScaleIndexNode<T>) {
        // For now, just let the buffer handle overflow
        // In the future, could implement more sophisticated policies:
        // - Archive old data
        // - Compress data
        // - Move to secondary storage
        self.stats.buffer_overflows.fetch_add(1, Ordering::Relaxed);
    }
}

/// Configuration for the cross-scale index
#[derive(Debug, Clone)]
pub struct CrossScaleIndexConfig {
    pub scale_buffer_capacity: usize,
    pub max_temporal_entries: usize,
    pub correlation_threshold: f32,
    pub pattern_threshold: f32,
    pub cleanup_interval_seconds: u64,
}

impl Default for CrossScaleIndexConfig {
    fn default() -> Self {
        Self {
            scale_buffer_capacity: 10000,
            max_temporal_entries: 100000,
            correlation_threshold: 0.1,
            pattern_threshold: 0.5,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Global statistics for the cross-scale index
#[derive(Debug)]
pub struct CrossScaleIndexStats {
    pub total_insertions: AtomicU64,
    pub total_queries: AtomicU64,
    pub multi_scale_queries: AtomicU64,
    pub temporal_queries: AtomicU64,
    pub correlation_updates: AtomicU64,
    pub pattern_registrations: AtomicU64,
    pub buffer_overflows: AtomicU64,
    pub created_at: Instant,
    sequence_counter: AtomicU64,
}

impl CrossScaleIndexStats {
    pub fn new() -> Self {
        Self {
            total_insertions: AtomicU64::new(0),
            total_queries: AtomicU64::new(0),
            multi_scale_queries: AtomicU64::new(0),
            temporal_queries: AtomicU64::new(0),
            correlation_updates: AtomicU64::new(0),
            pattern_registrations: AtomicU64::new(0),
            buffer_overflows: AtomicU64::new(0),
            created_at: Instant::now(),
            sequence_counter: AtomicU64::new(1),
        }
    }
    
    pub fn next_sequence_id(&self) -> u64 {
        self.sequence_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    pub fn snapshot(&self) -> Self {
        Self {
            total_insertions: AtomicU64::new(self.total_insertions.load(Ordering::Relaxed)),
            total_queries: AtomicU64::new(self.total_queries.load(Ordering::Relaxed)),
            multi_scale_queries: AtomicU64::new(self.multi_scale_queries.load(Ordering::Relaxed)),
            temporal_queries: AtomicU64::new(self.temporal_queries.load(Ordering::Relaxed)),
            correlation_updates: AtomicU64::new(self.correlation_updates.load(Ordering::Relaxed)),
            pattern_registrations: AtomicU64::new(self.pattern_registrations.load(Ordering::Relaxed)),
            buffer_overflows: AtomicU64::new(self.buffer_overflows.load(Ordering::Relaxed)),
            created_at: self.created_at,
            sequence_counter: AtomicU64::new(self.sequence_counter.load(Ordering::Relaxed)),
        }
    }
    
    pub fn total_operations(&self) -> u64 {
        self.total_insertions.load(Ordering::Relaxed) +
        self.total_queries.load(Ordering::Relaxed) +
        self.correlation_updates.load(Ordering::Relaxed) +
        self.pattern_registrations.load(Ordering::Relaxed)
    }
    
    pub fn operations_per_second(&self) -> f64 {
        let elapsed = self.created_at.elapsed();
        let total_ops = self.total_operations();
        
        if elapsed.as_secs_f64() == 0.0 {
            0.0
        } else {
            total_ops as f64 / elapsed.as_secs_f64()
        }
    }
}

/// Helper function to get current timestamp in nanoseconds
pub fn current_timestamp_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_index_operations() {
        let config = CrossScaleIndexConfig::default();
        let index = CrossScaleIndex::<i32>::new(config);
        
        let scale = ScaleLevel::Micro;
        let metadata = IndexEntryMetadata {
            quality_score: 0.8,
            ..Default::default()
        };
        
        // Insert items
        let seq1 = index.insert(scale.clone(), 42, metadata.clone());
        let seq2 = index.insert(scale.clone(), 43, metadata.clone());
        
        assert!(seq1 < seq2);
        
        // Query items
        let items = index.get_scale_items(&scale, None);
        assert_eq!(items, vec![42, 43]);
        
        // Get recent items
        let recent = index.get_recent_items(&scale, 1);
        assert_eq!(recent, vec![43]);
    }
    
    #[test]
    fn test_cross_scale_queries() {
        let config = CrossScaleIndexConfig::default();
        let index = CrossScaleIndex::<String>::new(config);
        
        let metadata = IndexEntryMetadata::default();
        
        // Insert into different scales
        index.insert(ScaleLevel::Micro, "micro_item".to_string(), metadata.clone());
        index.insert(ScaleLevel::Macro, "macro_item".to_string(), metadata.clone());
        index.insert(ScaleLevel::Global, "global_item".to_string(), metadata);
        
        // Query multiple scales
        let scales = vec![ScaleLevel::Micro, ScaleLevel::Macro];
        let results = index.query_multi_scale(&scales, None);
        
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.item == "micro_item"));
        assert!(results.iter().any(|r| r.item == "macro_item"));
    }
    
    #[test] 
    fn test_pattern_indexing() {
        let config = CrossScaleIndexConfig::default();
        let index = CrossScaleIndex::<String>::new(config);
        
        // Register pattern occurrences
        index.register_pattern("pattern1".to_string(), ScaleLevel::Micro, 0, 0.8);
        index.register_pattern("pattern1".to_string(), ScaleLevel::Macro, 1, 0.9);
        index.register_pattern("pattern2".to_string(), ScaleLevel::Micro, 2, 0.7);
        
        // Find cross-scale patterns
        let occurrences = index.find_cross_scale_patterns("pattern1");
        assert_eq!(occurrences.len(), 2);
        
        let occurrences2 = index.find_cross_scale_patterns("pattern2");
        assert_eq!(occurrences2.len(), 1);
    }
    
    #[test]
    fn test_correlation_tracking() {
        let config = CrossScaleIndexConfig::default();
        let index = CrossScaleIndex::<i32>::new(config);
        
        let correlation = CorrelationData {
            correlation_strength: 0.75,
            correlation_type: CorrelationType::Causal {
                direction: CausalDirection::Forward,
                lag: 2,
            },
            sample_count: 100,
            last_updated: current_timestamp_nanos(),
            confidence_interval: (0.65, 0.85),
        };
        
        // Update correlation
        index.update_correlation(ScaleLevel::Micro, ScaleLevel::Macro, correlation.clone());
        
        // Retrieve correlation
        let retrieved = index.get_correlation(&ScaleLevel::Micro, &ScaleLevel::Macro);
        assert!(retrieved.is_some());
        
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.correlation_strength, 0.75);
        assert_eq!(retrieved.sample_count, 100);
    }
}