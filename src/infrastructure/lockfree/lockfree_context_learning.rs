//! Lock-free context learning system for adaptive pattern recognition
//! Replaces RwLock-based learning with atomic operations and lock-free data structures

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Serialize, Deserialize};

use super::{ConcurrentMap, IndexedRingBuffer, AtomicContextAnalytics};

/// Lock-free context learning system
pub struct LockFreeContextLearningSystem {
    /// Learning parameters (atomic configuration)
    parameters: Arc<AtomicLearningParameters>,
    
    /// Training examples buffer with indexed access
    training_examples: Arc<IndexedRingBuffer<TrainingExample>>,
    
    /// Pattern weight adjustments (lock-free map)
    pattern_weights: Arc<ConcurrentMap<String, AtomicPatternWeight>>,
    
    /// Learning performance metrics (atomic)
    performance: Arc<AtomicLearningMetrics>,
    
    /// Model adaptation history
    adaptation_history: Arc<IndexedRingBuffer<AdaptationRecord>>,
    
    /// Feature importance scores
    feature_importance: Arc<ConcurrentMap<String, AtomicFeatureScore>>,
    
    /// Learning state
    learning_state: Arc<AtomicLearningState>,
    
    /// Configuration
    config: LockFreeLearningConfig,
}

/// Atomic learning parameters
#[derive(Debug)]
pub struct AtomicLearningParameters {
    /// Learning rate (scaled by 10000 for precision)
    pub learning_rate_x10000: AtomicU64,
    
    /// Adaptation threshold (scaled by 10000)
    pub adaptation_threshold_x10000: AtomicU64,
    
    /// Decay rate for old patterns (scaled by 10000)
    pub decay_rate_x10000: AtomicU64,
    
    /// Minimum confidence for pattern updates (scaled by 10000)
    pub min_confidence_x10000: AtomicU64,
    
    /// Maximum training examples to keep
    pub max_training_examples: AtomicUsize,
    
    /// Learning enabled flag
    pub learning_enabled: AtomicBool,
    
    /// Auto-adaptation enabled
    pub auto_adaptation_enabled: AtomicBool,
}

/// Atomic pattern weight
#[derive(Debug)]
pub struct AtomicPatternWeight {
    /// Current weight (scaled by 10000)
    pub weight_x10000: AtomicU64,
    
    /// Confidence in this weight (scaled by 10000)
    pub confidence_x10000: AtomicU64,
    
    /// Number of updates
    pub update_count: AtomicU64,
    
    /// Last update timestamp
    pub last_updated: AtomicU64,
    
    /// Success rate (scaled by 10000)
    pub success_rate_x10000: AtomicU64,
    
    /// Pattern stability metric (scaled by 10000)
    pub stability_x10000: AtomicU64,
}

/// Training example for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input_features: Vec<f32>,
    pub expected_output: f32,
    pub actual_output: Option<f32>,
    pub pattern_id: Option<String>,
    pub timestamp: u64,
    pub quality_score: f32,
    pub context_metadata: TrainingMetadata,
}

/// Training metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingMetadata {
    pub source: String,
    pub difficulty: f32,
    pub reliability: f32,
    pub tags: Vec<String>,
    pub custom_fields: std::collections::HashMap<String, serde_json::Value>,
}

/// Adaptation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecord {
    pub timestamp: u64,
    pub adaptation_type: AdaptationType,
    pub pattern_id: Option<String>,
    pub old_weight: f32,
    pub new_weight: f32,
    pub confidence_delta: f32,
    pub performance_impact: f32,
    pub reason: String,
}

/// Types of adaptations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    WeightIncrease,
    WeightDecrease,
    PatternDiscovery,
    PatternRemoval,
    ThresholdAdjustment,
    ParameterTuning,
}

/// Atomic feature score
#[derive(Debug)]
pub struct AtomicFeatureScore {
    /// Feature importance score (scaled by 10000)
    pub importance_x10000: AtomicU64,
    
    /// Usage frequency
    pub usage_count: AtomicU64,
    
    /// Success correlation (scaled by 10000)  
    pub success_correlation_x10000: AtomicU64,
    
    /// Last evaluation timestamp
    pub last_evaluated: AtomicU64,
}

/// Atomic learning metrics
#[derive(Debug)]
pub struct AtomicLearningMetrics {
    /// Total adaptations made
    pub total_adaptations: AtomicU64,
    
    /// Successful adaptations
    pub successful_adaptations: AtomicU64,
    
    /// Learning accuracy (scaled by 10000)
    pub learning_accuracy_x10000: AtomicU64,
    
    /// Convergence rate (scaled by 10000)
    pub convergence_rate_x10000: AtomicU64,
    
    /// Pattern discovery rate
    pub pattern_discovery_rate: AtomicU64,
    
    /// Total training examples processed
    pub examples_processed: AtomicU64,
    
    /// Average adaptation time in microseconds
    pub avg_adaptation_time_us: AtomicU64,
    
    /// Last learning session timestamp
    pub last_learning_session: AtomicU64,
}

/// Atomic learning state
#[derive(Debug)]
pub struct AtomicLearningState {
    /// Current learning phase
    pub learning_phase: AtomicU64, // 0=initialization, 1=exploration, 2=exploitation, 3=optimization
    
    /// Active patterns being learned
    pub active_pattern_count: AtomicUsize,
    
    /// System confidence level (scaled by 10000)
    pub system_confidence_x10000: AtomicU64,
    
    /// Learning stability indicator (scaled by 10000)
    pub stability_indicator_x10000: AtomicU64,
    
    /// Last major adaptation timestamp
    pub last_major_adaptation: AtomicU64,
    
    /// Consecutive successful predictions
    pub success_streak: AtomicU64,
    
    /// Learning session ID
    pub session_id: AtomicU64,
}

impl LockFreeContextLearningSystem {
    /// Create a new lock-free learning system
    pub fn new(config: LockFreeLearningConfig) -> Self {
        let parameters = AtomicLearningParameters {
            learning_rate_x10000: AtomicU64::new((config.initial_learning_rate * 10000.0) as u64),
            adaptation_threshold_x10000: AtomicU64::new((config.adaptation_threshold * 10000.0) as u64),
            decay_rate_x10000: AtomicU64::new((config.decay_rate * 10000.0) as u64),
            min_confidence_x10000: AtomicU64::new((config.min_confidence * 10000.0) as u64),
            max_training_examples: AtomicUsize::new(config.max_training_examples),
            learning_enabled: AtomicBool::new(config.learning_enabled),
            auto_adaptation_enabled: AtomicBool::new(config.auto_adaptation_enabled),
        };
        
        Self {
            parameters: Arc::new(parameters),
            training_examples: Arc::new(IndexedRingBuffer::new(config.max_training_examples)),
            pattern_weights: Arc::new(ConcurrentMap::new()),
            performance: Arc::new(AtomicLearningMetrics::default()),
            adaptation_history: Arc::new(IndexedRingBuffer::new(config.max_adaptation_history)),
            feature_importance: Arc::new(ConcurrentMap::new()),
            learning_state: Arc::new(AtomicLearningState::default()),
            config,
        }
    }
    
    /// Add a training example
    pub fn add_training_example(&self, example: TrainingExample) -> bool {
        if !self.parameters.learning_enabled.load(Ordering::Relaxed) {
            return false;
        }
        
        let success = self.training_examples.push(example.clone());
        
        if success {
            self.performance.examples_processed.fetch_add(1, Ordering::Relaxed);
            
            // Update feature importance scores
            self.update_feature_importance(&example);
            
            // Check if we should trigger adaptation
            if self.parameters.auto_adaptation_enabled.load(Ordering::Relaxed) {
                self.check_auto_adaptation(&example);
            }
        }
        
        success
    }
    
    /// Get or create pattern weight
    pub fn get_pattern_weight(&self, pattern_id: &str) -> f32 {
        if let Some(weight) = self.pattern_weights.get(&pattern_id.to_string()) {
            weight.weight_x10000.load(Ordering::Relaxed) as f32 / 10000.0
        } else {
            // Initialize new pattern with default weight
            let default_weight = AtomicPatternWeight::default();
            self.pattern_weights.insert(pattern_id.to_string(), default_weight);
            0.5 // Default weight
        }
    }
    
    /// Update pattern weight based on performance
    pub fn update_pattern_weight(&self, pattern_id: &str, performance_delta: f32, confidence: f32) -> bool {
        if !self.parameters.learning_enabled.load(Ordering::Relaxed) {
            return false;
        }
        
        let learning_rate = self.parameters.learning_rate_x10000.load(Ordering::Relaxed) as f32 / 10000.0;
        let min_confidence = self.parameters.min_confidence_x10000.load(Ordering::Relaxed) as f32 / 10000.0;
        
        if confidence < min_confidence {
            return false;
        }
        
        // Get or create pattern weight
        if !self.pattern_weights.contains_key(&pattern_id.to_string()) {
            self.pattern_weights.insert(pattern_id.to_string(), AtomicPatternWeight::default());
        }
        
        if let Some(weight_entry) = self.pattern_weights.get(&pattern_id.to_string()) {
            let current_weight = weight_entry.weight_x10000.load(Ordering::Relaxed) as f32 / 10000.0;
            
            // Apply learning update
            let weight_delta = learning_rate * performance_delta * confidence;
            let new_weight = (current_weight + weight_delta).clamp(0.0, 1.0);
            
            // Update atomic values
            weight_entry.weight_x10000.store((new_weight * 10000.0) as u64, Ordering::Relaxed);
            weight_entry.confidence_x10000.store((confidence * 10000.0) as u64, Ordering::Relaxed);
            weight_entry.update_count.fetch_add(1, Ordering::Relaxed);
            weight_entry.last_updated.store(current_timestamp_nanos(), Ordering::Relaxed);
            
            // Update success rate
            let success_indicator = if performance_delta > 0.0 { 1.0 } else { 0.0 };
            self.update_success_rate(&weight_entry, success_indicator);
            
            // Record adaptation
            let adaptation = AdaptationRecord {
                timestamp: current_timestamp_nanos(),
                adaptation_type: if weight_delta > 0.0 { AdaptationType::WeightIncrease } else { AdaptationType::WeightDecrease },
                pattern_id: Some(pattern_id.to_string()),
                old_weight: current_weight,
                new_weight,
                confidence_delta: confidence - (weight_entry.confidence_x10000.load(Ordering::Relaxed) as f32 / 10000.0),
                performance_impact: performance_delta,
                reason: format!("Performance-based adaptation: delta={:.4}", performance_delta),
            };
            
            self.adaptation_history.push(adaptation);
            self.performance.total_adaptations.fetch_add(1, Ordering::Relaxed);
            
            if performance_delta > 0.0 {
                self.performance.successful_adaptations.fetch_add(1, Ordering::Relaxed);
            }
            
            true
        } else {
            false
        }
    }
    
    /// Discover new patterns from recent training data
    pub fn discover_patterns(&self, min_frequency: f32, min_quality: f32) -> Vec<String> {
        let recent_examples = self.training_examples.get_last_n(self.config.pattern_discovery_window);
        let mut pattern_candidates = std::collections::HashMap::new();
        
        for example in &recent_examples {
            if example.quality_score >= min_quality {
                // Simple pattern discovery based on feature vectors
                let pattern_id = self.generate_pattern_id(&example.input_features);
                *pattern_candidates.entry(pattern_id).or_insert(0.0) += 1.0;
            }
        }
        
        let min_count = (recent_examples.len() as f32 * min_frequency) as usize;
        let mut new_patterns = Vec::new();
        
        for (pattern_id, count) in pattern_candidates {
            if count as usize >= min_count {
                if !self.pattern_weights.contains_key(&pattern_id) {
                    // New pattern discovered
                    self.pattern_weights.insert(pattern_id.clone(), AtomicPatternWeight::new_with_confidence(0.5, 0.3));
                    
                    let adaptation = AdaptationRecord {
                        timestamp: current_timestamp_nanos(),
                        adaptation_type: AdaptationType::PatternDiscovery,
                        pattern_id: Some(pattern_id.clone()),
                        old_weight: 0.0,
                        new_weight: 0.5,
                        confidence_delta: 0.3,
                        performance_impact: 0.0,
                        reason: format!("Pattern discovered with frequency {:.2}", count / recent_examples.len() as f32),
                    };
                    
                    self.adaptation_history.push(adaptation);
                    new_patterns.push(pattern_id);
                    
                    self.performance.pattern_discovery_rate.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        new_patterns
    }
    
    /// Get learning system statistics
    pub fn get_learning_stats(&self) -> LearningSystemStats {
        let total_adaptations = self.performance.total_adaptations.load(Ordering::Relaxed);
        let successful_adaptations = self.performance.successful_adaptations.load(Ordering::Relaxed);
        
        LearningSystemStats {
            total_patterns: self.pattern_weights.len(),
            active_patterns: self.learning_state.active_pattern_count.load(Ordering::Relaxed),
            total_adaptations,
            successful_adaptations,
            adaptation_success_rate: if total_adaptations > 0 {
                successful_adaptations as f32 / total_adaptations as f32
            } else {
                0.0
            },
            examples_processed: self.performance.examples_processed.load(Ordering::Relaxed),
            learning_accuracy: self.performance.learning_accuracy_x10000.load(Ordering::Relaxed) as f32 / 10000.0,
            system_confidence: self.learning_state.system_confidence_x10000.load(Ordering::Relaxed) as f32 / 10000.0,
            current_learning_rate: self.parameters.learning_rate_x10000.load(Ordering::Relaxed) as f32 / 10000.0,
            learning_phase: self.learning_state.learning_phase.load(Ordering::Relaxed),
            pattern_discovery_rate: self.performance.pattern_discovery_rate.load(Ordering::Relaxed),
        }
    }
    
    /// Optimize learning parameters based on performance
    pub fn optimize_parameters(&self) -> bool {
        let stats = self.get_learning_stats();
        let mut optimized = false;
        
        // Adaptive learning rate
        if stats.adaptation_success_rate > 0.8 {
            // High success rate - can increase learning rate slightly
            let current_rate = self.parameters.learning_rate_x10000.load(Ordering::Relaxed);
            let new_rate = std::cmp::min(current_rate + 100, 5000); // Cap at 0.5
            if new_rate != current_rate {
                self.parameters.learning_rate_x10000.store(new_rate, Ordering::Relaxed);
                optimized = true;
            }
        } else if stats.adaptation_success_rate < 0.4 {
            // Low success rate - decrease learning rate
            let current_rate = self.parameters.learning_rate_x10000.load(Ordering::Relaxed);
            let new_rate = std::cmp::max(current_rate.saturating_sub(100), 10); // Floor at 0.001
            if new_rate != current_rate {
                self.parameters.learning_rate_x10000.store(new_rate, Ordering::Relaxed);
                optimized = true;
            }
        }
        
        // Update learning phase
        if stats.examples_processed > self.config.exploration_threshold as u64 {
            if stats.system_confidence > 0.7 {
                self.learning_state.learning_phase.store(3, Ordering::Relaxed); // Optimization phase
            } else {
                self.learning_state.learning_phase.store(2, Ordering::Relaxed); // Exploitation phase
            }
        } else {
            self.learning_state.learning_phase.store(1, Ordering::Relaxed); // Exploration phase
        }
        
        if optimized {
            let adaptation = AdaptationRecord {
                timestamp: current_timestamp_nanos(),
                adaptation_type: AdaptationType::ParameterTuning,
                pattern_id: None,
                old_weight: 0.0,
                new_weight: 0.0,
                confidence_delta: 0.0,
                performance_impact: 0.0,
                reason: "Automatic parameter optimization".to_string(),
            };
            
            self.adaptation_history.push(adaptation);
            self.learning_state.last_major_adaptation.store(current_timestamp_nanos(), Ordering::Relaxed);
        }
        
        optimized
    }
    
    /// Private helper methods
    fn update_feature_importance(&self, example: &TrainingExample) {
        for (i, &feature_value) in example.input_features.iter().enumerate() {
            let feature_id = format!("feature_{}", i);
            
            if !self.feature_importance.contains_key(&feature_id) {
                self.feature_importance.insert(feature_id.clone(), AtomicFeatureScore::default());
            }
            
            if let Some(feature_score) = self.feature_importance.get(&feature_id) {
                feature_score.usage_count.fetch_add(1, Ordering::Relaxed);
                
                // Simple importance calculation based on feature magnitude
                let importance = feature_value.abs() * example.quality_score;
                let current_importance = feature_score.importance_x10000.load(Ordering::Relaxed) as f32 / 10000.0;
                let new_importance = (current_importance * 0.9 + importance * 0.1).clamp(0.0, 1.0);
                
                feature_score.importance_x10000.store((new_importance * 10000.0) as u64, Ordering::Relaxed);
                feature_score.last_evaluated.store(current_timestamp_nanos(), Ordering::Relaxed);
            }
        }
    }
    
    fn check_auto_adaptation(&self, example: &TrainingExample) {
        let threshold = self.parameters.adaptation_threshold_x10000.load(Ordering::Relaxed) as f32 / 10000.0;
        
        if let (Some(expected), Some(actual)) = (Some(example.expected_output), example.actual_output) {
            let error = (expected - actual).abs();
            
            if error > threshold {
                // Significant error - trigger adaptation
                if let Some(pattern_id) = &example.pattern_id {
                    let performance_delta = -error; // Negative because it's an error
                    self.update_pattern_weight(pattern_id, performance_delta, example.quality_score);
                }
            }
        }
    }
    
    fn update_success_rate(&self, weight: &AtomicPatternWeight, success: f32) {
        let current_rate = weight.success_rate_x10000.load(Ordering::Relaxed) as f32 / 10000.0;
        let alpha = 0.1; // Exponential moving average factor
        let new_rate = (current_rate * (1.0 - alpha) + success * alpha).clamp(0.0, 1.0);
        weight.success_rate_x10000.store((new_rate * 10000.0) as u64, Ordering::Relaxed);
    }
    
    fn generate_pattern_id(&self, features: &[f32]) -> String {
        // Simple pattern ID generation based on feature quantization
        let quantized: Vec<u8> = features
            .iter()
            .map(|&f| ((f.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8)
            .collect();
        
        // Simple hash function to avoid external dependencies
        let mut hash: u64 = 0;
        for &byte in &quantized {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        
        format!("pattern_{:x}", hash)
    }
}

impl Default for AtomicLearningParameters {
    fn default() -> Self {
        Self {
            learning_rate_x10000: AtomicU64::new(1000), // 0.1
            adaptation_threshold_x10000: AtomicU64::new(500), // 0.05
            decay_rate_x10000: AtomicU64::new(50), // 0.005
            min_confidence_x10000: AtomicU64::new(3000), // 0.3
            max_training_examples: AtomicUsize::new(10000),
            learning_enabled: AtomicBool::new(true),
            auto_adaptation_enabled: AtomicBool::new(true),
        }
    }
}

impl Default for AtomicPatternWeight {
    fn default() -> Self {
        Self {
            weight_x10000: AtomicU64::new(5000), // 0.5
            confidence_x10000: AtomicU64::new(1000), // 0.1
            update_count: AtomicU64::new(0),
            last_updated: AtomicU64::new(current_timestamp_nanos()),
            success_rate_x10000: AtomicU64::new(5000), // 0.5
            stability_x10000: AtomicU64::new(5000), // 0.5
        }
    }
}

impl AtomicPatternWeight {
    fn new_with_confidence(weight: f32, confidence: f32) -> Self {
        Self {
            weight_x10000: AtomicU64::new((weight * 10000.0) as u64),
            confidence_x10000: AtomicU64::new((confidence * 10000.0) as u64),
            update_count: AtomicU64::new(0),
            last_updated: AtomicU64::new(current_timestamp_nanos()),
            success_rate_x10000: AtomicU64::new(5000), // 0.5
            stability_x10000: AtomicU64::new(5000), // 0.5
        }
    }
}

impl Clone for AtomicPatternWeight {
    fn clone(&self) -> Self {
        Self {
            weight_x10000: AtomicU64::new(self.weight_x10000.load(Ordering::Relaxed)),
            confidence_x10000: AtomicU64::new(self.confidence_x10000.load(Ordering::Relaxed)),
            update_count: AtomicU64::new(self.update_count.load(Ordering::Relaxed)),
            last_updated: AtomicU64::new(self.last_updated.load(Ordering::Relaxed)),
            success_rate_x10000: AtomicU64::new(self.success_rate_x10000.load(Ordering::Relaxed)),
            stability_x10000: AtomicU64::new(self.stability_x10000.load(Ordering::Relaxed)),
        }
    }
}

impl Clone for AtomicFeatureScore {
    fn clone(&self) -> Self {
        Self {
            importance_x10000: AtomicU64::new(self.importance_x10000.load(Ordering::Relaxed)),
            usage_count: AtomicU64::new(self.usage_count.load(Ordering::Relaxed)),
            success_correlation_x10000: AtomicU64::new(self.success_correlation_x10000.load(Ordering::Relaxed)),
            last_evaluated: AtomicU64::new(self.last_evaluated.load(Ordering::Relaxed)),
        }
    }
}

impl Default for AtomicFeatureScore {
    fn default() -> Self {
        Self {
            importance_x10000: AtomicU64::new(1000), // 0.1
            usage_count: AtomicU64::new(0),
            success_correlation_x10000: AtomicU64::new(5000), // 0.5
            last_evaluated: AtomicU64::new(current_timestamp_nanos()),
        }
    }
}

impl Default for AtomicLearningMetrics {
    fn default() -> Self {
        Self {
            total_adaptations: AtomicU64::new(0),
            successful_adaptations: AtomicU64::new(0),
            learning_accuracy_x10000: AtomicU64::new(5000), // 0.5
            convergence_rate_x10000: AtomicU64::new(1000), // 0.1
            pattern_discovery_rate: AtomicU64::new(0),
            examples_processed: AtomicU64::new(0),
            avg_adaptation_time_us: AtomicU64::new(1000), // 1ms
            last_learning_session: AtomicU64::new(current_timestamp_nanos()),
        }
    }
}

impl Default for AtomicLearningState {
    fn default() -> Self {
        Self {
            learning_phase: AtomicU64::new(0), // Initialization phase
            active_pattern_count: AtomicUsize::new(0),
            system_confidence_x10000: AtomicU64::new(3000), // 0.3
            stability_indicator_x10000: AtomicU64::new(5000), // 0.5
            last_major_adaptation: AtomicU64::new(current_timestamp_nanos()),
            success_streak: AtomicU64::new(0),
            session_id: AtomicU64::new(1),
        }
    }
}

/// Configuration for lock-free learning
#[derive(Debug, Clone)]
pub struct LockFreeLearningConfig {
    pub initial_learning_rate: f32,
    pub adaptation_threshold: f32,
    pub decay_rate: f32,
    pub min_confidence: f32,
    pub max_training_examples: usize,
    pub max_adaptation_history: usize,
    pub pattern_discovery_window: usize,
    pub exploration_threshold: usize,
    pub learning_enabled: bool,
    pub auto_adaptation_enabled: bool,
}

impl Default for LockFreeLearningConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.1,
            adaptation_threshold: 0.05,
            decay_rate: 0.005,
            min_confidence: 0.3,
            max_training_examples: 10000,
            max_adaptation_history: 1000,
            pattern_discovery_window: 100,
            exploration_threshold: 1000,
            learning_enabled: true,
            auto_adaptation_enabled: true,
        }
    }
}

/// Learning system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSystemStats {
    pub total_patterns: usize,
    pub active_patterns: usize,
    pub total_adaptations: u64,
    pub successful_adaptations: u64,
    pub adaptation_success_rate: f32,
    pub examples_processed: u64,
    pub learning_accuracy: f32,
    pub system_confidence: f32,
    pub current_learning_rate: f32,
    pub learning_phase: u64,
    pub pattern_discovery_rate: u64,
}

/// Helper function to get current timestamp in nanoseconds
fn current_timestamp_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_learning_operations() {
        let config = LockFreeLearningConfig::default();
        let learning_system = LockFreeContextLearningSystem::new(config);
        
        // Add training examples
        let example1 = TrainingExample {
            input_features: vec![0.1, 0.2, 0.3],
            expected_output: 0.8,
            actual_output: Some(0.7),
            pattern_id: Some("test_pattern".to_string()),
            timestamp: current_timestamp_nanos(),
            quality_score: 0.9,
            context_metadata: TrainingMetadata::default(),
        };
        
        assert!(learning_system.add_training_example(example1));
        
        // Test pattern weight operations
        let initial_weight = learning_system.get_pattern_weight("test_pattern");
        assert_eq!(initial_weight, 0.5); // Default weight
        
        // Update pattern weight
        learning_system.update_pattern_weight("test_pattern", 0.1, 0.8);
        let updated_weight = learning_system.get_pattern_weight("test_pattern");
        assert!(updated_weight > initial_weight);
        
        let stats = learning_system.get_learning_stats();
        assert_eq!(stats.examples_processed, 1);
        assert_eq!(stats.total_adaptations, 1);
    }
    
    #[test]
    fn test_pattern_discovery() {
        let config = LockFreeLearningConfig {
            pattern_discovery_window: 10,
            ..Default::default()
        };
        let learning_system = LockFreeContextLearningSystem::new(config);
        
        // Add similar examples to trigger pattern discovery
        for i in 0..15 {
            let example = TrainingExample {
                input_features: vec![0.5, 0.5, 0.5], // Similar features
                expected_output: 0.8,
                actual_output: Some(0.8),
                pattern_id: None,
                timestamp: current_timestamp_nanos(),
                quality_score: 0.9,
                context_metadata: TrainingMetadata::default(),
            };
            learning_system.add_training_example(example);
        }
        
        let discovered_patterns = learning_system.discover_patterns(0.5, 0.8);
        assert!(!discovered_patterns.is_empty());
    }
    
    #[test]
    fn test_concurrent_learning() {
        use std::thread;
        use std::sync::Arc;
        
        let config = LockFreeLearningConfig::default();
        let learning_system = Arc::new(LockFreeContextLearningSystem::new(config));
        let num_threads = 4;
        let examples_per_thread = 100;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..num_threads {
            let system_clone = learning_system.clone();
            let handle = thread::spawn(move || {
                for i in 0..examples_per_thread {
                    let example = TrainingExample {
                        input_features: vec![
                            (thread_id as f32 + i as f32) * 0.01,
                            (thread_id as f32 - i as f32) * 0.01,
                            0.5,
                        ],
                        expected_output: 0.7,
                        actual_output: Some(0.6 + i as f32 * 0.001),
                        pattern_id: Some(format!("pattern_{}", thread_id)),
                        timestamp: current_timestamp_nanos(),
                        quality_score: 0.8,
                        context_metadata: TrainingMetadata::default(),
                    };
                    
                    system_clone.add_training_example(example);
                    
                    // Occasionally update pattern weights
                    if i % 10 == 0 {
                        system_clone.update_pattern_weight(
                            &format!("pattern_{}", thread_id),
                            (i as f32 * 0.01) - 0.5,
                            0.8,
                        );
                    }
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = learning_system.get_learning_stats();
        assert_eq!(stats.examples_processed, (num_threads * examples_per_thread) as u64);
        assert!(stats.total_adaptations > 0);
        assert_eq!(stats.total_patterns, num_threads);
    }
}