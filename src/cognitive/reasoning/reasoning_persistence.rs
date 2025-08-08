//! Reasoning persistence and learning system
//! 
//! This module handles persisting reasoning chains, learning from past reasoning,
//! and improving future reasoning performance based on historical data.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use tracing::{info, debug};

use super::{ReasoningChain, ReasoningType, ReasoningResult, ReasoningProblem};

/// Reasoning persistence manager
#[derive(Debug)]
pub struct ReasoningPersistence {
    /// Storage backend
    storage: Arc<RwLock<ReasoningStorage>>,
    
    /// Learning engine
    learning_engine: Arc<RwLock<ReasoningLearningEngine>>,
    
    /// Performance tracker
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    
    /// Configuration
    config: PersistenceConfig,
}

/// Configuration for reasoning persistence
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Storage directory
    pub storage_dir: PathBuf,
    
    /// Maximum number of reasoning chains to keep
    pub max_chains: usize,
    
    /// Enable automatic learning
    pub auto_learn: bool,
    
    /// Learning threshold (minimum confidence to learn from)
    pub learning_threshold: f64,
    
    /// Performance tracking window (in days)
    pub tracking_window_days: u32,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("data/reasoning"),
            max_chains: 10000,
            auto_learn: true,
            learning_threshold: 0.7,
            tracking_window_days: 30,
        }
    }
}

/// Storage backend for reasoning chains
#[derive(Debug)]
struct ReasoningStorage {
    /// In-memory cache
    cache: HashMap<String, StoredReasoningChain>,
    
    /// Storage path
    storage_path: PathBuf,
    
    /// Index by problem type
    type_index: HashMap<ReasoningType, Vec<String>>,
    
    /// Index by success (confidence > threshold)
    success_index: HashMap<bool, Vec<String>>,
}

/// Stored reasoning chain with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredReasoningChain {
    /// The reasoning chain
    pub chain: ReasoningChain,
    
    /// Problem that triggered this reasoning
    pub problem: ReasoningProblem,
    
    /// Result of the reasoning
    pub result: ReasoningResult,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Feedback score (if available)
    pub feedback_score: Option<f64>,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Was this chain successful?
    pub successful: bool,
}

/// Learning engine for improving reasoning
#[derive(Debug)]
struct ReasoningLearningEngine {
    /// Pattern database
    patterns: HashMap<String, LearnedPattern>,
    
    /// Strategy preferences by problem type
    strategy_preferences: HashMap<String, StrategyPreference>,
    
    /// Confidence calibration data
    calibration_data: CalibrationData,
}

/// Learned pattern from successful reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnedPattern {
    /// Pattern ID
    pub id: String,
    
    /// Pattern description
    pub description: String,
    
    /// Problem characteristics that match this pattern
    pub problem_features: Vec<String>,
    
    /// Successful reasoning steps
    pub successful_steps: Vec<String>,
    
    /// Average confidence when applied
    pub avg_confidence: f64,
    
    /// Number of times applied
    pub application_count: u32,
    
    /// Success rate
    pub success_rate: f64,
}

/// Strategy preference for different problem types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StrategyPreference {
    /// Problem type identifier
    pub problem_type: String,
    
    /// Preferred reasoning types and their weights
    pub reasoning_weights: HashMap<ReasoningType, f64>,
    
    /// Average processing time by type
    pub avg_processing_times: HashMap<ReasoningType, u64>,
    
    /// Success rates by type
    pub success_rates: HashMap<ReasoningType, f64>,
}

/// Confidence calibration data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CalibrationData {
    /// Predicted vs actual success rates
    pub calibration_curve: Vec<(f64, f64)>,
    
    /// Overconfidence factor
    pub overconfidence_factor: f64,
    
    /// Calibration samples
    pub sample_count: u32,
}

/// Performance tracking
#[derive(Debug, Default)]
struct PerformanceTracker {
    /// Daily performance metrics
    daily_metrics: HashMap<String, DailyMetrics>,
    
    /// Overall statistics
    overall_stats: OverallStats,
}

/// Daily performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct DailyMetrics {
    /// Date
    pub date: String,
    
    /// Total reasoning operations
    pub total_operations: u32,
    
    /// Successful operations
    pub successful_operations: u32,
    
    /// Average confidence
    pub avg_confidence: f64,
    
    /// Average processing time
    pub avg_processing_time_ms: u64,
    
    /// Reasoning type distribution
    pub type_distribution: HashMap<ReasoningType, u32>,
}

/// Overall statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverallStats {
    /// Total operations
    pub total_operations: u64,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Most successful reasoning type
    pub best_reasoning_type: Option<ReasoningType>,
    
    /// Average improvement over time
    pub improvement_rate: f64,
}

impl ReasoningPersistence {
    /// Create new reasoning persistence manager
    pub async fn new(config: PersistenceConfig) -> Result<Self> {
        // Ensure storage directory exists
        tokio::fs::create_dir_all(&config.storage_dir).await?;
        
        let storage = Arc::new(RwLock::new(ReasoningStorage::new(config.storage_dir.clone())));
        let learning_engine = Arc::new(RwLock::new(ReasoningLearningEngine::new()));
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::default()));
        
        let persistence = Self {
            storage,
            learning_engine,
            performance_tracker,
            config,
        };
        
        // Load existing data
        persistence.load_from_disk().await?;
        
        info!("🧠 Reasoning persistence initialized");
        Ok(persistence)
    }
    
    /// Store a reasoning result
    pub async fn store_reasoning(&self, problem: &ReasoningProblem, result: &ReasoningResult) -> Result<()> {
        debug!("💾 Storing reasoning result for session: {}", result.session_id);
        
        // Determine if successful based on confidence
        let successful = result.confidence >= self.config.learning_threshold;
        
        // Create stored chain for each reasoning chain in the result
        for chain in &result.reasoning_chains {
            let stored = StoredReasoningChain {
                chain: chain.clone(),
                problem: problem.clone(),
                result: result.clone(),
                timestamp: Utc::now(),
                feedback_score: None,
                tags: Self::extract_tags(problem, chain),
                successful,
            };
            
            // Store in memory
            let mut storage = self.storage.write().await;
            storage.add_chain(stored.clone()).await?;
        }
        
        // Update performance tracking
        self.update_performance_tracking(result).await?;
        
        // Trigger learning if enabled
        if self.config.auto_learn && successful {
            self.trigger_learning(problem, result).await?;
        }
        
        Ok(())
    }
    
    /// Retrieve similar reasoning chains
    pub async fn find_similar_reasoning(&self, problem: &ReasoningProblem) -> Result<Vec<StoredReasoningChain>> {
        let storage = self.storage.read().await;
        
        let mut similar_chains = Vec::new();
        
        // Find chains with similar problem characteristics
        for stored in storage.cache.values() {
            let similarity = Self::calculate_similarity(problem, &stored.problem);
            if similarity > 0.7 {
                similar_chains.push(stored.clone());
            }
        }
        
        // Sort by relevance (similarity * confidence * success)
        similar_chains.sort_by(|a, b| {
            let score_a = Self::calculate_relevance_score(problem, &a.problem, &a.chain);
            let score_b = Self::calculate_relevance_score(problem, &b.problem, &b.chain);
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        // Return top 5
        Ok(similar_chains.into_iter().take(5).collect())
    }
    
    /// Get learned patterns for a problem type
    pub async fn get_learned_patterns(&self, problem_type: &str) -> Result<Vec<LearnedPattern>> {
        let learning = self.learning_engine.read().await;
        
        let patterns: Vec<_> = learning.patterns.values()
            .filter(|p| p.problem_features.contains(&problem_type.to_string()))
            .cloned()
            .collect();
        
        Ok(patterns)
    }
    
    /// Get strategy recommendations
    pub async fn get_strategy_recommendations(&self, problem: &ReasoningProblem) -> Result<HashMap<ReasoningType, f64>> {
        let learning = self.learning_engine.read().await;
        
        // Extract problem type
        let problem_type = Self::classify_problem(problem);
        
        // Get strategy preference if exists
        if let Some(preference) = learning.strategy_preferences.get(&problem_type) {
            Ok(preference.reasoning_weights.clone())
        } else {
            // Return default weights
            Ok(Self::default_reasoning_weights())
        }
    }
    
    /// Update reasoning with feedback
    pub async fn update_feedback(&self, session_id: &str, feedback_score: f64) -> Result<()> {
        let mut storage = self.storage.write().await;
        
        // Find and update all chains from this session
        for stored in storage.cache.values_mut() {
            if stored.result.session_id == session_id {
                stored.feedback_score = Some(feedback_score);
                stored.successful = feedback_score >= 0.7;
            }
        }
        
        // Trigger re-learning with feedback
        if self.config.auto_learn {
            self.update_learning_with_feedback(session_id, feedback_score).await?;
        }
        
        Ok(())
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<OverallStats> {
        let tracker = self.performance_tracker.read().await;
        Ok(tracker.overall_stats.clone())
    }
    
    /// Get calibrated confidence
    pub async fn get_calibrated_confidence(&self, raw_confidence: f64) -> Result<f64> {
        let learning = self.learning_engine.read().await;
        let calibration = &learning.calibration_data;
        
        // Apply calibration curve
        let calibrated = raw_confidence * (1.0 - calibration.overconfidence_factor);
        
        Ok(calibrated.max(0.0).min(1.0))
    }
    
    // Helper methods
    
    /// Extract tags from problem and chain
    fn extract_tags(problem: &ReasoningProblem, chain: &ReasoningChain) -> Vec<String> {
        let mut tags = Vec::new();
        
        // Add problem domain tags
        tags.extend(problem.required_knowledge_domains.clone());
        
        // Add reasoning type
        tags.push(format!("{:?}", chain.chain_type));
        
        // Add complexity tags
        if problem.requires_creativity {
            tags.push("creative".to_string());
        }
        if problem.involves_uncertainty {
            tags.push("uncertain".to_string());
        }
        
        tags
    }
    
    /// Calculate similarity between problems
    fn calculate_similarity(p1: &ReasoningProblem, p2: &ReasoningProblem) -> f64 {
        let mut score = 0.0;
        let mut factors = 0.0;
        
        // Domain overlap
        let domains1: std::collections::HashSet<_> = p1.required_knowledge_domains.iter().collect();
        let domains2: std::collections::HashSet<_> = p2.required_knowledge_domains.iter().collect();
        let overlap = domains1.intersection(&domains2).count() as f64;
        let union = domains1.union(&domains2).count() as f64;
        if union > 0.0 {
            score += overlap / union;
            factors += 1.0;
        }
        
        // Variable similarity
        let vars1: std::collections::HashSet<_> = p1.variables.iter().collect();
        let vars2: std::collections::HashSet<_> = p2.variables.iter().collect();
        let var_overlap = vars1.intersection(&vars2).count() as f64;
        let var_union = vars1.union(&vars2).count() as f64;
        if var_union > 0.0 {
            score += var_overlap / var_union * 0.5;
            factors += 0.5;
        }
        
        // Boolean features
        if p1.requires_creativity == p2.requires_creativity {
            score += 0.2;
        }
        if p1.involves_uncertainty == p2.involves_uncertainty {
            score += 0.2;
        }
        factors += 0.4;
        
        score / factors
    }
    
    /// Calculate relevance score
    fn calculate_relevance_score(target: &ReasoningProblem, stored: &ReasoningProblem, chain: &ReasoningChain) -> f64 {
        let similarity = Self::calculate_similarity(target, stored);
        let confidence = chain.confidence;
        let recency_factor = 1.0; // Could decay based on age
        
        similarity * confidence * recency_factor
    }
    
    /// Classify problem type
    fn classify_problem(problem: &ReasoningProblem) -> String {
        if problem.requires_creativity {
            "creative".to_string()
        } else if problem.involves_uncertainty {
            "uncertain".to_string()
        } else if problem.required_knowledge_domains.is_empty() {
            "general".to_string()
        } else {
            problem.required_knowledge_domains.first().unwrap_or(&"general".to_string()).clone()
        }
    }
    
    /// Default reasoning weights
    fn default_reasoning_weights() -> HashMap<ReasoningType, f64> {
        let mut weights = HashMap::new();
        weights.insert(ReasoningType::Logical, 1.0);
        weights.insert(ReasoningType::Causal, 0.9);
        weights.insert(ReasoningType::Analogical, 0.8);
        weights.insert(ReasoningType::Contextual, 0.85);
        weights.insert(ReasoningType::Collaborative, 0.9);
        weights.insert(ReasoningType::Temporal, 0.8);
        weights.insert(ReasoningType::Deductive, 0.95);
        weights.insert(ReasoningType::Inductive, 0.7);
        weights.insert(ReasoningType::Abductive, 0.6);
        weights
    }
    
    /// Update performance tracking
    async fn update_performance_tracking(&self, result: &ReasoningResult) -> Result<()> {
        let mut tracker = self.performance_tracker.write().await;
        
        // Update daily metrics
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let daily = tracker.daily_metrics.entry(today).or_insert_with(DailyMetrics::default);
        
        daily.total_operations += 1;
        if result.confidence >= self.config.learning_threshold {
            daily.successful_operations += 1;
        }
        
        // Update averages
        let n = daily.total_operations as f64;
        daily.avg_confidence = (daily.avg_confidence * (n - 1.0) + result.confidence) / n;
        daily.avg_processing_time_ms = 
            ((daily.avg_processing_time_ms as f64 * (n - 1.0) + result.processing_time_ms as f64) / n) as u64;
        
        // Update type distribution
        for chain in &result.reasoning_chains {
            *daily.type_distribution.entry(chain.chain_type.clone()).or_insert(0) += 1;
        }
        
        // Update overall stats
        tracker.overall_stats.total_operations += 1;
        let success_rate = tracker.daily_metrics.values()
            .map(|d| d.successful_operations as f64 / d.total_operations.max(1) as f64)
            .sum::<f64>() / tracker.daily_metrics.len().max(1) as f64;
        tracker.overall_stats.success_rate = success_rate;
        
        Ok(())
    }
    
    /// Trigger learning from successful reasoning
    async fn trigger_learning(&self, problem: &ReasoningProblem, result: &ReasoningResult) -> Result<()> {
        let mut learning = self.learning_engine.write().await;
        
        // Extract patterns from successful chains
        for chain in &result.reasoning_chains {
            if chain.confidence >= self.config.learning_threshold {
                learning.learn_from_chain(problem, chain)?;
            }
        }
        
        // Update strategy preferences
        let problem_type = Self::classify_problem(problem);
        learning.update_strategy_preference(&problem_type, result)?;
        
        // Update confidence calibration
        learning.update_calibration(result.confidence, result.confidence >= self.config.learning_threshold)?;
        
        Ok(())
    }
    
    /// Update learning with feedback
    async fn update_learning_with_feedback(&self, session_id: &str, feedback_score: f64) -> Result<()> {
        let storage = self.storage.read().await;
        let mut learning = self.learning_engine.write().await;
        
        // Find chains from this session
        for stored in storage.cache.values() {
            if stored.result.session_id == session_id {
                // Update pattern success rates based on feedback
                learning.update_pattern_feedback(&stored.chain, feedback_score)?;
            }
        }
        
        Ok(())
    }
    
    /// Load from disk
    async fn load_from_disk(&self) -> Result<()> {
        let chains_path = self.config.storage_dir.join("chains.json");
        let patterns_path = self.config.storage_dir.join("patterns.json");
        let metrics_path = self.config.storage_dir.join("metrics.json");
        
        // Load chains
        if chains_path.exists() {
            let json = tokio::fs::read_to_string(&chains_path).await?;
            let chains: Vec<StoredReasoningChain> = serde_json::from_str(&json)?;
            
            let mut storage = self.storage.write().await;
            for chain in chains {
                storage.cache.insert(chain.chain.id.clone(), chain);
            }
        }
        
        // Load patterns
        if patterns_path.exists() {
            let json = tokio::fs::read_to_string(&patterns_path).await?;
            let patterns: HashMap<String, LearnedPattern> = serde_json::from_str(&json)?;
            
            let mut learning = self.learning_engine.write().await;
            learning.patterns = patterns;
        }
        
        // Load metrics
        if metrics_path.exists() {
            let json = tokio::fs::read_to_string(&metrics_path).await?;
            let metrics: OverallStats = serde_json::from_str(&json)?;
            
            let mut tracker = self.performance_tracker.write().await;
            tracker.overall_stats = metrics;
        }
        
        Ok(())
    }
    
    /// Save to disk
    pub async fn save_to_disk(&self) -> Result<()> {
        // Save chains
        let storage = self.storage.read().await;
        let chains: Vec<_> = storage.cache.values().cloned().collect();
        let json = serde_json::to_string_pretty(&chains)?;
        tokio::fs::write(self.config.storage_dir.join("chains.json"), json).await?;
        
        // Save patterns
        let learning = self.learning_engine.read().await;
        let json = serde_json::to_string_pretty(&learning.patterns)?;
        tokio::fs::write(self.config.storage_dir.join("patterns.json"), json).await?;
        
        // Save metrics
        let tracker = self.performance_tracker.read().await;
        let json = serde_json::to_string_pretty(&tracker.overall_stats)?;
        tokio::fs::write(self.config.storage_dir.join("metrics.json"), json).await?;
        
        Ok(())
    }
}

impl ReasoningStorage {
    fn new(storage_path: PathBuf) -> Self {
        Self {
            cache: HashMap::new(),
            storage_path,
            type_index: HashMap::new(),
            success_index: HashMap::new(),
        }
    }
    
    async fn add_chain(&mut self, stored: StoredReasoningChain) -> Result<()> {
        let chain_id = stored.chain.id.clone();
        let chain_type = stored.chain.chain_type.clone();
        let successful = stored.successful;
        
        // Add to cache
        self.cache.insert(chain_id.clone(), stored);
        
        // Update indices
        self.type_index.entry(chain_type).or_insert_with(Vec::new).push(chain_id.clone());
        self.success_index.entry(successful).or_insert_with(Vec::new).push(chain_id);
        
        Ok(())
    }
}

impl ReasoningLearningEngine {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            strategy_preferences: HashMap::new(),
            calibration_data: CalibrationData::default(),
        }
    }
    
    fn learn_from_chain(&mut self, problem: &ReasoningProblem, chain: &ReasoningChain) -> Result<()> {
        // Extract pattern features
        let pattern_id = format!("pattern_{}", uuid::Uuid::new_v4());
        let pattern = LearnedPattern {
            id: pattern_id.clone(),
            description: format!("Pattern from {} reasoning", chain.chain_type.clone() as i32),
            problem_features: problem.required_knowledge_domains.clone(),
            successful_steps: chain.steps.iter().map(|s| s.description.clone()).collect(),
            avg_confidence: chain.confidence,
            application_count: 1,
            success_rate: 1.0,
        };
        
        self.patterns.insert(pattern_id, pattern);
        Ok(())
    }
    
    fn update_strategy_preference(&mut self, problem_type: &str, result: &ReasoningResult) -> Result<()> {
        let preference = self.strategy_preferences.entry(problem_type.to_string())
            .or_insert_with(|| StrategyPreference {
                problem_type: problem_type.to_string(),
                reasoning_weights: HashMap::new(),
                avg_processing_times: HashMap::new(),
                success_rates: HashMap::new(),
            });
        
        // Update weights based on confidence
        for chain in &result.reasoning_chains {
            let weight = preference.reasoning_weights.entry(chain.chain_type.clone()).or_insert(0.0);
            *weight = (*weight + chain.confidence) / 2.0;
            
            // Update processing times
            let time = preference.avg_processing_times.entry(chain.chain_type.clone()).or_insert(0);
            *time = (*time + chain.processing_time_ms) / 2;
            
            // Update success rates
            let rate = preference.success_rates.entry(chain.chain_type.clone()).or_insert(0.0);
            *rate = (*rate + if chain.confidence > 0.7 { 1.0 } else { 0.0 }) / 2.0;
        }
        
        Ok(())
    }
    
    fn update_calibration(&mut self, predicted: f64, actual_success: bool) -> Result<()> {
        self.calibration_data.sample_count += 1;
        
        // Update calibration curve
        let actual = if actual_success { 1.0 } else { 0.0 };
        self.calibration_data.calibration_curve.push((predicted, actual));
        
        // Update overconfidence factor (simple exponential smoothing)
        let error = predicted - actual;
        self.calibration_data.overconfidence_factor = 
            self.calibration_data.overconfidence_factor * 0.9 + error.abs() * 0.1;
        
        Ok(())
    }
    
    fn update_pattern_feedback(&mut self, chain: &ReasoningChain, feedback: f64) -> Result<()> {
        // Find patterns that match this chain's steps
        for pattern in self.patterns.values_mut() {
            let step_match = chain.steps.iter()
                .filter(|s| pattern.successful_steps.contains(&s.description))
                .count();
            
            if step_match > 0 {
                // Update pattern metrics
                pattern.application_count += 1;
                let n = pattern.application_count as f64;
                pattern.success_rate = (pattern.success_rate * (n - 1.0) + feedback) / n;
                pattern.avg_confidence = (pattern.avg_confidence * (n - 1.0) + chain.confidence) / n;
            }
        }
        
        Ok(())
    }
}