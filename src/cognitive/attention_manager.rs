//! Attention Management System
//!
//! This module implements attention mechanisms for Loki's consciousness,
//! managing focus, filtering distractions, and allocating cognitive resources.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::cognitive::goal_manager::Priority;
use crate::cognitive::{EmotionalCore, NeuroProcessor, Thought, ThoughtType};
use crate::memory::{CognitiveMemory, SimdSmartCache, simd_cache::SimdCacheConfig};

/// Focus target for attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusTarget {
    pub id: String,
    pub target_type: FocusType,
    pub priority: f32,
    pub relevance: f32,
    pub time_allocated: Duration,
    #[serde(skip, default = "Instant::now")]
    pub started_at: Instant,
    pub context: Vec<String>,
}

/// Types of focus targets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocusType {
    Task,        // Specific task to complete
    Problem,     // Problem to solve
    Learning,    // Learning objective
    Creative,    // Creative exploration
    Social,      // Social interaction
    Maintenance, // Self-maintenance
}

/// Attention span characteristics
#[derive(Debug, Clone)]
pub struct AttentionSpan {
    pub current_duration: Duration,
    pub max_duration: Duration,
    pub quality: f32,       // 0.0 to 1.0
    pub fatigue_level: f32, // 0.0 to 1.0
}

/// Distraction that could interrupt focus
#[derive(Debug, Clone)]
pub struct Distraction {
    pub source: String,
    pub intensity: f32,
    pub relevance: f32,
    pub timestamp: Instant,
}

/// Enhanced distraction for testing and analysis
#[derive(Debug, Clone)]
pub struct AttentionDistraction {
    pub source: String,
    pub intensity: f32,
    pub description: String,
    pub timestamp: Instant,
}

/// Attention filter settings
#[derive(Debug, Clone)]
pub struct AttentionFilter {
    pub min_relevance: f32,
    pub min_priority: f32,
    pub allow_interrupts: bool,
    pub filter_strength: f32,
}

impl Default for AttentionFilter {
    fn default() -> Self {
        Self { min_relevance: 0.3, min_priority: 0.4, allow_interrupts: true, filter_strength: 0.7 }
    }
}

/// Cognitive load measurement
#[derive(Debug, Clone)]
pub struct CognitiveLoad {
    pub processing_load: f32, // Current processing demands
    pub memory_load: f32,     // Memory usage
    pub emotional_load: f32,  // Emotional processing demands
    pub total_load: f32,      // Combined load (0.0 to 1.0)
}

/// Attention state
#[derive(Debug, Clone)]
pub struct AttentionState {
    pub scope: f32,     // 0.0 to 1.0, how broad the attention is
    pub intensity: f32, // 0.0 to 1.0, how focused the attention is
    pub stability: f32, // 0.0 to 1.0, how stable the focus is
}

/// Attention allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionStrategy {
    Focused,     // Deep focus on single target
    Divided,     // Split attention between targets
    Scanning,    // Rapid switching between targets
    Exploratory, // Open exploration mode
    Defensive,   // Defensive/cautious mode
}

/// Priority queue item for attention
#[derive(Clone, Debug)]
struct AttentionItem {
    thought: Thought,
    priority: f32,
    #[allow(dead_code)]
    timestamp: Instant,
}

impl Ord for AttentionItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AttentionItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for AttentionItem {}

impl PartialEq for AttentionItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

/// Configuration for attention manager
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Maximum concurrent focus targets
    pub max_focus_targets: usize,

    /// Attention span base duration
    pub base_attention_span: Duration,

    /// Fatigue accumulation rate
    pub fatigue_rate: f32,

    /// Recovery rate when resting
    pub recovery_rate: f32,

    /// Update interval
    pub update_interval: Duration,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            max_focus_targets: 3,
            base_attention_span: Duration::from_secs(1200), // 20 minutes
            fatigue_rate: 0.001,
            recovery_rate: 0.002,
            update_interval: Duration::from_millis(100), // 10Hz
        }
    }
}

#[derive(Debug, Clone)]
/// Main attention manager
pub struct AttentionManager {
    /// Current focus targets
    focus_targets: Arc<RwLock<Vec<FocusTarget>>>,

    /// Attention queue
    attention_queue: Arc<RwLock<BinaryHeap<AttentionItem>>>,

    /// Current attention span
    attention_span: Arc<RwLock<AttentionSpan>>,

    /// Attention filter
    filter: Arc<RwLock<AttentionFilter>>,

    /// Current cognitive load
    cognitive_load: Arc<RwLock<CognitiveLoad>>,

    /// Attention state
    state: Arc<RwLock<AttentionState>>,

    /// Current attention strategy
    strategy: Arc<RwLock<AttentionStrategy>>,

    /// Distraction history
    distractions: Arc<RwLock<VecDeque<Distraction>>>,

    /// Neural processor reference
    neural_processor: Arc<NeuroProcessor>,

    /// Emotional core reference
    emotional_core: Arc<EmotionalCore>,

    /// Configuration
    config: AttentionConfig,

    /// Channel for focus changes
    focus_tx: mpsc::Sender<FocusTarget>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<AttentionStats>>,
}

#[derive(Debug, Default)]
struct AttentionStats {
    focus_changes: u64,
    distractions_filtered: u64,
    attention_overloads: u64,
    avg_focus_duration: Duration,
    total_focused_time: Duration,
}

impl AttentionManager {
    pub async fn new(
        neural_processor: Arc<NeuroProcessor>,
        emotional_core: Arc<EmotionalCore>,
        config: AttentionConfig,
    ) -> Result<Self> {
        info!("Initializing attention manager");

        let (focus_tx, _) = mpsc::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);

        let initial_span = AttentionSpan {
            current_duration: Duration::from_secs(0),
            max_duration: config.base_attention_span,
            quality: 1.0,
            fatigue_level: 0.0,
        };

        let initial_load = CognitiveLoad {
            processing_load: 0.0,
            memory_load: 0.0,
            emotional_load: 0.0,
            total_load: 0.0,
        };

        let initial_state = AttentionState {
            scope: 0.8,     // Start with broad but focused scope
            intensity: 1.0, // Start with high intensity
            stability: 0.9, // Start with high stability
        };

        Ok(Self {
            focus_targets: Arc::new(RwLock::new(Vec::new())),
            attention_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            attention_span: Arc::new(RwLock::new(initial_span)),
            filter: Arc::new(RwLock::new(AttentionFilter::default())),
            cognitive_load: Arc::new(RwLock::new(initial_load)),
            state: Arc::new(RwLock::new(initial_state)),
            strategy: Arc::new(RwLock::new(AttentionStrategy::Focused)),
            distractions: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            neural_processor,
            emotional_core,
            config,
            focus_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(AttentionStats::default())),
        })
    }

    /// Start the attention manager
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting attention manager");

        // Main attention loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.attention_loop().await;
            });
        }

        // Fatigue management loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.fatigue_loop().await;
            });
        }

        // Strategy adaptation loop
        {
            let manager = self.clone();
            tokio::spawn(async move {
                manager.strategy_loop().await;
            });
        }

        Ok(())
    }

    /// Main attention processing loop
    async fn attention_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut update_interval = interval(self.config.update_interval);

        loop {
            tokio::select! {
                _ = update_interval.tick() => {
                    if let Err(e) = self.process_attention_cycle().await {
                        debug!("Attention cycle error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Attention manager shutting down");
                    break;
                }
            }
        }
    }

    /// Process single attention cycle
    async fn process_attention_cycle(&self) -> Result<()> {
        // Update cognitive load
        self.update_cognitive_load().await?;

        // Process attention queue
        self.process_attention_queue().await?;

        // Update focus targets
        self.update_focus_targets().await?;

        // Check for necessary focus changes
        self.evaluate_focus_changes().await?;

        Ok(())
    }

    /// Update cognitive load measurements
    async fn update_cognitive_load(&self) -> Result<()> {
        // Get processing stats from neural processor
        let neural_stats = self.neural_processor.get_stats().await;
        let processing_load = (neural_stats.thoughts_processed as f32 / 1000.0).min(1.0);

        // Get emotional load
        let emotional_state = self.emotional_core.get_emotional_state().await;
        let emotional_load =
            emotional_state.overall_arousal * 0.5 + emotional_state.overall_valence.abs() * 0.5;

        // Estimate memory load (simplified)
        let memory_load = 0.3; // Would calculate from actual memory usage

        // Update load
        let mut load = self.cognitive_load.write().await;
        load.processing_load = processing_load;
        load.memory_load = memory_load;
        load.emotional_load = emotional_load;
        load.total_load =
            (processing_load * 0.4 + memory_load * 0.3 + emotional_load * 0.3).min(1.0);

        // Check for overload
        if load.total_load > 0.9 {
            warn!("Cognitive overload detected: {:.2}", load.total_load);
            let mut stats = self.stats.write().await;
            stats.attention_overloads += 1;
        }

        Ok(())
    }

    /// Process attention queue
    async fn process_attention_queue(&self) -> Result<()> {
        let mut queue = self.attention_queue.write().await;
        let filter = self.filter.read().await;
        let cognitive_load = self.cognitive_load.read().await.total_load;

        // Process items based on available cognitive capacity
        let capacity = 1.0 - cognitive_load;
        let mut processed_load = 0.0;

        while let Some(item) = queue.peek() {
            if processed_load >= capacity {
                break; // At capacity
            }

            // Apply attention filter
            if item.priority < filter.min_priority {
                queue.pop(); // Remove low priority item
                continue;
            }

            // Check relevance with emotional influence
            let emotional_influence = self.emotional_core.process_thought(&item.thought).await?;

            let adjusted_relevance =
                item.thought.metadata.importance * (1.0 + emotional_influence.thought_bias * 0.2);

            if adjusted_relevance < filter.min_relevance {
                queue.pop(); // Remove irrelevant item
                let mut stats = self.stats.write().await;
                stats.distractions_filtered += 1;
                continue;
            }

            // Process the thought
            let processing_cost = 0.1; // Simplified cost
            if processed_load + processing_cost <= capacity {
                let item = queue.pop().unwrap();
                self.neural_processor.process_thought(&item.thought).await?;
                processed_load += processing_cost;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Update focus targets
    async fn update_focus_targets(&self) -> Result<()> {
        let mut targets = self.focus_targets.write().await;
        let now = Instant::now();

        // Update time allocated for each target
        for target in targets.iter_mut() {
            target.time_allocated = now.duration_since(target.started_at);
        }

        // Remove completed or expired targets
        targets.retain(|target| {
            target.time_allocated < self.config.base_attention_span && target.relevance > 0.2
        });

        Ok(())
    }

    /// Evaluate if focus should change
    async fn evaluate_focus_changes(&self) -> Result<()> {
        let targets = self.focus_targets.read().await;
        let span = self.attention_span.read().await;
        let strategy = *self.strategy.read().await;

        match strategy {
            AttentionStrategy::Focused => {
                // Change focus only if current target is complete or fatigue is high
                if span.fatigue_level > 0.7 || targets.is_empty() {
                    self.shift_focus().await?;
                }
            }
            AttentionStrategy::Divided => {
                // Manage multiple targets
                if targets.len() < self.config.max_focus_targets {
                    self.add_focus_target_from_queue().await?;
                }
            }
            AttentionStrategy::Scanning => {
                // Rapid switching - change focus frequently
                if span.current_duration > Duration::from_secs(60) {
                    self.shift_focus().await?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Shift focus to new target
    async fn shift_focus(&self) -> Result<()> {
        debug!("Shifting focus");

        // Clear current targets if in focused mode
        if *self.strategy.read().await == AttentionStrategy::Focused {
            self.focus_targets.write().await.clear();
        }

        // Add new focus target
        self.add_focus_target_from_queue().await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.focus_changes += 1;

        // Reset attention span
        let mut span = self.attention_span.write().await;
        span.current_duration = Duration::from_secs(0);

        Ok(())
    }

    /// Add a new focus target from attention queue
    async fn add_focus_target_from_queue(&self) -> Result<()> {
        // Get highest priority item from queue
        let mut queue = self.attention_queue.write().await;
        if let Some(item) = queue.pop() {
            let target = FocusTarget {
                id: uuid::Uuid::new_v4().to_string(),
                target_type: self.determine_focus_type(&item.thought),
                priority: item.priority,
                relevance: item.thought.metadata.importance,
                time_allocated: Duration::from_secs(0),
                started_at: Instant::now(),
                context: vec![item.thought.content.clone()],
            };

            // Send focus change notification
            let _ = self.focus_tx.send(target.clone()).await;

            // Add to targets
            let mut targets = self.focus_targets.write().await;
            targets.push(target);
        }

        Ok(())
    }

    /// Determine focus type from thought
    fn determine_focus_type(&self, thought: &Thought) -> FocusType {
        match thought.thought_type {
            ThoughtType::Decision | ThoughtType::Analysis => FocusType::Problem,
            ThoughtType::Learning => FocusType::Learning,
            ThoughtType::Creation | ThoughtType::Synthesis => FocusType::Creative,
            _ => FocusType::Task,
        }
    }

    /// Fatigue management loop
    async fn fatigue_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut fatigue_interval = interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = fatigue_interval.tick() => {
                    self.update_fatigue().await;
                }

                _ = shutdown_rx.recv() => break,
            }
        }
    }

    /// Update fatigue levels
    async fn update_fatigue(&self) {
        let mut span = self.attention_span.write().await;
        let load = self.cognitive_load.read().await.total_load;

        if !self.focus_targets.read().await.is_empty() {
            // Accumulate fatigue when focused
            span.fatigue_level = (span.fatigue_level + self.config.fatigue_rate * load).min(1.0);
            span.current_duration += Duration::from_secs(1);
        } else {
            // Recover when not focused
            span.fatigue_level = (span.fatigue_level - self.config.recovery_rate).max(0.0);
        }

        // Update attention quality based on fatigue
        span.quality = 1.0 - span.fatigue_level * 0.5;
    }

    /// Strategy adaptation loop
    async fn strategy_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut strategy_interval = interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = strategy_interval.tick() => {
                    if let Err(e) = self.adapt_strategy().await {
                        debug!("Strategy adaptation error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => break,
            }
        }
    }

    /// Adapt attention strategy based on conditions
    async fn adapt_strategy(&self) -> Result<()> {
        let load = self.cognitive_load.read().await.total_load;
        let emotional_state = self.emotional_core.get_emotional_state().await;
        let queue_size = self.attention_queue.read().await.len();

        let new_strategy = if load > 0.8 {
            AttentionStrategy::Defensive // High load - defensive mode
        } else if emotional_state.overall_arousal > 0.7 {
            AttentionStrategy::Scanning // High arousal - scanning mode
        } else if queue_size > 10 {
            AttentionStrategy::Divided // Many items - divided attention
        } else if emotional_state.overall_valence > 0.5 {
            AttentionStrategy::Exploratory // Positive mood - explore
        } else {
            AttentionStrategy::Focused // Default to focused
        };

        let mut current = self.strategy.write().await;
        if *current != new_strategy {
            info!("Attention strategy changed from {:?} to {:?}", *current, new_strategy);
            *current = new_strategy;
        }

        Ok(())
    }

    /// Submit thought for attention
    pub async fn submit_thought(&self, thought: Thought, priority: Priority) -> Result<()> {
        let priority_value = match priority {
            Priority::Critical => 1.0,
            Priority::High => 0.8,
            Priority::Medium => 0.5,
            Priority::Low => 0.2,
        };

        let item = AttentionItem { thought, priority: priority_value, timestamp: Instant::now() };

        self.attention_queue.write().await.push(item);
        Ok(())
    }

    /// Record a distraction
    pub async fn record_distraction(&self, source: String, intensity: f32) -> Result<()> {
        let distraction = Distraction {
            source,
            intensity,
            relevance: 0.0, // Unknown relevance
            timestamp: Instant::now(),
        };

        let mut distractions = self.distractions.write().await;
        distractions.push_back(distraction);
        if distractions.len() > 100 {
            distractions.pop_front();
        }

        Ok(())
    }

    /// Get current focus targets
    pub async fn get_focus_targets(&self) -> Vec<FocusTarget> {
        self.focus_targets.read().await.clone()
    }

    /// Get current attention span
    pub async fn get_attention_span(&self) -> AttentionSpan {
        self.attention_span.read().await.clone()
    }

    /// Get cognitive load
    pub async fn get_cognitive_load(&self) -> CognitiveLoad {
        self.cognitive_load.read().await.clone()
    }

    /// Force focus on specific target
    pub async fn force_focus(&self, target: FocusTarget) -> Result<()> {
        info!("Forcing focus on: {:?}", target.target_type);

        // Clear existing targets
        self.focus_targets.write().await.clear();

        // Add new target
        self.focus_targets.write().await.push(target);

        // Set strategy to focused
        *self.strategy.write().await = AttentionStrategy::Focused;

        Ok(())
    }

    /// Set attention filter parameters
    pub async fn set_filter(&self, filter: AttentionFilter) -> Result<()> {
        *self.filter.write().await = filter;
        debug!(
            "Updated attention filter: min_relevance={:.2}, min_priority={:.2}, \
             allow_interrupts={}",
            self.filter.read().await.min_relevance,
            self.filter.read().await.min_priority,
            self.filter.read().await.allow_interrupts
        );
        Ok(())
    }

    /// Get current attention strategy
    pub async fn get_strategy(&self) -> AttentionStrategy {
        *self.strategy.read().await
    }

    /// Set attention strategy
    pub async fn set_strategy(&self, strategy: AttentionStrategy) -> Result<()> {
        *self.strategy.write().await = strategy;
        info!("Changed attention strategy to: {:?}", strategy);
        Ok(())
    }

    /// Reduce attention scope during high cognitive load
    pub async fn reduce_scope(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.scope = (state.scope * 0.8).max(0.1); // Reduce scope but keep minimum
        info!("Reduced attention scope to {:.2}", state.scope);
        Ok(())
    }

    /// Expand attention scope during low cognitive load
    pub async fn expand_scope(&self) -> Result<()> {
        let mut state = self.state.write().await;
        state.scope = (state.scope * 1.2).min(1.0); // Expand scope but cap at maximum
        info!("Expanded attention scope to {:.2}", state.scope);
        Ok(())
    }

    /// Add focus target (referenced by tests)
    pub async fn add_focus_target(&self, target: FocusTarget) -> Result<()> {
        let mut targets = self.focus_targets.write().await;
        
        // Respect maximum focus targets
        if targets.len() >= self.config.max_focus_targets {
            // Remove least relevant target
            targets.sort_by(|a, b| a.relevance.partial_cmp(&b.relevance).unwrap());
            targets.remove(0);
        }
        
        targets.push(target);
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.focus_changes += 1;
        
        Ok(())
    }

    /// Get current focus target (referenced by tests)
    pub async fn get_current_focus(&self) -> Result<Option<FocusTarget>> {
        let targets = self.focus_targets.read().await;
        
        // Return highest priority target
        Ok(targets.iter()
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap())
            .cloned())
    }

    /// Check if distraction should interrupt current focus (referenced by tests)
    pub async fn should_interrupt_for_distraction(&self, distraction: &AttentionDistraction) -> bool {
        let filter = self.filter.read().await;
        let current_focus = self.get_current_focus().await.ok().flatten();
        
        // Don't interrupt if interrupts are disabled
        if !filter.allow_interrupts {
            return false;
        }
        
        // High intensity distractions can interrupt
        if distraction.intensity > 0.8 {
            return true;
        }
        
        // If we have a high priority focus, be more resistant to interruption
        if let Some(focus) = current_focus {
            if focus.priority > 0.8 && distraction.intensity < 0.6 {
                return false;
            }
        }
        
        // Moderate distractions can interrupt if focus is low priority
        distraction.intensity > 0.5
    }

    /// Update attention quality (referenced by tests)
    pub async fn update_attention_quality(&self, quality: f32) {
        let mut span = self.attention_span.write().await;
        span.quality = quality.clamp(0.0, 1.0);
        
        // Quality affects fatigue accumulation rate
        if quality < 0.5 {
            span.fatigue_level = (span.fatigue_level + 0.05).min(1.0);
        }
    }

    /// Accumulate fatigue (referenced by tests)
    pub async fn accumulate_fatigue(&self, amount: f32) {
        let mut span = self.attention_span.write().await;
        span.fatigue_level = (span.fatigue_level + amount).min(1.0);
        
        // Update quality based on fatigue
        span.quality = (1.0 - span.fatigue_level * 0.7).max(0.1);
    }

    /// Restore attention after rest (referenced by tests)
    pub async fn restore_attention(&self) -> Result<()> {
        let mut span = self.attention_span.write().await;
        
        // Restore based on recovery rate
        span.fatigue_level = (span.fatigue_level - self.config.recovery_rate * 10.0).max(0.0);
        span.quality = (1.0 - span.fatigue_level * 0.5).max(0.5);
        
        // Reset current duration
        span.current_duration = Duration::from_secs(0);
        
        info!("Attention restored: quality={:.2}, fatigue={:.2}", span.quality, span.fatigue_level);
        Ok(())
    }

    /// Calculate cognitive load distribution across targets (referenced by tests)
    pub async fn calculate_cognitive_load_distribution(&self) -> HashMap<String, f32> {
        let targets = self.focus_targets.read().await;
        let mut allocation = HashMap::new();
        
        if targets.is_empty() {
            return allocation;
        }
        
        // Calculate total weighted importance
        let total_weight: f32 = targets.iter()
            .map(|t| t.priority * t.relevance)
            .sum();
        
        if total_weight == 0.0 {
            // Equal distribution if no weights
            let equal_share = 1.0 / targets.len() as f32;
            for target in targets.iter() {
                allocation.insert(target.id.clone(), equal_share);
            }
        } else {
            // Weighted distribution
            for target in targets.iter() {
                let weight = target.priority * target.relevance;
                let share = weight / total_weight;
                allocation.insert(target.id.clone(), share);
            }
        }
        
        allocation
    }

    /// Process input and return attention analysis result
    pub async fn process_input(&self, input: &str) -> Result<AttentionResult> {
        debug!("🎯 Processing input for attention analysis: {}", input);
        
        // Analyze input for attention requirements
        let attention_demand = self.analyze_attention_demand(input).await?;
        
        // Extract focus areas from input
        let focus_areas = self.extract_focus_areas(input).await?;
        
        // Calculate attention score based on current state and demands
        let current_state = self.state.read().await;
        let current_load = self.cognitive_load.read().await;
        
        // Calculate focus score based on attention intensity and cognitive availability
        let focus_score = current_state.intensity * (1.0 - current_load.total_load);
        
        // Generate attention recommendations
        let recommendations = self.generate_attention_recommendations(&attention_demand, &focus_areas).await?;
        
        let result = AttentionResult {
            focus_areas,
            focus_score,
            attention_demand,
            recommendations,
            cognitive_load: current_load.total_load,
        };
        
        // Update attention state based on processing
        self.update_attention_from_input(input).await?;
        
        info!("✅ Attention processing completed - focus score: {:.3}", focus_score);
        Ok(result)
    }

    /// Analyze the attention demand of input text
    async fn analyze_attention_demand(&self, input: &str) -> Result<f32> {
        // Simple heuristic-based attention demand calculation
        let word_count = input.split_whitespace().count();
        let complexity_indicators = ["complex", "difficult", "analyze", "understand", "think", "consider"];
        
        let complexity_score = complexity_indicators.iter()
            .map(|&indicator| if input.to_lowercase().contains(indicator) { 0.2 } else { 0.0 })
            .sum::<f32>();
        
        // Calculate demand based on length and complexity
        let length_factor = (word_count as f32 / 100.0).min(1.0); // Normalize to 0-1
        let demand = (length_factor + complexity_score).min(1.0);
        
        Ok(demand)
    }

    /// Extract focus areas from input text
    async fn extract_focus_areas(&self, input: &str) -> Result<Vec<String>> {
        // Simple keyword extraction for focus areas
        let focus_keywords = ["problem", "task", "goal", "issue", "question", "analyze", "understand", "create", "build"];
        let words: Vec<&str> = input.split_whitespace().collect();
        
        let mut focus_areas = Vec::new();
        
        // Look for focus-related keywords and extract context
        for (i, word) in words.iter().enumerate() {
            if focus_keywords.contains(&word.to_lowercase().as_str()) {
                // Extract context around focus keyword
                let start = i.saturating_sub(2);
                let end = (i + 3).min(words.len());
                let context = words[start..end].join(" ");
                focus_areas.push(context);
            }
        }
        
        // If no specific focus areas found, extract key nouns
        if focus_areas.is_empty() {
            focus_areas = words.iter()
                .filter(|word| word.len() > 4) // Likely to be meaningful nouns
                .take(3)
                .map(|&s| s.to_string())
                .collect();
        }
        
        Ok(focus_areas)
    }

    /// Generate attention recommendations
    async fn generate_attention_recommendations(&self, demand: &f32, focus_areas: &[String]) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        let current_load = self.cognitive_load.read().await;
        
        if *demand > 0.7 && current_load.total_load > 0.6 {
            recommendations.push("High attention demand detected - consider breaking into smaller tasks".to_string());
        }
        
        if focus_areas.len() > 3 {
            recommendations.push("Multiple focus areas detected - prioritize most important aspects".to_string());
        }
        
        if current_load.total_load > 0.8 {
            recommendations.push("Cognitive load high - consider attention restoration break".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Attention resources available - good conditions for focused work".to_string());
        }
        
        Ok(recommendations)
    }

    /// Update attention state based on input processing
    async fn update_attention_from_input(&self, input: &str) -> Result<()> {
        let mut state = self.state.write().await;
        let mut load = self.cognitive_load.write().await;
        
        // Increase processing load based on input complexity
        let complexity = self.analyze_attention_demand(input).await?;
        load.processing_load = (load.processing_load + complexity * 0.1).min(1.0);
        load.total_load = (load.processing_load + load.memory_load + load.emotional_load) / 3.0;
        
        // Adjust attention intensity based on load
        if load.total_load > 0.8 {
            state.intensity = (state.intensity - 0.1).max(0.3);
        } else {
            state.intensity = (state.intensity + 0.05).min(1.0);
        }
        
        Ok(())
    }

    /// Create a new attention manager with default dependencies
    pub async fn new_with_defaults() -> Result<Self> {
        // Create minimal dependencies for testing/standalone use
        let memory = CognitiveMemory::new_minimal().await?;
        
        let emotional_config = crate::cognitive::emotional_core::EmotionalConfig::default();
        let emotional_core = Arc::new(crate::cognitive::emotional_core::EmotionalCore::new(
            memory.clone(), 
            emotional_config
        ).await?);
        
        let cache_config = SimdCacheConfig::default();
        let cache = Arc::new(SimdSmartCache::new(cache_config));
        let neural_processor = Arc::new(NeuroProcessor::new(cache).await?);
        
        let config = AttentionConfig::default();
        
        Self::new(neural_processor, emotional_core, config).await
    }
}

/// Result from attention processing
#[derive(Debug, Clone)]
pub struct AttentionResult {
    /// Focus areas identified in the input
    pub focus_areas: Vec<String>,
    
    /// Overall focus score (0.0 to 1.0)
    pub focus_score: f32,
    
    /// Attention demand of the input
    pub attention_demand: f32,
    
    /// Recommendations for attention management
    pub recommendations: Vec<String>,
    
    /// Current cognitive load
    pub cognitive_load: f32,
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_attention_filtering() {
        // Test attention filter functionality with comprehensive scenarios
        use super::*;
        
        // Create mock neural processor and emotional core
        // Note: In a real test, these would be proper mocks
        use crate::cognitive::{NeuroProcessor, EmotionalCore};
        use crate::memory::CognitiveMemory;
        
        // Create mock dependencies
        let memory = Arc::new(CognitiveMemory::new(
            crate::memory::MemoryConfig::default(),
        ).await.unwrap());
        
        let emotional_core = Arc::new(EmotionalCore::new(
            memory.clone(),
            crate::cognitive::emotional_core::EmotionalConfig::default(),
        ).await.unwrap());
        
        let cache_config = SimdCacheConfig::default();
        let cache = Arc::new(SimdSmartCache::new(cache_config));
        let neural_processor = Arc::new(NeuroProcessor::new(cache).await.unwrap());
        
        let config = AttentionConfig::default();
        let manager = AttentionManager::new(
            neural_processor,
            emotional_core,
            config,
        ).await.unwrap();
        
        // Test priority-based filtering
        let high_priority_target = FocusTarget {
            id: "high_priority_task".to_string(),
            target_type: FocusType::Task,
            priority: 0.9,
            relevance: 0.8,
            time_allocated: Duration::from_secs(30 * 60),
            started_at: Instant::now(),
            context: vec!["urgent".to_string(), "critical".to_string()],
        };
        
        let low_priority_target = FocusTarget {
            id: "low_priority_task".to_string(),
            target_type: FocusType::Learning,
            priority: 0.3,
            relevance: 0.5,
            time_allocated: Duration::from_secs(15 * 60),
            started_at: Instant::now(),
            context: vec!["optional".to_string()],
        };
        
        // Add targets to attention queue
        manager.add_focus_target(high_priority_target.clone()).await.unwrap();
        manager.add_focus_target(low_priority_target.clone()).await.unwrap();
        
        // Test that high priority target is selected first
        let current_focus = manager.get_current_focus().await.unwrap();
        assert!(current_focus.is_some());
        assert_eq!(current_focus.unwrap().id, "high_priority_task");
        
        // Test distraction filtering
        let distraction = AttentionDistraction {
            source: "notification".to_string(),
            intensity: 0.4, // Moderate distraction
            description: "New email arrived".to_string(),
            timestamp: Instant::now(),
        };
        
        // Should be filtered out during high-priority focus
        let should_interrupt = manager.should_interrupt_for_distraction(&distraction).await;
        assert!(!should_interrupt, "Low-intensity distraction should not interrupt high-priority task");
        
        // Test high-intensity distraction
        let urgent_distraction = AttentionDistraction {
            source: "emergency".to_string(),
            intensity: 0.95, // Very high distraction
            description: "System alert".to_string(),
            timestamp: Instant::now(),
        };
        
        let should_interrupt_urgent = manager.should_interrupt_for_distraction(&urgent_distraction).await;
        assert!(should_interrupt_urgent, "High-intensity distraction should interrupt focus");
    }
    
    #[tokio::test]
    async fn test_attention_span_management() {
        // Test attention span tracking and fatigue management
        use super::*;
        
        // Create mock dependencies
        let memory = Arc::new(CognitiveMemory::new(
            crate::memory::MemoryConfig::default(),
        ).await.unwrap());
        
        let emotional_core = Arc::new(EmotionalCore::new(
            memory.clone(),
            crate::cognitive::emotional_core::EmotionalConfig::default(),
        ).await.unwrap());
        
        let cache_config = SimdCacheConfig::default();
        let cache = Arc::new(SimdSmartCache::new(cache_config));
        let neural_processor = Arc::new(NeuroProcessor::new(cache).await.unwrap());
        
        let config = AttentionConfig::default();
        let manager = AttentionManager::new(
            neural_processor,
            emotional_core,
            config,
        ).await.unwrap();
        
        // Create a long-duration focus target
        let focus_target = FocusTarget {
            id: "long_task".to_string(),
            target_type: FocusType::Problem,
            priority: 0.7,
            relevance: 0.8,
            time_allocated: Duration::from_secs(2 * 60 * 60),
            started_at: Instant::now(),
            context: vec!["complex".to_string()],
        };
        
        manager.add_focus_target(focus_target).await.unwrap();
        
        // Simulate extended focus period
        manager.update_attention_quality(0.9).await; // Start with high quality
        
        // Simulate fatigue accumulation over time
        for _ in 0..10 {
            manager.accumulate_fatigue(0.1).await; // Add 10% fatigue each iteration
        }
        
        let span = manager.get_attention_span().await;
        assert!(span.fatigue_level > 0.5, "Fatigue should accumulate over time");
        assert!(span.quality < 0.9, "Attention quality should decrease with fatigue");
        
        // Test attention restoration
        manager.restore_attention().await.unwrap();
        let restored_span = manager.get_attention_span().await;
        assert!(restored_span.fatigue_level < 0.3, "Fatigue should decrease after restoration");
        assert!(restored_span.quality > 0.7, "Quality should improve after restoration");
    }
    
    #[tokio::test] 
    async fn test_cognitive_load_balancing() {
        // Test cognitive resource allocation and load balancing
        use super::*;
        
        // Create mock dependencies
        let memory = Arc::new(CognitiveMemory::new(
            crate::memory::MemoryConfig::default(),
        ).await.unwrap());
        
        let emotional_core = Arc::new(EmotionalCore::new(
            memory.clone(),
            crate::cognitive::emotional_core::EmotionalConfig::default(),
        ).await.unwrap());
        
        let cache_config = SimdCacheConfig::default();
        let cache = Arc::new(SimdSmartCache::new(cache_config));
        let neural_processor = Arc::new(NeuroProcessor::new(cache).await.unwrap());
        
        let config = AttentionConfig::default();
        let manager = AttentionManager::new(
            neural_processor,
            emotional_core,
            config,
        ).await.unwrap();
        
        // Create multiple competing focus targets
        let targets = vec![
            FocusTarget {
                id: "task_a".to_string(),
                target_type: FocusType::Task,
                priority: 0.8,
                relevance: 0.7,
                time_allocated: Duration::from_secs(45 * 60),
                started_at: Instant::now(),
                context: vec!["analytical".to_string()],
            },
            FocusTarget {
                id: "creative_b".to_string(),
                target_type: FocusType::Creative,
                priority: 0.6,
                relevance: 0.9,
                time_allocated: Duration::from_secs(30 * 60),
                started_at: Instant::now(),
                context: vec!["artistic".to_string()],
            },
            FocusTarget {
                id: "social_c".to_string(),
                target_type: FocusType::Social,
                priority: 0.7,
                relevance: 0.6,
                time_allocated: Duration::from_secs(20 * 60),
                started_at: Instant::now(),
                context: vec!["communication".to_string()],
            },
        ];
        
        // Add all targets
        for target in targets {
            manager.add_focus_target(target).await.unwrap();
        }
        
        // Test load balancing algorithm
        let allocation = manager.calculate_cognitive_load_distribution().await;
        
        // Verify that allocation sums to approximately 1.0 (100%)
        let total_allocation: f32 = allocation.values().sum();
        assert!((total_allocation - 1.0).abs() < 0.1, "Total cognitive allocation should be ~100%");
        
        // Verify that higher priority/relevance tasks get more allocation
        assert!(allocation.get("task_a").unwrap_or(&0.0) > allocation.get("creative_b").unwrap_or(&0.0),
                "Higher priority task should get more cognitive resources");
    }
}
