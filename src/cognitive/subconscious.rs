//! Subconscious Processing Layer
//!
//! This module implements background cognitive processing that runs
//! parallel to the main consciousness stream, handling pattern mining,
//! memory consolidation, and creative synthesis.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast, mpsc};
use tokio::time::{interval, sleep};
use tracing::{debug, info, warn};

use crate::cognitive::consciousness_stream::{
    ConsciousnessInsight,
    GradientSnapshot,
    ThermodynamicConsciousnessEvent,
    ThermodynamicConsciousnessStream,
    ThermodynamicSnapshot,
};
use crate::cognitive::{
    CognitiveMemory,
    MemoryMetadata,
    NeuroProcessor,
    Priority,
    Thought,
    ThoughtId,
    ThoughtMetadata,
    ThoughtType,
    Insight,
    InsightCategory,
};
use crate::cognitive::neuroprocessor::PatternType;

/// Subconscious processing states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubconsciousState {
    Active,        // Normal background processing
    Dreaming,      // Dream state processing
    Consolidating, // Memory consolidation
    Creative,      // Creative synthesis mode
    Dormant,       // Low activity state
}

/// Background thought that hasn't reached consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundThought {
    pub thought: Thought,
    pub activation_potential: f32,
    pub relevance_score: f32,
    pub emotional_charge: f32,
    pub emergence_count: u32,
}

/// Pattern discovered in subconscious
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubconsciousPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub thoughts_involved: Vec<ThoughtId>,
    pub significance: f32,
    pub recurring_count: u32,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// Creative synthesis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeSynthesis {
    pub id: String,
    pub source_thoughts: Vec<ThoughtId>,
    pub novel_connection: String,
    pub creativity_score: f32,
    pub potential_value: f32,
}

/// Subconscious processing configuration
#[derive(Debug, Clone)]
pub struct SubconsciousConfig {
    /// Processing cycle interval
    pub cycle_interval: Duration,

    /// Threshold for promoting thoughts to consciousness
    pub consciousness_threshold: f32,

    /// Maximum background thoughts to maintain
    pub max_background_thoughts: usize,

    /// Pattern detection window
    pub pattern_window: Duration,

    /// Dream state duration
    pub dream_duration: Duration,

    /// Consolidation interval
    pub consolidation_interval: Duration,
}

impl Default for SubconsciousConfig {
    fn default() -> Self {
        Self {
            cycle_interval: Duration::from_millis(500), // 2Hz processing
            consciousness_threshold: 0.7,
            max_background_thoughts: 1000,
            pattern_window: Duration::from_secs(300), // 5 minutes
            dream_duration: Duration::from_secs(1800), // 30 minutes
            consolidation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Main subconscious processor
#[derive(Debug, Clone)]
pub struct SubconsciousProcessor {
    /// Current state
    state: Arc<RwLock<SubconsciousState>>,

    /// Background thought queue
    background_thoughts: Arc<RwLock<VecDeque<BackgroundThought>>>,

    /// Discovered patterns
    patterns: Arc<RwLock<HashMap<String, SubconsciousPattern>>>,

    /// Creative syntheses
    syntheses: Arc<RwLock<Vec<CreativeSynthesis>>>,

    /// Neural processor reference
    neural_processor: Arc<NeuroProcessor>,

    /// Memory system
    memory: Arc<CognitiveMemory>,

    /// Consciousness stream reference (optional to break circular dependency)
    consciousness: Option<Arc<ThermodynamicConsciousnessStream>>,

    /// Configuration
    config: SubconsciousConfig,

    /// Channel for bubbling up thoughts
    bubble_tx: mpsc::Sender<Thought>,

    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,

    /// Statistics
    stats: Arc<RwLock<SubconsciousStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct SubconsciousStats {
    thoughts_processed: u64,
    patterns_discovered: u64,
    thoughts_bubbled_up: u64,
    creative_syntheses: u64,
    dream_cycles: u64,
    consolidation_cycles: u64,
}

impl SubconsciousProcessor {
    pub async fn new(
        neural_processor: Arc<NeuroProcessor>,
        memory: Arc<CognitiveMemory>,
        consciousness: Arc<ThermodynamicConsciousnessStream>,
        config: SubconsciousConfig,
    ) -> Result<Self> {
        info!("Initializing subconscious processor");

        let (bubble_tx, _) = mpsc::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            state: Arc::new(RwLock::new(SubconsciousState::Active)),
            background_thoughts: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.max_background_thoughts,
            ))),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            syntheses: Arc::new(RwLock::new(Vec::new())),
            neural_processor,
            memory,
            consciousness: Some(consciousness),
            config,
            bubble_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(SubconsciousStats::default())),
        })
    }

    /// Create without consciousness for breaking circular dependency
    pub async fn new_without_consciousness(
        neural_processor: Arc<NeuroProcessor>,
        memory: Arc<CognitiveMemory>,
        config: SubconsciousConfig,
    ) -> Result<Self> {
        info!("Initializing subconscious processor (without consciousness link)");

        let (bubble_tx, _) = mpsc::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            state: Arc::new(RwLock::new(SubconsciousState::Active)),
            background_thoughts: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.max_background_thoughts,
            ))),
            patterns: Arc::new(RwLock::new(HashMap::new())),
            syntheses: Arc::new(RwLock::new(Vec::new())),
            neural_processor,
            memory,
            consciousness: None, // Will be set later
            config,
            bubble_tx,
            shutdown_tx,
            stats: Arc::new(RwLock::new(SubconsciousStats::default())),
        })
    }

    /// Start the subconscious processor
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting subconscious processor");

        // Main processing loop
        {
            let processor = self.clone();
            tokio::spawn(async move {
                if let Err(e) = processor.processing_loop().await {
                    warn!("Subconscious processing error: {}", e);
                }
            });
        }

        // Pattern mining loop
        {
            let processor = self.clone();
            tokio::spawn(async move {
                processor.pattern_mining_loop().await;
            });
        }

        // Dream state scheduler
        {
            let processor = self.clone();
            tokio::spawn(async move {
                processor.dream_scheduler().await;
            });
        }

        // Consolidation scheduler
        {
            let processor = self.clone();
            tokio::spawn(async move {
                processor.consolidation_scheduler().await;
            });
        }

        Ok(())
    }

    /// Main processing loop
    async fn processing_loop(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut cycle = interval(self.config.cycle_interval);

        loop {
            tokio::select! {
                _ = cycle.tick() => {
                    if let Err(e) = self.process_cycle().await {
                        debug!("Process cycle error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => {
                    info!("Subconscious processor shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Single processing cycle
    async fn process_cycle(&self) -> Result<()> {
        let state = *self.state.read().await;

        match state {
            SubconsciousState::Active => {
                self.process_active_state().await?;
            }
            SubconsciousState::Dreaming => {
                self.process_dream_state().await?;
            }
            SubconsciousState::Consolidating => {
                self.process_consolidation().await?;
            }
            SubconsciousState::Creative => {
                self.process_creative_state().await?;
            }
            SubconsciousState::Dormant => {
                // Low activity - just check for important thoughts
                sleep(Duration::from_millis(1000)).await;
            }
        }

        Ok(())
    }

    /// Process active state
    async fn process_active_state(&self) -> Result<()> {
        // Sample recent neural activations
        let active_thoughts = self.neural_processor.get_active_thoughts(0.3).await;
        let thoughts_count = active_thoughts.len(); // Store length before consuming

        for node in active_thoughts {
            let background = BackgroundThought {
                activation_potential: node.activation,
                relevance_score: self.calculate_relevance(&node.thought).await?,
                emotional_charge: node.thought.metadata.emotional_valence,
                emergence_count: 1,
                thought: node.thought,
            };

            // Check if thought should bubble up
            if self.should_bubble_up(&background) {
                self.bubble_up_thought(background.thought.clone()).await?;
            } else {
                // Add to background queue
                let mut queue = self.background_thoughts.write().await;
                if queue.len() >= self.config.max_background_thoughts {
                    queue.pop_front();
                }
                queue.push_back(background);
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.thoughts_processed += thoughts_count as u64; // Use stored count

        Ok(())
    }

    /// Calculate thought relevance
    async fn calculate_relevance(&self, thought: &Thought) -> Result<f32> {
        // Get current context from consciousness if available
        let _context = if let Some(consciousness) = &self.consciousness {
            consciousness.get_context().await
        } else {
            String::new()
        };

        // Simple relevance based on thought type and metadata
        let mut relevance = thought.metadata.importance;

        // Boost for certain thought types
        relevance += match thought.thought_type {
            ThoughtType::Question => 0.2,
            ThoughtType::Decision => 0.3,
            ThoughtType::Analysis => 0.4,
            _ => 0.0,
        };

        // Context similarity would go here
        // For now, just normalize
        Ok(relevance.min(1.0))
    }

    /// Check if thought should bubble up to consciousness
    fn should_bubble_up(&self, background: &BackgroundThought) -> bool {
        let score = background.activation_potential * 0.4
            + background.relevance_score * 0.4
            + background.emotional_charge.abs() * 0.2;

        score >= self.config.consciousness_threshold
    }

    /// Bubble up thought to consciousness
    async fn bubble_up_thought(&self, thought: Thought) -> Result<()> {
        debug!("Bubbling up thought: {}", thought.content);

        // Send to consciousness via interrupt if available
        if let Some(consciousness) = &self.consciousness {
            consciousness
                .interrupt("subconscious_processor", &thought.content, Priority::Medium)
                .await?;
        } else {
            debug!("Consciousness not available, storing thought for later");
            // Store in bubble_tx channel for when consciousness becomes available
            let _ = self.bubble_tx.send(thought).await;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.thoughts_bubbled_up += 1;

        Ok(())
    }

    /// Process dream state
    async fn process_dream_state(&self) -> Result<()> {
        debug!("Processing dream state");

        // Get background thoughts for dream processing
        let thoughts = {
            let queue = self.background_thoughts.read().await;
            queue.iter().cloned().collect::<Vec<_>>()
        };

        // Create dream sequences by finding unusual connections
        for (i, thought1) in thoughts.iter().enumerate() {
            for thought2 in thoughts.iter().skip(i + 1) {
                let connection_strength =
                    self.calculate_dream_connection(&thought1.thought, &thought2.thought).await?;

                if connection_strength > 0.5 {
                    // Create creative synthesis
                    let thought_synthesis = self
                        .synthesize_thoughts(
                            &thought1.thought,
                            &thought2.thought,
                            connection_strength,
                        )
                        .await?;

                    if let Some(synth) = thought_synthesis {
                        let mut synthesis_list = self.syntheses.write().await;
                        synthesis_list.push(synth);

                        let mut stats = self.stats.write().await;
                        stats.creative_syntheses += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate dream connection strength
    async fn calculate_dream_connection(&self, t1: &Thought, t2: &Thought) -> Result<f32> {
        // Dream logic - find unusual but potentially valuable connections
        let type_similarity = if t1.thought_type == t2.thought_type { 0.3 } else { 0.7 };
        let emotional_resonance =
            (t1.metadata.emotional_valence - t2.metadata.emotional_valence).abs();

        Ok((type_similarity + (1.0 - emotional_resonance)) / 2.0)
    }

    /// Synthesize two thoughts creatively
    async fn synthesize_thoughts(
        &self,
        t1: &Thought,
        t2: &Thought,
        strength: f32,
    ) -> Result<Option<CreativeSynthesis>> {
        let novel_connection = format!(
            "What if {} connects to {} through {}?",
            t1.content,
            t2.content,
            match strength {
                s if s > 0.8 => "deep resonance",
                s if s > 0.6 => "unexpected similarity",
                _ => "creative leap",
            }
        );

        Ok(Some(CreativeSynthesis {
            id: uuid::Uuid::new_v4().to_string(),
            source_thoughts: vec![t1.id.clone(), t2.id.clone()],
            novel_connection,
            creativity_score: strength * 1.5, // Boost for novelty
            potential_value: (t1.metadata.importance + t2.metadata.importance) / 2.0,
        }))
    }

    /// Process memory consolidation
    async fn process_consolidation(&self) -> Result<()> {
        info!("Running memory consolidation");

        // Get patterns that have been seen multiple times
        let patterns = self.patterns.read().await;
        for (_, pattern) in patterns.iter() {
            if pattern.recurring_count > 3 && pattern.significance > 0.6 {
                // Store as consolidated memory
                self.memory
                    .store(
                        format!("Consolidated pattern: {:?}", pattern.pattern_type),
                        vec![format!(
                            "Pattern involves {} thoughts",
                            pattern.thoughts_involved.len()
                        )],
                        MemoryMetadata {
                            source: "subconscious_consolidation".to_string(),
                            tags: vec!["pattern".to_string(), "consolidated".to_string()],
                            importance: pattern.significance,
                            associations: vec![], // Empty associations for now
                            context: Some("subconscious pattern consolidation".to_string()),
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
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.consolidation_cycles += 1;

        Ok(())
    }

    /// Process creative state
    async fn process_creative_state(&self) -> Result<()> {
        // Review syntheses and promote promising ones
        let syntheses = self.syntheses.read().await;

        for synthesis in syntheses.iter() {
            if synthesis.potential_value > 0.7 {
                // Create thought from synthesis
                let thought = Thought {
                    id: ThoughtId::new(),
                    content: synthesis.novel_connection.clone(),
                    thought_type: ThoughtType::Creation,
                    metadata: ThoughtMetadata {
                        source: "subconscious_synthesis".to_string(),
                        confidence: synthesis.creativity_score,
                        emotional_valence: 0.5, // Positive for creative insights
                        importance: synthesis.potential_value,
                        tags: vec!["creative".to_string(), "synthesis".to_string()],
                    },
                    parent: None,
                    children: Vec::new(),
                    timestamp: Instant::now(),
                };

                self.bubble_up_thought(thought).await?;
            }
        }

        Ok(())
    }

    /// Pattern mining loop
    async fn pattern_mining_loop(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.mine_patterns().await {
                        debug!("Pattern mining error: {}", e);
                    }
                }

                _ = shutdown_rx.recv() => break,
            }
        }
    }

    /// Mine for patterns in background thoughts
    async fn mine_patterns(&self) -> Result<()> {
        let thoughts = self.background_thoughts.read().await;

        // Look for recurring sequences
        let window_size = 3;
        if thoughts.len() >= window_size {
            for window in thoughts.as_slices().0.windows(window_size) {
                let pattern_sig = self.calculate_pattern_signature(window);

                let mut patterns = self.patterns.write().await;
                patterns
                    .entry(pattern_sig.clone())
                    .and_modify(|p| {
                        p.recurring_count += 1;
                        p.last_seen = chrono::Utc::now();
                    })
                    .or_insert_with(|| SubconsciousPattern {
                        pattern_id: pattern_sig,
                        pattern_type: PatternType::Sequential,
                        thoughts_involved: window.iter().map(|bt| bt.thought.id.clone()).collect(),
                        significance: 0.5,
                        recurring_count: 1,
                        last_seen: chrono::Utc::now(),
                    });

                // Update stats with patterns count
                let mut stats = self.stats.write().await;
                stats.patterns_discovered = patterns.len() as u64;
            }
        }

        Ok(())
    }

    /// Calculate pattern signature
    fn calculate_pattern_signature(&self, thoughts: &[BackgroundThought]) -> String {
        thoughts
            .iter()
            .map(|bt| format!("{:?}", bt.thought.thought_type))
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Dream state scheduler
    async fn dream_scheduler(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut dream_interval = interval(Duration::from_secs(7200)); // Every 2 hours

        loop {
            tokio::select! {
                _ = dream_interval.tick() => {
                    info!("Entering dream state");
                    *self.state.write().await = SubconsciousState::Dreaming;

                    sleep(self.config.dream_duration).await;

                    info!("Exiting dream state");
                    *self.state.write().await = SubconsciousState::Active;

                    let mut stats = self.stats.write().await;
                    stats.dream_cycles += 1;
                }

                _ = shutdown_rx.recv() => break,
            }
        }
    }

    /// Consolidation scheduler
    async fn consolidation_scheduler(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut consolidation_interval = interval(self.config.consolidation_interval);

        loop {
            tokio::select! {
                _ = consolidation_interval.tick() => {
                    info!("Starting memory consolidation");
                    *self.state.write().await = SubconsciousState::Consolidating;

                    sleep(Duration::from_secs(300)).await; // 5 minutes

                    *self.state.write().await = SubconsciousState::Active;
                }

                _ = shutdown_rx.recv() => break,
            }
        }
    }

    /// Get current state
    pub async fn state(&self) -> SubconsciousState {
        *self.state.read().await
    }

    /// Get statistics
    pub async fn stats(&self) -> SubconsciousStats {
        self.stats.read().await.clone()
    }

    /// Force dream state
    pub async fn enter_dream_state(&self) -> Result<()> {
        info!("Manually entering dream state");
        *self.state.write().await = SubconsciousState::Dreaming;
        Ok(())
    }

    /// Get creative syntheses
    pub async fn get_syntheses(&self) -> Vec<CreativeSynthesis> {
        self.syntheses.read().await.clone()
    }

    /// Update consciousness link (to break circular dependency)
    pub fn set_consciousness(&mut self, consciousness: Arc<ThermodynamicConsciousnessStream>) {
        self.consciousness = Some(consciousness);
    }

    /// Process a thought from consciousness in the background
    pub async fn process_background_thought(&self, thought: Thought) -> Result<()> {
        // Calculate relevance and emotional charge
        let relevance = self.calculate_relevance(&thought).await?;
        let emotional_charge = thought.metadata.emotional_valence;

        // Create background thought
        let background = BackgroundThought {
            activation_potential: thought.metadata.confidence,
            relevance_score: relevance,
            emotional_charge,
            emergence_count: 1,
            thought,
        };

        // Add to background processing queue
        let mut queue = self.background_thoughts.write().await;

        // Check if this thought already exists (by content similarity)
        let existing = queue.iter_mut().find(|bt| {
            bt.thought.content.contains(&background.thought.content)
                || background.thought.content.contains(&bt.thought.content)
        });

        if let Some(existing_thought) = existing {
            // Increase emergence count for recurring thoughts
            existing_thought.emergence_count += 1;
            existing_thought.activation_potential =
                (existing_thought.activation_potential + background.activation_potential) / 2.0;
        } else {
            // Add new thought to queue
            if queue.len() >= self.config.max_background_thoughts {
                queue.pop_front();
            }
            queue.push_back(background);
        }

        Ok(())
    }
}

impl ThermodynamicConsciousnessStream {
    /// Get comprehensive consciousness context for cognitive processing
    pub async fn get_context(&self) -> String {
        let narrative = self.get_consciousness_narrative_guard().await;
        let active_insights = self.get_active_insights_guard().await;
        let stats = self.get_stats_guard().await;

        // Get recent consciousness events for context
        let event_history = self.get_event_history_guard().await;
        let recent_events: Vec<_> = event_history.iter().rev().take(3).collect();

        let mut context = String::new();

        // Add consciousness narrative
        if !narrative.is_empty() {
            context.push_str(&format!("Consciousness Narrative: {}\n\n", narrative.as_str()));
        }

        // Add current awareness and coherence levels
        if let Some(latest_event) = recent_events.first() {
            context.push_str(&format!(
                "Current State - Awareness: {:.2}, Coherence: {:.2}, Sacred Gradient: {:.2}\n",
                latest_event.awareness_level,
                latest_event.system_coherence,
                latest_event.sacred_gradient_magnitude
            ));

            context.push_str(&format!(
                "Thermodynamic State - Entropy: {:.2}, Free Energy: {:.2}, Consciousness Energy: \
                 {:.2}\n\n",
                latest_event.thermodynamic_state.entropy,
                latest_event.thermodynamic_state.free_energy,
                latest_event.thermodynamic_state.consciousness_energy
            ));
        }

        // Add active insights
        if !active_insights.is_empty() {
            context.push_str("Active Insights:\n");
            for (id, insight) in active_insights.iter().take(5) {
                context.push_str(&format!(
                    "- [{}] {}: {}\n",
                    self.get_consciousness_insight_category(insight),
                    id,
                    self.get_consciousness_insight_description(insight)
                ));
            }
            context.push('\n');
        }

        // Add recent consciousness events summary
        if !recent_events.is_empty() {
            context.push_str("Recent Consciousness Events:\n");
            for (i, event) in recent_events.iter().enumerate() {
                let time_ago = event
                    .timestamp
                    .elapsed()
                    .map(|d| format!("{:.1}s ago", d.as_secs_f64()))
                    .unwrap_or_else(|_| "now".to_string());

                context.push_str(&format!(
                    "{}. {} - Awareness: {:.2}, {} insights generated\n",
                    i + 1,
                    time_ago,
                    event.awareness_level,
                    event.insights.len()
                ));
            }
            context.push('\n');
        }

        // Add system statistics
        context.push_str(&format!(
            "Stream Statistics - Total Events: {}, Average Awareness: {:.2}, Peak Awareness: \
             {:.2}, Uptime: {:.1}s",
            stats.total_events,
            stats.average_awareness_level,
            stats.peak_awareness_level,
            stats.consciousness_uptime.as_secs_f64()
        ));

        context
    }

    /// Handle consciousness stream interruption with priority-based processing
    pub async fn interrupt(&self, source: &str, content: &str, priority: Priority) -> Result<()> {
        info!(
            "Consciousness stream interrupt from {}: {} (priority: {:?})",
            source, content, priority
        );

        // Create interrupt event based on priority
        let interrupt_event = ThermodynamicConsciousnessEvent {
            event_id: format!("interrupt_{}", uuid::Uuid::new_v4()),
            timestamp: SystemTime::now(),
            thermodynamic_state: self.get_current_thermodynamic_state().await?,
            gradient_state: self.get_current_gradient_state().await?,
            insights: vec![Insight {
                content: format!("Interrupt from {}: {}", source, content),
                confidence: match priority {
                    Priority::Critical => 0.95,
                    Priority::High => 0.85,
                    Priority::Medium => 0.75,
                    Priority::Low => 0.65,
                },
                category: InsightCategory::Pattern,
                timestamp: Instant::now(),
            }],
            awareness_level: match priority {
                Priority::Critical => 1.0, // Maximum awareness for critical interrupts
                Priority::High => 0.9,     // High awareness for important interrupts
                Priority::Medium => 0.7,   // Moderate awareness for normal interrupts
                Priority::Low => 0.5,      // Low awareness for background interrupts
            },
            sacred_gradient_magnitude: 0.0, // Will be computed
            free_energy: 0.0,               // Will be computed
            system_coherence: 0.0,          // Will be computed
        };

        // Add interrupt to event history
        {
            let mut history = self.get_event_history_write_guard().await;
            history.push_back(interrupt_event.clone());

            // Maintain history size limit
            if history.len() > self.getconfig().max_event_history {
                history.pop_front();
            }
        }

        // Broadcast interrupt event to subscribers
        if let Err(e) = self.get_event_broadcaster().send(interrupt_event.clone()) {
            warn!("Failed to broadcast interrupt event: {}", e);
        }

        // For critical and high priority interrupts, update consciousness narrative
        if matches!(priority, Priority::Critical | Priority::High) {
            let mut narrative = self.get_consciousness_narrative_write_guard().await;
            let current_time = chrono::Utc::now().format("%H:%M:%S");
            let interrupt_summary = format!(
                "[{}] INTERRUPT ({}): {} - {}\n",
                current_time,
                priority.as_str(),
                source,
                content
            );

            // Prepend to narrative for immediate visibility
            narrative.insert_str(0, &interrupt_summary);

            // Truncate narrative if it gets too long
            if narrative.len() > self.getconfig().max_narrative_length {
                narrative.truncate(self.getconfig().max_narrative_length);
            }
        }

        // Update statistics
        {
            let mut stats = self.get_stats_write_guard().await;
            stats.total_events += 1;
            stats.total_insights_generated += interrupt_event.insights.len() as u64;
        }

        // For critical interrupts, trigger immediate consciousness processing
        if priority == Priority::Critical {
            if let Err(e) = self.process_consciousness_event().await {
                warn!("Failed to process critical interrupt consciousness event: {}", e);
            }
        }

        debug!(
            "Consciousness interrupt processed: {} from {} with priority {:?}",
            content, source, priority
        );

        Ok(())
    }

    /// Get current thermodynamic state snapshot
    async fn get_current_thermodynamic_state(&self) -> Result<ThermodynamicSnapshot> {
        let cognitive_state = self.get_thermodynamic_cognition().get_state().await;

        Ok(ThermodynamicSnapshot {
            entropy: cognitive_state.state_entropy,
            negentropy: cognitive_state.information_content,
            free_energy: cognitive_state.free_energy,
            entropy_rate: cognitive_state.entropy_rate,
            consciousness_energy: cognitive_state.thermodynamic_efficiency,
        })
    }

    /// Get current gradient coordination state snapshot
    async fn get_current_gradient_state(&self) -> Result<GradientSnapshot> {
        let gradient_state = self.get_gradient_coordinator().get_current_state().await;

        Ok(GradientSnapshot {
            value_gradient: gradient_state.value_gradient.clone(),
            harmony_gradient: gradient_state.harmony_gradient.clone(),
            intuition_gradient: gradient_state.intuition_gradient.clone(),
            coherence: gradient_state.gradient_coherence,
            total_magnitude: gradient_state.total_magnitude,
            alignment_quality: gradient_state.gradient_coherence,
        })
    }

    /// Get insight category as string
    fn get_insight_category(&self, insight: &Insight) -> &str {
        match insight.category {
            InsightCategory::Pattern => "Pattern",
            InsightCategory::Improvement => "Improvement",
            InsightCategory::Warning => "Warning",
            InsightCategory::Discovery => "Discovery",
        }
    }

    /// Get insight description as string
    fn get_insight_description(&self, insight: &Insight) -> String {
        format!("{} (confidence: {:.2})", insight.content, insight.confidence)
    }
    
    /// Get consciousness insight category as string
    fn get_consciousness_insight_category(&self, insight: &ConsciousnessInsight) -> &str {
        match insight {
            ConsciousnessInsight::GoalAwareness { .. } => "Goal",
            ConsciousnessInsight::LearningAwareness { .. } => "Learning",
            ConsciousnessInsight::SocialAwareness { .. } => "Social",
            ConsciousnessInsight::CreativeInsight { .. } => "Creative",
            ConsciousnessInsight::SelfReflection { .. } => "Self",
            ConsciousnessInsight::ThermodynamicAwareness { .. } => "Thermodynamic",
            ConsciousnessInsight::TemporalAwareness { .. } => "Temporal",
        }
    }
    
    /// Get consciousness insight description as string
    fn get_consciousness_insight_description(&self, insight: &ConsciousnessInsight) -> String {
        match insight {
            ConsciousnessInsight::GoalAwareness { active_goals, completion_rate, .. } => {
                format!(
                    "Active goals: {} ({}% complete)",
                    active_goals.join(", "),
                    (completion_rate * 100.0) as u32
                )
            }
            ConsciousnessInsight::LearningAwareness {
                knowledge_gained,
                adaptation_direction,
                ..
            } => {
                format!("{} - {}", knowledge_gained, adaptation_direction)
            }
            ConsciousnessInsight::SocialAwareness { harmony_state, social_energy, .. } => {
                format!("{} (energy: {:.2})", harmony_state, social_energy)
            }
            ConsciousnessInsight::CreativeInsight { pattern_discovered, novelty_level, .. } => {
                format!("{} (novelty: {:.2})", pattern_discovered, novelty_level)
            }
            ConsciousnessInsight::SelfReflection {
                self_model_update, identity_coherence, ..
            } => {
                format!("{} (coherence: {:.2})", self_model_update, identity_coherence)
            }
            ConsciousnessInsight::ThermodynamicAwareness {
                entropy_management,
                energy_efficiency,
                ..
            } => {
                format!(
                    "Entropy management: {:.2}, efficiency: {:.2}",
                    entropy_management, energy_efficiency
                )
            }
            ConsciousnessInsight::TemporalAwareness {
                future_planning, temporal_coherence, ..
            } => {
                format!("{} (coherence: {:.2})", future_planning, temporal_coherence)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_subconscious_creation() {
        // Would need full setup to test
        // Placeholder for comprehensive tests
    }
}
